"""Controlled SSE transport tests, not model or GPU verification."""

import asyncio
import json
import unittest
from unittest.mock import patch

import httpx

from benchmarks.continuous_batching import _scenario, measure, new_result


WORKLOAD = {
    "model": "controlled-stream",
    "generation": {"temperature": 0, "top_p": 1},
    "eos_token_ids": [99],
}
CASES = [
    {
        "name": name, "max_tokens": cap,
        "messages": [{"role": "user", "content": "Controlled prompt"}],
        "prompt_tokens": 2, "prompt_token_ids": [11, 12],
    }
    for name, cap in (("A", 16), ("B", 128), ("C", 32))
]


def event(value):
    return ("data: " + (value if isinstance(value, str) else json.dumps(value)) + "\n\n").encode()


def text(name, count, prompt_ids=None):
    frame = {
        "choices": [{
            "index": 0, "delta": {"content": name}, "finish_reason": None,
            "token_ids": list(range(count)),
        }],
    }
    if prompt_ids is not None:
        frame["prompt_token_ids"] = prompt_ids
    return event(frame)


def terminal(count, reason="length"):
    return (
        event({"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]})
        + event({"choices": [], "usage": {
            "prompt_tokens": 2, "completion_tokens": count, "total_tokens": 2 + count,
        }})
        + event("[DONE]")
    )


class Stream(httpx.AsyncByteStream):
    def __init__(self, body):
        self.body = body
        self.closed = asyncio.Event()

    async def __aiter__(self):
        async for frame in self.body:
            yield frame

    async def aclose(self):
        await self.body.aclose()
        self.closed.set()


class Tokenizer:
    chat_template = "Controlled template"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        return "user: " + messages[0]["content"] + "\nassistant:"

    def __call__(self, formatted, *, add_special_tokens):
        return {"input_ids": [11, 12]}


class ContinuousBatchingTests(unittest.IsolatedAsyncioTestCase):
    async def run_refill(self, mode):
        read = {name: asyncio.Event() for name in ("A", "B", "C")}
        streams = {}
        starts = []
        blocked = asyncio.Event()
        scenario = {"name": "refill"}

        async def body(name, count):
            if name == "C" and mode == "queued_c":
                read["C"].set()
                await streams["B"].closed.wait()
            yield text(name, count)
            read[name].set()  # Resuming after yield means the client consumed this text.
            if name == "A":
                await read["B"].wait()
                if mode == "b_first":
                    await streams["B"].closed.wait()
                elif mode in ("failure", "cancel"):
                    await blocked.wait()
            elif name == "B":
                await read["A"].wait()
                if mode == "failure":
                    yield event({"error": {"code": "generation_failed"}})
                    return
                if mode == "cancel":
                    await blocked.wait()
                elif mode != "b_first":
                    await read["C"].wait()
            yield terminal(count)

        def handler(request):
            count = json.loads(request.content)["max_tokens"]
            name = {16: "A", 128: "B", 32: "C"}[count]
            starts.append(name)
            if name == "C":
                self.assertTrue(streams["A"].closed.is_set())
            streams[name] = Stream(body(name, count))
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=streams[name])

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            task = asyncio.create_task(_scenario(
                client, "http://controlled.invalid", WORKLOAD, CASES, scenario, refill=True,
            ))
            try:
                if mode == "cancel":
                    await asyncio.wait_for(asyncio.gather(read["A"].wait(), read["B"].wait()), 5)
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                elif mode == "failure":
                    with self.assertRaisesRegex(RuntimeError, "generation_failed"):
                        await asyncio.wait_for(task, 5)
                else:
                    await asyncio.wait_for(task, 5)
            finally:
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        self.assertTrue(all(stream.closed.is_set() for stream in streams.values()))
        return scenario, starts

    async def test_refill_admits_c_after_a_and_before_b_finishes(self):
        scenario, starts = await self.run_refill("refill")
        rows = {row["case"]: row for row in scenario["rows"]}
        self.assertEqual(starts, ["A", "B", "C"])
        self.assertEqual(scenario["max_in_flight"], 2)
        self.assertTrue(scenario["checks"]["refill_demonstrated"])
        self.assertLess(rows["C"]["timeline"]["first_text_ms"], rows["B"]["timeline"]["done_ms"])
        self.assertGreaterEqual(rows["C"]["timeline"]["request_start_ms"], rows["A"]["timeline"]["done_ms"])
        self.assertEqual(scenario["aggregate"]["successful_completion_tokens"], 176)
        self.assertEqual([row["completion_tokens"] for row in scenario["rows"]], [16, 128, 32])

    async def test_admission_overlap_does_not_claim_refill_when_c_text_is_queued(self):
        scenario, _ = await self.run_refill("queued_c")
        self.assertTrue(scenario["checks"]["initial_text_overlap"])
        self.assertTrue(scenario["checks"]["b_active_at_c_admission"])
        self.assertFalse(scenario["checks"]["c_first_text_before_b_done"])
        self.assertFalse(scenario["checks"]["refill_demonstrated"])
        self.assertEqual(scenario["status"], "completed")

    async def test_b_finishing_before_a_retains_rows_without_claiming_refill(self):
        scenario, starts = await self.run_refill("b_first")
        self.assertEqual(starts, ["A", "B", "C"])
        self.assertFalse(scenario["checks"]["b_active_at_c_admission"])
        self.assertFalse(scenario["checks"]["refill_demonstrated"])
        self.assertEqual(scenario["status"], "completed")
        self.assertEqual(scenario["aggregate"]["successful_completion_tokens"], 176)

    async def test_initial_failure_cancels_peer_without_admitting_c(self):
        scenario, starts = await self.run_refill("failure")
        self.assertEqual(starts, ["A", "B"])
        self.assertEqual(scenario["status"], "failed")
        self.assertEqual([row["status"] for row in scenario["rows"]], ["canceled", "error"])
        self.assertEqual([row["text"] for row in scenario["rows"]], ["A", "B"])
        self.assertFalse(scenario["checks"]["refill_demonstrated"])

    async def test_cancellation_closes_both_streams_and_retains_partial_text(self):
        scenario, starts = await self.run_refill("cancel")
        self.assertEqual(starts, ["A", "B"])
        self.assertEqual(scenario["status"], "failed")
        self.assertEqual([row["status"] for row in scenario["rows"]], ["canceled", "canceled"])
        self.assertEqual([row["text"] for row in scenario["rows"]], ["A", "B"])

    async def test_serial_baseline_does_not_admit_a_successor_until_stream_closes(self):
        streams = []
        starts = []

        async def body(name, count):
            yield text(name, count)
            yield terminal(count)

        def handler(request):
            self.assertTrue(all(stream.closed.is_set() for stream in streams))
            count = json.loads(request.content)["max_tokens"]
            starts.append(count)
            stream = Stream(body(str(count), count))
            streams.append(stream)
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

        scenario = {"name": "sequential"}
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await _scenario(client, "http://controlled.invalid", WORKLOAD, CASES, scenario)
        self.assertEqual(starts, [16, 128, 32])
        self.assertEqual(scenario["max_in_flight"], 1)
        rows = scenario["rows"]
        self.assertTrue(all(
            before["timeline"]["done_ms"] <= after["timeline"]["request_start_ms"]
            for before, after in zip(rows, rows[1:])
        ))

    async def test_fixed_count_and_prompt_id_mismatch_fail_with_partial_record(self):
        for mismatch in ("count", "prompt"):
            with self.subTest(mismatch=mismatch):
                calls = []

                async def body(payload):
                    count = payload["max_tokens"]
                    # The discarded warmup succeeds; the first measured row fails validation.
                    bad = len(calls) == 2
                    if bad and mismatch == "count":
                        count -= 1
                    ids = [11, 13] if bad and mismatch == "prompt" else [11, 12]
                    yield text("Partial evidence", count, ids)
                    yield terminal(count, "stop" if bad and mismatch == "count" else "length")

                def handler(request):
                    payload = json.loads(request.content)
                    calls.append(payload)
                    return httpx.Response(
                        200, headers={"content-type": "text/event-stream"}, stream=Stream(body(payload)),
                    )

                client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
                record = new_result()
                with patch("httpx.AsyncClient", return_value=client):
                    with self.assertRaisesRegex(RuntimeError, "request checks failed"):
                        await measure("http://controlled.invalid", record, Tokenizer(), [99])
                self.assertEqual(record["status"], "failed")
                self.assertEqual([entry["status"] for entry in record["scenarios"]], ["completed", "failed"])
                row = record["scenarios"][1]["rows"][0]
                self.assertEqual(row["text"], "Partial evidence")
                key = "fixed_completion_count_matches_cap" if mismatch == "count" else "prompt_token_ids_match"
                self.assertFalse(row["checks"][key])
                self.assertFalse(record["checks"]["all_passed"])
                self.assertEqual(len(calls), 2)

    async def test_natural_eos_uses_stop_and_usage_not_last_returned_id(self):
        for count, reason, expected in ((2, "stop", True), (32, "length", False)):
            with self.subTest(count=count):
                async def body():
                    yield text("Hello", count)
                    yield terminal(count, reason)

                transport = httpx.MockTransport(lambda request: httpx.Response(
                    200, headers={"content-type": "text/event-stream"}, stream=Stream(body()),
                ))
                scenario = {"name": "natural_eos"}
                async with httpx.AsyncClient(transport=transport) as client:
                    await _scenario(
                        client, "http://controlled.invalid", WORKLOAD, [CASES[2]], scenario, fixed=False,
                    )
                self.assertEqual(scenario["checks"]["early_eos"], expected)
                self.assertFalse(scenario["checks"]["last_returned_token_is_eos"])
                self.assertTrue(scenario["checks"]["valid_usage_and_terminal_stream"])
                self.assertIsNone(scenario["rows"][0]["checks"]["prompt_token_ids_match"])


if __name__ == "__main__":
    unittest.main()
