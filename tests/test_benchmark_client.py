import json
import unittest
from unittest.mock import patch

import httpx

from benchmarks.client import run_request


CASE = {
    "name": "short-input-short-output",
    "messages": [{"role": "user", "content": "Synthetic prompt"}],
    "prompt_tokens": 12,
    "max_tokens": 8,
}
WORKLOAD = {"model": "Qwen/Qwen2.5-0.5B-Instruct", "generation": {"temperature": 0, "top_p": 1}}


def usage(prompt=12, completion=5, total=None):
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion if total is None else total,
    }


def chunk(delta=None, finish=None, counts=None):
    value = {"choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}
    if counts is not None:
        value["usage"] = counts
    return value


def event(value):
    data = value if isinstance(value, str) else json.dumps(value)
    return f"data: {data}\n\n".encode()


class TimedStream(httpx.SyncByteStream):
    def __init__(self, schedule, clock):
        self.schedule = schedule
        self.clock = clock

    def __iter__(self):
        for timestamp, data in self.schedule:
            self.clock[0] = timestamp
            if isinstance(data, BaseException):
                raise data
            yield data


class BenchmarkClientTests(unittest.TestCase):
    def request(self, schedule, status=200, content_type="text/event-stream"):
        clock = [0.0]
        transport = httpx.MockTransport(lambda request: httpx.Response(
            status,
            headers={"content-type": content_type, "x-request-id": "request-1"},
            stream=TimedStream(schedule, clock),
        ))
        with httpx.Client(transport=transport) as client, patch(
            "benchmarks.client.perf_counter", side_effect=lambda: clock[0]
        ):
            return run_request(client, "http://benchmark.invalid", WORKLOAD, CASE, "measured", 1)

    def test_first_text_excludes_role_and_empty_chunks_and_usage_is_not_chunk_count(self):
        row = self.request([
            (0.1, event(chunk({"role": "assistant"}))),
            (0.2, event(chunk({"content": ""}))),
            (0.3, event(chunk({"content": "Several tokens in one text chunk "}))),
            (0.7, event(chunk({"content": "你好。"}))),
            (0.9, event(chunk(finish="stop", counts=usage()))),
            (1.0, event("[DONE]")),
            # HTTP body closure can lag the application-level completion marker.
            (1.4, b""),
        ])
        self.assertEqual(row["status"], "success")
        self.assertAlmostEqual(row["first_text_ms"], 300)
        self.assertAlmostEqual(row["e2e_ms"], 1000)
        self.assertEqual((row["prompt_tokens"], row["completion_tokens"], row["total_tokens"]), (12, 5, 17))
        self.assertEqual(row["finish_reason"], "stop")
        self.assertNotIn("Several tokens", json.dumps(row))

    def test_separate_final_usage_and_fragmented_multiline_sse(self):
        counts = usage(completion=8)
        counts["completion_tokens_details"] = {"reasoning_tokens": 0}
        frames = (
            b": keepalive\r\n\r\n"
            b"event: message\r\ndata: {\r\ndata: \"choices\": "
            b"[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\r\n\r\n"
            + event(chunk({"content": "Text"}))
            + event(chunk(finish="length"))
            + event({"choices": [], "usage": counts})
            + event("[DONE]")
        )
        row = self.request([(0.1, frames[:23]), (0.4, frames[23:127]), (0.8, frames[127:])])
        self.assertEqual(row["status"], "success")
        self.assertEqual((row["completion_tokens"], row["finish_reason"]), (8, "length"))

    def test_role_only_zero_completion_has_no_first_text(self):
        row = self.request([
            (0.1, event(chunk({"role": "assistant", "content": None}))),
            (0.2, event(chunk(finish="stop", counts=usage(completion=0)))),
            (0.3, event("[DONE]")),
        ])
        self.assertEqual(row["status"], "success")
        self.assertIsNone(row["first_text_ms"])
        self.assertEqual(row["completion_tokens"], 0)

    def test_truncated_or_incomplete_success_never_fabricates_counts(self):
        terminal = event(chunk(finish="stop", counts=usage()))
        for frames, code in (
            ([event(chunk({"content": "Partial"}))], "truncated_stream"),
            ([terminal], "truncated_stream"),
            ([terminal, b"data: [DONE]\n"], "truncated_stream"),
            ([event("[DONE]")], "missing_terminal"),
            ([event(chunk(finish="stop")), event("[DONE]")], "missing_usage"),
        ):
            with self.subTest(code=code, frames=frames):
                row = self.request(list(enumerate(frames, 1)))
                self.assertEqual((row["status"], row["error_code"]), ("error", code))
                self.assertIsNone(row["completion_tokens"])
                self.assertIsNone(row["finish_reason"])

    def test_usage_must_be_integer_consistent_and_match_the_workload(self):
        invalid = [
            {**usage(), "completion_tokens": True},
            {**usage(), "completion_tokens": -1},
            {**usage(), "completion_tokens": 5.0},
            {**usage(), "total_tokens": None},
            usage(total=99),
            usage(prompt=13),
            usage(completion=9),
            {"prompt_tokens": 12, "completion_tokens": 5},
        ]
        for counts in invalid:
            with self.subTest(counts=counts):
                row = self.request([
                    (0.1, event(chunk(finish="stop", counts=counts))),
                    (0.2, event("[DONE]")),
                ])
                self.assertEqual((row["status"], row["error_code"]), ("error", "invalid_usage"))
                self.assertIsNone(row["completion_tokens"])

    def test_length_requires_cap_but_stop_can_occur_on_cap(self):
        for finish, count, status in (("length", 7, "error"), ("stop", 8, "success")):
            with self.subTest(finish=finish):
                row = self.request([
                    (0.1, event(chunk(finish=finish, counts=usage(completion=count)))),
                    (0.2, event("[DONE]")),
                ])
                self.assertEqual(row["status"], status)

    def test_content_cannot_claim_zero_generated_tokens(self):
        row = self.request([
            (0.1, event(chunk({"content": "Text"}))),
            (0.2, event(chunk(finish="stop", counts=usage(completion=0)))),
            (0.3, event("[DONE]")),
        ])
        self.assertEqual((row["status"], row["error_code"]), ("error", "invalid_usage"))

    def test_malformed_or_conflicting_terminals_are_errors(self):
        terminal = event(chunk(finish="stop", counts=usage()))
        for frames in (
            [b"data: not-json\n\n"],
            [terminal, terminal],
            [terminal, event("[DONE]")],
            [terminal, event("[DONE]"), event({"error": {"code": "generation_failed"}})],
            [terminal, event({"choices": [], "usage": usage()})],
            [terminal, event(chunk({"content": "Late text"}))],
            [event(chunk({"content": "Text"}, counts=usage()))],
            [event(chunk(finish="tool_calls", counts=usage()))],
        ):
            with self.subTest(frames=frames):
                row = self.request(list(enumerate([*frames, event("[DONE]")], 1)))
                self.assertEqual(row["status"], "error")
                self.assertIsNone(row["completion_tokens"])

    def test_stream_error_after_text_does_not_turn_done_into_success(self):
        row = self.request([
            (0.1, event(chunk({"content": "Private output"}))),
            (0.2, event({"error": {"code": "generation_failed", "message": "Private failure"}})),
            (0.3, event("[DONE]")),
        ])
        self.assertEqual((row["status"], row["error_code"]), ("error", "generation_failed"))
        self.assertAlmostEqual(row["first_text_ms"], 100)
        self.assertIsNone(row["completion_tokens"])
        self.assertNotIn("Private", json.dumps(row))

    def test_http_error_timeout_disconnect_and_cancellation_are_accounted(self):
        for failure, status, code in (
            (httpx.ReadTimeout("Private timeout"), "error", "timeout"),
            (httpx.RemoteProtocolError("Private disconnect"), "error", "transport_error"),
            (KeyboardInterrupt(), "canceled", "interrupted"),
        ):
            with self.subTest(code=code):
                row = self.request([(0.1, event(chunk({"content": "Private output"}))), (0.4, failure)])
                self.assertEqual((row["status"], row["error_code"]), (status, code))
                self.assertAlmostEqual(row["e2e_ms"], 400)
                self.assertIsNone(row["completion_tokens"])
                self.assertNotIn("Private", json.dumps(row))
        row = self.request([
            (0.4, json.dumps({"error": {"code": "server_busy", "message": "Private error"}}).encode())
        ], status=429, content_type="application/json")
        self.assertEqual((row["http_status"], row["status"], row["error_code"]), (429, "error", "server_busy"))
        self.assertAlmostEqual(row["e2e_ms"], 400)
        self.assertNotIn("Private", json.dumps(row))


if __name__ == "__main__":
    unittest.main()
