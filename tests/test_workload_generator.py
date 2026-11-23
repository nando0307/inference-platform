import asyncio
import json
import unittest
from pathlib import Path
from random import Random

import httpx

from benchmarks.workload_generator.arrivals import arrivals
from benchmarks.workload_generator.config import load_scenario, validate_scenario
from benchmarks.workload_generator.executor import _request
from benchmarks.workload_generator.prompts import PromptProvider
from benchmarks.workload_generator.recorder import summarize_results
from benchmarks.workload_generator.trace import generate_trace, metadata, read_trace, write_trace

ROOT = Path(__file__).resolve().parents[1]
S = "Qwen/Qwen2.5-0.5B-Instruct"
L = "Qwen/Qwen3-1.7B"


class WorkloadGeneratorTests(unittest.TestCase):
    def test_all_scenario_files_validate(self):
        paths = sorted((ROOT / "configs/workloads").glob("*.json"))
        self.assertEqual(len(paths), 15)
        for path in paths:
            with self.subTest(path=path.name):
                scenario = load_scenario(path)
                self.assertEqual(scenario.phases[0].start_s, 0)
                self.assertEqual(scenario.phases[-1].end_s, scenario.duration_s)

    def test_fifteen_minute_phase_boundaries_and_zero_traffic(self):
        scenario = load_scenario(ROOT / "configs/workloads/mixed_15min.json")
        self.assertEqual([(phase.start_s, phase.end_s) for phase in scenario.phases],
                         [(0, 180), (180, 360), (360, 540), (540, 720), (720, 900)])
        records = generate_trace(scenario)
        self.assertTrue(records)
        self.assertLessEqual(max(row["scheduled_offset_s"] for row in records), 720)
        self.assertEqual(scenario.duration_s, 900)

    def test_independent_seed_streams_keep_arrivals_stable_when_prompt_changes(self):
        scenario = load_scenario(ROOT / "configs/workloads/balanced.json")
        first = generate_trace(scenario, prompt_provider=PromptProvider())
        class DifferentPrompts(PromptProvider):
            def prompt(self, model, profile, index):
                messages, count = super().prompt(model, profile, index)
                return [{"role": "user", "content": "different"}], count
        second = generate_trace(scenario, prompt_provider=DifferentPrompts())
        self.assertEqual([row["scheduled_offset_s"] for row in first],
                         [row["scheduled_offset_s"] for row in second])
        self.assertEqual([row["model"] for row in first], [row["model"] for row in second])
        self.assertNotEqual([row["messages"] for row in first], [row["messages"] for row in second])

    def test_same_seed_produces_same_trace(self):
        scenario = load_scenario(ROOT / "configs/workloads/mixed_15min.json")
        self.assertEqual(generate_trace(scenario), generate_trace(scenario))

    def test_trace_metadata_is_separate_and_round_trips(self):
        scenario = load_scenario(ROOT / "configs/workloads/balanced.json")
        records = generate_trace(scenario)
        path = ROOT / "tests/.tmp-workload-trace.jsonl"
        try:
            write_trace(records, metadata(scenario, records), path)
            self.assertEqual(read_trace(path), records)
            saved = json.loads(path.with_suffix(".meta.json").read_text())
            self.assertEqual(saved["trace_schema_version"], 1)
            self.assertEqual(saved["request_count"], len(records))
        finally:
            path.unlink(missing_ok=True)
            path.with_suffix(".meta.json").unlink(missing_ok=True)

    def test_arrival_processes_are_seeded(self):
        self.assertEqual(list(arrivals(0, 10, 1, "poisson", Random(7))),
                         list(arrivals(0, 10, 1, "poisson", Random(7))))
        self.assertNotEqual(list(arrivals(0, 10, 1, "poisson", Random(7))),
                            list(arrivals(0, 10, 1, "poisson", Random(8))))

    def test_accuracy_summary_flags_client_limits_and_percentiles(self):
        results = [
            {"status": "success", "actual_submit_offset_s": 0.1, "scheduling_delay_ms": 1},
            {"status": "not_submitted", "actual_submit_offset_s": None, "scheduling_delay_ms": None},
        ]
        summary = summarize_results(results, intended_count=2)
        self.assertEqual(summary["scheduled_requests"], 2)
        self.assertEqual(summary["submitted_requests"], 1)
        self.assertTrue(summary["limited_by_client"])
        self.assertEqual(summary["scheduling_delay_ms"]["p99"], 1)


class ExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_request_executor_parses_existing_sse_contract(self):
        async def handler(request):
            body = json.loads(request.content)
            model = body["model"]
            common = {"id": "chatcmpl-test", "object": "chat.completion.chunk", "model": model}
            content = (
                f"data: {json.dumps({**common, 'choices': [{'index': 0, 'delta': {'content': 'ok'}, 'finish_reason': None}]})}\n\n"
                f"data: {json.dumps({**common, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'length'}]})}\n\n"
                f"data: {json.dumps({**common, 'choices': [], 'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}})}\n\n"
                "data: [DONE]\n\n"
            )
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=content)

        record = {
            "request_id": "request-1", "scenario_id": "test", "phase_id": "phase",
            "scheduled_offset_s": 0, "model": S, "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1, "temperature": 0, "top_p": 1, "stream": True,
            "request_timeout_s": 2,
        }
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False) as client:
            result = await _request(client, "http://gateway", record, 0)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["completion_tokens"], 1)
        self.assertIsNotNone(result["first_text_ms"])


    async def test_stream_requires_done_and_matching_model(self):
        async def handler(request):
            body = json.loads(request.content)
            wrong = {"id": "x", "model": "other", "choices": [{"delta": {"content": "x"}}]}
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=(f"data: {json.dumps(wrong)}\n\n").encode())

        record = {"request_id": "request-2", "scenario_id": "test", "phase_id": "phase",
                  "scheduled_offset_s": 0, "model": S, "messages": [], "max_tokens": 1,
                  "temperature": 0, "top_p": 1, "stream": True, "request_timeout_s": 2}
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False) as client:
            result = await _request(client, "http://gateway", record, 0)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_code"], "model_mismatch")

        async def incomplete(request):
            chunk = {"id": "x", "model": S, "choices": [{"delta": {"content": "x"}}]}
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=(f"data: {json.dumps(chunk)}\n\n").encode())

        async with httpx.AsyncClient(transport=httpx.MockTransport(incomplete), trust_env=False) as client:
            result = await _request(client, "http://gateway", record, 0)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_code"], "missing_terminal_marker")

if __name__ == "__main__":
    unittest.main()
