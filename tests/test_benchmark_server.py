import json
import logging
import unittest

from benchmarks.server import SchedulingMetrics


class SchedulingMetricsTests(unittest.TestCase):
    def test_only_terminal_scheduling_allowlisted_metadata_reaches_metrics(self):
        records = []
        handler = SchedulingMetrics(records.append)
        logger = logging.Logger("isolated-benchmark-metrics")
        logger.addHandler(handler)
        try:
            logger.info("PRIVATE raw exception text")
            logger.info(json.dumps({"event": "producer_finished", "text": "PRIVATE output"}))
            logger.info(json.dumps({
                "event": "scheduling_finished", "request_id": "request-one", "state": "canceled",
                "queue_wait_ms": 0, "cancellation_delay_ms": 25,
                "cancellation_reason": "client_disconnected", "messages": "PRIVATE prompt",
                "generated_ids": [12345], "text": "PRIVATE output",
            }))
            logger.info(json.dumps({
                "event": "scheduling_finished", "request_id": "request-two", "state": "canceled",
                "queue_wait_ms": None, "cancellation_delay_ms": None,
                "cancellation_reason": "shutdown",
            }))
        finally:
            logger.removeHandler(handler)
            handler.close()
        self.assertEqual(records, [
            dict(type="scheduling", request_id="request-one", state="canceled", queue_wait_ms=0,
                 cancellation_delay_ms=25, cancellation_reason="client_disconnected"),
            dict(type="scheduling", request_id="request-two", state="canceled", queue_wait_ms=None,
                 cancellation_delay_ms=None, cancellation_reason="shutdown"),
        ])
        self.assertNotIn("PRIVATE", json.dumps(records))
        self.assertNotIn("generated_ids", json.dumps(records))


if __name__ == "__main__":
    unittest.main()
