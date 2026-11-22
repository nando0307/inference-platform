import asyncio
import signal
import unittest
from pathlib import Path
from unittest.mock import patch

from src import worker_process

ROOT = Path(__file__).resolve().parents[1]


class WorkerProcessTests(unittest.IsolatedAsyncioTestCase):
    async def test_stop_signal_during_startup_still_stops_the_engine(self):
        loop = asyncio.get_running_loop()
        handlers = {}
        started = asyncio.Event()
        finish_start = asyncio.Event()
        engine = None

        class SlowEngine:
            def __init__(self, worker=None):
                nonlocal engine
                self.worker = worker
                self.handlers_at_start = None
                self.stopped = False
                engine = self

            async def start(self, model):
                self.handlers_at_start = dict(handlers)
                started.set()
                await finish_start.wait()

            async def stop(self):
                self.stopped = True

        def capture_handler(loop_self, signum, callback, *args):
            handlers[signum] = callback

        with patch.object(type(loop), "add_signal_handler", capture_handler), \
                patch.object(worker_process, "VLLMEngine", SlowEngine):
            serving = asyncio.create_task(
                worker_process.serve(ROOT / "configs/serving-a.json", "small-0")
            )
            await asyncio.wait_for(started.wait(), 2)
            self.assertIn(signal.SIGTERM, engine.handlers_at_start)
            self.assertIn(signal.SIGINT, engine.handlers_at_start)
            engine.handlers_at_start[signal.SIGTERM]()
            finish_start.set()
            await asyncio.wait_for(serving, 2)

        self.assertTrue(engine.stopped)
        self.assertIn(signal.SIGTERM, handlers)


if __name__ == "__main__":
    unittest.main()
