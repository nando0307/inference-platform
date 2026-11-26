import asyncio
import signal
import unittest
from pathlib import Path
from unittest.mock import patch

from src import worker_process

ROOT = Path(__file__).resolve().parents[1]


class WorkerProcessTests(unittest.IsolatedAsyncioTestCase):
    async def test_crashed_startup_cleans_once_without_reporting_ready(self):
        engine = None
        output = []

        class CrashedEngine:
            def __init__(self, worker=None):
                nonlocal engine
                engine = self
                self.stops = 0

            async def start(self, model):
                raise RuntimeError("partial startup")

            async def stop(self):
                self.stops += 1

        loop = asyncio.get_running_loop()
        with patch.object(type(loop), "add_signal_handler"), \
                patch.object(worker_process, "VLLMEngine", CrashedEngine), \
                patch("builtins.print", lambda *args, **kwargs: output.append(args[0])):
            with self.assertRaisesRegex(RuntimeError, "partial startup"):
                await worker_process.serve(ROOT / "configs/serving-a.json", "small-0")
        self.assertEqual(engine.stops, 1)
        self.assertEqual(output, [])

    async def test_stop_signal_after_ready_cleans_once(self):
        handlers = {}
        engine = None
        output = []

        class RunningEngine:
            def __init__(self, worker=None):
                nonlocal engine
                engine = self
                self.stops = 0

            async def start(self, model):
                pass

            async def stop(self):
                self.stops += 1

        loop = asyncio.get_running_loop()
        with patch.object(type(loop), "add_signal_handler",
                          lambda loop, signum, callback: handlers.__setitem__(signum, callback)), \
                patch.object(worker_process, "VLLMEngine", RunningEngine), \
                patch("builtins.print", lambda *args, **kwargs: output.append(args[0])):
            serving = asyncio.create_task(worker_process.serve(ROOT / "configs/serving-a.json", "small-0"))
            async with asyncio.timeout(1):
                while not output:
                    await asyncio.sleep(0)
            handlers[signal.SIGTERM]()
            await asyncio.wait_for(serving, 1)
        self.assertEqual(engine.stops, 1)
        self.assertTrue(output)

    async def test_stop_signal_interrupts_a_blocked_startup_and_still_stops_the_engine(self):
        loop = asyncio.get_running_loop()
        handlers = {}
        started = asyncio.Event()
        finish_start = asyncio.Event()
        engine = None
        printed = []

        class SlowEngine:
            def __init__(self, worker=None):
                nonlocal engine
                engine = self
                self.worker = worker
                self.handlers_at_start = None
                self.stopped = False
                self.start_cancelled = False

            async def start(self, model):
                self.handlers_at_start = dict(handlers)
                started.set()
                try:
                    await finish_start.wait()
                except asyncio.CancelledError:
                    self.start_cancelled = True
                    raise

            async def stop(self):
                self.stopped = True

        def capture_handler(loop_self, signum, callback, *args):
            handlers[signum] = callback

        with patch.object(type(loop), "add_signal_handler", capture_handler), \
                patch.object(worker_process, "VLLMEngine", SlowEngine), \
                patch("builtins.print", lambda *args, **kwargs: printed.append(args[0])):
            serving = asyncio.create_task(
                worker_process.serve(ROOT / "configs/serving-a.json", "small-0")
            )
            await asyncio.wait_for(started.wait(), 2)
            self.assertIn(signal.SIGTERM, engine.handlers_at_start)
            self.assertIn(signal.SIGINT, engine.handlers_at_start)
            engine.handlers_at_start[signal.SIGTERM]()
            await asyncio.wait_for(serving, 2)

        self.assertTrue(engine.stopped)
        self.assertTrue(engine.start_cancelled)
        self.assertEqual(printed, [])
        self.assertIn(signal.SIGTERM, handlers)


if __name__ == "__main__":
    unittest.main()
