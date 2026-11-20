"""Model-free checks of queue deadlines, admission, cancellation and split/retry."""

import asyncio
import threading
import unittest
from dataclasses import dataclass

from speed.scheduler import Overloaded, Scheduler, Unavailable


@dataclass
class Job:
    value: int
    tokens: int = 1
    batch_kind = "image"


class Worker:
    def __init__(self):
        self.calls = []
        self.entered, self.release = threading.Event(), threading.Event()
        self.oom = 100

    def __call__(self, jobs):
        self.entered.set()
        self.release.wait(3)
        self.calls.append([j.value for j in jobs])
        if len(jobs) > self.oom:
            raise MemoryError()
        return [j.value for j in jobs]

    is_oom = staticmethod(lambda exc: isinstance(exc, MemoryError))

    def recover(self):
        pass


class Tests(unittest.IsolatedAsyncioTestCase):
    async def test_split_and_route(self):
        w = Worker()
        w.release.set()
        w.oom = 2
        s = Scheduler(w, max_tokens=5, wait_ms=1)
        s.start()
        try:
            result = await asyncio.gather(*(s.submit(Job(i)) for i in range(12)))
            self.assertEqual(result, list(range(12)))
            self.assertEqual(max(map(len, w.calls)), 5)
            self.assertGreater(s.counts["oom"], 0)
            w.oom = 0
            with self.assertRaises(Unavailable):
                await s.submit(Job(14))
        finally:
            await s.close()
        with self.assertRaises(Unavailable):
            await s.submit(Job(13))

    async def test_backpressure_deadline_disconnect_and_drain(self):
        w = Worker()
        s = Scheduler(w, queue_limit=1, max_batch=1)
        s.start()
        first = asyncio.create_task(s.submit(Job(1)))
        await asyncio.to_thread(w.entered.wait, 2)
        expired = asyncio.create_task(s.submit(Job(2), timeout=0.02))
        await asyncio.sleep(0)
        with self.assertRaises(Overloaded):
            await s.submit(Job(3))
        with self.assertRaises(asyncio.TimeoutError):
            await expired
        cancelled = asyncio.create_task(s.submit(Job(4)))
        await asyncio.sleep(0)
        cancelled.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await cancelled
        queued = asyncio.create_task(s.submit(Job(5)))
        await asyncio.sleep(0)
        drain = asyncio.create_task(s.close())
        await asyncio.sleep(0)
        with self.assertRaises(Unavailable):
            await s.submit(Job(6))
        w.release.set()
        self.assertEqual(await first, 1)
        self.assertEqual(await queued, 5)
        await drain
        self.assertEqual(w.calls, [[1], [5]])


if __name__ == "__main__":
    unittest.main()
