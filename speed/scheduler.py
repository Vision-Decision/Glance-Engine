"""FIFO admission and deadline-bounded batching for a single device owner."""

import asyncio
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import monotonic


class Overloaded(Exception):
    pass


class Unavailable(Exception):
    pass


@dataclass
class Pending:
    request: object
    future: asyncio.Future
    arrived: float


class Scheduler:
    def __init__(self, worker, max_tokens=8192, wait_ms=0.5, queue_limit=256, max_batch=64):
        if min(max_tokens, queue_limit, max_batch) < 1 or wait_ms < 0:
            raise ValueError("invalid scheduler limits")
        self.worker = worker
        self.max_tokens, self.window = max_tokens, wait_ms / 1000
        self.queue_limit, self.max_batch = queue_limit, max_batch
        self.pending = deque()
        self.changed = asyncio.Event()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="glance-device")
        self.ready = False
        self.task = None
        self.counts, self.batches, self.latencies = Counter(), Counter(), Counter()

    def start(self):
        if self.task is not None:
            raise RuntimeError("scheduler already started")
        self.ready = True
        self.task = asyncio.create_task(self._run())

    async def submit(self, request, timeout=120):
        if not self.ready:
            raise Unavailable("server is draining")
        if len(self.pending) >= self.queue_limit:
            self.counts["rejected"] += 1
            raise Overloaded("request queue is full")
        job = Pending(request, asyncio.get_running_loop().create_future(), monotonic())
        self.pending.append(job)
        self.changed.set()
        try:
            return await asyncio.wait_for(job.future, timeout)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            self.counts["abandoned"] += 1
            try:
                self.pending.remove(job)
            except ValueError:
                pass
            raise

    async def close(self):
        self.ready = False
        self.changed.set()
        if self.task is not None:
            await self.task
        self.pool.shutdown(wait=True)

    def metrics(self):
        return {
            "queue_depth": len(self.pending),
            "queue_limit": self.queue_limit,
            "requests": dict(self.counts),
            "batch_sizes": dict(self.batches),
            "latency_ms_histogram": dict(self.latencies),
        }

    async def _run(self):
        while self.ready or self.pending:
            if not self.pending:
                self.changed.clear()
                await self.changed.wait()
                continue
            deadline = self.pending[0].arrived + self.window
            batch, tokens = [], 0
            while len(batch) < self.max_batch:
                if self.pending:
                    job = self.pending[0]
                    cost = job.request.tokens
                    if batch and (
                        cost + tokens > self.max_tokens or job.request.batch_kind != batch[0].request.batch_kind
                    ):
                        break
                    self.pending.popleft()
                    if not job.future.cancelled():
                        batch.append(job)
                        tokens += cost
                    if tokens >= self.max_tokens:
                        break
                elif len(batch) == 1 and getattr(batch[0].request, "prefer_immediate", False):
                    # Other submissions already ready on this event-loop turn joined above.
                    break
                elif batch and self.ready and monotonic() < deadline:
                    self.changed.clear()
                    try:
                        await asyncio.wait_for(self.changed.wait(), deadline - monotonic())
                    except asyncio.TimeoutError:
                        break
                else:
                    break
            await self._execute(batch)

    async def _execute(self, batch):
        batch = [j for j in batch if not j.future.cancelled()]
        if not batch:
            return
        loop = asyncio.get_running_loop()
        self.batches[len(batch)] += 1
        try:
            result = await loop.run_in_executor(self.pool, self.worker, [j.request for j in batch])
            if len(result) != len(batch):
                raise RuntimeError("incorrect result count from device worker")
        # Device failures must resolve every waiter and leave the admission loop alive.
        except Exception as exc:  # noqa: BLE001
            oom = self.worker.is_oom(exc)
            split_invalid = isinstance(exc, (ValueError, KeyError)) and len(batch) > 1
            if not oom and not split_invalid:
                for job in batch:
                    if not job.future.done():
                        job.future.set_exception(exc)
                self.counts["failed"] += len(batch)
                return
        else:
            for job, value in zip(batch, result):
                if not job.future.done():
                    job.future.set_result(value)
                    self.counts["completed"] += 1
                    elapsed = (monotonic() - job.arrived) * 1000
                    upper = next(
                        (v for v in (10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000) if elapsed <= v), "+Inf"
                    )
                    self.latencies[str(upper)] += 1
            return
        # Release the failing forward's traceback before reclaiming GPU memory.
        if oom:
            self.counts["oom"] += 1
            await loop.run_in_executor(self.pool, self.worker.recover)
        if len(batch) == 1:
            if not batch[0].future.done():
                batch[0].future.set_exception(Unavailable("insufficient device memory"))
        else:
            middle = len(batch) // 2
            await self._execute(batch[:middle])
            await self._execute(batch[middle:])
