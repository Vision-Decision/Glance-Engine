"""Transport fault tests. No GPU or model weights required."""

import asyncio
import threading
import unittest

from aiohttp.test_utils import TestClient, TestServer

from speed.decision_worker import Decision, DecisionWorker, PrefixCache
from speed.server import application


class BlockingWorker:
    max_tokens = 100

    def __init__(self):
        self.cache = PrefixCache(0)
        self.entered, self.release = threading.Event(), threading.Event()

    def __call__(self, jobs):
        self.entered.set()
        self.release.wait(3)
        return [[[]] for _ in jobs]

    is_oom = staticmethod(lambda exc: False)


class ServerTests(unittest.IsolatedAsyncioTestCase):
    async def test_shutdown_drains_a_request_still_being_parsed(self):
        from unittest.mock import patch

        from speed.server import parse

        worker = BlockingWorker()
        worker.release.set()
        started, resume = threading.Event(), threading.Event()

        def slow_parse(path, body):
            started.set()
            resume.wait(3)
            return parse(path, body)

        body = {"image_ids": ["aa" * 32], "questions": [{"type": "noul", "instr": "visible?"}]}
        app = application(worker, wait_ms=0)
        async with TestClient(TestServer(app)) as client:
            with patch("speed.server.parse", slow_parse):
                pending = asyncio.create_task(client.post("/v1/decisions", json=body))
                self.assertTrue(await asyncio.to_thread(started.wait, 2))
                shutdown = asyncio.create_task(app.shutdown())
                await asyncio.sleep(0)
                try:
                    self.assertEqual((await client.get("/ready")).status, 503)
                    self.assertEqual((await client.post("/v1/decisions", json=body)).status, 503)
                finally:
                    resume.set()
                self.assertEqual((await pending).status, 200)
                await shutdown

    async def test_admission_timeout_disconnect_and_recovery(self):
        worker = BlockingWorker()
        server = TestServer(application(worker, timeout=0.5, queue_limit=1, wait_ms=0), handler_cancellation=True)
        body = {"image_ids": ["aa" * 32], "questions": [{"type": "noul", "instr": "visible?"}]}
        async with TestClient(server) as client:

            async def submit():
                return await client.post("/v1/decisions", json=body)

            first = asyncio.create_task(submit())
            self.assertTrue(await asyncio.to_thread(worker.entered.wait, 2))
            queued = asyncio.create_task(submit())
            for _ in range(20):
                metrics = await (await client.get("/metrics")).json()
                if metrics["queue_depth"] == 1:
                    break
                await asyncio.sleep(0.005)
            full = await submit()
            self.assertEqual(full.status, 429)
            queued.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await queued
            self.assertEqual((await first).status, 504)
            worker.release.set()
            self.assertEqual((await submit()).status, 200)
            self.assertTrue((await (await client.get("/ready")).json())["ok"])
            malformed = await client.post("/v1/decisions", data=b"{")
            self.assertEqual(malformed.status, 400)
            self.assertEqual((await malformed.json())["error"]["type"], "invalid_request")

    def test_one_large_request_splits_at_question_boundaries(self):
        from types import SimpleNamespace

        from PIL import Image

        tok = SimpleNamespace(user_open=[1], vs=2, ve=3, box_block=lambda text: [4, 5, 6])
        engine = SimpleNamespace(
            tok=tok,
            proc=SimpleNamespace(image_processor=SimpleNamespace(patch_size=16)),
            fv=None,
            frame_tower=None,
            _plen=lambda item: 4,
        )
        worker = DecisionWorker(engine, max_tokens=10)
        steps = []

        def record(units, hit, results):
            steps.append(len(units))
            for u in units:
                results[u.request_index][u.image_index].extend(q["text"] for q in u.item["qs"])

        worker._step = record
        req = Decision([Image.new("RGB", (32, 32))], [{"type": "box", "text": str(i)} for i in range(5)])
        self.assertEqual(worker([req]), [[["0", "1", "2", "3", "4"]]])
        self.assertEqual(steps, [1, 1, 1])

    def test_resent_pixels_share_keys_across_decodes_and_detect_edits(self):
        from PIL import Image
        from speed.decision_worker import image_key

        cache = PrefixCache(4096)
        im = Image.new("RGB", (12, 8), color=(20, 10, 2))
        original = cache.key_for_image(im)
        self.assertEqual(original, image_key(im))
        self.assertEqual(original, cache.key_for_image(im.copy()))
        im.putpixel((0, 0), (20, 10, 3))
        self.assertNotEqual(original, cache.key_for_image(im))
        self.assertEqual(image_key(im), cache.key_for_image(im))
        transposed = Image.frombytes("RGB", (8, 12), im.tobytes())
        self.assertNotEqual(image_key(im), cache.key_for_image(transposed))
        cache.clear()
        self.assertIsNone(cache.last_rgb)

    def test_cache_bytes_and_recency(self):
        c = PrefixCache(4)
        self.assertTrue(c.put("a", 1, 2))
        self.assertTrue(c.put("b", 2, 2))
        self.assertEqual(c.get("a"), 1)
        self.assertFalse(c.put("a", 9, 5))
        c.put("c", 3, 2)
        self.assertIsNone(c.get("b"))
        self.assertEqual(c.get("a"), 1)
        self.assertEqual(c.used, 4)
        c.clear()
        self.assertEqual(c.used, 0)


if __name__ == "__main__":
    unittest.main()
