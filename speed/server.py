"""Async Glance HTTP server. Run python -m speed.server --ckpt PATH.

POST /v1/decisions with {"image": "<base64>", "questions": [{"type":"box","text":"dog"}]}.
An images array broadcasts all questions over each frame. cache=true retains an image
prefix for subsequent requests with the same pixels. Results are image-major, then
question-major. /v1/chat/completions accepts one multimodal user message with data
URLs and a yes/no question. /encode, /scan and /score_clips implement the vdm API.
"""

import argparse
import asyncio
import base64
import json
import logging
import math
import struct
import time
import uuid
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO

import numpy as np
from aiohttp import web
from dv.data.dataset import resize_dims
from PIL import Image

from scan import video_ops as VO
from speed.decision_worker import Decision, DecisionWorker, ImageRef
from speed.scheduler import Overloaded, Scheduler, Unavailable


def check(test, message):
    if not test:
        raise ValueError(message)


def string(value, name):
    check(isinstance(value, str) and 0 < len(value.strip()) <= 8192, f"invalid {name}")
    return value


def array(value, name, maximum):
    check(isinstance(value, list) and 0 < len(value) <= maximum, f"invalid {name}")
    return value


def questions(raw, video=False):
    result = []
    for q in array(raw, "questions", 64):
        check(isinstance(q, dict), "expected a question object")
        kind = q.get("type")
        check(
            kind in (("noul", "choice", "score") if video else ("box", "noul", "choice", "score")),
            "unknown question type",
        )
        key = "text" if kind == "box" else "instr"
        clean = {"type": kind, key: string(q.get(key), key)}
        if kind in ("choice", "score"):
            clean["options"] = [string(s, "option") for s in array(q.get("options"), "options", 26)]
        if q.get("block") == "short" and kind == "noul":
            clean["block"] = "short"
        result.append(clean)
    return result


class Images:
    def __init__(self):
        self.decoded_pixels = 0

    def decode(self, data):
        check(isinstance(data, str), "expected base64 image")
        if data.startswith("data:"):
            header, sep, data = data.partition(",")
            check(
                sep and header.startswith("data:image/") and header.endswith(";base64"), "invalid image data URL"
            )
        raw = base64.b64decode(data, validate=True)
        with Image.open(BytesIO(raw)) as im:
            self.decoded_pixels += im.width * im.height
            check(
                im.width * im.height <= 32_000_000 and self.decoded_pixels <= 96_000_000,
                "decoded images too large",
            )
            return im.convert("RGB")


@dataclass
class VideoRequest:
    path: str
    video: str
    clips: list
    digest: bytes
    dt: float
    times: list
    questions: list
    single: bool
    ops: list
    alpha: float
    q: float

    batch_kind = "video"
    # Video mutations must preserve FIFO order, including /encode immediately followed by /scan.
    tokens = 1 << 60


def _directed(decision, direction):
    if direction is not None:
        decision.direction = direction
    return decision


def parse(path, body):
    check(isinstance(body, dict), "expected a JSON object")
    decoder = Images()
    if path == "/v1/chat/completions":
        check(not body.get("stream"), "streaming is unavailable for decision readouts")
        messages = array(body.get("messages"), "messages", 1)
        msg = messages[0]
        check(isinstance(msg, dict) and msg.get("role") == "user", "expected one user message")
        content = array(msg.get("content"), "content", 257)
        check(
            all(isinstance(c, dict) and c.get("type") in ("image_url", "text") for c in content),
            "invalid content part",
        )
        images = [c["image_url"]["url"] for c in content if c["type"] == "image_url"]
        instruction = " ".join(string(c.get("text"), "text") for c in content if c["type"] == "text")
        body = {
            **body,
            "images": images,
            "questions": body.get("questions", [{"type": "noul", "instr": instruction}]),
        }
    if path.startswith("/v1/"):
        qs = questions(body.get("questions"))
        direction = body.get("direction")  # Glance-X reading direction; v1 models read image first
        check(direction in (None, "image_first", "text_first", "joint"), "invalid direction")
        if "image_ids" in body:
            check("image" not in body and "images" not in body, "send image_ids or image data")
            ids = array(body["image_ids"], "image_ids", 256)
            check(all(isinstance(s, str) and len(s) == 64 for s in ids), "invalid image id")
            return _directed(Decision([ImageRef(bytes.fromhex(s)) for s in ids], qs), direction)
        encoded = array(body.get("images", [body.get("image")]), "images", 256)
        max_vis = body.get("max_vis", 640)
        check(type(max_vis) is int and 16 <= max_vis <= 4096, "max_vis must be between 16 and 4096")
        cache = body.get("cache", False)
        check(type(cache) is bool, "cache must be a boolean")
        images = []
        for entry in encoded:
            im = decoder.decode(entry)
            size = resize_dims(*im.size, max_vis, 2.0)
            images.append(im if size == im.size else im.resize(size, Image.Resampling.BICUBIC))
        return _directed(Decision(images, qs, cache and direction in (None, "image_first")), direction)
    video = string(body.get("video"), "video") if path != "/score_clips" else ""
    dt = float(body.get("dt", 0.5))
    check(math.isfinite(dt) and dt > 0, "dt must be positive")
    times, clips, digest = [], [], sha256(struct.pack("!d", dt))
    if path != "/scan":
        for group in array(body.get("clips"), "clips", 256):
            frames = [decoder.decode(f) for f in array(group, "frames", 128)]
            w, h = (min(im.size[d] for im in frames) for d in (0, 1))
            clip = np.stack([np.asarray(im)[:h, :w] for im in frames])
            digest.update(struct.pack("!III", len(clip), h, w))
            digest.update(clip)
            clips.append(clip)
        times = body.get("times")
        if times is None or times == []:
            times = [[float(i), i + dt * (len(c) - 1)] for i, c in enumerate(clips)]
        check(isinstance(times, list) and len(times) == len(clips), "one time interval required per clip")
        for interval in times:
            check(
                isinstance(interval, list)
                and len(interval) == 2
                and all(type(t) in (int, float) and math.isfinite(t) for t in interval)
                and interval[0] <= interval[1],
                "invalid time interval",
            )
    qs, single = [], False
    if path == "/score_clips":
        single = body.get("questions") is None or body["questions"] == []
        if single:
            kind = body.get("kind", "yesno")
            check(kind in ("yesno", "choice"), "invalid score kind")
            qs = questions(
                [
                    {
                        "type": "noul" if kind == "yesno" else "choice",
                        "instr": body.get("question"),
                        "options": body.get("options"),
                    }
                ],
                video=True,
            )
        else:
            qs = questions(body["questions"], video=True)
    elif path == "/scan":
        qs = [{"type": "noul", "instr": string(body.get("query"), "query")}]
    ops = body.get("ops", ["curve", "ever", "when"])
    check(
        isinstance(ops, list) and all(op in ("curve", "ever", "when", "count") for op in ops),
        "invalid scan operation",
    )
    alpha, q = float(body.get("alpha", 0.1)), float(body.get("q", 0.1))
    check(0 < alpha < 1 and 0 < q < 1, "alpha and q must be in (0,1)")
    return VideoRequest(path, video, clips, digest.digest(), dt, times, qs, single, ops, alpha, q)


class VideoService:
    def __init__(self, worker, calib=None, short=False):
        self.worker, self.calib, self.short = worker, calib, short
        self.stats = {"clips_encoded": 0, "queries": 0, "gpu_sec": 0.0, "bank_hits": 0}

    def __call__(self, r):
        started = time.monotonic()
        try:
            return self._run(r)
        finally:
            self.stats["gpu_sec"] += time.monotonic() - started

    def _run(self, r):
        eng, store = self.worker.engine, self.worker.cache
        key = ("video", r.video) if r.video else ("clips", r.digest)
        old = store.get(key)
        hit = old is not None and (r.path == "/scan" or old[0] == r.digest)
        if r.path == "/scan" and not hit:
            raise KeyError("unknown video, POST /encode first")
        if hit:
            self.stats["bank_hits"] += 1
            _, bank, times = old
        else:
            eng.dt = r.dt
            largest = max((len(c) + 1) // 2 * (eng.clip_max_vis + 16) for c in r.clips)
            bank = eng.encode(r.clips, chunk=max(1, min(64, self.worker.max_tokens // largest)), readout="typed")
            self.stats["clips_encoded"] += bank.N
            times = r.times
        if r.path == "/encode" or not hit:
            size = bank.nbytes()
            stored = store.put(key, (r.digest, bank, r.times), size)
            if r.path == "/encode":
                if not stored:
                    raise ValueError("video prefix exceeds the cache budget (--cache-gib); send fewer clips")
                return {"n": bank.N, "cached": hit}
        # Bank-bound CUDA graphs belong to the bank. Eager execution keeps evictable bytes explicit.
        self.stats["queries"] += len(r.questions)
        graphs, eng.graphs = eng.graphs, False
        try:
            if r.path == "/score_clips":
                rows = [[] for _ in range(bank.N)]
                for question in r.questions:
                    logits = eng.query(bank, question)["typed_logits"]
                    for row, lg in zip(rows, logits):
                        p = lg[lg.isfinite()].softmax(-1).tolist()
                        row.append(p[1] if question["type"] == "noul" else p)
                return {"results": [r[0] for r in rows] if r.single else rows, "cached": hit}
            question = dict(r.questions[0])
            if self.short:
                question["block"] = "short"
            logits = eng.query(bank, question)["typed_logits"]
            scores = (logits[:, 1] - logits[:, 0]).numpy()
        finally:
            eng.graphs = graphs
        result = {"n_clips": bank.N}
        if "curve" in r.ops:
            result["curve"] = VO.curve(scores, times, self.calib)
        if self.calib is None:
            result["ever"] = {"ever": bool((scores > 0).any()), "method": "naive p>0.5", "guarantee": None}
        else:
            if "ever" in r.ops:
                result["ever"] = VO.ever(scores, self.calib, r.alpha)
            for op in ("when", "count"):
                if op in r.ops:
                    result[op] = getattr(VO, op)(scores, times, self.calib, r.q, r.alpha)
        return result


def application(worker, *, timeout=120, queue_limit=256, wait_ms=0.5, max_batch=64, calib=None, short=False):
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    scheduler = Scheduler(
        worker, max_tokens=worker.max_tokens, wait_ms=wait_ms, max_batch=max_batch, queue_limit=queue_limit
    )
    worker.video = VideoService(worker, calib, short)
    active = 0
    draining = False
    idle = asyncio.Event()
    idle.set()

    def error(status, code, message):
        return web.json_response({"error": {"type": code, "message": message}}, status=status)

    async def post(request):
        nonlocal active
        path = request.path.rstrip("/")
        if draining:
            return error(503, "unavailable", "server is draining")
        if active >= queue_limit + 1:
            scheduler.counts["rejected"] += 1
            return error(429, "overloaded", "request queue is full")
        active += 1
        idle.clear()
        t0 = time.monotonic()

        async def handle():
            raw = await request.read()
            job = await asyncio.to_thread(lambda: parse(path, json.loads(raw)))
            if getattr(job, "direction", None) not in (None, "image_first") and \
                    getattr(worker.engine, "family", None) != "x":
                raise ValueError("reading directions other than image_first need a Glance-X model")
            out = await scheduler.submit(job, timeout)
            sec = time.monotonic() - t0
            if path == "/v1/chat/completions":
                return {
                    "id": "chatcmpl-" + uuid.uuid4().hex,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": "glance",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": json.dumps(out)},
                        }
                    ],
                }
            if isinstance(job, Decision):
                return {
                    "object": "glance.decisions",
                    "model": "glance",
                    "results": out,
                    "sec": sec,
                    **({"image_ids": job.image_ids} if job.cache else {}),
                }
            return {**out, "sec": sec}

        try:
            return web.json_response(await asyncio.wait_for(handle(), timeout))
        except web.HTTPRequestEntityTooLarge:
            return error(413, "request_too_large", "request body exceeds limit")
        except (asyncio.TimeoutError, TimeoutError):
            return error(504, "timeout", "request deadline exceeded")
        except Overloaded as exc:
            return error(429, "overloaded", str(exc))
        except Unavailable as exc:
            return error(503, "unavailable", str(exc))
        except KeyError as exc:
            return error(
                404 if path == "/scan" or "evicted" in str(exc) else 400, "invalid_request", str(exc)
            )
        except (ValueError, TypeError, OSError, Image.DecompressionBombError, OverflowError) as exc:
            return error(400, "invalid_request", str(exc)[:240])
        except Exception:
            logging.getLogger(__name__).exception("decision request failed")
            return error(500, "internal_error", "inference failed")
        finally:
            active -= 1
            if active == 0:
                idle.set()

    async def status(request):
        path = request.path.rstrip("/")
        if path == "/metrics":
            return web.json_response({**scheduler.metrics(), "active": active, "cache_bytes": worker.cache.used})
        ready = scheduler.ready and not draining
        return web.json_response(
            {
                "ok": ready,
                "version": "glance-scheduler",
                "calibrated": calib is not None,
                "banks": sum(key[0] in ("video", "clips") for key in worker.cache.entries.copy()),
                **worker.video.stats,
            },
            status=503 if path == "/ready" and not ready else 200,
        )

    async def start(app):
        scheduler.start()

    async def stop(app):
        nonlocal draining
        draining = True
        await idle.wait()
        await scheduler.close()

    app = web.Application(client_max_size=64 << 20)
    for path in ("/health", "/ready", "/metrics"):
        app.router.add_get(path, status)
        app.router.add_get(path + "/", status)
    for path in ("/v1/decisions", "/v1/chat/completions", "/encode", "/scan", "/score_clips"):
        app.router.add_post(path, post)
        app.router.add_post(path + "/", post)
    app.on_startup.append(start)
    app.on_shutdown.append(stop)
    return app


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8002)
    p.add_argument("--graphs", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--group-vision", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--shared-prefill", choices=("packed", "padded"), default="packed")
    p.add_argument("--max-batch", type=int, default=64)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--wait-ms", type=float, default=0.5)
    p.add_argument("--queue-limit", type=int, default=256)
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--cache-gib", type=float, default=8)
    p.add_argument("--max-vis", type=int, default=96, help="clip tokens per temporal group")
    p.add_argument("--calib")
    p.add_argument("--readout", choices=("yes", "short"), default="yes")
    a = p.parse_args()
    worker = DecisionWorker.load(
        a.ckpt,
        graphs=a.graphs,
        max_tokens=a.max_tokens,
        cache_bytes=int(a.cache_gib * 2**30),
        packed_shared=a.shared_prefill == "packed",
        group_vision=a.group_vision,
    )
    worker.engine.clip_max_vis = a.max_vis
    calib = None
    if a.calib:
        with open(a.calib) as f:
            calib = VO.Calib.from_json(json.load(f))
    app = application(
        worker,
        timeout=a.timeout,
        queue_limit=a.queue_limit,
        max_batch=a.max_batch,
        wait_ms=a.wait_ms,
        calib=calib,
        short=a.readout == "short",
    )
    web.run_app(app, host=a.host, port=a.port, handler_cancellation=True)
