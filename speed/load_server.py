"""Load-test an existing server or exercise all endpoints on a local ephemeral port."""

import argparse
import asyncio
import base64
import json
import time
from io import BytesIO
from pathlib import Path

import numpy as np
from aiohttp import ClientSession, web
from PIL import Image


async def run(args):
    runner = None
    if args.self_test:
        from speed.server import DecisionWorker, application

        worker = DecisionWorker.load(args.ckpt)
        app = application(worker, queue_limit=32, timeout=120)
        runner = web.AppRunner(app, handler_cancellation=True)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        url = "http://127.0.0.1:" + str(runner.addresses[0][1])
    else:
        url = args.url.rstrip("/")
    rng = np.random.default_rng(101)
    encoded = []
    for _ in range(args.clients):
        data = BytesIO()
        Image.fromarray(rng.integers(0, 256, (192, 256, 3), dtype=np.uint8)).save(data, format="JPEG")
        encoded.append(base64.b64encode(data.getvalue()).decode())
    try:
        async with ClientSession() as session:

            async def get(path):
                async with session.get(url + path) as response:
                    assert response.status == 200
                    return await response.json()

            async def post(path, body, status=200):
                async with session.post(url + path, json=body) as response:
                    obj = await response.json()
                    assert response.status == status, (path, response.status, obj)
                    return obj

            assert (await get("/ready"))["ok"]

            async def client(i):
                t = time.perf_counter()
                out = await post(
                    "/v1/decisions", {"image": encoded[i], "questions": [{"type": "box", "text": "a car"}]}
                )
                assert len(out["results"][0][0]["box"]) == 4
                return (time.perf_counter() - t) * 1000

            await asyncio.gather(*(client(i) for i in range(args.clients)))
            times, bursts = [], []
            for _ in range(args.runs):
                t = time.perf_counter()
                times.extend(await asyncio.gather(*(client(i) for i in range(args.clients))))
                bursts.append((time.perf_counter() - t) * 1000)
            res = dict(
                clients=args.clients,
                runs=args.runs,
                burst_ms=bursts,
                request_ms=times,
                requests_s=len(times) * 1000 / sum(bursts),
                **{f"p{p}_ms": float(np.percentile(times, p)) for p in (50, 90, 99)},
            )
            if args.self_test:
                qs = [{"type": "noul", "instr": "a car?"}]
                cached = await post("/v1/decisions", {"image": encoded[0], "questions": qs, "cache": True})
                from_id = await post("/v1/decisions", {"image_ids": cached["image_ids"], "questions": qs})
                self_diff = abs(cached["results"][0][0]["probs"][1] - from_id["results"][0][0]["probs"][1])
                assert self_diff <= 0.0233531148
                assert "error" in await post("/v1/decisions", {"image_ids": ["00" * 32], "questions": qs}, 404)
                bad = await post(
                    "/v1/decisions", {"image": "?", "questions": [{"type": "box", "text": "dog"}]}, 400
                )
                assert bad["error"]["type"] == "invalid_request"
                assert "error" in await post("/scan", {"video": "missing", "query": "car?"}, 404)
                clips = [[encoded[0]] * 2, [encoded[-1]] * 2]
                req = {"video": "v", "clips": clips, "times": [[2, 3], [4, 5]]}
                assert (await post("/encode", req))["cached"] is False
                assert (await post("/encode", req))["cached"] is True
                out = await post("/scan", {"video": "v", "query": "car?", "ops": ["curve", "ever"]})
                assert [r["t0"] for r in out["curve"]] == [2, 4] and out["ever"]["guarantee"] is None
                req = {"clips": clips, "dt": 0.25, "question": "car?", "kind": "yesno"}
                cold = await post("/score_clips", req)
                warm = await post("/score_clips/", {**req, "questions": [], "times": None})
                assert not cold["cached"] and warm["cached"] and cold["results"] == warm["results"]
                req["dt"] = 0.5
                assert not (await post("/score_clips", req))["cached"]
                mixed = await post(
                    "/score_clips",
                    {
                        "clips": clips,
                        "questions": [
                            {"type": "noul", "instr": "car?"},
                            {"type": "choice", "instr": "Which?", "options": ["bus", "car", "train"]},
                        ],
                    },
                )
                assert len(mixed["results"]) == 2 and len(mixed["results"][1][1]) == 3
                chat = await post(
                    "/v1/chat/completions",
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": "data:image/jpeg;base64," + encoded[0]},
                                    },
                                    {"type": "text", "text": "Is there a car?"},
                                ],
                            }
                        ]
                    },
                )
                assert len(json.loads(chat["choices"][0]["message"]["content"])[0][0]["probs"]) == 2
                assert (await get("/health/"))["ok"]
                res["endpoint_checks"] = "passed"
            res["metrics"] = await get("/metrics")
            args.out.write_text(json.dumps(res, indent=2))
            print(json.dumps({k: v for k, v in res.items() if k not in ("request_ms", "burst_ms")}))
    finally:
        if runner is not None:
            await runner.cleanup()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--self-test", action="store_true")
    group.add_argument("--url")
    p.add_argument("--ckpt", default="/content/ckpt/v1/ckpt_final.pt")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--clients", type=int, default=16)
    p.add_argument("--runs", type=int, default=5)
    asyncio.run(run(p.parse_args()))
