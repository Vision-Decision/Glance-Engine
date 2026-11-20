"""Simultaneous logical clients, timed from burst arrival to each client's last answer.

Each backend receives the same decoded, resized images and questions. Media are
content-unique across cold bursts, including when the source corpus wraps.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np

from speed import workload as W


def burst(pool, workload, clients, iteration, serial, mark=True):
    result = []
    for j in range(clients):
        n = 32 if workload == "fresh" else 1
        idx = ((iteration * clients + j) * n) % len(pool)
        if workload == "fresh":
            from scan.data import QUERIES

            qs = [{"type": "noul", "instr": f"Is there {QUERIES[(iteration + j) % len(QUERIES)]} in the image?"}]
        else:
            kind = {"yesno": "noul", "multi": "mixed"}.get(workload, "box")
            qs = W.questions(pool, idx, 8 if workload == "multi" else 1, kind)
        images = []
        for k in range(n):
            im = pool[(idx + k) % len(pool)]["img"]
            if mark:
                im = im.copy()
                serial += 1
                im.putpixel((0, 0), (serial & 255, (serial >> 8) & 255, (serial >> 16) & 255))
            images.append(im)
        result.append((images, qs))
    return result, serial


class Generative:
    def __init__(self, a):
        from transformers import AutoProcessor

        from speed import bench_serve as serve

        if a.model:
            serve.MODEL = a.model
        self.proc = AutoProcessor.from_pretrained(serve.MODEL)
        self.engine = (serve.VLLM if a.backend == "vllm" else serve.SGLang)(a)
        self.info = self.engine.info

    async def reset(self):
        await self.engine.reset()
        eng = self.engine.eng
        if hasattr(eng, "reset_encoder_cache"):
            await eng.reset_encoder_cache()

    async def answer(self, images, questions, warm=False):
        async def one(im, q):
            text, count = W.gen_prompt(q)
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
            prompt = self.proc.apply_chat_template(
                msgs, add_generation_prompt=True, enable_thinking=False, tokenize=False
            )
            return await self.engine.gen(prompt, im, 1 if count == 2 else count)

        await asyncio.gather(*(one(im, q) for im in images for q in questions))

    async def close(self):
        self.engine.close()


class Serial:
    def __init__(self, a):
        from scan.engine import ScanEngine
        from speed.bench_glance import Runner

        self.runner = Runner(a)
        self.runner.set_mode("eng_fused")
        r = self.runner
        self.scan = ScanEngine(r.net, r.heads, r.tok, r.proc, fast_ops=True)
        self.prefixes = {}
        self.info = {"backend": "serial"}

    async def reset(self):
        self.prefixes.clear()

    async def answer(self, images, qs, warm=False):
        if len(images) > 1:
            self.scan.fresh(images, qs[0])
        elif warm:
            self.runner.call(images, [qs], Nv=640, keep_prefix=True)
            self.prefixes[id(images[0])] = self.runner._prefix
        elif id(images[0]) in self.prefixes:
            self.runner._prefix = self.prefixes[id(images[0])]
            self.runner.followup([qs])
        else:
            max_vis = max(640, images[0].width * images[0].height // 1024)
            self.runner.call(images, [qs], Nv=2048 if max_vis > 640 else 640)

    async def close(self):
        pass


class Scheduled:
    def __init__(self, a):
        from speed.decision_worker import DecisionWorker
        from speed.scheduler import Scheduler

        self.worker = DecisionWorker.load(a.ckpt, max_tokens=a.tokens, group_vision=a.group_vision)
        self.scheduler = Scheduler(self.worker, max_tokens=a.tokens, wait_ms=a.wait_ms)
        self.scheduler.start()
        self.by_id = a.by_id
        self.refs = {}
        self.info = {
            "backend": "scheduled",
            "max_tokens": a.tokens,
            "wait_ms": a.wait_ms,
            "group_vision": a.group_vision,
            "flash_gdn": self.worker.engine.flash_gdn is not None,
        }

    async def reset(self):
        self.worker.cache.clear()
        self.refs.clear()

    async def answer(self, images, questions, warm=False):
        from speed.decision_worker import Decision, ImageRef

        cache = warm or self.followup
        inputs = [self.refs[id(im)] for im in images] if self.by_id and self.followup and not warm else images
        req = Decision(inputs, questions, cache=cache and inputs is images)
        await self.scheduler.submit(req, timeout=900)
        if warm and self.by_id:
            self.refs.update({id(im): ImageRef(bytes.fromhex(key)) for im, key in zip(images, req.image_ids)})

    async def close(self):
        await self.scheduler.close()


async def run(a):
    import torch

    cls = {"vllm": Generative, "sglang": Generative, "serial": Serial, "scheduled": Scheduled}[a.backend]
    be = cls(a)
    res = {"info": be.info, "gpu": torch.cuda.get_device_name(), "warmups": a.warmups, "runs": a.runs, "modes": {}}
    res["warm_shapes"] = a.warm_shapes
    res["by_id"] = a.by_id
    if a.out.exists():
        prior = json.loads(a.out.read_text())
        if prior["info"] != res["info"] or prior["runs"] != a.runs or prior["warmups"] != a.warmups:
            raise ValueError("output file is from a different configuration")
        if prior.get("warm_shapes", False) != a.warm_shapes or prior.get("by_id", False) != a.by_id:
            raise ValueError("output belongs to a different cache/warmup protocol")
        res = prior
    serial = 0
    try:
        for work in a.parts.split(","):
            if work == "gui":
                pool = W.load_gui(a.gui, 40)
            elif work == "fresh":
                from scan.data import load_images

                pool = load_images(a.prep, a.image_root, 100, 128, frame=True)
            else:
                pool = W.load_coco(a.prep, a.image_root, 100)
            for clients in map(int, a.clients.split(",")):
                key = f"{work}_{clients}"
                serial = (
                    ("box", "yesno", "multi", "followup", "gui", "fresh").index(work) * 4
                    + (1, 4, 16, 64).index(clients)
                ) * 100000
                if key in res["modes"]:
                    continue
                finishes, totals, stages = [], [], []
                # Glance may lazily capture a new graph for each shape. Warm those exact
                # shapes on different pixels before measuring the content-cold requests.
                shape_runs = list(range(a.warmups, a.warmups + a.runs)) if a.warm_shapes else []
                for index, it in enumerate(shape_runs + list(range(a.warmups + a.runs))):
                    if index < len(shape_runs):
                        calls, _ = burst(pool, work, clients, it, 8000000 + serial + index * 2048)
                    else:
                        calls, serial = burst(pool, work, clients, it, serial)
                    await be.reset()
                    be.followup = work == "followup"
                    if be.followup:
                        await asyncio.gather(
                            *(
                                be.answer(ims, [{"type": "noul", "instr": "Is this an image?"}], warm=True)
                                for ims, _ in calls
                            )
                        )
                    if isinstance(be, Scheduled):
                        be.worker.reset_timings()
                    t0 = time.perf_counter()

                    async def client(ims, qs, start=t0):
                        await be.answer(ims, qs)
                        return (time.perf_counter() - start) * 1000

                    elapsed = await asyncio.gather(*(client(ims, qs) for ims, qs in calls))
                    if index >= len(shape_runs) + a.warmups:
                        finishes.extend(elapsed)
                        totals.append(max(elapsed))
                        if isinstance(be, Scheduled):
                            stages.append(dict(be.worker.timings))
                row = dict(
                    burst_ms=totals,
                    request_ms=finishes,
                    burst_median_ms=float(np.median(totals)),
                    requests_s=1000 * clients * len(totals) / sum(totals),
                    **{f"p{p}_ms": float(np.percentile(finishes, p)) for p in (50, 90, 99)},
                )
                if stages:
                    row["stages"] = stages
                res["modes"][key] = row
                tmp = a.out.with_suffix(".tmp")
                tmp.write_text(json.dumps(res, indent=1))
                tmp.replace(a.out)
                print(key, {k: round(v, 2) for k, v in row.items() if isinstance(v, float)}, flush=True)
    finally:
        await be.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["serial", "scheduled", "vllm", "sglang"], required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--parts", default="box,yesno,multi,followup,gui,fresh")
    p.add_argument("--clients", default="1,4,16,64")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--warmups", type=int, default=2)
    p.add_argument("--group-vision", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--by-id", action="store_true")
    p.add_argument("--warm-shapes", action="store_true")
    p.add_argument("--tokens", type=int, default=8192)
    p.add_argument("--wait-ms", type=float, default=0.5)
    p.add_argument("--ckpt", default="/content/ckpt/v1/ckpt_final.pt")
    p.add_argument("--prep", default="data/prep/eval_refcoco_val.jsonl")
    p.add_argument("--image-root", default="/content/data/coco/train2014")
    p.add_argument("--gui", default="/content/gui/sspro40.jsonl")
    p.add_argument("--model", help="local snapshot for generative engines")
    p.set_defaults(
        max_model_len=8192, gpu_mem=0.85, max_num_seqs=256, max_batched=16384, quant="", spec="", async_sched=True
    )
    asyncio.run(run(p.parse_args()))
