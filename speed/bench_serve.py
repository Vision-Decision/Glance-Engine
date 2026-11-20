"""Generative baseline (Qwen3.5-4B answering the same typed questions) served by vLLM or SGLang, in-process async engine.

/content/vllm_env/bin/python -m speed.bench_serve --backend vllm --out OUT.json [--spec mtp] [--quant fp8] [--quick]
/content/sglang_env/bin/python -m speed.bench_serve --backend sglang --out OUT.json

Workloads = speed/bench_glance.py (speed/workload.py): same images (already resized to <= max_vis visual tokens),
same questions; box questions ask for '[x1, y1, x2, y2]' or 'none' (max 24 new tokens, greedy), typed questions
for one answer token (max_tokens=1, i.e. the label is read from the first generated token, like Glance's readout).
Every call submits all its requests at once (the engine batches them, prefix caching shares the image prefix once
computed); the prefix cache is reset before each call (cold) unless the regime is "followup" (the image was
already asked about once: warm prefix + encoder cache, the best case for the baseline).
Timing per call: from building the prompts (chat template; the engine runs its image processor) until the last
answer; TTFA = first finished request.
"""
import argparse, asyncio, json, os, sys, time, uuid
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from speed import workload as W  # noqa: E402

MODEL = "Qwen/Qwen3.5-4B"


class VLLM:
    def __init__(self, a):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM
        from vllm import SamplingParams
        import vllm
        self.SP = SamplingParams
        kw = dict(model=MODEL, enable_prefix_caching=True, max_model_len=a.max_model_len, limit_mm_per_prompt={"image": 1},
                  gpu_memory_utilization=a.gpu_mem, seed=0, max_num_seqs=a.max_num_seqs,
                  max_num_batched_tokens=a.max_batched)
        if a.quant:
            kw["quantization"] = a.quant
        if a.spec:
            kw["speculative_config"] = dict(method=a.spec, num_speculative_tokens=a.spec_k)
        if a.async_sched:
            kw["async_scheduling"] = True
        self.info = dict(backend="vllm", version=vllm.__version__, engine_args={k: str(v) for k, v in kw.items()})
        self.eng = AsyncLLM.from_engine_args(AsyncEngineArgs(**kw))

    async def gen(self, text, img, max_tokens):
        sp = self.SP(temperature=0.0, max_tokens=max_tokens, logprobs=None)
        final = None
        async for out in self.eng.generate({"prompt": text, "multi_modal_data": {"image": img}}, sp, request_id=uuid.uuid4().hex):
            final = out
        o = final.outputs[0]
        return o.text, len(o.token_ids), getattr(final, "num_cached_tokens", None) or 0, len(final.prompt_token_ids)

    async def reset(self):
        await self.eng.reset_prefix_cache()
        try:
            await self.eng.reset_mm_cache()
        except Exception:
            pass

    def close(self):
        try:
            self.eng.shutdown()
        except Exception:
            pass


class SGLang:
    def __init__(self, a):
        import sglang as sgl
        kw = dict(model_path=MODEL, mem_fraction_static=a.gpu_mem, context_length=a.max_model_len,
                  random_seed=0, max_running_requests=a.max_num_seqs, chunked_prefill_size=a.max_batched)
        if a.quant:
            kw["quantization"] = a.quant
        if a.spec:
            kw.update(speculative_algorithm=a.spec, speculative_num_steps=a.spec_k, speculative_eagle_topk=1,
                      speculative_num_draft_tokens=a.spec_k + 1)
        self.info = dict(backend="sglang", version=sgl.__version__, engine_args={k: str(v) for k, v in kw.items()})
        self.eng = sgl.Engine(**kw)

    async def gen(self, text, img, max_tokens):
        out = await self.eng.async_generate(prompt=text, image_data=img,
                                            sampling_params=dict(temperature=0.0, max_new_tokens=max_tokens))
        mi = out.get("meta_info", {})
        return out["text"], mi.get("completion_tokens", 0), mi.get("cached_tokens", 0), mi.get("prompt_tokens", 0)

    async def reset(self):
        # Engine.flush_cache() drives the loop synchronously; this caller already owns it.
        result = await self.eng.tokenizer_manager.flush_cache()
        if not result.success:
            raise RuntimeError(f"SGLang cache reset failed: {result}")

    def close(self):
        try:
            self.eng.shutdown()
        except Exception:
            pass


async def run(a):
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(MODEL)
    be = VLLM(a) if a.backend == "vllm" else SGLang(a)
    import torch
    res = dict(be.info, gpu=torch.cuda.get_device_name(), modes={}, args=vars(a))

    def prompt(q):
        text, mt = W.gen_prompt(q)
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        mt = 1 if (a.one_token or mt == 2) else mt
        return proc.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=False), mt

    async def one(text, img, mt, t0):
        r = await be.gen(text, img, mt)
        return (time.perf_counter() - t0,) + r

    async def call(imgs, qss, reset=True):
        if reset:
            await be.reset()
        t0 = time.perf_counter()
        tasks = []
        for im, qs in zip(imgs, qss):
            for q in qs:
                text, mt = prompt(q)
                tasks.append(one(text, im, mt, t0))
        outs = await asyncio.gather(*tasks)
        return (time.perf_counter() - t0) * 1e3, outs

    def summ(Ts, Fs, outs_all, nq):
        t, f = np.array(Ts), np.array(Fs)
        return dict(total_ms_median=float(np.median(t)), total_ms_p90=float(np.percentile(t, 90)),
                    ttfa_ms_median=float(np.median(f)), q_per_s=float(nq * len(t) / (t.sum() / 1e3)), n_calls=len(t),
                    questions_per_call=nq, gen_tokens_mean=float(np.mean([o[2] for o in outs_all])),
                    cached_prompt_tokens_mean=float(np.mean([o[3] for o in outs_all])),
                    prompt_tokens_mean=float(np.mean([o[4] for o in outs_all])),
                    examples=[o[1] for o in outs_all[:6]])

    async def sweep(images, cfgs, n_calls, followup=False):
        for name, ipb, nq, kind in cfgs:
            calls = []
            for c in range(n_calls):
                idx = [(c * ipb + j) % len(images) for j in range(ipb)]
                calls.append(([images[i]["img"] for i in idx], [W.questions(images, i, nq, kind) for i in idx], idx))
            Ts, Fs, oo = [], [], []
            for c, (im, qs, idx) in enumerate(calls[:3] + calls):
                if followup:   # warm: the same image was asked one (different) question before
                    await call(im, [W.questions(images, i, 1, "box", seed=1) for i in idx], reset=True)
                T, outs = await call(im, qs, reset=not followup)
                if c >= 3:
                    Ts.append(T); Fs.append(min(o[0] for o in outs) * 1e3); oo += outs
            res["modes"][name] = summ(Ts, Fs, oo, ipb * nq)
            print(name, json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in res["modes"][name].items() if k != "examples"}), flush=True)
            json.dump(res, open(a.out, "w"), indent=1)

    parts = a.parts.split(",")
    imgs = W.load_coco(a.prep, a.image_root, a.n_images)
    QS = [1, 2, 4, 8, 16, 32, 64]
    nc = a.n_calls
    if "lat" in parts:
        await sweep(imgs, [(f"bs1_box_N{n}", 1, n, "box") for n in QS], nc)
    if "mixed" in parts:
        await sweep(imgs, [(f"bs1_mixed_N{n}", 1, n, "mixed") for n in (4, 8, 16, 32, 64)], nc)
    if "followup" in parts:
        await sweep(imgs, [(f"followup_{k}_N{n}", 1, n, k) for n in (1, 8, 32) for k in ("box", "mixed")], nc, followup=True)
    if "ceiling" in parts:
        await sweep(imgs, [(f"bs1_box_N{q}", 1, q, "box") for q in (1, 8, 32)] +
                    [(f"bs1_mixed_N{q}", 1, q, "mixed") for q in (8, 32)], nc)
    if "ceiling" in parts or "video1" in parts:
        frames = W.load_coco(a.prep, a.image_root, a.n_images, max_vis=256)
        await sweep(frames, [(f"video_f{f}_N1", f, 1, "box") for f in (32, 64)], max(6, nc // 4))
    if "clients" in parts:
        await sweep(imgs, [(f"clients_{c}", c, 1, "box") for c in (1, 4, 16, 64)], 12)
    if "tput" in parts:
        await sweep(imgs, [(f"bs{b}_box_N{n}", b, n, "box") for b in (2, 4, 8, 16, 32) for n in (1, 8)], max(6, nc // 4))
    if "tput2" in parts:          # saturation with many questions per image
        await sweep(imgs, [(f"bs{b}_box_N{n}", b, n, "box") for b, n in ((4, 32), (8, 32), (16, 32), (4, 64), (8, 64))], max(6, nc // 4))
    if "video" in parts:
        frames = W.load_coco(a.prep, a.image_root, a.n_images, max_vis=256)
        await sweep(frames, [(f"video_f{b}_N{n}", b, n, "mixed" if n > 1 else "box") for b in (8, 32, 64) for n in (1, 4)], max(6, nc // 4))
    if "gui" in parts and os.path.exists(a.gui):
        for nv in (2048, 4096):
            g = W.load_gui(a.gui, 40, max_vis=nv)
            await sweep(g, [(f"gui{nv}_box_N{n}", 1, n, "box") for n in (1, 8, 32)], min(nc, 30))
    json.dump(res, open(a.out, "w"), indent=1)
    be.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="vllm", choices=["vllm", "sglang"])
    ap.add_argument("--prep", default="data/prep/eval_refcoco_val.jsonl")
    ap.add_argument("--image_root", default="/content/data/coco/train2014")
    ap.add_argument("--gui", default="/content/gui/sspro40.jsonl")
    ap.add_argument("--parts", default="lat,mixed,followup,tput,video,gui")
    ap.add_argument("--n_images", type=int, default=100)
    ap.add_argument("--n_calls", type=int, default=50)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--max_num_seqs", type=int, default=256)
    ap.add_argument("--max_batched", type=int, default=16384)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--quant", default="")
    ap.add_argument("--spec", default="")
    ap.add_argument("--spec_k", type=int, default=2)
    ap.add_argument("--async_sched", action="store_true", help="vLLM: overlap scheduling with execution")
    ap.add_argument("--one_token", action="store_true",
                    help="every question generates exactly 1 token: prefill + one step = the cost of a decode-free "
                         "readout inside this engine (an engine-matched bound, NOT a baseline: it returns no box)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.quick:
        a.n_images, a.n_calls = 20, 8
    asyncio.run(run(a))


if __name__ == "__main__":
    main()
