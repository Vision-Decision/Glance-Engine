"""Fair baseline for scan: Qwen3.5-4B in vLLM answering the same N (image, query) pairs, one request per pair, all
submitted at once (continuous batching), image first in the prompt so prefix caching shares it.

/content/vllm_env/bin/python -m scan.bench_vllm --out OUT.json [--quant fp8] [--spec mtp] [--parts cached,fresh]

Readouts (both per pair):
  yesno  'Is there {query} in the image? Answer yes or no.'  max_tokens=1, logprobs -> P(yes)   (decode-free bound:
         what a FRAG / VQAScore-style scanner costs; no box)
  box    the speed/ box prompt ('[x1, y1, x2, y2]' or 'none'), greedy, <= 24 new tokens (what Glance returns)
Regimes (as scan/bench_scan.py):
  cached  every image was already asked one different question (warm prefix + encoder cache: vLLM's best case);
          time = the new query over all N images
  fresh   prefix and encoder caches reset before each call
"""
import argparse, asyncio, json, math, os, sys, time, uuid
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scan.data import load_images, QUERIES  # noqa: E402
from speed.workload import BOX_PROMPT  # noqa: E402

MODEL = "Qwen/Qwen3.5-4B"


async def main_async(a):
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm import SamplingParams
    from transformers import AutoProcessor
    import vllm, torch
    proc = AutoProcessor.from_pretrained(MODEL)
    kw = dict(model=MODEL, enable_prefix_caching=True, max_model_len=4096, limit_mm_per_prompt={"image": 1, "video": 1},
              gpu_memory_utilization=a.gpu_mem, seed=0, max_num_seqs=a.max_num_seqs, max_num_batched_tokens=a.max_batched,
              async_scheduling=True, mm_processor_kwargs=dict(max_pixels=4096 * 32 * 32, min_pixels=16 * 32 * 32))
    if a.quant:
        kw["quantization"] = a.quant
    if a.spec:
        kw["speculative_config"] = dict(method=a.spec, num_speculative_tokens=2)
    if a.engine_kw:          # E1: e.g. a smaller hybrid (mamba) cache block, '{"mamba_block_size": 64}'
        kw.update(json.loads(a.engine_kw))
    eng = AsyncLLM.from_engine_args(AsyncEngineArgs(**kw))
    res = dict(backend="vllm", version=vllm.__version__, gpu=torch.cuda.get_device_name(), engine_args={k: str(v) for k, v in kw.items()},
               args=vars(a), modes={})
    tokz = proc.tokenizer
    yes_ids = {tokz.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes")}
    no_ids = {tokz.encode(w, add_special_tokens=False)[0] for w in ("no", "No")}

    def prompt(text, video=False):
        msgs = [{"role": "user", "content": [{"type": "video" if video else "image"}, {"type": "text", "text": text}]}]
        return proc.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=False)

    tail_ids = tokz.encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False)

    async def gen(text, img, kind):
        sp = SamplingParams(temperature=0.0, max_tokens=1 if kind == "yesno" else 24, logprobs=20 if kind == "yesno" else None)
        final = None
        if isinstance(img, list):            # token-only bound: prefix token ids, no vision input
            ids = img + tokz.encode(text, add_special_tokens=False) + tail_ids
            async for o in eng.generate({"prompt_token_ids": ids}, sp, request_id=uuid.uuid4().hex):
                final = o
            out = final.outputs[0]
            lp = out.logprobs[0] if out.logprobs else {}
            y = sum(math.exp(v.logprob) for t, v in lp.items() if t in yes_ids)
            n = sum(math.exp(v.logprob) for t, v in lp.items() if t in no_ids)
            return out.text, y / max(y + n, 1e-12), getattr(final, "num_cached_tokens", 0) or 0, len(final.prompt_token_ids)
        if isinstance(img, np.ndarray):          # a clip [T,H,W,3] at 2 fps: native video input
            T = len(img)
            mm = {"video": (img, dict(fps=2.0, frames_indices=list(range(T)), total_num_frames=T, duration=T / 2.0,
                                      video_backend="opencv", do_sample_frames=False))}
        else:
            mm = {"image": img}
        async for o in eng.generate({"prompt": text, "multi_modal_data": mm}, sp, request_id=uuid.uuid4().hex):
            final = o
        out = final.outputs[0]
        p_yes = None
        if kind == "yesno" and out.logprobs:
            lp = out.logprobs[0]
            y = sum(math.exp(v.logprob) for t, v in lp.items() if t in yes_ids)
            n = sum(math.exp(v.logprob) for t, v in lp.items() if t in no_ids)
            p_yes = y / max(y + n, 1e-12)
        return out.text, p_yes, getattr(final, "num_cached_tokens", 0) or 0, len(final.prompt_token_ids)

    def text_for(q, kind):
        return f"Is there {q} in the image? Answer yes or no." if kind == "yesno" else BOX_PROMPT.format(expr=q)


    async def call(imgs, q, kind):
        t0 = time.perf_counter()
        outs = await asyncio.gather(*[gen(text_for(q, kind) if isinstance(im, list) else prompt(text_for(q, kind), isinstance(im, np.ndarray)), im, kind) for im in imgs])
        return (time.perf_counter() - t0) * 1e3, outs

    async def reset():
        if await eng.reset_prefix_cache() is False:
            raise RuntimeError("vLLM prefix cache reset failed")
        if hasattr(eng, "reset_mm_cache"):
            await eng.reset_mm_cache()
        # vLLM 0.34 keeps computed visual embeddings in a separate encoder cache.
        if not a.retain_encoder_cache and hasattr(eng, "reset_encoder_cache"):
            await eng.reset_encoder_cache()

    fresh_serial = 0

    async def regime(name, imgs_pool, N, kind, cached, n_calls, unique_images=False):
        nonlocal fresh_serial
        Ts, cached_toks, prompt_toks = [], [], []
        for c in range(n_calls + 3):
            ims = [imgs_pool[(c * N + j) % len(imgs_pool)] for j in range(N)] if not cached else imgs_pool[:N]
            if unique_images:
                # Vary image content outside the timed request, including across batch sizes.
                ims = [im.copy() for im in ims]
                for im in ims:
                    fresh_serial += 1
                    im.putpixel((0, 0), (fresh_serial & 255, (fresh_serial >> 8) & 255, (fresh_serial >> 16) & 255))
            q = QUERIES[c % len(QUERIES)]
            if not cached:
                await reset()
            elif c == 0:          # warm: each image asked one different question first
                await reset()
                await call(ims, "an object", kind)
            T, outs = await call(ims, q, kind)
            if c >= 3:
                Ts.append(T); cached_toks += [o[2] for o in outs]; prompt_toks += [o[3] for o in outs]
        r = dict(total_ms=float(np.median(Ts)), total_ms_p90=float(np.percentile(Ts, 90)), n_calls=len(Ts),
                 images_per_s=N / (float(np.median(Ts)) / 1e3),
                 cached_prompt_tokens_mean=float(np.mean(cached_toks)), prompt_tokens_mean=float(np.mean(prompt_toks)))
        res["modes"][name] = r
        print(name, json.dumps({k: round(v, 2) for k, v in r.items()}), flush=True)
        json.dump(res, open(a.out, "w"), indent=1)

    parts = a.parts.split(",")
    Ns = [int(x) for x in a.Ns.split(",")]
    kinds = a.kinds.split(",")
    if "tokens" in parts:         # engine-only lower bound: same prompt lengths, prefix = fixed random text tokens
        rs = np.random.RandomState(0)
        pre = [[int(x) for x in rs.randint(1000, 100000, a.prefix_len)] for _ in range(max(Ns))]
        for N in Ns:
            await regime(f"tokens_p{a.prefix_len}_N{N}_yesno", pre, N, "yesno", True, a.n_calls)
    if "cached" in parts:
        for Tv, frame in ((640, False), (256, True), (64, True)):
            pool = [x["img"] for x in load_images(a.prep, a.image_root, max(Ns), Tv, frame=frame)]
            for N in Ns:
                for kind in kinds:
                    await regime(f"cached_Tv{Tv}_N{N}_{kind}", pool, N, kind, True, a.n_calls)
    if "clips" in parts or "cached_clips" in parts:
        from scan.data import serving_clips
        for name, clips in serving_clips(a.prep, a.image_root, max(Ns), a.clip_cfgs):
            for N in Ns:
                await regime(f"clips_{name}_N{N}_yesno", clips, N, "yesno", True, a.n_calls)
            if "clips" in parts:
                for N in (8, 16, 32, 64):
                    await regime(f"clips_fresh_{name}_N{N}_yesno", clips, N, "yesno", False, a.n_calls)
    if "fresh" in parts:
        for Tv in [int(x) for x in a.Tvs.split(",")]:
            pool = [x["img"] for x in load_images(a.prep, a.image_root, max([64] + [int(x) for x in a.Ns_fresh.split(",")]), Tv, frame=True)]  # unique images per call
            for N in [int(x) for x in a.Ns_fresh.split(",")]:
                if N * Tv > a.max_fresh_tokens:
                    continue
                for kind in kinds:
                    await regime(f"fresh_Tv{Tv}_N{N}_{kind}", pool, N, kind, False, a.n_calls, a.unique_fresh)
    json.dump(res, open(a.out, "w"), indent=1)
    eng.shutdown()
    print("VLLM_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", default="data/prep/eval_refcoco_val.jsonl")
    ap.add_argument("--image_root", default="/content/data/coco/train2014")
    ap.add_argument("--clip_cfgs", default="4x96", help="frames x visual tokens per temporal group")
    ap.add_argument("--parts", default="clips,cached,fresh")
    ap.add_argument("--retain_encoder_cache", action="store_true",
                    help="reproduce the historical prefix/processor-only cache reset")
    ap.add_argument("--unique_fresh", action="store_true",
                    help="give every fresh image a unique content hash outside the timed request")
    ap.add_argument("--kinds", default="yesno,box")
    ap.add_argument("--Ns", default="1,8,16,32,64,128,256")
    ap.add_argument("--Ns_fresh", default="1,2,4,8,16,32,64")
    ap.add_argument("--Tvs", default="64,128,256,640")
    ap.add_argument("--max_fresh_tokens", type=int, default=16384)
    ap.add_argument("--n_calls", type=int, default=20)
    ap.add_argument("--max_num_seqs", type=int, default=256)
    ap.add_argument("--max_batched", type=int, default=16384)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--quant", default="")
    ap.add_argument("--spec", default="")
    ap.add_argument("--prefix_len", type=int, default=197, help="tokens part: prefix length (= clip prefix)")
    ap.add_argument("--engine_kw", default="", help="JSON dict merged into the AsyncEngineArgs")
    ap.add_argument("--out", required=True)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
