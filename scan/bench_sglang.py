"""E1 baseline: Qwen3.5-4B in SGLang (radix cache with hybrid / mamba state checkpoints) answering the same N
(clip or image, query) pairs as scan/bench_scan.py and scan/bench_vllm.py: one request per pair, all submitted at once,
visual input first in the prompt so the radix cache can share it; yes/no readout = 1 new token with top logprobs.

/content/sglang_env/bin/python -m scan.bench_sglang --out OUT.json [--parts clips,cached] [--server_kw '{"...": ...}']

Regimes:
  clips   N 4-frame clips (same frames / size as bench_vllm part clips), native video input (mp4, 2 fps); every clip
          was already asked one different question (warm cache); time = one new query over all N clips
  cached  N images at 640 visual tokens, same warm-cache protocol
Records SGLang's cached_tokens per request (how much of the prompt the radix cache served).
"""
import argparse, asyncio, json, math, os, sys, tempfile, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scan.data import load_images, QUERIES  # noqa: E402

MODEL = "Qwen/Qwen3.5-4B"


def write_clip_mp4(frames, path, fps=2):
    import av
    c = av.open(path, "w")
    s = c.add_stream("libx264", rate=fps)
    s.width, s.height = frames[0].shape[1], frames[0].shape[0]
    s.pix_fmt = "yuv420p"
    s.options = {"crf": "0", "preset": "ultrafast"}
    for f in frames:
        for p in s.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
            c.mux(p)
    for p in s.encode():
        c.mux(p)
    c.close()


async def main_async(a):
    import sglang as sgl
    from transformers import AutoProcessor
    import torch
    proc = AutoProcessor.from_pretrained(MODEL)
    kw = dict(model_path=MODEL, mem_fraction_static=a.gpu_mem, context_length=4096, random_seed=0,
              max_running_requests=a.max_running, chunked_prefill_size=a.max_batched, log_level="error")
    if a.server_kw:
        kw.update(json.loads(a.server_kw))
    eng = sgl.Engine(**kw)
    res = dict(backend="sglang", version=sgl.__version__, gpu=torch.cuda.get_device_name(),
               engine_args={k: str(v) for k, v in kw.items()}, args=vars(a), modes={})
    tokz = proc.tokenizer
    yes_ids = {tokz.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes")}
    no_ids = {tokz.encode(w, add_special_tokens=False)[0] for w in ("no", "No")}

    def prompt(text, video):
        msgs = [{"role": "user", "content": [{"type": "video" if video else "image"}, {"type": "text", "text": text}]}]
        return proc.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=False)

    sp = dict(temperature=0.0, max_new_tokens=1)
    tail_ids = tokz.encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False)

    async def gen(text, item, video):
        if isinstance(item, list):        # token-only bound: item = prefix token ids (no vision input at all)
            ids = item + tokz.encode(text, add_special_tokens=False) + tail_ids
            kwargs = dict(input_ids=ids, sampling_params=sp, return_logprob=True, top_logprobs_num=20)
        else:
            kwargs = dict(prompt=prompt(text, video), sampling_params=sp, return_logprob=True, top_logprobs_num=20)
            kwargs["video_data" if video else "image_data"] = item
        o = await eng.async_generate(**kwargs)
        mi = o["meta_info"]
        top = mi.get("output_top_logprobs") or [[]]
        y = sum(math.exp(lp) for lp, tid, *_ in top[0] if tid in yes_ids)
        n = sum(math.exp(lp) for lp, tid, *_ in top[0] if tid in no_ids)
        return y / max(y + n, 1e-12), mi.get("cached_tokens", 0) or 0, mi.get("prompt_tokens", 0) or 0

    def text_for(q, video):
        return f"Is there {q} in the image? Answer yes or no."  # same text as bench_vllm (clips too)

    async def call(items, q, video):
        t0 = time.perf_counter()
        outs = await asyncio.gather(*[gen(text_for(q, video), it, video) for it in items])
        return (time.perf_counter() - t0) * 1e3, outs

    def uniq(path, c, j):
        """a copy of the image with one pixel changed per (call, slot): a new content hash, so no cross-call cache hit"""
        from PIL import Image
        q = f"{tmp}/u{c}_{j}.png"
        im = Image.open(path).convert("RGB")
        im.putpixel((0, 0), (c % 256, j % 256, (c * 7 + j) % 256))
        im.save(q)
        return q

    async def flush():
        result = await eng.tokenizer_manager.flush_cache()
        if not result.success:
            raise RuntimeError(f"SGLang cache reset failed: {result}")

    async def regime(name, items, video, n_calls, fresh_pool=None):
        Ts, cached, prompt_toks = [], [], []
        await flush()
        if fresh_pool is None:
            await call(items, "an object", video)          # warm: each unit asked one different question first
        for c in range(n_calls + 3):
            if fresh_pool is not None:                     # E1b: new images every call, nothing cached
                await flush()                                    # (flush alone did not always take: images are also made unique)
                N = len(items)
                items = [uniq(fresh_pool[(c * N + j) % len(fresh_pool)], c, j) for j in range(N)]
            T, outs = await call(items, QUERIES[c % len(QUERIES)], video)
            if c >= 3:
                Ts.append(T); cached += [o[1] for o in outs]; prompt_toks += [o[2] for o in outs]
        r = dict(total_ms=float(np.median(Ts)), total_ms_p90=float(np.percentile(Ts, 90)), n_calls=len(Ts),
                 images_per_s=len(items) / (float(np.median(Ts)) / 1e3),
                 cached_prompt_tokens_mean=float(np.mean(cached)), prompt_tokens_mean=float(np.mean(prompt_toks)))
        res["modes"][name] = r
        print(name, json.dumps({k: round(v, 2) for k, v in r.items()}), flush=True)
        json.dump(res, open(a.out, "w"), indent=1)

    parts = a.parts.split(",")
    Ns = [int(x) for x in a.Ns.split(",")]
    tmp = tempfile.mkdtemp(prefix="sgl_units_")
    if "clips" in parts or "cached_clips" in parts:
        from scan.data import serving_clips
        for name, clips in serving_clips(a.prep, a.image_root, max(Ns), a.clip_cfgs):
            paths = []
            for c, frames in enumerate(clips):
                path = f"{tmp}/{name}_{c}.mp4"
                write_clip_mp4(frames, path)
                paths.append(path)
            for N in Ns:
                await regime(f"clips_{name}_N{N}_yesno", paths[:N], True, a.n_calls)
    if "tokens" in parts:         # engine-only lower bound: same prompt lengths, prefix = fixed random text tokens
        rs = np.random.RandomState(0)
        pre = [[int(x) for x in rs.randint(1000, 100000, a.prefix_len)] for _ in range(max(Ns))]
        for N in Ns:
            await regime(f"tokens_p{a.prefix_len}_N{N}_yesno", pre[:N], False, a.n_calls)
    if "cached_box" in parts:
        from speed.workload import BOX_PROMPT
        for tv in (640, 256, 64):
            pool = load_images(a.prep, a.image_root, max(Ns), tv, frame=tv != 640)
            paths = []
            for i, image in enumerate(pool):
                path = f"{tmp}/box{tv}_{i}.png"
                image["img"].save(path)
                paths.append(path)
            async def box_call(n, expr):
                text = BOX_PROMPT.format(expr=expr)
                return await asyncio.gather(*[eng.async_generate(
                    prompt=prompt(text, False), image_data=path,
                    sampling_params=dict(temperature=0.0, max_new_tokens=24)) for path in paths[:n]])
            for n in Ns:
                await flush()
                await box_call(n, "an object")
                times, cached = [], []
                for c in range(a.n_calls + 3):
                    tick = time.perf_counter()
                    answers = await box_call(n, QUERIES[c % len(QUERIES)])
                    if c >= 3:
                        times.append((time.perf_counter() - tick) * 1000)
                        cached.extend(x["meta_info"].get("cached_tokens", 0) for x in answers)
                key = f"cached_Tv{tv}_N{n}_box"
                res["modes"][key] = dict(total_ms=float(np.median(times)), samples_ms=times,
                                         cached_prompt_tokens_mean=float(np.mean(cached)))
                print(key, res["modes"][key], flush=True)
                with open(a.out, "w") as file:
                    json.dump(res, file, indent=1)
    if "cached" in parts:
        pool = load_images(a.prep, a.image_root, max(Ns), 640)
        paths = []
        for i, x in enumerate(pool):
            p = f"{tmp}/img{i}.png"; x["img"].save(p); paths.append(p)
        for N in Ns:
            await regime(f"cached_Tv640_N{N}_yesno", paths[:N], False, a.n_calls)
    if "fresh" in parts:          # E1b: N new frames (frame-sized like bench_scan part fresh), one question, no cache
        from dv.data.dataset import resize_dims  # noqa: F401
        for Tv in [int(x) for x in a.Tvs.split(",")]:
            pool = load_images(a.prep, a.image_root, max(Ns) * 2, Tv, frame=True)
            paths = []
            for i, x in enumerate(pool):
                p = f"{tmp}/fresh{Tv}_{i}.png"; x["img"].save(p); paths.append(p)
            for N in Ns:
                await regime(f"fresh_Tv{Tv}_N{N}_yesno", paths[:N], False, a.n_calls, fresh_pool=paths)
    json.dump(res, open(a.out, "w"), indent=1)
    eng.shutdown()
    print("SGLANG_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", default="data/prep/eval_refcoco_val.jsonl")
    ap.add_argument("--image_root", default="/content/data/coco/train2014")
    ap.add_argument("--clip_cfgs", default="4x96", help="frames x visual tokens per temporal group")
    ap.add_argument("--parts", default="clips,cached")
    ap.add_argument("--Ns", default="1,8,32,64,128,256")
    ap.add_argument("--n_calls", type=int, default=12)
    ap.add_argument("--Tvs", default="128,256")
    ap.add_argument("--prefix_len", type=int, default=197, help="tokens part: prefix length (= clip prefix)")
    ap.add_argument("--max_running", type=int, default=256)
    ap.add_argument("--max_batched", type=int, default=16384)
    ap.add_argument("--gpu_mem", type=float, default=0.80)
    ap.add_argument("--server_kw", default="", help="JSON dict merged into sgl.Engine(**kw)")
    ap.add_argument("--out", required=True)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
