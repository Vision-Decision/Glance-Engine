"""vLLM side of the 256-image search: Qwen3.5-4B answers the same box query for each of the same 256 images.

/content/vllm_env/bin/python demo/search_vllm.py --pick demo/data/coco_pick.json --image_dir /content/demo_imgs --config bf16 --out OUT.json

Protocol = scan/bench_vllm.py "cached" regime, the vLLM best case: one request per (image, query), all submitted at
once, image first in the prompt; every image was already asked one different question ("an object"), so its prefix
and visual embedding are cached. Time = the new query over all 256 images, prompt building to the last answer.
Each request's completion time is recorded (the progress bar in the GIF).
"""
import argparse, asyncio, json, os, sys, time, uuid
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from speed.workload import BOX_PROMPT  # noqa: E402
from dv.data.dataset import resize_dims  # noqa: E402
from demo.race_vllm import parse_box  # noqa: E402

MODEL = "Qwen/Qwen3.5-4B"


def article(c):
    return ("an " if c[0] in "aeiou" else "a ") + c


async def run(a):
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm import SamplingParams
    from transformers import AutoProcessor
    import vllm, torch
    pick = json.load(open(a.pick))["search"]
    imgs = []
    for r in pick["bank"]:
        im = Image.open(os.path.join(a.image_dir, r["file"])).convert("RGB")
        imgs.append(im.resize(resize_dims(*im.size, 640, 2.0), Image.BICUBIC))
    proc = AutoProcessor.from_pretrained(MODEL)
    kw = dict(model=MODEL, enable_prefix_caching=True, max_model_len=4096, limit_mm_per_prompt={"image": 1},
              gpu_memory_utilization=0.85, seed=0, max_num_seqs=256, max_num_batched_tokens=16384, async_scheduling=True,
              mm_processor_kwargs=dict(max_pixels=4096 * 32 * 32, min_pixels=16 * 32 * 32))
    if a.config == "fp8mtp":
        kw.update(quantization="fp8", speculative_config=dict(method="mtp", num_speculative_tokens=2))
    eng = AsyncLLM.from_engine_args(AsyncEngineArgs(**kw))
    res = dict(backend="vllm", version=vllm.__version__, gpu=torch.cuda.get_device_name(), config=a.config,
               engine_args={k: str(v) for k, v in kw.items()}, n_images=len(imgs), queries={})

    def prompt(t):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": BOX_PROMPT.format(expr=t)}]}]
        return proc.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=False)

    async def one(t, im, t0):
        final = None
        async for o in eng.generate({"prompt": prompt(t), "multi_modal_data": {"image": im}},
                                    SamplingParams(temperature=0.0, max_tokens=24), request_id=uuid.uuid4().hex):
            final = o
        out = final.outputs[0]
        return dict(t_ms=(time.perf_counter() - t0) * 1e3, text=out.text, n_tokens=len(out.token_ids),
                    cached_tokens=getattr(final, "num_cached_tokens", 0) or 0, prompt_tokens=len(final.prompt_token_ids))

    async def call(t):
        t0 = time.perf_counter()
        outs = await asyncio.gather(*[one(t, im, t0) for im in imgs])
        return (time.perf_counter() - t0) * 1e3, outs

    await eng.reset_prefix_cache()
    await call("an object")                          # warm: every image seen once (prefix + encoder cache)
    held = pick["held"]
    queries = [article(pick["target"])] + [article(c) for c in held[a.n_cal:]]
    for t in queries:
        await call(t)                                # untimed pass per query text
        calls = []
        for _ in range(a.n_calls):
            T, outs = await call(t)
            calls.append(dict(total=T, answers_at_ms=[o["t_ms"] for o in outs], n_tokens=[o["n_tokens"] for o in outs],
                              cached_tokens=[o["cached_tokens"] for o in outs], prompt_tokens=[o["prompt_tokens"] for o in outs]))
        last = outs
        tot = [c["total"] for c in calls]
        res["queries"][t] = dict(total_ms_median=float(np.median(tot)), total_ms_p10=float(np.percentile(tot, 10)),
                                 total_ms_p90=float(np.percentile(tot, 90)),
                                 cached_fraction=float(np.sum([c["cached_tokens"] for c in calls]) / np.sum([c["prompt_tokens"] for c in calls])),
                                 gen_tokens_mean=float(np.mean([x for c in calls for x in c["n_tokens"]])),
                                 calls=calls, answers=[dict(text=o["text"], box=parse_box(o["text"])) for o in last])
        q = res["queries"][t]
        print(t, round(q["total_ms_median"], 1), "cached", round(q["cached_fraction"], 3), "tokens", round(q["gen_tokens_mean"], 2),
              "found", sum(x["box"] is not None for x in q["answers"]), flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
    eng.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pick", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--config", default="bf16", choices=["fp8mtp", "bf16"])
    ap.add_argument("--n_cal", type=int, default=34)
    ap.add_argument("--n_calls", type=int, default=5)
    ap.add_argument("--out", required=True)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
