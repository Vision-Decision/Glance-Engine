"""vLLM side of the 32-question race: Qwen3.5-4B answers the same box questions about the same resized image.

/content/vllm_env/bin/python demo/race_vllm.py --image IMG.jpg --questions demo/race_questions.json --config fp8mtp --out OUT.json

Same protocol as speed/bench_serve.py: one request per question, all submitted at once (continuous batching; prefix
caching shares the image prefix), greedy, box prompt speed.workload.BOX_PROMPT, <= 24 new tokens. Before every call
the prefix, multimodal-processor and encoder caches are reset (a new image). Timing: from building the prompts to
the last answer; each request's own completion time is recorded, which is the timeline the GIF replays.
configs: fp8mtp = FP8 weights + MTP speculative decoding (k=2) + async scheduling (the strongest setting measured in
outputs/sprint_speed); bf16 = async scheduling only.
"""
import argparse, asyncio, json, os, re, sys, time, uuid
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from speed.workload import BOX_PROMPT  # noqa: E402
from dv.data.dataset import resize_dims  # noqa: E402

MODEL = "Qwen/Qwen3.5-4B"


def parse_box(text):
    m = re.findall(r"-?\d+(?:\.\d+)?", text)
    if "none" in text.lower() or len(m) < 4:
        return None
    return [round(float(v) / 1000.0, 5) for v in m[:4]]


async def run(a):
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm import SamplingParams
    from transformers import AutoProcessor
    import vllm, torch
    exprs = json.load(open(a.questions))["questions"]
    img0 = Image.open(a.image).convert("RGB")
    img = img0.resize(resize_dims(*img0.size, 640, 2.0), Image.BICUBIC)
    proc = AutoProcessor.from_pretrained(MODEL)
    kw = dict(model=MODEL, enable_prefix_caching=True, max_model_len=8192, limit_mm_per_prompt={"image": 1},
              gpu_memory_utilization=0.85, seed=0, max_num_seqs=256, max_num_batched_tokens=16384, async_scheduling=True)
    if a.config == "fp8mtp":
        kw.update(quantization="fp8", speculative_config=dict(method="mtp", num_speculative_tokens=2))
    eng = AsyncLLM.from_engine_args(AsyncEngineArgs(**kw))
    res = dict(backend="vllm", version=vllm.__version__, gpu=torch.cuda.get_device_name(), config=a.config,
               engine_args={k: str(v) for k, v in kw.items()}, image=a.image, input_wh=list(img.size), questions=exprs,
               runs={})

    def prompt(e):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": BOX_PROMPT.format(expr=e)}]}]
        return proc.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=False)

    async def one(e, t0):
        sp = SamplingParams(temperature=0.0, max_tokens=24)
        final = None
        async for o in eng.generate({"prompt": prompt(e), "multi_modal_data": {"image": img}}, sp, request_id=uuid.uuid4().hex):
            final = o
        out = final.outputs[0]
        return dict(t_ms=(time.perf_counter() - t0) * 1e3, text=out.text, n_tokens=len(out.token_ids),
                    cached_tokens=getattr(final, "num_cached_tokens", 0) or 0)

    async def reset():
        await eng.reset_prefix_cache()
        if hasattr(eng, "reset_mm_cache"):
            await eng.reset_mm_cache()
        if hasattr(eng, "reset_encoder_cache"):
            await eng.reset_encoder_cache()

    async def call(es):
        await reset()
        t0 = time.perf_counter()
        outs = await asyncio.gather(*[one(e, t0) for e in es])
        return (time.perf_counter() - t0) * 1e3, outs

    for N in [int(x) for x in a.Ns.split(",")]:
        es = exprs[:N]
        for _ in range(5):
            await call(es)
        calls = []
        for _ in range(a.n_calls):
            T, outs = await call(es)
            calls.append(dict(total=T, answers_at_ms=[o["t_ms"] for o in outs], n_tokens=[o["n_tokens"] for o in outs],
                              cached_tokens=[o["cached_tokens"] for o in outs], texts=[o["text"] for o in outs]))
        tot = [c["total"] for c in calls]
        last = calls[-1]
        res["runs"][f"vllm_{a.config}_N{N}"] = dict(
            total_ms_median=float(np.median(tot)), total_ms_p10=float(np.percentile(tot, 10)),
            total_ms_p90=float(np.percentile(tot, 90)), first_answer_ms_median=float(np.median([min(c["answers_at_ms"]) for c in calls])),
            gen_tokens_mean=float(np.mean([t for c in calls for t in c["n_tokens"]])),
            cached_tokens_mean=float(np.mean([t for c in calls for t in c["cached_tokens"]])),
            calls=calls, answers=[dict(text=t, box=parse_box(t)) for t in last["texts"]])
        r = res["runs"][f"vllm_{a.config}_N{N}"]
        print(N, round(r["total_ms_median"], 1), round(r["first_answer_ms_median"], 1), r["gen_tokens_mean"], r["cached_tokens_mean"], flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
    eng.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--config", default="fp8mtp", choices=["fp8mtp", "bf16"])
    ap.add_argument("--Ns", default="1,8,32")
    ap.add_argument("--n_calls", type=int, default=50)
    ap.add_argument("--out", required=True)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
