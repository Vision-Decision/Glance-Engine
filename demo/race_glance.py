"""Glance side of the 32-question race: one COCO image, N box questions, one call. Records every call's timeline.

PYTHONPATH=. python demo/race_glance.py --ckpt /content/ckpt/v1/ckpt_final.pt --image IMG.jpg --questions demo/race_questions.json --out OUT.json

Timing boundary = speed/bench_glance.py (and the vLLM side): from the resized PIL image + question texts to all
answers on the host (image processor, tokenisation, H2D, vision, LM, heads, D2H). Engine: eng_fused (bf16, LoRA
merged, CUDA graphs, fused ops). Every answer of a call arrives at the same instant (one forward), so the per-question
completion time equals the call time. The shipped reference path (ref_lora, PEFT LoRA unmerged, eager) is run first
on the same inputs to check that the engine's decisions match.
"""
import argparse, json, time, types
import numpy as np
import torch
from PIL import Image

from dv.data.dataset import resize_dims
from speed.bench_glance import Runner


def answers(out, n):
    p = torch.sigmoid(out["presence_logit"][:n]).tolist()
    iou = torch.sigmoid(out["iou_logit"][:n]).tolist()
    return [dict(p_present=round(p[k], 5), box=[round(float(v), 5) for v in out["box_pred"][k]], p_iou50=round(iou[k], 5))
            for k in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_calls", type=int, default=50)
    ap.add_argument("--Ns", default="1,8,32")
    a = ap.parse_args()
    exprs = json.load(open(a.questions))["questions"]
    img0 = Image.open(a.image).convert("RGB")
    w, h = resize_dims(*img0.size, 640, 2.0)
    img = img0.resize((w, h), Image.BICUBIC)
    R = Runner(types.SimpleNamespace(ckpt=a.ckpt))
    res = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, image=a.image, input_wh=[w, h],
               orig_wh=list(img0.size), questions=exprs, n_calls=a.n_calls, runs={})

    qs = [dict(type="box", text=e, present=True, box=None) for e in exprs]
    ref, _ = R.call([img], [qs], Nv=640)                 # shipped reference path, each answer from the same forward
    ref_ans = answers(ref, len(qs))
    R.set_mode("eng_fused")
    for N in [int(x) for x in a.Ns.split(",")]:
        q = qs[:N]
        for _ in range(20):
            R.call([img], [q], Nv=640)
        calls = []
        for _ in range(a.n_calls):
            out, T = R.call([img], [q], Nv=640)
            calls.append(dict(T, answers_at_ms=[T["total"]] * N))
        tot = [c["total"] for c in calls]
        ans = answers(out, N)
        res["runs"][f"glance_bf16_N{N}"] = dict(
            total_ms_median=float(np.median(tot)), total_ms_p10=float(np.percentile(tot, 10)),
            total_ms_p90=float(np.percentile(tot, 90)),
            stages_ms_median={k: float(np.median([c[k] for c in calls])) for k in ("prep", "vision", "lm", "total")},
            calls=calls, answers=ans)
        print(N, json.dumps({k: round(v, 2) for k, v in res["runs"][f"glance_bf16_N{N}"]["stages_ms_median"].items()}), flush=True)
    eng = res["runs"][f"glance_bf16_N{len(qs)}"]["answers"]
    agree = [(r["p_present"] > .5) == (e["p_present"] > .5) for r, e in zip(ref_ans, eng)]
    dbox = [float(np.abs(np.array(r["box"]) - np.array(e["box"])).max()) for r, e in zip(ref_ans, eng)]
    res["reference"] = dict(path="ref_lora (dv.model.v1forward.run_model_v1, PEFT LoRA unmerged, eager, bf16)",
                            answers=ref_ans, presence_decisions_agree=int(sum(agree)), n=len(agree),
                            max_abs_box_diff=max(dbox))
    print("reference agreement", sum(agree), "/", len(agree), "max box diff", max(dbox), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
