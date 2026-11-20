"""Numerical equivalence of every inference variant against an fp32 run of the shipped model.

python -m speed.diag_equiv --ckpt /content/ckpt/v1/ckpt_final.pt --out OUT.json [--n_img 60]

Reference = the shipped path (PEFT LoRA unmerged, run_model_v1) with all base weights in fp32 and autocast off.
Variants (all bf16):  ref_lora (shipped), ref_merged (LoRA merged), and the engine with each change toggled:
  eng_plain  engine forward with HF vision tower and F.conv1d (only the token-fused reorganisation)
  eng_taps   + 4-tap fused conv;   eng_fv  + graph-capturable vision tower;   eng_full = both (+ CUDA graphs)
  eng_fp8    eng_full with FP8 W8A8 language-model linears;  eng_fp8_vis  + FP8 vision-tower linears
Per variant vs fp32: max / mean |dP(present)|, max |d typed prob|, box max-coordinate diff (px), IoU of boxes,
decision agreement (presence at 0.5 / typed argmax), and Acc@0.5 against the RefCOCO ground truth.
"""
import argparse, json
import numpy as np
import torch

from dv.model.v1forward import run_model_v1
from speed import workload as W
from speed import engine as EN
from speed.bench_glance import Runner, equiv_outputs, compare


def fp32_call_factory(R):
    """Runner.call replacement: shipped path, no autocast (weights already fp32)."""
    orig_autocast = torch.autocast

    class _NoAutocast:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def call(imgs, qss, Nv=None, keep_prefix=False):
        torch.autocast = _NoAutocast
        try:
            return Runner.call(R, imgs, qss, Nv=Nv)
        finally:
            torch.autocast = orig_autocast
    return call


def set_dtype(hf, dt):
    for n, p in hf.named_parameters():
        if "lora_" not in n:
            p.data = p.data.to(dt)


def diagnostic_compare(rows, reference):
    stats = compare(rows, reference)
    disagreements = []
    center_agreed = box_count = 0
    all_margins, flip_margins = [], []
    for i, (row, ref) in enumerate(zip(rows, reference, strict=True)):
        if row["t"] == "box":
            box_count += 1
            same = row["center_idx"] == ref["center_idx"]
            center_agreed += same
            if "center_margin" in ref:
                all_margins.append(ref["center_margin"])
                if not same:
                    flip_margins.append(ref["center_margin"])
            a, b = row["p"], ref["p"]
            agree = (a >= .5) == (b >= .5)
        else:
            a, b = row["probs"], ref["probs"]
            agree = np.argmax(a) == np.argmax(b)
        if not agree:
            disagreements.append(dict(row=i, question_type=row["t"], actual=a, reference=b))
    if all_margins:
        quantiles = (0, 10, 25, 50, 75, 90, 100)
        stats["reference_margin_quantiles"] = dict(zip(quantiles, np.percentile(all_margins, quantiles).tolist()))
        stats["reference_exact_ties"] = all_margins.count(0.)
        stats["flipped_center_margins"] = sorted(flip_margins)
    return dict(stats, decision_agreed=len(rows) - len(disagreements), count=len(rows),
                center_agreed=center_agreed, box_count=box_count, disagreements=disagreements)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prep", default="data/prep/eval_refcoco_val.jsonl")
    ap.add_argument("--image_root", default="/content/data/coco/train2014")
    ap.add_argument("--n_img", type=int, default=60)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = Runner(a)
    R.trace_centers = True
    imgs = W.load_coco(a.prep, a.image_root, a.n_img)
    rows = {}
    rows["ref_lora"] = equiv_outputs(R, imgs, a.n_img)
    set_dtype(R.net.hf, torch.float32)
    import dv.model.v1forward as V
    R.call = fp32_call_factory(R)
    rows["fp32"] = equiv_outputs(R, imgs, a.n_img)
    del R.call
    set_dtype(R.net.hf, torch.bfloat16)
    rows["ref_lora_again"] = equiv_outputs(R, imgs, a.n_img)       # sanity: dtype round trip is exact
    R.set_mode("ref_merged")
    rows["ref_merged"] = equiv_outputs(R, imgs, a.n_img)
    for name, taps, fv, graphs in (("eng_plain", False, False, False), ("eng_taps", True, False, False),
                                   ("eng_fv", False, True, False), ("eng_full", True, True, True)):
        EN.USE_TAPS = taps
        R.eng = EN.GlanceEngine(R.net, R.heads, graphs=graphs, fast_vision=fv)
        R.eng.trace_centers = True
        R.mode = "eng_x"
        rows[name] = equiv_outputs(R, imgs, a.n_img)
    EN.USE_TAPS = True
    EN.fuse_weights_(R.net.lm, R.net.layer_types)
    R.eng = EN.GlanceEngine(R.net, R.heads, graphs=True, fast_vision=True, fast_ops=True)
    R.eng.trace_centers = True
    rows["eng_fused"] = equiv_outputs(R, imgs, a.n_img)
    EN.fuse_weights_(R.net.lm, R.net.layer_types, keep_ab_separate=True)
    R.n_fp8 = EN.fp8_language_model_(R.net.lm)
    R.eng = EN.GlanceEngine(R.net, R.heads, graphs=True, fast_vision=True, fast_ops=True)
    R.eng.trace_centers = True
    rows["eng_fp8"] = equiv_outputs(R, imgs, a.n_img)
    EN.fp8_language_model_(R.net.visual, skip=("patch_embed",))       # + FP8 vision tower
    R.eng = EN.GlanceEngine(R.net, R.heads, graphs=True, fast_vision=True, fast_ops=True)
    R.eng.trace_centers = True
    rows["eng_fp8_vis"] = equiv_outputs(R, imgs, a.n_img)
    res = dict(gpu=torch.cuda.get_device_name(), n_img=a.n_img, n_rows=len(rows["fp32"]),
               n_box_gt=sum("gt" in r for r in rows["fp32"]), vs_fp32={}, vs_ref_lora={})
    for k, v in rows.items():
        if k != "fp32":
            res["vs_fp32"][k] = diagnostic_compare(v, rows["fp32"])
        if k != "ref_lora":
            res["vs_ref_lora"][k] = diagnostic_compare(v, rows["ref_lora"])
        print(k, "vs fp32", json.dumps(res["vs_fp32"].get(k)), flush=True)
    res["acc50_fp32"] = res["vs_fp32"]["ref_lora"]["acc50_ref"]
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
