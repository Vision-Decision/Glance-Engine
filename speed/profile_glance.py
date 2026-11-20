"""Profile one Glance call (1 image, N questions) per mode: wall time per stage, GPU kernel time by category.

python -m speed.profile_glance --ckpt /content/ckpt/v1/ckpt_final.pt --out OUT.json [--modes ref_lora,ref_merged,eng_eager,eng_graph]
"""
import argparse, collections, json, time
import torch
from torch.profiler import profile, ProfilerActivity

from speed import workload as W
from speed.bench_glance import Runner


def cat(name):
    n = name.lower()
    for k, c in (("gemm", "gemm"), ("cutlass", "gemm"), ("sm90", "gemm"), ("sm100", "gemm"), ("sm120", "gemm"),
                 ("cublas", "gemm"), ("nvjet", "gemm"), ("flash", "attention"), ("fmha", "attention"),
                 ("attention", "attention"), ("efficient", "attention"), ("chunk", "gdn(fla)"), ("fwd_kernel", "gdn(fla)"),
                 ("gated_delta", "gdn(fla)"), ("l2norm", "gdn(fla)"), ("solve_tril", "gdn(fla)"), ("conv", "conv"),
                 ("memcpy", "memcpy"), ("copy", "copy/cat"), ("cat", "copy/cat"), ("index", "index"), ("gather", "index"),
                 ("reduce", "reduce/norm"), ("norm", "reduce/norm"), ("softmax", "softmax")):
        if k in n:
            return c
    return "elementwise/other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prep", default="data/prep/eval_refcoco_val.jsonl")
    ap.add_argument("--image_root", default="/content/data/coco/train2014")
    ap.add_argument("--modes", default="ref_lora,ref_merged,eng_eager,eng_graph")
    ap.add_argument("--nq", default="1,8")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    a.modes_list = a.modes.split(",")
    R = Runner(a)
    imgs = W.load_coco(a.prep, a.image_root, 12)
    res = dict(gpu=torch.cuda.get_device_name(), modes={})
    for mode in a.modes_list:
        R.set_mode(mode)
        res["modes"][mode] = {}
        for nq in map(int, a.nq.split(",")):
            im, qs = [imgs[0]["img"]], [W.questions(imgs, 0, nq, "box")]
            for _ in range(5):
                R.call(im, qs, Nv=640)
            ts = [R.call(im, qs, Nv=640)[1] for _ in range(10)]
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                R.call(im, qs, Nv=640)
            by, n_k = collections.Counter(), 0
            for e in prof.key_averages():
                if e.device_type == torch.autograd.DeviceType.CUDA or getattr(e, "self_device_time_total", 0) > 0:
                    t = getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0)
                    if t > 0 and not e.key.startswith("aten::") and not e.key.startswith("cuda"):
                        by[cat(e.key)] += t / 1e3
                        n_k += e.count
            top = sorted(((getattr(e, "self_device_time_total", 0) / 1e3, e.count, e.key[:90]) for e in prof.key_averages()
                          if getattr(e, "self_device_time_total", 0) > 0 and not e.key.startswith("aten::")), reverse=True)[:25]
            d = dict(stages_ms={k: sorted(t[k] for t in ts)[len(ts) // 2] for k in ts[0]},
                     gpu_kernel_ms_by_category={k: round(v, 3) for k, v in by.most_common()},
                     gpu_kernel_ms_total=round(sum(by.values()), 3), n_kernel_launches=n_k, top_kernels=top)
            res["modes"][mode][f"N{nq}"] = d
            print(mode, nq, json.dumps(d["stages_ms"]), "gpu", d["gpu_kernel_ms_total"], "launches", n_k, flush=True)
            print("   ", json.dumps(d["gpu_kernel_ms_by_category"]), flush=True)
            for t in top[:12]:
                print("      %.3f ms  x%d  %s" % t, flush=True)
            json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
