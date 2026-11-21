"""Print the markdown tables of outputs/video_p1/summary.md from the raw JSON (no GPU needed).
  python -m speed.video_p1.report outputs/video_p1/raw"""
import json
import os
import sys


def load(d, name):
    p = os.path.join(d, name)
    return json.load(open(p)) if os.path.exists(p) else None


def m2_table(runs):
    print("| 底座 | 精度 | 输入 | token | CUDA graph P50 | 无 graph P50 |")
    print("|---|---|---:|---:|---:|---:|")
    for m, d in runs:
        t = {(r["prec"], r["graphs"], r["side"], r["mode"]): r for r in d["m2"]}
        for prec in ("bf16", "fp8"):
            for side in (224, 256, 448):
                for mode in ("single", "sliding_pair"):
                    g, e = t.get((prec, True, side, mode)), t.get((prec, False, side, mode))
                    if not g:
                        continue
                    lab = "单帧" if mode == "single" else "滑动双帧"
                    print(f"| {m} | {prec.upper()} | {side}² {lab} | {g['tokens']} | {g['p50']:.2f} | {e['p50']:.2f} |")
        ref = t.get(("bf16", False, 256, "hf_eager_reference"))
        if ref:
            print(f"| {m} | BF16 | 256² HF 原生 eager | 64 | — | {ref['p50']:.2f} |")


def m3_table(runs):
    print("| 底座 | 帧 token | 问题块 | token 总数 | BF16 graph | BF16 无 graph | FP8 graph | FP8 无 graph |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for m, d in runs:
        t = {(r["prec"], r["T"], r["M"], r["graphs"]): r.get("p50") for r in d["m3"]}
        for T in (64, 128, 256):
            for M in (0, 8, 32):
                c = [t.get((p, T, M, g)) for p in ("bf16", "fp8") for g in (True, False)]
                f = lambda x: "—" if x is None else f"{x:.2f}"
                print(f"| {m} | {T} | {M} | {T + 20 * M} | " + " | ".join(f(x) for x in c) + " |")


def m4_table(runs):
    print("| 底座 | KV 窗口 | KV/路 | 全注意力层数 | 带 mask SDPA | 拆分 flash+局部 | 纯窗口 flash 下界 | 整步 graph（拆分） | 整步 graph（mask） |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m, d in runs:
        for r in d["m4"]:
            print(f"| {m} | {r['W'] / 1024:.0f}k | {r['kv_bytes_per_stream'] / 2**20:.0f} MiB | {r['n_attn_layers']} | "
                  f"{r['attn_masked_per_frame_ms']:.2f} | {r['attn_split_per_frame_ms']:.2f} | "
                  f"{r['attn_window_kernel_per_frame_ms']:.2f} | {r['step_graph_p50']:.2f} | {r['step_graph_masked_p50']:.2f} |")


def m9_table(rows):
    print("| 配置 | 底座 | 路数 | 端到端 P50 | 端到端 P99 | 服务时间 P50 | 解码 | 预处理 | ViT | LM+头 | 跟得上 30fps |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for m, r in rows:
        if "error" in r:
            print(f"| {r.get('config')} | {m} | {r.get('B')} | 错误: {r['error'][:60]} |")
            continue
        s = r["stages"]
        e2e = (f"{r['e2e']['p50']:.1f} | {r['e2e']['p99']:.1f}" if r["keeps_up"] else "积压 | 积压")
        print(f"| {r['config']} | {m} | {r['B']} | {e2e} | {r['service']['p50']:.1f} | "
              f"{s['decode']['p50']:.2f} | {s['prep']['p50']:.2f} | {s['vit']['p50']:.2f} | {s['lm_heads']['p50']:.2f} | "
              f"{'是' if r['keeps_up'] else '否'} |")


def profile_table(runs):
    print("| 底座 | 形状 | GEMM | 注意力 | GDN | 逐元素/其他 | kernel 数/步 |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for m, d in runs:
        for k, v in sorted(d.items()):
            if not k.startswith("profile_"):
                continue
            ms = v["ms_per_step"]
            print(f"| {m} | {k[8:]} | {ms.get('gemm', 0):.2f} | {ms.get('attention', 0):.2f} | {ms.get('gdn', 0):.2f} | "
                  f"{ms.get('elementwise/other', 0):.2f} | {v['kernels_per_step']:.0f} |")


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "outputs/video_p1/raw"
    runs = [(m, load(d, f"m1234_{m}.json")) for m in ("0.8B", "4B")]
    runs = [(m, x) for m, x in runs if x]
    prof = [(m, load(d, f"profile_{m}.json") or x) for m, x in runs]
    print("## M2\n"); m2_table(runs)
    print("\n## M3\n"); m3_table(runs + [(m, x) for m, x in [("2B", load(d, "m39_2B.json"))] if x and "m3" in x])
    print("\n## profile\n"); profile_table(prof)
    print("\n## M4\n"); m4_table(runs)
    rows = []
    for m, f in (("4B", "m9_4B.json"), ("4B", "m9_4B_extra.json"), ("2B", "m39_2B.json"), ("0.8B", "m9_0.8B.json"),
                 ("0.8B", "m9_0.8B_extra.json")):
        x = load(d, f)
        if x:
            rows += [(m, r) for r in x.get("m9", [])]
    rows.sort(key=lambda t: (["4B", "2B", "0.8B"].index(t[0]), t[1].get("config", ""), t[1].get("B", 0)))
    print("\n## M9\n"); m9_table(rows)
    m5 = load(d, "m5.json")
    if m5:
        print("\n## M5\n")
        print("| 后端 | 解码 P50 | 解码 P99 | 吞吐 fps | 预处理 256² P50 | 预处理 448² P50 |")
        print("|---|---:|---:|---:|---:|---:|")
        for k in ("cuda", "cuda_beta", "cpu"):
            r = m5.get(k)
            if r and "decode" in r:
                print(f"| {k} | {r['decode']['p50']:.2f} | {r['decode']['p99']:.2f} | {r['decode_throughput_fps']:.0f} | "
                      f"{r['prep']['256x256']['p50']:.2f} | {r['prep']['448x448']['p50']:.2f} |")
        for B, r in m5.get("multi_stream_decode_prep_256", {}).items():
            print(f"| {B} 路串行解码+预处理 | {r['p50']:.2f} | {r['p99']:.2f} | | | |")


if __name__ == "__main__":
    main()
