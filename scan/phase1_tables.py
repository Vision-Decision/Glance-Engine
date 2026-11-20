"""Phase-1 tables (Glance scan vs vLLM 0.34) from the raw JSONs in outputs/scan/raw/<session>/.

python3 scan/phase1_tables.py outputs/scan/raw/scan-a > outputs/scan/phase1_tables.md
"""
import json, os, sys

D = sys.argv[1]
L = lambda f: json.load(open(os.path.join(D, f))) if os.path.exists(os.path.join(D, f)) else {}
g, ge = L("bench_g4.json"), L("bench_eqfresh.json")
vb, vf = L("vllm_bf16.json").get("modes", {}), L("vllm_fp8mtp.json").get("modes", {})
ms = lambda d, k: (d.get(k) or {}).get("total_ms")
f = lambda x: "—" if x is None else (f"{x:.0f}" if x >= 100 else f"{x:.1f}")
ratio = lambda a, b: "—" if (a is None or b is None) else f"{b / a:.1f}×"

print(f"GPU: {g.get('gpu')}; Glance v1 (bf16, merged LoRA, fused ops, CUDA graphs); vLLM {L('vllm_bf16.json').get('version')}"
      " (prefix caching, async scheduling, CUDA graphs). Medians, ms, from the query to all N answers on the host.\n")

print("## (A) cached clips: the video's clips were encoded once; one new yes/no question over N clips\n")
print("Clip = 4 frames (2 temporal groups x <= 96 visual tokens, 197 prefix tokens). vLLM: same clips as native video,"
      " every clip already asked one question (warm cache); max_tokens=1 with logprobs.\n")
print("| N clips | Glance | vLLM bf16 | vLLM FP8 | Glance speed-up vs best vLLM | vLLM cached tokens |\n|---|---|---|---|---|---|")
for N in (1, 8, 16, 32, 64, 128, 256):
    a = ms(g.get("clips", {}), f"f4_v96_N{N}_noul")
    b, c = ms(vb, f"clips_f4_v96_N{N}_yesno"), ms(vf, f"clips_f4_v96_N{N}_yesno")
    best = min([x for x in (b, c) if x is not None], default=None)
    ct = (vb.get(f"clips_f4_v96_N{N}_yesno") or {}).get("cached_prompt_tokens_mean")
    print(f"| {N} | {f(a)} | {f(b)} | {f(c)} | {ratio(a, best)} | {f(ct)} |")

print("\n## (A) cached images: one new question over N cached images\n")
print("| Tv | N | Glance box (box + presence + IoU) | Glance yes/no | vLLM bf16 yes/no | vLLM bf16 box | vLLM FP8+MTP yes/no | vLLM FP8+MTP box | vLLM cached tokens |\n|---|---|---|---|---|---|---|---|---|")
for Tv in (640, 256, 64):
    for N in (1, 8, 16, 32, 64, 128, 256):
        c = g.get("cached", {})
        row = [ms(c, f"Tv{Tv}_N{N}_box"), ms(c, f"Tv{Tv}_N{N}_noul"), ms(vb, f"cached_Tv{Tv}_N{N}_yesno"),
               ms(vb, f"cached_Tv{Tv}_N{N}_box"), ms(vf, f"cached_Tv{Tv}_N{N}_yesno"), ms(vf, f"cached_Tv{Tv}_N{N}_box")]
        ct = (vb.get(f"cached_Tv{Tv}_N{N}_yesno") or {}).get("cached_prompt_tokens_mean")
        if any(x is not None for x in row):
            print(f"| {Tv} | {N} | " + " | ".join(f(x) for x in row) + f" | {f(ct)} |")

print("\n## (B) fresh frames: N new frames + 1 question, end to end (Glance: box readout, unpadded prefill)\n")
print("Cells: Glance / vLLM bf16 yes-no (1 token) / vLLM bf16 box. Bold = Glance < 100 ms.\n")
Ns = (1, 2, 4, 8, 16, 32)
print("| Tv \\ N | " + " | ".join(map(str, Ns)) + " |\n|---" + "|---" * len(Ns) + "|")
for Tv in (64, 128, 256):
    cells = []
    for N in Ns:
        a = ms(ge.get("fresh", {}), f"Tv{Tv}_N{N}")
        s = f(a) if a is None or a >= 100 else f"**{f(a)}**"
        cells.append(f"{s} / {f(ms(vb, f'fresh_Tv{Tv}_N{N}_yesno'))} / {f(ms(vb, f'fresh_Tv{Tv}_N{N}_box'))}")
    print(f"| {Tv} | " + " | ".join(cells) + " |")
print("\n(Tv=640 fresh, padded run: N=1 43 ms, N=2 76 ms, N=4 145 ms.)")

print("\n## Clip encode cost and bank size (Glance)\n")
c = g.get("clips", {})
for k in ("f4_v96", "f4_v48", "f2_v96"):
    print(f"- {k}: {c.get(k + '_prefix_tokens')} prefix tokens, encode {f(c.get(k + '_encode_ms_per_clip'))} ms/clip (eager, batched), "
          f"bank {f(c.get(k + '_bank_MB_per_clip'))} MB/clip; N=64 query {f(ms(c, k + '_N64_noul'))} ms, N=256 {f(ms(c, k + '_N256_noul'))} ms")

print("\n## Exactness on GPU (bf16, real weights, real images) vs the shipped reference path, each image alone\n")
print("| set / query / path | max ΔP | mean ΔP | decision agreement |\n|---|---|---|---|")
for k, d in ge.get("equiv", {}).items():
    if "p_absdiff_max" in d:
        print(f"| {k} | {d['p_absdiff_max']:.4f} | {d['p_absdiff_mean']:.4f} | {d['decision_agree']:.3f} |")
    else:
        print(f"| {k} | {d['prob_absdiff_max']:.4f} | — | {d['decision_agree']:.3f} |")
