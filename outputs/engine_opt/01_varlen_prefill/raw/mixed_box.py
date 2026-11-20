"""Trace the failing raw box comparisons on the mixed-length scan fixture with fixed vision features."""
import json
from pathlib import Path

from dv.data.dataset import resize_dims
from scan.varlen import packed_forward

src = Path("scan/test_scan_equivalence.py").read_text()
exec(compile(src[:src.index("masks = ")], "scan/test_scan_equivalence.py", "exec"))
results = {}
keys = ("presence_logit", "iou_logit", "box_pred")
with torch.no_grad():
    for mode in ("causal", "block_bidir"):
        cfg = Config().update(dict(task="v1", attn_mask=mode, max_vis_tokens=640, grad_ckpt=False))
        net = SharedPrefixLM(hf, cfg).eval()
        tok = ChatTok(proc, cfg)
        eng = ScanEngine(net, heads, tok, proc, graphs=False, fast_ops=False)
        ims = [im.resize(resize_dims(*im.size, cfg.max_vis_tokens, 2.0), Image.BICUBIC) for im in images[:2]]
        mixed = [ims[0].resize((160, 128)), ims[1].resize((64, 64)), ims[0].resize((160, 128))]
        q = QS[0]
        batch, tensors, host = eng.build(mixed, [[q]] * len(mixed))
        feats = eng.feats(batch)
        out = packed_forward(eng, FreshPack(batch,eng.dev), feats)
        out = {k:out[k].float().cpu() for k in keys}
        ref = reference(net, heads, tok, mixed, q)
        legacy = eng._core(eng.to_device(tensors, host, feats), host)[0]
        variants = dict(reference=ref, separate=eng.separate(mixed, q), legacy=legacy)
        single = {key:[] for key in ("separate_fixed","packed_fixed","packed_rerun_vision")}
        for im, feat in zip(mixed, feats.split(batch["p_img_mask"].sum(1).tolist())):
            one, e, h = eng.build([im], [[q]])
            single["separate_fixed"].append(eng._core(eng.to_device(e, h, feat), h)[0])
            single["packed_fixed"].append(packed_forward(eng, FreshPack(one, eng.dev), feat))
            single["packed_rerun_vision"].append(eng.fresh([im], q))
        for tag, rows in single.items():
            variants[tag] = {k:torch.cat([r[k][:1].float().cpu() for r in rows]) for k in keys}
        row = {}
        for tag, variant in variants.items():
            data = {k: variant[k][:3].float().cpu() for k in keys}
            row[tag] = dict(relative_by_output={k:rel(out[k],data[k]) for k in keys},
                            max_dP=(out["presence_logit"].sigmoid()-data["presence_logit"].sigmoid()).abs().max().item())
        results[mode] = row
Path("/content/out/scan/item1_mixed_box.json").write_text(json.dumps(results, indent=1))
