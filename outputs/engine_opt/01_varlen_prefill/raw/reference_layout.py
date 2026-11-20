"""Isolate layout sensitivity on the existing real-weight fixture. Run in the worktree on the G4."""
import json
from pathlib import Path
from unittest.mock import patch

# Reuse the exact seeded setup, reference function, images and trained head from the test.
src = Path("scan/test_scan_equivalence.py").read_text()
exec(compile(src[:src.index("masks = ")], "scan/test_scan_equivalence.py", "exec"))
from dv.data.dataset import resize_dims

result = {}
for mode in ("causal", "block_bidir"):
    cfg = Config().update(dict(task="v1", attn_mask=mode, max_vis_tokens=640, grad_ckpt=False))
    tok = ChatTok(proc, cfg)
    net = SharedPrefixLM(hf, cfg).eval()
    eng = ScanEngine(net, heads, tok, proc, graphs=False, fast_ops=False)
    ims = [im.resize(resize_dims(*im.size, cfg.max_vis_tokens, 2.0), Image.BICUBIC) for im in images]
    q = QS[0]
    scope = {}
    def record(name, fn):
        patches, margins = [], []
        def hook(module, args, out):
            patches.extend(out["pred_idx"].tolist())
            top2 = out["center_logits"].topk(2).values
            margins.extend((top2[:, 0] - top2[:, 1]).tolist())
        handle = heads.box.register_forward_hook(hook)
        try:
            out = fn()
        finally:
            handle.remove()
        scope[name] = dict(patches=patches, margins=margins,
                           **{k: out[k].flatten(1).tolist() if out[k].dim() > 1 else out[k].tolist()
                              for k in ("presence_logit", "box_pred", "iou_logit")})
    for bucket in (64, 1):
        with patch.dict(reference.__globals__, {"collate_mixed": lambda items, tok, fn=collate_mixed: fn(items, tok, pad_to=bucket)}):
            record(f"reference_pad{bucket}", lambda: reference(net, heads, tok, ims, q))
    for fv in ("fast", "hf"):
        if fv == "hf":
            eng.fv = None
        eng.fresh_varlen = False
        record(f"old_{fv}", lambda: eng.fresh(ims, q))
        eng.fresh_varlen = True
        record(f"packed_{fv}", lambda: eng.fresh(ims, q))
        record(f"single_{fv}", lambda: eng.separate(ims, q))
    result[mode] = scope
Path("/content/out/scan/item1_reference_layout.json").write_text(json.dumps(result, indent=1))
