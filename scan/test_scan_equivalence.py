"""Exactness of the scan engine: one query over N images in one forward == each image run alone.

Checks (tiny random Qwen3.5 config, fp32, CPU by default; --real uses the real weights + a checkpoint on GPU, bf16):
  cached   ScanEngine.encode (chunked, banks of different prefix lengths concatenated) + query()  vs  run_model_v1
           on each (image, question) alone (dv.model reference path)
  fresh    ScanEngine.fresh (all N images + blocks in one call)                                  vs  same reference
  hf       (causal mask only) the plain HF forward over [prefix + block] with the box head applied to its hidden
           states                                                                                  vs  cached
  isolate  changing image k changes only answer k
  repeat   querying the same bank twice gives identical answers (the cached state is not mutated)
  qfirst   query-first prefix: fresh() == each image alone with the same prefix

python -m scan.test_scan_equivalence [--real --ckpt PATH] [--n 6]
python -m scan.test_scan_equivalence --real --ckpt PATH --measure-reference noise.json

The real suite checks presence and typed decisions, probability deltas and normalized
box coordinates. The calibration mode compares only equivalent shipped-reference layouts.
Both tiny FP32 masks retain the original relative tolerance.
"""
import argparse, itertools, json, os, sys
from contextlib import nullcontext
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor, AutoModelForImageTextToText

from dv.config import Config
from dv.data.mixed import ChatTok, collate_mixed
from dv.model.backbone import SharedPrefixLM
from dv.model.build import V1Heads
from dv.model.v1forward import run_model_v1
import speed.engine as SE
from scan.engine import ScanEngine, make_item, make_clip_item
from scan.equivalence_limits import BF16_LIMIT, CENTER_LIMIT, MIXED_BOX_LIMIT
from scan.varlen import FreshPack

ap = argparse.ArgumentParser()
ap.add_argument("--model_id", default="Qwen/Qwen3.5-4B")
ap.add_argument("--real", action="store_true")
ap.add_argument("--ckpt", default="")
ap.add_argument("--n", type=int, default=6)
ap.add_argument("--mask", default="both", choices=["both", "causal", "block_bidir"])
ap.add_argument("--fused-prefill", choices=("reference", "optimized"),
                help="also run the suite on a separate merged BF16 engine with either packed LM implementation")
ap.add_argument("--w12", help="test an offline W12 model with optimized merged prefill")
ap.add_argument("--measure-reference", metavar="JSON", help="measure only shipped BF16 reference layout spread")
ap.add_argument("--calibration-head", choices=("blocks", "oracle"), default="blocks",
                help="measure the production box head or its independent FP64-dot oracle on reference hidden states")
ap.add_argument("--verify-head-layout", choices=("blocks", "tiled"), help="replay the same hidden inputs through the engine head alone and batched")
ap.add_argument("--probe-final-norm", action="store_true", help="calibration control retaining FP32 final RMSNorm outputs")
ap.add_argument("--probe-residual", action="store_true", help="calibration control with FP32 LM residual streams and BF16 mixers")
ap.add_argument("--probe-fp32", action="store_true", help="calibration control retaining FP32 for the whole backbone")
ap.add_argument("--inject", choices=("fault", "small_fault"), help="real-suite negative control")
a = ap.parse_args()
if (a.verify_head_layout or a.probe_final_norm or a.probe_fp32 or a.probe_residual) and not a.measure_reference:
    ap.error("precision controls require --measure-reference")
if a.inject and (not a.real or a.measure_reference):
    ap.error("--inject requires the normal real-weight suite")
if a.w12 and a.fused_prefill != "optimized":
    ap.error("--w12 requires --fused-prefill optimized")
if a.fused_prefill and not a.real:
    ap.error("--fused-prefill needs --real")
if a.measure_reference and not a.real:
    ap.error("--measure-reference needs --real")
if a.real and not torch.cuda.is_available():
    ap.error("--real needs CUDA")
torch.manual_seed(0)
dev = "cuda" if (a.real and torch.cuda.is_available()) else "cpu"
proc = AutoProcessor.from_pretrained(a.model_id)

if a.real:
    from dv.model.build import build, load_ckpt
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg0 = Config().update({k: v for k, v in sd["cfg"].items() if k in Config.__dataclass_fields__})
    net0, heads, _ = build(cfg0, dev)
    load_ckpt(a.ckpt, net0, heads)
    hf = net0.hf
    if a.probe_residual:
        # Preserve the hidden-state residual through every addition. The projections
        # still run under shipped BF16 autocast and the rotary tables stay BF16.
        net0.lm.embed_tokens.register_forward_hook(lambda module, inputs, out: out.float())
        net0.lm.rotary_emb.register_forward_hook(
            lambda module, inputs, out: tuple(part.bfloat16() for part in out))
    if a.probe_fp32:
        hf.float()
        # The shipped reference enters its own BF16 autocast context in run_model_v1.
        # Scope this override to the standalone calibration process, as in speed.diag_equiv.
        original_autocast = torch.autocast
        def no_autocast(*args, **kwargs):
            kwargs["enabled"] = False
            return original_autocast(*args, **kwargs)
        torch.autocast = no_autocast
    if a.probe_final_norm:
        net0.lm.norm.register_forward_pre_hook(lambda module, inputs: (inputs[0].float(),))
    if a.inject == "fault":
        init = FreshPack.__init__
        def shifted_pack(self, *args):
            init(self, *args)
            if self.ids.is_cuda:
                self.heads["q_last_packed"] -= 1
        FreshPack.__init__ = shifted_pack
else:
    c = AutoConfig.from_pretrained(a.model_id)
    t = c.text_config
    t.hidden_size, t.intermediate_size, t.num_hidden_layers = 128, 256, 8
    t.layer_types = t.layer_types[:8]
    t.num_attention_heads, t.num_key_value_heads, t.head_dim = 4, 2, 32
    t.linear_num_key_heads, t.linear_num_value_heads, t.linear_key_head_dim, t.linear_value_head_dim = 2, 4, 16, 16
    v = c.vision_config
    v.depth, v.hidden_size, v.intermediate_size, v.num_heads, v.out_hidden_size = 1, 64, 128, 2, 128
    hf = AutoModelForImageTextToText.from_config(c, dtype=torch.float32).to(dev)
    SE.USE_TAPS = False          # torch.compile'd conv taps are a GPU fast path; CPU uses F.conv1d (reference op)
hf.eval()

rng = np.random.RandomState(0)
sizes = [(320, 240), (200, 300), (256, 256), (320, 240), (160, 480), (288, 224), (224, 224), (192, 320)][: a.n]
images = [Image.fromarray((rng.rand(H, W, 3) * 255).astype("uint8")) for W, H in sizes]
QS = [dict(type="box", text="the red car on the left"),
      dict(type="noul", instr="Is there a person falling down in the image?"),
      dict(type="choice", instr="What is shown?", options=["a car", "a dog", "nothing"])]

errs, ok = {}, True


def rel(x, y):
    x, y = x.float(), y.float()
    fin = torch.isfinite(y)
    return ((x - y).abs()[fin].max() / y.abs()[fin].max().clamp(min=1e-6)).item() if fin.any() else 0.0


def reference(net, heads, tok, images, q, query_first=None, clip_kw=None, group=1, pad_to=64, traces=None):
    """Shipped forward; the canonical result uses one example at a time and its default padding."""
    rows = []
    for start in range(0, len(images), group):
        items = [make_clip_item(tok, proc, im, [q], query_first=query_first, **clip_kw) if clip_kw else
                 make_item(tok, proc, im, [q], query_first) for im in images[start:start + group]]
        b = collate_mixed(items, tok, pad_to=pad_to)
        b.pop("meta", None)
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
        geometry = []
        control = nullcontext()
        if a.real and q["type"] == "box":
            from scan.readout_probe import reference_hook
            calibration_head = a.calibration_head if a.measure_reference else "oracle"
            control = heads.box.register_forward_hook(reference_hook(geometry, traces, a.verify_head_layout, calibration_head))
        with torch.no_grad(), control:
            r = run_model_v1(net, heads, b, train=False)
        if q["type"] == "box":
            row = {k: r["box"][k].float().cpu() for k in ("presence_logit", "box_pred", "iou_logit")}
            if a.real:
                row["geometry"] = geometry
                row["center_idx"] = r["box"]["pred_idx"].cpu()
                # Per-image widths vary with padding. Keep logits ragged for the selection error.
                row["scores"] = [scores[mask].cpu() for scores, mask in zip(r["box"]["center_logits"], b["vis_valid"])]
            rows.append(row)
        else:
            rows.append(dict(typed_logits=r["typed_logits"].float().cpu()))
    result = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        result[key] = list(itertools.chain.from_iterable(values)) if key in ("scores", "geometry") else torch.cat(values)
    return result


def readout_stats(out, ref):
    """Maximum absolute errors in output units; decisions use the public threshold or valid-label argmax."""
    errors, decisions = {}, []
    for key, target in ref.items():
        if key in ("center_idx", "scores", "geometry"):
            continue
        actual = out[key]
        if key == "typed_logits":
            actual = actual[:, :target.shape[1]]
        assert actual.shape == target.shape, (key, actual.shape, target.shape)
        if key == "typed_logits":
            # Non-reference control paths pad invalid labels with -inf.
            assert torch.isneginf(actual).equal(torch.isneginf(target))
            x, y = actual.softmax(-1), target.softmax(-1)
            decisions.append(x.argmax(-1) == y.argmax(-1))
            errors["typed_prob"] = float((x-y).abs().max())
        elif key in ("presence_logit", "iou_logit"):
            errors[key] = float((actual-target).abs().max())
            x, y = actual.sigmoid(), target.sigmoid()
            errors[key.replace("logit", "prob")] = float((x-y).abs().max())
            if key == "presence_logit":
                decisions.append((x >= 0.5) == (y >= 0.5))
        elif key == "box_pred":
            errors[key] = float((actual-target).abs().max())
    if "center_idx" in ref:
        idx = out["center_idx"].long()
        assert idx.shape == ref["center_idx"].shape
        if "scores" in ref:
            assert all(0 <= int(i) < len(scores) for scores, i in zip(ref["scores"], idx, strict=True))
        same = idx == ref["center_idx"]
        if same.any():
            errors["same_center_box"] = float((out["box_pred"][same] - ref["box_pred"][same]).abs().max())
            errors["same_center_iou"] = float((out["iou_logit"][same].sigmoid() - ref["iou_logit"][same].sigmoid()).abs().max())
        if "geometry" in ref:
            from scan.readout_probe import reference_geometry
            expected_box, expected_iou = reference_geometry(ref, idx)
            errors["patch_box"] = float((out["box_pred"] - expected_box).abs().max())
            errors["patch_iou"] = float((out["iou_logit"].sigmoid() - expected_iou).abs().max())
        if "scores" in ref:
            errors["center_regret"] = max(float(s.max() - s[i]) for s, i in zip(ref["scores"], idx, strict=True))
    assert all(np.isfinite(v) for v in errors.values()), errors
    decisions = torch.cat(decisions)
    return errors, int(decisions.sum()), decisions.numel()


def measure_reference(mode, net, heads, tok, fixtures, result):
    """Pairwise envelope, including both unpadded singletons and heterogeneous batches."""
    for name, units, question, kwargs in fixtures:
        layouts, traces = {}, {}
        box = question["type"] == "box"
        for group, pad in ((1, 64), (1, 1), (2, 64), (len(units), 64), (len(units), 1)):
            label = f"batch{group}_pad{pad}"
            if box:
                traces[label] = []
            layouts[label] = reference(net, heads, tok, units, question, **kwargs,
                                       group=group, pad_to=pad, traces=traces.get(label))
        spreads = {}
        for (first, a_out), (second, b_out) in itertools.combinations(layouts.items(), 2):
            errors, agree, count = readout_stats(a_out, b_out)
            entry = dict(errors=errors, agreed=agree, count=count)
            if box:
                from scan.readout_probe import trace_pair
                # Regret is directional. Calibrate both choices against the opposite layout.
                reverse, _, _ = readout_stats(b_out, a_out)
                for key in ("center_regret", "patch_box", "patch_iou"):
                    errors[key] = max(errors[key], reverse[key])
                entry["center_agreed"] = int((a_out["center_idx"] == b_out["center_idx"]).sum())
                entry["stages"] = trace_pair(heads.box, traces[first], traces[second])
            spreads[first + "_" + second] = entry
            for k, v in errors.items():
                result["max"][k] = max(v, result["max"].get(k, 0.))
        result["fixtures"][mode + "/" + name] = spreads
        if box:
            from scan.readout_probe import summarize_rows
            result.setdefault("margins", {})[mode + "/" + name] = {k: summarize_rows(v) for k, v in traces.items()}
    with open(a.measure_reference, "w") as f:
        json.dump(result, f, indent=2)


decision_counts = [0, 0]
center_counts = [0, 0]


def compare(tag, out, ref, limits=None):
    global ok
    if not a.real:
        e = max(rel(out[k][:, :ref[k].shape[1]] if out[k].dim() > 1 else out[k], ref[k]) for k in ref)
        errs[tag] = e
        ok &= e < 1e-4
        return
    limits = (BF16_LIMIT | CENTER_LIMIT) if limits is None else (BF16_LIMIT | CENTER_LIMIT | limits)
    deltas, agree, count = readout_stats(out, ref)
    if "center_idx" in ref:
        matches = out["center_idx"] == ref["center_idx"]
        if limits["center_regret"] == 0 and not matches.all():
            print(f"FAIL {tag}/center_idx: mixed-prefix reference centers are layout-invariant")
            ok = False
        center_counts[0] += int(matches.sum())
        center_counts[1] += matches.numel()
    for k, v in deltas.items():
        errs[f"{tag}/{k}"] = v
        if k in ("presence_logit", "iou_logit"):
            continue
        if v > limits[k]:
            print(f"FAIL {tag}/{k}: {v:.6g} exceeds {limits[k]:.6g}")
            ok = False
    errs[tag + "/decision_agreement"] = agree / count
    decision_counts[0] += agree
    decision_counts[1] += count
    ok &= agree == count


masks = ["causal", "block_bidir"] if a.mask == "both" else [a.mask]
noise = {"max": {}, "fixtures": {}, "calibration_head": a.calibration_head}
for mode in masks:
    cfg = Config().update(dict(task="v1", attn_mask=mode, max_vis_tokens=64 if not a.real else 640, grad_ckpt=False))
    net = SharedPrefixLM(hf, cfg).eval()
    tok = ChatTok(proc, cfg)
    if not a.real:
        heads = V1Heads(net.hidden, cfg, hf.get_output_embeddings().weight, tok.label_token_groups()).to(dev).eval()
        for p in heads.parameters():          # non-trivial typed residual (zero-init in training)
            if p.dim() == 2 and p.shape[0] < 40:
                torch.nn.init.normal_(p, std=0.02)
    # resize like the real pipeline (multiples of 32 px)
    from dv.data.dataset import resize_dims
    ims = [im.resize(resize_dims(*im.size, cfg.max_vis_tokens, 2.0), Image.BICUBIC) for im in images]
    if a.measure_reference:
        mixed = [ims[0].resize((160, 128)), ims[1].resize((64, 64)), ims[0].resize((160, 128))]
        clips = [[Image.fromarray((rng.rand(96, 128, 3)*255).astype("uint8")) for _ in range(n)]
                 for n in (4, 4, 4, 4, 4, 6)]
        clip_kw = dict(clip_kw=dict(dt=0.5, max_vis=16))
        fixtures = [(name+"/"+q["type"], units, q, {}) for name, units in (("image", ims), ("mixed", mixed))
                    for q in QS]
        fixtures += [("clip/"+q["type"], clips, q, clip_kw) for q in QS[1:]]
        api_images = [im.resize(resize_dims(*im.size, 64, 2.0), Image.BICUBIC) for im in images]
        fixtures += [("api/box", api_images, dict(type="box", text="a dog"), {}),
                     ("image/qfirst", ims, QS[0], dict(query_first=QS[0]["text"])),
                     ("clip/qfirst", clips, QS[1], dict(clip_kw, query_first=QS[1]["instr"])),
                     ("clip/short", clips, dict(type="noul", instr="a person falling", block="short"), clip_kw)]
        measure_reference(mode, net, heads, tok, fixtures, noise)
        print("Reference layout maxima", mode, noise["max"], flush=True)
        continue
    if a.fused_prefill:
        from copy import deepcopy
        engine_net = deepcopy(net)
        SE.merge_lora_(engine_net.hf)
        SE.fuse_weights_(engine_net.lm, engine_net.layer_types)
        if a.w12:
            from speed.w12 import install
            install(engine_net.lm, a.w12)
        eng = ScanEngine(engine_net, heads, tok, proc, graphs=False, fast_ops=True, lm_prefill=a.fused_prefill == "optimized")
        assert eng.lm_prefill == (a.fused_prefill == "optimized")
    else:
        eng = ScanEngine(net, heads, tok, proc, graphs=False, fast_ops=False)
    eng.trace_centers = a.real
    bank = eng.encode(ims, chunk=2)                        # several banks, different prefix lengths -> concat
    for q in QS:
        ref = reference(net, heads, tok, ims, q)
        compare(f"{mode}/{q['type']}/cached_vs_alone", eng.query(bank, q), ref)
        fresh = eng.fresh(ims, q)
        if a.inject == "small_fault" and mode == "causal" and q["type"] == "box":
            # Localized postprocessing error in a regular fixture with a broad raw-coordinate gate.
            fresh["box_pred"][0, 0] += .04
        compare(f"{mode}/{q['type']}/fresh_vs_alone", fresh, ref)
        o1, o2 = eng.query(bank, q), eng.query(bank, q)
        repeat_key = f"{mode}/{q['type']}/repeat"
        fields = ("presence_logit", "box_pred", "iou_logit", "typed_logits")
        errs[repeat_key] = float(any(not torch.equal(o1[k], o2[k]) for k in fields))
        ok &= errs[repeat_key] == 0
    # Deliberately unequal prefixes in non-sorted order, with one repeated image at a different offset.
    mixed = [ims[0].resize((160, 128)), ims[1].resize((64, 64)), ims[0].resize((160, 128))]
    for q in QS:
        items = [make_item(tok, proc, im, [q]) for im in mixed]
        batch = collate_mixed(items, tok)
        pack = FreshPack(batch, torch.device("cpu"))
        assert len(set(pack.prefix.lengths)) == 2
        assert len(pack.ids) == batch["n_tokens"] == pack.full.cpu[-1].item()
        assert pack.ids[pack.img_rows].eq(tok.pad_img).all()
        assert pack.ids[pack.heads["q_last_packed"]].tolist() == [it["q_ids"][0][-1] for it in items]
        out = eng.fresh(mixed, q)
        ref = reference(net, heads, tok, mixed, q)
        refs = {"reference": ref, "separate": eng.separate(mixed, q)}
        eng.fresh_varlen = False
        refs["old"] = eng.fresh(mixed, q)
        eng.fresh_varlen = True
        limits = (BF16_LIMIT | MIXED_BOX_LIMIT) if q["type"] == "box" else BF16_LIMIT
        for name, other in refs.items():
            tag = f"{mode}/varlen_mixed/{q['type']}/{name}"
            compare(tag, out, {k: other[k] for k in ref if k not in ("geometry", "scores") or k in other}, limits)
            if a.real:
                continue  # compare() already checks every valid decision and probability.
            if q["type"] == "box":
                p, pr = (v["presence_logit"].sigmoid() for v in (out, other))
                agree = (p >= 0.5).eq(pr >= 0.5).all()
            else:
                width = ref["typed_logits"].shape[1]
                p, pr = (v["typed_logits"][:, :width].softmax(-1) for v in (out, other))
                agree = p.argmax(-1).eq(pr.argmax(-1)).all()
            errs[tag + "/max_dP"] = (p - pr).abs().max().item()
            errs[tag + "/decision_agreement"] = float(agree)
            ok &= bool(agree) and errs[tag + "/max_dP"] < 1e-4
        # Force more than one LM chunk and verify image order and the oversize-singleton case.
        eng._fresh_token_budget = lambda: pack.full.lengths[1]
        chunked = eng.fresh(mixed, q)
        del eng._fresh_token_budget
        assert eng.last_fresh_chunks == len(mixed)
        compare(f"{mode}/varlen_mixed/{q['type']}/chunked", chunked, {k: out[k] for k in ref if k not in ("geometry", "scores")}, limits)
    # isolation: perturb image 1, re-encode; only answer 1 may change
    q = QS[0]
    ims2 = list(ims)
    ims2[1] = Image.fromarray(255 - np.asarray(ims[1]))
    o_a, o_b = eng.query(bank, q), eng.query(eng.encode(ims2, chunk=2), q)
    d = (o_a["presence_logit"] - o_b["presence_logit"]).abs()
    errs[f"{mode}/isolate_others_maxdiff"] = float(torch.cat([d[:1], d[2:]]).max())
    errs[f"{mode}/isolate_changed"] = float(d[1])
    ok &= errs[f"{mode}/isolate_others_maxdiff"] < (1e-5 if not a.real else 1e-2) and errs[f"{mode}/isolate_changed"] > 0
    # query-first prefix (mode C)
    qf = "the red car on the left"
    compare(f"{mode}/qfirst_fresh_vs_alone", eng.fresh(ims, q, query_first=True), reference(net, heads, tok, ims, q, query_first=qf))
    # plain HF forward (only the causal block mask is an ordinary causal LM)
    if mode == "causal":
        out = eng.query(bank, q)
        diffs = []
        for i, im in enumerate(ims):
            it = make_item(tok, proc, im, [q])
            b = collate_mixed([it], tok)
            pv = b["p_valid"][0]
            ids = torch.cat([b["p_ids"][0][pv], b["q_ids"][0][b["q_valid"][0]]])[None].to(dev)
            mm = (ids == hf.config.image_token_id).int()
            with torch.no_grad():
                # The unmerged adapters have fp32 master weights. Match run_model_v1's bf16 forward.
                with eng._autocast():
                    h = hf.model(input_ids=ids, pixel_values=b["pixel_values"].to(dev),
                                 image_grid_thw=b["grid_thw"].to(dev), mm_token_type_ids=mm).last_hidden_state[0]
                vis = h[(ids[0] == hf.config.image_token_id)][None]
                n = vis.shape[1]
                r = heads.box(h[-1:], vis, torch.ones(1, n, dtype=torch.bool, device=dev),
                              b["centers"][:, :n].to(dev), torch.zeros(1, dtype=torch.long, device=dev))
            if a.real:
                fields = ("presence_logit", "box_pred", "iou_logit")
                raw = {k: r[k].float().cpu() for k in fields}
                raw.update(center_idx=r["pred_idx"].cpu(), scores=[r["center_logits"][0].float().cpu()])
                compare(f"causal/box/image{i}/cached_vs_hf_plain_forward",
                        {k: out[k][i:i + 1] for k in (*fields, "center_idx")}, raw)
                continue
            diffs.append(rel(out["presence_logit"][i:i + 1], r["presence_logit"].float().cpu()))
            diffs.append(rel(out["box_pred"][i:i + 1], r["box_pred"].float().cpu()))
        if not a.real:
            errs["causal/box/cached_vs_hf_plain_forward"] = max(diffs)
            ok &= max(diffs) < 1e-4

# clip units (Glance-Video native video prefix): 5 clips of 4 frames (2 temporal groups), one clip of 6 frames
    clips = [[Image.fromarray((rng.rand(96, 128, 3) * 255).astype("uint8")) for _ in range(4)] for _ in range(5)]
    clips.append([Image.fromarray((rng.rand(96, 128, 3) * 255).astype("uint8")) for _ in range(6)])
    eng.dt, eng.clip_max_vis = 0.5, 16
    ckw = dict(dt=0.5, max_vis=16)
    cbank = eng.encode(clips, chunk=2)
    for q in QS[1:]:
        ref = reference(net, heads, tok, clips, q, clip_kw=ckw)
        compare(f"{mode}/clip/{q['type']}/cached_vs_alone", eng.query(cbank, q), ref)
        compare(f"{mode}/clip/{q['type']}/fresh_vs_alone", eng.fresh(clips, q), ref)
    qf = "a person falling"
    compare(f"{mode}/clip/qfirst_fresh_vs_alone", eng.fresh(clips, QS[1], query_first=True),
            reference(net, heads, tok, clips, QS[1], query_first=QS[1]["instr"], clip_kw=ckw))

if a.measure_reference:
    if a.verify_head_layout:
        for layouts in noise["margins"].values():
            for rows in layouts.values():
                for row in rows:
                    assert not any(row["frozen_replay_max_abs"].values()), row
    sys.exit(0)

# demo API smoke test (last engine): per-image answers + calibrated set answer
from scan.api import Scanner
sc = Scanner(eng, max_vis=64)
r1 = sc.scan("a red car", images, mode="cached")
r2 = sc.scan("a dog", images, mode="cached")                    # bank reused
r3 = sc.scan("a dog", images, mode="fresh")
negs = [np.random.RandomState(i).normal(-2, 1, len(images)) for i in range(50)]
sc.calibrate(negs)
r4 = sc.scan("a dog", images, mode="cached")
errs["api/cached_vs_fresh_p"] = max(abs(a["p_present"] - b["p_present"]) for a, b in zip(r2["images"], r3["images"]))
ok &= errs["api/cached_vs_fresh_p"] < (BF16_LIMIT["presence_prob"] if a.real else 1e-4) and "encode" not in r2["latency_ms"] and "none_conformal" in r4["set"]
rc = sc.scan("a person falling", clips, mode="cached", readout="short")
rf = sc.scan("a person falling", clips, mode="fresh", readout="short")
errs["api/clips_short_cached_vs_fresh"] = max(abs(x["p_yes"] - y["p_yes"]) for x, y in zip(rc["images"], rf["images"]))
ok &= errs["api/clips_short_cached_vs_fresh"] < (BF16_LIMIT["typed_prob"] if a.real else 1e-4) and len(rc["images"]) == len(clips)
if a.real:
    for first, second, field in ((r2, r3, "p_present"), (rc, rf, "p_yes")):
        decisions = [(x[field] >= .5) == (y[field] >= .5) for x, y in zip(first["images"], second["images"])]
        ok &= all(decisions)
        decision_counts[0] += sum(decisions)
        decision_counts[1] += len(decisions)
    print(f"Real-weight box-center comparisons: {center_counts[0]}/{center_counts[1]}")
    print(f"Real-weight decision comparisons: {decision_counts[0]}/{decision_counts[1]}")
print("api example:", {k: v for k, v in r4["set"].items() if k != "curve"}, r4["latency_ms"])
# vdm scan server end to end (HTTP, tiny model): /encode, /scan (curve/ever/when/count), /score_clips (vdm1 API).
# The server lives in Vision-Decision/glance-video (video/vdm/gv/scan_server.py); point SCAN_SERVER_DIR at that
# directory to include it. Without it this part is skipped (glance-video's tests run it).
SERVER_DIR = os.environ.get("SCAN_SERVER_DIR", "")
if not a.real and not SERVER_DIR:
    print("scan server check: skipped (set SCAN_SERVER_DIR=<glance-video>/video/vdm/gv to run it)")
if not a.real and SERVER_DIR:
    import base64, io, json, threading, urllib.request, urllib.error
    from http.server import ThreadingHTTPServer
    sys.path.insert(0, os.path.abspath(SERVER_DIR))
    import scan_server as SS
    from scan import video_ops as VO
    SS.ENG = eng
    eng.dt, eng.clip_max_vis = 0.5, 16

    def b64(im):
        b = io.BytesIO(); im.save(b, "JPEG", quality=90); return base64.b64encode(b.getvalue()).decode()
    cl_b64 = [[b64(f) for f in c[:4]] for c in clips]
    srv = ThreadingHTTPServer(("127.0.0.1", 0), SS.H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    def post(p, body):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(url + p, json.dumps(body).encode(),
                                                                           {"content-type": "application/json"})))
        except urllib.error.HTTPError as e:
            return json.load(e)
    r0 = post("/scan", dict(video="v1", query="a person falling"))
    times = [[i * 1.0, i * 1.0 + 1.5] for i in range(len(clips))]
    r1 = post("/encode", dict(video="v1", clips=cl_b64, times=times, dt=0.5))
    r2 = post("/scan", dict(video="v1", query="a person falling", ops=["curve", "ever", "when", "count"]))
    rng2 = np.random.RandomState(1)
    SS.CAL = VO.Calib(rng2.normal(-2, 1, 2000), [rng2.normal(-2, 1, len(clips)) for _ in range(60)])
    r3 = post("/scan", dict(video="v1", query="a person falling", ops=["curve", "ever", "when", "count"], alpha=0.1))
    r4 = post("/score_clips", dict(question="Is a person falling?", kind="yesno", dt=0.5, clips=cl_b64))
    r5 = post("/score_clips", dict(question="Is a person falling?", kind="yesno", dt=0.5, clips=cl_b64))
    # the server's clips are re-decoded JPEGs, so compare the server against itself + the direct engine on the same JPEGs
    dec = [[Image.open(io.BytesIO(base64.b64decode(f))).convert("RGB") for f in c] for c in cl_b64]
    direct = eng.fresh(dec, dict(type="noul", instr="Is a person falling?"))["typed_logits"]
    p_direct = torch.softmax(direct[:, :2], -1)[:, 1]
    errs["server/score_clips_vs_direct"] = float(max(abs(x - float(y)) for x, y in zip(r4["results"], p_direct)))
    ok &= ("error" in r0 and "unknown video" in r0["error"]) and r1["n"] == len(clips) and len(r2["curve"]) == len(clips)
    ok &= r2["ever"]["guarantee"] is None and "guarantee" in r3["ever"] and "intervals" in r3["when"] and "count" in r3["count"]
    ok &= (not r4["cached"]) and r5["cached"] and errs["server/score_clips_vs_direct"] < 1e-4
    print("server:", {k: r3[k] for k in ("ever", "count")}, "cached second call:", r5["cached"])
    srv.shutdown()

for k, v in errs.items():
    print(f"{k:45s} {v:.3e}")
print(f"N={len(images)} images, bank Lp ok")
print("SCAN_EQUIVALENCE_OK" if ok else "SCAN_EQUIVALENCE_FAIL")

if not ok:
    sys.exit(1)
