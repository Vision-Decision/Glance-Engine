"""Demo API: one text query against many images / frames, per-image answers + a calibrated set answer, with latency.

    from scan.api import Scanner
    sc = Scanner.load("/content/ckpt/v1/ckpt_final.pt")             # merged LoRA, fused ops, CUDA graphs (bf16)
    sc.calibrate(neg_scans)                                          # optional: list of per-frame score arrays of
                                                                     # scans known NOT to contain their query
    r = sc.scan("a person falling down", frames, mode="cached")     # mode: cached | fresh | query_first
    r = sc.scan("the person opens the fridge", clips, readout="short")  # clips: lists of frames; short scan readout
    r["images"][i]  -> {p_present, box [x1,y1,x2,y2] in 0..1, p_iou50}
    r["set"]        -> {any (bool), none_conformal (bool, if calibrated), argmax, top (BH-selected indices),
                        first (first frame over the calibrated threshold), count_frames, count_events, curve}
    r["latency_ms"] -> {encode, query, total}
    r2 = sc.scan("what is he holding", frames, mode="cached")       # same frames: bank reused, query only

Images are PIL images (any size; resized to <= max_vis visual tokens, sides multiples of 32). For video, pass frames
in time order. 'cached' keeps the frames' bank keyed by id(list) + length (re-encodes only for a new list).
Set answers (scan/setlevel.py): without calibration, 'any' is the naive per-image rule (some P(present) > .5; its
false alarm grows with N); after calibrate(), 'none_conformal' has P(false alarm on a new negative scan) <= alpha.
"""
import time
import numpy as np
import torch
from PIL import Image

from dv.data.dataset import resize_dims
from scan import setlevel as SL
from scan.engine import _sync


class Scanner:
    def __init__(self, eng, max_vis=256, alpha=0.1, q=0.1):
        self.eng, self.max_vis, self.alpha, self.q = eng, max_vis, alpha, q
        self._bank_key, self._bank = None, None
        self.none_rule = self.select_rule = self.first_rule = None

    @classmethod
    def load(cls, ckpt, fp8=False, w12=None, **kw):
        from scan.pilot_charades import load_glance
        eng, _ = load_glance(ckpt, fp8=fp8, w12=w12)
        eng.graphs = True
        eng.fv.graphs = True
        return cls(eng, **kw)

    def calibrate(self, neg_scans, pos_scans=None, onsets=None):
        """neg_scans: list of per-frame presence-logit arrays from scans without the target (same N distribution)."""
        self.none_rule = SL.MaxStatByN(self.alpha).fit(neg_scans) if len({len(s) for s in neg_scans}) > 1 else \
            SL.MaxStat(self.alpha).fit(neg_scans)
        self.select_rule = SL.Select(self.q).fit(np.concatenate(neg_scans))
        if pos_scans is not None:
            self.first_rule = SL.FirstOccurrence(self.alpha).fit(list(neg_scans) + list(pos_scans),
                                                                [None] * len(neg_scans) + list(onsets))
        return self

    def _prep(self, images):
        from scan.engine import is_clip
        # clips are passed through (the video processor resizes each 2-frame group to eng.clip_max_vis tokens)
        return [im if is_clip(im) else im.convert("RGB").resize(resize_dims(*im.size, self.max_vis, 2.0), Image.BICUBIC)
                for im in images]

    @staticmethod
    def _question(query, readout):
        if readout == "box":       # images: presence + box + P(IoU>=.5)
            return dict(type="box", text=query)
        if readout == "short":     # scan readout, ~query + 4 tokens (needs a model trained for it: scan/distill_*)
            return dict(type="noul", instr=query, block="short")
        return dict(type="noul", instr=query)          # "yes": the full yes/no block

    @torch.no_grad()
    def scan(self, query, images, mode="cached", readout="box"):
        """images: PIL images, or clips (lists of PIL frames, time order). readout: box | yes | short."""
        q = self._question(query, readout)
        t0 = time.perf_counter()
        lat = {}
        if mode == "cached":
            key = (id(images), len(images), self.max_vis)
            if key != self._bank_key:
                te = time.perf_counter()
                self._bank = self.eng.encode(self._prep(images), chunk=64)
                _sync()
                lat["encode"] = (time.perf_counter() - te) * 1e3
                self._bank_key = key
            T = {}
            out = self.eng.query(self._bank, q, timings=T)
            lat["query"] = T["total"]
        else:
            T = {}
            out = self.eng.fresh(self._prep(images), q, query_first=(mode == "query_first"), timings=T)
            lat.update(prep=T["prep"], vision=T["vision"], lm=T["lm"])
        lat["total"] = (time.perf_counter() - t0) * 1e3
        if readout == "box":
            s = out["presence_logit"].numpy()
            p = 1 / (1 + np.exp(-s))
            per = [dict(p_present=round(float(p[i]), 4), box=[round(float(v), 4) for v in out["box_pred"][i]],
                        p_iou50=round(float(torch.sigmoid(out["iou_logit"][i])), 4)) for i in range(len(s))]
        else:
            tl = out["typed_logits"]
            s = (tl[:, 1] - tl[:, 0]).numpy()
            p = 1 / (1 + np.exp(-s))
            per = [dict(p_yes=round(float(x), 4)) for x in p]
        st = dict(any=bool((p > 0.5).any()), argmax=int(s.argmax()), curve=[round(float(x), 4) for x in p],
                  count_frames_naive=int((p > 0.5).sum()))
        if self.none_rule is not None:
            st["none_conformal"] = not self.none_rule.alarm(s)
            sel = self.select_rule(s)
            st["top"] = [int(i) for i in sel]
            st["count_frames"] = len(sel)
            st["count_events"] = SL.count_events(sel)
        if self.first_rule is not None:
            st["first"] = self.first_rule(s)
        return dict(images=per, set=st, latency_ms={k: round(v, 2) for k, v in lat.items()}, mode=mode, n=len(s))
