"""One request over many images with a Glance-X model; the same calls and result format as :class:`scan.api.Scanner`.

    scanner = glance.scanner(max_vis=640)               # a Glance-X model
    r = scanner.scan("a black cow", images)             # cached (image_first): first call on this list encodes
    r = scanner.scan("a red tractor", images)           # same list object: only the request blocks run
    r = scanner.scan("a black cow", images, mode="text_first")   # request read first, images read it (no bank)

Modes: ``cached`` (bank of image prefixes keyed by the list object and length; the request's modality stage runs
once for the whole bank), ``fresh`` (image_first without keeping a bank), ``text_first`` and ``joint`` (where the
checkpoint was trained in them). ``r["images"][i]`` adds ``boxes`` and ``box_scores`` (every target) to the v1
fields; ``r["set"]`` and ``calibrate`` are unchanged (``scan/setlevel.py``; scores are presence logits).
"""

from __future__ import annotations

import time

import numpy as np
import torch

from scan import setlevel as SL


class XScanner:
    def __init__(self, eng, max_vis=640, alpha=0.1, q=0.1, chunk=256):
        self.eng, self.max_vis, self.alpha, self.q, self.chunk = eng, max_vis, alpha, q, chunk
        self._bank_key, self._bank = None, None
        self.none_rule = self.select_rule = self.first_rule = None

    def calibrate(self, neg_scans, pos_scans=None, onsets=None):
        """neg_scans: per-image presence-logit arrays of scans without the target (as ``scan.api.Scanner``)."""
        self.none_rule = SL.MaxStatByN(self.alpha).fit(neg_scans) if len({len(s) for s in neg_scans}) > 1 else \
            SL.MaxStat(self.alpha).fit(neg_scans)
        self.select_rule = SL.Select(self.q).fit(np.concatenate(neg_scans))
        if pos_scans is not None:
            self.first_rule = SL.FirstOccurrence(self.alpha).fit(list(neg_scans) + list(pos_scans),
                                                                [None] * len(neg_scans) + list(onsets))
        return self

    def _prep(self, images):
        if any(isinstance(im, (list, tuple)) for im in images):
            raise ValueError("Glance-X scans images; clips need a Glance v1 model")
        return self.eng.fit_many(images, self.max_vis)

    def _sync(self):
        if self.eng.dev.type == "cuda":
            torch.cuda.synchronize(self.eng.dev)

    @torch.no_grad()
    def scan(self, query, images, mode="cached", readout="box"):
        if readout != "box":
            raise ValueError("Glance-X has the box readout only")
        eng = self.eng
        ids = eng.text_ids(query)
        t0 = time.perf_counter()
        lat = {}
        if mode in ("cached", "fresh"):
            eng.direction("image_first")
            key = (id(images), len(images), self.max_vis)
            bank = self._bank if mode == "cached" and key == self._bank_key else None
            if bank is None:
                te = time.perf_counter()
                prepped = self._prep(images)
                enc = 16  # images per encode forward (the largest CUDA-graph bucket)
                prefixes = [p for a in range(0, len(prepped), enc) for p in eng.encode(prepped[a:a + enc])]
                # stacked once per bank: later requests only run their blocks
                bank = [eng.stack(prefixes[a:a + self.chunk]) for a in range(0, len(prefixes), self.chunk)]
                del prefixes
                self._sync()
                lat["encode" if mode == "cached" else "vision_prefix"] = (time.perf_counter() - te) * 1e3
                if mode == "cached":
                    self._bank_key, self._bank = key, bank
            tq = time.perf_counter()
            rows = []
            for part in bank:
                rows += eng.answer(part, [(i, ids) for i in range(len(part))])
            self._sync()
            lat["query"] = (time.perf_counter() - tq) * 1e3
        elif mode in ("text_first", "joint"):
            prepped = self._prep(images)
            step = 64  # images per forward (the largest CUDA-graph bucket for pairs)
            rows = [r for a in range(0, len(prepped), step)
                    for r in eng.pairs(prepped[a:a + step], [[query]] * len(prepped[a:a + step]), mode)]
            self._sync()
        else:
            raise ValueError("mode must be cached, fresh, text_first or joint")
        lat["total"] = (time.perf_counter() - t0) * 1e3
        s = np.array([r["presence_logit"] for r in rows])
        p = 1 / (1 + np.exp(-s))
        per = [dict(p_present=round(float(r["p_present"]), 4), box=[round(float(v), 4) for v in r["box"]],
                    p_iou50=round(float(r["p_iou50"]), 4), boxes=[[round(float(v), 4) for v in b] for b in r["boxes"]],
                    box_scores=[round(float(v), 4) for v in r["box_scores"]]) for r in rows]
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
