"""Batched Glance-X decisions with the v1 worker's interface, so ``Glance`` and ``glance-serve`` drive it unchanged.

``XDecisionWorker(engine)(requests)`` takes the same :class:`speed.decision_worker.Decision` requests and returns the
same nested result rows (request -> image -> question). Box rows carry two more fields than v1: ``boxes`` and
``box_scores`` from the multi-box head (every target, normalized xyxy). A request may set ``direction``
(``image_first`` by default; ``text_first`` and ``joint`` where the checkpoint was trained in them).

``image_first`` work is split into cache lookups, one batched encode of the missing images, and one batched answer
of every request block against its image. With ``cache=True`` encoded prefixes go into the same byte-bounded
:class:`~speed.decision_worker.PrefixCache` and are addressed by the same 64-hex image ids.
"""

from __future__ import annotations

from collections import Counter
from time import perf_counter

import torch

from glance_engine.x.engine import as_image, image_tokens

_PUBLIC = ("p_present", "box", "p_iou50", "boxes", "box_scores")


class XDecisionWorker:
    def __init__(self, engine, max_tokens=8192, cache_bytes=8 << 30, max_rows=256, **_ignored):
        from speed.decision_worker import PrefixCache

        self.engine, self.max_tokens, self.max_rows = engine, max_tokens, max_rows
        self.cache = PrefixCache(cache_bytes)
        self.video = None
        self.reset_timings()

    def reset_timings(self):
        self.timings = Counter()

    @staticmethod
    def is_oom(exc):
        return isinstance(exc, torch.OutOfMemoryError)

    def recover(self):
        self.engine.free_graphs()

    # ---------------------------------------------------------------------------------------- requests
    @torch.no_grad()
    def __call__(self, requests):
        if requests[0].batch_kind != "image":
            raise ValueError("Glance-X serves image decisions; video and clip endpoints need a Glance v1 model")
        for req in requests:
            for q in req.questions:
                if q.get("type") != "box":
                    raise ValueError(f"Glance-X answers box questions only, not {q.get('type')!r}")
        results = [[[] for _ in r.images] for r in requests]
        started = perf_counter()
        first, other = [], []
        for i, r in enumerate(requests):
            d = self.engine.direction(getattr(r, "direction", None) or "image_first")
            (first if d == "image_first" else other).append(i)
        if first:
            self._image_first(first, requests, results)
        for i in other:
            self._pairs(requests[i], i, results)
        self.timings.update(total=(perf_counter() - started) * 1000, steps=1)
        return results

    def _image_first(self, indices, requests, results):
        from speed.decision_worker import ImageRef, image_key

        eng = self.engine
        started = perf_counter()
        # 1. resolve every image to a cached prefix or a pending encode (identical pixels are encoded once)
        units, pending, pending_index = [], [], {}
        raw = [(i, j) for i in indices for j, im in enumerate(requests[i].images) if not isinstance(im, ImageRef)]
        fitted = dict(zip(raw, eng.map(lambda ij: eng.fit(as_image(requests[ij[0]].images[ij[1]])), raw)))
        for i in indices:
            req = requests[i]
            req.image_ids = [None] * len(req.images)
            for j, im in enumerate(req.images):
                if isinstance(im, ImageRef):
                    key = ("RGB", im.digest)
                    prefix = self.cache.get(key)
                    if prefix is None:
                        raise KeyError("unknown or evicted image prefix")
                    req.image_ids[j] = im.digest.hex()
                    units.append((i, j, req, prefix, None))
                    continue
                im = fitted[i, j]
                key = image_key(im) if req.cache else None
                prefix = self.cache.get(key) if key else None
                if key:
                    req.image_ids[j] = key[1].hex()
                if prefix is not None:
                    units.append((i, j, req, prefix, None))
                    continue
                tag = key if key else ("pending", i, j)
                if tag not in pending_index:
                    pending_index[tag] = len(pending)
                    pending.append((im, key))
                units.append((i, j, req, None, pending_index[tag]))
        t1 = perf_counter()
        # 2. encode the missing images in token-bounded batches
        encoded = []
        batch, tokens = [], 0
        for im, key in pending:
            cost = image_tokens(im) + 2
            if batch and tokens + cost > self.max_tokens:
                encoded += self._encode(batch)
                batch, tokens = [], 0
            batch.append((im, key))
            tokens += cost
        if batch:
            encoded += self._encode(batch)
        t2 = perf_counter()
        # 3. answer every request block against its image
        work = []
        for i, j, req, prefix, p in units:
            prefix = prefix if prefix is not None else encoded[p]
            for k, q in enumerate(req.questions):
                work.append((i, j, k, prefix, eng.text_ids(q["text"])))
        for row, out in zip(work, self._answer(work)):
            i, j, k = row[:3]
            results[i][j].append(out)
        for i in indices:
            req = requests[i]
            if req.cache:
                req.image_ids = [s if s and ("RGB", bytes.fromhex(s)) in self.cache.entries else None
                                 for s in req.image_ids]
        self.timings.update(prep=(t1 - started) * 1000, encode=(t2 - t1) * 1000, lm=(perf_counter() - t2) * 1000)

    def _encode(self, batch):
        try:
            prefixes = self.engine.encode([im for im, _ in batch])
        except torch.OutOfMemoryError:
            if len(batch) == 1:
                raise
            self.recover()
            mid = len(batch) // 2
            return self._encode(batch[:mid]) + self._encode(batch[mid:])
        for (_, key), prefix in zip(batch, prefixes):
            if key is not None:
                self.cache.put(key, prefix, prefix.nbytes())
        return prefixes

    def _answer(self, work):
        """Answer rows in chunks of at most ``max_rows`` blocks and a bounded number of distinct prefixes."""
        out = []
        start = 0
        while start < len(work):
            end, seen = start, {}
            while end < len(work) and end - start < self.max_rows:
                seen.setdefault(id(work[end][3]), work[end][3])
                if len(seen) > 64:
                    break
                end += 1
            out += self._answer_chunk(work[start:end])
            start = end
        return out

    def _answer_chunk(self, rows):
        index, prefixes = {}, []
        for row in rows:
            if id(row[3]) not in index:
                index[id(row[3])] = len(prefixes)
                prefixes.append(row[3])
        try:
            res = self.engine.answer(prefixes, [(index[id(r[3])], r[4]) for r in rows])
        except torch.OutOfMemoryError:
            if len(rows) == 1:
                raise
            self.recover()
            mid = len(rows) // 2
            return self._answer_chunk(rows[:mid]) + self._answer_chunk(rows[mid:])
        return [{k: r[k] for k in _PUBLIC} for r in res]

    def _pairs(self, req, i, results):
        from speed.decision_worker import ImageRef

        if any(isinstance(im, ImageRef) for im in req.images):
            raise ValueError("cached image ids answer image_first questions only; send the pixels for this direction")
        req.image_ids = [None] * len(req.images)
        images = self.engine.map(lambda im: self.engine.fit(as_image(im)), req.images)
        texts = [q["text"] for q in req.questions]
        t0 = perf_counter()
        per_image = max(1, self.max_tokens // max(image_tokens(im) + 2 + 48 for im in images) // len(texts))
        for a in range(0, len(images), per_image):
            chunk = images[a:a + per_image]
            res = self.engine.pairs(chunk, [texts] * len(chunk), req.direction)
            for n, r in enumerate(res):
                results[i][a + n // len(texts)].append({k: r[k] for k in _PUBLIC})
        self.timings.update(lm=(perf_counter() - t0) * 1000)
