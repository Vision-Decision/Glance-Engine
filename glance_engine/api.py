"""The Python entry point: one prefill per image, any number of typed questions, probabilities back.

    from glance_engine import Glance, Box, YesNo, Choice

    glance = Glance.from_pretrained()                    # Vision-Decision/Glance-4B-preview, BF16 on CUDA
    image_id = glance.add_image("street.jpg")            # vision tower + prefill once; the prefix is kept
    answers = glance.ask(image_id, [Box("the red car"), YesNo("Is it raining?"),
                                    Choice("What time of day is it?", ["day", "night"])])

``Glance`` wraps the same :class:`speed.decision_worker.DecisionWorker` that serves HTTP requests,
so the Python and HTTP paths batch, cache and read out identically. It is not thread-safe;
for concurrent clients run the server (``glance-serve``), whose scheduler owns the device.

Glance-X models (``Glance.from_pretrained("Vision-Decision/Glance-X-Lite")``) run on the X backend
(:mod:`glance_engine.x`) behind the same calls. They answer ``Box`` questions, add every target
(``BoxAnswer.boxes``) and take a reading ``direction``: ``image_first`` (default, cacheable),
``text_first`` or ``joint``, where the checkpoint was trained in it.
"""

from __future__ import annotations

import os
import re
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

from glance_engine import checkpoints as C
from glance_engine.questions import Answer, Box, Question, YesNo, make_answer, to_wire
from glance_engine.x.model import is_x_engine

_ID = re.compile(r"^[0-9a-f]{64}$")


class Answers(list):
    """A list of answers in question order. ``image_id`` names the cached prefix for follow-ups."""

    image_id: str | None = None


class Glance:
    def __init__(self, engine, *, temperatures=None, max_vis=640, cache_gib=8.0, max_tokens=8192,
                 calibrated=True):
        if is_x_engine(engine):
            from glance_engine.x.worker import XDecisionWorker as DecisionWorker
        else:
            from speed.decision_worker import DecisionWorker

        self.worker = DecisionWorker(engine, max_tokens=max_tokens, cache_bytes=int(cache_gib * 2**30))
        self.temperatures = dict(temperatures or {})
        self.calibrated = calibrated
        self.max_vis = max_vis
        self._sizes = {}

    # ---------------------------------------------------------------------------------------- loading
    @classmethod
    def from_pretrained(cls, model: str = C.DEFAULT_MODEL, *, device: str = "cuda", fp8: bool = False,
                        w12: str | None = None, graphs: bool = True, exact: bool = False, **kw) -> "Glance":
        """Load a release repo/dir, a training ``.pt`` (local or ``repo:path``), ``"tiny"`` or ``"tiny-x"``.

        ``fp8`` swaps in W8A8 FP8 language-model projections (changes numerics; reported separately).
        ``w12`` points at an offline lossless W12 export (see ``speed/w12.py``).
        Glance-X checkpoints (``glancex_config.json`` releases or glancex ``.pt``) load the X backend;
        ``exact=True`` keeps their LoRA unmerged.
        """
        tiny = model in ("tiny", "tiny-x")
        loaded = C.load(model, device=None if tiny else device, fp8=fp8, w12=w12, graphs=graphs, exact=exact)
        kw.setdefault("max_vis", loaded.max_vis)
        return cls(loaded.engine, temperatures=loaded.temperatures, **kw)

    @classmethod
    def tiny(cls, **kw) -> "Glance":
        """Random tiny model on CPU: the full engine path, meaningless answers. For tests and smoke runs."""
        loaded = C.tiny()
        kw.setdefault("max_vis", loaded.max_vis)
        return cls(loaded.engine, **kw)

    @classmethod
    def tiny_x(cls, **kw) -> "Glance":
        """Random tiny Glance-X on CPU (K=2, all three directions). For tests and smoke runs."""
        loaded = C.tiny_x()
        kw.setdefault("max_vis", loaded.max_vis)
        return cls(loaded.engine, **kw)

    @property
    def engine(self):
        return self.worker.engine

    @property
    def family(self) -> str:
        """``"x"`` for Glance-X models, ``"v1"`` otherwise."""
        return "x" if is_x_engine(self.engine) else "v1"

    # ---------------------------------------------------------------------------------------- images
    def _image(self, image):
        from dv.data.dataset import resize_dims

        if isinstance(image, (str, os.PathLike)):
            with Image.open(image) as im:
                image = im.convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if not isinstance(image, Image.Image):
            raise TypeError("expected a PIL image, a path, an RGB array or an image id")
        image = image.convert("RGB")
        size = resize_dims(*image.size, self.max_vis, 2.0)
        return image.size, image if size == image.size else image.resize(size, Image.Resampling.BICUBIC)

    def _unit(self, image):
        from speed.decision_worker import ImageRef

        if isinstance(image, str) and _ID.match(image):
            return self._sizes.get(image), ImageRef(bytes.fromhex(image))
        return self._image(image)

    def add_image(self, image) -> str:
        """Run the vision tower and prefill once and keep the image's prefix. Returns its id."""
        probe = Box("an object") if self.family == "x" else YesNo("Is there anything in the image?")
        return self.ask(image, [probe]).image_id

    def forget(self, image_id: str) -> None:
        self.worker.cache.entries.pop(("RGB", bytes.fromhex(image_id)), None)
        self.worker.cache.used = sum(size for _, size in self.worker.cache.entries.values())
        self._sizes.pop(image_id, None)

    # ---------------------------------------------------------------------------------------- questions
    def ask(self, image, questions: Sequence[Question], *, cache: bool = True, direction: str | None = None) -> Answers:
        """Answer every question about one image (a PIL image, path, array, or an id from ``add_image``).

        All questions share one prefill. With ``cache=True`` the image prefix is kept, and the
        returned ``Answers.image_id`` answers later questions without the vision tower or prefill.
        A missing or evicted id raises ``KeyError``.

        ``direction`` (Glance-X only): ``"image_first"`` (default), ``"text_first"`` or ``"joint"``.
        Only ``image_first`` prefixes can be cached.
        """
        return self._run([image], questions, cache, direction)[0]

    def ask_each(self, images: Iterable, questions: Sequence[Question], *, cache: bool = False,
                 direction: str | None = None) -> list:
        """Broadcast the same questions over several images in one batched call (for example video frames)."""
        return self._run(list(images), questions, cache, direction)

    def _run(self, images, questions, cache, direction=None):
        from speed.decision_worker import Decision

        qs = [to_wire(q) for q in questions]
        if not qs:
            raise ValueError("ask at least one question")
        if direction not in (None, "image_first") and self.family != "x":
            raise ValueError("reading directions other than image_first need a Glance-X model")
        if direction not in (None, "image_first"):
            cache = False
        sizes, units = zip(*(self._unit(im) for im in images))
        req = Decision(list(units), qs, cache)
        if direction is not None:
            req.direction = direction
        [rows] = self.worker([req])
        temps = self.temperatures if self.calibrated else None
        out = []
        for j, (size, row) in enumerate(zip(sizes, rows)):
            answers = Answers(make_answer(q, r, size, temps) for q, r in zip(qs, row))
            if cache and req.image_ids[j]:
                answers.image_id = req.image_ids[j]
                if size is not None:
                    self._sizes[answers.image_id] = size
            out.append(answers)
        return out

    # ---------------------------------------------------------------------------------------- scanning
    def scanner(self, *, max_vis: int = 256, alpha: float = 0.1, q: float = 0.1):
        """One question over many images or clips. See :class:`scan.api.Scanner`.

        ``scan(query, images)`` encodes a bank once per image list and answers every later
        query from it. After ``calibrate(negative_scans)`` the set answer includes
        ``none_conformal``: a "not in any of these" verdict whose false-alarm rate on scans
        without the target is at most ``alpha`` (averaged over calibration draws).
        """
        if self.family == "x":
            from glance_engine.x.scanner import XScanner

            return XScanner(self.engine, max_vis=max_vis, alpha=alpha, q=q)
        from scan.api import Scanner

        return Scanner(self.engine, max_vis=max_vis, alpha=alpha, q=q)


__all__ = ["Answers", "Glance", "Answer"]
