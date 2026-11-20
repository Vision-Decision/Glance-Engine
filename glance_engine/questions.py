"""Typed questions and the answers the engine returns for them.

Glance answers with read-out probabilities, never with generated text. Each question
type maps to one readout head:

=========  ==========================================  =====================================
Question   Wire format (HTTP ``questions`` entry)       Answer
=========  ==========================================  =====================================
Box        ``{"type": "box", "text": ...}``            P(present), box, P(IoU >= 0.5)
YesNo      ``{"type": "noul", "instr": ...}``          P(yes)
Choice     ``{"type": "choice", "instr", "options"}``  one probability per option
Score      ``{"type": "score", "instr", "options"}``   one probability per level
=========  ==========================================  =====================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence, Union

MAX_TEXT = 8192
MAX_OPTIONS = 26


@dataclass(frozen=True)
class Box:
    """Find the region a referring expression describes, or say it is absent."""

    text: str

    def wire(self) -> dict:
        return {"type": "box", "text": _text(self.text, "text")}


@dataclass(frozen=True)
class YesNo:
    """A yes/no question about the image."""

    question: str
    short: bool = False  # the distilled scan readout; needs a checkpoint trained for it

    def wire(self) -> dict:
        q = {"type": "noul", "instr": _text(self.question, "question")}
        if self.short:
            q["block"] = "short"
        return q


@dataclass(frozen=True)
class Choice:
    """Pick one of several options."""

    question: str
    options: Sequence[str]

    def wire(self) -> dict:
        return {"type": "choice", "instr": _text(self.question, "question"), "options": _options(self.options)}


@dataclass(frozen=True)
class Score:
    """Rate on an ordered scale; ``levels`` are the labels in order (for example "1".."5")."""

    question: str
    levels: Sequence[str]

    def wire(self) -> dict:
        return {"type": "score", "instr": _text(self.question, "question"), "options": _options(self.levels)}


Question = Union[Box, YesNo, Choice, Score, Mapping]


def to_wire(q: Question) -> dict:
    """Convert a typed question (or an already wire-format dict) to the engine's record format."""
    if hasattr(q, "wire"):
        return q.wire()
    if not isinstance(q, Mapping):
        raise TypeError(f"expected Box, YesNo, Choice, Score or a dict, got {type(q).__name__}")
    kind = q.get("type")
    if kind == "box":
        return Box(q.get("text")).wire()
    if kind == "noul":
        return YesNo(q.get("instr"), q.get("block") == "short").wire()
    if kind == "choice":
        return Choice(q.get("instr"), q.get("options")).wire()
    if kind == "score":
        return Score(q.get("instr"), q.get("options")).wire()
    raise ValueError(f"unknown question type {kind!r}; expected box, noul, choice or score")


def _text(value, name):
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_TEXT:
        raise ValueError(f"invalid {name}: expected a non-empty string of at most {MAX_TEXT} characters")
    return value


def _options(values):
    if isinstance(values, str) or not isinstance(values, Sequence) or not 0 < len(values) <= MAX_OPTIONS:
        raise ValueError(f"expected a list of 1 to {MAX_OPTIONS} options")
    return [_text(v, "option") for v in values]


# ------------------------------------------------------------------------------------------------ answers
@dataclass
class BoxAnswer:
    """``box`` is [x1, y1, x2, y2] normalized to the image (0..1); ``box_px`` is in input pixels.

    Glance-X models also return every target found by their multi-box head: ``boxes`` (normalized),
    ``boxes_px`` and ``box_scores`` (objectness). They stay ``None`` for Glance v1.
    """

    question: str
    p_present: float
    box: list
    p_iou50: float
    box_px: list | None = None
    boxes: list | None = None
    box_scores: list | None = None
    boxes_px: list | None = None

    @property
    def present(self) -> bool:
        return self.p_present >= 0.5


@dataclass
class YesNoAnswer:
    question: str
    p_yes: float

    @property
    def answer(self) -> bool:
        return self.p_yes >= 0.5


@dataclass
class ChoiceAnswer:
    """``probs`` maps each option (or score level) to its probability, in the order given."""

    question: str
    probs: dict = field(default_factory=dict)

    @property
    def answer(self) -> str:
        return max(self.probs, key=self.probs.get)


Answer = Union[BoxAnswer, YesNoAnswer, ChoiceAnswer]


def _logit(p, eps=1e-12):
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def _sigmoid(x):
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def _tempered(probs, t):
    """softmax(log p / t) == softmax(z / t) for the logits z behind p."""
    logs = [math.log(max(p, 1e-30)) / t for p in probs]
    top = max(logs)
    exps = [math.exp(v - top) for v in logs]
    total = sum(exps)
    return [v / total for v in exps]


def make_answer(q: dict, row: dict, size=None, temperatures: Mapping | None = None) -> Answer:
    """Turn one worker result row into a typed answer, applying per-type temperatures if given."""
    t = temperatures or {}
    if q["type"] == "box":
        p = row["p_present"]
        if "presence" in t:
            p = _sigmoid(_logit(p) / t["presence"])
        box = [float(v) for v in row["box"]]
        px = None
        if size is not None:
            w, h = size
            px = [box[0] * w, box[1] * h, box[2] * w, box[3] * h]
        answer = BoxAnswer(q["text"], float(p), box, float(row["p_iou50"]), px)
        if "boxes" in row:  # Glance-X multi-box head
            answer.boxes = [[float(v) for v in b] for b in row["boxes"]]
            answer.box_scores = [float(v) for v in row["box_scores"]]
            if size is not None:
                w, h = size
                answer.boxes_px = [[b[0] * w, b[1] * h, b[2] * w, b[3] * h] for b in answer.boxes]
        return answer
    probs = [float(v) for v in row["probs"]]
    if q["type"] in t:
        probs = _tempered(probs, t[q["type"]])
    if q["type"] == "noul":
        return YesNoAnswer(q["instr"], probs[1])
    return ChoiceAnswer(q["instr"], dict(zip(q["options"], probs)))
