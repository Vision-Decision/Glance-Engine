"""Glance engine: a serving engine for decision models.

One prefill per image serves any number of typed questions; answers are read out by
heads, not generated. See README.md and docs/ for the architecture and guarantees.
"""

from glance_engine.api import Answers, Glance
from glance_engine.questions import Box, BoxAnswer, Choice, ChoiceAnswer, Score, YesNo, YesNoAnswer

__version__ = "0.2.0"

__all__ = [
    "Answers", "Box", "BoxAnswer", "Choice", "ChoiceAnswer", "Glance", "Score", "YesNo", "YesNoAnswer", "__version__",
]
