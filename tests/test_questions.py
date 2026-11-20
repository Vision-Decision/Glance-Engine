import math

import pytest

from glance_engine.questions import (
    Box,
    BoxAnswer,
    Choice,
    ChoiceAnswer,
    Score,
    YesNo,
    YesNoAnswer,
    _tempered,
    make_answer,
    to_wire,
)


def test_wire_format_matches_the_http_schema():
    assert to_wire(Box("the red car")) == {"type": "box", "text": "the red car"}
    assert to_wire(YesNo("Is it raining?")) == {"type": "noul", "instr": "Is it raining?"}
    assert to_wire(YesNo("a person falling", short=True))["block"] == "short"
    choice = {"type": "choice", "instr": "Color?", "options": ["red", "blue"]}
    assert to_wire(Choice("Color?", ["red", "blue"])) == choice
    assert to_wire(Score("Sharp?", ["1", "2", "3"]))["type"] == "score"


def test_dicts_are_validated_like_typed_questions():
    assert to_wire({"type": "noul", "instr": "Is it day?"}) == {"type": "noul", "instr": "Is it day?"}
    with pytest.raises(ValueError):
        to_wire({"type": "caption", "instr": "describe"})
    with pytest.raises(ValueError):
        to_wire(Box(""))
    with pytest.raises(ValueError):
        to_wire(Choice("Pick", "ab"))  # a string is not a list of options
    with pytest.raises(ValueError):
        to_wire(Choice("Pick", [str(i) for i in range(27)]))
    with pytest.raises(TypeError):
        to_wire("Is it day?")


def test_answers_carry_types_and_pixels():
    box = make_answer(to_wire(Box("dog")), {"p_present": 0.8, "box": [0.1, 0.2, 0.5, 1.0], "p_iou50": 0.7}, (200, 100))
    assert isinstance(box, BoxAnswer) and box.present and box.box_px == pytest.approx([20, 20, 100, 100])
    yes = make_answer(to_wire(YesNo("day?")), {"probs": [0.3, 0.7]})
    assert isinstance(yes, YesNoAnswer) and yes.p_yes == 0.7 and yes.answer
    pick = make_answer(to_wire(Choice("c?", ["a", "b", "c"])), {"probs": [0.2, 0.5, 0.3]})
    assert isinstance(pick, ChoiceAnswer) and pick.answer == "b" and list(pick.probs) == ["a", "b", "c"]


def test_temperatures_rescale_logits():
    raw = {"p_present": 0.9, "box": [0, 0, 1, 1], "p_iou50": 0.5}
    q = to_wire(Box("dog"))
    assert make_answer(q, raw, temperatures={"presence": 1.0}).p_present == pytest.approx(0.9)
    hot = make_answer(q, raw, temperatures={"presence": 2.0}).p_present
    assert hot == pytest.approx(1 / (1 + math.exp(-math.log(9) / 2)))
    probs = _tempered([0.7, 0.2, 0.1], 0.5)
    assert sum(probs) == pytest.approx(1) and probs[0] > 0.7
    yes = make_answer(to_wire(YesNo("day?")), {"probs": [0.3, 0.7]}, temperatures={"noul": 0.889068})
    assert yes.p_yes > 0.7  # T < 1 sharpens
