"""The public Python API on the tiny CPU model: one prefill serves many questions, cached == fresh."""

import pytest

from glance_engine import Box, BoxAnswer, Choice, ChoiceAnswer, Score, YesNo, YesNoAnswer

QS = [Box("the red car"), YesNo("Is it raining?"), Choice("What is shown?", ["a car", "a dog", "nothing"]),
      Score("How sharp is it?", ["1", "2", "3", "4", "5"])]


def test_typed_answers_in_question_order(glance, images):
    answers = glance.ask(images[0], QS)
    assert [type(a) for a in answers] == [BoxAnswer, YesNoAnswer, ChoiceAnswer, ChoiceAnswer]
    assert 0 <= answers[0].p_present <= 1 and len(answers[0].box) == 4 and len(answers[0].box_px) == 4
    assert sum(answers[2].probs.values()) == pytest.approx(1, abs=1e-5)
    assert list(answers[3].probs) == ["1", "2", "3", "4", "5"]
    assert answers.image_id and len(answers.image_id) == 64


def test_follow_up_from_cached_prefix_matches_fresh(glance, images):
    image_id = glance.add_image(images[1])
    cached = glance.ask(image_id, QS)
    fresh = glance.ask(images[1], QS, cache=False)
    assert fresh.image_id is None
    for c, f in zip(cached, fresh):
        if isinstance(c, BoxAnswer):
            assert c.p_present == pytest.approx(f.p_present, abs=1e-5)
            assert c.box == pytest.approx(f.box, abs=1e-4)
            assert c.box_px == pytest.approx(f.box_px, abs=1e-2)  # original size remembered for the id
        elif isinstance(c, YesNoAnswer):
            assert c.p_yes == pytest.approx(f.p_yes, abs=1e-5)
        else:
            assert list(c.probs.values()) == pytest.approx(list(f.probs.values()), abs=1e-5)


def test_broadcast_equals_one_by_one(glance, images):
    together = glance.ask_each(images, [YesNo("Is there a person?")])
    alone = [glance.ask(im, [YesNo("Is there a person?")], cache=False) for im in images]
    assert [t[0].p_yes for t in together] == pytest.approx([a[0].p_yes for a in alone], abs=1e-5)


def test_forget_evicts_and_unknown_ids_fail(glance, images):
    image_id = glance.add_image(images[2])
    glance.forget(image_id)
    with pytest.raises(KeyError):
        glance.ask(image_id, [YesNo("Is it day?")])


def test_rejects_empty_and_unknown_inputs(glance, images):
    with pytest.raises(ValueError):
        glance.ask(images[0], [])
    with pytest.raises(TypeError):
        glance.ask(42, [YesNo("Is it day?")])


def test_temperatures_are_applied_only_when_calibrated(glance, images):
    raw = glance.ask(images[0], [YesNo("Is it day?")], cache=False)[0].p_yes
    glance.temperatures = {"noul": 0.5}
    try:
        sharp = glance.ask(images[0], [YesNo("Is it day?")], cache=False)[0].p_yes
        glance.calibrated = False
        off = glance.ask(images[0], [YesNo("Is it day?")], cache=False)[0].p_yes
    finally:
        glance.temperatures, glance.calibrated = {}, True
    assert abs(sharp - 0.5) > abs(raw - 0.5) and off == pytest.approx(raw)


def test_scanner_cached_matches_fresh(glance, images):
    scanner = glance.scanner(max_vis=64)
    first = scanner.scan("a red car", images)
    again = scanner.scan("a dog", images)
    fresh = scanner.scan("a dog", images, mode="fresh")
    assert "encode" in first["latency_ms"] and "encode" not in again["latency_ms"]
    assert [x["p_present"] for x in again["images"]] == pytest.approx([x["p_present"] for x in fresh["images"]],
                                                                       abs=1e-4)


def test_scanner_does_not_reuse_a_bank_for_a_new_list(glance, images):
    scanner = glance.scanner(max_vis=64)
    first = scanner.scan("a dog", [images[0], images[1]])
    other = scanner.scan("a dog", [images[2], images[0]])  # same length, new list object
    assert "encode" in first["latency_ms"] and "encode" in other["latency_ms"]
