"""Glance-X on the engine, on the tiny CPU model: exact against glancex's own forward, cached == fresh, every path."""

import asyncio
import base64
import io

import pytest
import torch

pytest.importorskip("glancex")

from glance_engine import Box, BoxAnswer, Choice, Glance, YesNo  # noqa: E402

REQUESTS = ["the red car", "a dog on the left", "左边的人", "the red car"]


@pytest.fixture(scope="module")
def gx():
    return Glance.tiny_x()


def reference(eng, images, requests, mode):
    """glancex's GlanceXBackbone.forward + heads on one collated batch (what glancex.GlanceX(fast=False) runs)."""
    from glancex.data import collate

    b = collate(eng._items([eng.fit(im) for im in images], [requests] * len(images)), eng.tok, eng.cfg, mode)
    b.pop("meta")
    with torch.no_grad():
        feats = eng.net.image_features(b["pixel_values"], b["grid_thw"])
        o = eng.net(b, feats, mode)
        q_img = b["q_img"]
        rows = slice(None) if mode == "i2t" else q_img
        valid, centers = b["vis_valid"][rows], b["centers"][rows]
        out = eng.heads(o["r"], o["vis"], valid, centers, o["vis_index"])
    return out["presence_logit"].sigmoid(), out["box_pred"].clamp(0, 1), out["iou_logit"].sigmoid()


def flat(answers):
    return [a for row in answers for a in row]


@pytest.mark.parametrize("direction,mode", [("image_first", "i2t"), ("text_first", "t2i"), ("joint", "bi")])
def test_matches_glancex_forward(gx, images, direction, mode):
    got = flat(gx.ask_each(images, [Box(t) for t in REQUESTS], direction=direction))
    p, box, iou = reference(gx.engine, images, REQUESTS, mode)
    assert [a.p_present for a in got] == pytest.approx(p.tolist(), abs=1e-5)
    assert [a.p_iou50 for a in got] == pytest.approx(iou.tolist(), abs=1e-5)
    for a, b in zip(got, box.tolist()):
        assert a.box == pytest.approx(b, abs=1e-5)


def test_answers_carry_every_target(gx, images):
    answers = gx.ask(images[0], [Box("the red car"), Box("a dog")])
    assert gx.family == "x" and len(answers.image_id) == 64
    for a in answers:
        assert isinstance(a, BoxAnswer) and 0 <= a.p_present <= 1 and len(a.box_px) == 4
        assert isinstance(a.boxes, list) and len(a.boxes) == len(a.box_scores) == len(a.boxes_px)
        assert all(len(b) == 4 and all(0 <= v <= 1 for v in b) for b in a.boxes)


def test_cached_follow_up_matches_fresh(gx, images):
    image_id = gx.add_image(images[1])
    cached = gx.ask(image_id, [Box(t) for t in REQUESTS])
    fresh = gx.ask(images[1], [Box(t) for t in REQUESTS], cache=False)
    assert fresh.image_id is None
    for c, f in zip(cached, fresh):
        assert c.p_present == pytest.approx(f.p_present, abs=1e-5)
        assert c.box == pytest.approx(f.box, abs=1e-5)
        assert c.box_px == pytest.approx(f.box_px, abs=1e-2)
    gx.forget(image_id)
    with pytest.raises(KeyError):
        gx.ask(image_id, [Box("a dog")])


def test_mixed_cached_and_new_images_in_one_call(gx, images):
    image_id = gx.add_image(images[2])
    together = gx.ask_each([image_id, images[0]], [Box("a dog")])
    alone = [gx.ask(images[2], [Box("a dog")], cache=False), gx.ask(images[0], [Box("a dog")], cache=False)]
    assert [t[0].p_present for t in together] == pytest.approx([a[0].p_present for a in alone], abs=1e-5)


def test_small_token_budget_encodes_in_batches_with_the_same_answers(gx, images):
    from glance_engine.x.worker import XDecisionWorker
    from speed.decision_worker import Decision

    qs = [{"type": "box", "text": t} for t in REQUESTS]
    tight = XDecisionWorker(gx.engine, max_tokens=70)  # one tiny image per encode batch
    [rows] = tight([Decision(list(images) * 2, qs, False)])
    alone = [gx.ask(im, [Box(t) for t in REQUESTS], cache=False) for im in images] * 2
    assert len(rows) == 6 and all(len(r) == len(REQUESTS) for r in rows)
    assert [x["p_present"] for r in rows for x in r] == pytest.approx([a.p_present for r in alone for a in r], abs=1e-5)


def test_refuses_typed_questions_and_untrained_directions(gx, images):
    with pytest.raises(ValueError, match="box questions only"):
        gx.ask(images[0], [YesNo("Is it raining?")])
    with pytest.raises(ValueError, match="box questions only"):
        gx.ask(images[0], [Choice("What is shown?", ["a", "b"])])
    with pytest.raises(ValueError, match="direction"):
        gx.ask(images[0], [Box("a dog")], direction="sideways")
    eng = gx.engine
    trained = eng.directions
    eng.directions = ("image_first", "text_first")  # as Glance-X-Lite
    try:
        with pytest.raises(ValueError, match="not trained"):
            gx.ask(images[0], [Box("a dog")], direction="joint")
    finally:
        eng.directions = trained
    image_id = gx.add_image(images[0])
    with pytest.raises(ValueError, match="image_first"):
        gx.ask(image_id, [Box("a dog")], direction="text_first")


def test_v1_refuses_directions(glance, images):
    with pytest.raises(ValueError, match="Glance-X"):
        glance.ask(images[0], [Box("a dog")], direction="text_first")


def test_scanner_modes_agree(gx, images):
    scanner = gx.scanner(max_vis=64)
    first = scanner.scan("a red car", images)
    again = scanner.scan("a dog", images)
    fresh = scanner.scan("a dog", images, mode="fresh")
    assert "encode" in first["latency_ms"] and "encode" not in again["latency_ms"]
    assert [x["p_present"] for x in again["images"]] == pytest.approx([x["p_present"] for x in fresh["images"]],
                                                                       abs=1e-4)
    one_by_one = [gx.ask(im, [Box("a dog")], cache=False)[0].p_present for im in images]
    assert [x["p_present"] for x in fresh["images"]] == pytest.approx(one_by_one, abs=1e-4)
    t2i = scanner.scan("a dog", images, mode="text_first")
    ref = [a[0].p_present for a in gx.ask_each(images, [Box("a dog")], direction="text_first")]
    assert [x["p_present"] for x in t2i["images"]] == pytest.approx(ref, abs=1e-4)
    assert set(t2i["images"][0]) == {"p_present", "box", "p_iou50", "boxes", "box_scores"}
    scanner.calibrate([[-5.0, -4.0, -3.0]] * 20)
    assert "none_conformal" in scanner.scan("a dog", images)["set"]


def test_bank_filled_in_place_equals_stacked(gx, images):
    from glance_engine.x.engine import image_tokens

    eng = gx.engine
    fitted = [eng.fit(im) for im in images]
    prefixes = eng.encode(fitted)
    stacked = eng.stack(prefixes)
    tokens = [image_tokens(im) for im in fitted]
    filled = eng.empty_bank(len(prefixes), prefixes[0], max(tokens) + 2, max(tokens))
    for i, p in enumerate(prefixes):
        eng.put(filled, i, p)
    for a, b in zip([t for st in stacked.states for t in st], [t for st in filled.states for t in st]):
        assert torch.equal(a, b)
    for name in ("p_valid", "vis", "vis_valid", "centers"):
        assert torch.equal(getattr(stacked, name), getattr(filled, name))


def test_loader_recognises_sources(tmp_path):
    from glance_engine.x.model import x_state

    assert x_state("tiny-x") is None
    assert x_state(str(tmp_path)) is None                       # a folder without glancex_config.json
    pt = tmp_path / "ckpt.pt"
    torch.save({"cfg": {"bidir_k": 2}, "lora": {}, "heads": {}}, pt)
    assert x_state(str(pt))["cfg"]["bidir_k"] == 2
    torch.save({"cfg": {"task": "v1"}, "lora": {}, "heads": {}}, pt)
    assert x_state(str(pt)) is None


# ------------------------------------------------------------------------------------------------ HTTP
def b64(im):
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_http_decisions(gx, images):
    pytest.importorskip("aiohttp")
    asyncio.run(_http(gx, images))


async def _http(gx, images):
    from aiohttp.test_utils import TestClient, TestServer

    from glance_engine.x.worker import XDecisionWorker
    from speed.server import application

    worker = XDecisionWorker(gx.engine, cache_bytes=1 << 28)
    async with TestClient(TestServer(application(worker))) as client:
        qs = [{"type": "box", "text": "the red car"}, {"type": "box", "text": "a dog"}]
        r = await client.post("/v1/decisions", json={"image": b64(images[0]), "cache": True, "questions": qs})
        assert r.status == 200
        body = await r.json()
        assert set(body["results"][0][0]) == {"p_present", "box", "p_iou50", "boxes", "box_scores"}
        r = await client.post("/v1/decisions", json={"image_ids": body["image_ids"], "questions": qs})
        assert r.status == 200
        follow = await r.json()
        assert follow["results"][0][0]["p_present"] == pytest.approx(body["results"][0][0]["p_present"], abs=1e-5)
        r = await client.post("/v1/decisions", json={"image": b64(images[1]), "direction": "text_first",
                                                     "questions": qs[:1]})
        assert r.status == 200
        r = await client.post("/v1/decisions", json={"image": b64(images[0]), "questions": [
            {"type": "noul", "instr": "Is it raining?"}]})
        assert r.status == 400 and "box questions" in (await r.json())["error"]["message"]
        r = await client.post("/v1/decisions", json={"image": b64(images[0]), "direction": "up", "questions": qs})
        assert r.status == 400


def test_scanner_does_not_reuse_a_bank_for_a_new_list(gx, images):
    scanner = gx.scanner(max_vis=64)
    first = scanner.scan("a dog", [images[0], images[1]])
    other = scanner.scan("a dog", [images[2], images[0]])   # same length, new list object
    assert "encode" in first["latency_ms"] and "encode" in other["latency_ms"]
    assert other["images"][0]["p_present"] == pytest.approx(
        gx.ask(images[2], [Box("a dog")], cache=False)[0].p_present, abs=1e-4)


def test_calibration_lengths_and_options_are_checked(gx, images):
    scanner = gx.scanner(max_vis=64).calibrate([[-5.0, -4.0], [-5.0, -4.0, -3.0]] * 10)
    with pytest.raises(ValueError, match="calibrated for scans"):
        scanner.scan("a dog", images[:1])
    with pytest.raises(ValueError, match="distinct"):
        Choice("What is shown?", ["a car", "a car"]).wire()
