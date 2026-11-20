"""HTTP end to end on the tiny model: the real worker, scheduler and aiohttp app (no mocks)."""

import asyncio
import base64
import io

import pytest

pytest.importorskip("aiohttp")
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402


def b64(im):
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture
def worker(glance):
    from speed.decision_worker import DecisionWorker

    return DecisionWorker(glance.engine, cache_bytes=1 << 28)


def test_decisions_and_cached_follow_up(worker, images):
    asyncio.run(_decisions(worker, images))


async def _decisions(worker, images):
    from speed.server import application

    async with TestClient(TestServer(application(worker))) as client:
        qs = [{"type": "box", "text": "the red car"}, {"type": "noul", "instr": "Is it raining?"},
              {"type": "choice", "instr": "What is shown?", "options": ["a car", "a dog"]}]
        r = await client.post("/v1/decisions", json={"image": b64(images[0]), "cache": True, "questions": qs})
        assert r.status == 200
        body = await r.json()
        box, yes, pick = body["results"][0]
        assert set(box) == {"p_present", "box", "p_iou50"} and len(yes["probs"]) == 2 and len(pick["probs"]) == 2
        r = await client.post("/v1/decisions", json={"image_ids": body["image_ids"], "questions": qs[1:2]})
        assert (await r.json())["results"][0][0]["probs"] == pytest.approx(yes["probs"], abs=1e-5)
        r = await client.post("/v1/decisions", json={"image_ids": ["0" * 64], "questions": qs[1:2]})
        assert r.status == 404
        r = await client.post("/v1/decisions", json={"image": b64(images[0]), "questions": [{"type": "caption"}]})
        assert r.status == 400
        metrics = await (await client.get("/metrics")).json()
        assert metrics["active"] == 0


def test_chat_completions_returns_typed_json(worker, images):
    asyncio.run(_chat(worker, images))


async def _chat(worker, images):
    import json

    from speed.server import application

    url = "data:image/png;base64," + b64(images[1])
    msg = {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                       {"type": "text", "text": "Is there a dog?"}]}
    async with TestClient(TestServer(application(worker))) as client:
        r = await client.post("/v1/chat/completions", json={"messages": [msg]})
        assert r.status == 200
        content = json.loads((await r.json())["choices"][0]["message"]["content"])
        assert len(content[0][0]["probs"]) == 2
