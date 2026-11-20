"""``glance-serve``: the async HTTP decision server on one device.

    glance-serve                                   # Vision-Decision/Glance-4B-preview on CUDA
    glance-serve --model ckpt_final.pt --port 8002
    glance-serve --tiny                            # random tiny model on CPU, for smoke tests
    glance-serve --model Vision-Decision/Glance-X-Lite   # a Glance-X model (box questions, three directions)

Endpoints (see docs/api.md): POST /v1/decisions, /v1/chat/completions, /encode, /scan,
/score_clips; GET /health, /ready, /metrics. The server binds to loopback by default and
drains accepted requests on shutdown. ``speed.server`` remains the benchmark entry point.
"""

from __future__ import annotations

import argparse
import json


def build_parser():
    from glance_engine.checkpoints import DEFAULT_MODEL

    p = argparse.ArgumentParser(prog="glance-serve", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--model", default=DEFAULT_MODEL,
                     help="release repo or directory (Glance or Glance-X), training .pt, 'org/repo:path.pt', "
                          "or 'tiny-x' (default: %(default)s)")
    src.add_argument("--tiny", action="store_true", help="random tiny model on CPU (meaningless answers)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8002)
    p.add_argument("--graphs", action=argparse.BooleanOptionalAction, default=True, help="CUDA graph replay")
    p.add_argument("--group-vision", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--shared-prefill", choices=("packed", "padded"), default="packed")
    p.add_argument("--max-batch", type=int, default=64, help="requests per device step")
    p.add_argument("--max-tokens", type=int, default=8192, help="admission budget per device step")
    p.add_argument("--wait-ms", type=float, default=0.5, help="gathering window before a step")
    p.add_argument("--queue-limit", type=int, default=256)
    p.add_argument("--timeout", type=float, default=120, help="per-request deadline, seconds")
    p.add_argument("--cache-gib", type=float, default=8, help="prefix and bank cache budget")
    p.add_argument("--max-vis", type=int, default=96, help="clip tokens per temporal group (video endpoints)")
    p.add_argument("--calib", help="calibration JSON for /scan verdicts (scan.video_ops.Calib)")
    p.add_argument("--readout", choices=("yes", "short"), default="yes", help="clip scan readout")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    from aiohttp import web

    from glance_engine import checkpoints as C
    from scan import video_ops as VO
    from speed.decision_worker import DecisionWorker
    from speed.server import application

    loaded = C.load("tiny" if a.tiny else a.model, graphs=a.graphs)
    if loaded.family == "x":
        from glance_engine.x.worker import XDecisionWorker as DecisionWorker
    worker = DecisionWorker(loaded.engine, max_tokens=a.max_tokens, cache_bytes=int(a.cache_gib * 2**30),
                            packed_shared=a.shared_prefill == "packed", group_vision=a.group_vision)
    worker.engine.clip_max_vis = a.max_vis
    calib = None
    if a.calib:
        with open(a.calib) as f:
            calib = VO.Calib.from_json(json.load(f))
    app = application(worker, timeout=a.timeout, queue_limit=a.queue_limit, max_batch=a.max_batch,
                      wait_ms=a.wait_ms, calib=calib, short=a.readout == "short")
    print(f"glance-serve: {loaded.source} on {worker.engine.dev}, http://{a.host}:{a.port}", flush=True)
    web.run_app(app, host=a.host, port=a.port, handler_cancellation=True, print=None)


if __name__ == "__main__":
    main()
