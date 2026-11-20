"""Scheduler, device worker and HTTP application (stable import path).

==================  ===============================================================  ==========================
Name                What it does                                                     Implementation
==================  ===============================================================  ==========================
Scheduler           FIFO admission, deadline-bounded batching for one device owner   ``speed/scheduler.py``
DecisionWorker      batched BF16 decisions, prefix cache, one device thread          ``speed/decision_worker.py``
Decision            one request: images (or cached ids) x typed questions             ``speed/decision_worker.py``
PrefixCache         byte-bounded LRU shared by image prefixes and video banks        ``speed/decision_worker.py``
application         aiohttp app: /v1/decisions, /v1/chat/completions, /encode, ...   ``speed/server.py``
==================  ===============================================================  ==========================
"""

from speed.decision_worker import Decision, DecisionWorker, ImageRef, PrefixCache
from speed.scheduler import Overloaded, Scheduler, Unavailable


def application(*args, **kwargs):
    """``speed.server.application``; imported lazily because it needs the ``[serve]`` extra (aiohttp)."""
    from speed.server import application as app

    return app(*args, **kwargs)


__all__ = [
    "Decision", "DecisionWorker", "ImageRef", "Overloaded", "PrefixCache", "Scheduler", "Unavailable", "application",
]
