"""Glance-X models on the Glance engine.

Glance-X (`glancex <https://github.com/Vision-Decision/Glance-X-MLX>`_) is a box model with one backbone and three
reading directions. It differs from Glance v1 in ways the v1 fused engine cannot absorb, so it has its own backend
here, built on glancex's exact prefix/block layer kernels:

* **Modality stage.** In the first ``K`` language-model layers (``cfg.bidir_k``: 8 for Glance-X-4B and X2, 2 for
  X-Lite) the image stream and the request stream run separately. A request block only joins its image prefix from
  layer ``K + 1``. The backend runs the modality stage once per distinct request and shares it across every image.
* **Token layout.** No chat template: ``<vision_start> image <vision_end>`` with M-RoPE from 0, and
  ``<ref> request </ref> <box_start>`` from the fixed position ``cfg.text_pos0``. The readout is ``<box_start>``.
* **Directions.** ``image_first`` (image prefix, request blocks; cacheable), ``text_first`` (request prefix, image
  blocks) and ``joint`` (no prefix). A checkpoint only answers in the directions it was trained in.
* **Heads.** Box questions only: P(present), one box, P(IoU >= 0.5), plus every target from the multi-box head.

What it shares with the v1 engine: the ``Glance`` Python API, typed ``Box`` questions and answers, the byte-bounded
prefix cache with image ids, the request scheduler and ``glance-serve`` HTTP server, and the ``Scanner`` result
format including the conformal set answers.

==================  ====================================================================
``XEngine``         encode image prefixes, answer request blocks, all three directions
``XDecisionWorker`` drop-in for ``speed.decision_worker.DecisionWorker`` (API and server)
``XScanner``        drop-in for ``scan.api.Scanner``: cached banks, fresh, text_first
``load`` / ``tiny`` checkpoints: Hugging Face release, exported folder, training ``.pt``
==================  ====================================================================
"""

from glance_engine.x.model import X_CONFIG, is_x_engine, load, tiny

__all__ = ["X_CONFIG", "XDecisionWorker", "XEngine", "XScanner", "is_x_engine", "load", "tiny"]


def __getattr__(name):  # keep `import glance_engine` light: glancex / torch load on first use
    if name == "XEngine":
        from glance_engine.x.engine import XEngine

        return XEngine
    if name == "XDecisionWorker":
        from glance_engine.x.worker import XDecisionWorker

        return XDecisionWorker
    if name == "XScanner":
        from glance_engine.x.scanner import XScanner

        return XScanner
    raise AttributeError(name)
