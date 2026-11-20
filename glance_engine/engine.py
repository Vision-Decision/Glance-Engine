"""Engine core: the stable import path for the fused engine and the scan engine.

=====================  ===========================================================  ======================
Name                   What it does                                                 Implementation
=====================  ===========================================================  ======================
GlanceEngine           one image, many questions: fused prefill + question blocks   ``speed/engine.py``
ScanEngine             one question, many images/clips: banks, fresh packed scan    ``scan/engine.py``
FreshPack              packed variable-length prefill of N prefixes + questions     ``scan/varlen.py``
merge_lora_ ...        weight preparation used by every benchmark                   ``speed/engine.py``
=====================  ===========================================================  ======================
"""

from scan.engine import ScanEngine, image_item, make_clip_item, make_item
from scan.varlen import FreshPack, packed_forward
from speed.engine import GlanceEngine, fp8_language_model_, fuse_weights_, merge_lora_

__all__ = [
    "FreshPack", "GlanceEngine", "ScanEngine", "fp8_language_model_", "fuse_weights_", "image_item",
    "make_clip_item", "make_item", "merge_lora_", "packed_forward",
]
