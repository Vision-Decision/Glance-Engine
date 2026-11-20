"""Scan banks: retained per-image or per-clip prefix state (stable import path).

A bank stores, for every unit, the GDN recurrent state and conv tail of each linear-attention
layer, the K/V of each full-attention layer, the visual hidden states and the box head's
query-independent keys. ``ScanEngine.encode(units)`` builds one; ``ScanEngine.query(bank, q)``
answers a new question from it.

================  ==========================================================  ======================
Name              What it does                                                Implementation
================  ==========================================================  ======================
StoredBank        compact device storage (lossless by default)                ``scan/bank.py``
FORMATS           legacy, compact, state_packed (lossless); bf16/fp8/int8     ``scan/bank.py``
encode/decode     exact exponent-window packing of FP32 recurrent state       ``scan/bank_codec.py``
HostBank          pinned host tier streamed through two device slots          ``scan/host_bank.py``
================  ==========================================================  ======================
"""

from scan.bank import FORMATS, StoredBank
from scan.bank_codec import decode, encode
from scan.host_bank import HostBank

__all__ = ["FORMATS", "HostBank", "StoredBank", "decode", "encode"]
