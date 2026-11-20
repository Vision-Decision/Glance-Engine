"""The tiny-config exactness suite (scan/test_scan_equivalence.py) on CPU in FP32.

Checks cached, fresh and query-first scans, packed variable-length prefill, clips and the
Scanner API against running each image alone through the glance-core reference forward
and the plain HF forward. See docs/exactness.md.
"""

import os
import subprocess
import sys

import pytest


@pytest.mark.slow
def test_scan_equivalence_tiny_fp32():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    run = subprocess.run([sys.executable, "-m", "scan.test_scan_equivalence"], cwd=root, env=env,
                         capture_output=True, text=True, timeout=1800)
    tail = run.stdout[-4000:] + run.stderr[-2000:]
    assert run.returncode == 0 and "SCAN_EQUIVALENCE_OK" in run.stdout, tail
