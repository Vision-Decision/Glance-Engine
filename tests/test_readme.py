"""The README quickstart runs as written (on the tiny model), and every relative link in README and docs resolves."""

import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]


def python_blocks(text):
    return re.findall(r"```python\n(.*?)```", text, re.S)


@pytest.mark.slow
def test_readme_quickstart_runs_on_tiny(monkeypatch):
    blocks = python_blocks((ROOT / "README.md").read_text())
    assert len(blocks) >= 2, "README should show the image and the scan examples"
    code = "\n".join(blocks[:2])
    assert 'Glance.from_pretrained("Vision-Decision/Glance-4B-preview")' in code
    code = code.replace('Glance.from_pretrained("Vision-Decision/Glance-4B-preview")', "Glance.tiny()")
    monkeypatch.chdir(ROOT)
    scope = {}
    exec(compile(code, "README.md", "exec"), scope)
    assert len(scope["answers"]) == 3 and len(scope["answers"].image_id) == 64
    assert "encode" not in scope["result"]["latency_ms"]


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_relative_links_resolve(doc):
    text = doc.read_text()
    targets = re.findall(r"\]\(([^)\s]+)\)", text) + re.findall(r'(?:src|srcset)="([^"]+)"', text)
    targets += re.findall(r"^\[[^\]]+\]:\s*(\S+)", text, re.M)
    missing = []
    for t in targets:
        if re.match(r"[a-z]+:", t) or t.startswith("#"):
            continue
        path = t.split("#")[0]
        if path and not (doc.parent / path).exists():
            missing.append(t)
    assert not missing, f"{os.path.relpath(doc, ROOT)}: {missing}"
