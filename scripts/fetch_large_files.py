"""Restore the >5 MB data and result files that live on Hugging Face instead of in git.

    python scripts/fetch_large_files.py            # all files listed in LARGE_FILES.tsv
    python scripts/fetch_large_files.py data/prep  # only paths under a prefix
    python scripts/fetch_large_files.py --check    # verify files already in place

Each row of LARGE_FILES.tsv is `path  bytes  sha256`. The file is stored at
Vision-Decision/glance-train-data (dataset, private) under git_large_files/<path>, and is written back to <path>
in this checkout after its sha256 is verified. Needs a token with read access to the Vision-Decision org.
"""
import argparse, hashlib, os, shutil, sys

REPO, PREFIX = "Vision-Decision/glance-train-data", "git_large_files"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefixes", nargs="*")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    rows = [l.rstrip("\n").split("\t") for l in open(os.path.join(ROOT, "LARGE_FILES.tsv")) if l.strip() and not l.startswith("#")]
    rows = [r for r in rows if not a.prefixes or any(r[0].startswith(p) for p in a.prefixes)]
    bad = 0
    for path, size, digest in rows:
        dst = os.path.join(ROOT, path)
        if os.path.exists(dst) and sha256(dst) == digest:
            print("ok     ", path); continue
        if a.check:
            print("MISSING", path); bad += 1; continue
        from huggingface_hub import hf_hub_download
        src = hf_hub_download(REPO, f"{PREFIX}/{path}", repo_type="dataset")
        if sha256(src) != digest:
            print("SHA MISMATCH", path, file=sys.stderr); bad += 1; continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        print("fetched", path, f"({int(size) / 1e6:.1f} MB)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
