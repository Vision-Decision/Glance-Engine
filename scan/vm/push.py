"""Push files to HF right away: python scan/vm/push.py PREFIX FILE [FILE ...]
Results go to the org dataset Vision-Decision/collection-scan-results; if the org's private storage is full (403), to the
fallback dataset Vision-Decision/collection-scan-data (same paths)."""
import os, sys
from huggingface_hub import HfApi
api = HfApi()
for f in sys.argv[2:]:
    if not os.path.exists(f):
        continue
    for repo in ("Vision-Decision/collection-scan-results", "Vision-Decision/collection-scan-data"):
        try:
            api.upload_file(path_or_fileobj=f, path_in_repo=f"{sys.argv[1]}/{os.path.basename(f)}", repo_id=repo, repo_type="dataset")
            print("pushed", f, "->", repo, flush=True)
            break
        except Exception as e:
            print("push failed", f, repo, str(e)[:120], flush=True)
