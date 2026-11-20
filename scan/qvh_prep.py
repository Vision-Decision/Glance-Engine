"""QVHighlights val -> scan pilot data in the Charades format (scan/charades_prep.py), for E3/E4.

python scan/qvh_prep.py --tar videos.tar.gz --ann highlight_val_release.jsonl --out /content/qvh_pilot [--fps 2]

Outputs in --out:
  frames/{vid}/{k:03d}.jpg   t = k / fps, longer side <= 640 px
  meta.json  videos: {vid: {n_frames, duration, classes: [], source}}, queries: [{qid, vid, text, cls, instances}],
             class_obj: {}
Every query is its own "class" (cls = 'q<qid>'), its positive windows are the annotated relevant_windows.
`source` = the YouTube id (vid minus its _start_end suffix): negative scans must come from another source (the same
vlog often contains the same scene); this is enforced in scan/pilot_charades.py when `source` is present.
Label caveat: a negative video can still contain the queried content (open-vocabulary queries); measured false
alarms on QVHighlights are therefore upper bounds.
"""
import argparse, json, os, subprocess, tarfile
from concurrent.futures import ProcessPoolExecutor


def extract(args):
    mp4, vid, out, fps = args
    d = f"{out}/frames/{vid}"
    os.makedirs(d, exist_ok=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-i", mp4, "-vf",
                    f"fps={fps},scale='if(gt(iw,ih),min(640,iw),-2)':'if(gt(iw,ih),-2,min(640,ih))'",
                    "-q:v", "3", f"{d}/%03d.jpg"], check=False)
    os.remove(mp4)
    fs = sorted(os.listdir(d))
    for i, fn in enumerate(fs):
        os.rename(f"{d}/{fn}", f"{d}/{i:03d}.jpg.tmp")
    for i in range(len(fs)):
        os.rename(f"{d}/{i:03d}.jpg.tmp", f"{d}/{i:03d}.jpg")
    return vid, len(fs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", required=True)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(f"{a.out}/tmp", exist_ok=True)
    Q = [json.loads(l) for l in open(a.ann)]
    want = {q["vid"] for q in Q}
    jobs, nf = [], {}
    with ProcessPoolExecutor(a.workers) as ex, tarfile.open(a.tar, "r|gz") as tf:
        futs = []
        for m in tf:
            if not m.isfile() or not m.name.endswith(".mp4"):
                continue
            vid = os.path.splitext(os.path.basename(m.name))[0]
            if vid not in want:
                continue
            p = f"{a.out}/tmp/{vid}.mp4"
            with open(p, "wb") as f:
                f.write(tf.extractfile(m).read())
            futs.append(ex.submit(extract, (p, vid, a.out, a.fps)))
            if len(futs) % 200 == 0:
                print("queued", len(futs), flush=True)
        for f in futs:
            vid, n = f.result()
            nf[vid] = n
    videos = {v: dict(n_frames=n, duration=n / a.fps, classes=[], source=v.rsplit("_", 2)[0]) for v, n in nf.items() if n}
    queries = [dict(qid=q["qid"], vid=q["vid"], text=q["query"].strip().rstrip("."), cls=f"q{q['qid']}",
                    instances=[[float(s), float(e)] for s, e in q["relevant_windows"]]) for q in Q if q["vid"] in videos]
    json.dump(dict(fps=a.fps, videos=videos, queries=queries, class_obj={}), open(f"{a.out}/meta.json", "w"))
    print("DONE videos", len(videos), "queries", len(queries), "frames", sum(nf.values()), flush=True)


if __name__ == "__main__":
    main()
