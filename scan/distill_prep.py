"""Clips + queries for the short-readout self-distillation (Charades TRAIN split only; never a test split).

python3 scan/distill_prep.py --root ~/scan/charades --out ~/scan/distill [--clips_per_video 4] [--workers 28]

Each clip = 4 frames at 2 fps (vdm1 layout: 2 s, dt 0.5), short side <= 336, written as a tiny mp4
(clips/{vid}_{k}.mp4, 4 frames). Queries per clip: 2 STA-train sentences of the same video (often relevant, sometimes
not, depending on the clip time) + 2 sentences of random other videos. Output records.jsonl, one line per clip:
  {"video": "clips/X_k.mp4", "frames": [0,1,2,3], "dt": 0.5, "max_vis": 96, "t0": start_s, "vid": X,
   "queries": [{"text": sentence, "same_video": bool, "moment": [s, e] | null}, ...]}
Teacher probabilities are added on the GPU (scan/distill_teacher.py).
"""
import argparse, json, os, random, subprocess, zipfile
from concurrent.futures import ProcessPoolExecutor


def cut(args):
    zp, member, vid, out, starts = args
    tmp = f"{out}/tmp_{vid}.mp4"
    done = []
    try:
        with zipfile.ZipFile(zp) as z, open(tmp, "wb") as f:
            f.write(z.read(member))
        for k, t0 in enumerate(starts):
            dst = f"{out}/clips/{vid}_{k}.mp4"
            if not os.path.exists(dst):
                subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", f"{t0:.2f}", "-i", tmp, "-vf",
                                "fps=2,scale='if(gt(iw,ih),-2,336)':'if(gt(iw,ih),336,-2)'", "-frames:v", "4",
                                "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-y", dst], check=False)
            if os.path.exists(dst):
                done.append((k, t0))
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return vid, done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips_per_video", type=int, default=4)
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    os.makedirs(f"{a.out}/clips", exist_ok=True)
    sta = {}
    for l in open(f"{a.root}/charades_sta_train.txt"):
        head, text = l.strip().split("##")
        vid, s, e = head.split()
        sta.setdefault(vid, []).append((text.strip().rstrip("."), float(s), float(e)))
    import csv
    length = {r["id"]: float(r["length"]) for r in csv.DictReader(open(f"{a.root}/Charades/Charades_v1_train.csv"))}
    zp = f"{a.root}/Charades_v1_480.zip"
    names = {os.path.splitext(os.path.basename(n))[0]: n for n in zipfile.ZipFile(zp).namelist() if n.endswith(".mp4")}
    vids = sorted(v for v in sta if v in names and v in length)
    jobs = []
    for v in vids:
        L = max(2.5, length[v])
        starts = sorted(rng.uniform(0, L - 2.0) for _ in range(a.clips_per_video))
        jobs.append((zp, names[v], v, a.out, starts))
    all_sent = [(v, t) for v in vids for t in sta[v]]
    n = 0
    with ProcessPoolExecutor(a.workers) as ex, open(f"{a.out}/records.jsonl", "w") as f:
        for i, (vid, done) in enumerate(ex.map(cut, jobs, chunksize=4)):
            for k, t0 in done:
                own = rng.sample(sta[vid], min(2, len(sta[vid])))
                qs = [dict(text=t, same_video=True, moment=[s, e]) for t, s, e in own]
                while len(qs) < 4:
                    ov, (t, s, e) = rng.choice(all_sent)
                    if ov != vid:
                        qs.append(dict(text=t, same_video=False, moment=None))
                f.write(json.dumps(dict(video=f"clips/{vid}_{k}.mp4", frames=[0, 1, 2, 3], dt=0.5, max_vis=96, t0=t0,
                                        vid=vid, queries=qs)) + "\n")
                n += 1
            if i % 500 == 0:
                print("videos", i, len(jobs), "clips", n, flush=True)
    print("DONE clips", n, flush=True)


if __name__ == "__main__":
    main()
