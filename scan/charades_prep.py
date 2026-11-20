"""Charades-STA (test) frame-scan pilot data. Runs on a CPU host (gen-host); nothing large goes to the laptop.

python3 scan/charades_prep.py --root ~/scan/charades --out ~/scan/charades_pilot [--fps 1] [--workers 28]

Inputs in --root: charades_sta_test.txt (3720 query-moment pairs, 1334 videos), Charades/Charades_v1_test.csv (per-video
action instances 'cXXX start end'), Charades/Charades_v1_mapping.txt (class -> object, verb), Charades_v1_480.zip.
Outputs in --out:
  frames/{vid}/{k:03d}.jpg   frame at t = k / fps seconds (ffmpeg fps filter), longer side <= 640 px, JPEG q 90
  meta.json  videos: {vid: {n_frames, duration, classes}}, queries: [{qid, vid, text, cls, s, e, instances}],
             class_obj: {cls: object id}
Each STA query is mapped to the Charades action class whose instance in the same video best overlaps (s, e)
(temporal IoU >= 0.5, else the query is dropped); its positive frames are ALL instances of that class in the video
(so first occurrence and counts use the full annotation, not only the STA moment).
"""
import argparse, csv, json, os, subprocess, zipfile
from concurrent.futures import ProcessPoolExecutor


def tiou(a, b):
    i = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    u = max(a[1], b[1]) - min(a[0], b[0])
    return i / u if u > 0 else 0.0


def extract(args):
    zp, member, vid, out, fps = args
    d = f"{out}/frames/{vid}"
    if os.path.isdir(d) and len(os.listdir(d)) > 0:
        return vid, len(os.listdir(d))
    os.makedirs(d, exist_ok=True)
    tmp = f"{out}/tmp_{vid}.mp4"
    with zipfile.ZipFile(zp) as z, open(tmp, "wb") as f:
        f.write(z.read(member))
    subprocess.run(["ffmpeg", "-loglevel", "error", "-i", tmp, "-vf",
                    f"fps={fps},scale='if(gt(iw,ih),min(640,iw),-2)':'if(gt(iw,ih),-2,min(640,ih))'",
                    "-q:v", "3", f"{d}/%03d.jpg"], check=False)
    os.remove(tmp)
    fs = sorted(os.listdir(d))                  # ffmpeg numbers from 1; rename to 0-based k (t = k / fps)
    for i, fn in enumerate(fs):
        os.rename(f"{d}/{fn}", f"{d}/{i:03d}.jpg.tmp")
    for i in range(len(fs)):
        os.rename(f"{d}/{i:03d}.jpg.tmp", f"{d}/{i:03d}.jpg")
    return vid, len(fs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=28)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    acts, length = {}, {}
    for r in csv.DictReader(open(f"{a.root}/Charades/Charades_v1_test.csv")):
        inst = []
        for x in (r["actions"] or "").split(";"):
            if x.strip():
                c, s, e = x.split()
                inst.append((c, float(s), float(e)))
        acts[r["id"]] = inst
        length[r["id"]] = float(r["length"])
    class_obj = {}
    for l in open(f"{a.root}/Charades/Charades_v1_mapping.txt"):
        c, o, v = l.split()
        class_obj[c] = o
    queries, dropped = [], 0
    for i, l in enumerate(open(f"{a.root}/charades_sta_test.txt")):
        head, text = l.strip().split("##")
        vid, s, e = head.split()
        s, e = float(s), float(e)
        best = max(acts.get(vid, []), key=lambda x: tiou((s, e), x[1:]), default=None)
        if best is None or tiou((s, e), best[1:]) < 0.5:
            dropped += 1
            continue
        cls = best[0]
        inst = sorted([[x[1], x[2]] for x in acts[vid] if x[0] == cls])
        queries.append(dict(qid=i, vid=vid, text=text.strip().rstrip("."), cls=cls, s=s, e=e, instances=inst))
    vids = sorted({q["vid"] for q in queries})
    print(f"queries {len(queries)} (dropped {dropped}), videos {len(vids)}", flush=True)
    zp = f"{a.root}/Charades_v1_480.zip"
    names = {os.path.splitext(os.path.basename(n))[0]: n for n in zipfile.ZipFile(zp).namelist() if n.endswith(".mp4")}
    # frames of every Charades TEST video (negative scans come from videos without the query's class)
    test_vids = sorted(v for v in acts if v in names)
    jobs = [(zp, names[v], v, a.out, a.fps) for v in test_vids]
    nf = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for k, (vid, n) in enumerate(ex.map(extract, jobs, chunksize=4)):
            nf[vid] = n
            if k % 200 == 0:
                print("frames", k, len(jobs), flush=True)
    videos = {v: dict(n_frames=nf[v], duration=length[v], classes=sorted({x[0] for x in acts[v]})) for v in test_vids if nf.get(v)}
    json.dump(dict(fps=a.fps, videos=videos, queries=[q for q in queries if q["vid"] in videos], class_obj=class_obj),
              open(f"{a.out}/meta.json", "w"))
    print("DONE videos", len(videos), "frames", sum(v["n_frames"] for v in videos.values()), flush=True)


if __name__ == "__main__":
    main()
