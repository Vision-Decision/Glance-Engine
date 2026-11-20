"""Teacher targets for the short-readout self-distillation, then training records for train.py.

python -m scan.distill_teacher --ckpt /content/ckpt/v1/ckpt_final.pt --data /content/distill --out /content/distill/train.jsonl

For every clip (scan/distill_prep.py) and each of its queries, the frozen model's P(yes) on the LONG yes/no block
  "Does this clip show: {sentence}?"  (+ "Answer yes or no." + assistant template)
becomes the soft target of two questions in the training record:
  long  : the same long block (anchors the original readout: the model must not drift)
  short : {"type": "noul", "instr": sentence, "block": "short"}  ->  '\\n<ref>{sentence}</ref><im_end>'
Records are dv/data/video.py video records (typed questions only), so train.py consumes them unchanged.
"""
import argparse, json, os
import numpy as np
import torch

from dv.data import video as V
from scan.pilot_charades import load_glance


def long_q(text):
    return dict(type="noul", instr=f"Does this clip show: {text}?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(f"{a.data}/records.jsonl")]
    if a.limit:
        recs = recs[: a.limit]
    eng, cfg = load_glance(a.ckpt)
    eng.dt, eng.clip_max_vis = 0.5, 96
    n_q, ps = 0, []
    with open(a.out, "w") as f, torch.no_grad():
        for s in range(0, len(recs), a.chunk):
            rr = recs[s:s + a.chunk]
            clips = [V.decode_clip(f"{a.data}/{r['video']}")[:4] for r in rr]
            keep = [i for i, c in enumerate(clips) if len(c) >= 2]
            rr, clips = [rr[i] for i in keep], [clips[i] for i in keep]
            V._CACHE.clear()
            qss = [[long_q(q["text"]) for q in r["queries"]] for r in rr]
            batch, e, host = eng.build(clips, qss)
            o, _, _ = eng._core(eng.to_device(e, host, eng.feats(batch)), host)
            tl = o["typed_logits"].float().cpu()
            p = torch.softmax(tl[:, :2], -1)[:, 1].numpy()
            m = 0
            for r, qs in zip(rr, qss):
                questions = []
                for q in r["queries"]:
                    pt = float(p[m]); m += 1
                    ps.append(pt)
                    questions.append(dict(long_q(q["text"]), target=pt))
                    questions.append(dict(type="noul", instr=q["text"], block="short", target=pt))
                f.write(json.dumps(dict(video=r["video"], frames=r["frames"], dt=r["dt"], max_vis=r["max_vis"],
                                        plen=V.est_prefix_tokens(4, 448, 336, 96), questions=questions)) + "\n")
                n_q += len(r["queries"])
            if (s // a.chunk) % 50 == 0:
                print(f"clips {s + len(rr)}/{len(recs)} queries {n_q} mean p {np.mean(ps):.3f} "
                      f"frac>0.5 {np.mean(np.array(ps) > 0.5):.3f}", flush=True)
    print("TEACHER_DONE", n_q, "mean p", float(np.mean(ps)), flush=True)


if __name__ == "__main__":
    main()
