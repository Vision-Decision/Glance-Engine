"""Analyse the Charades-STA scan pilot (CPU, small files).

python -m scan.pilot_analyze IN.json.gz [--out outputs/scan/pilot_charades.json] [--splits 50] [--alpha 0.1]

For every scorer:
  frame AUROC        positive frames (inside the query's action) vs frames of negative scans
  scan AUROC         max-over-frames of positive vs negative scans
  none-of-these      scans of length N (N in 8, 16, 32 by uniform subsampling; 64, 128, 256 by concatenating the
                     query's negative scans over several videos: branches are independent, so a long scan's scores
                     are exactly the concatenation). Rules: Naive (logit > 0 = p > .5), Bonferroni, MaxStat (per N).
                     Calibration / test split BY QUERY (random half), repeated `splits` times: test false alarm
                     (mean, and the fraction of splits above alpha) and test miss rate on positive scans of the same N.
  first occurrence   FirstOccurrence(alpha) on natural-length scans: P(alarm before onset or on a negative scan),
                     detection rate, median |first - onset| in frames
"""
import argparse, gzip, json, math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scan.setlevel import MaxStat, Bonferroni, Naive, FirstOccurrence, Select, count_events  # noqa: E402


def auroc(pos, neg):
    s = np.concatenate([pos, neg]); y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    return float((r[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def subsample(x, N, rng):
    if len(x) >= N:
        st = rng.integers(0, max(1, len(x) // N))
        idx = np.linspace(st, len(x) - 1, N).round().astype(int)
        return x[idx]
    return None


def build_sets(scans, scorer, N, rng):
    """-> dict qid -> (list of negative scans of length N, list of (pos scan, labels) of length N)."""
    by = {}
    for s in scans:
        by.setdefault(s["qid"], []).append(s)
    out = {}
    for qid, ss in by.items():
        negs = [np.asarray(s["scores"][scorer]) for s in ss if not s["pos"]]
        poss = [(np.asarray(s["scores"][scorer]), np.asarray(s["labels"])) for s in ss if s["pos"]]
        if N <= 32:
            nn = [subsample(x, N, rng) for x in negs]
            pp = []
            for x, y in poss:
                if len(x) >= N and y.any():
                    idx = np.linspace(0, len(x) - 1, N).round().astype(int)
                    if y[idx].any():
                        pp.append(x[idx])
            nn = [x for x in nn if x is not None]
        else:   # concatenate this query's negative videos (cycled) to length N; positives: own video + negative fillers
            cat = np.concatenate(negs) if negs else np.zeros(0)
            nn = [cat[:N]] if len(cat) >= N else []          # distinct frames only (no recycling)
            pp = []
            for x, y in poss:
                if y.any() and len(x) + len(cat) >= N:
                    pp.append(np.concatenate([x, cat[: N - len(x)]]) if len(x) < N else x[:N])
        out[qid] = (nn, pp)
    return out


def none_of_these(scans, scorer, Ns, alpha, splits, seed=0):
    res = {}
    for N in Ns:
        rng = np.random.default_rng(seed + N)
        sets = build_sets(scans, scorer, N, rng)
        qids = np.array(sorted(sets))
        acc = {k: dict(fa=[], miss=[]) for k in ("naive", "bonferroni", "maxstat")}
        for sp in range(splits):
            r = np.random.default_rng(1000 + sp)
            perm = r.permutation(len(qids))
            cal, test = qids[perm[: len(qids) // 2]], qids[perm[len(qids) // 2:]]
            cal_neg = [x for q in cal for x in sets[q][0]]
            if len(cal_neg) < 20:
                continue
            rules = dict(naive=Naive(0.0), bonferroni=Bonferroni(alpha).fit(np.concatenate(cal_neg)),
                         maxstat=MaxStat(alpha).fit(cal_neg))
            tn = [x for q in test for x in sets[q][0]]
            tp = [x for q in test for x in sets[q][1]]
            for k, rule in rules.items():
                acc[k]["fa"].append(np.mean([rule.alarm(x) for x in tn]))
                if tp:
                    acc[k]["miss"].append(np.mean([not rule.alarm(x) for x in tp]))
        n_neg = sum(len(v[0]) for v in sets.values()); n_pos = sum(len(v[1]) for v in sets.values())
        res[N] = dict(n_neg_scans=n_neg, n_pos_scans=n_pos, **{
            k: dict(false_alarm=float(np.mean(v["fa"])) if v["fa"] else None,
                    false_alarm_sd=float(np.std(v["fa"])) if v["fa"] else None,
                    frac_splits_fa_above_alpha=float(np.mean(np.array(v["fa"]) > alpha)) if v["fa"] else None,
                    miss=float(np.mean(v["miss"])) if v["miss"] else None) for k, v in acc.items()})
    return res


def first_occ(scans, scorer, alpha, splits):
    by = {}
    for s in scans:
        by.setdefault(s["qid"], []).append(s)
    qids = np.array(sorted(by))
    out = dict(early_or_false=[], detect=[], abs_err=[])
    for sp in range(splits):
        r = np.random.default_rng(2000 + sp)
        perm = r.permutation(len(qids))
        cal, test = qids[perm[: len(qids) // 2]], qids[perm[len(qids) // 2:]]
        get = lambda qs: [(np.asarray(s["scores"][scorer]), s["onset_frame"] if s["pos"] else None) for q in qs for s in by[q]
                          if (not s["pos"]) or (s["onset_frame"] is not None and s["onset_frame"] < len(s["scores"][scorer]))]
        c = get(cal)
        fo = FirstOccurrence(alpha).fit([x for x, _ in c], [o for _, o in c])
        bad, det, err = [], [], []
        for x, o in get(test):
            f = fo(x)
            if o is None:
                bad.append(f is not None)
            else:
                bad.append(f is not None and f < o)
                det.append(f is not None and f >= o)
                if f is not None and f >= o:
                    err.append(f - o)
        out["early_or_false"].append(np.mean(bad)); out["detect"].append(np.mean(det)); out["abs_err"] += err
    return dict(early_or_false_alarm=float(np.mean(out["early_or_false"])), detect_rate=float(np.mean(out["detect"])),
                median_frames_after_onset=float(np.median(out["abs_err"])) if out["abs_err"] else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("--out", default="")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--splits", type=int, default=50)
    ap.add_argument("--Ns", default="8,16,32,64,128,256")
    a = ap.parse_args()
    d = json.load(gzip.open(a.inp, "rt"))
    scans = [s for s in d["scans"] if "scores" in s]
    scorers = sorted(scans[0]["scores"])
    Ns = [int(x) for x in a.Ns.split(",")]
    res = dict(n_scans=len(scans), n_pos=sum(s["pos"] for s in scans), n_queries=len({s["qid"] for s in scans}),
               timing=d.get("timing"), alpha=a.alpha, scorers={})
    for sc in scorers:
        pf = np.concatenate([np.asarray(s["scores"][sc])[np.asarray(s["labels"])] for s in scans if s["pos"]])
        nf = np.concatenate([np.asarray(s["scores"][sc]) for s in scans if not s["pos"]])
        pm = np.array([max(s["scores"][sc]) for s in scans if s["pos"]]); nm = np.array([max(s["scores"][sc]) for s in scans if not s["pos"]])
        r = dict(frame_auroc=auroc(pf, nf), scan_max_auroc=auroc(pm, nm),
                 frame_false_alarm_naive=float((nf > 0).mean()), frame_hit_naive=float((pf > 0).mean()),
                 none_of_these=none_of_these(scans, sc, Ns, a.alpha, a.splits),
                 first_occurrence=first_occ(scans, sc, a.alpha, a.splits))
        res["scorers"][sc] = r
        print(sc, json.dumps({k: v for k, v in r.items() if k != "none_of_these"}), flush=True)
        for N, v in r["none_of_these"].items():
            print(f"  N={N:4d}", {k: (round(x["false_alarm"], 3) if x["false_alarm"] is not None else None,
                                     round(x["miss"], 3) if x["miss"] is not None else None)
                                  for k, x in v.items() if isinstance(x, dict)}, flush=True)
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
