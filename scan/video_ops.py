"""Agent-facing video ops over one scan (one query x all clips of a video): curve, when, count, ever.

Input: per-clip scores s_i (logit scale, larger = more likely the event is in clip i) and clip times [t0_i, t1_i].
Calibration (Calib) comes from offline scans with known ground truth (scan/pilot_charades.py output): a pool of
per-clip scores of clips WITHOUT the event, and whole negative scans. It is model- and clip-format-specific.

  curve(s)          -> [(t0, t1, p)]                     p = sigmoid(s / T) (T: temperature fitted on the pool)
  ever(s, alpha)    -> {ever: bool, threshold, max_score, guarantee}
                       max-statistic conformal threshold for this N (negative calibration scans cut/concatenated to
                       N clips): P(ever=True | the event never happens) <= alpha for scans like the calibration ones.
                       Without enough calibration scans of length >= N, falls back to Bonferroni over the per-clip
                       pool (valid for any N, more conservative).
  when(s, q)        -> intervals: clips selected by BH over conformal p-values (FDR <= q among selected clips),
                       consecutive selected clips merged (gap <= 1 clip); each interval has [t0, t1], max p, argmax time.
                       Empty if ever() is False (the two answers are kept consistent: no interval without an alarm).
  count(s, q)       -> number of merged intervals from when() (events), plus the number of selected clips.
                       No guarantee on the count itself; repeated actions closer than one clip merge.
"""
import math
import numpy as np

from scan import setlevel as SL


class Calib:
    def __init__(self, null_pool, neg_scans, temperature=1.0, name=""):
        self.pool = np.sort(np.asarray(null_pool, dtype=float))
        self.negs = [np.asarray(s, dtype=float) for s in neg_scans]
        self.T, self.name = temperature, name
        self._tau = {}

    @classmethod
    def from_pilot(cls, scans, scorer, name=""):
        """scans: pilot rows (scan/pilot_charades.py): negatives give whole null scans, positives' negative frames are
        NOT used (they may be adjacent to the event)."""
        negs = [np.asarray(s["scores"][scorer]) for s in scans if not s["pos"]]
        pos_f = np.concatenate([np.asarray(s["scores"][scorer])[np.asarray(s["labels"])] for s in scans if s["pos"]])
        pool = np.concatenate(negs)
        return cls(pool, negs, temperature=fit_temperature(pos_f, pool), name=name)

    def to_json(self):
        return dict(pool=self.pool.round(4).tolist(), negs=[n.round(4).tolist() for n in self.negs], T=self.T, name=self.name)

    @classmethod
    def from_json(cls, d):
        return cls(d["pool"], d["negs"], d.get("T", 1.0), d.get("name", ""))

    def tau_max(self, N, alpha, seed=0):
        """max-statistic conformal threshold for scans of N clips; None if too few calibration scans."""
        k = (N, alpha)
        if k not in self._tau:
            rng = np.random.default_rng(seed)
            cat = [n for n in self.negs if len(n)]
            maxima = []
            if all(len(n) >= N for n in cat):
                maxima = [n[:N].max() for n in cat]                     # first N clips of each negative scan
            else:                                                       # concatenate negative scans to length N
                order = rng.permutation(len(cat))
                buf = []
                for i in order:
                    buf.append(cat[i])
                    if sum(map(len, buf)) >= N:
                        maxima.append(np.concatenate(buf)[:N].max()); buf = []
            need = math.ceil(1 / alpha) - 1
            self._tau[k] = SL.conformal_quantile(maxima, alpha) if len(maxima) >= need else None
        return self._tau[k]


def fit_temperature(pos, neg, grid=np.exp(np.linspace(-2, 2, 81))):
    """Temperature minimising the NLL of 'clip contains the event' on the calibration clips (balanced)."""
    s = np.r_[pos, neg]; y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    w = np.where(y == 1, 0.5 / max(len(pos), 1), 0.5 / max(len(neg), 1))
    best, bt = 1e9, 1.0
    for T in grid:
        z = s / T
        nll = (w * (np.logaddexp(0, -z) * y + np.logaddexp(0, z) * (1 - y))).sum()
        if nll < best:
            best, bt = nll, float(T)
    return bt


def curve(s, times, calib=None):
    T = calib.T if calib else 1.0
    p = 1 / (1 + np.exp(-np.asarray(s, dtype=float) / T))
    return [dict(t0=float(a), t1=float(b), p=round(float(x), 4)) for (a, b), x in zip(times, p)]


def ever(s, calib, alpha=0.1):
    s = np.asarray(s, dtype=float)
    N = len(s)
    tau = calib.tau_max(N, alpha)
    if tau is not None:
        return dict(ever=bool(s.max() > tau), threshold=float(tau), max_score=float(s.max()), method="maxstat",
                    guarantee=f"P(ever=True | never happens) <= {alpha} (max-statistic conformal, N={N})")
    p = SL.conformal_pvalues(s, calib.pool, presorted=True)
    return dict(ever=bool(p.min() <= alpha / N), min_p=float(p.min()), max_score=float(s.max()), method="bonferroni",
                guarantee=f"P(ever=True | never happens) <= {alpha} (Bonferroni over conformal p-values, N={N})")


def when(s, times, calib, q=0.1, alpha=0.1, gap=1, method="bh"):
    """method: 'bh' = BH over conformal p-values (FDR <= q vs the null pool); 'fwer' = clips above the set-level
    max-statistic threshold of ever() (P(any false clip) <= alpha on event-free videos; stricter, fewer clips)."""
    s = np.asarray(s, dtype=float)
    ev = ever(s, calib, alpha)
    if not ev["ever"]:
        return dict(intervals=[], ever=ev)
    if method == "fwer" and ev.get("method") == "maxstat":
        sel = np.where(s > ev["threshold"])[0]
    else:
        sel = np.sort(SL.Select(q).fit(calib.pool)(s))
    if len(sel) == 0:                       # alarm but nothing passes BH: report the argmax clip alone
        sel = np.array([int(s.argmax())])
    T = calib.T
    p = 1 / (1 + np.exp(-s / T))
    runs, cur = [], [int(sel[0])]
    for i in sel[1:]:
        if i - cur[-1] <= gap + 1:
            cur.append(int(i))
        else:
            runs.append(cur); cur = [int(i)]
    runs.append(cur)
    out = []
    for r in runs:
        lo, hi = r[0], r[-1]
        j = lo + int(np.argmax(s[lo:hi + 1]))
        out.append(dict(t0=float(times[lo][0]), t1=float(times[hi][1]), p_max=round(float(p[j]), 4),
                        t_peak=float((times[j][0] + times[j][1]) / 2), n_clips=len(r)))
    out.sort(key=lambda d: -d["p_max"])
    g = (f"selected clips: expected false fraction <= {q} (BH on conformal p-values)" if method != "fwer" else
         f"selected clips: P(any selected clip on an event-free video) <= {alpha} (max-statistic)")
    return dict(intervals=out, ever=ev, n_selected=int(len(sel)), method=method, guarantee=g)


def count(s, times, calib, q=0.1, alpha=0.1, gap=1, method="bh"):
    w = when(s, times, calib, q, alpha, gap, method)
    return dict(count=len(w["intervals"]), n_selected_clips=w.get("n_selected", 0), intervals=w["intervals"],
                note="events = runs of selected clips (gap <= 1 clip merged); repeats closer than a clip merge")
