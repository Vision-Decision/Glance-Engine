"""Set-level answers over N per-image scores, with finite-sample guarantees (split conformal, distribution-free).

Inputs are per-image scores s_i (e.g. Glance's presence logit or P(present); any score, larger = more likely present).
A "set" is one scan: N images/frames for one query. Calibration data are scans of the same kind (same query/video
distribution, same N or N-distribution) whose ground truth is known. Guarantees hold for a new scan exchangeable with
the calibration scans; they are marginal over scans (not conditional on a given query).

none_of_these  "the target is in none of these N images"
  MaxStat      T = max_i s_i over a NEGATIVE scan; tau = the ceil((n+1)(1-alpha))-th smallest of the n calibration
               maxima. Alarm iff max_i s_i > tau.  P(alarm | new negative scan) <= alpha, and >= alpha - 1/(n+1)
               (no ties).  No independence assumption between the N images: correlation between adjacent frames is
               absorbed by the max (this is the Westfall-Young / max-T idea, with a conformal quantile).
  Bonferroni   per-image conformal p-values from a pool of negative IMAGES (n_img of them):
               p_i = (1 + #{cal >= s_i}) / (n_img + 1); alarm iff min_i p_i <= alpha / N. Valid for any N and any
               dependence (union bound) if each negative test image is exchangeable with the pooled calibration
               images; conservative when frames are correlated.
  Naive        alarm iff max_i sigmoid(s_i) > 0.5 (per-image calibrated threshold, no set-level control).

select (top-k with abstention)   conformal p-values (as Bonferroni) + Benjamini-Hochberg at level q:
               E[#false selections / max(#selections, 1)] <= q  (conformal selection, Jin & Candes 2023: the
               p-values of null test images are PRDS given the calibration set). May return nothing.

first_occurrence   tau_first = conformal quantile of F_j = max over frames strictly BEFORE the onset (positive scans;
               the whole scan for negative scans). first = min{t : s_t > tau_first}.
               P(first alarm is earlier than the true onset, or any alarm on a negative scan) <= alpha.

count          images: number selected by `select`; events: number of runs of consecutive selected frames, merging
               gaps of <= `gap` frames. (No guarantee on the count itself; the selected set has FDR <= q.)
"""
import math
import numpy as np


def conformal_quantile(cal, alpha):
    """ceil((n+1)(1-alpha))-th smallest of cal; +inf if that rank exceeds n (too few calibration points)."""
    cal = np.sort(np.asarray(cal, dtype=float))
    n = len(cal)
    k = math.ceil((n + 1) * (1 - alpha))
    return float("inf") if k > n else float(cal[k - 1])


def conformal_pvalues(scores, cal_null, presorted=False):
    """p_i = (1 + #{cal_null >= s_i}) / (n + 1) (one-sided: large scores are evidence of presence)."""
    c = cal_null if presorted else np.sort(np.asarray(cal_null, dtype=float))
    s = np.asarray(scores, dtype=float)
    ge = len(c) - np.searchsorted(c, s, side="left")
    return (1.0 + ge) / (len(c) + 1.0)


def bh(p, q):
    """Benjamini-Hochberg: indices of rejected hypotheses (sorted by p)."""
    p = np.asarray(p)
    m = len(p)
    o = np.argsort(p)
    ok = np.nonzero(p[o] <= q * np.arange(1, m + 1) / m)[0]
    return o[: ok[-1] + 1] if len(ok) else o[:0]


class MaxStat:
    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, neg_scans):
        self.n = len(neg_scans)
        self.tau = conformal_quantile([np.max(s) for s in neg_scans], self.alpha)
        return self

    def alarm(self, scores):
        return bool(np.max(scores) > self.tau)


class MaxStatByN:
    """One MaxStat per scan length (scans of different N have different max distributions)."""

    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, neg_scans):
        by = {}
        for s in neg_scans:
            by.setdefault(len(s), []).append(s)
        self.m = {n: MaxStat(self.alpha).fit(v) for n, v in by.items()}
        return self

    def alarm(self, scores):
        if len(scores) not in self.m:
            raise ValueError(f"calibrated for scans of {sorted(self.m)} images, not {len(scores)}; "
                             "calibrate with negative scans of this length")
        return self.m[len(scores)].alarm(scores)


class Bonferroni:
    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, neg_images):
        self.cal = np.sort(np.asarray(neg_images, dtype=float))
        return self

    def alarm(self, scores):
        return bool(conformal_pvalues(scores, self.cal, presorted=True).min() <= self.alpha / len(scores))


class Naive:
    def __init__(self, thr=0.0):          # 0.0 on logits = 0.5 on probabilities
        self.thr = thr

    def fit(self, *_):
        return self

    def alarm(self, scores):
        return bool(np.max(scores) > self.thr)


class Select:
    """Top-k with abstention: BH over conformal p-values; returns selected indices ordered by score (<= k)."""

    def __init__(self, q, k=None):
        self.q, self.k = q, k

    def fit(self, neg_images):
        self.cal = np.sort(np.asarray(neg_images, dtype=float))
        return self

    def __call__(self, scores):
        s = np.asarray(scores, dtype=float)
        sel = bh(conformal_pvalues(s, self.cal, presorted=True), self.q)
        sel = sel[np.argsort(-s[sel])]
        return sel[: self.k] if self.k else sel


class FirstOccurrence:
    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, scans, onsets):
        """scans: list of score arrays (time order); onsets: index of the first positive frame, or None (negative)."""
        F = [np.max(s[:o]) if (o is not None and o > 0) else (-np.inf if o == 0 else np.max(s)) for s, o in zip(scans, onsets)]
        self.tau = conformal_quantile(F, self.alpha)
        return self

    def __call__(self, scores):
        hit = np.nonzero(np.asarray(scores) > self.tau)[0]
        return int(hit[0]) if len(hit) else None


def count_events(selected, gap=1):
    """Number of runs of selected frame indices, merging gaps of <= gap missing frames."""
    s = np.sort(np.asarray(selected))
    if len(s) == 0:
        return 0
    return int(1 + np.sum(np.diff(s) > gap + 1))


def family_false_alarm_independent(alpha_img, N):
    """1 - (1 - alpha)^N: the set-level false alarm of a per-image threshold if image errors were independent."""
    return 1 - (1 - alpha_img) ** N


# ------------------------------------------------------------------------------------------------ self-check
def _simulate(seed=0, trials=400, n_cal=200, alpha=0.1, rho=0.8, Ns=(8, 32, 128)):
    """Correlated AR(1) negative scans (frame scores ~ N(-2, 1) marginally, logit scale): empirical false alarm of
    every rule vs N. MaxStat must stay <= alpha (+ noise); Naive grows with N."""
    rng = np.random.default_rng(seed)

    def scan(N):
        e = np.empty(N)
        e[0] = rng.standard_normal()
        for t in range(1, N):
            e[t] = rho * e[t - 1] + math.sqrt(1 - rho ** 2) * rng.standard_normal()
        return -2.0 + e

    out = {}
    for N in Ns:
        fa = {"maxstat": [], "bonferroni": [], "naive": []}
        for _ in range(trials):
            cal = [scan(N) for _ in range(n_cal)]
            rules = dict(maxstat=MaxStat(alpha).fit(cal), bonferroni=Bonferroni(alpha).fit(np.concatenate(cal)),
                         naive=Naive().fit())
            test = scan(N)
            for k, r in rules.items():
                fa[k].append(r.alarm(test))
        out[N] = {k: float(np.mean(v)) for k, v in fa.items()}
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(_simulate(), indent=1))
