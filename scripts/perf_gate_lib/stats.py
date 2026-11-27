"""Statistics for perf_gate: medians, a bimodality test, a rank test, and the
tolerance a verdict is judged against.

Everything here is small-sample on purpose (N = 7 to 21 per side): no
distributional assumptions, only order statistics and ranks.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

# MAD -> standard deviation for a normal sample
_MAD_SIGMA = 1.4826
# a median's standard error is about this many sigma / sqrt(n)
_MEDIAN_SE = 1.2533
# timer, interpreter and pipe jitter below which no two statements can be told apart
ABS_FLOOR_MS = 0.02


def median(xs: Sequence[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def quantile(xs: Sequence[float], q: float) -> float:
    s = sorted(xs)
    if not s:
        return float("nan")
    pos = q * (len(s) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def rel_sigma(xs: Sequence[float]) -> float:
    """Robust relative spread: MAD-based sigma over the median."""
    m = median(xs)
    if not xs or m <= 0:
        return 0.0
    mad = median([abs(x - m) for x in xs])
    return _MAD_SIGMA * mad / m


def bimodal(xs: Sequence[float], min_ratio: float = 1.25) -> Optional[Dict]:
    """Two clusters, or None. Apple-silicon timings under load sit in one of
    two modes (docs/TRANSPARENT_DESIGN.md §9.1, "two modes of a short kernel"):
    a median over both describes neither. The samples are split at their
    widest gap; the split counts when both sides hold at least two samples,
    the gap is wider than either cluster, and the upper cluster's median is at
    least `min_ratio` times the lower one's."""
    s = sorted(xs)
    n = len(s)
    if n < 4:
        return None
    best = None
    for i in range(2, n - 1):
        gap = s[i] - s[i - 1]
        if best is None or gap > best[0]:
            best = (gap, i)
    gap, i = best
    lo, hi = s[:i], s[i:]
    if len(hi) < 2 or len(lo) < 2:
        return None
    if gap <= max(lo[-1] - lo[0], hi[-1] - hi[0]):
        return None
    mlo, mhi = median(lo), median(hi)
    if mlo <= 0 or mhi / mlo < min_ratio:
        return None
    return {"fast_median": mlo, "slow_median": mhi, "fast_n": len(lo), "slow_n": len(hi)}


def central(xs: Sequence[float]) -> float:
    """The figure a verdict compares: the median, or the fast mode's median
    when the samples are bimodal (the slow mode is the machine's state, not
    the statement's cost; the row is flagged so nobody reads it as one)."""
    b = bimodal(xs)
    return b["fast_median"] if b else median(xs)


def mann_whitney_greater(a: Sequence[float], b: Sequence[float]) -> float:
    """One-sided p-value that `a` is stochastically GREATER (slower) than `b`.
    Exact by enumeration up to 10 per side would be costly; the normal
    approximation with tie and continuity corrections is adequate at n >= 5."""
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return 1.0
    pooled = sorted([(x, 0) for x in a] + [(x, 1) for x in b])
    ranks = [0.0] * len(pooled)
    i = 0
    ties = 0.0
    while i < len(pooled):
        j = i
        while j + 1 < len(pooled) and pooled[j + 1][0] == pooled[i][0]:
            j += 1
        r = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[k] = r
        t = j - i + 1
        ties += t ** 3 - t
        i = j + 1
    r1 = sum(r for r, (_, g) in zip(ranks, pooled) if g == 0)
    u1 = r1 - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    n = n1 + n2
    var = n1 * n2 / 12.0 * ((n + 1) - ties / (n * (n - 1))) if n > 1 else 0.0
    if var <= 0:
        return 1.0
    z = (u1 - mu - 0.5) / math.sqrt(var)
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def williams(k: int) -> List[List[int]]:
    """A Williams design: orders of k sides in which every side takes every
    position equally often AND follows every other side equally often (k rows
    for even k, 2k for odd k). Plain rotation keeps the cyclic order, so the
    side after `native` would always be the one that finds the device idle
    after a CPU-only statement — measured in the first A/A run: that side
    alone 12-45% slower than its two identical twins on rewritten TPC-H SF1
    statements of 2-6 ms (Q21 5.43 ms against 3.79 and 3.69)."""
    if k <= 1:
        return [list(range(k))]
    first, lo, hi = [0], 1, k - 1
    while len(first) < k:
        first.append(lo)
        lo += 1
        if len(first) < k:
            first.append(hi)
            hi -= 1
    rows = [[(x + r) % k for x in first] for r in range(k)]
    if k % 2:
        rows += [list(reversed(r)) for r in rows]
    return rows


def bucket(ms: float) -> str:
    if ms < 2:
        return "<2 ms"
    if ms < 20:
        return "2-20 ms"
    if ms < 200:
        return "20-200 ms"
    return ">=200 ms"


BUCKETS = ("<2 ms", "2-20 ms", "20-200 ms", ">=200 ms")


def noise_floor(aa_pairs: List[Dict]) -> Dict[str, Dict]:
    """From the A/A pass: per duration bucket, how far two processes of the
    SAME wrapper land apart. `aa_pairs` holds {"base": ms, "aa": ms} per
    statement. The floor is the 90th percentile of |log(aa/base)| in the
    bucket, or over every statement when a bucket has fewer than five."""
    allv = [abs(math.log(p["aa"] / p["base"])) for p in aa_pairs if p["base"] > 0 and p["aa"] > 0]
    out = {}
    for b in BUCKETS:
        vs = [abs(math.log(p["aa"] / p["base"])) for p in aa_pairs
              if p["base"] > 0 and p["aa"] > 0 and bucket(p["base"]) == b]
        src = vs if len(vs) >= 5 else allv
        out[b] = {"n": len(vs), "pooled": len(vs) < 5,
                  "p90": quantile(src, 0.9) if src else 0.05,
                  "median": median(src) if src else 0.0}
    out["all"] = {"n": len(allv), "pooled": False,
                  "p90": quantile(allv, 0.9) if allv else 0.05,
                  "median": median(allv) if allv else 0.0}
    return out


def tolerance(base: Sequence[float], cand: Sequence[float], aa: Sequence[float],
              floor: Dict[str, Dict]) -> Dict:
    """The tolerance one statement's verdict is judged against, as a log ratio
    (0.10 ~ 10.5%), and where it came from. The largest of:
      - the bucket's A/A noise floor (how far two identical processes land);
      - this statement's own A/A distance;
      - three standard errors of the difference of two medians, from the
        samples' own robust spread.
    Plus ABS_FLOOR_MS in absolute terms (applied by the caller)."""
    cb, cc, ca = central(base), central(cand), central(aa) if aa else None
    n = max(1, min(len(base), len(cand)))
    se = _MEDIAN_SE * math.sqrt(rel_sigma(base) ** 2 + rel_sigma(cand) ** 2) / math.sqrt(n)
    parts = {"floor": floor.get(bucket(cb), floor.get("all", {})).get("p90", 0.05),
             "aa": abs(math.log(ca / cb)) if ca and cb > 0 and ca > 0 else 0.0,
             "spread": 3.0 * se}
    which = max(parts, key=parts.get)
    return {"log": parts[which], "from": which, "parts": parts}
