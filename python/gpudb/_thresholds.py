"""Per-backend thresholds for the transparent path (§9.1): which shapes the
rewrite may fire on, given the key's distinct-count estimate and, with a
WHERE, the measured selectivity. Rule 1 (never slower than native) is
enforced by scripts/transparent_gate.py over a sweep; these numbers are read
off that table and keep the rewrite away from the shapes that lost or came
out even there. The losing side stays printed in the gate output.

Metal (Apple M4 Max), TPC-H SF1 lineitem, 2026-09-17, min of 5, statement
vs statement through the wrapper:
  * 7 groups: 0.23–0.67× on every form               → min_groups
  * plain (every group returned): 1.04–1.48× at 10K groups, 1.11–1.12× at
    200K without a WHERE, 0.94–1.07× (noise around even) at 200K under any
    WHERE, 0.96–0.99× at 1.5M groups (output-bound)  → plain_max_groups,
    plain_max_groups_where, plain_min_selectivity
  * HAVING on the device: 1.17–6.2× from 10K groups, except 0.75× at 1.5M
    groups under a 9% WHERE and 0.99× at 10K groups under a 25% WHERE
    (2.1–2.6× at ≥ 200K groups there)                → having_min_selectivity,
    having_min_selectivity_big
  * top-k: 4.1× at 200K groups, 1.42× at 1.5M, 0.55–0.98× at 10K groups,
    0.71–0.72× under a 9% WHERE and 0.98–1.03× (noise) at 1.5M groups under
    a 25% WHERE                                      → topk_min_groups,
    topk_min_selectivity
CUDA: the same table until scripts/transparent_gate.py has run on the
Linux box (the CUDA exact kernels do not exist yet, so the wrapper never
takes the exact path there today).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class Thresholds:
    min_groups: int                 # below this every form runs native
    plain_max_groups: int           # plain form: every group comes back through DuckDB
    plain_max_groups_where: int     # plain form under a WHERE (the mask + the output both cost)
    plain_min_selectivity: float    # plain form under a WHERE
    having_min_selectivity: float   # device HAVING under a WHERE
    having_min_selectivity_big: float   # ... when the key has >= topk_min_groups distinct values
    topk_min_groups: int
    topk_min_selectivity: float


METAL = Thresholds(min_groups=1_000, plain_max_groups=300_000, plain_max_groups_where=50_000,
                   plain_min_selectivity=0.5,
                   having_min_selectivity=0.3, having_min_selectivity_big=0.2,
                   topk_min_groups=100_000, topk_min_selectivity=0.3)
CUDA = METAL
TABLE = {"METAL": METAL, "CUDA": CUDA}


def decide(backend: str, form: str, est_groups: Optional[int], selectivity: Optional[float],
           has_where: bool) -> Tuple[bool, str]:
    """(ok, detail). form: plain | having | topk. est_groups None = unknown
    (declines: a miss never rewrites). selectivity None = no WHERE."""
    t = TABLE.get((backend or "").upper())
    if t is None:
        return False, f"no thresholds for backend {backend!r}"
    if est_groups is None:
        return False, "no distinct-count estimate for the key"
    if est_groups < t.min_groups:
        return False, f"{est_groups} groups < {t.min_groups}"
    sel = selectivity if has_where else 1.0
    if sel is None:
        return False, "selectivity unknown"
    if form == "plain":
        if est_groups > t.plain_max_groups:
            return False, f"{est_groups} groups returned > {t.plain_max_groups} (output-bound)"
        if has_where and est_groups > t.plain_max_groups_where:
            return False, f"{est_groups} groups returned under a WHERE > {t.plain_max_groups_where}"
        if has_where and sel < t.plain_min_selectivity:
            return False, f"selectivity {sel:.2f} < {t.plain_min_selectivity} for the plain form"
        return True, ""
    if form == "having":
        bound = t.having_min_selectivity_big if est_groups >= t.topk_min_groups else t.having_min_selectivity
        if has_where and sel < bound:
            return False, f"selectivity {sel:.2f} < {bound} for HAVING"
        return True, ""
    if form == "topk":
        if est_groups < t.topk_min_groups:
            return False, f"{est_groups} groups < {t.topk_min_groups} for top-k"
        if has_where and sel < t.topk_min_selectivity:
            return False, f"selectivity {sel:.2f} < {t.topk_min_selectivity} for top-k"
        return True, ""
    return False, f"unknown form {form}"
