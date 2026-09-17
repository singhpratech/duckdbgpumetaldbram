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
    0.71–0.72× under a 9% WHERE, 0.98–1.06× (noise) at 1.5M groups under a
    25–64% WHERE, 1.22× at 91%                       → topk_min_groups,
    topk_min_selectivity
Key joins (§4.8), same machine and data, lineitem x orders [x customer],
lineitem x part, 2026-09-17: native has to run the join whatever the group
count, so there is no lower bound on groups (25 groups: 2.3–4.4×). HAVING
1.4–7.4× and top-k 1.6–4.0× on every row (top-k over <= 65K groups takes the
operator's host pass). The plain form 1.25–1.28× at 100K groups without a
WHERE, 1.09–1.13× under a 9–49% WHERE, 0.92–0.99× at 32K groups under a 3%
WHERE (a selective dimension filter makes native's join tiny while the
resident operator still masks every fact row; what is left is output on both
sides). SF10 (60M x 15M rows): every one of 78 rows wins, the plain form
1.08–1.28× at 1M groups returned, 1.12–1.14× at 320K under the 3% WHERE,
HAVING / top-k 2.0–9.9×                               → join_plain_max_groups,
join_plain_small_groups, join_plain_min_selectivity
Several payload columns (§4.9; three of them, same machine, SF1): the fused
operator shares the mask and the grouping, so HAVING / top-k keep their wins
(1.06–3.0× single table, up to 4.8× over joins), but every extra aggregate is
another output column on both sides and the plain form's margin shrinks to
1.01–1.09× (one 0.93× at 10K groups under a three-term WHERE), 1.00–1.08× at
100K groups over a join                               → multi_plain_max_groups,
multi_join_plain_max_groups; the plain form under a WHERE runs native
CUDA: the same table until scripts/transparent_gate.py has run on the
Linux box (the CUDA exact kernels do not exist yet, so the wrapper never
takes the exact path there today).

The bounds predict the win before a statement runs. After the FIRST rewritten
run of a template the wrapper also reads the operator's rows_out
(gpu_last_stats) and declines the template when a HAVING or top-k kept more
groups than the plain form's bound: a filter that passes most groups is
output-bound like the plain form (measured 0.97x on TPC-H orders by
o_custkey HAVING count(*) >= 3, 59K of ~100K groups surviving).
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
    # statements over a key join (§4.8)
    join_plain_max_groups: int = 1_000_000      # measured up to 1M groups returned (SF10), 1.08x there
    join_plain_small_groups: int = 20_000       # at or below this the plain form wins at any selectivity
    join_plain_min_selectivity: float = 0.08    # above it, the plain form under a WHERE needs this much kept
    # statements aggregating several payload columns (§4.9)
    multi_plain_max_groups: int = 50_000        # plain form, single table, no WHERE (under a WHERE: native)
    multi_join_plain_max_groups: int = 20_000   # plain form over a key join


METAL = Thresholds(min_groups=1_000, plain_max_groups=300_000, plain_max_groups_where=50_000,
                   plain_min_selectivity=0.5,
                   having_min_selectivity=0.3, having_min_selectivity_big=0.2,
                   topk_min_groups=100_000, topk_min_selectivity=0.8)
CUDA = METAL
TABLE = {"METAL": METAL, "CUDA": CUDA}


def decide(backend: str, form: str, est_groups: Optional[int], selectivity: Optional[float],
           has_where: bool, join: bool = False, payloads: int = 1) -> Tuple[bool, str]:
    """(ok, detail). form: plain | having | topk. est_groups None = unknown
    (declines: a miss never rewrites). selectivity None = no WHERE. join:
    the statement is over a key join (its own table above)."""
    t = TABLE.get((backend or "").upper())
    if t is None:
        return False, f"no thresholds for backend {backend!r}"
    if est_groups is None:
        return False, "no distinct-count estimate for the key"
    if payloads > 1 and form == "plain" and est_groups is not None:
        if join and est_groups > t.multi_join_plain_max_groups:
            return False, (f"{est_groups} groups x {payloads} payload columns returned over a join "
                           f"> {t.multi_join_plain_max_groups} (output-bound)")
        if not join and has_where:
            return False, f"plain form with {payloads} payload columns under a WHERE"
        if not join and est_groups > t.multi_plain_max_groups:
            return False, f"{est_groups} groups x {payloads} payload columns returned > {t.multi_plain_max_groups}"
    if join:
        if form != "plain":
            return True, ""
        if est_groups > t.join_plain_max_groups:
            return False, f"{est_groups} groups returned over a join > {t.join_plain_max_groups} (output-bound)"
        if has_where and est_groups > t.join_plain_small_groups:
            if selectivity is None:
                return False, "selectivity unknown"
            if selectivity < t.join_plain_min_selectivity:
                return False, (f"selectivity {selectivity:.2f} < {t.join_plain_min_selectivity} for the plain form "
                               f"over a join with {est_groups} groups")
        return True, ""
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
