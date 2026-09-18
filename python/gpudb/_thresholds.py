"""Per-backend thresholds for the transparent path (§9.1): which shapes the
rewrite may fire on, given the key's distinct-count estimate and, with a
WHERE, the measured selectivity. Rule 1 (never slower than native) is
enforced by scripts/transparent_gate.py over a sweep; these numbers are read
off that table and keep the rewrite away from the shapes that lost or came
out even there. The losing side stays printed in the gate output.

Metal (Apple M4 Max), TPC-H SF1 lineitem, 2026-09-17, min of 5, statement
vs statement through the wrapper:
  * 7 groups: 0.23–0.67× on every form               → min_groups
    (after the block-level reduce, 2026-09-17: 0.5–1.6× at SF1, 0.77–2.2× at
    SF10 for 3–7 groups — native answers these in 1.5–3 ms per 6M rows, so
    the bound stays; over a join the few-group shapes win and have no bound)
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
Few groups, by key type (2026-09-17, SF1, 1 to 5 payload columns with an
expression payload, after the block-level reduce): an INTEGER key with 7
groups loses or ties on every form (0.62–1.28×; native aggregates a tiny
integer domain through a perfect hash in 1.5–5 ms), a VARCHAR key with 3
groups WINS 1.02–1.71× with no WHERE (native hashes the strings on every
row). Under a WHERE it depends on the payload: with an EXPRESSION payload
(native evaluates it per row, the resident lane is free) 1.12–1.48× at 98%
kept and 0.62–0.82× at 9%; with a plain column payload 0.78–1.19× at 55–91%
kept — the mask costs the device what the expression costs native. TPC-H Q1
(two VARCHAR keys, eight aggregates over expressions, 98% of the rows):
12.2 → 5.9 ms. ONE expression payload under a WHERE is a coin flip: the same
cell measured 1.07x median over 12 process starts with 3 to 5 of them below
1.0x (the device time is bimodal per process start, 3.5 vs 4.2 ms, native sits at
3.8), and 0.58–1.35x across 55–91% kept. From TWO expression payloads on,
native pays per expression and the device does not: 1.08–1.59x on all 27
cells (2–4 payloads x 55/64/91% kept x 3 process starts).
                                → string_key_min_selectivity,
                                  string_key_min_computed_payloads
count(DISTINCT x) (§4.17; SF1, x = l_shipmode with 7 values, beside a sum):
the device groups by (key, x) and DuckDB re-aggregates every pair, ~0.25 ms
per 1K pairs. Up to 17K pairs 1.3–7.1×; 70K pairs 2.1–2.5× without a WHERE
but 0.82–1.13× under a 9–10% one; 700K pairs 0.39–0.51×; 441 pairs under a
1% WHERE 0.84× (native is 5 ms there)                → reagg_max_pairs,
reagg_max_pairs_where, reagg_min_selectivity
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

The direct grouped reduce (v0.8, `docs/RESIDENT_COLUMNS_DESIGN.md` §7) did not
move any of these, and the measurement says why. Inside the backend, a key with
few distinct values now answers in one row-order pass over a group-id lane
instead of masking, sorting and gathering: at SF10 that is 1.65× to 8.57×
faster on every shape it serves (BENCHMARK.md), and TPC-H Q1 goes from 37.2 ms
to 15.2. But the bounds here are group counts, not row counts, and what they
have to hold at is SF1, where `lineitem` is 6M rows and a few-group statement is
not bound by the reduce at all. `scripts/transparent_gate.py --subqueries
--exprs --no-thresholds` on this build, SF1 — the first 56 rows of the sweep,
which are the ones these bounds decide (the run was stopped there; the machine
was needed for the gate proper):

  * `l_linenumber` (7 groups, INTEGER key): 0.80-0.94x on plain / HAVING /
    top-k / projected / nested with no WHERE, 0.52-0.55x under a 9% one.
    Native answers a 6M-row, 7-group aggregate in 1.5-2.5 ms and the wrapper's
    own round trip is about 2.5 ms      → min_groups stays at 1000
  * `l_returnflag` (3 groups, VARCHAR key, already exempt from min_groups with
    no WHERE): 1.01-1.39x with no WHERE, 0.57-0.62x under the 9% WHERE — which
    string_key_min_selectivity = 0.5 already declines
                                        → string_key_min_selectivity and
                                          string_key_min_computed_payloads stay
  * the many-group rows (`l_suppkey` 10K, `l_partkey` 200K, `l_orderkey` 1.5M)
    read the same sort path as before and measure where they measured

The backend has its own rule for the same reason at a smaller scale: the direct
pass is only taken at 3+ groups and 6M+ rows x (payloads + terms), because below
that its per-threadgroup setup outweighs what it saves (BENCHMARK.md). That is
backend-private and invisible here; it changes which algorithm answers, never
whether the statement is rewritten.

So nothing here is relaxed on this evidence: a bound that admitted the
few-group shapes at SF1 would admit them at 0.5x, and rule 1 is not a ratio
averaged over scale factors. What the direct path changes is the SF10 and SF50
end of every shape that is ALREADY admitted, which is where it shows in the
TPC-H tables.

The bounds predict the win before a statement runs; the wrapper also MEASURES
it: every statement runs native while its set is being uploaded, so its own
native time is known, and when the best of the first three rewritten runs is
not faster than the best native run the template is declined from then on
(connection._note_timing). Millisecond-scale statements on Metal are bimodal
run to run (one join cell measured 2.2 ms in most processes and 4.2 ms in
others, against 4.6 ms native), which no static bound captures. After the FIRST rewritten
run of a template the wrapper also reads the operator's rows_out
(gpu_last_stats) and declines the template when a HAVING or top-k kept more
groups than the plain form's bound: a filter that passes most groups is
output-bound like the plain form (measured 0.97x on TPC-H orders by
o_custkey HAVING count(*) >= 3, 59K of ~100K groups surviving).

Cutting the per-statement fixed cost (2026-09-18) did not move these either,
and what the sweep said is worth keeping. The rewritten statement is planned
once instead of per run (docs/TRANSPARENT_DESIGN.md §3.2), which takes
0.06-0.10 ms off a few-group statement and 0.37-0.43 ms off the 10K-group and
join forms; the whole fixed cost outside the operator at SF1 is now 0.12 ms of
a 0.69 ms statement (BENCHMARK.md). `scripts/transparent_gate.py
--no-thresholds --keys l_linenumber,l_returnflag --joins none`, SF1, N=9, run
on the state before the change and on the state after:

  * `l_linenumber` (7 groups, INTEGER key) measures 1.19-2.50x on plain /
    HAVING / top-k / projected / nested, on BOTH states, at every selectivity
  * the same sweep recorded 0.80-0.94x for those rows earlier the same day

Nothing between the two runs changed those statements by more than 0.1 ms, so
the difference is the process's mode (§9.1: a 0.45 ms kernel runs at three
times that when the rest of the process keeps waking), and the slow mode did
not reproduce on demand. One mode's worth of evidence does not relax a bound
that has to hold in both, so min_groups stays at 1000 and the VARCHAR-key
rules stay as they are; the continuous measured rule 1
(connection._note_timing) keeps deciding these shapes at run time. The only
rows below 1.0x in that sweep are the global-aggregate form at SF1, which its
own rule (rows x (1 + terms) >= 60M) already declines.
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
    # a VARCHAR key is exempt from min_groups / topk_min_groups without a WHERE, and under a WHERE
    # when at least this many payloads are computed expressions and at least this much survives
    string_key_min_selectivity: float = 0.5
    string_key_min_computed_payloads: int = 2
    # count(DISTINCT x): (key, x) pairs the device returns for DuckDB to re-aggregate
    reagg_max_pairs: int = 100_000
    reagg_max_pairs_where: int = 20_000
    reagg_min_selectivity: float = 0.05
    # aggregates without GROUP BY (§4.12): one row out, so there is no
    # output-size bound at all. What decides is how much work native does per
    # row — a vectorised filter and sum is memory-bandwidth-optimal on the CPU,
    # and the device only pulls ahead once there are enough rows AND enough
    # predicate terms to evaluate. Measured on lineitem slices of 3M to 60M
    # rows (M4 Max, min of 9, warm), the win appears at rows * (1 + terms,
    # counting at most three) around 60M and nowhere below it: 15M rows x 3
    # terms 1.18-1.49x, 20M x 3 1.36-1.57x, 30M x 1 1.51-1.75x, 60M x 0 1.42x,
    # 60M x 5 (TPC-H Q6) 2.48x — while 10M x 3 0.93x, 10M x 5 0.97x,
    # 20M x 1 0.99x and 30M x 0 0.99x all lose. The row floor sits above the
    # one cell that is inside the noise (15M x 3 measured 1.06-1.49x over
    # three runs), so every cell the bound admits measured at least 1.35x. Over a JOIN the comparison is against native's join, not
    # its scan, and the device wins from SF1 on (7-190x), so no floor there.
    global_min_rows: int = 16_000_000
    global_min_row_terms: int = 60_000_000
    global_join_min_rows: int = 0


METAL = Thresholds(min_groups=1_000, plain_max_groups=300_000, plain_max_groups_where=50_000,
                   plain_min_selectivity=0.5,
                   having_min_selectivity=0.3, having_min_selectivity_big=0.2,
                   topk_min_groups=100_000, topk_min_selectivity=0.8)
CUDA = METAL
TABLE = {"METAL": METAL, "CUDA": CUDA}


def decide(backend: str, form: str, est_groups: Optional[int], selectivity: Optional[float],
           has_where: bool, join: bool = False, payloads: int = 1, string_key: bool = False,
           limited: bool = False, computed_payloads: int = 0,
           reaggregated: bool = False, global_agg: bool = False, rows: int = 0,
           where_terms: int = 0) -> Tuple[bool, str]:
    """(ok, detail). form: plain | having | topk. est_groups None = unknown
    (declines: a miss never rewrites). selectivity None = no WHERE. join:
    the statement is over a key join (its own table above)."""
    t = TABLE.get((backend or "").upper())
    if t is None:
        return False, f"no thresholds for backend {backend!r}"
    if global_agg:
        # §4.12: one group, one row — no output-size risk and no distinct count
        if join:
            if rows < t.global_join_min_rows:
                return False, f"{rows} rows < {t.global_join_min_rows} for an aggregate without GROUP BY over a join"
            return True, ""
        if rows < t.global_min_rows:
            return False, f"{rows} rows < {t.global_min_rows} for an aggregate without GROUP BY"
        # past three terms native's per-row cost stops growing with the
        # conjunction (it short-circuits), so the bound stops counting there
        work = rows * (1 + min(3, max(0, where_terms)))
        if work < t.global_min_row_terms:
            return False, (f"{rows} rows x {where_terms} WHERE term(s) = {work} < {t.global_min_row_terms} "
                           f"for an aggregate without GROUP BY over a single table")
        return True, ""
    if est_groups is None:
        return False, "no distinct-count estimate for the key"
    if reaggregated:
        # est_groups counts (key, x) pairs: every one of them goes back through DuckDB
        if est_groups > t.reagg_max_pairs:
            return False, f"{est_groups} (key, value) pairs to re-aggregate > {t.reagg_max_pairs}"
        if has_where and est_groups > t.reagg_max_pairs_where:
            return False, f"{est_groups} (key, value) pairs under a WHERE > {t.reagg_max_pairs_where}"
        if has_where and (selectivity is None or selectivity < t.reagg_min_selectivity):
            return False, f"selectivity below {t.reagg_min_selectivity} for count(DISTINCT)"
    # (a result of a few hundred groups is never output-bound, however many columns it has)
    if payloads > 1 and form == "plain" and est_groups >= t.min_groups:
        if join and est_groups > t.multi_join_plain_max_groups:
            return False, (f"{est_groups} groups x {payloads} payload columns returned over a join "
                           f"> {t.multi_join_plain_max_groups} (output-bound)")
        if not join and has_where:
            return False, f"plain form with {payloads} payload columns under a WHERE"
        if not join and est_groups > t.multi_plain_max_groups:
            return False, f"{est_groups} groups x {payloads} payload columns returned > {t.multi_plain_max_groups}"
    if join:
        # `limited`: the statement ends in a LIMIT that was not pushed as a top-k (ORDER BY
        # several terms): every group still leaves the operator, but only LIMIT rows reach
        # the client, so the fetch-bound reasoning behind the plain-form bounds does not
        # apply (TPC-H Q3: 11K groups under a 1% WHERE, LIMIT 10 — 1.5-1.7x)
        if form != "plain" or limited:
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
    # (count(DISTINCT) beside one expression payload keeps its win there: 1.26–1.84x, 0 of 8 below 1.0x —
    # native builds a second hash table for the DISTINCT)
    few_ok = string_key and (not has_where or ((computed_payloads >= t.string_key_min_computed_payloads
                                                or (reaggregated and computed_payloads >= 1))
                                               and selectivity is not None
                                               and selectivity >= t.string_key_min_selectivity))
    # (plain form only: with three groups HAVING / top-k save nothing and measured 0.98–1.07x)
    if est_groups < t.min_groups and not (few_ok and form == "plain"):
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
