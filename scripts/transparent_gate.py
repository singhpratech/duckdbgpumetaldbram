#!/usr/bin/env python3
"""transparent_gate.py — rule 1 for the transparent path (docs/TRANSPARENT_DESIGN.md
§9.1 / §9.3): every rewritable shape, transparent vs native, same process,
warm, min of N, through the Python wrapper (the path users take).

For every (key column, WHERE predicate, form) cell over TPC-H lineitem the
script runs the statement natively (transparent=False) and transparently,
checks the rows are identical (sorted), records min-of-N statement times and
the ratio native/transparent, and prints the table in BENCHMARK.md form.
Rows the wrapper rewrote and that came out below RATIO_MIN (default 1.0)
FAIL; rows the wrapper declined are printed with the reason and never fail.
The losing side of every sweep stays printed.

A cell is measured through the wrapper entry point `--path` names, and both
sides of the ratio go through the same one. `execute()` runs the statement
inside the call; `sql()` hands back a lazy relation and therefore runs the
statement's own guards on side cursors first (docs/TRANSPARENT_DESIGN.md §3.3
and §4.24), which is real cost for the user and is what the `gpudb` shell
pays. Measuring only `execute()` says nothing about the other path: it is how
a 2x top-k could turn into 0.9x through `sql()` and the gate still pass
(docs/RESEARCH_NOTES.md, 2026-09-20). The default `auto` therefore measures
every form through `execute()` and the top-k forms — the ones that carry a
device-pass guard — through both, which costs about one extra minute of the
gate's ~12.

Usage:
  python3 scripts/transparent_gate.py [--db data/tpch_sf1/tpch.duckdb] [--n 5]
                                      [--min-ratio 1.0] [--keys l_orderkey,l_partkey,...]
                                      [--path auto|execute|sql|both]
Needs a built extension (build-macos / build-linux) or GPUDB_EXTENSION_PATH.
"""
from __future__ import annotations
import argparse
import re
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import gpudb  # noqa: E402

# key column -> (label, approximate groups at SF1)
KEYS = {
    "l_linenumber": "7 groups",
    "l_returnflag": "3 groups, VARCHAR key",      # few groups are only rewritten for string keys (_thresholds.py)
    "l_suppkey":    "10K groups",
    "l_partkey":    "200K groups",
    "l_orderkey":   "1.5M groups",
}
# WHERE predicate -> label; selectivity is measured, not assumed
WHERES = {
    "": "no WHERE",
    "l_discount < 0.01": "~9%",
    "l_linenumber = 1": "~25%",
    "l_linenumber <= 3": "~55%",
    "l_discount <= 0.09": "~90%",
    "l_discount BETWEEN 0.02 AND 0.08 AND l_linenumber <> 4": "mixed",
}
# projected: expressions over aggregates (§4.11); nested: a rewritable GROUP BY inside a statement DuckDB keeps (§4.14)
# distinct: count(DISTINCT x) beside a sum (§4.17: the device groups by (key, x))
FORMS = ("plain", "having", "topk", "projected", "nested", "distinct")
# --ctes: the same aggregate read through a WITH (§4.22). `cte` names the whole
# FROM, `cte_arm` is one arm of a join — the shape a project-and-join CTE folds
# into. Both are measured against the same statement written without the CTE.
CTE_FORMS = ("cte", "cte_arm")
# --inner: the SAME GROUP BY, but its result is consumed inside DuckDB instead
# of by the client (§4.23). The plain and join forms' output bounds were swept
# on statements whose groups the client materialises; these four say what the
# same groups cost when DuckDB reads them out of the table function. Three
# consumers REDUCE the result (one row, or far fewer rows, leave), one does not.
#   inner_agg    an outer aggregate over the derived table
#   inner_group  an outer GROUP BY over the inner aggregate (TPC-H Q13's shape)
#   inner_scalar a CTE compared against a scalar subquery over itself (Q15's shape)
#   inner_join   a join back to the key's own table — every inner group still
#                reaches the client, so this is the consumer that does NOT reduce
INNER_FORMS = ("inner_agg", "inner_group", "inner_scalar", "inner_join")
# key column -> (table, its key column, a payload column) for the join-back consumer
BACK_JOIN = {
    "l_orderkey": ("orders", "o_orderkey", "o_orderstatus"),
    "l_partkey":  ("part", "p_partkey", "p_brand"),
    "l_suppkey":  ("supplier", "s_suppkey", "s_nationkey"),
    "o_custkey":  ("customer", "c_custkey", "c_nationkey"),
    "c_custkey":  ("customer", "c_custkey", "c_nationkey"),
}
# key joins (§4.8): label -> (FROM clause, key column); the WHERE list below
# applies where its table is part of the join
JOINS = {
    # `l_orderkey` is the high-cardinality key over a join (1.5M groups at SF1): the shape
    # the join output bounds and the inner-statement bounds (§4.23) both decide
    "li x orders":            ("lineitem JOIN orders ON l_orderkey = o_orderkey",
                               ["o_custkey", "o_orderdate", "l_suppkey", "l_orderkey"]),
    "li x orders x customer": ("lineitem JOIN orders ON l_orderkey = o_orderkey JOIN customer ON o_custkey = c_custkey",
                               ["c_nationkey", "c_custkey"]),
    "li x part":              ("lineitem JOIN part ON l_partkey = p_partkey", ["p_brand", "p_size"]),
    # §4.13: joins answered from an upload of the join's result (LEFT JOIN with an ON predicate; a composite key)
    "li left orders":         ("lineitem LEFT JOIN orders ON l_orderkey = o_orderkey AND o_orderpriority = '1-URGENT'",
                               ["o_orderdate", "l_suppkey"]),
    "li x partsupp":          ("lineitem JOIN partsupp ON l_partkey = ps_partkey AND l_suppkey = ps_suppkey",
                               ["ps_availqty", "l_suppkey"]),
}
JOIN_WHERES = {          # predicate -> the table it needs in the join ('' = any)
    "": "",
    "o_orderdate < DATE '1995-03-15'": "orders",
    "o_orderdate >= DATE '1998-06-01'": "orders",
    "l_discount < 0.01": "lineitem",
    "p_size <= 10 AND l_linenumber <= 3": "part",
}


EXTRA_PAYLOADS = ["sum(l_extendedprice)", "max(l_tax)", "sum(l_discount)", "min(l_partkey)"]

# --lane-floor (§4.23): what the row floor should count. Each cell reads a
# SMALL table and carries a subquery lane (§4.18) over a much larger one, so
# native's work is the lane and the statement's own FROM says nothing about it.
# (table, its rows at SF1, key column, payload column, its own key, the table
#  the lane reads, that table's join column, and a filter for the IN form)
LANE_SOURCES = [
    ("supplier", "s_nationkey",      "s_acctbal",      "s_suppkey", "lineitem", "l_suppkey",
     "l_discount > 0.09 AND l_quantity > 49 AND l_shipdate > DATE '1998-10-01'"),
    ("customer", "c_nationkey",      "c_acctbal",      "c_custkey", "orders",   "o_custkey",  "o_totalprice > 100000"),
    ("customer", "c_mktsegment",     "c_acctbal",      "c_custkey", "orders",   "o_custkey",  "o_totalprice > 100000"),
    ("part",     "p_brand",          "p_retailprice",  "p_partkey", "lineitem", "l_partkey",  "l_quantity > 45"),
    ("part",     "p_size",           "p_retailprice",  "p_partkey", "lineitem", "l_partkey",  "l_quantity > 45"),
    ("partsupp", "ps_availqty",      "ps_supplycost",  "ps_partkey", "lineitem", "l_partkey", "l_quantity > 45"),
    ("orders",   "o_orderpriority",  "o_totalprice",   "o_orderkey", "lineitem", "l_orderkey", "l_discount > 0.09"),
    ("orders",   "o_custkey",        "o_totalprice",   "o_orderkey", "lineitem", "l_orderkey", "l_discount > 0.09"),
]
LANE_KINDS = ("not exists", "in", "scalar")


PAYLOAD = "l_quantity"            # --exprs swaps it for REVENUE (a computed lane, §4.10)
SOURCE_PAYLOAD = {}               # --lane-floor: a FROM table that is not lineitem has its own
REVENUE = "l_extendedprice * (1 - l_discount)"


def decline_note(lr) -> str:
    """The reason a cell was not rewritten and, for the reasons that carry
    one, the sentence behind it. `error`, `memory` and `not_resident` are the
    three a reader can do nothing with on their own: 'error' does not say what
    raised, and 'memory' does not say what did not fit against what."""
    reason = lr.get("reason") or "?"
    if reason not in ("error", "memory", "not_resident"):
        return reason
    first = (lr.get("detail") or "").splitlines()
    return f"{reason}: {first[0][:160]}" if first and first[0] else reason


def build(key: str, where: str, form: str, having_thr: str, source: str = "lineitem", payloads: int = 1) -> str:
    """`payloads` > 1 adds aggregates over further columns (§4.9: one device
    pass per payload column)."""
    w = f" WHERE {where}" if where else ""
    pay = SOURCE_PAYLOAD.get(source, PAYLOAD)
    more = "".join(", " + e for e in EXTRA_PAYLOADS[:payloads - 1])
    if form == "plain":
        return f"SELECT {key}, sum({pay}){more}, count(*) FROM {source}{w} GROUP BY {key}"
    if form == "distinct":
        return (f"SELECT {key}, count(DISTINCT l_shipmode) AS modes, sum({pay}) AS q{more} "
                f"FROM {source}{w} GROUP BY {key}")
    if form == "nested":
        return (f"SELECT count(*) AS groups, max(q) AS top, min(q) AS low FROM (SELECT {key} AS kk, sum({pay}) AS q{more} "
                f"FROM {source}{w} GROUP BY {key} HAVING sum({pay}) > {having_thr}) gpudb_x")
    if form == "inner_agg":       # §4.23: the groups are reduced again, inside DuckDB
        return (f"SELECT count(*) AS groups, max(q) AS top, min(q) AS low "
                f"FROM (SELECT {key} AS kk, sum({pay}) AS q{more} FROM {source}{w} GROUP BY {key}) gpudb_i")
    if form == "inner_group":     # §4.23: an outer GROUP BY over the inner aggregate (Q13's shape)
        return (f"SELECT c, count(*) AS n FROM (SELECT {key} AS kk, count(*) AS c{more} "
                f"FROM {source}{w} GROUP BY {key}) gpudb_i GROUP BY c ORDER BY n DESC, c DESC")
    if form == "inner_scalar":    # §4.23: a CTE against a scalar subquery over itself (Q15's shape)
        return (f"WITH gpudb_i AS (SELECT {key} AS kk, sum({pay}) AS q{more} FROM {source}{w} GROUP BY {key}) "
                f"SELECT kk, q FROM gpudb_i WHERE q = (SELECT max(q) FROM gpudb_i)")
    if form == "inner_join":      # §4.23: the consumer that does NOT reduce — every group reaches the client
        tbl, pk, col = BACK_JOIN[key]
        return (f"SELECT gpudb_i.kk, gpudb_i.q, {tbl}.{col} "
                f"FROM (SELECT {key} AS kk, sum({pay}) AS q{more} FROM {source}{w} GROUP BY {key}) gpudb_i "
                f"JOIN {tbl} ON gpudb_i.kk = {tbl}.{pk}")
    if form == "cte":             # §4.22: a project-and-join CTE that is the whole FROM
        return (f"WITH gpudb_src AS (SELECT {key} AS gpudb_k, {pay} AS gpudb_v FROM {source}{w}) "
                f"SELECT gpudb_k, sum(gpudb_v), count(*) FROM gpudb_src GROUP BY gpudb_k")
    if form == "cte_arm":         # §4.22: the same CTE as one ARM of a join
        return (f"WITH gpudb_src AS (SELECT l_orderkey AS gpudb_ok, {key} AS gpudb_k, {pay} AS gpudb_v "
                f"FROM lineitem{w}) "
                f"SELECT gpudb_k, sum(gpudb_v), count(*) FROM gpudb_src, orders "
                f"WHERE gpudb_ok = o_orderkey AND o_orderstatus = 'F' GROUP BY gpudb_k")
    if form == "global":          # no GROUP BY (§4.12): the global masked aggregate
        return f"SELECT sum({pay}) AS q, count(*){more} FROM {source}{w}"
    if form == "projected":
        return (f"SELECT {key}, sum({pay}) / count(*) AS mean{more} FROM {source}{w} GROUP BY {key} "
                f"HAVING sum({pay}) > {having_thr} AND count(*) > 1")
    if form == "having":
        return (f"SELECT {key}, sum({pay}) AS q{more} FROM {source}{w} GROUP BY {key} "
                f"HAVING sum({pay}) > {having_thr}")
    return (f"SELECT {key}, sum({pay}) AS q{more} FROM {source}{w} GROUP BY {key} "
            f"ORDER BY q DESC LIMIT 10")


def lane_where(kind: str, own_key: str, big: str, big_key: str, filt: str, payload: str) -> str:
    """The §4.18 subquery predicate of a --lane-floor cell."""
    if kind == "not exists":
        return f"NOT EXISTS (SELECT 1 FROM {big} WHERE {big_key} = {own_key} AND {filt})"
    if kind == "in":
        return f"{own_key} IN (SELECT {big_key} FROM {big} WHERE {filt})"
    # a correlated scalar: the lane DuckDB evaluates once, during the upload
    col = {"lineitem": "l_extendedprice", "orders": "o_totalprice"}[big]
    return f"{payload} < (SELECT 0.5 * avg(gpudb_b.{col}) FROM {big} gpudb_b WHERE gpudb_b.{big_key} = {own_key})"


PACE_S = 0.0     # --pace-ms: idle gap before EVERY timed statement

# The two entry points a client has, and what a form is measured through by
# default. A pushed top-k is the shape whose `sql()` guard is a device pass of
# its own (§4.24), so it is the one `auto` measures on both paths.
BOTH_FORMS = ("topk",)


def run_path(con, sql: str, path: str):
    """One execution of `sql` through the entry point `path` names, rows
    fetched — the whole of what a client pays on that path."""
    if path == "sql":
        rel = con.sql(sql)
        return rel.fetchall() if rel is not None else []
    return con.execute(sql).fetchall()


def paths_for(form: str, arg: str) -> tuple:
    if arg == "auto":
        return ("execute", "sql") if form in BOTH_FORMS else ("execute",)
    return ("execute", "sql") if arg == "both" else (arg,)


def time_min(run, n: int) -> float:
    """Hot loop: the minimum of n back-to-back runs. Paced (--pace-ms): the
    MEDIAN of n runs, each after an idle gap — an interactive session's
    cadence, where clocks have dropped between statements (on Apple silicon
    the first statement after a 0.2 s pause runs 2-3x slower on both paths
    and the device needs more back-to-back runs than the CPU to recover)."""
    times = []
    for _ in range(n):
        if PACE_S > 0:
            time.sleep(PACE_S)
        t0 = time.perf_counter()
        run()
        times.append((time.perf_counter() - t0) * 1000.0)
    if PACE_S > 0:
        times.sort()
        return times[len(times) // 2]
    return min(times)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/tpch_sf1/tpch.duckdb")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--min-ratio", type=float, default=1.0)
    ap.add_argument("--keys", default=",".join(KEYS))
    ap.add_argument("--wheres", default="all", help="'all' or a ';'-separated list of predicates ('' = none)")
    ap.add_argument("--joins", default="all", help="'all', 'none' or a ';'-separated list of JOINS labels")
    ap.add_argument("--no-single", action="store_true", help="skip the single-table sweep")
    ap.add_argument("--pace-ms", type=float, default=0.0,
                    help="idle gap before every timed statement; reports medians (an interactive cadence)")
    ap.add_argument("--exprs", action="store_true",
                    help="aggregate l_extendedprice * (1 - l_discount) and add an expression WHERE (computed lanes)")
    ap.add_argument("--ctes", action="store_true",
                    help="add the WITH forms: a project-and-join CTE as the FROM and as a join arm (§4.22)")
    ap.add_argument("--inner", action="store_true",
                    help="add the inner-statement forms: the same GROUP BY consumed inside DuckDB (§4.23)")
    ap.add_argument("--lane-floor", action="store_true",
                    help="sweep small tables carrying a subquery lane over a large one — what the row floor "
                         "should count (§4.23); replaces the lineitem sweep unless --no-single is absent")
    ap.add_argument("--subqueries", action="store_true",
                    help="add EXISTS / IN / correlated scalar subquery predicates (BOOLEAN lanes, §4.18)")
    ap.add_argument("--forms", default="all",
                    help="'all' or a ','-separated list of form names to restrict the sweep to")
    ap.add_argument("--path", default="auto", choices=("auto", "execute", "sql", "both"),
                    help="the wrapper entry point both sides of a cell are measured through: "
                         "execute(), sql() (the lazy relation the gpudb shell uses), both, or "
                         "'auto' (default) — both for the top-k forms, execute() elsewhere")
    ap.add_argument("--payloads", type=int, default=1, help="aggregate this many payload columns per statement (1-5)")
    ap.add_argument("--memory-budget", default=None,
                    help="device memory budget for the run (§5.5). The default is the wrapper's own — what "
                         "ships, and therefore what the gate has to measure. Pass 'unlimited' to lift the cap "
                         "and see every shape the engine accepts, knowing no user runs that way; that WAS the "
                         "default here, and a gate run with the budget switched off is how a budget nobody was "
                         "enforcing went unnoticed (docs/RESEARCH_NOTES.md, 2026-09-20)")
    ap.add_argument("--no-thresholds", action="store_true",
                    help="rewrite every shape the engine accepts (data collection for the thresholds; "
                         "rows below the bound are reported, the exit code still fails on them)")
    args = ap.parse_args()
    global PACE_S
    PACE_S = args.pace_ms / 1000.0
    if args.exprs:
        global PAYLOAD
        PAYLOAD = REVENUE
        WHERES["l_commitdate < l_receiptdate AND (l_shipmode = 'AIR' OR l_quantity > 40)"] = "computed"
        WHERES["extract(year FROM l_shipdate) = 1995"] = "computed"
        JOIN_WHERES["o_orderpriority LIKE '1-%' OR o_orderpriority LIKE '2-%'"] = "orders"
    if args.subqueries:
        WHERES["EXISTS (SELECT 1 FROM orders WHERE o_orderkey = l_orderkey AND o_orderpriority = '1-URGENT')"] = "exists"
        WHERES["l_partkey NOT IN (SELECT p_partkey FROM part WHERE p_size <= 10)"] = "not in"
        WHERES["l_quantity < (SELECT 0.2 * avg(l2.l_quantity) FROM lineitem l2 WHERE l2.l_partkey = lineitem.l_partkey)"] = "scalar"
        JOIN_WHERES["EXISTS (SELECT 1 FROM lineitem l2 WHERE l2.l_orderkey = o_orderkey AND l2.l_commitdate < l2.l_receiptdate "
                    "AND l2.l_suppkey <> lineitem.l_suppkey)"] = "orders"
        JOIN_WHERES["l_suppkey IN (SELECT s_suppkey FROM supplier WHERE s_nationkey < 5)"] = "lineitem"
    if not os.path.exists(args.db):
        print(f"missing {args.db} — SF=1 ./scripts/gen_tpch.sh", file=sys.stderr)
        return 2

    con = gpudb.connect(args.db, read_only=True, residency="eager", floor_rows=0,
                        thresholds=not args.no_thresholds, memory_budget=args.memory_budget)
    info = con._raw.execute("SELECT gpu_build_info()").fetchone()[0]
    rows_total = con._raw.execute("SELECT count(*) FROM lineitem").fetchone()[0]
    print(f"# transparent_gate — {args.db} ({rows_total:,} rows), {info}, N={args.n}, min ratio {args.min_ratio}, "
          f"payload columns {args.payloads}, path {args.path}, "
          + (f"paced {args.pace_ms:.0f} ms (medians)" if PACE_S > 0 else "hot loop (minimums)"))
    print()
    print("| key | WHERE | selectivity | form | path | rows out | native ms | transparent ms | ratio | result |")
    print("|---|---|---|---|---|---|---|---|---|---|")

    keys = [k for k in args.keys.split(",") if k]
    wheres = list(WHERES) if args.wheres == "all" else args.wheres.split(";")
    fails = []
    cells = []          # (source label, FROM clause, key, where, forms or None)
    if not args.no_single:
        cells += [("", "lineitem", key, where, None) for where in wheres for key in keys]
    if args.joins != "none":
        labels = list(JOINS) if args.joins == "all" else args.joins.split(";")
        for label in labels:
            source, jkeys = JOINS[label]
            for where, table in JOIN_WHERES.items():
                if table and not re.search(rf"\b{table}\b", source):
                    continue
                cells += [(label, source, key, where, None) for key in jkeys]
    if args.lane_floor:
        for src, key, pay, own, big, bkey, filt in LANE_SOURCES:
            for kind in LANE_KINDS:
                SOURCE_PAYLOAD[src] = pay
                cells.append((f"{src} + a {kind} lane over {big}", src, key,
                              lane_where(kind, own, big, bkey, filt, pay), ("plain", "having")))
    sel_cache = {}
    for label, source, key, where, only_forms in cells:
        w = f" WHERE {where}" if where else ""
        if (source, where) not in sel_cache:
            kept, total = con._raw.execute(
                f"SELECT count(*) FILTER (WHERE {where or 'true'}), count(*) FROM {source}").fetchone()
            sel_cache[(source, where)] = f"{100.0 * kept / total:.0f}%" if total else "—"
        sel = sel_cache[(source, where)]
        key_label = f"{label}: {key}" if label else key
        pay = SOURCE_PAYLOAD.get(source, PAYLOAD)
        # a HAVING threshold that keeps roughly 1% of the groups
        thr = con._raw.execute(
            f"SELECT quantile_cont(q, 0.99) FROM (SELECT sum({pay}) q FROM {source}{w} GROUP BY {key})"
        ).fetchone()[0]
        thr_s = f"{thr:.2f}" if thr is not None else "0"
        if True:
            # the global form (§4.12) reads no key, so over a single table one
            # cell per (source, WHERE) is the whole sweep — it rides on the first key
            forms = FORMS + (("global",) if (label or key == keys[0]) else ())
            if args.ctes:
                # `cte_arm` joins lineitem to orders itself, so it is only run over
                # the single-table sweep (where the WHERE names lineitem alone)
                forms += CTE_FORMS if not label else ("cte",)
            if args.inner:
                # the join-back consumer needs a table the key is a key OF
                forms += tuple(f for f in INNER_FORMS if f != "inner_join" or key in BACK_JOIN)
            if only_forms is not None:
                forms = only_forms
            if args.forms != "all":
                want = args.forms.split(",")
                forms = tuple(f for f in forms if f in want)
            for form, path in [(f, p) for f in forms for p in paths_for(f, args.path)]:
                sql = build(key, where, form, thr_s, source, args.payloads)
                # native, through the same entry point as the transparent side
                con.transparent = False
                nat = run_path(con, sql, path)
                t_nat = time_min(lambda: run_path(con, sql, path), args.n)
                # transparent (first call may upload; timed calls are warm)
                con.transparent = True
                got = run_path(con, sql, path)
                lr = con.last_rewrite()
                if not lr["rewritten"]:
                    print(f"| {key_label} | {where or '—'} | {sel} | {form} | {path} | {len(nat)} | {t_nat:.1f} | — | — | "
                          f"declined ({lr['reason']}) |")
                    continue
                if form in ("global", "nested"):
                    identical = got == nat
                elif form == "topk":
                    # ORDER BY <agg> LIMIT k without a tiebreaker: which of the
                    # groups tied at the k-th value are returned is unspecified
                    # in SQL and differs between native runs too — compare the
                    # multiset of aggregate values (the kept ORDER BY makes the
                    # order native's problem, §2).
                    # (column 1 is the ORDER BY aggregate in every statement build() makes)
                    identical = sorted(str(r[1]) for r in got) == sorted(str(r[1]) for r in nat)
                else:
                    identical = sorted(map(str, got)) == sorted(map(str, nat))
                t_tr = time_min(lambda: run_path(con, sql, path), args.n)
                lr2 = con.last_rewrite()
                if not lr2["rewritten"]:
                    # the once-per-template output-size check sent the template
                    # back to native after its first rewritten run — or, on the
                    # sql() path, the measured rule declined it there (§9.1):
                    # the timed runs were native, there is no ratio to report
                    print(f"| {key_label} | {where or '—'} | {sel} | {form} | {path} | {len(nat)} | {t_nat:.1f} | — | — | "
                          f"{'declined after the first run' if identical else 'FAIL rows differ'} ({lr2['reason']}) |")
                    if not identical:
                        fails.append((key_label, where, form, 0.0, identical))
                    continue
                ratio = t_nat / t_tr if t_tr > 0 else float("inf")
                # A millisecond-scale statement can time a whole batch slow
                # (device clock state, a background task). Before a row fails
                # on speed it is re-measured, native and transparent
                # interleaved, and the minimum over every run is kept for both.
                # (Apple silicon: the same ms-scale statement runs in one of two
                # modes for seconds at a time, ~2.8 or ~4.2 ms here, depending on
                # what the SoC did just before — hence the pause between rounds.)
                # A hot loop is meant to be warm on BOTH sides, and the device needs
                # ~10 back-to-back runs to clock up where the CPU needs a few: each
                # re-measurement first warms the side it is about to time.
                for _ in range(3):
                    if not identical or ratio >= args.min_ratio or PACE_S > 0:
                        break
                    for transparent in (False, True):
                        con.transparent = transparent
                        for _w in range(20):
                            run_path(con, sql, path)
                        t = time_min(lambda: run_path(con, sql, path), args.n)
                        if transparent:
                            t_tr = min(t_tr, t)
                        else:
                            t_nat = min(t_nat, t)
                    con.transparent = True
                    ratio = t_nat / t_tr if t_tr > 0 else float("inf")
                ok = identical and ratio >= args.min_ratio
                result = "PASS" if ok else ("FAIL rows differ" if not identical else "FAIL")
                if not ok:
                    fails.append((f"{key_label} [{path}]", where, form, ratio, identical))
                print(f"| {key_label} | {where or '—'} | {sel} | {form} | {path} | {len(got)} | {t_nat:.1f} | {t_tr:.1f} | "
                      f"{ratio:.2f}× | {result} |")
                sys.stdout.flush()
    print()
    if fails:
        print(f"{len(fails)} row(s) below {args.min_ratio}× or not identical:")
        for key, where, form, ratio, identical in fails:
            print(f"  - {key} / {where or 'no WHERE'} / {form}: {ratio:.2f}×" + ("" if identical else " (rows differ)"))
        return 1
    print("all rewritten rows at or above the bound and identical to native")
    return 0


if __name__ == "__main__":
    sys.exit(main())
