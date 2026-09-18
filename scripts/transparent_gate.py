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

`--probation` (§9.1) adds a second pass over the cells a SOFT threshold
declined: the script keeps issuing the statement (which keeps running on
DuckDB, as it would for any user) until the wrapper's own trial on a side
cursor has either promoted it or given up, then measures the cell again the
same way the first pass did. A cell that was promoted and comes out below
RATIO_MIN in that second measurement FAILS like any other rewritten row — that
is the whole point of the mode. Without the flag nothing about the output
changes.

Usage:
  python3 scripts/transparent_gate.py [--db data/tpch_sf1/tpch.duckdb] [--n 5]
                                      [--min-ratio 1.0] [--keys l_orderkey,l_partkey,...]
                                      [--probation [--probation-timeout 90]]
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
# key joins (§4.8): label -> (FROM clause, key column); the WHERE list below
# applies where its table is part of the join
JOINS = {
    "li x orders":            ("lineitem JOIN orders ON l_orderkey = o_orderkey", ["o_custkey", "o_orderdate", "l_suppkey"]),
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


PAYLOAD = "l_quantity"            # --exprs swaps it for REVENUE (a computed lane, §4.10)
REVENUE = "l_extendedprice * (1 - l_discount)"


def build(key: str, where: str, form: str, having_thr: str, source: str = "lineitem", payloads: int = 1) -> str:
    """`payloads` > 1 adds aggregates over further columns (§4.9: one device
    pass per payload column)."""
    w = f" WHERE {where}" if where else ""
    more = "".join(", " + e for e in EXTRA_PAYLOADS[:payloads - 1])
    if form == "plain":
        return f"SELECT {key}, sum({PAYLOAD}){more}, count(*) FROM {source}{w} GROUP BY {key}"
    if form == "distinct":
        return (f"SELECT {key}, count(DISTINCT l_shipmode) AS modes, sum({PAYLOAD}) AS q{more} "
                f"FROM {source}{w} GROUP BY {key}")
    if form == "nested":
        return (f"SELECT count(*) AS groups, max(q) AS top, min(q) AS low FROM (SELECT {key} AS kk, sum({PAYLOAD}) AS q{more} "
                f"FROM {source}{w} GROUP BY {key} HAVING sum({PAYLOAD}) > {having_thr}) gpudb_x")
    if form == "global":          # no GROUP BY (§4.12): the global masked aggregate
        return f"SELECT sum({PAYLOAD}) AS q, count(*){more} FROM {source}{w}"
    if form == "projected":
        return (f"SELECT {key}, sum({PAYLOAD}) / count(*) AS mean{more} FROM {source}{w} GROUP BY {key} "
                f"HAVING sum({PAYLOAD}) > {having_thr} AND count(*) > 1")
    if form == "having":
        return (f"SELECT {key}, sum({PAYLOAD}) AS q{more} FROM {source}{w} GROUP BY {key} "
                f"HAVING sum({PAYLOAD}) > {having_thr}")
    return (f"SELECT {key}, sum({PAYLOAD}) AS q{more} FROM {source}{w} GROUP BY {key} "
            f"ORDER BY q DESC LIMIT 10")


PACE_S = 0.0     # --pace-ms: idle gap before EVERY timed statement


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


def same_rows(got, nat, form):
    """The comparison the first pass makes for this form (§2: top-k without a
    tiebreaker is only defined up to which of the tied groups come back)."""
    if form in ("global", "nested"):
        return got == nat
    if form == "topk":
        return sorted(str(r[1]) for r in got) == sorted(str(r[1]) for r in nat)
    return sorted(map(str, got)) == sorted(map(str, nat))


def probation_pass(con, sql, nat, t_nat, n, min_ratio, timeout, form):
    """One cell the thresholds declined softly (§9.1). Keep issuing the
    statement — it runs on DuckDB throughout, exactly as it would for a user —
    until the wrapper's trial promotes it or stops trying, then measure the cell
    again the way the first pass did. Returns (verdict, ratio, native ms,
    transparent ms, identical, rounds)."""
    con.transparent = True
    con._manager.wait_idle(timeout)          # the set uploads in idle windows first
    deadline = time.time() + timeout
    rounds = 0
    while time.time() < deadline:
        con.execute(sql).fetchall()
        if con.last_rewrite()["rewritten"]:
            break
        d = getattr(con, "_timing_decision", None)      # this statement's own decision
        rounds = getattr(d, "probe_rounds", 0)
        if d is None or getattr(d, "probation", "") in ("", "retired"):
            break
        # a round that lost puts the next one behind a back-off; if that lands after the
        # deadline there is nothing left to wait for
        if rounds and getattr(d, "next_probe_at", 0.0) - time.monotonic() > deadline - time.time():
            break
        time.sleep(0.02)
        if getattr(d, "probation", "") in ("candidate", "waiting"):
            # the trial's set uploads only while the connection is genuinely quiet
            # (§9.1: 250 ms, against 20 ms for a set a statement is waiting on), which a
            # loop issuing a statement every 20 ms never gives it. Hand it the quiet.
            con._manager.wait_idle(min(10.0, max(0.0, deadline - time.time())))
    got = con.execute(sql).fetchall()
    if not con.last_rewrite()["rewritten"]:
        return "stayed native", None, t_nat, None, True, rounds
    identical = same_rows(got, nat, form)
    # the same instrument the first pass used: native and transparent interleaved,
    # each side warmed before it is timed, minimum over every run kept for both
    t_tr = time_min(lambda: con.execute(sql).fetchall(), n)
    for _ in range(3):
        if not identical or (t_nat / t_tr if t_tr > 0 else 0) >= min_ratio:
            break
        for transparent in (False, True):
            con.transparent = transparent
            for _w in range(20):
                con.execute(sql).fetchall()
            t = time_min(lambda: con.execute(sql).fetchall(), n)
            if transparent:
                t_tr = min(t_tr, t)
            else:
                t_nat = min(t_nat, t)
        con.transparent = True
    return "promoted", (t_nat / t_tr if t_tr > 0 else float("inf")), t_nat, t_tr, identical, rounds


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
    ap.add_argument("--subqueries", action="store_true",
                    help="add EXISTS / IN / correlated scalar subquery predicates (BOOLEAN lanes, §4.18)")
    ap.add_argument("--payloads", type=int, default=1, help="aggregate this many payload columns per statement (1-5)")
    ap.add_argument("--memory-budget", default="unlimited",
                    help="device memory budget for the run (§5.5); the gate sweeps more distinct sets than a "
                         "session ever holds, so the default lifts the cap — pass e.g. 16GB to test the budget")
    ap.add_argument("--no-thresholds", action="store_true",
                    help="rewrite every shape the engine accepts (data collection for the thresholds; "
                         "rows below the bound are reported, the exit code still fails on them)")
    ap.add_argument("--probation", action="store_true",
                    help="give the wrapper's side-cursor trial (§9.1) time to act on every SOFT-declined "
                         "cell, then measure that cell again; a promoted cell below the bound fails")
    ap.add_argument("--probation-timeout", type=float, default=90.0,
                    help="seconds a cell is given before its trial is called off (default 90)")
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
          f"payload columns {args.payloads}, "
          + (f"paced {args.pace_ms:.0f} ms (medians)" if PACE_S > 0 else "hot loop (minimums)"))
    print()
    print("| key | WHERE | selectivity | form | rows out | native ms | transparent ms | ratio | result |")
    print("|---|---|---|---|---|---|---|---|---|")

    keys = [k for k in args.keys.split(",") if k]
    wheres = list(WHERES) if args.wheres == "all" else args.wheres.split(";")
    fails = []
    probation_rows = []
    cells = []          # (source label, FROM clause, key, where)
    if not args.no_single:
        cells += [("", "lineitem", key, where) for where in wheres for key in keys]
    if args.joins != "none":
        labels = list(JOINS) if args.joins == "all" else args.joins.split(";")
        for label in labels:
            source, jkeys = JOINS[label]
            for where, table in JOIN_WHERES.items():
                if table and not re.search(rf"\b{table}\b", source):
                    continue
                cells += [(label, source, key, where) for key in jkeys]
    sel_cache = {}
    for label, source, key, where in cells:
        w = f" WHERE {where}" if where else ""
        if (source, where) not in sel_cache:
            kept, total = con._raw.execute(
                f"SELECT count(*) FILTER (WHERE {where or 'true'}), count(*) FROM {source}").fetchone()
            sel_cache[(source, where)] = f"{100.0 * kept / total:.0f}%" if total else "—"
        sel = sel_cache[(source, where)]
        key_label = f"{label}: {key}" if label else key
        # a HAVING threshold that keeps roughly 1% of the groups
        thr = con._raw.execute(
            f"SELECT quantile_cont(q, 0.99) FROM (SELECT sum({PAYLOAD}) q FROM {source}{w} GROUP BY {key})"
        ).fetchone()[0]
        thr_s = f"{thr:.2f}" if thr is not None else "0"
        if True:
            # the global form (§4.12) reads no key, so over a single table one
            # cell per (source, WHERE) is the whole sweep — it rides on the first key
            for form in FORMS + (("global",) if (label or key == keys[0]) else ()):
                sql = build(key, where, form, thr_s, source, args.payloads)
                # native
                con.transparent = False
                nat = con.execute(sql).fetchall()
                t_nat = time_min(lambda: con.execute(sql).fetchall(), args.n)
                # transparent (first call may upload; timed calls are warm)
                con.transparent = True
                got = con.execute(sql).fetchall()
                lr = con.last_rewrite()
                if not lr["rewritten"]:
                    soft = lr["reason"] == "threshold" and lr.get("detail", "").startswith(("soft", "probation"))
                    print(f"| {key_label} | {where or '—'} | {sel} | {form} | {len(nat)} | {t_nat:.1f} | — | — | "
                          f"declined ({lr['reason']}{', soft' if soft else ''}) |")
                    sys.stdout.flush()
                    if args.probation and soft:
                        verdict, ratio, t_nat2, t_tr, identical, rounds = probation_pass(
                            con, sql, nat, t_nat, args.n, args.min_ratio, args.probation_timeout, form)
                        ok = verdict == "stayed native" or (identical and ratio >= args.min_ratio)
                        note = (f"{verdict} {ratio:.2f}×" if ratio is not None
                                else f"{verdict} ({rounds} round(s))")
                        if not ok:
                            fails.append((key_label + " [probation]", where, form, ratio or 0.0, identical))
                        probation_rows.append((key_label, where, form, verdict, ratio, t_nat2, t_tr,
                                               rounds, identical))
                        print(f"| {key_label} | {where or '—'} | {sel} | {form} | {len(nat)} | {t_nat2:.1f} | "
                              f"{'—' if t_tr is None else f'{t_tr:.1f}'} | "
                              f"{'—' if ratio is None else f'{ratio:.2f}×'} | probation: {note}"
                              + ("" if ok else (" FAIL rows differ" if not identical else " FAIL")) + " |")
                        sys.stdout.flush()
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
                t_tr = time_min(lambda: con.execute(sql).fetchall(), args.n)
                lr2 = con.last_rewrite()
                if not lr2["rewritten"]:
                    # the once-per-template output-size check sent the template
                    # back to native after its first rewritten run: the timed
                    # runs were native, there is no ratio to report
                    print(f"| {key_label} | {where or '—'} | {sel} | {form} | {len(nat)} | {t_nat:.1f} | — | — | "
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
                            con.execute(sql).fetchall()
                        t = time_min(lambda: con.execute(sql).fetchall(), args.n)
                        if transparent:
                            t_tr = min(t_tr, t)
                        else:
                            t_nat = min(t_nat, t)
                    con.transparent = True
                    ratio = t_nat / t_tr if t_tr > 0 else float("inf")
                ok = identical and ratio >= args.min_ratio
                result = "PASS" if ok else ("FAIL rows differ" if not identical else "FAIL")
                if not ok:
                    fails.append((key_label, where, form, ratio, identical))
                print(f"| {key_label} | {where or '—'} | {sel} | {form} | {len(got)} | {t_nat:.1f} | {t_tr:.1f} | "
                      f"{ratio:.2f}× | {result} |")
                sys.stdout.flush()
    print()
    if args.probation:
        promoted = [r for r in probation_rows if r[3] == "promoted"]
        print(f"## probation (§9.1): {len(probation_rows)} softly declined cell(s), {len(promoted)} promoted")
        print()
        print("| key | WHERE | form | outcome | native ms | transparent ms | ratio | rounds |")
        print("|---|---|---|---|---|---|---|---|")
        for key, where, form, verdict, ratio, t_nat2, t_tr, rounds, identical in probation_rows:
            print(f"| {key} | {where or '—'} | {form} | {verdict}{'' if identical else ' (ROWS DIFFER)'} | "
                  f"{t_nat2:.1f} | {'—' if t_tr is None else f'{t_tr:.1f}'} | "
                  f"{'—' if ratio is None else f'{ratio:.2f}×'} | {rounds} |")
        pb = con.probation()
        print()
        print(f"probe budget: {pb['budget']['spent_ms_last_minute']:.1f} ms spent in the last minute of "
              f"{pb['budget']['ms_per_min']:.0f} ms allowed; sets held for trials: "
              f"{pb['bytes'] / 2 ** 20:.0f} MiB")
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
