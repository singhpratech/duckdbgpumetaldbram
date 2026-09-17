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

Usage:
  python3 scripts/transparent_gate.py [--db data/tpch_sf1/tpch.duckdb] [--n 5]
                                      [--min-ratio 1.0] [--keys l_orderkey,l_partkey,...]
Needs a built extension (build-macos / build-linux) or GPUDB_EXTENSION_PATH.
"""
from __future__ import annotations
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import gpudb  # noqa: E402

# key column -> (label, approximate groups at SF1)
KEYS = {
    "l_linenumber": "7 groups",
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
FORMS = ("plain", "having", "topk", "projected")   # projected: expressions over aggregates (§4.11)
# key joins (§4.8): label -> (FROM clause, key column); the WHERE list below
# applies where its table is part of the join
JOINS = {
    "li x orders":            ("lineitem JOIN orders ON l_orderkey = o_orderkey", ["o_custkey", "o_orderdate", "l_suppkey"]),
    "li x orders x customer": ("lineitem JOIN orders ON l_orderkey = o_orderkey JOIN customer ON o_custkey = c_custkey",
                               ["c_nationkey", "c_custkey"]),
    "li x part":              ("lineitem JOIN part ON l_partkey = p_partkey", ["p_brand", "p_size"]),
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
    if form == "projected":
        return (f"SELECT {key}, sum({PAYLOAD}) / count(*) AS mean{more} FROM {source}{w} GROUP BY {key} "
                f"HAVING sum({PAYLOAD}) > {having_thr} AND count(*) > 1")
    if form == "having":
        return (f"SELECT {key}, sum({PAYLOAD}) AS q{more} FROM {source}{w} GROUP BY {key} "
                f"HAVING sum({PAYLOAD}) > {having_thr}")
    return (f"SELECT {key}, sum({PAYLOAD}) AS q{more} FROM {source}{w} GROUP BY {key} "
            f"ORDER BY q DESC LIMIT 10")


def time_min(run, n: int) -> float:
    best = float("inf")
    for _ in range(n):
        t0 = time.perf_counter()
        run()
        best = min(best, (time.perf_counter() - t0) * 1000.0)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/tpch_sf1/tpch.duckdb")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--min-ratio", type=float, default=1.0)
    ap.add_argument("--keys", default=",".join(KEYS))
    ap.add_argument("--wheres", default="all", help="'all' or a ';'-separated list of predicates ('' = none)")
    ap.add_argument("--joins", default="all", help="'all', 'none' or a ';'-separated list of JOINS labels")
    ap.add_argument("--no-single", action="store_true", help="skip the single-table sweep")
    ap.add_argument("--exprs", action="store_true",
                    help="aggregate l_extendedprice * (1 - l_discount) and add an expression WHERE (computed lanes)")
    ap.add_argument("--payloads", type=int, default=1, help="aggregate this many payload columns per statement (1-5)")
    ap.add_argument("--no-thresholds", action="store_true",
                    help="rewrite every shape the engine accepts (data collection for the thresholds; "
                         "rows below the bound are reported, the exit code still fails on them)")
    args = ap.parse_args()
    if args.exprs:
        global PAYLOAD
        PAYLOAD = REVENUE
        WHERES["l_commitdate < l_receiptdate AND (l_shipmode = 'AIR' OR l_quantity > 40)"] = "computed"
        WHERES["extract(year FROM l_shipdate) = 1995"] = "computed"
        JOIN_WHERES["o_orderpriority LIKE '1-%' OR o_orderpriority LIKE '2-%'"] = "orders"
    if not os.path.exists(args.db):
        print(f"missing {args.db} — SF=1 ./scripts/gen_tpch.sh", file=sys.stderr)
        return 2

    con = gpudb.connect(args.db, read_only=True, residency="eager", floor_rows=0,
                        thresholds=not args.no_thresholds)
    info = con._raw.execute("SELECT gpu_build_info()").fetchone()[0]
    rows_total = con._raw.execute("SELECT count(*) FROM lineitem").fetchone()[0]
    print(f"# transparent_gate — {args.db} ({rows_total:,} rows), {info}, N={args.n}, min ratio {args.min_ratio}, "
          f"payload columns {args.payloads}")
    print()
    print("| key | WHERE | selectivity | form | rows out | native ms | transparent ms | ratio | result |")
    print("|---|---|---|---|---|---|---|---|---|")

    keys = [k for k in args.keys.split(",") if k]
    wheres = list(WHERES) if args.wheres == "all" else args.wheres.split(";")
    fails = []
    cells = []          # (source label, FROM clause, key, where)
    if not args.no_single:
        cells += [("", "lineitem", key, where) for where in wheres for key in keys]
    if args.joins != "none":
        labels = list(JOINS) if args.joins == "all" else args.joins.split(";")
        for label in labels:
            source, jkeys = JOINS[label]
            for where, table in JOIN_WHERES.items():
                if table and table not in source:
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
            for form in FORMS:
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
                    print(f"| {key_label} | {where or '—'} | {sel} | {form} | {len(nat)} | {t_nat:.1f} | — | — | "
                          f"declined ({lr['reason']}) |")
                    continue
                if form == "topk":
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
                for _ in range(2):
                    if not identical or ratio >= args.min_ratio:
                        break
                    for _ in range(args.n):
                        con.transparent = False
                        t_nat = min(t_nat, time_min(lambda: con.execute(sql).fetchall(), 1))
                        con.transparent = True
                        t_tr = min(t_tr, time_min(lambda: con.execute(sql).fetchall(), 1))
                    ratio = t_nat / t_tr if t_tr > 0 else float("inf")
                ok = identical and ratio >= args.min_ratio
                result = "PASS" if ok else ("FAIL rows differ" if not identical else "FAIL")
                if not ok:
                    fails.append((key_label, where, form, ratio, identical))
                print(f"| {key_label} | {where or '—'} | {sel} | {lr['form']} | {len(got)} | {t_nat:.1f} | {t_tr:.1f} | "
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
