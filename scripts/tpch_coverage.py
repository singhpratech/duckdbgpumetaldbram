#!/usr/bin/env python3
"""tpch_coverage.py — which of the 22 TPC-H queries does the transparent path
answer, and why not the others?

Runs every official query (text from DuckDB's `tpch` extension) through
`gpudb.connect()` and natively, in one process: reports rewritten / native
with the decline reason, checks rewritten results against native, and times
both (hot loop, minimum of N; rewritten queries are warmed first). It is a
coverage map, not a gate: nothing fails on a declined query. A rewritten
query whose rows differ from native DOES exit non-zero.

Usage:
  python3 scripts/tpch_coverage.py [--db data/tpch_sf1/tpch.duckdb] [--n 5] [--no-thresholds]
"""
from __future__ import annotations
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import duckdb  # noqa: E402
import gpudb   # noqa: E402


def best(run, n):
    b = float("inf")
    for _ in range(n):
        t0 = time.perf_counter()
        r = run()
        b = min(b, (time.perf_counter() - t0) * 1000.0)
    return r, b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/tpch_sf1/tpch.duckdb")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--no-thresholds", action="store_true", help="rewrite every shape the engine accepts")
    args = ap.parse_args()
    if not os.path.exists(args.db):
        print(f"missing {args.db} — SF=1 ./scripts/gen_tpch.sh", file=sys.stderr)
        return 2
    c0 = duckdb.connect()
    try:
        c0.execute("LOAD tpch")
    except Exception:
        c0.execute("INSTALL tpch")
        c0.execute("LOAD tpch")
    queries = c0.execute("SELECT query_nr, query FROM tpch_queries() ORDER BY 1").fetchall()
    logs = []
    # the shipping configuration, floor included (tables under 1M rows are never parsed);
    # --no-thresholds also drops the floor, to see every shape the engine accepts
    con = gpudb.connect(args.db, read_only=True, residency="eager",
                        floor_rows=0 if args.no_thresholds else 1_000_000,
                        thresholds=not args.no_thresholds, log=logs.append)
    info = con._raw.execute("SELECT gpu_build_info()").fetchone()[0]
    print(f"# TPC-H coverage — {args.db}, {info}, thresholds {'off' if args.no_thresholds else 'on'}, N={args.n}")
    print()
    print("| query | path | native ms | transparent ms | ratio | identical | note |")
    print("|---|---|---|---|---|---|---|")
    rewritten = wrong = 0
    for nr, sql in queries:
        sql = sql.strip().rstrip(";")
        con.transparent = False
        nat, t_nat = best(lambda: con.execute(sql).fetchall(), args.n)
        con.transparent = True
        mark = len(logs)
        got = con.execute(sql).fetchall()
        lr = con.last_rewrite()
        if not lr["rewritten"]:
            why = [x for x in logs[mark:] if "declined" in x or "threshold" in x or "split" in x]
            note = (why[-1] if why else "").replace("|", "/")[:110]
            print(f"| Q{nr} | native ({lr['reason']}) | {t_nat:.1f} | — | — | — | {note} |")
            continue
        for _ in range(15):
            con.execute(sql).fetchall()
        got, t_tr = best(lambda: con.execute(sql).fetchall(), args.n)
        same = got == nat or sorted(map(str, got)) == sorted(map(str, nat))
        rewritten += 1
        wrong += 0 if same else 1
        print(f"| Q{nr} | GPU ({lr['form']}) | {t_nat:.1f} | {t_tr:.1f} | {t_nat / t_tr:.2f}× | {same} | |")
    print()
    print(f"{rewritten} of {len(queries)} queries answered on the device; {wrong} with rows that differ from native")
    return 1 if wrong else 0


if __name__ == "__main__":
    sys.exit(main())
