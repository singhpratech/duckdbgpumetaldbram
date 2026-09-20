#!/usr/bin/env python3
"""avg_decimal_parity.py — does avg() over a DECIMAL payload return native's rows?

DuckDB finalises an average as a quotient in `long double`. On arm64 that is a
plain double (53-bit mantissa); on x86-64 it is the 80-bit type (64-bit
mantissa). The wrapper derives avg over a DECIMAL(p, s) payload in SQL, as
`double(unscaled sum) / (count * 10^s)`, which is native's own expression only
where long double IS double — so on x86 the rewritten statement can round
differently. The difference appears only once a group's unscaled 128-bit sum
passes 2^53, which is why a small fixture (and TPC-H Q1 at SF1) can look
identical while the shape is in fact unsafe.

This script builds a table whose group sums are well past 2^53 and compares
the wrapper's rows against native's, row by row. It is the before / after
reproducer for the guard in `_rewrite.check_types`:

  before the guard is live   x86: rewritten, N groups differ  -> the bug
  after  the guard is live   x86: declined (shape, long double), 0 differ
                             arm64 (avgf=53): rewritten, 0 differ

Usage:
  python3 scripts/avg_decimal_parity.py [--rows 4000000] [--groups 5000]
                                        [--force-avg-bits 64|53]

--force-avg-bits pretends the extension reported that width, which simulates
the other platform on this one. Exits non-zero if a REWRITTEN statement's rows
differ from native's (rule 2), or if a run that was supposed to decline did not.
"""
from __future__ import annotations
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import gpudb   # noqa: E402

QUERY = "SELECT k, avg(amt) FROM d GROUP BY k ORDER BY k"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=4_000_000)
    ap.add_argument("--groups", type=int, default=5_000)
    ap.add_argument("--force-avg-bits", type=int, default=0,
                    help="pretend gpu_build_info() reported this mantissa width (64 = x86, 53 = arm64)")
    args = ap.parse_args()

    con = gpudb.connect(residency="eager")
    info = con._raw.execute("SELECT gpu_build_info()").fetchone()[0]
    print(f"# {info}")
    print(f"# reported avg mantissa: {con._avg_float_bits} bits"
          + (f" — forced to {args.force_avg_bits} for this run" if args.force_avg_bits else ""))
    con.execute(f"""CREATE TABLE d AS
        SELECT (i % {args.groups})::BIGINT AS k,
               (((i * 7919) % 99991) * 10000000000 / 100.0 + (i % 997))::DECIMAL(18,2) AS amt
        FROM range({args.rows}) r(i)""")
    if args.force_avg_bits:
        con._avg_float_bits = args.force_avg_bits
        con._invalidate_all("SET")

    widest = con._raw.execute(
        "SELECT max(s) FROM (SELECT abs(sum(amt) * 100) AS s FROM d GROUP BY k)").fetchone()[0]
    past = float(widest) > 2.0 ** 53
    print(f"# widest unscaled group sum: {widest} (past 2^53: {past})")
    if not past:
        print("# WARNING: no group sum passes 2^53, so this data cannot show the difference")

    want = con._raw.execute(QUERY).fetchall()
    got = con.execute(QUERY).fetchall()
    lr = con.last_rewrite()
    differ = [(a, b) for a, b in zip(got, want) if a != b]
    print(f"rewritten: {lr['rewritten']}  reason: {lr['reason'] or '-'}")
    if lr["detail"]:
        print(f"detail:    {lr['detail']}")
    print(f"groups:    {len(want)}   differing from native: {len(differ)}")
    for a, b in differ[:5]:
        print(f"  k={a[0]}  gpudb {a[1]!r}  native {b[1]!r}")

    bad = 0
    if lr["rewritten"] and differ:
        print("FAIL: the statement ran on the device and returned different rows (rule 2)")
        bad = 1
    elif lr["rewritten"]:
        print("ok: rewritten, and every group matches native")
    else:
        if "long double" not in (lr["detail"] or ""):
            print("FAIL: declined, but not by the long double guard — check the reason")
            bad = 1
        else:
            print("ok: declined by the long double guard, and DuckDB's own rows came back")
    con.close()
    return bad


if __name__ == "__main__":
    sys.exit(main())
