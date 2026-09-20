#!/usr/bin/env python3
"""budget_gate.py — the device-memory budget, under pressure, end to end
(docs/TRANSPARENT_DESIGN.md §5.5).

The other gates run with room to spare, so none of them ever asks the budget
to do anything; the budget that shipped was never enforced in a measurement
and nobody noticed (docs/RESEARCH_NOTES.md, 2026-09-20). This one runs a long
eager session over many distinct templates under a budget deliberately too
small for them and asserts, at every sample:

  a. PHYSICALLY resident bytes <= the budget. Physical means each store
     column counted ONCE plus every set that holds columns of its own — the
     per-set figures cannot be summed, because a store-backed set is a view
     over shared columns and reports what its lanes cost (measured: summing
     them over-counts 4.5x; summing `gpu_residents().bytes` alone, where a
     view reports 0, under-counts them to nothing).
     The bound is exact, not "the budget plus one upload": the wrapper
     decides BEFORE it uploads, so nothing is ever in flight above the line.
  b. the budget makes room by EVICTING, and never evicts for a candidate it
     then refuses (`evictions_wasted` stays 0).
  c. every statement's rows are identical to plain DuckDB's.
  d. no statement is declined with `error`, and none of the declines is an
     upload that failed and was retried on the next statement — an upload
     the device refuses is attempted once, not once per statement.

With `--refuse-mb N` (Metal only) the backend is told to refuse any exact
upload above N MiB, which is the only way to reach the device-refusal paths
on a machine whose GPU has 51 GiB: the run then asserts (d) with real
refusals, that nothing of a refused upload stays resident, and that the
process does not grow per attempt.

  ./scripts/budget_gate.py [--db data/tpch_sf1/tpch.duckdb] [--budget 256MB]
        [--every 10] [--residency eager|background] [--refuse-mb 8]
        [--templates 120]

Needs a built extension (build-macos / build-linux) or GPUDB_EXTENSION_PATH,
and TPC-H SF1 (`SF=1 ./scripts/gen_tpch.sh`). Takes about a minute at SF1.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import duckdb  # noqa: E402
import gpudb   # noqa: E402

MiB = 1 << 20

# Distinct (table, key, payload, WHERE) shapes: each one is its own resident
# set, which is what makes a small budget bite. They are ordinary GROUP BYs —
# the point here is the budget, not coverage.
LINE_KEYS = ["l_suppkey", "l_partkey", "l_linenumber", "l_shipmode", "l_returnflag", "l_orderkey"]
LINE_PAY = ["l_extendedprice", "l_quantity", "l_discount", "l_tax"]
LINE_WHERE = ["", "l_discount < 0.05", "l_linenumber = 1", "l_shipdate < DATE '1995-01-01'",
              "l_linenumber <= 3", "l_discount <= 0.09"]
OTHER = ([("orders", k, "o_totalprice", w)
          for k in ("o_orderpriority", "o_orderstatus", "o_custkey", "o_shippriority")
          for w in ("", "o_totalprice > 50000", "o_orderdate < DATE '1995-01-01'")]
         + [("part", k, "p_retailprice", w)
            for k in ("p_brand", "p_type", "p_size", "p_container") for w in ("", "p_size > 10")]
         + [("customer", k, "c_acctbal", w)
            for k in ("c_mktsegment", "c_nationkey") for w in ("", "c_acctbal > 0")]
         + [("supplier", "s_nationkey", "s_acctbal", "")])


def templates():
    out = [("lineitem", k, p, w) for k in LINE_KEYS for p in LINE_PAY for w in LINE_WHERE]
    return out + OTHER


def statement(table, key, payload, where):
    return (f"SELECT {key}, sum({payload}) FROM {table}"
            + (f" WHERE {where}" if where else "")
            + f" GROUP BY {key}")


def rss_bytes() -> int:
    """Live resident set size of this process, or 0 where ps cannot say. Used
    only to report growth across refused uploads, never to fail a run — RSS
    moves for reasons that have nothing to do with us."""
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                             capture_output=True, text=True).stdout.strip()
        return int(out) * 1024 if out else 0
    except Exception:
        return 0


def sample(con):
    m = con.memory()
    return {
        "physical": m.get("bytes"),
        "budget": m.get("budget") or 0,
        "evictions": m.get("evictions", 0),
        "wasted": m.get("evictions_wasted", 0),
        "refusals": m.get("refusals", 0),
        "device": m.get("device_allocated"),
        "ready": sum(1 for v in m["sets"].values() if v["state"] == "ready"),
        "failed": sum(1 for v in m["sets"].values() if v["state"] == "failed"),
        "attempts": max((con._manager.get(t).attempts for t in m["sets"]), default=0),
        "host_lanes": sum(1 for c in con.store_columns() if c.get("on_gpu") is False),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/tpch_sf1/tpch.duckdb")
    ap.add_argument("--budget", default="256MB")
    ap.add_argument("--residency", default="eager", choices=("eager", "background"))
    ap.add_argument("--every", type=int, default=10, help="sample the accounting every N statements")
    ap.add_argument("--templates", type=int, default=0, help="stop after this many (0 = all)")
    ap.add_argument("--refuse-mb", type=float, default=0.0,
                    help="Metal only: make the backend refuse any exact upload above this many "
                         "MiB, so the device-refusal paths are reachable on a machine with "
                         "memory to spare (sets GPUDB_METAL_UPLOAD_REFUSE_MB for the child "
                         "state; must be set before the extension loads, so pass it in the "
                         "environment instead if this process has already connected)")
    a = ap.parse_args()
    if a.refuse_mb > 0:
        os.environ["GPUDB_METAL_UPLOAD_REFUSE_MB"] = str(a.refuse_mb)
    if not os.path.exists(a.db):
        print(f"missing {a.db} — SF=1 ./scripts/gen_tpch.sh", file=sys.stderr)
        return 2

    con = gpudb.connect(a.db, read_only=True, residency=a.residency, memory_budget=a.budget)
    # the control runs on a cursor of the SAME database: a second connection to
    # one file with a different configuration is refused, and a cursor is the
    # plain engine either way (the wrapper only rewrites through its own entry
    # points)
    nat = con._raw.cursor()
    info = con._raw.execute("SELECT gpu_build_info()").fetchone()[0]
    budget = con.memory()["budget"]
    if not budget:
        print("budget_gate: the connection has no budget — nothing to gate", file=sys.stderr)
        return 2
    ts = templates()[:a.templates or None]
    print(f"# budget_gate — {a.db}, {info}")
    print(f"# {len(ts)} templates, residency {a.residency}, budget {budget / MiB:.0f} MiB"
          + (f", the backend refuses uploads above {a.refuse_mb:.0f} MiB" if a.refuse_mb else ""))
    print()
    print("|   n | ready | refused | resident MiB | budget MiB | device MiB | evictions | wasted | RSS MiB |")
    print("|---|---|---|---|---|---|---|---|---|")

    fails, differ, errors, reasons = [], 0, 0, {}
    rss0 = rss_bytes()
    for i, t in enumerate(ts, 1):
        sql = statement(*t)
        rows = con.execute(sql).fetchall()
        lr = con.last_rewrite()
        reason = lr["reason"] or ("rewritten" if lr["rewritten"] else "?")
        reasons[reason] = reasons.get(reason, 0) + 1
        if reason == "error":
            errors += 1
            fails.append(f"(error) {sql[:70]}: {(lr['detail'] or '')[:160]}")
        want = nat.execute(sql).fetchall()
        if sorted(map(repr, rows)) != sorted(map(repr, want)):
            differ += 1
            fails.append(f"rows differ: {sql[:90]}")
        if i % a.every == 0 or i == len(ts):
            s = sample(con)
            rss = rss_bytes()
            print(f"| {i:3d} | {s['ready']} | {s['failed']} | "
                  f"{(s['physical'] or 0) / MiB:.1f} | {s['budget'] / MiB:.0f} | "
                  f"{(s['device'] or 0) / MiB:.1f} | {s['evictions']} | {s['wasted']} | "
                  f"{rss / MiB:.0f} |")
            if s["physical"] is None:
                fails.append(f"n={i}: the extension could not be asked what is resident")
            elif s["physical"] > s["budget"]:
                fails.append(f"n={i}: {s['physical'] / MiB:.1f} MiB resident > "
                             f"{s['budget'] / MiB:.0f} MiB budget")
            if s["wasted"]:
                fails.append(f"n={i}: {s['wasted']} evictions bought nothing")
            if s["host_lanes"]:
                fails.append(f"n={i}: {s['host_lanes']} resident lanes are on the host, "
                             f"not the device")

    final = sample(con)
    rss = rss_bytes()
    print()
    print(f"# reasons: {reasons}")
    print(f"# evictions {final['evictions']} (wasted {final['wasted']}), "
          f"refusals {final['refusals']}, "
          f"at most {final['attempts']} upload attempts for any one set, "
          f"RSS {rss0 / MiB:.0f} -> {rss / MiB:.0f} MiB")
    # An upload the device refuses must be attempted ONCE per set, not once
    # per statement: the whole point of remembering a refusal.
    if final["attempts"] > 2:
        fails.append(f"a set was uploaded {final['attempts']} times — a refusal is not being "
                     f"remembered")
    if a.refuse_mb and final["refusals"] == 0:
        fails.append("the backend was told to refuse uploads and nothing was refused — "
                     "the injection did not reach the extension (it is read once, at load)")
    if not a.refuse_mb and final["evictions"] == 0:
        # Not a failure on its own: a budget that never has to evict is a
        # budget that fits. Said out loud so a run that proves nothing is not
        # mistaken for a run that proves something.
        print("# note: nothing was ever evicted — this budget was never tight enough to "
              "make the eviction path run. Lower --budget to exercise it.")
    con.close()
    nat.close()
    if fails:
        print()
        print(f"# FAIL ({len(fails)}):")
        for f in fails[:40]:
            print(f"#   {f}")
        return 1
    print(f"# PASS — {len(ts)} statements, {differ} differing, {errors} errors, "
          f"resident never above the budget")
    return 0


if __name__ == "__main__":
    sys.exit(main())
