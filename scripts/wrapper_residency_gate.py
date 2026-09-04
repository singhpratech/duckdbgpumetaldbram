#!/usr/bin/env python3
"""wrapper_residency_gate.py — the residency-active gate rows
(docs/TRANSPARENT_DESIGN.md §9.3): native shapes run through the Python
wrapper while the residency manager uploads a resident set in idle segments
(§5.5, milestone 0c), against the same shapes with residency='manual'.

For every native shape: N statements at the cadence of an interactive
session (a uniform 0–50 ms gap before each), statement-vs-statement on one
clock, reported as median / p99 / max. min-of-N is printed but never used
(it hides contention entirely). Three passes per shape — manual, background,
manual — so the two manual passes establish the noise band:

    pass  <=>  p99(background) <= max(p99(manual A), p99(manual B)) / thresh

The background pass first sights the rewritable shape over lineitem (which
runs native and schedules the upload), then issues the native statements;
the upload session runs only in the gaps. The pass also reports whether the
set became ready during the run, how many segments and interrupts it took,
and the per-segment scan time, so a stalled or an intrusive session is
visible either way.

  ./scripts/wrapper_residency_gate.py [--db data/tpch_sf10/tpch.duckdb]
        [--iters 200] [--thresh 0.9] [--idle-ms 20] [--gap-ms 50]
"""
from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import duckdb  # noqa: E402
import gpudb   # noqa: E402

SIGHT = "SELECT l_orderkey, sum(l_quantity) FROM lineitem GROUP BY l_orderkey HAVING sum(l_quantity) > 300"
SHAPES = [
    # a Q18-like GROUP BY that stays native (the WHERE keeps it off the rewrite path)
    ("q18_native", "SELECT l_orderkey, sum(l_quantity) AS q FROM lineitem WHERE l_linenumber > 0 "
                   "GROUP BY l_orderkey HAVING sum(l_quantity) > 300", 200),
    ("small_scan", "SELECT count(*), sum(l_extendedprice) FROM lineitem WHERE l_shipdate < DATE '1992-03-01'", 200),
    ("point_lookup", "SELECT o_totalprice FROM orders WHERE o_orderkey = 4", 400),
]


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]


def fmt(xs):
    return (f"n={len(xs)} min={min(xs):.1f} med={statistics.median(xs):.1f} "
            f"p99={pct(xs, 0.99):.1f} max={max(xs):.1f} ms")


def run_shape(con, sql, iters, gap_ms, rng, ready_probe=None):
    for _ in range(5):
        con.execute(sql).fetchall()
    lat, starts, ready_at = [], [], None
    t_run = time.perf_counter()
    for i in range(iters):
        time.sleep(rng.uniform(0, gap_ms / 1000.0))
        t0 = time.perf_counter()
        starts.append(time.monotonic())
        con.execute(sql).fetchall()
        lat.append((time.perf_counter() - t0) * 1000.0)
        if ready_probe is not None and ready_at is None and ready_probe():
            ready_at = (i, (time.perf_counter() - t_run) * 1000.0)
    return lat, ready_at, starts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/tpch_sf10/tpch.duckdb")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--thresh", type=float, default=0.9)
    ap.add_argument("--idle-ms", type=float, default=20.0)
    ap.add_argument("--gap-ms", type=float, default=50.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--only", default="", help="run one shape by label")
    args = ap.parse_args()
    if not os.path.exists(args.db):
        print(f"missing {args.db} — SF=10 ./scripts/gen_tpch.sh"); return 2
    os.environ.setdefault("GPUDB_UPLOAD_POOL_MAX_MB", "8192")
    rng = random.Random(args.seed)
    print(f"wrapper residency gate: db={args.db} iters={args.iters} gap=0-{args.gap_ms:.0f} ms "
          f"idle={args.idle_ms:.0f} ms  pass: p99(background) <= max(p99 manual A, B) / {args.thresh}")
    fails = 0
    for label, sql, iters in SHAPES:
        if args.only and label != args.only:
            continue
        iters = max(iters, args.iters)
        results = {}
        for mode in ("manual", "background", "manual2"):
            residency = "manual" if mode.startswith("manual") else "background"
            con = gpudb.connect(args.db, read_only=True, residency=residency, idle_ms=args.idle_ms)
            probe = None
            if residency == "background":
                con.execute("SELECT gpu_invalidate('gpudb:v1')").fetchall()
                con.execute(SIGHT).fetchall()          # first sighting: native, schedules the upload
                lr = con.last_rewrite()
                tag = lr["tag"]
                if lr["rewritten"] or lr["reason"] != "not_resident" or not tag:
                    print(f"[{label}] sighting did not schedule an upload: {lr}"); return 2
                probe = lambda: con._manager.is_ready(tag)   # noqa: E731
            lat, ready_at, starts = run_shape(con, sql, iters, args.gap_ms, rng, probe)
            results[mode] = lat
            print(f"[{label}/{mode}] native {fmt(lat)}")
            if residency == "background":
                pr = con._manager.progress()[tag]
                seg = pr["seg_ms"]
                seg_s = (f"segments={pr['segments']}/{pr['planned']} scan min/med/max="
                         f"{min(seg):.1f}/{statistics.median(seg):.1f}/{max(seg):.1f} ms" if seg else "no segment landed")
                where = (f"ready after statement {ready_at[0]} ({ready_at[1]:.0f} ms into the run)"
                         if ready_at else f"NOT ready during the run (state={pr['state']})")
                print(f"[{label}/background] upload: {where}; {seg_s}; interrupts={pr['interrupts']} "
                      f"attempts={pr['attempts']} session={pr['session_ms']:.0f} ms")
                f0, f1 = pr["finish_window"]
                if f1 > f0:
                    inside = [lat[i] for i, st in enumerate(starts) if f0 - 0.001 <= st <= f1]
                    print(f"[{label}/background] finish call {1000*(f1-f0):.0f} ms; statements started inside it: "
                          f"{len(inside)} " + (f"({fmt(inside)})" if inside else "") +
                          f"; slowest statement overall {max(lat):.1f} ms at #{lat.index(max(lat))}"
                          f"{' (inside finish)' if f0 - 0.001 <= starts[lat.index(max(lat))] <= f1 else ''}")
                if not ready_at:
                    con._manager.wait_idle(120)
                    pr = con._manager.progress()[tag]
                    print(f"[{label}/background] after the run: state={pr['state']} segments={pr['segments']}/{pr['planned']} "
                          f"interrupts={pr['interrupts']} session={pr['session_ms']:.0f} ms")
            con.close()
        a, b, bg = results["manual"], results["manual2"], results["background"]
        band = max(pct(a, 0.99), pct(b, 0.99))
        ratio = band / pct(bg, 0.99) if pct(bg, 0.99) else float("inf")
        ok = pct(bg, 0.99) <= band / args.thresh
        print(f"gate: {label} p99 manual A/B {pct(a, 0.99):.1f}/{pct(b, 0.99):.1f} ms, background {pct(bg, 0.99):.1f} ms "
              f"({ratio:.2f}x); max manual {max(max(a), max(b)):.1f} vs background {max(bg):.1f} ms")
        print(f"  [{'pass' if ok else 'FAIL'}] {label}")
        fails += 0 if ok else 1
    print("----")
    print(f"{fails} failing rows")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
