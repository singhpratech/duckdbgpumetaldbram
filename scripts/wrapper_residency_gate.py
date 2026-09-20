#!/usr/bin/env python3
"""wrapper_residency_gate.py — the residency-active gate rows
(docs/TRANSPARENT_DESIGN.md §9.3): native shapes run through the Python
wrapper while the residency manager uploads a resident set in idle segments
(§5.5, milestone 0c), against the same shapes with residency='manual'.

For every native shape: N statements at the cadence of an interactive
session (a uniform 0–50 ms gap before each, the same gap sequence in every
pass of a round), statement-vs-statement on one clock, reported as median /
p99 / max. min-of-N is printed but never used (it hides contention
entirely). Three passes per ROUND — manual, background, manual — so the two
manual passes measure the same thing twice and the pair is the control's own
uncertainty. The verdict is taken on the POOLED samples of every round run
so far:

    pass  p99(background) <= min(p99 manual A, p99 manual B) / thresh
    LOSS  p99(background) >  max(p99 manual A, p99 manual B) / thresh
    else  the control pair straddles the decision

A pass ends the shape. Anything else — a loss or a straddle — is
re-measured: another round is run and the verdict taken again on everything
measured so far, up to --max-rounds, so a row only FAILS on a loss that
survived the re-measurement (the pattern transparent_gate.py uses for a
losing cell). A shape still straddling after the last round is reported
INCONCLUSIVE (exit 3): not a pass and not a failure, because at that point
the machine did not hold still enough to resolve a `thresh` margin at all.
Every round's numbers are printed, losing rounds included.

The re-measurement is what a sub-millisecond shape needs: `point_lookup`
runs at a median of 0.4 ms in one machine state and 1.4 ms in the other
(the two modes of §9.1, 2026-09-18), and which state a 10 s pass lands in
moves its p99 further than the 10% this gate is testing for.

The background pass first sights the rewritable shape over lineitem (which
runs native and schedules the upload), then issues the native statements;
the upload session runs only in the gaps. The pass also reports whether the
set became ready during the run, how many segments and interrupts it took,
and the per-segment scan time, so a stalled or an intrusive session is
visible either way.

Every timed shape must stay NATIVE in every pass — that is what the gate
compares. If the wrapper rewrites one (its coverage grew into the shape),
the statements are no longer the same work on both sides and the row is
void: the gate stops with exit 2 and the shape has to be replaced with one
the rewriter still declines. See the 2026-09-19 entry in
docs/RESEARCH_NOTES.md.

  ./scripts/wrapper_residency_gate.py [--db data/tpch_sf10/tpch.duckdb]
        [--iters 200] [--thresh 0.9] [--idle-ms 20] [--gap-ms 50]
        [--max-rounds 3] [--warm-ms 1500] [--dump raw.json]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import duckdb  # noqa: E402
import gpudb   # noqa: E402

SIGHT = "SELECT l_orderkey, sum(l_quantity) FROM lineitem GROUP BY l_orderkey HAVING sum(l_quantity) > 300"
# Every shape here must be one the rewriter DECLINES, in both residency modes
# and after the sighting's set is resident — the gate times native statements
# against native statements. `stddev` and `quantile_cont` are the declines used
# below (no device implementation), not a WHERE that happens to be off the
# rewrite path today: the WHERE forms the wrapper declined in 2026-09 it now
# rewrites, which is what rotted the previous q18_native / small_scan shapes.
SHAPES = [
    # a Q18-like GROUP BY over the key the upload is reading, that stays native
    ("q18_native", "SELECT l_orderkey, stddev(l_quantity) AS q FROM lineitem "
                   "WHERE l_shipdate < DATE '1993-01-01' GROUP BY l_orderkey HAVING stddev(l_quantity) > 20", 200),
    ("small_scan", "SELECT quantile_cont(l_extendedprice, 0.9) FROM lineitem "
                   "WHERE l_shipdate < DATE '1992-03-01'", 200),
    ("point_lookup", "SELECT o_totalprice FROM orders WHERE o_orderkey = 4", 400),
]


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]


def fmt(xs):
    return (f"n={len(xs)} min={min(xs):.1f} med={statistics.median(xs):.1f} "
            f"p99={pct(xs, 0.99):.1f} max={max(xs):.1f} ms")


def warm(con, sql, warm_ms, label, mode):
    """Warm the pass until its statement time has settled: at least 5 runs, then
    on until warm_ms of wall time has passed (capped at 200 runs). A pass opens
    its own connection, so the first statements of the FIRST pass also pay the
    file cache — which is what made manual A the slowest pass of every round."""
    t0 = time.perf_counter()
    n = 0
    while n < 5 or (time.perf_counter() - t0) * 1000.0 < warm_ms:
        if n >= 200:
            break
        con.execute(sql).fetchall()
        n += 1
    if con.last_rewrite()["rewritten"]:
        print(f"[{label}/{mode}] the wrapper REWROTE this shape — it is no longer a native shape and the "
              f"row would compare a device statement against a native one. Replace the shape.")
        return False
    return True


def run_shape(con, sql, iters, gap_ms, rng, ready_probe=None):
    """Times `iters` statements and counts how many of them the wrapper
    rewrote: a shape the gate calls native must be declined every time, in
    every pass, or the two sides are not the same work (`rewrote`)."""
    lat, starts, ready_at, rewrote = [], [], None, 0
    t_run = time.perf_counter()
    for i in range(iters):
        time.sleep(rng.uniform(0, gap_ms / 1000.0))
        t0 = time.perf_counter()
        starts.append(time.monotonic())
        con.execute(sql).fetchall()
        lat.append((time.perf_counter() - t0) * 1000.0)
        rewrote += 1 if con._last.rewritten else 0
        if ready_probe is not None and ready_at is None and ready_probe():
            ready_at = (i, (time.perf_counter() - t_run) * 1000.0)
    return lat, ready_at, starts, rewrote


def run_round(args, label, sql, iters, rnd):
    """One round: manual, background, manual — same gap sequence in each.
    Returns the three latency lists, or None if a shape stopped being native."""
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
                print(f"[{label}] sighting did not schedule an upload: {lr}")
                con.close()
                return None
            probe = lambda: con._manager.is_ready(tag)   # noqa: E731
        if not warm(con, sql, args.warm_ms, label, mode):
            con.close()
            return None
        # the same gap sequence in every pass of the round: the passes differ in
        # what the manager is doing, not in when the statements arrive
        rng = random.Random(args.seed + 1000 * rnd)
        lat, ready_at, starts, rewrote = run_shape(con, sql, iters, args.gap_ms, rng, probe)
        if rewrote:
            print(f"[{label}/{mode}] the wrapper REWROTE {rewrote} of {iters} statements of this shape — it is "
                  f"no longer a native shape and the row would compare device statements against native ones. "
                  f"Replace the shape.")
            con.close()
            return None
        results[mode] = lat
        print(f"[{label}/{mode}#{rnd}] native {fmt(lat)}")
        if residency == "background":
            pr = con._manager.progress()[tag]
            seg = pr["seg_ms"]
            seg_s = (f"segments={pr['segments']}/{pr['planned']} scan min/med/max="
                     f"{min(seg):.1f}/{statistics.median(seg):.1f}/{max(seg):.1f} ms" if seg else "no segment landed")
            where = (f"ready after statement {ready_at[0]} ({ready_at[1]:.0f} ms into the run)"
                     if ready_at else f"NOT ready during the run (state={pr['state']})")
            print(f"[{label}/background#{rnd}] upload: {where}; {seg_s}; interrupts={pr['interrupts']} "
                  f"attempts={pr['attempts']} session={pr['session_ms']:.0f} ms")
            # the pass split where the upload ended: a row that loses because of
            # the session loses HERE, and one that loses anyway does not
            if ready_at and 0 < ready_at[0] < len(lat) - 1:
                print(f"[{label}/background#{rnd}] while the session ran {fmt(lat[:ready_at[0]])}; "
                      f"after it {fmt(lat[ready_at[0]:])}")
            f0, f1 = pr["finish_window"]
            if f1 > f0:
                inside = [lat[i] for i, st in enumerate(starts) if f0 - 0.001 <= st <= f1]
                print(f"[{label}/background#{rnd}] finish call {1000*(f1-f0):.0f} ms; statements started inside it: "
                      f"{len(inside)} " + (f"({fmt(inside)})" if inside else "") +
                      f"; slowest statement overall {max(lat):.1f} ms at #{lat.index(max(lat))}"
                      f"{' (inside finish)' if f0 - 0.001 <= starts[lat.index(max(lat))] <= f1 else ''}")
            if not ready_at:
                con._manager.wait_idle(120)
                pr = con._manager.progress()[tag]
                print(f"[{label}/background#{rnd}] after the run: state={pr['state']} segments={pr['segments']}/{pr['planned']} "
                      f"interrupts={pr['interrupts']} session={pr['session_ms']:.0f} ms")
        con.close()
        sys.stdout.flush()
    return results


def judge(results, thresh):
    """(verdict, p99 A, p99 B, p99 background, and its ratio against the faster
    and the slower control). The two manual passes are one measurement of the
    same thing made twice, so the pair is the control's own uncertainty, and
    the background pass is only judged where it falls outside that:

        pass  p99(background) <= min(p99 A, p99 B) / thresh — within the
              tolerance of even the FASTER control
        LOSS  p99(background) >  max(p99 A, p99 B) / thresh — outside it
              against even the slower control
        else  the control pair straddles the decision: nothing is claimed and
              another round is run

    Fed the POOLED samples of every round so far: p99 over 200 statements is
    the third slowest of the pass, an order statistic that moved 30% between
    two control passes of an idle machine, and pooling is what makes it an
    estimate of the shape's tail rather than one draw of it."""
    a, b, bg = results["manual"], results["manual2"], results["background"]
    pa, pb, pbg = pct(a, 0.99), pct(b, 0.99), pct(bg, 0.99)
    band, floor = max(pa, pb), min(pa, pb)
    r_floor = floor / pbg if pbg else float("inf")
    r_band = band / pbg if pbg else float("inf")
    if pbg <= floor / thresh:
        verdict = "pass"
    elif pbg > band / thresh:
        verdict = "LOSS"
    else:
        verdict = "inconclusive"
    return verdict, pa, pb, pbg, r_floor, r_band


def report(results, label, what, thresh):
    verdict, pa, pb, pbg, r_floor, r_band = judge(results, thresh)
    a, b, bg = results["manual"], results["manual2"], results["background"]
    print(f"gate: {label} {what}: p99 manual A/B {pa:.1f}/{pb:.1f} ms, background {pbg:.1f} ms "
          f"({r_floor:.2f}x/{r_band:.2f}x of the faster/slower control); "
          f"max manual {max(max(a), max(b)):.1f} vs background {max(bg):.1f} ms -> {verdict}")
    sys.stdout.flush()
    return verdict, r_floor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/tpch_sf10/tpch.duckdb")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--thresh", type=float, default=0.9)
    ap.add_argument("--idle-ms", type=float, default=20.0)
    ap.add_argument("--gap-ms", type=float, default=50.0)
    ap.add_argument("--warm-ms", type=float, default=1500.0,
                    help="wall time to warm each pass before its statements are timed")
    ap.add_argument("--max-rounds", type=int, default=3,
                    help="rounds (manual, background, manual) per shape; a shape stops at its first "
                         "passing round and re-measures otherwise, and every round is printed")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--only", default="", help="run one shape by label")
    ap.add_argument("--dump", default="", help="write every measured latency to this JSON file")
    args = ap.parse_args()
    if not os.path.exists(args.db):
        print(f"missing {args.db} — SF=10 ./scripts/gen_tpch.sh"); return 2
    os.environ.setdefault("GPUDB_UPLOAD_POOL_MAX_MB", "8192")
    print(f"wrapper residency gate: db={args.db} iters={args.iters} gap=0-{args.gap_ms:.0f} ms "
          f"idle={args.idle_ms:.0f} ms warm={args.warm_ms:.0f} ms rounds<={args.max_rounds}\n"
          f"  pooled over the rounds run so far: pass at p99(background) <= min(p99 manual A, B) / {args.thresh}, "
          f"LOSS above max(p99 manual A, B) / {args.thresh}, another round in between")
    fails, unresolved, dump = 0, 0, {}
    for label, sql, iters in SHAPES:
        if args.only and label != args.only:
            continue
        iters = max(iters, args.iters)
        verdicts = []
        pooled = {"manual": [], "manual2": [], "background": []}
        for rnd in range(1, args.max_rounds + 1):
            results = run_round(args, label, sql, iters, rnd)
            if results is None:
                return 2
            if args.dump:
                dump[f"{label}#{rnd}"] = results
            for mode in pooled:
                pooled[mode] += results[mode]
            report(results, label, f"round {rnd} alone", args.thresh)
            verdict, ratio = report(pooled, label, f"after {rnd} round(s), pooled", args.thresh)
            verdicts.append((verdict, ratio))
            # a pass is taken at once; anything else is re-measured until the
            # rounds run out, and the verdict is the pooled one at that point
            if verdict == "pass":
                break
        final = verdicts[-1][0]
        if final == "inconclusive":
            state = "INCONCLUSIVE"
            unresolved += 1
        else:
            state = "pass" if final == "pass" else "FAIL"
            fails += 0 if state == "pass" else 1
        rounds_s = ", ".join(f"{v}({r:.2f}x)" for v, r in verdicts)
        print(f"  [{state}] {label} — {len(verdicts)} round(s): {rounds_s}")
    print("----")
    print(f"{fails} failing rows" + (f", {unresolved} inconclusive (the machine could not resolve the "
                                     f"margin — rerun it idle)" if unresolved else ""))
    if args.dump:
        with open(args.dump, "w") as f:
            json.dump(dump, f)
        print(f"raw latencies: {args.dump}")
    return 1 if fails else (3 if unresolved else 0)


if __name__ == "__main__":
    sys.exit(main())
