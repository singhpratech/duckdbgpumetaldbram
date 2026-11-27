#!/usr/bin/env python3
"""release_gate.py — rule "never different" for the Python wrapper, over
statements nobody wrote by hand.

It generates statements over deliberately hostile data (NULLs, duplicates,
each type's extreme values, infinities, NaN, -0.0, empty and case- or
accent-variant strings, values that collide after rounding, skewed and
uniform keys from 1 to thousands of groups, unique and non-unique join
dimensions, a view and a CTE), runs every one through gpudb and through
native DuckDB in several modes, and fails if any result differs or any valid
statement raises. See docs/DEVELOPMENT.md, "The release gate".

  python3 scripts/release_gate.py [--quick | --full] [--seed 1] [--modes eager,lazy,...]
                                  [--baseline scripts/release_gate_baseline.json]
                                  [--write-baseline FILE] [--out results.jsonl]
                                  [--replay ID | --only SUBSTRING] [--wrapper DIR] [--cpu-reference]

Exit status: 0 when nothing differs and nothing raised (or, with --baseline,
when every failure is a listed known-open one AND every listed one that ran
still fails); 1 otherwise; 2 when the gate could not run.

Needs the extension: a local build (build-macos / build-linux) or
GPUDB_EXTENSION_PATH. `--wrapper` points at another checkout's `python/`
directory (an older release, a fix branch) to gate that wrapper instead.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import platform
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MODES = ["eager", "lazy", "background", "write-same", "write-sibling", "write-raw", "settings", "warm", "cursor",
         "shipped"]
# Statement counts per mode. `None` = every generated statement of the pool.
BUDGETS = {
    "quick": dict(random=0, literal=600, lazy=250, background=60, write=60, settings_per=1, warm=150, cursor=60,
                  shipped=16, big_rows=1_100_000, settings_relevant=True),
    "full": dict(random=1500, literal=None, lazy=None, background=500, write=300, settings_per=12, warm=800,
                 cursor=300, shipped=40, big_rows=1_100_000),
}


def build_tasks(args, stmts_mod, modes_mod):
    """Every (mode, protocol, statement, parameters) the run executes. A pure
    function of the seed, the budget and the mode list (never of results), so
    a replay regenerates exactly the task it names."""
    b = BUDGETS[args.budget]
    pair = stmts_mod.generate_pairwise(args.seed)
    rnd = stmts_mod.generate_random(args.seed, b["random"]) if b["random"] else []
    pool = pair + rnd + stmts_mod.generate_key_predicates(args.seed)

    def sample(n, src=pool, salt=""):
        """The first n of `src` in an order fixed by each statement's own text:
        a smaller budget's sample is a subset of a larger one's, so one
        baseline written from a --full run serves a --quick run too."""
        if n is None or n >= len(src):
            return list(src)
        return sorted(src, key=lambda st: hashlib.sha1(f"{args.seed}|{salt}|{st.template}|{st.lits!r}"
                                                         .encode()).hexdigest())[:n]

    lit = sample(b["literal"], stmts_mod.generate_literal_matrix(args.seed), "literal")

    tasks = []
    want = set(args.modes)
    if "eager" in want:
        tasks += [("eager", "basic", st, None) for st in pool + lit]
    if "lazy" in want:
        tasks += [("lazy", "basic", st, None) for st in sample(b["lazy"], salt="lazy")]
    if "background" in want:
        tasks += [("background", "basic", st, None) for st in sample(b["background"], salt="background")]
    for ch in ("same", "sibling", "raw"):
        if f"write-{ch}" in want:
            tasks += [(f"write-{ch}", "write", st, (ch, j)) for j, st in enumerate(sample(b["write"], salt="write"))]
    if "settings" in want:
        # prefer statements whose answer a setting can move: an ORDER BY without a direction,
        # a VARCHAR key or comparison, a division, a TIMESTAMPTZ
        sens = [s for s in pool if any(x in s.template for x in (" ORDER BY ", "c_s", "/ 2", "c_tz", "c_fl", "c_db"))]
        own = stmts_mod.generate_settings(args.seed)
        # what each setting can move; a --quick run keeps only those statements of the family
        moves = {"default_collation": ("c_s", "d.name"), "default_order": ("ORDER BY",), "default_null_order": ("k7",),
                 "integer_division": ("/",), "TimeZone": ("c_tz",), "ieee_floating_point_ops": ("/ (w - w)",)}
        for name, value in modes_mod.SETTINGS:
            fam = [st for st in own if not b.get("settings_relevant")
                   or any(m in st.template for m in moves.get(name, ("",)))]
            for ch in modes_mod.CHANNELS:
                for st in fam + sample(b["settings_per"], sens, f"{name}:{ch}"):
                    tasks.append((f"settings:{name}:{ch}", "setting", st, (name, value, ch)))
    if "warm" in want:
        tasks += [("warm", "warm", st, None) for st in sample(b["warm"], salt="warm")]
    if "cursor" in want:
        tasks += [("cursor", "cursor", st, None) for st in sample(b["cursor"], salt="cursor")]
    if "shipped" in want:
        tasks += [("shipped", "basic", st, None) for st in sample(b["shipped"], stmts_mod.generate_shipped(args.seed),
                                                                 "shipped")]
    return tasks


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--quick", dest="budget", action="store_const", const="quick")
    g.add_argument("--full", dest="budget", action="store_const", const="full")
    ap.set_defaults(budget="full")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rows", type=int, default=5000, help="rows of the small fact table")
    ap.add_argument("--big-rows", type=int, default=None, help="rows of the shipped-defaults table (>= 1M)")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--write-baseline", default=None)
    ap.add_argument("--out", default=None, help="JSON lines: one record per statement and mode")
    ap.add_argument("--replay", default=None, help="run only the statement with this id (as printed)")
    ap.add_argument("--only", default=None, help="run only statements whose SQL contains this text")
    ap.add_argument("--wrapper", default=os.path.join(HERE, "..", "python"))
    ap.add_argument("--cpu-reference", action="store_true",
                    help="on a CPU-only build, let the transparent path lower onto the CPU reference operators")
    ap.add_argument("--list", action="store_true", help="print the task counts and exit")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--max-fail", type=int, default=25)
    args = ap.parse_args(argv)
    args.modes = [m for m in args.modes.split(",") if m]
    bad = [m for m in args.modes if m not in MODES]
    if bad:
        ap.error(f"unknown modes {bad}; known: {MODES}")

    sys.path.insert(0, os.path.abspath(args.wrapper))
    import duckdb
    import gpudb
    from release_gate_lib import data, modes as M, report as R, stmts as S, runner

    t_gen = time.time()
    tasks = build_tasks(args, S, M)
    if args.replay:
        tasks = [t for t in tasks if t[2].sid(t[0]) == args.replay]
    if args.only:
        tasks = [t for t in tasks if args.only in t[2].sql(0) or args.only in t[2].sql(1)]
    gen_s = time.time() - t_gen
    counts = collections.Counter(t[0].split(":")[0] for t in tasks)
    if args.list or not tasks:
        print(f"{len(tasks)} tasks ({gen_s:.1f}s to generate): {dict(counts)}")
        return 0 if tasks or args.list else 2

    big_rows = args.big_rows or BUDGETS[args.budget]["big_rows"]
    base_setup = data.setup_sql(args.seed, args.rows)
    header = (f"== release gate: gpudb {getattr(gpudb, '__version__', '?')} ({os.path.dirname(gpudb.__file__)}) | "
              f"DuckDB {duckdb.__version__} | {platform.system()} {platform.machine()} | seed {args.seed} | "
              f"budget {args.budget} | {args.rows} + {big_rows} rows")
    print(header, flush=True)
    print(f"   {len(tasks)} tasks ({gen_s:.1f}s to generate): {dict(counts)}", flush=True)

    results = []
    elapsed = collections.Counter()
    engines = {}
    backend = "?"
    probe = M.Engine(gpudb, M.TEST_KW, [], args.cpu_reference)
    backend = probe.backend
    probe.close()
    if backend in ("", None):
        print("the gpudb extension is not loaded (set GPUDB_EXTENSION_PATH or build it); nothing to gate")
        return 2
    if backend == "CPU":
        print("backend is CPU: every statement declines with reason 'backend' and the gate proves only that "
              "declining is exact; pass --cpu-reference to lower onto the CPU reference operators")

    def engine(mode):
        group = mode.split(":")[0]
        key = group if not group.startswith("settings") else "settings"
        if key not in engines:
            if key == "shipped":
                if args.cpu_reference:
                    return None
                engines[key] = M.Engine(gpudb, {}, data.big_setup_sql(args.seed, big_rows), args.cpu_reference)
            elif key == "background":
                engines[key] = M.Engine(gpudb, M.BG_KW, base_setup, args.cpu_reference)
            elif key.startswith("write"):
                engines[key] = M.Engine(gpudb, M.TEST_KW, base_setup + data.write_setup_sql(), args.cpu_reference)
            else:
                engines[key] = M.Engine(gpudb, M.TEST_KW, base_setup, args.cpu_reference)
        return engines[key]

    protos = {}

    def proto(mode):
        group = mode.split(":")[0]
        if mode not in protos:
            api = "sql" if group == "lazy" else "execute"
            # eager uploads on sight, but a set dropped by a write or a setting reports
            # not_resident for the execution that schedules its rebuild
            wait = {"background": 6.0, "shipped": 20.0, "settings": 0.3, "write-same": 0.3, "write-sibling": 0.3,
                    "write-raw": 0.3}.get(group, 0.0)
            tmap = {"ft": "fw", "vf": "vfw"} if group.startswith("write") else None
            protos[mode] = M.Protocol(mode, api, wait, tmap, nondet_check=not group.startswith(("write", "settings")),
                                      verbose=args.verbose)
        return protos[mode]

    out_f = open(args.out, "w") if args.out else None
    last_print = time.time()
    done = collections.Counter()
    for j, (mode, kind, st, par) in enumerate(tasks):
        group = mode.split(":")[0]
        eng = engine(mode)
        if eng is None:
            r = M.StmtResult(st.sid(mode), mode, st, outcome=runner.SKIP, note="shipped mode needs a device")
            results.append(r)
            continue
        p = proto(mode)
        t0 = time.time()
        try:
            if kind == "basic":
                r = p.basic(eng, st)
            elif kind in ("warm", "cursor"):
                r = p.basic(eng, st, how=kind)
            elif kind == "write":
                r = p.write(eng, st, par[0], par[1])
            else:
                r = p.setting(eng, st, *par)
        except Exception as e:      # noqa: BLE001 — a harness failure must not pass as a gate pass
            r = M.StmtResult(st.sid(mode), mode, st, outcome=runner.RAISED,
                             note=f"gate harness: {type(e).__name__}: {e}")
            r.runs.append({"label": "harness", "outcome": runner.RAISED, "rewritten": False, "fallback": False,
                           "reason": "", "form": "", "engine": "", "sql": st.sql(0), "why": r.note,
                           "native": "", "gpudb": "", "try": 0})
        elapsed[group] += time.time() - t0
        results.append(r)
        done[group] += 1
        if out_f:
            out_f.write(json.dumps({"sid": r.sid, "mode": r.mode, "family": st.family, "outcome": r.outcome,
                                    "device": r.device, "reasons": r.reasons, "note": r.note, "tags": st.tags,
                                    "sql": st.sql(0), "sql_rerun": st.sql(1),
                                    "runs": r.runs if r.outcome in R.FAIL else None}, default=str) + "\n")
        if time.time() - last_print > 30:
            last_print = time.time()
            nf = sum(1 for x in results if x.outcome in R.FAIL)
            print(f"   ... {j + 1}/{len(tasks)} ({dict(done)}), {nf} failing so far", flush=True)
    for e in engines.values():
        e.close()
    if out_f:
        out_f.close()

    header2 = header + f" | backend {backend}"
    summ = R.summarize(results, MODES)
    R.print_report(results, summ, header2, elapsed)
    fails = [r for r in results if r.outcome in R.FAIL]
    ver = duckdb.__version__
    rc = 0
    if args.baseline:
        base = R.load_baseline(args.baseline)
        new, known, stale, stale_flaky, not_run = R.check_baseline(base, results, ver, backend)
        print(f"\n== baseline {args.baseline}: {len(new)} NEW, {len(known)} known-open, {len(stale)} STALE "
              f"(listed but now passing), {len(stale_flaky)} listed-flaky now passing, {len(not_run)} listed but not run")
        areas = base.get("areas", {})
        ka = collections.Counter(base["entries"][r.sid]["area"] for r in known)
        for a, n in ka.most_common():
            print(f"   known-open {a}: {n} — fix: {areas.get(a, {}).get('fix', '?')}")
        for s, e in stale[:20]:
            print(f"   STALE [{s}] {e['area']} {e['mode']}: {e['sql'][:150]}")
        if new:
            R.print_failures(new, max_full=args.max_fail)
        rc = 1 if (new or stale) else 0
    else:
        if fails:
            R.print_failures(fails, max_full=args.max_fail)
        rc = 1 if fails else 0
    if args.replay:
        for r in results:
            None if r.outcome in R.FAIL else print(f"  [{r.sid}] {r.mode}: {r.outcome}; runs: " +
                                                            "; ".join(f"{x['label']}={x['outcome']}/"
                                                                      f"{'device' if x['rewritten'] else x['reason']}"
                                                                      for x in r.runs))
            print(f"     first : {r.st.sql(0)}\n     rerun : {r.st.sql(1)}")
    if args.write_baseline:
        base = R.load_baseline(args.write_baseline)
        R.write_baseline(args.write_baseline, base, results, ver, backend,
                         {"seed": args.seed, "rows": args.rows, "budget": args.budget})
        print(f"baseline written: {args.write_baseline} ({len(base['entries'])} entries)")
    print(f"\nexit {rc}: {len(fails)} DIFFERENT/RAISED statements" + (" (see baseline above)" if args.baseline else ""))
    return rc


if __name__ == "__main__":
    sys.exit(main())
