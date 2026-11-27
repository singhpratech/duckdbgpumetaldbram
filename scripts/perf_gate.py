#!/usr/bin/env python3
"""perf_gate.py — the speed and device-coverage half of the release gate.

Compares a CANDIDATE gpudb Python wrapper (and, optionally, a candidate
extension binary) with a BASELINE release, statement by statement, against
raw DuckDB on the same statements. The correctness half — generated
statements against native — is scripts/release_gate.py; this script compares
result digests only to say that two sides timed the same work.

Every side is a process of its own (two gpudb packages are never imported
into one process), and every ROLE runs as two replicate processes: `native`
and `native#2` are raw DuckDB with no gpudb at all, `baseline` and
`baseline#2` the baseline wrapper (the A/A pair the noise floor is derived
from), `<candidate>` and `<candidate>#2` each candidate. Per statement every
side warms up, then the sides take turns, one run each per round, in a
carryover-balanced order (stats.williams), for at least N rounds. Each
side's times are taken inside its own process around the statement alone
(rows fetched). A difference counts only when it holds for both candidate
processes against both baseline processes.

Verdicts per statement and candidate (docs/DEVELOPMENT.md, "The performance
gate"): SLOWER_THAN_NATIVE (on the device and slower than DuckDB alone),
SLOWER_THAN_BASELINE, COVERAGE_LOST (the baseline ran it on the device, the
candidate does not) and CANDIDATE_ERROR fail the run (exit 1) unless the
loss is recorded with --accept. BASELINE_SLOWER_THAN_NATIVE (the same rule
breach, already in the baseline) fails it under --absolute, which a release
run uses. FIRST_EXECUTION, TIME_TO_RESIDENT, MEMORY
and DURING_UPLOAD regressions are reported; ROUTE_CHANGED, COVERAGE_GAINED,
COVERAGE_UNSTABLE, ROWS_DIFFER, BIMODAL and NOISY are informational.

Usage:
  python -I scripts/perf_gate.py --baseline <python/ tree | installed> --candidate <python/ tree> [...]
                                 [--ext FILE | --baseline-ext FILE --candidate-ext FILE]
                                 [--quick] [--n 7] [--warm 5] [--only 'tpch.sf1.*']
                                 [--opentoid-db FILE] [--allow-noisy [--no-lock]]
  python -I scripts/perf_gate.py --accept <statement id or glob> --verdict COVERAGE_LOST \\
                                 --reason "why the loss is deliberate" [--for-candidate NAME]

The DuckDB measured is the one the interpreter running this script imports.
Exit: 0 clean, 1 an unaccepted breach, 2 a setup error, 3 the run is INVALID
(the machine was not quiet, or the benchmark lock is held by someone else).
"""
from __future__ import annotations

import argparse
import atexit
import fnmatch
import json
import math
import os
import platform
import random
import shutil
import signal
import sys
import time
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from perf_gate_lib import quiet, report, stats, verdicts, workload  # noqa: E402
from perf_gate_lib.sides import Worker, WorkerError  # noqa: E402

def default_lock() -> str:
    """The machine's benchmark lock: GPUDB_BENCH_LOCK, else `.bench.lock` in
    the directory that holds the main checkout (a worktree resolves to its
    main checkout, so every worktree on a machine shares the one lock)."""
    env = os.environ.get("GPUDB_BENCH_LOCK")
    if env:
        return env
    try:
        import subprocess
        common = subprocess.run(["git", "-C", REPO, "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                capture_output=True, text=True, timeout=10).stdout.strip()
        if common:
            return os.path.join(os.path.dirname(os.path.dirname(common)), ".bench.lock")
    except Exception:
        pass
    return os.path.join(os.path.dirname(REPO), ".bench.lock")
ACCEPTED_FILE = os.path.join(HERE, "perf_gate_accepted.json")


def label_for(path: str) -> str:
    if path == "installed":
        return "installed"
    p = os.path.abspath(path).rstrip("/")
    return os.path.basename(os.path.dirname(p)) if os.path.basename(p) == "python" else os.path.basename(p)


def say(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---- the trees under test ----
def tree_hash(tree: str) -> str:
    """Content hash of a python/ tree's gpudb package (every file, sorted by path)."""
    import hashlib
    h = hashlib.sha256()
    root = os.path.join(tree, "gpudb")
    for dp, dns, fns in os.walk(root):
        dns[:] = sorted(d for d in dns if d != "__pycache__")
        for fn in sorted(fns):
            if fn.endswith((".pyc", ".pyo")):
                continue
            p = os.path.join(dp, fn)
            h.update(os.path.relpath(p, root).encode() + b"\0")
            with open(p, "rb") as f:
                h.update(f.read())
    return h.hexdigest()[:16]


def snapshot(tree: str, dest: str) -> str:
    """Copy a python/ tree's gpudb package into the run directory, so every
    process of the run imports the same code even while the tree is being
    edited (a candidate worktree can hold uncommitted work in progress)."""
    if tree == "installed":
        return tree
    out = os.path.join(dest, "python")
    shutil.copytree(os.path.join(tree, "gpudb"), os.path.join(out, "gpudb"),
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return out


# ---- data ------------------------------------------------------------------------------------
def tpch_texts() -> List:
    import duckdb
    c = duckdb.connect()
    try:
        c.execute("LOAD tpch")
    except Exception:
        c.execute("INSTALL tpch")
        c.execute("LOAD tpch")
    rows = c.execute("SELECT query_nr, query FROM tpch_queries() ORDER BY 1").fetchall()
    c.close()
    return [(int(n), q) for n, q in rows]


def ensure_tpch(sf: int, tpch_dir: str, work_dir: str, generate: bool) -> str:
    p = os.path.join(tpch_dir, f"tpch_sf{sf}", "tpch.duckdb")
    if os.path.exists(p):
        return p
    q = os.path.join(work_dir, f"tpch_sf{sf}.duckdb")
    if os.path.exists(q):
        return q
    if not generate:
        raise SystemExit(f"missing {p} (SF={sf} ./scripts/gen_tpch.sh), or pass --generate-tpch")
    import duckdb
    say(f"generating TPC-H SF{sf} into {q}")
    tmp = q + ".tmp"
    if os.path.exists(tmp):
        os.unlink(tmp)
    c = duckdb.connect(tmp)
    c.execute("INSTALL tpch")
    c.execute("LOAD tpch")
    c.execute(f"CALL dbgen(sf={sf})")
    c.close()
    os.rename(tmp, q)
    return q


def ensure_micro(work_dir: str, tables: List[str]) -> (str, Dict[str, str]):
    """The generated tables, built once per workload version and DuckDB version."""
    import duckdb
    path = os.path.join(work_dir, f"micro_{workload.WORKLOAD_VERSION}_duckdb{duckdb.__version__}.duckdb")
    c = duckdb.connect(path)
    c.execute("CREATE TABLE IF NOT EXISTS perf_gate_meta(tbl VARCHAR, key VARCHAR, value VARCHAR)")
    have = {r[0] for r in c.execute("SELECT DISTINCT tbl FROM perf_gate_meta").fetchall()}
    for t in tables:
        if t in have:
            continue
        say(f"building micro table {t} ({workload.MICRO_SIZES[t]:,} rows) in {path}")
        t0 = time.time()
        for s in (f"DROP VIEW IF EXISTS v_{t}", f"DROP TABLE IF EXISTS d_{t}", f"DROP TABLE IF EXISTS {t}"):
            c.execute(s)
        c.execute(workload.MICRO_TABLE_SQL.format(t=t, rows=workload.MICRO_SIZES[t]))
        c.execute(workload.MICRO_DIM_SQL.format(t=t))
        c.execute(workload.MICRO_VIEW_SQL.format(t=t))
        c.execute(workload.MICRO_META_SQL.format(t=t))
        c.execute("CHECKPOINT")
        say(f"  {time.time() - t0:.1f} s")
    having = dict(c.execute("SELECT tbl, value FROM perf_gate_meta WHERE key = 'having_200k'").fetchall())
    c.close()
    return path, having


# ---- measurement -----------------------------------------------------------------------------
class Run:
    def __init__(self, args, sides: Dict[str, Dict], out_dir: str):
        self.args = args
        self.sides = sides            # name -> {"wrapper":..., "ext":...}
        self.out_dir = out_dir
        self.sampler = quiet.CpuSampler()
        self.records: List[Dict] = []
        self.side_info: Dict[str, Dict] = {n: dict(s) for n, s in sides.items()}
        self.seq = 0
        self.roles: Dict[str, List[str]] = {}
        for n in sides:
            self.roles.setdefault(n.split("#")[0], []).append(n)
        # fresh-process figures: both processes of every wrapper role (the same
        # replicate rule as the statements), one of native
        self.fresh_sides = [n for n in sides if n != "native#2"]

    def spawn(self, name: str, tag: str) -> Worker:
        s = self.sides[name]
        safe = name.replace("#", "_")
        return Worker(name, s["wrapper"], s["ext"], os.path.join(self.out_dir, "logs", f"{tag}.{safe}.log"),
                      os.path.join(self.args.work_dir, "pycache"))

    def open_ctx(self, w: Worker, g: Dict, residency: str = None) -> Dict:
        db = g["db"].replace("{side}", w.side.replace("#", "_") + f"-{os.getpid()}-{self.seq}")
        if "{side}" in g["db"]:
            for ext in ("", ".wal"):
                if os.path.exists(db + ext):
                    os.unlink(db + ext)
        return w.call(op="open", ctx=g["name"], db=db, read_only=g["read_only"],
                      residency=residency or g["residency"], memory_budget=g["memory_budget"],
                      floor_rows=g["floor_rows"], setup=g["setup"], threads=self.args.threads)

    def measure(self, workers: Dict[str, Worker], g: Dict, st: Dict) -> Dict:
        a = self.args
        names = list(workers)
        rec = {"id": st["id"], "group": g["name"], "kind": st["kind"], "path": st["path"], "sql": st["sql"],
               "sides": {n: {"times": [], "routes": []} for n in names}}
        self.sampler.sample()
        load0 = quiet.load_average()
        base_n = st.get("n_offset", 0)
        for n in names:
            try:
                r = workers[n].call(op="run", ctx=g["name"], sql=st["sql"], path=st["path"], reps=a.warm,
                                    vary=st["vary"], n0=base_n, digest=True)
                rec["sides"][n].update({"warm": r["times"], "warm_routes": r["routes"],
                                        "digest": r["digest"], "rows": r["rows"]})
            except WorkerError as e:
                rec["sides"][n]["error"] = str(e)
        live = [n for n in names if "error" not in rec["sides"][n]]
        counter = base_n + a.warm

        square = stats.williams(len(live))

        def rounds(k: int, counter: int) -> int:
            # whole squares only: every side runs in every position, and right
            # after every other side, equally often (see stats.williams)
            k = -(-k // len(square)) * len(square)
            for i in range(k):
                for n in [live[j] for j in square[i % len(square)]]:
                    if "error" in rec["sides"][n]:
                        continue
                    try:
                        r = workers[n].call(op="run", ctx=g["name"], sql=st["sql"], path=st["path"], reps=1,
                                            vary=st["vary"], n0=counter)
                        rec["sides"][n]["times"] += r["times"]
                        rec["sides"][n]["routes"] += r["routes"]
                    except WorkerError as e:
                        rec["sides"][n]["error"] = str(e)
                counter += 1
            return counter

        counter = rounds(a.n, counter)
        if self.suspect(rec):
            rec["remeasured"] = True
            rounds(a.n, counter)
        for n in names:
            s = rec["sides"][n]
            s["route"] = verdicts.summarize_routes(s["routes"]) if n != "native" else None
        cpu = self.sampler.sample()
        rec["others_share"] = cpu["others_share"]
        rec["load1"] = max(load0, quiet.load_average())
        rec["noisy"] = cpu["others_share"] > a.max_share
        if rec["noisy"]:
            rec["top"] = f"{quiet.top_command(cpu['top_pid'])} ({cpu['top_cores']:.1f} cores)"
        return rec

    def suspect(self, rec: Dict) -> bool:
        """Before the verdicts can be computed (they need the whole run's A/A
        floor), decide whether a statement deserves a second set of rounds: a
        candidate that looks slower than the baseline, or slower than native
        while on the device, by more than 5% or its own spread."""
        s = rec["sides"]

        def cents(role):
            return [stats.central(s[x]["times"]) for x in self.roles[role] if s.get(x, {}).get("times")]
        b, nat = cents("baseline"), cents("native")
        if not b:
            return False
        pool_b = sum((s[x]["times"] for x in self.roles["baseline"] if s.get(x, {}).get("times")), [])
        for role in self.roles:
            if role in ("native", "baseline"):
                continue
            c = cents(role)
            if not c:
                continue
            pool_c = sum((s[x]["times"] for x in self.roles[role] if s.get(x, {}).get("times")), [])
            tol = max(0.05, 3 * 1.2533 * math.sqrt(stats.rel_sigma(pool_b) ** 2 + stats.rel_sigma(pool_c) ** 2)
                      / math.sqrt(len(pool_c)))
            if min(c) > max(b) * math.exp(tol):
                return True
            if nat and min(c) > max(nat) * math.exp(tol):
                return True
        return False

    def run_group(self, g: Dict) -> None:
        sts = g["statements"]
        if not sts:
            return
        say(f"[{g['name']}] {len(sts)} statements, sides {', '.join(self.sides)}")
        workers = {}
        try:
            for n in self.sides:
                workers[n] = self.spawn(n, g["name"])
                self.side_info[n].update({k: v for k, v in workers[n].info.items() if k in ("gpudb", "duckdb")})
            for n, w in workers.items():
                o = self.open_ctx(w, g)
                if n != "native" and o.get("build_info"):
                    self.side_info[n]["build_info"] = o["build_info"]
                    if o.get("extension_note"):
                        self.side_info[n]["extension_note"] = o["extension_note"]
            for i, st in enumerate(sts):
                t0 = time.time()
                rec = self.measure(workers, g, st)
                self.records.append(rec)
                b = rec["sides"].get("baseline", {})
                line = f"  {st['id']:<44} " + " ".join(
                    f"{n}={stats.median(s['times']):.2f}" if s.get("times") else f"{n}=ERR"
                    for n, s in rec["sides"].items())
                if b.get("route"):
                    line += f"  [{verdicts.route_label(b['route'])}]"
                line += f"  ({time.time() - t0:.1f} s{', noisy' if rec['noisy'] else ''}"
                line += ", re-measured)" if rec.get("remeasured") else ")"
                say(line)
            for n, w in workers.items():
                if n == "native":
                    continue
                try:
                    m = w.call(op="memory", ctx=g["name"])
                    self.side_info[n].setdefault("memory", {})[g["name"]] = m
                except WorkerError:
                    pass
        finally:
            for w in workers.values():
                w.quit()

    def confirm_coverage(self, groups: List[Dict]) -> None:
        """A coverage loss is re-checked before it can fail the run: the
        statement runs again (warm-ups + N) in a fresh pair of processes of
        the baseline and of the candidate, and only the routes are kept. Some
        decisions are taken once per process and do not come out the same in
        every process (TPC-H SF10 Q11 declined in 1 of 2, 2 of 2 and 1 of 2
        processes of three different candidates while 6 of 6 baseline
        processes ran it on the device), so two processes agreeing once is
        not yet a loss."""
        by_id = {st["id"]: (g, st) for g in groups for st in g["statements"]}
        for rec in self.records:
            S = rec["sides"]
            if rec["id"] not in by_id:
                continue
            rb = [S[x].get("route") for x in self.roles["baseline"]]
            if not all(r and r["device"] and r["frac"] == 1.0 for r in rb):
                continue
            for role, names in self.roles.items():
                if role in ("native", "baseline"):
                    continue
                rc = [S[x].get("route") for x in names]
                if not all(r and not r["device"] for r in rc):
                    continue
                g, st = by_id[rec["id"]]
                say(f"[confirm coverage] {rec['id']} for {role}")
                conf = rec.setdefault("confirm", {})
                for n in self.roles["baseline"] + names:
                    if n in conf:
                        continue
                    self.seq += 1
                    try:
                        w = self.spawn(n, "confirm")
                    except WorkerError as e:
                        say(f"  {n}: {e}")
                        continue
                    try:
                        self.open_ctx(w, g)
                        r = w.call(op="run", ctx=g["name"], sql=st["sql"], path=st["path"],
                                   reps=self.args.warm + self.args.n, vary=st["vary"], n0=st.get("n_offset", 0))
                        conf[n] = verdicts.summarize_routes(r["routes"][self.args.warm:])
                    except WorkerError as e:
                        say(f"  {n}: {e}")
                    finally:
                        w.quit()

    # ---- fresh processes ----
    def fresh(self, groups: List[Dict], probe_ids: Dict[str, List[str]], reps: int) -> Dict:
        by_id = {st["id"]: (g, st) for g in groups for st in g["statements"]}
        out: Dict[str, Dict[str, Dict[str, List]]] = {}
        names = self.fresh_sides
        for kind in ("first", "ttr"):
            for sid in probe_ids[kind]:
                if sid not in by_id:
                    continue
                g, st = by_id[sid]
                say(f"[fresh {kind}] {sid}")
                for r in range(reps):
                    order = names[r % len(names):] + names[:r % len(names)]
                    for n in order:
                        if kind == "ttr" and n == "native":
                            continue
                        self.seq += 1
                        try:
                            w = self.spawn(n, f"fresh-{kind}")
                        except WorkerError as e:
                            say(f"  {n}: {e}")
                            continue
                        try:
                            o = self.open_ctx(w, g, residency="background" if kind == "ttr" else None)
                            per = out.setdefault(sid, {})
                            if kind == "first":
                                res = w.call(op="run", ctx=g["name"], sql=st["sql"], path=st["path"], reps=1,
                                             vary=st["vary"], n0=st.get("n_offset", 0))
                                mem = w.call(op="memory", ctx=g["name"])
                                per.setdefault("connect ms", {}).setdefault(n, []).append(o["connect_ms"])
                                per.setdefault("first execution ms", {}).setdefault(n, []).append(res["times"][0])
                                per.setdefault("peak host MiB", {}).setdefault(n, []).append(mem["peak_rss"] / 2**20)
                                if mem.get("resident_bytes") is not None:
                                    per.setdefault("resident MiB", {}).setdefault(n, []).append(
                                        mem["resident_bytes"] / 2**20)
                                if mem.get("device_allocated") is not None:
                                    per.setdefault("device allocated MiB", {}).setdefault(n, []).append(
                                        mem["device_allocated"] / 2**20)
                                rt = res["routes"][0]
                                per.setdefault("first route", {}).setdefault(n, []).append(
                                    None if rt is None else (rt["form"] if rt["rewritten"] else rt["reason"]))
                            else:
                                res = w.call(op="ttr", ctx=g["name"], sql=st["sql"], path=st["path"],
                                             max_s=self.args.ttr_max_s)
                                per.setdefault("time to resident s", {}).setdefault(n, []).append(res["ready_s"])
                                per.setdefault("first sighting ms (background)", {}).setdefault(n, []).append(
                                    res["first_ms"])
                                ra = res["route_after"]
                                per.setdefault("route once resident", {}).setdefault(n, []).append(
                                    None if ra is None else (ra["form"] if ra["rewritten"] else ra["reason"]))
                        except WorkerError as e:
                            say(f"  {n}: {e}")
                        finally:
                            w.quit()
        return out

    def during_upload(self, g: Dict, reps: int) -> Dict:
        """Statements at an interactive cadence while a background upload runs."""
        rnd = random.Random(7)
        gaps = [rnd.uniform(0, 50) for _ in range(self.args.bg_statements)]
        out = {}
        names = self.fresh_sides
        say(f"[during upload] {g['name']}: {len(gaps)} statements at 0-50 ms gaps")
        for r in range(reps):
            for n in names[r % len(names):] + names[:r % len(names)]:
                self.seq += 1
                w = self.spawn(n, "during-upload")
                try:
                    self.open_ctx(w, g, residency="background")
                    res = w.call(op="during_bg", ctx=g["name"], trigger=workload.BG_TRIGGER,
                                 sql=workload.BG_TARGET, gaps_ms=gaps)
                    out.setdefault(n, []).append(res["samples"])
                except WorkerError as e:
                    say(f"  {n}: {e}")
                finally:
                    w.quit()
        return out


# ---- analysis --------------------------------------------------------------------------------
def analyse(run: Run, cands: List[str], accepted: List[Dict], fresh: Dict, during: Dict) -> Dict:
    recs = run.records
    pairs = []
    native_pairs = []
    for rec in recs:
        s = rec["sides"]
        b, a = s.get("baseline", {}), s.get("baseline#2", {})
        # a statement whose route differs between the two processes, or flips
        # inside one, measures the wrapper's routing, not the machine's noise
        steady = all(x.get("route") and x["route"]["frac"] in (0.0, 1.0) for x in (b, a)) and \
            b["route"]["device"] == a["route"]["device"]
        if b.get("times") and a.get("times") and steady:
            pairs.append({"base": stats.central(b["times"]), "aa": stats.central(a["times"])})
        b, a = s.get("native", {}), s.get("native#2", {})
        if b.get("times") and a.get("times"):
            native_pairs.append({"base": stats.central(b["times"]), "aa": stats.central(a["times"])})
    floor = stats.noise_floor(pairs)
    # The A/A pair judged the way a single-process gate would judge it (one
    # baseline process against one more): how many false alarms the replicate
    # rule is there to prevent. Informational.
    single = {"native": ["native"], "baseline": ["baseline"], "baseline#2": ["baseline#2"]}
    aa_breaches = []
    for rec in recs:
        vs = verdicts.judge_statement(dict(rec, judged={}), "baseline#2", floor, single)
        if any(v["verdict"] in ("SLOWER_THAN_BASELINE", "COVERAGE_LOST") for v in vs):
            aa_breaches.append(rec["id"])
    out = {}
    for c in cands:
        rows, counts = [], {}
        for rec in recs:
            vs = verdicts.judge_statement(rec, c, floor, run.roles)
            for v in vs:
                acc = verdicts.accepted_for(accepted, rec["id"], v["verdict"], c)
                if acc and v["verdict"] in verdicts.BREACHES + verdicts.BASELINE_BREACHES + verdicts.WARNINGS:
                    v["accepted"] = acc
                    acc["matched"] = True
                key = v["verdict"] + (" (accepted)" if v.get("accepted") else "")
                counts[key] = counts.get(key, 0) + 1
            s = rec["sides"]
            j = rec.get("judged", {}).get(c, {})

            def pooled(role):
                return sum((s.get(x, {}).get("times", []) for x in run.roles[role]), [])
            rows.append({"id": rec["id"], "verdicts": vs,
                         "native": pooled("native"), "baseline": pooled("baseline"), "candidate": pooled(c),
                         "ratio": f"{math.exp(j['log_ratio']):.3f}x" if "log_ratio" in j else "—"})
        # fresh-process figures
        for sid, measures in fresh.items():
            fv = []
            for meas, kind, min_tol, unit in (("first execution ms", "FIRST_EXECUTION", 0.25, "ms"),
                                              ("connect ms", "FIRST_EXECUTION", 0.25, "ms (connect)"),
                                              ("time to resident s", "TIME_TO_RESIDENT", 0.25, "s"),
                                              # peak RSS sits at one of two levels per process
                                              # whatever the code (+27 MiB on a 200 MiB peak in
                                              # 3 of 10 processes, one of them identical to the
                                              # baseline); the resident bytes are exact
                                              ("peak host MiB", "MEMORY", 0.25, "MiB peak host"),
                                              ("resident MiB", "MEMORY", 0.02, "MiB resident"),
                                              ("device allocated MiB", "MEMORY", 0.05, "MiB device")):
                if meas in measures:
                    v = verdicts.judge_fresh(kind, measures[meas], run.roles[c], run.roles["baseline"],
                                             min_tol, unit)
                    if v:
                        fv.append(v)
            for meas, what in (("first route", "first execution"), ("route once resident", "once resident")):
                rt = measures.get(meas, {})
                rb = {x for n in run.roles["baseline"] for x in rt.get(n, [])}
                rc = {x for n in run.roles[c] for x in rt.get(n, [])}
                if len(rb) == 1 and len(rc) == 1 and rb != rc:
                    fv.append({"verdict": "ROUTE_CHANGED", "note": f"{what}: {rb.pop()} -> {rc.pop()}"})
            if fv:
                for v in fv:
                    acc = verdicts.accepted_for(accepted, f"fresh.{sid}", v["verdict"], c)
                    if acc:
                        v["accepted"] = acc
                        acc["matched"] = True
                    counts[v["verdict"]] = counts.get(v["verdict"], 0) + 1
                rows.append({"id": f"fresh.{sid}", "verdicts": fv, "native": [], "baseline": [], "candidate": [],
                             "ratio": "—"})
        if during:
            def busy(side):
                return [x["ms"] for rnd in during.get(side, []) for x in rnd if x["busy"] or side == "native"]
            per_b = [busy(n) for n in run.roles["baseline"]]
            per_c = [busy(n) for n in run.roles[c]]
            if all(per_b) and all(per_c):
                mb, mc = [stats.median(x) for x in per_b], [stats.median(x) for x in per_c]
                tb, tc = sum(per_b, []), sum(per_c, [])
                tol = max(0.10, abs(math.log(max(mb) / min(mb))))
                lr = math.log(min(mc) / max(mb))
                if lr > tol and stats.mann_whitney_greater(tc, tb) < verdicts.P_MAX:
                    v = {"verdict": "DURING_UPLOAD",
                         "note": f"{' / '.join(f'{x:.2f}' for x in mc)} vs {' / '.join(f'{x:.2f}' for x in mb)} "
                                 f"ms per process while uploading ({len(tc)} / {len(tb)} statements, tolerance "
                                 f"{math.exp(tol):.2f}x)"}
                    counts["DURING_UPLOAD"] = counts.get("DURING_UPLOAD", 0) + 1
                    rows.append({"id": "during_upload", "verdicts": [v], "native": busy("native"),
                                 "baseline": tb, "candidate": tc, "ratio": f"{math.exp(lr):.3f}x"})
        out[c] = {"rows": rows, "counts": counts}
    return {"floor": floor, "native_floor": stats.noise_floor(native_pairs), "aa_breaches": aa_breaches,
            "candidates": out}


def finish(result: Dict, run, cands: List[str], accepted: List[Dict], absolute: bool, invalid: bool,
           allow_noisy: bool) -> (int, int):
    """Verdicts into `result`; returns (breaches, exit code)."""
    an = analyse(run, cands, accepted, result.get("fresh") or {}, result.get("during_upload") or {})
    failing = verdicts.BREACHES + (verdicts.BASELINE_BREACHES if absolute else ())
    breaches = sum(v for c in an["candidates"].values() for k, v in c["counts"].items() if k in failing)
    status = "INVALID (machine not quiet)" if invalid else ("BREACH" if breaches else "CLEAN")
    if allow_noisy:
        status += " — PROVISIONAL"
    result.update({"status": status, "noise_floor": an["floor"], "native_floor": an["native_floor"],
                   "aa_breaches": an["aa_breaches"], "candidates": an["candidates"], "accepted": accepted,
                   "absolute": absolute})
    return breaches, (3 if invalid else (1 if breaches else 0))


def rejudge(args) -> int:
    from types import SimpleNamespace
    with open(args.rejudge) as f:
        result = json.load(f)
    for rec in result["statements"]:
        rec.pop("judged", None)
    run = SimpleNamespace(records=result["statements"], roles=result["roles"])
    result["header"]["accepted_file"] = args.accepted_file
    result["header"]["rejudged"] = time.strftime("%Y-%m-%d %H:%M")
    accepted = verdicts.load_accepted(args.accepted_file)
    breaches, code = finish(result, run, list(result["candidates"]), accepted,
                            args.absolute or result.get("absolute", False), result.get("invalid", False),
                            result.get("provisional", False))
    print(report.render(result))
    say(f"{result['status']}: {breaches} breach(es) (re-judged {args.rejudge} against {args.accepted_file})")
    return code


# ---- main ------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--baseline", help="a python/ tree, or 'installed' (the gpudb this interpreter imports)")
    ap.add_argument("--candidate", action="append", default=[], help="a python/ tree; repeat to compare several")
    ap.add_argument("--ext", default="", help="extension binary for every side (shorthand for the two below)")
    ap.add_argument("--baseline-ext", default="")
    ap.add_argument("--candidate-ext", default="")
    ap.add_argument("--quick", action="store_true",
                    help="CI-sized: TPC-H SF1, the 2M-row microbenchmarks, the declined set; N=7, W=3, R=1")
    ap.add_argument("--sf", default=None, help="TPC-H scale factors, comma-separated (default 1,10; --quick: 1)")
    ap.add_argument("--micro", default=None, help="microbenchmark tables (default m2,m20; --quick: m2; 'none')")
    ap.add_argument("--opentoid-db", default="", help="the OS Open TOID database (opened read-only)")
    ap.add_argument("--only", default="", help="comma-separated id globs: measure only these statements")
    ap.add_argument("--n", type=int, default=7, help="interleaved rounds per statement (>= 7)")
    ap.add_argument("--warm", type=int, default=None, help="warm-up runs per side (default 5; --quick 3)")
    ap.add_argument("--fresh-reps", type=int, default=None, help="fresh-process repeats (default 3; --quick 1)")
    ap.add_argument("--no-fresh", action="store_true", help="skip first execution / time to resident / upload")
    ap.add_argument("--bg-statements", type=int, default=60)
    ap.add_argument("--ttr-max-s", type=float, default=120.0)
    ap.add_argument("--threads", type=int, default=0, help="SET threads on every side (default: DuckDB's own)")
    ap.add_argument("--tpch-dir", default=os.path.join(REPO, "data"))
    ap.add_argument("--generate-tpch", action="store_true", help="generate a missing TPC-H database into --work-dir")
    ap.add_argument("--work-dir", default=os.path.join(REPO, "data", "perf_gate"))
    ap.add_argument("--out", default="", help="result directory (default <work-dir>/runs/<timestamp>)")
    ap.add_argument("--max-load", type=float, default=max(2.0, 0.15 * quiet.CPUS))
    ap.add_argument("--max-share", type=float, default=0.10,
                    help="CPU share of the machine other processes may use (0..1)")
    ap.add_argument("--allow-noisy", action="store_true", help="run anyway; every figure is labelled provisional")
    ap.add_argument("--no-lock", action="store_true",
                    help="with --allow-noisy only: do not take the benchmark lock (provisional figures are not "
                         "worth stopping other work for); a lock that exists still refuses the run")
    ap.add_argument("--lock-file", default=None, help="default: GPUDB_BENCH_LOCK, else .bench.lock beside the "
                                                      "main checkout")
    ap.add_argument("--accepted-file", default=ACCEPTED_FILE)
    ap.add_argument("--accept", default="", help="record a deliberate loss for this statement id (or glob) and exit")
    ap.add_argument("--verdict", default="COVERAGE_LOST", help="with --accept: the verdict kind accepted")
    ap.add_argument("--reason", default="", help="with --accept: why the loss is deliberate (required)")
    ap.add_argument("--for-candidate", default="*", help="with --accept: candidate label (default any)")
    ap.add_argument("--absolute", action="store_true",
                    help="also fail on rule breaches the baseline already has (BASELINE_SLOWER_THAN_NATIVE): "
                         "a release run passes this")
    ap.add_argument("--render", default="", help="print the Markdown report of a saved result.json and exit")
    ap.add_argument("--rejudge", default="",
                    help="judge a saved result.json again (current verdict rules, current --accepted-file) "
                         "without measuring anything; prints the report and exits as the run would")
    args = ap.parse_args()

    if args.render:
        with open(args.render) as f:
            print(report.render(json.load(f)))
        return 0
    if args.rejudge:
        return rejudge(args)
    if args.accept:
        if not args.reason.strip():
            ap.error("--accept needs --reason")
        if args.verdict not in verdicts.BREACHES + verdicts.WARNINGS + ("*",):
            ap.error(f"--verdict must be one of {', '.join(verdicts.BREACHES + verdicts.WARNINGS)}")
        e = verdicts.record_accepted(args.accepted_file, args.accept, args.verdict, args.reason.strip(),
                                     args.for_candidate)
        print(f"recorded in {args.accepted_file}: {json.dumps(e)}")
        return 0
    if not args.baseline or not args.candidate:
        ap.error("--baseline and at least one --candidate are required")
    if args.n < 7:
        ap.error("--n must be at least 7")
    if args.no_lock and not args.allow_noisy:
        ap.error("--no-lock is only for provisional (--allow-noisy) runs")
    warm = args.warm if args.warm is not None else (3 if args.quick else 5)
    args.warm = warm
    fresh_reps = args.fresh_reps if args.fresh_reps is not None else (1 if args.quick else 3)
    sfs = [int(x) for x in (args.sf or ("1" if args.quick else "1,10")).split(",") if x]
    micro_tables = [] if args.micro == "none" else (args.micro or ("m2" if args.quick else "m2,m20")).split(",")
    for p in [args.baseline] + args.candidate:
        if p != "installed" and not os.path.isdir(os.path.join(p, "gpudb")):
            ap.error(f"{p}: not a python/ tree (no gpudb/ inside)")
    for e in (args.ext, args.baseline_ext, args.candidate_ext):
        if e and not os.path.exists(e):
            ap.error(f"{e}: no such extension binary")

    sides = {"native": {"wrapper": "none", "ext": ""}, "native#2": {"wrapper": "none", "ext": ""},
             "baseline": {"wrapper": args.baseline, "ext": args.baseline_ext or args.ext},
             "baseline#2": {"wrapper": args.baseline, "ext": args.baseline_ext or args.ext}}
    cands = []
    for p in args.candidate:
        name = label_for(p).replace("#", "-")
        while name in sides:
            name += "+"
        for suffix in ("", "#2"):
            sides[name + suffix] = {"wrapper": p, "ext": args.candidate_ext or args.ext}
        cands.append(name)
    for s in sides.values():
        if s["wrapper"] not in ("none", "installed"):
            s["source"] = os.path.abspath(s["wrapper"])
            s["tree_hash"] = tree_hash(s["wrapper"])

    # ---- the lock and the quiet check ----
    args.lock_file = args.lock_file or default_lock()
    lock = quiet.BenchLock(args.lock_file)
    if lock.exists():
        say(f"INVALID: the benchmark lock {args.lock_file} exists ({lock.holder() or 'no holder recorded'}); "
            f"another measurement is running. Not starting.")
        return 3
    say("quiet check (5 s) ...")
    pre = quiet.pre_check(args.max_load, args.max_share)
    say(f"  load {pre['load1']:.1f} (bound {args.max_load:.1f}), other processes {pre['others_share']:.0%} "
        f"(bound {args.max_share:.0%}), busiest {quiet.top_command(pre['top_pid'])} {pre['top_cores']:.1f} cores")
    if not pre["quiet"] and not args.allow_noisy:
        say("INVALID: the machine is not quiet. Stop the other work, or pass --allow-noisy for provisional figures.")
        return 3
    take_lock = not args.quick and not args.no_lock
    if take_lock:
        if not lock.acquire():
            say(f"INVALID: could not create {args.lock_file} (someone took it just now)")
            return 3
        atexit.register(lock.release)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, lambda *_: (lock.release(), sys.exit(130)))
        say(f"holding {args.lock_file} for the run")

    t_start = time.time()
    os.makedirs(args.work_dir, exist_ok=True)
    out_dir = args.out or os.path.join(args.work_dir, "runs", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    rw_dir = os.path.join(out_dir, "rw")
    os.makedirs(rw_dir, exist_ok=True)
    snaps = {}
    for n, s in sides.items():
        if s.get("source"):
            role = n.split("#")[0]
            if role not in snaps:
                snaps[role] = snapshot(s["source"], os.path.join(out_dir, "trees", role))
            s["wrapper"] = snaps[role]

    import duckdb
    texts = tpch_texts()
    tpch_dbs = {sf: ensure_tpch(sf, args.tpch_dir, args.work_dir, args.generate_tpch or args.quick) for sf in sfs}
    micro_db, having = (ensure_micro(args.work_dir, micro_tables) if micro_tables else (None, {}))
    groups = workload.build(tpch_dbs=tpch_dbs, tpch_texts=texts, micro_db=micro_db, micro_tables=micro_tables,
                            micro_having=having, rw_dir=rw_dir, opentoid_db=args.opentoid_db or None,
                            quick=args.quick)
    if args.only:
        globs = [x for x in args.only.split(",") if x]
        for g in groups:
            g["statements"] = [s for s in g["statements"] if any(fnmatch.fnmatchcase(s["id"], x) for x in globs)]
    n_statements = sum(len(g["statements"]) for g in groups)
    say(f"workload {workload.WORKLOAD_VERSION}: {n_statements} statements in {len(groups)} groups; "
        f"sides: {', '.join(sides)}; out {out_dir}")

    run = Run(args, sides, out_dir)
    for g in groups:
        run.run_group(g)
        # kept as it goes: a run that dies in its last phase keeps every statement measured
        with open(os.path.join(out_dir, "partial.json"), "w") as f:
            json.dump({"statements": run.records, "sides": run.side_info, "roles": run.roles}, f, default=str)
    run.confirm_coverage(groups)
    fresh, during = {}, {}
    if not args.no_fresh:
        pr = workload.probes(args.quick, 10 in tpch_dbs, micro_tables)
        if args.only:
            globs = [x for x in args.only.split(",") if x]
            pr = {k: [s for s in v if any(fnmatch.fnmatchcase(s, x) for x in globs)] for k, v in pr.items()}
        fresh = run.fresh(groups, pr, fresh_reps)
        bg_sf = 10 if (10 in tpch_dbs and not args.quick) else min(tpch_dbs)
        if not args.only or any(fnmatch.fnmatchcase("during_upload", x) for x in args.only.split(",")):
            bg_group = workload.group(f"bg_sf{bg_sf}", tpch_dbs[bg_sf], [])
            during = run.during_upload(bg_group, max(1, min(fresh_reps, 2)))

    accepted = verdicts.load_accepted(args.accepted_file)
    noisy = [r["id"] for r in run.records if r["noisy"]]
    invalid = bool((not pre["quiet"] or noisy) and not args.allow_noisy)
    provisional = bool(args.allow_noisy)
    env = {k: v for k, v in os.environ.items() if k.startswith("GPUDB_") and k != "GPUDB_EXTENSION_PATH"}
    header = {"date": time.strftime("%Y-%m-%d %H:%M"), "workload": workload.WORKLOAD_VERSION, "n": args.n,
              "warm": warm, "fresh_reps": fresh_reps, "no_fresh": args.no_fresh, "statements": n_statements, "quick": args.quick,
              "python": platform.python_version(), "duckdb": duckdb.__version__,
              "machine": f"{platform.system()} {platform.machine()} {platform.processor() or ''}".strip(),
              "cpus": quiet.CPUS, "sides": run.side_info, "quiet": pre, "noisy_statements": len(noisy),
              "worst_share": max([r["others_share"] for r in run.records] or [0.0]),
              "gpudb_env": env, "wall_s": time.time() - t_start, "accepted_file": os.path.relpath(
                  args.accepted_file, REPO) if args.accepted_file.startswith(REPO) else args.accepted_file,
              "argv": sys.argv[1:], "tpch_dbs": tpch_dbs, "micro_db": micro_db}
    result = {"header": header, "provisional": provisional, "invalid": invalid, "roles": run.roles,
              "statements": run.records, "fresh": fresh, "fresh_sides": run.fresh_sides, "during_upload": during}
    breaches, code = finish(result, run, cands, accepted, args.absolute, invalid, args.allow_noisy)
    status = result["status"]
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=1, default=str)
    md = report.render(result)
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write(md + "\n")
    print(md)
    say(f"\n{status}: {breaches} breach(es); {out_dir}/result.json, {out_dir}/report.md; "
        f"{header['wall_s'] / 60:.1f} min")
    shutil.rmtree(rw_dir, ignore_errors=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
