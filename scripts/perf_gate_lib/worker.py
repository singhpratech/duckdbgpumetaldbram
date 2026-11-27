#!/usr/bin/env python3
"""perf_gate worker: one side of a perf_gate measurement, in a process of its own.

Started by scripts/perf_gate.py as

    <python> -I worker.py --side NAME --wrapper PATH|installed|none [--ext FILE]

and driven over stdin/stdout with one JSON object per line. `--wrapper none`
is raw DuckDB (no gpudb import, no extension): the native side. A python/
tree is put first on sys.path, so two different gpudb packages are never in
one process; the worker checks that the gpudb it imported is the one it was
given and refuses otherwise.

Every time this file reports is taken here, around the statement alone
(rows fetched), so the pipe to the driver is never inside a timed region.

Deliberately self-contained (standard library + duckdb + the gpudb under
test): `-I` keeps the caller's directory and PYTHONPATH out of the import
path.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import re
import resource
import sys
import time
import traceback


_TAG_RE = re.compile(r"'gpudb:[^']*'")


def _peak_rss() -> int:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if sys.platform == "darwin" else r * 1024      # bytes on macOS, KiB on Linux


def _digest(rows) -> str:
    """Order-insensitive digest of a result: the driver only uses it to say
    that the sides timed the same work (the correctness gate owns equality)."""
    h = hashlib.sha256()

    def cell(v):
        # a DOUBLE aggregate's last bits follow DuckDB's parallel summation order
        return f"{v:.10g}" if isinstance(v, float) else repr(v)
    for s in sorted("|".join(map(cell, r)) for r in rows):
        h.update(s.encode())
        h.update(b"\n")
    return h.hexdigest()[:16]


class Side:
    def __init__(self, args):
        self.name = args.side
        self.wrapper = args.wrapper
        self.ext = args.ext or ""
        self.native = args.wrapper == "none"
        if self.ext:
            os.environ["GPUDB_EXTENSION_PATH"] = self.ext
        if not self.native and args.wrapper != "installed":
            tree = os.path.abspath(args.wrapper)
            sys.path.insert(0, tree)
        import duckdb  # noqa: F401  (the interpreter's own)
        self.duckdb = duckdb
        self.gpudb = None
        if not self.native:
            import gpudb
            if args.wrapper != "installed":
                got = os.path.dirname(os.path.dirname(os.path.abspath(gpudb.__file__)))
                if os.path.realpath(got) != os.path.realpath(os.path.abspath(args.wrapper)):
                    raise RuntimeError(f"imported gpudb from {got}, not from {args.wrapper}")
            self.gpudb = gpudb
        self.ctx = {}          # name -> connection
        self.logs = {}         # name -> list of wrapper log lines (bounded)

    # ---- protocol ----
    def hello(self, _m):
        out = {"side": self.name, "pid": os.getpid(), "python": sys.version.split()[0],
               "duckdb": self.duckdb.__version__, "native": self.native}
        if self.gpudb is not None:
            out["gpudb"] = getattr(self.gpudb, "__version__", "?")
            out["gpudb_file"] = self.gpudb.__file__
        return out

    def open(self, m):
        name = m["ctx"]
        db = m["db"]
        t0 = time.perf_counter()
        if self.native:
            con = self.duckdb.connect(db, read_only=m.get("read_only", True))
        else:
            logs = self.logs.setdefault(name, [])

            def log(line, logs=logs):
                logs.append(line)
                del logs[:-200]
            con = self.gpudb.connect(db, read_only=m.get("read_only", True),
                                     residency=m.get("residency", "eager"),
                                     memory_budget=m.get("memory_budget"),
                                     floor_rows=m.get("floor_rows", 1_000_000),
                                     log=log)
        connect_ms = (time.perf_counter() - t0) * 1000.0
        raw = con if self.native else con._raw
        if m.get("threads"):
            raw.execute(f"SET threads TO {int(m['threads'])}")
        for s in m.get("setup", []):
            raw.execute(s)
        self.ctx[name] = con
        out = {"connect_ms": connect_ms}
        if not self.native:
            try:
                out["build_info"] = raw.execute("SELECT gpu_build_info()").fetchone()[0]
            except Exception as e:          # no extension: every statement runs native
                out["build_info"] = f"(no extension: {str(e).splitlines()[0][:120]})"
            out["extension_note"] = getattr(con, "extension_note", "")
        return out

    def close(self, m):
        con = self.ctx.pop(m["ctx"], None)
        if con is not None:
            con.close()
        return {}

    def _once(self, con, sql, path):
        """One execution through `path`, rows fetched. Returns (ms, rows, rewrite)."""
        if path == "cursor":
            t0 = time.perf_counter()
            cur = con.cursor()
            rows = cur.execute(sql).fetchall()
            ms = (time.perf_counter() - t0) * 1000.0
            lr = cur.last_rewrite() if not self.native else None
            cur.close()
            return ms, rows, lr
        t0 = time.perf_counter()
        if path == "sql":
            rel = con.sql(sql)
            rows = rel.fetchall() if rel is not None else []
        else:
            rows = con.execute(sql).fetchall()
        ms = (time.perf_counter() - t0) * 1000.0
        return ms, rows, (con.last_rewrite() if not self.native else None)

    @staticmethod
    def _route(lr):
        if lr is None:
            return None
        detail = (lr.get("detail") or "").splitlines()
        # A resident set's tag carries hashes of its lanes and of the state it
        # was built under; two wrappers that render the same statement over
        # differently named sets are not on different routes.
        rendered = _TAG_RE.sub("'<tag>'", lr.get("sql") or "")
        return {"rewritten": bool(lr.get("rewritten")), "reason": lr.get("reason") or "",
                "form": lr.get("form") or "", "engine": lr.get("engine") or "",
                "fallback": bool(lr.get("fallback")),
                "detail": detail[0][:240] if detail else "",
                "error": (lr.get("error") or "").splitlines()[0][:200] if lr.get("error") else "",
                "rendered": hashlib.sha256(rendered.encode()).hexdigest()[:12] if rendered else ""}

    def run(self, m):
        """`reps` executions of one statement. `vary` substitutes a per-run
        integer for {n} (the same sequence on every side)."""
        con = self.ctx[m["ctx"]]
        sql, path, reps = m["sql"], m.get("path", "execute"), int(m.get("reps", 1))
        start = int(m.get("n0", 0))
        times, routes = [], []
        digest = rows_n = None
        for i in range(reps):
            text = sql.replace("{n}", str(start + i)) if m.get("vary") else sql
            gc.collect()
            ms, rows, lr = self._once(con, text, path)
            times.append(ms)
            routes.append(self._route(lr))
            if m.get("digest") and digest is None:
                digest, rows_n = _digest(rows), len(rows)
        return {"times": times, "routes": routes, "digest": digest, "rows": rows_n}

    def memory(self, m):
        out = {"peak_rss": _peak_rss()}
        con = self.ctx.get(m.get("ctx"))
        if con is not None and not self.native:
            try:
                mem = con.memory()
                out["resident_bytes"] = mem.get("bytes")
                out["device_allocated"] = mem.get("device_allocated")
                out["budget"] = mem.get("budget")
            except Exception as e:
                out["memory_error"] = str(e).splitlines()[0][:160]
        return out

    def logs_tail(self, m):
        return {"lines": self.logs.get(m["ctx"], [])[-int(m.get("n", 20)):]}

    def ttr(self, m):
        """Time to resident in background mode: the first sighting of `sql`
        schedules the upload; the connection then stays idle (polled every
        `poll_ms`) until every set it holds is ready, failed, or `max_s`
        passes, and the statement runs once more to see whether it now goes
        to the device."""
        con = self.ctx[m["ctx"]]
        path = m.get("path", "execute")
        first_ms, _, lr0 = self._once(con, m["sql"], path)
        t0 = time.perf_counter()
        poll = m.get("poll_ms", 100) / 1000.0
        states = {}
        ready_s = None
        while time.perf_counter() - t0 < m.get("max_s", 60):
            time.sleep(poll)
            if self.native:
                break
            states = con.residents()
            if states and all(s in ("ready", "failed", "missing", "stale") for s in states.values()):
                ready_s = time.perf_counter() - t0
                break
        after_ms, _, lr1 = self._once(con, m["sql"], path)
        return {"first_ms": first_ms, "ready_s": ready_s, "states": states,
                "after_ms": after_ms, "route_first": self._route(lr0), "route_after": self._route(lr1)}

    def during_bg(self, m):
        """Statements issued while a background upload is in flight: `trigger`
        runs once (its first sighting schedules the upload), then `sql` runs
        `count` times at the gap sequence the driver gives (the same on every
        side). Each time is kept with whether an upload was in flight when
        the statement started."""
        con = self.ctx[m["ctx"]]
        self._once(con, m["trigger"], "execute")
        out = []
        for g in m["gaps_ms"]:
            time.sleep(g / 1000.0)
            busy = False
            if not self.native:
                st = con.residents()
                busy = any(s in ("pending", "uploading") for s in st.values())
            ms, _, lr = self._once(con, m["sql"], "execute")
            out.append({"ms": ms, "busy": busy,
                        "rewritten": bool(lr and lr.get("rewritten"))})
        return {"samples": out}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", required=True)
    ap.add_argument("--wrapper", required=True)
    ap.add_argument("--ext", default="")
    args = ap.parse_args()
    # Nothing but protocol lines on the pipe. The extension prints from C++
    # straight to file descriptor 1, so the protocol gets a duplicate of it and
    # descriptor 1 itself is pointed at stderr before anything is imported.
    sys.stdout.flush()
    out = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    try:
        side = Side(args)
    except Exception as e:
        out.write(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}) + "\n")
        out.flush()
        return 2
    out.write(json.dumps({"ok": True, "ready": side.hello({})}) + "\n")
    out.flush()
    for line in sys.stdin:
        if not line.strip():
            continue
        m = json.loads(line)
        op = m.get("op")
        if op == "quit":
            out.write(json.dumps({"ok": True}) + "\n")
            out.flush()
            break
        try:
            res = getattr(side, op)(m)
            out.write(json.dumps({"ok": True, **res}) + "\n")
        except Exception as e:
            out.write(json.dumps({"ok": False, "error": f"{type(e).__name__}: {str(e)[:500]}",
                                  "trace": traceback.format_exc()[-1500:]}) + "\n")
        out.flush()
    return 0


if __name__ == "__main__":
    random.seed(0)
    sys.exit(main())
