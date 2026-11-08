#!/usr/bin/env python3
"""gpu_functions_check.py — does gpu_functions() describe every gpudb function,
and does every example it gives actually run?

gpu_functions() (src/extension/gpu_functions.cpp) is the extension's own
documentation table, there because DuckDB's stable C API cannot attach a
description, an example or parameter names to a function. This script holds
it to two things:

  1. coverage — the (function_name, function_type, parameter_types,
     return_type) rows of gpu_functions() and of duckdb_functions() WHERE
     function_name LIKE 'gpu\\_%' are the same set, with no duplicate in
     gpu_functions(). duckdb_functions() has no column that names the
     extension a C-API function came from (DuckDB 1.5 has none; 2.0's
     extension_name is NULL for them), so gpudb's functions are told apart by
     the gpu_ prefix — and the check first confirms that a database WITHOUT
     the extension has no function with that prefix, so the prefix selects
     gpudb's functions and nothing else. test/sql/gpu_functions.test makes the
     same comparison inside the SQL suite;
  2. examples — every `example` is run in a FRESH in-memory database after
     LOAD and must not raise. Examples are self-contained statement sequences,
     so each one starts from nothing and leaves nothing behind for the next.

Two ways to run it:

  python3 scripts/gpu_functions_check.py
      through the pip `duckdb` module, whatever version is installed
  python3 scripts/gpu_functions_check.py --cli /path/to/duckdb
      through a stock DuckDB CLI (started with -unsigned), one process per
      example — the way the extension is used from the shell

The extension is --extension, else GPUDB_EXTENSION_PATH, else the first
*.duckdb_extension in build-macos/ or build-linux/ (./scripts/build.sh).
Exits 0 iff every check passes.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# The gpu_ functions as duckdb_functions() reports them, and as gpu_functions()
# does — the same four columns, so EXCEPT compares like with like.
OURS = "SELECT function_name, function_type, parameter_types, return_type FROM gpu_functions()"
CATALOG = ("SELECT function_name, function_type, parameter_types, return_type "
           "FROM duckdb_functions() WHERE function_name LIKE 'gpu\\_%' ESCAPE '\\'")
COVERAGE = (
    "SELECT side, function_name, function_type, CAST(parameter_types AS VARCHAR) AS parameter_types FROM ("
    f"SELECT 'registered but not in gpu_functions()' AS side, * FROM ({CATALOG} EXCEPT {OURS}) "
    "UNION ALL "
    f"SELECT 'in gpu_functions() but not registered', * FROM ({OURS} EXCEPT {CATALOG})"
    ") ORDER BY ALL"
)
DUPLICATES = (
    "SELECT function_name, function_type, CAST(parameter_types AS VARCHAR) AS parameter_types "
    f"FROM ({OURS}) GROUP BY ALL HAVING count(*) > 1 ORDER BY ALL"
)
BARE = "SELECT count(*) AS n FROM duckdb_functions() WHERE function_name LIKE 'gpu\\_%' ESCAPE '\\'"
EXAMPLES = "SELECT function_name, function_type, example FROM gpu_functions() ORDER BY ALL"


def find_extension(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    env = os.environ.get("GPUDB_EXTENSION_PATH")
    if env:
        return env
    for build in ("build-macos", "build-linux"):
        found = sorted(glob.glob(os.path.join(REPO, build, "src", "extension", "*.duckdb_extension")))
        if found:
            return found[0]
    return None


class PyEngine:
    """The pip duckdb module: one in-memory database per call."""

    def __init__(self, ext: str):
        import duckdb   # noqa: WPS433 — imported here so --cli needs no module
        self.duckdb = duckdb
        self.ext = ext
        self.version = duckdb.__version__

    def _connect(self, load: bool):
        con = self.duckdb.connect(config={"allow_unsigned_extensions": "true"})
        if load:
            con.execute(f"LOAD '{self.ext}'")
        return con

    def rows(self, sql: str, load: bool = True) -> list[tuple]:
        con = self._connect(load)
        try:
            return [tuple(r) for r in con.execute(sql).fetchall()]
        finally:
            con.close()

    def run(self, sql: str) -> str | None:
        """None when `sql` ran, else the error text."""
        try:
            self.rows(sql)
            return None
        except Exception as e:   # noqa: BLE001 — any error is the finding
            return f"{type(e).__name__}: {e}"


class CliEngine:
    """A stock DuckDB CLI: one process (= one in-memory database) per call."""

    def __init__(self, cli: str, ext: str):
        self.cli = cli
        self.ext = ext
        self.env = dict(os.environ)
        out = subprocess.run([cli, "--version"], capture_output=True, text=True, env=self.env)
        # DuckDB 2.0's CLI switches to an "agent mode" output format when it
        # detects it is driven by a coding agent and stdout is not a terminal;
        # this script parses -json output, so it turns that off with -no-agent
        # where the CLI has the flag (1.5 has neither the mode nor the flag).
        probe = subprocess.run([cli, "-no-agent", "-c", "SELECT 1"], capture_output=True, text=True, env=self.env)
        self.flags = ["-no-agent"] if probe.returncode == 0 else []
        self.version = (out.stdout or out.stderr).strip()

    def _exec(self, sql: str, load: bool, fmt: list[str]) -> subprocess.CompletedProcess:
        prefix = f"LOAD '{self.ext}'; " if load else ""
        return subprocess.run([self.cli, *self.flags, "-unsigned", *fmt, "-c", prefix + sql],
                              capture_output=True, text=True, env=self.env, timeout=120)

    def rows(self, sql: str, load: bool = True) -> list[tuple]:
        p = self._exec(sql, load, ["-json"])
        err = self._error(p)
        if err:
            raise RuntimeError(err)
        # A LOAD prints no result set, so stdout holds the query's alone; an
        # empty result prints nothing at all.
        text = p.stdout.strip()
        return [tuple(r.values()) for r in json.loads(text)] if text else []

    def run(self, sql: str) -> str | None:
        return self._error(self._exec(sql, True, []))

    @staticmethod
    def _error(p: subprocess.CompletedProcess) -> str | None:
        lines = [ln for ln in p.stderr.splitlines() if not ln.startswith("[gpudb]")]
        if p.returncode != 0 or any("Error" in ln for ln in lines):
            return "\n".join(lines).strip() or f"exit status {p.returncode}"
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cli", help="run through this DuckDB CLI instead of the pip duckdb module")
    ap.add_argument("--extension", help="path to gpudb.<platform>.duckdb_extension")
    args = ap.parse_args()

    ext = find_extension(args.extension)
    if not ext or not os.path.exists(ext):
        print("gpu_functions_check: no extension found — build it (./scripts/build.sh), "
              "or pass --extension / set GPUDB_EXTENSION_PATH", file=sys.stderr)
        return 2
    eng = CliEngine(args.cli, ext) if args.cli else PyEngine(ext)
    print(f"==> gpu_functions() against {'CLI ' + args.cli if args.cli else 'pip duckdb'} "
          f"(DuckDB {eng.version}), extension {ext}")

    fails: list[str] = []

    # 1. the prefix selects gpudb's functions and nothing else
    bare = eng.rows(BARE, load=False)[0][0]
    if int(bare) != 0:
        fails.append(f"a database without gpudb already has {bare} gpu_ function(s): "
                     "the prefix no longer identifies gpudb's functions")
    print(f"  {'ok  ' if int(bare) == 0 else 'FAIL'} without the extension: {bare} gpu_ functions")

    # 2. coverage, both directions, and no duplicate overload
    missing = eng.rows(COVERAGE)
    for side, name, ftype, ptypes in missing:
        fails.append(f"{side}: {name} ({ftype}) {ptypes}")
    dups = eng.rows(DUPLICATES)
    for name, ftype, ptypes in dups:
        fails.append(f"gpu_functions() lists {name} ({ftype}) {ptypes} more than once")
    n_cat = eng.rows(f"SELECT count(*) FROM ({CATALOG})")[0][0]
    print(f"  {'ok  ' if not missing and not dups else 'FAIL'} coverage: {n_cat} registered gpu_ overloads, "
          f"{len(missing)} differ from gpu_functions(), {len(dups)} duplicated")
    for f in fails:
        print("       ", f)

    # 3. every example runs, from a fresh database
    examples = eng.rows(EXAMPLES)
    ran = 0
    for name, ftype, example in examples:
        if f"{name}(" not in example:
            msg = f"example for {name} ({ftype}) does not call {name}: {example}"
            fails.append(msg)
            print("  FAIL", msg)
            continue
        err = eng.run(example)
        if err:
            msg = f"example for {name} ({ftype}) failed: {example}\n         {err}"
            fails.append(msg)
            print("  FAIL", msg)
        else:
            ran += 1
    print(f"  {'ok  ' if ran == len(examples) else 'FAIL'} examples: {ran} / {len(examples)} ran")

    print(f"==> {'all checks passed' if not fails else str(len(fails)) + ' check(s) failed'}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
