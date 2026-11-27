"""The modes a statement runs under, and the execution protocol in each.

Every mode has its own gpudb connection and its own native DuckDB
connection over the same data. Native mirrors the wrapper's handle topology
(root, the statement's cursor, a sibling cursor, a raw cursor), so a setting
or a write made through one handle reaches native exactly where plain DuckDB
would make it reach: the expected answer is never the gate's own idea of
DuckDB's scoping rules.

The basic protocol is: first execution (literal set 0, drawn by the
extension's renderer), re-run with literal set 1 (drawn by the Python
reference renderer from the cached decision), then literal set 0 again. Each
execution is compared with native for its own literals.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import duckdb

from . import data
from .runner import (AMBIG, BOTH_ERR, DIFF, PASS, RAISED, SKIP, WAIT_REASONS, Res, compare, diff_detail,
                     run_gpu, run_native, worst)

TEST_KW = dict(residency="eager", floor_rows=0, thresholds=False)
BG_KW = dict(residency="background", floor_rows=0, thresholds=False)

# (setting, value): each changes what a statement returns or how it orders, binds or casts
SETTINGS = [("default_order", "'DESC'"), ("default_null_order", "'nulls_first'"),
            ("default_collation", "'nocase'"), ("integer_division", "true"), ("TimeZone", "'Asia/Tokyo'"),
            ("ieee_floating_point_ops", "false")]
CHANNELS = ["before", "same", "sibling", "raw", "global"]


def fresh_decisions(g) -> None:
    """Forget the wrapper's per-statement decisions, so the next execution is
    a FIRST execution (the extension's renderer) rather than a re-run of a
    template an earlier statement decided (the Python renderer). Resident sets
    stay: uploading them again per statement would measure the upload, not
    the answer. Private attributes, read defensively: an older or newer
    wrapper without them simply keeps its cache."""
    for name in ("_cache", "_nested_cache", "_flat_cache", "_split_cache"):
        c = getattr(g, "__dict__", {}).get(name)
        if isinstance(c, dict):
            c.clear()


@dataclass
class StmtResult:
    sid: str
    mode: str
    st: Any
    outcome: str = PASS
    device: bool = False
    reasons: List[str] = field(default_factory=list)
    runs: List[Dict[str, Any]] = field(default_factory=list)
    note: str = ""
    nondet: bool = False


class Engine:
    """One gpudb connection and its native mirror, both with the gate's data."""

    def __init__(self, gpudb, kw: Dict[str, Any], setup: List[str], cpu_reference: bool):
        self.g = gpudb.connect(**kw)
        self.n = duckdb.connect()
        if cpu_reference and getattr(self.g, "_backend", "") == "CPU":
            # the transparent path lowers onto the extension's CPU reference operators:
            # every exactness check holds, no speed claim is made
            self.g._backend = "CPU-REFERENCE"
        self.backend = getattr(self.g, "_backend", "")
        for s in setup:
            self.g.execute(s)
            self.n.execute(s)
        self.g_sib, self.n_sib = self.g.cursor(), self.n.cursor()
        self.g_raw, self.n_raw = self.g._raw.cursor(), self.n.cursor()

    def close(self):
        for c in (self.g_sib, self.n_sib, self.g_raw, self.n_raw):
            try:
                c.close()
            except Exception:       # noqa: BLE001
                pass
        try:
            self.g.close()
        except Exception:           # noqa: BLE001
            pass
        self.n.close()


class Protocol:
    def __init__(self, mode: str, api: str, wait_s: float = 0.0, table_map=None, nondet_check=True,
                 verbose=False, fold=False):
        self.mode, self.api, self.wait_s = mode, api, wait_s
        self.table_map = table_map
        self.nondet_check = nondet_check
        self.verbose = verbose
        self.fold = fold
        self.native_cache: Dict[str, Res] = {}

    # -- one execution ---------------------------------------------------------------------------
    def native(self, nh, sql, cache=True) -> Res:
        if cache and sql in self.native_cache:
            return self.native_cache[sql]
        r = run_native(nh, sql, self.api == "sql")
        if cache:
            self.native_cache[sql] = r
        return r

    def one(self, res: StmtResult, label: str, gh, nh, k: int, cache=True, fold=None) -> None:
        st = res.st
        sql = st.sql(k, self.table_map)
        nat = self.native(nh, sql, cache)
        nondet = False
        if self.nondet_check and nat.kind == "rows" and st.ordered and not st.total:
            again = run_native(nh, sql, False)
            nondet = again.kind != "rows" or again.rows != nat.rows
        full = None
        if st.limit_free and nat.kind == "rows" and st.ordered and not st.total:
            full = self.native(nh, st.full_sql(k, self.table_map), cache)
        deadline = time.time() + self.wait_s
        tries = 0
        while True:
            tries += 1
            got = run_gpu(gh, sql, self.api, self.api == "sql")
            o, why = compare(st, nat, got, nondet, full, self.fold if fold is None else fold)
            lr = got.lr or {}
            rw = bool(lr.get("rewritten")) and not lr.get("fallback")
            self._record(res, label, sql, o, why, nat, got, lr, tries)
            if rw or not self.wait_s or time.time() > deadline or \
                    str(lr.get("reason") or "") not in WAIT_REASONS:
                break
            # the background upload runs only while the connection is idle, and on a
            # machine whose cores are busy it waits for a longer quiet stretch
            # (GPUDB_UPLOAD_QUIET_MS): back off, so the retries do not keep it busy
            time.sleep(min(1.0, 0.05 * 2 ** (tries - 1)))
        res.nondet = res.nondet or nondet

    def _record(self, res, label, sql, o, why, nat, got, lr, tries):
        rw = bool(lr.get("rewritten")) and not lr.get("fallback")
        res.device = res.device or rw
        res.reasons.append("device" if rw else (str(lr.get("reason") or "") or "-"))
        res.outcome = worst(res.outcome, o)
        run = {"label": label, "outcome": o, "rewritten": bool(lr.get("rewritten")), "fallback": bool(lr.get("fallback")),
               "reason": lr.get("reason"), "form": lr.get("form"), "engine": lr.get("engine"), "try": tries}
        if o in (DIFF, RAISED) or (o == AMBIG and self.verbose):
            run.update({"sql": sql, "why": why, "native": nat.summary(), "gpudb": got.summary(),
                        "detail": str(lr.get("detail") or "")[:300], "rewritten_sql": str(lr.get("sql") or "")[:1500],
                        "error": str(lr.get("error") or "")[:300], "diff": diff_detail(nat, got) if o == DIFF else ""})
            if not res.note:
                res.note = why
        res.runs.append(run)

    # -- protocols -------------------------------------------------------------------------------
    def basic(self, eng: Engine, st, how: str = "root") -> StmtResult:
        """how: 'root' (the connection, its decisions forgotten first), 'warm'
        (the connection, decisions kept from every earlier statement), 'cursor'
        (a fresh `con.cursor()` per statement)."""
        res = StmtResult(st.sid(self.mode), self.mode, st)
        if how == "cursor":
            gh, nh = eng.g.cursor(), eng.n.cursor()
        else:
            gh, nh = eng.g, eng.n
            if how == "root":
                fresh_decisions(gh)
        try:
            self.one(res, "first", gh, nh, 0)
            self.one(res, "rerun", gh, nh, 1)
            self.one(res, "back", gh, nh, 0)
        finally:
            if how == "cursor":
                _close(gh, nh)
        return res

    def write(self, eng: Engine, st, channel: str, m: int) -> StmtResult:
        res = StmtResult(st.sid(self.mode), self.mode, st)
        gh, nh = eng.g, eng.n
        fresh_decisions(gh)
        try:
            self.one(res, "first", gh, nh, 0, cache=False)
            kinds = ["insert", "delete"] if channel == "raw" else ["insert", "update", "delete"]
            kind = kinds[m % len(kinds)]
            w = {"insert": data.WRITES_INSERT, "update": data.WRITES_UPDATE, "delete": data.WRITES_DELETE}[kind]
            w = w.replace("{m}", str(m % 89))
            gw, nw = {"same": (gh, nh), "sibling": (eng.g_sib, eng.n_sib), "raw": (eng.g_raw, eng.n_raw)}[channel]
            try:
                nw.execute(w)
            except duckdb.Error as e:
                res.note = f"write skipped: {e}"
                return res
            gw.execute(w)
            self.one(res, f"after-{kind}", gh, nh, 0, cache=False)
            self.one(res, f"after-{kind}-rerun", gh, nh, 1, cache=False)
        finally:
            pass
        return res

    def setting(self, eng: Engine, st, name: str, value: str, channel: str) -> Optional[StmtResult]:
        res = StmtResult(st.sid(self.mode), self.mode, st)
        gh, nh = eng.g, eng.n
        fresh_decisions(gh)
        glob = channel == "global"
        pick = {"before": (gh, nh), "same": (gh, nh), "sibling": (eng.g_sib, eng.n_sib),
                "raw": (eng.g_raw, eng.n_raw), "global": (eng.g_sib, eng.n_sib)}
        gs, ns = pick[channel]
        stmt = f"SET {'GLOBAL ' if glob else ''}{name} = {value}"
        reset = f"RESET {'GLOBAL ' if glob else ''}{name}"
        fold = name == "default_collation"
        try:
            if channel != "before":
                self.one(res, "first", gh, nh, 0, cache=False, fold=False)
            try:
                ns.execute(stmt)
            except duckdb.Error as e:
                res.outcome, res.note = SKIP, f"{stmt}: {str(e).splitlines()[0][:100]}"
                return res
            gs.execute(stmt)
            label = "under" if channel == "before" else "after-set"
            self.one(res, label, gh, nh, 0, cache=False, fold=fold)
            self.one(res, label + "-rerun", gh, nh, 1, cache=False, fold=fold)
        finally:
            for h in (ns, gs):
                try:
                    h.execute(reset)
                except Exception:   # noqa: BLE001
                    pass
        return res


def _close(*hs):
    for h in hs:
        try:
            h.close()
        except Exception:           # noqa: BLE001
            pass
