"""The client wrapper (§3.3): the same call surface as duckdb.connect, with
every statement passing through classification, name resolution, the
template cache and the rewrite before DuckDB sees it."""
from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import duckdb

from . import _classify, _exprs, _flatten, _join, _resolve, _rewrite, _split, _thresholds
from ._residency import ResidencyManager

GPUDB_EXTENSION_ENV = "GPUDB_EXTENSION_PATH"
_GROUP_BY_RE = re.compile(r"\bGROUP\s+BY\b", re.IGNORECASE)
# an aggregate without GROUP BY is only worth parsing over a join (§4.12): on a
# single table native's filter + sum wins (measured), so the fast path keeps it
_GLOBAL_AGG_JOIN_RE = re.compile(
    r"\b(?:sum|count|min|max|avg)\s*\(.*\bFROM\b.*(?:\bJOIN\b|,)", re.IGNORECASE | re.DOTALL)


def _maybe_aggregate(sql: str) -> bool:
    return bool(_GROUP_BY_RE.search(sql) or _GLOBAL_AGG_JOIN_RE.search(sql))
_SELECT_START_RE = re.compile(r"^\s*(SELECT|FROM|VALUES)\b", re.IGNORECASE)
_TABLE_REF_RE = re.compile(r'\bFROM\s+((?:"[^"]+"|[A-Za-z_][A-Za-z0-9_]*)(?:\.(?:"[^"]+"|[A-Za-z_][A-Za-z0-9_]*)){0,2})', re.IGNORECASE)
_MAX_STATEMENT_BYTES = 16 * 1024
_LITERAL_RE = re.compile(r"""('(?:[^']|'')*')|(\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b)""")
_WS_RE = re.compile(r"\s+")
STALE_MARKER = "GPUDB_STALE"
# a rewritten statement this slow is far outside a device answer (1-5 ms): when the session
# never saw it run native (eager residency), native is timed once to compare
_MEASURE_NATIVE_ABOVE_MS = 20.0
REASONS = ("shape", "not_resident", "threshold", "backend", "double", "nulls", "overflow",
           "decimal", "collation", "too_long", "transaction", "view", "temp", "ambiguous",
           "not_found", "manual", "error", "off", "params", "multi")


@dataclass
class Decision:
    """Cached per (normalised template, identity, settings)."""
    rewritten: bool
    reason: str = ""
    plan: Optional[_rewrite.Plan] = None
    fqn: str = ""
    tag: str = ""
    upload_sql: str = ""
    form: str = ""
    literals: Tuple[str, ...] = ()     # literal values of the template this was decided on
    scalar_sql: str = ""               # gpu_rewrite_ast's rendering for exactly those literals
    output_checked: bool = False       # the first rewritten run's rows_out was compared to the plain-form bound
    join: Optional[Any] = None         # _join.JoinResidency for a statement over a key join (§4.8)
    # computed lanes (§4.10) that embed a literal: the normalised template does
    # not identify the statement, so each literal tuple gets its own decision
    literal_sensitive: bool = False
    # expressions over aggregates (§4.11): the rewritten INNER group by is
    # spliced into this outer statement text at the placeholder
    wrap: Optional[Tuple[str, str]] = None
    is_join: bool = False              # the statement's FROM is a join (decides the §4.13 fallback)
    sentinels: List[Any] = field(default_factory=list)   # §4.18: row-count sentinels of tables read by subquery lanes
    # measured rule 1 (§9.1): this statement's own native time, seen while it
    # was not resident yet, against its first rewritten runs
    native_ms: Optional[float] = None
    rewritten_ms: List[float] = field(default_factory=list)
    timing_checked: bool = False
    variants: Dict[Tuple[str, ...], "Decision"] = field(default_factory=dict)


@dataclass
class LastRewrite:
    statement: str = ""
    rewritten: bool = False
    reason: str = ""
    form: str = ""
    tag: str = ""
    sql: str = ""
    fallback: bool = False       # rewritten statement raised GPUDB_STALE, native re-run
    round_trip_ms: float = 0.0
    engine: str = ""             # 'scalar' (gpu_rewrite_ast) | 'python' (reference renderer)

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def _num(x: Optional[str]):
    if x is None:
        return None
    try:
        return int(x)
    except ValueError:
        try:
            return float(x)
        except ValueError:
            return None


def _find_extension(explicit: Optional[str]) -> Optional[str]:
    if explicit:
        return explicit
    env = os.environ.get(GPUDB_EXTENSION_ENV)
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    for build in ("build-macos", "build-linux"):
        d = os.path.join(root, build, "src", "extension")
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.endswith(".duckdb_extension"):
                    return os.path.join(d, f)
    return None


class Connection:
    """Wraps a duckdb.DuckDBPyConnection. Attributes not defined here are
    delegated to it unchanged (relational API, appenders, fetch helpers)."""

    def __init__(self, raw: duckdb.DuckDBPyConnection, *, transparent: bool = True,
                 residency: str = "background", floor_rows: int = 1_000_000,
                 idle_ms: float = 20.0, log=None, _parent: Optional["Connection"] = None):
        self._raw = raw
        self._transparent = transparent
        self._residency_mode = residency
        self._floor_rows = floor_rows
        self._log = log or (lambda m: None)
        self._tx_open = False
        self._last = LastRewrite()
        self._cache: Dict[Tuple[str, str], Decision] = {}
        self._unique_cache: Dict[Tuple[int, str], bool] = {}     # (table oid, column) -> unique among non-NULLs
        self._nested_cache: Dict[Tuple[str, str], Any] = {}      # exact statement text -> nested plan | False
        self._flat_cache: Dict[str, str] = {}                    # statement text -> the same with SPJ derived tables folded in
        self._last_tags: List[str] = []
        self._function_stability: Optional[Dict[str, bool]] = None   # name -> every overload is a CONSISTENT scalar
        self._expr_types: Dict[Tuple[str, str], str] = {}        # (table fqn, expression sql) -> DuckDB type
        self._select_template: Optional[dict] = None
        self._settings: Dict[str, str] = {}
        self._settings_key = ""
        self._backend = ""
        self._has_rewrite_scalar = False
        self._refresh_after = False
        self._big_tables: Optional[set] = None    # names of tables above the floor (§0)
        self._parent = _parent
        if _parent is None:
            self._manager = ResidencyManager(lambda: self._raw.cursor(), mode=residency,
                                             idle_ms=idle_ms, log=self._log)
            self._refresh_settings()
            self._probe_extension()
        else:
            self._manager = _parent._manager
            self._settings = dict(_parent._settings)
            self._settings_key = _parent._settings_key
            self._backend = _parent._backend
            self._has_rewrite_scalar = _parent._has_rewrite_scalar

    # ---- settings ----
    @property
    def transparent(self) -> bool:
        return self._transparent

    @transparent.setter
    def transparent(self, v: bool) -> None:
        self._transparent = bool(v)

    @property
    def residency(self) -> str:
        return self._residency_mode

    def last_rewrite(self) -> Dict[str, Any]:
        return self._last.as_dict()

    def residents(self) -> Dict[str, str]:
        return self._manager.snapshot()

    def _refresh_settings(self) -> None:
        row = self._raw.execute(
            "SELECT current_setting('default_order'), current_setting('default_null_order'), "
            "current_setting('default_collation'), current_setting('search_path'), "
            "current_database()").fetchone()
        self._settings = {"default_order": row[0], "default_null_order": row[1],
                          "default_collation": row[2] or "", "search_path": row[3] or "",
                          "database": row[4]}
        self._settings_key = "|".join(self._settings.values())
        self._cache.clear()

    def _probe_extension(self) -> None:
        try:
            info = self._raw.execute("SELECT gpu_build_info()").fetchone()[0]
        except Exception:
            self._backend = ""
            return
        m = re.search(r"runtime=(\w+)", info)
        self._backend = (m.group(1) if m else "").upper()   # CPU | METAL | CUDA
        self._exact = "exact=true" in info                   # the v0.7 exact path runs on the GPU side
        self._join = self._exact and "join=true" in info     # ... and so does the materialised key join (§4.8)
        try:
            self._raw.execute("SELECT gpu_rewrite_ast('{}', '{}')").fetchall()
            self._has_rewrite_scalar = True
        except Exception:
            self._has_rewrite_scalar = False

    # ---- duckdb surface ----
    def cursor(self) -> "Connection":
        c = Connection(self._raw.cursor(), transparent=self._transparent,
                       residency=self._residency_mode, floor_rows=self._floor_rows,
                       log=self._log, _parent=self)
        c._thresholds = getattr(self, "_thresholds", True)
        return c

    def duplicate(self) -> "Connection":
        return self.cursor()

    def close(self) -> None:
        if self._parent is None:
            self._manager.close()
        self._raw.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def interrupt(self) -> None:
        self._raw.interrupt()

    def begin(self):
        self._tx_open = True
        return self._raw.begin()

    def commit(self):
        self._invalidate_all("commit")
        self._tx_open = False
        return self._raw.commit()

    def rollback(self):
        self._invalidate_all("rollback")
        self._tx_open = False
        return self._raw.rollback()

    def register(self, name, obj):
        self._invalidate_all("register")
        return self._raw.register(name, obj)

    def unregister(self, name):
        self._invalidate_all("unregister")
        return self._raw.unregister(name)

    def append(self, table, df, **kw):
        self._invalidate_all("append")
        return self._raw.append(table, df, **kw)

    def executemany(self, query, parameters=None):
        self._invalidate_all("executemany")
        self._manager.statement_begin()
        try:
            self._raw.executemany(query, parameters)
        finally:
            self._manager.statement_end()
        return self

    def _note_timing(self, ms: float, query=None, parameters=None) -> None:
        """Rule 1, measured: the thresholds PREDICT the win. When this
        session has also seen the statement run native (it did, every time
        before its set became resident), compare: if the best of the first
        three rewritten runs is not faster than the best native run, the
        template runs native from now on (reason 'threshold'). A session
        that never saw it native (eager residency) times native ONCE, on its
        own cursor, when the rewritten runs are slow enough to be suspect."""
        d = getattr(self, "_timing_decision", None)
        if d is None or d.timing_checked or not getattr(self, "_thresholds", True) or self._last.fallback:
            return
        if not self._last.rewritten:
            if self._last.reason == "not_resident":
                d.native_ms = ms if d.native_ms is None else min(d.native_ms, ms)
            return
        d.rewritten_ms.append(ms)
        if len(d.rewritten_ms) < 3:
            return
        if d.native_ms is None:
            if min(d.rewritten_ms) < _MEASURE_NATIVE_ABOVE_MS or not isinstance(query, str):
                return
            try:
                cur = self._raw.cursor()       # the caller still fetches from self._raw
                t0 = time.perf_counter()
                cur.execute(query, parameters)
                d.native_ms = (time.perf_counter() - t0) * 1000.0
                cur.close()
            except duckdb.Error as e:
                self._log(f"native timing probe failed: {e}")
                d.timing_checked = True
                return
        d.timing_checked = True
        best = min(d.rewritten_ms)
        if best >= d.native_ms:
            self._log(f"threshold: measured {best:.2f} ms rewritten vs {d.native_ms:.2f} ms native — "
                      f"template declined from now on")
            d.rewritten = False
            d.reason = "threshold"

    def execute(self, query, parameters=None):
        sql = self._route(query, parameters)
        self._manager.statement_begin()
        try:
            try:
                t0 = time.perf_counter()
                self._raw.execute(sql, parameters)
                self._note_timing((time.perf_counter() - t0) * 1000.0, query, parameters)
                if self._last.rewritten:
                    self._check_output_size()
            except duckdb.Error as e:
                if self._last.rewritten and STALE_MARKER in str(e):
                    self._on_stale(sql)
                    self._raw.execute(query, parameters)
                else:
                    raise
        finally:
            self._manager.statement_end()
            self._after()
        return self

    def _check_output_size(self) -> None:
        """Once per template, after its first rewritten run: the shape's
        thresholds predict the win from the key's distinct count and the
        WHERE selectivity, but a HAVING or top-k that keeps most groups is
        output-bound like the plain form (§9.1). Read the operator's rows_out
        from gpu_last_stats() on a side cursor (the user's result stays
        untouched) and, when it exceeds the plain-form bound, decline this
        template from now on with reason 'threshold'."""
        d = getattr(self, "_last_decision", None)
        if d is None or d.output_checked or not getattr(self, "_thresholds", True):
            return
        d.output_checked = True
        t = _thresholds.TABLE.get((self._backend or "").upper())
        if t is None:
            return
        try:
            cur = self._raw.cursor()
            try:
                line = cur.execute("SELECT gpu_last_stats()").fetchone()[0] or ""
            finally:
                cur.close()
        except Exception:
            return
        m = re.search(r"rows_out=(\d+)", line)
        if not m:
            return
        rows_out = int(m.group(1))
        has_where = bool(d.plan is not None and d.plan.where)
        if d.join is not None:
            bound = t.join_plain_max_groups          # a key join has its own measured bound (§4.8)
        else:
            bound = t.plain_max_groups_where if has_where else t.plain_max_groups
        if rows_out > bound:
            self._log(f"threshold: {rows_out} rows returned by the resident operator > {bound} — "
                      f"template declined from now on")
            d.rewritten = False
            d.reason = "threshold"

    def sql(self, query, **kw):
        sql = self._route(query, None)
        self._manager.statement_begin()
        try:
            try:
                rel = self._raw.sql(sql, **kw)
                # a lazy relation binds now; force the guard to run here so a
                # stale set falls back inside this call, not at fetch time
                if self._last.rewritten:
                    rel = self._raw.sql(sql, **kw)
                    _ = rel.columns
                return rel
            except duckdb.Error as e:
                if self._last.rewritten and STALE_MARKER in str(e):
                    self._on_stale(sql)
                    return self._raw.sql(query, **kw)
                raise
        finally:
            self._manager.statement_end()
            self._after()

    query = sql

    def _after(self) -> None:
        if self._refresh_after:
            self._refresh_after = False
            try:
                self._refresh_settings()
            except Exception:
                pass

    def __getattr__(self, name):
        return getattr(self._raw, name)

    # ---- the statement path ----
    def _on_stale(self, sql: str) -> None:
        self._last.fallback = True
        tags = [t for t in (getattr(self, "_last_tags", None) or [self._last.tag]) if t]
        for tag in tags:
            st = self._manager.get(tag)
            # a joined set: the error does not say which table moved
            for dep in (st.deps if st is not None else []):
                self._manager.invalidate(dep)
                ds = self._manager.get(dep)
                if ds is not None:
                    self._manager.note_candidate(dep, ds.upload_sql)
            self._manager.invalidate(tag)
            if st is not None:
                self._manager.note_candidate(tag, st.upload_sql)

    def _invalidate_all(self, why: str) -> None:
        self._manager.invalidate(None)
        self._cache.clear()
        self._nested_cache.clear()
        self._flat_cache.clear()
        self._unique_cache.clear()
        self._expr_types.clear()
        self._big_tables = None
        try:
            self._raw.execute("SELECT gpu_invalidate('gpudb:v1')").fetchall()
        except Exception:
            pass

    def _refresh_big_tables(self) -> set:
        """Tables whose estimated size is at or above the floor: only a
        statement naming one of them is ever split, parsed or cached, so a
        statement over a small table costs the wrapper two regex matches."""
        try:
            rows = self._raw.execute(
                "SELECT table_name FROM duckdb_tables() WHERE NOT internal AND NOT temporary "
                "AND estimated_size >= ?", [self._floor_rows]).fetchall()
            self._big_tables = {r[0] for r in rows}
        except Exception:
            self._big_tables = set()
        return self._big_tables

    def _names_big_table(self, sql: str) -> bool:
        """Does the statement name a table at or above the floor — anywhere
        (FROM a, b / JOIN b / a subquery), not only right after the first
        FROM? A word match: a false positive only costs one parse."""
        big = self._big_tables if self._big_tables is not None else self._refresh_big_tables()
        if not big:
            return False
        rx = getattr(self, "_big_tables_rx", None)
        if rx is None or rx[0] is not big:
            pat = "|".join(re.escape(t) for t in sorted(big, key=len, reverse=True))
            rx = (big, re.compile(r'(?<![A-Za-z0-9_])(?:' + pat + r')(?![A-Za-z0-9_])', re.IGNORECASE))
            self._big_tables_rx = rx
        return rx[1].search(sql) is not None

    def _route(self, query: Any, parameters) -> Any:
        """Return the SQL to run in place of `query`."""
        self._last = LastRewrite(statement=query if isinstance(query, str) else "")
        self._timing_decision = None
        self._last_tags = []
        if not isinstance(query, str):
            self._last.reason = "shape"
            return query
        # Fast path (rule 1 for statements that can never be rewritten): a
        # single SELECT-looking statement that does not name a table above
        # the floor, or has no GROUP BY, is passed through untouched — no
        # split, no parse, no cache. A statement that starts with SELECT/FROM/
        # VALUES and carries no ';' cannot be DML (a CTE prefix can, so WITH
        # takes the full path), so nothing is invalidated either.
        if ";" not in query and _SELECT_START_RE.match(query) and (
                not _maybe_aggregate(query) or not self._names_big_table(query)):
            self._last.reason = "threshold" if _maybe_aggregate(query) else "shape"
            return query
        stmts = _classify.split(self._raw, query)
        if stmts is None:
            self._last.reason = "error"
            return query
        non_select = [s for s in stmts if not (_classify.is_select(s) or _classify.is_explain(s))]
        for s in non_select:
            if _classify.is_transaction(s):
                self._tx_open = _classify.transaction_opens(s)
            self._invalidate_all(s.type)
            if s.type in ("SET", "VARIABLE_SET", "ATTACH", "DETACH", "PRAGMA", "LOAD"):
                self._refresh_after = True      # settings change when it has RUN
        if not self._transparent:
            self._last.reason = "off"
            return query
        if len(stmts) != 1:
            self._last.reason = "multi" if stmts else "shape"
            return query
        if parameters:
            self._last.reason = "params"
            return query
        stmt = stmts[0]
        prefix, inner = "", stmt.query
        if _classify.is_explain(stmt):
            prefix, inner = _classify.strip_explain(stmt.query)
            if prefix is None:
                self._last.reason = "shape"
                return query
        elif not _classify.is_select(stmt):
            self._last.reason = "shape"
            return query
        rewritten = self._rewrite_select(inner)
        if rewritten is None:
            return query
        return prefix + rewritten

    def _rewrite_select(self, sql: str) -> Optional[str]:
        if self._tx_open:
            self._last.reason = "transaction"
            return None
        if self._backend in ("", "CPU"):
            self._last.reason = "backend"
            return None
        if len(sql) > _MAX_STATEMENT_BYTES:
            self._last.reason = "too_long"
            return None
        if not _maybe_aggregate(sql):
            self._last.reason = "shape"
            return None
        out = self._rewrite_text(sql)
        # the statement as a whole is not a transparent shape (or reads a CTE / view /
        # derived table, which has no identity of its own): a SELECT inside it may be
        if out is None and getattr(self, "_exact", False) and \
                self._last.reason in ("shape", "not_found", "view", "temp", "ambiguous", "double", "decimal"):
            whole = self._last.reason
            out = self._rewrite_nested(sql)
            if out is None and self._last.reason == "shape":
                self._last.reason = whole
        return out

    # ---- nested rewriting (§4.14): subqueries, derived tables and CTEs that are rewritable themselves ----
    _AGG_NAMES = ("sum", "count", "count_star", "min", "max", "avg")

    @classmethod
    def _is_aggregate_select(cls, node) -> bool:
        if not isinstance(node, dict) or node.get("type") != "SELECT_NODE":
            return False
        if node.get("group_expressions"):
            return True
        def has_agg(e):
            if isinstance(e, dict):
                if e.get("class") == "FUNCTION" and (e.get("function_name") or "").lower() in cls._AGG_NAMES:
                    return True
                return any(has_agg(v) for k, v in e.items() if k != "subquery")
            if isinstance(e, list):
                return any(has_agg(v) for v in e)
            return False
        return has_agg(node.get("select_list") or [])

    def _rewrite_nested(self, sql: str) -> Optional[str]:
        """The statement as a whole is not a transparent shape, but a SELECT
        inside it may be: `o_orderkey IN (SELECT l_orderkey FROM lineitem GROUP
        BY l_orderkey HAVING sum(l_quantity) > 300)`, a derived table, a CTE, a
        scalar subquery. Each such SELECT is rewritten on its own (its own
        decision, thresholds, residency and guards) and spliced back; the
        outer statement stays DuckDB's. A correlated subquery does not bind on
        its own and declines by itself."""
        key = (sql, self._settings_key)
        entry = self._nested_cache.get(key)
        if entry is False:
            self._last.reason = "shape"
            return None
        first = entry is None
        if first:
            try:
                tree = json.loads(self._serialize(sql))
            except Exception:
                self._nested_cache[key] = False
                return None
            stmts = tree.get("statements") or []
            if len(stmts) != 1:
                self._nested_cache[key] = False
                self._last.reason = "shape"
                return None
            subs: List[Tuple[list, str]] = []

            def walk(e, path, top):
                if isinstance(e, dict):
                    if not top and self._is_aggregate_select(e):
                        try:
                            sub_sql = self._raw.execute("SELECT json_deserialize_sql(?)", [json.dumps(
                                {"error": False, "statements": [{"node": e, "named_param_map": []}]})]).fetchone()[0]
                        except Exception:
                            sub_sql = None
                        if sub_sql:
                            self._rewrite_text(sub_sql)
                            if self._last.rewritten or self._last.reason == "not_resident":
                                subs.append((list(path), sub_sql))
                                return                      # rewritten as a whole: do not descend
                    for k, v in e.items():
                        walk(v, path + [k], False)
                elif isinstance(e, list):
                    for i, v in enumerate(e):
                        walk(v, path + [i], False)

            walk(stmts[0].get("node"), ["statements", 0, "node"], True)
            if not subs or len(self._nested_cache) > 512:
                self._nested_cache[key] = False
                self._last = LastRewrite(statement=sql, reason="shape")
                return None
            entry = {"tree": tree, "subs": subs, "final": {}, "decision": Decision(True, form="nested")}
            self._nested_cache[key] = entry
        d = entry["decision"]
        if not d.rewritten:
            self._last = LastRewrite(statement=sql, reason=d.reason or "threshold")
            return None
        outs, tags, pending = [], [], False
        for _path, sub_sql in entry["subs"]:
            o = self._rewrite_text(sub_sql)
            if o is None:
                pending = pending or self._last.reason == "not_resident"
                outs.append(None)
            else:
                outs.append(o)
                tags.append(self._last.tag)
        self._last = LastRewrite(statement=sql)
        self._timing_decision = None
        self._last_decision = None
        if not any(outs):
            self._last.reason = "not_resident" if pending else "threshold"
            if pending:
                self._timing_decision = d            # its native runs are this statement's native time
            return None
        fkey = tuple(outs)
        final = entry["final"].get(fkey)
        if final is None:
            try:
                tree = json.loads(json.dumps(entry["tree"]))
                for (path, _sub_sql), o in zip(entry["subs"], outs):
                    if o is None:
                        continue
                    node = json.loads(self._serialize(o))["statements"][0]["node"]
                    holder = tree
                    for k in path[:-1]:
                        holder = holder[k]
                    holder[path[-1]] = node
                final = self._raw.execute("SELECT json_deserialize_sql(?)", [json.dumps(tree)]).fetchone()[0]
                want = [(r[0], r[1]) for r in self._raw.execute("DESCRIBE " + sql).fetchall()]
                have = [(r[0], r[1]) for r in self._raw.execute("DESCRIBE " + final).fetchall()]
                if want != have:
                    raise ValueError(f"names / types changed: {want} -> {have}")
            except Exception as e:
                self._log(f"nested rewrite failed: {str(e)[:160]}")
                self._nested_cache[key] = False
                self._last.reason = "shape"
                return None
            if len(entry["final"]) < 32:
                entry["final"][fkey] = final
        self._timing_decision = d
        self._last.rewritten = True
        self._last.form = "nested"
        self._last.tag = tags[0] if tags else ""
        self._last_tags = tags
        self._last.engine = "nested"
        self._last.sql = final
        return final

    def _folded(self, sql: str) -> str:
        """§4.16: the statement with select-project-join derived tables folded
        into it — what the rewrite decides on and builds from. The original
        text is what runs when the rewrite declines."""
        flat = self._flat_cache.get(sql)
        if flat is None:
            flat = sql
            if "(" in sql or sql.lstrip()[:4].upper() == "WITH":
                try:
                    folded = None
                    tree = self._serialize(sql)
                    if _flatten.fold(tree) is not None:                 # cheap structural test first
                        want = [(r[0], r[1]) for r in self._raw.execute("DESCRIBE " + sql).fetchall()]
                        folded = _flatten.fold(tree, [w[0] for w in want])
                    if folded is not None:
                        cand = self._raw.execute("SELECT json_deserialize_sql(?)", [folded]).fetchone()[0]
                        have = [(r[0], r[1]) for r in self._raw.execute("DESCRIBE " + cand).fetchall()]
                        if want == have:
                            flat = cand
                        else:
                            self._log(f"fold: names / types changed ({want} -> {have}); not folded")
                except Exception as e:
                    self._log(f"fold failed: {str(e)[:120]}")
            if len(self._flat_cache) > 1024:
                self._flat_cache.clear()
            self._flat_cache[sql] = flat
        return flat

    def _rewrite_text(self, sql: str) -> Optional[str]:
        """The rewritten SQL of ONE statement, or None with _last.reason set."""
        self._last = LastRewrite(statement=sql)
        sql = self._folded(sql)
        t0 = time.perf_counter()
        template, literals = self._normalise(sql)
        key = (template, self._settings_key)
        d = self._cache.get(key)
        if d is None:
            d = self._decide(sql)
            d.literals = literals
            self._cache[key] = d
        elif d.literal_sensitive and literals != d.literals:
            # a computed lane embeds a literal (substr(s, 1, 2), a LIKE pattern,
            # an OR of comparisons): another literal tuple is another statement
            v = d.variants.get(literals)
            if v is None:
                if len(d.variants) >= 16:
                    self._last.reason = "threshold"
                    return None
                v = self._decide(sql)
                v.literals = literals
                d.variants[literals] = v
            d = v
        self._last.round_trip_ms = (time.perf_counter() - t0) * 1000.0
        if not d.rewritten:
            self._last.reason = d.reason
            return None
        self._timing_decision = d
        self._last.form = d.form
        self._last.tag = d.tag
        self._last_decision = d
        # residency
        if self._residency_mode == "manual":
            if not self._manager.is_ready(d.tag) and not self._extension_has_ready_set(d.tag):
                self._last.reason = "manual"
                return None
        elif not self._manager.is_ready(d.tag):
            if d.join is not None:
                for b in d.join.base:
                    if b.sentinel:
                        self._manager.note_candidate(b.tag, "", steps=[b.upload_sql])
                    else:
                        self._manager.note_candidate(b.tag, b.upload_sql, fqn=b.fqn)
            for b in d.sentinels:
                self._manager.note_candidate(b.tag, "", steps=[b.upload_sql])
            st = self._manager.note_candidate(d.tag, d.upload_sql, fqn=d.fqn,
                                              deps=([b.tag for b in d.join.base] if d.join is not None
                                                    else [b.tag for b in d.sentinels] or None),
                                              steps=d.join.steps_sql if d.join is not None else None)
            if self._residency_mode == "eager" and st.state == "pending":
                self._manager.upload_now(d.tag, lambda s: self._raw.execute(s).fetchall())
            if not self._manager.is_ready(d.tag):
                self._last.reason = "not_resident"
                return None
        if d.scalar_sql and literals == d.literals:
            out = d.scalar_sql
            self._last.engine = "scalar"
        else:
            # literals differ from the cached template (or no scalar): re-render
            # from the current statement's tree with the reference renderer
            # (one serialize, no catalog work, no scalar call)
            plan = d.plan
            if literals != d.literals and (plan.having is not None or plan.limit is not None or plan.where):
                plan = self._replan_literals(sql, plan)
                if plan is None:
                    self._last.reason = "shape"
                    return None
            out = _rewrite.render(plan, d.fqn, self._settings["default_order"])
            self._last.engine = "python"
        if d.wrap is not None:
            out = d.wrap[0] + out + d.wrap[1]
        self._last.rewritten = True
        self._last.sql = out
        return out

    def _extension_has_ready_set(self, tag: str) -> bool:
        try:
            row = self._raw.execute(
                "SELECT state FROM gpu_residents() WHERE name = ? AND origin = 'managed'",
                [tag]).fetchone()
        except Exception:
            return False
        if row and row[0] == "ready":
            self._manager.note_candidate(tag, "")
            self._manager.mark_ready(tag)
            return True
        return False

    @staticmethod
    def _normalise(sql: str) -> Tuple[str, Tuple[str, ...]]:
        lits: List[str] = []

        def sub(m):
            lits.append(m.group(0))
            return "'?'" if m.group(1) else "?"
        return _WS_RE.sub(" ", _LITERAL_RE.sub(sub, sql)).strip(), tuple(lits)

    def _serialize(self, sql: str) -> str:
        return self._raw.execute("SELECT json_serialize_sql(?)", [sql]).fetchone()[0]

    # ---- key joins (§4.8) ----
    def _is_unique(self, ident: _resolve.Identity, col: str) -> bool:
        """Is `col` unique among its non-NULL values? A PRIMARY KEY / UNIQUE
        constraint answers from the catalog; otherwise one scan, cached until
        the next statement that can change data."""
        key = (ident.oid, col)
        hit = self._unique_cache.get(key)
        if hit is not None:
            return hit
        ok = False
        try:
            rows = self._raw.execute(
                "SELECT constraint_column_names FROM duckdb_constraints() WHERE table_oid = ? "
                "AND constraint_type IN ('PRIMARY KEY', 'UNIQUE')", [ident.oid]).fetchall()
            ok = any(list(r[0] or []) == [col] for r in rows)
            if not ok:
                ok = bool(self._raw.execute(
                    f'SELECT count("{col}") = count(DISTINCT "{col}") FROM {ident.fqn}').fetchone()[0])
        except Exception as e:
            self._log(f"uniqueness probe failed: {e}")
            ok = False
        self._unique_cache[key] = ok
        return ok

    def _lower_join(self, tree: str) -> "_join.Lowered":
        return _join.lower(
            tree,
            lambda c, sch, t: _resolve.resolve(self._raw, c, sch, t),
            lambda ident: self._raw.execute(f"SELECT count(*) FROM {ident.fqn}").fetchone()[0],
            self._is_unique)

    # ---- computed lanes (§4.10) ----
    # Listed CONSISTENT, but they read session state a resident lane must not depend on.
    _SESSION_FUNCTIONS = {"current_setting", "getvariable", "getenv", "version", "current_schema", "current_schemas",
                          "current_database", "current_catalog", "current_user", "user", "session_user",
                          "current_query", "in_search_path", "current_role"}

    def _function_ok(self, name: str) -> bool:
        """Is `name` safe inside a computed lane? Every overload must be a
        scalar function DuckDB lists as CONSISTENT; a built-in macro (nullif,
        ...) is accepted when its definition names no function that is not."""
        if self._function_stability is None:
            ok: Dict[str, bool] = {}
            try:
                rows = self._raw.execute(
                    "SELECT lower(function_name), function_type, stability, internal, macro_definition "
                    "FROM duckdb_functions()").fetchall()
                bad = {r[0] for r in rows if r[1] == "scalar" and r[2] != "CONSISTENT"} | self._SESSION_FUNCTIONS
                macros: Dict[str, List[str]] = {}
                for fname, ftype, stab, internal, mdef in rows:
                    if ftype == "scalar":
                        ok[fname] = ok.get(fname, True) and stab == "CONSISTENT" and fname not in bad
                    elif ftype == "macro" and internal:
                        macros.setdefault(fname, []).append(mdef or "")
                    else:
                        ok[fname] = False
                for fname, defs in macros.items():
                    if fname in ok and not ok[fname]:
                        continue
                    tokens = set()
                    for d in defs:
                        tokens |= {t.lower() for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", d)}
                    ok[fname] = ok.get(fname, True) and not (tokens & bad) and "select" not in tokens
            except Exception as e:
                self._log(f"function catalog probe failed: {e}")
            self._function_stability = ok
        return self._function_stability.get(name.lower(), False)

    def _expr_to_sql(self, expr_json: str) -> str:
        if self._select_template is None:
            self._select_template = json.loads(self._serialize("SELECT 1"))
        stmt = json.loads(json.dumps(self._select_template))
        stmt["statements"][0]["node"]["select_list"] = [json.loads(expr_json)]
        return self._raw.execute("SELECT json_deserialize_sql(?)", [json.dumps(stmt)]).fetchone()[0]

    def _expr_type(self, fqn: str, sql: str) -> str:
        key = (fqn, sql)
        t = self._expr_types.get(key)
        if t is None:
            try:
                t = self._raw.execute(f"DESCRIBE SELECT {sql} AS c FROM {fqn}").fetchall()[0][1]
            except Exception as e:
                raise _rewrite.Decline("shape", f"expression does not bind: {str(e)[:80]}")
            self._expr_types[key] = t
        return t

    def _lower_exprs(self, tree: str, low: Optional["_join.Lowered"]):
        """(tree with expressions lowered to virtual columns, {name: Computed},
        single-table identity or None)."""
        upload = isinstance(low, _join.LoweredUpload)
        if upload:
            # §4.13: lanes are expressions over the JOINED row — any tables, qualified names
            columns = low.columns
            table_of = lambda c: 0                           # noqa: E731
            real_of = lambda c: low.tables[low.colmap[c][0]].alias + "\x00" + low.colmap[c][1]   # noqa: E731
        elif low is not None:
            columns = low.columns
            idents = [t.ident for t in low.tables]
            table_of = lambda c: low.colmap[c][0]            # noqa: E731
            real_of = lambda c: low.colmap[c][1]             # noqa: E731
        else:
            j = json.loads(tree)
            ft = ((j.get("statements") or [{}])[0].get("node") or {}).get("from_table") or {}
            if ft.get("type") != "BASE_TABLE":
                raise _rewrite.Decline("shape", "from is not a base table")
            ident, why = _resolve.resolve(self._raw, ft.get("catalog_name") or "", ft.get("schema_name") or "",
                                          ft.get("table_name") or "")
            if ident is None:
                raise _rewrite.Decline(why or "shape", "table")
            columns, idents = ident.columns, [ident]
            table_of = lambda c: 0                           # noqa: E731
            real_of = lambda c: c                            # noqa: E731
        if upload:
            src = low.from_text + (f" WHERE {low.edge_where}" if low.edge_where else "")
            lw = _exprs.Lowerer(columns, table_of, real_of, probe_of=real_of,
                                deserialize=self._expr_to_sql,
                                describe=lambda ti, sql: self._expr_type(src, sql),
                                function_ok=self._function_ok, identity=lambda ti: src,
                                resolve_table=lambda c, sch, t: _resolve.resolve(self._raw, c, sch, t))
        else:
            oa = _join.OUTER_ALIAS
            lw = _exprs.Lowerer(columns, table_of, real_of, probe_of=lambda c: c,
                                deserialize=self._expr_to_sql,
                                describe=lambda ti, sql: self._expr_type(f"{idents[ti].fqn} AS {oa}", sql),
                                function_ok=self._function_ok,
                                identity=lambda ti: idents[ti].fqn,
                                resolve_table=lambda c, sch, t: _resolve.resolve(self._raw, c, sch, t),
                                sql_qualifier=oa, probe_qualifier=oa if low is None else "gpudb_j0")
        return lw.lower(tree), lw.computed

    def _validate_join_condition(self, e) -> None:
        """An ON conjunct of an uploaded join (§4.13) is evaluated by DuckDB
        during the upload: it must be deterministic like a computed lane."""
        if isinstance(e, dict):
            cls = e.get("class")
            if cls in ("SUBQUERY", "WINDOW", "PARAMETER", "LAMBDA", "STAR"):
                raise _rewrite.Decline("shape", f"{cls} in a join condition")
            if cls == "FUNCTION" and not self._function_ok(e.get("function_name") or ""):
                raise _rewrite.Decline("shape", f"function {e.get('function_name')} in a join condition")
            for v in e.values():
                self._validate_join_condition(v)
        elif isinstance(e, list):
            for v in e:
                self._validate_join_condition(v)

    def _lower_join_upload(self, tree: str) -> "_join.LoweredUpload":
        return _join.lower_upload(
            tree,
            lambda c, sch, t: _resolve.resolve(self._raw, c, sch, t),
            lambda ident: self._raw.execute(f"SELECT count(*) FROM {ident.fqn}").fetchone()[0],
            lambda stmt: self._raw.execute("SELECT json_deserialize_sql(?)", [stmt]).fetchone()[0],
            self._validate_join_condition)

    def _match(self, sql: str, mode: str = "device"):
        """(plan, lowered join or None, computed lanes). Raises _rewrite.Decline.
        mode 'device': a join is materialised on the device (§4.8); 'upload':
        the join's result is uploaded (§4.13)."""
        tree = self._serialize(sql)
        order, nulls = self._settings["default_order"], self._settings["default_null_order"]
        try:
            return _rewrite.match(tree, order, nulls), None, {}
        except _rewrite.Decline as e:
            first = e
        low = None
        if _join.is_join_statement(tree):
            if not getattr(self, "_join", False):
                raise first
            low = self._lower_join(tree) if mode == "device" else self._lower_join_upload(tree)
            tree = low.tree_json
            try:
                return _rewrite.match(tree, order, nulls), low, {}
            except _rewrite.Decline as e:
                first = e
        if not getattr(self, "_exact", False):
            raise first
        tree2, computed = self._lower_exprs(tree, low)
        if not computed and tree2 == tree:
            raise first
        if low is not None:
            low.tree_json = tree2
        plan = _rewrite.match(tree2, order, nulls)
        plan.lowered_tree = tree2
        return plan, low, computed

    def _replan_literals(self, sql: str, cached: _rewrite.Plan) -> Optional[_rewrite.Plan]:
        try:
            plan, _low, _computed = self._match(sql, "upload" if cached.tag and ":joinu-" in cached.tag else "device")
        except _rewrite.Decline:
            return None
        if plan.keys != cached.keys:
            return None
        plan.guards = cached.guards
        plan.key_type, plan.val_type, plan.scale = cached.key_type, cached.val_type, cached.scale
        plan.outputs, plan.tag = cached.outputs, cached.tag
        plan.exact, plan.pred_types = cached.exact, cached.pred_types
        if plan.vals != cached.vals:
            return None
        plan.val_types, plan.scales = cached.val_types, cached.scales
        plan.keys, plan.key_types, plan.pack, plan.dict_key = cached.keys, cached.key_types, cached.pack, cached.dict_key
        if plan.pred_cols != cached.pred_cols:
            return None
        if plan.form == "topk" and cached.form != "topk":
            plan.form = cached.form
        return plan

    def _decide_split(self, sql: str) -> Optional[Decision]:
        """§4.11: answer the GROUP BY on the device and let DuckDB evaluate
        the expressions over its aggregates (and a compound HAVING) on top."""
        try:
            names = [r[0] for r in self._raw.execute("DESCRIBE " + sql).fetchall()]
            tmpl = json.loads(self._serialize(f"SELECT 1 FROM {_split.PLACEHOLDER}"))
            tree = self._serialize(sql)
            parts = _split.split(tree, names, tmpl)                       # §4.11 / §4.12
            reagg = parts is None
            if parts is None:
                parts = _split.split_distinct(tree, names, tmpl)          # §4.17
            if parts is None:
                return None
            inner_sql, outer_sql = (self._raw.execute("SELECT json_deserialize_sql(?)", [x]).fetchone()[0]
                                    for x in parts[:2])
            is_global = parts[2]
        except Exception as e:
            self._log(f"split failed: {e}")
            return None
        if outer_sql.count(_split.PLACEHOLDER) != 1:
            return None
        d = self._decide(inner_sql, allow_split=False, reagg=reagg)
        if not d.rewritten:
            self._log(f"split: the inner GROUP BY declined ({d.reason})")
            return Decision(False, d.reason)
        head, tail = outer_sql.split(_split.PLACEHOLDER)
        if is_global:
            # no GROUP BY: one row even when nothing qualifies, as native
            d.wrap = (head + "(SELECT 1) AS gpudb_one LEFT JOIN (", ") AS gpudb_q ON (true)" + tail)
        else:
            d.wrap = (head + "(", ") AS gpudb_q" + tail)
        d.literal_sensitive = True          # the outer text carries this statement's literals
        if d.form == "plain":
            d.form = "projected"
        return d

    def _decide(self, sql: str, allow_split: bool = True, reagg: bool = False) -> Decision:
        """Device path first; a join it cannot express falls back to uploading
        the join's result (§4.13); expressions over aggregates fall back to the
        split (§4.11), whose inner statement comes back through here."""
        d = self._decide_once(sql, "device", reagg)
        if d.rewritten or d.reason != "shape" or not getattr(self, "_exact", False):
            return d
        if d.is_join and getattr(self, "_join", False):
            du = self._decide_once(sql, "upload", reagg)
            if du.rewritten or du.reason != "shape":
                return du
        if allow_split:
            ds = self._decide_split(sql)
            if ds is not None:
                return ds
        return d

    def _decide_once(self, sql: str, mode: str, reagg: bool = False) -> Decision:
        d = self._decide_body(sql, mode, reagg)
        try:
            d.is_join = _join.is_join_statement(self._serialize(sql))
        except Exception:
            pass
        return d

    def _decide_body(self, sql: str, mode: str, reagg: bool = False) -> Decision:
        try:
            plan, low, computed = self._match(sql, mode)
        except _rewrite.Decline as e:
            if e.detail:
                self._log(f"declined ({e.reason}, {mode}): {e.detail}")
            return Decision(False, e.reason)
        except Exception as e:
            self._log(f"rewrite error: {e}")
            return Decision(False, "error")
        if low is None:
            ident, why = _resolve.resolve(self._raw, plan.catalog, plan.schema, plan.table)
            if ident is None:
                return Decision(False, why)
            columns, probe_from = ident.columns, ident.fqn
            idents = [ident]
            col_home = lambda c: (ident, c)                     # noqa: E731
            base_from = f"{ident.fqn} AS {_join.OUTER_ALIAS}"
        else:
            # a key join lowered to one virtual table: the root (fact) table
            # carries the identity, every column knows its own table
            ident = low.tables[low.root].ident
            columns = low.columns
            probe_from = (low.from_sql + " gpudb_j") if not isinstance(low, _join.LoweredUpload) else ""
            idents = [t.ident for t in low.tables]
            col_home = lambda c: (low.tables[low.colmap[c][0]].ident, low.colmap[c][1])   # noqa: E731
            base_from = (low.from_sql + " gpudb_j0") if not isinstance(low, _join.LoweredUpload) else ""
        upload_mode = isinstance(low, _join.LoweredUpload)
        if upload_mode:
            # §4.13: the set holds the join's result; how big is it? (one native join, once per template)
            where = f" WHERE {low.edge_where}" if low.edge_where else ""
            try:
                join_rows = self._raw.execute(f"SELECT count(*) FROM {low.from_text}{where}").fetchone()[0]
            except Exception as e:
                self._log(f"join count failed: {e}")
                return Decision(False, "error", is_join=True)
            biggest = max(t.rows for t in low.tables)
            if join_rows > min(4 * biggest, 0xFFFFFFFF - 64):
                self._log(f"declined (threshold): the join returns {join_rows} rows from tables of at most {biggest}")
                return Decision(False, "threshold", is_join=True)
            columns = dict(columns)
            for cname, comp in computed.items():
                columns[cname] = comp.lane_type
            names = list(dict.fromkeys(list(plan.keys or [plan.key]) + list(plan.vals or ([plan.val] if plan.val else []))
                                       + list(plan.pred_cols)))
            probe_from = low.derived(names, computed, with_rowid=False) + " gpudb_c"
        elif computed:
            # computed lanes (§4.10) are columns of the statement from here on;
            # decision-time probes read them from a derived table
            columns = dict(columns)
            for cname, comp in computed.items():
                columns[cname] = comp.lane_type
            proj = ", ".join(
                (f"CAST({c.probe_sql} AS TINYINT)" if c.native_type == "BOOLEAN" else c.probe_sql) + f' AS "{n}"'
                for n, c in computed.items())
            probe_from = f"(SELECT *, {proj} FROM {base_from}) gpudb_c"
        try:
            _rewrite.check_types(plan, columns, exact=getattr(self, "_exact", False))
        except _rewrite.Decline as e:
            return Decision(False, e.reason)
        # thresholds: the row count floor (§9.1); group estimate comes from
        # the resident set once it exists
        nrows = (max(t.rows for t in low.tables) if low is not None     # the largest joined table decides
                 else self._raw.execute(f"SELECT count(*) FROM {ident.fqn}").fetchone()[0])
        if nrows < self._floor_rows:
            return Decision(False, "threshold")
        # NULLs and the overflow bound from zonemap statistics
        stats: Dict[str, Dict[str, Any]] = {}
        stat_cols = list(plan.keys or [plan.key]) + ([plan.val] if plan.val else []) + list(plan.pred_cols)
        for col in stat_cols:
            if col in computed:
                st = None
                if col in (plan.keys or [plan.key]):
                    # a computed key: bounds and the distinct estimate from one scan
                    try:
                        mn, mx, uq, nn = self._raw.execute(
                            f'SELECT min("{col}"), max("{col}"), approx_count_distinct("{col}"), '
                            f'count(*) - count("{col}") FROM {probe_from}').fetchone()
                        st = _resolve.ColumnStats(has_null=bool(nn), min=None if mn is None else str(mn),
                                                  max=None if mx is None else str(mx), approx_unique=int(uq))
                    except Exception as e:
                        self._log(f"computed key statistics failed: {e}")
            else:
                st = _resolve.column_stats(self._raw, *col_home(col))
            if plan.exact:
                # the exact path keeps NULLs and never wraps: statistics are
                # informative only (thresholds), never a gate
                if st is not None:
                    stats[col] = {"has_null": st.has_null, "min": _num(st.min), "max": _num(st.max),
                                  "approx_unique": st.approx_unique,
                                  "min_raw": st.min, "max_raw": st.max}   # temporal bounds stay text
                continue
            if st is None or st.has_null:
                return Decision(False, "nulls")
            stats[col] = {"has_null": st.has_null, "min": _num(st.min), "max": _num(st.max),
                          "approx_unique": st.approx_unique}
            if col == plan.val and plan.needs_sum:
                try:
                    bound = max(abs(float(st.min)), abs(float(st.max))) * (10 ** plan.scale)
                except ValueError:
                    return Decision(False, "overflow")
                if nrows * bound >= 2.0 ** 63:
                    return Decision(False, "overflow")
        if plan.dict_key or any(t in _rewrite._STRING_TYPES for t in plan.pred_types.values()):
            # byte-wise dictionary: only under binary collation, column and session
            coll = (self._settings["default_collation"] or "").lower()
            if coll and coll != "binary":
                return Decision(False, "collation")
            try:
                for idn in idents:
                    ddl = self._raw.execute("SELECT sql FROM duckdb_tables() WHERE table_oid = ?", [idn.oid]).fetchone()
                    if ddl and ddl[0] and "COLLATE" in ddl[0].upper():
                        return Decision(False, "collation")
            except Exception:
                return Decision(False, "collation")
        # packed keys (§4.4): each component's integer image bounds from stats()
        if plan.exact and plan.packed:
            pack = []
            for kc, kt in zip(plan.keys, plan.key_types):
                st = stats.get(kc) or {}
                lo, hi = _rewrite.stat_image(st.get("min_raw"), kt), _rewrite.stat_image(st.get("max_raw"), kt)
                if lo is None or hi is None or hi < lo:
                    return Decision(False, "shape")
                pack.append((lo, hi - lo + 2))
            stride, packs = 1, []
            for mn, rng in reversed(pack):
                if stride > (2 ** 63 - 1) // rng:
                    return Decision(False, "shape")      # does not fit in 64 bits: native
                packs.append((mn, rng, stride))
                stride *= rng
            plan.pack = list(reversed(packs))
        # per-backend thresholds (§9.1): distinct-count estimate of the key and,
        # under a WHERE, the selectivity of THIS statement's literals (one
        # count(*) scan at decision time, cached with the template)
        if plan.exact and getattr(self, "_thresholds", True):
            est = (stats.get(plan.key) or {}).get("approx_unique")
            if len(plan.keys) > 1:
                est = 1
                for kc in plan.keys:
                    u = (stats.get(kc) or {}).get("approx_unique")
                    est = None if (u is None or est is None) else est * u
                if est is not None:
                    est = min(est, nrows)
            if est is None:
                # no zone-map estimate (VARCHAR columns carry none): count the key's distinct
                # values once — one scan per statement template
                try:
                    cols = ", ".join(f'"{kc}"' for kc in plan.keys)
                    src = probe_from if probe_from else ident.fqn
                    est = int(self._raw.execute(
                        f"SELECT approx_count_distinct(hash({cols})) FROM {src}").fetchone()[0])
                except Exception as e:
                    self._log(f"distinct-count probe failed: {e}")
            sel = None
            if plan.where:
                try:
                    kept, total = self._raw.execute(
                        f"SELECT count(*) FILTER (WHERE {_rewrite.where_sql(plan)}), count(*) "
                        f"FROM {probe_from}").fetchone()
                    sel = (kept / total) if total else 0.0
                    # there cannot be more groups than rows that survive the WHERE (the
                    # product of several keys' distinct counts overshoots wildly under a filter)
                    if est is not None:
                        # few rows survive next to the number of distinct key tuples: a dictionary
                        # key decodes per returned row, not by joining the whole dictionary
                        # (Q18: 57 groups out of 1.5M entries, 416 ms vs 4 ms; all 1.5M groups
                        # out: 1410 ms joined vs 2200 ms per key — the crossover is near a third)
                        plan.decode_per_key = bool(plan.dict_key) and int(kept) * 4 <= est
                        est = max(1, min(est, int(kept)))
                except Exception as e:
                    self._log(f"selectivity probe failed: {e}")
                    return Decision(False, "threshold")
            # a lane with a subquery (EXISTS / IN / correlated scalar, §4.18) makes native run a
            # join too, whatever the FROM says: the join bounds apply
            ok, why = _thresholds.decide(self._backend, plan.form, est, sel, bool(plan.where),
                                         join=low is not None or any(c.dep_tables for c in computed.values()), payloads=max(1, len(plan.vals)),
                                         string_key=bool(plan.dict_key),
                                         limited=plan.limit is not None and plan.limit <= 10_000,
                                         computed_payloads=sum(1 for v in set(plan.vals) if v in computed),
                                         reaggregated=reagg)
            if not ok:
                self._log(f"threshold: {why}")
                return Decision(False, "threshold")
        try:
            described = self._raw.execute("DESCRIBE " + sql).fetchall()
            _rewrite.apply_describe(plan, [(r[0], r[1]) for r in described])
        except _rewrite.Decline as e:
            return Decision(False, e.reason)
        except Exception as e:
            self._log(f"describe failed: {e}")
            return Decision(False, "error")
        q = (lambda c: computed[c].sql if c in computed else f'"{c}"')   # noqa: E731
        if low is None:
            try:
                plan.tag = ident.tag(plan.upload_columns)
            except ValueError:
                return Decision(False, "shape")
            d = Decision(True, "", plan=plan, fqn=ident.fqn, tag=plan.tag,
                         upload_sql=_rewrite.upload_sql(plan, f"{ident.fqn} AS {_join.OUTER_ALIAS}", q),
                         form=plan.form)
        else:
            try:
                jr = (_join.plan_upload_residency(low, plan, computed) if isinstance(low, _join.LoweredUpload)
                      else _join.plan_residency(low, plan, computed))
            except _rewrite.Decline as e:
                self._log(f"declined ({e.reason}): {e.detail}")
                return Decision(False, e.reason)
            except ValueError:
                return Decision(False, "shape")          # an identifier the tag cannot carry
            plan.tag = jr.tag
            plan.guards = [(g, f) for g, f, _i in jr.guards]
            d = Decision(True, "", plan=plan, fqn=jr.root_fqn or ident.fqn, tag=plan.tag,
                         upload_sql=jr.upload_sql, form=plan.form, join=jr)
        d.literal_sensitive = any(c.has_constant for c in computed.values())
        # §4.18: tables read only inside subquery lanes get a row-count sentinel and a guard
        covered = {(i.catalog, i.oid) for i in idents}
        extra: List[_resolve.Identity] = []
        for c in computed.values():
            for t in c.dep_tables:
                if (t.catalog, t.oid) not in covered:
                    covered.add((t.catalog, t.oid))
                    extra.append(t)
        if extra:
            try:
                sent = [_join.BaseSet(table=0, tag=t.tag(["rows"]) + ":sentinel", fqn=t.fqn, sentinel=True,
                                      upload_sql="SELECT gpu_note_rows('%s', (SELECT count(*) FROM %s))" % (
                                          (t.tag(["rows"]) + ":sentinel").replace("'", "''"), t.fqn))
                        for t in extra]
            except ValueError:
                return Decision(False, "shape")
            if d.join is not None:
                d.join.base = list(d.join.base) + sent
                d.join.guards = list(d.join.guards) + [(b.tag, b.fqn, t) for b, t in zip(sent, extra)]
                plan.guards = [(g, f) for g, f, _i in d.join.guards]
            else:
                d.sentinels = sent
                plan.guards = [(plan.tag, ident.fqn)] + [(b.tag, b.fqn) for b in sent]
        if self._has_rewrite_scalar:
            # The extension's pure scalar is the authority on the decision and
            # renders the statement for these literals; `ready` is passed as
            # true because residency is enforced here, not in the scalar, and
            # the decision must not depend on it.
            ctx = {
                "tag": plan.tag,
                "exact": plan.exact,
                "keys": [{"name": kc, "type": kt, **({"min": mn, "max": mn + rng - 2} if plan.packed else {})}
                         for kc, kt, (mn, rng, _st) in zip(plan.keys, plan.key_types,
                                                          plan.pack if plan.packed else [(0, 2, 1)] * len(plan.keys))],
                "table": {"catalog": ident.catalog, "schema": ident.schema, "name": ident.table,
                          "oid": ident.oid},
                "columns": {c: {"type": t, "scale": (_rewrite.decimal_scale(t) or (0, 0))[1]}
                            for c, t in columns.items()},
                "backend": self._backend, "rows": nrows, "ready": True,
                "decode_per_key": bool(getattr(plan, "decode_per_key", False)),
                "default_order": "DESC" if self._settings["default_order"].upper().startswith("DESC") else "ASC",
                "default_null_order": self._settings["default_null_order"],
                "default_collation": self._settings["default_collation"],
                "outputs": [{"name": o.name, "type": o.native_type} for o in plan.outputs],
                "thresholds": {"min_rows": self._floor_rows},
                "stats": stats,
            }
            if low is not None:
                ctx["guards"] = [{"tag": g, "catalog": i.catalog, "schema": i.schema, "table": i.table}
                                 for g, _f, i in d.join.guards]
            elif d.sentinels:
                ctx["guards"] = [{"tag": plan.tag, "catalog": ident.catalog, "schema": ident.schema,
                                  "table": ident.table}] + \
                                [{"tag": b.tag, "catalog": t.catalog, "schema": t.schema, "table": t.table}
                                 for b, t in zip(d.sentinels, extra)]
            try:
                tree = (getattr(plan, "lowered_tree", None)
                        or (low.tree_json if low is not None else self._serialize(sql)))
                row = self._raw.execute(
                    "SELECT r, json_deserialize_sql(r) FROM (SELECT gpu_rewrite_ast(?, ?) AS r) s",
                    [tree, json.dumps(ctx)]).fetchone()
                info = (json.loads(row[0]).get("gpudb") or {})
            except Exception as e:
                self._log(f"gpu_rewrite_ast failed: {e}")
                return Decision(False, "error")
            if not info.get("rewritten"):
                if info.get("reason") in ("not_resident", "threshold"):
                    pass                       # static checks passed; residency is ours
                else:
                    self._log(f"scalar declined ({info.get('reason')}: {info.get('detail', '')}); "
                              f"reference matcher accepted — following the scalar")
                    return Decision(False, info.get("reason") or "shape")
            else:
                d.scalar_sql = row[1]
                d.form = info.get("form") or d.form
        return d


def connect(database: str = ":memory:", read_only: bool = False, config: Optional[dict] = None,
            *, extension: Optional[str] = None, transparent: bool = True,
            residency: str = "background", floor_rows: int = 1_000_000,
            idle_ms: float = 20.0, thresholds: bool = True, log=None) -> Connection:
    """duckdb.connect with the gpudb extension loaded and the transparent path
    on. `residency`: 'background' (upload in short row-id segments, each only
    while the connection is idle for `idle_ms`; §5.5), 'eager' (upload on first
    sighting, synchronously), 'manual' (v0.6 behaviour: only sets uploaded
    under an identity tag are used). `floor_rows`: tables smaller than this
    are never parsed (§0). `thresholds`: apply the per-backend shape thresholds
    (§9.1, gpudb/_thresholds.py); False rewrites every exact shape regardless
    of the predicted win — for parity testing, never for production."""
    if residency not in ("background", "eager", "manual"):
        raise ValueError("residency must be 'background', 'eager' or 'manual'")
    cfg = dict(config or {})
    ext = _find_extension(extension)
    if ext:
        cfg.setdefault("allow_unsigned_extensions", "true")
    raw = duckdb.connect(database, read_only=read_only, config=cfg)
    if ext:
        raw.execute(f"LOAD '{ext}'")
    con = Connection(raw, transparent=transparent, residency=residency,
                     floor_rows=floor_rows, idle_ms=idle_ms, log=log)
    con._thresholds = thresholds
    return con
