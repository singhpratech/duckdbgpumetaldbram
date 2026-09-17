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

from . import _classify, _join, _resolve, _rewrite, _thresholds
from ._residency import ResidencyManager

GPUDB_EXTENSION_ENV = "GPUDB_EXTENSION_PATH"
_GROUP_BY_RE = re.compile(r"\bGROUP\s+BY\b", re.IGNORECASE)
_SELECT_START_RE = re.compile(r"^\s*(SELECT|FROM|VALUES)\b", re.IGNORECASE)
_TABLE_REF_RE = re.compile(r'\bFROM\s+((?:"[^"]+"|[A-Za-z_][A-Za-z0-9_]*)(?:\.(?:"[^"]+"|[A-Za-z_][A-Za-z0-9_]*)){0,2})', re.IGNORECASE)
_MAX_STATEMENT_BYTES = 16 * 1024
_LITERAL_RE = re.compile(r"""('(?:[^']|'')*')|(\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b)""")
_WS_RE = re.compile(r"\s+")
STALE_MARKER = "GPUDB_STALE"
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

    def execute(self, query, parameters=None):
        sql = self._route(query, parameters)
        self._manager.statement_begin()
        try:
            try:
                self._raw.execute(sql, parameters)
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
        tag = self._last.tag
        if tag:
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
        self._unique_cache.clear()
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
        big = self._big_tables if self._big_tables is not None else self._refresh_big_tables()
        if not big:
            return False
        for m in _TABLE_REF_RE.finditer(sql):
            last = m.group(1).split(".")[-1].strip('"')
            if last in big:
                return True
        return False

    def _route(self, query: Any, parameters) -> Any:
        """Return the SQL to run in place of `query`."""
        self._last = LastRewrite(statement=query if isinstance(query, str) else "")
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
                not _GROUP_BY_RE.search(query) or not self._names_big_table(query)):
            self._last.reason = "threshold" if _GROUP_BY_RE.search(query) else "shape"
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
        if not _GROUP_BY_RE.search(sql):
            self._last.reason = "shape"
            return None
        t0 = time.perf_counter()
        template, literals = self._normalise(sql)
        key = (template, self._settings_key)
        d = self._cache.get(key)
        if d is None:
            d = self._decide(sql)
            d.literals = literals
            self._cache[key] = d
        self._last.round_trip_ms = (time.perf_counter() - t0) * 1000.0
        if not d.rewritten:
            self._last.reason = d.reason
            return None
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
                    self._manager.note_candidate(b.tag, b.upload_sql, fqn=b.fqn)
            st = self._manager.note_candidate(d.tag, d.upload_sql, fqn=d.fqn,
                                              deps=[b.tag for b in d.join.base] if d.join is not None else None,
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

    def _match(self, sql: str):
        """(plan, lowered join or None). Raises _rewrite.Decline."""
        tree = self._serialize(sql)
        try:
            return _rewrite.match(tree, self._settings["default_order"],
                                  self._settings["default_null_order"]), None
        except _rewrite.Decline:
            if not (getattr(self, "_join", False) and _join.is_join_statement(tree)):
                raise
        low = self._lower_join(tree)
        return _rewrite.match(low.tree_json, self._settings["default_order"],
                              self._settings["default_null_order"]), low

    def _replan_literals(self, sql: str, cached: _rewrite.Plan) -> Optional[_rewrite.Plan]:
        try:
            plan, _low = self._match(sql)
        except _rewrite.Decline:
            return None
        plan.guards = cached.guards
        plan.key_type, plan.val_type, plan.scale = cached.key_type, cached.val_type, cached.scale
        plan.outputs, plan.tag = cached.outputs, cached.tag
        plan.exact, plan.pred_types = cached.exact, cached.pred_types
        plan.keys, plan.key_types, plan.pack, plan.dict_key = cached.keys, cached.key_types, cached.pack, cached.dict_key
        if plan.pred_cols != cached.pred_cols:
            return None
        if plan.form == "topk" and cached.form != "topk":
            plan.form = cached.form
        return plan

    def _decide(self, sql: str) -> Decision:
        try:
            plan, low = self._match(sql)
        except _rewrite.Decline as e:
            if e.detail:
                self._log(f"declined ({e.reason}): {e.detail}")
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
        else:
            # a key join lowered to one virtual table: the root (fact) table
            # carries the identity, every column knows its own table
            ident = low.tables[low.root].ident
            columns, probe_from = low.columns, low.from_sql + " gpudb_j"
            idents = [t.ident for t in low.tables]
            col_home = lambda c: (low.tables[low.colmap[c][0]].ident, low.colmap[c][1])   # noqa: E731
        try:
            _rewrite.check_types(plan, columns, exact=getattr(self, "_exact", False))
        except _rewrite.Decline as e:
            return Decision(False, e.reason)
        # thresholds: the row count floor (§9.1); group estimate comes from
        # the resident set once it exists
        nrows = (low.tables[low.root].rows if low is not None
                 else self._raw.execute(f"SELECT count(*) FROM {ident.fqn}").fetchone()[0])
        if nrows < self._floor_rows:
            return Decision(False, "threshold")
        # NULLs and the overflow bound from zonemap statistics
        stats: Dict[str, Dict[str, Any]] = {}
        stat_cols = list(plan.keys or [plan.key]) + ([plan.val] if plan.val else []) + list(plan.pred_cols)
        for col in stat_cols:
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
            if plan.packed:
                est = 1
                for kc in plan.keys:
                    u = (stats.get(kc) or {}).get("approx_unique")
                    est = None if (u is None or est is None) else est * u
                if est is not None:
                    est = min(est, nrows)
            sel = None
            if plan.where:
                try:
                    kept, total = self._raw.execute(
                        f"SELECT count(*) FILTER (WHERE {_rewrite.where_sql(plan)}), count(*) "
                        f"FROM {probe_from}").fetchone()
                    sel = (kept / total) if total else 0.0
                except Exception as e:
                    self._log(f"selectivity probe failed: {e}")
                    return Decision(False, "threshold")
            ok, why = _thresholds.decide(self._backend, plan.form, est, sel, bool(plan.where),
                                         join=low is not None)
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
        if low is None:
            plan.tag = ident.tag(plan.upload_columns)
            d = Decision(True, "", plan=plan, fqn=ident.fqn, tag=plan.tag,
                         upload_sql=_rewrite.upload_sql(plan, ident.fqn), form=plan.form)
        else:
            try:
                jr = _join.plan_residency(low, plan)
            except _rewrite.Decline as e:
                self._log(f"declined ({e.reason}): {e.detail}")
                return Decision(False, e.reason)
            except ValueError:
                return Decision(False, "shape")          # an identifier the tag cannot carry
            plan.tag = jr.tag
            plan.guards = [(g, f) for g, f, _i in jr.guards]
            d = Decision(True, "", plan=plan, fqn=ident.fqn, tag=plan.tag, upload_sql="",
                         form=plan.form, join=jr)
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
            try:
                tree = low.tree_json if low is not None else self._serialize(sql)
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
