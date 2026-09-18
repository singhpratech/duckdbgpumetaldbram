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

from . import (_aggs, _classify, _exprs, _flatten, _join, _probation, _resolve, _rewrite, _split,
               _syntax, _thresholds, _views)
from ._residency import MEMORY_ERROR, ResidencyManager

GPUDB_EXTENSION_ENV = "GPUDB_EXTENSION_PATH"
_GROUP_BY_RE = re.compile(r"\bGROUP\s+BY\b", re.IGNORECASE)
# an aggregate without GROUP BY (§4.12). Over a join it was always worth
# parsing; over a SINGLE table it is worth parsing since the global masked
# aggregate exists (one fused pass, no key, no sort cache) — on a build whose
# backend reports that operator, which is what the wider regex is gated on.
_GLOBAL_AGG_JOIN_RE = re.compile(
    r"\b(?:sum|count|min|max|avg)\s*\(.*\bFROM\b.*(?:\bJOIN\b|,)", re.IGNORECASE | re.DOTALL)
_GLOBAL_AGG_RE = re.compile(
    r"\b(?:sum|count|min|max|avg)\s*\(.*\bFROM\b", re.IGNORECASE | re.DOTALL)


_SELECT_DISTINCT_RE = re.compile(r"\bSELECT\s+DISTINCT\b(?!\s+ON\b)", re.IGNORECASE)   # a GROUP BY in disguise (§4.21)


def _maybe_aggregate(sql: str, single_global: bool = False) -> bool:
    glob = _GLOBAL_AGG_RE if single_global else _GLOBAL_AGG_JOIN_RE
    return bool(_GROUP_BY_RE.search(sql) or glob.search(sql) or _SELECT_DISTINCT_RE.search(sql))
_SELECT_START_RE = re.compile(r"^\s*(SELECT|FROM|VALUES)\b", re.IGNORECASE)
_TABLE_REF_RE = re.compile(r'\bFROM\s+((?:"[^"]+"|[A-Za-z_][A-Za-z0-9_]*)(?:\.(?:"[^"]+"|[A-Za-z_][A-Za-z0-9_]*)){0,2})', re.IGNORECASE)
_MAX_STATEMENT_BYTES = 16 * 1024
_LITERAL_RE = re.compile(r"""('(?:[^']|'')*')|(\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b)""")
_WS_RE = re.compile(r"\s+")
STALE_MARKER = "GPUDB_STALE"
# measured rule 1 (§9.1): a template is re-measured against native at most this often — one
# side-cursor probe per template per interval, the user's own statement never the experiment
_REMEASURE_S = 60.0
REASONS = ("shape", "not_resident", "threshold", "backend", "double", "nulls", "overflow",
           "decimal", "collation", "too_long", "transaction", "view", "temp", "ambiguous",
           "not_found", "manual", "error", "off", "params", "multi", "memory")


def _host_memory_bytes() -> int:
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return 0


def parse_memory_budget(value) -> Optional[int]:
    """None -> the default for the backend; 0 / 'unlimited' -> no cap; an int is
    bytes; a string takes a unit: '512MB', '16GB', '1.5 GiB'."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if value < 0:
            raise ValueError("memory_budget must be >= 0")
        return int(value)
    m = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*([KMGT]?)I?B?\s*", str(value).upper())
    if str(value).strip().lower() in ("unlimited", "none", "off"):
        return 0
    if not m:
        raise ValueError(f"memory_budget: cannot read {value!r} (examples: 8589934592, '512MB', '16GB', 'unlimited')")
    return int(float(m.group(1)) * 1024 ** " KMGT".index(m.group(2) or " "))


def default_memory_budget(backend: str, device_bytes: int = 0) -> int:
    """§5.5: a quarter of unified memory on Apple silicon (the GPU shares it
    with DuckDB and everything else), half of device memory on a discrete GPU.
    `device_bytes` is what gpu_build_info() reports (Metal: the recommended
    working-set size; CUDA: total device memory once that backend reports it).
    When it is 0, a discrete GPU gets the smaller of a quarter of host memory
    and 8 GiB — conservative on purpose. GPUDB_MEMORY_BUDGET_MB overrides."""
    env = os.environ.get("GPUDB_MEMORY_BUDGET_MB")
    if env:
        try:
            return max(0, int(float(env) * 2 ** 20))
        except ValueError:
            pass
    host = _host_memory_bytes()
    if backend == "CUDA":
        if device_bytes > 0:
            return device_bytes // 2
        return min(host // 4, 8 * 2 ** 30) if host else 8 * 2 ** 30
    # unified memory: a quarter of host memory; never above what Metal recommends keeping resident
    quarter = host // 4 if host else 8 * 2 ** 30
    return min(quarter, device_bytes) if device_bytes > 0 else quarter


# An UPPER bound on a lane's storage width, from the DuckDB type of the column
# it holds (docs/RESIDENT_COLUMNS_DESIGN.md §6: since stage C a lane is stored
# at the narrowest signed width its values fit, which the wrapper cannot know
# before the upload — the type is what bounds it). A computed lane, a DECIMAL
# image and a string hash fill the range and take 8.
_LANE_WIDTH = {"BOOLEAN": 1, "TINYINT": 1, "UTINYINT": 1,
               "SMALLINT": 2, "USMALLINT": 2,
               "INTEGER": 4, "UINTEGER": 4, "DATE": 4}


def lane_width(col_type: str) -> int:
    return _LANE_WIDTH.get((col_type or "").upper(), 8)


def estimate_set_bytes(rows: int, lanes: int, widths=None, key_width: int = 8) -> int:
    """What a resident set costs on the device: the lanes, a validity bit per
    row and lane, the key's sort cache (the sorted key plus a u32 row id per
    row) and one row-sized scratch lane the exact operators keep beside it.
    `widths` is the per-lane upper bound in bytes (see lane_width) and is only
    passed on a backend that stores lanes narrow; without it every lane is
    charged 8. `key_width` 0 = the set has no key (§4.12): no sort cache.
    It must stay an UPPER bound — it is the budget's admission rule."""
    rows, lanes = max(0, int(rows)), max(1, int(lanes))
    if widths:
        lanes = max(lanes, len(widths))
        per = sum(max(1, int(w)) for w in widths)
    else:
        per = 8 * lanes
    cache = (max(1, int(key_width)) + 4) if key_width else 0
    return rows * (per + cache + 8) + (rows * lanes) // 8 + (64 << 10)


def _tag_lanes(tag: str) -> int:
    """Lanes of a set from its identity tag (…:oid:<k>,<v>,<pred>,…[:suffix]); a
    several-column key (a+b+c) is one lane."""
    parts = tag.split(":")
    return parts[6].count(",") + 1 if len(parts) > 6 and parts[6] else 1


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
    set_rows: int = 0                                     # §5.5: rows the resident set will hold (the memory budget's estimate)
    store_key: str = ""                                   # stage B: the table store a single-table set is a view over
    store_lanes: List[str] = field(default_factory=list)  # ... and the lanes it reads there
    store_all: List[tuple] = field(default_factory=list)  # (name, sql, kind) of every lane, for the upload of missing ones
    # §5.5 + docs/RESIDENT_COLUMNS_DESIGN.md §6: an upper bound on each store
    # lane's storage width, from its DuckDB type, and which lane carries the
    # sort cache ("" = none, §4.12). Only used where the backend stores narrow.
    lane_widths: Dict[str, int] = field(default_factory=dict)
    key_lane: str = ""
    base_rows: Dict[str, int] = field(default_factory=dict)   # ... and of each base set of a device join, by tag
    # measured rule 1 (§9.1): this statement's own native time, seen while it
    # was not resident yet, against its first rewritten runs
    native_ms: Optional[float] = None
    rewritten_ms: List[float] = field(default_factory=list)
    timing_checked: bool = False
    measured_declined: bool = False      # sent native by a measurement, not by the thresholds
    next_check_at: float = 0.0           # when the next side-cursor probe may run
    probe_sql: str = ""                  # the rewritten statement, for the probe of a declined template
    variants: Dict[Tuple[str, ...], "Decision"] = field(default_factory=dict)
    # probation (§9.1, gpudb/_probation.py): a template only a SOFT threshold
    # keeps on DuckDB. `threshold_kind` is what _thresholds.decide said about
    # the bound that declined it ('hard' | 'soft'); `probation` is where the
    # trial stands ('' = not on probation, else one of _probation.STATES).
    threshold_kind: str = ""
    threshold_why: str = ""
    probation: str = ""
    native_obs: List[float] = field(default_factory=list)   # native ms of the user's OWN runs, newest last
    sightings: int = 0                   # times the user ran this template (admission, §9.1)
    round_probes: List[float] = field(default_factory=list)   # probes of the round in progress
    probe_rounds: int = 0                # rounds run
    probe_wins: int = 0                  # consecutive winning rounds
    probes_spent: int = 0
    probe_ms_spent: float = 0.0
    probe_ms: Optional[float] = None     # the last round's figure (the median of its probes)
    next_probe_at: float = 0.0
    backoff_s: float = 0.0


@dataclass
class LastRewrite:
    statement: str = ""
    rewritten: bool = False
    reason: str = ""
    form: str = ""
    tag: str = ""
    sql: str = ""
    fallback: bool = False       # the rewritten statement raised (GPUDB_STALE or any other error), native re-run
    error: str = ""              # ... the error text, when it was not staleness
    round_trip_ms: float = 0.0
    engine: str = ""             # 'scalar' (gpu_rewrite_ast) | 'python' (reference renderer)
    # More about `reason`, never instead of it. For reason 'threshold' it says
    # which kind of bound declined the statement ('hard: …' / 'soft: …') and,
    # once the shape is being tried on a side cursor, where that stands:
    # 'probation', 'promoted' or 'demoted', with the measured pair (§9.1).
    detail: str = ""

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
                 idle_ms: float = 20.0, log=None, memory_budget=None, read_only: bool = False,
                 _parent: Optional["Connection"] = None):
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
        self._flat_views: Dict[str, Dict[str, str]] = {}         # statement text -> {view: definition} it was built on (§4.20)
        self._split_cache: Dict[str, Any] = {}                   # statement text -> DuckDB's split of it (§5.2)
        self._plans: Dict[str, str] = {}                         # rendered statement -> the name it is PREPAREd under
        self._plan_seen: Dict[str, bool] = {}                    # rendered statements seen once (see _planned)
        self._plan_seq = 0
        self._plan_mu = threading.Lock()
        self._resnap_after = False                                # take a new file snapshot after this statement (our own write)
        self._read_only = False
        self._watch_files: List[str] = []
        self._write_snapshot: tuple = ()
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
            self._probe_budget = _probation.Budget()
            self._manager = ResidencyManager(lambda: self._raw.cursor(), mode=residency,
                                             idle_ms=idle_ms, log=self._log)
            self._refresh_settings()
            self._probe_extension()
            self._read_only = bool(read_only)
            self._watch_files = self._watched_files()
            self._write_snapshot = self._snapshot_files()
            budget = parse_memory_budget(memory_budget)
            if budget is None:
                budget = default_memory_budget(self._backend, getattr(self, "_device_bytes", 0))
            self._manager.memory_budget = budget or None       # 0 = no cap
        else:
            self._probe_budget = _parent._probe_budget      # one probe budget per process
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

    def memory(self) -> Dict[str, Any]:
        """The memory budget (§5.5) and, per resident set, the size the wrapper
        expected and the size the extension reports."""
        return self._manager.memory()

    def residents(self) -> Dict[str, str]:
        return self._manager.snapshot()

    def probation(self) -> Dict[str, Any]:
        """The shapes a SOFT threshold keeps on DuckDB that are being TRIED on a
        side cursor (§9.1, gpudb/_probation.py), one entry per statement
        template of this connection: `state` (candidate / waiting / probing /
        promoted / demoted / retired), `rounds` and `wins`, `probes` and
        `probe_ms` spent on it, `round_ms` (the last round's median) against
        `native_ms` (the median of the user's own runs), the `bytes` its
        resident set holds, and the bound that declined it. `budget` is the process-wide probe budget and
        what of it the last minute spent."""
        now = time.monotonic()
        mem = self._manager.memory().get("sets") or {}
        out: Dict[str, Any] = {}
        for (template, _settings), d in list(self._cache.items()):
            for dd in [d] + list(d.variants.values()):
                if not dd.probation:
                    continue
                native = _probation.median(dd.native_obs)
                best = dd.probe_ms
                out[template] = {
                    "state": dd.probation, "form": dd.form, "tag": dd.tag,
                    "rounds": dd.probe_rounds, "wins": dd.probe_wins,
                    "probes": dd.probes_spent, "probe_ms": round(dd.probe_ms_spent, 3),
                    "round_ms": None if best is None else round(best, 3),
                    "native_ms": None if native is None else round(native, 3),
                    "ratio": None if not (best and native) else round(native / best, 3),
                    "bytes": int((mem.get(dd.tag) or {}).get("bytes") or 0),
                    "resident": self._manager.is_ready(dd.tag),
                    "threshold": f"{dd.threshold_kind}: {dd.threshold_why}" if dd.threshold_why else dd.threshold_kind,
                    "next_probe_in_s": max(0.0, round(dd.next_probe_at - now, 1)),
                }
        return {"templates": out,
                "budget": {"ms_per_min": self._probe_budget.ms_per_min,
                           "spent_ms_last_minute": round(self._probe_budget.spent_ms(now), 3)},
                "bytes": self._manager.probation_bytes()}

    def _refresh_settings(self) -> None:
        row = self._raw.execute(
            "SELECT current_setting('default_order'), current_setting('default_null_order'), "
            "current_setting('default_collation'), current_setting('search_path'), "
            "current_database()").fetchone()
        self._settings = {"default_order": row[0], "default_null_order": row[1],
                          "default_collation": row[2] or "", "search_path": row[3] or "",
                          "database": row[4]}
        self._settings_key = "|".join(self._settings.values())
        # a prepared plan keeps the ORDER BY direction it was bound under
        self._drop_plans()
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
        self._store = self._exact and "store=true" in info   # stage B: per-table column store (docs/RESIDENT_COLUMNS_DESIGN.md)
        # §4.12: the global masked aggregate on the GPU side; it answers over a
        # store view, so it needs the store too
        self._global = self._exact and self._store and "global=true" in info
        # stage C: lanes stored at their narrowest width — the memory estimate sizes them by type
        self._narrow = "narrow=true" in info
        dm = re.search(r"device_memory=(\d+)", info)
        self._device_bytes = int(dm.group(1)) if dm else 0   # 0 = the backend does not report it (§5.5)
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
        self._close_probe_cursor()
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
        """Rule 1, measured: the thresholds PREDICT the win; the process is the
        judge. After a template's first three rewritten runs its native time is
        known — seen while the set was not resident yet, or timed once on a side
        cursor — and if the best rewritten run is not faster, the template runs
        native from then on (reason 'threshold'). The decision is not final:
        a short kernel runs up to 3x slower on Apple silicon while other threads
        of the process keep waking (DuckDB's idle workers do) or after the device
        idled, and that state comes and goes. So at most once per _REMEASURE_S a
        kept template re-times native, and a declined one re-times its rewritten
        form, on a side cursor; whichever is faster in the process's current
        state wins. The user's own statement is never the experiment."""
        d = getattr(self, "_timing_decision", None)
        if d is None or not getattr(self, "_thresholds", True) or self._last.fallback:
            return
        now = time.monotonic()
        if not self._last.rewritten:
            if self._last.reason in ("not_resident", "threshold"):
                d.native_ms = ms if d.native_ms is None else min(d.native_ms, ms)
            if d.probation:
                # this run of the user's own statement IS the native measurement
                # probation compares against — it never times native itself
                d.native_obs.append(ms)
                del d.native_obs[:-_probation.NATIVE_OBSERVATIONS]
                if self._may_probe(d, now):
                    self._probation_probe(d, now)
                return
            if d.measured_declined and now >= d.next_check_at and d.probe_sql and isinstance(query, str):
                d.next_check_at = now + _REMEASURE_S
                probe = self._probe_ms(d.probe_sql, None)
                if probe is not None and probe < ms:
                    self._log(f"threshold: measured {probe:.2f} ms rewritten vs {ms:.2f} ms native — "
                              f"template rewritten again")
                    d.rewritten, d.reason, d.measured_declined = True, "", False
                    d.rewritten_ms = [probe]
                    d.native_ms = ms
            return
        d.rewritten_ms.append(ms)
        del d.rewritten_ms[:-5]
        if not d.probe_sql:
            d.probe_sql = self._last.sql or ""
        if not d.timing_checked:
            if len(d.rewritten_ms) < 3:
                return
            if d.native_ms is None:
                if not isinstance(query, str):
                    return
                d.native_ms = self._probe_ms(query, parameters)
                if d.native_ms is None:
                    d.timing_checked = True
                    return
            d.timing_checked = True
            d.next_check_at = now + _REMEASURE_S
            native = d.native_ms
        elif now >= d.next_check_at and isinstance(query, str):
            d.next_check_at = now + _REMEASURE_S
            native = self._probe_ms(query, parameters)
            if native is None:
                return
            d.native_ms = native
        else:
            return
        best = min(d.rewritten_ms[-3:])
        if best >= native:
            self._log(f"threshold: measured {best:.2f} ms rewritten vs {native:.2f} ms native — "
                      f"template declined (re-measured in {_REMEASURE_S:.0f} s)")
            d.rewritten = False
            d.reason = "threshold"
            d.measured_declined = True
            if d.probation:
                # a promoted template is an ordinary one, so the 60 s re-measure is what
                # takes it back when the process turns slow. It returns to the trial, but
                # behind a doubling back-off: a shape that flaps must not burn probes.
                d.probation = "demoted"
                d.probe_wins, d.round_probes = 0, []
                d.probe_ms = best
                d.backoff_s = _probation.next_backoff(d.backoff_s)
                d.next_probe_at = now + d.backoff_s
                self._manager.mark_probation(d.tag, True)
                self._log(f"probation: demoted — the next round is in {d.backoff_s:.0f} s")

    def _probe_ms(self, sql: str, parameters) -> Optional[float]:
        """Time one execution of `sql` on a side cursor (the caller still
        fetches from self._raw); None when it fails, with the error text left in
        `self._probe_error` for the caller that cares which failure it was."""
        self._probe_error = ""
        try:
            cur = self._raw.cursor()
            t0 = time.perf_counter()
            cur.execute(sql, parameters)
            out = (time.perf_counter() - t0) * 1000.0
            cur.close()
            return out
        except duckdb.Error as e:
            self._probe_error = str(e)
            self._log(f"timing probe failed: {e}")
            return None

    def _probe_rewritten_ms(self, sql: str) -> Optional[float]:
        """Time the rewritten form on the trial's OWN cursor, which is kept
        between probes. `_probe_ms` opens a fresh one every time, and a fresh
        DuckDB connection pays about a millisecond on its first statement:
        measured on the SF1 7-group statement, 0.88 ms best but 1.93 ms median
        on a new cursor per run against 0.91 ms median on a warm one, with
        native at 2.07 ms — enough to read a 2.3x win as a loss. That is a
        property of the instrument, not of the statement; a promoted template
        runs on the session's own warm connection. The statement is still run as
        TEXT, so the plan cache (§3.2) stays out of the comparison on purpose."""
        self._probe_error = ""
        cur = getattr(self, "_probe_cur", None)
        try:
            if cur is None:
                # one untimed run to warm the new cursor: the native runs on the other
                # side of the comparison are warm, and the first statement on a fresh
                # connection is not
                cur = self._probe_cur = self._raw.cursor()
                t0 = time.perf_counter()
                cur.execute(sql)
                self._probe_budget.spend((time.perf_counter() - t0) * 1000.0)
            t0 = time.perf_counter()
            cur.execute(sql)
            return (time.perf_counter() - t0) * 1000.0
        except duckdb.Error as e:
            self._probe_error = str(e)
            self._log(f"probation probe failed: {e}")
            self._close_probe_cursor()
            return None

    def _close_probe_cursor(self) -> None:
        cur, self._probe_cur = getattr(self, "_probe_cur", None), None
        if cur is not None:
            try:
                cur.close()
            except Exception:
                pass

    # ---- probation (§9.1, gpudb/_probation.py) ----
    def _probation_on(self) -> bool:
        """Is the mechanism available at all on this connection? It is a
        background mechanism: it needs the residency manager to be allowed to
        upload, the thresholds to be the thing declining statements, and no
        open transaction (whose writes the trial's set would not see). It is
        also off inside the nested pass (§4.14): what is being decided there is
        a SELECT lifted out of the user's statement, and the user's statement is
        the only thing whose native time is ever measured."""
        return (bool(getattr(self, "_thresholds", True)) and self._transparent
                and self._residency_mode != "manual" and not self._tx_open
                and not getattr(self, "_in_nested", False)
                and bool(getattr(self, "_exact", False)))

    def _may_probe(self, d: "Decision", now: float) -> bool:
        """May a round start right now? Everything here is about not taking
        anything from the user: the set has to be there already (no upload is
        waited for), no other statement of this connection may be running (a
        probe on a side cursor shares DuckDB's worker pool with it), the
        template's own back-off has to have expired, and the process's probe
        budget has to have room."""
        return (d.probation in ("candidate", "waiting", "probing", "demoted")
                and self._probation_on() and now >= d.next_probe_at
                and bool(d.probe_sql)
                and len(d.native_obs) >= _probation.MIN_NATIVE_OBSERVATIONS
                and self._manager.is_ready(d.tag)
                and self._manager.others_in_flight() == 0
                and self._probe_budget.may_probe(now))

    def _probation_probe(self, d: "Decision", now: float) -> None:
        """ONE run of the rewritten form on a side cursor. A round is
        PROBES_PER_ROUND of them, one per user statement rather than three back
        to back, so what probation adds to the statement the user is waiting on
        is one extra execution — the same order of cost the measured re-measure
        above has always had.

        The probe runs the statement as TEXT, the way the native runs it is
        compared against were measured — a promoted template then also gets the
        cached plan (§3.2) and is faster than the probe said, so the error the
        comparison makes is the one rule 1 can afford."""
        t = self._probe_rewritten_ms(d.probe_sql)
        if t is None:
            if STALE_MARKER in (getattr(self, "_probe_error", "") or ""):
                # some connection wrote to the table. The probe is how a template on
                # probation finds out, since its own statements run native and never
                # touch the guard. The set goes, and the trial starts again on the new
                # rows — what it had measured was for rows that are gone.
                self._probation_stale(d)
            else:
                # the rewritten form does not even run here (a set dropped behind the
                # wrapper's back, a defect): this shape is not a candidate at all
                d.probation = "retired"
                d.round_probes = []
                self._manager.mark_probation(d.tag, False)
            return
        self._probe_budget.spend(t)
        d.probes_spent += 1
        d.probe_ms_spent += t
        d.round_probes.append(t)
        if len(d.round_probes) < _probation.PROBES_PER_ROUND:
            return
        self._probation_judge(d, now)

    def _probation_judge(self, d: "Decision", now: float) -> None:
        """A round is complete: its median against the median of the template's
        own recent native runs (`_probation` says why the medians and not the
        minimums)."""
        native = _probation.median(d.native_obs)
        got = _probation.median(d.round_probes)
        d.round_probes = []
        d.probe_ms = got
        d.probe_rounds += 1
        d.probation = "probing"
        if got < _probation.PROMOTE_RATIO * native:
            d.probe_wins += 1
            d.next_probe_at = now + _probation.ROUND_INTERVAL_S
            self._log(f"probation: round {d.probe_rounds} won, measured {got:.2f} ms rewritten vs "
                      f"{native:.2f} ms native ({d.probe_wins} of {_probation.ROUNDS_TO_PROMOTE} rounds)")
            if d.probe_wins >= _probation.ROUNDS_TO_PROMOTE:
                self._promote(d, got, native, now)
            return
        d.probe_wins = 0
        d.backoff_s = _probation.next_backoff(d.backoff_s)
        d.next_probe_at = now + d.backoff_s
        self._log(f"probation: round {d.probe_rounds} lost, measured {got:.2f} ms rewritten vs "
                  f"{native:.2f} ms native — template stays native (next round in {d.backoff_s:.0f} s)")
        if d.probe_rounds >= _probation.MAX_ROUNDS:
            d.probation = "retired"
            d.round_probes = []
            self._manager.mark_probation(d.tag, False)
            self._log(f"probation: {d.probe_rounds} rounds without a win — this template is not tried again")

    def _probation_stale(self, d: "Decision") -> None:
        """Drop the trial's set the way a rewritten statement's fallback does
        (§5.4), and put its template back at the start of the trial.

        Everything here runs on a cursor of its own. A probe happens after the
        user's statement has executed but BEFORE they have fetched from it, and
        anything issued on `self._raw` in that window replaces the result they
        are about to read. (Cached plans are deliberately left alone for the
        same reason: `DEALLOCATE` is per-connection, so it cannot be moved to a
        side cursor — and it is not needed, because a plan over a set this just
        invalidated raises GPUDB_STALE on its next execution and takes the
        ordinary `_on_stale` path.)"""
        if d.tag.startswith("gpudb:v1:"):
            try:
                cur = self._raw.cursor()
                try:
                    cur.execute("SELECT gpu_invalidate(?)", [":".join(d.tag.split(":")[:6])]).fetchall()
                finally:
                    cur.close()
            except Exception:
                pass
        self._close_probe_cursor()
        st = self._manager.get(d.tag)
        self._manager.invalidate(d.tag)
        if st is not None:
            self._manager.note_candidate(d.tag, st.upload_sql, probation=True)
        self._reset_probation([d.tag])
        self._log(f"probation: the table under {d.tag} moved — the trial starts again")

    def _promote(self, d: "Decision", best: float, native: float, now: float) -> None:
        """The shape won ROUNDS_TO_PROMOTE rounds spread over more than the
        seconds a process's mode lasts: it is an ordinary rewritten template
        from now on, handed to the measured rule 1 with what the trial already
        knows — so its next 60 s re-measure against native is what keeps it."""
        d.rewritten, d.reason, d.measured_declined = True, "", False
        d.probation = "promoted"
        d.rewritten_ms = [best]
        d.native_ms = native
        d.timing_checked = True
        d.next_check_at = now + _REMEASURE_S
        d.backoff_s, d.round_probes = 0.0, []
        self._manager.mark_probation(d.tag, False)
        self._log(f"probation: promoted — {best:.2f} ms rewritten vs {native:.2f} ms native "
                  f"({native / max(best, 1e-9):.2f}x) over {_probation.ROUNDS_TO_PROMOTE} rounds; "
                  f"re-measured against native every {_REMEASURE_S:.0f} s from here")

    def execute(self, query, parameters=None):
        sql = self._route(query, parameters)
        self._manager.statement_begin()
        try:
            try:
                t0 = time.perf_counter()
                # A rewritten statement DuckDB has planned before runs as
                # EXECUTE; an EXPLAIN (whose sql carries its prefix) and
                # everything native run as text. The run that PREPAREs pays for
                # it inside the timed region, so the measured rule 1 sees it.
                run = self._planned(sql) if (self._last.rewritten and not parameters
                                             and sql == self._last.sql) else sql
                self._raw.execute(run, parameters)
                self._note_timing((time.perf_counter() - t0) * 1000.0, query, parameters)
                if self._last.rewritten:
                    self._check_output_size()
            except duckdb.Error as e:
                if self._last.rewritten and STALE_MARKER in str(e):
                    self._on_stale(sql)
                    self._raw.execute(query, parameters)
                elif self._last.rewritten and "INTERRUPT" not in str(e).upper():
                    # Rule 2: the user's statement is the ORIGINAL one. Whatever went wrong in the
                    # rewritten form (a device allocation that failed, a set dropped behind the
                    # wrapper's back, a bug), DuckDB answers the original — and if that raises too,
                    # it is DuckDB's own error for the user's own statement.
                    self._on_rewrite_error(e)
                    self._raw.execute(query, parameters)
                else:
                    raise
        finally:
            self._manager.statement_end()
            self._after()
        return self

    def _on_rewrite_error(self, e: Exception) -> None:
        """The rewritten statement failed for a reason that is not staleness:
        answer natively now, and keep this template native from here on (its
        sets are re-noted as stale so a healthy upload can replace them)."""
        self._last.fallback = True
        self._drop_plans()
        self._last.error = str(e)[:300]
        self._log(f"rewritten statement failed, answered natively: {str(e)[:160]}")
        d = getattr(self, "_last_decision", None)
        if d is not None:
            d.rewritten = False
            d.reason = "error"
            d.probation = ""        # a form that raises is not a candidate for anything
        for tag in [t for t in (getattr(self, "_last_tags", None) or [self._last.tag]) if t]:
            self._manager.invalidate(tag)

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
            d.threshold_kind, d.threshold_why = _thresholds.HARD, f"{rows_out} rows returned > {bound} (output-bound)"
            d.probation = ""                     # output-bound: never tried on a side cursor

    def sql(self, query, **kw):
        sql = self._route(query, None)
        self._manager.statement_begin()
        try:
            try:
                rel = self._raw.sql(sql, **kw)
                # the relation binds here; reading its columns is what forces
                # that bind to have happened, so a rewritten statement that
                # cannot even bind fails inside this call. (The guard itself
                # runs when the relation is executed — measured: neither one
                # bind nor two raise on a stale set.) Binding it a second time
                # cost a second bind of two or three relations and changed
                # nothing.
                if self._last.rewritten:
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
                if not self._parent:
                    self._watch_files = self._watched_files()
            except Exception:
                pass
        if self._resnap_after:
            self._resnap_after = False
            root = self._parent or self
            root._write_snapshot = root._snapshot_files()

    # ---- foreign writes (§5.9): the database file and its WAL change on every committed write ----
    def _watched_files(self) -> List[str]:
        """The files whose size / mtime move on a committed write: every attached
        database file and its write-ahead log. Nothing for :memory: databases and
        nothing for a read-only connection (nobody can write the file while it is
        open read-only)."""
        if getattr(self, "_read_only", False):
            return []
        out: List[str] = []
        try:
            for (path,) in self._raw.execute(
                    "SELECT path FROM duckdb_databases() WHERE NOT internal AND path IS NOT NULL AND path <> ''").fetchall():
                if path and path != ":memory:":
                    out += [path, path + ".wal"]
        except Exception:
            pass
        return out

    def _snapshot_files(self) -> tuple:
        snap = []
        for f in self._watch_files:
            try:
                st = os.stat(f)
                snap.append((st.st_size, st.st_mtime_ns))
            except OSError:
                snap.append(None)
        return tuple(snap)

    def _foreign_write_seen(self) -> bool:
        """Did any watched file change since the last snapshot? 2-3 us per
        statement. A change means some connection committed a write (an
        in-place UPDATE keeps the row count and passes gpu_assert_rows: this is
        the check that catches it); the sets are dropped and rebuilt."""
        root = self._parent or self                # cursors share the family's snapshot
        if not root._watch_files:
            return False
        now = root._snapshot_files()
        if now == root._write_snapshot:
            return False
        root._write_snapshot = now
        return True

    def __getattr__(self, name):
        return getattr(self._raw, name)

    # ---- the statement path ----
    def _on_stale(self, sql: str) -> None:
        self._last.fallback = True
        self._drop_plans()
        tags = [t for t in (getattr(self, "_last_tags", None) or [self._last.tag]) if t]
        self._reset_probation(tags)
        # stage B: the table's STORE holds the stale columns — drop it (and every view on it)
        # in the extension, or the next upload would find its lanes "already there"
        for prefix in {":".join(t.split(":")[:6]) for t in tags if t.startswith("gpudb:v1:")}:
            try:
                self._raw.execute("SELECT gpu_invalidate(?)", [prefix]).fetchall()
            except Exception:
                pass
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

    def _reset_probation(self, tags: List[str]) -> None:
        """The data under these sets moved (§5.4). Every trial over them starts
        again: the times measured were for rows that are gone, and a template
        promoted on them has to earn it on the new data. (A write the wrapper
        sees goes through `_invalidate_all`, which drops the decisions outright;
        this is the path for a write the staleness guard caught.)"""
        hit = set(tags)
        for d in list(self._cache.values()):
            for dd in [d] + list(d.variants.values()):
                if not dd.probation or dd.tag not in hit:
                    continue
                if dd.probation == "promoted":
                    dd.rewritten, dd.reason, dd.measured_declined = False, "threshold", False
                    dd.rewritten_ms, dd.native_ms, dd.timing_checked = [], None, False
                dd.probation = "candidate"
                dd.probe_wins = dd.probe_rounds = dd.sightings = 0
                dd.probe_ms = None
                dd.round_probes = []
                dd.native_obs, dd.probe_sql = [], ""
                dd.backoff_s, dd.next_probe_at = 0.0, 0.0

    def _invalidate_all(self, why: str) -> None:
        self._resnap_after = True                 # the file changes when THIS write commits: not foreign
        if why in ("ATTACH", "DETACH"):
            self._refresh_after = True
        self._manager.invalidate(None)
        self._drop_plans()
        self._cache.clear()
        self._nested_cache.clear()
        self._flat_cache.clear()
        self._unique_cache.clear()
        self._expr_types.clear()
        self._big_tables = None
        self._view_catalog = None
        self._flat_views.clear()
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
            # a view whose definition names a big table stands for it (§4.20)
            vc = self._view_catalog = _views.ViewCatalog(self._raw)
            for vname, (body, _cols) in vc.views.items():
                if any(re.search(r"(?<![A-Za-z0-9_])" + re.escape(t) + r"(?![A-Za-z0-9_])", body, re.I)
                       for t in list(self._big_tables)):
                    self._big_tables.add(vname)
        except Exception:
            self._big_tables = set()
        return self._big_tables

    def _names_view(self, sql: str) -> bool:
        vc = getattr(self, "_view_catalog", None)
        if vc is None:
            vc = self._view_catalog = _views.ViewCatalog(self._raw)
        return any(re.search(r"(?<![A-Za-z0-9_])" + re.escape(v) + r"(?![A-Za-z0-9_])", sql, re.I) for v in vc.views)

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

    # ---- cached plans: DuckDB plans a rewritten statement once, not per run ----
    _MAX_PLANS = 48

    def _planned(self, sql: str) -> str:
        """`EXECUTE <name>` for a rewritten statement DuckDB has already
        planned, or the text itself the first time it is seen.

        DuckDB is handed the rewritten statement as TEXT on every run, so it
        parses, binds and optimises two or three relations every time, while
        the text of a warm template is byte-identical run to run. A statement
        that comes back is PREPAREd once and EXECUTEd after that.

        What a plan may never do is freeze the answer, and it does not: the
        staleness guard is a volatile scalar over a live `count(*)` of the base
        table, so it runs inside the plan on every execution (a write behind a
        prepared plan still raises GPUDB_STALE), the table function's `init`
        runs again and re-acquires the set, and DuckDB re-binds a prepared
        statement when the catalog moves under it (DDL, a view redefinition).
        What it does NOT re-bind is a session setting or a name resolution, so
        the plans die wherever a decision dies — `_invalidate_all` (every
        non-SELECT the wrapper sees: DDL, DML, SET, ATTACH, a transaction, a
        foreign write), `_refresh_settings`, `_on_stale` and
        `_on_rewrite_error`.

        The first sighting of a rendered text runs as text: a statement whose
        literals change on every execution renders a new text every time, and
        preparing each one would be pure loss."""
        with self._plan_mu:
            name = self._plans.get(sql)
            if name is not None:
                return "EXECUTE " + name
            if sql not in self._plan_seen:
                if len(self._plan_seen) > self._MAX_PLANS:
                    self._plan_seen.clear()
                self._plan_seen[sql] = True
                return sql
            # A name is never reused for a different statement — two threads on
            # one connection must not be able to EXECUTE each other's plan. Old
            # names are deallocated, and a thread that reaches a name a moment
            # after it was deallocated gets an error, which execute() answers
            # natively like any other error from a rewritten statement.
            self._plan_seq += 1
            name = "gpudb_plan_%d" % self._plan_seq
            try:
                self._raw.execute("PREPARE " + name + " AS " + sql)
            except duckdb.Error as e:
                self._log(f"plan cache: {str(e).splitlines()[0][:120]}")
                return sql
            self._plans[sql] = name
            if len(self._plans) > self._MAX_PLANS:
                old = next(iter(self._plans))
                self._deallocate([self._plans.pop(old)])
            return "EXECUTE " + name

    def _deallocate(self, names) -> None:
        if not names:
            return
        try:
            self._raw.execute("; ".join("DEALLOCATE " + n for n in names))
        except duckdb.Error:
            pass

    def _drop_plans(self) -> None:
        """Every cached plan goes: the statements it was built on are being
        re-decided, and a plan DuckDB does not re-bind by itself (a session
        setting, a name resolution) would otherwise outlive its decision."""
        with self._plan_mu:
            names = list(self._plans.values())
            self._plans.clear()
            self._plan_seen.clear()
        self._deallocate(names)

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
        if ";" not in query and _SELECT_START_RE.match(query):
            big = self._names_big_table(query)
            # a view stands for its definition: the aggregate may be inside it (§4.20)
            sg = getattr(self, "_global", False)
            agg = _maybe_aggregate(query, sg)         # two DOTALL regexes over the text: run once
            if not big or (not agg and not self._names_view(query)):
                self._last.reason = "threshold" if agg else "shape"
                return query
        # DuckDB's splitter is a round trip, and what it answers depends on the
        # text alone — so the same text is split once (§5.2 is unchanged: every
        # statement is still classified, and a non-SELECT still invalidates
        # before it runs)
        stmts = self._split_cache.get(query, False)
        if stmts is False:
            stmts = _classify.split(self._raw, query)
            if len(self._split_cache) > 256:
                self._split_cache.clear()
            self._split_cache[query] = stmts
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
        if not _maybe_aggregate(sql, getattr(self, "_global", False)) and not self._names_view(sql):   # a view may hold the aggregate (§4.20)
            self._last.reason = "shape"
            return None
        out = self._rewrite_text(sql)
        # the statement as a whole is not a transparent shape (or reads a CTE / view /
        # derived table, which has no identity of its own): a SELECT inside it may be
        if out is None and getattr(self, "_exact", False) and \
                self._last.reason in ("shape", "not_found", "view", "temp", "ambiguous", "double", "decimal"):
            whole = self._last.reason
            # the nested pass walks the statement with its views spliced in (§4.20): a view's
            # GROUP BY is a rewritable SELECT like any derived table's
            out = self._rewrite_nested(self._inline_views(sql) if self._names_view(sql) else sql)
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
                            self._in_nested = True
                            try:
                                self._rewrite_text(sub_sql)
                            finally:
                                self._in_nested = False
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
            self._in_nested = True
            try:
                o = self._rewrite_text(sub_sql)
            finally:
                self._in_nested = False
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
        if flat is not None and sql in self._flat_views and not self._views_unchanged(self._flat_views[sql]):
            # a view the statement uses was redefined (any connection): rebuild from the new definition
            self._flat_cache.pop(sql, None); self._flat_views.pop(sql, None)
            self._view_catalog = None
            self._cache.pop((self._normalise(flat)[0], self._settings_key), None)
            flat = None
        if flat is None:
            flat = self._inline_views(sql)
            sql_v = flat
            if "(" in flat or flat.lstrip()[:4].upper() == "WITH":
                try:
                    folded = None
                    tree = self._serialize(sql_v)
                    if _flatten.fold(tree) is not None:                 # cheap structural test first
                        want = [(r[0], r[1]) for r in self._raw.execute("DESCRIBE " + sql_v).fetchall()]
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
            flat = self._normalise_aggs(self._normalise_syntax(flat))
            if len(self._flat_cache) > 1024:
                self._flat_cache.clear()
            self._flat_cache[sql] = flat
        return flat

    def _inline_views(self, sql: str) -> str:
        """§4.20: views named by the statement become derived tables holding their
        definition — the statement is then a statement over base tables and the
        fold / nested passes apply. Names pinned, names + types verified with
        DESCRIBE; anything else keeps the text."""
        vc = getattr(self, "_view_catalog", None)
        if vc is None:
            vc = self._view_catalog = _views.ViewCatalog(self._raw)
        named = [v for v in vc.views
                 if re.search(r"(?<![A-Za-z0-9_])" + re.escape(v) + r"(?![A-Za-z0-9_])", sql, re.I)]
        if not named:
            return sql
        try:
            want = [(r[0], r[1]) for r in self._raw.execute("DESCRIBE " + sql).fetchall()]
            out = _views.inline(self._serialize(sql), vc, self._serialize, [w[0] for w in want])
            if out is None:
                return sql
            cand = self._raw.execute("SELECT json_deserialize_sql(?)", [out]).fetchone()[0]
            have = [(r[0], r[1]) for r in self._raw.execute("DESCRIBE " + cand).fetchall()]
            if want == have:
                # remembered so a redefinition of any of these views (from any connection) is noticed
                self._flat_views[sql] = {v: vc.views[v][0] for v in named}
                return cand
            self._log(f"views: names / types changed ({want} -> {have}); left as written")
        except Exception as e:
            self._log(f"views: {str(e)[:120]}")
        return sql

    def _views_unchanged(self, snapshot: Dict[str, str]) -> bool:
        """One catalog read: do the views a cached statement was built on still have
        the same definition? (~0.2 ms; a view is the one object whose text can change
        under a running session without any table changing.)"""
        try:
            rows = self._raw.execute(
                "SELECT lower(view_name), sql FROM duckdb_views() WHERE NOT internal AND NOT temporary "
                "AND database_name = current_database() AND schema_name = current_schema() "
                "AND lower(view_name) IN (" + ",".join("?" * len(snapshot)) + ")", list(snapshot)).fetchall()
        except Exception:
            return False
        now = {}
        for name, sql in rows:
            m = _views._CREATE_RE.match(sql or "")
            now[name] = m.group("body") if m else None
        return all(now.get(v) == body for v, body in snapshot.items())

    _SYNTAX_RE = re.compile(r"\b(?:GROUP|ORDER)\s+BY\s+(?:ALL\b|\d|[^;]*?,\s*\d+\s*(?:,|$|\)|ASC|DESC|NULLS|LIMIT|HAVING|OFFSET))"
                            r"|\bSELECT\s+DISTINCT\b|\bRIGHT\s+(?:OUTER\s+)?JOIN\b", re.IGNORECASE)

    def _normalise_syntax(self, sql: str) -> str:
        """§4.21: GROUP BY ALL / ordinals and ORDER BY ALL / ordinals spelled out as
        the select items they stand for. Names pinned, names + types verified."""
        if not self._SYNTAX_RE.search(sql):
            return sql
        try:
            want = [(r[0], r[1]) for r in self._raw.execute("DESCRIBE " + sql).fetchall()]
            out = _syntax.normalise(self._serialize(sql), [w[0] for w in want])
            if out is None:
                return sql
            cand = self._raw.execute("SELECT json_deserialize_sql(?)", [out]).fetchone()[0]
            have = [(r[0], r[1]) for r in self._raw.execute("DESCRIBE " + cand).fetchall()]
            if want == have:
                return cand
            self._log(f"syntax: names / types changed ({want} -> {have}); left as written")
        except Exception as e:
            self._log(f"syntax: {str(e)[:120]}")
        return sql

    _AGG_SPELLING_RE = re.compile(r"\bFILTER\s*\(|\bcount_if\s*\(|\bbool_(?:and|or)\s*\(", re.IGNORECASE)

    def _normalise_aggs(self, sql: str) -> str:
        """§4.19: FILTER aggregates, count_if and bool_and / bool_or as the
        plain aggregates over a CASE / cast they are equal to. Names are pinned
        and names + types verified with DESCRIBE; anything else keeps the text."""
        if not self._AGG_SPELLING_RE.search(sql):
            return sql
        try:
            want = [(r[0], r[1]) for r in self._raw.execute("DESCRIBE " + sql).fetchall()]
            out = _aggs.normalise(self._serialize(sql), [w[0] for w in want])
            if out is None:
                return sql
            cand = self._raw.execute("SELECT json_deserialize_sql(?)", [out]).fetchone()[0]
            have = [(r[0], r[1]) for r in self._raw.execute("DESCRIBE " + cand).fetchall()]
            if want == have:
                return cand
            self._log(f"aggregate spelling: names / types changed ({want} -> {have}); left as written")
        except Exception as e:
            self._log(f"aggregate spelling failed: {str(e)[:120]}")
        return sql

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
                    self._last.detail = f"{_thresholds.HARD}: 16 literal variants of one template"
                    return None
                v = self._decide(sql)
                v.literals = literals
                d.variants[literals] = v
            d = v
        self._last.round_trip_ms = (time.perf_counter() - t0) * 1000.0
        if not d.rewritten:
            self._last.reason = d.reason
            self._last.detail = self._detail(d)
            if d.measured_declined or d.probation:
                self._timing_decision = d        # its native runs keep its native time; it may be re-measured
            if d.probation:
                # §9.1: the statement runs on DuckDB exactly as it would without any of
                # this. What happens here is that the set is asked for in the background
                # and, once it is there, the rewritten form is rendered for the side cursor.
                self._probation_step(d, sql, literals)
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
                    elif getattr(self, "_store", False) and b.store_lanes:
                        # stage B: a base set is a view over its table's store
                        parts = b.tag.split(":")
                        skey = ":".join(parts[:6])
                        try:
                            have = {r[0] for r in self._raw.execute(
                                "SELECT \"column\" FROM gpu_store_columns() WHERE store = ?", [skey]).fetchall()}
                        except Exception:
                            have = set()
                        missing = [l for l in b.store_lanes if l[0] not in have]
                        bident = _resolve.Identity(parts[2], parts[3], parts[4], int(parts[5]), {})
                        up_name, up_sql = _rewrite.store_upload_sql(bident.tag, missing, f"{b.fqn} AS {_join.OUTER_ALIAS}") if missing else ("", "")
                        self._manager.note_candidate(b.tag, up_sql, fqn=b.fqn,
                                                     est_bytes=estimate_set_bytes(d.base_rows.get(b.tag, d.set_rows), max(1, len(missing))),
                                                     upload_name=up_name, store_key=skey,
                                                     store_lanes=[l[0] for l in b.store_lanes],
                                                     post_sql=["SELECT gpu_prepare_resident('%s')" % b.tag.replace("'", "''")])
                    else:
                        self._manager.note_candidate(b.tag, b.upload_sql, fqn=b.fqn,
                                                     est_bytes=estimate_set_bytes(d.base_rows.get(b.tag, d.set_rows),
                                                                                  _tag_lanes(b.tag)))
            for b in d.sentinels:
                self._manager.note_candidate(b.tag, "", steps=[b.upload_sql])
            if d.store_key and d.join is None:
                up_name, up_sql, missing = self._store_upload(d)
                # the sort cache is charged only when the key lane itself is being
                # uploaded: a lane already in the store carries its cache with it
                narrow = getattr(self, "_narrow", False)
                widths = [d.lane_widths.get(n, 8) for n in missing] if narrow and missing else None
                kw = (d.lane_widths.get(d.key_lane, 8) if narrow else 8) if d.key_lane in missing else 0
                st = self._manager.note_candidate(d.tag, up_sql, fqn=d.fqn,
                                                  deps=[b.tag for b in d.sentinels] or None,
                                                  est_bytes=estimate_set_bytes(d.set_rows, max(1, len(missing)),
                                                                               widths, kw),
                                                  upload_name=up_name, store_key=d.store_key,
                                                  store_lanes=list(d.store_lanes),
                                                  post_sql=["SELECT gpu_prepare_resident('%s')" % d.tag.replace("'", "''")])
            else:
                st = self._manager.note_candidate(d.tag, d.upload_sql, fqn=d.fqn,
                                                  deps=([b.tag for b in d.join.base] if d.join is not None
                                                        else [b.tag for b in d.sentinels] or None),
                                                  steps=d.join.steps_sql if d.join is not None else None,
                                                  est_bytes=estimate_set_bytes(d.set_rows, _tag_lanes(d.tag)))
            if self._residency_mode == "eager" and st.state == "pending":
                self._manager.upload_now(d.tag, lambda s: self._raw.execute(s).fetchall())
            if not self._manager.is_ready(d.tag):
                self._last.reason = ("memory" if st.state == "failed" and st.error.startswith(MEMORY_ERROR)
                                     else "not_resident")
                return None
        rendered = self._render(d, sql, literals)
        if rendered is None:
            self._last.reason = "shape"
            return None
        out, self._last.engine = rendered
        if self._foreign_write_seen():
            # some connection committed a write since the last look: every set may be stale
            # (an in-place UPDATE keeps the row count, so the guard would not know)
            self._log("foreign write: the database file changed — resident sets dropped, statement runs native")
            self._invalidate_all("foreign write")
            self._resnap_after = False
            self._last.reason = "not_resident"
            return None
        self._last.rewritten = True
        self._last.sql = out
        if d.probation:
            self._last.detail = self._detail(d)     # 'promoted', with the pair the trial measured
        return out

    def _render(self, d: "Decision", sql: str, literals) -> Optional[Tuple[str, str]]:
        """(rewritten text, which renderer made it) for this statement's
        literals, or None when the literals cannot be re-planned."""
        if d.scalar_sql and literals == d.literals:
            out, engine = d.scalar_sql, "scalar"
        else:
            # literals differ from the cached template (or no scalar): re-render
            # from the current statement's tree with the reference renderer
            # (one serialize, no catalog work, no scalar call)
            plan = d.plan
            if literals != d.literals and (plan.having is not None or plan.limit is not None or plan.where):
                plan = self._replan_literals(sql, plan)
                if plan is None:
                    return None
            out, engine = _rewrite.render(plan, d.fqn, self._settings["default_order"]), "python"
        return ((d.wrap[0] + out + d.wrap[1]) if d.wrap is not None else out), engine

    # ---- probation: the decline side (§9.1) ----
    def _detail(self, d: "Decision") -> str:
        """`last_rewrite()["detail"]`: which kind of bound declined the statement,
        and where its trial stands if it has one (§9.1)."""
        bound = f"{d.threshold_kind}: {d.threshold_why}" if d.threshold_why else d.threshold_kind
        if not d.probation:
            return bound
        pair = ""
        if d.probe_ms is not None and d.native_obs:
            pair = f" ({d.probe_ms:.2f} ms rewritten vs {_probation.median(d.native_obs):.2f} ms native)"
        state = "probation" if d.probation in ("candidate", "waiting", "probing") else d.probation
        return f"{state}{pair}; {bound}" if bound else f"{state}{pair}"

    def _probation_step(self, d: "Decision", sql: str, literals) -> None:
        """Keep the trial of a soft-declined template moving, without touching
        the user's statement: ask the background path for the set (under the
        probation flag, so it never evicts anything that earned its place), and
        once the set is there render the rewritten form for the side cursor.
        The probe itself runs from `_note_timing`, after the user's statement
        has returned."""
        if d.probation == "retired" or not self._probation_on():
            return
        d.sightings += 1
        if not self._manager.is_ready(d.tag):
            if d.probation == "candidate" and d.sightings >= _probation.MIN_SIGHTINGS:
                budget = self._manager.memory_budget
                up_name, up_sql, missing = self._store_upload(d)
                narrow = getattr(self, "_narrow", False)
                widths = [d.lane_widths.get(n, 8) for n in missing] if narrow and missing else None
                kw = (d.lane_widths.get(d.key_lane, 8) if narrow else 8) if d.key_lane in missing else 0
                est = estimate_set_bytes(d.set_rows, max(1, len(missing)), widths, kw) if missing else 0
                ok, why = _probation.admissible_bytes(budget, self._manager.probation_bytes(), est)
                if not ok:
                    d.probation = "retired"
                    self._log(f"probation: not admitted ({why}) — {d.tag}")
                    return
                for b in d.sentinels:
                    self._manager.note_candidate(b.tag, "", steps=[b.upload_sql], probation=True)
                self._manager.note_candidate(d.tag, up_sql, fqn=d.fqn,
                                             deps=[b.tag for b in d.sentinels] or None,
                                             est_bytes=est, upload_name=up_name, store_key=d.store_key,
                                             store_lanes=list(d.store_lanes),
                                             post_sql=["SELECT gpu_prepare_resident('%s')" % d.tag.replace("'", "''")],
                                             probation=True)
                d.probation = "waiting"
            return
        if d.probation == "waiting":
            # The native runs seen so far were taken while the background upload was
            # scanning the table for this very set, so they are not what native costs
            # in the steady state (measured: 2.5 ms during the upload, 10 ms after it,
            # at an interactive cadence). The comparison starts from here.
            d.probation, d.native_obs = "probing", []
        # The rewritten text is rendered only when a probe could actually run next:
        # for a template whose literals move run to run, rendering means re-planning
        # them, and a statement on its way to DuckDB must not pay for that.
        now = time.monotonic()
        if (now >= d.next_probe_at
                and len(d.native_obs) + 1 >= _probation.MIN_NATIVE_OBSERVATIONS
                and self._probe_budget.may_probe(now)):
            rendered = self._render(d, sql, literals)
            d.probe_sql = rendered[0] if rendered is not None else ""

    def _store_upload(self, d: "Decision"):
        """(session name, upload statement, missing lanes) for a store-backed set:
        what the table's store lacks of the lanes the set reads (stage B). An
        empty statement means every lane is there and only the view's sort cache
        may be missing."""
        try:
            have = {r[0] for r in self._raw.execute(
                "SELECT \"column\" FROM gpu_store_columns() WHERE store = ?", [d.store_key]).fetchall()}
        except Exception:
            have = set()
        missing = [l for l in d.store_all if l[0] not in have]
        if not missing:
            return "", "", []
        ident = self._identity_of(d)
        tag, sql = _rewrite.store_upload_sql(ident.tag, missing, f"{d.fqn} AS {_join.OUTER_ALIAS}")
        return tag, sql, [l[0] for l in missing]

    def _identity_of(self, d: "Decision"):
        parts = d.store_key.split(":")
        return _resolve.Identity(parts[2], parts[3], parts[4], int(parts[5]), {})

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
        except duckdb.BinderException as e:
            # a subquery the nested pass lifted out of its statement can be correlated (TPC-H Q2:
            # `p_partkey` belongs to the outer query): it does not bind on its own, by design
            self._log(f"split: the statement does not bind on its own (correlated): {str(e).splitlines()[0][:120]}")
            return None
        except Exception as e:
            self._log(f"split failed: {e}")
            return None
        if outer_sql.count(_split.PLACEHOLDER) != 1:
            return None
        d = self._decide(inner_sql, allow_split=False, reagg=reagg, global_agg=is_global)
        if not d.rewritten and not d.probation:
            self._log(f"split: the inner GROUP BY declined ({d.reason})")
            return Decision(False, d.reason, threshold_kind=d.threshold_kind, threshold_why=d.threshold_why)
        # a softly declined inner statement keeps being built: what a trial has to time is
        # the WHOLE statement the user wrote, wrap and all, not the GROUP BY inside it (§9.1)
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

    def _decide(self, sql: str, allow_split: bool = True, reagg: bool = False,
                global_agg: bool = False) -> Decision:
        """Device path first; a join it cannot express falls back to uploading
        the join's result (§4.13); expressions over aggregates fall back to the
        split (§4.11), whose inner statement comes back through here."""
        d = self._decide_once(sql, "device", reagg, global_agg)
        if d.rewritten or d.reason != "shape" or not getattr(self, "_exact", False):
            return d
        if d.is_join and getattr(self, "_join", False):
            du = self._decide_once(sql, "upload", reagg, global_agg)
            if du.rewritten or du.reason != "shape":
                return du
        if allow_split:
            ds = self._decide_split(sql)
            if ds is not None:
                return ds
        return d

    def _decide_once(self, sql: str, mode: str, reagg: bool = False,
                     global_agg: bool = False) -> Decision:
        d = self._decide_body(sql, mode, reagg, global_agg)
        try:
            d.is_join = _join.is_join_statement(self._serialize(sql))
        except Exception:
            pass
        return d

    def _decide_body(self, sql: str, mode: str, reagg: bool = False,
                     global_agg: bool = False) -> Decision:
        # (kind, why) of a SOFT threshold decline whose decision is built to the
        # end anyway, so the shape can be tried on a side cursor (§9.1 probation)
        soft: Optional[Tuple[str, str]] = None
        if global_agg and not getattr(self, "_global", False):
            return Decision(False, "backend")
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
                return Decision(False, "threshold", is_join=True, threshold_kind=_thresholds.HARD,
                                threshold_why=f"the join returns {join_rows} rows from tables of at most {biggest}")
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
            return Decision(False, "threshold", threshold_kind=_thresholds.HARD,
                            threshold_why=f"{nrows} rows < the floor of {self._floor_rows} (§0)")
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
            # §4.12: an aggregate without GROUP BY has exactly one group and no
            # output-size risk, so neither the zone map nor the distinct-count
            # scan says anything about it
            est = 1 if global_agg else (stats.get(plan.key) or {}).get("approx_unique")
            if not global_agg and len(plan.keys) > 1:
                est = 1
                for kc in plan.keys:
                    u = (stats.get(kc) or {}).get("approx_unique")
                    est = None if (u is None or est is None) else est * u
                if est is not None:
                    est = min(est, nrows)
            if est is None and not global_agg:
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
                    return Decision(False, "threshold", threshold_kind=_thresholds.HARD,
                                    threshold_why="the WHERE's selectivity could not be measured")
            # a lane with a subquery (EXISTS / IN / correlated scalar, §4.18) makes native run a
            # join too, whatever the FROM says: the join bounds apply
            ok, why, kind = _thresholds.decide(self._backend, plan.form, est, sel, bool(plan.where),
                                         join=low is not None or any(c.dep_tables for c in computed.values()), payloads=max(1, len(plan.vals)),
                                         string_key=bool(plan.dict_key),
                                         limited=plan.limit is not None and plan.limit <= 10_000,
                                         computed_payloads=sum(1 for v in set(plan.vals) if v in computed),
                                         reaggregated=reagg, global_agg=global_agg, rows=nrows,
                                         where_terms=len(plan.where))
            if not ok:
                self._log(f"threshold ({kind}): {why}")
                # §9.1 probation: a SOFT bound is one the process's mode or a small fixed
                # cost decides, so the shape may be TRIED on a side cursor — but only the
                # cheapest kind of set is ever built for a trial: a view over one table's
                # column store. A join (device or uploaded) materialises a result the size
                # of the join; nothing like that is built for a shape nobody asked for.
                if not (kind == _thresholds.SOFT and low is None and getattr(self, "_store", False)
                        and self._probation_on()):
                    return Decision(False, "threshold", threshold_kind=kind, threshold_why=why)
                soft = (kind, why)
        try:
            described = self._raw.execute("DESCRIBE " + sql).fetchall()
            _rewrite.apply_describe(plan, [(r[0], r[1]) for r in described])
        except _rewrite.Decline as e:
            return Decision(False, e.reason)
        except Exception as e:
            self._log(f"describe failed: {e}")
            return Decision(False, "error")
        if global_agg:
            # §4.12: from here the plan has no GROUP BY. Over a single table the
            # key lane goes with it (nothing uploads or sorts it); over a join the
            # set is the join's materialised result, whose lane 0 stays — no
            # operator ever reads it as a key.
            _rewrite.make_global(plan, keep_key=low is not None)
        q = (lambda c: computed[c].sql if c in computed else f'"{c}"')   # noqa: E731
        if low is None:
            try:
                plan.tag = ident.tag(plan.upload_columns) + (":global" if plan.no_key else "")
            except ValueError:
                return Decision(False, "shape")
            d = Decision(True, "", plan=plan, fqn=ident.fqn, tag=plan.tag,
                         upload_sql=_rewrite.upload_sql(plan, f"{ident.fqn} AS {_join.OUTER_ALIAS}", q),
                         form=plan.form)
            if getattr(self, "_store", False):
                # stage B: the set is a view over the table's store; only lanes the store lacks are uploaded
                lanes = _rewrite.store_lanes(plan, q)
                # a lane that is a plain column is stored at most as wide as its type;
                # a computed lane, a packed or dictionary key and a DECIMAL image take 8
                if len(plan.keys) == 1 and plan.keys[0] not in computed and plan.key_types:
                    d.lane_widths[plan.key_field] = lane_width(plan.key_types[0])
                if plan.val and plan.val not in computed:
                    d.lane_widths[plan.val] = lane_width(plan.val_types.get(plan.val, plan.val_type))
                for c in plan.pred_cols:
                    if c not in computed:
                        d.lane_widths[c] = lane_width(plan.pred_types.get(c, ""))
                d.key_lane = "" if plan.no_key else (("k#" + plan.key_field) if plan.dict_key else plan.key_field)
                d.store_key = f"gpudb:v1:{ident.catalog}:{ident.schema}:{ident.table}:{ident.oid}"
                d.store_lanes = [l[0] for l in lanes]
                d.store_all = lanes
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
        # §5.5: what the set(s) will hold — an uploaded join its result, a device join at most its
        # largest (probe) table, each base set its own table
        d.set_rows = int(join_rows) if upload_mode else int(nrows)
        if d.join is not None and not upload_mode:
            d.base_rows = {b.tag: int(low.tables[b.table].rows) for b in d.join.base if not b.sentinel}
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
        # §4.12: the C++ statement rewriter has no global form — it matches a
        # GROUP BY and nothing else — so a global plan is rendered by the
        # reference renderer alone, as the split's outer statement already is.
        if self._has_rewrite_scalar and not plan.global_agg:
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
        if soft is not None:
            # the statement is declined exactly as before — reason 'threshold', DuckDB
            # answers it. The decision was built to the end only so that the rewritten
            # form exists to be TIMED on a side cursor (§9.1).
            d.rewritten, d.reason = False, "threshold"
            d.threshold_kind, d.threshold_why = soft
            d.probation = "candidate" if d.store_key and d.join is None else ""
        return d


def connect(database: str = ":memory:", read_only: bool = False, config: Optional[dict] = None,
            *, extension: Optional[str] = None, transparent: bool = True,
            residency: str = "background", floor_rows: int = 1_000_000,
            idle_ms: float = 20.0, thresholds: bool = True, log=None,
            memory_budget=None) -> Connection:
    """duckdb.connect with the gpudb extension loaded and the transparent path
    on. `residency`: 'background' (upload in short row-id segments, each only
    while the connection is idle for `idle_ms`; §5.5), 'eager' (upload on first
    sighting, synchronously), 'manual' (v0.6 behaviour: only sets uploaded
    under an identity tag are used). `floor_rows`: tables smaller than this
    are never parsed (§0). `thresholds`: apply the per-backend shape thresholds
    (§9.1, gpudb/_thresholds.py); False rewrites every exact shape regardless
    of the predicted win — for parity testing, never for production.
    `memory_budget`: device memory the resident sets may use — bytes, or a string
    such as '16GB'; 0 or 'unlimited' removes the cap. Default: a quarter of
    unified memory on Apple silicon, half of device memory on a discrete GPU
    (§5.5). Least recently used sets are evicted to make room; a set that cannot
    fit is not uploaded and its statements keep running on DuckDB
    (`last_rewrite()["reason"] == "memory"`)."""
    if residency not in ("background", "eager", "manual"):
        raise ValueError("residency must be 'background', 'eager' or 'manual'")
    cfg = dict(config or {})
    ext = _find_extension(extension)
    if ext:
        cfg.setdefault("allow_unsigned_extensions", "true")
    raw = duckdb.connect(database, read_only=read_only, config=cfg)
    if ext:
        raw.execute(f"LOAD '{ext}'")
    else:
        # no local build: the extension DuckDB itself has installed (INSTALL gpudb FROM community),
        # if any; without it every statement runs native and gpu_build_info() says so. A locally
        # INSTALLed unsigned build needs config={"allow_unsigned_extensions": "true"} to LOAD.
        try:
            raw.execute("LOAD gpudb")
        except duckdb.Error as e:
            if log:
                log(f"extension not loaded ({str(e).splitlines()[0][:120]}); every statement runs native")
    con = Connection(raw, transparent=transparent, residency=residency,
                     floor_rows=floor_rows, idle_ms=idle_ms, log=log, memory_budget=memory_budget,
                     read_only=read_only)
    con._thresholds = thresholds
    return con
