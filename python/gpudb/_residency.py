"""Automatic residency (§5.5): what becomes resident, and when it is ready.

The manager never uploads beside a user statement. A background upload is an
upload *session* in the extension (milestone 0c): `gpu_upload_begin(tag)`,
then one short statement per row-id segment of the table
(`... WHERE rowid >= a AND rowid < b`, the 8 MiB host segment the extension
buffers, a few ms of scan each), then `gpu_upload_finish(tag)` (one device
copy + prepare + publish). Nothing touches the device before finish.

Each segment statement is issued on the wrapper's own cursor only while no
statement of the wrapper is in flight and the connection has been idle for
`idle_ms`; the moment a user statement arrives the cursor is interrupted
(DuckDB checks interrupts between vectors, ~0.5 ms), so a user statement
overlaps the upload by at most that latency, never by a whole scan. An
interrupted segment is re-run; the segments before it are kept by the
session (an append is one atomic step at finalize, so an interrupted scan
never appends). A session that never goes idle never uploads and runs
native throughout — rule 1 holds, the win is simply not there yet.

Consecutive interrupts pause the session (50 ms, doubling per interrupt,
capped at 1 s) so a cadence whose statements keep landing on segments pays
the interrupt latency at most once per pause, not once per statement; a
completed segment resets the pause.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

SEGMENT_BYTES = 8 << 20          # the extension's host segment (gpu_resident.cpp)
RETRY_PAUSE_MS = 50.0            # after an interrupted segment: this, doubling per consecutive interrupt
EVICT_MIN_AGE_S = 60.0            # §5.5: a set is never evicted within this long of being uploaded (anti-thrash)
MEMORY_ERROR = "memory budget: "  # SetState.error prefix of a set the budget kept off the device
RETRY_PAUSE_MAX_MS = 1000.0      # the idle wait already yields to every statement; a longer cap only
                                 # delayed readiness (measured: 15 of 20 segments landed in 3 s, the
                                 # rest took 30+ s at a 5 s cap under a 0-10 ms statement cadence)


@dataclass
class SetState:
    tag: str
    upload_sql: str
    fqn: str = ""
    state: str = "missing"          # missing | pending | uploading | ready | stale | failed
    epoch: int = 0
    last_invalidate: float = 0.0
    last_upload_start: float = 0.0
    resume_at: float = 0.0          # a pending set is picked once time.monotonic() >= this
    attempts: int = 0               # sessions begun
    error: str = ""
    # progress of the current / last session
    segments: int = 0
    segments_planned: int = 0
    rows_seen: int = 0
    interrupts: int = 0             # interrupted segment statements, over the set's life
    session_ms: float = 0.0         # wall time of the last session, begin -> finish
    seg_ms: List[float] = field(default_factory=list)   # scan time of each landed segment (last session)
    finish_window: tuple = (0.0, 0.0)  # time.monotonic() start/end of the last gpu_upload_finish call
    # a DERIVED set (a materialised join, §4.8): no table scan of its own —
    # once every set in `deps` is ready, `steps` (gpu_join_materialize calls)
    # run in order on the device
    deps: List[str] = field(default_factory=list)
    steps: List[str] = field(default_factory=list)
    # §5.5 memory budget: what the wrapper expects the set to cost on the device
    # (before the upload) and what the extension reports it costs (after)
    est_bytes: int = 0
    bytes: int = 0
    # Stage B (docs/RESIDENT_COLUMNS_DESIGN.md): a set answered by a VIEW over its
    # table's store. The upload session runs under `upload_name` (a store tag
    # naming the missing lanes); `store_key` / `store_lanes` say what the view
    # needs; `post_sql` runs once the upload landed (the key's sort cache).
    upload_name: str = ""
    store_key: str = ""
    store_lanes: List[str] = field(default_factory=list)
    post_sql: List[str] = field(default_factory=list)

    @property
    def session_name(self) -> str:
        return self.upload_name or self.tag

    @property
    def derived(self) -> bool:
        return bool(self.steps)

    @property
    def pair(self) -> bool:
        return ("gpu_upload_pair(" in self.upload_sql or "gpu_upload_pair_exact(" in self.upload_sql
                or "gpu_upload_rows_exact(" in self.upload_sql)

    @property
    def segment_rows_default(self) -> int:
        return SEGMENT_BYTES // (16 if self.pair else 8)


def _is_interrupt(err: str) -> bool:
    return "INTERRUPT" in err.upper()


def _not_resident(err: str) -> bool:
    """The extension saying a set it was asked for is not there (or no longer
    what it was): a view whose lanes left the store, a source of a join that
    was replaced. Recoverable — the set is uploaded again."""
    return ("no resident set" in err or "no resident column" in err
            or "GPUDB_STALE" in err or "GPUDB_UPLOAD_DISCARDED" in err)


class ResidencyManager:
    def __init__(self, cursor_factory: Callable[[], object], *, mode: str = "background",
                 idle_ms: float = 20.0, quiet_s: float = 2.0, rate_s: float = 30.0,
                 max_attempts: int = 20, segment_rows: Optional[int] = None,
                 memory_budget: Optional[int] = None, evict_min_age_s: float = EVICT_MIN_AGE_S,
                 log: Optional[Callable[[str], None]] = None):
        self._cursor_factory = cursor_factory
        self.mode = mode
        self.idle_ms = idle_ms
        self.quiet_s = quiet_s
        self.rate_s = rate_s
        self.max_attempts = max_attempts
        self.segment_rows = segment_rows        # None: 8 MiB worth of rows for the set's kind
        self.memory_budget = memory_budget      # bytes of device memory for resident sets; None = no cap
        self.evict_min_age_s = evict_min_age_s
        self.evictions = 0
        self._col_uploaded: Dict[tuple, float] = {}   # (store, lane) -> monotonic time it landed (anti-thrash)
        self._log = log or (lambda m: None)
        self._sets: Dict[str, SetState] = {}
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._in_flight = 0
        self._last_activity = time.monotonic()
        self._upload_cursor = None
        self._uploading_tag: Optional[str] = None   # set only while a statement runs on the cursor
        self._closed = False
        self._thread: Optional[threading.Thread] = None

    # ---- statement activity (called by the connection on every statement) ----
    def statement_begin(self) -> None:
        with self._lock:
            self._in_flight += 1
            self._last_activity = time.monotonic()
            cur, tag = self._upload_cursor, self._uploading_tag
        if cur is not None and tag is not None:
            self._interrupt(cur)

    def statement_end(self) -> None:
        with self._cv:
            self._in_flight = max(0, self._in_flight - 1)
            self._last_activity = time.monotonic()
            self._cv.notify_all()

    @staticmethod
    def _interrupt(cur) -> None:
        try:
            cur.interrupt()
        except Exception:
            pass

    # ---- set bookkeeping ----
    def get(self, tag: str) -> Optional[SetState]:
        with self._lock:
            return self._sets.get(tag)

    def is_ready(self, tag: str) -> bool:
        s = self.get(tag)
        return s is not None and s.state == "ready"

    def note_candidate(self, tag: str, upload_sql: str, fqn: str = "",
                       deps: Optional[List[str]] = None, steps: Optional[List[str]] = None,
                       est_bytes: int = 0, upload_name: str = "", store_key: str = "",
                       store_lanes: Optional[List[str]] = None, post_sql: Optional[List[str]] = None) -> SetState:
        """A rewritable shape over a non-resident set was seen. With `steps`
        the set is derived from `deps` (note those first). `est_bytes` is what
        the set is expected to cost on the device (§5.5, the memory budget)."""
        with self._cv:
            s = self._sets.get(tag)
            if s is None:
                s = SetState(tag=tag, upload_sql=upload_sql, fqn=fqn,
                             deps=list(deps or []), steps=list(steps or []))
                self._sets[tag] = s
            elif fqn and not s.fqn:
                s.fqn = fqn
            if est_bytes:
                s.est_bytes = int(est_bytes)      # the table may have grown since the last sighting
            if s.state in ("missing", "stale", "failed", "pending"):
                # What the next upload must fetch may have changed (lanes evicted or
                # landed meanwhile), so a sighting's recipe replaces the one the set
                # holds — including an empty upload statement, which for a store-backed
                # set means "every lane the view reads is already there". A set that is
                # only queued (pending) is refreshed too: the sighting read the store a
                # moment ago, the queued recipe may be older than the last invalidation.
                if store_key:
                    s.upload_sql = upload_sql
                    s.upload_name = upload_name
                    s.store_key = store_key
                    s.store_lanes = list(store_lanes or [])
                    s.post_sql = list(post_sql or [])
                elif not s.store_key:
                    s.upload_sql = upload_sql or s.upload_sql
                    s.upload_name = upload_name
                    s.store_lanes = list(store_lanes or [])
                    s.post_sql = list(post_sql or [])
                # a store-backed set sighted without a recipe keeps the one it has:
                # erasing it would leave a view-backed set with nothing to upload and
                # no sort cache to build, i.e. ready for free over columns that are gone
            # a set the memory budget refused stays refused until its retry time: re-queueing it on
            # every sighting would hide the reason and make the worker ask again every few ms
            refused = (s.state == "failed" and s.error.startswith(MEMORY_ERROR)
                       and time.monotonic() < s.resume_at)
            if s.state in ("missing", "stale", "failed") and s.attempts < self.max_attempts and not refused:
                s.state = "pending"
                self._cv.notify_all()
            if self.mode == "background":
                self._ensure_thread()
            return s

    def mark_ready(self, tag: str) -> None:
        with self._lock:
            s = self._sets.get(tag)
            if s is not None:
                s.state = "ready"

    def invalidate(self, tag: Optional[str] = None, prefix: Optional[str] = None) -> None:
        """Local bookkeeping; the caller runs gpu_invalidate on the database
        (which also drops any open upload session under the name). `prefix` is
        the extension's own form: a table store's key stands for every set over
        it, the way gpu_invalidate('<store>') drops the store and every view on
        it (§5.4)."""
        def under(t: str) -> bool:
            return prefix is not None and (t == prefix or t.startswith(prefix + ":"))
        with self._cv:
            now = time.monotonic()
            hit = {t for t in self._sets if (tag is None and prefix is None) or t == tag or under(t)}
            # a derived set goes with any of its sources
            hit |= {t for t, s in self._sets.items() if any(d in hit for d in s.deps)}
            for t, s in self._sets.items():
                if t in hit:
                    s.epoch += 1
                    s.last_invalidate = now
                    s.resume_at = now + self.quiet_s
                    if s.state in ("ready", "uploading", "pending"):
                        s.state = "stale"
            cur, utag = self._upload_cursor, self._uploading_tag
            self._cv.notify_all()
        if cur is not None and utag is not None and utag in hit:
            self._interrupt(cur)

    def requeue(self, tag: str) -> None:
        """Put a set back in the queue without touching its recipe. What the
        next upload must fetch depends on what the store holds now, which only
        a sighting knows (`note_candidate` recomputes it); this is the state
        change alone, for a caller that has just dropped the set's backing."""
        with self._cv:
            s = self._sets.get(tag)
            if s is None:
                return
            refused = (s.state == "failed" and s.error.startswith(MEMORY_ERROR)
                       and time.monotonic() < s.resume_at)
            if s.state in ("missing", "stale", "failed") and s.attempts < self.max_attempts and not refused:
                s.state = "pending"
                self._cv.notify_all()

    def snapshot(self) -> Dict[str, str]:
        with self._lock:
            return {t: s.state for t, s in self._sets.items()}

    def progress(self) -> Dict[str, Dict[str, object]]:
        """Test / diagnostics helper: per set, the session progress counters."""
        with self._lock:
            return {t: {"state": s.state, "segments": s.segments, "planned": s.segments_planned,
                        "rows_seen": s.rows_seen, "interrupts": s.interrupts,
                        "attempts": s.attempts, "session_ms": round(s.session_ms, 1),
                        "seg_ms": [round(x, 1) for x in s.seg_ms],
                        "finish_window": s.finish_window}
                    for t, s in self._sets.items()}

    # ---- memory budget (§5.5) ----
    def _protected(self, s: SetState) -> set:
        """Tags that must stay while `s` uploads: itself and, transitively, its sources."""
        keep, todo = {s.tag}, list(s.deps)
        while todo:
            t = todo.pop()
            if t not in keep:
                keep.add(t)
                d = self._sets.get(t)
                todo.extend(d.deps if d is not None else [])
        return keep

    def _make_room(self, run: Callable[[str], List[tuple]], s: SetState) -> bool:
        """Before an upload: does the set fit the budget, evicting least
        recently used sets if it has to? False (and s.error says why) when it
        does not — the statement then keeps running native, which is what rule
        1 asks for when the device cannot hold the data.

        The extension is the source of truth for what is resident and what it
        costs (`gpu_residents().bytes` counts lanes, validity and derived
        structures); sets uploaded by hand count toward the total and are
        never evicted. Not evictable: a set in use by a running operator, a
        source of `s`, a source of another resident set (its dependents go
        first), and anything uploaded less than evict_min_age_s ago."""
        budget = self.memory_budget
        if not budget or s.est_bytes <= 0:
            return True
        if s.est_bytes > budget:
            with self._lock:
                s.error = (f"{MEMORY_ERROR}the set needs about {s.est_bytes / 2**20:.0f} MiB, "
                           f"the budget is {budget / 2**20:.0f} MiB")
            self._log(f"not uploaded: {s.tag}: {s.error}")
            return False
        try:
            rows = run("SELECT name, origin, bytes, refs, epoch_ms(last_used_at) AS used_ms "
                       "FROM gpu_residents()")
            # stage B: the store's columns are what the views cost (a view reports 0)
            crows = run("SELECT store, \"column\", bytes, epoch_ms(last_used_at) AS used_ms FROM gpu_store_columns()")
        except Exception as e:
            self._log(f"memory budget: gpu_residents() failed ({str(e)[:80]}); uploading without a check")
            return True
        keep = self._protected(s)
        live = {r[0]: r for r in rows}
        used = sum(int(r[2] or 0) for r in rows if r[0] != s.tag)
        # columns: (store, lane) -> (bytes, last used)
        cols = {(r[0], r[1]): (int(r[2] or 0), r[3]) for r in crows}
        used += sum(b for b, _u in cols.values())
        with self._lock:
            keep_cols = {(o.store_key, l) for t, o in self._sets.items() if t in keep for l in o.store_lanes}
            young_cols = {c for c, at in self._col_uploaded.items() if time.monotonic() - at < self.evict_min_age_s}
        while used + s.est_bytes > budget:
            now = time.monotonic()
            with self._lock:
                is_source = {d for t, o in self._sets.items() if t in live and t not in keep for d in o.deps}
                # age by this manager's own clock (a set it did not upload is old by definition)
                young = {t for t, o in self._sets.items()
                         if o.last_upload_start and now - o.last_upload_start < self.evict_min_age_s}
            victims = [r for r in live.values()
                       if r[1] == "managed" and r[0] not in keep and r[0] not in is_source
                       and r[0] not in young and int(r[3] or 0) == 0 and int(r[2] or 0) > 0]
            col_victims = [(c, b, u) for c, (b, u) in cols.items()
                           if c not in keep_cols and c not in young_cols and b > 0]
            if col_victims and (not victims or
                                min(v[4] or 0 for v in victims) > min((u or 0) for _c, _b, u in col_victims)):
                # the least recently used thing is a column: drop it (and the views on it)
                c, b, _u = min(col_victims, key=lambda x: (x[2] is not None, x[2] or 0))
                try:
                    run("SELECT gpu_drop_column('%s', '%s')" % (c[0].replace("'", "''"), c[1].replace("'", "''")))
                except Exception as e:
                    self._log(f"memory budget: could not evict column {c}: {str(e)[:80]}")
                    cols.pop(c, None)
                    continue
                used -= b
                cols.pop(c, None)
                with self._lock:
                    self.evictions += 1
                    self._col_uploaded.pop(c, None)
                    gone = set()
                    for t, o in self._sets.items():
                        if o.store_key == c[0] and c[1] in o.store_lanes and o.state == "ready":
                            o.state = "missing"          # its next sighting uploads the missing lane
                            o.bytes = 0
                            gone.add(t)
                    self._demote_dependents_locked(gone)
                self._log(f"evicted (least recently used): column {c[1]} of {c[0]} ({b / 2**20:.0f} MiB) for {s.tag}")
                continue
            if not victims:
                managed = [r for r in live.values() if r[1] == "managed" and int(r[2] or 0) > 0]
                why = (f"{len(managed)} managed sets: {sum(r[0] in keep for r in managed)} needed by this upload, "
                       f"{sum(r[0] in is_source for r in managed)} sources of resident sets, "
                       f"{sum(r[0] in young for r in managed)} younger than {self.evict_min_age_s:.0f} s, "
                       f"{sum(int(r[3] or 0) > 0 for r in managed)} in use")
                with self._lock:
                    s.error = (f"{MEMORY_ERROR}{used / 2**20:.0f} MiB resident + about "
                               f"{s.est_bytes / 2**20:.0f} MiB needed > {budget / 2**20:.0f} MiB, "
                               f"and nothing can be evicted yet ({why})")
                self._log(f"not uploaded: {s.tag}: {s.error}")
                return False
            v = min(victims, key=lambda r: (r[4] is not None, r[4] or 0))  # never used, then least recently used
            try:
                run("SELECT gpu_drop_resident('%s')" % v[0].replace("'", "''"))
            except Exception as e:
                self._log(f"memory budget: could not evict {v[0]}: {str(e)[:80]}")
                live.pop(v[0], None)
                continue
            used -= int(v[2] or 0)
            live.pop(v[0], None)
            with self._lock:
                self.evictions += 1
                o = self._sets.get(v[0])
                if o is not None and o.state == "ready":
                    o.state = "missing"            # a later sighting uploads it again
                    o.bytes = 0
                    self._demote_dependents_locked({v[0]})
            self._log(f"evicted (least recently used): {v[0]} ({int(v[2] or 0) / 2**20:.0f} MiB) for {s.tag}")
        return True

    def _store_holds(self, run: Callable[[str], List[tuple]], s: SetState) -> bool:
        """Stage B: does the store still hold every lane the set's view reads?

        A view is not a durable object — it is synthesised from the store on
        lookup and exists only while every lane it names is there — so `ready`
        for a view-backed set is never a stored fact. The one place that would
        otherwise take it on trust is a session with nothing to upload, which
        is where this is asked; it runs once per cold set, never per statement.
        True when the question cannot be answered: an upload is then attempted
        the usual way and the extension has the last word."""
        if not s.store_key or not s.store_lanes:
            return True
        try:
            rows = run("SELECT \"column\" FROM gpu_store_columns() WHERE store = '%s'"
                       % s.store_key.replace("'", "''"))
        except Exception:
            return True
        have = {r[0] for r in rows}
        return all(l in have or ("k#" + l) in have for l in s.store_lanes)

    def _extension_holds(self, run: Callable[[str], List[tuple]], s: SetState) -> bool:
        """The extension's own answer to 'is this set there': the store's lanes
        for a view, the registry for a set of its own."""
        if s.store_key:
            return self._store_holds(run, s)
        try:
            row = run("SELECT state FROM gpu_residents() WHERE name = '%s'" % s.tag.replace("'", "''"))
        except Exception:
            return True
        return bool(row) and row[0][0] == "ready"

    def _demote_dependents_locked(self, gone: set) -> None:
        """A derived set was materialised FROM its sources: when one of them is
        no longer resident the derived set is not ready either, however
        recently it answered. Caller holds the lock."""
        while gone:
            nxt = set()
            for t, o in self._sets.items():
                if o.state == "ready" and t not in gone and any(d in gone for d in o.deps):
                    o.state = "missing"        # its next sighting materialises it again
                    o.bytes = 0
                    nxt.add(t)
            gone = nxt

    def _note_columns(self, s: SetState) -> None:
        if not s.store_key:
            return
        with self._lock:
            now = time.monotonic()
            for l in s.store_lanes:
                self._col_uploaded.setdefault((s.store_key, l), now)

    def _note_bytes(self, run: Callable[[str], List[tuple]], s: SetState) -> None:
        try:
            if s.store_key:
                # a view owns nothing: what its lanes cost in the store (shared with other views)
                rows = run("SELECT \"column\", bytes FROM gpu_store_columns() WHERE store = '%s'" % s.store_key.replace("'", "''"))
                have = {r[0]: int(r[1]) for r in rows}
                total = sum(have.get(l, have.get("k#" + l, 0)) for l in s.store_lanes)
            else:
                row = run("SELECT bytes FROM gpu_residents() WHERE name = '%s'" % s.tag.replace("'", "''"))
                total = int(row[0][0]) if row else 0
            with self._lock:
                s.bytes = total
        except Exception:
            pass

    def memory(self) -> Dict[str, object]:
        """Diagnostics: the budget, and per set the estimate and the extension's figure."""
        with self._lock:
            return {"budget": self.memory_budget, "evictions": self.evictions,
                    "sets": {t: {"state": s.state, "est_bytes": s.est_bytes, "bytes": s.bytes,
                                 "error": s.error} for t, s in self._sets.items()}}

    # ---- synchronous upload (residency='eager', and tests) ----
    def upload_now(self, tag: str, run: Callable[[str], None]) -> bool:
        """One statement over the whole table on the caller's connection: the
        caller asked for it (eager), so it runs beside nothing."""
        s = self.get(tag)
        if s is None:
            return False
        for d in s.deps:
            if not self.is_ready(d) and not self.upload_now(d, run):
                ds = self.get(d)
                with self._lock:
                    # a source refused by the memory budget: this set is refused for the same reason
                    # (the statement's reason must read 'memory', not 'not_resident')
                    if ds is not None and ds.error.startswith(MEMORY_ERROR):
                        s.state = "failed"
                        s.error = f"{MEMORY_ERROR}source set {d}: {ds.error[len(MEMORY_ERROR):]}"
                        s.resume_at = ds.resume_at
                    elif ds is not None and ds.state == "missing":
                        # the source has to be uploaded again first and knows how; this set
                        # is simply not there yet, and its next sighting builds the chain
                        s.state = "missing"
                        s.error = f"source set {d} is being uploaded again: {ds.error[:120]}"
                    else:
                        s.state = "failed"
                        s.error = f"source set {d} is not resident"
                return False
        if not self._make_room(run, s):
            with self._lock:
                s.state = "failed"
                s.attempts += 1
            return False
        if not s.derived and not s.upload_sql and not self._store_holds(run, s):
            # the recipe says every lane is in the store and the store says otherwise:
            # it predates an invalidation, so the next sighting must recompute it
            with self._lock:
                s.state = "missing"
                s.error = self._store_gone_error(s)
            self._log(f"not uploaded: {s.tag}: {s.error}")
            return False
        with self._lock:
            s.state = "uploading"
            s.attempts += 1
            s.last_upload_start = time.monotonic()
            epoch = s.epoch
        try:
            for stmt in (s.steps if s.derived else ([s.upload_sql] if s.upload_sql else [])):
                run(stmt)
            for stmt in s.post_sql:
                run(stmt)
        except Exception as e:
            err = str(e)
            if "GPUDB_UPLOAD_DISCARDED" in err:
                with self._lock:
                    s.state = "stale"
                    s.error = err[:200]
                return False
            if _not_resident(err):
                # a set this one is built from is not there after all: say so where the
                # statement can read it, re-derive the sources from the extension, and
                # leave this set where its next sighting builds it again
                self._recheck_sources(run, s, err)
                with self._lock:
                    s.state = "missing"
                    s.error = err[:200]
                self._log(f"not uploaded: {s.tag}: {err[:120]}")
                return False
            with self._lock:
                s.state = "failed"
                s.error = err[:200]
            return False
        self._note_bytes(run, s)
        self._note_columns(s)
        with self._lock:
            if s.epoch == epoch and s.state == "uploading":
                s.state = "ready"
                return True
            s.state = "stale"
            return False

    # ---- background worker ----
    def _ensure_thread(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._worker, name="gpudb-residency",
                                            daemon=True)
            self._thread.start()

    def _pick(self) -> Optional[SetState]:
        now = time.monotonic()
        for s in self._sets.values():
            if s.state != "pending" or now < s.resume_at:
                continue
            if s.deps:
                deps = [self._sets.get(d) for d in s.deps]
                if any(d is None or d.state == "failed" for d in deps):
                    s.state = "failed"
                    bad = next((d for d in deps if d is not None and d.state == "failed"), None)
                    if bad is not None and bad.error.startswith(MEMORY_ERROR):
                        s.error = f"{MEMORY_ERROR}source set {bad.tag}: {bad.error[len(MEMORY_ERROR):]}"
                        s.resume_at = bad.resume_at
                    else:
                        s.error = "a source set failed to upload"
                    continue
                if any(d.state != "ready" for d in deps):
                    continue                      # its sources first
            return s
        return None

    def _idle_ms(self) -> float:
        return (time.monotonic() - self._last_activity) * 1000.0

    def _wait_idle(self, s: SetState, epoch: int, idle_ms: float,
                   not_before: float = 0.0) -> bool:
        """Block (lock held) until no statement is in flight, the wrapper has
        been idle for idle_ms, and time.monotonic() >= not_before. False when
        the session must stop: manager closed, or the set invalidated (epoch
        moved) meanwhile."""
        while True:
            if self._closed or s.epoch != epoch or s.state != "uploading":
                return False
            now = time.monotonic()
            if now >= not_before and self._in_flight == 0 and self._idle_ms() >= idle_ms:
                return True
            wait_s = max(0.002, idle_ms / 1000.0)
            if now < not_before:
                wait_s = max(wait_s, not_before - now)
            self._cv.wait(timeout=wait_s)

    def _run(self, cur, s: SetState, sql: str, params=None) -> List[tuple]:
        """Execute one statement on the upload cursor, marking it interruptible
        for the duration. Raises whatever the cursor raises."""
        with self._lock:
            self._uploading_tag = s.tag
        try:
            return cur.execute(sql, params).fetchall() if params else cur.execute(sql).fetchall()
        finally:
            with self._lock:
                self._uploading_tag = None

    def _session_status(self, cur, s: SetState) -> Dict[str, object]:
        """The extension's view of the open session. Not interruptible (it
        is a sub-millisecond scalar with no device work, so it may run beside
        a user statement): an interrupted status read as "session closed"
        would throw away every segment that already landed. A late interrupt
        aimed at the segment statement can still hit it, hence the retries."""
        for _ in range(5):
            try:
                row = cur.execute("SELECT gpu_upload_status(?)", [s.session_name]).fetchall()
                return json.loads(row[0][0])
            except Exception as e:
                if not _is_interrupt(str(e)):
                    break
        return {"open": False}

    def _abort(self, cur, s: SetState) -> None:
        try:
            cur.execute("SELECT gpu_upload_abort(?)", [s.session_name]).fetchall()
        except Exception:
            pass

    def _extension_ready(self, cur, s: SetState) -> Optional[int]:
        """rows_seen of the set if the extension holds it ready, else None."""
        try:
            if s.store_key:
                rows = cur.execute("SELECT \"column\", rows FROM gpu_store_columns() WHERE store = ?",
                                   [s.store_key]).fetchall()
                have = {r[0]: int(r[1]) for r in rows}
                if all(l in have for l in s.store_lanes) and s.store_lanes:
                    return have[s.store_lanes[0]]
                return None
            row = cur.execute("SELECT state, rows_seen FROM gpu_residents() WHERE name = ?",
                              [s.tag]).fetchone()
            return int(row[1]) if row and row[0] == "ready" else None
        except Exception:
            return None

    def _session(self, cur, s: SetState, epoch: int) -> str:
        """One upload session for the set. Returns the outcome: ready | pending
        (retry later) | recheck (the recipe predates an invalidation) | stale
        (invalidated meanwhile) | failed."""
        t_start = time.monotonic()
        run = lambda q: self._run(cur, s, q)     # noqa: E731
        if s.derived:
            return self._session_derived(cur, s, epoch, t_start)
        if not s.upload_sql:
            # every lane is already in the store: only the view's sort cache is missing
            if not self._store_holds(run, s):
                with self._lock:
                    s.error = self._store_gone_error(s)
                self._log(f"not uploaded: {s.tag}: {s.error}")
                return "recheck"
            try:
                for stmt in s.post_sql:
                    self._run(cur, s, stmt)
            except Exception as e:
                err = str(e)
                if _is_interrupt(err):
                    return "pending"
                if _not_resident(err):
                    # the acquire the sort cache does is the second reader of the store,
                    # and it has just said the lanes are not there after all
                    with self._lock:
                        s.error = self._store_gone_error(s)
                    return "recheck"
                return self._fail(s, e)
            return "ready"
        fqn = s.fqn or s.upload_sql.split(" FROM ", 1)[1]
        seg_rows = self.segment_rows or s.segment_rows_default
        idle_ms = self.idle_ms
        # 1. bounds (metadata-fast; the rowid range is stable while no write
        #    lands, and every write through the wrapper invalidates the set)
        try:
            row = self._run(cur, s, f"SELECT max(rowid) FROM {fqn}")
        except Exception as e:
            return "pending" if _is_interrupt(str(e)) else self._fail(s, e)
        max_rowid = row[0][0] if row else None
        if max_rowid is None:
            return self._fail(s, RuntimeError("table is empty"))
        planned = (int(max_rowid) // seg_rows) + 1
        with self._lock:
            s.segments = 0
            s.segments_planned = planned
            s.rows_seen = 0
            s.seg_ms = []
        # 2. begin
        try:
            self._run(cur, s, "SELECT gpu_upload_begin(?)", [s.session_name])
        except Exception as e:
            return "pending" if _is_interrupt(str(e)) else self._fail(s, e)
        # 3. segments, each only in an idle window
        a, done, consecutive_interrupts, not_before = 0, 0, 0, 0.0
        while a <= max_rowid:
            with self._cv:
                ok = self._wait_idle(s, epoch, idle_ms, not_before)
            if not ok:
                self._abort(cur, s)
                return "closed" if self._closed else "stale"
            b = a + seg_rows
            seg_sql = f"{s.upload_sql} WHERE rowid >= {a} AND rowid < {b}"
            try:
                t_seg = time.monotonic()
                self._run(cur, s, seg_sql)
                landed = True
                with self._lock:
                    s.seg_ms.append((time.monotonic() - t_seg) * 1000.0)
            except Exception as e:
                if not _is_interrupt(str(e)):
                    self._abort(cur, s)
                    return self._fail(s, e)
                # interrupted: did the segment land before the interrupt was seen?
                st = self._session_status(cur, s)
                if not st.get("open") or st.get("invalidated"):
                    self._abort(cur, s)
                    return "stale"
                landed = int(st.get("segments", 0)) == done + 1
                consecutive_interrupts += 1
                pause = min(RETRY_PAUSE_MS * (2.0 ** (consecutive_interrupts - 1)),
                            RETRY_PAUSE_MAX_MS)
                not_before = time.monotonic() + pause / 1000.0
                with self._lock:
                    s.interrupts += 1
            if landed:
                a, done = b, done + 1
                consecutive_interrupts, not_before = 0, 0.0
                with self._lock:
                    s.segments = done
        # 4. finish: device copy + prepare + publish (one scalar call; an
        #    interrupt arriving during it is only seen after it returns)
        with self._cv:
            ok = self._wait_idle(s, epoch, self.idle_ms)
        if not ok:
            self._abort(cur, s)
            return "closed" if self._closed else "stale"
        t_fin = time.monotonic()
        try:
            row = self._run(cur, s, "SELECT gpu_upload_finish(?)", [s.session_name])
            rows = int(row[0][0])
        except Exception as e:
            err = str(e)
            with self._lock:
                s.finish_window = (t_fin, time.monotonic())
            if "GPUDB_UPLOAD_DISCARDED" in err:
                return "stale"
            if _is_interrupt(err):
                # the interrupt was seen after the scalar returned: the set
                # may already be published — then it is ready, not lost
                published = self._extension_ready(cur, s)
                if published is not None:
                    rows = published
                else:
                    self._abort(cur, s)
                    return "pending"
            else:
                return self._fail(s, e)
        try:
            for stmt in s.post_sql:                  # the view's sort cache, once the columns are there
                self._run(cur, s, stmt)
        except Exception as e:
            if not _is_interrupt(str(e)):
                return self._fail(s, e)
        with self._lock:
            s.rows_seen = rows
            s.finish_window = (t_fin, time.monotonic())
            s.session_ms = (time.monotonic() - t_start) * 1000.0
        return "ready"

    def _session_derived(self, cur, s: SetState, epoch: int, t_start: float) -> str:
        """Materialise a derived set: each step is one scalar call on the
        device (tens of ms), issued only in an idle window like a finish."""
        rows = 0
        for stmt in s.steps:
            with self._cv:
                ok = self._wait_idle(s, epoch, self.idle_ms)
            if not ok:
                return "closed" if self._closed else "stale"
            try:
                row = self._run(cur, s, stmt)
                if "gpu_join_materialize" in stmt and row:
                    rows = int(row[0][0])
            except Exception as e:
                err = str(e)
                if _is_interrupt(err):
                    return "pending"              # steps are idempotent: run the chain again
                if _not_resident(err):
                    self._recheck_sources(lambda q: self._run(cur, s, q), s, err)
                    return "stale"                # a source changed under the chain
                return self._fail(s, e)
        with self._lock:
            s.rows_seen = rows
            s.segments = s.segments_planned = len(s.steps)
            s.session_ms = (time.monotonic() - t_start) * 1000.0
        return "ready"

    @staticmethod
    def _store_gone_error(s: SetState) -> str:
        return (f"the view's lanes ({', '.join(s.store_lanes)}) are no longer in the store "
                f"{s.store_key} — the set is uploaded again on its next sighting")

    def _recheck_sources(self, run: Callable[[str], List[tuple]], s: SetState, err: str) -> None:
        """A materialise step said a source is gone or changed. A source is a
        VIEW over its table's store, so its `ready` is not a stored fact: ask
        the extension about each one and put back in the queue whichever it no
        longer holds, or the chain would be retried against the same absence
        for as long as the statement keeps being seen."""
        for d in s.deps:
            o = self._sets.get(d)
            if o is None or o.state != "ready" or self._extension_holds(run, o):
                continue
            with self._lock:
                o.state = "missing"           # its next sighting uploads it again
                o.bytes = 0
                o.error = (self._store_gone_error(o) if o.store_key else
                           "the extension no longer holds this set — it is uploaded again on its next sighting")
                self._demote_dependents_locked({d})
            self._log(f"source not resident: {d} ({err[:100]})")
        with self._lock:
            s.error = err[:200]

    def _fail(self, s: SetState, e: Exception) -> str:
        with self._lock:
            s.error = str(e)[:200]
        self._log(f"upload failed: {s.tag}: {str(e)[:120]}")
        return "failed"

    def _worker(self) -> None:
        while True:
            with self._cv:
                while not self._closed:
                    s = self._pick()
                    if s is not None and self._in_flight == 0 and self._idle_ms() >= self.idle_ms:
                        break
                    self._cv.wait(timeout=max(0.005, self.idle_ms / 1000.0))
                if self._closed:
                    return
                s.state = "uploading"
                s.attempts += 1
                s.last_upload_start = time.monotonic()
                s.error = ""
                epoch = s.epoch
                if self._upload_cursor is None:
                    try:
                        self._upload_cursor = self._cursor_factory()
                    except Exception as e:
                        s.state = "failed"
                        s.error = str(e)[:200]
                        continue
                cur = self._upload_cursor
            run = lambda q, _c=cur: _c.execute(q).fetchall()      # noqa: E731  (sub-millisecond, no device work)
            if not self._make_room(run, s):
                outcome = "failed"
            else:
                outcome = self._session(cur, s, epoch)
                if outcome == "ready":
                    self._note_bytes(run, s)
                    self._note_columns(s)
            with self._cv:
                now = time.monotonic()
                if outcome == "ready" and s.epoch == epoch and s.state == "uploading":
                    s.state = "ready"
                    self._log(f"resident: {s.tag} ({s.segments} segments, "
                              f"{s.interrupts} interrupts, {s.session_ms:.0f} ms)")
                elif outcome == "pending" and s.epoch == epoch and s.state == "uploading":
                    s.state = "pending"           # retry once idle again; no rate wait
                    s.resume_at = now
                elif outcome == "recheck" and s.epoch == epoch and s.state == "uploading":
                    s.state = "missing"           # the recipe is out of date: the next sighting recomputes it
                elif outcome == "failed":
                    s.state = "failed"
                    s.resume_at = s.last_upload_start + self.rate_s
                elif outcome == "closed":
                    s.state = "pending"
                else:
                    # invalidated while uploading (epoch moved): the caller
                    # already set stale + resume_at; a new sighting re-queues it
                    if s.state == "uploading":
                        s.state = "stale"
                        s.resume_at = max(s.resume_at, now + self.quiet_s)
                self._cv.notify_all()

    def close(self) -> None:
        with self._cv:
            self._closed = True
            cur, tag = self._upload_cursor, self._uploading_tag
            self._cv.notify_all()
        if cur is not None and tag is not None:
            self._interrupt(cur)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self._upload_cursor is not None:
            try:
                self._upload_cursor.close()
            except Exception:
                pass
            self._upload_cursor = None

    def wait_idle(self, timeout: float = 30.0) -> bool:
        """Test helper: block until no set is pending/uploading."""
        deadline = time.monotonic() + timeout
        with self._cv:
            while any(s.state in ("pending", "uploading") for s in self._sets.values()):
                if time.monotonic() > deadline:
                    return False
                self._cv.wait(timeout=0.05)
        return True
