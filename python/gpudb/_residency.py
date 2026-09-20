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

Two steps of a session an interrupt CANNOT stop: `gpu_upload_finish` (the
device copy, ~110-140 ms for a lineitem-sized set) and the sort cache built
after it (~150-170 ms). They are the whole of the upload's measured p99 cost
to a user statement, and they only cost anything when the machine's cores
are oversubscribed (2026-09-20, docs/RESEARCH_NOTES.md). So the session
measures, for free, how many cores the machine is giving it — CPU seconds
per wall second of a completed segment scan — and when that says the cores
are contended, a step an interrupt cannot stop waits for a genuinely quiet
connection before it starts. The window it asks for decays to the ordinary
idle threshold over `quiet_max_s`, so the step takes the best window the
connection offers and, if the connection never offers one, runs anyway at
the deadline: readiness is delayed by at most `quiet_max_s` per such step,
never withheld.
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple

SEGMENT_BYTES = 8 << 20          # the extension's host segment (gpu_resident.cpp)
RETRY_PAUSE_MS = 50.0            # after an interrupted segment: this, doubling per consecutive interrupt
EVICT_MIN_AGE_S = 60.0            # §5.5: a set is never evicted within this long of being uploaded (anti-thrash)
MEMORY_ERROR = "memory budget: "  # SetState.error prefix of a set the budget kept off the device
POST_MAX_ATTEMPTS = 8            # interrupted sort-cache attempts before the set goes ready without it
RETRY_PAUSE_MAX_MS = 1000.0      # the idle wait already yields to every statement; a longer cap only
                                 # delayed readiness (measured: 15 of 20 segments landed in 3 s, the
                                 # rest took 30+ s at a 5 s cap under a 0-10 ms statement cadence)

# ---- a step an interrupt cannot stop, on a machine whose cores are taken ----
# The quiet window such a step asks for. The longest one measured at SF10 is
# the sort cache at 171 ms, the device copy is 110-140 ms; 400 ms is the pair
# with margin, and it is a gap a connection that is between bursts — rather
# than in one — offers without trying.
HEAVY_QUIET_MS = 400.0
# ... and the longest it waits for that window. The window asked for decays
# linearly to `idle_ms` across this, so the step takes the best gap the
# connection offers in the meantime and runs unconditionally at the end. This
# is the whole of the mechanism's worst case: a session is delayed by at most
# this per step an interrupt cannot stop (one device copy plus one sort cache
# for a single-table set, so at most 2x this), never by more, and never
# indefinitely.
HEAVY_QUIET_MAX_S = 20.0
# How many cores a completed segment scan gets when nothing else wants them,
# as a fraction of os.cpu_count(): measured on this machine 10.1 of 16 idle
# and 6.5 of 16 with 8 cores busy, so half the machine separates the two with
# no overlap (idle median 10.1 against a loaded maximum of 7.1).
CONTENDED_CORES_FRACTION = 0.5
CONTENTION_SEGMENTS = 16         # completed segments the estimate is taken over (the most recent ones)
CONTENTION_MIN_SAMPLES = 3       # ... below which the machine is called free, i.e. today's behaviour
# A scan DuckDB did not parallelise says nothing about how many cores the
# machine has to spare, and DuckDB parallelises a table scan by ROW GROUP
# (122,880 rows). Measured: a 100K-row segment — less than one row group, so a
# serial scan — reports 2.6-2.7 cores of 16 with nothing else running, at 1.4 ms
# and at 5 ms alike, while the 1M-row segments of a lineitem-sized set report
# 9.2-10.1. So a segment votes only when it spans at least four row groups and
# ran long enough to have been worth parallelising; otherwise the estimate stays
# empty and the manager behaves as it did before this mechanism existed.
DUCKDB_ROW_GROUP = 122880
CONTENTION_MIN_SEG_ROWS = 4 * DUCKDB_ROW_GROUP
CONTENTION_MIN_SEG_MS = 2.0

# ---- a segment that does not fit the window the workload leaves ----
# A segment's cost is host work — lanes x rows, plus validity — so the same
# `segment_rows` is a different number of milliseconds on different machines
# and for different sets, while the window a workload leaves between its
# statements is whatever it is. When the two do not fit, almost every attempt
# is interrupted and the session makes no progress (measured on the x86 box,
# 2026-09-20: 2 of 69 attempts landed at a 2.5 ms mean window against a 4.9 ms
# segment, where this machine lands 58 of 95). So the size follows the yield:
# halve it when attempts stop landing, double it back when they do.
SEG_YIELD_WINDOW = 8             # attempts a decision is taken over
# 1 of 8 or fewer land -> halve the segment. The threshold is low on purpose:
# this machine's yield is ~0.6, where a window of 8 falls to 3 or fewer 17% of
# the time, so a 3-of-8 rule made the size oscillate here for no reason (58
# segments became 75-81 under load). At 1 of 8 a yield of 0.6 trips 0.9% of
# the time and the starving box's 0.03 trips at once — which is the case the
# rule is for. It is a starvation guard, not a tuning knob.
SEG_YIELD_LOW = 0.125
SEG_YIELD_HIGH = 0.875           # 7 of 8 or more land -> give the halving back
SEG_SCALE_MAX = 32               # 1/32 of the 8 MiB segment: the floor while nothing is measured

# ---- ... and how small is worth going, which the machine has to say ----
# A segment has a FIXED cost no smaller segment escapes: the statement's own
# parse and bind, and DuckDB's scan set-up over every row group of the table.
# Measured through this manager on an M4 Max over a 2M-row table (2026-09-20,
# docs/RESEARCH_NOTES.md): 0.40 ms for a segment of 8192 rows against 2.6 ms
# for one of 524288 — a sixty-fourth of the rows for a sixth of the time, so
# the same table costs 98 ms of scanning in small pieces where it costs 10 ms
# in default ones. So a halving is judged on the TOTAL work it adds, and the
# shrinking stops at the smallest size whose total for this table stays within
# SEG_TOTAL_MAX_RATIO of the default's (`_segment_floor_locked`). Where a
# segment really does cost `fixed + rows x per_row` that is the same size at
# which the two parts are equal. A connection that leaves no window even at
# the floor is STARVED rather than ground finer: the set does not become
# resident, every statement keeps its native answer, and progress() says so
# (`starved`, with the window and the segment cost it was judged on).
#
# How much more total scanning the background upload may do to fit a workload
# that leaves small gaps, against what the table costs at the default size.
# Four, because that is what the two machines' measured curves say the useful
# range is worth: on the x86 box the table costs 46 ms in default segments and
# 152 ms in 32768-row ones (3.3x) — which is where the constant floor of
# 1/SEG_SCALE_MAX happened to sit, now derived instead of assumed — and 427 ms
# at 8192 rows, which buys a segment only 30% cheaper to attempt. On the M4 Max
# the same budget stops at 65536 rows. Raising it trades background CPU for a
# chance at residency under a tighter cadence; nothing else changes.
SEG_TOTAL_MAX_RATIO = 4.0
# What an UNTRIED halving is assumed to buy, until a segment of that size has
# landed and said. A halving that buys less than a quarter of the cost is not
# worth taking at all, so an untried one is priced at exactly that: the most it
# could be worth without being refused outright. With the budget above it lets
# the size fall three halvings below the last measured one before the total
# stops it — and every size it passes through is measured on the way, so the
# machine's real curve takes over from this guess within a few segments.
SEG_BLIND_GAIN = 0.75
SEG_COST_SAMPLES = 8             # landed segments kept per size, for the fit
SEG_WINDOW_SAMPLES = 16          # recent windows the workload left, for the report
SEG_MAX_SEGMENTS = 2048          # ... and a table is never cut into more pieces than this

# ---- what a resident set is WORTH (§5.5, value-aware residency) ----
# A set's value is a decaying rate: the milliseconds it saves per second of
# wall time, `sum over uses of saved_ms * exp(-age / VALUE_TAU_S) / VALUE_TAU_S`.
# It is kept as one float updated in O(1) per use, so yesterday's hot set stops
# holding memory a few minutes after it goes quiet. Divided by the bytes the
# set costs it becomes a value DENSITY, which is what admission compares.
VALUE_TAU_S = 300.0
# A candidate must beat the least valuable thing it would evict by this much
# before anything is evicted at all; without a margin two sets of nearly equal
# value would take turns evicting each other.
EVICT_HYSTERESIS = 0.25
# ... and by this much to evict something the minimum age still protects. What
# the minimum age is FOR is that a set gets the chance to pay back its upload,
# and what paying back means is answering a statement — so a unit is protected
# while it is younger than evict_min_age_s AND has not yet saved anybody
# anything. After that it competes on value like everything else, however
# young: measured, a set is no longer a guess that needs shielding. (The
# 22-query run is the case this was decided on: it fills the budget inside
# 60 s, and every set in it has already answered by the time the next
# statement arrives.) An unused set still has this ratio as its one way out,
# so a speculative upload that nothing ever reads cannot hold memory against a
# statement that would use it.
YOUNG_OVERRIDE_RATIO = 2.0
# A use that ran NATIVE (the set was not resident, so the saving is an
# estimate, not a measurement) is worth this fraction of a resident use. It is
# what makes residency sticky: at equal true value a resident set's accumulator
# grows twice as fast as the value of the candidate trying to displace it, so
# the two cannot trade places.
NATIVE_USE_WEIGHT = 0.5


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
    # what waiting for a quiet connection cost this set, for the gate to print:
    # how many steps an interrupt cannot stop had to wait, for how long in
    # total, and how many of them gave up waiting and ran at the deadline
    quiet_waits: int = 0
    quiet_wait_ms: float = 0.0
    quiet_forced: int = 0
    segment_rows_used: int = 0      # the size the yield settled on for the last session
    table_rows: int = 0             # max(rowid) + 1, from the session's bounds statement
    # what a landed segment of each size COST this set, in ms — the two ends of
    # it are the fixed part and the per-row part of a segment on this machine
    seg_cost: Dict[int, List[float]] = field(default_factory=dict)
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
    # §5.5 value: the decaying saved-ms-per-second accumulator, the clock it was
    # last brought up to date on, and the use counters behind it
    value: float = 0.0
    value_at: float = 0.0
    uses: int = 0
    measured_uses: int = 0          # uses whose saving was measured, not estimated
    last_use: float = 0.0
    override_evicted_at: float = 0.0  # ... when it was last evicted by a minimum-age override

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


def _env_float(name: str, default: float) -> float:
    try:
        v = os.environ.get(name)
        return default if v is None or v == "" else max(0.0, float(v))
    except ValueError:
        return default


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
                 log: Optional[Callable[[str], None]] = None,
                 clock: Optional[Callable[[], float]] = None,
                 value_tau_s: float = VALUE_TAU_S):
        self._cursor_factory = cursor_factory
        self._clock = clock or time.monotonic
        self.value_tau_s = value_tau_s
        self.mode = mode
        self.idle_ms = idle_ms
        self.quiet_s = quiet_s
        self.rate_s = rate_s
        self.max_attempts = max_attempts
        self.segment_rows = segment_rows        # None: 8 MiB worth of rows for the set's kind
        self.memory_budget = memory_budget      # bytes of device memory for resident sets; None = no cap
        self.evict_min_age_s = evict_min_age_s
        self.evictions = 0
        # evictions that bought nothing: a candidate that was refused AFTER
        # something had already been dropped for it. Plan-then-execute makes
        # this structurally unreachable except when a drop itself fails, and it
        # is counted rather than assumed (§5.5).
        self.evictions_wasted = 0
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
        # ---- contention (§5.5): how many cores the machine is giving us ----
        self._cpus = float(os.cpu_count() or 1)
        # CPU seconds per wall second of the most recent completed segment
        # scans. Only COMPLETED segments count: a segment a statement arrived
        # on was interrupted, and its CPU time would be the statement's too.
        self._seg_cores: List[float] = []
        # the segment size the machine and the workload have settled on, as a
        # power-of-two divisor of the 8 MiB default, and the recent attempts
        # (landed or interrupted) it is decided from
        self._seg_scale = 1
        self._seg_landed: List[bool] = []
        # the windows the workload actually left, in ms: for an interrupted
        # attempt, how long it ran before the statement arrived. What a segment
        # has to fit into, measured rather than assumed.
        self._seg_windows: List[float] = []
        # the yield asked for a smaller segment and the measured floor refused:
        # this connection leaves no window a segment of any worthwhile size
        # fits (see `_segment_floor_locked`)
        self._seg_starved = False
        self._plan_seq = 0               # names the session's PREPAREd segment statement
        # the quiet window a step an interrupt cannot stop asks for on a
        # contended machine; 0 turns the mechanism off (GPUDB_UPLOAD_QUIET_MS)
        self.quiet_ms = _env_float("GPUDB_UPLOAD_QUIET_MS", HEAVY_QUIET_MS)
        self.quiet_max_s = _env_float("GPUDB_UPLOAD_QUIET_MAX_S", HEAVY_QUIET_MAX_S)
        # GPUDB_RESIDENCY_TRACE=1: one stderr line per segment attempt and per
        # quiet wait — how long the session waited for its window, how long the
        # scan then ran, how many cores it got, and whether it landed. This is
        # what a box where segments starve has to be read with.
        self._trace = os.environ.get("GPUDB_RESIDENCY_TRACE", "") not in ("", "0")

    def _now(self) -> float:
        """The clock the residency POLICY runs on — ages, decay, cooldowns.
        `time.monotonic` everywhere but in the policy's own tests, which drive
        it by hand so a minute of anti-thrash takes no time to test. The upload
        session's waits keep using the real clock: they are waits, not policy."""
        return self._clock()

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
                       and self._now() < s.resume_at)
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
            now = self._now()
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
                       and self._now() < s.resume_at)
            if s.state in ("missing", "stale", "failed") and s.attempts < self.max_attempts and not refused:
                s.state = "pending"
                self._cv.notify_all()

    def snapshot(self) -> Dict[str, str]:
        with self._lock:
            return {t: s.state for t, s in self._sets.items()}

    def progress(self) -> Dict[str, Dict[str, object]]:
        """Test / diagnostics helper: per set, the session progress counters."""
        with self._lock:
            cores = (statistics.median(self._seg_cores)
                     if len(self._seg_cores) >= CONTENTION_MIN_SAMPLES else 0.0)
            window = statistics.median(self._seg_windows) if self._seg_windows else 0.0
            return {t: {"state": s.state, "segments": s.segments, "planned": s.segments_planned,
                        "rows_seen": s.rows_seen, "interrupts": s.interrupts,
                        "attempts": s.attempts, "session_ms": round(s.session_ms, 1),
                        "seg_ms": [round(x, 1) for x in s.seg_ms],
                        "finish_window": s.finish_window,
                        # §5.5 back-off: the machine's answer to how many cores
                        # it has to spare, and what waiting for quiet cost
                        "cores": round(cores, 2), "cpus": self._cpus,
                        "quiet_waits": s.quiet_waits,
                        "quiet_wait_ms": round(s.quiet_wait_ms, 1),
                        "quiet_forced": s.quiet_forced,
                        # ... and the segment size the yield settled on (inline:
                        # segment_rows_for would take the lock this holds)
                        "segment_rows": (s.segment_rows_used or int(self.segment_rows or 0)
                                         or max(1, s.segment_rows_default // self._seg_scale)),
                        "seg_scale": self._seg_scale,
                        # §5.5 the segment that fits: what a segment of this set
                        # costs here (fixed part and per-row part, fitted to the
                        # segments that landed), the smallest size still worth
                        # taking, the window this workload has been leaving, and
                        # whether the two have stopped meeting
                        "fixed_ms": round(model[0], 3) if model else 0.0,
                        "per_row_us": round(model[1] * 1000.0, 4) if model else 0.0,
                        "floor_rows": self._segment_floor_locked(s),
                        "window_ms": round(window, 2),
                        "starved": self._seg_starved}
                    for t, s, model in ((t, s, self.segment_cost_model(s))
                                        for t, s in self._sets.items())}

    # ---- value: what a set is worth (§5.5) ----
    def _decay_locked(self, s: SetState, now: float) -> None:
        """Bring the accumulator up to `now`. Caller holds the lock."""
        if s.value_at and s.value:
            dt = now - s.value_at
            if dt > 0.0:
                s.value *= math.exp(-dt / self.value_tau_s)
        s.value_at = now

    def _value_locked(self, s: SetState, now: float) -> float:
        """The set's value AT `now` without writing it back. Caller holds the lock."""
        if not s.value_at or not s.value:
            return 0.0
        dt = now - s.value_at
        return s.value * math.exp(-dt / self.value_tau_s) if dt > 0.0 else s.value

    def note_use(self, tags: Iterable[str], saved_ms: float, *, measured: bool = True) -> None:
        """One statement read these sets — or would have, had they been
        resident — and saved `saved_ms` by doing so.

        This is the only bookkeeping on the statement path, and it is O(1) per
        set: one dict lookup, one exp() and a handful of stores under the
        manager's own lock. A use of a set that is NOT resident is an estimate
        of a saving nobody measured, so it counts for NATIVE_USE_WEIGHT of a
        resident use; that difference is what keeps a resident set in place
        against a candidate of the same worth.

        A set a statement read is credited together with everything it is
        built FROM (a materialised join's base sets, a subquery lane's
        sentinel). Value is deliberately not additive down a chain: each
        member gets the whole saving, because none of them could be dropped
        without losing it, and each is then divided by its own bytes — so the
        small source a join cannot do without is dense and safe, and a large
        one is judged on its size like anything else."""
        if not saved_ms or saved_ms <= 0.0:
            saved_ms = 0.0
        now = self._now()
        with self._lock:
            todo, seen = list(tags), set()
            while todo:
                t = todo.pop()
                if t in seen:
                    continue
                seen.add(t)
                s = self._sets.get(t)
                if s is None:
                    continue
                todo.extend(s.deps)
                self._decay_locked(s, now)
                w = saved_ms if s.state == "ready" else saved_ms * NATIVE_USE_WEIGHT
                s.value += w / self.value_tau_s
                s.uses += 1
                if measured:
                    s.measured_uses += 1
                s.last_use = now

    @staticmethod
    def _col_key(cols: dict, store: str, lane: str):
        """The store-column key a set's lane is held under, or None: a key lane
        is stored under 'k#<lane>' (it carries the sort cache), a plain one
        under its own name."""
        if (store, lane) in cols:
            return (store, lane)
        if (store, "k#" + lane) in cols:
            return (store, "k#" + lane)
        return None

    def _column_values(self, cols: dict, now: float) -> Dict[tuple, float]:
        """A set's value spread over the store columns it reads, in proportion
        to what each one costs; a column read by several sets carries the sum
        of their shares.

        Bytes are what eviction frees, and a column's bytes are freed only when
        NO set needs it any more — so the thing whose value must be known is
        the column, not the set. Sharing by bytes gives every set the same
        value density over each of its columns, which is exactly the quantity
        admission compares. Caller must not hold the lock."""
        out: Dict[tuple, float] = {}
        with self._lock:
            for _t, o in self._sets.items():
                if o.state != "ready" or not o.store_key:
                    continue
                mine = [c for c in (self._col_key(cols, o.store_key, l) for l in o.store_lanes) if c]
                total = sum(cols[c][0] for c in mine)
                if total <= 0:
                    continue
                v = self._value_locked(o, now)
                if v <= 0.0:
                    continue
                for c in mine:
                    out[c] = out.get(c, 0.0) + v * (cols[c][0] / total)
        return out

    @staticmethod
    def _per_gib(density: float) -> float:
        """A density (ms saved per second, per byte) in the units the log and
        the refusal sentence print: ms per second per GiB."""
        return density * float(2 ** 30)

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
        """Before an upload: does the set fit the budget, and if not, is it
        worth more than what it would take to make it fit? False (and s.error
        says why) when it is not — the statement then keeps running native,
        which is what rule 1 asks for when the device cannot hold the data.

        Three steps, in this order, and the order is the point. A SNAPSHOT of
        what is resident and what it costs (the extension is the source of
        truth: `gpu_residents().bytes` counts lanes, validity and derived
        structures; sets uploaded by hand count toward the total and are never
        evicted). Then a PLAN over that snapshot, which drops nothing: the
        whole set of units that would have to go, and the decision whether
        they should. Only then the drops. A plan that cannot reach the budget,
        or is not worth executing, refuses with everything still resident —
        the earlier shape evicted one unit at a time and could run out of
        acceptable victims after it had already destroyed several for a
        candidate it then declined anyway.

        Not evictable at all: a set an operator is using, a source of `s`, a
        source of another resident set (its dependents go first), and that
        source's store columns."""
        budget = self.memory_budget
        if not budget or s.est_bytes <= 0:
            return True
        if s.est_bytes > budget:
            with self._lock:
                s.error = (f"{MEMORY_ERROR}the set needs about {s.est_bytes / 2**20:.0f} MiB, "
                           f"the budget is {budget / 2**20:.0f} MiB")
            self._log(f"not uploaded: {s.tag}: {s.error}")
            return False
        with self._lock:
            before = self.evictions
        for _attempt in range(3):
            snap = self._snapshot(run, s)
            if snap is None:
                return True                      # cannot ask; upload without a check
            live, cols, used = snap
            if used + s.est_bytes <= budget:
                return True
            plan = self._plan(s, budget, used, live, cols)
            if plan is None:
                self._note_wasted(before)        # nothing was evicted; this records that
                return False
            if self._evict(run, s, plan, live, cols):
                return True
            self._log(f"memory budget: a drop failed for {s.tag} — re-planning "
                      f"against what is resident now")
        with self._lock:
            s.error = (f"{MEMORY_ERROR}the room could not be freed: three eviction plans "
                       f"in a row had a drop fail under them")
        self._log(f"not uploaded: {s.tag}: {s.error}")
        self._note_wasted(before)
        return False

    def _note_wasted(self, before: int) -> None:
        """This call is refusing. Anything it dropped on the way bought nothing."""
        with self._lock:
            self.evictions_wasted += self.evictions - before

    def _snapshot(self, run: Callable[[str], List[tuple]], s: SetState):
        """What the extension holds right now: (sets by name, columns by
        (store, lane), bytes in use). None when it cannot be asked."""
        try:
            rows = run("SELECT name, origin, bytes, refs, epoch_ms(last_used_at) AS used_ms "
                       "FROM gpu_residents()")
            # stage B: the store's columns are what the views cost (a view reports 0)
            crows = run("SELECT store, \"column\", bytes, epoch_ms(last_used_at) AS used_ms FROM gpu_store_columns()")
        except Exception as e:
            self._log(f"memory budget: gpu_residents() failed ({str(e)[:80]}); uploading without a check")
            return None
        live = {r[0]: r for r in rows}
        cols = {(r[0], r[1]): (int(r[2] or 0), r[3]) for r in crows}
        used = (sum(int(r[2] or 0) for r in rows if r[0] != s.tag)
                + sum(b for b, _u in cols.values()))
        return live, cols, used

    def _plan(self, s: SetState, budget: int, used: int, live: dict, cols: dict):
        """The whole eviction plan for `s`, or None with `s.error` saying why
        there is none. Drops nothing.

        Two different quantities do two different jobs, and conflating them was
        the earlier mistake:

          * **value per byte decides WHO goes** — the units are taken cheapest
            first, because a byte freed from a low-density unit costs the least
            value; between two of equal density the one used longest ago, and a
            unit the minimum age still protects comes last whatever its value.
          * **total value decides WHETHER** — the plan runs only if the
            candidate is worth more than everything in it put together, by
            EVICT_HYSTERESIS. A candidate that needs three victims must be
            worth more than the three of them, not merely denser than each.
            That is the exchange that makes the resident population better
            rather than only differently arranged.

        A plan containing a unit the minimum age protects has one more
        condition, because such a unit has never been read and its value says
        nothing: the candidate must be worth YOUNG_OVERRIDE_RATIO times the
        typical resident density."""
        now = self._now()
        keep = self._protected(s)
        col_value = self._column_values(cols, now)
        with self._lock:
            is_source = {d for t, o in self._sets.items() if t in live and t not in keep for d in o.deps}
            # age by this manager's own clock (a set it did not upload is old by definition)
            young = {t for t, o in self._sets.items()
                     if o.last_upload_start and now - o.last_upload_start < self.evict_min_age_s}
            young_cols = {c for c, at in self._col_uploaded.items()
                          if now - at < self.evict_min_age_s}
            set_value = {t: self._value_locked(o, now) for t, o in self._sets.items()}
            # a set that must stay takes its store columns with it: this upload's
            # own sources, and the sources of every resident derived set (their
            # dependents go first — dropping a lane under one would un-ready the
            # join built from it)
            keep_cols = {(o.store_key, l) for t, o in self._sets.items()
                         if t in keep or t in is_source for l in o.store_lanes}
            cand_value = self._value_locked(s, now)
            # a set an override took off the device may not take one back
            # immediately: the margin already rules out a straight swap, this
            # rules out a swap through a third set whose value moved meanwhile
            cooling = bool(s.override_evicted_at) and now - s.override_evicted_at < self.evict_min_age_s
        # every evictable unit: (value per byte, least-recently-used key, kind, key, bytes, protected, value)
        units: List[tuple] = []
        for r in live.values():
            if (r[1] != "managed" or r[0] in keep or r[0] in is_source
                    or int(r[3] or 0) != 0 or int(r[2] or 0) <= 0):
                continue
            b, v = int(r[2]), set_value.get(r[0], 0.0)
            # protected while young AND worth nothing yet: it has not had the
            # chance to pay back its upload (see YOUNG_OVERRIDE_RATIO)
            units.append((v / b, (r[4] is not None, r[4] or 0),
                          "set", r[0], b, r[0] in young and v <= 0.0, v))
        for c, (b, u) in cols.items():
            if c in keep_cols or b <= 0:
                continue
            v = col_value.get(c, 0.0)
            units.append((v / b, (u is not None, u or 0),
                          "col", c, b, c in young_cols and v <= 0.0, v))
        # the typical worth of what is resident, over everything measured
        known = sorted(d for d in
                       ([set_value.get(r[0], 0.0) / int(r[2]) for r in live.values()
                         if r[1] == "managed" and int(r[2] or 0) > 0]
                        + [col_value.get(c, 0.0) / b for c, (b, _u) in cols.items() if b > 0])
                       if d > 0.0)
        median = known[len(known) // 2] if known else 0.0
        # a candidate nobody has measured is priced at that median: absent
        # evidence, a set is worth what this connection's sets are typically
        # worth, which lets it displace the cheapest thing there is and nothing
        # better. One statement later the measurement replaces the guess.
        priced = False
        if cand_value <= 0.0 and median > 0.0:
            cand_value, priced = median * s.est_bytes, True
        cand_density = cand_value / s.est_bytes
        allow_young = cand_density > 0.0 and not cooling
        pool = sorted(units, key=lambda u: (u[5], u[0], u[1]))
        if not allow_young:
            pool = [u for u in pool if not u[5]]
        need = used + s.est_bytes - budget
        chosen: List[tuple] = []
        freed, lost, takes_young = 0, 0.0, False
        for u in pool:
            if freed >= need:
                break
            chosen.append(u)
            freed += u[4]
            lost += u[6]
            takes_young = takes_young or u[5]
        ctx = (s, used, budget, live, cols, keep, is_source, young, young_cols,
               allow_young, cand_density, cand_value, priced)
        if freed < need:
            self._refuse(ctx, None)
            return None
        if takes_young and cand_density < median * YOUNG_OVERRIDE_RATIO:
            self._refuse(ctx, ("young", chosen, lost, median * YOUNG_OVERRIDE_RATIO))
            return None
        if lost > 0.0 and cand_value < lost * (1.0 + EVICT_HYSTERESIS):
            self._refuse(ctx, ("value", chosen, lost, lost * (1.0 + EVICT_HYSTERESIS)))
            return None
        return chosen, cand_density, lost, priced

    def _evict(self, run: Callable[[str], List[tuple]], s: SetState, plan, live: dict,
               cols: dict) -> bool:
        """Execute a plan the budget has already accepted. False when a drop
        failed — the caller re-plans against what is resident then, rather than
        carrying on against a picture that is no longer true."""
        chosen, cand_density, lost, priced = plan
        now = self._now()
        for density, _lru, kind, key, nbytes, is_young, value in chosen:
            worth = (f"worth {self._per_gib(density):.2f} ms/s per GiB, this set "
                     f"{self._per_gib(cand_density):.2f}"
                     + (" as an unmeasured set is priced" if priced else "")
                     if (density or cand_density)
                     else "nothing measured either way, least recently used")
            if len(chosen) > 1:
                worth += (f"; {len(chosen)} units worth {self._per_gib(lost / max(1, s.est_bytes)):.2f} "
                          f"together over this set's bytes")
            if is_young:
                worth += f"; younger than {self.evict_min_age_s:.0f} s and not used yet, overridden"
            if kind == "col":
                try:
                    run("SELECT gpu_drop_column('%s', '%s')" % (key[0].replace("'", "''"),
                                                                key[1].replace("'", "''")))
                except Exception as e:
                    self._log(f"memory budget: could not evict column {key}: {str(e)[:80]}")
                    return False
                cols.pop(key, None)
                with self._lock:
                    self.evictions += 1
                    self._col_uploaded.pop(key, None)
                    gone = set()
                    for t, o in self._sets.items():
                        if o.store_key == key[0] and key[1] in o.store_lanes and o.state == "ready":
                            o.state = "missing"      # its next sighting uploads the missing lane
                            o.bytes = 0
                            if is_young:
                                o.override_evicted_at = now
                            gone.add(t)
                    self._demote_dependents_locked(gone)
                self._log(f"evicted ({worth}): column {key[1]} of {key[0]} "
                          f"({nbytes / 2**20:.0f} MiB) for {s.tag}")
                continue
            try:
                run("SELECT gpu_drop_resident('%s')" % key.replace("'", "''"))
            except Exception as e:
                self._log(f"memory budget: could not evict {key}: {str(e)[:80]}")
                return False
            live.pop(key, None)
            with self._lock:
                self.evictions += 1
                o = self._sets.get(key)
                if o is not None and o.state == "ready":
                    o.state = "missing"              # a later sighting uploads it again
                    o.bytes = 0
                    if is_young:
                        o.override_evicted_at = now
                    self._demote_dependents_locked({key})
            self._log(f"evicted ({worth}): {key} ({nbytes / 2**20:.0f} MiB) for {s.tag}")
        return True

    def _refuse(self, ctx: tuple, beat: Optional[tuple]) -> None:
        """Why this set stays off the device, with the numbers: what it needed,
        what is resident, and either what the plan would have cost against what
        the set is worth, or what stood in the way of a plan at all. One
        sentence — it is what `last_rewrite()['detail']` shows a person.
        Nothing has been evicted when this is called."""
        (s, used, budget, live, cols, keep, is_source, young, young_cols,
         allow_young, cand_density, cand_value, priced) = ctx
        managed = [r for r in live.values() if r[1] == "managed" and int(r[2] or 0) > 0]
        head = (f"{MEMORY_ERROR}{used / 2**20:.0f} MiB resident + about "
                f"{s.est_bytes / 2**20:.0f} MiB needed > {budget / 2**20:.0f} MiB")
        mine = (f"is worth {cand_value:.3f} ms/s ({self._per_gib(cand_density):.2f} per GiB)"
                if not priced else
                f"has nothing measured yet and is priced at the median, "
                f"{cand_value:.3f} ms/s ({self._per_gib(cand_density):.2f} per GiB)")
        if beat is not None:
            kind, chosen, lost, floor = beat
            first = chosen[0]
            name = (first[3][1] + " of " + first[3][0]) if first[2] == "col" else first[3]
            what = (f"{len(chosen)} units, the cheapest of them {name}" if len(chosen) > 1
                    else name)
            if kind == "young":
                why = (f"{head}, and this set {mine}, under the "
                       f"{self._per_gib(floor):.2f} per GiB it would take to interrupt {what} — "
                       f"younger than {self.evict_min_age_s:.0f} s and not used yet, so it would "
                       f"take {YOUNG_OVERRIDE_RATIO:.0f}x the typical set. Nothing was evicted")
            else:
                why = (f"{head}, and this set {mine} against {lost:.3f} ms/s for {what}, "
                       f"which is what making room would cost (a swap takes "
                       f"{1.0 + EVICT_HYSTERESIS:.2f}x). Nothing was evicted")
        else:
            n_cols = sum(1 for c, (b, _u) in cols.items() if b > 0)
            why = (f"{head}, and no plan reaches the budget ({len(managed)} managed sets, "
                   f"{n_cols} resident columns: "
                   f"{sum(r[0] in keep for r in managed)} sets needed by this upload, "
                   f"{sum(r[0] in is_source for r in managed)} sources of resident sets, "
                   f"{sum(int(r[3] or 0) > 0 for r in managed)} in use, "
                   f"{sum(r[0] in young for r in managed) + sum(1 for c in cols if c in young_cols)} "
                   f"younger than {self.evict_min_age_s:.0f} s"
                   + ("" if allow_young else " and this set may not override that")
                   + "). Nothing was evicted")
        with self._lock:
            s.error = why
        self._log(f"not uploaded: {s.tag}: {why}")

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
            now = self._now()
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
        """Diagnostics: the budget, and per set the estimate, the extension's
        figure, and what the set is worth — `value` in ms saved per second of
        wall time, `density` the same per GiB it holds, which is the quantity
        admission compares."""
        with self._lock:
            now = self._now()
            out = {}
            for t, s in self._sets.items():
                v = self._value_locked(s, now)
                size = s.bytes or s.est_bytes
                out[t] = {"state": s.state, "est_bytes": s.est_bytes, "bytes": s.bytes,
                          "error": s.error, "value": v, "uses": s.uses,
                          "density": self._per_gib(v / size) if size else 0.0}
            return {"budget": self.memory_budget, "evictions": self.evictions,
                    "evictions_wasted": self.evictions_wasted, "sets": out}

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
            s.last_upload_start = self._now()
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
            # Wake when the test could FIRST pass — idle_ms after the last
            # statement, or `not_before`, whichever is later — and not a whole
            # idle_ms from here, which is what this waited before 2026-09-20.
            # The window a segment then has is the whole of the gap the
            # workload leaves minus idle_ms, rather than that minus up to
            # 2 x idle_ms: on the x86 box segments were starting at 5.1-9.9 ms
            # of idle against a threshold of 5 (median 5.26), and the gap they
            # were competing for is itself only a few ms.
            target = max(not_before, self._last_activity + idle_ms / 1000.0)
            self._cv.wait(timeout=min(max(target - now, 0.0005), 0.05))

    def _trace_line(self, msg: str) -> None:
        if self._trace:
            print(f"[gpudb/residency] {msg}", file=sys.stderr, flush=True)

    # ---- contention, and the steps an interrupt cannot stop (§5.5) ----
    def note_cores(self, cores: float) -> None:
        """One completed segment scan used `cores` cores (CPU seconds per wall
        second). Two clock reads per segment, which is the whole cost of
        knowing whether this machine has cores to spare."""
        with self._lock:
            self._seg_cores.append(cores)
            del self._seg_cores[:-CONTENTION_SEGMENTS]

    def _cores_locked(self) -> float:
        """The estimate, for a caller that already holds the lock (`self._cv`
        shares it and it is not re-entrant, so a locked caller must never go
        through `cores_seen`)."""
        xs = self._seg_cores
        return statistics.median(xs) if len(xs) >= CONTENTION_MIN_SAMPLES else 0.0

    def cores_seen(self) -> float:
        """The median of those, or 0.0 while too few have been measured."""
        with self._lock:
            return self._cores_locked()

    def contended(self) -> bool:
        """Is something else on this machine taking the cores? Measured, not
        configured: a segment scan that gets less than half the machine while
        nothing of ours is running beside it means the cores are spoken for.
        Unknown (too few segments) reads as free — that is today's behaviour,
        and the conservative direction is to protect nothing we cannot show
        needs protecting."""
        if self.quiet_ms <= 0.0:
            return False
        seen = self.cores_seen()
        return 0.0 < seen < CONTENDED_CORES_FRACTION * self._cpus

    def note_segment(self, landed: bool, s: Optional[SetState] = None, *,
                     rows: int = 0, ms: float = 0.0, window_ms: float = 0.0) -> int:
        """One segment attempt ended. Returns the divisor the next segment
        should use: the size follows the YIELD, not the clock, because what
        decides whether a segment lands is its cost against the window the
        workload leaves, and neither of those is knowable in advance — only
        the outcome is. A halving is given back as readily as it is taken, so
        a session that is only briefly crowded does not stay small.

        What the attempt MEASURED comes in with it: `ms` is what a landed
        segment of `rows` rows cost (the pair of sizes behind the fit of the
        fixed and per-row parts), `window_ms` is how long an interrupted one
        ran before the statement arrived, which is the window the workload is
        leaving. Both are optional — with neither, the floor is the constant
        backstop and the rule is exactly what it was.

        Both bounds are hard: a halving that is not worth taking is refused
        (`segment_floor_rows`, and the set is then reported STARVED rather
        than ground finer), and a landed segment is progress that is never
        undone, so a session terminates at the floor however busy the
        connection is."""
        if self.segment_rows:
            return 1                     # the size is pinned; there is nothing to decide
        with self._lock:
            if landed and s is not None and rows > 0 and ms > 0.0:
                xs = s.seg_cost.setdefault(int(rows), [])
                xs.append(float(ms))
                del xs[:-SEG_COST_SAMPLES]
            if not landed and window_ms > 0.0:
                self._seg_windows.append(float(window_ms))
                del self._seg_windows[:-SEG_WINDOW_SAMPLES]
            self._seg_landed.append(bool(landed))
            if len(self._seg_landed) < SEG_YIELD_WINDOW:
                return self._seg_scale
            yld = sum(self._seg_landed) / float(len(self._seg_landed))
            if yld <= SEG_YIELD_LOW:
                base = s.segment_rows_default if s is not None else 0
                floor = self._segment_floor_locked(s)
                nxt = max(1, base // (self._seg_scale * 2)) if base else 0
                if base and nxt < floor:
                    # smaller would not be meaningfully cheaper, so it would not
                    # land either: this connection leaves no window, and saying
                    # so is more use than grinding the segment finer
                    self._seg_starved = True
                    del self._seg_landed[:-SEG_YIELD_WINDOW]
                elif not base and self._seg_scale >= SEG_SCALE_MAX:
                    self._seg_starved = True
                    del self._seg_landed[:-SEG_YIELD_WINDOW]
                else:
                    self._seg_scale *= 2
                    self._seg_landed = []
                    self._seg_starved = False
            elif yld >= SEG_YIELD_HIGH and self._seg_scale > 1:
                self._seg_scale //= 2
                self._seg_landed = []
                self._seg_starved = False
            else:
                del self._seg_landed[:-SEG_YIELD_WINDOW]
                if yld > SEG_YIELD_LOW:
                    self._seg_starved = False
            return self._seg_scale

    @staticmethod
    def segment_costs(s: SetState) -> List[Tuple[int, float]]:
        """What a segment of each size COSTS this set here, ascending by size:
        the FASTEST of the segments of that size that landed.

        The fastest and not the median because the quantity wanted is the work
        a segment is, not what the machine was doing beside it: a segment that
        lands while a user statement is being answered on the other cores is
        descheduled part of the time and reads slower than it is (measured
        under the wrapper suite's starved cadence: 16384 rows timed at 2.1 ms
        where the same segment unraced costs 0.55). Interrupted attempts are
        not in it at all — they were cut short and say nothing about cost."""
        pts, top = [], 0.0
        for r, xs in sorted(s.seg_cost.items()):
            if not xs:
                continue
            # ... and never less than a smaller segment: a bigger one reads
            # everything it reads, so an inversion is noise, not a measurement
            top = max(top, min(xs))
            pts.append((r, top))
        return pts

    @classmethod
    def segment_cost_model(cls, s: SetState) -> Optional[Tuple[float, float]]:
        """(fixed ms, ms per row) near the SMALL end of the measured sizes, or
        None while fewer than two sizes have landed.

        Near the small end, and not over the whole range, because the curve is
        not a straight line: a segment large enough to span several row groups
        is scanned in parallel and costs far less per row than a small one
        (measured on an M4 Max: 0.0051 us/row at 524288 rows against 0.0488 at
        8192). The two parts of the cost are only meaningful locally, and the
        place the floor cares about is the bottom."""
        pts = cls.segment_costs(s)
        if len(pts) < 2:
            return None
        (r0, t0), (r1, t1) = pts[0], pts[1]
        slope = (t1 - t0) / float(r1 - r0)
        if slope <= 0.0:
            return None
        return max(0.0, t0 - r0 * slope), slope

    @staticmethod
    def _cost_at(pts: List[Tuple[int, float]], r: int) -> float:
        """What a segment of `r` rows costs, from the sizes that landed: the
        measurement where there is one, else the nearest ones carried along
        their local slope. A size below everything measured is never priced
        under SEG_BLIND_GAIN per halving of the smallest measurement — the
        fixed part is what a smaller segment cannot escape, and pricing it away
        is the one error that would let the size run off to nothing."""
        d = dict(pts)
        if r in d:
            return d[r]
        if r < pts[0][0]:
            r0, t0 = pts[0]
            steps = max(1, int(math.ceil(math.log(r0 / float(max(1, r)), 2))))
            blind = t0 * (SEG_BLIND_GAIN ** steps)
            if len(pts) < 2:
                return blind
            r1, t1 = pts[1]
            slope = max(0.0, (t1 - t0) / float(r1 - r0))
            return max(blind, t0 - (r0 - r) * slope)
        if len(pts) == 1:
            r0, t0 = pts[0]
            return t0 * (r / float(r0))
        if r > pts[-1][0]:
            (r0, t0), (r1, t1) = pts[-2], pts[-1]
            slope = max(0.0, (t1 - t0) / float(r1 - r0))
            return t1 + (r - r1) * slope
        lo = max(p for p in pts if p[0] <= r)
        hi = min(p for p in pts if p[0] >= r)
        if hi[0] == lo[0]:
            return lo[1]
        return lo[1] + (r - lo[0]) / float(hi[0] - lo[0]) * (hi[1] - lo[1])

    def _segment_floor_locked(self, s: Optional[SetState]) -> int:
        """The smallest segment still worth taking for this set, in rows.
        Caller holds the lock.

        What a halving costs is TOTAL work: the fixed part of a segment — the
        statement's own bind and DuckDB's scan set-up over the table's row
        groups — is paid once per segment however few rows it asks for, so
        halving the size adds a whole fixed part per pair of segments. The
        floor is therefore the smallest size whose predicted total for this
        table stays within SEG_TOTAL_MAX_RATIO of the total at the default
        size, priced from the segments that actually landed here. Measured on
        an M4 Max over a 2M-row table (2026-09-20): 2.6 ms per 524288-row
        segment and 0.4 ms per 8192-row one, i.e. 10 ms for the table at the
        default and 98 ms in 8192-row pieces — ten times the work for a
        segment that is only 6x cheaper to attempt. Where the cost really is
        `fixed + rows x per_row` this is exactly the size at which the two
        parts are equal, which is the same statement in other words.

        Until a segment has landed it is the constant backstop, 1/SEG_SCALE_MAX
        of the default — what this was before any of it was measured. A table
        is never cut into more than SEG_MAX_SEGMENTS pieces either way."""
        if s is None:
            return 0
        base = s.segment_rows_default
        rows = float(s.table_rows or base)
        hard = max(1, -(-int(rows) // SEG_MAX_SEGMENTS))
        pts = self.segment_costs(s)
        at_base = dict(pts).get(base, 0.0)
        if at_base <= 0.0:
            # nothing landed at the default size, so there is no baseline to
            # judge a smaller one against: the constant backstop, as before
            return min(base, max(hard, max(1, base // SEG_SCALE_MAX)))
        budget = SEG_TOTAL_MAX_RATIO * (rows / base) * at_base
        floor = r = base
        while r // 2 >= hard:
            r //= 2
            if (rows / r) * self._cost_at(pts, r) > budget:
                break
            floor = r
        return min(base, max(1, floor))

    def segment_floor_rows(self, s: SetState) -> int:
        with self._lock:
            return self._segment_floor_locked(s)

    def window_seen_ms(self) -> float:
        """The median window this connection has been leaving between its
        statements, from the interrupted attempts; 0.0 when none was seen."""
        with self._lock:
            return statistics.median(self._seg_windows) if self._seg_windows else 0.0

    def starved(self) -> bool:
        """The yield asked for a smaller segment and the floor refused: no
        segment worth taking fits this connection's gaps. Correct and never
        slower — the set is simply not becoming resident (KNOWN_ISSUES.md)."""
        with self._lock:
            return self._seg_starved

    def segment_rows_for(self, s: SetState) -> int:
        """Rows in the next segment of this set: its own 8 MiB worth, divided
        by what the yield has settled on, never below the measured floor. An
        explicit `segment_rows` (a test, a sweep) is taken as given and never
        adapted."""
        if self.segment_rows:
            return int(self.segment_rows)
        with self._lock:
            rows = max(1, s.segment_rows_default // self._seg_scale)
            return max(rows, self._segment_floor_locked(s))

    def quiet_want_ms(self, need_ms: float, waited_s: float) -> float:
        """The idle stretch a step an interrupt cannot stop still insists on,
        `waited_s` after it first asked for `need_ms`. Decays linearly to the
        ordinary idle threshold at `quiet_max_s` — so the step takes the best
        gap the connection offers along the way, and at the deadline it asks
        for no more than any segment does, which is what bounds the wait."""
        if need_ms <= self.idle_ms or self.quiet_max_s <= 0.0:
            return self.idle_ms
        frac = min(1.0, max(0.0, waited_s / self.quiet_max_s))
        return need_ms + (self.idle_ms - need_ms) * frac

    def _post_step(self, cur, s: SetState, epoch: int, stmt: str) -> str:
        """One post-upload step — the view's sort cache — retried while it is
        interrupted. Returns "" when the session may go on, or the outcome it
        must return instead.

        `ready` is defined as uploaded AND PREPARED (§5.5). An interrupt used
        to end the loop silently, which broke that in two visible ways: the
        first statement to use the set paid the sort, and the set had no row
        in `gpu_residents()` at all, because a store-backed set is a VIEW the
        extension only synthesises when something acquires it — and the sort
        cache is what acquires it here. Measured under the wrapper suite's
        `big:` cadence (~130 statements/s, 8 of 16 cores busy), that happened
        in 3 of 40 runs. Retrying is safe: prepare() is idempotent."""
        pause_until, since = 0.0, time.monotonic()
        for attempt in range(POST_MAX_ATTEMPTS):
            # `since` is shared by every attempt: the quiet window decays once,
            # so the whole step is bounded by quiet_max_s plus the interrupt
            # back-off, not by quiet_max_s per retry
            if not self._wait_quiet(s, epoch, not_before=pause_until, since=since):
                return "closed" if self._closed else "stale"
            try:
                self._run(cur, s, stmt)
                return ""
            except Exception as e:
                err = str(e)
                if _not_resident(err):
                    # the acquire the sort cache does is the second reader of the
                    # store, and it has just said the lanes are not there after all
                    with self._lock:
                        s.error = self._store_gone_error(s)
                    return "recheck"
                if not _is_interrupt(err):
                    return self._fail(s, e)
                pause_until = time.monotonic() + min(
                    RETRY_PAUSE_MS * (2.0 ** attempt), RETRY_PAUSE_MAX_MS) / 1000.0
        # the columns are resident either way; only the cache is missing, and the
        # first statement to use the set builds it. Said out loud rather than
        # left as a set that claims to be prepared and is not.
        self._log(f"sort cache not built for {s.tag}: {POST_MAX_ATTEMPTS} attempts in a row were "
                  f"interrupted; the first statement to use the set builds it")
        return ""

    def _wait_quiet(self, s: SetState, epoch: int, not_before: float = 0.0,
                    since: float = 0.0) -> bool:
        """`_wait_idle`, but for a step an interrupt cannot stop. On a machine
        with cores to spare this IS `_wait_idle` (measured: such a step costs a
        user statement nothing there). On a contended one it holds the step
        back until the connection goes quiet for `quiet_want_ms`, which decays
        to the idle threshold over `quiet_max_s` — the bound on how long
        readiness can be delayed by one step."""
        need = self.quiet_ms if self.contended() else 0.0
        if need <= self.idle_ms:
            with self._cv:
                return self._wait_idle(s, epoch, self.idle_ms, not_before)
        t_enter = time.monotonic()
        t0 = since or t_enter          # the decay's clock; the accounting's is t_enter
        with self._cv:
            while True:
                if self._closed or s.epoch != epoch or s.state != "uploading":
                    return False
                now = time.monotonic()
                waited = now - t0
                want = self.quiet_want_ms(need, waited)
                if now >= not_before and self._in_flight == 0 and self._idle_ms() >= want:
                    s.quiet_waits += 1
                    s.quiet_wait_ms += (now - t_enter) * 1000.0
                    if waited >= self.quiet_max_s:
                        s.quiet_forced += 1
                    if self._trace:
                        self._trace_line(
                            f"{s.tag}: a step an interrupt cannot stop asked for {need:.0f} ms of quiet "
                            f"on {self._cores_locked():.1f} of {self._cpus:.0f} cores and waited "
                            f"{waited * 1000.0:.0f} ms for {want:.0f} ms"
                            f"{' (took it at the deadline)' if waited >= self.quiet_max_s else ''}")
                    return True
                self._cv.wait(timeout=max(0.002, min(0.01, not_before - now) if now < not_before else 0.01))

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

    def _abort(self, cur, s: SetState, plan: str = "") -> None:
        try:
            cur.execute("SELECT gpu_upload_abort(?)", [s.session_name]).fetchall()
        except Exception:
            pass
        self._deallocate(cur, plan)

    def _prepare_segment(self, cur, s: SetState) -> str:
        """Hand the segment statement to the database ONCE for the session and
        keep only its row range per segment, or "" when that cannot be done and
        the segment statement is rendered in full each time (what this did
        before 2026-09-20).

        A segment's cost has a fixed part that no smaller segment escapes, and
        part of it is the statement's own parse and bind — measured on an M4
        Max over a 13-lane upload, 121 us of a 935 us segment, and 35-50 us of
        a 2-lane one. A PREPARE is where DuckDB keeps a plan, so the session
        makes one and EXECUTEs it; the row range stays a literal, which is what
        lets the scan go on pruning row groups by it (the wrapper's parameter
        path is measurably SLOWER than the literal one — 497 us against 368 —
        so it is not used here).

        Everything else about the statement is unchanged, so the rows it reads
        and the order it reads them in are unchanged: the accounting checks
        (rows_seen == rows, every row exactly once) are what prove that."""
        if "$" in s.upload_sql:
            return ""                     # a $ in the text would be read as a parameter
        with self._lock:
            self._plan_seq += 1
            name = f"gpudb_seg_{self._plan_seq}"
        try:
            self._run(cur, s, f"PREPARE {name} AS {s.upload_sql} "
                              f"WHERE rowid >= $1 AND rowid < $2")
            return name
        except Exception as e:
            # an interrupt, an older engine, a shape PREPARE will not take: the
            # session simply renders each segment, which costs the parse back
            self._trace_line(f"{s.tag}: the segment statement was not prepared "
                             f"({str(e)[:80]}); rendering each segment instead")
            return ""

    def _deallocate(self, cur, plan: str) -> None:
        if not plan:
            return
        try:
            cur.execute(f"DEALLOCATE {plan}").fetchall()
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
            for stmt in s.post_sql:                  # the sort cache: an interrupt cannot stop it
                bad = self._post_step(cur, s, epoch, stmt)
                if bad:
                    return bad
            return "ready"
        fqn = s.fqn or s.upload_sql.split(" FROM ", 1)[1]
        seg_rows = self.segment_rows_for(s)
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
            s.table_rows = int(max_rowid) + 1
            s.quiet_waits = s.quiet_forced = 0
            s.quiet_wait_ms = 0.0
        # 2. begin, and hand the segment statement's PLAN to the database once
        #    for the whole session rather than per segment
        try:
            self._run(cur, s, "SELECT gpu_upload_begin(?)", [s.session_name])
        except Exception as e:
            return "pending" if _is_interrupt(str(e)) else self._fail(s, e)
        plan = self._prepare_segment(cur, s)
        # 3. segments, each only in an idle window
        a, done, consecutive_interrupts, not_before = 0, 0, 0, 0.0
        while a <= max_rowid:
            t_want = time.monotonic()
            with self._cv:
                ok = self._wait_idle(s, epoch, idle_ms, not_before)
            if not ok:
                self._abort(cur, s, plan)
                return "closed" if self._closed else "stale"
            b = a + seg_rows
            seg_sql = (f"EXECUTE {plan}({a}, {b})" if plan else
                       f"{s.upload_sql} WHERE rowid >= {a} AND rowid < {b}")
            cores, scan_ms = 0.0, 0.0
            try:
                t_seg, c_seg = time.monotonic(), time.process_time()
                self._run(cur, s, seg_sql)
                landed = True
                wall = time.monotonic() - t_seg
                scan_ms = wall * 1000.0
                with self._lock:
                    s.seg_ms.append(scan_ms)
                # nothing of ours ran beside this one (a statement that arrives
                # interrupts, and an interrupted segment does not land here), so
                # the CPU it burned per wall second is the machine's answer to
                # "how many cores can you give me right now"
                if wall > 0.0:
                    cores = (time.process_time() - c_seg) / wall
                    if (seg_rows >= CONTENTION_MIN_SEG_ROWS
                            and wall * 1000.0 >= CONTENTION_MIN_SEG_MS):
                        self.note_cores(cores)
            except Exception as e:
                if not _is_interrupt(str(e)):
                    self._abort(cur, s, plan)
                    return self._fail(s, e)
                # how long the segment got before the statement arrived: the
                # window this workload is leaving, measured rather than assumed
                window = (time.monotonic() - t_seg) * 1000.0
                # interrupted: did the segment land before the interrupt was seen?
                st = self._session_status(cur, s)
                if not st.get("open") or st.get("invalidated"):
                    self._abort(cur, s, plan)
                    return "stale"
                landed = int(st.get("segments", 0)) == done + 1
                consecutive_interrupts += 1
                pause = min(RETRY_PAUSE_MS * (2.0 ** (consecutive_interrupts - 1)),
                            RETRY_PAUSE_MAX_MS)
                not_before = time.monotonic() + pause / 1000.0
                with self._lock:
                    s.interrupts += 1
            self.note_segment(landed, s, rows=seg_rows, ms=scan_ms,
                              window_ms=(0.0 if landed else window))
            if self._trace:
                starving = ("" if not self.starved() else
                            f"; starved: no segment above the floor of "
                            f"{self.segment_floor_rows(s)} rows fits the "
                            f"{self.window_seen_ms():.1f} ms this connection leaves")
                self._trace_line(
                    f"{s.tag} segment {done}/{planned} rows [{a},{b}) ({seg_rows} rows): waited "
                    f"{(t_seg - t_want) * 1000.0:.1f} ms for an idle window, scan "
                    f"{(time.monotonic() - t_seg) * 1000.0:.1f} ms on {cores:.1f} of "
                    f"{self._cpus:.0f} cores, "
                    f"{'landed' if landed else 'interrupted'}"
                    f"{'' if landed else f' (consecutive {consecutive_interrupts}, pausing {pause:.0f} ms)'}"
                    f"{starving}")
            if landed:
                a, done = b, done + 1
                consecutive_interrupts, not_before = 0, 0.0
                with self._lock:
                    s.segments = done
            # the size the yield has settled on, for the segments still to come,
            # and what the plan now says (the count changes with the size, and
            # an exhausted range plans nothing — `done` is the whole of it)
            seg_rows = self.segment_rows_for(s)
            left = max(0, int(max_rowid) - a + 1)
            with self._lock:
                s.segment_rows_used = seg_rows
                s.segments_planned = done + (left + seg_rows - 1) // seg_rows
                planned = s.segments_planned
        self._deallocate(cur, plan)      # every segment is in; the plan has no more use
        # 4. finish: device copy + prepare + publish (one scalar call; an
        #    interrupt arriving during it is only seen after it returns, so on
        #    a machine whose cores are taken it waits for a quiet connection)
        ok = self._wait_quiet(s, epoch)
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
        with self._lock:
            # the window the finish CALL occupied. Closed here and not after the
            # sort cache: the cache now waits for a quiet connection of its own,
            # and a window that swallowed that wait would tell the gate that a
            # 20 s call had statements inside it.
            s.finish_window = (t_fin, time.monotonic())
        for stmt in s.post_sql:                  # the view's sort cache, once the columns are there
            # the sort cache is the session's OTHER step an interrupt cannot
            # stop, and the longer of the two (171 ms at SF10)
            bad = self._post_step(cur, s, epoch, stmt)
            if bad:
                return bad
        with self._lock:
            s.rows_seen = rows
            s.session_ms = (time.monotonic() - t_start) * 1000.0
        return "ready"

    def _session_derived(self, cur, s: SetState, epoch: int, t_start: float) -> str:
        """Materialise a derived set: each step is one scalar call on the
        device (tens of ms), issued only in an idle window like a finish."""
        rows = 0
        for stmt in s.steps:
            # each step is device work an interrupt cannot stop, so it waits
            # for a quiet connection on a machine whose cores are taken
            if not self._wait_quiet(s, epoch):
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
                s.last_upload_start = self._now()
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
                now = self._now()
                if outcome == "ready" and s.epoch == epoch and s.state == "uploading":
                    s.state = "ready"
                    self._log(f"resident: {s.tag} ({s.segments} segments, "
                              f"{s.interrupts} interrupts, {s.session_ms:.0f} ms"
                              + (f", {s.quiet_wait_ms / 1000.0:.1f} s waiting for a quiet "
                                 f"connection on {self._cores_locked():.1f} of {self._cpus:.0f} cores"
                                 if s.quiet_wait_ms >= 1.0 else "") + ")")
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
