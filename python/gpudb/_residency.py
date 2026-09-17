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
capped at 5 s) so a cadence whose statements keep landing on segments pays
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
RETRY_PAUSE_MAX_MS = 5000.0


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

    @property
    def pair(self) -> bool:
        return "gpu_upload_pair(" in self.upload_sql

    @property
    def segment_rows_default(self) -> int:
        return SEGMENT_BYTES // (16 if self.pair else 8)


def _is_interrupt(err: str) -> bool:
    return "INTERRUPT" in err.upper()


class ResidencyManager:
    def __init__(self, cursor_factory: Callable[[], object], *, mode: str = "background",
                 idle_ms: float = 20.0, quiet_s: float = 2.0, rate_s: float = 30.0,
                 max_attempts: int = 20, segment_rows: Optional[int] = None,
                 log: Optional[Callable[[str], None]] = None):
        self._cursor_factory = cursor_factory
        self.mode = mode
        self.idle_ms = idle_ms
        self.quiet_s = quiet_s
        self.rate_s = rate_s
        self.max_attempts = max_attempts
        self.segment_rows = segment_rows        # None: 8 MiB worth of rows for the set's kind
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

    def note_candidate(self, tag: str, upload_sql: str, fqn: str = "") -> SetState:
        """A rewritable shape over a non-resident set was seen."""
        with self._cv:
            s = self._sets.get(tag)
            if s is None:
                s = SetState(tag=tag, upload_sql=upload_sql, fqn=fqn)
                self._sets[tag] = s
            elif fqn and not s.fqn:
                s.fqn = fqn
            if s.state in ("missing", "stale", "failed") and s.attempts < self.max_attempts:
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

    def invalidate(self, tag: Optional[str] = None) -> None:
        """Local bookkeeping; the caller runs gpu_invalidate on the database
        (which also drops any open upload session under the name)."""
        with self._cv:
            now = time.monotonic()
            for t, s in self._sets.items():
                if tag is None or t == tag:
                    s.epoch += 1
                    s.last_invalidate = now
                    s.resume_at = now + self.quiet_s
                    if s.state in ("ready", "uploading", "pending"):
                        s.state = "stale"
            cur, utag = self._upload_cursor, self._uploading_tag
            self._cv.notify_all()
        if cur is not None and utag is not None and (tag is None or tag == utag):
            self._interrupt(cur)

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

    # ---- synchronous upload (residency='eager', and tests) ----
    def upload_now(self, tag: str, run: Callable[[str], None]) -> bool:
        """One statement over the whole table on the caller's connection: the
        caller asked for it (eager), so it runs beside nothing."""
        s = self.get(tag)
        if s is None:
            return False
        with self._lock:
            s.state = "uploading"
            s.attempts += 1
            s.last_upload_start = time.monotonic()
            epoch = s.epoch
        try:
            run(s.upload_sql)
        except Exception as e:
            with self._lock:
                s.state = "failed" if "GPUDB_UPLOAD_DISCARDED" not in str(e) else "stale"
                s.error = str(e)[:200]
            return False
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
            if s.state == "pending" and now >= s.resume_at:
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
        try:
            row = self._run(cur, s, "SELECT gpu_upload_status(?)", [s.tag])
            return json.loads(row[0][0])
        except Exception:
            return {"open": False}

    def _abort(self, cur, s: SetState) -> None:
        try:
            cur.execute("SELECT gpu_upload_abort(?)", [s.tag]).fetchall()
        except Exception:
            pass

    def _extension_ready(self, cur, s: SetState) -> Optional[int]:
        """rows_seen of the set if the extension holds it ready, else None."""
        try:
            row = cur.execute("SELECT state, rows_seen FROM gpu_residents() WHERE name = ?",
                              [s.tag]).fetchone()
            return int(row[1]) if row and row[0] == "ready" else None
        except Exception:
            return None

    def _session(self, cur, s: SetState, epoch: int) -> str:
        """One upload session for the set. Returns the outcome:
        ready | pending (retry later) | stale (invalidated meanwhile) | failed."""
        t_start = time.monotonic()
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
            self._run(cur, s, "SELECT gpu_upload_begin(?)", [s.tag])
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
            row = self._run(cur, s, "SELECT gpu_upload_finish(?)", [s.tag])
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
            s.rows_seen = rows
            s.finish_window = (t_fin, time.monotonic())
            s.session_ms = (time.monotonic() - t_start) * 1000.0
        return "ready"

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
            outcome = self._session(cur, s, epoch)
            with self._cv:
                now = time.monotonic()
                if outcome == "ready" and s.epoch == epoch and s.state == "uploading":
                    s.state = "ready"
                    self._log(f"resident: {s.tag} ({s.segments} segments, "
                              f"{s.interrupts} interrupts, {s.session_ms:.0f} ms)")
                elif outcome == "pending" and s.epoch == epoch and s.state == "uploading":
                    s.state = "pending"           # retry once idle again; no rate wait
                    s.resume_at = now
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
