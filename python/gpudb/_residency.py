"""Automatic residency (§5.5): what becomes resident, and when it is ready.

The manager never uploads beside a user statement. An upload runs on the
wrapper's own cursor only while every connection of the wrapper has been
idle for `idle_ms`; the moment a user statement arrives the upload cursor is
interrupted (DuckDB checks interrupts between vectors, ~0.5 ms), the upload
is retried after the quiet period, and the user's statement runs alone.
Until the extension gains upload sessions (milestone 0c) a retried upload
starts over; a session that never goes idle therefore never uploads and
runs native throughout — rule 1 holds, the win is simply not there yet.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional


@dataclass
class SetState:
    tag: str
    upload_sql: str
    state: str = "missing"          # missing | pending | uploading | ready | stale | failed
    epoch: int = 0
    last_invalidate: float = 0.0
    last_upload_start: float = 0.0
    attempts: int = 0
    error: str = ""


class ResidencyManager:
    def __init__(self, cursor_factory: Callable[[], object], *, mode: str = "background",
                 idle_ms: float = 20.0, quiet_s: float = 2.0, rate_s: float = 30.0,
                 max_attempts: int = 20, log: Optional[Callable[[str], None]] = None):
        self._cursor_factory = cursor_factory
        self.mode = mode
        self.idle_ms = idle_ms
        self.quiet_s = quiet_s
        self.rate_s = rate_s
        self.max_attempts = max_attempts
        self._log = log or (lambda m: None)
        self._sets: Dict[str, SetState] = {}
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._in_flight = 0
        self._last_activity = time.monotonic()
        self._upload_cursor = None
        self._uploading_tag: Optional[str] = None
        self._closed = False
        self._thread: Optional[threading.Thread] = None

    # ---- statement activity (called by the connection on every statement) ----
    def statement_begin(self) -> None:
        with self._lock:
            self._in_flight += 1
            self._last_activity = time.monotonic()
            cur, tag = self._upload_cursor, self._uploading_tag
        if cur is not None and tag is not None:
            try:
                cur.interrupt()
            except Exception:
                pass

    def statement_end(self) -> None:
        with self._cv:
            self._in_flight = max(0, self._in_flight - 1)
            self._last_activity = time.monotonic()
            self._cv.notify_all()

    # ---- set bookkeeping ----
    def get(self, tag: str) -> Optional[SetState]:
        with self._lock:
            return self._sets.get(tag)

    def is_ready(self, tag: str) -> bool:
        s = self.get(tag)
        return s is not None and s.state == "ready"

    def note_candidate(self, tag: str, upload_sql: str) -> SetState:
        """A rewritable shape over a non-resident set was seen."""
        with self._cv:
            s = self._sets.get(tag)
            if s is None:
                s = SetState(tag=tag, upload_sql=upload_sql)
                self._sets[tag] = s
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
        """Local bookkeeping; the caller runs gpu_invalidate on the database."""
        with self._cv:
            now = time.monotonic()
            for t, s in self._sets.items():
                if tag is None or t == tag:
                    s.epoch += 1
                    s.last_invalidate = now
                    if s.state in ("ready", "uploading", "pending"):
                        s.state = "stale"
            cur, utag = self._upload_cursor, self._uploading_tag
        if cur is not None and utag is not None and (tag is None or tag == utag):
            try:
                cur.interrupt()
            except Exception:
                pass

    def snapshot(self) -> Dict[str, str]:
        with self._lock:
            return {t: s.state for t, s in self._sets.items()}

    # ---- synchronous upload (residency='eager', and tests) ----
    def upload_now(self, tag: str, run: Callable[[str], None]) -> bool:
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
            if s.state != "pending":
                continue
            if now - s.last_invalidate < self.quiet_s:
                continue
            if now - s.last_upload_start < self.rate_s and s.attempts > 0 and s.error == "":
                continue
            return s
        return None

    def _worker(self) -> None:
        while True:
            with self._cv:
                while not self._closed:
                    s = self._pick()
                    idle = (time.monotonic() - self._last_activity) * 1000.0
                    if s is not None and self._in_flight == 0 and idle >= self.idle_ms:
                        break
                    self._cv.wait(timeout=max(0.005, self.idle_ms / 1000.0))
                if self._closed:
                    return
                s.state = "uploading"
                s.attempts += 1
                s.last_upload_start = time.monotonic()
                epoch = s.epoch
                if self._upload_cursor is None:
                    try:
                        self._upload_cursor = self._cursor_factory()
                    except Exception as e:
                        s.state = "failed"
                        s.error = str(e)[:200]
                        continue
                self._uploading_tag = s.tag
                cur = self._upload_cursor
            ok, err = True, ""
            try:
                cur.execute(s.upload_sql).fetchall()
            except Exception as e:
                ok, err = False, str(e)
            with self._cv:
                self._uploading_tag = None
                if ok and s.epoch == epoch and s.state == "uploading":
                    s.state = "ready"
                    s.error = ""
                    self._log(f"resident: {s.tag}")
                elif not ok and ("INTERRUPT" in err.upper() or "GPUDB_UPLOAD_DISCARDED" in err):
                    s.state = "pending"        # a user statement arrived; try again when idle
                    s.error = ""
                    s.last_invalidate = time.monotonic()   # honour the quiet period
                elif not ok:
                    s.state = "failed"
                    s.error = err[:200]
                    self._log(f"upload failed: {s.tag}: {err[:120]}")
                else:
                    s.state = "stale"          # invalidated while uploading
                self._cv.notify_all()

    def close(self) -> None:
        with self._cv:
            self._closed = True
            cur, tag = self._upload_cursor, self._uploading_tag
            self._cv.notify_all()
        if cur is not None and tag is not None:
            try:
                cur.interrupt()
            except Exception:
                pass
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
