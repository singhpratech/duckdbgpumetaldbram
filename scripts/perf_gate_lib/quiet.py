"""The quiet-machine guard and the benchmark lock for perf_gate.

Two measures, both portable between macOS and Linux:
  - the one-minute load average (os.getloadavg), read before the run starts,
    when none of the gate's own processes exist yet;
  - the CPU share of OTHER processes: two `ps -A -o pid=,ppid=,time=`
    snapshots, the difference of every process's accumulated CPU time, minus
    the gate's own process tree, over the wall time between them and the
    number of cores. Accumulated time, not `ps`'s %CPU column, which is a
    decaying average on macOS and a lifetime average on Linux.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from typing import Dict, Iterable, Optional, Set, Tuple

CPUS = os.cpu_count() or 1


def _parse_cputime(s: str) -> float:
    """'[[dd-]hh:]mm:ss[.cc]' (Linux and macOS spell it differently) -> seconds."""
    s = s.strip()
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    parts = [float(p) for p in s.split(":")]
    secs = 0.0
    for p in parts:
        secs = secs * 60.0 + p
    return days * 86400.0 + secs


def _snapshot() -> Dict[int, Tuple[int, float]]:
    try:
        out = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,time="], capture_output=True,
                             text=True, timeout=10).stdout
    except Exception:
        return {}
    snap = {}
    for line in out.splitlines():
        f = line.split()
        if len(f) < 3:
            continue
        try:
            snap[int(f[0])] = (int(f[1]), _parse_cputime(f[2]))
        except ValueError:
            continue
    return snap


def _tree(snap: Dict[int, Tuple[int, float]], roots: Iterable[int]) -> Set[int]:
    mine = set(roots)
    grew = True
    while grew:
        grew = False
        for pid, (ppid, _) in snap.items():
            if ppid in mine and pid not in mine:
                mine.add(pid)
                grew = True
    return mine


class CpuSampler:
    """CPU share of processes outside the gate, between two calls."""

    def __init__(self, own_roots: Optional[Iterable[int]] = None):
        self.roots = set(own_roots or [os.getpid()])
        self._t = time.monotonic()
        self._snap = _snapshot()

    def sample(self) -> Dict[str, float]:
        """Share of the machine (0..1) other processes used since the last
        call, and the busiest of them."""
        t = time.monotonic()
        snap = _snapshot()
        wall = max(1e-3, t - self._t)
        mine = _tree(snap, self.roots) | _tree(self._snap, self.roots)
        others = 0.0
        top = (0.0, 0)
        for pid, (_, cpu) in snap.items():
            if pid in mine:
                continue
            before = self._snap.get(pid)
            # a process that started in between: all of its time is inside the window
            d = cpu - before[1] if before else cpu
            d = max(0.0, min(d, wall * CPUS))
            others += d
            if d > top[0]:
                top = (d, pid)
        self._t, self._snap = t, snap
        return {"others_share": others / (wall * CPUS), "wall_s": wall,
                "top_pid": top[1], "top_cores": top[0] / wall}


def load_average() -> float:
    try:
        return os.getloadavg()[0]
    except OSError:
        return 0.0


def pre_check(max_load: float, max_share: float, seconds: float = 5.0) -> Dict:
    """Before the run: load average and other processes' CPU share over
    `seconds` of the gate doing nothing."""
    s = CpuSampler()
    time.sleep(seconds)
    cpu = s.sample()
    load = load_average()
    quiet = load <= max_load and cpu["others_share"] <= max_share
    return {"load1": load, "max_load": max_load, "others_share": cpu["others_share"],
            "max_share": max_share, "top_pid": cpu["top_pid"], "top_cores": cpu["top_cores"],
            "cpus": CPUS, "quiet": quiet}


def top_command(pid: int) -> str:
    try:
        comm = subprocess.run(["ps", "-o", "comm=", "-p", str(pid)], capture_output=True,
                              text=True, timeout=5).stdout.strip()
        return f"{os.path.basename(comm)[:40]} (pid {pid})"
    except Exception:
        return "?"


class BenchLock:
    """The machine-wide benchmark lock. While it exists other jobs on the
    machine hold off heavy work. Created exclusively (O_EXCL), removed on exit
    however the run ends; a lock that already exists is never taken over."""

    def __init__(self, path: str):
        self.path = path
        self.held = False

    def exists(self) -> bool:
        return os.path.exists(self.path)

    def holder(self) -> str:
        try:
            with open(self.path) as f:
                return f.read().strip()[:300]
        except OSError:
            return ""

    def acquire(self) -> bool:
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps({"holder": "scripts/perf_gate.py", "pid": os.getpid(),
                                "host": socket.gethostname(),
                                "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                "argv": sys.argv[1:]}) + "\n")
        self.held = True
        return True

    def release(self) -> None:
        if not self.held:
            return
        try:
            with open(self.path) as f:
                mine = json.loads(f.read() or "{}").get("pid") == os.getpid()
        except (OSError, ValueError):
            mine = False
        if mine:
            try:
                os.unlink(self.path)
            except OSError:
                pass
        self.held = False
