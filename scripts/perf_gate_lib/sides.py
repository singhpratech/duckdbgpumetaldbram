"""Worker processes as seen from the driver: start one per side, send it one
JSON line, read one back."""
from __future__ import annotations

import json
import os
import select
import subprocess
import sys
from typing import Dict, Optional

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worker.py")


class WorkerError(RuntimeError):
    pass


class Worker:
    def __init__(self, side: str, wrapper: str, ext: str, log_path: str, pycache: str,
                 env: Optional[Dict] = None):
        self.side = side
        self.log = open(log_path, "ab")
        # Bytecode goes to a cache of the gate's own, never into a tree under
        # test, and every tree is on the same footing: a tree that happens to
        # carry a __pycache__ of its own would otherwise start ~2 ms faster
        # and ~10 MiB smaller than one that does not (measured).
        cmd = [sys.executable, "-I", "-X", f"pycache_prefix={pycache}", WORKER, "--side", side,
               "--wrapper", wrapper]
        if ext:
            cmd += ["--ext", ext]
        e = dict(os.environ)
        e.update(env or {})
        # the extension binary is chosen per side, by --ext alone: a value
        # inherited from the caller's shell would silently put both sides on it
        e.pop("GPUDB_EXTENSION_PATH", None)
        if ext:
            e["GPUDB_EXTENSION_PATH"] = ext
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.log, env=e, bufsize=0)
        first = self._read(120)
        if not first.get("ok"):
            raise WorkerError(f"{side}: {first.get('error')}")
        self.info = first["ready"]

    def _read(self, timeout: float) -> Dict:
        buf = b""
        while not buf.endswith(b"\n"):
            r, _, _ = select.select([self.proc.stdout], [], [], timeout)
            if not r:
                self.kill()
                raise WorkerError(f"{self.side}: no answer in {timeout:.0f} s")
            chunk = os.read(self.proc.stdout.fileno(), 1 << 20)
            if not chunk:
                raise WorkerError(f"{self.side}: worker exited (code {self.proc.poll()})")
            buf += chunk
        return json.loads(buf.decode())

    def call(self, timeout: float = 1800, **msg) -> Dict:
        self.proc.stdin.write((json.dumps(msg) + "\n").encode())
        self.proc.stdin.flush()
        res = self._read(timeout)
        if not res.get("ok"):
            raise WorkerError(f"{self.side}: {msg.get('op')}: {res.get('error')}")
        return res

    def quit(self) -> None:
        try:
            if self.proc.poll() is None:
                self.proc.stdin.write(b'{"op": "quit"}\n')
                self.proc.stdin.flush()
                self.proc.wait(timeout=60)
        except Exception:
            self.kill()
        finally:
            self.log.close()

    def kill(self) -> None:
        try:
            self.proc.kill()
            self.proc.wait(timeout=10)
        except Exception:
            pass
