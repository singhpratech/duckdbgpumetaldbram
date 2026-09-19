"""Throwaway probe: what a terminal does with a ^D that arrives while the
program is NOT inside `input()`.

No gpudb here at all — the child is eight lines of stock Python that loop over
`input()` and sleep for a second when told to `work`. That is the whole of the
suspicion: `input()` puts the terminal in readline's own (non-canonical) mode
and puts it back when it returns, so a ^D typed while a statement runs reaches
a canonical terminal, and the next `input()` flips the mode again before anyone
reads it.

    python3 python/tests/probe_eof_window.py

Three cases, five times each:
  idle       ^D at a prompt that is waiting            (must leave)
  busy       ^D while the child is working             (?)
  typeahead  `work\\n` and ^D in ONE write at a prompt  (?)
"""
import fcntl
import os
import pty
import select
import subprocess
import sys
import termios
import time

CHILD = r"""
import sys, time
try:
    import readline
    sys.stderr.write("readline: %s\n" % (readline.__doc__ or "").strip())
except ImportError:
    pass
print("READY", flush=True)
while True:
    try:
        line = input("p> ")
    except EOFError:
        print("EOF", flush=True)
        break
    if line.strip() == "work":
        time.sleep(1.0)
        print("DONE", flush=True)
print("BYE", flush=True)
"""


def child():
    def own_the_terminal():
        os.setsid()
        fcntl.ioctl(0, termios.TIOCSCTTY, 0)

    primary, secondary = pty.openpty()
    p = subprocess.Popen([sys.executable, "-u", "-c", CHILD], stdin=secondary,
                         stdout=secondary, stderr=subprocess.STDOUT,
                         close_fds=True, preexec_fn=own_the_terminal)
    os.close(secondary)
    return p, primary


def reader(primary, out):
    def read_until(text, timeout=10.0):
        end = time.time() + timeout
        while time.time() < end:
            r, _, _ = select.select([primary], [], [], 0.2)
            if r:
                try:
                    chunk = os.read(primary, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                out.append(chunk.decode("utf-8", "replace"))
            if text in "".join(out):
                return True
        return False
    return read_until


def mode(primary):
    try:
        a = termios.tcgetattr(primary)
        return ("canonical" if a[3] & termios.ICANON else "raw") + \
               (" +echo" if a[3] & termios.ECHO else " -echo")
    except Exception as e:                                   # noqa: BLE001
        return f"<{type(e).__name__}>"


def one(case):
    p, primary = child()
    out = []
    until = reader(primary, out)
    note = []
    try:
        until("READY")
        until("p> ")
        if case == "idle":
            note.append(f"terminal at the prompt: {mode(primary)}")
            os.write(primary, b"\x04")
        elif case == "busy":
            os.write(primary, b"work\n")
            time.sleep(0.4)                      # the child is inside time.sleep now
            note.append(f"terminal while working: {mode(primary)}")
            os.write(primary, b"\x04")
            until("DONE")
            until("p> ")
            note.append(f"terminal at the next prompt: {mode(primary)}")
        elif case == "typeahead":
            note.append(f"terminal at the prompt: {mode(primary)}")
            os.write(primary, b"work\n\x04")     # both bytes while readline is armed
            until("DONE")
        left = until("EOF", timeout=6.0)
        try:
            rc = p.wait(timeout=6)
        except subprocess.TimeoutExpired:
            rc = None
        return rc, left, note, "".join(out)
    finally:
        if p.poll() is None:
            p.kill()
            p.wait(timeout=5)
        os.close(primary)


def main():
    print(f"{sys.platform} python {sys.version.split()[0]}", flush=True)
    bad = 0
    for case in ("idle", "busy", "typeahead"):
        results = []
        last = None
        for _ in range(5):
            rc, left, note, text = one(case)
            results.append("left" if rc == 0 else f"STAYED(rc={rc})")
            last = (note, text)
        print(f"\n== {case}: {results}")
        for line in last[0]:
            print(f"   {line}")
        print(f"   transcript: {last[1][-220:]!r}", flush=True)
        if any(r != "left" for r in results):
            bad += 1
    return 0 if bad == 0 else 0      # informational: never fail the job


if __name__ == "__main__":
    sys.exit(main())
