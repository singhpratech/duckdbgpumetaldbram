"""Throwaway probe: run the shell's interactive (pty) sequence in a loop and,
when Ctrl-D does not leave the shell, print everything knowable about the child
at that moment.

    python3 python/tests/probe_ctrl_d.py [iterations]

The window under suspicion is microseconds wide, so nothing between the reads
and the writes that bound it does any work: marks are (label, time, index into
the transcript) tuples, formatted only when a round hangs.

Not part of the suite; it lives only on the diagnosis branch.
"""
import fcntl
import os
import pty
import select
import shutil
import subprocess
import sys
import termios
import time

PKG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, PKG)
_PP = os.environ.get("PYTHONPATH")
ENV = dict(os.environ, PYTHONPATH=os.pathsep.join([PKG] + ([_PP] if _PP else [])))


def run_cmd(argv, timeout=20):
    try:
        p = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True, timeout=timeout)
        return f"$ {' '.join(argv)}\n{p.stdout.strip()}"
    except Exception as e:                                   # noqa: BLE001
        return f"$ {' '.join(argv)}  -> {type(e).__name__}: {e}"


def lflags(attrs):
    names = ("ICANON", "ECHO", "ECHOE", "ECHOK", "ECHONL", "ISIG", "IEXTEN", "NOFLSH", "TOSTOP")
    lflag = attrs[3]
    on = [n for n in names if lflag & getattr(termios, n, 0)]
    cc = attrs[6]
    return (f"lflag={lflag:#x} on={on} VEOF={cc[termios.VEOF]!r} "
            f"VINTR={cc[termios.VINTR]!r} VMIN={cc[termios.VMIN]!r} VTIME={cc[termios.VTIME]!r}")


class Round:
    def __init__(self, n):
        self.n = n
        self.out = []          # cleared by the sequence, the way the suite does
        self.all = []          # everything, never cleared
        self.marks = []        # (label, t, index into self.all)
        self.t0 = time.time()
        self.primary, secondary = pty.openpty()
        self.slave = os.ttyname(secondary)

        def own_the_terminal():
            os.setsid()
            fcntl.ioctl(0, termios.TIOCSCTTY, 0)

        self.p = subprocess.Popen([sys.executable, "-m", "gpudb"], stdin=secondary,
                                  stdout=secondary, stderr=subprocess.STDOUT,
                                  env=dict(ENV, TERM="dumb"), close_fds=True,
                                  preexec_fn=own_the_terminal)
        os.close(secondary)

    def mark(self, label):
        # a shallow copy of the (few, short) chunks the suite has not cleared:
        # cheap enough not to move the race, enough to say what was on screen
        self.marks.append((label, time.time() - self.t0, len(self.all), tuple(self.out)))

    def send(self, text):
        data = text if isinstance(text, bytes) else text.encode()
        while data:
            data = data[os.write(self.primary, data):]

    def read_until(self, text, timeout=60.0):
        end = time.time() + timeout
        while time.time() < end:
            r, _, _ = select.select([self.primary], [], [], 0.2)
            if r:
                try:
                    chunk = os.read(self.primary, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                chunk = chunk.decode("utf-8", "replace")
                self.out.append(chunk)
                self.all.append(chunk)
            if text in "".join(self.out):
                return True
        return False

    def drain(self, seconds):
        end, got = time.time() + seconds, []
        while time.time() < end:
            r, _, _ = select.select([self.primary], [], [], 0.2)
            if r:
                try:
                    chunk = os.read(self.primary, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                got.append(chunk.decode("utf-8", "replace"))
        return "".join(got)

    # ---- the sequence the suite runs, step for step ----
    def play(self):
        out, send, until = self.out, self.send, self.read_until
        until("Enter .help for usage.")
        until("gpudb> ")
        send(b"SELECT 1 AS a,\n")
        until("...> ")
        send(b"       2 AS b;\n")
        until("│")
        until(" ms")
        del out[:]
        until("gpudb> ")
        self.mark("first statement done")

        del out[:]
        send(b"SELECT 1 AS thrown_away")
        until("thrown_away")
        send(b"\x03")
        until("gpudb> ")
        del out[:]
        send(b"SELECT 2 AS after_ctrl_c;\n")
        until("after_ctrl_c")
        self.mark("^C on a half-typed line")

        del out[:]
        until("gpudb> ")
        del out[:]
        send(b"SELECT count(*) FROM range(100000000000) r(i) WHERE i % 7 = 3;\n")
        until("\x00", timeout=2.0)
        self.mark("long statement running")
        send(b"\x03")
        until("INTERRUPT", timeout=60)
        del out[:]
        send(b"SELECT 3 AS still_here;\n")
        until("still_here", timeout=60)
        self.mark("still_here matched")
        del out[:]
        until("gpudb> ")
        self.mark("prompt matched")
        del out[:]
        send(b"\x04")
        self.mark("^D sent")
        until("\x00", timeout=30)
        try:
            return self.p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            return None

    # ---- what the child looks like when it will not leave ----
    def diagnose(self):
        say = [f"child pid={self.p.pid} poll={self.p.poll()} slave={self.slave}"]
        whole = "".join(self.all)
        cuts = []
        for label, t, idx, seen in self.marks:
            cuts.append(f"  +{t:6.2f}s {label:<28} seen so far ends: "
                        f"{''.join(self.all[:idx])[-90:]!r}")
        say.append("timeline:\n" + "\n".join(cuts))
        say.append(f"whole transcript ({len(whole)} chars):\n{whole!r}")
        for name in ("status", "wchan", "syscall", "stack"):
            path = f"/proc/{self.p.pid}/{name}"
            try:
                with open(path) as fh:
                    txt = fh.read()
                if name == "status":
                    keep = ("State", "Threads", "SigQ", "SigPnd", "SigBlk", "SigIgn", "SigCgt")
                    txt = "\n".join(l for l in txt.splitlines() if l.split(":")[0] in keep)
                say.append(f"--- {path}\n{txt.strip()}")
            except Exception as e:                           # noqa: BLE001
                say.append(f"--- {path}: {type(e).__name__}: {e}")
        try:
            for tid in sorted(os.listdir(f"/proc/{self.p.pid}/task")):
                bits = []
                for name in ("stat", "wchan", "comm"):
                    try:
                        with open(f"/proc/{self.p.pid}/task/{tid}/{name}") as fh:
                            raw = fh.read().strip()
                        bits.append(raw.split(") ", 1)[-1].split()[0] if name == "stat" else raw)
                    except Exception as e:                   # noqa: BLE001
                        bits.append(f"<{type(e).__name__}>")
                say.append(f"thread {tid}: comm={bits[2]} state={bits[0]} wchan={bits[1]}")
        except Exception as e:                               # noqa: BLE001
            say.append(f"tasks: {type(e).__name__}: {e}")
        try:
            say.append("master termios: " + lflags(termios.tcgetattr(self.primary)))
        except Exception as e:                               # noqa: BLE001
            say.append(f"master termios: {type(e).__name__}: {e}")
        say.append(run_cmd(["stty", "-a", "-F", self.slave]))
        spy = shutil.which("py-spy") or os.path.expanduser("~/.local/bin/py-spy")
        say.append(run_cmd(["sudo", "-n", spy, "dump", "--pid", str(self.p.pid), "--nonblocking"]))

        # a SECOND ^D out (the byte was lost)?  a \n first (the line was not
        # empty)?  a whole command (it is not reading at all)?
        for label, keys, wait in (("second ^D", [b"\x04"], 6),
                                  ("newline then ^D", [b"\n", b"\x04"], 6),
                                  (".quit", [b".quit\n"], 6)):
            if self.p.poll() is not None:
                break
            for k in keys:
                self.send(k)
                time.sleep(0.4)
            got = self.drain(wait)
            say.append(f"after {label}: alive={self.p.poll() is None} saw={got[-200:]!r}")
        return "\n".join(say)

    def cleanup(self):
        if self.p.poll() is None:
            self.p.kill()
            try:
                self.p.wait(timeout=10)
            except Exception:                                # noqa: BLE001
                pass
        try:
            os.close(self.primary)
        except OSError:
            pass


def shape(r):
    """Which prompt the round matched: the one readline prints when it arms
    (nothing rendered yet) or the one after the answer."""
    for label, _, _, seen in r.marks:
        if label == "prompt matched":
            return "answered" if "│" in "".join(seen) else "armed"
    return "?"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    print(run_cmd([sys.executable, "-c",
                   "import readline;print(readline.__doc__, "
                   "getattr(readline,'_READLINE_LIBRARY_VERSION','?'))"]), flush=True)
    fails, shapes = 0, {}
    for i in range(1, n + 1):
        r = Round(i)
        try:
            rc = r.play()
            s = shape(r)
            shapes[s] = shapes.get(s, 0) + 1
            if rc == 0:
                print(f"[{i}/{n}] ok   ({time.time() - r.t0:4.1f}s, prompt={s})", flush=True)
            else:
                fails += 1
                print(f"[{i}/{n}] HANG rc={rc} prompt={s}\n{r.diagnose()}\n{'=' * 70}", flush=True)
        finally:
            r.cleanup()
    print(f"\n{fails} hangs in {n} rounds; prompt shapes {shapes}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
