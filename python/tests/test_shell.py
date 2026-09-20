"""Tests for the `gpudb` shell (python/gpudb/_shell.py).
Plain script: `python3 python/tests/test_shell.py` (no pytest needed); also
collected by pytest if present. The shell is always run as a subprocess, the
way a user runs it. Everything here passes both with a built extension and
without one (in CI the wheel job has none): the tests that need the device
check `gpu_build_info()` first and say so when they are skipped."""
import os
import re
import subprocess
import sys
import tempfile
import time

PKG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, PKG)
import duckdb                      # noqa: E402
import gpudb                       # noqa: E402

FAILS = []
SKIPS = []
_PP = os.environ.get("PYTHONPATH")
ENV = dict(os.environ, PYTHONPATH=os.pathsep.join([PKG] + ([_PP] if _PP else [])))
# a big enough table for the wrapper to look at a statement at all (floor_rows)
BIG = "CREATE TABLE t AS SELECT (i%20000)::BIGINT k, i::BIGINT v FROM range(2000000) r(i)"
FEW = "CREATE TABLE f AS SELECT (i%7)::BIGINT k, i::BIGINT v FROM range(2000000) r(i)"


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print("  FAIL", msg)
    else:
        print("  ok  ", msg)


def skip(msg):
    SKIPS.append(msg)
    print("  skip", msg)


def shell(*args, stdin="", timeout=300):
    """Run the shell as a subprocess. Returns (returncode, stdout, stderr)."""
    p = subprocess.run([sys.executable, "-m", "gpudb"] + list(args),
                       input=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       universal_newlines=True, env=ENV, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def field(out, key):
    """A `key:` line of `.gpu` / `.memory`, whose label is padded to a column."""
    for line in out.splitlines():
        if line.startswith(key + ":"):
            return line[len(key) + 1:].strip()
    return None


def footers(out):
    return " / ".join(l for l in out.splitlines() if l.endswith(" ms"))


def backend():
    """What this build offers, read the way the shell reads it."""
    try:
        con = gpudb.connect()
        info = con.execute("SELECT gpu_build_info()").fetchone()[0]
        con.close()
        return info
    except Exception:
        return ""


def run():
    info = backend()
    gpu = "exact=true" in info and "runtime=cpu" not in info
    print(f"== extension: {info or 'not loaded (degraded plain-DuckDB mode)'}")

    print("== -c")
    rc, out, err = shell("-c", "SELECT 42 AS answer")
    check(rc == 0 and "42" in out and "answer" in out, f"-c runs one statement (rc={rc})")
    check("│" in out, "-c renders DuckDB's box")
    rc, out, _ = shell("-c", "SELECT 1 AS a", "-c", "SELECT 2 AS b")
    check(rc == 0 and out.index(" a ") < out.index(" b "),
          "-c twice runs both, in order")
    rc, out, _ = shell("-c", "SELECT 1 AS a; SELECT 2 AS b;")
    check(rc == 0 and " a " in out and " b " in out, "-c runs several statements in one string")
    rc, out, _ = shell("-c", "SELECT 3 AS trailing")   # no trailing semicolon
    check(rc == 0 and "trailing" in out, "-c runs a statement with no trailing `;`")

    print("== stdin")
    rc, out, _ = shell(stdin="SELECT 7 AS piped;\n")
    check(rc == 0 and "piped" in out, "SQL piped on stdin runs non-interactively")
    rc, out, _ = shell(stdin="SELECT\n  11 AS spread,\n  12 AS over_lines;\nSELECT 13 AS second;\n")
    check(rc == 0 and "spread" in out and "over_lines" in out and "second" in out,
          "a statement spanning lines, then a second statement")
    rc, out, _ = shell(stdin="SELECT 'a;b' AS s;\n")
    check(rc == 0 and "a;b" in out, "a `;` inside a string does not split the statement")
    rc, out, _ = shell(stdin="-- a comment with ; in it\nSELECT 5 AS c; /* ; */ SELECT 6 AS d;\n")
    check(rc == 0 and " c " in out and " d " in out, "comments (line and block) do not split statements")
    rc, out, _ = shell(stdin="SELECT $tag$ ; $tag$ AS dollar;\n")
    check(rc == 0 and "dollar" in out, "dollar-quoting does not split the statement")

    print("== -f and .read")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "script.sql")
        with open(path, "w") as fh:
            fh.write("CREATE TABLE s(a INTEGER);\nINSERT INTO s VALUES (1),(2);\n"
                     "SELECT sum(a) AS from_file FROM s;\n")
        rc, out, _ = shell("-f", path)
        check(rc == 0 and "from_file" in out and "3" in out, "-f runs the statements in a file")
        rc, out, _ = shell(stdin=f".read {path}\nSELECT count(*) AS after_read FROM s;\n")
        check(rc == 0 and "from_file" in out and "after_read" in out, ".read runs a file")
        rc, out, err = shell(stdin=".read /no/such/file.sql\n")
        check(rc == 1 and "No such file" in err, ".read of a missing file is an error")

    print("== errors")
    rc, out, err = shell("-c", "SELECT * FROM nope")
    check(rc == 1, f"a failing statement exits non-zero (rc={rc})")
    check("Catalog Error" in err and "Traceback" not in err,
          "the error is DuckDB's own message, and never a traceback")
    rc, out, _ = shell("-c", "SELECT * FROM nope", "-c", "SELECT 'reached' AS second")
    check(rc == 1 and "reached" not in out, "-c stops at the first error")
    rc, out, err = shell(stdin="SELECT * FROM nope;\nSELECT 'reached' AS second;\n")
    check(rc == 1 and "reached" not in out, "piped input stops at the first error")
    rc, out, err = shell("-c", "SELEKT 1")
    check(rc == 1 and "Parser Error" in err and "Traceback" not in err,
          "a syntax error is DuckDB's message, no traceback")

    print("== rendering")
    rc, out, _ = shell("-c", "SELECT NULL AS n, 'x' AS s, 1.5::DECIMAL(4,2) AS d, DATE '2020-01-02' AS dt")
    check(rc == 0 and "NULL" in out and "decimal(4,2)" in out.lower() and "2020-01-02" in out,
          "NULLs and column types render as DuckDB renders them")
    rc, out, _ = shell("-c", "SELECT i, i::VARCHAR AS s FROM range(1000) r(i)")
    check(rc == 0 and "1000 rows" in out and "40 shown" in out,
          "a long result is cut like the CLI, with the row footer")
    rc, out, _ = shell("-c", "CREATE TABLE q(a INTEGER)")
    check(rc == 0 and out.strip() == "", "a statement with no result prints nothing")

    print("== dot-commands")
    rc, out, _ = shell(stdin=".help\n")
    check(rc == 0 and ".residents" in out and ".schema" in out, ".help lists the commands")
    rc, out, err = shell(stdin=".nope\n")
    check(rc == 1 and ".help" in err and "Traceback" not in err,
          "an unknown dot-command points at .help")
    rc, out, _ = shell(stdin=".version\n")
    check(rc == 0 and gpudb.__version__ in out and duckdb.__version__ in out,
          ".version names both versions")
    rc, out, _ = shell(stdin="CREATE TABLE u(a INTEGER, b VARCHAR);\n.tables\n.schema u\n")
    check(rc == 0 and out.count("u") >= 2 and "VARCHAR" in out, ".tables and .schema (with a table)")
    rc, out, _ = shell(stdin="CREATE TABLE u(a INTEGER);\n.schema\n")
    check(rc == 0 and "CREATE TABLE u" in out, ".schema with no argument shows the DDL")
    rc, out, _ = shell(stdin=".residents\n.memory\n")
    check(rc == 0 and "Nothing is resident yet." in out and "budget:" in out
          and "residency:" in out, ".residents and .memory print the wrapper's own tables")
    rc, out, _ = shell("--residency", "eager", "-c", BIG,
                       "-c", "SELECT k, sum(v) FROM t GROUP BY k ORDER BY k LIMIT 1",
                       "-c", ".residents")
    if gpu:
        head = [l for l in out.splitlines()
                if l.startswith("table ") and "width" in l and "rows" in l]
        check(rc == 0 and len(head) == 1 and head[0].split()[:5] == [
            "table", "column", "dtype", "rows", "width"],
              f".residents heads a per-column table with width and rows ({head})")
        body = [l for l in out.splitlines() if l.startswith("t ") and " B " in l]
        check(bool(body) and all(any(w in l for w in (" 1 B", " 2 B", " 4 B", " 8 B", " - "))
                                 for l in body),
              f".residents gives every resident column a width and a row count ({body})")
        check("resident column" in out and "width is the bytes" in out,
              ".residents says what the width column means")
    else:
        skip("no exact GPU path on this build: the per-column .residents table")
        check(rc == 0, ".residents runs without a resident column too")
    rc, out, _ = shell(stdin=".quit\nSELECT 'never' AS x;\n")
    check(rc == 0 and "never" not in out, ".quit stops reading")
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "other.duckdb")
        rc, out, _ = shell(stdin=f".open {db}\nCREATE TABLE o(a INTEGER);\n.tables\n")
        check(rc == 0 and " o " in out and os.path.exists(db), ".open opens another database")

    print("== the timer footer")
    rc, out, _ = shell("-c", "SELECT 1 AS a")
    check("ms" not in out, "no footer in -c output by default")
    rc, out, _ = shell("--timer", "-c", "SELECT 1 AS a")
    check(" ms" in out and ("DuckDB" in out or "GPU" in out), "--timer prints one footer line")
    check("\x1b[" not in out, "no colour when the output is not a terminal")

    print("== the transparent switch")
    rc, out, _ = shell("--timer", "-c", BIG, "-c", "SELECT k, sum(v) FROM t GROUP BY k",
                       "-c", ".gpu")
    check(rc == 0 and "rewritten:" in out and "reason:" in out,
          ".gpu prints the whole of last_rewrite()")
    # every row's label is padded to the same column — `round_trip_ms:` is the
    # longest one and used to leave no space at all — and a time is printed as
    # a time, not as 17 digits of float repr
    rows = [re.sub(r"\x1b\[[0-9;]*m", "", l) for l in out.splitlines()
            if re.match(r"^[a-z_]+:", re.sub(r"\x1b\[[0-9;]*m", "", l))]
    check(rows and len({len(l) - len(l.split(":", 1)[1].lstrip()) for l in rows}) == 1,
          f".gpu: every label ends in the same column ({[l.split(':')[0] for l in rows]})")
    rt = [l for l in rows if l.startswith("round_trip_ms:")]
    check(not rt or re.fullmatch(r"round_trip_ms: +\d+\.\d{3} ms", rt[0]),
          f".gpu: a time reads as a time ({rt[0] if rt else 'not printed for this statement'})")
    rc, out, _ = shell("--timer", "--no-gpu", "-c", BIG,
                       "-c", "SELECT k, sum(v) FROM t GROUP BY k")
    check(rc == 0 and "DuckDB (off" in out, "--no-gpu leaves the statement on DuckDB")
    check("the transparent path is off" in out,
          f"the footer carries the wrapper's own detail for 'off'\n       {footers(out)}")
    rc, out, _ = shell("--timer", "-c", BIG, "-c", ".gpu off",
                       "-c", "SELECT k, sum(v) FROM t GROUP BY k")
    check(rc == 0 and "DuckDB (off" in out, ".gpu off makes the next statement run on DuckDB")

    print("== the device")
    if gpu:
        q = "SELECT k, sum(v) FROM t GROUP BY k ORDER BY k LIMIT 3"
        rc, out, _ = shell("--residency", "eager", "--timer", "-c", BIG, "-c", q, "-c", q, "-c", ".gpu")
        check(rc == 0 and "GPU (plain" in out,
              f"a resident GROUP BY runs on the GPU\n       {footers(out)}")
        check(field(out, "rewritten") == "True" and field(out, "form") == "plain"
              and "gpudb:v1:" in out,
              ".gpu after a rewritten statement names the form and the set")
        check((field(out, "detail") or "").startswith("the resident GROUP BY"),
              f".gpu shows the wrapper's detail, naming the path ({field(out, 'detail')!r})")
        check("GPU (plain: the resident GROUP BY" in out,
              f"the footer carries the detail beside the form\n       {footers(out)}")
        few = "SELECT k, sum(v) FROM f GROUP BY k ORDER BY k"
        rc, out, _ = shell("--residency", "eager", "--timer", "-c", FEW, "-c", few, "-c", ".gpu")
        check(rc == 0 and "DuckDB (threshold" in out and field(out, "rewritten") == "False"
              and field(out, "reason") == "threshold",
              "a shape under the thresholds stays on DuckDB, and .gpu says so")
        check("DuckDB (threshold: " in out,
              f"the footer carries the wrapper's own reason\n       {footers(out)}")
        rc, out, _ = shell("--residency", "eager", "--timer", "-c", BIG, "-c", q, "-c", q,
                           "-c", ".gpu off", "-c", q)
        check(rc == 0 and "DuckDB (off" in out.rsplit("GPU (plain", 1)[-1],
              ".gpu off turns a rewritten statement back to DuckDB")
    else:
        skip("no exact GPU path on this build: the GPU-side footer cases")
        rc, out, _ = shell("--timer", "-c", BIG,
                           "-c", "SELECT k, sum(v) FROM t GROUP BY k ORDER BY k LIMIT 3")
        check(rc == 0 and "sum(v)" in out and "99000000" in out and "DuckDB (" in out,
              "without the exact path the same statement answers on DuckDB, and the footer says so")

    print("== --readonly")
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "ro.duckdb")
        rc, out, _ = shell(db, "-c", "CREATE TABLE r(a INTEGER)")
        check(rc == 0, "a file database takes a write")
        rc, out, err = shell("--readonly", db, "-c", "INSERT INTO r VALUES (1)")
        check(rc == 1 and "read-only" in err.lower() and "Traceback" not in err,
              "--readonly refuses a write with DuckDB's own error")
        rc, out, _ = shell("--readonly", db, "-c", "SELECT count(*) AS n FROM r")
        check(rc == 0 and " n " in out, "--readonly still reads")
        # `.open` with no argument asks for the in-memory database, which
        # DuckDB refuses to open read-only ("Cannot launch in-memory database
        # in read-only mode!"). --readonly is a promise about the FILES the
        # session was pointed at, and an in-memory database is not one of them.
        rc, out, err = shell("--readonly", db, "-c", ".open", "-c", "CREATE TABLE m(a INTEGER)",
                             "-c", ".tables")
        check(rc == 0 and " m " in out and "in-memory" not in err,
              f"--readonly then `.open`: the in-memory database opens and takes a write ({err[:70]})")
        # ... and a FILE the same session opens still gets the flag
        rc, out, err = shell("--readonly", db, "-c", f".open {db}", "-c", "INSERT INTO r VALUES (1)")
        check(rc == 1 and "read-only" in err.lower(),
              "--readonly then `.open <file>`: the file is still read-only")

    print("== entry points")
    rc, out, _ = shell("--version")
    check(rc == 0 and out.startswith("gpudb ") and gpudb.__version__ in out,
          "python -m gpudb --version")
    check(console_script(), "the console script `gpudb` resolves after an install")

    print("== a bounded way out")
    bounded_close()

    print("== interactive (pty)")
    interactive()

    print()
    if SKIPS:
        print(f"{len(SKIPS)} skipped")
    print(f"{len(FAILS)} failures" if FAILS else "all shell tests passed")
    return 1 if FAILS else 0


def bounded_close():
    """The shell never waits long on the connection's close: after an
    interrupted statement DuckDB can still be unwinding, and the shell leaves."""
    import threading, time
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from gpudb import _shell

    class Stuck:
        def close(self):
            time.sleep(30)

    sh = _shell.Shell.__new__(_shell.Shell)
    sh.con = Stuck()
    saved = _shell.CLOSE_S
    _shell.CLOSE_S = 0.3
    try:
        t0 = time.time()
        finished = sh.close()
        took = time.time() - t0
    finally:
        _shell.CLOSE_S = saved
    check(finished is False and took < 3.0,
          f"a close that blocks is abandoned after the bound (finished={finished}, {took:.1f} s)")

    class Quick:
        def close(self):
            pass
    sh.con = Quick()
    check(sh.close() is True, "an ordinary close finishes and says so")


def console_script():
    """`pip install ./python` into a throwaway virtual environment and run the
    `gpudb` the entry point installs. --system-site-packages + --no-deps keeps
    it offline; anything missing here is a skip, not a failure."""
    import venv
    with tempfile.TemporaryDirectory() as tmp:
        try:
            venv.create(tmp, with_pip=True, system_site_packages=True)
        except Exception as e:
            skip(f"no virtual environment available ({e})")
            return True
        pip = os.path.join(tmp, "bin", "pip")
        exe = os.path.join(tmp, "bin", "gpudb")
        if not os.path.exists(pip):
            skip("the virtual environment has no pip")
            return True
        p = subprocess.run([pip, "install", "--quiet", "--no-deps", "--no-build-isolation", PKG],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True, timeout=600)
        if p.returncode != 0:
            skip(f"pip install of the package did not run here ({p.stdout.strip()[-200:]})")
            return True
        if not os.path.exists(exe):
            print("  FAIL console script not installed:", os.listdir(os.path.join(tmp, "bin")))
            return False
        p = subprocess.run([exe, "--version"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True, timeout=300)
        return p.returncode == 0 and p.stdout.startswith("gpudb ")


def interactive():
    """The prompt, the continuation prompt, ^C and Ctrl-D, through a
    pseudo-terminal. The shell has to OWN that terminal for ^C to become a
    signal at all, so the child gets its own session and claims it."""
    try:
        import fcntl
        import pty
        import termios
    except ImportError:
        skip("no pty on this platform: the interactive cases")
        return
    try:
        primary, secondary = pty.openpty()
    except OSError as e:
        skip(f"no pseudo-terminal available ({e}): the interactive cases")
        return

    def own_the_terminal():
        os.setsid()
        fcntl.ioctl(0, termios.TIOCSCTTY, 0)

    try:
        p = subprocess.Popen([sys.executable, "-m", "gpudb"], stdin=secondary, stdout=secondary,
                             stderr=subprocess.STDOUT, env=dict(ENV, TERM="dumb"),
                             close_fds=True, preexec_fn=own_the_terminal)
    except Exception as e:
        skip(f"the child cannot take the terminal here ({e}): the interactive cases")
        os.close(primary)
        os.close(secondary)
        return
    os.close(secondary)
    out = []

    def send(text):
        """A write to a terminal is short when its input queue fills up."""
        data = text if isinstance(text, bytes) else text.encode()
        while data:
            data = data[os.write(primary, data):]

    def interrupt_at_the_prompt(tries=8, timeout=5.0):
        """^C at the prompt, repeated until the shell answers with its own `^C`.

        `input()` looks for a signal only where the `select()` inside readline
        returns EINTR, so a SIGINT that lands in the instant readline is
        digesting the previous keystroke stays pending, unseen, until the next
        byte arrives — and it is THAT byte's line which is then thrown away.
        Nothing is printed while it is pending, so the only way to tell the
        shell is waiting again is to ask a second time."""
        for _ in range(tries):
            send(b"\x03")
            if read_until("^C", timeout=timeout):
                return True
            timeout = 2.0
        return False

    def read_until(text, timeout=60.0, after=None):
        """Wait for `text` in what the terminal has sent. `after` waits for it
        BEYOND an earlier marker — which clearing the buffer cannot do, because
        one read can already hold the marker and everything that follows it."""
        import select
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
            seen = "".join(out)
            if after is None:
                if text in seen:
                    return True
            else:
                head = seen.find(after)
                if head >= 0 and text in seen[head + len(after):]:
                    return True
        return False

    try:
        check(read_until("Enter .help for usage."), "the banner points at .help")
        text = "".join(out)
        check("database:" in text and "backend:" in text and "transparent:" in text
              and f"gpudb {gpudb.__version__}" in text,
              "the banner names the version, the backend and the database")
        dev = re.search(r"device='([^']*)'", backend())
        if dev and dev.group(1):
            line = [l for l in text.splitlines() if "backend:" in l]
            check(bool(line) and dev.group(1) in line[0],
                  f"the banner names the device the extension reports "
                  f"({dev.group(1)!r} in {line!r})")
        else:
            skip("this build reports no device name: the banner's device clause")
        check(read_until("gpudb> "), "the prompt")
        send(b"SELECT 1 AS a,\n")
        check(read_until("...> "), "a half-typed statement gets the continuation prompt")
        send(b"       2 AS b;\n")
        check(read_until("│"), "the finished statement renders its box")
        check(read_until(" ms"), "the timer footer is on by default at a terminal")
        # the NEW prompt: the one after the footer, not the one already read
        check(read_until("gpudb> ", after=" ms"), "the prompt comes back after a statement")

        del out[:]
        send(b"SELECT 1 AS thrown_away")
        read_until("thrown_away")
        took = interrupt_at_the_prompt()    # ^C with a half-typed line
        # the prompt that FOLLOWS the shell's own `^C` line, never one that was
        # already on screen: `after=` looks past a marker instead of clearing
        # the buffer, which cannot separate them
        read_until("gpudb> ", after="^C")
        del out[:]
        send(b"SELECT 2 AS after_ctrl_c;\n")
        boxed = read_until("│")
        seen = "".join(out)
        check(took and boxed and "after_ctrl_c" in seen and "thrown_away" not in seen,
              f"^C throws away the half-typed line, the next statement runs on its own "
              f"(^C answered={took}, box={boxed}, statement={'after_ctrl_c' in seen}, "
              f"old line gone={'thrown_away' not in seen}, saw {seen[-240:]!r})")

        # ^C flushes the terminal's queues, so the long statement has to be read
        # by the shell (prompt seen, line echoed) before the signal is sent
        read_until("gpudb> ", after="│")
        del out[:]
        send(b"SELECT count(*) FROM range(100000000000) r(i) WHERE i % 7 = 3;\n")
        read_until("\x00", timeout=2.0)                  # never matches: reads for 2 s

        send(b"\x03")                       # ^C with a statement running
        check(read_until("INTERRUPT", timeout=60),
              f"^C interrupts the running statement (saw {''.join(out)[-160:]!r})")
        # the terminal echoes what is typed at it, so the statement's own text
        # would match the echo and say nothing about the shell: wait for the
        # ANSWER, which only the shell can print
        del out[:]
        send(b"SELECT 3 AS still_here;\n")
        check(read_until("│", timeout=60) and "still_here" in "".join(out),
              "the shell is still there after the interrupt")

        # and then for the prompt that follows the answer. readline prints its
        # prompt the moment it takes the terminal, before it has read the line
        # already waiting in the terminal's queue, so an earlier `gpudb> ` does
        # not mean the shell is idle — and a ^D typed while a statement is
        # still running reaches a terminal that readline has put back in
        # canonical mode, where Linux records it as an end-of-file mark and
        # then drops that mark when readline switches the mode again.
        check(read_until("gpudb> ", after="│"), "the prompt is back")
        send(b"\x04")                       # Ctrl-D on an empty line
        read_until("\x00", timeout=30)     # never matches: reads until the shell closes the tty
        try:
            rc = p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()
            rc = None
        check(rc == 0, f"Ctrl-D leaves the shell (rc={rc}, tail={''.join(out)[-80:]!r})")
    finally:
        if p.poll() is None:
            p.kill()
        os.close(primary)


if __name__ == "__main__":
    sys.exit(run())


def test_shell():   # pytest entry
    assert run() == 0
