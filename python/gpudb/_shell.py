"""The `gpudb` shell — a terminal client for the transparent path.

    $ gpudb my.duckdb
    gpudb> SELECT l_returnflag, sum(l_quantity) FROM lineitem GROUP BY 1;

DuckDB's stable C extension API — the one the loadable extension uses on
purpose, so that one binary keeps working across DuckDB versions — has no hook
that sees a statement before it is planned. So `LOAD gpudb` in the stock
`duckdb` CLI can only add the explicit `gpu_*` functions; the transparent
path (ordinary SQL answered on the device when that is faster, by DuckDB
otherwise, identical rows either way) lives in a client wrapper instead
(docs/TRANSPARENT_DESIGN.md §6). A terminal user therefore needs a client,
and this is it: every statement goes through `gpudb.connect()`.

Nothing here parses SQL. Statements are split by DuckDB's own tokenizer and
`extract_statements`, results are printed by DuckDB's own box renderer, and
where a statement ran is read from the wrapper's `last_rewrite()`.
"""
import argparse
import os
import re
import shlex
import signal
import sys
import threading
import time
from typing import List, Optional, Tuple

import duckdb

from . import __version__, connect

PROMPT = "gpudb> "
CONTINUE = "  ...> "          # the `>` sits under the `>` of the prompt above it
MAX_ROWS = 40                 # what the DuckDB CLI shows of a long result
HISTORY = "~/.gpudb_history"
HISTORY_LEN = 2000
MAX_READ_DEPTH = 10           # `.read` of a file that reads itself
LABEL = 14                    # width of the `key:` column in the banner, `.gpu` and `.memory`
DIM = "2"
# a calm teal for the line that says the GPU answered; staying on DuckDB is
# normal, not a warning, so its line is only dim
TEAL_256, TEAL_8 = "38;5;36", "36"

HELP = """\
.help                 this list
.quit, .exit          leave the shell (Ctrl-D does too)
.timer on|off         the footer line after each statement
.gpu                  where the last statement ran, in full
.gpu on|off           the transparent path (off: every statement on DuckDB)
.residents            the resident sets and their state
.memory               the device-memory budget and what holds it
.read FILE            run the statements in FILE
.open [DATABASE]      open another database (no argument: in-memory)
.tables               the tables in the database
.schema [TABLE]       TABLE's columns, or every table's DDL
.version              gpudb and duckdb versions

Anything else is SQL: statements may span lines and end at `;`."""


def _accent() -> str:
    """The 256-colour teal where the terminal says it has 256 colours, the
    basic cyan where it does not."""
    return TEAL_256 if "256color" in (os.environ.get("TERM") or "") else TEAL_8


def _sep(stream) -> str:
    """`·` where the output can carry it, `-` where it cannot."""
    enc = getattr(stream, "encoding", None) or "ascii"
    try:
        "·".encode(enc)
    except (LookupError, UnicodeEncodeError):
        return "-"
    return "·"


def _terminators(text: str) -> List[int]:
    """Offsets of the `;` that end a statement — DuckDB's tokenizer decides, so
    a semicolon inside a string, a comment or a dollar-quoted body is not one
    (those are one token, or no token at all)."""
    try:
        tokens = duckdb.tokenize(text)
    except Exception:
        # an input the tokenizer cannot even scan: let the parser report it
        return [i for i, ch in enumerate(text) if ch == ";"]
    return [pos for pos, kind in tokens
            if kind == duckdb.token_type.operator and text[pos:pos + 1] == ";"]


def _split_complete(buf: str) -> Tuple[str, str]:
    """(what can run now, what is still being typed). The tail after the last
    terminator is left in the buffer unless it is only whitespace and
    comments — which the tokenizer reports by finding no token in it."""
    ends = _terminators(buf)
    if not ends:
        return "", buf
    tail = buf[ends[-1] + 1:]
    try:
        blank = not duckdb.tokenize(tail)
    except Exception:
        blank = not tail.strip()
    return (buf, "") if blank else (buf[:ends[-1] + 1], tail)



_INTERRUPTED = "INTERRUPT: the statement was interrupted"


class _Interrupted(Exception):
    """The shell interrupted the running statement and DuckDB raised nothing."""

class Shell:
    """One connection, one input buffer. `feed()` drives it from any source:
    the terminal, a file, `-c`, piped stdin."""

    def __init__(self, database: str = ":memory:", *, read_only: bool = False,
                 transparent: bool = True, residency: str = "background",
                 memory_budget=None, timer: bool = False, interactive: bool = False,
                 debug: bool = False, out=None, err=None):
        self.database = database
        self.opts = dict(read_only=read_only, transparent=transparent,
                         residency=residency, memory_budget=memory_budget)
        self.timer = timer
        self.interactive = interactive
        self.debug = debug
        self.out = out or sys.stdout
        self.err = err or sys.stderr
        self.exit_code = 0
        self.stop_on_error = not interactive
        self.buf = ""
        self.done = False
        self._depth = 0
        self._color = (self.out.isatty() if hasattr(self.out, "isatty") else False) \
            and not os.environ.get("NO_COLOR")
        self._accent = _accent()
        self._sep = _sep(self.out)
        # the wrapper's own log, kept only for the statement being run: it is
        # where a decline says how many groups or how few rows it was about,
        # which `last_rewrite()` does not carry
        self._log: List[str] = []
        self.con = connect(database, log=self._log.append, **self.opts)
        self.info = self._build_info()       # read once: asking again would overwrite last_rewrite()

    # ---- output ----
    def say(self, text: str = "") -> None:
        print(text, file=self.out)

    def paint(self, text: str, code: str = DIM) -> str:
        return f"\x1b[{code}m{text}\x1b[0m" if self._color else text

    def field(self, key: str, value: str) -> None:
        """A dim `key:` label with a plain value, the banner's and `.memory`'s shape."""
        self.say(self.paint(f"{key + ':':<{LABEL}}") + value)

    def fail(self, text: str) -> None:
        """A statement's error: DuckDB's own message, on stderr, never a
        traceback. It decides the exit code of a `-c` / `-f` / piped run; a
        session at a terminal keeps going and still leaves with 0."""
        print(text, file=self.err)
        self.err.flush()
        if not self.interactive:
            self.exit_code = 1

    def banner(self) -> None:
        self.say(f"gpudb {__version__}")
        self.field("backend", self._backend())
        self.field("transparent", self._path())
        self.field("database", self.database)
        self.say(self.paint("Enter .help for usage."))
        self.say()

    def _backend(self) -> str:
        """What the extension says it is running on, plus the device memory it
        reports (`gpu_build_info()` names no device, so neither does this)."""
        info = self.info
        if info is None:
            return "none — the extension is not loaded"
        m = re.search(r"runtime=(\w+)", info)
        name = {"metal": "Metal", "cuda": "CUDA", "cpu": "CPU"}.get(
            (m.group(1) if m else "").lower(), "unknown")
        mem = re.search(r"device_memory=(\d+)", info)
        size = int(mem.group(1)) if mem else 0
        return f"{name} {self._sep} {self._bytes(size)} device memory" if size else name

    def _path(self) -> str:
        info = self.info
        if not self.con.transparent:
            return "off — every statement goes straight to DuckDB"
        if info is None:
            return "a plain DuckDB shell — every statement goes straight to DuckDB"
        if "exact=true" not in info:
            return "not on this build — every statement goes straight to DuckDB"
        return "available — every statement goes through the wrapper"

    def _build_info(self) -> Optional[str]:
        try:
            return self.con.sql("SELECT gpu_build_info()").fetchone()[0]
        except Exception:
            return None

    # ---- input ----
    def feed(self, text: str, flush: bool = True) -> None:
        """Run everything in `text`. `flush` runs a trailing statement that has
        no `;`, the way `duckdb -c` and `duckdb -f` do."""
        for line in text.splitlines():
            self.line(line)
            if self.done:
                return
        if flush and self.buf.strip():
            stmts, self.buf = self.buf, ""
            self.run_sql(stmts)

    def line(self, line: str) -> None:
        """One line of input: a dot-command when nothing is half-typed, SQL
        otherwise. Runs whatever the buffer now completes."""
        if not self.buf.strip() and line.strip().startswith("."):
            self.buf = ""
            self.dot(line.strip())
            return
        self.buf += line + "\n"
        ready, self.buf = _split_complete(self.buf)
        if ready.strip():
            self.run_sql(ready)

    def repl(self) -> None:
        self._history()
        while not self.done:
            try:
                line = input(CONTINUE if self.buf.strip() else PROMPT)
            except EOFError:
                self.say()
                return
            except KeyboardInterrupt:
                # ^C at the prompt throws away what was being typed, like the DuckDB CLI
                self.buf = ""
                self.say("^C")
                continue
            self.line(line)

    def _history(self) -> None:
        """readline where the build has it; a history file that cannot be read
        or written is never worth an error."""
        try:
            import readline
        except ImportError:
            return
        path = os.path.expanduser(HISTORY)
        try:
            readline.read_history_file(path)
        except Exception:
            pass
        try:
            readline.set_history_length(HISTORY_LEN)
        except Exception:
            pass
        import atexit

        def save():
            try:
                readline.write_history_file(path)
            except Exception:
                pass
        atexit.register(save)

    # ---- statements ----
    def run_sql(self, text: str) -> None:
        try:
            stmts = duckdb.extract_statements(text)
        except duckdb.Error as e:
            self.fail(str(e))
            return
        for s in stmts:
            self.statement(s.query)
            if self.exit_code and self.stop_on_error:
                self.done = True
                return
            if self.done:
                return

    def statement(self, sql: str) -> None:
        sql = sql.strip()
        if not sql:
            return
        t0 = time.perf_counter()
        note = ""
        del self._log[:]          # the detail in the footer is this statement's, or none
        try:
            self.interruptible(lambda: self.render(self.con.sql(sql)))
        except _Interrupted:
            self.fail(_INTERRUPTED)
            return
        except duckdb.Error as e:
            note = self.recover(sql, e)
            if note is None:
                return
        except KeyboardInterrupt:
            self.say("^C")
            return
        except Exception as e:                       # noqa: BLE001 - a shell never shows a traceback
            if self.debug:
                raise
            self.fail(f"gpudb: {type(e).__name__}: {e}")
            return
        if self.timer:
            self.footer((time.perf_counter() - t0) * 1000.0, note)

    def recover(self, sql: str, e: duckdb.Error) -> Optional[str]:
        """The statement raised. When it was the REWRITTEN form that raised —
        the rows were being read, so the wrapper's own `sql()` was already past
        its guard — the user's statement is still the original one (rule 2,
        docs/TRANSPARENT_DESIGN.md §5.4). Hand it to the wrapper's `execute()`,
        which re-runs the original on DuckDB and drops the sets behind the
        failure, then render the healed statement. Returns a note for the
        footer, or None when the error was the user's own."""
        last = self.con.last_rewrite()
        if not last.get("rewritten") or "INTERRUPT" in str(e).upper():
            self.fail(str(e))
            return None
        why = str(e).splitlines()[0]
        try:
            self.interruptible(lambda: self.con.execute(sql))
            self.interruptible(lambda: self.render(self.con.sql(sql)))
        except _Interrupted:
            self.fail(_INTERRUPTED)
            return None
        except duckdb.Error as e2:
            self.fail(str(e2))
            return None
        except KeyboardInterrupt:
            self.say("^C")
            return None
        return f"the rewritten statement failed: {why}"

    def render(self, rel) -> None:
        """DuckDB's own box renderer. A statement with no result set (CREATE,
        INSERT, SET, …) gives no relation and prints nothing."""
        if rel is not None:
            rel.show(max_rows=MAX_ROWS)

    def interruptible(self, fn):
        """^C stops the statement, not the shell. A signal handler cannot run
        while the main thread sits in DuckDB, so at a terminal the statement
        runs on a worker and the main thread stays free to take the signal and
        call `interrupt()`. Non-interactive input keeps the default handling."""
        if not self.interactive:
            return fn()
        box = {}

        def body():
            try:
                box["value"] = fn()
            except BaseException as exc:             # noqa: BLE001 - re-raised below
                box["error"] = exc
        worker = threading.Thread(target=body, daemon=True)
        worker.start()
        interrupted = False
        while True:
            # the ^C may land anywhere in this wait, including between two
            # join() calls, and the worker is left running whatever happens —
            # so every interrupt is answered and then waited out
            try:
                while worker.is_alive():
                    worker.join(0.05)
                break
            except KeyboardInterrupt:
                interrupted = True
                self.con.interrupt()
        if "error" in box:
            raise box["error"]
        if interrupted:
            # DuckDB does not always raise for an interrupted statement (the client on some
            # platforms returns quietly); the shell sent the interrupt, so it says so itself
            raise _Interrupted()
        return box.get("value")

    def footer(self, ms: float, note: str = "") -> None:
        """One line: where the statement ran and how long it took. Where it ran
        is the wrapper's answer, never the shell's guess — `last_rewrite()` for
        the verdict, the wrapper's own log line for the `reason:` clause."""
        last = self.con.last_rewrite()
        if last.get("rewritten"):
            form = last.get("form") or ""
            inside = form
            colour = self._accent
        else:
            inside = last.get("reason") or ""
            detail = note or self._detail(inside)
            if detail:
                inside = f"{inside}: {detail}" if inside else detail
            colour = DIM
        where = ("GPU" if last.get("rewritten") else "DuckDB") + (f" ({inside})" if inside else "")
        self.say(self.paint(f"{where} {self._sep} {ms:.1f} ms", colour))

    def _detail(self, reason: str) -> str:
        """The clause after the reason. `last_rewrite()` carries no detail
        field, so it comes from what the wrapper logged while deciding THIS
        statement (a decision it took earlier and cached logs nothing, and then
        the footer simply names the reason), plus the two the wrapper's own
        tables answer."""
        if reason == "error" or self.con.last_rewrite().get("fallback"):
            err = self.con.last_rewrite().get("error") or ""
            if err:
                return "the rewritten statement failed: " + err.splitlines()[0]
        for line in self._log:
            if line.startswith(reason + ":"):                 # 'threshold: 7 groups < 1000'
                return line.split(":", 1)[1].strip()
            if line.startswith(f"declined ({reason}") and "):" in line:
                return line.split("):", 1)[1].strip()
        if reason == "not_resident":
            waiting = sum(1 for s in self.con.residents().values() if s != "ready")
            if waiting:
                return f"uploading {waiting} set{'s' if waiting > 1 else ''}"
        return ""

    # ---- dot-commands ----
    def dot(self, line: str) -> None:
        try:
            parts = shlex.split(line)
        except ValueError:
            parts = line.split()
        if not parts:
            return
        cmd, args = parts[0], parts[1:]
        if cmd in (".quit", ".exit"):
            self.done = True
        elif cmd == ".help":
            self.say(HELP)
        elif cmd == ".version":
            self.say(f"gpudb {__version__} (duckdb {duckdb.__version__})")
        elif cmd == ".timer":
            self._switch(args, "timer")
        elif cmd == ".gpu":
            if args:
                self._switch(args, "gpu")
            else:
                self.gpu()
        elif cmd == ".residents":
            self.residents()
        elif cmd == ".memory":
            self.memory()
        elif cmd == ".read":
            self.read(args)
        elif cmd == ".open":
            self.open(args[0] if args else ":memory:")
        elif cmd == ".tables":
            self.statement("SHOW TABLES")
        elif cmd == ".schema":
            self.statement(f"DESCRIBE {args[0]}" if args else
                           "SELECT table_name, sql FROM duckdb_tables() ORDER BY table_name")
        else:
            self.fail(f"gpudb: unknown command {cmd} — `.help` lists them")

    def _switch(self, args: List[str], which: str) -> None:
        if len(args) != 1 or args[0] not in ("on", "off"):
            self.fail(f"gpudb: .{which} takes `on` or `off` — `.help` lists them")
            return
        on = args[0] == "on"
        if which == "timer":
            self.timer = on
            self.say(self.paint(f"Timer {'on' if on else 'off'}."))
        else:
            self.con.transparent = on
            self.say(self.paint(f"GPU path on — residency: {self.con.residency}." if on else
                                "GPU path off — statements go straight to DuckDB."))

    def gpu(self) -> None:
        """The last statement's whole `last_rewrite()`, the fields it filled in."""
        last = self.con.last_rewrite()
        if not last.get("statement"):
            self.say(self.paint("No statement has run yet."))
            return
        for key, value in last.items():
            # a field this statement did not fill in; `rewritten` always shows,
            # False is the whole answer there
            if value == "" or value is None or (isinstance(value, float) and not value):
                continue
            if value is False and key != "rewritten":
                continue
            self.say(self.paint(f"{key + ':':<{LABEL}}") + str(value))

    def residents(self) -> None:
        """One line per resident set, from the wrapper's own table. The set's
        table and columns are read out of its identity tag
        (`gpudb:v1:<catalog>:<schema>:<table>:<oid>:<columns>[:extra]`)."""
        mem = self.con.memory()
        sets = mem.get("sets") or {}
        if not sets:
            self.say(self.paint("Nothing is resident yet."))
            return
        rows = []
        for tag, s in sets.items():
            parts = tag.split(":")
            table = ".".join(parts[3:5]) if len(parts) > 6 and parts[0] == "gpudb" else tag
            cols = parts[6] if len(parts) > 6 and parts[0] == "gpudb" else ""
            rows.append((table, cols, s.get("state", ""), self._bytes(s.get("bytes") or 0),
                         self._bytes(s.get("est_bytes") or 0), s.get("error") or ""))
        head = ("table", "columns", "state", "bytes", "estimated")
        width = [max(len(str(r[i])) for r in rows + [head]) for i in range(5)]
        self.say(self.paint("  ".join(h.ljust(width[i]) for i, h in enumerate(head))))
        for r in rows:
            self.say("  ".join(str(r[i]).ljust(width[i]) for i in range(5))
                     + (("  " + self.paint(r[5])) if r[5] else ""))
        held = sum((s.get("bytes") or 0) for s in sets.values())
        self.say(self.paint(f"{len(sets)} set{'s' if len(sets) > 1 else ''} {self._sep} "
                            f"{self._bytes(held)} held {self._sep} `.memory` for the budget"))

    def memory(self) -> None:
        mem = self.con.memory()
        sets = mem.get("sets") or {}
        budget = mem.get("budget") or 0
        held = sum((s.get("bytes") or 0) for s in sets.values())
        self.field("backend", self._backend())
        self.field("resident", f"{self._bytes(held)} in {len(sets)} set{'' if len(sets) == 1 else 's'}")
        self.field("budget", "unlimited" if not budget else self._bytes(budget))
        self.field("residency", self.con.residency)
        if mem.get("evictions"):
            self.field("evictions", str(mem["evictions"]))

    @staticmethod
    def _bytes(n: int) -> str:
        for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
            if n < 1024 or unit == "TiB":
                return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
            n /= 1024.0
        return str(n)

    def read(self, args: List[str]) -> None:
        if len(args) != 1:
            self.fail("gpudb: .read takes one file name")
            return
        if self._depth >= MAX_READ_DEPTH:
            self.fail("gpudb: .read nested too deeply")
            return
        try:
            with open(os.path.expanduser(args[0]), "r") as fh:
                text = fh.read()
        except OSError as e:
            self.fail(f"gpudb: {e}")
            return
        self._depth += 1
        outer, self.buf = self.buf, ""
        try:
            self.feed(text)
        finally:
            self._depth -= 1
            self.buf = outer

    def open(self, database: str) -> None:
        """Another database on a fresh connection; the settings stay."""
        try:
            con = connect(database, log=self._log.append, **self.opts)
        except duckdb.Error as e:
            self.fail(str(e))
            return
        try:
            self.con.close()
        except Exception:
            pass
        self.con, self.database = con, database
        self.info = self._build_info()
        if self.interactive:
            self.say(f"database: {database}")

    def close(self) -> None:
        try:
            self.con.close()
        except Exception:
            pass


class _Ordered(argparse.Action):
    """`-c` and `-f` kept in the order they were written on the command line."""

    def __call__(self, parser, namespace, value, option_string=None):
        kind = "file" if option_string in ("-f", "--file") else "sql"
        namespace.script = (namespace.script or []) + [(kind, value)]


def _parser():
    p = argparse.ArgumentParser(
        prog="gpudb", add_help=True,
        description="A SQL shell that answers plain DuckDB SQL on the GPU when that is "
                    "faster and on DuckDB otherwise, with identical results.")
    p.add_argument("database", nargs="?", default=":memory:",
                   help="database file (default: an in-memory database)")
    p.add_argument("-c", metavar="SQL", action=_Ordered, dest="script",
                   help="run SQL and exit; may be given more than once")
    p.add_argument("-f", "--file", metavar="FILE", action=_Ordered, dest="script",
                   help="run the statements in FILE and exit")
    p.add_argument("--readonly", "--read-only", action="store_true",
                   help="open the database read-only")
    p.add_argument("--no-gpu", action="store_true",
                   help="leave every statement on DuckDB (the transparent path off)")
    p.add_argument("--residency", choices=("background", "eager", "manual"),
                   default="background", help="when columns are uploaded (default: background)")
    p.add_argument("--memory-budget", metavar="SIZE", default=None,
                   help="device memory the resident sets may use, e.g. 16GB")
    p.add_argument("--timer", dest="timer", action="store_true", default=None,
                   help="print the footer line after each statement")
    p.add_argument("--no-timer", dest="timer", action="store_false",
                   help="never print the footer line")
    p.add_argument("--debug", action="store_true",
                   help="show the Python traceback of an internal error")
    p.add_argument("--version", action="version",
                   version=f"gpudb {__version__} (duckdb {duckdb.__version__})")
    p.set_defaults(script=None)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        piped = not sys.stdin.isatty()
    except Exception:
        piped = True
    interactive = not args.script and not piped
    try:
        signal.signal(signal.SIGINT, signal.default_int_handler)
    except ValueError:
        pass                                     # not the main thread; ^C is the caller's

    try:
        shell = Shell(args.database, read_only=args.readonly,
                      transparent=not args.no_gpu, residency=args.residency,
                      memory_budget=args.memory_budget,
                      timer=interactive if args.timer is None else args.timer,
                      interactive=interactive, debug=args.debug)
    except duckdb.Error as e:
        print(str(e), file=sys.stderr)
        return 1
    try:
        if args.script:
            for kind, value in args.script:
                if kind == "sql":
                    shell.feed(value)
                else:
                    shell.read([value])
                if shell.done or shell.exit_code:
                    break
        elif piped:
            shell.feed(sys.stdin.read())
        else:
            shell.banner()
            shell.repl()
    finally:
        shell.close()
    return shell.exit_code
