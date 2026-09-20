# gpudb — plain DuckDB SQL on the GPU

```python
import gpudb
con = gpudb.connect("my.duckdb")          # same surface as duckdb.connect
con.execute("SELECT k, sum(v) FROM t GROUP BY k").fetchall()
```

Ordinary DuckDB SQL, unchanged: statements the GPU answers faster are answered
on the device, everything else runs on DuckDB exactly as before, and the rows,
names and types are identical either way. `con.last_rewrite()` says what
happened to the last statement.

The wrapper needs the `gpudb` DuckDB extension. It looks, in order, for an
explicit `extension=` path, the `GPUDB_EXTENSION_PATH` environment variable, a
local build next to a source checkout, a copy bundled in this package, and
finally the extension installed in DuckDB itself (`INSTALL gpudb FROM
community; LOAD gpudb;`). The wheels built for a supported platform carry that
bundled copy; the `py3-none-any` wheel does not, and there the extension has to
come from DuckDB's own install.

Settings: `residency` (`background` | `eager` | `manual`), `floor_rows`,
`memory_budget` (e.g. `"16GB"`), `thresholds`. Design, limits and measurements:
`docs/TRANSPARENT_DESIGN.md`, `KNOWN_ISSUES.md` and `BENCHMARK.md` in the repository.

## The `gpudb` shell

```
$ gpudb my.duckdb
gpudb 0.7.0.dev0
backend:      Metal · Apple M4 Max · 51.8 GiB device memory
transparent:  available — every statement goes through the wrapper
database:     my.duckdb
Enter .help for usage.

gpudb> SELECT k, sum(v) FROM t GROUP BY k ORDER BY k LIMIT 3;
┌───────┬───────────┐
│   k   │  sum(v)   │
│ int64 │  int128   │
├───────┼───────────┤
│     0 │ 398000000 │
│     1 │ 398000200 │
│     2 │ 398000400 │
└───────┴───────────┘

GPU (plain: the resident GROUP BY) · 3.3 ms
```

The same shell runs a script or a single statement:
`gpudb my.duckdb -c "SELECT …"`, `gpudb -f script.sql`, `gpudb < script.sql`,
`python -m gpudb`. Options: `--readonly`, `--no-gpu`, `--residency`,
`--memory-budget`, `--timer` / `--no-timer`, `--version`, `--help`. A statement
that fails ends a `-c` / `-f` / piped run with a non-zero exit code; at the
terminal the session keeps going.

There is a shell because the transparent path cannot live in the extension.
DuckDB's stable C extension API — the one the loadable extension uses on
purpose, so that one binary keeps working across DuckDB versions — has no hook
that sees a statement before it is planned. `LOAD gpudb` in the stock `duckdb`
CLI therefore gives the explicit `gpu_*` functions and nothing more; ordinary
SQL is answered on the device only through a client that can look at the
statement first. This shell is that client: it splits statements with DuckDB's
own tokenizer, hands each one to `gpudb.connect()`, and prints the result with
DuckDB's own box renderer.

The banner's `backend:` line names the runtime, the device as the driver
reports it, and the device memory the budget plans against; a build without a
GPU backend names no device.

The dim line under each result says where the statement ran, why, and how long
it took — all of it from `con.last_rewrite()`, which carries both a short
`reason` code and a `detail` sentence explaining it:

```
GPU (plain: the resident GROUP BY) · 3.3 ms
GPU (topk: a key join materialised on the device) · 8.1 ms
DuckDB (threshold: 7 groups < 1000) · 5.0 ms
DuckDB (threshold: measured 4.20 ms rewritten vs 3.10 ms native (re-measured in 60 s)) · 3.2 ms
DuckDB (not_resident: the resident set is not ready yet) · 12.0 ms
DuckDB (off: the transparent path is off on this connection) · 12.0 ms
```

The footer is on at a terminal, off in `-c` / `-f` / piped output unless
`--timer` is given, and colour follows `NO_COLOR` and whether the output is a
terminal. `.gpu` prints the whole record, `detail` included.

`.residents` prints two tables: the wrapper's resident SETS — what a statement
is waiting on — and, under them, the COLUMNS those sets are views over, with
the rows each one holds and the width it is stored at. A lane is kept at the
narrowest signed width its values fit, so a column of small integers costs one
or two bytes a row rather than eight, and the table says which:

```
table  column  dtype  rows       width  bytes     state
t      k       I64    2,000,000  2 B    15.3 MiB  ready
t      v       I64    2,000,000  4 B    7.6 MiB   ready
```

Dot-commands, kept small on purpose — this is not an emulation of the DuckDB
CLI:

| | |
|---|---|
| `.help` | the list |
| `.quit`, `.exit` | leave (Ctrl-D does too) |
| `.timer on\|off` | the footer line |
| `.gpu` | the last statement's whole `last_rewrite()`, `detail` included |
| `.gpu on\|off` | the transparent path, live |
| `.residents` | the resident sets, then the columns behind them with rows and width |
| `.memory` | the device-memory budget and what holds it |
| `.read FILE`, `.open [DATABASE]` | run a file, open another database |
| `.tables`, `.schema [TABLE]` | plain SQL underneath (`SHOW TABLES`, `DESCRIBE`) |
| `.version` | gpudb and duckdb versions |

Statements may span lines and end at `;`; Ctrl-C stops the running statement
(or clears what you were typing) and Ctrl-D leaves. History is kept in
`~/.gpudb_history` when the Python build has `readline`.
