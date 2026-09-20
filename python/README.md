# gpudb — plain DuckDB SQL on the GPU

**New in v0.7 — plain DuckDB SQL runs on the GPU.** No `gpu_*` calls, no query
changes: write the SQL you already write, and the GPU answers it when that is
measured faster — DuckDB answers everything else, with the same rows either
way. Apple Silicon Metal and NVIDIA CUDA, from one extension.

Ordinary DuckDB SQL, unchanged: statements the GPU answers faster are answered
on the device, everything else runs on DuckDB exactly as before, and the rows,
names and types are identical either way.

## Install

```bash
pip install duckdb-gpudb        # the distribution is duckdb-gpudb; the import is gpudb
```

There are **two pieces**: this package is the wrapper — the `gpudb` command and
`gpudb.connect()` — and the GPU code itself is a DuckDB extension. On **Apple
Silicon (macOS 15 or later)** and **x86-64 Linux (glibc 2.34 or newer, e.g.
Ubuntu 22.04 and later)** the wheel carries both: the v0.7.0 extension binary
travels inside the package, so that one line is the whole install. No `INSTALL`,
no build, no environment variable. One binary, and it has been run under both
DuckDB 1.4.5 and 1.5.5.

Anywhere else `pip` installs the pure-Python wheel and the extension comes from
DuckDB's own install:

```sql
INSTALL gpudb FROM community;   -- in any DuckDB >= 1.5.5 client
LOAD gpudb;
```

The wrapper looks for the extension in this order: an explicit `extension=`
path, the `GPUDB_EXTENSION_PATH` environment variable, a `build-macos/` or
`build-linux/` directory next to a source checkout, the copy bundled in this
package, and finally the extension DuckDB itself has installed. A checkout's
own build comes before the bundled copy deliberately — someone who has just
built the extension is testing that binary. If it finds none — or finds one
older than this client — `con.extension_note` says so in one sentence and every
statement runs on DuckDB.

**Requires** Python >= 3.9 and the `duckdb` module >= 1.4. A bundled or
downloaded binary needs only DuckDB >= 1.2, because the loadable extension is
built against the stable C API v1.2.0; the registry builds gpudb separately for
each DuckDB version from 1.5.5 on. Apple silicon for the Metal backend, an
NVIDIA GPU for the CUDA one; plain SQL runs on the GPU by default on both, and
`GPUDB_CUDA_EXACT=0` turns the CUDA path off without a rebuild. On Linux the
extension needs `libgomp.so.1` at load time (`apt install libgomp1`), which the
wheel bundles and a registry install does not; a binary installed from the
community registry there reports `compiled=cpu` and carries no CUDA at all —
`SELECT gpu_build_info();` says which one you have.

## The `gpudb` shell

The first way in, and the one that shows its work. This is one session on an
M4 Max over TPC-H SF1, opened read-only, on the v0.7.0 release build of
2026-09-20 (the three `[gpudb] registered …` lines
are the extension announcing itself on stderr as DuckDB loads it):

```
$ gpudb data/tpch_sf1/tpch.duckdb --readonly
[gpudb] registered gpu_inner_join + gpu_join_rows_resident
[gpudb] registered gpu_groupby_{sum,sum_f64,count,exact}_resident[_having|_topk] + gpu_topk_resident[_f64]
[gpudb] registered gpu_sum / gpu_min / gpu_max streaming aggregates + resident-column functions (gpu_upload, gpu_*_resident) (backend=Metal)
gpudb 0.7.0
backend:       Metal · Apple M4 Max · 51.8 GiB device memory
transparent:   available — every statement goes through the wrapper
database:      data/tpch_sf1/tpch.duckdb
Enter .help for usage.

gpudb> SELECT l_partkey, sum(l_quantity) AS qty FROM lineitem GROUP BY l_partkey ORDER BY qty DESC LIMIT 5;
┌───────────┬───────────────┐
│ l_partkey │      qty      │
│   int64   │ decimal(38,2) │
├───────────┼───────────────┤
│    125009 │       1642.00 │
│    140633 │       1562.00 │
│     49981 │       1553.00 │
│    149443 │       1537.00 │
│     10426 │       1513.00 │
└───────────┴───────────────┘

DuckDB (not_resident: the resident set is not ready yet) · 25.7 ms
```

The first ask is on DuckDB on purpose: the columns are uploaded in short
row-id segments taken only while the connection is idle, so an upload never
runs a long scan beside a query you are waiting for. Ten seconds later, the
same statement, nine runs each way in the same session:

```
gpudb> .gpu off
GPU path off — statements go straight to DuckDB.
… DuckDB · 30.4, 24.5, 16.5, 19.5, 20.9, 22.3, 21.6, 22.1, 23.0 ms
gpudb> .gpu on
GPU path on — residency: background.
… GPU (topk: the resident GROUP BY) · 12.9, 47.3, 10.0, 11.4, 14.1, 14.4, 14.4, 14.5, 14.8 ms
```

Median 22.1 ms against 14.4 ms — 1.53×, with both series printed whole so the
warm-up runs and the wrapper's own measuring run stay visible. The nine GPU runs
spread from 10.0 to 14.8 ms with nothing changed between them: a statement this
short has more than one speed on Apple silicon depending on what else is waking,
which is why the wrapper measures in your process instead of trusting a
published ratio. Expect your own numbers rather than these.

The banner's `backend:` line names the runtime, the device as the driver
reports it, and the device memory the budget plans against; a build without a
GPU backend names no device, and `transparent:` says what is missing instead.

### Options and dot-commands

```bash
gpudb my.duckdb -c "SELECT …"     # one statement
gpudb -f script.sql               # a file (or: gpudb < script.sql)
python -m gpudb                   # the same entry point
```

| Option | |
|---|---|
| `-c SQL` | run a statement and exit, repeatable |
| `-f FILE`, `--file FILE` | run a file and exit, repeatable; `-c` and `-f` run in the order given |
| `--readonly`, `--read-only` | open read-only |
| `--no-gpu` | every statement on DuckDB |
| `--residency background\|eager\|manual` | when tables become resident (default `background`) |
| `--memory-budget SIZE` | e.g. `16GB`, or `unlimited` |
| `--timer` / `--no-timer` | force the footer line on or off |
| `--debug` | show tracebacks instead of one-line errors |
| `--version` | the gpudb and duckdb versions |
| `-h`, `--help` | the same list, from the program |

A statement that fails ends a `-c` / `-f` / piped run with a non-zero exit
code; at the terminal the session keeps going. The footer is on at a terminal
and off in scripted output unless `--timer` says otherwise; colour follows
`NO_COLOR` and whether the output is a terminal.

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

`.residents` prints two tables: the resident SETS — what a statement is waiting
on — and, under them, the COLUMNS those sets are views over. A lane is kept at
the narrowest signed width its values fit, so a column of small integers costs
one or two bytes a row rather than eight. Same session as above:

```
gpudb> .residents
table          columns               state  bytes     estimated  worth
main.lineitem  l_partkey,l_quantity  ready  80.1 MiB  161.7 MiB  3.99
1 set · 80.1 MiB held · worth is ms saved per second per GiB · `.memory` for the budget

table     column      dtype  rows       width  bytes     state
lineitem  l_partkey   I64    6,001,215  4 B    68.7 MiB  ready
lineitem  l_quantity  I64    6,001,215  2 B    11.4 MiB  preparing
2 resident columns · width is the bytes a row of the lane is stored at · `-` where the backend does not say
```

`worth` is the DuckDB time a set saves per second of wall time, per GiB it
holds — what the memory budget compares when it has to choose. On a column
line, `state` means something narrower: whether *that lane* has a sort cache.
Only a key lane ever needs one, so a payload lane reads `preparing` and stays
there; the set's own state is what a statement waits on. Every dot-command and
option, and the session these came from, are in
[the shell guide](https://github.com/singhpratech/duckdbgpumetaldbram/blob/main/docs/USING_THE_SHELL.md).

There is a shell because the transparent path cannot live in the extension.
DuckDB's stable C extension API — the one the loadable extension uses on
purpose, so that one binary keeps working across DuckDB versions — has no hook
that sees a statement before it is planned. `LOAD gpudb` in the stock `duckdb`
CLI therefore gives the explicit `gpu_*` functions and nothing more. This shell
is a client that can look first: it splits statements with DuckDB's own
tokenizer, hands each one to `gpudb.connect()`, and prints the result with
DuckDB's own box renderer.

## Python

```python
import gpudb
con = gpudb.connect("my.duckdb")          # same surface as duckdb.connect
con.execute("SELECT k, sum(v) FROM t GROUP BY k").fetchall()
con.last_rewrite()                        # what happened to that statement
```

`connect()` takes `database`, `read_only` and `config` as `duckdb.connect`
does, plus `extension=` (an explicit path to the `.duckdb_extension`),
`transparent=` (default `True`), `residency=` (`background` | `eager` |
`manual`), `floor_rows=` (default 1,000,000), `idle_ms=` (default 20.0),
`thresholds=` (default `True`; `False` rewrites every exact shape regardless of
the predicted win — for parity testing only), `memory_budget=` (bytes or
`"16GB"`; `0` / `"unlimited"` removes the cap) and `log=` (a callable that
receives the wrapper's decisions as text).

Anything the wrapper does not define itself is delegated to the underlying
`duckdb.DuckDBPyConnection`. What it does define:

| | |
|---|---|
| `con.last_rewrite()` | `rewritten`, `reason`, `detail`, `form`, `tag`, `sql`, `statement`, `engine`, `round_trip_ms`, `fallback`, `error` |
| `con.memory()` | `budget`, `evictions`, `evictions_wasted`, `sets` |
| `con.residents()` | `{identity tag: state}` — `missing`, `pending`, `uploading`, `ready`, `stale`, `failed` |
| `con.store_columns()` | one dict per resident column: `table`, `column`, `dtype`, `rows`, `width`, `bytes`, `prepared` |
| `con.transparent` | readable and settable; `False` leaves every statement on DuckDB without dropping anything |
| `con.residency` | read-only: the mode passed to `connect()` |
| `con.cursor()` / `con.duplicate()` | another connection sharing the resident sets, with its own `last_rewrite()` |
| `con.interrupt()` | DuckDB's interrupt, on this connection's handle only |
| `con.extension_note` | empty while the loaded extension can serve the client; one sentence saying why not otherwise |

Give each thread its own `con.cursor()`: `last_rewrite()` is one record per
connection object, so two threads sharing one connection overwrite each other's.

## Why a statement did not go to the GPU

The footer and `last_rewrite()["reason"]` carry a short code; `detail` is the
sentence behind it.

| Reason | What it means |
|---|---|
| `not_resident` | the columns are not on the device yet — the background uploader is working on it |
| `threshold` | a measured bound says DuckDB is faster for this shape and size, **or** this machine measured it slower and the template went back to DuckDB |
| `shape` | not a shape the rewrite expresses: a window function, `median`, `ROLLUP`, a set operation, a subquery in the select list |
| `double` | a `sum` / `avg` over `DOUBLE` or `FLOAT` — never rewritten, because native's own answer depends on the order the values are added |
| `ties` | a pushed `ORDER BY … LIMIT k` found two of the first *k* rows equal on the ordering value — which rows come back, and in what order, is DuckDB's to choose, and it answered the original. Decided against the data on every execution; a tie that keeps happening shows as `threshold` with the tie named in `detail`, until the 60-second re-measure. Through `sql()` — and so through the shell — the check runs on a side cursor inside the call, so the first statement of a data version asks the device once more and the verdict is remembered until the data changes |
| `backend` | this build has no GPU backend to rewrite for, or the installed extension is older than this client |
| `memory` | the set does not fit the device-memory budget; it is refused before the upload |
| `transaction` | a `BEGIN` is open, so the resident sets cannot be trusted |
| `params` | the statement takes prepared-statement parameters |
| `multi` | more than one statement in the call |
| `too_long` | the statement is longer than the 16 KB the wrapper parses |
| `error` | the rewritten statement raised and DuckDB answered the original — the text is in `error` |
| `off` / `manual` | the path is off (`.gpu off`, `--no-gpu`, `transparent=False`), or residency is `manual` and this set was not uploaded by hand |
| `nulls` `overflow` `decimal` `collation` `view` `temp` `ambiguous` `not_found` | a narrower refusal, each with its own sentence in `detail` |

What those look like in the shell's footer — these are the *shapes*, collected
from different statements and different sessions, not one run:

```
GPU (plain: the resident GROUP BY) · 3.3 ms
GPU (topk: a key join materialised on the device) · 8.1 ms
DuckDB (threshold: 7 groups < 1000) · 5.0 ms
DuckDB (threshold: measured 4.20 ms rewritten vs 3.10 ms native (re-measured in 60 s)) · 3.2 ms
DuckDB (not_resident: the resident set is not ready yet) · 12.0 ms
DuckDB (ties: two of the first 5 rows tie on qty, so which rows come back — and in what order — is DuckDB's to choose, and DuckDB answered the original) · 49.9 ms
DuckDB (off: the transparent path is off on this connection) · 12.0 ms
```

## The third way in: explicit `gpu_*` functions

The shell and `gpudb.connect()` are two of three routes. The third needs no
wrapper at all: `INSTALL gpudb FROM community; LOAD gpudb;` in **any** DuckDB
client, in any language, and then call the functions by name —
`gpu_upload` / `gpu_upload_pair` to make a column resident once, then
`gpu_sum_resident`, `gpu_groupby_sum_resident` (with `_having` and `_topk`
forms that filter on the device), `gpu_topk_resident`, `gpu_inner_join` and the
exact `gpu_groupby_exact_*` family to read it back. `gpu_last_stats()` says
which processor ran and for how long. v0.7 registers 65 of these; v0.6.0
registered 38, and every one of those is still there unchanged.

It is full manual control, and it is the only route from the stock `duckdb`
CLI. The repository's README documents the whole surface.

## Going deeper

- [The `gpudb` shell, end to end](https://github.com/singhpratech/duckdbgpumetaldbram/blob/main/docs/USING_THE_SHELL.md)
- [`gpudb.connect()`, every option](https://github.com/singhpratech/duckdbgpumetaldbram/blob/main/docs/USING_PYTHON.md)
- [Installing both pieces, platforms, troubleshooting](https://github.com/singhpratech/duckdbgpumetaldbram/blob/main/docs/INSTALL.md)
- [How a plain `SELECT` reaches the GPU](https://github.com/singhpratech/duckdbgpumetaldbram/blob/main/docs/TRANSPARENT_DESIGN.md) ·
  [every environment variable](https://github.com/singhpratech/duckdbgpumetaldbram/blob/main/docs/ENVIRONMENT.md)
- [Every measurement, losing cells included](https://github.com/singhpratech/duckdbgpumetaldbram/blob/main/BENCHMARK.md) ·
  [every documented trade-off](https://github.com/singhpratech/duckdbgpumetaldbram/blob/main/KNOWN_ISSUES.md)
- [The repository](https://github.com/singhpratech/duckdbgpumetaldbram)
