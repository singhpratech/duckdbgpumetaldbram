# gpudb — GPU-accelerated DuckDB on **Apple Silicon Metal + NVIDIA CUDA**

[![DuckDB Community Extension](https://img.shields.io/badge/DuckDB_Community_Extension-gpudb-FFF100?logo=duckdb&logoColor=black)](https://duckdb.org/community_extensions/extensions/gpudb)
[![Latest release](https://img.shields.io/github/v/release/singhpratech/duckdbgpumetaldbram?label=release)](https://github.com/singhpratech/duckdbgpumetaldbram/releases/latest)
[![CI](https://github.com/singhpratech/duckdbgpumetaldbram/actions/workflows/ci.yml/badge.svg)](https://github.com/singhpratech/duckdbgpumetaldbram/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Platforms](https://img.shields.io/badge/platforms-Apple_Silicon_Metal_%7C_Linux_CUDA-8A2BE2)](#quick-start)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/singhpratech/duckdbgpumetaldbram/blob/main/examples/gpudb_quickstart.ipynb)
[![GitHub stars](https://img.shields.io/github/stars/singhpratech/duckdbgpumetaldbram?style=social)](https://github.com/singhpratech/duckdbgpumetaldbram/stargazers)

> **The first SQL execution engine for Apple Silicon GPUs**, built as a DuckDB extension that *also* runs on NVIDIA CUDA. One codebase, two backends, your existing DuckDB queries.

Now an official [**DuckDB Community Extension**](https://duckdb.org/community_extensions/extensions/gpudb) — install it straight from any DuckDB ≥ 1.5.5, no flags, no downloads:

```sql
INSTALL gpudb FROM community;
LOAD gpudb;
SELECT gpu_sum(value::BIGINT) FROM range(1000000) AS t(value);
```

```sql
-- upload a column to GPU memory once, then every query runs at silicon speed
SELECT gpu_upload('qty', l_quantity::BIGINT) FROM lineitem;   -- once
SELECT gpu_sum_resident('qty');                               -- 600M rows: 10 ms vs 99 ms native
SELECT gpu_last_stats();                                      -- proof: which processor ran, kernel time
-- op=resident_i64 backend=Metal reason=Hot_GpuAlwaysWins rows=600037902
--   wall_ms=9.702 kernel_ms=9.533 transfer_ms=0.000
```

Apache-2.0 · v0.7.0 · Linux + macOS · DuckDB ≥ 1.5.5 from the community registry
(release binaries need only the v1.2.0 C API)

---

## v0.7 — plain DuckDB SQL on the GPU

Until now you called `gpu_*` functions by name. From v0.7 you write the SQL you
already write — in the `gpudb` shell, or through `gpudb.connect()` — and the GPU
answers it when that has been **measured** faster on your machine. Everything
else runs on DuckDB, untouched, at its usual speed. Same rows, same column
names, same column types, either way. No hints, no schema changes, nothing to
call.

On TPC-H at SF10 that is **19 of 22 queries on the device, 0 rows differing, up
to 52.9× on an M4 Max** <!-- RE-RUN -->; 17 of 22 at SF1. The three that stay
are declined on purpose, and this README says which and why.

**Nothing you use today goes away.** `gpu_upload`, `gpu_upload_pair`, the
`gpu_*_resident` scalars, the fused joins, the resident GROUP BY and top-k table
functions all work in v0.7 exactly as they did in v0.5 / v0.6, from any DuckDB
client, with the same names and the same results.

### Three ways in

| | What it is for |
|---|---|
| **The `gpudb` shell** | exploring, ad-hoc SQL, and seeing where every statement ran and why |
| **Python — `gpudb.connect()`** | applications, notebooks, pipelines |
| **Explicit `gpu_*` functions** | any client in any language, full manual control; the only route that works from the stock DuckDB CLI |

All three talk to the same extension and the same resident columns.

### The shell way

#### Install it

```bash
pip install duckdb-gpudb          # installs the `gpudb` command
```

From a checkout instead (what this repository's own runs use):

```bash
./scripts/build.sh                # builds the extension into build-macos/ or build-linux/
pip install -e python/
```

The wrapper looks for the extension in this order: an explicit
`gpudb.connect(extension="…")`, the `GPUDB_EXTENSION_PATH` environment
variable, a `build-macos/` or `build-linux/` directory next to a source
checkout, and finally whatever DuckDB itself has installed
(`INSTALL gpudb FROM community; LOAD gpudb;`). If nothing usable is found —
or the installed extension is older than the client — the banner's
`transparent:` line says so, `con.extension_note` carries the same sentence,
and every statement simply runs on DuckDB.

#### Start it

```bash
gpudb                             # in-memory database
gpudb my.duckdb                   # a file
gpudb my.duckdb --readonly        # nobody writes while you look
python -m gpudb                   # the same entry point
```

The banner is the machine's answer to "is this actually going to use the GPU":

<!-- RE-RUN: banner, device line and every timing in this section, on the release build -->

```
gpudb 0.7.0
backend:      Metal · Apple M4 Max · 51.8 GiB device memory
transparent:  available — every statement goes through the wrapper
database:     /data/tpch_sf1/tpch.duckdb
Enter .help for usage.
```

`backend:` names the runtime, the device as the driver reports it, and the
device memory the budget plans against. A build with no GPU backend names no
device, and `transparent:` says what is missing instead.

#### Type SQL, read the footer

Under each result is one dim line: where the statement ran, why, and how long
it took. Captured on an M4 Max over TPC-H SF1, default settings:

```
gpudb> SELECT l_orderkey, sum(l_quantity) AS qty FROM lineitem GROUP BY l_orderkey ORDER BY qty DESC LIMIT 5;
┌────────────┬───────────────┐
│ l_orderkey │      qty      │
│   int64    │ decimal(38,2) │
├────────────┼───────────────┤
│    4806726 │        328.00 │
│    2199712 │        327.00 │
│    4722021 │        323.00 │
│    4702759 │        320.00 │
│    1263015 │        320.00 │
└────────────┴───────────────┘

DuckDB (not_resident: the resident set is not ready yet) · 37.0 ms
```

**That first line is not a failure, and it is the one thing worth understanding
before anything else.** In the default `background` residency the wrapper
records the statement, leaves it with DuckDB, and uploads the columns it needs
in short row-id segments taken only while your connection is idle — so an
upload never runs a long scan beside a query you are waiting for. Ask again a
few seconds later and the same statement is on the device:

```
gpudb> SELECT l_orderkey, sum(l_quantity) AS qty FROM lineitem GROUP BY l_orderkey ORDER BY qty DESC LIMIT 5;
…
GPU (topk: the resident GROUP BY) · 35.7 ms
```

If you would rather pay for it now — a script that knows its workload, or a
benchmark — start with `--residency eager` and the upload happens inside the
statement that asked for it (that statement therefore reads in seconds).
`--residency manual` never uploads by itself; only sets you uploaded by hand
are used.

Statements the GPU does not take say so in the same place. Printing them is the
point:

```
gpudb> SELECT l_returnflag, median(l_quantity) FROM lineitem GROUP BY 1 ORDER BY 1;
…
DuckDB (shape: not a shape the rewrite expresses: select expression) · 56.4 ms
```

The reason codes you will actually meet:

| Reason | What it means |
|---|---|
| `not_resident` | the columns are not on the device yet — the background uploader is working on it |
| `threshold` | a measured bound says DuckDB is faster for this shape and size (`6 groups < 1000`), **or** this machine measured it slower: `measured 4.20 ms rewritten vs 3.10 ms native (re-measured in 60 s)` |
| `shape` | not a shape the rewrite expresses: a window function, `median`, `ROLLUP`, a set operation, a subquery in the select list |
| `double` | a `sum` / `avg` over `DOUBLE` or `FLOAT` — never rewritten, by design (see the two rules) |
| `backend` | this build has no GPU backend to rewrite for (a CPU-only binary), or the installed extension is older than the client |
| `memory` | the set does not fit the device-memory budget; it is refused before the upload |
| `transaction` | a `BEGIN` is open, so the resident sets cannot be trusted |
| `params` | the statement takes prepared-statement parameters |
| `error` | the rewritten statement raised and DuckDB answered the original — the text is in `.gpu` |
| `off` / `manual` | the path is off (`.gpu off`, `--no-gpu`), or residency is `manual` |

`shape` also covers the smaller refusals with their own codes — `nulls`,
`overflow`, `decimal`, `collation`, `too_long`, `multi`, `view`, `temp`,
`ambiguous`, `not_found` — each with its sentence in `detail`.

#### Look underneath

`.gpu` prints the whole record for the last statement, including the SQL that
actually ran:

```
gpudb> .gpu
statement:    SELECT l_orderkey, sum(l_quantity) AS qty FROM lineitem GROUP BY l_orderkey ORDER BY qty DESC LIMIT 5;
rewritten:    True
form:         topk
tag:          gpudb:v1:tpch:main:lineitem:20631:l_orderkey,l_quantity
sql:          SELECT "key" AS l_orderkey, (CAST(sum AS DECIMAL(36,0)) * 0.01) AS qty FROM gpu_groupby_exact_resident_topk('gpudb:v1:tpch:main:lineitem:20631:l_orderkey,l_quantity', 'sum', 5, 'desc') AS r , (SELECT gpu_assert_rows('gpudb:v1:tpch:main:lineitem:20631:l_orderkey,l_quantity', count_star()) AS ok FROM tpch.main.lineitem) AS gd WHERE gd.ok ORDER BY qty DESC LIMIT 5
round_trip_ms:4.444167003384791
engine:       scalar
detail:       the resident GROUP BY
```

`.gpu off` turns the path off live, which makes an honest A/B against DuckDB
one keystroke away (`.gpu on` puts it back; `--no-gpu` starts that way).
`.residents` shows what is on the device — the sets a statement waits on, then
the columns behind them with the width each lane is stored at:

```
gpudb> .residents
table          columns                  state  bytes      estimated  worth
main.lineitem  l_orderkey,l_quantity    ready  80.1 MiB   207.5 MiB  -
main.lineitem  l_returnflag,l_quantity  ready  131.6 MiB  161.0 MiB  -
2 sets · 211.8 MiB held · worth is ms saved per second per GiB · `.memory` for the budget

table     column          dtype  rows       width  bytes      state
lineitem  k#l_returnflag  STR    6,001,215  8 B    120.2 MiB  ready
lineitem  l_orderkey      I64    6,001,215  4 B    68.7 MiB   ready
lineitem  l_quantity      I64    6,001,215  2 B    11.4 MiB   preparing
3 resident columns · width is the bytes a row of the lane is stored at · `-` where the backend does not say
```

`.memory` is the budget side of the same picture:

```
gpudb> .memory
backend:      Metal · Apple M4 Max · 51.8 GiB device memory
resident:     200.3 MiB in 2 sets
budget:       16.0 GiB
residency:    eager
```

Raise or lower it with `--memory-budget 16GB` (`unlimited` removes the cap).
A set that does not fit is not uploaded, and its statements keep running on
DuckDB.

#### Scripts, options, keys

```bash
gpudb my.duckdb -c "SELECT count(*) FROM lineitem;"     # one statement
gpudb my.duckdb -f report.sql                           # a file
gpudb my.duckdb < report.sql                            # or on stdin
```

`-c` and `-f` may be repeated and run in the order given; `.read FILE` does the
same from inside a session. A statement that fails ends a `-c` / `-f` / piped
run with a non-zero exit code; at the terminal the session keeps going. The
footer line is on at a terminal and off in scripted output unless `--timer`
says otherwise (`--no-timer` forces it off, `.timer on|off` toggles it live);
colour follows `NO_COLOR`.

| Option | |
|---|---|
| `gpudb [DATABASE]` | open a database (no argument: in-memory) |
| `-c SQL`, `-f FILE` | run a statement or a file, repeatable, in order |
| `--readonly` | open read-only |
| `--no-gpu` | every statement on DuckDB — the control for a comparison |
| `--residency background\|eager\|manual` | when tables become resident (default `background`) |
| `--memory-budget SIZE` | e.g. `16GB`, or `unlimited` |
| `--timer` / `--no-timer` | force the footer line on or off |
| `--debug` | show tracebacks instead of one-line errors |
| `--version` | the gpudb version |

Dot-commands, kept small on purpose — this is the terminal client for the
transparent path, not an emulation of the DuckDB CLI (no `.mode`, `.output`,
`.import`, `.shell`):

| | |
|---|---|
| `.help` | the list |
| `.gpu` | where the last statement ran, in full |
| `.gpu on\|off` | the transparent path, live |
| `.residents` | the resident sets, then the columns behind them |
| `.memory` | the device-memory budget and what holds it |
| `.timer on\|off` | the footer line |
| `.read FILE`, `.open [DATABASE]` | run a file; open another database |
| `.tables`, `.schema [TABLE]` | plain SQL underneath (`SHOW TABLES`, `DESCRIBE`) |
| `.version` | gpudb and duckdb versions |
| `.quit`, `.exit` | leave |

Statements may span lines and end at `;`. **Ctrl-C** stops the running
statement, or clears what you were typing; **Ctrl-D** at a waiting prompt
leaves. History lives in `~/.gpudb_history` when the Python build has
`readline`. One platform quirk, measured and written down rather than papered
over: on Linux a Ctrl-D typed *while a statement is running* is swallowed by
the terminal's line discipline and the shell stays at the next prompt — stock
Python's `input()` does the same thing, and
[KNOWN_ISSUES.md](KNOWN_ISSUES.md) has the mechanism.

### The Python way

```python
import gpudb

con = gpudb.connect("my.duckdb")
con.execute("SELECT k, sum(v) FROM t GROUP BY k").fetchall()
```

`gpudb.connect()` returns a wrapper around a `duckdb.DuckDBPyConnection`.
Anything it does not define itself is delegated to that connection unchanged —
the relational API, appenders and the fetch helpers (`fetchall`, `fetchone`,
and DuckDB's own `df()` / `arrow()` where their optional dependencies are
installed). `con.sql()` takes the same decision as `con.execute()`, and
`con.cursor()` gives another connection over the same database that shares the
resident sets and the same decisions.

#### Every `connect()` option

| Argument | Default | |
|---|---|---|
| `database` | `":memory:"` | as `duckdb.connect` |
| `read_only` | `False` | as `duckdb.connect` |
| `config` | `None` | DuckDB config dict; the wrapper adds `allow_unsigned_extensions` when it loads a local build |
| `extension=` | `None` | an explicit path to the `.duckdb_extension` |
| `transparent=` | `True` | `False` leaves every statement on DuckDB |
| `residency=` | `"background"` | `"background"` \| `"eager"` \| `"manual"` |
| `floor_rows=` | `1_000_000` | tables smaller than this are never even parsed |
| `idle_ms=` | `20.0` | how long the connection must be idle before an upload segment runs |
| `thresholds=` | `True` | `False` rewrites every exact shape regardless of the predicted win — for parity testing, never for production |
| `log=` | `None` | a callable that receives the wrapper's decisions as text |
| `memory_budget=` | backend default | bytes, or `"16GB"`; `0` / `"unlimited"` removes the cap |

A short script that knows what it will read usually wants
`residency="eager"`. A long-lived service or a notebook wants the default:
the first statements answer from DuckDB and the uploads fill in behind them.

#### Seeing what happened

`con.last_rewrite()` returns the record for the last statement. Its keys are
`rewritten`, `reason`, `detail`, `form`, `tag`, `sql`, `statement`, `engine`,
`round_trip_ms`, `fallback` and `error`.

`con.memory()` returns `budget`, `evictions`, `evictions_wasted` (evictions
made for a candidate that was then declined — it is counted rather than
assumed, and measures 0) and `sets`, each with its state, the size the wrapper
expected, the size the extension reports, and what the set is worth.
`con.extension_note` is empty while the loaded extension can serve the client,
and one sentence saying why not otherwise.

#### A run, end to end

```python
import gpudb

con = gpudb.connect("data/tpch_sf1/tpch.duckdb", read_only=True, residency="eager")
print("extension_note:", repr(con.extension_note))

QUERIES = {
    "top 5 orders by quantity":
        "SELECT l_orderkey, sum(l_quantity) AS qty FROM lineitem "
        "GROUP BY l_orderkey ORDER BY qty DESC LIMIT 5",
    "pricing summary (few groups)":
        "SELECT l_returnflag, l_linestatus, sum(l_quantity) AS qty, count(*) AS n "
        "FROM lineitem WHERE l_shipdate <= DATE '1998-09-02' GROUP BY 1, 2 ORDER BY 1, 2",
    "median (no kernel)":
        "SELECT l_returnflag, median(l_quantity) FROM lineitem GROUP BY 1 ORDER BY 1",
}

for title, q in QUERIES.items():
    rows = con.execute(q).fetchall()
    r = con.last_rewrite()
    print(f"\n--- {title}")
    for row in rows[:3]:
        print("   ", row)
    print("    rewritten:", r["rewritten"], "| reason:", r["reason"], "| form:", repr(r["form"]))
    print("    detail:", r["detail"])

print("\nmemory:", con.memory())
con.close()
```

Its output on an M4 Max, TPC-H SF1, DuckDB 1.5.5 <!-- RE-RUN -->:

```
extension_note: ''

--- top 5 orders by quantity
    (4806726, Decimal('328.00'))
    (2199712, Decimal('327.00'))
    (4722021, Decimal('323.00'))
    rewritten: True | reason:  | form: 'topk'
    detail: the resident GROUP BY

--- pricing summary (few groups)
    ('A', 'F', Decimal('37734107.00'), 1478493)
    ('N', 'F', Decimal('991417.00'), 38854)
    ('N', 'O', Decimal('74476040.00'), 2920374)
    rewritten: False | reason: threshold | form: ''
    detail: 6 groups < 1000

--- median (no kernel)
    ('A', Decimal('26.00'))
    ('N', Decimal('25.00'))
    ('R', Decimal('26.00'))
    rewritten: False | reason: shape | form: ''
    detail: not a shape the rewrite expresses: select expression

memory: {'budget': 17179869184, 'evictions': 0, 'evictions_wasted': 0, 'sets': {'gpudb:v1:tpch:main:lineitem:20631:l_orderkey,l_quantity': {'state': 'ready', 'est_bytes': 217609579, 'bytes': 84017010, 'error': '', 'value': 0.0, 'uses': 1, 'density': 0.0}}}
```

Three statements, three different honest outcomes — a win, a size threshold,
and a shape with no kernel behind it.

#### Writes, transactions, parameters

- **Writes.** After an `INSERT` / `UPDATE` / `DELETE` the resident copy is
  invalidated and rebuilt, and statements run on DuckDB until it is ready
  again. On a file-backed database the wrapper stats the database file and its
  write-ahead log (2–3 µs) before every rewritten statement, so a committed
  write from **any** connection is noticed — not only its own. On an in-memory
  database there is no file to watch: writes through the wrapper's connection
  and its `cursor()`s are seen, a write through a raw `duckdb` cursor is not.
  Use the wrapper's own cursors there, or a file-backed database.
- **Transactions.** Between `BEGIN` and `COMMIT` nothing is rewritten; every
  statement runs on DuckDB (`reason == "transaction"`).
- **Parameters.** `con.execute(sql, params)` runs on DuckDB in v0.7
  (`reason == "params"`): a `PARAMETER` node is rejected by shape.
- **Errors.** If a rewritten statement raises, the wrapper re-runs your
  original statement on DuckDB and leaves that template native afterwards.
- **Closing.** `con.close()` as usual; `con.cursor()` gives a connection that
  shares the same resident sets and the same decisions.

### Explicit `gpu_*` functions — any DuckDB client, including the CLI

The 64 `gpu_*` functions are unchanged in v0.7 and need no wrapper. This is
also the only route from the stock `duckdb` CLI, because DuckDB's stable C
extension API — the one the loadable extension uses on purpose, so that one
binary keeps working across DuckDB versions — has no hook that sees a statement
before it is planned.

```sql
INSTALL gpudb FROM community;
LOAD gpudb;

SELECT gpu_build_info();
-- compiled=cpu,metal runtime=metal exact=true join=true global=true narrow=true
--   device_memory=55662788608 store=true rebuilds=0/0 device='Apple M4 Max'

CREATE TABLE sales AS
  SELECT (range % 1000)::BIGINT AS store, (range * 7 % 10007)::BIGINT AS amount
  FROM range(10000000);
SELECT gpu_upload_pair('sales_by_store', store, amount) FROM sales;   -- resident, once

SELECT * FROM gpu_groupby_sum_resident_topk('sales_by_store', 5, 'desc');
-- (282, 50037056, 10000) … top 5 stores by sum(amount): (store, sum, count)

SELECT gpu_last_stats();
-- op=groupby_sum_resident_topk backend=Metal reason=Hot_GpuAlwaysWins
--   rows_in=10000000 groups=1000 rows_out=5 wall_ms=39.551 kernel_ms=15.834 transfer_ms=0.000
```

The full surface, the identity-tag rules and the one footgun are further down
under [The resident model in 20 seconds](#the-resident-model-in-20-seconds).

### Measured — TPC-H, every row compared with native

<!-- RE-RUN: the whole of this section is re-measured on the release build -->

Apple M4 Max, 64 GiB unified memory, Metal backend · DuckDB v1.5.x · warm,
minimum of 5 runs, statement against statement in one process · measured
2026-09-19. These numbers are from one machine; yours will differ.
`python3 scripts/tpch_coverage.py --db data/tpch_sf10/tpch.duckdb` reproduces
the tables.

| | Queries answered on the GPU | Rows differing | Speed-up on those queries |
|---|---|---|---|
| TPC-H SF1 (6M-row `lineitem`) | 17 of 22 | 0 | 1.4× – 13.6× |
| TPC-H SF10 (60M-row `lineitem`) | 19 of 22 | 0 | 1.3× – 52.9× |

The queries that stay on DuckDB are declined on purpose. Q2 and Q20 hold a
correlated subquery that does not bind on its own; Q16's inner `GROUP BY`
declines on its own threshold (forced, it measures 0.02–0.08×); at SF1, Q6 and
Q11 are below the measured size floors. Each of those runs on DuckDB unchanged.

<details>
<summary>Query by query — scale factor 10</summary>

| Query | What it is | DuckDB ms | gpudb ms | Speed-up | Runs on |
|---|---|---:|---:|---:|---|
| Q1 | pricing summary | 109.9 | 14.4 | 7.6× | ✓ GPU |
| Q2 | minimum cost supplier | 17.6 | — | — | DuckDB — the inner statement is correlated and does not bind on its own |
| Q3 | shipping priority | 45.8 | 13.6 | 3.4× | ✓ GPU |
| Q4 | order priority | 44.1 | 2.6 | 17.1× | ✓ GPU |
| Q5 | local supplier volume | 46.7 | 0.9 | 52.9× | ✓ GPU |
| Q6 | forecasting revenue change | 14.1 | 5.9 | 2.4× | ✓ GPU |
| Q7 | volume shipping | 47.0 | 10.9 | 4.3× | ✓ GPU |
| Q8 | national market share | 66.7 | 7.8 | 8.5× | ✓ GPU |
| Q9 | product type profit | 145.0 | 5.6 | 26.1× | ✓ GPU |
| Q10 | returned items | 77.4 | 13.9 | 5.6× | ✓ GPU |
| Q11 | important stock | 10.4 | 6.6 | 1.6× | ✓ GPU |
| Q12 | shipping modes | 39.3 | 10.3 | 3.8× | ✓ GPU |
| Q13 | customer distribution | 157.3 | 13.6 | 11.6× | ✓ GPU |
| Q14 | promotion effect | 29.0 | 4.3 | 6.7× | ✓ GPU |
| Q15 | top supplier | 20.4 | 15.3 | 1.3× | ✓ GPU |
| Q16 | parts and suppliers | 37.6 | — | — | DuckDB — the inner GROUP BY declines on its own threshold |
| Q17 | small-quantity revenue | 49.3 | 3.2 | 15.6× | ✓ GPU |
| Q18 | large volume customers | 101.7 | 8.8 | 11.5× | ✓ GPU |
| Q19 | discounted revenue | 64.6 | 4.2 | 15.3× | ✓ GPU |
| Q20 | potential part promotion | 34.9 | — | — | DuckDB — the inner statement is correlated and does not bind on its own |
| Q21 | suppliers who kept orders waiting | 159.4 | 21.4 | 7.5× | ✓ GPU |
| Q22 | global sales opportunity | 27.1 | 0.7 | 36.9× | ✓ GPU |

</details>

<details>
<summary>Query by query — scale factor 1</summary>

The smaller the table, the more often DuckDB is already the faster answer, and
the size thresholds leave the statement with it.

| Query | What it is | DuckDB ms | gpudb ms | Speed-up | Runs on |
|---|---|---:|---:|---:|---|
| Q1 | pricing summary | 12.1 | 3.2 | 3.8× | ✓ GPU |
| Q2 | minimum cost supplier | 4.6 | — | — | DuckDB — below the size threshold |
| Q3 | shipping priority | 6.6 | 2.2 | 3.1× | ✓ GPU |
| Q4 | order priority | 7.3 | 0.8 | 9.4× | ✓ GPU |
| Q5 | local supplier volume | 6.8 | 1.1 | 6.1× | ✓ GPU |
| Q6 | forecasting revenue change | 1.9 | — | — | DuckDB — below the size threshold at 6M rows |
| Q7 | volume shipping | 7.4 | 2.1 | 3.5× | ✓ GPU |
| Q8 | national market share | 7.3 | 1.4 | 5.2× | ✓ GPU |
| Q9 | product type profit | 18.5 | 1.4 | 13.6× | ✓ GPU |
| Q10 | returned items | 18.0 | 2.9 | 6.3× | ✓ GPU |
| Q11 | important stock | 2.8 | — | — | DuckDB — below the size threshold |
| Q12 | shipping modes | 6.1 | 1.6 | 3.8× | ✓ GPU |
| Q13 | customer distribution | 18.6 | 1.7 | 11.2× | ✓ GPU |
| Q14 | promotion effect | 5.1 | 1.9 | 2.7× | ✓ GPU |
| Q15 | top supplier | 3.1 | 2.2 | 1.4× | ✓ GPU |
| Q16 | parts and suppliers | 12.2 | — | — | DuckDB — below the size threshold |
| Q17 | small-quantity revenue | 6.1 | 1.2 | 5.2× | ✓ GPU |
| Q18 | large volume customers | 13.3 | 1.5 | 9.0× | ✓ GPU |
| Q19 | discounted revenue | 10.3 | 1.4 | 7.1× | ✓ GPU |
| Q20 | potential part promotion | 7.7 | — | — | DuckDB — the inner statement is correlated and does not bind on its own |
| Q21 | suppliers who kept orders waiting | 21.4 | 2.9 | 7.3× | ✓ GPU |
| Q22 | global sales opportunity | 8.0 | 1.1 | 7.6× | ✓ GPU |

</details>

### The two rules

**Never slower than DuckDB.** Not on average — per statement, per shape, per
size. Which shapes may be rewritten at all comes from bounds a gate measures
against native before a release; one row under 1.0× stops it, and the losing
measurements stay published. Your machine is not the gate's, so the decision is
re-taken there: after a statement template's first three rewritten runs the
wrapper times the native form once on a side cursor in your own process, hands
the template back to DuckDB if the rewritten runs were not faster, and
re-measures every 60 seconds. Your own statement is never the experiment.

**Never a different answer.** A rewritten statement returns what native returns
— the same rows, the same order where native guarantees one, the same column
names, the same column types. Integer and `DECIMAL` aggregates are bit-exact,
with 128-bit sums on the device. Every scenario in the suite runs three ways in
one process — native, rewritten, and the explicit `gpu_*` calls — and all three
must agree on ordered rows, names and `typeof()` of every column; each rewrite
is also checked against the original statement text with `DESCRIBE` before it is
used. Where "the same as native" is not definable — `sum(DOUBLE)`, which DuckDB
itself computes in an order-dependent way — the shape is simply never rewritten.

### What runs on the GPU, and what stays on DuckDB

A row marked DuckDB is not a missing answer: the statement runs as it always
did, at its usual speed. Section numbers point at
[docs/TRANSPARENT_DESIGN.md](docs/TRANSPARENT_DESIGN.md).

| Feature | Runs on | Note |
|---|---|---|
| **Grouping and aggregation** | | |
| `sum` `count` `count(*)` `min` `max` `avg`, with `GROUP BY` | ✓ GPU | up to eight aggregated columns in one device pass (§4.9) |
| Aggregates with no `GROUP BY` | ✓ GPU | one fused pass; over a join at any size, over a single table above a measured row and predicate bound (§4.12) |
| Expressions inside aggregates | ✓ GPU | `sum(price * (1 - discount))`, `sum(CASE …)` (§4.10) |
| Expressions over aggregates, compound `HAVING` | ✓ GPU | §4.11 |
| `count(DISTINCT x)` and `DISTINCT` aggregates | ✓ GPU | an inner device `GROUP BY`, with a bound of its own (§4.17) |
| `median`, `stddev`, quantiles | DuckDB | no kernel for them |
| **Filtering, ordering, shorthand** | | |
| `WHERE`: comparisons, `BETWEEN`, `IN`, `IS [NOT] NULL` | ✓ GPU | one fused mask pass over resident predicate lanes (§4.6) |
| `WHERE`: `OR`, `LIKE`, functions, column vs column | ✓ GPU | each becomes a computed lane (§4.10) |
| `HAVING`, `ORDER BY … LIMIT k` | ✓ GPU | top-k on the device; the `ORDER BY` node is never removed, so order and NULL order stay DuckDB's (§2) |
| `SELECT DISTINCT`, `GROUP BY ALL`, ordinals, `ORDER BY ALL` | ✓ GPU | spelled out by DuckDB's own definitions before the decision (§4.21) |
| `FILTER (WHERE …)`, `count_if`, `bool_and`, `bool_or` | ✓ GPU | rewritten to the aggregate they are in disguise (§4.19) |
| `ROLLUP`, `CUBE`, `GROUPING SETS`, `QUALIFY`, `DISTINCT ON` | DuckDB | rejected on the matched node (§2) |
| **Types and keys** | | |
| Integer keys and payloads, `DATE`, `TIMESTAMP` | ✓ GPU | int64-representable as keys and as predicates (§2) |
| `DECIMAL` | ✓ GPU | scaled integers, 128-bit sums (§4.2, §4.3) |
| `VARCHAR` keys | ✓ GPU | through a dictionary, binary collation only (§4.5) |
| Multi-column, wide and `DECIMAL` keys | ✓ GPU | up to three packed into one 64-bit key where the ranges allow, a hashed tuple above that (§4.4, §4.15) |
| `min` / `max` / `count` over `DOUBLE` | ✓ GPU | exact (§4.7) |
| `sum` / `avg` over `DOUBLE` or `FLOAT` | DuckDB | by design: native `sum(DOUBLE)` depends on the order the values are added, so "the same as native" is not definable. Never rewritten, so rounding can never differ (§4.7). The explicit `gpu_sum` keeps its stated 1e-9 relative tolerance |
| `UBIGINT` / `HUGEINT` keys | DuckDB | not int64-orderable (§2) |
| **Joins, subqueries, views, CTEs** | | |
| `INNER JOIN` on a unique key | ✓ GPU | the fact-to-dimension shape, materialised on the device (§4.8) |
| `LEFT` / `RIGHT`, many-to-many, `USING`, composite keys | ✓ GPU | answered from an upload of the join's result (§4.13, §4.21) |
| `EXISTS` / `IN` / a correlated scalar subquery in `WHERE` | ✓ GPU | lowered to a predicate lane DuckDB fills once per row (§4.18) |
| Derived tables, views, CTEs | ✓ GPU | folded or spliced in first, then checked against the original with `DESCRIBE` (§4.16, §4.20, §4.22) |
| Aggregation nested inside a statement DuckDB keeps | ✓ GPU | the inner `SELECT` gets its own decision and its own guards (§4.14) |
| `FULL` join, semi / anti joins, cross products | DuckDB | §2 |
| `WITH RECURSIVE`, `AS MATERIALIZED` | DuckDB | left as written (§4.22) |
| `UNION` / `UNION ALL` as the whole statement | DuckDB | an aggregating `SELECT` inside an arm is still offered to the GPU (§2) |
| **Session and statement handling** | | |
| Window functions | DuckDB | the `WINDOW` class is rejected (§2) |
| Prepared statements with parameters | DuckDB | `PARAMETER` nodes are rejected by shape (§2) |
| Statements inside an explicit transaction | DuckDB | nothing is rewritten while a `BEGIN` is open (§5.4) |
| A set that does not fit the memory budget | DuckDB | refused before the upload; the statement runs natively (§5.5) |

Every trade-off above, with its reason, is in
[KNOWN_ISSUES.md](KNOWN_ISSUES.md).

### How it decides

- The statement is rewritten **before DuckDB plans it**, through DuckDB's own
  parser (`json_serialize_sql`) and a pure function in the extension
  (`gpu_rewrite_ast`) — no plan surgery, no C++ API.
- Size bounds come from a measured sweep (`python/gpudb/_thresholds.py`, from
  `scripts/transparent_gate.py`): tables under `floor_rows` (default 1,000,000)
  are never parsed; keys with fewer than 1,000 distinct values never rewrite;
  the plain form returns at most 300K groups (50K under a `WHERE`); an
  aggregate without `GROUP BY` over a single table needs 16M rows. The shapes
  outside those bounds measured 0.23–0.99× against native, and those rows stay
  in the gate output.
- Then the run-time measurement above overrides the bounds in either direction.
  `last_rewrite()["detail"]` names the rule that decided, in both directions.
- Any error on the rewritten path re-runs the user's original statement on
  DuckDB. An error there can never reach you as a different or a missing answer.
- Every rewritten statement carries a staleness guard that re-counts the rows of
  each table it reads inside the same transaction, and on a file-backed database
  the database file and its write-ahead log are stat'ed (2–3 µs) before every
  rewritten statement, so a committed write from any connection is noticed.

### Residency and the memory budget

Tables become resident in the background, in short row-id segments taken only
while your connection is idle, so an upload never runs a long scan beside a
query. A table's columns live in one per-table store, each lane kept at the
narrowest signed width its values fit — the 22 TPC-H queries at SF10 hold
22.7 GiB where they held 44.9 before narrow lanes.

The memory budget defaults to a quarter of unified memory on Apple silicon and
half of device memory on a discrete GPU (`memory_budget=`, or
`GPUDB_MEMORY_BUDGET_MB`). A set that does not fit is refused **before** the
upload and its statements run on DuckDB. What is kept under pressure is decided
by measured value per byte rather than by recency: the sets that save the most
DuckDB time per byte stay. `con.memory()` and the shell's `.memory` print the
budget and what holds it; `.residents` prints the sets and the columns behind
them.

### Platforms and install

| | Metal (Apple Silicon) | CUDA (NVIDIA) | No GPU |
|---|---|---|---|
| Plain SQL on the GPU, through the shell or `gpudb.connect()` | yes | opt-in, see below | no — everything runs on DuckDB |
| Explicit `gpu_*` functions | yes | yes | yes, on the CPU backend, same answers |
| From the community registry | yes | the registry's Linux binary is CPU-only | yes |

<!-- CUDA-DEFAULT: one row + the paragraph below are the conservative (opt-in) statement.
     If CUDA ships on by default at release, change the CUDA cell above to "yes" and
     replace the paragraph below with the measured CUDA table. -->

On NVIDIA hardware every operator the transparent path needs is implemented —
exact `GROUP BY`, the `WHERE` mask, the global aggregate and the materialised
join — and the SQL suite and the TPC-H coverage run there with rows identical to
native. It is **opt-in** in this release: set `GPUDB_CUDA_EXACT=1` to let the
CUDA backend answer plain SQL. Without it, a CUDA machine gets the explicit
`gpu_*` functions and leaves plain SQL to DuckDB — correct, with no speed-up.

`SELECT gpu_build_info();` tells any binary apart. The install routes are in
[Quick start](#quick-start) below: the community registry, a release binary, or
a build from source. The Python wrapper and the `gpudb` shell come from
`pip install duckdb-gpudb`.

### Honest limits

- [KNOWN_ISSUES.md](KNOWN_ISSUES.md) — every documented trade-off, reason by
  reason, including what a resident column does not carry and where the GPU
  path declines.
- [BENCHMARK.md](BENCHMARK.md) — the whole measurement record, **including the
  shapes where the GPU loses**. Low-cardinality `GROUP BY` on Metal, whole-column
  `min`/`max` against DuckDB's zonemaps, and row materialisation across PCIe are
  all in there with their numbers.
- A write made through a raw `duckdb` cursor on an **in-memory** database is not
  seen by the guard (a file-backed database is watched); use the wrapper's own
  cursors there.

### Research and design notes

The work is written down as it happens — what was tried, what was measured,
what lost, and why each decision was taken:

- **[docs/RESEARCH_NOTES.md](docs/RESEARCH_NOTES.md)** — the dated research journal.
- **[docs/TRANSPARENT_DESIGN.md](docs/TRANSPARENT_DESIGN.md)** — how a plain `SELECT` reaches the GPU and how the answer is kept identical.
- **[docs/RESIDENT_COLUMNS_DESIGN.md](docs/RESIDENT_COLUMNS_DESIGN.md)** — how columns live on the device.
- **[docs/README.md](docs/README.md)** — a reading guide to all of it, and how to reproduce a number.

---

## What you'd use it for

gpudb is for workloads that **ask the same aggregate questions of the same big data, over and over**. Upload a column to GPU memory once; every query after runs at memory-bandwidth speed with zero transfer.

- **📊 Dashboards & monitoring** — a metrics dashboard re-runs `SUM`s every few seconds. Resident columns turn a 99 ms scan into 10 ms (measured, 600M rows). Refresh every 5 s and the one-time upload pays for itself inside 10 minutes — then every refresh is 4–25× cheaper, forever.
- **🔬 Notebook exploration** — an analyst slicing data re-aggregates on every idea. Upload at the top of the notebook; the whole session runs GPU-speed. On a Mac this is capability that exists nowhere else: no other SQL engine uses the Apple Silicon GPU.
- **⚡ Serving APIs** — endpoints answering "total X for Y" thousands of times a day. The resident column is a cache that never goes stale-wrong: exact answers, 5–25× lower latency, re-upload in seconds when data refreshes.
- **📈 High-cardinality GROUP BY / top-k** — "quantity per order", "spend per customer", "top 10 by amount" over tens of millions of keys, re-asked as the data is explored. `gpu_groupby_sum_resident` returns `(key, sum, count)` rows (`gpu_groupby_count_resident` returns `(key, count)`) from a cached device sort, and the `_having(name, cmp, threshold)` / `_topk(name, k, order)` forms evaluate `HAVING` and `ORDER BY sum LIMIT k` **on the device** so only the survivors come back: **5.4–6.1× (Metal)** on TPC-H Q18's inner query at SF10–SF50, 6–8× on `HAVING count(*) >= 7`, 3.6–4.1× on the top-10 groups by sum — statement time vs statement time, same process. On CUDA the same device-side HAVING is **13–24× (SF50) / 17× (SF10)** and the top-10 groups 13–17×, because the 1.8 GB result copy over PCIe simply no longer happens. Returning all 15M–75M groups is 2.3–2.9× (Metal) / 1.5× end-to-end on CUDA (~32× on-device, PCIe-bound). Low-cardinality GROUP BY stays a native win on Metal (measured, kept in the table).
- **💰 DECIMAL/financial data** — money columns are stored DECIMAL, and native DuckDB re-casts every value on every scan. `gpu_upload` stores the cast once — it's why our biggest measured wins (9.9× Metal, 25× CUDA) came from the most accounting-shaped column in TPC-H.
- **🎯 Membership at scale (semi / anti join)** — "how much did *these* customers spend?", "which transactions hit the blocklist?", "how many events came from outside the cohort?" Keep the big fact side resident (`gpu_upload_pair`), re-upload only the small, changing set, and ask with `gpu_semi_join_*` / `gpu_anti_join_*`: **22× (Metal) / ~376× (CUDA)** over native on TPC-H SF10 (measured, v0.5.0).
- **🔗 Fact ⋈ dimension rollups** — revenue joined to a filtered orders/customers/dates set, re-asked per filter. `gpu_join_sum_resident` / `gpu_left_join_count_resident` run the fused join-aggregate on the device with the sorted build side cached: **11.7× / 27–37×** at SF50. DOUBLE payloads via the `_f64` variants (within the 1e-9 relative tolerance contract; measured ≤4e-11).

**Not for:** one-shot queries on cold data (transfer loses — the streaming `gpu_sum/min/max` deliberately match native there), or `min`/`max` where DuckDB's statistics answer without scanning, or joins that must return the matched *rows* at scale (`gpu_join_rows_resident` works, but native DuckDB wins on discrete GPUs — use the aggregate variants). [KNOWN_ISSUES.md](KNOWN_ISSUES.md) lists every trade-off honestly.

## Numbers — measured, not promised

TPC-H `lineitem`, warm cache, every result verified equal to native before
timing counted. GROUP BY rows: statement against statement inside the same
embedded DuckDB v1.5.2 process, after the one-time upload and sort; aggregate
and join rows: DuckDB CLI (v1.5.5 for CUDA, v1.5.2 for the Metal joins),
5-run medians. Full grid + reproduction:
**[BENCHMARK.md](BENCHMARK.md)**.

| TPC-H | Workload | Hardware | Native | gpudb | |
|---|---|---|---:|---:|:---|
| **SF50** (300M rows → 75M groups) | Q18 inner: `GROUP BY l_orderkey HAVING sum > 300`, HAVING on the device | RTX 4090 Laptop · CUDA | 1012–1039 ms | **42–78 ms** (kernel 37–72, bimodal laptop clocks) | **13–24× 🚀** |
| **SF10** (60M rows → 15M groups) | Q18 inner, HAVING on the device | RTX 4090 Laptop · CUDA | 208–210 ms | **11.7–12.9 ms** (kernel 8–10) | **16–18× 🚀** |
| **SF50** (300M rows → 75M groups) | top-10 groups by `SUM` (`ORDER BY sum DESC LIMIT 10`) | RTX 4090 Laptop · CUDA | 1009–1023 ms | **60–64 ms** (kernel 52–56) | **16–17× 🚀** |
| **SF50** (300M rows → 75M groups) | Q18 inner: `GROUP BY l_orderkey HAVING sum > 300`, HAVING on the device | MacBook M4 Max · Metal | 462–476 ms | **78 ms** (kernel 77) | **5.9–6.1× 🚀** |
| **SF50** (300M rows → 75M groups) | `HAVING count(*) >= 7` (10.7M groups survive) | MacBook M4 Max · Metal | 413–438 ms | **56 ms** (kernel 50) | **7.3–7.8× 🚀** |
| **SF50** (300M rows → 75M groups) | top-10 groups by `SUM` (`ORDER BY sum DESC LIMIT 10`) | MacBook M4 Max · Metal | 463–471 ms | **114 ms** (kernel 112) | **4.0× 🚀** |
| **SF50** (300M rows → 75M groups) | `GROUP BY l_orderkey` + `SUM` BIGINT, all 75M groups returned | MacBook M4 Max · Metal | 347–355 ms | **120–123 ms** (operator 87–90) | **2.8–2.9× 🚀** |
| **SF50** (300M rows → 75M groups) | `GROUP BY l_orderkey` + `SUM` BIGINT, all 75M groups returned | RTX 4090 Laptop · CUDA | 681–718 ms | **458–471 ms** (kernel 21 ms) | **1.5× end-to-end, ~32× on-device** |
| **SF50** (300M ⋈ 75M) | `JOIN` + `SUM` BIGINT | RTX 4090 Laptop · CUDA | 998 ms | **27–37 ms** | **27–37× 🚀** |
| **SF10** (60M ⋈ 15M) | `EXISTS` semi-join + `SUM` DOUBLE | RTX 4090 Laptop · CUDA | 640 ms | **1.7 ms** | **~376× 🚀** |
| **SF50** (300M ⋈ 75M) | `JOIN` + `SUM` BIGINT | MacBook M4 Max · Metal | 429 ms | **37 ms** | **11.7× 🚀** |
| **SF10** (60M ⋈ 15M) | `EXISTS` semi-join + `SUM` DOUBLE | MacBook M4 Max · Metal | 182 ms | **8.2 ms** | **22.2× 🚀** |
| **SF50** (300M rows) | `SUM` BIGINT | RTX 4090 Laptop · CUDA | 99 ms | **4 ms** | **25× 🚀** |
| **SF100** (600M rows) | `SUM` DOUBLE | RTX 4090 Laptop · CUDA | 196 ms | **9 ms** | **22× 🚀** |
| **SF100** (600M rows) | `SUM` BIGINT | MacBook M4 Max · Metal | 99 ms | **10 ms** | **9.9× 🚀** |
| **SF50** (300M rows) | `SUM` BIGINT | MacBook M4 Max · Metal | 46 ms | **5 ms** | **8.5× 🚀** |
| **SF10** (60M rows) | `SUM` BIGINT | MacBook M4 Max · Metal | 9.6 ms | **1.4 ms** | **6.9× 🚀** |

**The bigger the scale factor, the bigger the win** — the resident kernel runs
at the memory-bandwidth ceiling of the silicon (563 GB/s CUDA, 503 GB/s
Metal) while native's scan grows linearly. Ratio curve on Metal:
2× (SF1) → 5× → 6.9× → 6.8× → 8.5× → 9.9× (SF100); on CUDA: 4× → 10× →
20× → 24× → 25× → 21×.

Honest asterisks, on the table not under it: TPC-H's DECIMAL-stored columns
make native pay a cast per scan while the resident column stores it once —
already-BIGINT columns win 3.3–3.7× (Metal) / 5.6–10× (CUDA). Whole-column
`min`/`max` on stored tables stays a **native win** (zonemap statistics answer
without scanning). One-time upload breaks even after ~100–150 repeated
aggregate queries; the GROUP BY rows assume the pair is resident — on Metal the
upload is ~1 s at SF10 / ~6 s at SF50 and the first call pays the sort
(0.3–3 s), on CUDA the upload is 4–5 s / 19–25 s — which the device-side
HAVING / top-k rows recoup after ~20–25 repeated queries and the all-groups
rows after ~40 (Metal) to ~90 (CUDA). Returning every group is 1.5–3× — the larger ratios come from
running the `HAVING` / `LIMIT` on the device; the CUDA all-groups row is
bounded by copying 24 bytes per group over PCIe, which unified memory
does not pay. `DOUBLE` sums filter on the host on Metal (1.4–2.6×). Low-cardinality GROUP BY (a handful of groups) is a
**native win on Metal** (0.56–0.65× at SF10–SF50, a tie at SF1) and stays in the table. All in
[BENCHMARK.md](BENCHMARK.md) and [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

## The resident model in 20 seconds

```sql
INSTALL gpudb FROM community; LOAD gpudb;

SELECT gpu_upload('sales', amount::BIGINT) FROM orders;  -- pay the transfer once
SELECT gpu_sum_resident('sales');                        -- every query after: GPU speed
SELECT gpu_min_resident('sales'), gpu_max_resident('sales');
SELECT gpu_sum_resident_f64('price');                    -- DOUBLE flavor
SELECT gpu_resident_info('sales');                       -- dtype / rows / device
SELECT gpu_last_stats();                                 -- which backend ran + kernel time
SELECT gpu_build_info();                                 -- which backends this binary carries
SELECT gpu_drop_resident('sales');                       -- free device memory

-- v0.7 registry (milestone 0b): every resident set, with identity, state, size, hits
SELECT * FROM gpu_residents();
SELECT gpu_prepare_resident('l');          -- build the sort cache now instead of on the first GROUP BY
SELECT gpu_invalidate('l');                -- mark a set stale (exact name, or an identity-tag prefix)
-- staleness guard for a rewritten statement: raises "GPUDB_STALE: ..." unless the
-- table still has the row count the upload saw; the one-row derived table runs once
SELECT r.key, r.sum
FROM gpu_groupby_sum_resident('gpudb:v1:memory:main:lineitem:1:l_orderkey,l_quantity') r,
     (SELECT gpu_assert_rows('gpudb:v1:memory:main:lineitem:1:l_orderkey,l_quantity', count(*)) AS ok
      FROM lineitem) gd
WHERE gd.ok;
```

A set name of the form `gpudb:v1:<catalog>:<schema>:<table>:<table_oid>:<col1>[,<col2>…][:<extra>]`
is an **identity tag**: the set is `managed` (what the v0.7 transparent path
consumes), its fields show in `gpu_residents()`, and it is prepared at upload
so the first query pays no sort. Any other name is an `explicit` set with the
v0.6 behaviour. A pair is one registry entry (`'l.k'` / `'l.v'` address its
columns), an operator call keeps its set alive until it finishes, and uploads
never block queries on another connection.

⚠️ **One footgun:** in a single-statement upload-and-query, the outer query
must reference the upload's result column (e.g. `SELECT u.n, gpu_sum_resident('x') FROM (SELECT gpu_upload('x', col) AS n FROM t) u`) —
an unreferenced `gpu_upload` is pruned by DuckDB's optimizer and never runs.
Uploads are capped at 4 GB of buffering by default
(`GPUDB_UPLOAD_POOL_MAX_MB` to raise); the streaming `gpu_sum/min/max`
aggregates work in any query shape (GROUP BY, windows, FILTER) at native
parity.

## Why this exists

Every standalone GPU database from 2013-2024 was acqui-hired or pivoted (HEAVY.AI → NVIDIA 2025, BlazingSQL dormant, Voltron Data 50% layoff). Building "another GPU SQL engine" is not a viable bet.

What's open in 2026: **no published SQL engine targets Apple Silicon GPUs**. Sirius (UW + NVIDIA, CIDR 2026) is CUDA-only. cuDF is CUDA-only. So is everything else. Apple Silicon's unified memory architecture (up to 512 GB at 819 GB/s on M3 Ultra) is a genuine architectural advantage that nobody has wired into a database.

`gpudb` is a DuckDB *extension* (not a fork, not a new database) that closes that gap with a real dual-backend implementation. Operator-level benchmarks (GROUP BY, multi-aggregate fusion, hash join) live in [BENCHMARK.md](BENCHMARK.md)'s earlier entries.

## Quick start

### Option A — install from the DuckDB community repo (recommended)

```sql
INSTALL gpudb FROM community;
LOAD gpudb;
SELECT gpu_sum(value::BIGINT) FROM range(1000000) AS t(value);
-- -> 499999500000
```

Works in any DuckDB ≥ 1.5.5 client (CLI, Python, etc.), signed, no flags needed.
The registry binary carries the **full Metal backend on Apple Silicon**. On
Linux the registry binary is **CPU-only** — the community build machines have
no CUDA toolchain, so every `gpu_*` function works and returns the same
results, but `gpu_last_stats()` will say `backend=CPU`. For the CUDA backend
on Linux use the release binary (Option B; statically linked CUDA runtime,
needs only a driver) or build from source with `nvcc`. Check any binary with
`SELECT gpu_build_info();`.
The registry serves the **v0.6.0** build (merged 2026-08-29), including the
resident GROUP BY / top-k table functions (`gpu_groupby_*_resident`,
`gpu_topk_resident`), the full resident-column surface (`gpu_upload`, `gpu_sum_resident`, `gpu_build_info`,
…) and the GPU join functions (`gpu_upload_pair`, `gpu_join_*_resident`,
`gpu_join_rows_resident`). Installed an earlier version? `UPDATE EXTENSIONS;`
pulls the latest.

### Option B — load a prebuilt release binary

Download the platform binary from the [latest release](https://github.com/singhpratech/duckdbgpumetaldbram/releases/latest), then:

```bash
# Linux (RTX/CUDA)
duckdb -unsigned -c "LOAD '/path/to/gpudb.linux_amd64.duckdb_extension'; \
  SELECT gpu_sum(value::BIGINT) FROM range(1000000) AS t(value);"
# -> [gpudb] registered gpu_sum / gpu_min / gpu_max (BIGINT,DOUBLE) streaming aggregates (backend=CUDA)
# -> 499999500000
```

Requires DuckDB ≥ 1.2 (C API v1.2.0); release binaries track the latest tag.
`LOAD` needs `-unsigned` here because release-page binaries are unsigned —
the community install above doesn't.

### Option C — build from source

```bash
git clone https://github.com/singhpratech/duckdbgpumetaldbram.git
cd duckdbgpumetaldbram

# Linux (CUDA): one-time toolkit install if needed
# sudo apt install -y cuda-toolkit-13-0
# export PATH=/usr/local/cuda/bin:$PATH

# macOS (Metal): brew install cmake

# build (auto-detects CUDA on Linux, Metal on macOS, CPU-only otherwise).
# Produces a loadable .duckdb_extension with metadata footer attached.
./scripts/build.sh

# load + query via DuckDB CLI
duckdb -unsigned -c "LOAD '$(pwd)/build-linux/src/extension/gpudb.linux_amd64.duckdb_extension'; \
  SELECT gpu_sum(range::BIGINT) FROM range(1000000);"

# OR run via the embedded SQL CLI shipped in this repo
./build-linux/bin/gpudb-sql --sql "SELECT gpu_sum(range::BIGINT) FROM range(1000000);"
```

#### CUDA requirements (build from source on Linux)

| | Supported | Notes |
|---|---|---|
| **CUDA Toolkit** | **13.0** (verified: 13.0.88 / CUB 3.0.1, all benchmarks) | Runtime API plus the CUB that ships with the toolkit (`DeviceReduce`, `DeviceSelect`, `DeviceRadixSort`, `DeviceScan` on explicit temp storage; Thrust only for iterators; no cooperative groups). 64-bit item counts need **CUB ≥ 2.1, i.e. CUDA 12.2 or newer**; older 12.x narrows counts to 32-bit (fine below 2^31 rows) and 11.x is not supported. Only 13.0 is tested by us — if you build on 12.x, please open an issue with your `nvcc --version` either way. C++17 host + device. |
| **NVIDIA driver** | **580.x** (verified) | Any driver that supports your toolkit (NVIDIA's minimum for 13.0 is R580; for 12.x, R525+). Runtime linking: `-DGPUDB_CUDA_STATIC_RUNTIME=ON` (what the registry build in the root `Makefile` uses; off by default in `scripts/build.sh`) bakes `cudart` into the extension, so the only runtime dependency is `libcuda.so` from the driver — and the extension still loads on machines with no GPU/driver, falling back to CPU. |
| **GPUs** | **sm_75 – sm_90**: Turing (T4, RTX 20xx), Ampere (A100, RTX 30xx), Ada (RTX 40xx, L4/L40), Hopper (H100) | Default fatbin: `75;80;86;89;90`, each with SASS + PTX. Newer parts (Blackwell / RTX 50xx, sm_100+) load via PTX JIT from `compute_90` — should work, not yet measured. **Volta (sm_70) and older are not supported**: CUDA 13 dropped them from `nvcc`. Override with `-DCMAKE_CUDA_ARCHITECTURES=...` or `CUDAARCHS=...` (the Colab notebook builds `CUDAARCHS=75` for its T4). |

Verified configuration: RTX 4090 Laptop (sm_89, 16 GB), CUDA 13.0.88, driver
580.x, Linux — every CUDA number in this README and BENCHMARK.md comes from
that box. `SELECT gpu_build_info();` reports whether any given binary was
compiled with CUDA and which backend it picked at runtime.

### Option D (TPC-H reproducibility)

```bash
# get TPC-H SF1 data (downloads DuckDB CLI to .tools/, ~1 GB lineitem)
SF=1 ./scripts/gen_tpch.sh

duckdb -unsigned -c "LOAD '$(pwd)/build-linux/src/extension/gpudb.linux_amd64.duckdb_extension'; \
  SELECT gpu_sum(v) FROM read_parquet('data/tpch_sf1/lineitem_orderkey.parquet') t(v);"
# -> 18005322964949
```

## Run it in CI or a notebook

GitHub's `macos-14`/`macos-15` hosted runners are **Apple Silicon machines** —
gpudb's Metal path runs in free GitHub Actions with zero setup, which makes it
(as far as we know) the only DuckDB extension that does anything special
there. Copy-paste workflows for Apple Silicon runners, Linux runners, Docker,
and self-hosted CUDA boxes: **[docs/CI_RECIPES.md](docs/CI_RECIPES.md)**.

Prefer a notebook? **[examples/gpudb_quickstart.ipynb](examples/gpudb_quickstart.ipynb)**
opens directly in Google Colab — registry install + parity checks anywhere,
plus an optional build-from-source section that runs the CUDA benchmarks on
Colab's free T4 GPU.

## What you get

After build, five CLI tools:

| Tool | What it does |
|---|---|
| **`gpudb-sql`** | Embeds DuckDB, registers `gpu_sum` / `gpu_min` / `gpu_max`, runs SQL from `--sql` or stdin. **Demo this.** |
| `gpudb-bench` | Microbench SUM/MIN/MAX across CPU + CUDA + Metal, cold vs hot resident, on synthetic or `.gpudb` files |
| `gpudb-groupby-bench` | Microbench GROUP BY hash aggregate at varying cardinality |
| `gpudb-window-bench` | Microbench window functions (running sum, partitioned, unbounded frame) |
| `gpudb-hashjoin-bench` | Microbench inner equi-join build × probe across CPU + CUDA |

And a static library `libgpudb` you can embed in any C++ project. See `src/extension/gpu_sum_extension.{cpp,hpp}` for the DuckDB-aware wrapper.

## Architecture

```
┌──────────────────────────────────────────┐
│  DuckDB (host)                           │
│  Parser → Optimizer → Plan → Executor    │
│            │                             │
│            ↓ aggregate function call     │
│  ┌────────────────────────────────────┐  │
│  │  gpudb extension                   │  │
│  │  - gpu_sum / gpu_min / gpu_max     │  │
│  │  - streaming aggregate states      │  │
│  │      ↓ (operator-level / join)     │  │
│  │  ┌───────────────────────────────┐ │  │
│  │  │  libgpudb backend dispatch    │ │  │
│  │  │  ┌───────┐ ┌──────┐ ┌──────┐  │ │  │
│  │  │  │ CUDA  │ │Metal │ │ CPU  │  │ │  │
│  │  │  └───────┘ └──────┘ └──────┘  │ │  │
│  │  └───────────────────────────────┘ │  │
│  └────────────────────────────────────┘  │
└──────────────────────────────────────────┘
```

Backend selection is automatic: CUDA if a device is found at runtime, else Metal if compiled-in, else CPU.

Two SQL paths, by design (v0.4.0):

- **Streaming** `gpu_sum/min/max` — CPU-shaped running accumulators, native
  parity in any query shape. Deliberate: the v0.2.0 numbers in
  [BENCHMARK.md](BENCHMARK.md) showed per-query buffering-for-GPU loses
  3×–110× through this interface.
- **Resident** `gpu_upload` + `gpu_*_resident` — the GPU path with substance:
  pay the transfer once, then reductions run on-device (CUDA and Metal) at
  memory-bandwidth speed with `transfer_ms=0.000`. This is where the
  4–25× numbers above come from.
- **Resident joins (v0.5.0)** `gpu_upload_pair` + `gpu_[left_|semi_|anti_]join_{sum,count}_resident[_f64]`
  — fused join + reduction against a device-cached sorted build side; the
  11–376× join rows above. Row-returning `gpu_join_rows_resident` exists as
  the composability primitive (wins on unified memory, loses to native across
  PCIe — [BENCHMARK.md](BENCHMARK.md) states both).

## Testing
```bash
./build-linux/test/test_gpudb        # unit checks across the backends present at build time (118 CUDA / 137 Metal)
./scripts/run_sql_tests.sh           # SQL-level suite: gpu_sum / min / max / GROUP BY / window / resident / joins
./scripts/join_parity_check.sh       # 11 adversarial join scenarios, native vs gpudb in the same statement
./scripts/local_check.sh             # everything CI would run, end to end
```

The SQL test suite lives in `test/sql/*.test`. Each file is plain SQL with
`-- expect:` lines after each query; the runner reports per-query
PASS / FAIL / GUARDRAIL / SKIP. As of v0.6.0: 126 queries, 0 failures — 113
result checks plus 13 GUARDRAIL cases (deliberate misuse such as a `DOUBLE` join
key or a `NULL` upload name, which the extension must reject with a clear
error; the suite fails if one of them unexpectedly succeeds) — plus full
resident-surface coverage in the community-CI sqllogic suite.

**Reproducibility entry point:** [`scripts/local_check.sh`](scripts/local_check.sh) runs the full pipeline end-to-end (configure → build → unit tests → smoke benchmarks → SQL suite → join parity harness). The hosted CI workflow lives at [`.github/workflows/ci.yml`](.github/workflows/ci.yml) (Linux + macos-15) and runs on every push to `main`.

## Roadmap

### Latest — v0.6.0
- [x] **Resident GROUP BY / top-k from SQL, both backends** — `gpu_groupby_sum_resident` / `gpu_groupby_sum_resident_f64` / `gpu_groupby_count_resident` return `(key, sum, count)` rows sorted by key; `gpu_topk_resident[_f64]` returns `(idx, value)` for `ORDER BY … LIMIT k`. Rides the upload-once model and the same cached device sort the joins use as a build side (one sort serves both). Segmented reduce with no hash table and no atomics on Metal; CUB `reduce_by_key` on CUDA. `_having(name, cmp, threshold)` and `_topk(name, k, order)` forms evaluate `HAVING` / `ORDER BY aggregate LIMIT k` on the device (Metal: block compaction + 8-pass radix select; CUDA: CUB select + 8-pass radix select) so only survivors cross to DuckDB. Verified against native both ways on TPC-H SF1/10/50 — statement time against statement time in the same process: **5.4–6.1× (Metal)** on Q18's inner query with the HAVING on the device, 6–8× on `HAVING count(*) >= 7`, 3.6–4.1× on the top-10 groups by sum; on CUDA the device-side HAVING is **13–24× (SF50) / 17× (SF10)** and the top-10 groups 8–12×; returning all 15M–75M groups is 2.3–2.9× (Metal) and 1.5–2.0× end-to-end on CUDA (~32–47× on-device, bounded by copying 24 bytes per group over PCIe). Honest losing rows kept: low-cardinality GROUP BY on Metal, the first top-k call vs native's zonemap top-k.
- [x] **Composable results** — the GPU produces the rows, DuckDB does the rest: `SELECT key, sum FROM gpu_groupby_sum_resident('l') WHERE sum > 300 ORDER BY sum DESC LIMIT 10` is plain SQL over a small result.
- [x] **Adversarial parity harness for GROUP BY** — `scripts/groupby_parity_check.sh`: 11 scenarios × 7 checks, incl. runs placed exactly on the kernels' 64-chunk / 256-block boundaries; SQL suite gained a `-- setup:` directive so table functions are tested in the documented sequential form.
- [x] **Metal radix-sort fix** — the sort behind the v0.5 join build cache skipped a byte pass whenever min and max agreed on that byte; wrong for keys between them that differ there (TPC-H returnflag/linestatus packed keys). Fixed, regression scenarios in both parity harnesses; exposure of the v0.5.0 Metal binary stated in [KNOWN_ISSUES.md](KNOWN_ISSUES.md).
- [x] **Metal device memory is now released** — the Metal host code was built without ARC since v0.1, so every `MTLBuffer` (resident columns, sort caches, scratch) leaked until process exit; `gpu_drop_resident` now actually frees GPU memory.
- [x] **Pre-release adversarial audit** — 65-agent find/verify pass over the sort, kernels, C-API layer, hybrid planner, SQL semantics vs native (NULLs, overflow, NaN/-0.0), the v0.5 join surface after the sort fix, the CUDA branch (static) and every documentation claim; all confirmed findings fixed or documented in [KNOWN_ISSUES.md](KNOWN_ISSUES.md) before the tag.
- [x] Design: [docs/GROUPBY_RESIDENT_DESIGN.md](docs/GROUPBY_RESIDENT_DESIGN.md).

### Shipped in v0.5.0
- [x] **GPU joins from SQL, both backends** — `gpu_upload_pair` + fused `gpu_[left_|semi_|anti_]join_{sum,count}_resident[_f64]` (inner / left / semi / anti; right / full as documented compositions) and the row-returning `gpu_join_rows_resident`. Sorted-build + binary-search probe with the sorted side cached on the device. Verified against native DuckDB's hash join end-to-end on TPC-H SF10/SF50: **11.7× (Metal) / 27–37× (CUDA)** inner join-sum at SF50, **22× / ~376×** on the EXISTS semi-join; i64 bit-exact, f64 within the 1e-9 relative tolerance contract (measured ≤4e-11). Honest losing row kept: row materialisation across PCIe loses to native on discrete GPUs.
- [x] **Adversarial parity harness** — `scripts/join_parity_check.sh`: 11 scenarios × 12 checks (dup-heavy, Knuth-hash, Zipf skew, int64 boundaries, negative keys, no-match, all-match, inverted sizes), native and gpudb computed in the same statement; passes on both machines.
- [x] **Metal hash join + hybrid join planner + on-device segment reduce** — contributed by [@lmangani](https://github.com/lmangani) ([PR #43](https://github.com/singhpratech/duckdbgpumetaldbram/pull/43)); this release lands that commit as the base of the join stack.
- [x] **Colab notebook runs real CUDA** — requests a T4 runtime automatically, builds with `GPUDB_REQUIRE_CUDA=1` (configure fails loudly instead of silently falling back to CPU), and the test cell asserts the CUDA backend actually ran.

### Shipped in v0.4.0
- [x] **Resident-column SQL surface** — `gpu_upload` / `gpu_sum_resident` / `gpu_min_resident` / `gpu_max_resident` / `gpu_sum_resident_f64` / `gpu_resident_info` / `gpu_last_stats` / `gpu_drop_resident` / `gpu_build_info`. The GPU genuinely executes SQL reductions on both CUDA and Metal — up to **25×** over native (see Numbers). Hardened by a three-reviewer adversarial pass pre-release: buffer-pool cap (window-frame O(n²) OOM → clean error), mixed-name/NULL-name guards, defined overflow wrap, truthful dispatch stats.
- [x] **CUDA-ready community build** — the root Makefile auto-detects nvcc with a statically linked CUDA runtime (no libcuda/libcudart dynamic deps; loads clean on GPU-less machines) so the registry's Linux binary flips to CUDA automatically when the registry's build tooling ships its CUDA toolchain. Also fixed a CMake ordering bug that had every prior CUDA build shipping single-arch fatbins.
- [x] **Full dual-platform benchmark record** — TPC-H SF1→SF100, six columns, correctness-gated, both backends, in [BENCHMARK.md](BENCHMARK.md).
- [x] [Community Extensions PR #2503](https://github.com/duckdb/community-extensions/pull/2503) **merged** (2026-08-17) — the registry now serves **v0.4.0**, resident-column surface included.

### In flight
- [x] **v0.7 milestone 0b — residency prerequisite** (PR): the resident registry keyed per database with identity tags and origin, per-set state (`ready` = uploaded *and* prepared, `stale`, epoch, hits, references), `gpu_residents()`, `gpu_assert_rows()` as the in-statement staleness guard, `gpu_invalidate()`, `gpu_prepare_resident()`; a lock-free `gpu_upload` update path (the one contended atomic it had cost a 60M-row upload 1.8 s → 0.36 s and starved concurrent native queries 3–7×), segmented host buffers with no combine copy, and on CUDA the pair split on the device and uploads on the column's own stream. `scripts/residency_gate.sh` measures a native statement during a concurrent upload against DuckDB's own `list()` as the control. Design: [docs/TRANSPARENT_DESIGN.md](docs/TRANSPARENT_DESIGN.md) §5.3–§5.6.
- [x] **v0.7 milestone 2 (extension side) — the statement rewriter** (PR): `gpu_rewrite_ast(tree, context)`, a pure scalar over DuckDB's own `json_serialize_sql` tree, turns `SELECT k, sum(v) FROM t GROUP BY k [HAVING sum(v) > c] [ORDER BY sum(v) DESC LIMIT n]` into the resident table function plus the `gpu_assert_rows` guard, with native output names and types (HUGEINT sums, DECIMAL(38,s) for DECIMAL payloads with exactly rescaled thresholds), and returns every other statement unchanged with a reason. Verified three ways (native / rewritten / explicit `gpu_*`, `scripts/rewrite_parity_check.sh`) incl. TPC-H SF1 Q18-inner; 0.045 ms to reject a statement, 0.14 ms to rewrite one. The Python wrapper that drives it (classification, name resolution, template cache, fallback) is the macOS side's PR.
- [x] **v0.7 milestone 0c — upload sessions** (PR): `gpu_upload_begin(tag)` / segment `gpu_upload_pair`/`gpu_upload` statements append to the open session / `gpu_upload_finish(tag)` → one device copy + `prepare()`, `gpu_upload_abort`, `gpu_upload_status` (JSON). The wrapper uploads a table as short row-id segments (4.7–12.9 ms of scan each at SF10) instead of one multi-second scan, so an upload never blocks a query on another connection; a failed or interrupted segment leaves the session untouched and re-runs. Design: [docs/TRANSPARENT_DESIGN.md](docs/TRANSPARENT_DESIGN.md) §5.5.
- [ ] **Community packaging phase 2** — `requires_toolchains: "python3;cuda"` registry update (the build side shipped in v0.4.0; the token activates when the registry adopts CUDA-capable ci-tools — `SELECT gpu_build_info();` shows what any given binary carries).

### Next (v0.7.0 — directional)
- [ ] **Transparent execution** — plain `SELECT k, sum(v) FROM t GROUP BY k` routed to the resident operators without calling `gpu_*` functions, with everything unsupported running native. The loadable extension stays on the stable DuckDB **C API**: the statement is rewritten before DuckDB plans it, using DuckDB's own parser (`json_serialize_sql` → a pure `gpu_rewrite_ast` scalar in the extension → `json_deserialize_sql`), driven by a thin client wrapper (`gpudb.connect()`, Python first). No C++ extension API, no per-DuckDB-version binaries. The v0.6 table functions are the rewrite targets. Design: [docs/TRANSPARENT_DESIGN.md](docs/TRANSPARENT_DESIGN.md). Research journal (what was tried, measured and decided, in order): [docs/RESEARCH_NOTES.md](docs/RESEARCH_NOTES.md).
- [ ] GROUP BY over join results as a fused op (join multiplicity × payload → segmented reduce); composite join keys
- [ ] Resident f64 min/max (a small ABI entry)

### Beyond (exploratory)
- [ ] Window functions on GPU as proper operators (not just aggregate-as-window)
- [ ] String / regex operators (libcudf-class functionality on Metal where it doesn't exist)

<details>
<summary><b>Earlier releases</b> (v0.1.0 → v0.3.0 + community-extension milestones)</summary>

### Shipped in v0.3.0
- [x] **Streaming aggregate states** — the SQL aggregate path rewritten from "buffer every value, reduce at finalize" to running accumulators, the same algorithmic shape as native DuckDB. End-to-end on rewritten TPC-H Q6/Q1 and high-cardinality GROUP BY: parity with native (the v0.2.0 buffered path lost 3×–110×; the SF10 GROUP BY cell alone went from 11.05 s to 0.110 s). Full before/after in [BENCHMARK.md](BENCHMARK.md). `GPUDB_FORCE_BACKEND` is a no-op on this path now (it routed the deleted machinery).
- [x] **`gpu_min(DOUBLE)` / `gpu_max(DOUBLE)`** — all three aggregates are now overload sets carrying `(BIGINT)->BIGINT` and `(DOUBLE)->DOUBLE`. No backend-interface change was needed under the streaming design. NaN ordering matches native (NaN sorts greatest). Type matrix in [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

### DuckDB Community Extension milestones
- [x] [Community Extensions PR #1898](https://github.com/duckdb/community-extensions/pull/1898) **merged** — `INSTALL gpudb FROM community` is live (no `-unsigned` flag needed), and gpudb is [listed on duckdb.org](https://duckdb.org/community_extensions/extensions/gpudb).
- [x] [Community Extensions PR #2404](https://github.com/duckdb/community-extensions/pull/2404) **merged** — the community build ships **v0.3.0** (streaming aggregates + DOUBLE overloads on all four platforms).

### Shipped in v0.2.0
- [x] **SQL-correct NULL semantics** (PR #44) — `gpu_sum`/`gpu_min`/`gpu_max` over empty or all-NULL input now return SQL `NULL` (not 0), matching native DuckDB on every path: plain aggregate, GROUP BY groups, and window frames.
- [x] **`gpu_sum(DOUBLE) -> DOUBLE`** (PR #45) — a real second overload via the C API aggregate function set. Doubles ride the existing int64 state machinery as raw bit patterns (zero state-layout change); only the finalize differs. `INTEGER`/`SMALLINT`/`TINYINT` work via DuckDB's implicit widening to the `BIGINT` overload (locked in by tests). Type matrix in [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

### Shipped in v0.1.3
- [x] **Hybrid Metal GROUP BY** — 32K-partition slot-lock + radix-opt with auto-dispatch (env override `GPUDB_METAL_GROUPBY_PATH`). Flipped TPC-H SF10 `l_orderkey` (15M unique) from CPU 1.78× faster to Metal 1.30× faster vs DuckDB CPU 16-thread. 9 wins / 1 honest loss on the lineitem scorecard.
- [x] Prebuilt v0.1.3 binaries (Linux CUDA + macOS Metal) attached to the [v0.1.3 release](https://github.com/singhpratech/duckdbgpumetaldbram/releases/tag/v0.1.3).

### Shipped in v0.1.2
- [x] **All 4 known window/GROUP BY bugs fixed** (PR #18, #20, #21, #22 — see KNOWN_ISSUES.md)
- [x] **DuckDB loadable extension actually loads** — `duckdb -unsigned -c "LOAD '/path/to/gpudb.<platform>.duckdb_extension'"` works on Linux (CUDA) and macOS (Metal).

### Shipped v0.1.0 – v0.1.2 (foundation)
- [x] CUDA backend: SUM/MIN/MAX (one-shot + resident)
- [x] CUDA GROUP BY hash aggregate (open-addressing + atomicCAS, ~520 GiB/s on RTX 4090)
- [x] **CUDA hash join probe** (1M build × 10M probe @ 97% sel: 3.7× wall, 107× kernel over CPU)
- [x] Metal backend: SUM/MIN/MAX i64 with real compute pipelines (~470 GiB/s on M4 Max)
- [x] **Metal GROUP BY** via GPU-resident radix sort (wins 4.4–4.8× over CPU at 100M-500M × 1M groups)
- [x] **Multi-aggregate fusion** (SUM+MIN+MAX+COUNT in one pass, 5.3× over CPU fused)
- [x] **Hybrid CPU/GPU planner** (HybridAggregator + DispatchDecision, beats both pure-CPU and pure-GPU at the 1M×1M sweet spot)
- [x] DuckDB extension: gpu_sum / gpu_min / gpu_max with NULL handling + GPUDB_FORCE_BACKEND env var
- [x] CLI: gpudb-bench, gpudb-groupby-bench, gpudb-window-bench, gpudb-hashjoin-bench, gpudb-sql

</details>

## Why DuckDB? Why not a new database?

The 2013-2024 GPU-DB graveyard is real. The wedge that *isn't* in the graveyard:

1. **Apple Silicon backend** — empty field, defining differentiator
2. **DuckDB-native** — no migration, just `LOAD`
3. **Hybrid CPU/GPU planner** — picks CPU when it wins (low cardinality), GPU when it doesn't
4. **Window functions** — Sirius lacks them; high-value for analytics

This combination is unique as of May 2026. See [GOAL.md](GOAL.md) for the full positioning and [BENCHMARK.md](BENCHMARK.md) for reproducible numbers.

## Citing

If you use this project in research or commercial work:
```
gpudb: GPU-accelerated DuckDB extension for NVIDIA CUDA and Apple Silicon Metal.
2026. https://github.com/singhpratech/duckdbgpumetaldbram
```

## Author / blog

Build process, design tradeoffs, and ongoing benchmarks are posted at **[theaivibe.org](https://theaivibe.org)**.

## License

Apache-2.0. See [LICENSE](LICENSE).
