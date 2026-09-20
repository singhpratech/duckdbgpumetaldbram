# The Python way — `gpudb.connect()`

The second of the three ways in: the same decision as the shell, for
applications, notebooks and pipelines, over Apple Silicon Metal or NVIDIA CUDA.
[← back to the README](../README.md) · [the `gpudb` shell](USING_THE_SHELL.md) ·
[installing](INSTALL.md)

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

## Every `connect()` option

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

## Seeing what happened

`con.last_rewrite()` returns the record for the last statement. Its keys are
`rewritten`, `reason`, `detail`, `form`, `tag`, `sql`, `statement`, `engine`,
`round_trip_ms`, `fallback` and `error` — the same fields the shell's `.gpu`
prints, described in [the shell guide](USING_THE_SHELL.md#look-underneath).

`con.memory()` returns `budget`, `evictions`, `evictions_wasted` (evictions
made for a candidate that was then declined — it is counted rather than
assumed, and measures 0) and `sets`, each with its state, the size the wrapper
expected, the size the extension reports, and what the set is worth.
`con.extension_note` is empty while the loaded extension can serve the client,
and one sentence saying why not otherwise.

## The rest of the object

`gpudb.connect()` returns a wrapper, not a subclass, so everything below is
the whole of what it adds. Everything else is DuckDB's.

| | |
|---|---|
| `con.residents()` | `{identity tag: state}` for every set the wrapper is managing. State is `missing`, `pending`, `uploading`, `ready`, `stale` or `failed`. |
| `con.store_columns()` | one dict per resident column — `table`, `column`, `dtype`, `rows`, `width` (bytes a row of the lane is stored at), `bytes`, `prepared`. This is what the shell's `.residents` prints underneath. Returns `[]` when no usable extension is loaded. |
| `con.transparent` | readable and settable. `con.transparent = False` leaves every statement on DuckDB from then on; `True` puts it back. Nothing is dropped either way — resident sets and decisions survive the round trip, which is what makes it usable as an A/B switch. `reason == "off"` is what a statement that would otherwise have been *decided* reports; one that never reaches the decision — `SELECT 1`, or an aggregate over a table below the row floor — still reports `shape` or `threshold`. |
| `con.residency` | read-only: `"background"`, `"eager"` or `"manual"`, as passed to `connect()`. To change it, open another connection. |
| `con.interrupt()` | DuckDB's own interrupt, on this connection's handle. It stops the statement running on **this** connection; a `cursor()` has its own handle and is not affected. |
| `con.duplicate()` | exactly `con.cursor()`, under DuckDB's name for it. |
| `con.cursor()` | another connection over the same database, sharing the resident sets and the memory budget, with its own `last_rewrite()` and its own decision cache. |
| `con.close()` | closes the connection; on the original (not a cursor) it also stops the background uploader. |

## Threads and several connections

Four guarantees, each of them tested:

- **One `Connection` per thread.** `last_rewrite()` is a single record per
  connection object, so two threads sharing one connection overwrite each
  other's. Give each thread its own `con.cursor()`: it shares the resident
  sets, and the shared residency state is the part that is lock-protected.
  That is the shape the suite tests — two threads, twenty statements each on
  one cached template, forty identical answers.
- **Prepared plans are not shared across threads by name.** A plan name is
  never reused for a different statement, and a thread that reaches a name
  just after it was deallocated gets an error, which the wrapper answers by
  running your original statement on DuckDB, like any other error.
- **`interrupt()` is per handle**, as above. It does not stop a background
  upload — the uploader yields to your statements by itself.
- **Writes from another connection** are noticed on a file-backed database
  (the file and its write-ahead log are stat'ed before every rewritten
  statement). On an **in-memory** database there is no file to watch, so a
  write through a raw `duckdb` cursor is invisible to the guard; use the
  wrapper's own cursors there, or a file-backed database.

## A run, end to end

```python
import gpudb

con = gpudb.connect("data/tpch_sf1/tpch.duckdb", read_only=True, residency="eager")
print("extension_note:", repr(con.extension_note))

QUERIES = {
    "top 5 parts by quantity shipped":
        "SELECT l_partkey, sum(l_quantity) AS qty FROM lineitem "
        "GROUP BY l_partkey ORDER BY qty DESC LIMIT 5",
    "pricing summary (a handful of groups)":
        "SELECT l_returnflag, l_linestatus, sum(l_quantity) AS qty, count(*) AS n "
        "FROM lineitem WHERE l_shipdate <= DATE '1998-09-02' GROUP BY 1, 2 ORDER BY 1, 2",
    "median (no kernel)":
        "SELECT l_returnflag, median(l_quantity) FROM lineitem GROUP BY 1 ORDER BY 1",
}

for title, q in QUERIES.items():
    rows = con.execute(q).fetchall()
    r = con.last_rewrite()
    print(f"\n--- {title}")
    for row in rows:
        print("   ", row)
    print("    rewritten:", r["rewritten"], "| reason:", r["reason"], "| form:", repr(r["form"]))
    print("    detail:", r["detail"])

print("\nresidents:", con.residents())
print("memory:", con.memory())
con.close()
```

Run as written on an M4 Max, TPC-H SF1, DuckDB 1.5.5 (the three `[gpudb]
registered …` lines the extension writes to stderr on load come first, as in
the shell):

```
extension_note: ''

--- top 5 parts by quantity shipped
    (125009, Decimal('1642.00'))
    (140633, Decimal('1562.00'))
    (49981, Decimal('1553.00'))
    (149443, Decimal('1537.00'))
    (10426, Decimal('1513.00'))
    rewritten: True | reason:  | form: 'topk'
    detail: the resident GROUP BY

--- pricing summary (a handful of groups)
    ('A', 'F', Decimal('37734107.00'), 1478493)
    ('N', 'F', Decimal('991417.00'), 38854)
    ('N', 'O', Decimal('74476040.00'), 2920374)
    ('R', 'F', Decimal('37719753.00'), 1478870)
    rewritten: False | reason: threshold | form: ''
    detail: 6 groups < 1000

--- median (no kernel)
    ('A', Decimal('26.00'))
    ('N', Decimal('25.00'))
    ('R', Decimal('26.00'))
    rewritten: False | reason: shape | form: ''
    detail: not a shape the rewrite expresses: select expression

residents: {'gpudb:v1:tpch:main:lineitem:20631:l_partkey,l_quantity': 'ready'}
memory: {'budget': 17179869184, 'evictions': 0, 'evictions_wasted': 0, 'sets': {'gpudb:v1:tpch:main:lineitem:20631:l_partkey,l_quantity': {'state': 'ready', 'est_bytes': 217609579, 'bytes': 84017010, 'error': '', 'value': 0.0, 'uses': 1, 'density': 0.0}}}
```

Three statements, three outcomes — a win, a size threshold, and a shape with
no kernel behind it. `6 groups` in the second is the wrapper's
*estimate* of the key's distinct count (three return flags × two line
statuses); the statement actually returns four rows, and a statement is
decided before it runs, on the estimate. Why that estimate declines here while
TPC-H Q1 over the very same two keys runs on the GPU is in
[How it decides](../README.md#how-it-decides).

## Writes, transactions, parameters

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
