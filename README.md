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

On Apple Silicon this path is on by default; on NVIDIA the same path is
implemented and **opt-in** — set `GPUDB_CUDA_EXACT=1` — because the thresholds
it decides with were swept on Metal hardware rather than on CUDA
([Platforms and install](docs/INSTALL.md#platforms-and-install) has the detail).

On TPC-H at SF10 that is **19 of 22 queries on the device, 0 rows differing,
up to 52.9× on an M4 Max** — warm, with the tables already resident — and
17 of 22 at SF1. The ones that stay are declined on purpose, and this README
says which and why.

**Nothing you use goes away.** `gpu_upload`, `gpu_upload_pair`, the
`gpu_*_resident` scalars, the fused joins, the resident GROUP BY and top-k table
functions all work in v0.7 exactly as they did in v0.5 / v0.6, from any DuckDB
client, with the same names and the same results.

### Try it in a minute

```bash
pip install duckdb-gpudb                        # the `gpudb` command and the gpudb module
gpudb -c "INSTALL gpudb FROM community;"        # the extension, into DuckDB
gpudb my.duckdb                                 # a shell whose footer says where each statement ran
```

> **Where this stands today.** The `duckdb-gpudb` package is not on PyPI yet
> and the community registry serves v0.6.0; both change when v0.7.0 is
> released. Until then, route 2 — [from a checkout](docs/INSTALL.md#installing-in-full) — is
> the one that works end to end. On Linux the three lines above install a
> binary that may carry no CUDA at all: `SELECT gpu_build_info();` says what
> the one in front of you has, and [Platforms and
> install](docs/INSTALL.md#platforms-and-install) says what to do about it.

There are **two pieces**, and you need both: the *extension*, which is the GPU
code and lives inside DuckDB, and the *wrapper*, which is the `gpudb` command
and `gpudb.connect()`. The three lines above install one each. Already have an
older gpudb extension? `FORCE INSTALL gpudb FROM community;` replaces it.
[How to tell it is working](docs/USING_THE_SHELL.md#how-to-tell-it-is-working) is two commands that
say which piece is missing when one is, and [Installing, in
full](docs/INSTALL.md#installing-in-full) has both routes, the lookup order and the supported
versions.

### What's in this README

| | |
|---|---|
| [Three ways in](#three-ways-in) | the shell, Python, and the `gpu_*` functions |
| [The shell way](#the-shell-way) | install, start it, read the footer, `.gpu` / `.residents` / `.memory` |
| [The Python way](#the-python-way) | `gpudb.connect()`, every option, a run end to end |
| [Installing, in full](docs/INSTALL.md#installing-in-full) · [Troubleshooting](docs/INSTALL.md#troubleshooting) · [Upgrade, uninstall, turning it off](docs/INSTALL.md#upgrade-uninstall-and-turning-it-off) · [Reporting a bug](docs/INSTALL.md#reporting-a-bug) | both routes, and what to do when it is not doing what you expect |
| [Explicit `gpu_*` functions](#explicit-gpu_-functions--any-duckdb-client-including-the-cli) | the route that needs no wrapper |
| [Measured — TPC-H](#measured--tpc-h-every-row-compared-with-native) | 22 queries at SF1 and SF10, query by query |
| [The two rules](#the-two-rules) · [What runs on the GPU](#what-runs-on-the-gpu-and-what-stays-on-duckdb) · [How it decides](#how-it-decides) | never slower, never different, and the bounds that enforce it |
| [Residency and the memory budget](#residency-and-the-memory-budget) · [Platforms and install](docs/INSTALL.md#platforms-and-install) · [Limits](#limits-and-where-the-gpu-loses) | what lives on the device, and where it loses |
| [What you'd use it for](#what-youd-use-it-for) · [Numbers](#numbers--measured-not-promised) · [The resident model](#the-resident-model-in-20-seconds) | the explicit surface, with its own measurements |
| [Quick start](#quick-start) · [Architecture](#architecture) · [Testing](#testing) · [Release history](#release-history) | the four install routes, and the record |

### Three ways in

| | What it is for |
|---|---|
| **The `gpudb` shell** | exploring, ad-hoc SQL, and seeing where every statement ran and why |
| **Python — `gpudb.connect()`** | applications, notebooks, pipelines |
| **Explicit `gpu_*` functions** | any client in any language, full manual control; the only route that works from the stock DuckDB CLI |

All three talk to the same extension and the same resident columns.

### The shell way

The `gpudb` shell, end to end — how to tell it is working, starting it, the
real session, reading the footer and its reason codes, `.gpu` / `.residents` /
`.memory`, uploading by hand, scripts, options and keys:
**[docs/USING_THE_SHELL.md](docs/USING_THE_SHELL.md)**.

### The Python way

`gpudb.connect()` in full — every option, `last_rewrite()` and `memory()`, the
rest of the object, threads and several connections, a run end to end with its
real output, and writes / transactions / parameters:
**[docs/USING_PYTHON.md](docs/USING_PYTHON.md)**.

### Installing, in full

Both install routes, the lookup order, the supported versions, the platform
detail, troubleshooting, upgrading, what to do when GPU memory is full, and
how to report a bug: **[docs/INSTALL.md](docs/INSTALL.md)**.

### Explicit `gpu_*` functions — any DuckDB client, including the CLI

**Everything you use keeps working.** v0.6.0 registered 38 `gpu_*` functions;
v0.7 registers 65. Nothing was removed and nothing changed shape — every v0.6
function is registered in v0.7 under the same name, with the same return type
and the same parameter types. None of them needs the wrapper. This is
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
--   avgf=53
-- (avgf is the mantissa bits of the host's long double, which is what native
--  finalises an avg in; 53 on arm64, 64 on x86-64)

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

Apple M4 Max, Metal backend (the backend reports 51.8 GiB of device memory) ·
warm, minimum of 5 runs, statement against statement in one process, with
every table the query reads **already resident** · measured 2026-09-19. These
numbers are from one machine; yours will differ. The conditions, the five
queries that stay and what each run did not record are in
[BENCHMARK.md](BENCHMARK.md) under *TPC-H coverage, the whole 22 through the
transparent path*.

```bash
PYTHONPATH=python python3 scripts/tpch_coverage.py --db data/tpch_sf1/tpch.duckdb
PYTHONPATH=python python3 scripts/tpch_coverage.py --db data/tpch_sf10/tpch.duckdb \
    --memory-budget 200GB      # SF10 holds 18.7 GiB; the default budget is smaller
```

The script needs DuckDB's own `tpch` extension for the query texts, and
installs it over the network the first time if `LOAD tpch` fails — so the
first run wants a connection, and later runs do not.

| | Queries answered on the GPU | Rows differing | Speed-up on those queries |
|---|---|---|---|
| TPC-H SF1 (6M-row `lineitem`) | 17 of 22 | 0 | 1.4× – 13.6× |
| TPC-H SF10 (60M-row `lineitem`) | 19 of 22 | 0 | 1.3× – 52.9× |

The queries that stay on DuckDB are declined on purpose, and which ones stay
depends on the scale factor. **At SF10, three**: Q2 and Q20 each read a
subquery from inside another subquery — the inner statement is correlated and
does not bind on its own, so there is nothing to hand the device — and Q16's
inner `GROUP BY` declines on its own threshold; forced past it, Q16 measures
0.02–0.08×. **At SF1, five**: those three, and Q6 and Q11, which sit
below the measured size floors at 6M rows (Q2 declines on a size floor there
too, before its shape is ever looked at). Each of them runs on DuckDB
unchanged, at DuckDB's speed.

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
did, at its usual speed. A `§n` in the Note column is a section of
[docs/TRANSPARENT_DESIGN.md](docs/TRANSPARENT_DESIGN.md), where that row is
argued out with its measurements.

| Feature | Runs on | Note |
|---|---|---|
| **Grouping and aggregation** | | |
| `sum` `count` `count(*)` `min` `max` `avg`, with `GROUP BY` | ✓ GPU | up to eight aggregated columns in one device pass (§4.9). `avg` over `DECIMAL` is finalised by the extension's own `gpu_avg_decimal`, which is how it matches native bit for bit on every platform; against an extension too old to provide it the column is derived in SQL, and the shape declines only where that derivation is not native's own arithmetic (`gpu_build_info()` reporting `avgf=` anything but 53 — x86-64) |
| Aggregates with no `GROUP BY` | ✓ GPU | one fused pass; over a join at any size, over a single table above a measured row and predicate bound — 16M rows, and rows × predicate terms ≥ 60M (§4.12). A bare `count(*)` over a whole table is never rewritten: DuckDB answers it from the table's own row count |
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
| `INNER JOIN` on a unique key, and the comma form | ✓ GPU | the fact-to-dimension shape, materialised on the device (§4.8) |
| `LEFT` / `RIGHT`, many-to-many, `USING`, composite keys | ✓ GPU | answered from an upload of the join's result (§4.13, §4.21) |
| `EXISTS` / `IN` / a correlated scalar subquery in `WHERE` | ✓ GPU | lowered to a predicate lane DuckDB fills once per row (§4.18) |
| Derived tables, views, CTEs | ✓ GPU | folded or spliced in first, then checked against the original with `DESCRIBE` (§4.16, §4.20, §4.22) |
| Aggregation nested inside a statement DuckDB keeps | ✓ GPU | the inner `SELECT` gets its own decision and its own guards (§4.14) |
| `FULL` join, `SEMI` / `ANTI` join **syntax**, `NATURAL` join, cross products | DuckDB | the `EXISTS` / `IN` **forms** above are rewritten; the join keywords are not (§2). `USING` is rewritten over base tables, and declines over a derived table that renamed the join column |
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
- **Size bounds are per form, not one rule.** Every statement needs a table of
  at least `floor_rows` rows (default 1,000,000) behind its answer; above that
  floor the plain `GROUP BY`, `HAVING`, top-k, join, global-aggregate,
  inner-statement and `count(DISTINCT)` forms each carry their own group floor,
  output-size cap and selectivity bound, measured by
  `scripts/transparent_gate.py` and written down in
  `python/gpudb/_thresholds.py`. All of them, with the exact numbers and the
  sentence each one prints, are in [KNOWN_ISSUES.md — the size bounds, form by
  form](KNOWN_ISSUES.md#the-size-bounds-form-by-form).
- Two of those bounds explain most of what you will see. A key estimated at
  fewer than 1,000 distinct values does not rewrite on a single table — native
  aggregates a tiny integer domain through a perfect hash in 1.5–5 ms per 6M
  rows — but a `VARCHAR` key is exempt for the **plain** form, with no `WHERE`
  at all or under a `WHERE` that keeps at least half the rows with at least two
  computed-expression payloads, because native hashes the strings and evaluates
  the expressions on every row. That is why TPC-H Q1 (two `VARCHAR` keys, eight
  aggregates over expressions, 98% of rows kept) is on the GPU at 3.8× while
  the plain `sum` and `count(*)` over the *same* two keys in the example above
  declines at `6 groups < 1000`: column payloads, no expressions, no exemption.
  And over a **join** there is no group floor at all — native has to run the
  join whatever the group count, so a join returning one group is rewritten.
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
18.7 GiB where they held 44.9 before narrow lanes and shedding.

The memory budget defaults to a quarter of unified memory on Apple silicon and
half of device memory on a discrete GPU (`memory_budget=`, or
`GPUDB_MEMORY_BUDGET_MB`). A set that does not fit is refused **before** the
upload and its statements run on DuckDB. What is kept under pressure is decided
by measured value per byte rather than by recency: the sets that save the most
DuckDB time per byte stay. `con.memory()` and the shell's `.memory` print the
budget and what holds it; `.residents` prints the sets and the columns behind
them.

### Platforms and install

Which platform gets what, why the CUDA path is opt-in behind
`GPUDB_CUDA_EXACT=1`, and the install routes:
**[docs/INSTALL.md](docs/INSTALL.md#platforms-and-install)**.
### Limits, and where the GPU loses

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

These are the shapes the explicit `gpu_*` surface was built for: **the same aggregate questions asked of the same big data, over and over**. Upload a column to GPU memory once, and every query after it runs at memory-bandwidth speed with no transfer. The numbers below are measured; the losing shapes are in the same tables.

- **📊 Dashboards & monitoring** — a metrics dashboard re-runs `SUM`s every few seconds. Resident columns turn a 99 ms scan into 10 ms (measured, 600M rows). Refresh every 5 s and the one-time upload pays for itself inside 10 minutes — then every refresh is 4–25× cheaper for as long as the column stays resident.
- **🔬 Notebook exploration** — an analyst slicing data re-aggregates on every idea. Upload at the top of the notebook; the whole session runs GPU-speed. On a Mac this is capability that exists nowhere else: no other SQL engine uses the Apple Silicon GPU.
- **⚡ Serving APIs** — endpoints answering "total X for Y" thousands of times a day. The resident column is a cache that never goes stale-wrong: exact answers, 5–25× lower latency, re-upload in seconds when data refreshes.
- **📈 High-cardinality GROUP BY / top-k** — "quantity per order", "spend per customer", "top 10 by amount" over tens of millions of keys, re-asked as the data is explored. `gpu_groupby_sum_resident` returns `(key, sum, count)` rows (`gpu_groupby_count_resident` returns `(key, count)`) from a cached device sort, and the `_having(name, cmp, threshold)` / `_topk(name, k, order)` forms evaluate `HAVING` and `ORDER BY sum LIMIT k` **on the device** so only the survivors come back: **5.4–6.1× (Metal)** on TPC-H Q18's inner query at SF10–SF50, 6–8× on `HAVING count(*) >= 7`, 3.6–4.1× on the top-10 groups by sum — statement time vs statement time, same process. On CUDA the same device-side HAVING is **13–24× (SF50) / 17× (SF10)** and the top-10 groups 13–17×, because the 1.8 GB result copy over PCIe simply no longer happens. Returning all 15M–75M groups is 2.3–2.9× (Metal) / 1.5× end-to-end on CUDA (~32× on-device, PCIe-bound). Low-cardinality GROUP BY stays a native win on Metal (measured, kept in the table).
- **💰 DECIMAL/financial data** — money columns are stored DECIMAL, and native DuckDB re-casts every value on every scan. `gpu_upload` stores the cast once — it's why our biggest measured wins (9.9× Metal, 25× CUDA) came from the most accounting-shaped column in TPC-H.
- **🎯 Membership at scale (semi / anti join)** — "how much did *these* customers spend?", "which transactions hit the blocklist?", "how many events came from outside the cohort?" Keep the big fact side resident (`gpu_upload_pair`), re-upload only the small, changing set, and ask with `gpu_semi_join_*` / `gpu_anti_join_*`: **22× (Metal) / ~376× (CUDA)** over native on TPC-H SF10 (measured, v0.5.0).
- **🔗 Fact ⋈ dimension rollups** — revenue joined to a filtered orders/customers/dates set, re-asked per filter. `gpu_join_sum_resident` / `gpu_left_join_count_resident` run the fused join-aggregate on the device with the sorted build side cached: **11.7× / 27–37×** at SF50. DOUBLE payloads via the `_f64` variants (within the 1e-9 relative tolerance contract; measured ≤4e-11).

**Not for:** one-shot queries on cold data (transfer loses — the streaming `gpu_sum/min/max` deliberately match native there), or `min`/`max` where DuckDB's statistics answer without scanning, or joins that must return the matched *rows* at scale (`gpu_join_rows_resident` works, but native DuckDB wins on discrete GPUs — use the aggregate variants). [KNOWN_ISSUES.md](KNOWN_ISSUES.md) lists every trade-off.

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

-- v0.7 registry: every resident set, with its identity, state, size and hits
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
parity. Every environment variable the build honours — one line each — is in
[docs/ENVIRONMENT.md](docs/ENVIRONMENT.md). None of them changes an answer:
they change which path runs, how much memory it may use, or what it prints.

### GROUP BY / HAVING / top-k on the device (v0.6.0, unchanged in v0.7)

```sql
-- Upload a (key, payload) pair once; the GPU sorts it once, then every GROUP BY is a
-- segmented reduce over that order, and HAVING / ORDER BY … LIMIT k run on the device
-- so only the survivors come back.
SELECT gpu_upload_pair('l', l_orderkey, l_quantity::BIGINT) FROM lineitem;        -- once
SELECT * FROM gpu_groupby_sum_resident_having('l', '>', 300);   -- TPC-H Q18 inner: 75M groups → 3,182 rows
--   SF50: 78 ms vs 462–476 ms native on M4 Max (6×), 42–78 ms vs ~1 s on RTX 4090 (13–24×)
SELECT * FROM gpu_groupby_sum_resident_topk('l', 10, 'desc');   -- top-10 groups by sum: 4× Metal, 16–17× CUDA
SELECT key, sum, count FROM gpu_groupby_sum_resident('l');      -- or all 75M groups: 2.8× Metal
-- statement vs statement in the same process; verified equal to native GROUP BY both ways
```

### Fused resident joins (v0.5.0, unchanged in v0.7)

```sql
-- Upload key+payload pairs once; the GPU joins and reduces in one pass against
-- a cached sorted build side — no join output materialised.
SELECT gpu_upload_pair('l', l_orderkey, (l_extendedprice*100)::BIGINT) FROM lineitem;  -- once
SELECT gpu_upload('o', o_orderkey) FROM orders;                                          -- once
SELECT gpu_join_sum_resident('l.k', 'l.v', 'o');   -- = sum(...) FROM lineitem JOIN orders
--   SF50: 37 ms vs 429 ms native on M4 Max (11.7×), 27-37 ms vs 998 ms on RTX 4090 (~30×)
-- inner / left / semi / anti × sum(BIGINT) / sum(DOUBLE) / count — all bit-exact
-- or within 1e-9 of native, verified by scripts/join_parity_check.sh
```

## Why this exists

Two things are missing from the GPU-database landscape, and both are addressed
by writing an extension rather than an engine.

**No published SQL engine targets Apple Silicon GPUs.** Sirius (UW + NVIDIA,
CIDR 2026) is CUDA-only; cuDF is CUDA-only. Apple Silicon's unified memory —
up to 512 GB at 819 GB/s on an M3 Ultra — is a column store's natural home:
there is no PCIe hop, so a table the CPU already holds is a table the GPU can
read. `gpudb` wires that into a database, with the same operators on CUDA.

**An engine is mostly not the operators.** A parser, an optimizer, a storage
format, a type system and a client ecosystem are the bulk of the work, and
DuckDB has them. As an extension, gpudb adds the GPU underneath them and
nothing else: DuckDB still answers every statement the GPU does not take, at
its usual speed, and the GPU takes only the shapes it is measured to win. The
extension reaches DuckDB through its stable C API, so one binary keeps working
across DuckDB versions, and turning it off leaves a plain DuckDB session.

Operator-level benchmarks (GROUP BY, multi-aggregate fusion, hash join) live in [BENCHMARK.md](BENCHMARK.md)'s earlier entries.

## Quick start

### Option A — install from the DuckDB community repo (recommended)

```sql
INSTALL gpudb FROM community;
LOAD gpudb;
SELECT gpu_sum(value::BIGINT) FROM range(1000000) AS t(value);
-- -> 499999500000
```

Works in any DuckDB ≥ 1.5.5 client (CLI, Python, etc.), signed, no flags
needed — that is the range the registry builds gpudb for. The registry binary
carries the **full Metal backend on Apple Silicon**. A registry Linux binary
may report `compiled=cpu`, in which case every `gpu_*` function works and
returns the same results, with `gpu_last_stats()` saying `backend=CPU`.
**`SELECT gpu_build_info();` is the answer for whichever binary is in front of
you**: `compiled=` lists what it was built with and `runtime=` the backend it
chose. For the CUDA backend take the release binary (Option B; statically
linked CUDA runtime, needs only a driver) or build from source with `nvcc`.
The v0.7.0 build carries all 65 `gpu_*` functions: the
streaming aggregates, the full resident-column surface (`gpu_upload`,
`gpu_sum_resident`, `gpu_residents`, `gpu_build_info`, …), the GPU join
functions (`gpu_upload_pair`, `gpu_join_*_resident`, `gpu_join_rows_resident`,
`gpu_inner_join`), the resident GROUP BY / top-k table functions
(`gpu_groupby_*_resident`, `gpu_topk_resident`) and the exact family the
transparent path uses (`gpu_upload_*_exact`, `gpu_groupby_exact_*`,
`gpu_agg_exact_global`, `gpu_rewrite_ast`). Installed an earlier version?
`UPDATE EXTENSIONS;` pulls the latest.

The transparent path is not reached by `LOAD gpudb` alone — that gives the
explicit functions. For plain SQL on the GPU, use the `gpudb` shell or
`gpudb.connect()` from `pip install duckdb-gpudb` over this same extension.

### Option B — load a prebuilt release binary

Download the platform binary from the [latest release](https://github.com/singhpratech/duckdbgpumetaldbram/releases/latest), then:

```bash
# Linux (RTX/CUDA)
duckdb -unsigned -c "LOAD '/path/to/gpudb.linux_amd64.duckdb_extension'; \
  SELECT gpu_sum(value::BIGINT) FROM range(1000000) AS t(value);"
# -> [gpudb] registered gpu_sum / gpu_min / gpu_max (BIGINT,DOUBLE) streaming aggregates (backend=CUDA)
# -> 499999500000
```

The loadable extension is built against the stable C API v1.2.0, so a release
binary loads in any DuckDB ≥ 1.2; the community install above is what needs
≥ 1.5.5, because the registry builds gpudb separately for each DuckDB version
from 1.5.5 on.
Release binaries track the latest tag. `LOAD` needs `-unsigned` here because
release-page binaries are unsigned — the community install above does not.

### Option C — build from source

```bash
git clone https://github.com/singhpratech/duckdbgpumetaldbram.git
cd duckdbgpumetaldbram

# Linux (CUDA): one-time toolkit install if needed
# sudo apt install -y cuda-toolkit-13-0
# export PATH=/usr/local/cuda/bin:$PATH

# macOS (Metal): brew install cmake

# fetch pre-built libduckdb + headers into third_party/duckdb-libs/.
# build.sh only builds the loadable extension when these are present.
./scripts/get_duckdb_libs.sh

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
| **GPUs** | **sm_75 – sm_90**: Turing (T4, RTX 20xx), Ampere (A100, RTX 30xx), Ada (RTX 40xx, L4/L40), Hopper (H100) | Default fatbin: `75;80;86;89;90`, each with SASS + PTX. Newer parts (Blackwell / RTX 50xx, sm_100+) load via PTX JIT from `compute_90` — should work; not measured here. **Volta (sm_70) and older are not supported**: CUDA 13 dropped them from `nvcc`. Override with `-DCMAKE_CUDA_ARCHITECTURES=...` or `CUDAARCHS=...` (the Colab notebook builds `CUDAARCHS=75` for its T4). |

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
./build-macos/test/test_gpudb        # unit checks across the backends present at build time
./scripts/run_sql_tests.sh           # SQL-level suite: gpu_sum / min / max / GROUP BY / window / resident / joins / exact
./scripts/join_parity_check.sh       # 11 adversarial join scenarios, native vs gpudb in the same statement
./scripts/groupby_parity_check.sh    # 11 GROUP BY scenarios, including the ones that must NOT rewrite
./scripts/local_check.sh             # everything CI would run, end to end

PYTHONPATH=python python3 -m pytest python/tests/test_wrapper.py   # the transparent path
PYTHONPATH=python python3 scripts/tpch_coverage.py                 # the 22 TPC-H queries
```

| Suite | Result | Where |
|---|---|---|
| `test_wrapper.py` — the transparent path | 1157 checks, 0 skipped, 0 failing, under DuckDB 1.4.5 and under 1.5.5 | M4 Max |
| `tpch_coverage.py` | SF1 17 of 22 on the device, SF10 19 of 22, 0 rows differing | M4 Max |
| `tpch_coverage.py` with `GPUDB_CUDA_EXACT=1` | SF1 17 of 22 on the device, 0 rows differing | RTX 4090 Laptop |
| `test_gpudb` — unit checks | 730 / 730 | RTX 4090 Laptop, CPU + CUDA |
| `run_sql_tests.sh` | 224 passing, 0 failing with `GPUDB_CUDA_EXACT=1` | RTX 4090 Laptop |
| `run_sql_tests.sh` guardrails | 45 `expected_fail` cases across the 18 files in `test/sql/` | — |

The wrapper and coverage rows were re-measured on the M4 Max on this commit;
the unit and SQL rows are the RTX 4090's, which is where they were last run
against this code. `test_gpudb` and the SQL suite on the M4 Max run in CI on
every push and were not re-run by hand here. On the x86-64 box the wrapper
suite carries **4 failures** in the segmented-upload cases — the background
uploader never finds a quiet window under that test's statement cadence — with
the measurements in [BENCHMARK.md](BENCHMARK.md). They are unrelated to the
exact path (they reproduce with the flag unset) and do not appear on Apple
silicon.

The SQL test suite lives in `test/sql/*.test`. Each file is plain SQL with
`-- expect:` lines after each query; the runner reports per-query
PASS / FAIL / GUARDRAIL / SKIP. The guardrail cases are deliberate misuse — a
`DOUBLE` join key, a `NULL` upload name — that the extension must reject with a
clear error; the suite fails if one of them unexpectedly succeeds.
`test/sqllogic/` is a separate suite in DuckDB's sqllogictest format, run by
the community-CI `make test` path.

**Reproducibility entry point:** [`scripts/local_check.sh`](scripts/local_check.sh) runs the full pipeline end-to-end (configure → build → unit tests → smoke benchmarks → SQL suite → join parity harness). The hosted CI workflow lives at [`.github/workflows/ci.yml`](.github/workflows/ci.yml) (Linux + macos-15) and runs on every push to `main`.

## Release history

### Latest — v0.7.0

- [x] **Plain DuckDB SQL on the GPU** — a statement is rewritten before DuckDB
  plans it, through DuckDB's own parser and the pure `gpu_rewrite_ast` scalar in
  the extension, and answered on the device when that is measured faster.
  Driven by a client that sees the statement first: the `gpudb` shell and
  `gpudb.connect()` (Python). The loadable extension stays on the stable C API —
  no C++ extension API, no plan surgery, no per-DuckDB-version binaries.
- [x] **Exact operators** — `GROUP BY` with `sum` / `count` / `count(*)` /
  `min` / `max` / `avg`, native NULL semantics, 128-bit sums, `DECIMAL` as
  scaled integers, up to eight payloads in one device pass; a fused `WHERE`
  mask; an aggregate with no `GROUP BY` in one pass; device `HAVING` and top-k.
  CPU reference, Metal, and CUDA (opt-in, `GPUDB_CUDA_EXACT=1`).
- [x] **Keys that real SQL uses** — integer, `DATE`, `TIMESTAMP`, `DECIMAL` and
  `VARCHAR` keys; one to eight of them, up to three packed into a 64-bit key and
  a hashed tuple with a dictionary above that; `GROUP BY ALL`, ordinals,
  `ORDER BY ALL`, `SELECT DISTINCT`, `FILTER (WHERE …)`, `count_if`,
  `bool_and` / `bool_or`, expressions inside and over aggregates, compound
  `HAVING`, `count(DISTINCT x)`.
- [x] **Joins, subqueries, views, CTEs** — inner equi-joins onto a unique key
  materialised on the device; other INNER / LEFT / RIGHT and many-to-many joins
  answered from an upload of the join's result; `EXISTS` / `IN` / correlated
  scalar subqueries as `WHERE` terms lowered to predicate lanes; derived tables,
  views and CTEs folded in and checked against the original with `DESCRIBE`;
  an aggregating `SELECT` nested inside a statement DuckDB keeps gets its own
  decision.
- [x] **A resident column store** — one copy per column in row-id order, each
  lane at the narrowest signed width its values fit (the 22 TPC-H queries at
  SF10 hold 18.7 GiB where they held 44.9), shared between statements, built in
  idle row-id segments so an upload never runs beside a query you are waiting
  for, admitted by a memory budget that keeps what saves the most DuckDB time
  per byte.
- [x] **Rule 1 as a running measurement, not only a table** — per-machine
  re-measurement of each statement template on a side cursor, re-checked every
  60 seconds, with `last_rewrite()["detail"]` naming the rule in both
  directions.
- [x] **The `gpudb` shell and the Python package** — a SQL shell whose footer
  says where each statement ran and why, `.gpu` / `.residents` / `.memory`, and
  the `duckdb-gpudb` distribution (import name `gpudb`).
- [x] Design: [docs/TRANSPARENT_DESIGN.md](docs/TRANSPARENT_DESIGN.md),
  [docs/RESIDENT_COLUMNS_DESIGN.md](docs/RESIDENT_COLUMNS_DESIGN.md);
  journal: [docs/RESEARCH_NOTES.md](docs/RESEARCH_NOTES.md);
  release notes: [docs/RELEASE_NOTES_v0.7.md](docs/RELEASE_NOTES_v0.7.md).

### Shipped in v0.6.0
- [x] **Resident GROUP BY / top-k from SQL, both backends** — `gpu_groupby_sum_resident` / `gpu_groupby_sum_resident_f64` / `gpu_groupby_count_resident` return `(key, sum, count)` rows sorted by key; `gpu_topk_resident[_f64]` returns `(idx, value)` for `ORDER BY … LIMIT k`. Rides the upload-once model and the same cached device sort the joins use as a build side (one sort serves both). Segmented reduce with no hash table and no atomics on Metal; CUB `reduce_by_key` on CUDA. `_having(name, cmp, threshold)` and `_topk(name, k, order)` forms evaluate `HAVING` / `ORDER BY aggregate LIMIT k` on the device (Metal: block compaction + 8-pass radix select; CUDA: CUB select + 8-pass radix select) so only survivors cross to DuckDB. Verified against native both ways on TPC-H SF1/10/50 — statement time against statement time in the same process: **5.4–6.1× (Metal)** on Q18's inner query with the HAVING on the device, 6–8× on `HAVING count(*) >= 7`, 3.6–4.1× on the top-10 groups by sum; on CUDA the device-side HAVING is **13–24× (SF50) / 17× (SF10)** and the top-10 groups 8–12×; returning all 15M–75M groups is 2.3–2.9× (Metal) and 1.5–2.0× end-to-end on CUDA (~32–47× on-device, bounded by copying 24 bytes per group over PCIe). The losing rows are kept: low-cardinality GROUP BY on Metal, and the first top-k call against native's zonemap top-k.
- [x] **Composable results** — the GPU produces the rows, DuckDB does the rest: `SELECT key, sum FROM gpu_groupby_sum_resident('l') WHERE sum > 300 ORDER BY sum DESC LIMIT 10` is plain SQL over a small result.
- [x] **Adversarial parity harness for GROUP BY** — `scripts/groupby_parity_check.sh`: 11 scenarios × 7 checks, incl. runs placed exactly on the kernels' 64-chunk / 256-block boundaries; SQL suite gained a `-- setup:` directive so table functions are tested in the documented sequential form.
- [x] **Metal radix-sort fix** — the sort behind the v0.5 join build cache skipped a byte pass whenever min and max agreed on that byte; wrong for keys between them that differ there (TPC-H returnflag/linestatus packed keys). Fixed, regression scenarios in both parity harnesses; exposure of the v0.5.0 Metal binary stated in [KNOWN_ISSUES.md](KNOWN_ISSUES.md).
- [x] **Metal device memory is now released** — the Metal host code was built without ARC since v0.1, so every `MTLBuffer` (resident columns, sort caches, scratch) leaked until process exit; `gpu_drop_resident` now actually frees GPU memory.
- [x] **Pre-release adversarial audit** — a find/verify pass over the sort, kernels, C-API layer, hybrid planner, SQL semantics vs native (NULLs, overflow, NaN/-0.0), the v0.5 join surface after the sort fix, the CUDA branch (static) and every documentation claim; all confirmed findings fixed or documented in [KNOWN_ISSUES.md](KNOWN_ISSUES.md) before the tag.
- [x] Design: [docs/GROUPBY_RESIDENT_DESIGN.md](docs/GROUPBY_RESIDENT_DESIGN.md).

### Shipped in v0.5.0
- [x] **GPU joins from SQL, both backends** — `gpu_upload_pair` + fused `gpu_[left_|semi_|anti_]join_{sum,count}_resident[_f64]` (inner / left / semi / anti; right / full as documented compositions) and the row-returning `gpu_join_rows_resident`. Sorted-build + binary-search probe with the sorted side cached on the device. Verified against native DuckDB's hash join end-to-end on TPC-H SF10/SF50: **11.7× (Metal) / 27–37× (CUDA)** inner join-sum at SF50, **22× / ~376×** on the EXISTS semi-join; i64 bit-exact, f64 within the 1e-9 relative tolerance contract (measured ≤4e-11). The losing row is kept: row materialisation across PCIe loses to native on discrete GPUs.
- [x] **Adversarial parity harness** — `scripts/join_parity_check.sh`: 11 scenarios × 12 checks (dup-heavy, Knuth-hash, Zipf skew, int64 boundaries, negative keys, no-match, all-match, inverted sizes), native and gpudb computed in the same statement; passes on both machines.
- [x] **Metal hash join + hybrid join planner + on-device segment reduce** — contributed by [@lmangani](https://github.com/lmangani) ([PR #43](https://github.com/singhpratech/duckdbgpumetaldbram/pull/43)); this release lands that commit as the base of the join stack.
- [x] **Colab notebook runs real CUDA** — requests a T4 runtime automatically, builds with `GPUDB_REQUIRE_CUDA=1` (configure fails loudly instead of silently falling back to CPU), and the test cell asserts the CUDA backend actually ran.

### Shipped in v0.4.0
- [x] **Resident-column SQL surface** — `gpu_upload` / `gpu_sum_resident` / `gpu_min_resident` / `gpu_max_resident` / `gpu_sum_resident_f64` / `gpu_resident_info` / `gpu_last_stats` / `gpu_drop_resident` / `gpu_build_info`. The GPU genuinely executes SQL reductions on both CUDA and Metal — up to **25×** over native (see Numbers). Hardened by a three-reviewer adversarial pass pre-release: buffer-pool cap (window-frame O(n²) OOM → clean error), mixed-name/NULL-name guards, defined overflow wrap, truthful dispatch stats.
- [x] **CUDA-ready community build** — the root Makefile auto-detects nvcc with a statically linked CUDA runtime (no libcuda/libcudart dynamic deps; loads clean on GPU-less machines) so the registry's Linux binary flips to CUDA automatically when the registry's build tooling ships its CUDA toolchain. Also fixed a CMake ordering bug that had every prior CUDA build shipping single-arch fatbins.
- [x] **Full dual-platform benchmark record** — TPC-H SF1→SF100, six columns, correctness-gated, both backends, in [BENCHMARK.md](BENCHMARK.md).
- [x] [Community Extensions PR #2503](https://github.com/duckdb/community-extensions/pull/2503) **merged** (2026-08-17) — the registry now serves **v0.4.0**, resident-column surface included.

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
- [x] **Hybrid Metal GROUP BY** — 32K-partition slot-lock + radix-opt with auto-dispatch (env override `GPUDB_METAL_GROUPBY_PATH`). Flipped TPC-H SF10 `l_orderkey` (15M unique) from CPU 1.78× faster to Metal 1.30× faster vs DuckDB CPU 16-thread. 9 wins / 1 loss on the lineitem scorecard, both published.
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

Because the hard parts — a parser, a planner, a storage format, a type
system, a client ecosystem — already exist and are good. What is missing is
the GPU underneath them, and three things follow from putting it there
instead of beside it:

1. **An Apple Silicon backend.** No other published SQL engine has one.
2. **No migration.** `LOAD` for the explicit functions, one wrapper for plain SQL. Your tables, your clients, your queries.
3. **A decision, not a mode.** The CPU answers where the CPU wins — low cardinality, small tables, selective filters — and the measurement that says so is published, losing rows included.

Where that sits against the other GPU query engines, on the axes that are
checkable from their own documentation:

| | Sirius | cuDF / RAPIDS | HeavyDB | gpudb |
|---|:-:|:-:|:-:|:-:|
| Apple Silicon (Metal) backend | no | no | no | **yes** |
| Runs as a DuckDB extension (no migration) | yes | no | no | **yes** |
| CUDA backend | yes | yes | yes | yes |
| Falls back to the CPU per statement, on a measurement | partial | no | no | **yes** |
| Window functions on the GPU | no | partial | yes | no — they run on DuckDB |
| Apache-2.0 | yes | yes | yes | yes |

[BENCHMARK.md](BENCHMARK.md) has the reproducible numbers behind our column.

## Credits

- **Metal hash join, the hybrid join planner and the on-device segment reduce**
  — contributed by [@lmangani](https://github.com/lmangani) in
  [PR #43](https://github.com/singhpratech/duckdbgpumetaldbram/pull/43), and the
  base of the join stack from v0.5.0 onward.
- The DuckDB team, for a stable C extension API that one binary can keep
  working against across versions, and for `json_serialize_sql` — the
  transparent path is built on DuckDB's own parser rather than a second one.

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
