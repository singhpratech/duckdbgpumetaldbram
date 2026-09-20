# gpudb — GPU-accelerated DuckDB on **Apple Silicon Metal + NVIDIA CUDA**

[![DuckDB Community Extension](https://img.shields.io/badge/DuckDB_Community_Extension-gpudb-FFF100?logo=duckdb&logoColor=black)](https://duckdb.org/community_extensions/extensions/gpudb)
[![Latest release](https://img.shields.io/github/v/release/singhpratech/duckdbgpumetaldbram?label=release)](https://github.com/singhpratech/duckdbgpumetaldbram/releases/latest)
[![CI](https://github.com/singhpratech/duckdbgpumetaldbram/actions/workflows/ci.yml/badge.svg)](https://github.com/singhpratech/duckdbgpumetaldbram/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Platforms](https://img.shields.io/badge/platforms-Apple_Silicon_Metal_%7C_Linux_CUDA-8A2BE2)](#quick-start)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/singhpratech/duckdbgpumetaldbram/blob/main/examples/gpudb_quickstart.ipynb)
[![GitHub stars](https://img.shields.io/github/stars/singhpratech/duckdbgpumetaldbram?style=social)](https://github.com/singhpratech/duckdbgpumetaldbram/stargazers)

> **The first SQL execution engine for Apple Silicon GPUs**, built as a DuckDB extension that *also* runs on NVIDIA CUDA. One codebase, two backends, your existing DuckDB queries.

> [!IMPORTANT]
> **New in v0.7 — plain DuckDB SQL runs on the GPU.** No `gpu_*` calls, no query changes: write the SQL you already write, and the GPU answers it when that is measured faster — DuckDB answers everything else, with the same rows either way.

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

## What you'd use it for

gpudb is for **the analytical SQL you already run in DuckDB, over big tables**, and pays most where the same questions come back. The GPU answers when a table is past the one-million-row floor, resident and measured faster; otherwise DuckDB does, with the same rows.

- **🔌 Existing DuckDB apps, BI tools, ORMs** — `duckdb.connect` becomes `gpudb.connect()`; the SQL is untouched. → [Python](docs/USING_PYTHON.md)
- **🐚 Ad-hoc analysis** — the `gpudb` shell, whose footer says where each statement ran, and why. → [The shell](docs/USING_THE_SHELL.md)
- **🔗 Whole multi-table queries** — joins, `EXISTS`/`IN` subqueries, CTEs, views, derived tables, `HAVING`, `ORDER BY … LIMIT`, `count(DISTINCT)`, `DECIMAL`/`DATE`/`VARCHAR` keys: TPC-H SF10 in plain SQL is **19 of 22 on the device, 0 rows differing, 1.3–52.9×** (Metal). → [what runs where](#what-runs-on-the-gpu-and-what-stays-on-duckdb)
- **📊 Dashboards & monitoring** — the same aggregates every few seconds, on the device once the table is resident; asked by name, a resident `SUM` turns a 99 ms scan into **10 ms** at 600M rows (Metal). → [Numbers](#numbers--measured-not-promised)
- **🔬 Notebook exploration** — a whole session on `gpudb.connect()`. On a Mac it exists nowhere else: no other SQL engine uses the Apple Silicon GPU. → [Python](docs/USING_PYTHON.md)
- **📈 High-cardinality GROUP BY / top-k** — `HAVING` and `ORDER BY … LIMIT k` run on the device, so only the survivors come back: TPC-H Q18 in plain SQL is **11.5×** at SF10 (Metal). → [BENCHMARK.md](BENCHMARK.md)
- **💰 DECIMAL / financial data** — scaled integers, 128-bit sums, bit-exact against native, and no per-scan cast. Asked by name, a `DECIMAL`-stored column reduces at **9.9× (Metal) / 25× (CUDA)** where an already-BIGINT one gets 3.3–3.7× / 5.6–10×. → [BENCHMARK.md](BENCHMARK.md)
- **🎯 Membership at scale** — `EXISTS`/`IN` in a `WHERE` becomes a predicate lane on the device (plain-SQL Q4, SF10: **17.1×**); asked by name, `gpu_semi_join_*` measures **22× (Metal) / ~376× (CUDA)** at SF10. → [BENCHMARK.md](BENCHMARK.md)
- **🧰 From any DuckDB client, by name** — the `gpu_*` functions, from the stock CLI or any binding; measured that way, fused join-aggregates run at **11.7× (Metal) / 27–37× (CUDA)** at SF50. → [Three ways in](#three-ways-in)

**What DuckDB keeps answering:** nothing is lost there — same session, same rows, DuckDB's own speed. Window functions, `sum`/`avg` over `DOUBLE`, `median`/`stddev`, prepared-statement parameters, statements inside an explicit transaction, tables under the floor, a handful of groups, and sets over the memory budget all stay on DuckDB ([what runs where](#what-runs-on-the-gpu-and-what-stays-on-duckdb)). The one thing not to do: upload a column by name to ask it a single question — a plain scan wins. [KNOWN_ISSUES.md](KNOWN_ISSUES.md) has every trade-off.

## Plain DuckDB SQL on the GPU — new in v0.7

Plain SQL reaches the GPU through a client that sees the statement before
DuckDB plans it: the `gpudb` shell, or `gpudb.connect()` from Python — the
extension alone gives the functions, not the rewrite. Until now you called
`gpu_*` functions by name; they all still work, from any client, and now the
same questions can be asked in the SQL you already write. Same rows, same
column names, same column types as native, either way. No hints, no schema
changes, nothing to call.

This path is **on by default on both GPUs** — Apple Silicon Metal and NVIDIA
CUDA. It was turned on for CUDA once the measurement was there: the full gate
on an RTX 4090 Laptop ran 1630 cells with **0 slower than native and 0
differing**. `GPUDB_CUDA_EXACT=0` turns the CUDA path off again without a
rebuild ([Platforms and install](docs/INSTALL.md#platforms-and-install) has the
detail, including what a registry install on Linux gives you).

### Try it in a minute

```bash
pip install duckdb-gpudb          # the `gpudb` command, the gpudb module, and the extension
gpudb my.duckdb                   # a shell whose footer says where each statement ran
```
```
gpudb> SELECT l_partkey, sum(l_quantity) FROM lineitem GROUP BY l_partkey;
```

`pip install duckdb-gpudb` is the shortest way to plain SQL on the GPU, because
it carries the matching binary: on Apple Silicon (macOS 15 or later) and on
x86-64 Linux (glibc 2.34 or newer — Ubuntu 22.04 and later) the wheel bundles
the v0.7.0 extension itself. No `INSTALL`, no build, no environment variable.
On any other platform `pip` installs the wrapper alone and the extension comes
from DuckDB's own install:

```sql
INSTALL gpudb FROM community;     -- in any DuckDB ≥ 1.5.5 client
LOAD gpudb;
```

That registry build is what gives **any** DuckDB client the explicit `gpu_*`
functions, wrapper or no wrapper. `FORCE INSTALL gpudb FROM community;` — or
`UPDATE EXTENSIONS;` — replaces an already-installed copy with the newest the
registry has. On Linux a registry binary may carry no CUDA at all: `SELECT
gpu_build_info();` says what the one in front of you has, and [Platforms and
install](docs/INSTALL.md#platforms-and-install) says what to do about it.

There are **two pieces**: the *extension*, which is the GPU code and lives
inside DuckDB, and the *wrapper*, which is the `gpudb` command and
`gpudb.connect()` — the piece that puts plain SQL on the device. A platform
wheel is both in one install. [How to tell it is
working](docs/USING_THE_SHELL.md#how-to-tell-it-is-working) is two commands that
say which piece is missing when one is.

One session on an M4 Max over TPC-H SF1, opened read-only. The first ask lands
on DuckDB while the columns go to the device in idle segments; the next one is
on the GPU:

```
gpudb> SELECT l_partkey, sum(l_quantity) AS qty FROM lineitem GROUP BY l_partkey ORDER BY qty DESC LIMIT 5;
┌───────────┬───────────────┐
│ l_partkey │      qty      │
│   int64   │ decimal(38,2) │
├───────────┼───────────────┤
│    125009 │       1642.00 │
…
└───────────┴───────────────┘

DuckDB (not_resident: the resident set is not ready yet) · 24.1 ms
…
GPU (topk: the resident GROUP BY) · 39.2 ms
```

Nine more runs of it each way, same session, same connection: **median 14.9 ms
with the path off against 8.7 ms with it on**, both series printed whole so the
warm-ups and the wrapper's own measuring run stay visible. The whole session —
the banner, every reason code the footer prints, `.gpu` / `.residents` /
`.memory`, and those eighteen runs — is in the shell guide, with what a
different machine state does to the ratio.

**Nothing you use goes away.** `gpu_upload`, `gpu_upload_pair`, the
`gpu_*_resident` scalars, the fused joins, the resident GROUP BY and top-k table
functions all work in v0.7 exactly as they did in v0.5 / v0.6, from any DuckDB
client, with the same names and the same results.

→ **[The `gpudb` shell](docs/USING_THE_SHELL.md)** ·
**[The Python way](docs/USING_PYTHON.md)** ·
**[Installing, in full](docs/INSTALL.md)**

---

| Where to look | |
|---|---|
| [What you'd use it for](#what-youd-use-it-for) · [Numbers — measured, not promised](#numbers--measured-not-promised) | whether it is for you, and what has been measured |
| [Three ways in](#three-ways-in) | the `gpudb` shell, `gpudb.connect()`, the explicit `gpu_*` functions |
| [The two rules](#the-two-rules) · [What runs on the GPU](#what-runs-on-the-gpu-and-what-stays-on-duckdb) · [How it decides](#how-it-decides) | never slower, never different, and the bounds that enforce it |
| [The resident model](#the-resident-model-in-20-seconds) · [Quick start](#quick-start) · [Architecture](#architecture) · [Testing](#testing) · [Release history](#release-history) | the explicit surface, the install routes, the record |
| **Guides** · [the shell](docs/USING_THE_SHELL.md) · [Python](docs/USING_PYTHON.md) · [installing](docs/INSTALL.md) | every option and dot-command, both install routes, troubleshooting |
| **Measured** · [BENCHMARK.md](BENCHMARK.md) · [KNOWN_ISSUES.md](KNOWN_ISSUES.md) | every number, including the losing ones; every trade-off, reason by reason |
| **Design** · [TRANSPARENT_DESIGN.md](docs/TRANSPARENT_DESIGN.md) · [RESIDENT_COLUMNS_DESIGN.md](docs/RESIDENT_COLUMNS_DESIGN.md) · [RESEARCH_NOTES.md](docs/RESEARCH_NOTES.md) · [the reading guide](docs/README.md) | how a plain `SELECT` reaches the GPU, how columns live there, and the dated journal |

## Numbers — measured, not promised

**Plain SQL, through the transparent path.** TPC-H, every row compared with
native before any time was counted, with every table the query reads **already
resident**. These numbers are from two machines; yours will differ.

| TPC-H | Machine | Queries answered on the GPU | Rows differing | Speed-up on those queries |
|---|---|---:|---:|---:|
| SF1 (6M-row `lineitem`) | MacBook M4 Max · Metal | 17 of 22 | 0 | 1.4× – 13.6× |
| SF10 (60M-row `lineitem`) | MacBook M4 Max · Metal | 19 of 22 | 0 | 1.3× – 52.9× |
| SF1 (6M-row `lineitem`) | RTX 4090 Laptop · CUDA | 17 of 22 | 0 | 0.95× – 23.3× |

The queries that stay on DuckDB are declined on purpose, and which ones stay
depends on the scale factor. **At SF10, three**: Q2 and Q20 each read a
subquery from inside another subquery — the inner statement is correlated and
does not bind on its own, so there is nothing to hand the device — and Q16's
inner `GROUP BY` declines on its own threshold; forced past it, Q16 measures
0.02–0.08×. **At SF1, five**: those three, and Q6 and Q11, which sit
below the measured size floors at 6M rows (Q2 declines on a size floor there
too, before its shape is ever looked at). Each of them runs on DuckDB
unchanged, at DuckDB's speed. The SF10 row is measured at the **default memory
budget** — nothing has to be raised for those 19 queries.

The CUDA row's low end is one query: **Q1 sits at parity on that box** and has
measured either side of 1.0× across runs of the same build, which is exactly
the case the measured rule decides per process. The thresholds both GPUs use
are the Metal-measured ones, verified on one CUDA machine rather than measured
for every GPU — the gate that verified them is the 1630-cell run named above.

Query by query at both scale factors, the exact conditions and what each run
did not record: [BENCHMARK.md](BENCHMARK.md), *TPC-H coverage, the whole 22
through the transparent path* and *the transparent path on CUDA*; the commands
that take both runs again are in the [reading guide](docs/README.md#reproducing-a-number).

### The operators underneath

The explicit `gpu_*` surface, measured on its own — this is where the ratios
above come from.

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

Every row in this section is re-measured on the release commit, on the machine
named in it: the Metal rows on the M4 Max, the CUDA rows on the RTX 4090.

## Three ways in

| | What it is for |
|---|---|
| **The `gpudb` shell** | exploring, ad-hoc SQL, and seeing where every statement ran and why |
| **Python — `gpudb.connect()`** | applications, notebooks, pipelines |
| **Explicit `gpu_*` functions** | any client in any language, full manual control; the only route that works from the stock DuckDB CLI |

All three talk to the same extension and the same resident columns.

### The `gpudb` shell

```bash
gpudb                             # in-memory database
gpudb my.duckdb                   # a file
gpudb my.duckdb --readonly        # nobody writes while you look
python -m gpudb                   # the same entry point
```

→ **[the full guide](docs/USING_THE_SHELL.md)** — the banner, the footer and its
reason codes, `.gpu` / `.residents` / `.memory`, uploading by hand, scripts,
options and keys.

### Python — `gpudb.connect()`

```python
import gpudb

con = gpudb.connect("my.duckdb")
con.execute("SELECT k, sum(v) FROM t GROUP BY k").fetchall()
con.last_rewrite()                # what happened to that statement
```

→ **[the full guide](docs/USING_PYTHON.md)** — every `connect()` option,
`last_rewrite()` and `memory()`, threads and several connections, a run end to
end, writes and transactions.

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

## The two rules

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
An `ORDER BY <aggregate> LIMIT k` whose ordering values tie inside the first *k*
rows goes back to DuckDB for the same reason (`reason == "ties"`): native has no
tie order of its own there, and plain DuckDB above one thread returned 3
different row sets — and up to 6 different orderings — over 20 runs of one such
statement. The device is asked for one row past the limit and stops itself when
it sees the tie, so that is decided against the data on every execution rather
than guessed from the shape. Through `sql()` — the path the `gpudb` shell takes
— the check cannot be a clause of your statement, so the first call of a data
version asks the device once more and the verdict is then remembered until the
data changes: one extra device pass per data version, not one per call.

## What runs on the GPU, and what stays on DuckDB

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

**The size bounds, in two sentences.** Every statement needs a table of at
least `floor_rows` rows (default 1,000,000) behind its answer; above that floor
the plain `GROUP BY`, `HAVING`, top-k, join, global-aggregate, inner-statement
and `count(DISTINCT)` forms each carry their own group floor, output-size cap
and selectivity bound, measured by `scripts/transparent_gate.py` and written
down in `python/gpudb/_thresholds.py`. All of them, with the exact numbers and
the sentence each one prints, are in [KNOWN_ISSUES.md — the size bounds, form
by form](KNOWN_ISSUES.md#the-size-bounds-form-by-form).

### How it decides

- The statement is rewritten **before DuckDB plans it**, through DuckDB's own
  parser (`json_serialize_sql`) and a pure function in the extension
  (`gpu_rewrite_ast`) — no plan surgery, no C++ API.
- Two of the bounds above explain most of what you will see. A key estimated at
  fewer than 1,000 distinct values does not rewrite on a single table — native
  aggregates a tiny integer domain through a perfect hash in 1.5–5 ms per 6M
  rows — but a `VARCHAR` key is exempt for the **plain** form, with no `WHERE`
  at all or under a `WHERE` that keeps at least half the rows with at least two
  computed-expression payloads, because native hashes the strings and evaluates
  the expressions on every row. That is why TPC-H Q1 (two `VARCHAR` keys, eight
  aggregates over expressions, 98% of rows kept) is on the GPU at 3.8× while
  the plain `sum` and `count(*)` over the *same* two keys — the second
  statement of [the run end to end](docs/USING_PYTHON.md#a-run-end-to-end) —
  declines at `6 groups < 1000`: column payloads, no expressions, no exemption.
  And over a **join** there is no group floor at all — native has to run the
  join whatever the group count, so a join returning one group is rewritten.
- Then the run-time measurement above overrides the bounds in either direction.
  `last_rewrite()["detail"]` names the rule that decided, in both directions.
- That measurement times **the path the statement arrived on**. `execute()` and
  `sql()` — the one the shell uses — reach the same templates, but `sql()` hands
  back a lazy relation and so pays for its guards inside the call. What is
  compared against native is therefore what a caller on that path actually pays,
  and a template that only loses through one door is declined at that door
  alone.
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

An upload has two steps an interrupt cannot stop — the copy to the device and
the sort cache built after it — and on a machine whose cores are already taken
those two wait for a quiet moment rather than run beside your statements; the
wait is bounded at 20 seconds per step, after which the step runs anyway, so
residency is delayed on a busy machine and never withheld. Segments also adapt
their size when they rarely fit the window a workload leaves between its
statements, and they are priced from what **this** machine measures: a segment
has a fixed cost no smaller one escapes, so the size stops halving at the
smallest one still worth taking here. When even that does not fit the pauses a
workload leaves, the manager says so — `progress()` reports `starved`, with the
window it measured and where the floor is — and the statements stay on DuckDB:
correct and at native speed, simply not resident. `residency="eager"`, or a
pause long enough for one segment, is the way out.

The memory budget defaults to a quarter of unified memory on Apple silicon and
half of the card's own memory on a discrete GPU (`memory_budget=`, or
`GPUDB_MEMORY_BUDGET_MB`). What it is compared with is the **physical** total —
every store column counted once, plus whatever a set holds of its own — not the
sum of the per-set figures, which are ranking quantities and count a shared
column once per set that reads it. A set that does not fit is refused **before**
the upload, and the refusal is **remembered**: one upload attempt, not one per
statement, until something that could change the answer changes — the data, the
budget, the resident population or the anti-thrash window. Those statements run
on DuckDB, and `last_rewrite()` says `memory` with both sizes in its detail.
What is kept under pressure is decided by measured value per byte rather than by
recency: the sets that save the most DuckDB time per byte stay.

An upload the **device** refuses fails cleanly rather than quietly landing
somewhere slower: the set is not placed in host memory, no store is left with
lanes on both sides, and DuckDB answers the statement. And a statement that runs
out of device **working** memory — a reduce's scratch, a sort's temporaries,
none of which is resident and so none of which a budget over resident bytes can
see — is answered by DuckDB, after which the budget holds back exactly the
shortfall the backend reported as headroom, so the next statement does not walk
into the same wall.

Where the GPU loses is published in the same place it wins: low-cardinality
`GROUP BY` on Metal, whole-column `min` / `max` against DuckDB's zonemaps and
row materialisation across PCIe are all in [BENCHMARK.md](BENCHMARK.md) with
their numbers, and every trade-off, reason by reason, is in
[KNOWN_ISSUES.md](KNOWN_ISSUES.md). One to know: a write made through a raw
`duckdb` cursor on an **in-memory** database is not seen by the staleness guard
(a file-backed database is watched); use the wrapper's own cursors there.

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
needed. The v0.7.0 build carries all 65 `gpu_*` functions — the streaming
aggregates, the resident-column surface, the joins, the resident GROUP BY and
top-k table functions and the exact family the transparent path uses. Which
backends the binary in front of you actually carries is what `SELECT
gpu_build_info();` answers; [docs/INSTALL.md](docs/INSTALL.md#platforms-and-install)
has that and the rest, and `UPDATE EXTENSIONS;` replaces an earlier version.

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

A release binary loads in any DuckDB ≥ 1.2 and needs `-unsigned`; the
community install above needs neither. [Why, and which DuckDB versions the
registry builds for](docs/INSTALL.md#platforms-and-install).

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

**Building with CUDA** needs a supported toolkit, driver and architecture:
[docs/INSTALL.md — CUDA requirements](docs/INSTALL.md#cuda-requirements-build-from-source-on-linux).

### Option D (TPC-H reproducibility)

```bash
# get TPC-H SF1 data (downloads DuckDB CLI to .tools/, ~1 GB lineitem)
SF=1 ./scripts/gen_tpch.sh

duckdb -unsigned -c "LOAD '$(pwd)/build-linux/src/extension/gpudb.linux_amd64.duckdb_extension'; \
  SELECT gpu_sum(v) FROM read_parquet('data/tpch_sf1/lineitem_orderkey.parquet') t(v);"
# -> 18005322964949
```

### The shell and the Python wrapper

The three ways in above are one extension and one wrapper. The wrapper — the
`gpudb` command and `gpudb.connect()`, which is what puts plain SQL on the
device — installs on its own:

```bash
pip install duckdb-gpudb          # the `gpudb` command and the gpudb module
```

On Apple Silicon (macOS 15+) and x86-64 Linux (glibc 2.34+) that wheel also
bundles the v0.7.0 extension, so it is the whole install; everywhere else the
extension comes from `INSTALL gpudb FROM community;` as in Option A, and a
checkout's own build is preferred over the bundled copy when you have one.

Both routes in full, the lookup order, the supported versions, troubleshooting
and upgrading: **[docs/INSTALL.md](docs/INSTALL.md)**.

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

The explicit surface is two SQL paths by design — streaming `gpu_sum/min/max`
at native parity in any query shape, and the resident `gpu_upload` +
`gpu_*_resident` family where the 4–25× numbers come from, with the fused
resident joins on top of it. Each one, and why it is shaped that way:
**[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#the-two-sql-paths)**.

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
| `test_wrapper.py` — the transparent path | 1238 checks, 0 skipped, 0 failing, under DuckDB 1.4.5 and under 1.5.5 | M4 Max |
| `test_residency_policy.py` — the residency policy, on a driven clock | 105 checks, 0 failing | M4 Max |
| `tpch_coverage.py` | SF1 17 of 22 on the device, SF10 19 of 22 at the default budget, 0 rows differing | M4 Max |
| `tpch_coverage.py` | SF1 17 of 22 on the device, 0 rows differing, through `--path execute` and `--path sql` alike | RTX 4090 Laptop |
| `test_wrapper.py` — the transparent path | 1258 checks, 0 skipped, 0 failing | RTX 4090 Laptop |
| `test_gpudb` — unit checks | 752 / 752 | RTX 4090 Laptop, CPU + CUDA |
| `run_sql_tests.sh` | 225 passing, 0 failing | RTX 4090 Laptop |
| `run_sql_tests.sh` guardrails | 45 `expected_fail` cases across the 18 files in `test/sql/` | — |
| `budget_gate.py` — the memory budget under pressure | PASS: 169 statements, 0 rows differing, 0 errors, resident never above the budget | M4 Max and RTX 4090 Laptop |

Each row names the machine that took it. The unit and SQL rows are the RTX
4090's, run there when the CUDA path was turned on by default; `test_gpudb` and
the SQL suite on the M4 Max run in CI on every push and were not re-run by hand
here, and the Mac's two check counts above were taken before the memory-budget
work added its cases, so both are re-counted on the release build. The x86-64 box used
to carry four failures in the segmented-upload cases — the background uploader
never found a quiet window under that test's statement cadence — and once a
segment was priced from what each machine measures rather than from a constant,
that box came back fully green. The check count differs between the two boxes
because the `avg`-over-`DECIMAL` section branches on the host's `long double`.

The SQL suite lives in `test/sql/*.test` — plain SQL with `-- expect:` lines,
reported per query as PASS / FAIL / GUARDRAIL / SKIP; `test/sqllogic/` is a
separate suite in DuckDB's sqllogictest format, run by the community-CI `make
test` path.

**Reproducibility entry point:** [`scripts/local_check.sh`](scripts/local_check.sh) runs the full pipeline end-to-end (configure → build → unit tests → smoke benchmarks → SQL suite → join parity harness). The hosted CI workflow lives at [`.github/workflows/ci.yml`](.github/workflows/ci.yml) (Linux + macos-15) and runs on every push to `main`.

## Release history

**v0.7.0** (2026-09-20) — plain DuckDB SQL on the GPU: a statement rewritten
before DuckDB plans it, the exact operators behind it, the resident column store
with narrow lanes and a value-aware budget, rule 1 as a running measurement, and
the `gpudb` shell and Python package, whose platform wheels carry the extension
binary with them. What shipped, theme by theme:
**[docs/RELEASE_NOTES_v0.7.md](docs/RELEASE_NOTES_v0.7.md)**. Every release
and every entry before it, in full:
**[docs/RELEASE_HISTORY.md](docs/RELEASE_HISTORY.md)**.

## Why DuckDB? Why not a new database?

Because the hard parts — a parser, a planner, a storage format, a type
system, a client ecosystem — already exist and are good. What is missing is
the GPU underneath them, and three things follow from putting it there
instead of beside it:

1. **An Apple Silicon backend.** No other published SQL engine has one.
2. **No migration.** `LOAD` for the explicit functions, one wrapper for plain SQL. Your tables, your clients, your queries.
3. **A decision, not a mode.** The CPU answers where the CPU wins — low cardinality, small tables, selective filters — and the measurement that says so is published, losing rows included.

Where that sits against the other GPU query engines, on axes that are checkable
from their own documentation (checked 2026-09-20; the sources are under the
table):

| | Sirius | cuDF / RAPIDS | HeavyDB | gpudb |
|---|---|---|---|---|
| Runs on an Apple Silicon GPU | no — requires an NVIDIA GPU, compute capability 7.5+ | no — requires an NVIDIA GPU, compute capability 7.0+ | no — NVIDIA GPUs; CPU-only on x86, Power and ARM | **yes — Metal** |
| Runs as a DuckDB extension | yes — loaded into DuckDB, statements intercepted by an optimizer hook | no — a CUDA C++ and Python dataframe library | no — a standalone SQL engine | **yes — loaded into DuckDB; a client rewrites the statement before DuckDB plans it, over the stable C API** |
| CUDA backend | yes | yes | yes | **yes — including plain SQL on the GPU, on by default** |
| What sends work back to the CPU | operators it does not support | an operation cuDF does not implement, or one that raises | operations that cannot run on GPU, and steps needing more memory than the GPU has | **a per-statement speed measurement**, re-taken on your own machine, as well as the shapes it does not express |
| SQL window functions on the GPU | not in its published supported-operator list | not applicable — a dataframe library; it documents a rolling-window API | supported in SQL; the documentation states they are computed in CPU mode | no — they run on DuckDB |
| Apache-2.0 | yes | yes | yes | yes |

**Sources.** Sirius: [README](https://github.com/sirius-db/sirius) (requirements, supported operators, `LOAD … sirius.duckdb_extension`, CPU fallback). cuDF / RAPIDS: [README](https://github.com/rapidsai/cudf), [system requirements](https://docs.nvidia.com/datascience/install/), [how `cudf.pandas` falls back](https://docs.nvidia.com/cudf/latest/cudf_pandas/how-it-works/), [`DataFrame.rolling`](https://docs.nvidia.com/cudf/latest/cudf/api_docs/api/cudf.DataFrame.rolling/index.html). HeavyDB: [README](https://github.com/heavyai/heavydb), [window functions](https://docs.nvidia.com/heavyai/sql/data-manipulation-dml/window-functions), [configuration parameters](https://docs.nvidia.com/heavyai/installation-and-configuration/config-parameters/configuration-parameters-for-heavydb). gpudb's own cells are the coverage table above and the code behind it.

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
gpudb: GPU-accelerated DuckDB extension for Apple Silicon Metal and NVIDIA CUDA.
2026. https://github.com/singhpratech/duckdbgpumetaldbram
```

## Author / blog

Build process, design tradeoffs, and ongoing benchmarks are posted at **[theaivibe.org](https://theaivibe.org)**.

## License

Apache-2.0. See [LICENSE](LICENSE).
