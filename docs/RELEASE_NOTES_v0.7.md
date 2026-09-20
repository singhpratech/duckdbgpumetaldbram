# gpudb v0.7.0 — plain DuckDB SQL on the GPU

Until v0.6 you called `gpu_*` functions by name. From v0.7 you write the SQL
you already write — in the `gpudb` shell, or through `gpudb.connect()` — and
the GPU answers it when that has been measured faster on your machine.
Everything else runs on DuckDB, untouched. Same rows, same column names, same
column types, either way.

Everything the explicit surface did in v0.5 / v0.6 still works, unchanged:
`gpu_upload`, `gpu_upload_pair`, the `gpu_*_resident` scalars, the fused joins,
the resident GROUP BY and top-k table functions. The loadable extension still
touches DuckDB through the stable C API only (the C_STRUCT ABI): it links no
libduckdb, includes no DuckDB C++ headers and does no plan surgery.

**Measured** on an Apple M4 Max (Metal), TPC-H, warm, with every table the
query reads already resident, minimum of 5 runs, every row compared with
native (conditions and the per-query tables: `BENCHMARK.md`):

| | Queries on the GPU | Rows differing | Speed-up on those queries |
|---|---|---|---|
| SF1 | 17 of 22 | 0 | 1.4× – 13.6× |
| SF10 | 19 of 22 | 0 | 1.3× – 52.9× |

---

## The two rules

**Never slower than DuckDB.** Per statement, per shape, per size. The bounds a
statement is admitted by come from a sweep measured against native
(`scripts/transparent_gate.py`); one row under 1.0× stops a release, and the
losing rows stay published in `BENCHMARK.md`. On your machine the decision is
re-taken: after a template's first three rewritten runs the wrapper times the
native form once on a side cursor in your own process, hands the template back
to DuckDB if the rewritten runs were not faster, and re-measures every 60
seconds (#107, #101, #124).

**Never a different answer.** Integer and `DECIMAL` aggregates are bit-exact,
with 128-bit sums on the device. Every scenario runs three ways in one process
— native, rewritten, and the explicit `gpu_*` calls — and all three must agree
on ordered rows, names and `typeof()` of every column; each rewrite is also
checked against the original text with `DESCRIBE`. Where "the same as native"
is not definable — `sum(DOUBLE)`, which DuckDB itself computes
order-dependently — the shape is never rewritten.

## The statement rewrite

- **`gpu_rewrite_ast`** (#86) — a pure C++ scalar over DuckDB's own
  `json_serialize_sql` tree. Everything it decides comes from a `context` the
  client builds; it touches no catalog and records nothing, so DuckDB may cache
  it. 0.14 ms to rewrite a statement, 0.045 ms to reject one.
- **The Python wrapper** (#87, #91, #121) — classification, name resolution,
  the template cache, the transaction rule, typed fallback, settings, and a
  `pip`-installable package that finds the extension by an explicit path, by
  `GPUDB_EXTENSION_PATH`, next to a source checkout, or as the copy DuckDB
  itself installed.
- **Shapes the rewrite reaches**: `WHERE` (#93, #94), multi-column packed keys
  (#99), `DATE` / `TIMESTAMP` keys and predicates (#98), `VARCHAR` keys through
  a dictionary (#100), several payloads in one pass (#105), computed lanes for
  arbitrary row-local expressions (#106), expressions over aggregates and
  compound `HAVING` (#107), aggregates with no `GROUP BY` (#109), nested
  rewriting of an inner `SELECT` (#111), folded derived tables and
  `count(DISTINCT)` (#112), subquery predicates as lanes (#113), views (#117),
  `GROUP BY ALL` / ordinals / `ORDER BY ALL` (#118), `SELECT DISTINCT`,
  `RIGHT JOIN` and further DISTINCT forms (#119), `FILTER (WHERE …)`,
  `count_if`, `bool_and` / `bool_or` (#115), and CTEs (#143, #155).
- **Joins** (#103, #104, #109) — an inner equi-join onto a unique key is
  materialised on the device as a new exact set; other INNER / LEFT / RIGHT and
  many-to-many joins, composite keys, `USING`, extra `ON` predicates and
  cross-table expressions are answered from an upload of the join's result.

## Exact operators

- **Exact GROUP BY** (#90 CPU reference, #92 Metal, #152 CUDA) — NULL-aware
  uploads, a NULL-key group last, 128-bit sums that never wrap, `count` and
  `count(*)` distinguished, `min` / `max` / `avg`, device `HAVING` and top-k.
- **The `WHERE` mask** (#93 CPU, #94 Metal, #152 CUDA) — conjunctions of
  comparisons, `IN`, `BETWEEN`, `IS [NOT] NULL` over row-aligned predicate
  lanes, with projection pushdown on the table functions.
- **The global masked aggregate** (#126 core, #153 CUDA) — an aggregate with no
  `GROUP BY` in one fused pass: no key, no sort cache, no permutation.
- **The materialised key join** (#103 core, #154 CUDA) — `gpu_join_materialize`
  and the `gpu_inner_join` table function.
- **`avg` finalised the way DuckDB finalises it** (#146, #149, #157, #160) —
  native computes `avg` as a `long double` quotient. Over integers that is
  exact everywhere. Over `DECIMAL` it is not expressible in SQL at all, since
  the host's `long double` is 80-bit on x86-64 and SQL has no 80-bit type, so
  the division moved into C++: `gpu_avg_decimal(sum HUGEINT, count BIGINT,
  scale BIGINT)`. The shape is now rewritten on every platform. Against an
  extension too old to provide that function the column is derived in SQL, and
  the wrapper declines the shape only where that derivation is not native's own
  arithmetic — where `gpu_build_info()` reports `avgf=` anything but 53. So the
  change is additive across version skew: nothing that was right becomes wrong.

## Kernels

- **Few keys do not need a sort** (#127, #129, #130) — a direct row-order
  grouped reduce for keys with at most 512 distinct values, admitted by a
  measured rule and gated on the device's identity before anything is compiled
  (a refused pipeline build on a virtualised Apple GPU used to poison that
  process's Metal compiler).
- **The `WHERE` stage in one pass** (#139) — one fused `gpred_eval` per
  statement instead of one pass per term, and the counting pass's gather kept
  as a mask in sorted order so the reduce stops re-gathering it per payload:
  1.06× to 2.38× on the sort path's kernel over 98 measured cells at SF10,
  nothing slower.
- **Block-level reduce** (#108) — a second reduction level so a group spanning
  millions of rows is not merged by one thread; 7 groups over 6M rows went from
  6.1 ms to 1.1 ms of kernel time, bit-identical.
- **Sort caches and device memory** (#116, #135) — backends report device
  memory for the budget, and a few-group key sheds its sort cache and its key
  lane once the direct path has made them unnecessary.
- **Statement overhead** (#132) — a cached plan per rendered statement and a
  leaner per-statement path in the extension.
- **The fused reduce on CUDA** (#163) — `agg_all_i64` (SUM + MIN + MAX + COUNT
  in one pass) had been a throwing stub on CUDA since v0.1 while CPU and Metal
  implemented it. The fused kernel reads the column once instead of three
  times: 2.150 ms of kernel time down to **0.719 ms** over 50M int64 rows on an
  RTX 4090 Laptop, 517.8 GiB/s — about 90% of that part's theoretical
  bandwidth. The same fusion on 20 CPU threads is 1.16×, which is the contrast
  worth keeping: fusing pays in proportion to how much of the time was spent
  moving bytes.
- **The CUDA exact upload, double-buffered** (#162) — one staging buffer with a
  stream synchronise after every span serialised the copy against the kernel,
  leaving the PCIe link idle for each kernel and the GPU idle for each copy. A
  buffer each, and an event instead of a barrier: 18.67 ms down to **12.75 ms**
  over 2M rows in 996 spans, 32% off the upload. Both exact uploads take it.

## Residency and memory

- **Resident columns, stages A–C** (#123, #124, #125) — exact columns keep the
  table's row order with NULLs under a bitmap; one per-table store holds a
  single copy of each column, which statements are views over; each lane is
  stored at the narrowest signed width its values fit. The 22 TPC-H queries at
  SF10 went from 44.9 GiB to 22.7 — and to 18.7 once a key lane with few
  distinct values stopped keeping a sort cache.
- **The memory budget** (#115, #122, #144) — a pre-upload estimate that is an
  upper bound by construction, admission before the upload rather than eviction
  after it, a `memory` reason and `detail` when a set is refused, and eviction
  by measured value per byte rather than by recency. A refused candidate costs
  nothing, and `memory()["evictions_wasted"]` counts that rather than assuming
  it.
- **Uploads that do not disturb a query** — short idle row-id segments through
  the extension's upload sessions (#91), carrying exact segments (#95).
- **Writes from any connection** (#120, #137) — the database file and its
  write-ahead log are stat'ed before every rewritten statement (2–3 µs), so a
  committed write from a connection the wrapper does not own is noticed; the
  row-count guard alone would miss an in-place `UPDATE`.
- **Inner-statement bounds** (#144) — a `GROUP BY` whose groups DuckDB consumes
  itself, rather than the client, is decided by a different rule; measured, with
  the reduction ratio as the condition.
- **Joins as index vectors** (#142) — the mechanism, and what it costs to read
  a lane through an index, measured. No SQL path takes it: every query's
  behaviour is what the rest of these notes describe.

## Using it

There are **three ways in**, and they all talk to the same extension and the
same resident columns:

1. **The `gpudb` shell** (#138) — plain DuckDB SQL from a terminal, DuckDB's own
   box renderer, and a footer under each result saying where the statement ran
   and why. `.gpu`, `.gpu on|off`, `.residents`, `.memory`, `.timer`, `.read`,
   `.open`, `.tables`, `.schema`, `.version`.
2. **Python — `gpudb.connect()`** — the same decision from an application, a
   notebook or a pipeline. **`last_rewrite()["detail"]`** (#140) gives the
   decision in a sentence next to the reason code, `sql()` decides on
   `execute()`'s path, and the device is reported by name.
3. **Explicit `gpu_*` functions** — any DuckDB client in any language,
   including the stock CLI. This is the only route that needs no wrapper,
   because DuckDB's stable C API has no hook that sees a statement before it is
   planned. v0.6.0 registered 38 of these functions; v0.7 registers 65, and
   every v0.6 one is still there under the same name, return type and parameter
   types.

Installing both pieces:

```bash
pip install duckdb-gpudb          # the `gpudb` command and the gpudb module
```
```sql
INSTALL gpudb FROM community;     -- the extension, into DuckDB
LOAD gpudb;
```

To replace an extension DuckDB already has, `INSTALL` alone is not enough:
`FORCE INSTALL gpudb FROM community;`, or `UPDATE EXTENSIONS;`. The wrapper
upgrades with `pip install -U duckdb-gpudb`.

**Distribution name `duckdb-gpudb`** (#136), import name `gpudb`;
Apache-2.0 in the package metadata to match the repository (#133), and the
full licence text so GitHub detects it (#89).

## Tests, CI and packaging

- `python/tests/test_wrapper.py`: 1157 checks, 0 skipped, green under DuckDB
  1.4.5 and under 1.5.5 on an M4 Max (#150 makes the suite run to the end on a
  backend without the exact path; #159 and #160 replaced its host gating with a
  probe for the function that decides). On the RTX 4090 box the same suite is
  1154 ok with 4 failures, all in the segmented-upload cases and all unrelated
  to the exact path — measured and written down in `BENCHMARK.md` (#161).
- `test_gpudb`: 730 / 730 checks on CPU + CUDA (RTX 4090 Laptop) — 711 before
  #163 gave CUDA a fused `agg_all` and the suite stopped skipping that block.
- `run_sql_tests.sh`: 224 passing, 0 failing with `GPUDB_CUDA_EXACT=1` on the
  RTX 4090; 45 `expected_fail` guardrail cases across the 18 files in
  `test/sql/`.
- `scripts/tpch_coverage.py`: SF1 17 of 22, SF10 19 of 22, 0 rows differing, on
  the M4 Max.
- The SQL suite now runs on Linux with the DuckDB libs pinned (#151), and the
  wrapper suite runs against the built extension in CI (#158).
- sqllogic coverage for the exact surface and a guard against an older
  extension (#148); parity and residency scripts default to the platform's
  build directory (#141); the residency gate measures native shapes (#156).

## What stays on DuckDB

Window functions; `FULL` joins, `SEMI` / `ANTI` **join syntax**, `NATURAL`
joins and cross products (the `EXISTS` / `IN` *forms* are rewritten — see Joins
above);
`median`, `stddev` and quantiles; `sum` / `avg` over `DOUBLE` or `FLOAT`;
prepared-statement parameters; statements inside an explicit transaction;
`WITH RECURSIVE` and `AS MATERIALIZED` CTEs; `ROLLUP` / `CUBE` /
`GROUPING SETS` / `QUALIFY` / `DISTINCT ON`; a set operation as the whole
statement; and any shape the measured bounds decline. Each one is answered —
by DuckDB, at DuckDB's speed. `docs/TRANSPARENT_DESIGN.md` §10 gives the
reason for each, and `KNOWN_ISSUES.md` has the rest.

## Platforms

| | Metal (Apple Silicon) | CUDA (NVIDIA) | No GPU |
|---|---|---|---|
| Plain SQL on the GPU | yes | opt-in: `GPUDB_CUDA_EXACT=1` | no — everything runs on DuckDB |
| Explicit `gpu_*` functions | yes | yes | yes, on the CPU backend, same answers |
| From the community registry | yes | a registry Linux binary may report `compiled=cpu`; `gpu_build_info()` answers it for whichever binary is in front of you | yes |

Every operator the transparent path needs is implemented on CUDA (#152, #153,
#154). On an RTX 4090 Laptop with the path enabled, the unit suite is 730 / 730,
the SQL suite 224 passing and 0 failing, and TPC-H at SF1 is 17 of 22 on the
device with 0 rows differing — the same coverage and the same five declines as
Metal (#161). It stays opt-in in this release for one reason:
`scripts/transparent_gate.py` has not been swept on that machine, so a CUDA
build would be deciding with Metal's thresholds — and rule 1 is a measurement,
not an assumption.

## Credits

The Metal hash join, the hybrid join planner and the on-device segment reduce
were contributed by [@lmangani](https://github.com/lmangani) in
[PR #43](https://github.com/singhpratech/duckdbgpumetaldbram/pull/43), and have
been the base of the join stack since v0.5.0.

## Reading further

- [docs/README.md](README.md) — a reading guide to the journal and the design documents
- [docs/TRANSPARENT_DESIGN.md](TRANSPARENT_DESIGN.md) — the rewrite, the rules, the thresholds
- [docs/RESIDENT_COLUMNS_DESIGN.md](RESIDENT_COLUMNS_DESIGN.md) — how columns live on the device
- [docs/RESEARCH_NOTES.md](RESEARCH_NOTES.md) — the dated journal
- [BENCHMARK.md](../BENCHMARK.md) — every measurement, losing cells included
- [KNOWN_ISSUES.md](../KNOWN_ISSUES.md) — every documented trade-off
