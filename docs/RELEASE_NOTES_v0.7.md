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

**Measured** on an Apple M4 Max (Metal), TPC-H, every row compared with
native <!-- RE-RUN -->:

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
  scale BIGINT)`. The shape is now rewritten on every platform. An extension
  too old to provide that function still declines it rather than deriving it
  in SQL, so the change is additive across version skew.

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

## Residency and memory

- **Resident columns, stages A–C** (#123, #124, #125) — exact columns keep the
  table's row order with NULLs under a bitmap; one per-table store holds a
  single copy of each column, which statements are views over; each lane is
  stored at the narrowest signed width its values fit. The 22 TPC-H queries at
  SF10 went from 44.9 GiB to 22.7.
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
- **Joins as index vectors** (#142) — the mechanism and what it costs to read a
  lane through an index, measured. It is not wired into SQL, and nothing claims
  it is.

## Using it

- **The `gpudb` shell** (#138) — plain DuckDB SQL from a terminal, DuckDB's own
  box renderer, and a footer under each result saying where the statement ran
  and why. `.gpu`, `.gpu on|off`, `.residents`, `.memory`, `.timer`, `.read`,
  `.open`, `.tables`, `.schema`, `.version`.
- **`last_rewrite()["detail"]`** (#140) — the decision in a sentence next to the
  reason code, `sql()` deciding on `execute()`'s path, and the device reported
  by name.
- **Distribution name `duckdb-gpudb`** (#136), import name `gpudb`;
  Apache-2.0 in the package metadata to match the repository (#133), and the
  full licence text so GitHub detects it (#89).

## Tests, CI and packaging

- `test_gpudb`: 3026 / 3026 checks on CPU + Metal; 711 / 711 on CPU + CUDA.
- `run_sql_tests.sh`: 224 passing cases and 45 expected-fail guardrails, on
  Metal and on CUDA with the exact path on.
- `python/tests/test_wrapper.py`: 1158 checks, green under DuckDB 1.4.5 and
  1.5.5 (#150 makes the suite run to the end on a backend without the exact
  path).
- `scripts/tpch_coverage.py`: SF1 17 of 22, SF10 19 of 22, 0 rows differing.
- The SQL suite now runs on Linux with the DuckDB libs pinned (#151), and the
  wrapper suite runs against the built extension in CI (#158).
- sqllogic coverage for the exact surface and a guard against an older
  extension (#148); parity and residency scripts default to the platform's
  build directory (#141); the residency gate measures native shapes (#156).

## What stays on DuckDB

Window functions; `FULL` joins, `SEMI` / `ANTI` **join syntax** and cross
products (the `EXISTS` / `IN` *forms* are rewritten — see Joins above);
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
| From the community registry | yes | the registry's Linux binary is CPU-only | yes |

<!-- CUDA-DEFAULT: the CUDA row and the paragraph below are the conservative
     (opt-in) statement. If CUDA ships on by default, change the cell to "yes"
     and replace the paragraph with the measured CUDA coverage table. -->

Every operator the transparent path needs is implemented on CUDA (#152, #153,
#154), and the suites and the TPC-H coverage pass there. It stays opt-in in
this release because `scripts/transparent_gate.py` has not been swept on that
machine, so a CUDA build would be using Metal's thresholds — and rule 1 is a
measurement, not an assumption.

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
