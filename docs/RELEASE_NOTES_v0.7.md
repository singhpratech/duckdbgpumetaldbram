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

**Ties in a pushed top-k go back to DuckDB** (#164). An `ORDER BY <aggregate>
LIMIT k` whose ordering values tie inside the first *k* rows has no single
native answer: plain DuckDB above one thread returned 3 different row sets, and
up to 6 different orderings, over 20 runs of one such statement on TPC-H SF1,
and is deterministic only at `threads=1`. The rewritten statement therefore asks
the device for one row past the limit and carries a guard that stops it when
`rank()` and `row_number()` disagree at or above rank *k*; the wrapper then
answers the original on DuckDB with `reason == "ties"`. It is decided against
the data on every execution, and because the fallback is a genuine loss (a
device pass plus DuckDB's own run) the template is measured-declined like any
other losing template, with the tie named in `detail`.

Through `sql()` — the path the `gpudb` shell takes — that guard cannot be a
clause of the statement: a lazy relation is read after the call has returned,
where a raise could not be answered, so the guard runs on a side cursor inside
the call, and for a pushed top-k the side cursor IS the device pass. **The
verdict is therefore remembered** (#170), per rendered statement, and dropped
wherever the resident sets are — any write, DDL, `SET`, `ATTACH` or foreign
write the wrapper sees, a row count that moved under the set, a rewrite that
raised. One device pass per data version instead of one per call: the shape went
from 17.1 ms back to 8.7 ms through `sql()`, and in the shell to 5.7 ms against
17.8 ms native. A *tie* verdict is never cached — it raises, and the template is
measured-declined, which is the stronger answer.

**The measured rule times the path** (#170). `execute()` and `sql()` reach the
same templates, but `sql()` pays for its guards inside the call, so what is
compared against native is what a caller on that path actually pays. A template
that loses only through `sql()` is declined there alone; one whose statement
loses is declined on both, since `sql()` costs what `execute()` costs plus the
guards. `scripts/transparent_gate.py` takes `--path execute|sql|both|auto` and
`scripts/tpch_coverage.py` takes `--path execute|sql`: a gate that measures one
entry point proves nothing about the other, which is how this cost reached a
release branch at all.

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
- **Narrow lanes on CUDA** (#165) — the exact path's I64 lanes are stored at the
  narrowest signed width their values fit on NVIDIA hardware too, which is what
  `gpu_build_info()` now reports as `narrow=true` there: 58% off the lanes at
  TPC-H SF1 on an RTX 4090 Laptop, with the widths chosen in the upload and every
  kernel reading through one helper.
- **The CUDA exact sort cache, narrowed** (#166) — its companion. The exact
  sort cache held an i64 key and an i64 row id; it now holds the key at the
  lane's width and the row id as u32, which is the layout Apple Silicon Metal
  has had since the lane narrowing landed there. 44% off the derived structures
  and 31% off the resident total at SF1 (257.5 MiB to 177.4), lanes untouched.
  The v0.6 `gpu_upload_pair` path's cache is still i64 + i64 on CUDA.
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
- **The budget counts what is there, and a refusal is remembered** (#172) — the
  number the budget is compared with is now the physical total, every store
  column once plus what sets own (`con.memory()["bytes"]`, beside
  `device_allocated`, what the driver says the process holds). The per-set
  figures are ranking quantities and are not summable — a store-backed set is a
  view over shared columns, and summing them measured 4.4× the truth. The
  pre-upload estimate narrows each plain column's lane from DuckDB's own
  zone-map statistics instead of charging it the width of its type, which was
  2.4× the truth and refused sets that fit. A refusal is now remembered, keyed
  on the things that could change the answer, so a statement whose set cannot
  fit costs **one** upload attempt rather than one per statement. An upload the
  device refuses **fails** (`GPUDB_DEVICE_UPLOAD_REFUSED`) instead of landing on
  the CPU reference, so a set is never half-placed or answered from host memory,
  and `gpu_store_columns()` gained `on_gpu` as the second line of that defence.
  An execution failure for device memory becomes a refusal plus a learned
  headroom — where the backend reports what it needed against what was free, the
  budget holds the difference back from then on — and the fallback now reports
  the real exception's first line in `last_rewrite()["detail"]`. New gate:
  `scripts/budget_gate.py` (56 s, in `local_check.sh`), which asserts at every
  sample that physical resident stays under the budget, that no lane is on the
  host, that no set is uploaded twice, and that every row equals native's.
  Measured after it: **TPC-H SF10 is 19 of 22 at the default budget**, with no
  `memory` declines — the raised budget the earlier runs used is no longer
  needed.
- **An operator that cannot get working memory says how much it wanted** (#171)
  — on CUDA, `out of memory (needs N MiB of working memory, M MiB free)`, with
  a forced-refusal unit test behind it. CUDA also reports its device memory, so
  the wrapper's budget there is derived from the card rather than from host RAM.
  `scripts/vram_sampler.sh` samples device memory while a long run proceeds,
  which is how a plateau is told from a climb.
- **Uploads that do not disturb a query** — short idle row-id segments through
  the extension's upload sessions (#91), carrying exact segments (#95).
- **An upload that steps back on a busy machine** (#167) — measured per
  statement, the cost a background upload imposes is not the segments (a
  statement that lands on one is not slower than the control, and the interrupt
  is honoured in 0.27 ms median with 8 of 16 cores busy) but the two steps an
  interrupt cannot stop: the device copy and the sort cache built after it. The
  session now measures for free how many cores it is being given — CPU seconds
  per wall second of a completed segment scan — and on a contended machine those
  two steps wait for a quiet connection first. The window asked for decays to the
  ordinary idle threshold over 20 seconds, so a step takes the best gap offered
  and runs unconditionally at the end: residency is delayed on a busy machine,
  never withheld, and an idle machine takes exactly the old path. A segment's
  size also follows its yield, halving when almost none of the recent attempts
  land and doubling back when almost all do, for the box where the segment does
  not fit the window the workload leaves. An interrupted sort cache is retried
  rather than skipped, so `ready` means uploaded *and* prepared.
  `GPUDB_UPLOAD_QUIET_MS`, `GPUDB_UPLOAD_QUIET_MAX_S` and
  `GPUDB_RESIDENCY_TRACE` are the controls ([ENVIRONMENT.md](ENVIRONMENT.md)).
- **A segment priced from measurement, and a session that says when none fits**
  (#169) — a segment has a fixed cost no smaller segment escapes: the
  statement's own parse and bind, and DuckDB's scan set-up over every row group
  of the table (measured 0.40 ms for 8,192 rows on an M4 Max against 2.6 ms for
  524,288; about 1.5 ms on an x86 box). The floor the size halves down to is now
  the smallest size whose predicted total for the table stays within reach of
  the total at the default size, fitted to the segments that landed *here* —
  which derives 32,768 rows on the x86 curve, where the old constant sat, and
  65,536 on the M4 Max. Pricing a halving takes two landed sizes, so until there
  are two the size halves on the yield rule alone down to the old constant of
  1/32 of the default, and starvation is declared there. At the floor a session
  whose windows still ask for something smaller is **starved** and says so:
  `progress()` reports `starved` with `window_ms`, `fixed_ms`, `per_row_us` and
  `floor_rows`. Nothing is forced — the set stays off the device and every
  statement keeps its native answer. Two measured cuts to the fixed cost came
  with it: the segment statement is `PREPARE`d once per session (0.50 → 0.40 ms
  for an 8,192-row segment), and the idle wait now wakes when the idle test can
  first pass rather than a whole `idle_ms` later.
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
   `.open`, `.tables`, `.schema`, `.version`. `.open` with no argument works in
   a `--readonly` session as well (the in-memory database it opens is
   read-write; a named file keeps the session's read-only setting), and `.gpu`
   prints every time as a time — `4.172 ms`, not seventeen digits — with its
   labels in one column (#164).
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
upgrades with `pip install -U duckdb-gpudb`. A wrapper that finds an extension
older than itself says exactly that and gives exactly that advice, ending in a
new session, since neither form reaches a process that has already loaded the
old copy (#164).

**Distribution name `duckdb-gpudb`** (#136), import name `gpudb`;
Apache-2.0 in the package metadata to match the repository (#133), and the
full licence text so GitHub detects it (#89).

## Tests, CI and packaging

- `python/tests/test_wrapper.py`: green under DuckDB 1.4.5 and under 1.5.5 on
  an M4 Max, and **1258 of 1258** on the RTX 4090 Laptop (#150 makes the suite
  run to the end on a backend without the exact path; #159 and #160 replaced its
  host gating with a probe for the function that decides; #164 added the
  reported tie and its fallbacks; #170 asserts the remembered verdict as a count
  of device passes rather than as a time; #171 and #172 added the refusal,
  placement and accounting cases). The two boxes collect different counts,
  because the `avg`-over-`DECIMAL` section branches on the host's `long double`,
  so each number is named with its machine. The x86 box used to carry 4
  failures, all of them segmented-upload cases unrelated to the exact path
  (#161); once #169 priced a segment from what each machine measures, it came
  back fully green.
- `python/tests/test_residency_policy.py`: 105 checks, 0 failing — the residency
  policy on a clock the test drives, so the yield rule, the quiet-window bound
  and the measured segment floor are asserted directly rather than raced against
  a real machine (#167, #169).
- `test_gpudb`: 752 / 752 checks on CPU + CUDA (RTX 4090 Laptop) — 711 before
  #163 gave CUDA a fused `agg_all` and the suite stopped skipping that block,
  and #171 added the forced-refusal case.
- `run_sql_tests.sh`: 225 passing, 0 failing on the RTX 4090 with the default
  build; 45 `expected_fail` guardrail cases across the 18 files in `test/sql/`.
- `scripts/tpch_coverage.py`: SF1 17 of 22 and SF10 19 of 22 on the M4 Max,
  SF1 17 of 22 on the RTX 4090, 0 rows differing anywhere. SF10 runs at the
  **default** memory budget since #172.
- `scripts/budget_gate.py` (#172): 169 statements under a budget too small for
  them — physical resident at or under the budget at every sample, 0 evictions
  wasted, at most one upload attempt per set, 0 rows differing, 0 errors.
- `scripts/transparent_gate.py` now drives both entry points (#170). The top-k
  and plain subset was re-run at SF1 through `execute()` and through `sql()`:
  every row at or above its bound and identical either way, with the `l_partkey`
  top-k reading 2.15× through `execute()` and 2.12× through `sql()`.
- The SQL suite now runs on Linux with the DuckDB libs pinned (#151), and the
  wrapper suite runs against the built extension in CI (#158).
- sqllogic coverage for the exact surface and a guard against an older
  extension (#148); parity and residency scripts default to the platform's
  build directory (#141); the residency gate measures native shapes (#156); the
  libduckdb fetch is retried when the connection is reset (#164).

## What stays on DuckDB

Window functions; `FULL` joins, `SEMI` / `ANTI` **join syntax**, `NATURAL`
joins and cross products (the `EXISTS` / `IN` *forms* are rewritten — see Joins
above);
`median`, `stddev` and quantiles; `sum` / `avg` over `DOUBLE` or `FLOAT`;
prepared-statement parameters; statements inside an explicit transaction;
`WITH RECURSIVE` and `AS MATERIALIZED` CTEs; `ROLLUP` / `CUBE` /
`GROUPING SETS` / `QUALIFY` / `DISTINCT ON`; a set operation as the whole
statement; a pushed `ORDER BY … LIMIT k` whose ordering values tie inside the
first *k* rows (`ties`, above); and any shape the measured bounds decline. Each
one is answered — by DuckDB, at DuckDB's speed. A bare `SELECT count(*) FROM t`
is on this list too and now says why it is: DuckDB answers it from the table's
own row count without reading a column, and the device plan needs a column to
build its constant key from (#164 fixed the sentence, not the decision).
`docs/TRANSPARENT_DESIGN.md` §10 gives the reason for each, and
`KNOWN_ISSUES.md` has the rest.

## Platforms

| | Metal (Apple Silicon) | CUDA (NVIDIA) | No GPU |
|---|---|---|---|
| Plain SQL on the GPU | yes | yes, on by default — from a binary that carries CUDA | no — everything runs on DuckDB |
| Explicit `gpu_*` functions | yes | yes | yes, on the CPU backend, same answers |
| From the community registry | yes | a registry Linux binary may report `compiled=cpu`; `gpu_build_info()` answers it for whichever binary is in front of you | yes |

Every operator the transparent path needs is implemented on CUDA (#152, #153,
#154) and the path is **on by default** there (#168). On an RTX 4090 Laptop the
unit suite is 752 / 752, the SQL suite 225 passing and 0 failing, the wrapper
suite 1258 passing and 0 failing, and TPC-H at SF1 is 17 of 22 on the device
with 0 rows differing through both entry points — the same coverage and the same
five declines as Metal (#161, #168). What turned it on was the evidence the flip
had been waiting for: the full gate on that box, at the wrapper's own memory
budget, ran **1630 cells with 0 slower than native and 0 differing**, minimum
ratio 1.07×. `GPUDB_CUDA_EXACT=0` turns it off again without a rebuild.

Two caveats stay: the thresholds are the Metal-measured ones **verified on one
CUDA machine** rather than measured for every GPU (`_thresholds.TABLE["CUDA"]`
is still `METAL`, and says so as a dated measured fact), and TPC-H Q1 sits at
parity on that box — the per-process measured rule is what decides it. And the
path needs a binary that carries CUDA at all: a Linux binary from the community
registry may report `compiled=cpu`, in which case CUDA comes from a release
binary or a source build. `SELECT gpu_build_info();` is what tells them apart.

## Credits

The Metal hash join, the hybrid join planner and the on-device segment reduce
were contributed by [@lmangani](https://github.com/lmangani) in
[PR #43](https://github.com/singhpratech/duckdbgpumetaldbram/pull/43), and have
been the base of the join stack since v0.5.0.

## Reading further

- [docs/USING_THE_SHELL.md](USING_THE_SHELL.md) — the `gpudb` shell, end to end
- [docs/USING_PYTHON.md](USING_PYTHON.md) — `gpudb.connect()`, every option
- [docs/INSTALL.md](INSTALL.md) — both install routes, platforms, troubleshooting
- [docs/README.md](README.md) — a reading guide to the journal and the design documents
- [docs/TRANSPARENT_DESIGN.md](TRANSPARENT_DESIGN.md) — the rewrite, the rules, the thresholds
- [docs/RESIDENT_COLUMNS_DESIGN.md](RESIDENT_COLUMNS_DESIGN.md) — how columns live on the device
- [docs/RESEARCH_NOTES.md](RESEARCH_NOTES.md) — the dated journal
- [BENCHMARK.md](../BENCHMARK.md) — every measurement, losing cells included
- [KNOWN_ISSUES.md](../KNOWN_ISSUES.md) — every documented trade-off
