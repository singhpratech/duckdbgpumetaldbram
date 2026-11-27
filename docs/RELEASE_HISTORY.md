# Release history

What shipped in each release, newest first.

### Latest — pip 0.7.5 (2026-11-26)

Upgrade if you group by a text column that contains NULLs and filter on that
same column with `<>` or `IS [NOT] NULL`. There is no 0.7.4.

- [x] **A WHERE on a VARCHAR GROUP BY key drops the NULL group** (pip
  `duckdb-gpudb` 0.7.5) — a single VARCHAR key is uploaded as its tuple text,
  in which a NULL key is a valid cell, and a WHERE term on that key read that
  lane. `SELECT s, count(*) FROM t WHERE s <> 'x' GROUP BY s` and
  `... WHERE s IS NOT NULL GROUP BY s` returned a NULL group DuckDB does not,
  and `IS NULL` read a validity bit that is always set. With shipped defaults
  on 20M rows and a 4000-value key with NULLs, pip 0.7.3 returned 4000 and
  4001 rows where DuckDB returns 3999 and 4000. A `<>` or `IS [NOT] NULL` term
  on such a key now reads a lane of its own, whose NULLs are NULL.
  `=`, `IN`, `NOT IN`, `IS DISTINCT FROM`, `LIKE`, multi-column keys and
  DECIMAL / DATE / integer keys were already right. Present in pip 0.7.2 and
  0.7.3.
- [x] **A DECIMAL key with a WHERE stays on the device for new literals** — a
  statement with a single DECIMAL key and a WHERE on it, re-run with different
  literals, went back to DuckDB on every re-run; its rows were right, and it
  now stays on the device.
- [x] **Checked** — on Apple Silicon, with the extension binary the wheels
  bundle: the fix adds 186 checks, 42 of which fail on
  the 0.7.3 wrapper, and the wrapper suite passes with 1594 checks on DuckDB
  2.0.1 and 2.0.0 and 1624 on 1.5.6, none failing. TPC-H coverage is unchanged on DuckDB 2.0.1: 16 of 22 queries on the device at scale factor 1 and 18 of 22 at scale factor 10, no rows that differ.

  On an RTX 4090 Laptop (CUDA), with the same extension binary, pip 0.7.3
  returned the same extra NULL group for both statements above with shipped
  defaults (4000 and 4001 rows where DuckDB returns 3999 and 4000), and with
  the fix both match DuckDB. The wrapper suite passes there with 1594 checks
  on DuckDB 2.0.1 and 2.0.0 and 1624 on 1.5.6, none failing, and the fix's own
  checks fail on the 0.7.3 wrapper there as they do on Apple Silicon.
- [x] **Still open** — wrong-answer cases in the published package that this
  release does not fix:
  - A TIMESTAMP column that holds `infinity` or `-infinity`: a statement that
    filters on it, or on an expression over it such as `date_trunc`, can count
    those rows as NULL (pull request #197).
  - An ORDER BY name that is both a select alias and a different column
    (`SELECT k AS kk, sum(v) AS k ... ORDER BY k`): gpudb can order by the
    column where DuckDB orders by the alias, which with a LIMIT returns
    different rows; giving the alias a name no column has avoids it.
  - A setting changed on another cursor or connection of the same database
    (`default_collation`, `default_order`, `default_null_order`) is not seen
    by a cached decision until the connection itself runs a statement that
    changes settings, schema or data (pull request #196).
  - When a `LIMIT` is not selected on the device and the rows at its boundary
    tie on every `ORDER BY` column, gpudb and DuckDB can each return a
    different, equally valid choice among the tied rows; adding a column that
    makes the order unique avoids it.

  `gpudb.connect(transparent=False)` (or `con.transparent = False` on an open
  connection) hands every statement to DuckDB unchanged, for anyone who needs
  to rule these out; the explicit `gpu_*` functions stay available.
- [x] **Versions** — wrapper only. The extension stays at v0.7.1 in the
  community registry; the pip 0.7.5 platform wheels bundle the same v0.7.0
  extension binary as pip 0.7.1, 0.7.2 and 0.7.3.

### pip 0.7.3 (2026-11-26)

- [x] **An explicit NULL order survives a re-render** (pip `duckdb-gpudb`
  0.7.3) — `json_serialize_sql` writes the NULL order as `NULLS FIRST` /
  `NULLS LAST`, and the Python renderer read only `NULLS_FIRST` /
  `NULLS_LAST`. The first run of a statement was right; run again with
  different literals, an explicit NULL order was dropped and the NULL group
  came back where `default_null_order` puts it. Both spellings are now read.
- [x] **A top-k with NULLs first keeps the NULL group** — `ORDER BY sum(x)
  DESC NULLS FIRST LIMIT k` (and `ASC NULLS FIRST`, `min` / `max`, and NULLs
  first through `default_null_order`) came back without the group whose
  aggregate is NULL: the device top-k ranks a NULL aggregate after every
  value. That top-k is no longer pushed; the GROUP BY can still run on the
  device, with DuckDB ordering and limiting the groups.
- [x] **Who it affects** — only statements with an explicit NULL order or a
  non-default `default_null_order`; statements with NULLs last (DuckDB's
  default) were already exact. Anyone who writes `NULLS FIRST` /
  `NULLS LAST` or sets `default_null_order` should upgrade. A corpus of 55
  statement templates under 5 order settings with 3 literal sets each: 184 of
  825 runs differed from native on 0.7.2, 0 on 0.7.3. Wrapper suite: 1408
  checks on DuckDB 2.0.1 and 2.0.0, 1438 on 1.5.6, 0 failures. TPC-H coverage
  is unchanged (SF1 16 of 22, SF10 18 of 22 on DuckDB 2.0.1, no rows that
  differ).
- [x] **Versions** — wrapper only. The extension stays at v0.7.1 in the
  community registry; the pip 0.7.3 platform wheels bundle the same v0.7.0
  extension binary as pip 0.7.1 and 0.7.2.

### pip 0.7.2 (2026-11-25)

- [x] **DuckDB 2.0.1 in the Python wrapper** (pip `duckdb-gpudb` 0.7.2) —
  DuckDB 2.0.1 lists the TIMESTAMP WITH TIME ZONE overloads of `+`, `-`,
  `date_part`, `year`, `date_trunc`, `strftime` and others (47 names) as
  `CONSISTENT_WITHIN_QUERY`. The wrapper judged stability per function name,
  so on 2.0.1 every statement using one of those names declined with `shape`
  and ran on DuckDB: TPC-H at SF10 went from 18 of 22 on the GPU to 9, with
  identical rows. The wrapper now admits a name that has a `CONSISTENT`
  overload and judges the overload DuckDB binds, on the bound plan, for every
  lane and every uploaded join. Measured at SF10 on Apple Silicon: 18 of 22 on
  2.0.1 and 2.0.0, 19 of 22 on 1.5.6, no rows that differ.
- [x] **Time-zone-aware values stay on DuckDB** — on every DuckDB version, an
  expression that produces or reads a TIMESTAMPTZ / TIMETZ value is not
  computed as a resident lane. `current_localtime()` and
  `current_localtimestamp()` are treated as session functions: DuckDB 1.5 and
  2.0.0 list them `CONSISTENT`, and a lane computed from one kept the clock
  value of its upload, which gave rows that differed from native.
- [x] **Versions** — wrapper only. The extension stays at v0.7.1 in the
  community registry; the pip 0.7.2 platform wheels bundle the same v0.7.0
  extension binary as pip 0.7.1.

### v0.7.1 (2026-11-08)

- [x] **DuckDB 2.0 in the Python wrapper** (pip `duckdb-gpudb` 0.7.1,
  2026-10-29) — DuckDB 2.0 changed the shape of `json_serialize_sql`, so on
  2.0.0 every statement declined with `shape` and ran on DuckDB. The wrapper
  now converts a 2.0 tree to the shape its readers take where the tree enters
  it (`gpudb/_ast_compat.py`), and back where 2.0's deserializer needs it;
  `python/tests/test_ast_compat.py` pins 115 statements as 1.5.6 fixtures and runs on
  both engines. TPC-H on the M4 Max on DuckDB 2.0.0: 16 of 22 on the GPU at
  SF1 and 18 of 22 at SF10, 0 rows differing; on 1.5.6, 17 and 19 (the SF10
  figure measured before the change).
- [x] **`avg` over `DECIMAL` stays on DuckDB 2** — 2.0's optimizer finalises
  `avg(DECIMAL)` as `CAST(sum(x) AS DOUBLE) / count(x)` in some plans and as
  before in others, depending on NULL statistics at plan time; the two differ
  in the last bit on up to 53% of groups. The wrapper declines the shape on
  DuckDB 2 (reason `shape`) rather than return a different double, which moves
  TPC-H Q1 to DuckDB there. DuckDB 1.4 / 1.5 are unchanged.
  [KNOWN_ISSUES.md](../KNOWN_ISSUES.md).
- [x] **`gpu_functions()`** (issue #176) — one row per gpudb overload: the
  real parameter names, the parameter and return types exactly as
  `duckdb_functions()` renders them (the two join on `(function_name,
  function_type, parameter_types)`), a description, a runnable example and
  categories. The stable C API gives a function no description or parameter
  names, so this is where they live. The extension now registers 66 `gpu_*`
  names and 73 overloads. `test/sql/gpu_functions.test` and
  `test/sqllogic/gpudb.test` compare it with `duckdb_functions()` both ways,
  and `scripts/gpu_functions_check.py` runs every example.
- [x] **CI follows DuckDB 2.0's renamed CLI download** — v2.0.0 publishes
  `duckdb-cli-<plat>.tar.gz` where v1.x published `duckdb_cli-<plat>.zip`.
  `registry-smoke.yml`, `scripts/gen_tpch.sh` and the Linux recipes in
  `docs/CI_RECIPES.md` fetch the tarball; `duckdb-compat.yml` picks the asset
  by tag and gains v2.0.0 legs; `scripts/get_duckdb_libs.sh` keeps its v1.5.5
  pin and says so, exiting 1, when asked for a v2 tag.
- [x] **Versions** — extension v0.7.1 from the community registry
  (`INSTALL gpudb FROM community;`); the pip 0.7.1 platform wheels bundle the
  v0.7.0 extension binary, which has everything above except
  `gpu_functions()`. Journal: [docs/RESEARCH_NOTES.md](RESEARCH_NOTES.md).

### Shipped in v0.7.0 (2026-09-20)

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
  CPU reference, Metal and CUDA — on by default on both GPUs, with
  `GPUDB_CUDA_EXACT=0` as the way to turn the CUDA path off.
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
  says where each statement ran and why, `.gpu` / `.residents` / `.memory`
  (which reports the physical device total, plus what the driver says the
  process holds), and the `duckdb-gpudb` distribution (import name `gpudb`).
- [x] **`pip` carries the binary** — the platform wheels for Apple Silicon
  (macOS 15 or later) and x86-64 Linux (glibc 2.34 or newer) bundle the v0.7.0
  extension, so `pip install duckdb-gpudb` is the whole install; one binary
  loads under both DuckDB 1.4.5 and 1.5.5. macOS binaries now declare a 15.0
  deployment target — the floor of the binary the registry has been serving —
  and the Linux extension links libstdc++ and libgcc statically so it carries
  no `GLIBCXX_` floor from its build machine. The unit binary skips the
  CUDA-only device checks where there is no device, and the root `Makefile`
  refuses to build outside a git checkout, where the version stamp would be
  git's error text.
- [x] Design: [docs/TRANSPARENT_DESIGN.md](TRANSPARENT_DESIGN.md),
  [docs/RESIDENT_COLUMNS_DESIGN.md](RESIDENT_COLUMNS_DESIGN.md);
  journal: [docs/RESEARCH_NOTES.md](RESEARCH_NOTES.md);
  release notes: [docs/RELEASE_NOTES_v0.7.md](RELEASE_NOTES_v0.7.md).

### Shipped in v0.6.0
- [x] **Resident GROUP BY / top-k from SQL, both backends** — `gpu_groupby_sum_resident` / `gpu_groupby_sum_resident_f64` / `gpu_groupby_count_resident` return `(key, sum, count)` rows sorted by key; `gpu_topk_resident[_f64]` returns `(idx, value)` for `ORDER BY … LIMIT k`. Rides the upload-once model and the same cached device sort the joins use as a build side (one sort serves both). Segmented reduce with no hash table and no atomics on Metal; CUB `reduce_by_key` on CUDA. `_having(name, cmp, threshold)` and `_topk(name, k, order)` forms evaluate `HAVING` / `ORDER BY aggregate LIMIT k` on the device (Metal: block compaction + 8-pass radix select; CUDA: CUB select + 8-pass radix select) so only survivors cross to DuckDB. Verified against native both ways on TPC-H SF1/10/50 — statement time against statement time in the same process: **5.4–6.1× (Metal)** on Q18's inner query with the HAVING on the device, 6–8× on `HAVING count(*) >= 7`, 3.6–4.1× on the top-10 groups by sum; on CUDA the device-side HAVING is **13–24× (SF50) / 17× (SF10)** and the top-10 groups 8–12×; returning all 15M–75M groups is 2.3–2.9× (Metal) and 1.5–2.0× end-to-end on CUDA (~32–47× on-device, bounded by copying 24 bytes per group over PCIe). The losing rows are kept: low-cardinality GROUP BY on Metal, and the first top-k call against native's zonemap top-k.
- [x] **Composable results** — the GPU produces the rows, DuckDB does the rest: `SELECT key, sum FROM gpu_groupby_sum_resident('l') WHERE sum > 300 ORDER BY sum DESC LIMIT 10` is plain SQL over a small result.
- [x] **Adversarial parity harness for GROUP BY** — `scripts/groupby_parity_check.sh`: 11 scenarios × 7 checks, incl. runs placed exactly on the kernels' 64-chunk / 256-block boundaries; SQL suite gained a `-- setup:` directive so table functions are tested in the documented sequential form.
- [x] **Metal radix-sort fix** — the sort behind the v0.5 join build cache skipped a byte pass whenever min and max agreed on that byte; wrong for keys between them that differ there (TPC-H returnflag/linestatus packed keys). Fixed, regression scenarios in both parity harnesses; exposure of the v0.5.0 Metal binary stated in [KNOWN_ISSUES.md](../KNOWN_ISSUES.md).
- [x] **Metal device memory is now released** — the Metal host code was built without ARC since v0.1, so every `MTLBuffer` (resident columns, sort caches, scratch) leaked until process exit; `gpu_drop_resident` now actually frees GPU memory.
- [x] **Pre-release adversarial audit** — a find/verify pass over the sort, kernels, C-API layer, hybrid planner, SQL semantics vs native (NULLs, overflow, NaN/-0.0), the v0.5 join surface after the sort fix, the CUDA branch (static) and every documentation claim; all confirmed findings fixed or documented in [KNOWN_ISSUES.md](../KNOWN_ISSUES.md) before the tag.
- [x] Design: [docs/GROUPBY_RESIDENT_DESIGN.md](GROUPBY_RESIDENT_DESIGN.md).

### Shipped in v0.5.0
- [x] **GPU joins from SQL, both backends** — `gpu_upload_pair` + fused `gpu_[left_|semi_|anti_]join_{sum,count}_resident[_f64]` (inner / left / semi / anti; right / full as documented compositions) and the row-returning `gpu_join_rows_resident`. Sorted-build + binary-search probe with the sorted side cached on the device. Verified against native DuckDB's hash join end-to-end on TPC-H SF10/SF50: **11.7× (Metal) / 27–37× (CUDA)** inner join-sum at SF50, **22× / ~376×** on the EXISTS semi-join; i64 bit-exact, f64 within the 1e-9 relative tolerance contract (measured ≤4e-11). The losing row is kept: row materialisation across PCIe loses to native on discrete GPUs.
- [x] **Adversarial parity harness** — `scripts/join_parity_check.sh`: 11 scenarios × 12 checks (dup-heavy, Knuth-hash, Zipf skew, int64 boundaries, negative keys, no-match, all-match, inverted sizes), native and gpudb computed in the same statement; passes on both machines.
- [x] **Metal hash join + hybrid join planner + on-device segment reduce** — contributed by [@lmangani](https://github.com/lmangani) ([PR #43](https://github.com/singhpratech/duckdbgpumetaldbram/pull/43)); this release lands that commit as the base of the join stack.
- [x] **Colab notebook runs real CUDA** — requests a T4 runtime automatically, builds with `GPUDB_REQUIRE_CUDA=1` (configure fails loudly instead of silently falling back to CPU), and the test cell asserts the CUDA backend actually ran.

### Shipped in v0.4.0
- [x] **Resident-column SQL surface** — `gpu_upload` / `gpu_sum_resident` / `gpu_min_resident` / `gpu_max_resident` / `gpu_sum_resident_f64` / `gpu_resident_info` / `gpu_last_stats` / `gpu_drop_resident` / `gpu_build_info`. The GPU genuinely executes SQL reductions on both CUDA and Metal — up to **25×** over native (see Numbers). Hardened by a three-reviewer adversarial pass pre-release: buffer-pool cap (window-frame O(n²) OOM → clean error), mixed-name/NULL-name guards, defined overflow wrap, truthful dispatch stats.
- [x] **CUDA-ready community build** — the root Makefile auto-detects nvcc with a statically linked CUDA runtime (no libcuda/libcudart dynamic deps; loads clean on GPU-less machines) so the registry's Linux binary flips to CUDA automatically when the registry's build tooling ships its CUDA toolchain. Also fixed a CMake ordering bug that had every prior CUDA build shipping single-arch fatbins.
- [x] **Full dual-platform benchmark record** — TPC-H SF1→SF100, six columns, correctness-gated, both backends, in [BENCHMARK.md](../BENCHMARK.md).
- [x] [Community Extensions PR #2503](https://github.com/duckdb/community-extensions/pull/2503) **merged** (2026-08-17) — the registry now serves **v0.4.0**, resident-column surface included.


### Shipped in v0.3.0
- [x] **Streaming aggregate states** — the SQL aggregate path rewritten from "buffer every value, reduce at finalize" to running accumulators, the same algorithmic shape as native DuckDB. End-to-end on rewritten TPC-H Q6/Q1 and high-cardinality GROUP BY: parity with native (the v0.2.0 buffered path lost 3×–110×; the SF10 GROUP BY cell alone went from 11.05 s to 0.110 s). Full before/after in [BENCHMARK.md](../BENCHMARK.md). `GPUDB_FORCE_BACKEND` is a no-op on this path now (it routed the deleted machinery).
- [x] **`gpu_min(DOUBLE)` / `gpu_max(DOUBLE)`** — all three aggregates are now overload sets carrying `(BIGINT)->BIGINT` and `(DOUBLE)->DOUBLE`. No backend-interface change was needed under the streaming design. NaN ordering matches native (NaN sorts greatest). Type matrix in [KNOWN_ISSUES.md](../KNOWN_ISSUES.md).

### DuckDB Community Extension milestones
- [x] [Community Extensions PR #1898](https://github.com/duckdb/community-extensions/pull/1898) **merged** — `INSTALL gpudb FROM community` is live (no `-unsigned` flag needed), and gpudb is [listed on duckdb.org](https://duckdb.org/community_extensions/extensions/gpudb).
- [x] [Community Extensions PR #2404](https://github.com/duckdb/community-extensions/pull/2404) **merged** — the community build ships **v0.3.0** (streaming aggregates + DOUBLE overloads on all four platforms).

### Shipped in v0.2.0
- [x] **SQL-correct NULL semantics** (PR #44) — `gpu_sum`/`gpu_min`/`gpu_max` over empty or all-NULL input now return SQL `NULL` (not 0), matching native DuckDB on every path: plain aggregate, GROUP BY groups, and window frames.
- [x] **`gpu_sum(DOUBLE) -> DOUBLE`** (PR #45) — a real second overload via the C API aggregate function set. Doubles ride the existing int64 state machinery as raw bit patterns (zero state-layout change); only the finalize differs. `INTEGER`/`SMALLINT`/`TINYINT` work via DuckDB's implicit widening to the `BIGINT` overload (locked in by tests). Type matrix in [KNOWN_ISSUES.md](../KNOWN_ISSUES.md).

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
