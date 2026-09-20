# Release history

What shipped in each release, newest first.

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
  says where each statement ran and why, `.gpu` / `.residents` / `.memory`, and
  the `duckdb-gpudb` distribution (import name `gpudb`).
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
