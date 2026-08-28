# Known issues

Status as of 2026-08-22, `v0.5.0`. **All known functional bugs resolved.**

If you do hit something here, fall back to native DuckDB `sum()` / `min()` / `max()` / `JOIN` / window functions for that query — every shape the GPU path does not cover runs natively.

---

## Open issues

None.

---

## Documented intentional divergences from SQL spec

None. The `MIN/MAX/SUM` empty-input / all-NULL → `0` divergence was resolved in
`v0.2.0` (see the resolved table below); those aggregates now return SQL `NULL`,
matching native DuckDB, and `IS NULL` on such a result is now `true`.

### Design notes (v0.3.0)

| Behavior | Reason |
|---|---|
| The SQL aggregate path (`gpu_sum`/`gpu_min`/`gpu_max` called from SQL) does not dispatch to the GPU | Deliberate v0.3.0 design, not a fallback. DuckDB feeds aggregates pre-grouped 2048-row chunks; on unified memory, copying those out to feed a GPU reduction is pure overhead (measured 3×–110× end-to-end loss in v0.2.0 — see BENCHMARK.md). v0.3.0 streams running accumulators instead and reaches parity with native. GPU reductions remain at operator level (`gpudb-bench` etc.); GPU value on the SQL path moves to the join track. |
| Apple GPUs have no IEEE-754 double precision in MSL | Still true and still relevant to operator-level f64 work on Metal (runs on host there). No longer affects the SQL aggregate path (see row above). |
| `GPUDB_FORCE_BACKEND` no longer affects SQL aggregates | The env var routed the deleted buffered/batched path; it is a silent no-op on the aggregate path as of v0.3.0. Operator-level tools still honor backend selection. |

### Design notes — resident-column surface (v0.4.0)

| Behavior | Reason |
|---|---|
| `gpu_sum_resident` (and the streaming `gpu_sum`) wrap on int64 overflow where native `sum(BIGINT)` promotes to `HUGEINT` | The GPU kernels and the C ABI accumulate in 64-bit; wrap is two's-complement (uint64 accumulate — defined behavior, identical on CPU/CUDA/Metal). Use native `sum()` if your column can exceed int64. |
| One `gpu_upload` call = one column name; a name column with mixed values errors | Values from different names silently merging into one column is data corruption; use a constant name or `GROUP BY` the name column. |
| `gpu_upload` names may not be NULL; read-side NULL names yield SQL `NULL` rows | Upload with NULL name used to silently register the column under `''`. |
| Total memory buffered by in-flight `gpu_upload` states is capped (default 4 GB, `GPUDB_UPLOAD_POOL_MAX_MB`) | `gpu_upload` inside a running window frame buffers O(n²) (combine per output row) and would otherwise OOM the host from a KB-scale input. The cap turns that into a clean error. |
| An unreferenced `gpu_upload` subquery column is pruned and the upload never runs | DuckDB's unused-column elimination; always reference the upload count in the outer query (see `src/extension/gpu_resident.cpp` header). |
| Resident f64 columns run on the host on Apple Silicon (`reason=Resident_OnCpu`) | No IEEE-754 doubles in MSL. The host path is parallel (chunked reduction) as of v0.4.0 — measured faster than a native DuckDB scan on M4 Max — and f64 sums are deterministic per upload, not per dataset (parallel upload order affects rounding, same as native). |
| Whole-column `gpu_min_resident`/`gpu_max_resident` are slower than native `min()`/`max()` on unfiltered persistent tables | Native answers those from zonemap statistics without scanning — a metadata lookup no engine can beat. The resident path wins where a real reduction happens (sums, or data DuckDB has no stats for). |

### Design notes — fused resident joins (v0.5.0)

| Behavior | Reason |
|---|---|
| Join keys must be `BIGINT` resident columns; payloads may be `BIGINT` or `DOUBLE` | The v0.5 join ABI is single-column i64 equi-join. `DOUBLE`/`DECIMAL`/`VARCHAR` keys, composite keys, and non-equi joins are not on the GPU path — write them as native `JOIN`s. |
| `gpu_upload` skips NULL values and `gpu_upload_pair` skips a row when **either** its key or its payload is NULL — resident columns carry no NULLs | There is no validity mask on the device. Consequences: `gpu_join_*_count_resident` counts non-NULL rows only, LEFT / ANTI never emit NULL-key probe rows, and (v0.6) `gpu_groupby_*_resident('p')` equals native `SELECT k, sum(v), count(v) FROM t WHERE k IS NOT NULL AND v IS NOT NULL GROUP BY k` — no NULL-key group, `count` is `COUNT(payload)` not `COUNT(*)`, a key whose payloads are all NULL is absent (native returns `(k, NULL, n)`), and top-k never returns NULL rows. Filter or `coalesce` before upload if the native semantics matter. |
| An upload whose input yields zero non-NULL rows (or pairs) registers nothing and does **not** replace a previously uploaded column of the same name | The upload aggregate never sees a row, so finalize has nothing to register; the old column stays. Pipelines that refresh a name should `gpu_drop_resident(name)` (both `name.k` and `name.v` for pairs) first, or check the upload's returned row count. |
| Keys and payloads must be uploaded together with `gpu_upload_pair(name, key, payload)` — registers `name.k` and `name.v` | Two separate `gpu_upload` calls over a parallel scan do not preserve row alignment between key and payload (DuckDB scans chunks in parallel); a misaligned pair silently sums the wrong rows. `gpu_upload_pair` interleaves both in one scan. |
| `gpu_join_sum_resident` returns SQL `NULL` when no rows join; `gpu_join_count_resident` returns `0` | Matches native `sum()` / `count(*)` over an empty join. |
| INNER and LEFT carry full build-side multiplicity (a probe row joined to 3 equal build keys contributes 3×); SEMI/ANTI contribute each probe row at most once | Native join semantics. `matched` in `gpu_last_stats()` counts the rows contributing for that kind (ANTI counts non-matching probe rows). |
| RIGHT / FULL OUTER are compositions, not separate functions | Probe-payload sum equals INNER / LEFT respectively; the missing-side `COUNT(*)` term is `gpu_anti_join_count_resident` with the sides swapped. |
| `DOUBLE` join sums are compared to native within a relative tolerance (1e-9 contract; measured ≤ 1e-11 on both backends), never bit-for-bit | Summation order differs per backend (CUDA: on-device atomics; Metal: GPU multiplicity + host stream). `BIGINT` sums are bit-exact on both. |
| On Apple Silicon the `DOUBLE` payload reduction runs on the host after the GPU computes per-row multiplicity ("GPU searches, host streams") | No IEEE-754 doubles in MSL. Still 5–6× faster than native at SF50 because the random-access part is on the GPU and the host pass is a sequential stream. |
| `gpu_join_rows_resident` (row-returning) wins modestly on unified memory (1.16× at 60M pairs on M4 Max) and **loses to native on discrete GPUs** (0.5× on RTX 4090: ~96% of wall is the ~960 MB device→host copy) | Materialising 60M index pairs across PCIe costs more than DuckDB's native hash join. Use the fused aggregate variants on discrete GPUs; the rows function is the composability primitive for unified-memory machines. Output capped at `GPUDB_JOIN_ROWS_MAX_M` (default 100M rows) with a clean error. |
| The build side's sorted-key + permutation cache (16 bytes per build row) lives on the device outside the `GPUDB_UPLOAD_POOL_MAX_MB` accounting | Device OOM surfaces as a clean `runtime_error` on the first join call rather than at upload. |
| TPC-H SF50-scale pair uploads need `GPUDB_UPLOAD_POOL_MAX_MB` raised (10–12 GB) | Default 4 GB pool refuses with the documented error; same guardrail as resident columns. |
| `gpu_join_rows_resident` needs the uploads to run as earlier statements (`gpudb-sql --multi`, or separate statements on one connection) | A table function's bind/init cannot be sequenced after an upload in the same statement. The scalar join-aggregate functions have no such restriction. |
| `probe_idx` / `build_idx` from `gpu_join_rows_resident` refer to **upload order**, which equals table order only when the upload scan was not parallel | Same ordering caveat as any parallel scan; use the indices to look up rows, not as row numbers of the source table. |
| Multi-table join chains and `GROUP BY` over join results are not on the GPU path | Roadmap (v0.7). `GROUP BY` and `ORDER BY … LIMIT k` over resident columns are on it as of v0.6.0 (next table). Unsupported shapes run natively — gpudb never breaks a query. |

### Design notes — resident GROUP BY / top-k (v0.6.0)

| Behavior | Reason |
|---|---|
| `gpu_groupby_sum_resident_f64_having` / `_f64_topk` filter on the host, not the device | `DOUBLE` sums are finished on the host on Metal (no doubles in MSL) and the filter runs there through the reference implementation; the saving is only the row streaming (measured 1.4–2.6× vs native, against 5–8× for the BIGINT HAVING forms and 4× for BIGINT top-k on Metal). CUDA filters f64 on the device. |
| `_topk` tie order is unspecified | When several groups share the k-th aggregate, which of them are returned — and in what order among equals — is backend-defined (Metal/CPU: key ascending; CUDA: radix-sort order). An `ORDER BY` on the result fixes only the order among the returned equals; if *which* tied groups come back matters, ask for k plus the expected ties, or use the `_having` form and rank in SQL. Cross-backend checks compare multisets. |
| Metal `_topk` with k close to the group count is slower than the plain form + `ORDER BY` | The Metal top-k always runs the 8-pass radix select and then orders the k rows on the host; for k a large fraction of the groups that host sort dominates. CUDA switches to a full device sort for k > groups/8. On Metal use `gpu_groupby_sum_resident` + SQL `ORDER BY … LIMIT` when k is not small. |
| The first filtered call on a backend allocates device scratch for the finalized groups | 24 B per group (1.8 GB at 75M groups) allocated once and reused by every later filtered call on any pair; the first (B′) statement at SF50 on Metal took 0.9 s, later ones 78 ms. Counted in the process's device memory, outside `GPUDB_UPLOAD_POOL_MAX_MB`. |
| Every Metal device allocation made before v0.6.0 (resident columns, sort caches, scratch) was never released — `gpu_drop_resident` freed the host-side accounting but not the GPU buffer | The Metal host code was compiled without ARC; v0.6.0 turns it on (`-fobjc-arc`), so dropped columns and their caches now release device memory. Long-lived processes on v0.5.0 that upload/drop repeatedly grow until restart. |
| `gpu_groupby_*_resident` / `gpu_topk_resident[_f64]` are table functions; the uploads they read must run as earlier statements on the same connection | A table function's bind/init cannot be ordered after an aggregate in the same statement (a single cross-product reference happens to work, two do not — DuckDB's pipeline ordering gives no guarantee). The result types are fixed per function (`_f64` variants) for the same reason: bind cannot inspect a column that a same-statement upload has not produced yet. |
| Rows come out sorted by key ascending on every backend | The GROUP BY is a sort + segmented reduce over the cached radix sort (the join build cache), so order is free and cross-backend parity is a plain row diff. Need a different order? `ORDER BY` the result — it is small. |
| `sum` of a `BIGINT` payload wraps on int64 overflow where native `sum()` promotes to `HUGEINT` | Same uint64 wrap rule as `gpu_sum_resident` and the join sums; bit-identical on CPU/CUDA/Metal. Use native `sum()` if a group can exceed int64. |
| `DOUBLE` group sums compare to native within a relative tolerance (1e-9 contract), never bit-for-bit; on Apple Silicon they run on the host after the GPU sorts and gathers | Summation order is backend-defined; no IEEE-754 doubles in MSL. Same rule as the f64 join. |
| Top-k tie order among equal values is unspecified | Same as SQL `ORDER BY` without a tiebreaker; the sort is not stable across backends. Cross-backend checks compare the multiset of values, not `idx`. |
| Group count is capped at `GPUDB_GROUPBY_ROWS_MAX_M` million (default 100) with a clean error naming the actual count | Checked before any device→host copy on every backend, never truncated — the same guardrail as `GPUDB_JOIN_ROWS_MAX_M`. |
| The sort cache costs 16 bytes per row per sorted column on the device (sorted keys + permutation), outside the `GPUDB_UPLOAD_POOL_MAX_MB` accounting; on Metal the radix sorter additionally keeps its ping-pong scratch (32 bytes × the largest row count sorted so far) for the life of the process, on CUDA the sort scratch is transient | Built lazily on the first GROUP BY / top-k / join-build use of a column, shared by all of them, freed with the column. Device OOM surfaces as a clean error on that first call (CUDA: `cudaMalloc join cache (sorted keys) failed: out of memory`). Worked example: at SF50 two resident pairs (4.8 GB each) plus one sort cache do not fit a 16 GB GPU — `gpu_drop_resident` what you no longer need. |
| `gpu_groupby_*_resident` / `gpu_topk_resident*` results reflect what was uploaded: a NULL key or NULL payload row was never uploaded (see the v0.5 table above) | Same rule as every resident op; stated here because native `GROUP BY` would show a NULL group and `count(*)` would include NULL payloads. Test 14 in `test/sql/gpu_groupby_resident.test` pins the behaviour. |
| ~~On CUDA, a device fault raised inside a Thrust call aborted the process instead of surfacing as a SQL error~~ | **Resolved in v0.6.0**: the resident CUDA paths run CUB primitives on explicit temp storage (no Thrust temporaries), so a sticky device fault is mapped to a `runtime_error` naming the CUDA error and the statement fails cleanly; pinned by a unit test that re-executes the binary with a fault-injection hook and checks the child exits normally. Linux + CUDA only. |
| Low-cardinality GROUP BY over a `BIGINT` key differs between backends: CUDA wins with the data resident (7 groups at SF10: 31 ms native vs 5.9 ms), Metal loses (4 groups at SF10: 31–35 ms native vs 47 ms) | Both run the same sort + segmented reduce; on Metal the gather of every value through the sort permutation is the cost and there is no fp64/atomic hash path to fall back to. Native's streaming aggregate is scan-bound and very fast at low cardinality. Rows in BENCHMARK.md. |
| On discrete GPUs (CUDA) the device→host copy of the `(key, sum, count)` result dominates wall time for high-cardinality outputs (24 bytes/group); the kernel is a few percent of wall | Same honest split as `gpu_join_rows_resident`. `gpu_last_stats()` shows `kernel_ms` vs `wall_ms`. On unified memory (Metal) there is no copy: the GPU writes the result vectors in place. |
| Low-cardinality GROUP BY (a handful of groups over tens of millions of rows) is not the target shape | The sort is wasted work there; native's streaming aggregate is scan-bound and very fast. Measured rows for both shapes are in BENCHMARK.md — pick by cardinality. |

### Type support (v0.3.0)

| Aggregate | Supported input types | Notes |
|---|---|---|
| `gpu_sum` | `BIGINT`, `DOUBLE` (and `INTEGER`/`SMALLINT`/`TINYINT` via DuckDB's implicit widening to the `BIGINT` overload) | Smaller integer types carry no dedicated overload — they widen to `BIGINT`, which is why the result type of `gpu_sum(INTEGER)` is `BIGINT`. `BIGINT` sums wrap on int64 overflow where native `sum()` promotes to `HUGEINT`. |
| `gpu_min` / `gpu_max` | `BIGINT`, `DOUBLE` (and smaller integers via implicit widening) | `DOUBLE` overloads added in v0.3.0. NaN ordering matches native DuckDB: NaN sorts greatest, so `gpu_max` over a NaN-containing group is `NaN`, `gpu_min` prefers finite values, and an all-NaN group is `NaN`. |

**Casting gotchas (all three aggregates):** `HUGEINT`, `DECIMAL`, and `FLOAT` arguments implicitly cast to the `DOUBLE` overload — values beyond 2^53 lose integer/scale precision where native aggregates are exact, and the result type is `DOUBLE` (native preserves `HUGEINT`/`DECIMAL`). `VARCHAR` arguments are a binder error (native `min`/`max` do lexicographic comparison). Use native aggregates where those types matter.

---

## Resolved issues

| Issue | Fixed in |
|---|---|
| Metal radix sort (the sorted build cache behind the v0.5 fused / row joins, and the v0.6 GROUP BY / top-k) produced an unsorted result for key sets whose min and max agree on a byte that other keys differ in — e.g. TPC-H `l_returnflag`/`l_linestatus` packed to one `BIGINT` (`0x4146`, `0x4E46`, `0x4E4F`, `0x5246`: min and max share low byte `0x46`, `0x4E4F` does not) came back as 66,085 groups instead of 4 — and, less exotically, any dense key range whose min and max share a low byte, such as `0..256`, `0..65536` or `0..2^20`; a Metal join against such build keys could miss matches. Affects the **v0.5.0 Metal binary** for such key sets (CUDA unaffected; ranges like `1..150000` or TPC-H `orderkey`/`custkey` differ in every low byte and were not hit). The same heuristic existed in a second copy inside the non-resident Metal `GroupByAggregator` radix path (C++ API / `gpudb-groupby-bench` only, not reachable from SQL) and is fixed there too | v0.6.0 — the pass-skip heuristic compared min and max byte-by-byte and skipped every pass where they matched; that is only valid for bytes above the highest differing byte (the prefix every key in `[min, max]` shares). Now: all passes at or below the highest differing byte run. Regression coverage in the unit suite and in both parity harnesses (`join_parity_check.sh`, `groupby_parity_check.sh`) |
| Metal v1 `HashJoinProbe` lost a small number of matches on GitHub's macOS runner (`Apple Paravirtual device`): 199,973 vs 199,998 on a 50k×200k join | v0.5.0 — the slot-lock tables spun on a locked slot for a bounded number of attempts (a GPU gives no cross-threadgroup forward-progress guarantee, so the spinner could give up and silently drop its row), read another thread's key through relaxed atomics, and chained init → build → probe in one compute encoder with no memory barrier. Now: wait-free insert (never spin, never read a foreign key during build), probe scans the whole run and keeps the smallest build index (exact first-row-wins on both paths), `memoryBarrierWithScope:MTLBarrierScopeBuffers` between dependent dispatches. Regression tests: duplicate-heavy 97-key build and a 700k-row 5-key build on the partitioned path |
| CUDA v1 `HashJoinProbe` (operator-level, `gpudb-hashjoin-bench`) emitted every build row for duplicate build keys, diverging from the CPU reference and the header contract (first build row wins) | v0.5.0 — one slot per key, `atomicMin` on the smallest build index. Behaviour change only for callers relying on the old multimap output of the v1 probe; the v0.5 fused joins carry full multiplicity by design and are unaffected |
| `gpu_min/gpu_max(DOUBLE)` leaked the fold identity (`+inf`/`-inf`) on all-NaN groups, and `gpu_max` dropped NaN where native returns NaN | v0.3.0 — caught by pre-release adversarial review, fixed before the overloads ever shipped: f64 min/max folds now use a NaN-aware total order matching native (NaN sorts greatest) |
| `MIN/MAX/SUM(empty input)` returned `0` instead of SQL `NULL` | v0.2.0 (PR #44) — finalize marks the output row invalid via `duckdb_vector_ensure_validity_writable` + `duckdb_validity_set_row_invalid` when a state accumulated zero non-NULL values; `set_special_handling` registered so DuckDB does not short-circuit our NULL handling |
| `MIN/MAX/SUM(all NULLs)` returned `0` instead of SQL `NULL` | v0.2.0 (PR #44) — same fix; a state with zero non-NULL values (all input rows NULL) yields SQL `NULL`, including per-group in GROUP BY |
| `gpu_sum(v) OVER ()` unbounded-frame SIGSEGV | PR #22 (BufferPool POD state + magic-word probe defends update() against `WindowConstantAggregator`'s CONSTANT_VECTOR state) |
| `gpu_sum(v) OVER (PARTITION BY g ORDER BY k)` non-determinism / wrong values | PR #18 (combine() copy-not-move) + PR #22 (POD state) |
| `gpu_sum(v) OVER (ORDER BY k)` running-sum chunk-boundary state loss | PR #18 |
| Mid-cardinality (50–63 groups) GROUP BY wrong totals | PR #21 (CPU per-state in mid-card regime) |
| `metal_aggregator.mm` duplicate decls (build broken on macOS) | PR #20 |
| `scripts/run_sql_tests.sh` couldn't run on macOS (missing `timeout`) | PR #20 |

---

## How to verify

```bash
git clone https://github.com/singhpratech/duckdbgpumetaldbram
cd duckdbgpumetaldbram
./scripts/get_duckdb_libs.sh
./scripts/build.sh
./scripts/local_check.sh        # builds + cpp tests + smoke benchmarks
./scripts/run_sql_tests.sh      # 9 .test files, 0 fail (4 GUARDRAIL cases assert that misuse is rejected)
./scripts/join_parity_check.sh  # 11 adversarial join scenarios, native vs gpudb in the same statement
```
