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
| Multi-table join chains, `GROUP BY` over join results, and `ORDER BY` are not on the GPU path | Roadmap (v0.6). Unsupported shapes run natively — gpudb never breaks a query. |

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
./scripts/run_sql_tests.sh      # 9 .test files, 0 fail (guardrail cases are asserted expected-fails)
./scripts/join_parity_check.sh  # 11 adversarial join scenarios, native vs gpudb in the same statement
```
