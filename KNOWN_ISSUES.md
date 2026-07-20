# Known issues

Status as of 2026-07-19, `v0.3.0`. **All known functional bugs resolved.**

If you do hit something here, fall back to native DuckDB `sum()` / `min()` / `max()` / window functions for that query.

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
./scripts/run_sql_tests.sh      # 5 .test files, 60 queries, 0 fail / 0 xfail
```
