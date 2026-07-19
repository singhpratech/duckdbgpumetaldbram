# Known issues

Status as of 2026-05-09 night, `v0.1.0` shipped. **All known functional bugs resolved.**

If you do hit something here, fall back to native DuckDB `sum()` / `min()` / `max()` / window functions for that query — the hybrid planner inside the extension already does this automatically wherever it can.

---

## Open issues

None.

---

## Documented intentional divergences from SQL spec

None. The `MIN/MAX/SUM` empty-input / all-NULL → `0` divergence was resolved in
`v0.2.0` (see the resolved table below); those aggregates now return SQL `NULL`,
matching native DuckDB, and `IS NULL` on such a result is now `true`.

### Hardware-imposed limits (won't fix)

| Behavior | Reason |
|---|---|
| `gpu_sum(DOUBLE)` falls back to host on Apple Silicon | Apple GPUs do not implement IEEE-754 double precision in MSL. Result is correct; just no GPU acceleration. |

---

## Resolved issues

| Issue | Fixed in |
|---|---|
| `MIN/MAX/SUM(empty input)` returned `0` instead of SQL `NULL` | v0.2.0 (pending PR) — finalize marks the output row invalid via `duckdb_vector_ensure_validity_writable` + `duckdb_validity_set_row_invalid` when a state accumulated zero non-NULL values; `set_special_handling` registered so DuckDB does not short-circuit our NULL handling |
| `MIN/MAX/SUM(all NULLs)` returned `0` instead of SQL `NULL` | v0.2.0 (pending PR) — same fix; empty state buffer (all input rows NULL) now yields SQL `NULL`, including per-group in GROUP BY (single, batched-GPU, and per-state paths) |
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
./scripts/run_sql_tests.sh      # 5 .test files, 46 queries, 0 fail / 0 xfail
```
