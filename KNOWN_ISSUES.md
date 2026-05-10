# Known issues

Status as of 2026-05-09 night, `v0.1.0` shipped. **All known functional bugs resolved.**

If you do hit something here, fall back to native DuckDB `sum()` / `min()` / `max()` / window functions for that query — the hybrid planner inside the extension already does this automatically wherever it can.

---

## Open issues

None.

---

## Documented intentional divergences from SQL spec

These are **not** bugs — explicit design choices for v0.1.0. Will be revisited in v0.2.0.

| Behavior | Native DuckDB | Our extension |
|---|---|---|
| `MIN/MAX/SUM(empty input)` | NULL | 0 |
| `MIN/MAX/SUM(all NULLs)` | NULL | 0 |

Returning `0` is the C++ aggregator's natural behavior; properly returning SQL `NULL` requires using `duckdb_validity_set_row_invalid` on the output vector. SQL queries that test for `IS NULL` on a fully-empty/all-NULL aggregate result currently get `false`.

### Hardware-imposed limits (won't fix)

| Behavior | Reason |
|---|---|
| `gpu_sum(DOUBLE)` falls back to host on Apple Silicon | Apple GPUs do not implement IEEE-754 double precision in MSL. Result is correct; just no GPU acceleration. |

---

## Resolved issues (fixed in `v0.1.0`)

| Issue | Fixed in |
|---|---|
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
