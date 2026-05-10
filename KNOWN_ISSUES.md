# Known issues

Pre-alpha. Status as of 2026-05-09 night, post-`v0.1.0` launch candidate.

If you are bitten by something here, fall back to native DuckDB `sum()` / `min()` / `max()` / window functions for that query — the hybrid planner inside the extension already does this automatically wherever it can.

---

## Open issues

### `gpu_sum(v) OVER ()` — unbounded frame SEGFAULTS

```sql
SELECT k, gpu_sum(v) OVER () FROM (SELECT range AS k, range::BIGINT AS v FROM range(20));
```

Expected: every row shows the constant `190`.
Actual: SIGSEGV.

Workaround: use native `sum() OVER ()`. The query has been commented out from `test/sql/gpu_window.test` for v0.1; a parallel agent on `fix/window-bugs-followup` is investigating the root cause (suspected: state lifecycle for the unbounded window frame is different from the standard aggregate path).

Target fix: v0.1.1 (within 1 week of launch).

---

## Documented intentional divergences from SQL spec

These are **not** bugs — explicit design choices for v0.1.0. Will be revisited in v0.2.0.

| Behavior | Native DuckDB | Our extension |
|---|---|---|
| `MIN/MAX/SUM(empty input)` | NULL | 0 |
| `MIN/MAX/SUM(all NULLs)` | NULL | 0 |

Returning `0` is the C++ aggregator's natural behavior; properly returning SQL `NULL` requires using `duckdb_validity_set_row_invalid` on the output vector. SQL queries that test for `IS NULL` on a fully-empty/all-NULL aggregate result currently get `false`.

---

## Resolved issues (fixed in `v0.1.0` candidate)

| Issue | Fixed in |
|---|---|
| Window+PARTITION BY non-determinism | PR #18 (combine() copy-not-move) + PR #20 |
| Window OVER (ORDER BY) running-sum chunk-boundary state loss | PR #18 |
| Mid-cardinality (50-63 groups) GROUP BY wrong totals | PR #21 (CPU per-state in mid-card regime) |
| metal_aggregator.mm duplicate decls (build broken on macOS) | PR #20 |
| `scripts/run_sql_tests.sh` couldn't run on macOS (missing `timeout`) | PR #20 |

---

## How to verify

```bash
git clone https://github.com/singhpratech/duckdbgpumetaldbram
cd duckdbgpumetaldbram
./scripts/get_duckdb_libs.sh
./scripts/build.sh
./scripts/local_check.sh        # builds + 77 unit tests + smoke benchmarks
./scripts/run_sql_tests.sh      # 5 .test files, 44 queries, 0 fail / 0 xfail
```
