# Known issues

Pre-alpha. Tracks issues Agent C (the SQL test suite agent) discovered against the DuckDB extension on `main`. Each is reproducible, each will land a fix in `v0.1.1` (target: within one week of `v0.1.0` launch).

If you are bitten by one of these in the meantime, fall back to native DuckDB `sum()`/`min()`/`max()` for that query — the hybrid planner already does this automatically inside the extension whenever it can, but a few code paths still need the same treatment.

---

## Window functions

### `gpu_sum(v) OVER (ORDER BY k)` — running sum loses state across DuckDB chunk boundaries

```sql
SELECT k, gpu_sum(v) OVER (ORDER BY k) AS gs, sum(v) OVER (ORDER BY k) AS ns
FROM (SELECT range AS k, range::BIGINT AS v FROM range(50));
```

Expected: `gs` grows monotonically (0, 1, 3, 6, 10, 15, ..., 1225) matching `ns`.
Actual: starts correct, **resets to 0 around row 15-16** (DuckDB's small-chunk size for this query).

Workaround: use native `sum()`. Filed as `tests/sql/gpu_window.test` xfail q3.

### `gpu_sum(v) OVER ()` — unbounded frame SEGFAULTS

```sql
SELECT k, gpu_sum(v) OVER () FROM (SELECT range AS k, range::BIGINT AS v FROM range(20));
```

Expected: every row shows the constant `190`.
Actual: SIGSEGV (exit 139).

Workaround: use native `sum() OVER ()`. Filed as `tests/sql/gpu_window.test` xfail q4.

### `gpu_sum(v) OVER (PARTITION BY g ORDER BY k)` — non-deterministic

Sometimes returns wrong values, sometimes segfaults, sometimes hangs.
Reproducer: any small VALUES table with 3+ partitions, run the same query 10 times — at least one attempt fails.

Workaround: use native `sum() OVER (PARTITION BY ...)`. Filed as `tests/sql/gpu_window.test` xfail q5.

**Root cause analysis (in flight):** previous fix (PR #18) addressed combine() copy-vs-move, added a CPU short-circuit, and a GPU-call mutex. These addressed *some* states but not all. Window-mode state lifecycle in DuckDB is more complex than the standard aggregate path.

---

## Mid-cardinality GROUP BY

### `gpu_sum(...) GROUP BY ...` produces wrong totals when group count is between 50 and 63

```sql
SELECT count(*) AS groups, sum(g) AS total
FROM (
    SELECT range % 100 AS k, gpu_sum(range::BIGINT) AS g
    FROM range(10000000) GROUP BY range % 100
);
```

Expected: `groups=100`, `total=49999995000000`.
Actual: `groups=100` (correct), `total != 49999995000000` (wrong).

Both extremes are correct:
- count == 1 (no GROUP BY): correct
- count >= 64 (`kBatchedGroupByThreshold`): correct (batched single-call GPU path)
- 1 < count < 64: **wrong totals on the per-state finalize loop**

Workaround: use native `sum()`. Filed as `tests/sql/gpu_groupby.test` xfail q7.

**Root cause hypothesis:** singleton CUDA aggregator with shared device buffers — concurrent per-state calls inside finalize race on the input/output device pointers. Likely fix: serialize per-state calls under the same mutex used for the window-bug fix, or always-batch (raise threshold to 1).

---

## Documented intentional divergences from SQL spec

These are **not** bugs — they are explicit design choices for the v1 release. They will be revisited in `v0.2.0`.

| Behavior | Native DuckDB | Our extension |
|---|---|---|
| `MIN/MAX/SUM(empty)` | NULL | 0 |
| `MIN/MAX/SUM(all NULLs)` | NULL | 0 |

Returning `0` is the C++ aggregator's natural behavior; properly returning SQL `NULL` requires using `duckdb_validity_set_row_invalid` on the output vector. Not yet wired up. SQL queries that test for `IS NULL` on a fully-empty/all-NULL aggregate result will currently get `false`.

---

## How to verify these on your machine

```bash
git clone https://github.com/singhpratech/duckdbgpumetaldbram
cd duckdbgpumetaldbram
./scripts/get_duckdb_libs.sh
./scripts/build.sh
./scripts/run_sql_tests.sh        # 5 .test files, ~46 queries, 4 xfails, 0 fails
```

The xfails listed in the runner output map directly to the issues above.

---

## Status

| Issue | Status | Target |
|---|---|---|
| Window: chunk-boundary state loss | In flight (background fix agent) | v0.1.1 |
| Window: `OVER ()` segfault | In flight (background fix agent) | v0.1.1 |
| Window: PARTITION BY non-determinism | In flight (background fix agent) | v0.1.1 |
| Mid-cardinality GROUP BY wrong totals | In flight (background fix agent) | v0.1.1 |
| `MIN/MAX/SUM(empty/all-NULL)` returns 0 not NULL | Backlog | v0.2.0 |
