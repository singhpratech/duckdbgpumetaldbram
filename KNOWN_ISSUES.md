# Known issues

Status as of 2026-05-09 night, post-`v0.1.0` and post-PR #22.

If you are bitten by something here, fall back to native DuckDB `sum()` / `min()` / `max()` / window functions for that query — the hybrid planner inside the extension already does this automatically wherever it can.

---

## Open issues

None blocking `v0.1.0`. Window functions and GROUP BY at all tested cardinalities pass on both backends.

---

## Documented intentional divergences from SQL spec

These are **not** bugs — explicit design choices for v0.1.0. Will be revisited in v0.2.0.

| Behavior | Native DuckDB | Our extension |
|---|---|---|
| `MIN/MAX/SUM(empty input)` | NULL | 0 |
| `MIN/MAX/SUM(all NULLs)` | NULL | 0 |

Returning `0` is the C++ aggregator's natural behavior; properly returning SQL `NULL` requires using `duckdb_validity_set_row_invalid` on the output vector. SQL queries that test for `IS NULL` on a fully-empty/all-NULL aggregate result currently get `false`.

---

## Resolved issues (fixed in `v0.1.0`)

| Issue | Fixed in |
|---|---|
| Window+PARTITION BY non-determinism | PR #18 (combine() copy-not-move) + PR #20 |
| Window OVER (ORDER BY) running-sum chunk-boundary state loss | PR #18 |
| `gpu_sum(v) OVER ()` unbounded-frame SEGFAULT | PR #22 (POD state + global BufferPool, magic-word probe) |
| Mid-cardinality (50-63 groups) GROUP BY wrong totals | PR #21 (CPU per-state in mid-card regime) |
| metal_aggregator.mm duplicate decls (build broken on macOS) | PR #20 |
| `scripts/run_sql_tests.sh` couldn't run on macOS (missing `timeout`) | PR #20 |

### Why PR #22 needed a process-global BufferPool

DuckDB's `RadixPartitionedTupleData::Repartition()` raw-`memcpy`s aggregate state bytes between partitions without calling `state_init()` or any move/copy callback. Our previous layout — `struct GpuSumState { std::vector<int64_t> buf; }` — had its `buf.data()` pointer duplicated across partitions: both copies pointed at the same heap allocation, and when one was destroyed the other was left dangling. Plus `WindowConstantAggregator` (used for `OVER ()`) passes a `CONSTANT_VECTOR` state pointer that the C API doesn't flatten; reading `states[1..N-1]` is OOB → segfault.

The fix collapses state to a 16-byte POD `{uint64_t magic; uint64_t buf_id;}` indexing into a process-global mutex-guarded `unordered_map<uint64_t, vector<int64_t>>`. The magic word lets `update()` reject CONSTANT_VECTOR OOB reads. State bytes can now be freely copied; the actual data lives outside DuckDB's view.

---

## How to verify

```bash
git clone https://github.com/singhpratech/duckdbgpumetaldbram
cd duckdbgpumetaldbram
./scripts/get_duckdb_libs.sh
./scripts/build.sh
./scripts/local_check.sh        # builds + 77 unit tests + smoke benchmarks
./scripts/run_sql_tests.sh      # 5 .test files, 46 queries, 0 fail / 0 xfail
```
