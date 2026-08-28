# Resident GROUP BY / ORDER BY from SQL — design (v0.6)

The v0.5 join stack left one obvious hole in the SQL surface: the GPU could
join and reduce, but a `GROUP BY` still had to go through native DuckDB.
v0.6 closes it with **row-returning table functions** over resident columns,
so the GPU produces the `(key, aggregate)` rows and everything downstream —
`WHERE`, `ORDER BY`, `LIMIT`, further joins — is ordinary SQL over a small
result:

```sql
SELECT gpu_upload_pair('l', l_orderkey, l_quantity::BIGINT) FROM lineitem;   -- once
SELECT key, sum FROM gpu_groupby_sum_resident('l') WHERE sum > 300;           -- TPC-H Q18's inner query
SELECT idx, value FROM gpu_topk_resident('l', 10, 'desc');                    -- ORDER BY v DESC LIMIT 10
```

This is deliberately the same shape as `gpu_join_rows_resident`: upload once,
query many times, no output materialised beyond the groups themselves.

## SQL surface

| Function | Returns | Notes |
|---|---|---|
| `gpu_groupby_sum_resident(name)` | `(key BIGINT, sum BIGINT, count BIGINT)` | over `gpu_upload_pair` `name.k` / `name.v` (BIGINT payload) |
| `gpu_groupby_sum_resident_f64(name)` | `(key BIGINT, sum DOUBLE, count BIGINT)` | DOUBLE payload |
| `gpu_groupby_count_resident(name)` | `(key BIGINT, count BIGINT)` | `name` may be a pair (uses `name.k`) or a bare `gpu_upload` column |
| `gpu_topk_resident(name, k, 'asc'\|'desc')` | `(idx BIGINT, value BIGINT)` | ranks the payload of a pair, or a bare BIGINT column; `idx` = upload-order index |
| `gpu_topk_resident_f64(name, k, 'asc'\|'desc')` | `(idx BIGINT, value DOUBLE)` | same over a DOUBLE payload / column |

Rows come out **sorted by key ascending** on every backend; top-k rows come
out in the requested order. `gpu_last_stats()` reports the backend, dispatch
reason, `rows_in`, `groups`, and the wall / kernel / transfer split for the
last call, as for the other resident ops. Group count is capped at `GPUDB_GROUPBY_ROWS_MAX_M`
million (default 100) with a clean error naming the actual count — checked
before any device→host copy, never truncated.

## Backend contract (`gpu_backend.hpp`)

```cpp
GroupByResidentResult groupby_sum_resident_i64(keys, vals, max_groups);  // keys, sums, counts
GroupByResidentResult groupby_sum_resident_f64(keys, vals, max_groups);  // keys, sums_f64, counts
GroupByResidentResult groupby_count_resident(keys, max_groups);          // keys, counts
TopKResult            topk_resident(col, k, descending);                 // idx + values
```

- i64 sums accumulate in uint64 two's-complement (defined wrap; bit-exact
  across CPU / CUDA / Metal). f64 sums are backend-ordered and compared
  under the 1e-9 relative tolerance contract. Counts are exact.
- Top-k tie order among equal values is unspecified (SQL `ORDER BY` without
  a tiebreaker); cross-backend checks compare the multiset of values, not
  `idx`. f64 ordering is total with NaN greatest (native DuckDB order).
- The ops live on `Aggregator` (like the join ops) because a
  `ResidentColumn` is bound to the aggregator that uploaded it. Base
  implementations throw "not implemented"; the hybrid planner forwards to
  the GPU with `Hot_GpuAlwaysWins` when the columns are resident there.
- The CPU implementation is the executable reference (sort pairs, run-length
  reduce); `test_backend()` compares every backend against a host map and
  treats a "not implemented" throw as SKIP.

## Algorithms

All three backends share one idea: **a sorted key column is a grouped key
column.** The v0.5 join already keeps a radix-sorted copy of a key column
plus its permutation (original upload indices) cached on the column. GROUP BY
reuses that cache, so the second query on the same pair pays no sort, and a
column used as a join build side is already grouped for free.

### Metal (Apple Silicon, UMA)

No 64-bit atomics needed, no hash table, output already ordered:

1. `ensure_sorted_cache` — radix sort `(key, index)` once (cached).
2. `gb_block_counts_i64` — per 256-element block, count run starts
   (`keys[i] != keys[i-1]`). Host exclusive-scans the ~n/256 block counts
   (trivial) → block offsets and the total group count. **Cap checked here.**
3. `gb_run_starts_i64` — each run start writes its sorted position to
   `starts[block_offset + in-block prefix]` (simd prefix sum) → `starts[]`
   ordered by key.
4. `gb_chunk_sum_i64` — one thread per 64 sorted positions sums
   `vals[perm[i]]` per run. Runs fully inside the chunk are written straight
   to the output by their exclusive owner; a run that crosses the chunk's
   boundary leaves a head (started before the chunk) or tail (started in it,
   continues past) partial. No atomics.
5. `gb_finalize_i64` — one thread per group: key and count from `starts`;
   for boundary-crossing groups `sum = tail[c0] + Σ head[c0+1..c1]` over the
   chunks it spans. Total work O(n/64 + groups).

DOUBLE payloads: no fp64 in MSL, so `gb_gather_i64` lays the payload out in
sorted order as raw 64-bit words and the host streams one sequential
per-segment sum (parallel over segments) — the "GPU sorts, host streams"
pattern from the f64 join. Output vectors are page-aligned so the GPU writes
the result in place (`newBufferWithBytesNoCopy`), no copy-back.

Top-k = the first/last `k` entries of the cached sort. For DOUBLE columns
the sort runs on an order-preserving integer transform (NaN canonicalised
and greatest), cached on the column like the i64 sort.

### CUDA (discrete GPU)

Same cached sorted-build + permutation; values gathered through the
permutation; `cub::DeviceReduce::ReduceByKey` with a `(sum, count)` pair in
one pass (`cub::DeviceRunLengthEncode` for count-only); group count from a
run-count pass first so the cap is checked before any output allocation.
The device→host copy of the `(key, sum, count)` arrays dominates wall time
at Q18 sizes (75M groups at SF50 ≈ 1.8 GB over PCIe); `kernel_ms` vs
`wall_ms` in `gpu_last_stats()` shows the split, as with the row join.

### CPU (reference)

`std::sort` on `(key, value)` pairs, run-length reduce; `partial_sort` for
top-k. Not a performance path.

## Where it wins and where it doesn't

- **High-cardinality GROUP BY** (Q18's `GROUP BY l_orderkey`: 15M groups at
  SF10, 75M at SF50) is the target: native DuckDB's hash aggregate thrashes
  caches; the GPU sort is bandwidth-bound and already cached.
- **Low-cardinality GROUP BY** (Q1's 4 groups) will not beat native's
  streaming aggregate on any backend — the sort is wasted work — and we say
  so in BENCHMARK.md rather than hide the row.
- On discrete GPUs the output copy is the ceiling; on unified memory it is
  free. Same honest split as the row-returning join.

## Statement sequencing

Uploads and reads are separate statements on one connection (`gpudb-sql
--multi`, or any client running statements sequentially); the result types
are fixed per function so binding never depends on a column that a
same-statement upload has not produced yet. A one-statement form (upload
aggregate on the right-hand side of a cross product with the table
function) happens to work for a single reference but is not guaranteed by
DuckDB's pipeline ordering — the SQL suite uses `-- setup:` statements
(runner support added for this release: `gpudb-sql --multi-last`).

## Tests

- `test/cpp/test_aggregator.cpp` — every backend vs a host map: dup-heavy
  keys, int64 boundary keys, wrapping sums, f64 tolerance, cap error text,
  top-k both directions, k clamping.
- `test/sql/gpu_groupby_resident.test` — SQL surface, composition with
  WHERE/ORDER BY/LIMIT, same-statement parity with native, guardrails.
- `scripts/groupby_parity_check.sh` — 10 adversarial scenarios × 5 checks
  against native in the same process (dup-heavy, all-unique, single group,
  Knuth-hash, Zipf skew, negatives, int64 boundaries, tiny, and runs placed
  exactly on the 64-chunk / 256-block boundaries the Metal kernels use).

## Deferred (v0.7)

Composite keys (pack into one BIGINT for now), `GROUP BY` over join results
as a fused op, resident f64 min/max, and transparent operator substitution
via a DuckDB optimizer extension so plain `GROUP BY` SQL routes here without
calling `gpu_*` functions.
