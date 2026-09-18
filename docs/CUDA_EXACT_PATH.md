# The CUDA backend and the v0.7 exact path — what to implement, and the tests that prove it

Status 2026-09-17. Everything the transparent path needs from a backend exists on
the CPU reference and on Metal; the CUDA backend still runs the v0.6 operators only
(`exact_supported() == false`, `join_supported() == false`). Until it opts in, the
Python wrapper leaves every statement on DuckDB on a CUDA machine — correct, no
speed-up. This page is the checklist for the Linux instance: the interface is
frozen, the semantics are pinned by tests that already run for every compiled
backend, and the measurements that decide the thresholds have a script.

Ownership per the contributor instructions at the repository root: `src/backends/cuda/**` and `src/include/cuda/**` are the
Linux instance's; the shared header `src/include/gpu_backend.hpp` changes only by PR.

## 1. Methods (all in `gpudb::Aggregator`, `src/include/gpu_backend.hpp`)

| method | contract (see the header's comment block) | reference | Metal notes |
|---|---|---|---|
| `upload_pair_exact(spans, n, vdt)` | key + payload with NULLs kept: rows in input order, a NULL key or payload is a zero bit in that column's validity bitmap (stage A of `docs/RESIDENT_COLUMNS_DESIGN.md`) | `cpu_aggregator.cpp` | `metal_aggregator.mm` |
| `upload_rows_exact(spans, n, dtypes, L)` | the general form: lane 0 key, lane 1 payload, lanes 2.. predicate columns (I64 or F64 in any lane — a store upload orders them rowid, ints, doubles, strings), one validity bitmap per lane, same row layout for every lane. Spans arrive in any number (an upload session hands over 8 MiB segments): copy them in parallel, the output layout must be identical to a serial copy. A span may carry `dst_row` (where its rows land; `kNext` = after the previous span) and `valid_bit` (the bit offset of its first row in the span's validity bitmaps) — the store places scan chunks at their row-id rank this way (stage B) | same | per-span parallel copy with prefix-sum offsets and a NULL-free fast path (2026-09-17) |
| `ResidentColumn::prepare()` / `prepared()` / `resident_bytes()` | build the derived structures now (sort cache), report memory INCLUDING them — the wrapper's budget reads `gpu_residents().bytes` | trivial | radix sort with the sorter's output buffers handed to the column, staging released after large sorts |
| `groupby_exact_resident(keys, vals, cap, filter)` | one row per distinct key ascending, NULL-key group last; exact 128-bit sums (`sums`/`sums_hi`), counts, count(*), min, max; `filter` = device HAVING / top-k | same | sort-based: run starts over the sorted prefix |
| `groupby_exact_masked_resident(keys, vals, preds, n_preds, cap, filter)` | the same under a WHERE program: conjunction of `Predicate`s over predicate lanes (EQ NE LT LE GT GE IsNull IsNotNull In) | same | mask → selection → reduce |
| `groupby_exact_masked_multi(keys, pays, n_pays, filter_payload, preds, n_preds, cap, filter)` | several payloads in ONE pass; `MultiPayload::columns` bits select what to produce per payload | default = one pass per payload (fine to start with) | fused: mask / selection / run starts once, reduce per payload |
| `join_materialize(probe_key, build_key, lanes, n)` | inner join onto a UNIQUE build key; returns row-aligned lanes (probe or build side), NULL-key rows dropped; `join_supported()` gates it | `cpu_aggregator.cpp` | sort-merge (Metal has no 64-bit atomic CAS); CUDA should use open-addressing with `atomicCAS` as `join_kernel.cu` already does for the v0.6 join |
| `device_memory_bytes()` | total device memory in bytes (`cudaMemGetInfo` total); reported by `gpu_build_info()`, sets the wrapper's default budget to half of it | 0 | `recommendedMaxWorkingSetSize` |
| `exact_supported()`, `join_supported()` | return true only when the above are complete — these are rule-1 gates: a backend that claims support and throws makes statements fall back at run time | — | — |

Not needed: anything in `GroupByAggregator`, `WindowAggregator`, `HashJoinProbe`
(v0.6 operators, unchanged).

## 2. Semantics that are easy to get wrong (each has a test)

- **NULL keys** form one group, sorted last; they sit anywhere in the column
  under its validity bitmap, and the sort cache covers the valid rows only
  (`sort_rows()`), its permutation holding row ids.
- **NULL payloads** are skipped by sum/count/min/max but counted by `count(*)`;
  a group whose payloads are all NULL has `counts == 0` and unspecified
  sum/min/max (the extension emits SQL NULL).
- **Sums are 128-bit** (`sums` low, `sums_hi` high), bit-identical across
  backends; DuckDB's `sum(BIGINT)` is HUGEINT.
- **Predicates on F64 lanes** compare doubles; on I64 lanes integers; `In`
  takes a list. NULL never satisfies a predicate except `IsNull`.
- **HAVING / top-k on the device** (`GroupByFilter`): `cmp` + `threshold` on
  `agg` (Sum / Count / CountStar / Min / Max), or `topk` with `topk_desc`;
  `columns` bits select the output columns.
- **Order of the output**: ascending by key; the wrapper never sorts on the host.
- **Big inputs**: 300M rows per set is real (TPC-H SF50); use 64-bit sizes;
  the Metal sort cache is limited to 2^32 rows and says so.

## 3. Tests that prove it (run for EVERY compiled backend, no per-test filter)

- `build-linux/test/test_gpudb` — `test/cpp/test_aggregator.cpp`: exact upload
  with NULL keys / payloads, WHERE programs, HAVING / top-k, several payloads,
  the join (unique keys, misses, NULL keys), the few-groups / giant-group block
  (~180K NULL-key rows), and `exact upload from many spans` (2.6M rows from nine
  uneven spans, NULL-free / 60% NULL keys / empty / stale bitmap bits, compared
  with the CPU reference fed one span). 804 checks on Metal; the CUDA count will
  differ only by the SKIP lines that become real checks.
- `./scripts/run_sql_tests.sh` — `test/sql/gpu_groupby_exact*.test`,
  `gpu_join_materialize.test`, `gpu_groupby_exact_multi.test`, `gpu_rewrite.test`
  (33 cases: the C++ rewriter's output, backend-independent).
- `python/tests/test_wrapper.py` — 808 checks through `gpudb.connect()`: parity
  against native DuckDB for every shape, staleness, background residency, the
  memory budget, error fallback. Needs the extension built with
  `third_party/duckdb-libs/` present (`./scripts/get_duckdb_libs.sh`).
- `scripts/tpch_coverage.py` — the 22 TPC-H queries, SF1: expect 15 of 22 on the
  device with identical rows (Metal); `--db data/tpch_sf10/tpch.duckdb` 16 of 22.

## 4. Rule 1 on CUDA: measure, then set the thresholds

The thresholds in `python/gpudb/_thresholds.py` carry a `METAL` table and
`CUDA = METAL` as a placeholder. They were derived from
`scripts/transparent_gate.py` on Apple silicon (unified memory, no PCIe): a
discrete GPU moves results over the bus, so the output-bound limits
(`plain_max_groups`, `reagg_max_pairs`, …) will differ. Procedure:

1. `PYTHONPATH=python python3 scripts/transparent_gate.py --no-thresholds --subqueries --exprs`
   (data collection: every shape rewritten, losing rows reported) at SF1 and SF10.
2. Read the break-even per form / selectivity / groups from the table; set the
   `CUDA` `Thresholds` accordingly (docstring in `_thresholds.py` records the
   Metal measurements as the model).
3. `scripts/transparent_gate.py --subqueries --exprs` must exit 0: nothing below
   1.0×, nothing differing. Then `scripts/tpch_coverage.py`.
4. `BENCHMARK.md` gets the CUDA tables next to the Metal ones; the README's
   comparison table at release is regenerated from both.

## 5. Upload path on a discrete GPU

The extension stages rows in host segments and hands `RowSpan`s to
`upload_rows_exact`. On Metal the copy into shared buffers is the whole
transfer; on CUDA it is a host-side de-interleave into pinned staging plus
`cudaMemcpyAsync` per lane — do the de-interleave in parallel (the Metal
version is the model) and overlap the copies on the column's own stream.
`prepare()` (sort cache: valid keys + permutation to row ids, compacted through
the validity bitmap first, `cub::DeviceRadixSort`) runs on that stream too; `prepared()` flips after the event completes. The memory
budget counts `resident_bytes()`, so include the sort cache and any scratch
that stays allocated.
