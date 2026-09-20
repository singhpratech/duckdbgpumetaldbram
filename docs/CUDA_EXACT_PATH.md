# The CUDA backend and the v0.7 exact path — the interface contract, and the tests that prove it

**Status 2026-09-19: implemented.** Every method below exists on the CPU
reference, on Metal and on CUDA (#152, #153, #154): exact `GROUP BY`, the
`WHERE` mask, the global masked aggregate and the materialised join. On the
RTX 4090 Laptop the unit suite is 730 / 730, the SQL suite 224 / 0 with the
path on, and `scripts/tpch_coverage.py` answers 17 of 22 TPC-H queries at SF1
on the device with 0 rows differing from native — the same coverage and the
same five declines as Metal at that scale factor (`BENCHMARK.md`, *the
transparent path on CUDA*). SF10 on CUDA is not recorded.

It is **opt-in in v0.7**: `GPUDB_CUDA_EXACT=1` makes the CUDA backend report
`exact_supported()`, and with it `global_supported()` and `join_supported()`,
true. The default is off: `scripts/transparent_gate.py` has not been run on that
machine, so the thresholds a CUDA build decides with are Metal's, and rule 1 is
a measurement rather than an assumption. Note that a
column is single-homed: a set resident on the GPU cannot fall back to the CPU
reference for an operator the GPU lacks.

This page stays the interface contract — what each method must do, and what
proves it. Read it as the specification the three backends are held to, not as
a to-do list.

Ownership per the contributor instructions at the repository root: `src/backends/cuda/**` and `src/include/cuda/**` are the
Linux instance's; the shared header `src/include/gpu_backend.hpp` changes only by PR.

## 1. Methods (all in `gpudb::Aggregator`, `src/include/gpu_backend.hpp`)

| method | contract (see the header's comment block) | reference | Metal notes |
|---|---|---|---|
| `upload_pair_exact(spans, n, vdt)` | key + payload with NULLs kept: rows in input order, a NULL key or payload is a zero bit in that column's validity bitmap (stage A of `docs/RESIDENT_COLUMNS_DESIGN.md`) | `cpu_aggregator.cpp` | `metal_aggregator.mm` |
| `upload_rows_exact(spans, n, dtypes, L)` | the general form: lane 0 key, lane 1 payload, lanes 2.. predicate columns (I64 or F64 in any lane — a store upload orders them rowid, ints, doubles, strings), one validity bitmap per lane, same row layout for every lane. Spans arrive in any number (an upload session hands over 8 MiB segments): copy them in parallel, the output layout must be identical to a serial copy. A span may carry `dst_row` (where its rows land; `kNext` = after the previous span) and `valid_bit` (the bit offset of its first row in the span's validity bitmaps) — the store places scan chunks at their row-id rank this way (stage B) | same | per-span parallel copy with prefix-sum offsets and a NULL-free fast path (2026-09-17) |
| `ResidentColumn::prepare()` / `prepared()` / `resident_bytes()` | build the derived structures now (sort cache), report memory INCLUDING them — the wrapper's budget reads `gpu_residents().bytes` | nothing to derive; only `resident_bytes()` is overridden | the valid rows compacted through the key's validity bitmap, then radix-sorted into a PACKED cache: the keys at the key's storage width (§6) and the permutation as u32 row ids, packed by a device kernel. The sorter's output buffers are handed to the column instead of copied, and the sort staging is released after large sorts |
| `groupby_exact_resident(keys, vals, cap, filter)` | one row per distinct key ascending, NULL-key group last; exact 128-bit sums (`sums`/`sums_hi`), counts, count(*), min, max; `filter` = device HAVING / top-k | same | sort-based: run starts over the sorted prefix |
| `groupby_exact_masked_resident(keys, vals, preds, n_preds, cap, filter)` | the same under a WHERE program: conjunction of `Predicate`s over predicate lanes (EQ NE LT LE GT GE IsNull IsNotNull In) | same | mask → selection → reduce |
| `groupby_exact_masked_multi(keys, pays, n_pays, filter_payload, preds, n_preds, cap, filter)` | several payloads in ONE pass; `MultiPayload::columns` bits select what to produce per payload | the base-class default in `src/backends/backend_factory.cpp` — one `groupby_exact_masked_resident` pass per payload, merged and cross-checked — so this method comes for free on CUDA as soon as the masked single-payload op works; fusing it is a later optimisation | fused: mask / selection / run starts once, reduce per payload |
| `aggregate_exact_masked(pays, n_pays, preds, n_preds)` | **optional, added 2026-09-18 (§4.12)**: aggregates without GROUP BY under a WHERE — one fused pass over the rows in storage order, no key, no sort cache, no permutation. Returns `GlobalAggResult`: per payload the exact 128-bit sum, `count(payload)`, min and max, plus one `count_star` of the surviving rows. Semantics are one group of `groupby_exact_masked_multi`; `n_pays` may be 0 (`count(*)` only) and `n_preds` may be 0. `global_supported()` is its rule-1 gate, defaulting to false — the CUDA backend builds and answers unchanged until it overrides both | `cpu_aggregator.cpp` (the reference) | `metal_aggregator.mm` + `gagg_masked_i64` in `kernels/sum.metal`: one lane table bound per distinct lane (payloads and predicate columns share slots, 12 of them), per-thread accumulators indexed `[group * n_pays + payload]`, threadgroup tree reduce, host merge |
| the direct grouped reduce | **backend-private, optional, added 2026-09-18; built on Metal only** (`docs/RESIDENT_COLUMNS_DESIGN.md` §7): a key with few distinct values gets a dense group-id lane derived from its sort cache (`gid[row]` = the rank of the row's key among the distinct valid keys ascending, a NULL key taking the reserved last id) and the exact operators answer with ONE row-order pass over it — the whole WHERE per row, every payload folded into `acc[gid]`, no mask buffer, no run starts, no permutation, no gather. **Nothing in `gpu_backend.hpp` changes**: same methods, same results, same order, same NULL-key group last, same absent-when-emptied groups, HAVING / top-k on the host over the group rows through `apply_group_filter_host`. The tests are the parity block in `test_aggregator.cpp` (every exact form, both algorithms, keys either side of the id-lane's limits) — CUDA gets them free. The one thing to report is which algorithm ran: `gpudb::exact_path_note()` (`src/include/exact_path_note.hpp`), printed by `gpu_last_stats()` as `path=…` | not implemented (the reference is always the sort path) | `metal_aggregator.mm` + `gdir_*` in `kernels/sum.metal`. CUDA has 64-bit atomics and `atomicCAS`, which Apple GPUs do not, so the accumulator design does NOT transfer: Metal replicates a slab of 32-bit atomics per threadgroup and carries the 128-bit sums by hand. A CUDA port should use shared-memory 64-bit atomics (or `cub::BlockReduce` per group) and will have a different crossover — measure it, do not copy the numbers. The admission rule Metal needs (`groups >= 3` and `rows x (payloads + WHERE terms) >= 6M`) exists because a threadgroup clears and folds its accumulator slab whatever the rows: sweep ROWS as well as groups, or the fixed cost hides at the top of the range. And gate the path on the DEVICE before asking it to compile anything: a kernel that will not build is a fallback, not an error — Metal marks the path unavailable for the aggregator's lifetime, keeps the compiler's text and answers through the sort path — but a fallback after the fact is not always enough. On a virtualised Apple GPU ("Apple Paravirtual device", a hosted macOS runner) ONE refused pipeline build left that process's Metal compiler unusable: kernels that had built moments before then failed too. So a backend must not ask an unsuitable device to compile an optional kernel at all; it decides from the device's identity and capabilities first, once, before any state the path would leave behind exists |
| `narrow_lanes()` | **optional**: does this backend store an exact I64 lane at the narrowest signed width its values fit (`docs/RESIDENT_COLUMNS_DESIGN.md` §6)? Storage width stays backend-private; the flag exists because the wrapper's PRE-upload memory estimate sizes lanes from their DuckDB type when it is true and charges 8 bytes a lane when it is false. Defaults to false, which is correct for a CUDA backend that stores 8-byte lanes | false | true |
| `join_materialize(probe_key, build_key, lanes, n)` | inner join onto a UNIQUE build key; returns row-aligned lanes (probe or build side), NULL-key rows dropped; `join_supported()` gates it | `cpu_aggregator.cpp` | sort-merge (Metal has no 64-bit atomic CAS); CUDA should use open-addressing with `atomicCAS` as `join_kernel.cu` already does for the v0.6 join |
| `device_memory_bytes()` | total device memory in bytes (`cudaMemGetInfo` total); reported by `gpu_build_info()`, sets the wrapper's default budget to half of it | 0 | `recommendedMaxWorkingSetSize` |
| `exact_supported()`, `join_supported()` | return true only when the above are complete — these are rule-1 gates: a backend that claims support and throws makes statements fall back at run time | — | — |

### 1.1 The WHERE stage: one fused pass (mirror this)

`groupby_exact_masked_resident` reaches the device as a mask stage, and the
shape of that stage is worth copying rather than rediscovering. Metal's first
version was one kernel per term over the rows, each reading a byte of the mask
and writing one back; at SF10 it cost 2.9 ms plus **3.3 ms per term**, the same
whatever the lane's width and whatever survived, because the pass is bound by
the byte it reads and the byte it writes, not by the comparison. TPC-H Q12's
five terms were 19.7 ms of a 23.3 ms kernel.

A CUDA port should write the mask in ONE pass that evaluates the whole
conjunction per row — the predicate program (lane, op, value, IN-list slice)
in constant or global memory, the lane table (pointer, validity bitmap, width,
is-f64) bound once, an early exit on the first failing term. On Metal that is
`gpred_eval` in `kernels/sum.metal`, one function called by the mask stage, by
the global aggregate (§4.12) and by the direct reduce; writing it once is what
made the third caller free. The second half is the same idea one level down:
the sort path reads the mask through the permutation, so gather `mask[perm[i]]`
ONCE, in the pass that counts the survivors of the key range, and keep the
result as a mask in sorted order that every payload's reduce then reads
sequentially — otherwise each payload re-gathers it (+1.4 to +2.6 ms per
payload at SF10).

Two constraints came with it. The number of lanes a kernel can bind is finite
(Metal: 12 distinct columns, payloads and predicate columns sharing slots), and
a WHERE over more than that must fall back to a per-term pass rather than
throw. And a pipeline that will not build is a fallback, not an error — see the
direct reduce's row below for why a backend asks the device what it is before
it asks it to compile anything. Evaluating the predicates AT `perm[i]` instead,
so no row-order mask exists at all, was measured and is slower: it trades one
sequential read per lane for one random gather per lane, and a lane gather
costs what a payload gather costs.

Not needed: anything in `GroupByAggregator`, `WindowAggregator`, `HashJoinProbe`
(v0.6 operators, unchanged). Also NOT part of this port: `topk_resident` and
the three resident joins (`join_sum_resident_i64` / `_f64`,
`join_rows_resident`) — the CUDA backend already implements all four, and the
extension calls them from many places; they keep working while the exact path
is being added.

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
- **The global aggregate returns one result, always** — over an empty column,
  over a mask that keeps nothing, over a payload whose every cell is NULL.
  `count_star` is then 0, every `counts[p]` is 0 and the sums / mins / maxs of
  those payloads are unspecified (the extension emits SQL NULL). Returning
  "no result" would be a different answer from native's.
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
  with the CPU reference fed one span). At the time of writing (2026-09-18) the
  binary reports 2397 checks on Metal, all of them passing; the CUDA count will
  differ only by the SKIP lines that become real checks.
- `test/cpp/test_aggregator.cpp`, block `global masked aggregate`: six payloads
  with NULL cells, every predicate op including `In` and `IsNull` / `IsNotNull`,
  an F64 predicate lane, lanes sitting on each narrow-width boundary, a mask
  that keeps nothing, an empty column, a 128-bit sum that leaves 64 bits, and
  2.6M rows so the parallel reduction runs many threadgroups — every case
  compared limb for limb against the CPU reference.
- `./scripts/run_sql_tests.sh` — the whole suite is 224 passing cases and 45
  expected fails, on Metal and on CUDA with the exact path on (2026-09-19). The exact path's
  own files: `test/sql/gpu_agg_exact_global.test` (the
  table function, including its guardrails), `test/sql/gpu_groupby_exact*.test`,
  `gpu_join_materialize.test`, `gpu_groupby_exact_multi.test`, `gpu_rewrite.test`
  (33 cases: the C++ rewriter's output, backend-independent).
- `python/tests/test_wrapper.py` — 1157 checks through `gpudb.connect()`: parity
  against native DuckDB for every shape, staleness, background residency, the
  memory budget, error fallback. Needs the extension built with
  `third_party/duckdb-libs/` present (`./scripts/get_duckdb_libs.sh`).
- `scripts/tpch_coverage.py` — the 22 TPC-H queries. SF1 gives 17 of 22 on the
  device with identical rows on both backends (on CUDA with
  `GPUDB_CUDA_EXACT=1`); on Metal, `--db data/tpch_sf10/tpch.duckdb
  --memory-budget 200GB` gives 19 of 22. SF10 on CUDA is not recorded.

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

## 6. Storage width is backend-private (stage C)

`src/include/gpu_backend.hpp` does not change for stage C of
`docs/RESIDENT_COLUMNS_DESIGN.md`: `dtype()` keeps returning `I64`, `rows()` and
`null_count()` keep their meanings, and every operator signature is what it was.
A backend is free to store an I64 lane narrower than 8 bytes and widen on load,
and `resident_bytes()` is what tells the budget it did.

Metal does it this way, and CUDA should mirror it:

- the width is chosen in the upload, from the lane's min and max over its valid
  cells (NULL cells hold 0, which fits any width) — the per-span copy already
  reads every value, so each span reduces its own partials and the width follows
  from the merged range, boundaries inclusive: 1 byte for `[-128, 127]`, 2 for
  `[-32768, 32767]`, 4 for `[-2147483648, 2147483647]`, else 8; a lane with no
  valid cell takes 1;
- a second parallel pass packs the staging into the chosen width — on CUDA that
  pass belongs in the de-interleave or on the device after the copy;
- F64 lanes stay 8, and so do string-hash lanes (the min/max rule lands them
  there by itself);
- the sort cache keeps the key's width and holds u32 row ids (Metal already
  refuses a column above 2^32 rows; CUDA should decide its own cap);
- a join's output lanes keep their source lanes' widths — a gather cannot widen
  a lane's range;
- every kernel that reads a lane, a sorted key or a permutation entry takes the
  width and loads through one helper. On Metal that is a uniform branch and it
  cost nothing measurable; on CUDA the same choice is a template parameter, and
  the reason to reach for one is a measurement, not a guess.

The proof is the same as for everything else here: the CPU reference stores I64,
`test/cpp/test_aggregator.cpp`'s "narrow lane storage" block compares a backend
against it bit for bit over lanes that sit exactly on the I8 / I16 / I32
boundaries and one past each, an all-NULL lane and a 64-bit hash lane, and reads
the widths back through `resident_bytes()`.

## 7. Shedding a derived structure is backend-private too (§9)

A backend that keeps a group-id lane (`docs/RESIDENT_COLUMNS_DESIGN.md` §7) may
release the structures that lane makes unnecessary — the key's sort cache, and,
for a lane the extension marks as a key and nothing else, the key lane itself,
since `key[row] = dkeys[gid[row]]` rebuilds it. `gpu_backend.hpp` says nothing
about any of this; what crosses the boundary is one note each way in
`src/include/resident_shed_note.hpp`: the extension sets a bit mask of key-only
lanes immediately before the upload call that creates them, and the backend
counts the rebuilds it was forced into. CUDA is free to implement none of it —
an unmarked lane and an empty counter are exactly today's behaviour — and if it
does implement it, the rules it must keep are: never shed where the direct path
is unavailable or where the dispatch rule would send this column's calls to the
sort path at this row count; rebuild on demand behind the single accessor that
returns the lane; pin what was rebuilt so a column pays at most one rebuild per
structure; and let `resident_bytes()` follow, since that is what the budget
reads. The parity proof is the ordinary one: `test/cpp/test_aggregator.cpp`'s
shedding block runs every exact form over a shed column against the CPU
reference, forces the sort path back onto it and compares again.

## Stage D1 — lanes read through an index (2026-09-19)

`gpu_backend.hpp` gained five additive things and changed nothing that exists.
CUDA builds and runs unchanged without implementing any of them: the defaults
refuse an index rather than ignore one, and `indexed_supported()` returns false,
which is the gate the SQL layer reads before it plans an indexed set.

What the CUDA side owes, when it comes to it:

- **`IndexedColumn`** — a lane plus the index vector it is read through. An
  index is an ordinary `Dtype::I64` resident column with one cell per row of the
  statement; the cell is the row POSITION in the lane's column, not a DuckDB
  rowid. `nullptr` is the identity and must cost exactly what a direct read
  costs today. A NULL index cell makes every lane read through it NULL — that is
  the unmatched side of an outer join, and it is the whole NULL story.
- **The `index` fields** on `Predicate`, `MultiPayload` and `JoinLane`, honoured
  in the mask kernels and the reduce kernels. The statement's row count is the
  index's when a lane has one, and the column's own row count is then only the
  bound its cells must respect. That bound is the implementor's to prove: the
  kernels use a cell as a row with no check, so an out-of-range cell is a
  planner bug that must be caught before a kernel sees it. Metal proves it on
  the host from the index column's min and max over its valid cells, computed
  once and cached on the column; the CPU reference checks every cell.
- **`join_index`** — the same probe, uniqueness check and class-1/class-2 split
  as `join_materialize`, stopping one kernel earlier: instead of one gather per
  output lane it returns `probe_rows` and `build_rows`, plus whatever lanes the
  caller still asked to materialise. `rows_out`, `null_key_rows` and the order
  within each class must agree with `join_materialize` exactly. `probe_index`
  is for a chained join, whose probe key is itself read through the previous
  step's vector; composing the earlier vectors needs no new operator, because
  gathering the old index at the new probe rows is an ordinary materialised
  lane.
- **`groupby_exact_masked_multi_indexed`** — exists only for an indexed KEY;
  payload and predicate indexes travel on their own structs. With every index
  null it must be `groupby_exact_masked_multi`, call for call.
- **`indexed_supported()`** — true only once the above are real on the device.

A note on where the index vectors are bound, because it cost a design revision
on Metal: they are bound as ordinary lanes, sharing the per-statement lane
budget, because the direct reduce had already used every argument-table slot the
API offers. CUDA has no such limit and can bind them separately; the interface
does not care either way.

The parity proof is `test/cpp/test_aggregator.cpp`'s stage-D1 block. It runs on
every backend that reports `indexed_supported()`, compares `join_index` plus
indexed reads against `join_materialize` plus direct reads lane for lane where
`join_index` exists, and otherwise holds the backend to the indexed read alone
against host-built index vectors — the width boundaries, an F64 lane with NaN /
±inf / -0.0, NULL cells in the lane, NULL cells in the index, and a refusal on
an out-of-range cell. One allowance the block makes: a backend may refuse a KEY
whose dtype it does not group on (Metal refuses an F64 key), and the block then
requires BOTH forms to refuse — a refusal is an answer only if the index did not
change it.

Two findings from building it on Metal that a CUDA implementor should have
before starting, because they are about shape rather than about Metal:

- **Do not put the index test inside the row loop.** Reading a lane through a
  runtime test cost the masked kernels 8–15% at SF10 with no statement indexed,
  because a row returned from a function is not the loop's induction variable
  and the address arithmetic for `lane[row]` and its validity word stops
  strength-reducing. Metal templates the read and every kernel that uses it on
  a compile-time flag, builds both, and picks one per dispatch from a value
  uniform over the grid. CUDA has the same property and the same fix available.
- **The indexed read is slower than the gathered lane, everywhere measured.**
  On an M4 Max, 16M probe rows against a 200k dimension: 1.16–1.24× slower at
  4–64 groups and 1.51–1.92× at 1000, growing with groups and payloads
  (docs/RESIDENT_COLUMNS_DESIGN.md §8.2 has the table). What the index buys is
  the residency build — the join that returns row vectors is 1.3–1.4× faster
  than the one that gathers lanes — and the bytes. So rule 1 for stage D is a
  memory rule, not a speed rule, and a backend that implements the mechanism
  has not thereby made anything faster.
