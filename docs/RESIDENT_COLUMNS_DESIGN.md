# Resident columns — the storage design

Status: designed 2026-09-18 and shipped in v0.7 — stage A merged (#123), stage B
merged (#124), stage C merged (#125), the group-id lane of §7 built 2026-09-18
(Metal). Stage D (a join result read through index vectors instead of copied)
is designed and measured in §8; the indexed read is not wired into SQL, and
the section says so where it says so. Companion to
`docs/TRANSPARENT_DESIGN.md` (the rewrite, the rules, the thresholds), which it
does not change: rule 1 (never slower than native) and rule 2 (never a
different answer) are enforced by the same gate and the same tests.

## 1. Why

In v0.7 every statement template uploads its own private *set*: a key lane, the
payload lanes, the predicate lanes and a sort cache, in a row order specific to
that set (rows with a NULL key are moved to a trailing block so the sort cache
covers a prefix). Measured on the 22 TPC-H queries at SF10: 24 sets, 102 lane
instances, 32.9 GiB of lanes plus sort caches — for a `lineitem` that is about
5 GB at its natural column widths. Two consequences:

- nothing is shared between statements, even when they read the same column
  (sharing was measured at 0.0 GiB of saving *within this design*, because the
  duplicates sit in join-result sets with their own row order);
- every write drops and rebuilds whole sets, and a table larger than the
  memory budget cannot be used at all.

The device is a cache of the table, so it should be organised like the table:
**one resident copy per column, in row-id order, at its natural width**, with
statements referring to columns rather than owning copies.

## 2. The design

```
TableStore  (catalog, table oid, epoch)            one per resident table
  ├── column "l_quantity"      rows 0..n-1, INT32 storage, validity bitmap
  ├── column "l_shipdate"      DATE as INT32
  ├── column "x_7ce4ca4c…"     a computed lane (DuckDB evaluated it), shared by every statement that uses it
  ├── column "h_c_name+…"      a hashed key tuple + its dictionary
  └── sort cache per column that is ever a GROUP BY key or a join build key
Statement plan = references into one TableStore (key column(s), payload columns,
predicate columns) — no data of its own.
Join result   = an index vector over the probe table's rows plus, per matched
                build row, the build row id — lanes are gathered through it.
Chunks        = row-id ranges; an INSERT is a new chunk; a table larger than the
                budget keeps hot chunks resident and streams the rest.
```

Invariants:
1. **Row order is table order** (rank of rowid among existing rows). Every
   column of a TableStore is row-aligned with every other; a NULL key is a bit
   in the key column's validity bitmap, never a moved row.
2. **A column belongs to the table, not to a statement.** Its identity is
   (catalog, oid, epoch, expression text); its memory is counted once.
3. **The sort cache belongs to the column** (sorted valid keys + permutation to
   row ids), shared by every statement grouping on it.
4. **Width is a storage detail.** The interface stays `I64` / `F64`; a backend
   may store a lane narrower when its values fit and widens on load.
5. **Everything above the backend interface stays as it is**: the rewrite, the
   thresholds, the guards, the gate.

## 3. Stages (each green on the gate and the suites before the next)

| stage | what | touches | done when |
|---|---|---|---|
| A | **Layout**: no NULL-key suffix; key columns carry a validity bitmap; the sort cache covers the valid keys and maps to row ids; rows stay in input order | Metal, CPU, the interface comments | all suites + gate unchanged; a layout unit test |
| B | **TableStore**: columns uploaded by (table, expression) in row-id order (`gpu_upload_columns`), single-table sets and join base sets become views over the store; the wrapper asks what is resident and uploads only the missing columns; budget and LRU per column | extension, wrapper | done — §5: the single-table and join-base lanes of the 22 TPC-H queries at SF10 are 30 shared columns; what is left uploaded is join results (stage D) |
| C | **Width**: I32 / I16 / I8 storage chosen from the upload's min/max; typed loads in the exact kernels | Metal | done — §6: the SF10 store is 4.91 GiB where it was 11.43, total resident 22.71 GiB against 44.91; no hot kernel lost time, so the runtime width stayed and no per-width PSO was needed |
| D | **Index-vector joins and chunks**: a device join returns row ids; INSERTs append chunks; cold chunks stream | extension, Metal, wrapper | appends resident in ms; a table above the budget answers on the device for the shapes that win |

CUDA implements the design once, after stage C, from `docs/CUDA_EXACT_PATH.md`
(revised for stage C: storage width is backend-private and the interface is
unchanged). §7 is a second derived structure beside the sort cache, also
backend-private, and does not belong to a stage: it can land before or after
stage D.

## 4. Stage A in detail

What the suffix layout bought was a sort cache over a prefix and a host fold
over a suffix for the NULL-key group. Both work over a bitmap:

- `build_sort_cache`: with a key bitmap, compact the valid (key, row id) pairs
  and sort those; `sort_rows()` = valid keys; the permutation holds row ids, so
  every gather stays as it is.
- the NULL-key group: fold the rows whose key bit is 0 (under the WHERE mask),
  in parallel over bitmap words, instead of a trailing range.
- the WHERE mask kernel already takes a bitmap per column; the "null from" cut
  is simply never set.
- `join_materialize`: the build side always uses the column's sort cache (it
  now covers valid keys whatever the layout); the probe side's NULL keys come
  from its bitmap; every output lane, the key included, carries its own bitmap.
- `upload_rows_exact` / `upload_pair_exact`: rows land in input order; the key
  lane gets a bitmap when it has NULLs. The parallel per-span copy keeps one
  cursor instead of two.

Nothing in the extension or the wrapper changes in stage A: `null_count()`
keeps its meaning (number of NULL rows), the operators' output contract is
unchanged, and the tests that compare every backend against the CPU reference
are the proof.

## 5. Stage B in detail — the store

`gpu_upload_columns(tag, rowid, ints BIGINT[], dbls DOUBLE[], strs VARCHAR[])`
uploads any number of lanes of one table in one scan. Lane 0 is DuckDB's
`rowid`; the tag names the lanes (`gpudb:v1:<catalog>:<schema>:<table>:<oid>:<lane,…>:store`)
in upload order (ints, doubles, strings), and each lane is a *store name*: a
column name, the virtual name of a computed lane (`x_<digest>`, the same text
every statement gets for the same expression), or `k#<field>` for a GROUP BY
key that is the tuple text of its columns (it is not the column, so it never
collides with a WHERE lane on the raw column). The columns land in the table's
`TableStore`, keyed by that name; a second upload of a name replaces it.

**Row-id order.** DuckDB scans in parallel, so chunks arrive in any order but
rows inside a scan chunk are ascending; `upload_columns_update` records one
chunk per (update call, segment) and `finish_upload_columns` sorts the chunk
records and hands each to the backend as a `RowSpan` with `dst_row` /
`valid_bit` — the rows are placed at their row-id rank without a sort of the
data. When a lane's expression is a correlated subquery DuckDB evaluates it
through a join and even that order is gone; then finish ranks every row by
its row id (a dense position table when the ids are dense, a sort when they
are not) and gathers the rows into fresh segments by destination segment in
parallel — the placed path, 2× the upload's host memory while it runs, the
same columns at the end.

**Views.** A statement's set is now a *view* over the store: `acquire()` of a
tag whose store holds every lane it names synthesises the `ResidentSet` —
key column (`k#name` first, then `name`), payload column (`-` = none: the key
stands in), predicate columns bucketed ints / doubles / strings — shares the
dictionaries, and prepares the key column's sort cache once for every view
on it. Views cost 0 bytes (`gpu_residents().kind = 'view'`); the memory is
the store's (`gpu_store_columns()`), and the budget counts, ages and evicts
*columns* (`gpu_drop_column`) — a view whose lane went is simply
re-synthesised after the wrapper uploads the lane again. Invalidating a table
drops its store with its sets; a store upload that finds another row count
replaces the store and marks every view on it stale.

**The wrapper** (`_rewrite.store_lanes` / `store_upload_sql`, `_residency`,
`connection._store_upload`) asks `gpu_store_columns()` what the table already
holds, uploads only the missing lanes, then runs `gpu_prepare_resident(tag)`
so the view exists and its sort cache is built before the statement is
rewritten. A device join's base sets (`…:join`) take the same route: their
lanes are the table's columns under the same names, so Q12's `l_orderkey` is
one column for the single-table statements and the join.

**Measured (SF10, 22 TPC-H queries back to back, unlimited budget, M4 Max,
16 of 22 on the device, all identical, on both).** On main every statement
owns its lanes: 32 sets, 45.5 GiB — 11 single-table sets 4.8 GiB, 8 join
base sets 7.2 GiB, 13 join-result sets 33.5 GiB. With the store: the
single-table and join-base lanes are 35 shared columns, 11.4 GiB (`lineitem`
16 columns 9.8 GiB, sort caches included), 11 views at 0 bytes, and the same
33.5 GiB of join results — 44.9 GiB. The saving on this workload is 0.6 GiB:
its sets barely overlapped in lanes. What is still uploaded per statement is
join *results* — the uploaded joins DuckDB evaluates for the shapes the
device join does not plan yet (composite keys, cross-table expressions, keys
from several tables) and the device join's own materialised results. That is
stage D, three quarters of the memory.

## 6. Stage C in detail — width

A lane's width is a storage detail (invariant 4): `ResidentColumn::dtype()`
keeps saying `I64`, `rows()` and `null_count()` mean what they meant, and
`gpu_backend.hpp` is untouched — the interface is frozen for CUDA. What
changed is inside the Metal backend.

**What is narrow.** Every I64 lane that arrives through the exact path —
`upload_rows_exact` (the store, join base sets, uploaded join results) and
`upload_pair_exact` — is stored at the narrowest signed width its values fit:
1, 2, 4 or 8 bytes. `MetalResidentColumn` carries that width; `resident_bytes()`
reports the real bytes.

**How the width is chosen.** The upload's own parallel per-span copy already
sees every value, so each span reduces a min and a max per lane over the cells
it writes, NULL cells excluded (a NULL is stored as 0, which fits any width).
The per-span partials merge and the width follows from the range, boundaries
inclusive: `[-128, 127]` → 1, `[-32768, 32767]` → 2,
`[-2147483648, 2147483647]` → 4, otherwise 8. A lane with no valid cell at all
takes 1 byte. A second parallel pass, over (lane, row range) tasks, packs each
narrowing lane into a buffer of that width and drops the I64 staging.

**The sort cache.** A column's cache is the sorted valid keys plus the
permutation to row ids. The keys keep the column's width; the permutation is
u32, which the Metal cache could always have been (it already refuses a column
above 2^32 rows). The radix sorter still sorts i64 pairs: the lane is widened
by the same per-thread copy that stages it, and the sorted pair is packed back
down to (width, u32) by one kernel — 3.68 ms against 13.4 ms for the identical
loop on the host, at 60M rows and width 2, so the device does it. A cache over
a key of width 2 costs 6 bytes a row where it used to cost 16.

**In the kernels.** Every kernel that reads a lane, a sorted key or a
permutation entry takes the width as a uniform and loads through one helper
(`ldw` in `sum.metal`; `stw` for the one kernel that writes a lane, the join's
gather). 20 kernels of `sum.metal`'s 41 take a width and two more changed only
because the permutation is u32; the branch is the same for every thread of a
dispatch, and one new kernel in `groupby.metal` packs the sort cache. A narrow
element offset is not a legal `MTLBuffer` offset, so where a kernel used to be
bound at `lo * 8` into the sorted keys it now takes `koff` as an index instead. The host reads a lane through the same widening
(`load_w`): the key-range binary search over the sorted cache, the NULL-key
fold, top-k, the row-returning join.

Measured at SF1 with `GPUDB_METAL_TRACE_EXACT=1` and `SET threads TO 1` (min of
10) no hot kernel lost time — `gbx_mask_i64` is within 1%, the reduce stages
are 8–68% faster because they read fewer bytes — so the runtime width stayed
and the function-constant specialisation the plan held in reserve was not
built. The numbers are in BENCHMARK.md.

**What is not narrow, and why.**

- **F64 lanes**: 8 bytes, always. The exact path never reduces a double on the
  device (no doubles in MSL) and there is no narrower IEEE image to keep.
- **The legacy columns** (`gpu_upload`, `gpu_upload_pair`,
  `upload_pair_interleaved`) and the v0.6 GROUP BY in `groupby.metal`: not on
  the exact path, no min/max pass, 8 bytes. Their sort caches do get the u32
  permutation, which is why `gpu_residents().bytes` for a prepared
  `gpu_upload_pair` set is 28 B/row rather than 32.
- **String lanes**: a string key is a 64-bit hash with a dictionary beside it.
  Hashes fill the range, so the min/max rule lands them at 8 by itself.
- **Join results**: they are narrow, but only as narrow as their source lanes
  — a gather cannot widen a lane's range, so each output lane keeps its
  source's width. The result is still a copy; that is stage D.

**The wrapper's estimate sizes lanes from their type (2026-09-18).** The
backend chooses a lane's width from the values, which the wrapper cannot know
before the upload; the column's DuckDB type bounds it, and that is what
`estimate_set_bytes` now charges where the build reports `narrow=true`
(`gpu_build_info()`, from `Aggregator::narrow_lanes()`): BOOLEAN / TINYINT 1,
SMALLINT 2, INTEGER / DATE 4, everything else — BIGINT, DECIMAL images, computed
lanes, string hashes — 8, plus the validity bit per row and lane, plus the key
lane's sort cache (the key's width and a u32 row id per row) when that lane is
the one being uploaded, plus one row-sized scratch lane. Since 2026-09-20 the
type is not the only bound: for a lane that is a plain column the wrapper also
reads the column's min and max from DuckDB's own zone-map statistics
(`SELECT stats(col) FROM t LIMIT 1`, metadata, once per template) and takes the
narrower of the two. The type alone was a poor bound — an `INTEGER` key holding
0-999 is stored at two bytes, a `BIGINT` payload holding 0-96 at one — and the
estimate came out 2.4x the truth, which is what the admission rule compares, so
it refused sets that fit. Statistics are BOUNDS, loose if anything, which is the
direction an upper bound needs; a lane whose statistics cannot be read, or whose
min/max are not plain integers, keeps its type's width. It stays an UPPER bound,
which is what the admission rule needs, and the wrapper test that asserts the
estimate never falls below what `gpu_residents()` / `gpu_store_columns()` report
covers a narrow-typed table (INTEGER key, DATE, SMALLINT) as well as a BIGINT
one. On a backend without narrow lanes — CUDA, the CPU reference — every lane is
charged 8 as before. A set with no key at all (`docs/TRANSPARENT_DESIGN.md`
§4.12) is charged no sort cache.

## 7. The group-id lane — a second derived structure

The sort cache (invariant 3) is what the exact GROUP BY reads: sorted valid
keys plus a permutation to row ids. It answers any key. A key with FEW
distinct values does not need an answer that general, and paying for one is
most of what those statements cost — at SF10 a flat 2.9–3.1 ms to find the
run starts and 5.4–7.1 ms to gather the first payload through the
permutation, with 2.5–3.7 ms for every further payload, for a result with
four rows in it.

So a key column gains a second derived structure beside the sort cache:

```
column "l_returnflag"   rows 0..n-1, hashed key lane, validity bitmap
  ├── sort cache        sorted valid keys + u32 row ids          (any key)
  └── group-id lane     gid[row] = rank of the row's key         (few keys only)
                        among the distinct valid keys ascending;
                        a NULL-key row takes the reserved id
                        n_distinct, which is the LAST group
      + the distinct keys, ascending, at the column's width
```

- **Stored at 1 or 2 bytes.** The width follows the largest id written — one
  byte while that is 255 or less, two above. A key with more distinct values
  than the direct reduce can hold gets no lane at all, and the column
  remembers that so no statement asks twice.
- **Built on the device from the sort cache**, under the column's own mutex:
  run starts over the sorted keys give each sorted position its run index (the
  block-offset plus simd-prefix scan the sort path already uses), the
  permutation scatters that id to row order, and each run's key is kept as
  that group's key. Rows with a NULL key are filled with the reserved id
  first.
- **Built by `prepare()`**, which runs inside the wrapper's upload where
  seconds are already being spent, and lazily by the first exact call that
  wants it otherwise. Counted in `resident_bytes()`; released with the column.
- **It is a backend-private detail.** `gpu_backend.hpp` is untouched:
  `dtype()`, `rows()` and `null_count()` mean what they meant, and which
  algorithm ran is a word in `gpu_last_stats()` (`path=direct` / `path=sort`)
  and in `GPUDB_METAL_TRACE_EXACT`, not a field of the interface.

What reads it is the **direct reduce**: one pass over the rows in storage
order, the whole `WHERE` evaluated per row by the same `gpred_eval` the global
aggregate uses (§4.12 of `docs/TRANSPARENT_DESIGN.md`), every payload folded
into the accumulator of `gid[row]`. No mask buffer, no run starts, no
permutation, no gather. The ids are ranks of the ascending distinct keys, so
the groups come out in the operator's order with the NULL-key group last, and
a group the `WHERE` emptied is simply absent — the sort path's contract,
unchanged.

**Where the accumulators live is the whole design.** Two shapes, measured at
SF10 and kept per range:

| accumulators | holds | measured |
|---|---|---|
| thread-private (`GAggAcc[cap]`, the global aggregate's layout with more than one group) | a handful of (group × payload) slots | 4.6 ms at 2 groups × 1 payload; 106 ms at 4 × 5, 215 ms at 25 × 3 — 256 threads times the slots is the working set and no cache holds it |
| one threadgroup slab of 32-bit atomics, replicated up to 32 times so a simdgroup's lanes never share a word | everything up to the id lane's limit | see BENCHMARK.md |

Apple GPUs (through Apple9 / M4) have no 64-bit atomics at all — device or
threadgroup — and no 64-bit simd reductions, so the slab is 32-bit
throughout: `count(*)` and `count(v)` are u32 adds, the 128-bit sum is four
u32 limbs whose carries each thread propagates itself (an atomic add returns
the old value, so a thread knows when its own add wrapped), and negative
values are added as their unsigned image and counted, the count being
subtracted at 2^64 when the limbs are assembled. `min` and `max` are u32
atomics over the order-preserving image of a 32-bit value, which is exact for
a payload lane stored at 4 bytes or fewer — stage C stores a lane at the
narrowest width its values fit, so that is every lane whose values fit int32.
A payload whose values do not, with `min` or `max` asked for, keeps the
thread-private accumulators while they hold it (32 slots) and the sort path
above that: the answer is the same either way, and only the speed of that one
shape is left. The exact version wants a second pass — the first gives the high
word of the extreme per group, the second the low word among the rows that
match it — and is worth building when something asks for it.

**Where it does not run at all.** The direct path is five compute pipelines,
and a GPU may compile the kernel library and then refuse to lower one of them —
the virtualised Apple device on a hosted macOS runner does. That is not an
error: the aggregator marks the path unavailable for its lifetime, keeps the
compiler's own text, says so once on stderr, and every exact call answers
through the sort path with `path=sort` and the reason in
`gpudb::exact_path_reason()`. A forced `GPUDB_METAL_GROUPBY_EXACT_PATH=direct`
answers the same way rather than failing. Nothing on the path throws out of an
operator or out of `prepare()`.

**The device is asked what it is before it is asked to compile anything.** On
the virtualised Apple GPU of a hosted macOS runner, ONE refused pipeline build
leaves that process's Metal compiler unusable: after `gdir_slab_i64` was
refused on `macos-15-arm64` ("Apple Paravirtual device", families Apple1–Apple7
Mac2 Common1–3 Metal3), `sum_i64`, `hashjoin_merge_sorted_i64` and
`bitonic_step_i64` all failed in the same process — kernels that had built
minutes earlier. No fallback can repair that, so the slab kernel is never
offered to a device we are unsure of. The gate, before any direct pipeline is
requested: the device must report `MTLGPUFamilyApple7` or above (where the slab
reduce's threadgroup atomics were measured), and a device whose name contains
`Paravirtual` is refused outright — a deny entry written from this evidence,
because that compiler's behaviour cannot be queried. A refusal compiles nothing,
probes nothing and builds no id lane; `GPUDB_METAL_DIRECT_DISABLE_PSO=unsupported`
simulates it and the unit tests assert that the count of pipelines asked of the
device is zero. `device_name()` carries the families, so every log says what the
machine reported.

**Every pipeline names itself when it will not build.** All four Metal
subsystems — the aggregator, the radix sorter, the v0.6 GROUP BY and the hash
join — put the function's name and the compiler's own text in the error or the
reason. A message that says only `Compilation failed` cost two round trips to
narrow down and cannot be produced any more. The unit binary line-buffers
stdout too, so a crash cannot swallow the lines that say where it was.

**The decision is made once, before anything is built for it.** The path needs
four pipelines — the id-lane pair, the merge and the slab reduce — and the
first time anything wants the path they are all asked for together. A device
that refuses any of them is latched unavailable before a single group-id lane
exists, which is what makes the fallback free rather than merely safe: an id
lane costs a byte or two per row and there is nothing to read it with. The
runner that found this refuses `gdir_slab_i64` alone and builds the other four,
so the ordering matters in practice and not just in principle. The
thread-private kernels are deliberately NOT in the required set: they serve
only a `min` / `max` over a payload lane wider than 4 bytes, they measured no
better than the sort path on any device here, and a slab-less mode built out of
them would be a shape nobody has numbers for. An optional pipeline that refuses
costs that one shape and nothing else — the call takes the sort path and every
other shape still goes direct, which `GPUDB_METAL_DIRECT_DISABLE_PSO=masked32`
and a 21-group wide-`min`/`max` case in the unit tests pin down.
The threadgroup budget and the threads a
pipeline will take are asked of the device rather than assumed, so a smaller
GPU narrows the slab instead of overrunning it; `GPUDB_METAL_DIRECT_DISABLE_PSO`
makes every pipeline refuse, which is how the fallback is tested where it would
otherwise work.

**Where it stops.** Two of the three limits are memory. The slab is
`n_groups × (6 × payloads + 1)` 32-bit words and a threadgroup may hold 30 KiB
of them, so 512 groups fit with one payload, 404 with three and 247 with five;
the id lane itself is built up to `GPUDB_METAL_DIRECT_MAX_GROUPS` distinct
values, 512 by default. Both are hard declines to the sort path and both measure
at 0.99–1.04×, the same algorithm on either side.

The third is a real crossover, and it is in ROWS, not in groups. Every
threadgroup clears its slab before its first row and folds it after its last,
and none of that is proportional to rows; below a few million rows it is the
whole call. Sizing the grid and the replicas to the row work removed most of it
(BENCHMARK.md), and what is left is an admission rule:

    groups >= 3   AND   rows × (payloads + WHERE terms) >= 6,000,000

Two groups are excluded because the sort path's reduce over two runs is a
sequential scan — its best case — while the direct path still reads the id lane;
the work bound is there because what the direct path saves is one gather per
payload and one mask pass per term. Of the 129 cells the sweep admits under this
rule, none is slower than the sort path. `GPUDB_METAL_GROUPBY_EXACT_PATH=direct`
forces past it.

## 8. Stage D — joins as index vectors

A join result today is a copy: every lane gathered into a new row-aligned set
in the join's order, 8 bytes per row and lane. At SF10 that is 33.5 GiB for
the TPC-H joins, three times the store that holds every base column.
The design replaces the copy with a **join set** = one index vector per joined
table (the row id of that table's row for each result row) plus, only when a
lane cannot come from a store, a lane in result order:

- a device join (`gpu_join_materialize`) returns the probe and build row ids
  instead of gathered lanes — 16 bytes per result row;
- an uploaded join (DuckDB evaluated it) uploads `(rowid_1, …, rowid_n)` per
  result row, and a cross-table expression lane (`x_…` over columns of two
  tables) as its own lane in result order — 8 bytes per table plus 8 per such
  lane, instead of 8 per lane read;
- the exact operators take an optional index per input column: a key,
  payload or predicate lane of a join set is `store_col[index[i]]`; the sort
  cache of a join set's key is built over the gathered key, as now.

The kernels gather through the index on unified memory (the index is mostly
in probe order, so the reads are sequential runs); the CPU reference does the
same, and the parity tests compare them. Chunks (appends, tables above the
budget) come with the same stage: an index vector already addresses rows by
id, and a chunk is a range of ids.

### 8.1 D1 — the mechanism (2026-09-19)

**What is built.** The interface additions are in `gpu_backend.hpp` and are
additive only, because the file is frozen for the CUDA port: `IndexedColumn`
(a lane and the index it is read through), a defaulted `index` field on
`Predicate`, `MultiPayload` and `JoinLane`, `join_index` with
`Aggregator::JoinIndexResult`, `indexed_supported()` as the rule-1 gate, and
`groupby_exact_masked_multi_indexed`. Nothing existing changed shape.

The CPU backend implements all of it as the obvious loop and is the parity
oracle. On Metal the indexed read lives in one helper, `gl_load` in
`sum.metal`, and reaches every operator that reads a lane through a `GLane`:
`gpred_eval` — and so every WHERE program, on the sort path, the direct path
and the global aggregate alike — the global masked aggregate itself, and both
shapes of the direct row-order reduce.

**Where the index vectors are bound.** The plan gave them a buffer of their
own. There is no room: the direct reduce already binds 0..30, which is every
argument-table slot Metal offers. An index vector is already shaped like a
lane — a storage width, a validity bitmap, and a NULL cell that means exactly
the NULL row an outer join wants — so it is bound as an ordinary lane and the
lane that reads through it names that slot, in the word `GLaneMeta` used as
padding. No new binding, no new pipeline, nothing new to compile, and so no
capability gate and no fallback to arrange: the kernels that gained the
indirection are the ones the backend already builds and already falls back
from. The cost is budget instead — the twelve distinct lanes a statement may
read now have to cover its index vectors too, and the operator refuses above
that, as it always has.

A slot is the pair *(column, index)*, not the column: one store column read
through two different indexes holds two different values per row.

**The bound.** A kernel uses an index cell as a row and has no bound of its
own. The bound is proved on the host from the index column's minimum and
maximum over its valid cells, computed once and kept on the column — an index
vector is read by every statement over its join set, so one sequential pass at
the lane's stage-C width amortises away. The CPU reference checks every cell.

**Composition.** A chained join composes index vectors,
`idx_new[d] = idx_old[probe_rows[d]]`. That is a gather of the old index at
the new probe rows, which is what the materialise path already does to any
lane, so a chain hands the previous step's index vector to the next step as an
ordinary materialised lane. No new operator.

**What was measured first.** The plan was drawn against a main that has since
changed, so the inventory was re-taken at SF10 before any code was written:
18.73 GiB resident after the 22 queries (33 sets at 14.69 GiB, 35 store
columns at 4.04 GiB), of which the three device-join sets are 3.16 GiB and the
ten uploaded ones 11.53. Q12's device set had fallen from 26 to 15 bytes a row
and Q11's from 18 to 12, because the direct reduce replaced the key's 12-byte
sort cache with a 1-byte group-id lane; Q21's is unchanged at 40, its key
having 100,000 groups and still taking the sort path.

The plan priced D1 as removing 97 ms of gather on Q12 and 190 ms on Q21. Those
figures are cold: hot, at SF10 and one thread, Q12 answers in 10.4 ms, Q21 in
20.5 ms and Q11 in 47.1 ms, and the join's gather runs once when the set is
built. So D1's saving is on the residency build and on the size of the
device-join sets, and the hot statement must only be no slower. That is the
honest claim for it.

### 8.2 D1 on Metal — what it cost, and what it is worth (2026-09-19)

**The read had to be compiled, not tested.** The first shape of the indexed
read put one test inside every kernel that reads a lane: *is this lane read
through an index?* No statement was indexed, so any difference was pure
overhead — and it was 8–15% at SF10 on the masked kernels, measured against
`main` with two interleaved runs of the 22 queries at one thread (Q19 1.148×,
Q14 1.118×, Q6 1.116×, Q12 1.110×, Q1 1.082×). The reason is not the branch.
`row` is the row loop's induction variable, so the address of `lane[row]` and
the word of its validity bitmap strength-reduce across the loop; a row that
comes back from a function does not, and every masked kernel lost its address
arithmetic.

So the read is a compile-time parameter. `gl_row` and `gpred_eval` are
templated on whether any lane of the dispatch is indexed, every kernel that
reads a lane is built twice, and each dispatch picks its instance from a flag
that is uniform over the whole grid (computed from the lane table the host
already writes — no new uniform, no new binding). The index slot reaches the
thread's lane table only when there is one. With the flag false the kernel is,
literally, the pre-stage-D kernel. Re-measured the same way: worst 1.012× at
SF10 and 1.011× at SF1, with 14 of 17 device queries at or below `main` — run
to run spread, and no query slower.

**What the index costs when it IS used.** A lane read through an index against
the same lane gathered into result order, on the exact grouped reduce, 16M
probe rows joined to a 200k dimension, min of 7 (`ixsweep`, M4 Max):

| groups | payloads | WHERE | join build ms (materialise / index) | reduce ms materialised | reduce ms indexed | indexed / materialised |
|---|---|---|---|---|---|---|
| 4 | 1 | no | 13.8 / 9.4 | 1.78 | 2.10 | 1.180× |
| 4 | 3 | no | 12.5 / 8.7 | 3.18 | 3.70 | 1.161× |
| 4 | 1 | yes | 11.7 / 8.6 | 1.96 | 2.40 | 1.224× |
| 64 | 1 | no | 11.9 / 8.6 | 1.77 | 2.06 | 1.163× |
| 64 | 3 | no | 11.9 / 8.7 | 2.88 | 3.57 | 1.240× |
| 64 | 1 | yes | 11.9 / 8.7 | 2.03 | 2.42 | 1.193× |
| 1000 | 1 | no | 12.2 / 8.9 | 3.74 | 5.65 | 1.512× |
| 1000 | 3 | no | 12.3 / 9.2 | 5.81 | 11.13 | 1.917× |
| 1000 | 1 | yes | 12.8 / 9.5 | 5.21 | 8.89 | 1.706× |

The indexed read loses on **every** shape tried, by 1.16× at the smallest and
1.92× at the largest, and the loss grows with the groups and the payloads —
the gather is random where the materialised read is sequential, and the more
work per row the more of it is gather. The join that returns row vectors is
1.3–1.4× faster to build than the one that gathers lanes, which together with
the bytes is what stage D was for.

**So the rule.** An index vector is a residency-build and memory win and a
hot-query loss, and rule 1 is written from the measurement, not from the
plan: a lane a hot statement READS keeps its materialised copy; the index is
for what a set carries but statements do not read per row. The step-0
inventory said D1 should be neutral on hot query time; it is not, and the
honest consequence is that the SQL layer must not simply swap the three
device sets over to index vectors. What that admission rule should key on —
memory pressure against the per-statement loss above — is D2's question.

**The sort path keeps the gather.** The sort path does not read a lane at the
row; it reads it at `perm[i]`. An index there is a double gather on a path
whose reduce is already gather-bound, so an indexed lane reaching the sort
path is gathered once, by the operator, and the operator that ran before
stage D then runs bit for bit. The indexed read on that path is not built and
is not claimed either way. The direct path reads every lane at the row, so
there the index is the one extra load the table above prices.

**The key is always a lane.** Both paths read a key through structures derived
from the key column and shared by every statement over it — the sort cache,
the group-id lane, the distinct keys. An index would make those per statement.
So an indexed key reaching `groupby_exact_masked_multi_indexed` is gathered
into a lane first, which is what §8 says a join set's key is.

**Built.** `join_index` on Metal: `join_materialize` and it now share one
probe stage — uniqueness, the binary-search probe, the class counts — so the
two agree term for term by construction, and the last dispatch is `jm_rows`
(the probe and build row of each kept output row, each at the narrowest width
its side's row count fits) instead of one gather per lane. The probe key, the
classifying key lane and every materialised lane may each be read through an
index, which is what a chained join needs. `groupby_exact_masked_multi_indexed`
on Metal, with the direct path reading payload and predicate lanes through
their index vectors and the sort path gathering them. Parity is checked
against the CPU reference and against `join_materialize` for the 1/2/4/8-byte
widths, F64 NaN / ±inf / −0.0, NULL lane cells, NULL index cells as payload
and as key, the chained three-step composition, and an out-of-range index
(refused, not read).

**Not built.** There is no `gpu_join_index()` in the extension, no join set
that owns index vectors, no reporting columns for one in `gpu_residents()`,
and the wrapper emits no index steps. Nothing in SQL takes the indexed path,
so the measured behaviour of every query is what the rest of this document
describes.

## 9. Shedding — a derived structure a column stops needing

Every structure §7 added is derived: the sort cache is derived from the lane,
the group-id lane and its distinct keys are derived from the sort cache. Once
the direct reduce runs, the derivation runs backwards too — `key[row] =
dkeys[gid[row]]` — so a column that the dispatch rule keeps sending to the
direct path can release what the direct path does not read and rebuild it if
anything ever asks.

Two things can go, and they are the ones the memory is in:

| released | costs today | needed by |
|---|---|---|
| the sort cache (`sorted_` + `perm_`) | `sort_width + 4` per valid row — 12 bytes for an 8-byte key | the sort path: the run starts, the permutation gathers, the key-range binary search, top-k, a join build side |
| the key lane itself | the storage width per row — 8 bytes for a hash key | every read of the key AS A VALUE: a WHERE on the GROUP BY key, a payload over it, the sort cache's own build |

**When.** The decision is made at the end of a call the direct path answered,
because that is the only moment the backend knows both the column and the shape
of the calls it serves. The rule is the dispatch rule (§7) evaluated at the
LEAST work a call can bring — one payload, no WHERE term:

    groups in [3, 512]   AND   rows >= 6,000,000

(`GPUDB_METAL_DIRECT_MIN_GROUPS` / `_MAX_GROUPS` / `_MIN_WORK`, the same
constants the dispatch rule reads). So a 2-group key does not shed — the sort
path's reduce over two runs is its best case and that is where its calls go; a
key above the id lane's limit has no id lane to rebuild from and does not shed;
an input below the work rule does not shed, because its next call takes the sort
path too; and a device where the direct path is unavailable never sheds, since
no call ever reaches the decision. The forms the direct path declines per call —
a payload shape beyond the slab, `min` / `max` on a lane wider than 4 bytes,
top-k by key, `count(*)` with no payload at all — are what the rebuild is for.

A join BUILD side cannot reach the rule at all, and that is a proof rather than
a hope: a build key is unique among its valid cells, so a column with at most
512 distinct values has at most 512 valid rows, and the row rule needs millions.

**The key lane needs one more thing: that it is a key and nothing else.** That
is a fact about the NAME the extension gave the lane, so the extension says it
(`src/include/resident_shed_note.hpp`, a note beside the upload call, the same
shape as `exact_path_note.hpp` — `gpu_backend.hpp` is frozen and carries no
field for it):

- a store column named `k#<field>` is the tuple TEXT of a GROUP BY key
  (`_rewrite.store_lanes`); the raw column a WHERE or a payload reads lives in
  the store under its own name, so nothing else can reach this lane;
- lane 0 of a join RESULT set (`…:join-<digest>`, `…:joinu-<digest>`) is that
  statement's key. An INTERMEDIATE of a chained device join is written
  `…:join-<digest>.<step>` and is explicitly NOT one — its lane 0 is the next
  step's probe key.

Everything else — a plain store column, a base set's key, a set uploaded by
hand — is never marked, and keeps its lane.

On top of that, the call that just ran must not have read the key lane itself.
A WHERE on the GROUP BY key is an ordinary shape (TPC-H Q12 is one), the direct
plan lists that column among its lanes, and a set whose statement has one
therefore keeps its lane from the first call instead of rebuilding it later.

**The rebuild.** `buffer()` is the one door to the lane and rebuilds it from
(group ids, distinct keys) when it was shed; `build_sort_cache` gets the lane
that way and sorts it as usual. A NULL-key row's cell comes back as zero: every
reader gates on the validity bitmap (stage A), so the value under a zero bit is
never part of an answer. Both rebuilds PIN what they rebuilt — a column that
something reads as values is a column the rule misread, and one rebuild is the
price of being wrong; a sort is worth more than the bytes. So a column pays this
at most once per structure, and `gpu_build_info()` reports the two counts as
`rebuilds=<caches>/<lanes>` so a workload can be checked;
`GPUDB_METAL_TRACE_EXACT=1` prints a line per rebuild, with the rows, the
groups and the width. Over the 22 TPC-H queries at SF10 both counts are zero,
and over the coverage runs at SF1 and SF10 too. `scripts/transparent_gate.py
--subqueries` — which sweeps every form over every key, including shapes no
TPC-H template produces — rebuilds 3 lanes and 4 caches in 782 cells, each
column once, 5.7 to 18.4 ms apiece, and stays exit 0 with every rewritten cell
at or above its bound.

**Reporting and the budget.** `resident_bytes()` counts what the column is
holding at the time, so `gpu_residents()` and `gpu_store_columns()` follow a
shed down and a rebuild back up, and the budget's LRU and eviction are
unchanged. `prepared()` stays true across a shed: what the column keeps derived
is built, and the rest comes back by itself. The wrapper's PRE-upload estimate
is deliberately not adjusted — it must stay an upper bound over a set that has
not been uploaded yet, so it keeps charging every lane 8 bytes (or its DuckDB
type's width) and the key a `width + 4` sort cache.

### 9.1 A set no key is ever read from

§4.12's global masked aggregate reads payload and predicate lanes and never a
key. Over a single table such a set is a view over the store that simply names
no key lane (`-` in lane 0 of its tag). Over a JOIN it used to carry one anyway:
the split hands the matcher a constant key so the statement reads as a GROUP BY,
and that constant became a lane of the uploaded or materialised result — 1 byte
a row, plus a 5-byte sort cache and a group-id lane over a column with exactly
one group, none of it ever read.

It is gone on both join paths. `make_global` now strips the key for a join as
well; the uploaded set's lane-0 slot arrives as `CAST(NULL AS BIGINT)` (the
row-major upload is positional, so the slot exists) and is dropped when the set
is published, and a device join gathers no key lane at all — its `gpu_join_
materialize` spec starts at the payload. The published set's `keys` points at
the payload so the set's invariants hold, `no_key` is set, and nothing prepares
or sorts it. Measured at SF10: Q14, Q17, Q19 lost 7 bytes a row each, Q11 lost 7
of 19.

### 9.2 Measured (SF10, the 22 TPC-H queries back to back, unlimited budget)

M4 Max, Metal, `data/tpch_sf10/tpch.duckdb`, `residency="eager"`, every answer
identical to native on both sides.

| | before | after |
|---|---:|---:|
| store columns | 5.319 GiB | 4.042 GiB |
| join-result sets (13) | 18.153 GiB | 14.693 GiB |
| **total** | **23.472 GiB** | **18.735 GiB** |

Per join-result set, bytes per row:

| Q | set | before | after | what left |
|---|---|---:|---:|---|
| 9 | `joinu-04b669f1…` | 26 | 6 | key lane 8 + cache 12, id lane +1 |
| 12 | `join-174ae33c…` | 27 | 15 | cache 12 (the WHERE reads the key lane), +1 |
| 17 | `joinu-…` | 28 | 21 | the constant key 1 + its cache 5, +1 |
| 19 | `joinu-…` | 28 | 21 | as Q17 |
| 8 | `joinu-242fbba5…` | 35 | 27 | key 2 + cache 6, +1 |
| 14 | `joinu-…` | 17 | 10 | the constant key and its cache |
| 11 | `join-…` | 19 | 12 | the constant key and its cache |
| 3, 5, 7, 10, 18, 21 | | — | unchanged | more distinct keys than the id lane holds, or too few rows |

Q7's key looks like a 4-group key in the ANSWER; the column has 1250 distinct
values before the WHERE, which is what the id lane is built over, so it keeps
everything. Q3, Q10, Q18 and Q21 have millions of groups. Q5's set is 2.4M rows,
below the work rule.
