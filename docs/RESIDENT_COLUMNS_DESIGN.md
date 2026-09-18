# Resident columns — the v0.8 storage design

Status: design 2026-09-18; stage A merged (#123), stage B merged (#124), stage C built
2026-09-18 (this PR, Metal); stage D next. Companion to
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
unchanged).

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

**The wrapper's estimate is now an over-estimate.** `estimate_set_bytes` still
charges 8 bytes per row and lane, so the budget admits a set on a figure that
can be three times what the set costs. Nothing is wrong — the budget is
conservative, never optimistic — but a table that would now fit can still be
refused. Sizing the estimate from the column's type (what the store knows
before the upload) belongs with stage D, where a set stops being a copy at all.

## 7. Stage D — joins as index vectors

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
