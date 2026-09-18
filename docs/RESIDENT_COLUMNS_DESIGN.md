# Resident columns — the v0.8 storage design

Status: design 2026-09-18; stage A in progress. Companion to
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
| B | **TableStore**: columns uploaded by (table, expression) in row-id order (`gpu_upload_columns`), sets become references; the wrapper asks what is resident and uploads only the missing columns; budget and LRU per column | extension, wrapper | the 22 TPC-H queries at SF10 fit a 16 GiB budget; 33 GiB → measured |
| C | **Width**: I32 / I16 / I8 storage chosen from the upload's min/max; typed loads in the exact kernels (function constants, one PSO per width) | Metal, CPU | memory and kernel bytes measured; gate unchanged |
| D | **Index-vector joins and chunks**: a device join returns row ids; INSERTs append chunks; cold chunks stream | extension, Metal, wrapper | appends resident in ms; a table above the budget answers on the device for the shapes that win |

CUDA implements the design once, after stage C, from `docs/CUDA_EXACT_PATH.md`
(to be revised at that point).

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
