# v0.7 — transparent GPU execution (design)

Status: proposal, 2026-09-01. Companion to `GROUPBY_RESIDENT_DESIGN.md` (v0.6).
Carries the v0.6.1 maintenance release plan in §11 because v0.7's build
migration inherits from it.

## 0. Two rules

**Rule 1 — never slower.** With gpudb loaded, a query is never slower than the
same query without it. Not on average: per query, per shape, per size, on
both backends. This is enforced by a benchmark gate (§9.3) that times every
shape the rewrite can produce against native on every release; a single row
under 1.0× blocks the release.

**Rule 2 — never different.** A rewritten query returns exactly what native
returns. Where the v0.6 resident operators and native DuckDB differ today
(NULLs, overflow, DECIMAL, summation order), v0.7 makes the device **exact**
rather than falling through around the difference. Falling through is
allowed for shapes we do not handle yet; it is not allowed as a way to avoid
handling a shape correctly.

The end state a user sees:

```sql
LOAD gpudb;
-- that's it. Their existing SQL, unchanged, gets faster where it can.
```

No `gpu_*` names, no `gpu_pin`, no cast to BIGINT. The v0.6 functions stay
available as the explicit form and as the rewrite targets.

## 1. What is being built

Five pieces, in dependency order:

| # | Piece | Why it exists |
|---|---|---|
| A | **C++ extension API migration** (§3) | the optimizer hook does not exist in the C API |
| B | **Exactness on the device** (§4): NULL-aware resident columns, 128-bit sums, DECIMAL as scaled integers, packed multi-column keys | rule 2 without fall-through |
| C | **Automatic residency** (§5): the extension decides what to keep on the device, uploads in the background, and invalidates on writes | no `gpu_pin`; rule 1 even on the first query |
| D | **The rewrite** (§6): `GROUP BY` with `WHERE`, `HAVING`, `ORDER BY … LIMIT`, over resident columns, with a device-side predicate mask | the user-visible feature |
| E | **The gate** (§9): break-even sweeps, thresholds per backend, `transparent_gate.sh`, three-way parity | rule 1 mechanically |

Joins on the transparent path are v0.8 (§10) and are designed for here so
nothing in A–E has to be redone for them.

## 2. Plan shapes covered

```sql
SELECT k1 [, k2, k3], sum(v) | count(v) | count(*) | min(v) | max(v) | avg(v)
FROM t
[WHERE <predicate on columns of t>]
GROUP BY k1 [, k2, k3]
[HAVING <agg> {> >= < <= = <>} <constant> [AND …]]
[ORDER BY <agg> | k ASC|DESC]
[LIMIT k]
```

- Keys: any integer family, DATE, TIMESTAMP (all 64-bit-representable), up to
  three of them packed into one 64-bit key when their ranges allow (§4.4),
  and VARCHAR through a dictionary when the column is low-cardinality enough
  that a dictionary is cheap (§4.5).
- Payloads: integer family, DECIMAL (§4.3), DOUBLE/FLOAT.
- Aggregates: `sum`, `count`, `count(*)`, `min`, `max`, `avg` (= sum/count,
  computed on the host from the two device results). Several aggregates over
  the same payload in one query are one device pass.
- `WHERE`: conjunctions of comparisons between a column of `t` and a
  constant, `IN (list)`, `BETWEEN`, `IS [NOT] NULL`. Evaluated on the device
  as a mask over the resident predicate columns (§4.6). Anything else (a
  function call, a subquery, a correlated reference) is not a resident shape
  and runs native.
- `HAVING`: any conjunction of aggregate-vs-constant comparisons (v0.6
  supports one; the `=`/`<>` cases and conjunctions are host-side over the
  device survivors until a kernel exists, which is still a device filter for
  the expensive comparison).
- `ORDER BY agg LIMIT k`: device top-k. `ORDER BY key`: rows already arrive
  sorted by key; the sort node is dropped. `ORDER BY` anything else: left
  native above the rewritten node.

Everything above the matched subtree (projection, aliases, joins to the
result, further `ORDER BY`) is untouched; the replacement node reproduces the
aggregate's column bindings (`ColumnBindingReplacer`).

## 3. C++ extension API migration (piece A)

The DuckDB C API exposes no optimizer hook. `OptimizerExtension`
(`duckdb/optimizer/optimizer_extension.hpp`; a struct holding an
`optimize_function(OptimizerExtensionInput&, unique_ptr<LogicalOperator>&)`
and an `optimizer_info`, appended to `DBConfig::optimizer_extensions` at load)
is C++ only. It is not in the `duckdb.hpp` amalgamation either — the spike
needs the DuckDB source tree, i.e. the `duckdb` submodule that
`extension-ci-tools`' C++ template flow expects anyway.

Consequences, accepted:

- **Per-DuckDB-version binaries.** The community registry already builds and
  serves per version (it serves v1.5.5 only today), so the C ABI's
  forward-compatibility was not reaching users through that channel. Release
  assets become per-version; the release checklist gains a DuckDB tag field.
- **Community CI time** rises from ~1 min to the usual 15–25 min per platform
  (DuckDB is compiled from the submodule). Every other C++ community
  extension pays this.
- **Local build.** `scripts/build.sh` gains `-DGPUDB_DUCKDB_SOURCE_DIR`
  (default: the submodule). The loadable links no libduckdb — symbols
  resolve from the host process (`-undefined dynamic_lookup` on macOS).
- **`gpudb-sql`** (the embedded CLI the SQL test runner drives) stops
  registering functions through the C API and `LOAD`s the freshly built
  loadable instead. One registration path, not two.
- **Removed in the same PR:** `third_party/duckdb_capi/`,
  `src/extension/duckdb_loadable.cpp`, the C-API template `Makefile`
  includes (replaced with the C++ template's).

This lands **alone** as PR 1 with the entire v0.6 verification surface green
on both machines (unit, SQL suite, both parity scripts, registry-smoke
probes, community `make configure && make release && make test`) and an
audit pass of the shape used before the v0.6 tag. Nothing stacks on it until
it is merged. It is the highest-risk step and it is isolated on purpose.

## 4. Exactness on the device (piece B)

Today's resident columns equal
`… WHERE k IS NOT NULL AND v IS NOT NULL` with 64-bit sums and BIGINT-only
payloads. Each gap below is closed on the device so the rewrite never has to
decline for a semantic reason.

### 4.1 NULL-aware resident columns
A resident column gains an optional validity bitmap (1 bit/row, uploaded
alongside; omitted when the source has no NULLs, which the pin records). The
sort/segmented-reduce paths treat a NULL key as its own group (sorted last,
as DuckDB orders NULLs in `GROUP BY` output — verified per backend by parity)
and a NULL payload as contributing nothing to `sum`/`min`/`max` and nothing
to `count(v)` while still counting for `count(*)`. Cost: one extra mask read
per element in the reduce; measured, and the bitmap is skipped entirely for
NULL-free columns so v0.6 numbers are unchanged on the benchmark tables.

### 4.2 128-bit sums
`sum(BIGINT)` in DuckDB is HUGEINT and never overflows. The device
accumulates in two 64-bit limbs (segmented reduce with carry; CUDA has native
128-bit arithmetic via CUB's reduce on a two-limb struct, Metal does the
carry by hand — both are already the shape of the existing reduce). Output
column is HUGEINT. The extra limb is skipped when the pin's static bound
`rows × max|v| < 2^63` proves it unnecessary, which is the case on every
benchmark table, so v0.6 numbers are unchanged.

### 4.3 DECIMAL
DuckDB stores `DECIMAL(p,s)` with `p ≤ 18` as an int64 scaled by `10^s`
(int16/int32 for smaller `p`). A DECIMAL payload is uploaded as its integer
representation; `sum` is the 4.2 path with the result typed `DECIMAL(38,s)`;
`min`/`max` are exact; `avg` is `DECIMAL` division on the host. This is what
makes TPC-H's `l_quantity` / `l_extendedprice` match with no cast — the
README examples currently write `l_quantity::BIGINT` and v0.7 stops needing
to. `DECIMAL(p>18)` is int128-backed and is uploaded as two limbs (same
machinery as 4.2).

### 4.4 Packed multi-column keys
`GROUP BY a, b [, c]` over integer/date/timestamp columns: at upload, each
column's `[min, max]` is read from DuckDB's statistics; if
`Π (range_i + 1) < 2^63` the tuple is packed as a mixed-radix integer
`((a−min_a)·R_b + (b−min_b))·R_c + (c−min_c)`, sorted and reduced as one
64-bit key, and unpacked on the way out. Exact, no new kernels. When the
product does not fit, the shape is not resident (native).

### 4.5 VARCHAR keys via dictionary
A VARCHAR key is uploaded as a dense int32 id against a host-side dictionary
built at upload (one pass, hash map). Results map ids back to strings. The
dictionary is only built when DuckDB's distinct-count estimate is below a
threshold measured in §9.1 (a 15M-distinct-string dictionary is not a win);
above it the shape runs native. Low-cardinality VARCHAR `GROUP BY` is
usually a native win anyway (§9.1 decides).

### 4.6 Device-side predicate mask (`WHERE`)
Most real `GROUP BY` queries filter first. The resident set for a table
includes the columns that appear in `WHERE` clauses of matched shapes (§5
decides which); at query time the conjunction of comparisons is compiled to a
small predicate program (column, op, constant) evaluated on the device into a
selection mask that the segmented reduce consumes. New kernel on each backend
(mask build + masked reduce); the masked reduce is the existing reduce with a
predicate read, so its cost is measured against the unmasked one and both
sides are in the gate.

### 4.7 DOUBLE summation
Native DuckDB's `sum(DOUBLE)` is itself order-dependent across thread counts.
The device result is "equal up to reassociation"; parity uses the tolerance
`groupby_parity_check.sh` already applies. This is the one place where
"exact" means "as exact as native is with itself", and it is stated in the
docs. Kahan-compensated device accumulation is implemented behind a setting
so users who need the tighter bound can turn it on and see the cost.

## 5. Automatic residency (piece C)

No `gpu_pin`. The extension keeps a **residency manager** per database:

**What becomes resident.** The optimizer pass sees every plan. When it sees a
rewritable shape (§2) over a table whose size and estimated group count pass
the §9.1 thresholds, and the needed columns are not resident, it records the
shape and **runs the query native** (rule 1: the first query pays nothing)
while scheduling an upload of those columns on a background thread through a
second connection on the same database instance. The second matching query
finds them resident and is rewritten. `SET gpudb_residency = 'eager'` uploads
on first sight instead (for scripts that know their workload);
`'manual'` restores v0.6 behaviour (`gpu_upload_pair` only).

**Memory budget.** `SET gpudb_memory_budget = '8GB'` (default: 50% of device
memory on a discrete GPU, 25% of unified memory on Apple silicon). Resident
sets are evicted LRU by last use; a set that does not fit is not uploaded.
`gpu_residents()` lists what is on the device, its size, hit count, and the
plans it served.

**Invalidation — the part that has to be right.** Three mechanisms, layered:

1. **DML seen in the optimizer.** `INSERT`, `UPDATE`, `DELETE`, `TRUNCATE`,
   `COPY … FROM`, `CREATE OR REPLACE`, `ALTER`, `DROP` all pass through the
   optimizer as logical plans naming their target table. The pass marks every
   resident set on that table invalid *before* the statement executes. If the
   transaction rolls back, the set is simply re-uploaded next time — the
   safe direction.
2. **Row-count check at rewrite time** (catches the Appender API and anything
   else that bypasses the planner and adds or removes rows): the table's
   current row count must equal the count at upload.
3. **Transaction visibility.** An upload runs in its own transaction and
   records the snapshot it read. A rewrite is applied only when the querying
   transaction started after that upload committed, so a connection with an
   older snapshot never sees newer resident data. (Verified in the spike;
   if the C++ API does not expose what is needed, uploads are done
   synchronously inside the first query's transaction under `'eager'` and
   the background path is dropped — rule 2 beats rule 1.)

What is not caught: in-place updates through a non-SQL path that keeps the
row count constant. DuckDB has no such public path today; KNOWN_ISSUES says
so and says what to do if one appears (`CALL gpu_invalidate('t')`).

**Shared with joins.** A resident set is `{table, columns, sorted key
permutation, validity}`. The v0.5 join build side is the same object, which
is what lets v0.8 route joins through the same manager (§10).

## 6. The rewrite (piece D)

`OptimizerExtension` pass, before DuckDB's own optimizers so that our node is
in place for the rest of the pipeline (filter pushdown, top-N, projection
pruning see a `LogicalGet` with known cardinality and sorted-by-key
property). Steps:

1. Match the §2 subtree bottom-up; on the first non-matching node, stop.
   Measured cost for an unmatched plan is under 0.05 ms (unit test over
   10,000 plans).
2. Check residency (§5) and the thresholds (§9.1); on a miss, record and
   return the plan untouched.
3. Build the replacement: a `LogicalGet` over the C++ `TableFunction` for the
   resident operator with bind data `{resident set, predicate program,
   aggregates, filter, top-k}`; the function reports cardinality
   (`groups_total` after first run, DuckDB's distinct estimate before) and
   the sorted-by-key property. Rewire bindings with `ColumnBindingReplacer`.
4. `EXPLAIN` shows `GPU_GROUPBY (resident: t.k,t.v; predicate: …; having: …;
   topk: …)`. `SET gpudb_explain_rewrites = true` logs, for every candidate,
   whether it was rewritten and if not why (shape / not resident / threshold
   / stale / budget), which is also what the gate parses.
5. `SET gpudb_transparent = false` disables the pass entirely.

`gpu_last_stats()` gains the plan-level fields (rewritten yes/no, reason,
residency hit/miss, predicate rows in/out).

## 7. Backend work (both, in parallel)

| Kernel / path | Metal (macOS instance) | CUDA (Linux instance) |
|---|---|---|
| validity-aware sort + segmented reduce (4.1) | extend `sum.metal` reduce; NULL group sorts last | CUB `reduce_by_key` over `(key, valid)` |
| two-limb sum (4.2) | manual carry in the segmented reduce | CUB reduce on a 128-bit struct |
| predicate mask + masked reduce (4.6) | new kernels | new kernels (CUB `DeviceSelect::Flagged` for the mask) |
| min/max group-by | segmented reduce with min/max ops (exists for scalar; extend) | CUB `reduce_by_key` with min/max |
| DOUBLE device-side HAVING/top-k (closes the v0.6 host path) | total-order radix select on the `gb_ord`-style key mapping for f64 | exists post-#77 |
| Kahan option (4.7) | compensated segmented reduce | same |

The CPU reference backend implements every one of these first, in plain C++,
and is the oracle the two GPU backends are checked against in unit tests
before parity against native.

## 8. Spike before the rest

Two things are unknown until tried and both are answered by one spike on the
C++ API with the source submodule:

1. Plan surgery: rewrite the plain `GROUP BY k, sum(v)` shape, get `EXPLAIN`
   and parity green, confirm `ColumnBindingReplacer` handles parents.
2. Invalidation hooks: confirm DML plans reach the optimizer with their
   target table, and what transaction/snapshot information the C++ API
   exposes to the pass (§5.3).

The spike is throwaway code on a branch; its findings update this document
before piece C/D are written.

## 9. The gate (piece E)

### 9.1 Break-even sweeps → thresholds
BENCHMARK.md gains, per backend, at SF1/SF10/SF50:
- group count: 1, 4, 100, 1K, 10K, 100K, 1M, 15M (all-groups, HAVING,
  top-k forms), with and without a `WHERE` mask at 1%/10%/50%/90%
  selectivity;
- VARCHAR dictionary cost vs distinct count (4.5);
- validity-bitmap and two-limb overheads on NULL-free vs 10%-NULL columns.

Thresholds in `hybrid_planner.cpp` for the transparent path are set from
these tables, per backend, and the losing side of every sweep stays printed.
Today's row (F) (4 groups, 0.56–0.65× on Metal) stops losing because the
rewrite does not fire there.

### 9.2 Three-way parity
`groupby_parity_check.sh` runs every scenario native
(`gpudb_transparent=false`), transparent, and explicit `gpu_*`, in one
process, and all three must agree (`EXCEPT` both ways for integer/DECIMAL;
tolerance for DOUBLE). Every row of §4 is a scenario: NULL keys, NULL
payloads, `count(*)` vs `count(v)`, sums past 2^63, DECIMAL(15,2) and
DECIMAL(38,4), packed 2- and 3-column keys at the range boundary, VARCHAR
keys, each `WHERE` operator, each `HAVING` operator, ties in top-k, and the
fall-through shapes (assert the plan did **not** change and the answer still
matches). Plus the write scenarios: insert / update / delete / rollback /
appender between two identical queries.

### 9.3 `scripts/transparent_gate.sh` (rule 1)
Every rewritable shape and every §9.1 sweep point, transparent vs native,
same process, warm, min-of-N, both backends. **Any row below 1.0× fails.**
Runs at SF1 in `local_check.sh`, at SF10/SF50 before every tag on both
machines, output pasted into BENCHMARK.md unedited. The first-query case
(residency miss) is a row in this table too: it must be ≥ 1.0× because it
runs native — the gate confirms the background upload does not steal from it.

### 9.4 Community path
`make configure && make release && make test` via the C++ template on Linux
(CPU-only, as the registry builds) plus the registry-smoke workflow with a
transparent probe (`EXPLAIN` contains the GPU node after `LOAD gpudb` on the
registry binary — the CPU backend rewrites too, at CPU-appropriate
thresholds, so the community binary demonstrates the mechanism).

## 10. v0.8, designed for now

- **Transparent joins**: `t1 JOIN t2 ON t1.k = t2.k` with aggregates above,
  routed to the v0.5 resident join on the same resident sets (§5). The
  matcher gains one more shape; the residency manager and gate are already
  there.
- **Fused aggregate → join** (TPC-H Q18 end to end: filtered groups joined
  back to `orders`/`customer` on the device without materialising the group
  rows). New kernel, both backends.
- **Window functions** over resident sorted columns (`WINDOW_FUNCTIONS_DESIGN.md`).

## 11. v0.6.1 — maintenance release (before PR 1)

Packaging only. No operator changes.

**Problem.** The v0.6.0 `gpudb.linux_amd64.duckdb_extension` release asset
was built on Ubuntu 24.04 and needs `GLIBCXX_3.4.32` / `GLIBC_2.38`, so it
does not load on Ubuntu 22.04 hosts (glibc 2.35), Google Colab included. It
is also built against CUDA 13, which needs a 580-series driver; Colab's T4
runtime has a CUDA 12.x-era driver, so it would fall back to CPU there even
if it loaded. The registry's Linux binary is unaffected (CPU-only, built by
community CI); this concerns the CUDA release asset only.

**Fix (built and verified, not released).**
- PR #82: `-static-libstdc++ -static-libgcc` for the loadable on Linux.
- Build in `nvidia/cuda:12.8.1-devel-ubuntu22.04` (glibc 2.35, CUB 2.7.0),
  static CUDA runtime → 12,645,358 B, glibc floor 2.34, NEEDED only
  `libgomp`, `libc`, `ld-linux`; unit 366/366, SQL suite, parity, CLI SF1
  smoke green; SHA-256
  `ae64f1e3f2dad2a1db73d1781e0eb80c204da44504d07f37653403b5a652f9c7`. The
  asset, `REPORT.md`, `container_build.sh`, `loadtest.sh` are parked outside
  the repo on the Linux machine.
- `scripts/build.sh` honours `GPUDB_CUDA_STATIC_RUNTIME` from the environment
  (today it must be set on the CMake cache by hand); add
  `scripts/container_build_linux.sh` so the portable build is reproducible
  from the repo.
- `examples/gpudb_quickstart.ipynb` (PR #81): loads the release asset, falls
  back to a direct `cmake --target gpudb_duckdb` build with Colab's `nvcc`.
  Needs one end-to-end T4 run that reaches the benchmark cells before it
  leaves draft.

**Release steps (each on the user's explicit go).**
1. Merge #82. Rebuild the portable asset from the final commit in the
   container; re-run `loadtest.sh` on Ubuntu 22.04 and 24.04.
2. Bump `CMakeLists.txt` to 0.6.1; tag; release notes = the problem
   statement, the glibc/CUDA floors, the SHA-256s. macOS asset rebuilt at the
   tag so both carry the same footer.
3. Registry `description.yml` ref → the v0.6.1 tag (same CPU-only binary;
   the ref should point at a tag that exists).
4. Un-draft #81 after the Colab run is seen.
5. README "Requirements": Linux asset glibc floor + CUDA 12.8/driver line;
   the CUDA-13 note moves to "building from source".

**Why before PR 1.** PR 1 changes how the loadable is built. The container
build and the static-runtime flags must be in the tree first so the C++
migration inherits them instead of re-deriving them.

## 12. Milestones (no calendar — each gated on being right)

| # | Deliverable | Gate |
|---|---|---|
| 0 | v0.6.1 (§11) | user go per step |
| 1 | C++ API migration, C API removed, full v0.6 verification + audit | PR, user review |
| 2 | Spike (§8): plain `GROUP BY` rewrite + invalidation hooks confirmed; this doc updated | findings written down |
| 3 | Exactness (§4): CPU reference first, then Metal + CUDA in parallel; unit + parity | per-kernel PRs |
| 4 | Residency manager (§5): budget, eviction, three-layer invalidation, write-scenario parity | PR |
| 5 | Rewrite (§6): full §2 shape, `EXPLAIN`, settings, `gpu_last_stats` fields | PR |
| 6 | Gate (§9): sweeps, thresholds, `transparent_gate.sh` all ≥ 1.0× on both machines, BENCHMARK/KNOWN_ISSUES rows | PR |
| 7 | Audit (v0.6 shape), tag v0.7.0, registry bump | user go per step |

Milestones 3 and 4 run in parallel across the two machines; 1, 2, 5, 6 are
shared-file work and go one at a time.
