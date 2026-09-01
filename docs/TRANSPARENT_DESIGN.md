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
- **`-static-libstdc++` is reverted in this PR.** It is safe for the C-API
  loadable (nothing C++ crosses the boundary) and is what v0.6.1 ships
  (§11). It is *not* safe for a C++-API loadable: `unique_ptr<LogicalOperator>`,
  `std::string`, exceptions and RTTI cross into the host's libstdc++, and two
  copies in one process is a known source of typeinfo/exception mismatches.
  The portable glibc floor for the C++ build comes from building in an
  old-glibc container with dynamic libstdc++ — exactly what the community CI
  does.
- **Link options are re-applied to the template's target.** The C++
  template's `build_loadable_extension` creates its own target; our
  `--exclude-libs,ALL`, static CUDA runtime and the CUDA architecture list
  must be attached to it, and the Linux instance runs
  `make configure && make release && make test` in the community container
  shape before PR 1 is opened.
- **Which DuckDB versions get a CUDA release asset.** One container build and
  one asset per version; DuckDB refuses a C++ extension whose footer version
  does not match. v0.7 ships CUDA assets for the version the registry serves
  and for the current `pip install duckdb` version when it differs (Colab's
  pip lags releases); the release checklist lists both, and the quickstart
  notebook selects the asset by `duckdb.__version__`.
- **`get_duckdb_libs.sh` pins the submodule's tag.** `gpudb-sql` can only
  `LOAD` a loadable built against the same DuckDB version it embeds.

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

There is no spare int64 value to make NULL "sort last", so the NULL-key rows
are partitioned out at upload (valid-key prefix, NULL-key suffix; one
`DevicePartition` on CUDA, one compaction pass on Metal) and only the prefix
is sorted. The suffix is one group at the end — `count(*)` and the sums of
its payloads — and is naturally excluded from join build/probe, which is what
SQL join semantics need (NULL never matches). One extra pass at upload, zero
cost at query time. NULL payloads are handled by identity injection in the
reduce (0 for sum, ±inf / INT64 limits for min/max) and the per-group tuple
becomes `(sum, count_v, count_star)`.

### 4.2 128-bit sums
`sum(BIGINT)` in DuckDB is HUGEINT and never overflows. The device
accumulates in two 64-bit limbs. On CUDA `nvcc` supports `__int128` in device code (11.5+, 64-bit
targets), so it is `reduce_by_key` with an `__int128` accumulator through a
transform iterator, no hand carry; Metal does the carry by hand in the
segmented reduce. Output column is HUGEINT. The extra limb costs 16 B per
element in the tree and per group out — expected to be a small measurable hit
at SF50 — so it is skipped when the pin's static bound `rows × max|v| < 2^63`
proves it unnecessary, which holds on every benchmark table. The bound needs
the payload's min/max at upload: one `DeviceReduce`, cheap, and the same
statistics pass §4.4 uses.

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
`Π (range_i + 2) < 2^63` (one extra slot per column for NULL, since DuckDB
groups NULL per column and there are up to 2^n NULL combinations) the tuple
is packed as a mixed-radix integer
`((a−min_a)·R_b + (b−min_b))·R_c + (c−min_c)`, sorted and reduced as one
64-bit key, and unpacked on the way out. Mixed radix preserves lexicographic
order, so the sorted-by-key property holds for `ORDER BY a, b, c`. DuckDB's
base-table min/max are conservative after updates (widened, never wrong),
which is the safe direction. Exact, no new kernels. When the product does not
fit, the shape is not resident (native).

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
selection mask. Two execution variants, both built and both in the gate:
(a) a masked reduce that reads one flag per element (cost ∝ rows in), and
(b) compaction of the sorted permutation (`DeviceSelect::Flagged` on CUDA,
the v0.6 block compaction on Metal) followed by the ordinary reduce over
survivors (cost ∝ rows out plus one select pass). At 1% selectivity (b) wins
by a wide margin, at 90% (a) wins; the §9.1 selectivity sweep fixes the
crossover per backend and it becomes a threshold. A predicate on the **key**
column (`WHERE k BETWEEN …`) is a binary search on the sorted key cache — a
contiguous range at zero per-row cost — and is handled before either variant.

### 4.7 DOUBLE summation
Native DuckDB's `sum(DOUBLE)` is itself order-dependent across thread counts.
The device result is "equal up to reassociation"; parity uses the tolerance
`groupby_parity_check.sh` already applies. This is the one place where
"exact" means "as exact as native is with itself", and it is stated in the
docs. Kahan summation is sequential by construction and cannot be a
`reduce_by_key` operator; the compensated option that fits a tree reduction
is double-double (TwoSum error-free transformation, 16 B state, ~106-bit
effective), implemented behind a setting so users who need the tighter bound
can turn it on and see the cost. Worth stating plainly: the device paths are
deterministic run to run (fixed reduction tree, no atomics on either
backend), unlike native DuckDB across thread counts — parity keeps its
tolerance against native, but bit-reproducibility is a property gpudb can
claim.

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
   else that bypasses the planner and adds or removes rows): the row count
   visible to the *querying transaction* must equal the count at upload —
   which makes this and (3) one check in practice.
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

**Streams and cost of an upload.** Upload and query work run on separate
device streams (CUDA streams; Metal command queues) with a completion event
recorded at the end of the upload; a rewrite may consume a resident set only
after that event reports complete. Today everything is on one stream. The
expensive part of an upload is not the host-to-device copy but the key sort
(`ensure_join_cache`; SF100 keys are seconds), and it competes with the
concurrent native query for host memory bandwidth and PCIe — so the
first-query row in the gate (§9.3) is measured with that sort running, not
after it. The `GPUDB_UPLOAD_POOL_MAX_MB` knob becomes `gpudb_memory_budget`
rather than a second setting.

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
| NULL-key partition at upload + identity-injected reduce (4.1) | compaction pass; extend `sum.metal` reduce tuple to `(sum, count_v, count_star)` | `DevicePartition` on validity; transform iterator injects identities |
| two-limb sum (4.2) | manual carry in the segmented reduce | `reduce_by_key` with an `__int128` accumulator |
| predicate mask, variants (a) masked reduce and (b) compact-then-reduce (4.6) | new kernels; (b) reuses the v0.6 block compaction | new kernels; (b) uses `DeviceSelect::Flagged` |
| min/max group-by | fused into the per-group tuple `(sum, count, min, max)`: one pass for all aggregates of a query | same, `reduce_by_key` on the tuple |
| DOUBLE device-side HAVING/top-k (closes the v0.6 host path) | total-order radix select on the `gb_ord`-style key mapping for f64 | exists post-#77 |
| double-double option (4.7) | compensated segmented reduce | same |

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
- validity-bitmap and two-limb overheads on NULL-free vs 10%-NULL columns,
  and the two-limb path forced on;
- mask variant (a) vs (b) across the selectivity sweep;
- optimizer-pass overhead: a plain `SELECT` (no `GROUP BY`) native vs
  `LOAD gpudb` with `gpudb_transparent=false` vs transparent — rule 1 has
  to hold for queries the pass never rewrites, and the §6.1 bound includes
  the residency lookup and the DML scan.

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
same process, warm, min-of-N with N ≥ 5 and the GPU clock printed beside
each row (the 4090's boost clock is bimodal, 1665–2400 MHz, and a 1.0× row
can flip between runs otherwise), both backends. **Any row below 1.0× fails.**
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
does not load on Ubuntu 22.04 hosts (glibc 2.35), Google Colab included. It is also built against CUDA 13, which needs an R580+ driver; CUDA 12.x
minor-version compatibility means a 12.8-built asset gets the GPU on any
R525+ driver, which is what Colab's T4 runtime has — that is the reason for
building with 12.8, and it is the line the README "Requirements" row should
carry. The registry's Linux binary is unaffected (CPU-only, built by
community CI); this concerns the CUDA release asset only.

**Fix (built and verified, not released).**
- PR #82: `-static-libstdc++ -static-libgcc` for the loadable on Linux.
- Build in `nvidia/cuda:12.8.1-devel-ubuntu22.04` (glibc 2.35, CUB 2.7.0),
  static CUDA runtime → 12,645,358 B, glibc floor 2.34 (plus `OMP_1.0` from
  libgomp; no CXXABI dependency), NEEDED only `libgomp.so.1`, `libc.so.6`,
  `ld-linux-x86-64.so.2`. Verified: unit 366/366, SQL 29/0, GROUP BY parity
  12/12, `LOAD` on ubuntu:22.04 (CPU) and on the 24.04 host, v1.5.5 CLI SF1
  smoke equal to native with `backend=CUDA` on the 4090. This is a
  **v0.6.0-based verification build** (footer `C_STRUCT, v0.6.0, v1.2.0,
  linux_amd64`; SHA-256
  `ae64f1e3f2dad2a1db73d1781e0eb80c204da44504d07f37653403b5a652f9c7`); the
  release asset is rebuilt at the v0.6.1 tag and will have a different
  SHA. The asset, `REPORT.md`, `container_build.sh`, `loadtest.sh` are
  parked outside the repo on the Linux machine.
- `scripts/build.sh` honours `GPUDB_CUDA_STATIC_RUNTIME` from the environment
  (today it must be set on the CMake cache by hand; the container script
  already does) and `scripts/container_build_linux.sh` is added so the
  portable build is reproducible from the repo — one PR from the Linux
  instance.
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
build and the static CUDA runtime must be in the tree first so the C++
migration inherits them. `-static-libstdc++` (#82) is C-API-era only and is
reverted in PR 1 (§3); v0.6.1 is the last release that carries it.

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
