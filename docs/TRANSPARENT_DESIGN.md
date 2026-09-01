# v0.7 — transparent GPU execution (design)

Status: proposal, 2026-09-01. Companion to `GROUPBY_RESIDENT_DESIGN.md` (v0.6).
Carries the v0.6.1 maintenance release plan in §11; its portable Linux
build is the build every later release uses.

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

```python
import gpudb                      # pip package: DuckDB + the gpudb extension + the rewrite
con = gpudb.connect()             # same surface as duckdb.connect
con.sql("SELECT l_orderkey, sum(l_quantity) FROM lineitem GROUP BY l_orderkey HAVING sum(l_quantity) > 300")
# their existing SQL, unchanged, gets faster where it can; native otherwise.
```

In the DuckDB CLI the explicit `gpu_*` functions remain the interface (§3.5).

No `gpu_*` names, no `gpu_pin`, no cast to BIGINT. The v0.6 functions stay
available as the explicit form and as the rewrite targets.

## 1. What is being built

Five pieces, in dependency order:

| # | Piece | Why it exists |
|---|---|---|
| A | **Parser-level rewrite through the C API** (§3): DuckDB's own `json_serialize_sql` / `json_deserialize_sql` plus a pure `gpu_rewrite_ast` function in the extension, driven by a thin client wrapper | no C++ API, no DuckDB-version coupling, nothing that can break the base extension |
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
result, further `ORDER BY`, enclosing CTEs) is untouched; the replacement
table-function reference exposes the same column names and aliases as the
aggregate it replaces.

## 3. Where the rewrite happens: DuckDB's parser, through the C API (piece A)

### 3.1 Constraint
The loadable extension stays on the **stable C API**. No C++ extension API,
no DuckDB source submodule, no per-DuckDB-version binaries, no internal
classes. This was checked against the C API headers of v1.2 (vendored),
v1.5.5 and DuckDB's `main` branch on 2026-09-01: the C API exposes no
optimizer, planner or parser hook, and `main` adds none. The only
interception mechanism it has — the replacement scan — fires only for table
names that do not resolve, so it cannot touch a query over a real table.

### 3.2 The mechanism
DuckDB exposes its own parser as SQL: `json_serialize_sql(sql)` returns the
statement as a structured JSON tree (select list, table reference, group
expressions, `HAVING` comparison, `ORDER BY`, `LIMIT`, `WHERE`, CTEs), and
`json_deserialize_sql(json)` turns such a tree back into SQL. Both are plain
SQL functions callable from any client and from the C API. The rewrite is
therefore done on the **statement before DuckDB plans it**:

```
statement text
  → regex pre-check (GROUP BY present, table of interest)      ~0.001 ms
  → json_serialize_sql                                          DuckDB's real parser
  → gpu_rewrite_ast(json) → json    (pure function in the gpudb extension, C API scalar)
  → json_deserialize_sql                                        DuckDB's real unparser
  → the rewritten statement runs; the result is cached by statement text
```

Proven end to end on 2026-09-01 against the v0.6.0 build (Metal): a
40-line rewriter over the serialized tree turned
`SELECT l_orderkey, sum(l_quantity) AS q FROM lineitem GROUP BY l_orderkey HAVING sum(l_quantity) > 300 ORDER BY q DESC LIMIT 10`
into
`SELECT "key" AS l_orderkey, sum AS q FROM gpu_groupby_sum_resident_having('lineitem', '>', 300) ORDER BY q DESC LIMIT 10`;
native and rewritten returned identical rows (6M rows, 1.5M groups,
`EXCEPT` both ways = 0), statement time 9–10 ms native vs 3–4 ms rewritten.
The parse-and-serialize round trip costs 0.15 ms and is paid once per
distinct statement text on a table of interest; every other statement pays
the regex only.

### 3.3 Where each part lives

| Part | Lives in | Interface |
|---|---|---|
| `gpu_rewrite_ast(json VARCHAR) → VARCHAR`: matches the §2 shape on the serialized tree, consults residency (§5) and thresholds (§9.1), returns the rewritten tree or the input unchanged with a reason in `gpu_last_rewrite()` | the `gpudb` extension | C API scalar function; string in, string out; no DuckDB internals |
| Kernels, resident columns, exactness (§4), residency manager (§5) | the `gpudb` extension | as today |
| Interception: the statement has to pass through something before DuckDB executes it | a thin client wrapper: Python first (`import gpudb; con = gpudb.connect(...)`, same call surface as `duckdb.connect`), then Node, R, JDBC — each is ~100 lines that call the three SQL functions above | no DuckDB ABI at all |
| DuckDB CLI | explicit `gpu_*` functions as in v0.6 (§3.5) | — |

The rewriter never parses SQL text itself; every decision is made on the
tree DuckDB produced, and the SQL that runs is the SQL DuckDB unparsed. A
rewriter bug can only make one statement return an error (the wrapper then
runs the original text natively and records the failure) — it cannot touch
any other statement or the process.

### 3.4 What this removes from the plan
No C++ extension API migration; `third_party/duckdb_capi/` and
`duckdb_loadable.cpp` stay; the community `Makefile` stays on the C-API
template (1-minute builds, forward-compatible binaries, one release asset per
platform regardless of DuckDB version); `-static-libstdc++` (#82) stays
correct, since nothing C++ ever crosses the boundary; `gpudb-sql` and
`get_duckdb_libs.sh` are unchanged. The C++ optimizer-hook design is kept in
git history only.

### 3.5 The CLI, and the upstream route
The DuckDB CLI offers no place to intercept a statement, so CLI users keep
the explicit v0.6 functions. Two routes close that gap later, neither
blocking v0.7:

1. **Upstream a C API hook.** Propose to DuckDB a statement-rewrite callback
   on the *serialized* statement (`duckdb_add_statement_rewriter(db,
   callback, data)`: JSON in, JSON or NULL out). It exposes no internal
   class, matches machinery they already ship, and would serve any extension
   that substitutes operators. If accepted, `gpu_rewrite_ast` plugs in
   unchanged and the CLI is covered with the same C-API binary.
2. **An optional, separate C++ shim extension** that only calls
   `gpu_rewrite_ast` from an optimizer hook. Isolated so its per-version
   breakage can never affect `gpudb`. Only if (1) is refused.

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

**What becomes resident.** `gpu_rewrite_ast` sees every candidate statement.
When it sees a rewritable shape (§2) over a table whose size and estimated
group count pass the §9.1 thresholds, and the needed columns are not
resident, it records the shape and **returns the statement unchanged** (rule
1: the first query pays nothing). The **wrapper** then runs the upload
(`SELECT gpu_upload_pair(...)`) on its own second cursor/thread — the
extension stays free of threads and hidden connections, and
`residency = 'manual' | 'eager' | 'background'` is a wrapper setting. The
upload moves into the extension only if a CLI-side hook is ever upstreamed
(§3.5). The second matching query
finds them resident and is rewritten. `SET gpudb_residency = 'eager'` uploads
on first sight instead (for scripts that know their workload);
`'manual'` restores v0.6 behaviour (`gpu_upload_pair` only).

**Memory budget.** `SET gpudb_memory_budget = '8GB'` (default: 50% of device
memory on a discrete GPU, 25% of unified memory on Apple silicon). Resident
sets are evicted LRU by last use; a set that does not fit is not uploaded.
`gpu_residents()` lists what is on the device, its size, hit count, and the
plans it served.

**Invalidation — the part that has to be right.** Three mechanisms, layered:

1. **DML seen by the wrapper.** Every statement passes through the wrapper
   before DuckDB executes it; `INSERT`, `UPDATE`, `DELETE`, `TRUNCATE`,
   `COPY … FROM`, `CREATE OR REPLACE`, `ALTER`, `DROP` naming a table with a
   resident set are recognised on the serialized tree (never by text
   matching) and the set is marked invalid *before* the statement executes
   (`CALL gpu_invalidate('t')`). The Appender API goes through the same
   connection object and is intercepted the same way. If the transaction
   rolls back, the set is simply re-uploaded next time — the safe direction.
2. **Row-count check inside the rewritten statement — mandatory.** The
   wrapper only sees its own connection; writers through other connections,
   processes or clients are invisible to it. So every rewritten statement
   carries the check itself: the resident table function takes the expected
   row count and the wrapper passes `(SELECT count(*) FROM t)` as that
   argument (stats-only on a base table in DuckDB, so it costs nothing). A
   mismatch raises a typed error the wrapper catches, marks the set stale,
   and re-runs the original text natively. Because the subquery evaluates
   inside the querying transaction, this *is* the snapshot rule below,
   with the C API and no transaction introspection. It is never optional:
   without it rule 2 has a hole for any non-wrapper writer.
3. **Transaction visibility.** An upload runs in its own transaction and
   records the row count it read. A rewrite is applied only when the
   querying transaction's own `count(*)` (layer 2) equals it, so a
   connection with an older or newer snapshot that differs in size never
   sees mismatched resident data. Same-size, different-content snapshots
   are the documented gap below; the wrapper additionally sees
   `BEGIN`/`COMMIT`/`ROLLBACK` on its own connection and does not rewrite
   inside a transaction that has written to the table.

What is not caught: writes made outside the wrapper (another connection,
process or client) that change values **without changing the row count**.
KNOWN_ISSUES states it and the remedy (`CALL gpu_invalidate('t')`, or route
writes through the wrapper). Every other outside write is caught by layer 2.

**Concurrency — prerequisite of the background path.** Today
`gpu_resident.cpp` holds one global mutex across an entire resident operator
call (`gpu_groupby_extension.cpp`, `gpu_join_extension.cpp`), and the shared
`HybridAggregator`'s backends keep non-thread-safe caches
(`ensure_join_cache` / `ensure_sort_cache`). Fine with one connection and
serialized calls; with a background upload on a second connection an SF50
upload (its key sort is seconds) would hold the lock while the hit query
blocks on it — the miss query never touches the lock, so rule 1 survives
for the miss, but the hit would lose. Before any background upload lands:
a short lock for registry lookup only; per-resident-set state
`{uploading, ready, completion event}`; upload work on its own stream
outside the global lock; a per-column (not global) mutex around the
sort-cache build. This is shared-extension plus CUDA-backend work carried as
one `feat/core-*` PR from the Linux instance; the Metal backend honours the
same per-set lock. Upload and query work run on separate device streams
(CUDA streams; Metal command queues) with a completion event recorded at
the end of the upload; a rewrite may consume a resident set only after that
event reports complete. Today everything is on one stream. The
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

`gpu_rewrite_ast` over the serialized statement:

1. Walk the tree for the §2 shape; on the first non-matching node, return
   the input unchanged. Unmatched trees cost under 0.05 ms in the function
   (unit test over 10,000 statements); statements that never reach it cost
   the wrapper's regex only.
2. Check residency (§5) and thresholds (§9.1); on a miss, record the shape
   and return unchanged.
3. Build the replacement tree: the base-table reference becomes a
   `TABLE_FUNCTION` reference to the resident operator with constant
   arguments `{resident set, predicate program, aggregates, filter, top-k}`;
   select-list expressions are remapped to the function's output columns
   with the user's aliases preserved; `GROUP BY`/`HAVING`/pushed `ORDER BY
   … LIMIT` are removed; everything else in the tree (CTEs, projections,
   further `ORDER BY`, joins to the result) is left as is.
4. `gpu_last_rewrite()` reports, for the last statement seen, whether it
   was rewritten and if not why (shape / not resident / threshold / stale /
   budget), which is also what the gate parses. `EXPLAIN` of the rewritten
   statement shows the `gpu_*` table function in the plan.
5. `SET gpudb_transparent = false` makes the wrapper pass every statement
   through untouched (a config option registered through the C API).

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

## 8. Spike

The plan-surgery spike in the previous revision is no longer needed: the
parser-level rewrite was run end to end on 2026-09-01 (§3.2). What remains
to confirm before piece C/D is written, on the same throwaway branch:

1. The serialized tree for every §2 construct (`WHERE` conjunctions,
   multi-column keys, `count(*)`, `min`/`max`/`avg`, DECIMAL literals,
   CTEs wrapping the shape) and that `json_deserialize_sql` reproduces each
   rewritten form.
2. The wrapper's statement cache and the invalidation path (§5.1) with
   multi-statement strings, prepared statements with parameters, and
   transactions.

Findings update this document before implementation.

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
- wrapper overhead: a plain `SELECT` (no `GROUP BY`) through `duckdb.connect`
  vs through `gpudb.connect` with `gpudb_transparent=false` vs transparent —
  rule 1 has to hold for statements that are never rewritten; and the
    first-sighting cost of a `GROUP BY` statement (regex + parse round trip +
  `gpu_rewrite_ast`, 0.15 ms measured) as its own row, measured on both
  machines against the SF10 hit case and against a fall-through shape at
  small size — the statement cache must make a fall-through statement cost
  zero extra on its second execution, and the embedded `count(*)` check must
  measure as zero on a base table.

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
Unchanged C-API template path (`make configure && make release && make
test`) on Linux plus the registry-smoke workflow with a rewrite probe:
`SELECT gpu_rewrite_ast(json_serialize_sql(...))` on the registry binary
returns the table-function form, and `json_deserialize_sql` of it executes
— the CPU backend rewrites too, at CPU-appropriate thresholds, so the
community binary demonstrates the mechanism without a wrapper.

## 10. v0.8, designed for now

- **Transparent joins**: `t1 JOIN t2 ON t1.k = t2.k` with aggregates above,
  routed to the v0.5 resident join on the same resident sets (§5). The
  matcher gains one more shape; the residency manager and gate are already
  there.
- **Fused aggregate → join** (TPC-H Q18 end to end: filtered groups joined
  back to `orders`/`customer` on the device without materialising the group
  rows). New kernel, both backends.
- **Window functions** over resident sorted columns (`WINDOW_FUNCTIONS_DESIGN.md`).

## 11. v0.6.1 — maintenance release (first)

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

**Why first.** The container build, the static CUDA runtime and
`-static-libstdc++` (#82) are the portable Linux build for every release
from here on — the loadable stays on the C API (§3.4), so nothing C++ ever
crosses into DuckDB and the static runtime remains correct.

## 12. Milestones (no calendar — each gated on being right)

| # | Deliverable | Gate |
|---|---|---|
| 0 | v0.6.1 (§11) | user go per step |
| 1 | Spike remainder (§8): tree shapes for every §2 construct, wrapper cache + invalidation with transactions/prepared statements; this doc updated | findings written down |
| 2 | `gpu_rewrite_ast` for the plain `GROUP BY` shape + the Python wrapper, three-way parity on the v0.6 operators as they are | PR |
| 3 | Exactness (§4): CPU reference first, then Metal + CUDA in parallel; unit + parity | per-kernel PRs |
| 4 | Residency manager (§5): budget, eviction, three-layer invalidation, write-scenario parity | PR |
| 5 | Full §2 shape in the rewriter, `gpu_last_rewrite`, settings, `gpu_last_stats` fields; Node/R/JDBC wrappers | PR |
| 6 | Gate (§9): sweeps, thresholds, `transparent_gate.sh` all ≥ 1.0× on both machines, BENCHMARK/KNOWN_ISSUES rows | PR; gate must be all ≥ 1.0× |
| 7 | Audit (v0.6 shape), tag v0.7.0, registry bump, `pip` package release | user go per step |
| — | Upstream proposal for a C API statement-rewrite hook (§3.5) | drafted after milestone 2 proves the interface; user sign-off before anything is posted |

Milestones 3 and 4 run in parallel across the two machines; 1, 2, 5, 6 are
shared-file work and go one at a time. The loadable extension's build,
ABI and release shape do not change at any milestone.
