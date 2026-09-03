# v0.7 — transparent GPU execution (design)

Status: proposal, 2026-09-01; revised 2026-09-03 after an adversarial review
(§13). Companion to `GROUPBY_RESIDENT_DESIGN.md` (v0.6). Carries the v0.6.1
maintenance release plan in §11; its portable Linux build is the build every
later release uses.

## 0. Two rules

**Rule 1 — never slower.** With gpudb loaded, a query is never slower than the
same query without it. Not on average: per query, per shape, per size, on
both backends. This is enforced by a benchmark gate (§9.3) that times every
shape the rewrite can produce against native on every release; a single row
under 1.0× blocks the release.

The one bounded exception, decided 2026-09-03: the wrapper cannot know
native's time before running a statement, so the **first sighting** of a new
statement template over a table of interest pays the parse round trip once
(0.12 ms measured, §3.2). The wrapper only parses statements that name a
table above the §9.1 floor, where native takes at least 50× the round trip,
so the one-time loss is under 2% of a single query; every later execution of
that template pays nothing. Tables below the floor are never parsed and are
therefore never slower. The gate carries the first-sighting row (§9.1) and
the bound is stated in the README.

**Rule 2 — never different.** A rewritten query returns exactly what native
returns: same rows, same order where native guarantees one, same column
names, same column types. Where the v0.6 resident operators and native
DuckDB differ today (NULLs, overflow, DECIMAL, output types), v0.7 makes the
device **exact** rather than falling through around the difference. Falling
through is allowed for shapes we do not handle yet; it is not allowed as a
way to avoid handling a shape correctly. Where exactness against native is
not definable (DOUBLE sums, §4.7) the shape is not rewritten.

The one documented gap, decided 2026-09-03: a writer in **another process**
that changes values without changing the row count is invisible to the
wrapper and to the in-statement row-count check (§5.4). DuckDB's
single-writer rule already excludes this when the wrapper's process holds the
write lock, so the requirement is: the wrapper opens the database read-write
(the default); a read-only wrapper beside an external writer is unsupported
and stated in KNOWN_ISSUES.

The end state a user sees:

```python
import gpudb                      # pip package: DuckDB + the gpudb extension + the rewrite
con = gpudb.connect()             # same surface as duckdb.connect
con.sql("SELECT l_orderkey, sum(l_quantity) FROM lineitem GROUP BY l_orderkey HAVING sum(l_quantity) > 300")
# their existing SQL, unchanged, gets faster where it can; native otherwise.
```

In the DuckDB CLI the explicit `gpu_*` functions remain the interface (§3.5).
No new function names to learn on the transparent path, no cast to BIGINT.
The v0.6 functions stay available as the explicit form and as the rewrite
targets.

## 1. What is being built

Five pieces, in dependency order:

| # | Piece | Why it exists |
|---|---|---|
| A | **Parser-level rewrite through the C API** (§3): DuckDB's own `json_serialize_sql` / `json_deserialize_sql` plus a pure `gpu_rewrite_ast` function in the extension, driven by a thin client wrapper | no C++ API, no DuckDB-version coupling, nothing that can break the base extension |
| B | **Exactness on the device** (§4): NULL-aware resident columns, 128-bit sums, DECIMAL as scaled integers, packed multi-column keys, native output types | rule 2 without fall-through |
| C | **Automatic residency** (§5): the wrapper decides what to keep on the device, uploads in the background, and invalidates on writes; the extension keys every set by table identity | rule 1 even on the first query; rule 2 across connections and catalogs |
| D | **The rewrite** (§6): `GROUP BY` with `WHERE`, `HAVING`, `ORDER BY … LIMIT`, over resident columns, with a device-side predicate mask | the user-visible feature |
| E | **The gate** (§9): break-even sweeps, thresholds per backend, `transparent_gate.sh`, three-way parity | rules 1 and 2 mechanically |

Joins on the transparent path are v0.8 (§10) and are designed for here so
nothing in A–E has to be redone for them.

## 2. Plan shapes covered

```sql
SELECT k1 [, k2, k3], sum(v) | count(v) | count(*) | min(v) | max(v) | avg(v)
FROM t
[WHERE <predicate on columns of t>]
GROUP BY k1 [, k2, k3]
[HAVING <agg> {> >= < <= = <>} <constant> [AND …]]
[ORDER BY <agg> | k [ASC|DESC] [NULLS FIRST|LAST]]
[LIMIT k]
```

- Keys: signed integer family, DATE, TIMESTAMP (all int64-representable), up
  to three of them packed into one 64-bit key when their ranges allow (§4.4),
  and VARCHAR through a dictionary when the column is low-cardinality enough
  that a dictionary is cheap and the collation is binary (§4.5). UBIGINT and
  HUGEINT keys are not int64-orderable and run native.
- Payloads: integer family, DECIMAL (§4.3). DOUBLE/FLOAT payloads are not
  rewritten (§4.7); they stay available through the explicit `gpu_*` calls.
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
  the expensive comparison). Thresholds at a finer scale than the payload are
  rescaled exactly (§6 step 3).
- `ORDER BY agg LIMIT k`: device top-k, pushed only when the direction and
  NULL order are explicit in the tree or the connection's `default_order` /
  `default_null_order` have been read. The `ORDER BY` node itself is **never
  removed** from the tree: sorting an already-sorted group list is trivial
  and keeping it is what makes the direction, NULL order, VARCHAR ordering
  and session defaults native's problem, not ours.

**Rejected by field, not by node.** The matcher reads every field of every
node on the path and declines the statement when any of these is present:
`GROUP BY ALL`, ordinal group references, `ROLLUP`/`CUBE`/`GROUPING SETS`,
`FILTER (WHERE …)` on an aggregate, `DISTINCT` inside an aggregate, `ORDER BY`
inside an aggregate, `SAMPLE`/`USING SAMPLE`, `AT` clauses, window
functions, `QUALIFY`, set operations, `PARAMETER` nodes, and any `cte_map`
at any scope that defines the table's name (§5.1). The unit test for the
matcher enumerates each of these as a serialized tree that must come back
unchanged.

Everything above the matched subtree (projection, aliases, joins to the
result, further `ORDER BY`, enclosing CTEs) is untouched; the replacement
table-function reference exposes the same column names and types as the
aggregate it replaces (§6 step 3).

## 3. Where the rewrite happens: DuckDB's parser, through the C API (piece A)

### 3.1 Constraint
The loadable extension stays on the **stable C API** at the v1.2.0 API
struct it targets today. No C++ extension API, no DuckDB source submodule,
no per-DuckDB-version binaries, no internal classes, no raise of the API
floor. This was checked against the C API headers of v1.2 (vendored), v1.5.5
and DuckDB's `main` branch on 2026-09-01: the C API exposes no optimizer,
planner or parser hook, and `main` adds none. The only interception mechanism
it has — the replacement scan — is observed to fire only for table names
that do not resolve, so it cannot touch a query over a real table.

### 3.2 The mechanism
DuckDB exposes its own parser as SQL: `json_serialize_sql(sql)` returns the
statement as a structured JSON tree (select list, table reference, group
expressions, `HAVING` comparison, `ORDER BY`, `LIMIT`, `WHERE`, CTEs), and
`json_deserialize_sql(json)` turns such a tree back into SQL. Both are plain
SQL functions callable from any client and from the C API. The rewrite is
therefore done on the **statement before DuckDB plans it**:

```
statement text
  → statement classification (DuckDB's own splitter, §5.2)          every statement
  → whitelist check: names a table above the §9.1 floor?              ~0.001 ms, regex on the text
  → template cache lookup (literal-normalised, §3.3)                  hit: 0 extra
  → one statement:
      SELECT json_deserialize_sql(gpu_rewrite_ast(json_serialize_sql(?), <context>))
                                                                       0.124 ms measured
  → the returned statement runs (rewritten, or the input unchanged)
```

Proven end to end on 2026-09-01 against the v0.6.0 build (Metal): a
40-line rewriter over the serialized tree turned
`SELECT l_orderkey, sum(l_quantity) AS q FROM lineitem GROUP BY l_orderkey HAVING sum(l_quantity) > 300 ORDER BY q DESC LIMIT 10`
into
`SELECT "key" AS l_orderkey, sum AS q FROM gpu_groupby_sum_resident_having('lineitem', '>', 300) ORDER BY q DESC LIMIT 10`;
native and rewritten returned identical rows (6M rows, 1.5M groups,
`EXCEPT` both ways = 0), statement time 9–10 ms native vs 3–4 ms rewritten.
The three functions as three round trips cost 0.238 ms; nested in one
statement 0.124 ms (measured 2026-09-02, Python `duckdb` 1.4.5). That is the
first-sighting cost in rule 1. Statements longer than 16 KB are not parsed
(a 0.6 MB `IN` list produces 16 MB of JSON and a 140 ms round trip); they
run native.

The tree is **unbound**: `FROM t` serializes as
`{"table_name":"t","schema_name":"","catalog_name":""}` whether `t` is a
base table, a same-named CTE, a temp table, a view, a registered DataFrame,
or a table in another attached database. Name resolution is therefore the
wrapper's job (§5.1), done once per template and cached with the template.

### 3.3 Where each part lives

| Part | Lives in | Interface |
|---|---|---|
| `gpu_rewrite_ast(json VARCHAR, context VARCHAR) → VARCHAR`: matches the §2 shape on the serialized tree and returns the rewritten tree or the input unchanged. **Pure**: no shared state, no residency lookup, no "last statement" recording (DuckDB may evaluate a scalar in parallel per vector). Everything it needs — the resolved table identity, the resident-set key, its column list, the effective backend, the connection's `default_order`/`default_null_order`/`default_collation`, thresholds — arrives in `context` as a JSON string built by the wrapper | the `gpudb` extension | C API scalar function; strings in, string out; no DuckDB internals |
| Kernels, resident columns, exactness (§4), the resident registry keyed by identity (§5.3) | the `gpudb` extension | as today, plus one `prepare()` entry on `ResidentColumn` (§5.5) |
| Interception, classification, name resolution, the template cache, residency decisions, upload scheduling, settings, fallback, `last_rewrite()` | a client wrapper: Python first (`import gpudb; con = gpudb.connect(...)`, same call surface as `duckdb.connect`), then Node, R, JDBC | no DuckDB ABI at all; plain SQL through the client's own API |
| DuckDB CLI | explicit `gpu_*` functions as in v0.6 (§3.5) | — |

The wrapper is more than a hundred lines. It has to cover `execute`,
`executemany`, `sql`, cursors, the default connection, and the relational
API (`con.table('t').aggregate(...)`, which never produces SQL text and
therefore simply runs native — no rewrite, no risk). Its statement cache is
keyed on a **literal-normalised template** (constants replaced by
placeholders) plus the resolved table identity plus the connection's
relevant settings; literals that change the rewrite (`HAVING` threshold,
`LIMIT`, `IN` list) are re-substituted per statement from the current text,
never taken from the cached entry.

The rewriter never parses SQL text itself; every decision is made on the
tree DuckDB produced, and the SQL that runs is the SQL DuckDB unparsed.
Failure containment (see §5.4 for the transaction rule): the rewritten
statement raises a **typed** error for a stale set (the `gpu_assert_rows`
text); on that error, and only on that error, the wrapper marks the set
stale and re-runs the original text natively. Any other error from the
rewritten statement surfaces to the user unchanged — a silent fallback on an
arbitrary error would hide a rule 2 bug.

### 3.4 What this removes from the plan
No C++ extension API migration; `third_party/duckdb_capi/` and
`duckdb_loadable.cpp` stay; the community `Makefile` stays on the C-API
template (1-minute builds, forward-compatible binaries, one release asset per
platform regardless of DuckDB version); `-static-libstdc++` (#82) stays
correct, since nothing C++ ever crosses the boundary; `gpudb-sql` and
`get_duckdb_libs.sh` are unchanged. No `SET gpudb_*` options: the v1.2.0
API struct has no config-option registration, so settings are wrapper-side
(§6 step 5). The C++ optimizer-hook design is kept in git history only.

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
   breakage can never affect `gpudb`. Only if (1) is refused. Note from the
   review: in v1.5.2 the `OperatorExtension` bind path runs only when normal
   binding fails, so even the shim would need the optimizer hook, not the
   bind hook.

## 4. Exactness on the device (piece B)

Today's resident columns equal
`… WHERE k IS NOT NULL AND v IS NOT NULL` with 64-bit sums and BIGINT-only
payloads and outputs. Each gap below is closed on the device so the rewrite
never has to decline for a semantic reason.

### 4.1 NULL-aware resident columns
A resident column gains an optional validity bitmap (1 bit/row, uploaded
alongside; omitted when the source has no NULLs, which the set records). The
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
reduce (0 for sum, ±limits for min/max) and the per-group tuple becomes
`(sum, count_v, count_star, min, max)`. A group whose payload is all NULL
returns **NULL** for `sum`/`min`/`max`/`avg` and 0 for `count(v)`, as
native does — `count_v = 0` selects NULL on output, never the identity.

### 4.2 128-bit sums, native output types
`sum(BIGINT)` in DuckDB is HUGEINT and never overflows. The device
accumulates in two 64-bit limbs. On CUDA `nvcc` supports `__int128` in
device code (11.5+, 64-bit targets), so it is `reduce_by_key` with an
`__int128` accumulator through a transform iterator, no hand carry; Metal
does the carry by hand in the segmented reduce. The table function emits
the limbs directly as a HUGEINT vector (`duckdb_hugeint` in the C API), no
host conversion. The extra limb costs 16 B per element in the tree and per
group out — expected to be a small measurable hit at SF50 — so the
accumulation is skipped when the set's static bound `rows × max|v| < 2^63`
proves it unnecessary (the output column is still HUGEINT). The bound needs
the payload's min/max at upload: one `DeviceReduce`, cheap, and the same
statistics pass §4.4 uses.

Output types follow native exactly: `sum` of any integer type is HUGEINT,
`sum` of `DECIMAL(p,s)` is `DECIMAL(38,s)`, `count`/`count(*)` are BIGINT,
`min`/`max` keep the input type, `avg` is DOUBLE for integers and DECIMAL
for DECIMAL inputs. The parity harness compares `typeof()` of every output
column (§9.2).

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
order, so the sorted-by-key property holds for `ORDER BY a, b, c`. A
key-range predicate is a contiguous range only on the **first** packed
component; predicates on later components go through the mask (§4.6).
DuckDB's base-table min/max are conservative after updates (widened, never
wrong), which is the safe direction. Exact, no new kernels. When the product
does not fit, the shape is not resident (native).

### 4.5 VARCHAR keys via dictionary
A VARCHAR key is uploaded as a dense int32 id against a host-side dictionary
built at upload (one pass, hash map). Results map ids back to strings. The
dictionary is byte-wise, so it is only built when the column's collation
(from `duckdb_columns()`) and the connection's `default_collation` are
binary; `NOCASE`, `NOACCENT` and any ICU collation merge groups natively
that a byte-wise dictionary would not, and those shapes run native. The
dictionary is only built when DuckDB's distinct-count estimate is below a
threshold measured in §9.1 (a 15M-distinct-string dictionary is not a win);
above it the shape runs native. Ids are hash-ordered, not string-ordered,
which is one more reason the `ORDER BY` node is kept (§2).

### 4.6 Device-side predicate mask (`WHERE`)
Most real `GROUP BY` queries filter first. The resident set for a table
includes the columns that appear in `WHERE` clauses of matched shapes (§5
decides which); at query time the conjunction of comparisons is compiled to a
small predicate program (column, op, constant) evaluated on the device into a
selection mask. Two execution variants, both built and both in the gate:
(a) a masked reduce that reads one flag per element (cost ∝ rows in), and
(b) compaction of the sorted permutation (`DeviceSelect::Flagged` on CUDA,
the v0.6 block compaction on Metal) followed by the ordinary reduce over
survivors (cost ∝ rows out plus one select pass). Variant (a) must drop
groups whose every row was masked out — native emits no row for an empty
group — which the `count_star = 0` test on the tuple gives for free. At 1%
selectivity (b) wins by a wide margin, at 90% (a) wins; the §9.1
selectivity sweep fixes the crossover per backend and it becomes a
threshold. A predicate on the **key** column (`WHERE k BETWEEN …`) is a
binary search on the sorted key cache — a contiguous range at zero per-row
cost — and is handled before either variant.

Comparisons on DOUBLE predicate columns use DuckDB's total order, not IEEE
compare: NaN is greatest, `NaN = NaN`, `-0.0 = 0.0`. The mask kernel maps
f64 to an order-preserving u64 (the mapping the f64 top-k already uses) and
compares integers. `'nan'::DOUBLE > 5` is true natively and must be true on
the device; the parity scenarios include NaN and ±inf rows under every
operator.

### 4.7 DOUBLE summation — not on the transparent path
Native `sum(DOUBLE)` is order-dependent: the same table gives different
last-ulp results across thread counts and insert orders, and a 1-ulp
difference flips a `HAVING` or a top-k tie, changing the row set. "Same as
native" is therefore not definable for DOUBLE sums, and v0.7 does **not**
rewrite statements whose aggregates have DOUBLE or FLOAT inputs (`sum`,
`avg`; `min`/`max`/`count` on DOUBLE are exact and are rewritten). Those
statements run native at native speed. The explicit `gpu_*` functions keep
their DOUBLE paths with the parity tolerance `groupby_parity_check.sh`
already applies, and the device result there is deterministic run to run
(fixed reduction tree, no atomics on either backend), which the docs may
state. The double-double option stays a setting on the explicit path.
Revisited in v0.8 only if a definition of exactness against native exists.

## 5. Automatic residency (piece C)

No pin call. The **wrapper** keeps a residency manager per connection
family (one per DuckDB database it opened); the extension keeps the
registry keyed by identity (§5.3) and knows nothing about workloads.

### 5.1 Name resolution — before anything else
The tree is unbound (§3.2), so before a template is considered the wrapper
resolves the table reference the way the binder will, on the same
connection:

1. If any `cte_map` at any scope in the tree defines the name, the
   statement runs native.
2. An unqualified name is resolved against `current_schemas(true)` (temp
   schema first) through `duckdb_tables()` and `duckdb_views()`; the first
   hit wins. A view, a registered DataFrame/Arrow object (a view in DuckDB),
   or no hit at all: native.
3. A qualified name is checked the same way against its catalog and schema.
4. The result is `(catalog, schema, table, table oid, column names)` — the
   **identity** — cached with the template together with the search path
   and catalog list that produced it.

Any statement the classifier (§5.2) marks as catalog-changing — `CREATE`
(including `TEMP`, `OR REPLACE`, `VIEW`), `DROP`, `ALTER`, `ATTACH`,
`DETACH`, `USE`, `SET search_path`, `SET default_*`, `register()` /
`unregister()` in the client — clears every cached resolution on that
connection. The rewritten statement always names the table fully qualified.

### 5.2 Statement classification — DuckDB's own splitter
`json_serialize_sql` serializes only `SELECT` statements; every other kind
returns an error, so DML cannot be recognised "on the tree". The wrapper
instead uses the classifier DuckDB ships in every client: in Python
`con.extract_statements(text)` returns one object per statement with its
`type`; in the C API `duckdb_extract_statements` +
`duckdb_prepared_statement_type`. Every incoming string is split; each
member is handled by type:

- `SELECT`: the rewrite path (§3.2).
- `EXPLAIN [ANALYZE] <select>`: the inner statement is rewritten and
  re-prefixed, so users can see the resident operator in the plan.
- `BEGIN`/`COMMIT`/`ROLLBACK`: transaction tracking (§5.4).
- **Anything else** — `INSERT`, `UPDATE`, `DELETE`, `MERGE`, `COPY`,
  `TRUNCATE`, `CREATE`, `ALTER`, `DROP`, `ATTACH`, `DETACH`, `USE`, `SET`,
  `CALL`, `PREPARE`/`EXECUTE`, `PRAGMA`, `LOAD`, `INSTALL`, `VACUUM`,
  `CHECKPOINT`, and any type added by a later DuckDB — invalidates **every**
  resident set on that database before the statement runs, and clears the
  resolution cache if it is catalog-changing. Conservative on purpose: a
  miss runs native, so over-invalidation costs at most a re-upload, never an
  answer.

The client's non-SQL write paths — `Appender`, `con.append`, `register` —
are wrapped by the same invalidation as an optimisation, but correctness does
not depend on the wrapper seeing them: the identity key (§5.1), the epoch
(§5.5) and the in-statement check (§5.4) stand on their own.

### 5.3 The registry, keyed by identity
`Globals::registry` in `gpu_resident.cpp` is today a process-global map
keyed by the free-form upload name, and re-upload replaces. Two databases
open in one process, or an `ATTACH`, with a same-count table of the same
name share one set, and any SQL user can replace a set with foreign data
through `gpu_upload_pair`. The registry becomes keyed by:

`(database handle, catalog, schema, table oid, column names, row count at upload, epoch at upload, origin)`

- The database handle comes from `duckdb_extension_info.get_database` at
  load and is attached as the function's extra_info, so one extension load
  serving two databases keeps two disjoint registries.
- The row count in the key means a set uploaded from a different-size
  snapshot is **unreachable** by construction, not merely detected.
- `origin ∈ {explicit, managed}`: sets created by a user's `gpu_upload*`
  call are `explicit` and are used only by the explicit functions; the
  rewriter consumes only `managed` sets, which the wrapper creates through
  the same upload functions with the identity tag as the name. A user
  cannot poison the transparent path from SQL.
- Today the upload functions do not know which table their input came from;
  the wrapper passes the identity as the set name. The registry stops
  treating the name as free-form and parses the tag.

`gpu_residents()` lists every set with its identity, origin, size, state
(§5.5), hit count and epoch. `gpu_last_stats()` is one process-global
string today and is meaningless with two connections; it becomes per-set
(a column of `gpu_residents()`) and the process-level string is documented
as "the last call in this process".

### 5.4 Invalidation — the part that has to be right
Four mechanisms, layered; every one is independent of the others.

1. **Classification** (§5.2): any non-`SELECT` through the wrapper
   invalidates every set on the database before it executes.
2. **Row-count check inside the rewritten statement — mandatory.** The
   wrapper only sees its own connection; writers through other connections
   in the same process are invisible to it. A C-API table function reads
   its arguments as constants at bind (`Table function cannot contain
   subqueries`; the `LATERAL` form is rejected too), so the check cannot be
   an argument. It is a one-row derived table cross-joined to the resident
   call:

   ```sql
   SELECT "key" AS l_orderkey, sum AS q
   FROM gpu_groupby_sum_resident_having('<identity tag>', '>', 300) r,
        (SELECT gpu_assert_rows('<identity tag>', count(*)) FROM main.lineitem) g
   ORDER BY q DESC LIMIT 10
   ```

   `gpu_assert_rows` is a volatile C-API scalar
   (`duckdb_scalar_function_set_volatile` exists in the vendored v1.2.0
   header) that raises a typed error when the count differs from the set's
   upload count. The derived table has exactly one row whatever the resident
   side returns, so the check runs exactly once even when the resident
   result is **empty** (a `WHERE`-form guard would never be evaluated on an
   empty result and would pass a stale empty answer). `count(*)` runs inside
   the querying transaction, so it is the snapshot rule with no transaction
   introspection. It is a row-group scan, not stats-only, and is not free on
   tables with deletes; §9.1 measures it and the threshold floor absorbs it.
   On the typed error the wrapper marks the set stale and re-runs the
   original text natively; on any other error it surfaces (§3.3).
3. **Epoch** (§5.5): every invalidation increments the set's epoch; an
   upload that started under an older epoch can never become ready.
4. **Transactions.** The wrapper tracks `BEGIN`/`COMMIT`/`ROLLBACK` on its
   connection and **never rewrites while an explicit transaction is open**:
   a rewritten statement that raises inside a transaction aborts it and the
   user's earlier writes are lost at `COMMIT`, and an upload scheduled from
   a writing transaction would read the committed snapshot and pass the
   count check after `COMMIT`. `COMMIT` on a connection that wrote
   invalidates every set again (the epoch moves), so a background upload
   that overlapped the transaction is discarded. Autocommit statements —
   the common case in Python — are rewritten as usual.

What is not caught: the out-of-process same-count write in §0. Everything
else — any write through the wrapper, any same-process write through
another connection that changes the count, any catalog change — is caught
by one of the four layers, and the write scenarios in §9.2 exercise each
one.

### 5.5 What becomes resident, and when it is ready
When the wrapper sees a rewritable shape over a resolved table above the
§9.1 floor whose columns are not resident, it records the template and runs
the statement **unchanged** (rule 1: the first query pays only the round
trip). The upload is scheduled on the wrapper's own second connection to
the same database — the extension stays free of threads and hidden
connections — under these rules:

- **Quiet period and rate cap.** No upload starts within 2 s of the last
  invalidation of that table, and no more than one upload per table per 30 s
  (wrapper settings). A write-heavy session therefore runs native rather
  than re-uploading after every write and stealing from itself.
- **Epoch capture.** The upload records the set's epoch at start; when the
  device work finishes, the set becomes ready only if the epoch is
  unchanged; otherwise the buffers are freed.
- **Ready means uploaded *and prepared*.** On CUDA the expensive part of a
  set is the key sort, built today lazily on the first query
  (`ensure_join_cache` / `ensure_sort_cache`, `cuda_aggregator.cpp`); a set
  marked ready at upload would make the first hit pay seconds of sorting.
  The upload path calls a new `ResidentColumn::prepare()` (one ABI entry on
  the shared interface; Metal implements it too) that builds the sort cache
  on the upload stream and records a completion event; the set flips to
  ready after that event completes.
- **Memory budget.** Wrapper setting `memory_budget` (default: 50% of device
  memory on a discrete GPU, 25% of unified memory on Apple silicon), passed
  to the upload as the cap. The budget counts key, payload, validity, sort
  cache and scratch, not just the columns. Sets are evicted LRU by last use;
  a set that does not fit is not uploaded; a set is never evicted within 60 s
  of being uploaded (anti-thrash). `GPUDB_UPLOAD_POOL_MAX_MB` stays what it
  is — the extension-side cap on **host** upload buffering — and is
  documented as such.
- `residency = 'manual' | 'eager' | 'background'` is a wrapper setting:
  `'eager'` uploads on first sight (for scripts that know their workload),
  `'manual'` restores v0.6 behaviour (`gpu_upload_pair` only, no managed
  sets, no rewrite).

### 5.6 Concurrency — prerequisite of the background path
`gpu_resident.cpp` takes the global mutex inside `upload_update` (line 248)
and `upload_combine` (line 302), which are DuckDB's per-thread aggregate
callbacks. DuckDB's worker pool is shared across connections in one
database instance, so every scan thread of the upload statement blocks on
that mutex and a native query on the other connection starves for workers.
Measured on SF10: Q18-inner 798–811 ms against an 80 ms baseline (0.10×)
while an upload ran, and small statements 1.5× slower for the whole window.
The fix is structural, not throttling:

- per-thread upload state with **no lock in `update`**; the lock is taken in
  `combine`/`finalize` only (already the contract for source states); pool
  accounting through an atomic;
- a short registry lock for lookup only; per-set state
  `{uploading, prepared, ready, stale, epoch, completion event, refcount}`;
- upload and prepare work on their own device stream (CUDA stream; Metal
  command queue) outside any global lock; a per-set (not global) mutex
  around the sort-cache build;
- an operator call holds a **reference** on the set for the duration of the
  call instead of the global lock; eviction waits for the refcount to reach
  zero, so LRU can never free a column mid-query.

This is shared-extension plus CUDA-backend work carried as one
`feat/core-*` PR from the Linux instance (milestone 0b, §12); the Metal
backend honours the same per-set state. The first-query row in the gate
(§9.3) is measured with the upload and prepare actually in flight, using
max and p99 rather than min-of-N, because min-of-N hides the contention
entirely (it reported 80 ms during the 800 ms window).

### 5.7 Shared with joins
A resident set is `{identity, columns, sorted key permutation, validity,
state}`. The v0.5 join build side is the same object, which is what lets
v0.8 route joins through the same manager (§10).

## 6. The rewrite (piece D)

`gpu_rewrite_ast(tree, context)` over the serialized statement, pure:

1. Walk the tree for the §2 shape, reading every field (§2 rejection
   list); on the first non-matching node or field, return the input
   unchanged. The cost of an unmatched tree is a target of 0.05 ms in the
   function, measured by a unit test over the §2 rejection corpus once it
   exists (§9.1 carries the row; nothing is measured yet).
2. Read residency and thresholds **from `context`** (the wrapper resolved
   the identity, looked up `gpu_residents()`, and passed the set key, its
   column list, the effective backend, the table's row count and the
   connection's `default_order` / `default_null_order` /
   `default_collation`). If the backend is CPU: unchanged, reason
   `backend`. If the set is missing or not ready: unchanged, reason
   `not_resident`. Below threshold: unchanged, reason `threshold`. A C-API
   scalar cannot read catalog statistics from a JSON string, which is why
   these come from the wrapper.
3. Build the replacement tree: the base-table reference becomes a
   `TABLE_FUNCTION` reference to the resident operator with constant
   arguments `{identity tag, predicate program, aggregates, filter, top-k}`
   cross-joined to the `gpu_assert_rows` derived table (§5.4). Every
   remapped output is wrapped in a `CAST` to the native type (§4.2) and
   carries the user's alias, or, when unaliased, DuckDB's auto-generated
   name (`sum(l_quantity)`, `count_star()` — obtained once per template by
   `json_serialize_sql` of the projection and cached). `HAVING` thresholds
   are rescaled exactly to the payload's scale: `>` floors, `>=` ceils, `<`
   ceils, `<=` floors; `=`/`<>` only when exactly representable, otherwise
   constant false/true; 128-bit thresholds on the two-limb path.
   `GROUP BY`/`HAVING` are removed; `ORDER BY … LIMIT` are **kept** and
   additionally pushed as top-k when direction and NULL order are known;
   everything else in the tree (CTEs, projections, further `ORDER BY`,
   joins to the result) is left as is.
4. `last_rewrite()` is a **wrapper** method (not a SQL function: the scalar
   is pure and may run in parallel) reporting, for the last statement on
   that connection, whether it was rewritten and if not why (`shape` /
   `not_resident` / `threshold` / `stale` / `budget` / `backend` /
   `transaction` / `too_long` / `double`), which is also what the gate
   parses. `EXPLAIN` of the rewritten statement shows the `gpu_*` table
   function in the plan.
5. Settings are wrapper-side: `gpudb.connect(transparent=True,
   residency='background', memory_budget=None, floor_rows=None)` and
   `con.gpudb.transparent = False` at runtime. For scripts, the wrapper also
   honours `SET VARIABLE gpudb_transparent = false` (read back with
   `getvariable()`, plain SQL on every DuckDB version). No `SET gpudb_*`
   config options: the v1.2.0 API struct cannot register them and the API
   floor does not move.

## 7. Backend work (both, in parallel)

| Kernel / path | Metal (macOS instance) | CUDA (Linux instance) |
|---|---|---|
| NULL-key partition at upload + identity-injected reduce, all-NULL → NULL (4.1) | compaction pass; extend `sum.metal` reduce tuple to `(sum, count_v, count_star, min, max)` | `DevicePartition` on validity; transform iterator injects identities |
| two-limb sum, HUGEINT/DECIMAL(38,s) output vectors (4.2) | manual carry in the segmented reduce | `reduce_by_key` with an `__int128` accumulator |
| predicate mask, variants (a) masked reduce with empty-group drop and (b) compact-then-reduce; f64 total-order compare (4.6) | new kernels; (b) reuses the v0.6 block compaction | new kernels; (b) uses `DeviceSelect::Flagged` |
| min/max group-by | fused into the per-group tuple: one pass for all aggregates of a query | same, `reduce_by_key` on the tuple |
| `ResidentColumn::prepare()` on the upload stream + completion event (5.5) | command-queue event | stream event; `ensure_*_cache` called from prepare |
| per-set state, refcount, lock-free `update` (5.6) | honours the shared state | owns the PR |

The CPU reference backend implements every one of these first, in plain C++,
and is the oracle the two GPU backends are checked against in unit tests
before parity against native. The CPU backend is an oracle only: on the
CPU-only registry binary the resident path is a single-threaded sort per
call, 108 ms against 16–18 ms native on 6M rows / 1.5M groups, so the
rewriter refuses on it (§6 step 2) and there is no CPU threshold to find.

## 8. Spike

The plan-surgery spike in the previous revision is no longer needed: the
parser-level rewrite was run end to end on 2026-09-01 (§3.2). What remains
to confirm before piece C/D is written, on the same throwaway branch:

1. The serialized tree for every §2 construct and every §2 rejection
   (`WHERE` conjunctions, multi-column keys, `count(*)`, `min`/`max`/`avg`,
   DECIMAL literals, CTEs wrapping the shape, `GROUP BY ALL`, ordinals,
   `FILTER`, `DISTINCT`, `ROLLUP`) and that `json_deserialize_sql`
   reproduces each rewritten form.
2. `gpu_assert_rows` in the cross-join form: that it is evaluated exactly
   once, before any row reaches the client, on a non-empty and on an empty
   resident result, under the streaming fetch path (`fetch_df_chunk`,
   `fetchmany`), and with the optimizer's filter pushdown and join
   reordering on; and its measured cost on a base table with and without
   deleted rows.
3. The wrapper's statement classification, cache and invalidation path with
   multi-statement strings, `EXPLAIN`, prepared statements with parameters,
   transactions, `register()`, temp tables and `ATTACH`.
4. The auto-generated column names and `typeof()` of every output for every
   aggregate/input-type pair in §4.2, against native.

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
- the `gpu_assert_rows` cross-join cost on a base table, clean and with 10%
  deleted rows, at each SF;
- wrapper overhead, measured on both machines: a plain `SELECT` (no
  `GROUP BY`) through `duckdb.connect` vs through `gpudb.connect` with
  `transparent=False` vs transparent — rule 1 has to hold for statements
  that are never rewritten; the **first-sighting** row (classification +
  whitelist + one nested round trip, 0.124 ms measured) against the SF10 hit
  case and against the smallest table above the floor; a
  **distinct-literal loop** (the same template with 1,000 different `HAVING`
  constants) which must hit the template cache and pay nothing after the
  first; a **two-connection** row; and the unmatched-tree cost of
  `gpu_rewrite_ast` over the §2 rejection corpus.

The **floor** is derived here: the smallest table size at which native
`GROUP BY` on that backend takes ≥ 50× the measured round trip. Tables
below it are never parsed. Thresholds in the wrapper (row count, group
estimate, selectivity crossover) are set from these tables, per backend,
and the losing side of every sweep stays printed. The group estimate comes
from the resident set itself once it exists (exact); before upload the
wrapper uses the row count only, which is why a miss never rewrites.
Today's row (F) (4 groups, 0.56–0.65× on Metal) stops losing because the
rewrite does not fire there. Numbers quoted in this section are from the
review's measurements and are replaced by BENCHMARK.md rows as they land;
thresholds are measured on both machines and the gate refuses to enable the
transparent path on a device whose margin at the floor is below 1.2×
(laptops, T4) rather than assuming the 4090 and M-series numbers transfer.

### 9.2 Three-way parity
`groupby_parity_check.sh` runs every scenario native
(`transparent=False`), transparent, and explicit `gpu_*`, in one process
through the Python wrapper (the CLI has no interception, §3.5), and all
three must agree on **ordered rows, column names and `typeof()` of every
column** — not `EXCEPT`, which misses duplicates, order, names and types.
Every row of §4 is a scenario: NULL keys, NULL payloads, all-NULL groups,
`count(*)` vs `count(v)`, sums past 2^63, DECIMAL(15,2) and DECIMAL(38,4),
`HAVING` thresholds at a finer scale than the payload, packed 2- and
3-column keys at the range boundary, VARCHAR keys under binary and `NOCASE`
collation (the latter must not rewrite), NaN/±inf under each `WHERE`
operator, each `HAVING` operator, ties in top-k, `default_order='DESC'` and
`default_null_order='NULLS FIRST'` set on the connection, and the
fall-through shapes (assert the plan did **not** change and the answer still
matches). Plus the write scenarios, each between two identical queries:
insert / update keeping the count / delete / rollback / appender /
`register()` of a same-named DataFrame / temp table shadowing / CTE
shadowing / `ATTACH` + `USE` / a second connection writing / a write inside
an open transaction with the query before and after `COMMIT`.

### 9.3 `scripts/transparent_gate.sh` (rule 1)
Every rewritable shape and every §9.1 sweep point, transparent vs native,
same process, warm, min-of-N with N ≥ 5 and the GPU clock printed beside
each row (the 4090's boost clock is bimodal, 1665–2400 MHz, and a 1.0× row
can flip between runs otherwise), both backends. **Any row below 1.0×
fails**, except the first-sighting row, whose bound is the §0 2% and is
printed as such. Runs at SF1 in `local_check.sh` through the Python
wrapper, at SF10/SF50 before every tag on both machines, output pasted into
BENCHMARK.md unedited. The first-query case (residency miss) is a row in
this table too, measured **while the upload and prepare are in flight** and
reported as max and p99, not min: it must be ≥ 1.0× because it runs native,
and this is the row that proves the background work does not steal from it.

### 9.4 Community path
Unchanged C-API template path (`make configure && make release && make
test`) on Linux plus the registry-smoke workflow. The registry's Linux
binary is CPU-only and the rewriter refuses on CPU (§6 step 2), so the
smoke probe asserts the **mechanism**, not a rewrite:
`SELECT gpu_rewrite_ast(json_serialize_sql(...), '{"backend":"CPU",...}')`
returns its input unchanged with reason `backend`, and separately the
explicit upload + `gpu_groupby_*` call returns the native answer. The GPU
node is demonstrated on the release assets (Metal, CUDA), not on the
community binary.

## 10. v0.8, designed for now

- **Transparent joins**: `t1 JOIN t2 ON t1.k = t2.k` with aggregates above,
  routed to the v0.5 resident join on the same resident sets (§5). The
  matcher gains one more shape; the residency manager and gate are already
  there.
- **Fused aggregate → join** (TPC-H Q18 end to end: filtered groups joined
  back to `orders`/`customer` on the device without materialising the group
  rows). New kernel, both backends.
- **Window functions** over resident sorted columns (`WINDOW_FUNCTIONS_DESIGN.md`).
- **DOUBLE sums**, only with a definition of exactness against native (§4.7).

## 11. v0.6.1 — maintenance release (first)

Packaging only. No operator changes.

**Problem.** The v0.6.0 `gpudb.linux_amd64.duckdb_extension` release asset
was built on Ubuntu 24.04 and needs `GLIBCXX_3.4.32` / `GLIBC_2.38`, so it
does not load on Ubuntu 22.04 hosts (glibc 2.35), Google Colab included. It
is also built against CUDA 13, which needs an R580+ driver; CUDA 12.x
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
- Two facts the README "Requirements" row must carry, since neither is
  DuckDB's own floor: the asset needs `libgomp.so.1` on the host (not
  present on minimal images; `apt install libgomp1`), and its glibc floor
  is 2.34, which is stricter than the DuckDB CLI's.
- `scripts/build.sh` honours `GPUDB_CUDA_STATIC_RUNTIME` from the environment
  (today it must be set on the CMake cache by hand; the container script
  already does) and `scripts/container_build_linux.sh` is added so the
  portable build is reproducible from the repo — one PR from the Linux
  instance.
- `examples/gpudb_quickstart.ipynb` (PR #81): loads the release asset, falls
  back to a direct `cmake --target gpudb_duckdb` build with Colab's `nvcc`.
  Needs one end-to-end T4 run that reaches the benchmark cells before it
  leaves draft. A PTX-only fallback does not help drivers older than the
  toolkit's JIT expects, so the asset carries SASS for the T4's sm_75.

**Release steps (each on the user's explicit go).**
1. Merge #82. Rebuild the portable asset from the final commit in the
   container; re-run `loadtest.sh` on Ubuntu 22.04 and 24.04.
2. Bump `CMakeLists.txt` to 0.6.1; tag; release notes = the problem
   statement, the glibc/CUDA/libgomp floors, the SHA-256s. macOS asset
   rebuilt at the tag so both carry the same footer.
3. Registry `description.yml` ref → the v0.6.1 tag (same CPU-only binary;
   the ref should point at a tag that exists).
4. Un-draft #81 after the Colab run is seen.
5. README "Requirements": Linux asset glibc floor + libgomp + CUDA
   12.8/driver line; the CUDA-13 note moves to "building from source".

**Why first.** The container build, the static CUDA runtime and
`-static-libstdc++` (#82) are the portable Linux build for every release
from here on — the loadable stays on the C API (§3.4), so nothing C++ ever
crosses into DuckDB and the static runtime remains correct.

## 12. Milestones (no calendar — each gated on being right)

| # | Deliverable | Gate |
|---|---|---|
| 0a | v0.6.1 (§11) | user go per step |
| 0b | Residency prerequisite PR (§5.3, §5.5, §5.6): identity-keyed registry with origin, per-set state + epoch + refcount, lock-free `update`, `ResidentColumn::prepare()` on the upload stream, `gpu_assert_rows`, `gpu_residents()` columns; Linux instance, Metal implements `prepare()` | unit + parity unchanged; SF10 concurrent-upload row ≥ 1.0× at p99 |
| 1 | Spike remainder (§8): tree shapes and rejections, `gpu_assert_rows` evaluation-order proof, classification/cache/invalidation, output names and types; this doc updated | findings written down |
| 2 | `gpu_rewrite_ast` (pure, context-driven) for the plain `GROUP BY` shape + the Python wrapper (classification, resolution, template cache, transaction rule, typed fallback, settings), three-way parity on the v0.6 operators as they are | PR |
| 3 | Exactness (§4): CPU reference first, then Metal + CUDA in parallel; native output types and names; unit + parity | per-kernel PRs |
| 4 | Residency manager in the wrapper (§5.5): budget, eviction, quiet period, epoch, all write scenarios in parity | PR |
| 5 | Full §2 shape in the rewriter, `HAVING` rescale, kept `ORDER BY` + top-k push, `last_rewrite()`, `gpu_residents()` stats; Node/R/JDBC wrappers | PR |
| 6 | Gate (§9): sweeps, floor and thresholds per backend, `transparent_gate.sh` all ≥ 1.0× on both machines (first-sighting row within its bound), BENCHMARK/KNOWN_ISSUES rows | PR; gate must pass |
| 7 | Audit (v0.6 shape), tag v0.7.0, registry bump, `pip` package release (per-platform assets; `allow_unsigned_extensions` requirement stated) | user go per step |
| — | Upstream proposal for a C API statement-rewrite hook (§3.5) | drafted after milestone 2 proves the interface; user sign-off before anything is posted |

Milestone 0b precedes everything that uploads in the background. Milestones
3 and 4 run in parallel across the two machines; 1, 2, 5, 6 are shared-file
work and go one at a time. The loadable extension's build, ABI (plus the one
`prepare()` entry) and release shape do not change at any milestone.

## 13. Review record (2026-09-02/03)

The 2026-09-01 revision was put through an adversarial review: eight
independent reviewers, one lens each (exactness, rewrite, invalidation,
rule 1, security, packaging, consistency, alternatives), 99 findings, each
argued against by two further reviewers; every kept finding was reproduced
on the v1.5.2 CLI with the local Metal build or on Python `duckdb` 1.4.5.
The Linux instance then checked the resulting changes against the CUDA code.

**Kept and folded in above:** the unbound tree (§3.2, §5.1); the row-count
check that could not bind and its empty-result hole (§5.4); DML invisible
to `json_serialize_sql` (§5.2); the transaction abort (§5.4);
the in-flight upload surviving an invalidation (§5.5); the process-global
registry (§5.3); rule 1 on first sighting and on distinct literals (§0,
§3.2, §9.1); upload contention and its cause in the `update` callback
(§5.6); no `SET gpudb_*` on the v1.2.0 struct (§3.4, §6); output names and
types (§4.2); session-dependent `ORDER BY` (§2); `HAVING` rescale, VARCHAR
collation, DOUBLE predicate total order, DOUBLE sums (§4.5–4.7); the CPU
backend never rewriting (§7, §9.4); the unmeasured 0.05 ms claim (§6);
ready-means-prepared (§5.5); and the medium items on shape rejection,
all-NULL groups, empty groups under the mask, memory accounting, `count(*)`
cost, per-device thresholds and the parity method.

**Rejected after argument (do not re-raise):** parameters baked into the
cache (`PARAMETER` nodes are rejected by shape); the C++ shim as primary
route (its bind hook runs only on bind failure); "no config-option API"
(the API exists from v1.5; the constraint is the v1.2.0 floor, which
stays); `sum(FLOAT)` in f32 (no f32 path exists); `gpu_rewrite_ast` as a
crash surface (the deep-nesting crash is DuckDB's own
`json_deserialize_sql`, reachable without gpudb); incomplete catch blocks
(every backend throw is `std::runtime_error`); the documented gap being
too narrow.

**Decided by the user, 2026-09-03:** the bounded first-sighting cost (§0)
and the out-of-process same-count write as the one documented gap (§0),
each over the stricter alternative that would have made the feature
impossible (no rewrite without a prior native timing; a full fingerprint
scan on every hit).
