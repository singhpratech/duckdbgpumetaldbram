# v0.7 — transparent GPU execution (design)

Status: proposal, 2026-09-01; revised 2026-09-03 after an adversarial review
and the milestone-1 spike (§13). Companion to `GROUPBY_RESIDENT_DESIGN.md` (v0.6). Carries the v0.6.1
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

The one documented gap, decided 2026-09-03: a writer on a **connection the
wrapper did not open** — another process, or a raw `duckdb.connect()` in
the same process — that changes values without changing the row count is
invisible to the wrapper and to the in-statement row-count check (§5.4).
DuckDB's single-writer rule already excludes the other-process case when the
wrapper's process holds the write lock, so the requirement is: the wrapper
opens the database read-write (the default) and every connection to that
database in the process is opened through the wrapper; anything else is
unsupported and stated in KNOWN_ISSUES.

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
FROM t [[INNER] JOIN d ON t.fk = d.key [JOIN ...]]      -- or: FROM t, d WHERE t.fk = d.key
[WHERE <predicate on columns of t, d, ...>]
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
- `FROM`: one base table, or a join of base tables. An inner equi-join tree
  in which every join lands on a column that is unique in its table — the
  fact-to-dimension joins of a star or snowflake schema — is materialised on
  the device (§4.8). Any other INNER / LEFT join of base tables tied together
  by equalities — many-to-many, `USING`, composite or non-integer keys, extra
  `ON` predicates, self joins, keys or expressions that mix columns of several
  tables — is answered from an upload of the join's result (§4.13, Python
  wrapper). RIGHT / FULL / semi / anti joins, cross products, subqueries as
  join inputs and joins whose result is more than four times the largest
  table run native.
- No `GROUP BY` at all (`SELECT sum(x), count(*) FROM … WHERE …`) is its own
  device operator, a single fused pass with no key (§4.12): accepted over a
  join at any size, and over a single table above a measured row and
  predicate-term bound.
- Payloads: integer family, DECIMAL (§4.3); DATE / TIMESTAMP for `min`, `max`
  and `count` (days / microseconds on the device, typed on the way out).
  `count(DISTINCT x)`, one DISTINCT column per statement (§4.17).
- A select-project-join derived table (or a single CTE) as the FROM is folded
  into the statement first (§4.16). DOUBLE/FLOAT payloads are not
  rewritten (§4.7); they stay available through the explicit `gpu_*` calls.
- Aggregates: `sum`, `count`, `count(*)`, `min`, `max`, `avg` (= sum/count,
  computed on the host from the two device results). Several aggregates over
  the same payload in one query are one device pass; aggregates over up to
  eight different payload columns share the mask and the grouping and add one
  reduce each (§4.9).
- `WHERE`: conjunctions of comparisons between a column of `t` and a
  constant, `IN (list)`, `BETWEEN`, `IS [NOT] NULL`. Evaluated on the device
  as a mask over the resident predicate columns (§4.6). Through the Python
  wrapper every other deterministic row-local conjunct — `OR`, `LIKE`, a
  function predicate, column-vs-column — is a computed BOOLEAN lane (§4.10);
  subqueries, correlated references and volatile functions run native.
- Expressions (§4.10, Python wrapper): a deterministic expression over the
  columns of one table may stand as an aggregate's argument
  (`sum(price * (1 - discount))`, `sum(CASE ...)`), as a GROUP BY key
  (`year(d)`, `substr(s, 1, 2)`, `a % 10`) and on the column side of a
  `WHERE` comparison.
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
node on the path and declines the statement when any of these is present
(field names as `json_serialize_sql` emits them, checked 2026-09-03 on
DuckDB 1.4.5): `GROUP BY ALL` (`aggregate_handling = FORCE_AGGREGATES`),
ordinal group references (a `CONSTANT` in `group_expressions`),
`ROLLUP`/`CUBE`/`GROUPING SETS` (more than one entry in `group_sets`),
`FILTER (WHERE …)` on an aggregate (`filter` on the function node),
`DISTINCT` inside an aggregate (`distinct: true`), `ORDER BY` inside an
aggregate (`order_bys` on the function node), `TABLESAMPLE` (`sample` on the
`BASE_TABLE`), `USING SAMPLE` (`sample` on the select node), `AT` clauses
(`at_clause` on the `BASE_TABLE`), window functions (class `WINDOW`),
`QUALIFY` (`qualify` non-null), set operations (`SET_OPERATION_NODE`),
`PARAMETER` nodes, and a non-empty `cte_map.map` at any scope that defines
the table's name (§5.1). The unit test for the matcher enumerates each of
these as a serialized tree that must come back unchanged.

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
  → statement classification (DuckDB's own splitter, §5.2)          every statement, memoised on the text
  → whitelist check: names a table above the §9.1 floor?              ~0.001 ms, regex on the text
  → template cache lookup (literal-normalised, §3.3)                  hit: 0 extra
  → one statement:
      SELECT json_deserialize_sql(gpu_rewrite_ast(json_serialize_sql(?), <context>))
                                                                       0.124 ms measured
  → the returned statement runs (rewritten, or the input unchanged)
       — as EXECUTE of a plan DuckDB already built, from its second sighting on
```

The rendered statement is **planned once, not per run**. DuckDB is handed SQL
text, so it parses, binds and optimises the rewritten two or three relations on
every execution; the text of a warm template is byte-identical run to run, so
the second sighting of a rendered statement is `PREPARE`d and every later one
is an `EXECUTE` (`connection.py:_planned`). Measured at SF1: 0.06–0.10 ms of a
0.7–0.9 ms few-group statement, 0.37–0.43 ms of a 6.5–7.6 ms one with more
relations (BENCHMARK.md, 2026-09-18). A statement whose literals change on
every execution renders a new text every time, never comes back, and is never
prepared.

A plan may not freeze the answer, and it does not. The staleness guard (§5.4)
is a volatile scalar over a live `count(*)` of the base table, inside the plan:
it runs on every execution, and a write behind a prepared plan raises
`GPUDB_STALE` exactly as before. The table function's `init` runs again and
re-acquires the set. DuckDB re-binds a prepared statement when the catalog
moves under it (DDL, a view redefinition). What it does **not** re-bind is a
session setting or a name resolution — a temp table that shadows the name, a
`USE`, `SET default_order` — so the plans are dropped wherever a decision is
dropped: `_invalidate_all` (every non-SELECT the wrapper sees, including SET,
ATTACH, DDL, a transaction and a foreign write), `_refresh_settings`,
`_on_stale` and `_on_rewrite_error`. Names are never reused for a different
statement, so two threads on one connection cannot execute each other's plan.

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

**Two algorithms, chosen inside the backend (2026-09-18).** Everything above
reads the key's sort cache: mask the rows, find the run starts of the sorted
keys, reduce each payload through the permutation. That is the right shape
for a key with many distinct values and the wrong one for a key with few —
at SF10 the run starts are a flat 2.9–3.1 ms whether the answer has four rows
or a hundred thousand, and the reduce gathers 60M rows through a permutation
to produce them. So a key whose distinct count is small enough gets a dense
**group-id lane** (`docs/RESIDENT_COLUMNS_DESIGN.md` §7) beside its sort
cache, and the exact operators run a **direct** pass instead: one pass over
the rows in storage order, the whole `WHERE` evaluated per row by §4.12's
`gpred_eval`, every payload folded into the accumulator of that row's group
id. No mask buffer, no run starts, no permutation, no gather.

Which of the two runs is decided inside the backend, and rule 1 applies there
too: the direct pass reads the id lane on top of the payloads, so it needs
`groups >= 3` and `rows × (payloads + WHERE terms) >= 6,000,000` before it is
worth taking — measured, BENCHMARK.md. Below that, or with two groups (where
the sort path's reduce over two runs is already a sequential scan), the sort
path runs and nothing is lost.

Nothing above the backend changes. The ids are the ranks of the ascending
distinct keys and a NULL key takes the reserved last id, so the output order,
the NULL-key group's place and the absence of a group the `WHERE` emptied are
what they were; `GroupByFilter` (HAVING, top-k) runs on the host over the few
group rows through `apply_group_filter_host`, the same reference the device
filter is tested against; and the interface is untouched
(`gpu_backend.hpp` is frozen for the CUDA port). Which algorithm ran is a word
in `gpu_last_stats()` — `path=direct` or `path=sort` — and in
`GPUDB_METAL_TRACE_EXACT`; `GPUDB_METAL_GROUPBY_EXACT_PATH=direct|sort|auto`
pins it for tests and sweeps.

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

Output types follow native exactly (`DESCRIBE` over every aggregate ×
input type, 2026-09-03): `sum` of any integer type, HUGEINT and UBIGINT
included, is HUGEINT; `sum` of `DECIMAL(p,s)` is `DECIMAL(38,s)`;
`count`/`count(*)` are BIGINT; `min`/`max` keep the input type; `avg` is
DOUBLE for **every** input, DECIMAL included. Native `avg` over integer and
DECIMAL inputs is the exact sum divided by the count (verified identical
across 1/2/8 threads and three insert orders, and equal to
`sum::DOUBLE / count`), so the device computes it the same way from the
two-limb sum and stays exact. Unaliased output names are DuckDB's
(`sum(v)`, `count_star()`, `avg(v)`); the wrapper obtains names and types
together from one `DESCRIBE <statement>` per template — bind only, no
execution, 0.08 ms measured. The parity harness compares `typeof()` and
the name of every output column (§9.2).

### 4.3 DECIMAL
DuckDB stores `DECIMAL(p,s)` with `p ≤ 18` as an int64 scaled by `10^s`
(int16/int32 for smaller `p`). A DECIMAL payload is uploaded as its integer
representation; `sum` is the 4.2 path with the result typed `DECIMAL(38,s)`;
`min`/`max` are exact; `avg` is DOUBLE, the exact sum over the count on the
host, as native does it (§4.2). This is what
makes TPC-H's `l_quantity` / `l_extendedprice` match with no cast — the
README examples currently write `l_quantity::BIGINT` and v0.7 stops needing
to. `DECIMAL(p>18)` is int128-backed and is uploaded as two limbs (same
machinery as 4.2).


`avg` over a DECIMAL(p ≤ 18, s) payload is computed the way native computes
it — ONE division by the scaled count, `double(unscaled sum) / (count ×
10^s)` — in the rewritten select list (and in a host-side `HAVING avg(...)`),
from the exact `sum` and `count` the table function returns. Verified against
native over thousands of groups at four scales; `(sum / count) / 10^s`,
`(sum / 10^s) / count` and `sum(decimal)::DOUBLE / count` each differ from
native on 20–30% of groups. Native does this arithmetic in `long double`
(80-bit on x86, double on ARM): the formula is verified on ARM and must be
re-verified on x86 with the CUDA port.

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

### 4.8 Key joins, materialised on the device
The join shapes analytics SQL is made of are fact-to-dimension: `lineitem
JOIN orders ON l_orderkey = o_orderkey`, then `orders JOIN customer`, and so
on. The dimension side is joined on a primary / unique key, so every fact row
has at most one match and the inner join is *a subset of the fact rows with
dimension columns attached*. That is an exact row set of the §4.6 kind, and
it is produced on the device once and kept:

`Aggregator::join_materialize(probe_key, build_key, out lanes)` takes two
resident row sets and returns a NEW one in the `upload_rows_exact` layout
(lane 0 = the GROUP BY key, NULL-key rows in a suffix, probe order kept).
Each output lane is a probe lane (copied) or a build lane (gathered through
the match). Every form of §4.1–§4.6 then runs over the joined set unchanged —
plain, `WHERE` on columns of either table, device `HAVING`, top-k — and an
output lane can probe a further dimension, so snowflake chains are repeated
calls. Semantics are native's: NULL join keys never match, a fact row
without a match is absent, a gathered cell is NULL iff its source is. The
result is bit-identical across backends, row for row.

The build key MUST be unique among its non-NULL cells; the operator verifies
it on the device (one pass over the sorted keys) and throws `build key not
unique` otherwise, and the statement then runs native. Many-to-many joins,
outer / semi / anti joins, non-equality conditions and joins whose result is
returned row by row (no GROUP BY) are not rewritten: the first multiplies
rows (a different operator), the last is bound by moving the rows back
through DuckDB (rule 1, same finding as the plain form at millions of
groups).

Metal: the build key's sort cache (sorted keys + permutation) is the index;
one kernel binary-searches every probe key and classifies the row, block
counts + a host scan give the destinations, one gather kernel per output
lane clears validity bits atomically. TPC-H SF1 on an M4 Max: 6.0M × 1.5M
rows joined with four output lanes in 12 ms warm (25 ms including the build
side's sort); SF1 statements over the joined set, identical to native:
GROUP BY o_custkey 1.44×, top-10 customers 4.29×, three tables by
c_nationkey under a date predicate 4.17× (min of 5, statement vs statement,
operator-level).

SQL: `gpu_join_materialize(out, probe, probe_lane, build, build_lane,
'p.<lane>, b.<lane>, ...')` publishes the joined set under `out`; it goes
stale as soon as a base set it was built from is stale, dropped or replaced
(checked by identity on every use). A joined set built from another joined
set inherits that set's base sets as its sources — the rows were copied — so
the intermediate sets of a chain are dropped as soon as the chain is done.

**Plain SQL (the wrapper).** A statement whose `FROM` is a join is *lowered*
to the single-table shape before anything else looks at it
(`python/gpudb/_join.py`): the join tree is flattened, every `a.x = b.y`
conjunct of the `ON` clauses and of `WHERE` between two tables becomes a join
edge (the other `ON` conjuncts of an inner join are `WHERE` conjuncts and
move there), column references are resolved to their tables (aliases,
unqualified names that are unique across the join) and rewritten to the
columns of one virtual table named as the root table. The matcher, the type
checks, the thresholds, the pure rewrite scalar and the renderer then run
unchanged on the lowered tree. The root is the largest table from which every
edge points at a unique column; uniqueness comes from a `PRIMARY KEY` /
`UNIQUE` constraint or from one `count(c) = count(DISTINCT c)` scan, cached
until the next statement that can change data. No such root → native.

Residency: one exact set per base table under its own identity tag
(`…:<lanes>:join`; a dimension's lane `k` is its unique key, so its sort
cache is the join index), then the `gpu_join_materialize` chain from the root
outwards, the last step laying the lanes out as `key, payload, predicate
lanes` — the order the rewriter addresses. In the residency manager the
joined set is a *derived* set: it is picked only when every source is ready,
its steps run in idle windows like an upload's finish, and invalidating a
source invalidates it. The rewritten statement carries one staleness guard
per base table (`gpu_assert_rows(<base tag>, (SELECT count(*) FROM <table>))`,
context field `guards`), so a write to any joined table from any connection
is caught exactly as for a single table; the wrapper then re-uploads that
table and materialises again.

Thresholds (`_thresholds.py`, from `scripts/transparent_gate.py`'s join
sweep, TPC-H SF1 on an M4 Max, identical rows on every line): native has to
run the join whatever the group count, so there is no lower bound on groups
(25 groups: 2.3–4.4×); HAVING 1.4–7.4×, top-k 1.6–4.0×; the plain form
1.25–1.28× at 100K groups returned, 1.09–1.13× under a 9–49% `WHERE`, and
0.92–0.99× at 32K groups under a 3% `WHERE` — a selective dimension filter
makes native's join tiny while the resident operator still masks every fact
row; declined. SF10 (60M × 15M rows): all 78 swept rows win, the plain form
1.08–1.28× at 1M groups returned, HAVING / top-k 2.0–9.9×, few-group shapes
up to 18×.

Not covered yet: a multi-column GROUP BY whose components come from
different tables (the packed key is computed per table at upload), composite
and non-integer join keys, and other language wrappers (the lowering lives in the Python wrapper; the
scalar only needs the `guards` field).

### 4.9 Several payload columns in one statement
`SELECT k, sum(a), min(b), avg(c) ... GROUP BY k` reads three payload columns.
The resident set needs no new layout: the first payload is lane `v`, every
other one is a BIGINT predicate lane of the same set (DECIMAL as its unscaled
integer, exactly like `v`), whether or not the `WHERE` also reads it — the
identity tag lists it among the set's columns, so one set serves both uses.

`Aggregator::groupby_exact_masked_multi(keys, payloads[], filter_payload,
preds, filter)` returns one row-aligned result per payload; the shared columns
(keys, `count(*)`) ride on the filtered payload's result. The default
implementation is one single-payload pass per column: every pass sees the
same keys and the same mask, hence the same groups in the same order, so the
passes are zipped by position, and under a HAVING / top-k the filtered
payload runs first and the others are looked up by key. Metal overrides it:
the mask, the selection (range, compaction) and the run starts are computed
once, only the reduce (chunk partials → finalized tuple) repeats per payload,
all extra payloads in one command buffer; the filter stage runs on the
filtered payload's tuple and the others are gathered from shared memory by
key. Measured with three payloads on TPC-H SF1: the unfused version lost 20
gate rows (0.56–0.99×, each extra payload repeating the mask and the
grouping), the fused one none.

SQL: `gpu_groupby_exact_multi(name, program, 'v, i0, i2', filter)` →
`key, count_star`, then per payload `sum<p> HUGEINT, count<p>, min<p>, max<p>,
avg<p> DOUBLE`; `filter` is `''`, `'having <p> <agg> <cmp> <threshold>'` or
`'topk <p> <agg> <k> <asc|desc>'`. Projection pushdown skips the payloads and
columns a statement does not read. Both rewrite engines emit it when a
statement aggregates more than one column (or one that is resident on a
predicate lane); a single payload on lane `v` keeps the single-payload
functions. Per-payload typing (DECIMAL scale, native `min`/`max` type) and the
HAVING threshold's rescale follow the payload the aggregate reads.

Thresholds: every extra aggregate is another output column on both sides, so
the plain form's margin shrinks (1.01–1.09× with three payloads, one 0.93×
under a three-term `WHERE`; 1.00–1.08× at 100K groups over a join) and
`_thresholds.py` sends the multi-payload plain form native under a `WHERE`,
above 50K groups on a single table and above 20K groups over a join. HAVING
and top-k keep their wins (1.06–3.0× single table, up to 4.8× over joins).

### 4.10 Computed lanes: expressions as columns
A deterministic, row-local expression over the columns of one table is, for a
GROUP BY, just another column of that table. The wrapper
(`python/gpudb/_exprs.py`) lowers such an expression to a *virtual column*
`x_<hash>` before the matcher sees the statement: DuckDB itself evaluates the
expression while the table is uploaded (`gpu_upload_rows_exact(..., <expr>,
...)`), so its semantics, NULL handling, casts and typing are native's by
construction, and everything downstream — matcher, type checks, thresholds,
the pure rewrite scalar, the renderer, the join planner — handles an ordinary
column. The scalar needed no change: it receives the lowered tree and a
context whose `columns` include the virtual ones.

Where an expression may stand:
- the argument of `sum` / `count` / `min` / `max` / `avg` → a payload lane;
- a GROUP BY expression, with its repeats in SELECT and ORDER BY → a key
  (BOOLEAN keys travel as 0 / 1 and come back typed BOOLEAN);
- `expr <op> constant`, `expr IN (...)`, `expr BETWEEN ...`,
  `expr IS [NOT] NULL` → a predicate lane compared on the device;
- any other `WHERE` conjunct (`OR`, `LIKE`, column-vs-column, a function
  predicate) → a BOOLEAN lane kept where it is TRUE, which is SQL's `WHERE`.

Accepted inside an expression: column references, constants, casts, `CASE`,
comparisons, `AND` / `OR` / `NOT`, `IS NULL`, `BETWEEN`, `IN` lists,
`COALESCE`, and every function `duckdb_functions()` lists as a scalar with
stability `CONSISTENT` in all its overloads (built-in macros such as `nullif`
when their definition names nothing else). Not accepted: `VOLATILE` and
`CONSISTENT_WITHIN_QUERY` functions (`random()`, `now()`, `nextval()` ...),
functions that read session state (`current_setting`, `getvariable`, ...),
user macros, subqueries, windows, lambdas, parameters, aggregates, and an
expression over columns of two joined tables (a lane belongs to one table's
set). The expression's type decides what it can be: integer family / DECIMAL
as a payload (DOUBLE declines as `double`, as for a column), those plus DATE /
TIMESTAMP / VARCHAR / BOOLEAN as a key, any of them plus DOUBLE as a
predicate. A DECIMAL wider than 18 digits is allowed for an expression (`a *
b` is typed wide but holds small values): a value that does not fit fails the
upload's BIGINT cast, the set is never published and the statement stays
native.

Identity: the virtual column's name is a hash of the table and the
expression's SQL, so the same expression in another statement reuses the
lane, and the identity tag lists it like a column. A literal *inside* an
expression (`substr(s, 1, 2)`, a LIKE pattern, the constants of an `OR`) is
part of that identity, while the statement cache normalises literals away —
so a decision that involves such a lane is marked literal-sensitive and each
literal tuple gets its own decision (at most 16 per template; beyond that the
statement runs native). Each distinct expression is its own resident lane:
an application that repeats its statements pays one upload per expression, an
ad-hoc stream of ever-new patterns never becomes resident and runs native
throughout.

Decision-time probes (selectivity, a computed key's bounds and distinct
estimate) read the virtual columns from a derived table `(SELECT *, <expr> AS
"x_..." FROM <table or join>)`; they run once per template.

TPC-H SF1 through the wrapper (M4 Max, identical rows): revenue by order
under a three-table join with date predicates (Q3 shape) 2.2×, revenue by
nation over four tables (Q5 shape) 3.8×, the Q12 `CASE` sums with
column-vs-column predicates 1.9×, `GROUP BY extract(year ...)` with `NOT
LIKE` 4.8×, `substr` key (Q22 shape) 2.3×; the gate's `--exprs` sweep
(`sum(l_extendedprice * (1 - l_discount))`, computed predicates) has 96
rewritten rows at 1.06–7.6×, none below native. Shapes with a handful of
groups over a single table (Q1) stay native under the existing `min_groups`
threshold: the resident reduce walks a huge group serially and loses there.

### 4.11 Expressions over aggregates, compound HAVING
`sum(a) / count(*)`, `100 * sum(x) / sum(y)`, `max(p) - min(p)`, `HAVING
sum(a) > 10 AND count(*) > 5`, `ORDER BY sum(b) - sum(a)` are not aggregates
the device computes; they are projections over aggregates it does. The
wrapper (`python/gpudb/_split.py`) splits such a statement in two: an INNER
GROUP BY that selects every group key (`__k<i>`) and every distinct plain
aggregate (`__g<i>`) the statement mentions, and an OUTER statement — the
original select list, HAVING and ORDER BY with those subtrees replaced by
column references — over `(<inner>) AS gpudb_q`. The inner statement is an
ordinary transparent shape (columns or computed lanes, joins, several
payloads) and goes through the whole existing path; the outer one is plain
SQL that DuckDB evaluates. Because the inner outputs carry native's values
AND types, the outer expressions see exactly what native's would and produce
the same values and types; their names are pinned by aliasing every outer
select item with the original statement's DESCRIBE name. A HAVING that is one
aggregate-vs-constant comparison stays in the inner statement (the device's
HAVING); anything else becomes the outer WHERE. The decision is
literal-sensitive (the outer text carries the literals). Declines: `avg` over
DECIMAL and DOUBLE aggregates inside the expression (the inner declines),
DISTINCT / FILTER aggregates, windows, subqueries, a bare column that is not
a group key.

### 4.12 Aggregates without GROUP BY
`SELECT sum(x), count(*) FROM a [JOIN b …] WHERE …` has one group — TPC-H Q6's
shape. It is not a GROUP BY with a constant key: there is no key at all, so
there is nothing to sort, nothing to permute and no sort cache to build. The
operator is `Aggregator::aggregate_exact_masked(pays, n_pays, preds, n_preds)`,
ONE pass over the rows in storage order: each thread evaluates the whole `WHERE`
program for its row on the (narrow) predicate lanes, folds the surviving row
into its payload accumulators — 128-bit sum, count, min, max per payload, plus
the shared `count(*)` — and the threadgroup reduces them into one partial block
that the host merges. `global_supported()` is its rule-1 gate, as
`exact_supported()` is the exact GROUP BY's. Semantics are exactly one group of
`groupby_exact_masked_multi`: a row failing the mask takes part in nothing
including `count(*)`, a NULL payload cell is skipped by the aggregates and
counted by `count(*)`, a payload with count 0 gives NULL sum / min / max / avg —
which is what native returns over an empty input, so the operator's single row
IS native's answer however selective the `WHERE` is.

SQL: `gpu_agg_exact_global(name, program, payloads)` → `count_star`, then per
payload `sum<p> HUGEINT, count<p>, min<p>, max<p>, avg<p> DOUBLE`; `payloads` is
`'v, i0, i2'` as §4.9 spells it, or `''` for `count(*)` alone; the `program` is
the WHERE program of §4.6. Projection pushdown skips the payload columns the
statement does not read. A set only global statements use carries the tag extra
`global` and writes lane 0 as `-`: the store view synthesised for it has no key
column, so neither `acquire()` nor the wrapper's `gpu_prepare_resident` post-step
sorts a 60M-row lane for nothing. Over a join the set is the join's materialised
result, whose lane 0 is an ordinary lane — it stays, and no operator reads it as
a key.

The statement still goes through the split of §4.11: the inner statement is the
aggregate, the outer selects it, and because a global aggregate's HAVING filters
one row it always sits in the outer statement (no row in, no row out). The outer
form is `FROM (SELECT 1) LEFT JOIN (<inner>) ON true` with `coalesce(count, 0)`,
which is now belt and braces — the operator returns its row whatever happens —
and costs nothing. The C++ statement rewriter has no global form (it matches a
GROUP BY), so a global plan is rendered by the reference renderer alone.

Thresholds. One row out means no output-size risk at all; what decides is how
much work native does per row, because a vectorised filter-and-sum is
memory-bandwidth-optimal on the CPU. Measured on `lineitem` slices of 3M to 60M
rows (M4 Max, warm, minimum of nine): the device wins from about
`rows × (1 + WHERE terms, counting at most three) = 60M` and nowhere below it —
20M × 3 terms 1.39×, 30M × 1 term 1.66×, 60M × 0 terms 1.42×, 60M × 5 terms
(Q6) 2.48× — while 10M × 3 terms 0.93×, 10M × 5 terms 0.97×, 20M × 1 term 0.99×
and 30M × 0 terms 0.99× all lose. `_thresholds.py` takes that bound with a row
floor of 16M under it, so TPC-H Q6 runs on the device at SF10 (15.0 ms native,
6.1 ms transparent, kernel 5.6 ms) and stays native at SF1, where `lineitem` is
6M rows and the two sides measure within noise of each other. Over a join the
comparison is against native's join rather than its scan and the device wins
from SF1 on — 3.9–7.1× at SF1, 10.5–17.7× at SF10 — so there is no floor there.

History: before this operator the shape was forced through the GROUP BY
machinery as a constant key, which built a sort cache and gathered a permutation
for one group and measured 2.9 ms against native's 2.0 at SF1 and 25 against 17
at SF10 — which is why the wrapper's fast path used to skip single-table
statements without a GROUP BY altogether.

### 4.13 Joins the device operator cannot express: upload the join's result
§4.8 covers joins that are "fact rows with dimension columns attached". A
LEFT JOIN, a many-to-many join, `USING`, a composite or non-integer key, an
`ON` clause with more than an equality, a GROUP BY whose keys come from two
tables, an expression that mixes columns of two tables (`CASE WHEN p_type LIKE
'PROMO%' THEN l_extendedprice * (1 - l_discount) …`, TPC-H Q14) are not that.
For them the computed-lane idea (§4.10) is applied to the FROM clause: DuckDB
executes the join — once, inside the background upload — and the resident set
holds lanes of the join's RESULT. Any lane is then an expression over the
joined row, evaluated by DuckDB, so semantics are native's by construction.

`_join.lower_upload` accepts INNER / LEFT joins (explicit or comma syntax) of
base tables in which every table is tied to the others by at least one
equality — a cross product is never uploaded — and lowers the statement to
the same single virtual table as §4.8. Equalities written in `WHERE` move
into the upload's `WHERE`; `ON` clauses stay where they are (a LEFT JOIN's
`ON` predicates decide matching, not filtering) and are validated like
computed lanes. `USING` columns resolve to the left side, which is the merged
column's value for the join types accepted. The upload statement is
`gpu_upload_rows_exact(...) FROM (SELECT <lanes>, <leftmost table>.rowid AS
rowid FROM <the FROM clause> WHERE <equalities>)`: every result row of an
INNER / LEFT join has exactly one row of the leftmost table, so the existing
row-id segmentation uploads the join in idle-time pieces like a table.

Staleness: a set built from a join has no table of its own to count. The
extension gains `gpu_note_rows(tag, n)`, a column-less SENTINEL set that
remembers a base table's row count at upload time; the rewritten statement
carries one `gpu_assert_rows(<sentinel>, (SELECT count(*) FROM <table>))` per
base table (the `guards` context field of §4.8), and the residency manager
makes the uploaded set depend on its sentinels.

Order of attempts: the device join first (its base sets are shared between
statements and re-materialise in milliseconds); when it declines for shape,
the upload; when the select list holds expressions over aggregates, the split
of §4.11 around whichever of the two answers the inner statement. A join
whose result exceeds four times its largest table (or 2^32 rows) is declined
before anything is uploaded (`threshold`).

Metal: a LEFT JOIN can put most rows into the NULL-key group (every unmatched
row has a NULL dimension key); that group is folded on the host, which cost
several ms serially (gate: 0.72–0.90× on such a shape) and is now folded by
up to eight threads. Gate, TPC-H SF1: LEFT JOIN with an `ON` predicate and a
composite-key join, all forms, 60 rewritten rows at 1.03–7.9×. Through the
wrapper: Q14 1.6×, a Q19-style `OR` across two tables 2.2×, the full Q3 with
its three-column GROUP BY over two tables 1.5×.

### 4.14 Nested rewriting: a rewritable SELECT inside a statement DuckDB keeps
`o_orderkey IN (SELECT l_orderkey FROM lineitem GROUP BY l_orderkey HAVING
sum(l_quantity) > 300)` (TPC-H Q18), a derived table that aggregates before
the outer statement groups again (Q13), a CTE, a scalar subquery, the arms of
a UNION: the statement as a whole is not a transparent shape, but a SELECT
inside it is. When the whole statement declines for shape (or reads something
without an identity of its own — a CTE, a view, a derived table), the wrapper
walks the tree top-down, and every SELECT node that aggregates is offered to
the ordinary path as a statement of its own (`json_deserialize_sql` of the
node): its own decision, thresholds, residency, guards and measured check. A
SELECT that is accepted is not descended into; its rewritten SQL is
serialized back and spliced in place of the node, the outer statement stays
DuckDB's, and the result is one SQL text. A correlated subquery does not bind
on its own, so it declines by itself. The composed statement is checked once
against the original with `DESCRIBE` (names and types), cached by exact
statement text (a nested plan is literal-sensitive by construction), and
guarded per sub-statement: a stale set in any of them falls the whole
statement back to native.

TPC-H SF1: Q13 18.4 → 2.0 ms (the inner LEFT JOIN + GROUP BY is an uploaded
join, §4.13), Q18 13.6 → 8.7 ms (the IN-subquery is the device HAVING over
1.5M groups; the rest of Q18 stays native).

`GROUP BY <select alias>` (`SELECT l_suppkey AS supplier_no … GROUP BY
supplier_no`, Q15) is resolved during expression lowering, only when no
column of the statement has that name, so the resolution cannot differ from
DuckDB's.

### 4.15 Wide keys and DECIMAL keys
Up to three integer / temporal keys are packed (§4.4). Four to eight keys of
any supported type, and any key with a VARCHAR or DECIMAL component, are one
hashed tuple with a dictionary (§4.5): every component travels as text
(`CAST(col AS VARCHAR)`, exact for the integer, temporal and DECIMAL types)
and is cast back to its native type on the way out. A `WHERE` on a component
reads that column through its own predicate lane; a single DECIMAL key is no
exception (its key lane holds the hash of the TEXT, which a numeric literal
cannot be compared with) and the pure rewrite function declines such a
`WHERE` when the context gives the column no lane.

The dictionary of a wide key is large (TPC-H Q10 groups by seven customer
columns including the comment: ~100K tuples), and `gpu_resident_dictionary`
copies and sorts all of it per statement — Q10 measured 0.46×. After a device
HAVING or top-k only a handful of keys are left, so those forms decode each
surviving key with the scalar `gpu_resident_dict_component(tag, key, i)`
instead of joining the dictionary: Q10 17.5 → 4.0 ms (4.4×). The plain form
returns every group and keeps the join. Columns without a zone-map distinct
estimate (VARCHAR) get one `approx_count_distinct(hash(keys))` scan per
statement template.

### 4.16 Select-project-join derived tables
`SELECT supp_nation, l_year, sum(volume) FROM (SELECT n1.n_name AS
supp_nation, extract(year FROM l_shipdate) AS l_year, l_extendedprice * (1 -
l_discount) AS volume FROM supplier, lineitem, … WHERE …) AS shipping GROUP BY
…` (TPC-H Q7, Q8, Q9): the derived table only renames and computes columns of
a join — no aggregate, DISTINCT, GROUP BY, LIMIT, window, set operation or
sample — so folding it into the outer statement changes nothing, and the
folded statement is an ordinary shape (joins, computed lanes, the split).
`python/gpudb/_flatten.py` replaces every outer reference to a derived column
by a copy of its defining expression, merges the two WHERE clauses, takes the
inner FROM, and aliases every select item with the ORIGINAL statement's
output name (folding changes auto-names: `sum(vv)` would become `sum((v *
2))`); innermost derived tables fold first, a single CTE that is the FROM
folds the same way. The folded text is checked once against the original with
DESCRIBE and is only used to decide and to build the rewritten statement:
when the rewrite declines, the original text runs.

Q8 joins eight tables, and it exposed the cost of the multi-table staleness
guard: eight `(SELECT count(*) FROM t)` scalar subqueries in one SELECT took
3.8 ms — DuckDB's join-order search over eight one-row relations — against
0.05 ms for one count and 2.7 ms for the whole device operator (Q8 measured
0.96×). The guard is now `SELECT bool_and(ok) FROM (SELECT gpu_assert_rows(tag,
count(*)) AS ok FROM t1 UNION ALL …)`: 0.27 ms for eight tables. Q8 2.5×, and
every multi-table statement gained (Q5 3.6 → 4.8×, Q9 6.7 → 8.2×).

### 4.17 count(DISTINCT x)
No kernel: `SELECT k, count(DISTINCT x), sum(v) … GROUP BY k` becomes an inner
device GROUP BY over (k, x) and an outer statement in which DuckDB counts the
pairs per key — `count(x)` over the inner rows skips a NULL x exactly as
`count(DISTINCT x)` does — and re-aggregates the rest exactly: sum of sums,
sum of counts cast back to BIGINT, min of mins, max of maxs. `avg` over a plain payload does not
decompose and declines. Since 2026-09-18 the pass also takes `sum` / `avg` /
`min` / `max` (DISTINCT x) — over rows unique per (keys, x) the plain
aggregate of the pair column is the DISTINCT one — several DISTINCT columns
(the device groups by the tuple, DuckDB keeps a DISTINCT per column over the
far smaller result) and, over a join, no GROUP BY at all. A single-table
`count(DISTINCT x)` without GROUP BY stays native: DuckDB does it in 4–13 ms
at SF1 and the device would first return every distinct value. Every (key, x) pair travels back through DuckDB, ~0.25 ms per 1K
pairs, so the form has its own bound (`_thresholds.py`): up to 17K pairs
1.3–7.1×, 70K pairs 2.1–2.5× without a WHERE but 0.82–1.13× under a 9–10% one,
700K pairs 0.39–0.51× — declined above 100K pairs, above 20K under a WHERE,
and under a WHERE that keeps less than 5%.

### 4.18 Subquery predicates: EXISTS / IN / a correlated scalar subquery as a lane
A WHERE term that contains a subquery — `EXISTS (…)`, `NOT EXISTS (…)`,
`x IN (SELECT …)`, `x NOT IN (SELECT …)`, `v < (SELECT 0.2 * avg(…) … WHERE
inner.k = outer.k)` — is row-local from the outer row's point of view: its
value depends on that row's columns and on the contents of the tables the
subquery reads. It is lowered like any other computed lane (§4.10): DuckDB
evaluates it once per row during the upload (decorrelating it into a join, as
it would natively), the BOOLEAN result is a predicate lane, and the device
answers `lane = 1`. Three-valued logic needs nothing special: a `NOT IN` over
a set that contains a NULL is NULL for every row, the lane holds NULL, and
`= 1` rejects it exactly as WHERE rejects NULL.

What the subquery may contain (`_exprs._validate_subquery`): a plain SELECT
over base tables — no LIMIT, ORDER BY, sample, QUALIFY, window, DISTINCT ON,
derived table or CTE reference; CONSISTENT scalar functions only; aggregates
whose value does not depend on evaluation order (`count min max sum avg
bool_and bool_or`, and `sum` / `avg` only over non-floating columns). GROUP BY
and HAVING inside are fine (TPC-H Q18). Anything else declines and the
statement runs native.

**Scoping.** The lowerings rename columns (a joined column becomes a lane of
the virtual table; a computed expression is re-emitted in the upload's SELECT).
Inside a subquery only the references that reach OUT may be renamed, and a
renamed outer reference must still not be captured by the subquery's own FROM:
`FROM t o WHERE EXISTS (SELECT 1 FROM t i WHERE i.k = o.k + 1)` rewritten
carelessly to `… WHERE i.k = k + 1` binds `k` to `i` and is true for no row —
a wrong answer with no error. `_scope.mark` resolves each reference the way
SQL does, innermost scope first, and tags the ones bound inside; every outer
reference inside a subquery is emitted qualified with an alias the outer
relation always carries (`gpudb_o` in uploads, DESCRIBE and single-table
probes; `gpudb_j0` in join probes). The test suite holds self-correlated cases
in which every column name exists on both sides.

**Staleness.** A lane computed from another table is stale when THAT table
changes. Every base table a subquery reads gets a sentinel set
(`gpu_note_rows`) and an arm in the statement's guard; the wrapper's own writes
to it invalidate the dependent set; a foreign write trips the guard, the
statement falls back to native, and the set is rebuilt.

**Thresholds.** Native has to run a join for such a statement whatever its FROM
says, so the join bounds apply (`join=True`).

Three fixes this work exposed, all general:
- *Decode choice for dictionary keys.* TPC-H Q18 returns 57 groups keyed by a
  five-column tuple whose dictionary holds 1.5M entries. Joining the dictionary
  cost 416 ms per statement; decoding per key costs 4 ms. With every group
  returned the order flips (1410 ms joined, 2200 ms per key). The wrapper
  passes `decode_per_key` in the context when the rows that survive the WHERE
  number at most a quarter of the distinct key tuples; device HAVING and top-k
  always decode per key (§4.15).
- *Measured rule 1 without a native observation.* A session with eager
  residency never sees the statement run native, so the measured check (§9.1)
  had nothing to compare with. After the first three rewritten runs native is
  timed once, on a cursor of its own, for every template — the short ones
  most of all, see "two modes of a short kernel" in §9.1 — and the usual
  comparison decides; the decision is re-measured every 60 s.

- *A VARCHAR key with few groups under a WHERE.* With ONE expression payload
  the cell is a coin flip (median 1.07×, below 1.0× in a quarter of process
  process starts, on the main branch as much as on this one); from two expression
  payloads on it wins every time (1.08–1.59×, 27 cells). The exemption from
  `min_groups` now requires two (`string_key_min_computed_payloads`);
  `count(DISTINCT)` beside one expression payload keeps its own (1.25–1.84×).

Measured (Metal, SF1, `transparent_gate.py --subqueries`): EXISTS on lineitem
3.1–8.3×, NOT IN 1.4–6.1×, the Q17-shaped correlated scalar 3.3–13.3×, EXISTS
over a join 2.6–36×, IN over a join 1.15–6.4×; 185 rewritten cells, none below
1.0×, 31 declined by the bounds. TPC-H: Q4, Q17, Q21 move to the device and
Q18 becomes one statement — 15 of 22.

### 4.19 Aggregate spellings: FILTER, count_if, bool_and / bool_or
Some aggregates are another aggregate in disguise, and the identity is SQL's own:

    agg(x) FILTER (WHERE c)      =  agg(CASE WHEN c THEN x END)        agg in sum count min max avg
    count(*) FILTER (WHERE c)    =  count(CASE WHEN c THEN 1 END)
    count_if(c)                  =  CAST(count(CASE WHEN c THEN 1 END) AS HUGEINT)
    bool_and(b) / bool_or(b)     =  CAST(min / max (CAST(b AS TINYINT)) AS BOOLEAN)

An aggregate skips NULL inputs and a CASE without ELSE is NULL for the rows the
filter rejects; over no qualifying row sum / min / max / avg / bool_* are NULL
and the counts 0 on both sides (a fixture pins that). `_aggs.normalise` rewrites
the top-level SELECT (select list, HAVING, ORDER BY) before the decision, in
the stage that folds derived tables (§4.16): output names are pinned as
aliases and names + types verified with DESCRIBE, otherwise the text is left
as written. The CASE / cast is a computed lane (§4.10), so no operator changed;
a CAST around the aggregate makes the statement the projected form (§4.11).
Left alone: DISTINCT or ordered aggregates with a FILTER, and anything the
lane rules refuse (a volatile function in the filter).

### 4.20 Views
`FROM revenue0` means `FROM (SELECT …) AS revenue0` — that is what a view is —
so a statement over a view is a statement over a derived table once the
definition is spliced in, and §4.14 / §4.16 apply unchanged. `_views.inline`
replaces every `BASE_TABLE` node that names a view of the CURRENT catalog and
schema with a `SUBQUERY` node holding the view's SELECT (the view's column list
becomes the column aliases, the reference's alias or the view name the alias),
views over views up to four deep. Left as written: a view in another schema (its
unqualified names bind in the schema it was created in), a name that is both a
table and a view, a view whose body is not one SELECT (a set operation), and
anything the DESCRIBE check (names and types) rejects. The fast path treats a
view that names a big table as naming it, and a statement that names a view as
possibly aggregating.

A view is the one object whose text can change under a session without any
table changing, so a statement built on views remembers their definitions and
re-reads `duckdb_views()` for those names on every run (~0.2 ms): a view
redefined from any connection — or replaced by a table of the same name — is
noticed on the next statement, which is rebuilt from the new definition.

### 4.21 Shorthand: GROUP BY ALL, ordinals, ORDER BY ALL
DuckDB's shorthand is spelled out before the decision (`_syntax.normalise`), by
DuckDB's own definitions: `GROUP BY ALL` becomes every select item that holds no
aggregate; `GROUP BY 1, 2` and `ORDER BY 3 DESC` become the select items they
stand for; `ORDER BY ALL` becomes every select item in order. Two more identities in the same pass: `SELECT DISTINCT a, b` (no aggregate,
no GROUP BY) is `GROUP BY a, b`, and `A RIGHT JOIN B ON c` is `B LEFT JOIN A ON
c` (column order is not observable: `SELECT *` is never rewritten). ROLLUP / CUBE /
GROUPING SETS, `DISTINCT ON`, `SELECT *` and an ordinal past the select list are
left as written. Names pinned and verified with DESCRIBE, as §4.19.

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
2. An unqualified name is looked up in `duckdb_tables()` and
   `duckdb_views()` across the temp catalog and every catalog and schema on
   the search path (`current_setting('search_path')`, `current_database()`,
   `current_schemas(true)`). Temp objects live in the `temp` catalog with
   `temporary = true`; a `register()`ed DataFrame/Arrow object is a
   non-temporary **view** in the current catalog (checked 2026-09-03). The
   rule is conservative: the statement is rewritten only when exactly
   **one** object of that name exists across all of those, and it is a base
   table. Two candidates (a temp table shadowing a base table, a view beside
   a table in another catalog), a view outside the current schema, or no hit
   at all: native (a view of the current schema is inlined first, §4.20). This
   avoids reproducing the binder's precedence rules.
3. A qualified name is checked the same way against its catalog and schema.
4. The result is `(catalog, schema, table, table oid, column names)` — the
   **identity** — cached with the template together with the search path
   and catalog list that produced it.

Any statement that is not a `SELECT` or `EXPLAIN` (§5.2) clears every
cached resolution on that connection, as do `register()` / `unregister()`
in the client; the splitter's types are too coarse to single out the
catalog-changing ones (`USE` is typed `SET`), and clearing is cheap. The
rewritten statement always names the table fully qualified.

### 5.2 Statement classification — DuckDB's own splitter
`json_serialize_sql` serializes only `SELECT` statements; every other kind
returns an error, so DML cannot be recognised "on the tree". The wrapper
instead uses the classifier DuckDB ships in every client: in Python
`con.extract_statements(text)` returns one object per statement with its
`type`; in the C API `duckdb_extract_statements` +
`duckdb_prepared_statement_type`. Every incoming string is split; each
member is handled by type:

- `SELECT`: the rewrite path (§3.2). `SHOW`, `DESCRIBE`, `SUMMARIZE`,
  `VALUES` and `FROM t` are typed `SELECT` too; `json_serialize_sql` may
  refuse some of them, and a refusal simply means native.
- `EXPLAIN [ANALYZE] <select>`: the inner statement is rewritten and
  re-prefixed, so users can see the resident operator in the plan.
- `TRANSACTION` (one type for `BEGIN`/`COMMIT`/`ROLLBACK`): invalidates
  every set, then the first keyword decides open (`BEGIN`, `START`) or
  closed (`COMMIT`, `END`, `ROLLBACK`, `ABORT`); anything else is treated
  as open, the direction that never rewrites (§5.4).
- **Anything else** — `INSERT`, `UPDATE`, `DELETE` (also `TRUNCATE`),
  `MERGE_INTO`, `COPY`, `CREATE`, `ALTER`, `DROP`, `ATTACH`, `DETACH`,
  `SET` (also `USE`, `PRAGMA`, `RESET`), `CALL` (also `CHECKPOINT`),
  `PREPARE`/`EXECUTE`, `LOAD` (also `INSTALL`), `VACUUM`, `EXPORT`, and
  any type added by a later DuckDB — invalidates **every** resident set on
  that database before the statement runs and clears the resolution cache.
  The parenthesised aliases are how DuckDB 1.4.5 actually types those
  statements (checked 2026-09-03). Conservative on purpose: a miss runs
  native, so over-invalidation costs at most a re-upload, never an answer.

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
        (SELECT gpu_assert_rows('<identity tag>', count(*)) AS ok FROM main.lineitem) gd
   WHERE gd.ok
   ORDER BY q DESC LIMIT 10
   ```

   `gpu_assert_rows` is a volatile C-API scalar
   (`duckdb_scalar_function_set_volatile` exists in the vendored v1.2.0
   header) returning BOOLEAN `true`, or raising a typed error when the count
   differs from the set's upload count. Two properties make this form
   correct, both established on 2026-09-03 with an `error()`-based
   stand-in on the real Metal resident function (2M rows, 100K groups):

   - The raising expression sits **inside** the one-row derived table, so it
     is evaluated in that subquery's own pipeline, once, before the cross
     product emits anything — with rows, with an **empty** resident result,
     under `fetchmany(1)` streaming, with `threads=1`, and with the optimizer
     disabled, the stale case raised every time and nothing reached the
     client. (A guard whose call is in the `WHERE` of the outer query is
     evaluated per output row and never fires on an empty result.)
   - `gd.ok` is **referenced** by the outer `WHERE`. Unreferenced, the
     optimizer's unused-column removal deletes the projection that carries
     the call, the plan contains no guard at all, and a stale set passes
     silently — the form in the first revision of this document had exactly
     that hole. The parity harness asserts on `EXPLAIN` that the `FILTER` on
     `ok` is present in every rewritten plan.

   `count(*)` runs inside the querying transaction, so it is the snapshot
   rule with no transaction introspection. Measured cost at SF1 (6M rows):
   0.18 ms clean, 0.47 ms after deleting 10% of rows; the whole guard adds
   0.3–0.5 ms to the rewritten statement. The §9.1 floor absorbs it. On the
   typed error the wrapper marks the set stale and re-runs the original text
   natively; on any other error it surfaces (§3.3).
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

What is not caught: the same-count write from a connection the wrapper did
not open (§0). Everything else — any write through the wrapper, any write
through any connection that changes the count, any catalog change — is
caught by one of the four layers, and the write scenarios in §9.2 exercise
each one.

### 5.5 What becomes resident, and when it is ready
When the wrapper sees a rewritable shape over a resolved table above the
§9.1 floor whose columns are not resident, it records the template and runs
the statement **unchanged** (rule 1: the first query pays only the round
trip). The upload runs on the wrapper's own second connection to the
same database — the extension stays free of threads and hidden connections
— under these rules:

- **Idle segments, never beside a user statement.** One DuckDB process has
  one worker pool, so any scan on the second connection slows a statement
  running at the same time on the first — DuckDB's own `list()` over the
  same columns costs a concurrent native query 25–33% at p99 (§5.6). A
  single long upload scan therefore violates rule 1 for every statement the
  user runs during it, however cheap the callbacks are. The upload is
  instead a sequence of short statements, each over one segment of the
  table (`WHERE rowid >= a AND rowid < b`, the 8 MiB host segment the
  extension already buffers, a few ms of scan), appended to an open upload
  session in the extension (`gpu_upload_begin(tag)`, the aggregate with the
  same tag, `gpu_upload_finish(tag)` → device copy + `prepare()`). The
  wrapper issues a segment only when none of its connections has a
  statement in flight and the connection has been idle for at least the
  idle threshold (default 20 ms, a wrapper setting); a user statement that
  arrives mid-segment overlaps it for at most one segment, and the wrapper
  calls `interrupt()` on the upload connection when it sees the statement
  arrive, so the overlap is bounded by DuckDB's interrupt latency (checked
  between vectors) rather than the segment length. An interrupted segment
  is re-run; progress before it is kept: an append is one atomic step at
  the aggregate's finalize, so an interrupted scan never appends, and the
  wrapper reads `gpu_upload_status(tag).segments` after every interrupt to
  learn whether the segment landed just before the interrupt was seen (it
  does happen) rather than appending it twice. Consecutive interrupts pause
  the session — 50 ms, doubling per interrupt, capped at 1 s, reset by a
  landed segment — so a cadence whose statements keep landing on segments
  pays the interrupt latency at most once per pause, not once per
  statement. (Raising the idle threshold instead was tried first and
  stalled a session under 0–10 ms gaps: a threshold above the gaps is never
  met, a pause always expires.) `gpu_upload_finish` is one scalar call an
  interrupt cannot stop; an interrupt seen after it returns is recognised by
  the set being `ready` in `gpu_residents()`, not treated as a lost session.
  A session that never goes idle never uploads and runs native throughout
  — rule 1 holds, the win is simply not there yet.
- **Quiet period and rate cap.** No upload session starts within 2 s of the
  last invalidation of that table, and no more than one session per table
  per 30 s (wrapper settings). A write-heavy session therefore runs native
  rather than re-uploading after every write.
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
- **Memory budget.** Wrapper setting `memory_budget` (bytes, or `'16GB'`;
  `0` / `'unlimited'` removes the cap; `GPUDB_MEMORY_BUDGET_MB` overrides the
  default). Default: 25% of unified memory on Apple silicon — the GPU shares
  it with DuckDB and everything else on the machine — and 50% of device
  memory on a discrete GPU (until the backend reports its device memory
  through `gpu_build_info()`, a discrete GPU gets the smaller of 25% of host
  memory and 8 GiB). Implemented in the wrapper (`_residency._make_room`),
  with the extension as the source of truth:
  - *Before an upload* the set's cost is estimated as `rows × (Σ lane widths +
    key width + 4 + 8) + rows × lanes / 8`: an upper bound on each lane from
    the DuckDB type of the column it holds (BOOLEAN / TINYINT 1, SMALLINT 2,
    INTEGER / DATE 4, everything else 8 — a backend stores a lane at the
    narrowest width its values fit and the wrapper cannot know that before the
    upload, `docs/RESIDENT_COLUMNS_DESIGN.md` §6), a validity bit per row and
    lane, the key lane's sort cache (its width plus a u32 row id) when that
    lane is the one being uploaded, and one row-sized scratch lane. On a
    backend that does not store lanes narrow (`gpu_build_info()` says
    `narrow=false`) every lane is charged 8; a set with no key (§4.12) is
    charged no sort cache. It is an UPPER bound by construction — that is what
    the admission rule needs — and a wrapper test pins it against what
    `gpu_residents()` / `gpu_store_columns()` report afterwards, on a BIGINT
    table and on a narrow-typed one. An uploaded join counts
    its result rows, a device join at most its probe table's.
  - *What is resident and what it costs* comes from `gpu_residents()`
    (`bytes` includes derived structures). Sets uploaded by hand count toward
    the total and are never evicted.
  - *Eviction* is least recently used by the extension's own `last_used_at`,
    through `gpu_drop_resident` — and, since the store (§5.10), through
    `gpu_drop_column` for the columns views share: a view costs nothing, its
    columns are what the budget counts (`gpu_store_columns()`), and a column
    goes when none of its views was used recently. Never evicted: a set an operator is using
    (`refs > 0`), a source of the set being uploaded, a source of another
    resident set (the derived set goes first — dropping a base table's set
    from under a join would turn every guard of that join stale), and any set
    uploaded less than 60 s ago (anti-thrash: two sets that do not fit
    together must not evict each other on alternate statements).
  - *A set that does not fit* — larger than the budget, or nothing evictable
    yet — is not uploaded. Its statements keep running on DuckDB with
    `last_rewrite()["reason"] == "memory"`; a refused set is asked about again
    no sooner than the failed-upload retry time (30 s), not on every sighting.
  `GPUDB_UPLOAD_POOL_MAX_MB` stays what it is — the extension-side cap on
  **host** upload buffering — and is documented as such.
- **Upload cost (measured 2026-09-17).** A statement's time on the device is
  kernel-bound (wrapper ≈ 0, guard 0.2–0.5 ms, table function ≈ the whole
  statement); the upload was not: 30–58 ms for two 6M-row lanes that DuckDB
  itself materialises in 5–8 ms. Three changes, none of them the C API:
  the Metal copy from staged rows into device buffers runs per span in
  parallel with a NULL-free fast path (prefix sums give every span its output
  ranges, so the layout is byte-identical); the sort cache takes the sorter's
  output buffers instead of copying them and stages in parallel
  (`MetalRadixSort::sort_iota_take`), and the sorter no longer keeps gigabytes
  of staging alive after a large sort; and the host staging of an EXACT upload
  has its own cap (`GPUDB_EXACT_UPLOAD_POOL_MAX_MB`, default half of physical
  memory) — the general 4 GB cap, a guardrail against window-frame buffering,
  had kept every set above ~250M row-lanes off the device (TPC-H SF50: 0 of 22
  resident before, 10 of 22 after).
  What the CUDA backend has to implement to join this path, and the tests that
  prove it, is listed in `docs/CUDA_EXACT_PATH.md`.
- **When the rewritten statement itself fails.** Staleness was the only
  error the wrapper recovered from. Any other error of the rewritten form — a
  device allocation that fails at query time despite the budget, a set
  dropped behind the wrapper's back, a defect — is now answered by running
  the user's ORIGINAL statement on DuckDB (rule 2: if that raises too, it is
  DuckDB's own error for the user's own statement), the template stays native
  afterwards (`reason == "error"`), and the text is kept in
  `last_rewrite()["error"]`. A user interrupt is not retried.
- `residency = 'manual' | 'eager' | 'background'` is a wrapper setting:
  `'eager'` uploads on first sight (for scripts that know their workload),
  `'manual'` restores v0.6 behaviour (`gpu_upload_pair` only, no managed
  sets, no rewrite).

### 5.6 Concurrency — prerequisite of the background path
Before milestone 0b, `gpu_resident.cpp` took the global mutex inside
`upload_update` and `upload_combine` (DuckDB's per-thread aggregate
callbacks) and incremented one process-wide atomic (the host pool-cap
accounting) **per row** from every scan thread. Measured on SF10: Q18-inner
798–811 ms against an 80 ms baseline (0.10×) while an upload ran, and small
statements 1.5× slower for the whole window. Milestone 0b (PR #84, Linux
gate on the 4090, 2026-09-03) found that the per-row atomic, not the mutex,
was the starvation: removing it alone took a 60M-row upload from 1.8 s to
0.36 s and the concurrent native Q18 from 575–1450 ms spikes to
311–327 ms. The mutex removal, the 8 MiB segments and the device-side split
each mattered for footprint (host peak 2.8 GB → 0.9 GB, finalize
478–1018 ms → 172–232 ms), not for starvation. The structural changes:

- per-thread upload state with **no lock in `update`**; the lock is taken in
  `combine`/`finalize` only (already the contract for source states); pool
  accounting batched per chunk, never per row;
- a short registry lock for lookup only; per-set state
  `{uploading, prepared, ready, stale, epoch, completion event, refcount}`;
- upload and prepare work on their own device stream (CUDA stream; Metal
  command queue) outside any global lock; a per-set (not global) mutex
  around the sort-cache build;
- an operator call holds a **reference** on the set for the duration of the
  call instead of the global lock; eviction waits for the refcount to reach
  zero, so LRU can never free a column mid-query.

This is shared-extension plus CUDA-backend work carried as one
`feat/core-*` PR from the Linux instance (milestone 0b, #84); the Metal
backend implements `prepare()` and the interleaved pair upload on its side
(#85).

**What the fixed upload still costs a neighbour, and why segments.** With
the callbacks fixed, a native statement on the other connection during a
whole-table upload measured 0.98× (Q18 inner), 0.94× (point lookup) and
0.82× (a 14 ms scan) of its p99 during a *control* — DuckDB's own `list()`
over the same two columns, no gpudb code — and that control itself costs
the native query 0.67–0.75× at p99 (a native `GROUP BY` 0.62×,
`count(DISTINCT)` 0.42×). One worker pool: any concurrent scan is a loss
for whoever shares it, and "≥ 1.0× at p99 during a background upload" is
not achievable in-process by any implementation. Swap, allocation, prepare
(41–51 ms on its own stream) and finalize were each ruled out as the cause.
So the upload is not allowed to be concurrent with the user's statements at
all: §5.5's idle segments with `interrupt()` bound the overlap to the
interrupt latency, and the gate row (§9.3) measures the user's statements
with the residency manager active, not a statement beside a running scan.
`scripts/residency_gate.sh` (from #84) remains the regression gate for the
callbacks themselves: the upload must stay no worse a neighbour than the
`list()` control, and its own duration is printed so the per-row atomic can
never come back unnoticed.

### 5.7 Shared with joins
A resident set is `{identity, columns, sorted key permutation, validity,
state}`. The v0.5 join build side is the same object, which is what lets
v0.8 route joins through the same manager (§10).

### 5.9 Foreign writes: the file is the change log
The staleness guard (`gpu_assert_rows`) compares row counts, so a write from
another connection that keeps the count — an in-place `UPDATE`, a delete and
an insert of the same size — passed it, and the device answered from stale
values. Found and closed 2026-09-18. DuckDB has no table version to read, and
its storage metadata (`pragma_storage_info`, whose `has_updates` and block
positions would tell) costs 2.8 ms at SF1 and 175 ms at SF50 per look — not a
per-statement guard. What is: the database file and its write-ahead log. Every
committed write appends to the WAL (or, at a checkpoint, rewrites the file);
reads and rolled-back transactions touch neither; and only connections of the
same process can write a DuckDB file that is open, and they all share that WAL.
So before every rewritten statement the wrapper stats the database files and
their `.wal` (2–3 µs): a change since the last snapshot means some connection
committed a write, every resident set is dropped and the statement runs native
(`reason == "not_resident"`); the next statement rebuilds. The wrapper's own
writes re-take the snapshot after they run, so they cost no second
invalidation. Cursors of one connection share the snapshot.

Not covered: an in-memory database (no file — writes made through a raw
`duckdb` cursor rather than the wrapper's `cursor()` are invisible; the
wrapper's own cursors are seen), and a read-only connection watches nothing
because nobody can write the file while it is open read-only. A checkpoint
from another connection looks like a write and costs one rebuild.

### 5.10 Resident columns: the store
The storage design that replaces per-statement sets with per-table columns —
one resident copy per column in row-id order, sort caches per key column, joins
as index vectors, chunks for appends and for tables above the budget — is
`docs/RESIDENT_COLUMNS_DESIGN.md`. Stage A (2026-09-18) changed the exact
columns' layout: rows stay in input order and a NULL key is a bit in the key's
validity bitmap, no longer a trailing block. Stage B (2026-09-18) is the
store: `gpu_upload_columns` uploads a table's lanes by name in row-id order
into its `TableStore`; a statement's set — and a device join's base set — is
a *view* synthesised from the store's columns on first use, sharing the
columns, the dictionaries and the key's sort cache with every other statement
on the table. The wrapper asks `gpu_store_columns()` what the table holds and
uploads only the missing lanes (`_rewrite.store_lanes` / `store_upload_sql`,
`connection._store_upload`); the budget counts and evicts columns
(`gpu_drop_column`) and a view whose lane went is re-synthesised once the lane
is back. Nothing in the rewrite, the guards or the gate changed; every suite
and the gate are the proof. Join results are still copies — stage D.

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
   cross-joined to the `gpu_assert_rows` derived table with `WHERE gd.ok`
   (§5.4). Every remapped output is wrapped in a `CAST` to the native type
   (§4.2) and carries the user's alias, or, when unaliased, DuckDB's
   auto-generated name (`sum(l_quantity)`, `count_star()`); names and types
   come from one `DESCRIBE` of the original statement per template, run by
   the wrapper and passed in `context`. `HAVING` thresholds
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
   Two forms the reference renderer and the scalar both produce, settled
   while building them (2026-09-03): a `HAVING count(*) <cmp> n` beside
   `sum(v)` uses the SUM function and carries the count predicate in the
   outer `WHERE (gd.ok AND r.count <cmp> n)`; an `ORDER BY <ordinal>` is
   resolved to the select-list item and pushed as top-k when it names the
   aggregate with a known direction and a `LIMIT`. Reason order in the
   scalar: shape, backend, nulls, overflow, threshold, not_resident — a
   shape miss is reported as `shape` even when the set is not ready.
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

Findings update this document before implementation. Status 2026-09-03:
item 1 done (every construct and rejection serialized and round-tripped;
fields recorded in §2; the `HAVING` literal `300.5` arrives as
`DECIMAL(4,1)` value `3005`, confirming the rescale in §6); item 2 done for
the evaluation-order and cost questions (§5.4) — the C-API scalar itself is
milestone 0b; item 3 done for classification (§5.2) and resolution sources
(§5.1), open for the wrapper's cache and transaction handling until the
wrapper exists; item 4 done (§4.2).

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
  deleted rows, at each SF (SF1 on the M-series: 0.18 / 0.47 ms for the
  count, 0.3–0.5 ms for the whole guard — re-measured 2026-09-18 on DuckDB
  1.4.5 at **0.05 ms**: `EXPLAIN ANALYZE` of the rewritten statement shows
  the guard arm as a `COLUMN_DATA_SCAN` of one row, because an unfiltered
  `count(*)` over a base table is answered from row-group metadata and never
  scans. The 0.18 / 0.47 ms figures stand for the case the optimizer cannot
  do that for);
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

**Two modes of a short kernel (measured 2026-09-18).** A 2 ms exact GROUP
BY kernel on the M4 Max runs at 2.0 ms or at 6.5 ms — every stage 3× slower,
`GPUStartTime`→`GPUEndTime` — depending on what the rest of the *process* is
doing: with DuckDB's idle worker threads waking every few milliseconds
(`threads = 16`, the default) the slow mode sets in after native queries and
stays; `SET threads TO 1` restores the fast mode at once; a fresh process is
fast while the slow one idles beside it; 15 Python threads that merely sleep
5 ms in a loop turn the fast mode into the slow one (2.16 → 6.78 ms) and back
(2.14 ms). A bandwidth-bound kernel is unaffected (0.4 ms either way); the
device idling a few seconds has the same effect for a few runs. This is the
SoC's power management, not the extension: it is the same on `main`, with or
without the store, in the embedded CLI without Python. What it means for rule
1: the hot-loop minimums the gate reports for statements under ~10 ms are the
fast mode, and a shape that wins 1.8× there loses (0.85×) in the slow mode.
So the thresholds are not the last word — the process is: the wrapper times
native once per template after its first three rewritten runs (a side cursor,
whatever the statement's size), declines the template when the rewritten
runs are not faster, and re-measures every 60 s — native for a kept template,
the rewritten form for a declined one — so the answer follows the process's
state. The user's own statement is never the experiment. In the gate a row
that measures slower in the slow mode reads `declined after the first run
(threshold)`: no ratio, and no statement ran slower for a user.

**A faster kernel is not a looser threshold (2026-09-18).** The direct grouped
reduce (§4.1, `docs/RESIDENT_COLUMNS_DESIGN.md` §7) makes a few-group exact
GROUP BY 1.65× to 8.57× faster at SF10, which is the obvious moment to ask
whether `min_groups` and the VARCHAR-key rules were set against a cost that no
longer exists. The sweep says no, and the reason is worth keeping: these bounds
are group counts, so they apply at every scale factor, and the scale factor
they have to hold at is the small one. At SF1 a 7-group aggregate over 6M rows
is 1.5–2.5 ms of native and about 2.5 ms of wrapper round trip; the reduce is
not what it is bound by, and the gate with `--no-thresholds` measures those
shapes at 0.80–0.94× with no WHERE and 0.52–0.55× under a 9% one, direct path
and all (the first 56 rows of that sweep; it was stopped there, the machine
being needed for the gate proper). A bound that admitted them would admit them at 0.5×. So the thresholds
are unchanged and the win lands where the shapes are already admitted — the
SF10 and SF50 end of the same rows.

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
BENCHMARK.md unedited. The **residency-active** rows are in this table
too: each native shape run through the wrapper with the residency manager
uploading in idle segments (§5.5) versus the same shape with
`residency='manual'`, reported as p99 and max over N ≥ 200 statements
issued at the cadence of an interactive session (gaps of 0–50 ms, so
segments do get scheduled between them), and must be ≥ 1.0× within the
noise band the control rows establish. This is the row that proves the
background work never steals from the user's statements; min-of-N is
printed but never used, because it hides contention entirely (it reported
80 ms inside the 800 ms window in §5.6). These rows are produced by
`scripts/wrapper_residency_gate.py` (manual pass, background pass with the
upload of `lineitem` scheduled by a first sighting, manual pass again for
the noise band; per pass the segment count, per-segment scan time,
interrupt count, time-to-ready and the statements that started inside the
`gpu_upload_finish` call). Metal, SF10, 2026-09-04: Q18-shape native p99
84.2/88.7 ms manual vs 85.3 ms background (1.04×), small scan 12.4/13.7 vs
11.6 (1.18×), point lookup 1.9/2.0 vs 2.0 (1.00×, max 3.4 vs 2.8); the set
was ready 2.5–7 s into each run after 115 segments of 2–14 ms scan; the
313 ms finish call had 12 point lookups start inside it at 0.4–0.5 ms.
CUDA (RTX 4090, SM 1815 MHz), SF10, 2026-09-04, same script and branch:
Q18-shape 213.7/212.5 vs 214.9 ms (0.99×, inside the band), small scan
14.1/14.5 vs 14.4 (1.01×), point lookup 1.9/1.8 vs 1.4 (1.31×); 115
segments of 3.8–9.7 ms scan, 48–56 interrupts; the finish call (211–484 ms:
one `cudaMemcpyAsync` of the held segments plus the sort, on the column's
own stream, no device lock a query needs) had 2–10 statements start inside
it at their baseline latency, and the slowest statement of every run fell
outside it. The Q18 run took 19 s to go ready because 0–50 ms gaps rarely
clear the 20 ms idle bar beside a 200 ms statement — expected: a session
that cannot find idle time waits, it does not intrude.

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
| 0c | Upload session in the extension (§5.5): `gpu_upload_begin` / segment append to an open session / `gpu_upload_finish` → device copy + `prepare()`, rows_seen accumulated across segments, discarded on epoch change; Linux instance, Metal follows | registry test rows for begin/append/finish/interrupt; unit + parity unchanged |
| 1 | Spike remainder (§8): tree shapes and rejections, `gpu_assert_rows` evaluation-order proof, classification/cache/invalidation, output names and types; this doc updated | findings written down |
| 2 | `gpu_rewrite_ast` (pure, context-driven) for the plain `GROUP BY` shape + the Python wrapper (classification, resolution, template cache, transaction rule, typed fallback, settings), three-way parity on the v0.6 operators as they are | PR |
| 3 | Exactness (§4): CPU reference first, then Metal + CUDA in parallel; native output types and names; unit + parity | per-kernel PRs |
| 4 | Residency manager in the wrapper (§5.5): budget, eviction, quiet period, epoch, all write scenarios in parity | PR |
| 5 | Full §2 shape in the rewriter, `HAVING` rescale, kept `ORDER BY` + top-k push, `last_rewrite()`, `gpu_residents()` stats; Node/R/JDBC wrappers | PR |
| 6 | Gate (§9): sweeps, floor and thresholds per backend, `transparent_gate.sh` all ≥ 1.0× on both machines (first-sighting row within its bound), BENCHMARK/KNOWN_ISSUES rows | PR; gate must pass |
| 7 | Audit (v0.6 shape), tag v0.7.0, registry bump, `pip` package release (per-platform assets; `allow_unsigned_extensions` requirement stated) | user go per step |
| — | Upstream proposal for a C API statement-rewrite hook (§3.5) | drafted after milestone 2 proves the interface; user sign-off before anything is posted |

Milestones 0b and 0c precede everything that uploads in the background. Milestones
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

**Milestone-1 spike, 2026-09-03 (Python `duckdb` 1.4.5 + the v0.6.0 Metal
build):** the cross-join guard as first written was pruned by the optimizer
and passed a stale set silently; the referenced form (`WHERE gd.ok`) raises
in every case including the empty result and streaming fetch (§5.4).
`avg` is DOUBLE for every input and native computes it exactly, so it stays
on the transparent path (§4.2). The splitter's statement types are coarser
than the SQL keywords (§5.2). `register()` creates a non-temporary view;
temp tables live in the `temp` catalog (§5.1). One `DESCRIBE` per template
gives native names and types at 0.08 ms (§4.2). Every §2 construct and
rejection has a distinguishing serialized field (§2).

**Milestone 0b gate, Linux, 2026-09-03:** the starvation was a per-row
atomic, not the mutex; with it fixed, a neighbouring native statement still
loses 2–18% at p99 to a whole-table upload because one process has one
worker pool — the same loss DuckDB's own `list()` inflicts. The user ruled
that loss unacceptable, so the upload became idle segments with
`interrupt()` (§5.5) and the gate row became "native shapes with the
residency manager active ≥ 1.0×" (§9.3); the extension gains an upload
session (milestone 0c).

**Milestone 0c, both machines, 2026-09-04:** upload sessions landed in the
extension (Linux: SF10 `lineitem` as 115 segment statements of 4.7/6.2/12.9
ms min/mean/max scan each, finish 285 ms) and the wrapper's residency
manager moved to them (Metal and CUDA: the gate rows above, all within
the band at p99, finish invisible to neighbours on both; a
20-segment session completes while statements flow with 0–10 ms gaps with
`rows_seen` equal to `count(*)` exactly; a write mid-session drops the
open session and the set comes back after the quiet period). Milestones
0b, 0c and 2 are therefore in: #85, #86 merged, #87 rebased and green, #88
and the segment-mode manager on their branches.

**Decided by the user, 2026-09-03:** the bounded first-sighting cost (§0)
and the out-of-process same-count write as the one documented gap (§0),
each over the stricter alternative that would have made the feature
impossible (no rewrite without a prior native timing; a full fingerprint
scan on every hit).
