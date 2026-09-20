# The `gpudb` shell

The first of the three ways in, and the one that shows its work: a SQL shell
whose footer says, under every result, where the statement ran and why. It
speaks to the same extension on Apple Silicon Metal and on NVIDIA CUDA.
[← back to the README](../README.md) · [the Python way](USING_PYTHON.md) ·
[installing](INSTALL.md)

## How to tell it is working

Two commands, in order. From the shell:

```
gpudb
```

and read the `backend:` and `transparent:` lines of the banner. `backend:
none — the extension is not loaded` means the extension is missing: install
it into DuckDB with `INSTALL gpudb FROM community;`, or point
`GPUDB_EXTENSION_PATH` at a binary you built ([Installing, in
full](INSTALL.md)). `transparent: available` means both pieces are
in place.

Then run a statement over a table of at least a million rows and read the
footer. `GPU (…)` is the GPU; `DuckDB (not_resident: …)` means it is on its
way there and the next ask will be; `DuckDB (threshold: …)` or `DuckDB
(shape: …)` mean the wrapper decided against it on purpose, and `.gpu` says
exactly which rule.

From SQL, `SELECT gpu_build_info();` tells you what any binary carries:
`compiled=` the backends it was built with, `runtime=` the one it picked,
`exact=true` that it has the operators the transparent path needs.

## Start it

```bash
gpudb                             # in-memory database
gpudb my.duckdb                   # a file
gpudb my.duckdb --readonly        # nobody writes while you look
python -m gpudb                   # the same entry point
```

The banner is the machine's answer to "is this actually going to use the GPU".
Every transcript in this section comes from one session on an M4 Max over
TPC-H SF1, opened read-only, default settings, DuckDB 1.5.5, 2026-09-20:

```
$ gpudb data/tpch_sf1/tpch.duckdb --readonly
[gpudb] registered gpu_inner_join + gpu_join_rows_resident
[gpudb] registered gpu_groupby_{sum,sum_f64,count,exact}_resident[_having|_topk] + gpu_topk_resident[_f64]
[gpudb] registered gpu_sum / gpu_min / gpu_max streaming aggregates + resident-column functions (gpu_upload, gpu_*_resident) (backend=Metal)
gpudb 0.7.0
backend:       Metal · Apple M4 Max · 51.8 GiB device memory
transparent:   available — every statement goes through the wrapper
database:      data/tpch_sf1/tpch.duckdb
Enter .help for usage.
```

Those first three lines are the extension announcing, on stderr, which
function families it registered and which backend it found. The extension
prints them itself as it loads, so you will see them from the stock CLI too.

`backend:` names the runtime, the device as the driver reports it, and the
device memory the budget plans against. A build with no GPU backend names no
device, and `transparent:` says what is missing instead.

## Type SQL, read the footer

Under each result is one dim line: where the statement ran, why, and how long
it took. The five parts with the most units shipped, out of 200,000:

```
gpudb> SELECT l_partkey, sum(l_quantity) AS qty FROM lineitem GROUP BY l_partkey ORDER BY qty DESC LIMIT 5;
┌───────────┬───────────────┐
│ l_partkey │      qty      │
│   int64   │ decimal(38,2) │
├───────────┼───────────────┤
│    125009 │       1642.00 │
│    140633 │       1562.00 │
│     49981 │       1553.00 │
│    149443 │       1537.00 │
│     10426 │       1513.00 │
└───────────┴───────────────┘

DuckDB (not_resident: the resident set is not ready yet) · 24.1 ms
```

**That first line is not a failure, and it is the one thing worth understanding
before anything else.** In the default `background` residency the wrapper
records the statement, leaves it with DuckDB, and uploads the columns it needs
in short row-id segments taken only while your connection is idle — so an
upload never runs a long scan beside a query you are waiting for. Ten seconds
later the same statement is on the device, and `.residents` (below) is how you
see that it got there:

```
gpudb> SELECT l_partkey, sum(l_quantity) AS qty FROM lineitem GROUP BY l_partkey ORDER BY qty DESC LIMIT 5;
… the same five rows …
GPU (topk: the resident GROUP BY) · 39.2 ms
```

That run is still the first one through the device pipelines. Nine more of it
against nine with the path off, same session, same connection, is the A/B:

```
gpudb> .gpu off
GPU path off — statements go straight to DuckDB.
… 9 runs …
DuckDB (off: the transparent path is off on this connection) · 20.7, 17.4, 15.5, 15.1, 14.8, 14.8, 14.9, 14.8, 14.8 ms
gpudb> .gpu on
GPU path on — residency: background.
… 9 runs …
GPU (topk: the resident GROUP BY) · 9.6, 32.3, 8.7, 8.7, 8.7, 8.7, 8.5, 8.4, 8.4 ms
```

**Median 14.9 ms on DuckDB against 8.7 ms on the GPU — 1.7×.** Both series are
printed whole on purpose: DuckDB's first three runs are its own warm-up, the
GPU's first is the pipeline's, and the 32.3 ms is the wrapper measuring this
template against native on a side cursor, which it does once and then every 60
seconds. The 24.1 ms at the top of the section is the true cost of asking
before the column was there.

Those numbers are one machine's *in one state*. A statement this short has two
speeds on Apple silicon — this same statement reads 5–6 ms on a quiet machine
and 8–9 ms when other threads are waking, whatever else is running, and the
journal entry *Two modes of a short kernel* has the experiment that pins it
down. That is exactly why the wrapper measures in your process rather than
trusting a published ratio, and why the footer prints a time at all: the five
measurements above are ones you can take on yours.

If you would rather pay the upload now — a script that knows its workload, or
a benchmark — start with `--residency eager` and the upload happens inside the
statement that asked for it (which therefore takes as long as a scan of the
column — 73 ms for this one at SF1, measured by starting the same
session with `--residency eager`). `--residency manual` never uploads by
itself; only sets you uploaded by hand are used, and ["Uploading by
hand"](#uploading-by-hand) is how.

Statements the GPU does not take say so in the same place. Printing them is the
point:

```
gpudb> SELECT l_returnflag, median(l_quantity) FROM lineitem GROUP BY 1 ORDER BY 1;
┌──────────────┬────────────────────┐
│ l_returnflag │ median(l_quantity) │
│   varchar    │   decimal(15,2)    │
├──────────────┼────────────────────┤
│ A            │              26.00 │
│ N            │              25.00 │
│ R            │              26.00 │
└──────────────┴────────────────────┘

DuckDB (shape: not a shape the rewrite expresses: select expression) · 46.5 ms
```

The reason codes you will actually meet:

| Reason | What it means |
|---|---|
| `not_resident` | the columns are not on the device yet — the background uploader is working on it |
| `threshold` | a measured bound says DuckDB is faster for this shape and size (`6001215 rows < 16000000 for an aggregate without GROUP BY`), **or** this machine measured it slower: `measured 4.20 ms rewritten vs 3.10 ms native (re-measured in 60 s)` |
| `shape` | not a shape the rewrite expresses: a window function, `median`, `ROLLUP`, a set operation, a subquery in the select list |
| `double` | a `sum` / `avg` over `DOUBLE` or `FLOAT` — never rewritten, by design (see [the two rules](../README.md#the-two-rules)) |
| `ties` | a pushed `ORDER BY … LIMIT k` found two of the first *k* rows equal on the ordering value, so which rows come back — and in what order — is DuckDB's to choose, and DuckDB answered the original |
| `backend` | this build has no GPU backend to rewrite for (a CPU-only binary), or the installed extension is older than the client |
| `memory` | the set does not fit the device-memory budget; it is refused before the upload |
| `transaction` | a `BEGIN` is open, so the resident sets cannot be trusted |
| `params` | the statement takes prepared-statement parameters |
| `error` | the rewritten statement raised and DuckDB answered the original — the text is in `.gpu` |
| `off` / `manual` | the path is off (`.gpu off`, `--no-gpu`), or residency is `manual` |

Ten more codes name a smaller refusal precisely where `shape` would only say
"no": `nulls`, `overflow`, `decimal`, `collation`, `too_long`, `multi`,
`view`, `temp`, `ambiguous`, `not_found`. Each is a `reason` of its own, with
its own sentence in `detail`. Twenty-two codes in all.

`ties` is the one of them decided against the data rather than the statement,
and it is worth seeing once. Three orders in TPC-H SF1 share the fourth-and-fifth
place quantity, so the top five is not one answer:

```
gpudb> SELECT l_orderkey, sum(l_quantity) AS qty FROM lineitem GROUP BY l_orderkey ORDER BY qty DESC LIMIT 5;
┌────────────┬───────────────┐
│ l_orderkey │      qty      │
│   int64    │ decimal(38,2) │
├────────────┼───────────────┤
│    4806726 │        328.00 │
│    2199712 │        327.00 │
│    4722021 │        323.00 │
│    1263015 │        320.00 │
│    4702759 │        320.00 │
└────────────┴───────────────┘

DuckDB (ties: two of the first 5 rows tie on qty, so which rows come back — and in what order — is DuckDB's to choose, and DuckDB answered the original) · 43.8 ms
```

Plain DuckDB does not have one answer here either — above one thread it returns
different row sets run to run — so there is nothing for the device to reproduce,
and the statement is handed back. A tie that keeps happening is also a cost: the
device pass is paid and then DuckDB answers anyway, so the template is declined
by the same measured rule that declines any losing template, and the next run of
it reads

```
DuckDB (threshold: two of the first 5 rows tie on qty, so DuckDB answers it — and the device pass costs 26.20 ms on top of whatever native costs, so the template is native from here (re-measured in 60 s)) · 25.7 ms
```

until the 60-second re-measure finds the tie gone. A `LIMIT` below the tie, or an
`ORDER BY` that is already total (`ORDER BY qty DESC, l_orderkey`), keeps the
device.

One thing about that check is particular to this shell. Inside `execute()` it is
a clause of your own statement and costs a row. The shell goes through `sql()`,
which hands back a relation read after the call has returned — too late for the
check to raise there — so it runs on a side cursor inside the call instead, and
for a pushed top-k that side cursor is the device pass itself. Paid on every
call it would cost the shape its whole win, so **the verdict is remembered**:
the first statement of a data version asks the device once more, and every call
after it reads the answer back until the data changes. Any write, DDL, `SET`,
`ATTACH` or a foreign write the wrapper notices drops it along with the resident
sets, and so does a row count that has moved, so a remembered "no tie" can never
outlive the rows it was taken over.

## Look underneath

`.gpu` prints the whole record for the last statement, including the SQL that
actually ran:

```
gpudb> .gpu
statement:     SELECT l_partkey, sum(l_quantity) AS qty FROM lineitem GROUP BY l_partkey ORDER BY qty DESC LIMIT 5;
rewritten:     True
form:          topk
tag:           gpudb:v1:tpch:main:lineitem:20631:l_partkey,l_quantity
sql:           SELECT "key" AS l_partkey, (CAST(sum AS DECIMAL(36,0)) * 0.01) AS qty FROM gpu_groupby_exact_resident_topk('gpudb:v1:tpch:main:lineitem:20631:l_partkey,l_quantity', 'sum', 6, 'desc') AS r , (SELECT gpu_assert_rows('gpudb:v1:tpch:main:lineitem:20631:l_partkey,l_quantity', count_star()) AS ok FROM tpch.main.lineitem) AS gd WHERE gd.ok QUALIFY CASE  WHEN (((rank() OVER (ORDER BY qty DESC) = row_number() OVER (ORDER BY qty DESC)) OR (rank() OVER (ORDER BY qty DESC) > 5))) THEN (CAST('t' AS BOOLEAN)) ELSE "error"('GPUDB_TIES: the first 5 rows are not ordered uniquely by qty') END ORDER BY qty DESC LIMIT 5
round_trip_ms: 0.006 ms
engine:        scalar
detail:        the resident GROUP BY
```

- `form` is which device shape answered: `plain`, `having`, `topk`,
  `projected`, or `nested` when the rewrite reached inside a bigger statement.
- `tag` is the **identity tag** of the resident set it used —
  `gpudb:v1:<catalog>:<schema>:<table>:<oid>:<columns>`. Two statements that
  need the same columns of the same table name the same tag and share one copy.
- The `LIMIT 5` above asks the device for **6** rows, and the `QUALIFY` is the
  tie guard: it compares `rank()` with `row_number()` over those rows and raises
  `GPUDB_TIES` if any of the first five is not uniquely ordered, which is what
  turns into the `ties` footer. It costs one extra row when nothing ties.
- `round_trip_ms` is the wrapper's own overhead: parse, decide, render. Not
  the statement's time, which the footer prints.
- `engine` says who produced the rewritten SQL — `scalar` for the extension's
  pure `gpu_rewrite_ast`, `python` for the wrapper's reference renderer,
  `nested` for the pass that rewrites a `SELECT` inside a larger statement.

`.gpu off` turns the path off live, which is what the A/B above is made of
(`.gpu on` puts it back; `--no-gpu` starts that way). `.residents` shows what
is on the device — the sets a statement waits on, then the columns behind
them. This is the same session, after those nineteen runs:

```
gpudb> .residents
table          columns               state  bytes     estimated  worth
main.lineitem  l_partkey,l_quantity  ready  80.1 MiB  207.5 MiB  2.13
1 set · 80.1 MiB held · worth is ms saved per second per GiB · `.memory` for the budget

table     column      dtype  rows       width  bytes     state
lineitem  l_partkey   I64    6,001,215  4 B    68.7 MiB  ready
lineitem  l_quantity  I64    6,001,215  2 B    11.4 MiB  preparing
2 resident columns · width is the bytes a row of the lane is stored at · `-` where the backend does not say
```

- The **set** line is what a statement waits on: `ready` means uploaded and
  its sort cache built. `estimated` is the upper bound the budget reserved
  against; `bytes` is what the extension actually holds. That is the lanes
  *plus the sort cache of a key lane* — which is why `l_partkey` reads 68.7 MiB
  and not the 22.9 MiB its 6,001,215 rows at 4 bytes each come to: 22.9 MiB of
  key data and 45.8 MiB of sorted keys and permutation, 8 bytes a row on top of
  the 4. A lane with no sort cache is exactly its width (`l_quantity`, 2 bytes
  a row, 11.4 MiB). Each lane is stored at the narrowest signed width its
  values fit, which is where those widths come from.
- A set **you uploaded by hand** reads `0 B` here: the wrapper never planned
  it, so it has no estimate of its own for it and no store columns underneath.
  It is counted against the memory budget all the same — admission sums every
  row of `gpu_residents()` — and it is never evicted.
- `worth` is the set's value: milliseconds of DuckDB time it saves per second
  of wall time, per GiB it holds. It reads `-` until the set has been used
  enough to have one, and it is what the budget compares when it has to
  choose. Nineteen runs of one statement put this one at 2.13.
- The **column** lines are the store underneath, and their `state` means
  something narrower: whether *that lane* has a sort cache. Only a key lane
  ever needs one, so a payload lane reads `preparing` and stays there. Read
  the set's state, not the payload's.

`.memory` is the budget side of the same picture:

```
gpudb> .memory
backend:       Metal · Apple M4 Max · 51.8 GiB device memory
resident:      80.1 MiB in 1 set
budget:        16.0 GiB
residency:     background
```

Raise or lower it with `--memory-budget 16GB` (`unlimited` removes the cap).
A set that does not fit is not uploaded, and its statements keep running on
DuckDB. [When GPU memory is full](INSTALL.md#when-gpu-memory-is-full) has the rest.

## Uploading by hand

Under `--residency manual` nothing is uploaded for you, and a statement that
would need a set it does not have declines with `manual`. Its `.gpu` record
still names the `tag:` it wanted, and that is the whole recipe: upload a set
under exactly that name and the statement is rewritten from then on.

```
gpudb> SELECT l_partkey, sum(l_quantity) AS qty FROM lineitem GROUP BY l_partkey ORDER BY qty DESC LIMIT 5;
DuckDB (manual: residency is manual and this set was not uploaded by hand) · …
gpudb> .gpu
…
tag:          gpudb:v1:tpch:main:lineitem:20631:l_partkey,l_quantity

gpudb> SELECT gpu_upload_pair_exact('gpudb:v1:tpch:main:lineitem:20631:l_partkey,l_quantity',
                                    l_partkey, (l_quantity*100)::BIGINT) FROM lineitem;
-- 6001215
gpudb> SELECT l_partkey, sum(l_quantity) AS qty FROM lineitem GROUP BY l_partkey ORDER BY qty DESC LIMIT 5;
GPU (topk: the resident GROUP BY) · …
```

The payload is scaled to an integer because a `DECIMAL(15,2)` lives on the
device as a scaled integer — `l_quantity` has two decimal places, so `* 100`.
An integer column needs no cast. This is an escape hatch for people who want
to decide exactly what is resident; the default `background` residency does
all of it for you.

## Scripts, options, keys

```bash
gpudb my.duckdb -c "SELECT count(*) FROM lineitem;"     # one statement
gpudb my.duckdb -f report.sql                           # a file
gpudb my.duckdb < report.sql                            # or on stdin
```

`-c` and `-f` may be repeated and run in the order given; `.read FILE` does the
same from inside a session. A statement that fails ends a `-c` / `-f` / piped
run with a non-zero exit code; at the terminal the session keeps going. The
footer line is on at a terminal and off in scripted output unless `--timer`
says otherwise (`--no-timer` forces it off, `.timer on|off` toggles it live);
colour follows `NO_COLOR`.

| Option | |
|---|---|
| `gpudb [DATABASE]` | open a database (no argument: in-memory) |
| `-c SQL` | run a statement and exit, repeatable |
| `-f FILE`, `--file FILE` | run a file and exit, repeatable; `-c` and `-f` run in the order given |
| `--readonly`, `--read-only` | open read-only |
| `--no-gpu` | every statement on DuckDB — the control for a comparison |
| `--residency background\|eager\|manual` | when tables become resident (default `background`) |
| `--memory-budget SIZE` | e.g. `16GB`, or `unlimited` |
| `--timer` / `--no-timer` | force the footer line on or off |
| `--debug` | show tracebacks instead of one-line errors |
| `--version` | the gpudb and duckdb versions |
| `-h`, `--help` | the same list, from the program |

Dot-commands, kept small on purpose — this is the terminal client for the
transparent path, not an emulation of the DuckDB CLI (no `.mode`, `.output`,
`.import`, `.shell`):

| | |
|---|---|
| `.help` | the list |
| `.gpu` | where the last statement ran, in full |
| `.gpu on\|off` | the transparent path, live |
| `.residents` | the resident sets, then the columns behind them |
| `.memory` | the device-memory budget and what holds it |
| `.timer on\|off` | the footer line |
| `.read FILE`, `.open [DATABASE]` | run a file; open another database (no argument: a fresh in-memory one) |
| `.tables`, `.schema [TABLE]` | plain SQL underneath (`SHOW TABLES`, `DESCRIBE`) |
| `.version` | gpudb and duckdb versions |
| `.quit`, `.exit` | leave |

`.open` with no argument works in a `--readonly` session as well: the in-memory
database it opens is read-write, because there is nothing on disk to protect,
while `.open` on a *named* file in such a session keeps the read-only setting
the session was started with.

Statements may span lines and end at `;`. **Ctrl-C** stops the running
statement, or clears what you were typing; **Ctrl-D** at a waiting prompt
leaves. History lives in `~/.gpudb_history` when the Python build has
`readline`. One platform quirk, measured and written down: on Linux a Ctrl-D
typed *while a statement is running* is swallowed by the terminal's line
discipline and the shell stays at the next prompt — stock Python's `input()`
does the same thing, and [KNOWN_ISSUES.md](../KNOWN_ISSUES.md) has the mechanism.
