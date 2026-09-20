# Reading guide — the research and design record

Everything that was tried, measured, kept or dropped while building gpudb is
written down in this repository. This page says where, and in what order to
read it.

## Using it

| If you want | Read |
|---|---|
| The `gpudb` shell, end to end — the banner, the footer and its reason codes, `.gpu` / `.residents` / `.memory` | **[USING_THE_SHELL.md](USING_THE_SHELL.md)** |
| `gpudb.connect()` — every option, `last_rewrite()` and `memory()`, threads, writes and transactions | **[USING_PYTHON.md](USING_PYTHON.md)** |
| Installing both pieces on Apple Silicon Metal or NVIDIA CUDA, the platform detail, troubleshooting | **[INSTALL.md](INSTALL.md)** |
| What shipped in each release | **[RELEASE_HISTORY.md](RELEASE_HISTORY.md)** · [RELEASE_NOTES_v0.7.md](RELEASE_NOTES_v0.7.md) for v0.7.0 by theme |

## The record

| If you want | Read |
|---|---|
| The story: what was asked, what was tried, what failed, what was decided | **[RESEARCH_NOTES.md](RESEARCH_NOTES.md)** — a dated lab journal, written to be read front to back |
| Every number, with the losing cells left in | **[../BENCHMARK.md](../BENCHMARK.md)** |
| What does not work, or is slower than DuckDB, today | **[../KNOWN_ISSUES.md](../KNOWN_ISSUES.md)** |
| How a plain `SELECT` reaches the GPU and how the answer is kept identical | **[TRANSPARENT_DESIGN.md](TRANSPARENT_DESIGN.md)** |
| How columns live on the device: the store, narrow lanes, shedding, index vectors | **[RESIDENT_COLUMNS_DESIGN.md](RESIDENT_COLUMNS_DESIGN.md)** |

The journal records *how we got there*; the design documents stay
authoritative for *how things are*.

## The two rules every entry is judged by

1. **Never slower than DuckDB.** A statement goes to the GPU only where
   measurement says it wins; otherwise DuckDB answers it. Thresholds come from
   sweeps, and a running connection keeps re-measuring its own statements.
2. **Never a different answer.** Row for row, including names and types —
   NULLs, 128-bit sums, NaN ordering, DECIMAL, stale data after a write.

A result that loses is printed next to the ones that win. Those cells are the
reason the thresholds look the way they do.

## The journal by theme

Entries are in date order in the file; this groups them by question.

**Exactness**
- Exactness first: NULLs and 128-bit sums
- Subqueries in WHERE, a silent wrong answer avoided, and a 0.03× caught by the coverage map
- The write the guard could not see
- A view is not a thing you can remember
- A guard that tested for a value the serializer never emits
- A formula verified bit-exact on one architecture and wrong on the other
- The tie at the k-th row, and what native actually does with one — twenty plain-DuckDB runs of one `ORDER BY … LIMIT 5`, and the three different row sets they returned

**Deciding when the GPU should answer**
- The gate: thresholds come from measurements
- Few groups, and what a benchmark loop hides
- Two modes of a short kernel: what the gate's three losing rows were
- The millisecond that was not there
- The second door into the wrapper
- A bound is a measurement of a situation, and three of ours were in the wrong one
- The gate whose native shapes had stopped being native

**Widening the SQL that qualifies**
- WHERE on the device
- Widening the key: dates, several columns, strings
- Joins · The join that needed no hash table, and the flag that could finally flip
- Several aggregated columns in one statement
- Expressions: computed lanes · Expressions over aggregates · Aggregates in disguise
- No GROUP BY, and the joins the device cannot do
- A coverage map: the 22 TPC-H queries · Working down the coverage map
- Derived tables, an expensive guard, count(DISTINCT)
- Views are derived tables with a name · GROUP BY ALL · Probing 36 shapes, taking four
- A CTE is two things, and only one of them wants inlining

**Memory: what has to stay on the device**
- The memory budget that was only on paper
- Lane sharing measured before it was built: not worth building
- Reinvention, stage A (row order), stage B (the store), stage C (the width the data asks for)
- The index that had nowhere to be bound (stage D, measured and not wired into SQL)
- What a column stops needing
- Least recently used knows when, not what for (value-aware residency)
- The lane got smaller and the cache did not — narrow lanes on CUDA, and the derived structure that had not followed them
- The number went the wrong way — narrowing the CUDA exact sort cache, and the double count the A/B caught
- It was never the segments — what a background upload actually costs a statement, and the back-off that followed from measuring it

**Kernels**
- Raw performance first: where a statement's time goes
- SF10, and "what is the gap?"
- One group is not a GROUP BY (the global masked aggregate)
- Few keys do not need a sort: the direct grouped reduce
- The WHERE that read the mask ten times (the fused mask)
- The aggregate that has no use for a sort
- A capability flag that was really a placement decision (CUDA)

**Using it**
- The Python wrapper, and the name it had to be published under — the entry keeps the name of the day in its title, "pip install gpudb"; the distribution is `duckdb-gpudb` and the import is `gpudb`
- A terminal is a client (the `gpudb` shell)
- The end-of-file that arrived while nobody was reading, and the Ctrl-C that had nobody to wake — two pty races that turned out to be stock Python's
- The other build path, unexercised since the rewriting began
- The SQL suite now runs on x86-64, and the DuckDB libs are pinned
- The documents still described a layout we had dropped
- A guard that was green on a fixture and dead on the path
- CI runs the wrapper suite, and what a machine without a GPU can prove
- A test that pretended to be the other machine
- The column SQL could not compute (`avg` over `DECIMAL`, derived in C++)
- Measurements that existed only in review (the CUDA numbers put into the repository)
- A hypothesis that was wrong, and a 32% win that was real (the CUDA exact upload, double-buffered)
- Three passes into one, and a ratio that checks itself (the fused `agg_all` on CUDA)

The journal ends with **Open questions** — the shapes the current design does
not answer, each with the measurement that says why. It is a record of where
the edges are, not a plan.

## Reference documents

| Document | What it covers |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Layers, backends, the abstract interface |
| [GROUPBY_RESIDENT_DESIGN.md](GROUPBY_RESIDENT_DESIGN.md) | The explicit `gpu_*` SQL functions and the resident GROUP BY |
| [WINDOW_FUNCTIONS_DESIGN.md](WINDOW_FUNCTIONS_DESIGN.md) | Window functions — a design note; they are not implemented |
| [CUDA_EXACT_PATH.md](CUDA_EXACT_PATH.md) | The interface contract for the exact path, and the tests that prove it on every backend |
| [ENVIRONMENT.md](ENVIRONMENT.md) | Every environment variable the code honours, one line each |
| [BENCHMARK_PLAN.md](BENCHMARK_PLAN.md) · [DATASETS.md](DATASETS.md) | How the benchmarks are run and on what data |
| [DEVELOPMENT.md](DEVELOPMENT.md) · [MACOS_EXTENSION_BUILD.md](MACOS_EXTENSION_BUILD.md) · [CI_RECIPES.md](CI_RECIPES.md) | Building, testing, CI |
| [RELEASE_NOTES_v0.7.md](RELEASE_NOTES_v0.7.md) · [RELEASE_HISTORY.md](RELEASE_HISTORY.md) | What v0.7 ships, by theme; and every release before it |
| [RELEASE_READINESS.md](RELEASE_READINESS.md) | Historical: the first community-extension submission's punch list |

## Reproducing a number

Every table names its command. The common ones:

```bash
SF=1 ./scripts/gen_tpch.sh                                  # data
./scripts/get_duckdb_libs.sh && ./scripts/build.sh          # extension + tools
PYTHONPATH=python python3 scripts/tpch_coverage.py          # the 22 TPC-H queries: who answers, how fast, identical?
PYTHONPATH=python python3 scripts/tpch_coverage.py \
    --db data/tpch_sf10/tpch.duckdb --memory-budget 200GB   # ... at SF10, which needs the budget raised
PYTHONPATH=python python3 scripts/transparent_gate.py \
    --subqueries --exprs                                    # the sweep behind the thresholds (about an hour)
```

Measurements are on an Apple M4 Max (Metal) unless an entry says otherwise.
Timings of statements under about 5 ms are taken with `SET threads TO 1`,
interleaved, min and median — the journal entry *Two modes of a short kernel*
explains why.
