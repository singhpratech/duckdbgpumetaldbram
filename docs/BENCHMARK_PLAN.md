# Benchmark plan — operators × scales × scenarios × backends

This document defines the comprehensive comparison framework for `libgpudb`
across the three backends (CPU, CUDA, Metal). It is the canvas on which
BENCHMARK.md numbers are produced, so we can honestly answer:

1. **Where do we beat CPU?**
2. **Where do we come close to CUDA?**
3. **Where do we beat CUDA?** (Metal-specific opportunities)
4. **Where do we honestly lose, and why?**

The aim is reproducibility, transparency, and a story that survives technical
review.

---

## 1. Standard suites (industry-recognized)

We use these because reviewers will ask for them:

| Suite | What it is | Why it matters | Status |
|---|---|---|---|
| **TPC-H** | 22 SQL queries on 8-table snowflake schema, multiple SFs | THE canonical analytical SQL benchmark. CUDA path already runs SF1 lineitem aggregations. | SF1 generated; SF10 next |
| **TPC-DS** | 99 queries on 17-table snowflake, more complex (CTEs, windows) | Modern decision-support; tests window functions and complex joins | Future (post hash-join) |
| **ClickBench** | 43 queries on real Yandex.Metrica web traffic (~14 GiB hits.parquet) | Modern OLAP leaderboard, lots of GROUP BY at varying cardinalities | Wired once DuckDB extension is fully end-to-end |
| **SSB (Star Schema Benchmark)** | Denormalized TPC-H, 13 queries | Dense aggregations + star joins; the workload `cuDF` paper uses | Planned |

Both CUDA and Metal numbers go in the **same** BENCHMARK.md table for each
TPC-H query so the comparison is apples-to-apples.

---

## 2. Operator categories

Every operator we ship gets benched independently first (micro), then in a
real query (TPC-H or ClickBench).

### 2.1 Single-column aggregates
- `SUM(int64)`, `SUM(float64)` — bandwidth-bound, the simplest case
- `MIN(int64)`, `MAX(int64)` — same shape as SUM
- `COUNT(*)` — trivial
- `AVG = SUM / COUNT` — derived, free if SUM bench has wall

**Status:** SUM/MIN/MAX shipped on Metal; FP64 on Apple GPU is a CPU fallback (no IEEE-754 doubles in MSL).

### 2.2 Multi-aggregate fusion
- `agg_all_i64` → `{sum, min, max, count}` from one pass over the column

**Why it matters:** same bandwidth as a single SUM, **4× more outputs per byte read**. This is where Metal can hit 10×+ effective speedup over CPU running 4 separate ops. CUDA path doesn't have this yet.

**Status:** in flight on `feat/metal-multiagg`.

### 2.3 GROUP BY (the wedge)
- `SUM(value) GROUP BY key` — i64 keys, i64 values
- Future: multi-column group keys, multiple aggregates per group, COUNT DISTINCT

**Status:** shipped on Metal with **GPU-resident radix sort + on-device scan + min-max pre-scan**. Peak 4.89× over CPU at 500M × 1M groups. Ties CUDA on TPC-H wall.

### 2.4 Hash join
- `inner_join(build_keys, probe_keys)` → matched (probe_idx, build_idx) pairs
- Future: outer joins, anti/semi-joins

**Status:** shipped on CUDA and Metal. Metal uses adaptive global slot-lock (small build) plus partitioned TG hash radix join (4.9× wall vs CPU @ 1M×10M M4). SQL: `gpu_inner_join`.

### 2.5 Window functions (the GOAL.md item 8 differentiator)
- `RANK()`, `ROW_NUMBER()` over partition
- `LAG`, `LEAD` with offset
- Sliding-window aggregates

**Status:** future. Sirius (CUDA-only) doesn't have these — open competitive gap.

### 2.6 Filter pushdown
- Predicate evaluation on column scan, output passing rows
- Combined with downstream aggregates

**Status:** future.

---

## 3. Scenarios (workload shapes that stress different things)

| Scenario | Shape | What it stresses | Expected Metal advantage |
|---|---|---|---|
| **COLD** (per-call upload) | Fresh column, no resident state | PCIe transfer (CUDA) vs UMA (Metal) | Metal wins decisively (no transfer) |
| **HOT** (resident column) | Column cached, repeated queries | Pure kernel bandwidth | CUDA wins (HBM ~3× M4 Max LPDDR5X) |
| **Tiny N** (≤ 1M rows) | Small batches, OLTP-like | Per-call setup overhead | CPU wins (Metal launch overhead) |
| **Mid N** (10M-100M rows) | TPC-H SF1-SF10 lineitem-class | Mix of bandwidth + setup | Metal wins on cardinality regimes |
| **Huge N** (1B+ rows) | Data lake / warehouse | Sustained bandwidth | Metal closes gap to CUDA (UMA) |
| **Low cardinality GROUP BY** (≤ 1K groups) | DAU, status counts | CPU cache locality | CPU often wins; Metal pulls even with min-max trick |
| **High cardinality GROUP BY** (~rows distinct) | Per-user stats, transaction IDs | Memory latency | **Metal's wedge** — both Metal and CUDA win 3-22× |
| **Multi-op same column** | "give me sum, min, max, count of price" | I/O efficiency | **Metal's wedge** with multi-agg fusion |
| **Star join + agg** | TPC-H Q1, Q5, Q19; SSB | Hash join + group-by | Both Metal and CUDA needed |

---

## 4. Apple-to-apple comparison matrix

For every (operator × scale × scenario) cell, capture:
- CPU wall (single-thread `unordered_map` for GROUP BY; scalar loop for SUM)
- CUDA wall (RTX 4090 Laptop, on the Linux machine)
- Metal wall (Apple M4 Max, on this Mac)
- Metal kernel-only throughput (GiB/s) — to compare to hardware peak
- Speedup over CPU (Metal/CPU and CUDA/CPU)
- Metal vs CUDA ratio

The matrix is published in BENCHMARK.md per release.

### What "honest" means
- Same input data sets across backends (e.g., the same generated `.gpudb` file)
- Same warm-up policy (median of 5+ runs, drop first if cold-start matters)
- Single-thread CPU baseline (so that GPU wins are fair vs a non-GPU world)
- Where CPU could be parallelized (e.g., OpenMP), report both — the parallel CPU number tells you "GPU vs CPU at full tilt", the serial one tells you "GPU vs naive CPU"

---

## 5. Where Metal CAN beat CUDA (Apple-specific opportunities)

Realistic, defensible:

1. **Power efficiency** — Metal SUM at 467 GiB/s on M4 Max draws ~30 W; CUDA RTX 4090 SUM at 1547 GiB/s draws ~150 W. **Per-watt, M4 Max is competitive or better** for memory-bound work. (Bench would need wall-power instrumentation.)
2. **Cold start** — UMA means no PCIe transfer cost. For one-shot queries (small N) where the user's data is in host RAM, Metal can beat CUDA on end-to-end wall.
3. **Multi-agg fusion shipped before CUDA** — this is a tactical gap; whichever backend ships it first wins on multi-op workloads until the other catches up.
4. **DuckDB-native install** — Metal extension means an out-of-the-box GPU OLAP install on every M-series Mac without any NVIDIA hardware. The TAM is significant (millions of MacBook Pro users) and CUDA can't address it.

The headline metric is **wall-time on a real query against real CPU**, not raw kernel GiB/s. We optimize for that.

---

## 6. Where we honestly lose (to either CPU or CUDA), and that's OK

1. **Single SUM at sub-100M rows on a single thread** — CPU's hardware prefetcher is too good at sequential memory.
2. **Tiny low-cardinality GROUP BY (<1M rows × <100 groups)** — CPU's `unordered_map` fits in L1; impossible to beat without ridiculous Metal setup overhead.
3. **Raw HBM-bound throughput** — CUDA wins per-watt and per-second on bandwidth.

Where we lose, we **say so in BENCHMARK.md** and explain why. Honesty buys credibility for the wins.

---

## 7. How to add a new operator to the comparison

1. Decide its category in §2.
2. Implement on CPU first (always), then Metal, then CUDA (Linux Claude).
3. Pick the scenarios from §3 it should be benched against.
4. Add a section to `BENCHMARK.md` with the matrix from §4.
5. Add the canonical SQL to TPC-H/ClickBench bindings if applicable.
6. Update this doc's status table in §2.

---

## 8. Status snapshot

| Operator | CPU | CUDA | Metal | TPC-H query example | BENCHMARK.md section |
|---|:-:|:-:|:-:|---|---|
| SUM/MIN/MAX i64 | ✅ | ✅ | ✅ | Q1 (SUM(l_quantity)) | "Metal SUM at scale" |
| SUM f64 | ✅ | ✅ | host fallback | Q1 (SUM(l_extendedprice)) | same |
| `agg_all_i64` (multi-agg) | 🔜 | — | 🔜 (`feat/metal-multiagg`) | derived from Q1 | (incoming) |
| GROUP BY hash | ✅ | ✅ | ✅ (radix sort) | Q1 GROUP BY l_returnflag | "Metal GROUP BY" / matrix |
| Hash join probe | ✅ | ✅ | ✅ adaptive + partitioned TG hash | Q3, Q5, Q12 | Metal hash join (2026-07-07) |
| Window functions | — | — | — | Q8 (window) — future | — |
| DuckDB extension wrapper | — | — | scaffold + `gpu_sum`, `gpu_min`, `gpu_max` (Linux-built; macOS validation: `feat/ext-macos-validate`) | All TPC-H | (when end-to-end) |
