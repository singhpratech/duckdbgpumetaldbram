# gpudb — GPU-accelerated DuckDB on **NVIDIA CUDA + Apple Silicon Metal**

> **The first SQL execution engine for Apple Silicon GPUs**, built as a DuckDB extension that *also* runs on NVIDIA CUDA. One codebase, two backends, your existing DuckDB queries.

```sql
-- Real query, real GPU, real result on RTX 4090 + Apple M4 Max
SELECT gpu_sum(l_orderkey) FROM read_parquet('lineitem.parquet');
[gpudb] registered gpu_sum (BIGINT,DOUBLE) / gpu_min / gpu_max  (backend=CUDA)
gpu_sum(l_orderkey)
18005322964949
```

Apache-2.0 · Pre-alpha · Linux + macOS · DuckDB ≥ 1.2

---

## Why this exists

Every standalone GPU database from 2013-2024 was acqui-hired or pivoted (HEAVY.AI → NVIDIA 2025, BlazingSQL dormant, Voltron Data 50% layoff). Building "another GPU SQL engine" is not a viable bet.

What's open in 2026: **no published SQL engine targets Apple Silicon GPUs**. Sirius (UW + NVIDIA, CIDR 2026) is CUDA-only. cuDF is CUDA-only. So is everything else. Apple Silicon's unified memory architecture (up to 512 GB at 819 GB/s on M3 Ultra) is a genuine architectural advantage that nobody has wired into a database.

`gpudb` is a DuckDB *extension* (not a fork, not a new database) that closes that gap with a real dual-backend implementation.

## Numbers (RTX 4090 Laptop, sm_89, CUDA 13.0)

| Workload | CPU baseline | CUDA cold | CUDA resident | Speedup |
|---|---:|---:|---:|---|
| SUM 100M int64 | 13.8 ms / 54 GiB/s | 80.6 ms (PCIe-bound) | **0.04 ms / 1187 GiB/s** | 17.9× over CPU |
| SUM 6M TPC-H lineitem.l_orderkey | 0.7 ms / 65 GiB/s | 4.9 ms | **0.04 ms / 1187 GiB/s** | 17.9× over CPU |
| GROUP BY 50M × 1M groups | 1067 ms (serial) | 130 ms | n/a | **9.6× over CPU** |
| GROUP BY 50M × 10M groups | 2321 ms | 188 ms | n/a | **13.7× over CPU** |
| GROUP BY 6M TPC-H lineitem (1.5M groups) | 54.1 ms | 15.0 ms | n/a | **3.6× over CPU** |

## Numbers — Apple Silicon (M4 Max, v0.1.3 Metal vs DuckDB CLI default)

DuckDB CLI uses 16 threads (12 P + 4 E cores) by default on M4 Max. These are vs `duckdb -c "SET threads=16"`, the actual user-visible baseline.

| Workload | DuckDB CPU mt | Metal v0.1.3 | **Speedup** |
|---|---:|---:|:---:|
| **TPC-H SF10 multi-agg fusion `l_quantity`** | 27 ms | **1.06 ms** | **25.5×** 🚀 |
| **TPC-H SF10 multi-agg fusion `l_extendedprice`** | 26 ms | **1.18 ms** | **22.0×** 🚀 |
| **TPC-H SF10 multi-agg fusion `l_orderkey`** | 11 ms | **1.13 ms** | **9.7×** 🚀 |
| **SF10 SUM `l_quantity` HOT** | 5 ms | **1.16 ms** | **4.3×** ✅ |
| **TPC-H SF10 GROUP BY `l_extendedprice` (1.35M unique)** | 113 ms | **28.9 ms** | **3.9×** ✅ |
| **500M × 1M GROUP BY synthetic** | 820 ms | **242 ms** | **3.4×** ✅ |
| **1B × 1M GROUP BY synthetic** | 2500 ms | **770 ms** | **3.2×** ✅ |
| **1B int64 SUM HOT** | 40 ms | **16.2 ms** | **2.6×** ✅ |
| **SF10 SUM `l_extendedprice` HOT** | 5 ms | **2.23 ms** | **2.2×** ✅ |
| **SF10 SUM `l_orderkey` HOT** | 3 ms | **1.68 ms** | **1.8×** ✅ |
| **TPC-H SF1 GROUP BY `l_orderkey` (1.5M unique)** | 8 ms | **5.71 ms** | **1.40×** ✅ |
| **TPC-H SF10 GROUP BY `l_orderkey` (15M unique)** | 56 ms | **42.95 ms** | **1.30×** ✅ |
| TPC-H SF10 GROUP BY `l_quantity` (50 unique) | 26 ms | 372 ms | CPU 14× ❌ structural |

**v0.1.3 ships a hybrid Metal GROUP BY** that auto-dispatches between a 32K-partition slot-lock hash aggregate (sweet spot at 1024 ≤ unique ≤ 16M) and an optimized multi-pass radix sort. Multi-aggregate fusion (`SELECT SUM(x), MIN(x), MAX(x), COUNT(x) FROM t`) reads the column once and computes all four in a single Metal pass at ~475 GiB/s (87% of LPDDR5X peak). Full numbers + reproduction in [BENCHMARK.md](BENCHMARK.md).

## Quick start

### Option A — load the prebuilt extension into DuckDB CLI

Download the platform binary from the [latest release](https://github.com/singhpratech/duckdbgpumetaldbram/releases/latest), then:

```bash
# Linux (RTX/CUDA)
duckdb -unsigned -c "LOAD '/path/to/gpudb.linux_amd64.duckdb_extension'; \
  SELECT gpu_sum(value::BIGINT) FROM range(1000000) AS t(value);"
# -> [gpudb] registered gpu_sum (BIGINT,DOUBLE) / gpu_min / gpu_max  (backend=CUDA)
# -> 499999500000
```

Requires DuckDB ≥ 1.2 (C API v1.2.0); community CI builds against v1.5.x. `LOAD` needs `-unsigned` because the
extension is unsigned community code; `INSTALL gpudb FROM community`
(no `-unsigned` needed) lights up after [community-extensions PR #1898](https://github.com/duckdb/community-extensions/pull/1898) merges.

### Option B — build from source

```bash
git clone https://github.com/singhpratech/duckdbgpumetaldbram.git
cd duckdbgpumetaldbram

# Linux (CUDA): one-time toolkit install if needed
# sudo apt install -y cuda-toolkit-13-0
# export PATH=/usr/local/cuda/bin:$PATH

# macOS (Metal): brew install cmake

# build (auto-detects CUDA on Linux, Metal on macOS, CPU-only otherwise).
# Produces a loadable .duckdb_extension with metadata footer attached.
./scripts/build.sh

# load + query via DuckDB CLI
duckdb -unsigned -c "LOAD '$(pwd)/build-linux/src/extension/gpudb.linux_amd64.duckdb_extension'; \
  SELECT gpu_sum(range::BIGINT) FROM range(1000000);"

# OR run via the embedded SQL CLI shipped in this repo
./build-linux/bin/gpudb-sql --sql "SELECT gpu_sum(range::BIGINT) FROM range(1000000);"
```

### Option C (TPC-H reproducibility)

```bash
# get TPC-H SF1 data (downloads DuckDB CLI to .tools/, ~1 GB lineitem)
SF=1 ./scripts/gen_tpch.sh

duckdb -unsigned -c "LOAD '$(pwd)/build-linux/src/extension/gpudb.linux_amd64.duckdb_extension'; \
  SELECT gpu_sum(v) FROM read_parquet('data/tpch_sf1/lineitem_orderkey.parquet') t(v);"
# -> 18005322964949
```

## What you get

After build, five CLI tools:

| Tool | What it does |
|---|---|
| **`gpudb-sql`** | Embeds DuckDB, registers `gpu_sum` / `gpu_min` / `gpu_max`, runs SQL from `--sql` or stdin. **Demo this.** |
| `gpudb-bench` | Microbench SUM/MIN/MAX across CPU + CUDA + Metal, cold vs hot resident, on synthetic or `.gpudb` files |
| `gpudb-groupby-bench` | Microbench GROUP BY hash aggregate at varying cardinality |
| `gpudb-window-bench` | Microbench window functions (running sum, partitioned, unbounded frame) |
| `gpudb-hashjoin-bench` | Microbench inner equi-join build × probe across CPU + CUDA |

And a static library `libgpudb` you can embed in any C++ project. See `src/extension/gpu_sum_extension.{cpp,hpp}` for the DuckDB-aware wrapper.

## Architecture

```
┌──────────────────────────────────────────┐
│  DuckDB (host)                           │
│  Parser → Optimizer → Plan → Executor    │
│            │                             │
│            ↓ aggregate function call     │
│  ┌────────────────────────────────────┐  │
│  │  gpudb extension                   │  │
│  │  - gpu_sum / gpu_min / gpu_max     │  │
│  │  - batched finalize for GROUP BY   │  │
│  │      ↓                             │  │
│  │  ┌───────────────────────────────┐ │  │
│  │  │  libgpudb backend dispatch    │ │  │
│  │  │  ┌───────┐ ┌──────┐ ┌──────┐  │ │  │
│  │  │  │ CUDA  │ │Metal │ │ CPU  │  │ │  │
│  │  │  └───────┘ └──────┘ └──────┘  │ │  │
│  │  └───────────────────────────────┘ │  │
│  └────────────────────────────────────┘  │
└──────────────────────────────────────────┘
```

Backend selection is automatic: CUDA if a device is found at runtime, else Metal if compiled-in, else CPU.

## Testing
```bash
./build-linux/test/test_gpudb        # 96 unit checks across the backends present at build time
./scripts/run_sql_tests.sh           # SQL-level suite: gpu_sum / min / max / GROUP BY / window
./scripts/local_check.sh             # everything CI would run, end to end
```

The SQL test suite lives in `test/sql/*.test`. Each file is plain SQL with
`-- expect:` lines after each query; the runner reports per-query
PASS / FAIL / XFAIL (expected fail) / SKIP. As of v0.2.0: 50 / 50 pass,
0 fail, 0 expected fail. The window-function bugs that were previously
xfail are now strict positive assertions (PR #22).

**Reproducibility entry point:** [`scripts/local_check.sh`](scripts/local_check.sh) runs the full pipeline end-to-end (configure → build → 96 unit tests → smoke benchmarks → 50-query SQL suite). The hosted CI workflow lives at [`.github/workflows/ci.yml`](.github/workflows/ci.yml) (Linux + macos-15) and runs on every push to `main`.

## Roadmap

### Shipped on `main` (v0.1.0 – v0.1.2)
- [x] CUDA backend: SUM/MIN/MAX (one-shot + resident)
- [x] CUDA GROUP BY hash aggregate (open-addressing + atomicCAS, ~520 GiB/s on RTX 4090)
- [x] **CUDA hash join probe** (1M build × 10M probe @ 97% sel: 3.7× wall, 107× kernel over CPU)
- [x] Metal backend: SUM/MIN/MAX i64 with real compute pipelines (~470 GiB/s on M4 Max)
- [x] **Metal GROUP BY** via GPU-resident radix sort (wins 4.4–4.8× over CPU at 100M-500M × 1M groups)
- [x] **Multi-aggregate fusion** (SUM+MIN+MAX+COUNT in one pass, 5.3× over CPU fused)
- [x] **Hybrid CPU/GPU planner** (HybridAggregator + DispatchDecision, beats both pure-CPU and pure-GPU at the 1M×1M sweet spot)
- [x] DuckDB extension: gpu_sum / gpu_min / gpu_max with NULL handling + GPUDB_FORCE_BACKEND env var
- [x] CLI: gpudb-bench, gpudb-groupby-bench, gpudb-window-bench, gpudb-hashjoin-bench, gpudb-sql

### Shipped in v0.1.2
- [x] **All 4 known window/GROUP BY bugs fixed** (PR #18, #20, #21, #22 — see KNOWN_ISSUES.md)
- [x] **DuckDB loadable extension actually loads** — `duckdb -unsigned -c "LOAD '/path/to/gpudb.<platform>.duckdb_extension'"` works on Linux (CUDA) and macOS (Metal).

### Shipped in v0.1.3
- [x] **Hybrid Metal GROUP BY** — 32K-partition slot-lock + radix-opt with auto-dispatch (env override `GPUDB_METAL_GROUPBY_PATH`). Flipped TPC-H SF10 `l_orderkey` (15M unique) from CPU 1.78× faster to Metal 1.30× faster vs DuckDB CPU 16-thread. 9 wins / 1 honest loss on the lineitem scorecard.
- [x] Prebuilt v0.1.3 binaries (Linux CUDA + macOS Metal) attached to the [v0.1.3 release](https://github.com/singhpratech/duckdbgpumetaldbram/releases/tag/v0.1.3).

### Shipped in v0.2.0
- [x] **SQL-correct NULL semantics** (PR #44) — `gpu_sum`/`gpu_min`/`gpu_max` over empty or all-NULL input now return SQL `NULL` (not 0), matching native DuckDB on every path: plain aggregate, GROUP BY groups, and window frames.
- [x] **`gpu_sum(DOUBLE) -> DOUBLE`** (PR #45) — a real second overload via the C API aggregate function set. Doubles ride the existing int64 state machinery as raw bit patterns (zero state-layout change); only the finalize differs. `INTEGER`/`SMALLINT`/`TINYINT` work via DuckDB's implicit widening to the `BIGINT` overload (locked in by tests). Type matrix in [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

### In flight
- [ ] [DuckDB Community Extensions PR #1898](https://github.com/duckdb/community-extensions/pull/1898) merged → `INSTALL gpudb FROM community` (no `-unsigned` flag needed). The submission is deliberately frozen at the v0.1.3-era ref while it awaits maintainer review; v0.2.0 ships as a follow-up bump after the first merge.

### Roadmap (v0.3.0)
Near-term, scoped against what the DuckDB C extension API actually allows:
- [ ] **`gpu_min(DOUBLE)` / `gpu_max(DOUBLE)`** — needs `min_f64`/`max_f64` added to the backend interface (`gpu_backend.hpp`); the registration side is ready (function sets already carry the sum overloads).
- [ ] **Batched GPU GROUP BY for DOUBLE** — a `groupby_sum_f64` backend entry point so high-cardinality DOUBLE GROUP BY takes the batched GPU path the BIGINT path already has.
- [ ] **Real Metal hash-join** (currently a CPU-fallback scaffold; CUDA hash-join is real).
- [ ] **GPU-resident segment reduce** for Metal GROUP BY at 1B+ rows (the cell where we currently lose 1.5×).
- [ ] **Community packaging phase 2** — add `cuda` to `requires_toolchains` so Linux community binaries ship the CUDA backend.

### Beyond (v0.4.0+)
- [ ] Resident-column SQL hooks: `gpu_cache(table, col)` table function so `gpu_sum` can run on data already loaded
- [ ] Window functions on GPU as proper operators (not just aggregate-as-window)
- [ ] String / regex operators (libcudf-class functionality on Metal where it doesn't exist)
- [ ] Transparent GPU operator substitution — **blocked upstream**: the DuckDB C API exposes no optimizer/planner/physical-operator hooks, so a loadable extension cannot silently replace plan operators today. A table-function-based `gpu_group_by(...)` is possible but doesn't compose with normal SQL; parked until the C API grows the needed hooks.

## Why DuckDB? Why not a new database?

The 2013-2024 GPU-DB graveyard is real. The wedge that *isn't* in the graveyard:

1. **Apple Silicon backend** — empty field, defining differentiator
2. **DuckDB-native** — no migration, just `LOAD`
3. **Hybrid CPU/GPU planner** — picks CPU when it wins (low cardinality), GPU when it doesn't
4. **Window functions** — Sirius lacks them; high-value for analytics

This combination is unique as of May 2026. See [GOAL.md](GOAL.md) for the full positioning and [BENCHMARK.md](BENCHMARK.md) for reproducible numbers.

## Citing

If you use this project in research or commercial work:
```
gpudb: GPU-accelerated DuckDB extension for NVIDIA CUDA and Apple Silicon Metal.
2026. https://github.com/singhpratech/duckdbgpumetaldbram
```

## Author / blog

Build process, design tradeoffs, and ongoing benchmarks are posted at **[theaivibe.org](https://theaivibe.org)**.

## License

Apache-2.0. See [LICENSE](LICENSE).
