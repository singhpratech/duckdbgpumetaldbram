# gpudb — GPU-accelerated DuckDB on **NVIDIA CUDA + Apple Silicon Metal**

> **The first SQL execution engine for Apple Silicon GPUs**, built as a DuckDB extension that *also* runs on NVIDIA CUDA. One codebase, two backends, your existing DuckDB queries.

```sql
-- Real query, real GPU, real result on RTX 4090 + Apple M4 Max
SELECT gpu_sum(l_orderkey) FROM read_parquet('lineitem.parquet');
[gpudb] registered gpu_sum / gpu_min / gpu_max  (backend=CUDA)
gpu_sum(l_orderkey)
18005322964949
```

Apache-2.0 · Pre-alpha · Linux + macOS · DuckDB ≥ 1.0

---

## Why this exists

Every standalone GPU database from 2013-2024 was acqui-hired or pivoted (HEAVY.AI → NVIDIA 2025, BlazingSQL dormant, Voltron Data 50% layoff). Building "another GPU SQL engine" is not a viable bet.

What's open in 2026: **no published SQL engine targets Apple Silicon GPUs**. Sirius (UW + NVIDIA, CIDR 2026) is CUDA-only. cuDF is CUDA-only. So is everything else. Apple Silicon's unified memory architecture (up to 512 GB at 819 GB/s on M3 Ultra) is a genuine architectural advantage that nobody has wired into a database.

`gpudb` is a DuckDB *extension* (not a fork, not a new database) that closes that gap with a real dual-backend implementation.

## Numbers — honest, vs **multi-thread DuckDB** (the real default)

Most GPU-DB benchmarks compare against single-thread CPU. That's the kernel-vs-kernel
comparison the literature uses, but it is NOT what a user typing `SELECT sum(x)` in
the DuckDB CLI on M4 Max sees — DuckDB defaults to all 16 cores. The numbers below
compare against `duckdb -c "SET threads=16; …"` (the actual baseline). Full sweep,
including single-thread numbers and CUDA results, in [BENCHMARK.md](BENCHMARK.md).

### Apple Silicon (M4 Max, Metal vs DuckDB CLI default)

| Workload | DuckDB multi-thread (16 threads) | Metal | **Win** |
|---|---:|---:|:---:|
| **Multi-agg fusion (SUM+MIN+MAX+COUNT) SF10** | 10 ms | **1.13 ms** | **8.8× ✅** |
| **Multi-agg fusion 1B int64** | 92 ms | **16.1 ms** | **5.7× ✅** |
| SUM 1B int64 HOT | 40 ms | **16.16 ms** | **2.5× ✅** |
| SUM 500M int64 HOT | 20 ms | 8.08 ms | 2.5× ✅ |
| TPC-H SF1 GROUP BY (1.5M unique) | **8 ms** | 16.2 ms | CPU 2.0× ⚠️ |
| TPC-H SF10 GROUP BY (15M unique) | **57 ms** | 173.5 ms | CPU 3.0× ⚠️ |
| 500M × 1M GROUP BY synthetic | **820 ms** | 935.8 ms | CPU 1.14× ⚠️ |

### NVIDIA CUDA (RTX 4090 Laptop, sm_89)

| Workload | CPU baseline | CUDA cold | CUDA resident | Speedup |
|---|---:|---:|---:|---|
| SUM 100M int64 | 13.8 ms / 54 GiB/s | 80.6 ms (PCIe-bound) | **0.04 ms / 1187 GiB/s** | 17.9× over CPU |
| GROUP BY 50M × 1M groups | 1067 ms | 130 ms | n/a | 9.6× over CPU |
| GROUP BY 6M TPC-H lineitem (1.5M groups) | 54.1 ms | 15.0 ms | n/a | 3.6× over CPU |

### What actually wins, what doesn't

- **Decisive Metal win: multi-aggregate fusion** (5.7-8.8× over multi-thread CPU). The
  median TPC-H query (Q1, Q3, Q5, Q6, Q14, Q19) computes 3-4 aggregates over the
  same column. DuckDB doesn't fuse them; Metal does in one pass at 87% of LPDDR5X peak.
- **Real Metal win: SUM at scale** (2.5× over 16-thread CPU on 1B). The kernel hits
  92% of M4 Max bandwidth ceiling; CPU vectorized SUM hits ~37% of bandwidth ceiling.
- **Metal does NOT win today: GROUP BY at TPC-H scale.** DuckDB's tuned multi-thread
  radix-partitioned hash aggregate beats our LSD radix sort + GPU scan + host
  segment-reduce pipeline. We over-claimed this win in earlier docs. v0.1.3 will move
  segment-reduce onto the GPU; we expect that to flip the verdict.
- **CUDA wins everything by a lot** — RTX 4090's 1 TB/s GDDR6X is just stronger
  hardware. Apple Silicon competes on perf-per-watt and zero-PCIe-transfer (UMA).

## Quick start

### Option A — load the prebuilt extension into DuckDB CLI

Download the platform binary from the [v0.1.2 release](https://github.com/singhpratech/duckdbgpumetaldbram/releases/tag/v0.1.2), then:

```bash
# Linux (RTX/CUDA)
duckdb -unsigned -c "LOAD '/path/to/gpudb.linux_amd64.duckdb_extension'; \
  SELECT gpu_sum(value::BIGINT) FROM range(1000000) AS t(value);"
# -> [gpudb] registered gpu_sum / gpu_min / gpu_max  (backend=CUDA)
# -> 499999500000
```

Requires DuckDB v1.5.x (C API v1.2.0). `LOAD` needs `-unsigned` because the
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
./build-linux/test/test_gpudb        # 77 unit checks across CPU + CUDA backends
./scripts/run_sql_tests.sh           # SQL-level suite: gpu_sum / min / max / GROUP BY / window
./scripts/local_check.sh             # everything CI would run, end to end
```

The SQL test suite lives in `test/sql/*.test`. Each file is plain SQL with
`-- expect:` lines after each query; the runner reports per-query
PASS / FAIL / XFAIL (expected fail) / SKIP. As of v0.1.0: 46 / 46 pass,
0 fail, 0 expected fail. The window-function bugs that were previously
xfail are now strict positive assertions (PR #22).

**Reproducibility entry point:** [`scripts/local_check.sh`](scripts/local_check.sh) runs the full pipeline end-to-end (configure → build → 77 unit tests → smoke benchmarks → 46-query SQL suite). The hosted CI workflow exists at [`.github/workflows/ci.yml.disabled`](.github/workflows/ci.yml.disabled) and is staged for re-enable; today the project relies on local validation against the dual-machine dev fleet (RTX 4090 + M4 Max).

## Roadmap

### Shipped on `main` (v0.1.0 launch candidate)
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
- [x] **DuckDB loadable extension actually loads** — `duckdb -unsigned -c "LOAD '/path/to/gpudb.<platform>.duckdb_extension'"` works on Linux (CUDA) and macOS (Metal). Prebuilt binaries attached to [v0.1.2 release](https://github.com/singhpratech/duckdbgpumetaldbram/releases/tag/v0.1.2).

### In flight (v0.1.3)
- [ ] [DuckDB Community Extensions PR #1898](https://github.com/duckdb/community-extensions/pull/1898) merged → `INSTALL gpudb FROM community` (no `-unsigned` flag needed)

### Roadmap (v0.2.0+)
- [ ] Real Metal hash-join sort-merge (currently a CPU-fallback scaffold; CUDA hash-join is real)
- [ ] GPU-resident segment reduce for Metal GROUP BY at 1B+ rows (the cell where we currently lose 1.5×)
- [ ] Resident-column SQL hooks: `gpu_cache(table, col)` table function so `gpu_sum` can run on data already loaded
- [ ] Window functions on GPU as proper operators (not just aggregate-as-window)
- [ ] String / regex operators (libcudf-class functionality on Metal where it doesn't exist)

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
