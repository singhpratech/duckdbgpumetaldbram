# gpudb — GPU-accelerated DuckDB on **Apple Silicon Metal + NVIDIA CUDA**

[![DuckDB Community Extension](https://img.shields.io/badge/DuckDB_Community_Extension-gpudb-FFF100?logo=duckdb&logoColor=black)](https://duckdb.org/community_extensions/extensions/gpudb)
[![Latest release](https://img.shields.io/github/v/release/singhpratech/duckdbgpumetaldbram?label=release)](https://github.com/singhpratech/duckdbgpumetaldbram/releases/latest)
[![CI](https://github.com/singhpratech/duckdbgpumetaldbram/actions/workflows/ci.yml/badge.svg)](https://github.com/singhpratech/duckdbgpumetaldbram/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Platforms](https://img.shields.io/badge/platforms-Apple_Silicon_Metal_%7C_Linux_CUDA-8A2BE2)](#quick-start)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/singhpratech/duckdbgpumetaldbram/blob/main/examples/gpudb_quickstart.ipynb)
[![GitHub stars](https://img.shields.io/github/stars/singhpratech/duckdbgpumetaldbram?style=social)](https://github.com/singhpratech/duckdbgpumetaldbram/stargazers)

> **The first SQL execution engine for Apple Silicon GPUs**, built as a DuckDB extension that *also* runs on NVIDIA CUDA. One codebase, two backends, your existing DuckDB queries.

Now an official [**DuckDB Community Extension**](https://duckdb.org/community_extensions/extensions/gpudb) — install it straight from any DuckDB ≥ 1.5.5, no flags, no downloads:

```sql
INSTALL gpudb FROM community;
LOAD gpudb;
SELECT gpu_sum(value::BIGINT) FROM range(1000000) AS t(value);
```

```sql
-- upload a column to GPU memory once, then every query runs at silicon speed
SELECT gpu_upload('qty', l_quantity::BIGINT) FROM lineitem;   -- once
SELECT gpu_sum_resident('qty');                               -- 600M rows: 10 ms vs 99 ms native
SELECT gpu_last_stats();                                      -- proof: which processor ran, kernel time
-- op=resident_i64 backend=Metal reason=Hot_GpuAlwaysWins rows=600037902
--   wall_ms=9.702 kernel_ms=9.533 transfer_ms=0.000
```

```sql
-- v0.6.0: GROUP BY / HAVING / top-k on the device. Upload a (key, payload) pair once;
-- the GPU sorts it once, then every GROUP BY is a segmented reduce over that order,
-- and HAVING / ORDER BY … LIMIT k run on the device so only the survivors come back.
SELECT gpu_upload_pair('l', l_orderkey, l_quantity::BIGINT) FROM lineitem;        -- once
SELECT * FROM gpu_groupby_sum_resident_having('l', '>', 300);   -- TPC-H Q18 inner: 75M groups → 3,182 rows
--   SF50: 78 ms vs 462–476 ms native on M4 Max (6×), 42–78 ms vs ~1 s on RTX 4090 (13–24×)
SELECT * FROM gpu_groupby_sum_resident_topk('l', 10, 'desc');   -- top-10 groups by sum: 4× Metal, 16–17× CUDA
SELECT key, sum, count FROM gpu_groupby_sum_resident('l');      -- or all 75M groups: 2.8× Metal
-- statement vs statement in the same process; verified equal to native GROUP BY both ways
```

```sql
-- v0.5.0: joins. Upload key+payload pairs once; the GPU joins and reduces
-- in one pass against a cached sorted build side — no join output materialised.
SELECT gpu_upload_pair('l', l_orderkey, (l_extendedprice*100)::BIGINT) FROM lineitem;  -- once
SELECT gpu_upload('o', o_orderkey) FROM orders;                                          -- once
SELECT gpu_join_sum_resident('l.k', 'l.v', 'o');   -- = sum(...) FROM lineitem JOIN orders
--   SF50: 37 ms vs 429 ms native on M4 Max (11.7×), 27-37 ms vs 998 ms on RTX 4090 (~30×)
-- inner / left / semi / anti × sum(BIGINT) / sum(DOUBLE) / count — all bit-exact
-- or within 1e-9 of native, verified by scripts/join_parity_check.sh
```

Apache-2.0 · v0.6.0 · Linux + macOS · DuckDB ≥ 1.2

---

## What you'd use it for

gpudb is for workloads that **ask the same aggregate questions of the same big data, over and over**. Upload a column to GPU memory once; every query after runs at memory-bandwidth speed with zero transfer.

- **📊 Dashboards & monitoring** — a metrics dashboard re-runs `SUM`s every few seconds. Resident columns turn a 99 ms scan into 10 ms (measured, 600M rows). Refresh every 5 s and the one-time upload pays for itself inside 10 minutes — then every refresh is 4–25× cheaper, forever.
- **🔬 Notebook exploration** — an analyst slicing data re-aggregates on every idea. Upload at the top of the notebook; the whole session runs GPU-speed. On a Mac this is capability that exists nowhere else: no other SQL engine uses the Apple Silicon GPU.
- **⚡ Serving APIs** — endpoints answering "total X for Y" thousands of times a day. The resident column is a cache that never goes stale-wrong: exact answers, 5–25× lower latency, re-upload in seconds when data refreshes.
- **📈 High-cardinality GROUP BY / top-k** — "quantity per order", "spend per customer", "top 10 by amount" over tens of millions of keys, re-asked as the data is explored. `gpu_groupby_sum_resident` returns `(key, sum, count)` rows (`gpu_groupby_count_resident` returns `(key, count)`) from a cached device sort, and the `_having(name, cmp, threshold)` / `_topk(name, k, order)` forms evaluate `HAVING` and `ORDER BY sum LIMIT k` **on the device** so only the survivors come back: **5.4–6.1× (Metal)** on TPC-H Q18's inner query at SF10–SF50, 6–8× on `HAVING count(*) >= 7`, 3.6–4.1× on the top-10 groups by sum — statement time vs statement time, same process. On CUDA the same device-side HAVING is **13–24× (SF50) / 17× (SF10)** and the top-10 groups 13–17×, because the 1.8 GB result copy over PCIe simply no longer happens. Returning all 15M–75M groups is 2.3–2.9× (Metal) / 1.5× end-to-end on CUDA (~32× on-device, PCIe-bound). Low-cardinality GROUP BY stays a native win on Metal (measured, kept in the table).
- **💰 DECIMAL/financial data** — money columns are stored DECIMAL, and native DuckDB re-casts every value on every scan. `gpu_upload` stores the cast once — it's why our biggest measured wins (9.9× Metal, 25× CUDA) came from the most accounting-shaped column in TPC-H.
- **🎯 Membership at scale (semi / anti join)** — "how much did *these* customers spend?", "which transactions hit the blocklist?", "how many events came from outside the cohort?" Keep the big fact side resident (`gpu_upload_pair`), re-upload only the small, changing set, and ask with `gpu_semi_join_*` / `gpu_anti_join_*`: **22× (Metal) / ~376× (CUDA)** over native on TPC-H SF10 (measured, v0.5.0).
- **🔗 Fact ⋈ dimension rollups** — revenue joined to a filtered orders/customers/dates set, re-asked per filter. `gpu_join_sum_resident` / `gpu_left_join_count_resident` run the fused join-aggregate on the device with the sorted build side cached: **11.7× / 27–37×** at SF50. DOUBLE payloads via the `_f64` variants (within the 1e-9 relative tolerance contract; measured ≤4e-11).

**Not for:** one-shot queries on cold data (transfer loses — the streaming `gpu_sum/min/max` deliberately match native there), or `min`/`max` where DuckDB's statistics answer without scanning, or joins that must return the matched *rows* at scale (`gpu_join_rows_resident` works, but native DuckDB wins on discrete GPUs — use the aggregate variants). [KNOWN_ISSUES.md](KNOWN_ISSUES.md) lists every trade-off honestly.

## Numbers — measured, not promised

TPC-H `lineitem`, warm cache, every result verified equal to native before
timing counted. GROUP BY rows: statement against statement inside the same
embedded DuckDB v1.5.2 process, after the one-time upload and sort; aggregate
and join rows: DuckDB CLI (v1.5.5 for CUDA, v1.5.2 for the Metal joins),
5-run medians. Full grid + reproduction:
**[BENCHMARK.md](BENCHMARK.md)**.

| TPC-H | Workload | Hardware | Native | gpudb | |
|---|---|---|---:|---:|:---|
| **SF50** (300M rows → 75M groups) | Q18 inner: `GROUP BY l_orderkey HAVING sum > 300`, HAVING on the device | RTX 4090 Laptop · CUDA | 1012–1039 ms | **42–78 ms** (kernel 37–72, bimodal laptop clocks) | **13–24× 🚀** |
| **SF10** (60M rows → 15M groups) | Q18 inner, HAVING on the device | RTX 4090 Laptop · CUDA | 208–210 ms | **11.7–12.9 ms** (kernel 8–10) | **16–18× 🚀** |
| **SF50** (300M rows → 75M groups) | top-10 groups by `SUM` (`ORDER BY sum DESC LIMIT 10`) | RTX 4090 Laptop · CUDA | 1009–1023 ms | **60–64 ms** (kernel 52–56) | **16–17× 🚀** |
| **SF50** (300M rows → 75M groups) | Q18 inner: `GROUP BY l_orderkey HAVING sum > 300`, HAVING on the device | MacBook M4 Max · Metal | 462–476 ms | **78 ms** (kernel 77) | **5.9–6.1× 🚀** |
| **SF50** (300M rows → 75M groups) | `HAVING count(*) >= 7` (10.7M groups survive) | MacBook M4 Max · Metal | 413–438 ms | **56 ms** (kernel 50) | **7.3–7.8× 🚀** |
| **SF50** (300M rows → 75M groups) | top-10 groups by `SUM` (`ORDER BY sum DESC LIMIT 10`) | MacBook M4 Max · Metal | 463–471 ms | **114 ms** (kernel 112) | **4.0× 🚀** |
| **SF50** (300M rows → 75M groups) | `GROUP BY l_orderkey` + `SUM` BIGINT, all 75M groups returned | MacBook M4 Max · Metal | 347–355 ms | **120–123 ms** (operator 87–90) | **2.8–2.9× 🚀** |
| **SF50** (300M rows → 75M groups) | `GROUP BY l_orderkey` + `SUM` BIGINT, all 75M groups returned | RTX 4090 Laptop · CUDA | 681–718 ms | **458–471 ms** (kernel 21 ms) | **1.5× end-to-end, ~32× on-device** |
| **SF50** (300M ⋈ 75M) | `JOIN` + `SUM` BIGINT | RTX 4090 Laptop · CUDA | 998 ms | **27–37 ms** | **27–37× 🚀** |
| **SF10** (60M ⋈ 15M) | `EXISTS` semi-join + `SUM` DOUBLE | RTX 4090 Laptop · CUDA | 640 ms | **1.7 ms** | **~376× 🚀** |
| **SF50** (300M ⋈ 75M) | `JOIN` + `SUM` BIGINT | MacBook M4 Max · Metal | 429 ms | **37 ms** | **11.7× 🚀** |
| **SF10** (60M ⋈ 15M) | `EXISTS` semi-join + `SUM` DOUBLE | MacBook M4 Max · Metal | 182 ms | **8.2 ms** | **22.2× 🚀** |
| **SF50** (300M rows) | `SUM` BIGINT | RTX 4090 Laptop · CUDA | 99 ms | **4 ms** | **25× 🚀** |
| **SF100** (600M rows) | `SUM` DOUBLE | RTX 4090 Laptop · CUDA | 196 ms | **9 ms** | **22× 🚀** |
| **SF100** (600M rows) | `SUM` BIGINT | MacBook M4 Max · Metal | 99 ms | **10 ms** | **9.9× 🚀** |
| **SF50** (300M rows) | `SUM` BIGINT | MacBook M4 Max · Metal | 46 ms | **5 ms** | **8.5× 🚀** |
| **SF10** (60M rows) | `SUM` BIGINT | MacBook M4 Max · Metal | 9.6 ms | **1.4 ms** | **6.9× 🚀** |

**The bigger the scale factor, the bigger the win** — the resident kernel runs
at the memory-bandwidth ceiling of the silicon (563 GB/s CUDA, 503 GB/s
Metal) while native's scan grows linearly. Ratio curve on Metal:
2× (SF1) → 5× → 6.9× → 6.8× → 8.5× → 9.9× (SF100); on CUDA: 4× → 10× →
20× → 24× → 25× → 21×.

Honest asterisks, on the table not under it: TPC-H's DECIMAL-stored columns
make native pay a cast per scan while the resident column stores it once —
already-BIGINT columns win 3.3–3.7× (Metal) / 5.6–10× (CUDA). Whole-column
`min`/`max` on stored tables stays a **native win** (zonemap statistics answer
without scanning). One-time upload breaks even after ~100–150 repeated
aggregate queries; the GROUP BY rows assume the pair is resident — on Metal the
upload is ~1 s at SF10 / ~6 s at SF50 and the first call pays the sort
(0.3–3 s), on CUDA the upload is 4–5 s / 19–25 s — which the device-side
HAVING / top-k rows recoup after ~20–25 repeated queries and the all-groups
rows after ~40 (Metal) to ~90 (CUDA). Returning every group is 1.5–3× — the larger ratios come from
running the `HAVING` / `LIMIT` on the device; the CUDA all-groups row is
bounded by copying 24 bytes per group over PCIe, which unified memory
does not pay. `DOUBLE` sums filter on the host on Metal (1.4–2.6×). Low-cardinality GROUP BY (a handful of groups) is a
**native win on Metal** (0.56–0.65× at SF10–SF50, a tie at SF1) and stays in the table. All in
[BENCHMARK.md](BENCHMARK.md) and [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

## The resident model in 20 seconds

```sql
INSTALL gpudb FROM community; LOAD gpudb;

SELECT gpu_upload('sales', amount::BIGINT) FROM orders;  -- pay the transfer once
SELECT gpu_sum_resident('sales');                        -- every query after: GPU speed
SELECT gpu_min_resident('sales'), gpu_max_resident('sales');
SELECT gpu_sum_resident_f64('price');                    -- DOUBLE flavor
SELECT gpu_resident_info('sales');                       -- dtype / rows / device
SELECT gpu_last_stats();                                 -- which backend ran + kernel time
SELECT gpu_build_info();                                 -- which backends this binary carries
SELECT gpu_drop_resident('sales');                       -- free device memory

-- v0.7 registry (milestone 0b): every resident set, with identity, state, size, hits
SELECT * FROM gpu_residents();
SELECT gpu_prepare_resident('l');          -- build the sort cache now instead of on the first GROUP BY
SELECT gpu_invalidate('l');                -- mark a set stale (exact name, or an identity-tag prefix)
-- staleness guard for a rewritten statement: raises "GPUDB_STALE: ..." unless the
-- table still has the row count the upload saw; the one-row derived table runs once
SELECT r.key, r.sum
FROM gpu_groupby_sum_resident('gpudb:v1:memory:main:lineitem:1:l_orderkey,l_quantity') r,
     (SELECT gpu_assert_rows('gpudb:v1:memory:main:lineitem:1:l_orderkey,l_quantity', count(*)) AS ok
      FROM lineitem) gd
WHERE gd.ok;
```

A set name of the form `gpudb:v1:<catalog>:<schema>:<table>:<table_oid>:<col1>[,<col2>…][:<extra>]`
is an **identity tag**: the set is `managed` (what the v0.7 transparent path
consumes), its fields show in `gpu_residents()`, and it is prepared at upload
so the first query pays no sort. Any other name is an `explicit` set with the
v0.6 behaviour. A pair is one registry entry (`'l.k'` / `'l.v'` address its
columns), an operator call keeps its set alive until it finishes, and uploads
never block queries on another connection.

⚠️ **One footgun:** in a single-statement upload-and-query, the outer query
must reference the upload's result column (e.g. `SELECT u.n, gpu_sum_resident('x') FROM (SELECT gpu_upload('x', col) AS n FROM t) u`) —
an unreferenced `gpu_upload` is pruned by DuckDB's optimizer and never runs.
Uploads are capped at 4 GB of buffering by default
(`GPUDB_UPLOAD_POOL_MAX_MB` to raise); the streaming `gpu_sum/min/max`
aggregates work in any query shape (GROUP BY, windows, FILTER) at native
parity.

## Why this exists

Every standalone GPU database from 2013-2024 was acqui-hired or pivoted (HEAVY.AI → NVIDIA 2025, BlazingSQL dormant, Voltron Data 50% layoff). Building "another GPU SQL engine" is not a viable bet.

What's open in 2026: **no published SQL engine targets Apple Silicon GPUs**. Sirius (UW + NVIDIA, CIDR 2026) is CUDA-only. cuDF is CUDA-only. So is everything else. Apple Silicon's unified memory architecture (up to 512 GB at 819 GB/s on M3 Ultra) is a genuine architectural advantage that nobody has wired into a database.

`gpudb` is a DuckDB *extension* (not a fork, not a new database) that closes that gap with a real dual-backend implementation. Operator-level benchmarks (GROUP BY, multi-aggregate fusion, hash join) live in [BENCHMARK.md](BENCHMARK.md)'s earlier entries.

## Quick start

### Option A — install from the DuckDB community repo (recommended)

```sql
INSTALL gpudb FROM community;
LOAD gpudb;
SELECT gpu_sum(value::BIGINT) FROM range(1000000) AS t(value);
-- -> 499999500000
```

Works in any DuckDB ≥ 1.5.5 client (CLI, Python, etc.), signed, no flags needed.
The registry binary carries the **full Metal backend on Apple Silicon**. On
Linux the registry binary is **CPU-only** — the community build machines have
no CUDA toolchain, so every `gpu_*` function works and returns the same
results, but `gpu_last_stats()` will say `backend=CPU`. For the CUDA backend
on Linux use the release binary (Option B; statically linked CUDA runtime,
needs only a driver) or build from source with `nvcc`. Check any binary with
`SELECT gpu_build_info();`.
The registry serves the **v0.5.0** build (merged 2026-08-24), including the
full resident-column surface (`gpu_upload`, `gpu_sum_resident`, `gpu_build_info`,
…) and the GPU join functions (`gpu_upload_pair`, `gpu_join_*_resident`,
`gpu_join_rows_resident`). Installed an earlier version? `UPDATE EXTENSIONS;`
pulls the latest.

### Option B — load a prebuilt release binary

Download the platform binary from the [latest release](https://github.com/singhpratech/duckdbgpumetaldbram/releases/latest), then:

```bash
# Linux (RTX/CUDA)
duckdb -unsigned -c "LOAD '/path/to/gpudb.linux_amd64.duckdb_extension'; \
  SELECT gpu_sum(value::BIGINT) FROM range(1000000) AS t(value);"
# -> [gpudb] registered gpu_sum / gpu_min / gpu_max (BIGINT,DOUBLE) streaming aggregates (backend=CUDA)
# -> 499999500000
```

Requires DuckDB ≥ 1.2 (C API v1.2.0); release binaries track the latest tag.
`LOAD` needs `-unsigned` here because release-page binaries are unsigned —
the community install above doesn't.

### Option C — build from source

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

#### CUDA requirements (build from source on Linux)

| | Supported | Notes |
|---|---|---|
| **CUDA Toolkit** | **13.0** (verified: 13.0.88 / CUB 3.0.1, all benchmarks) | Runtime API plus the CUB that ships with the toolkit (`DeviceReduce`, `DeviceSelect`, `DeviceRadixSort`, `DeviceScan` on explicit temp storage; Thrust only for iterators; no cooperative groups). 64-bit item counts need **CUB ≥ 2.1, i.e. CUDA 12.2 or newer**; older 12.x narrows counts to 32-bit (fine below 2^31 rows) and 11.x is not supported. Only 13.0 is tested by us — if you build on 12.x, please open an issue with your `nvcc --version` either way. C++17 host + device. |
| **NVIDIA driver** | **580.x** (verified) | Any driver that supports your toolkit (NVIDIA's minimum for 13.0 is R580; for 12.x, R525+). Runtime linking: `-DGPUDB_CUDA_STATIC_RUNTIME=ON` (what the registry build in the root `Makefile` uses; off by default in `scripts/build.sh`) bakes `cudart` into the extension, so the only runtime dependency is `libcuda.so` from the driver — and the extension still loads on machines with no GPU/driver, falling back to CPU. |
| **GPUs** | **sm_75 – sm_90**: Turing (T4, RTX 20xx), Ampere (A100, RTX 30xx), Ada (RTX 40xx, L4/L40), Hopper (H100) | Default fatbin: `75;80;86;89;90`, each with SASS + PTX. Newer parts (Blackwell / RTX 50xx, sm_100+) load via PTX JIT from `compute_90` — should work, not yet measured. **Volta (sm_70) and older are not supported**: CUDA 13 dropped them from `nvcc`. Override with `-DCMAKE_CUDA_ARCHITECTURES=...` or `CUDAARCHS=...` (the Colab notebook builds `CUDAARCHS=75` for its T4). |

Verified configuration: RTX 4090 Laptop (sm_89, 16 GB), CUDA 13.0.88, driver
580.x, Linux — every CUDA number in this README and BENCHMARK.md comes from
that box. `SELECT gpu_build_info();` reports whether any given binary was
compiled with CUDA and which backend it picked at runtime.

### Option D (TPC-H reproducibility)

```bash
# get TPC-H SF1 data (downloads DuckDB CLI to .tools/, ~1 GB lineitem)
SF=1 ./scripts/gen_tpch.sh

duckdb -unsigned -c "LOAD '$(pwd)/build-linux/src/extension/gpudb.linux_amd64.duckdb_extension'; \
  SELECT gpu_sum(v) FROM read_parquet('data/tpch_sf1/lineitem_orderkey.parquet') t(v);"
# -> 18005322964949
```

## Run it in CI or a notebook

GitHub's `macos-14`/`macos-15` hosted runners are **Apple Silicon machines** —
gpudb's Metal path runs in free GitHub Actions with zero setup, which makes it
(as far as we know) the only DuckDB extension that does anything special
there. Copy-paste workflows for Apple Silicon runners, Linux runners, Docker,
and self-hosted CUDA boxes: **[docs/CI_RECIPES.md](docs/CI_RECIPES.md)**.

Prefer a notebook? **[examples/gpudb_quickstart.ipynb](examples/gpudb_quickstart.ipynb)**
opens directly in Google Colab — registry install + parity checks anywhere,
plus an optional build-from-source section that runs the CUDA benchmarks on
Colab's free T4 GPU.

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
│  │  - streaming aggregate states      │  │
│  │      ↓ (operator-level / join)     │  │
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

Two SQL paths, by design (v0.4.0):

- **Streaming** `gpu_sum/min/max` — CPU-shaped running accumulators, native
  parity in any query shape. Deliberate: the v0.2.0 numbers in
  [BENCHMARK.md](BENCHMARK.md) showed per-query buffering-for-GPU loses
  3×–110× through this interface.
- **Resident** `gpu_upload` + `gpu_*_resident` — the GPU path with substance:
  pay the transfer once, then reductions run on-device (CUDA and Metal) at
  memory-bandwidth speed with `transfer_ms=0.000`. This is where the
  4–25× numbers above come from.
- **Resident joins (v0.5.0)** `gpu_upload_pair` + `gpu_[left_|semi_|anti_]join_{sum,count}_resident[_f64]`
  — fused join + reduction against a device-cached sorted build side; the
  11–376× join rows above. Row-returning `gpu_join_rows_resident` exists as
  the composability primitive (wins on unified memory, loses to native across
  PCIe — [BENCHMARK.md](BENCHMARK.md) states both).

## Testing
```bash
./build-linux/test/test_gpudb        # unit checks across the backends present at build time (118 CUDA / 137 Metal)
./scripts/run_sql_tests.sh           # SQL-level suite: gpu_sum / min / max / GROUP BY / window / resident / joins
./scripts/join_parity_check.sh       # 11 adversarial join scenarios, native vs gpudb in the same statement
./scripts/local_check.sh             # everything CI would run, end to end
```

The SQL test suite lives in `test/sql/*.test`. Each file is plain SQL with
`-- expect:` lines after each query; the runner reports per-query
PASS / FAIL / GUARDRAIL / SKIP. As of v0.6.0: 126 queries, 0 failures — 113
result checks plus 13 GUARDRAIL cases (deliberate misuse such as a `DOUBLE` join
key or a `NULL` upload name, which the extension must reject with a clear
error; the suite fails if one of them unexpectedly succeeds) — plus full
resident-surface coverage in the community-CI sqllogic suite.

**Reproducibility entry point:** [`scripts/local_check.sh`](scripts/local_check.sh) runs the full pipeline end-to-end (configure → build → unit tests → smoke benchmarks → SQL suite → join parity harness). The hosted CI workflow lives at [`.github/workflows/ci.yml`](.github/workflows/ci.yml) (Linux + macos-15) and runs on every push to `main`.

## Roadmap

### Latest — v0.6.0
- [x] **Resident GROUP BY / top-k from SQL, both backends** — `gpu_groupby_sum_resident` / `gpu_groupby_sum_resident_f64` / `gpu_groupby_count_resident` return `(key, sum, count)` rows sorted by key; `gpu_topk_resident[_f64]` returns `(idx, value)` for `ORDER BY … LIMIT k`. Rides the upload-once model and the same cached device sort the joins use as a build side (one sort serves both). Segmented reduce with no hash table and no atomics on Metal; CUB `reduce_by_key` on CUDA. `_having(name, cmp, threshold)` and `_topk(name, k, order)` forms evaluate `HAVING` / `ORDER BY aggregate LIMIT k` on the device (Metal: block compaction + 8-pass radix select; CUDA: CUB select + 8-pass radix select) so only survivors cross to DuckDB. Verified against native both ways on TPC-H SF1/10/50 — statement time against statement time in the same process: **5.4–6.1× (Metal)** on Q18's inner query with the HAVING on the device, 6–8× on `HAVING count(*) >= 7`, 3.6–4.1× on the top-10 groups by sum; on CUDA the device-side HAVING is **13–24× (SF50) / 17× (SF10)** and the top-10 groups 8–12×; returning all 15M–75M groups is 2.3–2.9× (Metal) and 1.5–2.0× end-to-end on CUDA (~32–47× on-device, bounded by copying 24 bytes per group over PCIe). Honest losing rows kept: low-cardinality GROUP BY on Metal, the first top-k call vs native's zonemap top-k.
- [x] **Composable results** — the GPU produces the rows, DuckDB does the rest: `SELECT key, sum FROM gpu_groupby_sum_resident('l') WHERE sum > 300 ORDER BY sum DESC LIMIT 10` is plain SQL over a small result.
- [x] **Adversarial parity harness for GROUP BY** — `scripts/groupby_parity_check.sh`: 11 scenarios × 7 checks, incl. runs placed exactly on the kernels' 64-chunk / 256-block boundaries; SQL suite gained a `-- setup:` directive so table functions are tested in the documented sequential form.
- [x] **Metal radix-sort fix** — the sort behind the v0.5 join build cache skipped a byte pass whenever min and max agreed on that byte; wrong for keys between them that differ there (TPC-H returnflag/linestatus packed keys). Fixed, regression scenarios in both parity harnesses; exposure of the v0.5.0 Metal binary stated in [KNOWN_ISSUES.md](KNOWN_ISSUES.md).
- [x] **Metal device memory is now released** — the Metal host code was built without ARC since v0.1, so every `MTLBuffer` (resident columns, sort caches, scratch) leaked until process exit; `gpu_drop_resident` now actually frees GPU memory.
- [x] **Pre-release adversarial audit** — 65-agent find/verify pass over the sort, kernels, C-API layer, hybrid planner, SQL semantics vs native (NULLs, overflow, NaN/-0.0), the v0.5 join surface after the sort fix, the CUDA branch (static) and every documentation claim; all confirmed findings fixed or documented in [KNOWN_ISSUES.md](KNOWN_ISSUES.md) before the tag.
- [x] Design: [docs/GROUPBY_RESIDENT_DESIGN.md](docs/GROUPBY_RESIDENT_DESIGN.md).

### Shipped in v0.5.0
- [x] **GPU joins from SQL, both backends** — `gpu_upload_pair` + fused `gpu_[left_|semi_|anti_]join_{sum,count}_resident[_f64]` (inner / left / semi / anti; right / full as documented compositions) and the row-returning `gpu_join_rows_resident`. Sorted-build + binary-search probe with the sorted side cached on the device. Verified against native DuckDB's hash join end-to-end on TPC-H SF10/SF50: **11.7× (Metal) / 27–37× (CUDA)** inner join-sum at SF50, **22× / ~376×** on the EXISTS semi-join; i64 bit-exact, f64 within the 1e-9 relative tolerance contract (measured ≤4e-11). Honest losing row kept: row materialisation across PCIe loses to native on discrete GPUs.
- [x] **Adversarial parity harness** — `scripts/join_parity_check.sh`: 11 scenarios × 12 checks (dup-heavy, Knuth-hash, Zipf skew, int64 boundaries, negative keys, no-match, all-match, inverted sizes), native and gpudb computed in the same statement; passes on both machines.
- [x] **Metal hash join + hybrid join planner + on-device segment reduce** — contributed by [@lmangani](https://github.com/lmangani) ([PR #43](https://github.com/singhpratech/duckdbgpumetaldbram/pull/43)); this release lands that commit as the base of the join stack.
- [x] **Colab notebook runs real CUDA** — requests a T4 runtime automatically, builds with `GPUDB_REQUIRE_CUDA=1` (configure fails loudly instead of silently falling back to CPU), and the test cell asserts the CUDA backend actually ran.

### Shipped in v0.4.0
- [x] **Resident-column SQL surface** — `gpu_upload` / `gpu_sum_resident` / `gpu_min_resident` / `gpu_max_resident` / `gpu_sum_resident_f64` / `gpu_resident_info` / `gpu_last_stats` / `gpu_drop_resident` / `gpu_build_info`. The GPU genuinely executes SQL reductions on both CUDA and Metal — up to **25×** over native (see Numbers). Hardened by a three-reviewer adversarial pass pre-release: buffer-pool cap (window-frame O(n²) OOM → clean error), mixed-name/NULL-name guards, defined overflow wrap, truthful dispatch stats.
- [x] **CUDA-ready community build** — the root Makefile auto-detects nvcc with a statically linked CUDA runtime (no libcuda/libcudart dynamic deps; loads clean on GPU-less machines) so the registry's Linux binary flips to CUDA automatically when the registry's build tooling ships its CUDA toolchain. Also fixed a CMake ordering bug that had every prior CUDA build shipping single-arch fatbins.
- [x] **Full dual-platform benchmark record** — TPC-H SF1→SF100, six columns, correctness-gated, both backends, in [BENCHMARK.md](BENCHMARK.md).
- [x] [Community Extensions PR #2503](https://github.com/duckdb/community-extensions/pull/2503) **merged** (2026-08-17) — the registry now serves **v0.4.0**, resident-column surface included.

### In flight
- [x] **v0.7 milestone 0b — residency prerequisite** (PR): the resident registry keyed per database with identity tags and origin, per-set state (`ready` = uploaded *and* prepared, `stale`, epoch, hits, references), `gpu_residents()`, `gpu_assert_rows()` as the in-statement staleness guard, `gpu_invalidate()`, `gpu_prepare_resident()`; a lock-free `gpu_upload` update path (the one contended atomic it had cost a 60M-row upload 1.8 s → 0.36 s and starved concurrent native queries 3–7×), segmented host buffers with no combine copy, and on CUDA the pair split on the device and uploads on the column's own stream. `scripts/residency_gate.sh` measures a native statement during a concurrent upload against DuckDB's own `list()` as the control. Design: [docs/TRANSPARENT_DESIGN.md](docs/TRANSPARENT_DESIGN.md) §5.3–§5.6.
- [x] **v0.7 milestone 2 (extension side) — the statement rewriter** (PR): `gpu_rewrite_ast(tree, context)`, a pure scalar over DuckDB's own `json_serialize_sql` tree, turns `SELECT k, sum(v) FROM t GROUP BY k [HAVING sum(v) > c] [ORDER BY sum(v) DESC LIMIT n]` into the resident table function plus the `gpu_assert_rows` guard, with native output names and types (HUGEINT sums, DECIMAL(38,s) for DECIMAL payloads with exactly rescaled thresholds), and returns every other statement unchanged with a reason. Verified three ways (native / rewritten / explicit `gpu_*`, `scripts/rewrite_parity_check.sh`) incl. TPC-H SF1 Q18-inner; 0.045 ms to reject a statement, 0.14 ms to rewrite one. The Python wrapper that drives it (classification, name resolution, template cache, fallback) is the macOS side's PR.
- [ ] **Community packaging phase 2** — `requires_toolchains: "python3;cuda"` registry update (the build side shipped in v0.4.0; the token activates when the registry adopts CUDA-capable ci-tools — `SELECT gpu_build_info();` shows what any given binary carries).

### Next (v0.7.0 — directional)
- [ ] **Transparent operator substitution** — plain `SELECT k, sum(v) FROM t GROUP BY k` and `… JOIN …` routed to the resident operators without calling `gpu_*` functions, behind a `SET` switch, with everything unsupported falling through to native. The DuckDB **C API** exposes no optimizer hooks, but the C++ extension API does (`OptimizerExtension`); this means moving the loadable off the stable C ABI onto per-version builds, which is what most community extensions do anyway. The v0.6 table functions are the rewrite targets.
- [ ] GROUP BY over join results as a fused op (join multiplicity × payload → segmented reduce); composite join keys
- [ ] Resident f64 min/max (a small ABI entry)

### Beyond (exploratory)
- [ ] Window functions on GPU as proper operators (not just aggregate-as-window)
- [ ] String / regex operators (libcudf-class functionality on Metal where it doesn't exist)

<details>
<summary><b>Earlier releases</b> (v0.1.0 → v0.3.0 + community-extension milestones)</summary>

### Shipped in v0.3.0
- [x] **Streaming aggregate states** — the SQL aggregate path rewritten from "buffer every value, reduce at finalize" to running accumulators, the same algorithmic shape as native DuckDB. End-to-end on rewritten TPC-H Q6/Q1 and high-cardinality GROUP BY: parity with native (the v0.2.0 buffered path lost 3×–110×; the SF10 GROUP BY cell alone went from 11.05 s to 0.110 s). Full before/after in [BENCHMARK.md](BENCHMARK.md). `GPUDB_FORCE_BACKEND` is a no-op on this path now (it routed the deleted machinery).
- [x] **`gpu_min(DOUBLE)` / `gpu_max(DOUBLE)`** — all three aggregates are now overload sets carrying `(BIGINT)->BIGINT` and `(DOUBLE)->DOUBLE`. No backend-interface change was needed under the streaming design. NaN ordering matches native (NaN sorts greatest). Type matrix in [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

### DuckDB Community Extension milestones
- [x] [Community Extensions PR #1898](https://github.com/duckdb/community-extensions/pull/1898) **merged** — `INSTALL gpudb FROM community` is live (no `-unsigned` flag needed), and gpudb is [listed on duckdb.org](https://duckdb.org/community_extensions/extensions/gpudb).
- [x] [Community Extensions PR #2404](https://github.com/duckdb/community-extensions/pull/2404) **merged** — the community build ships **v0.3.0** (streaming aggregates + DOUBLE overloads on all four platforms).

### Shipped in v0.2.0
- [x] **SQL-correct NULL semantics** (PR #44) — `gpu_sum`/`gpu_min`/`gpu_max` over empty or all-NULL input now return SQL `NULL` (not 0), matching native DuckDB on every path: plain aggregate, GROUP BY groups, and window frames.
- [x] **`gpu_sum(DOUBLE) -> DOUBLE`** (PR #45) — a real second overload via the C API aggregate function set. Doubles ride the existing int64 state machinery as raw bit patterns (zero state-layout change); only the finalize differs. `INTEGER`/`SMALLINT`/`TINYINT` work via DuckDB's implicit widening to the `BIGINT` overload (locked in by tests). Type matrix in [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

### Shipped in v0.1.3
- [x] **Hybrid Metal GROUP BY** — 32K-partition slot-lock + radix-opt with auto-dispatch (env override `GPUDB_METAL_GROUPBY_PATH`). Flipped TPC-H SF10 `l_orderkey` (15M unique) from CPU 1.78× faster to Metal 1.30× faster vs DuckDB CPU 16-thread. 9 wins / 1 honest loss on the lineitem scorecard.
- [x] Prebuilt v0.1.3 binaries (Linux CUDA + macOS Metal) attached to the [v0.1.3 release](https://github.com/singhpratech/duckdbgpumetaldbram/releases/tag/v0.1.3).

### Shipped in v0.1.2
- [x] **All 4 known window/GROUP BY bugs fixed** (PR #18, #20, #21, #22 — see KNOWN_ISSUES.md)
- [x] **DuckDB loadable extension actually loads** — `duckdb -unsigned -c "LOAD '/path/to/gpudb.<platform>.duckdb_extension'"` works on Linux (CUDA) and macOS (Metal).

### Shipped v0.1.0 – v0.1.2 (foundation)
- [x] CUDA backend: SUM/MIN/MAX (one-shot + resident)
- [x] CUDA GROUP BY hash aggregate (open-addressing + atomicCAS, ~520 GiB/s on RTX 4090)
- [x] **CUDA hash join probe** (1M build × 10M probe @ 97% sel: 3.7× wall, 107× kernel over CPU)
- [x] Metal backend: SUM/MIN/MAX i64 with real compute pipelines (~470 GiB/s on M4 Max)
- [x] **Metal GROUP BY** via GPU-resident radix sort (wins 4.4–4.8× over CPU at 100M-500M × 1M groups)
- [x] **Multi-aggregate fusion** (SUM+MIN+MAX+COUNT in one pass, 5.3× over CPU fused)
- [x] **Hybrid CPU/GPU planner** (HybridAggregator + DispatchDecision, beats both pure-CPU and pure-GPU at the 1M×1M sweet spot)
- [x] DuckDB extension: gpu_sum / gpu_min / gpu_max with NULL handling + GPUDB_FORCE_BACKEND env var
- [x] CLI: gpudb-bench, gpudb-groupby-bench, gpudb-window-bench, gpudb-hashjoin-bench, gpudb-sql

</details>

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
