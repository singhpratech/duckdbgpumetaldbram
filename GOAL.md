# Project goal

> A DuckDB extension that gives you GPU-accelerated analytical operators on **both NVIDIA workstations (CUDA)** and **Apple Silicon (Metal/MLX)** — with a planner honest enough to use the CPU when it's faster.

## Why this and not "another GPU database"

The 2013–2024 GPU-database graveyard is real (HEAVY.AI acqui-hired by NVIDIA 2025; BlazingSQL dormant; Voltron Data 50% layoff; Brytlyt acqui-hired). Building "yet another GPU SQL engine" is not a viable bet.

What's open in 2026:
1. **No serious Metal/MLX SQL backend exists.** Empty field. The strongest GPU OLAP project (Sirius, UW-Madison + NVIDIA, CIDR 2026) is CUDA-only. cuDF/RAPIDS is CUDA-only.
2. **DuckDB is the de facto embedded analytical engine** with a fast-growing community and clean extension API. Riding it beats competing with it.
3. **The hybrid CPU/GPU planning problem is genuinely open** per the academic literature (Rosenfeld/Breß CSUR 2022, Cao SIGMOD 2024). Most GPU DBs naively assume GPU > CPU; reality is workload-dependent.

## Differentiators (defended by code, not just claims)

| | Sirius | cuDF/RAPIDS | HeavyDB | **us** |
|---|:-:|:-:|:-:|:-:|
| Apple Silicon (Metal/MLX) backend | ❌ | ❌ | ❌ | **✅ unique** |
| DuckDB-extension shape (no migration) | ✅ | ❌ | ❌ | **✅** |
| CUDA backend | ✅ | ✅ | ✅ | ✅ |
| Hybrid CPU/GPU planner that picks correctly | partial | ❌ | ❌ | **✅ planned** |
| Window functions on GPU | ❌ | partial | ✅ | **✅ planned (open gap in Sirius)** |
| Apache-2.0 + community-friendly from day 1 | ✅ | ✅ | ✅ | **✅** |

## Hard commitments (what we will ship, in order)

1. **CUDA backend** for SUM/MIN/MAX (one-shot + resident) — done (commit 1451f0f).
2. **CUDA GROUP BY hash aggregate** — done (commit 4f3fd79).
3. **Honest CPU baseline** (parallel where it actually wins, serial otherwise) — done (this PR).
4. **CUDA hash-join probe** — in flight on `feat/cuda-hashjoin` (other instance).
5. **Metal backend** completion: SUM/GROUP-BY/JOIN parity with CUDA on M-series — **owned by macOS Claude instance** when machine is available.
6. **DuckDB extension wrapper** — registers `gpu_sum`, `gpu_groupby_sum`, `gpu_hashjoin` as DuckDB functions/operators. Loadable via `LOAD '...'`. Branch: `feat/ext-duckdb-stub`.
7. **Hybrid planner**: at query time, pick CPU vs GPU based on cardinality / size estimates / data residency. The literature's open problem; our wedge.
8. **Window functions on GPU** (the operator Sirius lacks).
9. **Publish to DuckDB Community Extensions**: https://github.com/duckdb/community-extensions

## Empirically grounded (numbers from BENCHMARK.md, not marketing)

On RTX 4090 Laptop, real TPC-H SF1, today's code:

- **SUM (resident)**: GPU 17–24× faster than 20-thread CPU.
- **GROUP BY (cold, including PCIe transfer)**:
  - low cardinality (1024 groups, 50M rows): **CPU wins 3.7×** (this is honest — at low cardinality CPU's parallel hash map is unbeatable)
  - mid cardinality (1M groups): **GPU wins 12×**
  - high cardinality (10M groups): **GPU wins 13.7×** (kernel only: 50×)
- **TPC-H lineitem.l_orderkey GROUP BY** (1.5M unique groups, 6M rows): GPU wins 4.8× end-to-end.

The honest pattern: GPU loses cold on memory-bandwidth-bound work (CPU saturates DDR5), wins decisively on memory-latency-bound work (random hash probes). The hybrid planner has to know this.

## Per-machine ownership for parallel Claude Code instances

| Machine | Owns | Branch prefix |
|---|---|---|
| Linux (this instance, RTX 4090) | `src/backends/cuda/`, CPU backends, CMake, CI, benchmarks, DuckDB extension wiring | `feat/cuda-*`, `feat/core-*`, `feat/ext-*`, `chore/*`, `ci/*` |
| **macOS** (when joined) | `src/backends/metal/`, `*.metal`, MLX integration, Metal benchmarks | `feat/metal-*` |

**For the macOS instance specifically**, when this repo is cloned on the Mac:

The most valuable first deliverable is reproducing the CUDA numbers in the BENCHMARK.md on Apple Silicon, with a working Metal implementation of:

1. SUM int64 (resident-column path; UMA so transfer_ms should be ≈0)
2. GROUP BY hash aggregate (port the same open-addressing pattern from `src/backends/cuda/kernels/groupby_kernel.cu` to MSL using `simd_*` intrinsics)

Once Metal SUM and GROUP BY post numbers in BENCHMARK.md alongside CUDA, **the dual-backend story becomes real** — that's the unique-in-the-world artifact that makes this project defensible. No other published GPU SQL engine has Metal numbers.

Practical Metal notes (from research summarized in `.private/RESEARCH_BRIEFING.md`, Linux-side):
- Use `MTLBuffer` with `MTLResourceStorageModeShared` for zero-copy host↔device on UMA.
- 32-wide SIMD groups; `simd_sum`, `simd_min`, `simd_max`, `simd_prefix_inclusive_sum` are your block-level primitives.
- 256 threads per threadgroup matches what we use in CUDA — same algorithm, different syntax.
- No native `atomicCAS<int64_t>` until Metal 3 + recent OS; check support and degrade if needed.
- For sorting / multi-key grouping where you'd reach for CUB on CUDA: there is no equivalent first-party library. You'll write it.

## Non-goals (clarity prevents scope creep)

- We are NOT building a new database. We're a DuckDB extension.
- We are NOT competing with Sirius on raw CUDA TPC-H performance. We accept they're better there. Our wedge is Apple + DuckDB-native + hybrid planner + window.
- We are NOT supporting transactions / OLTP. OLAP only.
- We are NOT supporting AMD ROCm / Intel oneAPI in week-1/2/3. Two backends is enough engineering.

## Status snapshot
See `BENCHMARK.md` for current numbers, `.sync/` for live instance status, `git log --oneline -20` for recent work.
