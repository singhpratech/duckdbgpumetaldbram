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

### Phase 1 — DONE (week 1, all merged or in-PR)
1. **CUDA backend** for SUM/MIN/MAX (one-shot + resident) — ✅ done.
2. **CUDA GROUP BY hash aggregate** — ✅ done.
3. **Honest parallel CPU baseline** (cardinality-aware switch) — ✅ done.
4. **Metal backend** for SUM/MIN/MAX i64 with real compute pipelines (UMA, no transfer) — ✅ done by macOS instance.
5. **DuckDB extension wrapper** — ✅ done. Registers `gpu_sum`, `gpu_min`, `gpu_max`. NULL handling via validity mask. Batched-finalize for GROUP BY queries. `gpudb-sql` CLI demos it. PR #1 open on `feat/ext-duckdb-stub`.

### Phase 2 — LAUNCH (next 2 weeks, ~3.5 hrs/week)
The owner has explicitly chosen positioning: **"someone who really solves problems"**, time budget **30 min/day**, primary publishing channel **aivibe.org**. Ship a publishable v1 then **enhance only after established as a publisher** (2-week milestone).

6. **DuckDB Community Extension submission** — get the `.duckdb_extension` metadata footer right (use the official `duckdb/extension-template` build pipeline) and submit YAML + extension to https://github.com/duckdb/community-extensions. **The single highest-leverage action for recognition.**
7. **Launch blog post on aivibe.org** — "The first SQL execution engine for Apple Silicon GPUs" or "Why every GPU database failed and what changes in 2026". Cross-post HN + r/dataengineering.
8. **README polish** for public consumption (already done in PR #1; revise after first round of comments).

### Phase 3 — ENHANCE (after week 2, once user is established as a publisher)
The owner stated: "after 2 weeks I will enhance once I've established myself as an enhancer." Treat this as the moment to add the operators that demonstrate continued shipping cadence.

9. **CUDA hash-join probe** — same pattern as GROUP BY hash table; biggest TPC-H impact since joins dominate query time.
10. **Resident-column SQL hooks** — `gpu_cache(table, col)` table function so users can pre-load columns and query them many times. Where the GPU wins are decisive.
11. **Metal GROUP BY** with real device-side hash table — open lane for macOS instance. Currently falls back to CPU. Sirius doesn't have this either; we'd be unique.
12. **Hybrid CPU/GPU planner**: at query time, pick CPU vs GPU based on cardinality / size / residency. Empirical thresholds are documented in BENCHMARK.md; needs to be wired into the extension.
13. **Window functions on GPU** (the operator Sirius lacks per their CIDR 2026 paper). Highest-value differentiator after Apple Silicon.
14. **String / regex operators** — libcudf-class functionality is weak; opportunity for a high-impact deep-dive blog post + working code.

## For the macOS Claude Code instance specifically

**Your highest-leverage next action**: implement the Metal version of GROUP BY hash aggregate, mirroring `src/backends/cuda/kernels/groupby_kernel.cu`. Apple GPUs:
- Have `simd_min`, `simd_max`, `simd_sum`, `atomic_compare_exchange_weak_explicit` (Metal 2.4+)
- Lack 64-bit atomic CAS until very recent macOS — check support, degrade if needed
- Use 32-wide simdgroups; threadgroup memory for per-tile reduction
- See `philipturner/metal-benchmarks` for microarchitectural reference

If 64-bit atomics aren't usable on the target chip:
- Use 32-bit hash slots (split int64 into two 32-bit lanes; atomic 32-bit ops are universal)
- OR sort-based GROUP BY (radix sort the keys, then reduce_by_key)

After that lands: the project becomes the **first SQL engine in the world** with both CUDA and Metal GROUP BY working, end-to-end. That's the artifact the launch post should lead with.

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
