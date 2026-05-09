# Architecture

## Goals
- One C++ codebase compiles on Linux (CUDA + CPU) and macOS (Metal + CPU).
- Backends are plug-in: `CPU`, `CUDA`, `METAL`. Adding a backend is one TU + one entry in the factory.
- The library is independent of DuckDB. The DuckDB extension is a thin wrapper.

## Layers

```
┌─────────────────────────────────────────────────────────────┐
│  src/extension/        DuckDB extension wrapper (optional)  │
├─────────────────────────────────────────────────────────────┤
│  src/operators/        operator-level dispatch              │
│                        (will grow: aggregate, join, sort)   │
├─────────────────────────────────────────────────────────────┤
│  src/include/gpu_backend.hpp    abstract Aggregator API     │
├─────────────────────────────────────────────────────────────┤
│  src/backends/cpu/     scalar + OpenMP                      │
│  src/backends/cuda/    CUDA kernels + host wrapper          │
│  src/backends/metal/   Metal kernels + host wrapper         │
└─────────────────────────────────────────────────────────────┘
```

## Backend selection

`gpudb::default_backend()` returns the best backend that compiled into the binary AND has a usable runtime device:

```
CUDA available?  → CUDA
else Metal?      → Metal
else             → CPU
```

`gpudb::available_backends()` returns the full list. Tests and bench iterate over it.

## Memory model
- **CUDA**: explicit `cudaMalloc` + `cudaMemcpyAsync` over a single stream. Reuses device buffers across calls (`ensure_buffers`).
- **Metal** (planned): `MTLBuffer` with `MTLResourceStorageModeShared` — Apple Silicon UMA means no transfer cost, just a CPU-visible pointer the GPU can read directly.
- **CPU**: zero-copy obviously.

## Why this shape (and not a full DuckDB extension first)

1. **Faster iteration.** Standalone library compiles in 5 seconds; full DuckDB compile is minutes.
2. **Decoupled testing.** We can verify GPU kernels are numerically correct without DuckDB in the loop.
3. **Reusable.** The same library can later back a Polars plug-in, a Python module, or a custom engine.
4. **Sirius pattern.** Sirius (CIDR 2026) is a DuckDB extension that delegates all heavy lifting to libcudf — same separation we're building here.

## Reduction algorithm

Two-pass, matches the reference design across CUDA and Metal:

```
Pass 1: per-block reduction
   blocks*BLOCK threads do grid-stride load → threadgroup memory
   intra-block tree reduction → 1 partial per block
Pass 2: single-block final reduction
   read partials → tree reduction → 1 scalar
```

This avoids needing CUB/Thrust for week 1. We'll swap to CUB once we add more operators (it has well-tuned `DeviceReduce::Sum`, `DeviceScan::ExclusiveSum`, `DeviceRadixSort` — all of which we'll need).

## Future shape (not yet implemented)
- `src/operators/group_by_aggregate.cpp` — hash group-by
- `src/operators/hash_join.cpp` — radix-partitioned probe
- `src/operators/window.cpp` — the differentiator vs Sirius
- `src/extension/duckdb_gpu_extension.cpp` — registers operator overrides via DuckDB's Substrait or operator-replacement API
