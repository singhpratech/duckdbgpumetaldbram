// sum.metal — Metal compute kernels for int64/float64 reductions.
//
// IMPLEMENTATION PLAN (for macOS Claude Code on `feat/metal-sum`):
//
// Strategy: two-pass reduction matching the CUDA implementation.
//   Pass 1: per-threadgroup reduction using simdgroup intrinsics
//           (simd_sum / simd_min / simd_max) and threadgroup memory.
//           Output one partial per threadgroup.
//   Pass 2: single-threadgroup reduction over partials.
//
// Apple GPUs use 32-wide SIMD groups (per philipturner/metal-benchmarks).
// Threadgroup size: 256 (matches CUDA).
//
// Caveats per Linebender GPU sorting wiki: Apple GPUs lack forward-progress
// guarantees; do not rely on threadgroup_barrier as a global sync. Keep
// kernels single-pass-per-tile.

#include <metal_stdlib>
using namespace metal;

constant uint BLOCK = 256;

// ---------- int64 sum ----------
kernel void sum_i64(
    device const long*  in       [[buffer(0)]],
    device long*        partials [[buffer(1)]],
    constant uint&      n        [[buffer(2)]],
    uint                tid      [[thread_position_in_threadgroup]],
    uint                gid      [[thread_position_in_grid]],
    uint                gsize    [[threads_per_grid]],
    uint                block_id [[threadgroup_position_in_grid]])
{
    threadgroup long shm[BLOCK];
    long local = 0;
    for (uint i = gid; i < n; i += gsize) local += in[i];
    shm[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) shm[tid] += shm[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) partials[block_id] = shm[0];
}

// TODO(metal): sum_f64, min_i64, max_i64, partials reduction kernel.
// Keep signatures C-callable from metal_aggregator.mm via MTLLibrary lookup.
