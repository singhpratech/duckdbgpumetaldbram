// sum.metal — Metal compute kernels for int64 / float64 reductions.
//
// Mirrors src/backends/cuda/kernels/sum_kernel.cu so the two backends
// produce identical results (modulo FP rounding).
//
// Strategy: two-pass reduction.
//   Pass 1: per-threadgroup reduction with grid-stride loop over input,
//           threadgroup memory + shared-memory tree reduction.
//           Output one partial per threadgroup.
//   Pass 2: single-threadgroup reduction over the partials → final scalar.
//
// All kernels use threadgroup size 256 to match the CUDA path.
// Apple GPUs have 32-wide SIMD groups; the tree reduction is naive (no
// simd_sum) for week 1 — we can swap in simd intrinsics later.

#include <metal_stdlib>
using namespace metal;

constant uint BLOCK = 256;

// ===================== int64 SUM =====================

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

kernel void sum_partials_i64(
    device const long* partials [[buffer(0)]],
    device long*       out      [[buffer(1)]],
    constant uint&     n        [[buffer(2)]],
    uint               tid      [[thread_position_in_threadgroup]])
{
    threadgroup long shm[BLOCK];
    long local = 0;
    for (uint i = tid; i < n; i += BLOCK) local += partials[i];
    shm[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) shm[tid] += shm[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) out[0] = shm[0];
}

// ===================== int64 MIN =====================

kernel void min_i64(
    device const long*  in       [[buffer(0)]],
    device long*        partials [[buffer(1)]],
    constant uint&      n        [[buffer(2)]],
    constant long&      init     [[buffer(3)]],
    uint                tid      [[thread_position_in_threadgroup]],
    uint                gid      [[thread_position_in_grid]],
    uint                gsize    [[threads_per_grid]],
    uint                block_id [[threadgroup_position_in_grid]])
{
    threadgroup long shm[BLOCK];
    long local = init;
    for (uint i = gid; i < n; i += gsize) local = min(local, in[i]);
    shm[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) shm[tid] = min(shm[tid], shm[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) partials[block_id] = shm[0];
}

kernel void min_partials_i64(
    device const long* partials [[buffer(0)]],
    device long*       out      [[buffer(1)]],
    constant uint&     n        [[buffer(2)]],
    uint               tid      [[thread_position_in_threadgroup]])
{
    threadgroup long shm[BLOCK];
    long local = partials[0];
    for (uint i = tid; i < n; i += BLOCK) local = min(local, partials[i]);
    shm[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) shm[tid] = min(shm[tid], shm[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) out[0] = shm[0];
}

// ===================== int64 MAX =====================

kernel void max_i64(
    device const long*  in       [[buffer(0)]],
    device long*        partials [[buffer(1)]],
    constant uint&      n        [[buffer(2)]],
    constant long&      init     [[buffer(3)]],
    uint                tid      [[thread_position_in_threadgroup]],
    uint                gid      [[thread_position_in_grid]],
    uint                gsize    [[threads_per_grid]],
    uint                block_id [[threadgroup_position_in_grid]])
{
    threadgroup long shm[BLOCK];
    long local = init;
    for (uint i = gid; i < n; i += gsize) local = max(local, in[i]);
    shm[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) shm[tid] = max(shm[tid], shm[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) partials[block_id] = shm[0];
}

kernel void max_partials_i64(
    device const long* partials [[buffer(0)]],
    device long*       out      [[buffer(1)]],
    constant uint&     n        [[buffer(2)]],
    uint               tid      [[thread_position_in_threadgroup]])
{
    threadgroup long shm[BLOCK];
    long local = partials[0];
    for (uint i = tid; i < n; i += BLOCK) local = max(local, partials[i]);
    shm[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) shm[tid] = max(shm[tid], shm[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) out[0] = shm[0];
}

// ===================== float64 SUM =====================
// Apple Silicon GPUs do NOT support double-precision floats in MSL.
// The CUDA backend has a true f64 kernel; on Metal we keep f64 sums on the
// CPU path inside metal_aggregator.mm (transfer cost is zero on UMA, so the
// overhead is just the host-side reduction). See the host file for the
// fallback implementation.
