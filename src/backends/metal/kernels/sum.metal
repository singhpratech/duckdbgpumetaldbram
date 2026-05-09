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

// ===================== int64 multi-agg fusion (sum+min+max+count) =====================
//
// Reads each int64 ONCE and produces four partials per threadgroup:
//   partials[block_id*4 + 0] = SUM
//   partials[block_id*4 + 1] = MIN
//   partials[block_id*4 + 2] = MAX
//   partials[block_id*4 + 3] = COUNT
//
// This is the wedge: separate sum_i64 / min_i64 / max_i64 calls each
// re-read the column. Fusing them halves (or quarters) DRAM traffic on a
// memory-bandwidth-bound workload.

constant long INIT_MIN = 0x7FFFFFFFFFFFFFFFL;   // INT64_MAX
constant long INIT_MAX = (long)0x8000000000000000L; // INT64_MIN as signed long

kernel void agg_all_i64(
    device const long*  in       [[buffer(0)]],
    device long*        partials [[buffer(1)]],   // 4 longs per block
    constant uint&      n        [[buffer(2)]],
    uint                tid      [[thread_position_in_threadgroup]],
    uint                gid      [[thread_position_in_grid]],
    uint                gsize    [[threads_per_grid]],
    uint                block_id [[threadgroup_position_in_grid]])
{
    threadgroup long shm_sum  [BLOCK];
    threadgroup long shm_min  [BLOCK];
    threadgroup long shm_max  [BLOCK];
    threadgroup long shm_cnt  [BLOCK];

    long local_sum = 0;
    long local_min = INIT_MIN;
    long local_max = INIT_MAX;
    long local_cnt = 0;

    for (uint i = gid; i < n; i += gsize) {
        long x = in[i];
        local_sum += x;
        local_min = min(local_min, x);
        local_max = max(local_max, x);
        local_cnt += 1;
    }
    shm_sum[tid] = local_sum;
    shm_min[tid] = local_min;
    shm_max[tid] = local_max;
    shm_cnt[tid] = local_cnt;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shm_sum[tid] += shm_sum[tid + s];
            shm_min[tid]  = min(shm_min[tid], shm_min[tid + s]);
            shm_max[tid]  = max(shm_max[tid], shm_max[tid + s]);
            shm_cnt[tid] += shm_cnt[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        partials[block_id * 4u + 0u] = shm_sum[0];
        partials[block_id * 4u + 1u] = shm_min[0];
        partials[block_id * 4u + 2u] = shm_max[0];
        partials[block_id * 4u + 3u] = shm_cnt[0];
    }
}

kernel void agg_all_partials_i64(
    device const long* partials [[buffer(0)]],   // 4*n longs
    device long*       out      [[buffer(1)]],   // 4 longs
    constant uint&     n        [[buffer(2)]],   // number of blocks
    uint               tid      [[thread_position_in_threadgroup]])
{
    threadgroup long shm_sum [BLOCK];
    threadgroup long shm_min [BLOCK];
    threadgroup long shm_max [BLOCK];
    threadgroup long shm_cnt [BLOCK];

    long local_sum = 0;
    long local_min = INIT_MIN;
    long local_max = INIT_MAX;
    long local_cnt = 0;

    for (uint i = tid; i < n; i += BLOCK) {
        local_sum += partials[i * 4u + 0u];
        local_min  = min(local_min, partials[i * 4u + 1u]);
        local_max  = max(local_max, partials[i * 4u + 2u]);
        local_cnt += partials[i * 4u + 3u];
    }
    shm_sum[tid] = local_sum;
    shm_min[tid] = local_min;
    shm_max[tid] = local_max;
    shm_cnt[tid] = local_cnt;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shm_sum[tid] += shm_sum[tid + s];
            shm_min[tid]  = min(shm_min[tid], shm_min[tid + s]);
            shm_max[tid]  = max(shm_max[tid], shm_max[tid + s]);
            shm_cnt[tid] += shm_cnt[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        out[0] = shm_sum[0];
        out[1] = shm_min[0];
        out[2] = shm_max[0];
        out[3] = shm_cnt[0];
    }
}

// ===================== float64 SUM =====================
// Apple Silicon GPUs do NOT support double-precision floats in MSL.
// The CUDA backend has a true f64 kernel; on Metal we keep f64 sums on the
// CPU path inside metal_aggregator.mm (transfer cost is zero on UMA, so the
// overhead is just the host-side reduction). See the host file for the
// fallback implementation.
