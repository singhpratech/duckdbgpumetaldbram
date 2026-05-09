// groupby.metal — Apple-native GROUP BY: bitonic sort on GPU.
//
// Why sort and not a hash table:
//   The CUDA path uses an open-addressing hash table whose insertion needs
//   64-bit atomic compare-and-swap. Apple Silicon GPUs implement neither
//   64-bit atomic CAS nor 64-bit atomic_fetch_add on `device` storage
//   (the MSL 3.2 SFINAE predicates reject both). The standard alternative
//   for GPU GROUP BY without atomic CAS is sort-then-segment-reduce.
//
// Algorithm: bitonic sort of (key, value) pairs by key, in-place, across
// the whole input. After this kernel runs, host-side does a one-pass
// segment-reduce. Padding with (INT64_MAX, 0) brings N up to a power of 2
// and is harmless: padding sorts to the end, contributes 0 to any sum.
//
// Two-tier dispatch strategy to minimize kernel-launch overhead:
//   1. `bitonic_local_sort_i64` fully sorts each CHUNK-sized window in
//      threadgroup memory. ALL stages with k ≤ CHUNK happen in this one
//      dispatch. log₂(CHUNK)·(log₂(CHUNK)+1)/2 stages collapsed.
//   2. For larger k:
//      a. `bitonic_step_i64` handles cross-block stages where j > CHUNK/2.
//      b. `bitonic_local_merge_i64` finishes the inner stages
//         j = CHUNK/2, CHUNK/4, ..., 1 inside threadgroup memory in one
//         dispatch.
//
// For N=1M, CHUNK=512: ~78 dispatches total (down from ~210 in the
// dispatch-per-stage version), since bigger chunks of inner work fit in
// shared memory with `threadgroup_barrier` providing in-tg ordering.

#include <metal_stdlib>
using namespace metal;

constant uint CHUNK    = 512;   // elements sorted per threadgroup
constant uint TGSIZE   = 256;   // threads per threadgroup; each handles 2 elements
constant long PAD_KEY  = 0x7FFFFFFFFFFFFFFFL;   // INT64_MAX

// One stage of bitonic sort. Used for cross-block stages (j > CHUNK/2).
kernel void bitonic_step_i64(
    device long*  keys     [[buffer(0)]],
    device long*  values   [[buffer(1)]],
    constant uint& n       [[buffer(2)]],
    constant uint& k       [[buffer(3)]],
    constant uint& j       [[buffer(4)]],
    uint           gid     [[thread_position_in_grid]])
{
    if (gid >= n) return;
    const uint partner = gid ^ j;
    if (partner <= gid) return;
    if (partner >= n)   return;

    const long ka = keys[gid];
    const long kb = keys[partner];
    const bool ascending = ((gid & k) == 0u);
    const bool out_of_order = ascending ? (ka > kb) : (ka < kb);
    if (out_of_order) {
        const long va = values[gid];
        const long vb = values[partner];
        keys[gid]       = kb;
        keys[partner]   = ka;
        values[gid]     = vb;
        values[partner] = va;
    }
}

// Pad the tail of (keys, values) with (INT64_MAX, 0) so length is a power of 2.
kernel void bitonic_pad_i64(
    device long*  keys    [[buffer(0)]],
    device long*  values  [[buffer(1)]],
    constant uint& n_real [[buffer(2)]],
    constant uint& n_pad  [[buffer(3)]],
    uint           gid    [[thread_position_in_grid]])
{
    const uint i = n_real + gid;
    if (i >= n_pad) return;
    keys[i]   = PAD_KEY;
    values[i] = 0L;
}

// Sort each contiguous CHUNK-element window in threadgroup memory.
// One dispatch covers k = 2, 4, ..., CHUNK (all stages with k ≤ CHUNK).
// Direction uses the GLOBAL index (base + local_i) so that chunks at even
// positions sort ascending and odd-positioned chunks sort descending,
// producing the bitonic structure the next k = 2·CHUNK stage expects.
kernel void bitonic_local_sort_i64(
    device long*  keys    [[buffer(0)]],
    device long*  values  [[buffer(1)]],
    uint          tid     [[thread_position_in_threadgroup]],
    uint          cid     [[threadgroup_position_in_grid]])
{
    threadgroup long shm_k[CHUNK];
    threadgroup long shm_v[CHUNK];

    const uint base = cid * CHUNK;

    // Each thread loads two elements.
    shm_k[tid]            = keys[base + tid];
    shm_k[tid + TGSIZE]   = keys[base + tid + TGSIZE];
    shm_v[tid]            = values[base + tid];
    shm_v[tid + TGSIZE]   = values[base + tid + TGSIZE];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Bitonic sort, all stages with k ≤ CHUNK.
    for (uint k = 2u; k <= CHUNK; k <<= 1) {
        for (uint j = k >> 1; j > 0u; j >>= 1) {
            const uint i        = (tid / j) * (2u * j) + (tid % j);
            const uint partner  = i + j;
            const uint global_i = base + i;
            const bool ascending = ((global_i & k) == 0u);

            const long ka = shm_k[i];
            const long kb = shm_k[partner];
            const bool out_of_order = ascending ? (ka > kb) : (ka < kb);
            if (out_of_order) {
                shm_k[i] = kb; shm_k[partner] = ka;
                const long va = shm_v[i];
                const long vb = shm_v[partner];
                shm_v[i] = vb; shm_v[partner] = va;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Write back
    keys[base + tid]            = shm_k[tid];
    keys[base + tid + TGSIZE]   = shm_k[tid + TGSIZE];
    values[base + tid]          = shm_v[tid];
    values[base + tid + TGSIZE] = shm_v[tid + TGSIZE];
}

// For an outer k > CHUNK, after the cross-block stages have brought j
// down to CHUNK/2, run the remaining j = CHUNK/2, CHUNK/4, ..., 1 stages
// inside threadgroup memory. Direction uses the global index because k
// is larger than CHUNK and the bit at position log₂(k) lives in `base`.
kernel void bitonic_local_merge_i64(
    device long*  keys     [[buffer(0)]],
    device long*  values   [[buffer(1)]],
    constant uint& k       [[buffer(2)]],   // current outer-k stage
    uint          tid      [[thread_position_in_threadgroup]],
    uint          cid      [[threadgroup_position_in_grid]])
{
    threadgroup long shm_k[CHUNK];
    threadgroup long shm_v[CHUNK];

    const uint base = cid * CHUNK;

    shm_k[tid]            = keys[base + tid];
    shm_k[tid + TGSIZE]   = keys[base + tid + TGSIZE];
    shm_v[tid]            = values[base + tid];
    shm_v[tid + TGSIZE]   = values[base + tid + TGSIZE];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Direction is constant across this chunk (since k > CHUNK).
    const bool ascending = ((base & k) == 0u);

    for (uint j = CHUNK >> 1; j > 0u; j >>= 1) {
        const uint i       = (tid / j) * (2u * j) + (tid % j);
        const uint partner = i + j;

        const long ka = shm_k[i];
        const long kb = shm_k[partner];
        const bool out_of_order = ascending ? (ka > kb) : (ka < kb);
        if (out_of_order) {
            shm_k[i] = kb; shm_k[partner] = ka;
            const long va = shm_v[i];
            const long vb = shm_v[partner];
            shm_v[i] = vb; shm_v[partner] = va;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    keys[base + tid]            = shm_k[tid];
    keys[base + tid + TGSIZE]   = shm_k[tid + TGSIZE];
    values[base + tid]          = shm_v[tid];
    values[base + tid + TGSIZE] = shm_v[tid + TGSIZE];
}
