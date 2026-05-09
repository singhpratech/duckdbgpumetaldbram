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

// ============================================================================
// LSD radix sort, 8-bit buckets, 8 passes for int64 keys.
// Used by metal_groupby.mm as the primary path; bitonic_* above kept as
// reference / fallback for tiny inputs.
//
// Per pass:
//   1. radix_histogram: 256-bucket per-block histogram into hist[bid*256+b].
//   2. Host computes scatter offsets (bucket-major exclusive prefix sum).
//   3. radix_scatter: stable scatter using threadgroup-memory + per-element
//      preceding-bucket count for in-block local position (O(B²) but B is
//      small constant; the lockstep simdgroup divergence gates the cost).
//
// Sign-bit flip wraps the 8 passes so signed-radix == unsigned-radix.

constant uint RADIX_BLOCK_SIZE     = 256;
constant uint RADIX_WORK_PER_BLOCK = 256;   // 1 elem/thread minimizes O(B²) scatter inner-loop
constant uint RADIX_BUCKETS        = 256;

kernel void radix_flip_sign_bit(
    device long*   keys [[buffer(0)]],
    constant uint& n    [[buffer(1)]],
    uint           gid  [[thread_position_in_grid]])
{
    if (gid >= n) return;
    keys[gid] = (long)((ulong)keys[gid] ^ 0x8000000000000000UL);
}

kernel void radix_histogram(
    device const long*  keys     [[buffer(0)]],
    constant uint&      n        [[buffer(1)]],
    constant uint&      shift    [[buffer(2)]],
    device atomic_uint* hist     [[buffer(3)]],
    uint                tid      [[thread_position_in_threadgroup]],
    uint                bid      [[threadgroup_position_in_grid]])
{
    threadgroup atomic_uint local_hist[RADIX_BUCKETS];

    if (tid < RADIX_BUCKETS) {
        atomic_store_explicit(&local_hist[tid], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint base = bid * RADIX_WORK_PER_BLOCK;
    constexpr uint per_thread = RADIX_WORK_PER_BLOCK / RADIX_BLOCK_SIZE;
    for (uint i = 0; i < per_thread; ++i) {
        const uint idx = base + i * RADIX_BLOCK_SIZE + tid;
        if (idx >= n) break;
        const ulong k      = (ulong)keys[idx];
        const uint  bucket = (uint)((k >> shift) & 0xFFu);
        atomic_fetch_add_explicit(&local_hist[bucket], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < RADIX_BUCKETS) {
        const uint count = atomic_load_explicit(&local_hist[tid], memory_order_relaxed);
        atomic_store_explicit(&hist[bid * RADIX_BUCKETS + tid], count, memory_order_relaxed);
    }
}

kernel void radix_scatter(
    device const long*  in_keys    [[buffer(0)]],
    device const long*  in_values  [[buffer(1)]],
    constant uint&      n          [[buffer(2)]],
    constant uint&      shift      [[buffer(3)]],
    device const uint*  scan       [[buffer(4)]],
    device long*        out_keys   [[buffer(5)]],
    device long*        out_values [[buffer(6)]],
    uint                tid        [[thread_position_in_threadgroup]],
    uint                bid        [[threadgroup_position_in_grid]])
{
    threadgroup long  shm_keys   [RADIX_WORK_PER_BLOCK];
    threadgroup long  shm_values [RADIX_WORK_PER_BLOCK];
    threadgroup uchar shm_buckets[RADIX_WORK_PER_BLOCK];
    threadgroup bool  shm_valid  [RADIX_WORK_PER_BLOCK];

    const uint base = bid * RADIX_WORK_PER_BLOCK;
    constexpr uint per_thread = RADIX_WORK_PER_BLOCK / RADIX_BLOCK_SIZE;

    for (uint i = 0; i < per_thread; ++i) {
        const uint pos = i * RADIX_BLOCK_SIZE + tid;
        const uint idx = base + pos;
        if (idx < n) {
            const long key = in_keys[idx];
            shm_keys   [pos] = key;
            shm_values [pos] = in_values[idx];
            shm_buckets[pos] = (uchar)(((ulong)key >> shift) & 0xFFu);
            shm_valid  [pos] = true;
        } else {
            shm_valid  [pos] = false;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = 0; i < per_thread; ++i) {
        const uint pos = i * RADIX_BLOCK_SIZE + tid;
        if (!shm_valid[pos]) continue;

        const uchar my_bucket = shm_buckets[pos];
        uint local_pos = 0;
        for (uint p = 0; p < pos; ++p) {
            if (shm_valid[p] && shm_buckets[p] == my_bucket) ++local_pos;
        }
        const uint global_pos = scan[bid * RADIX_BUCKETS + my_bucket] + local_pos;
        out_keys  [global_pos] = shm_keys  [pos];
        out_values[global_pos] = shm_values[pos];
    }
}

// ============================================================================
// On-device exclusive scan for the radix sort scatter offsets.
//
// Goal: replace the host-side cache-friendly scan (which dominates wall time
// at large N) with three GPU kernels that compute the bucket-major exclusive
// prefix sum entirely on the GPU. With this, all 8 radix passes can stay in
// a single command buffer and the wall converges to the kernel time.
//
// Layout: hist[bid * 256 + bucket]. Output: scan[bid * 256 + bucket] =
//   sum over (b' < bucket, all bid') of hist[bid'][b']
//   + sum over (bid' < bid) of hist[bid'][bucket]
//
// Three kernels:
//   1. radix_bucket_totals: 256 threadgroups (one per bucket). Each reduces
//      num_blocks values for its bucket to a single total.
//   2. radix_bucket_offsets: one 256-thread threadgroup, exclusive scan over
//      the 256 bucket totals → bucket_offset[256].
//   3. radix_per_bucket_scan: 256 threadgroups (one per bucket). Each does an
//      exclusive scan of hist[*][bucket] across all blocks, then adds
//      bucket_offset[bucket]. Writes the final scan[bid * 256 + bucket].
//
// The per-bucket scan handles arbitrary num_blocks via chunked Hillis-Steele
// in threadgroup memory: process 256 blocks at a time, accumulate a running
// total across chunks. O(num_blocks / 256 * log(256)) cycles per bucket.

kernel void radix_bucket_totals(
    device const uint* hist        [[buffer(0)]],
    constant uint&     num_blocks  [[buffer(1)]],
    device uint*       bucket_total [[buffer(2)]],
    uint               tid         [[thread_position_in_threadgroup]],
    uint               b           [[threadgroup_position_in_grid]])
{
    threadgroup uint shm[256];

    uint running = 0;
    for (uint chunk_start = 0; chunk_start < num_blocks; chunk_start += 256) {
        const uint bid = chunk_start + tid;
        running += (bid < num_blocks) ? hist[bid * 256u + b] : 0u;
    }
    shm[tid] = running;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) shm[tid] += shm[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) bucket_total[b] = shm[0];
}

kernel void radix_bucket_offsets(
    device const uint* bucket_total [[buffer(0)]],
    device uint*       bucket_offset [[buffer(1)]],
    uint               tid          [[thread_position_in_threadgroup]])
{
    threadgroup uint shm[256];
    shm[tid] = bucket_total[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Hillis-Steele inclusive scan
    for (uint offset = 1; offset < 256; offset <<= 1) {
        uint t = (tid >= offset) ? shm[tid - offset] : 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        shm[tid] += t;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Convert inclusive → exclusive
    bucket_offset[tid] = shm[tid] - bucket_total[tid];
}

kernel void radix_per_bucket_scan(
    device const uint* hist          [[buffer(0)]],
    constant uint&     num_blocks    [[buffer(1)]],
    device const uint* bucket_offset [[buffer(2)]],
    device uint*       scan          [[buffer(3)]],
    uint               tid           [[thread_position_in_threadgroup]],
    uint               b             [[threadgroup_position_in_grid]])
{
    threadgroup uint shm[256];

    uint running = bucket_offset[b];
    for (uint chunk_start = 0; chunk_start < num_blocks; chunk_start += 256) {
        const uint bid = chunk_start + tid;
        const uint val = (bid < num_blocks) ? hist[bid * 256u + b] : 0u;
        shm[tid] = val;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Inclusive Hillis-Steele scan within chunk
        for (uint offset = 1; offset < 256; offset <<= 1) {
            uint t = (tid >= offset) ? shm[tid - offset] : 0u;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            shm[tid] += t;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Exclusive within chunk = inclusive - own value. Plus running across chunks.
        if (bid < num_blocks) {
            scan[bid * 256u + b] = running + (shm[tid] - val);
        }

        // Pass running across chunks (chunk total = inclusive[255]).
        running += shm[255];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
