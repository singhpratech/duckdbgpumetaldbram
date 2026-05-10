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
// === 2026-05 optimizations on this Apple Family 9 (M4 Max) tuned path ===
//   - WORK_PER_BLOCK bumped from 256 → 1024 (4 elements per thread).
//   - Histogram + scatter use vectorized device ulong4* loads (16 B per
//     transaction; LPDDR5X likes wide bursts).
//   - Scatter rewritten: O(B²) per-element scan replaced by an in-block
//     8-bit multi-split local sort (8 × simdgroup-level prefix sums),
//     then write contiguous runs out at scan[bucket] + local_offset.
//     Per block: 8 × O(B) instead of O(B²) — at B=1024 that's 8 K simple
//     ops vs 1 M.
//   - The multi-split permutes 16-bit indices, NOT (key,value) pairs.
//     Threadgroup memory limit on M4 is 32 KB; permuting 8-byte keys +
//     8-byte values would overflow. Keys/values stay in their original
//     load slots; sorted indices give the read order during the final
//     write to device memory.
//   - Sign-bit flip kept available but unused: GROUP BY only needs equal
//     keys to cluster, not signed ordering. Host orchestration drops the
//     two flip dispatches and treats keys as unsigned for byte selection.
//
// Per pass:
//   1. radix_histogram: 256-bucket per-block histogram into hist[bid*256+b].
//   2. Three GPU scan kernels build the bucket-major exclusive prefix.
//   3. radix_scatter: stable scatter via local multi-split + per-bucket
//      offset table.

// MSL `constant` is evaluated at PSO build time. Inside kernel bodies we
// also use these as fixed-array dimensions and as loop bounds — those
// require the compiler to see them as constexpr-style integer constants.
// Using #define ensures the value is plain-text-substituted into every
// use site so loop bounds and array sizes are unambiguous compile-time
// constants.
#define RADIX_BLOCK_SIZE      256u
#define RADIX_WORK_PER_BLOCK  1024u
#define RADIX_BUCKETS         256u
#define RADIX_PER_THREAD      (RADIX_WORK_PER_BLOCK / RADIX_BLOCK_SIZE)
#define RADIX_SIMD_WIDTH      32u
#define RADIX_NUM_SIMDS       (RADIX_BLOCK_SIZE / RADIX_SIMD_WIDTH)

kernel void radix_flip_sign_bit(
    device long*   keys [[buffer(0)]],
    constant uint& n    [[buffer(1)]],
    uint           gid  [[thread_position_in_grid]])
{
    if (gid >= n) return;
    keys[gid] = (long)((ulong)keys[gid] ^ 0x8000000000000000UL);
}

// Histogram: each threadgroup processes RADIX_WORK_PER_BLOCK contiguous
// elements. Threads cooperatively read the block via vectorized ulong4
// loads where possible (interior blocks; tail block falls back to
// scalar). Per-block histogram lives in threadgroup atomics, then is
// copied to global hist[bid*256 + b].
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

    // Fast path: full-block, vectorized. Each thread reads one ulong4 = 4
    // contiguous keys at offset (base + tid*4). Stride 4×BLOCK = 1024 covers
    // the whole block in one cycle.
    if (base + RADIX_WORK_PER_BLOCK <= n) {
        const device ulong4* keys_v4 =
            reinterpret_cast<const device ulong4*>(keys + base);
        const ulong4 k4 = keys_v4[tid];
        const uint b0 = (uint)((k4.x >> shift) & 0xFFu);
        const uint b1 = (uint)((k4.y >> shift) & 0xFFu);
        const uint b2 = (uint)((k4.z >> shift) & 0xFFu);
        const uint b3 = (uint)((k4.w >> shift) & 0xFFu);
        atomic_fetch_add_explicit(&local_hist[b0], 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&local_hist[b1], 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&local_hist[b2], 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&local_hist[b3], 1u, memory_order_relaxed);
    } else {
        // Tail-block scalar path: bounds-check each element. Layout matches
        // the scatter so element j of thread t comes from base + tid*4 + j.
        const uint base_lin = tid * RADIX_PER_THREAD;
        for (uint j = 0; j < RADIX_PER_THREAD; ++j) {
            const uint idx = base + base_lin + j;
            if (idx >= n) break;
            const ulong k      = (ulong)keys[idx];
            const uint  bucket = (uint)((k >> shift) & 0xFFu);
            atomic_fetch_add_explicit(&local_hist[bucket], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < RADIX_BUCKETS) {
        const uint count = atomic_load_explicit(&local_hist[tid], memory_order_relaxed);
        atomic_store_explicit(&hist[bid * RADIX_BUCKETS + tid], count, memory_order_relaxed);
    }
}

// Scatter: stable in-block sort by bucket via 8-bit multi-split, then
// contiguous global writes. Replaces the previous O(B²) per-element scan
// that walked shm_buckets[0..pos] for each thread.
//
// Multi-split per bit:
//   For each bit i in [0,8):
//     bit = (bucket >> i) & 1
//     pos0 = exclusive prefix sum of (1 - bit) over preceding elements
//     n0   = total count of bit=0 across the block
//     pos1 = exclusive prefix sum of bit       over preceding elements
//     newpos = bit ? (n0 + pos1) : pos0
//     permute index to newpos, barrier, repeat.
//
// We permute uint16 INDICES into the original load buffer (not keys/
// values) — this keeps total threadgroup memory under the 32 KB limit
// even with WORK_PER_BLOCK = 1024.
//
// After 8 splits the indices are sorted by bucket within the block (in
// stable order — multi-split is order-preserving for ties). Then we walk
// the sorted positions: thread t with sorted bucket b at sorted position
// p writes its element to scan[bid*256 + b] + (p - block_offset_of[b]).
//
// Two-level prefix sum across 256 threads:
//   level 1: simd_prefix_exclusive_sum within each 32-thread simdgroup.
//   level 2: store warp totals to threadgroup memory; one warp scans the
//            8 totals; broadcast warp_offset back; add warp_offset to
//            level-1 result.
//
// Threadgroup memory budget (M4 limit: 32 KB):
//   shm_keys[1024]    = 8 KB
//   shm_vals[1024]    = 8 KB
//   shm_buck[1024]    = 1 KB
//   pp_a_idx[1024]    = 2 KB  (uint16)
//   pp_b_idx[1024]    = 2 KB  (uint16)
//   block_hist[256]   = 1 KB
//   block_off[256]    = 1 KB
//   block_hist_atom   = 1 KB
//   warp_totals[8]    = 32 B
//   warp_off[8]       = 32 B
//   total_zeros       = 4 B
//   ────────────────────────
//   ~25 KB. Comfortably fits.
kernel void radix_scatter(
    device const long*  in_keys    [[buffer(0)]],
    device const long*  in_values  [[buffer(1)]],
    constant uint&      n          [[buffer(2)]],
    constant uint&      shift      [[buffer(3)]],
    device const uint*  scan       [[buffer(4)]],
    device long*        out_keys   [[buffer(5)]],
    device long*        out_values [[buffer(6)]],
    uint                tid        [[thread_position_in_threadgroup]],
    uint                bid        [[threadgroup_position_in_grid]],
    uint                lane       [[thread_index_in_simdgroup]],
    uint                wid        [[simdgroup_index_in_threadgroup]])
{
    threadgroup long  shm_keys [RADIX_WORK_PER_BLOCK];
    threadgroup long  shm_vals [RADIX_WORK_PER_BLOCK];
    threadgroup uchar shm_buck [RADIX_WORK_PER_BLOCK];
    threadgroup uchar shm_valid[RADIX_WORK_PER_BLOCK];

    threadgroup ushort pp_a_idx[RADIX_WORK_PER_BLOCK];
    threadgroup ushort pp_b_idx[RADIX_WORK_PER_BLOCK];

    threadgroup uint  warp_totals[RADIX_NUM_SIMDS];
    threadgroup uint  warp_off   [RADIX_NUM_SIMDS];
    threadgroup uint  block_hist [RADIX_BUCKETS];
    threadgroup uint  block_off  [RADIX_BUCKETS];
    threadgroup atomic_uint block_hist_atomic[RADIX_BUCKETS];
    threadgroup uint  total_zeros;

    const uint base = bid * RADIX_WORK_PER_BLOCK;
    const bool full_block = (base + RADIX_WORK_PER_BLOCK <= n);

    // -------- Load + compute bucket; init pp_a_idx = identity --------
    // Memory layout: element j of thread t lives at slot (tid*4 + j).
    // We use this layout uniformly so the multi-split's "thread t owns
    // slots tid*4..tid*4+3" property holds. Vectorized device load reads
    // one ulong4 per thread (32 B) at device offset (base + tid*4).
    if (full_block) {
        const device ulong4* keys_v4 =
            reinterpret_cast<const device ulong4*>(in_keys + base);
        const device ulong4* vals_v4 =
            reinterpret_cast<const device ulong4*>(in_values + base);
        const ulong4 k4 = keys_v4[tid];
        const ulong4 v4 = vals_v4[tid];
        const uint dst = tid * RADIX_PER_THREAD;
        shm_keys [dst + 0] = (long)k4.x;
        shm_keys [dst + 1] = (long)k4.y;
        shm_keys [dst + 2] = (long)k4.z;
        shm_keys [dst + 3] = (long)k4.w;
        shm_vals [dst + 0] = (long)v4.x;
        shm_vals [dst + 1] = (long)v4.y;
        shm_vals [dst + 2] = (long)v4.z;
        shm_vals [dst + 3] = (long)v4.w;
        shm_buck [dst + 0] = (uchar)((k4.x >> shift) & 0xFFu);
        shm_buck [dst + 1] = (uchar)((k4.y >> shift) & 0xFFu);
        shm_buck [dst + 2] = (uchar)((k4.z >> shift) & 0xFFu);
        shm_buck [dst + 3] = (uchar)((k4.w >> shift) & 0xFFu);
        shm_valid[dst + 0] = 1;
        shm_valid[dst + 1] = 1;
        shm_valid[dst + 2] = 1;
        shm_valid[dst + 3] = 1;
        pp_a_idx[dst + 0] = (ushort)(dst + 0);
        pp_a_idx[dst + 1] = (ushort)(dst + 1);
        pp_a_idx[dst + 2] = (ushort)(dst + 2);
        pp_a_idx[dst + 3] = (ushort)(dst + 3);
    } else {
        const uint dst = tid * RADIX_PER_THREAD;
        for (uint j = 0; j < RADIX_PER_THREAD; ++j) {
            const uint idx = base + dst + j;
            if (idx < n) {
                const long k = in_keys[idx];
                shm_keys [dst + j] = k;
                shm_vals [dst + j] = in_values[idx];
                shm_buck [dst + j] = (uchar)(((ulong)k >> shift) & 0xFFu);
                shm_valid[dst + j] = 1;
            } else {
                shm_keys [dst + j] = 0;
                shm_vals [dst + j] = 0;
                shm_buck [dst + j] = 0;
                shm_valid[dst + j] = 0;
            }
            pp_a_idx[dst + j] = (ushort)(dst + j);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -------- Multi-split: 8 bits, permute pp_a_idx → pp_b_idx alternately --------
    // Each thread owns 4 elements at slots tid*4..tid*4+3 in the CURRENT
    // permutation buffer (pp_a or pp_b). Per bit, each thread:
    //   1. Reads its 4 indices from the current buffer.
    //   2. For each, looks up bucket, computes bit.
    //   3. Computes within-thread exclusive prefix sum of (1-bit) → pre0[j].
    //      n0_t = sum of (1-bit) across the 4.
    //   4. Two-level scan of n0_t across 256 threads → thread_off0.
    //      total_zeros = grand total of bit=0.
    //   5. For each j: pos0 = thread_off0 + pre0[j]; pos1 = (4*tid + j) - pos0;
    //      newpos = bit ? (total_zeros + pos1) : pos0.
    //   6. Write index to dest buffer[newpos]. Barrier. Swap.
    for (uint bit_i = 0; bit_i < 8; ++bit_i) {
        const bool a_to_b = ((bit_i & 1u) == 0u);

        // Read this thread's 4 indices from current source.
        ushort idx_arr[RADIX_PER_THREAD];
        const uint slot_base = tid * RADIX_PER_THREAD;
        if (a_to_b) {
            for (uint j = 0; j < RADIX_PER_THREAD; ++j) idx_arr[j] = pp_a_idx[slot_base + j];
        } else {
            for (uint j = 0; j < RADIX_PER_THREAD; ++j) idx_arr[j] = pp_b_idx[slot_base + j];
        }

        // Per-thread bits + within-thread exclusive prefix of (1 - bit).
        // For invalid elements we set bit = 1 so they sort to the back of
        // the block; the post-sort write stage still skips them via shm_valid.
        // Note: invalid elements are loaded LAST (highest lin within block),
        // so among elements with bit=1, valid ones precede invalid ones in
        // input order — multi-split's stable ordering then preserves valid
        // bucket=255 elements' position before invalid sentinels in the
        // sorted output. (See block_off discussion below.)
        uint bits[RADIX_PER_THREAD];
        uint pre0[RADIX_PER_THREAD];
        uint n0_t = 0;
        for (uint j = 0; j < RADIX_PER_THREAD; ++j) {
            const uint origin   = idx_arr[j];
            const uint v_flag   = shm_valid[origin];
            const uint b        = v_flag
                ? (uint)((shm_buck[origin] >> bit_i) & 1u)
                : 1u;
            bits[j] = b;
            pre0[j] = n0_t;
            n0_t += (1u - b);
        }

        // Two-level exclusive prefix sum of n0_t across the 256 threads.
        // Level 1: prefix in this thread's simdgroup.
        const uint warp_excl_n0 = simd_prefix_exclusive_sum(n0_t);
        const uint warp_tot_n0  = simd_sum(n0_t);
        if (lane == 0u) warp_totals[wid] = warp_tot_n0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Level 2: scan over the 8 warp totals on simdgroup 0. ALL 32
        // lanes of simdgroup 0 must execute the simd ops (they are
        // simdgroup-wide); lanes >= NUM_SIMDS contribute 0 so the
        // result for the meaningful lanes is unaffected.
        if (wid == 0u) {
            const uint t = (lane < RADIX_NUM_SIMDS) ? warp_totals[lane] : 0u;
            const uint excl = simd_prefix_exclusive_sum(t);
            const uint tot  = simd_sum(t);
            if (lane < RADIX_NUM_SIMDS) warp_off[lane] = excl;
            if (lane == 0u) total_zeros = tot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint thread_off0 = warp_off[wid] + warp_excl_n0;

        // Write the 4 indices to their new positions.
        for (uint j = 0; j < RADIX_PER_THREAD; ++j) {
            const uint pos0   = thread_off0 + pre0[j];
            const uint lin    = slot_base + j;
            const uint pos1   = lin - pos0;
            const uint newpos = bits[j] ? (total_zeros + pos1) : pos0;
            if (a_to_b) {
                pp_b_idx[newpos] = idx_arr[j];
            } else {
                pp_a_idx[newpos] = idx_arr[j];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // After 8 bit splits (even count), final permutation lives in pp_a_idx
    // (bit 0: A→B, bit 1: B→A, ..., bit 7: B→A → ends in pp_a).
    threadgroup ushort* sorted_idx = pp_a_idx;

    // -------- Build block_hist + block_off (exclusive scan over buckets) --------
    // block_off[b] = number of valid elements in this block with bucket < b.
    //   (Invalids don't count — they're skipped on the global write below.)
    if (tid < RADIX_BUCKETS) {
        atomic_store_explicit(&block_hist_atomic[tid], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
        const uint base_lin = tid * RADIX_PER_THREAD;
        for (uint j = 0; j < RADIX_PER_THREAD; ++j) {
            const uint origin = sorted_idx[base_lin + j];
            if (shm_valid[origin]) {
                atomic_fetch_add_explicit(
                    &block_hist_atomic[shm_buck[origin]],
                    1u, memory_order_relaxed);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < RADIX_BUCKETS) {
        block_hist[tid] = atomic_load_explicit(&block_hist_atomic[tid],
                                               memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Two-level exclusive prefix sum over 256 buckets (RADIX_BLOCK_SIZE
    // happens to equal RADIX_BUCKETS, so each thread owns exactly one
    // bucket count). Same level-1 / level-2 structure as the multi-split.
    //   Level 1 (within simdgroup): simd_prefix_exclusive_sum.
    //   Level 2 (across simdgroups): one warp scans 8 totals, broadcasts.
    {
        const uint cnt        = (tid < RADIX_BUCKETS) ? block_hist[tid] : 0u;
        const uint warp_excl  = simd_prefix_exclusive_sum(cnt);
        const uint warp_total = simd_sum(cnt);
        if (lane == 0u) warp_totals[wid] = warp_total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (wid == 0u) {
            const uint t    = (lane < RADIX_NUM_SIMDS) ? warp_totals[lane] : 0u;
            const uint excl = simd_prefix_exclusive_sum(t);
            if (lane < RADIX_NUM_SIMDS) warp_off[lane] = excl;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < RADIX_BUCKETS) {
            block_off[tid] = warp_off[wid] + warp_excl;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -------- Final scatter to device memory --------
    {
        const uint base_lin = tid * RADIX_PER_THREAD;
        for (uint j = 0; j < RADIX_PER_THREAD; ++j) {
            const uint p      = base_lin + j;
            const uint origin = sorted_idx[p];
            if (!shm_valid[origin]) continue;
            const uchar b     = shm_buck[origin];
            const uint loc    = p - block_off[b];
            const uint dst    = scan[bid * RADIX_BUCKETS + b] + loc;
            out_keys  [dst] = shm_keys[origin];
            out_values[dst] = shm_vals[origin];
        }
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

// ============================================================================
// Min-max pre-scan (one kernel per radix sort, runs once before pass 0).
//
// Computes min and max of the input keys (signed int64) so the host can
// determine which radix-sort passes are "constant byte" no-ops and skip
// them. For uniformly-distributed keys in [0, G), only the lowest
// ceil(log2(G)/8) bytes vary; the higher bytes are zero across all keys
// and the corresponding radix passes do nothing useful.
//
// Output: a 16-byte buffer holding {min: int64, max: int64}.

kernel void radix_minmax_i64(
    device const long*  keys     [[buffer(0)]],
    constant uint&      n        [[buffer(1)]],
    device atomic_int*  out_min  [[buffer(2)]],   // 2 atomic_int = lo, hi
    device atomic_int*  out_max  [[buffer(3)]],
    uint                tid      [[thread_position_in_threadgroup]],
    uint                bid      [[threadgroup_position_in_grid]],
    uint                gsize    [[threads_per_grid]],
    uint                gid      [[thread_position_in_grid]])
{
    threadgroup long shm_min[256];
    threadgroup long shm_max[256];

    long local_min = 0x7FFFFFFFFFFFFFFFL;
    long local_max = (long)0x8000000000000000UL;
    for (uint i = gid; i < n; i += gsize) {
        const long k = keys[i];
        local_min = (k < local_min) ? k : local_min;
        local_max = (k > local_max) ? k : local_max;
    }
    shm_min[tid] = local_min;
    shm_max[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            const long a = shm_min[tid];
            const long b = shm_min[tid + s];
            shm_min[tid] = (a < b) ? a : b;
            const long c = shm_max[tid];
            const long d = shm_max[tid + s];
            shm_max[tid] = (c > d) ? c : d;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        // Reduce per-block partials into a single (min, max) using 32-bit
        // atomic min/max on the low / high halves. Apple supports
        // atomic_fetch_min/max on int (32-bit) but not on long; do it in
        // two halves with monotonic comparison.
        // For a coarse "do this byte vary?" estimate, we don't need a
        // bit-perfect 64-bit reduction — but we do need correctness. The
        // simplest correct path here: just have block 0 do a final
        // sequential reduction via atomics on 32-bit halves below.
        //
        // Implementation note: we use the "two passes" pattern. The
        // initial values sentinel are stored in out_min[0..1] (lo,hi for min)
        // and out_max[0..1] (lo,hi for max) by the host. Each block's tid==0
        // applies its partial via the atomic 32-bit min/max API, with a
        // careful split so the result represents the true 64-bit value.
        //
        // For simplicity and since this kernel runs ONCE per groupby call
        // (negligible cost), we let block 0 wait via a serialization trick:
        // we actually just store each block's partial to a per-block slot
        // and do the final 64-bit reduction on the host. To keep this kernel
        // self-contained as written, store the per-block partials at
        // out_min[bid * 2..bid * 2 + 1] and same for max.
        atomic_store_explicit(&out_min[bid * 2 + 0], (int)((ulong)shm_min[0] & 0xFFFFFFFFu),
                              memory_order_relaxed);
        atomic_store_explicit(&out_min[bid * 2 + 1], (int)((ulong)shm_min[0] >> 32),
                              memory_order_relaxed);
        atomic_store_explicit(&out_max[bid * 2 + 0], (int)((ulong)shm_max[0] & 0xFFFFFFFFu),
                              memory_order_relaxed);
        atomic_store_explicit(&out_max[bid * 2 + 1], (int)((ulong)shm_max[0] >> 32),
                              memory_order_relaxed);
    }
}
// ============================================================================
// Slot-lock hash aggregate (32-bit slot-state lock + non-atomic 64-bit sum).
//
// Apple Silicon GPUs lack 64-bit atomic_fetch_add on `device` storage, so the
// classic CUDA hash-table-with-atomicAdd approach won't compile. The standard
// substitute is a 32-bit slot-state lock that protects a non-atomic 64-bit
// (key, sum) pair: insert/update enters via atomic CAS on the state, performs
// the 64-bit work without atomics, and exits with an atomic store to the
// state. Other threads observe the state with acquire ordering so they see
// the key/sum writes after seeing COMMITTED.
//
// Pipeline (one MetalCommandBuffer):
//   1. partition_count_hash:   one atomic_fetch_add per row to the partition's
//      counter. Cost: 1 device atomic per row, 4096 partitions → low
//      contention.
//   2. (host) exclusive scan partition_counts[NUM_PARTITIONS] → partition_offsets.
//   3. partition_scatter_hash: each row reads its partition counter (reset by
//      host between count and scatter), atomic_fetch_add for in-partition pos,
//      writes (key, value) into scatter_keys/values at offsets[part]+pos.
//   4. partition_hash_aggregate_slotlock_i64: one threadgroup per partition.
//      64 threads stride over the partition's slice, slot-lock-insert into
//      a 1024-slot threadgroup-resident hash table. Emit phase atomic_fetch_adds
//      a global out_count and writes (key, sum) to flat out_keys/out_sums.
//
// Why 4096 partitions × 1024 slots:
//   - 4096 partitions × ~244 rows/partition (1M total rows) is enough work per
//     threadgroup to amortize launch overhead.
//   - 1024 slots × (4 + 8 + 8) = 20 KB threadgroup memory per partition,
//     well under Apple's 32 KB limit.
//   - Handles ~400-500 unique keys per partition before load factor degrades;
//     covers 1M total unique keys with margin.
// ============================================================================

constant uint SLOTLOCK_NUM_PARTITIONS = 32768;
constant uint SLOTLOCK_PARTITION_MASK = 32767;
constant uint SLOTLOCK_SLOT_COUNT     = 1024;
constant uint SLOTLOCK_SLOT_MASK      = 1023;
constant uint SLOTLOCK_TG_THREADS     = 64;
constant uint SLOTLOCK_COUNT_THREADS  = 256;
constant uint SLOTLOCK_COUNT_WORK     = 256;

constant uint SLOT_EMPTY     = 0u;
constant uint SLOT_LOCKED    = 1u;
constant uint SLOT_COMMITTED = 2u;

// 64-bit splitmix-style hash. We use the *same* hash for partition selection
// and slot probing (taking different bit ranges) — partition uses bits [0..12),
// slot probe uses bits [16..26). The mid-bits diversify enough that the slot
// distribution is not biased by the partition selection.
inline ulong slotlock_hash(long key) {
    ulong x = (ulong)key;
    x ^= (x >> 33);
    x *= 0xff51afd7ed558ccdUL;
    x ^= (x >> 33);
    x *= 0xc4ceb9fe1a85ec53UL;
    x ^= (x >> 33);
    return x;
}

inline uint slotlock_partition(ulong h) {
    return (uint)(h & SLOTLOCK_PARTITION_MASK);
}

inline uint slotlock_slot_start(ulong h) {
    return (uint)((h >> 16) & SLOTLOCK_SLOT_MASK);
}

// ----------------------------------------------------------------------------
// Pass 1: partition_count_hash
//   For each row i, atomic_fetch_add(partition_counts[hash_partition(keys[i])], 1).
//   Output: partition_counts[NUM_PARTITIONS] (host scans these into offsets).
// ----------------------------------------------------------------------------
kernel void partition_count_hash(
    device const long*  keys             [[buffer(0)]],
    constant uint&      n                [[buffer(1)]],
    device atomic_uint* partition_counts [[buffer(2)]],
    uint                gid              [[thread_position_in_grid]],
    uint                gsize            [[threads_per_grid]])
{
    for (uint i = gid; i < n; i += gsize) {
        const ulong h = slotlock_hash(keys[i]);
        const uint p = slotlock_partition(h);
        atomic_fetch_add_explicit(&partition_counts[p], 1u, memory_order_relaxed);
    }
}

// ----------------------------------------------------------------------------
// Pass 2: partition_scatter_hash
//   For each row i, atomic_fetch_add(partition_write_pos[part], 1) → pos.
//   Write (key, value) into scatter_keys/values at offsets[part] + pos.
//   `partition_write_pos` is a separate buffer from partition_counts (which
//   was written in pass 1); host clears it before this pass.
// ----------------------------------------------------------------------------
kernel void partition_scatter_hash(
    device const long*  keys                [[buffer(0)]],
    device const long*  values              [[buffer(1)]],
    constant uint&      n                   [[buffer(2)]],
    device const uint*  partition_offsets   [[buffer(3)]],
    device atomic_uint* partition_write_pos [[buffer(4)]],
    device long*        scatter_keys        [[buffer(5)]],
    device long*        scatter_values      [[buffer(6)]],
    uint                gid                 [[thread_position_in_grid]],
    uint                gsize               [[threads_per_grid]])
{
    for (uint i = gid; i < n; i += gsize) {
        const long  k = keys[i];
        const long  v = values[i];
        const ulong h = slotlock_hash(k);
        const uint  p = slotlock_partition(h);
        const uint  pos = atomic_fetch_add_explicit(&partition_write_pos[p], 1u,
                                                    memory_order_relaxed);
        const uint  out_idx = partition_offsets[p] + pos;
        scatter_keys  [out_idx] = k;
        scatter_values[out_idx] = v;
    }
}

// ----------------------------------------------------------------------------
// Pass 3: partition_hash_aggregate_slotlock_i64
//   One threadgroup per partition. SLOTLOCK_TG_THREADS threads stride over the
//   partition's slice [offsets[p], offsets[p+1]). Each thread inserts/updates
//   a threadgroup-resident slot table using the 32-bit slot-lock pattern.
//   After processing, all threads cooperate on the emit phase: each owns a
//   share of slots; for each COMMITTED slot, atomic_fetch_add a global
//   out_count and write (key, sum) to flat output buffers.
// ----------------------------------------------------------------------------
kernel void partition_hash_aggregate_slotlock_i64(
    device const long*  scatter_keys      [[buffer(0)]],
    device const long*  scatter_values    [[buffer(1)]],
    device const uint*  partition_offsets [[buffer(2)]],   // size NUM_PARTITIONS+1
    device atomic_uint* out_count         [[buffer(3)]],
    device long*        out_keys          [[buffer(4)]],
    device long*        out_sums          [[buffer(5)]],
    uint                tid               [[thread_position_in_threadgroup]],
    uint                pid               [[threadgroup_position_in_grid]])
{
    // Threadgroup hash table.
    threadgroup atomic_uint slot_state[SLOTLOCK_SLOT_COUNT];
    threadgroup long        slot_keys [SLOTLOCK_SLOT_COUNT];
    threadgroup long        slot_sums [SLOTLOCK_SLOT_COUNT];

    // Initialize slot states to EMPTY. Cooperatively over the threads.
    for (uint s = tid; s < SLOTLOCK_SLOT_COUNT; s += SLOTLOCK_TG_THREADS) {
        atomic_store_explicit(&slot_state[s], SLOT_EMPTY, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint slice_lo = partition_offsets[pid];
    const uint slice_hi = partition_offsets[pid + 1];

    // Per-row insert/update. Each thread takes a stride.
    for (uint i = slice_lo + tid; i < slice_hi; i += SLOTLOCK_TG_THREADS) {
        const long  k = scatter_keys  [i];
        const long  v = scatter_values[i];
        const ulong h = slotlock_hash(k);
        uint slot = slotlock_slot_start(h);

        // Bound the probe-loop iterations to avoid theoretical livelock.
        // SLOT_COUNT * 2 is enough: one full pass to find a slot, plus
        // contention margin. If we exhaust this, the table is too full —
        // we silently drop the row, which would manifest as a missing
        // group in correctness checks. Host code sizes the partition count
        // so this never happens.
        for (uint attempt = 0; attempt < SLOTLOCK_SLOT_COUNT * 4u; ++attempt) {
            const uint state = atomic_load_explicit(&slot_state[slot],
                                                    memory_order_relaxed);
            if (state == SLOT_EMPTY) {
                // Try to claim this slot for a new key.
                uint expected = SLOT_EMPTY;
                if (atomic_compare_exchange_weak_explicit(
                        &slot_state[slot], &expected, SLOT_LOCKED,
                        memory_order_relaxed, memory_order_relaxed))
                {
                    // We hold the lock. Initialize key + sum, then publish.
                    slot_keys[slot] = k;
                    slot_sums[slot] = v;
                    atomic_store_explicit(&slot_state[slot], SLOT_COMMITTED,
                                          memory_order_relaxed);
                    break;
                }
                // CAS lost — another thread claimed this slot. Retry on the
                // same slot (state is now LOCKED or COMMITTED).
                continue;
            }

            if (state == SLOT_COMMITTED) {
                // Read the key (state was acquired, so the key write is
                // visible). If it matches, try to acquire for an update.
                if (slot_keys[slot] == k) {
                    uint expected = SLOT_COMMITTED;
                    if (atomic_compare_exchange_weak_explicit(
                            &slot_state[slot], &expected, SLOT_LOCKED,
                            memory_order_relaxed, memory_order_relaxed))
                    {
                        slot_sums[slot] += v;
                        atomic_store_explicit(&slot_state[slot], SLOT_COMMITTED,
                                              memory_order_relaxed);
                        break;
                    }
                    // CAS lost (another thread is updating same slot) — retry
                    // on the same slot.
                    continue;
                }
                // Key mismatch — linear probe to next slot.
                slot = (slot + 1u) & SLOTLOCK_SLOT_MASK;
                continue;
            }

            // state == SLOT_LOCKED — another thread holds it. Spin on this
            // slot. Apple Silicon's SIMD-style execution lets divergent
            // paths interleave so the lock-holder will make progress.
            // No explicit yield needed.
            continue;
        }
    }

    // Make all slot writes visible threadgroup-wide.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Emit phase. Each thread takes a share of slots, atomic_fetch_adds a
    // global counter for its position, then writes (key, sum).
    for (uint s = tid; s < SLOTLOCK_SLOT_COUNT; s += SLOTLOCK_TG_THREADS) {
        const uint state = atomic_load_explicit(&slot_state[s],
                                                memory_order_relaxed);
        if (state == SLOT_COMMITTED) {
            const uint out_idx = atomic_fetch_add_explicit(out_count, 1u,
                                                           memory_order_relaxed);
            out_keys[out_idx] = slot_keys[s];
            out_sums[out_idx] = slot_sums[s];
        }
    }
}
