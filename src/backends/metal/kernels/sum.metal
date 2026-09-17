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

// ===================== fused join-sum (v0.5) =====================
//
// Inner equi-join + SUM in one pass, against a build-side key column that
// has been radix-sorted once and cached on its resident column. Each probe
// element binary-searches its multiplicity m in the sorted build keys and
// contributes m * payload[i]; probe keys and payload stream sequentially.
// Multiply and accumulate are ulong (unsigned wrap — the cross-backend rule;
// the CPU reference does the same, so results are bit-identical).

// mode: 0=INNER (c=m), 1=LEFT (c=max(m,1)), 2=SEMI (c=m?1:0), 3=ANTI (c=m?0:1)
// — mirrors gpudb::JoinKind; see the multiplier table in gpu_backend.hpp.
kernel void join_sum_i64(
    device const long*  probe_keys   [[buffer(0)]],
    device const long*  payload      [[buffer(1)]],
    device const long*  build_sorted [[buffer(2)]],
    device long*        partials     [[buffer(3)]],   // 2 per block: sum, matched
    constant uint&      n_probe      [[buffer(4)]],
    constant uint&      n_build      [[buffer(5)]],
    constant uint&      mode         [[buffer(6)]],
    uint                tid          [[thread_position_in_threadgroup]],
    uint                gid          [[thread_position_in_grid]],
    uint                gsize        [[threads_per_grid]],
    uint                block_id     [[threadgroup_position_in_grid]])
{
    threadgroup long shm_sum[BLOCK];
    threadgroup long shm_cnt[BLOCK];

    ulong local_sum = 0;
    long  local_cnt = 0;
    for (uint i = gid; i < n_probe; i += gsize) {
        const long k = probe_keys[i];
        // lower_bound
        uint lo = 0, hi = n_build;
        while (lo < hi) {
            const uint mid = (lo + hi) >> 1;
            if (build_sorted[mid] < k) lo = mid + 1; else hi = mid;
        }
        const uint first = lo;
        // upper_bound, resuming from lower_bound
        hi = n_build;
        while (lo < hi) {
            const uint mid = (lo + hi) >> 1;
            if (build_sorted[mid] <= k) lo = mid + 1; else hi = mid;
        }
        const uint m = lo - first;
        uint c;
        switch (mode) {
            case 1:  c = m ? m : 1; break;   // LEFT
            case 2:  c = m ? 1 : 0; break;   // SEMI
            case 3:  c = m ? 0 : 1; break;   // ANTI
            default: c = m;         break;   // INNER
        }
        if (c != 0) {
            local_sum += (ulong)c * (ulong)payload[i];
            local_cnt += (long)c;
        }
    }
    shm_sum[tid] = (long)local_sum;
    shm_cnt[tid] = local_cnt;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shm_sum[tid] += shm_sum[tid + s];
            shm_cnt[tid] += shm_cnt[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        partials[2 * block_id]     = shm_sum[0];
        partials[2 * block_id + 1] = shm_cnt[0];
    }
}

kernel void join_sum_partials_i64(
    device const long* partials [[buffer(0)]],
    device long*       out      [[buffer(1)]],    // out[0]=sum, out[1]=matched
    constant uint&     n_blocks [[buffer(2)]],
    uint               tid      [[thread_position_in_threadgroup]])
{
    threadgroup long shm_sum[BLOCK];
    threadgroup long shm_cnt[BLOCK];
    long ls = 0, lc = 0;
    for (uint i = tid; i < n_blocks; i += BLOCK) {
        ls += partials[2 * i];
        lc += partials[2 * i + 1];
    }
    shm_sum[tid] = ls;
    shm_cnt[tid] = lc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shm_sum[tid] += shm_sum[tid + s];
            shm_cnt[tid] += shm_cnt[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        out[0] = shm_sum[0];
        out[1] = shm_cnt[0];
    }
}

// Multiplicity-only variant for the f64-payload join: the GPU performs the
// binary searches (the random-access part it is fast at) and writes each
// probe element's per-kind contribution count c[i]; the host then streams
// sum += c[i] * payload_f64[i] in one sequential pass (no doubles in MSL).
kernel void join_mult_i64(
    device const long*  probe_keys   [[buffer(0)]],
    device const long*  build_sorted [[buffer(1)]],
    device uint*        mult         [[buffer(2)]],
    constant uint&      n_probe      [[buffer(3)]],
    constant uint&      n_build      [[buffer(4)]],
    constant uint&      mode         [[buffer(5)]],
    uint                gid          [[thread_position_in_grid]],
    uint                gsize        [[threads_per_grid]])
{
    for (uint i = gid; i < n_probe; i += gsize) {
        const long k = probe_keys[i];
        uint lo = 0, hi = n_build;
        while (lo < hi) {
            const uint mid = (lo + hi) >> 1;
            if (build_sorted[mid] < k) lo = mid + 1; else hi = mid;
        }
        const uint first = lo;
        hi = n_build;
        while (lo < hi) {
            const uint mid = (lo + hi) >> 1;
            if (build_sorted[mid] <= k) lo = mid + 1; else hi = mid;
        }
        const uint m = lo - first;
        uint c;
        switch (mode) {
            case 1:  c = m ? m : 1; break;   // LEFT
            case 2:  c = m ? 1 : 0; break;   // SEMI
            case 3:  c = m ? 0 : 1; break;   // ANTI
            default: c = m;         break;   // INNER
        }
        mult[i] = c;
    }
}

// Lookup variant for the row-returning join: per probe element, write the
// build-side match count m[i] and the first-match position first[i] in the
// SORTED build keys (meaningful only when m[i] > 0). Kind-independent —
// the host applies the JoinKind emission rules using these two arrays plus
// the sort permutation.
kernel void join_lookup_i64(
    device const long*  probe_keys   [[buffer(0)]],
    device const long*  build_sorted [[buffer(1)]],
    device uint*        mcount       [[buffer(2)]],
    device uint*        first        [[buffer(3)]],
    constant uint&      n_probe      [[buffer(4)]],
    constant uint&      n_build      [[buffer(5)]],
    uint                gid          [[thread_position_in_grid]],
    uint                gsize        [[threads_per_grid]])
{
    for (uint i = gid; i < n_probe; i += gsize) {
        const long k = probe_keys[i];
        uint lo = 0, hi = n_build;
        while (lo < hi) {
            const uint mid = (lo + hi) >> 1;
            if (build_sorted[mid] < k) lo = mid + 1; else hi = mid;
        }
        const uint f = lo;
        hi = n_build;
        while (lo < hi) {
            const uint mid = (lo + hi) >> 1;
            if (build_sorted[mid] <= k) lo = mid + 1; else hi = mid;
        }
        mcount[i] = lo - f;
        first[i]  = f;
    }
}

// ===================== resident GROUP BY / top-k (v0.6) =====================
//
// Input is a key column already radix-sorted by the join build cache
// (sorted keys + perm = original upload indices). A "run" of equal sorted
// keys is one group. Pipeline (host orchestrates, see metal_aggregator.mm):
//   gb_block_counts_i64  per-256-block count of run starts
//   (host: exclusive scan of the block counts → block offsets, num_segs)
//   gb_run_starts_i64    starts[seg] = sorted position of each run start,
//                        in key order (block offset + in-block prefix scan)
//   gb_chunk_sum_i64     chunked segmented sum of vals[perm[i]]: runs fully
//                        inside a 64-element chunk are written directly by
//                        their exclusive owner; boundary-crossing runs leave
//                        a head / tail partial per chunk
//   gb_finalize_i64      keys + counts for every segment; sums for the
//                        boundary-crossing segments (tail of the first chunk
//                        + heads of the chunks it spans)
// Sums are ulong wrap-adds (the cross-backend rule). No atomics, no 64-bit
// CAS needed, output already sorted by key.

constant uint GB_CHUNK = 64;

kernel void gb_block_counts_i64(
    device const long* keys         [[buffer(0)]],
    constant uint&     n            [[buffer(1)]],
    device uint*       block_counts [[buffer(2)]],
    uint               tid          [[thread_position_in_threadgroup]],
    uint               gid          [[thread_position_in_grid]],
    uint               block_id     [[threadgroup_position_in_grid]])
{
    threadgroup uint shm[BLOCK];
    uint f = 0u;
    if (gid < n) f = (gid == 0u || keys[gid] != keys[gid - 1u]) ? 1u : 0u;
    shm[tid] = f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) shm[tid] += shm[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) block_counts[block_id] = shm[0];
}

kernel void gb_run_starts_i64(
    device const long* keys          [[buffer(0)]],
    constant uint&     n             [[buffer(1)]],
    device const uint* block_offsets [[buffer(2)]],
    device uint*       starts        [[buffer(3)]],
    uint               tid           [[thread_position_in_threadgroup]],
    uint               gid           [[thread_position_in_grid]],
    uint               block_id      [[threadgroup_position_in_grid]],
    uint               lane          [[thread_index_in_simdgroup]],
    uint               sg            [[simdgroup_index_in_threadgroup]],
    uint               sg_size       [[threads_per_simdgroup]])
{
    threadgroup uint sg_tot[BLOCK];
    uint f = 0u;
    if (gid < n) f = (gid == 0u || keys[gid] != keys[gid - 1u]) ? 1u : 0u;
    const uint lane_ex = simd_prefix_exclusive_sum(f);
    const uint sg_sum  = simd_sum(f);
    if (lane == 0) sg_tot[sg] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint sg_off = 0u;
    for (uint s = 0; s < sg; ++s) sg_off += sg_tot[s];
    if (f) starts[block_offsets[block_id] + sg_off + lane_ex] = gid;
    (void)tid; (void)sg_size;
}

kernel void gb_chunk_sum_i64(
    device const long* keys     [[buffer(0)]],   // sorted
    device const long* perm     [[buffer(1)]],   // sorted pos -> original index
    device const long* vals     [[buffer(2)]],   // original order
    device const uint* starts   [[buffer(3)]],
    constant uint&     n        [[buffer(4)]],
    constant uint&     num_segs [[buffer(5)]],
    device long*       out_sums [[buffer(6)]],
    device long*       head_sum [[buffer(7)]],
    device long*       tail_sum [[buffer(8)]],
    uint               gid      [[thread_position_in_grid]])
{
    const uint a = gid * GB_CHUNK;
    if (a >= n) return;
    const uint b = (n - a < GB_CHUNK) ? n : a + GB_CHUNK;   // no uint wrap near 2^32
    // segment containing position a: upper_bound(starts, a) - 1
    uint lo = 0, hi = num_segs;
    while (lo < hi) {
        const uint mid = (lo + hi) >> 1;
        if (starts[mid] <= a) lo = mid + 1; else hi = mid;
    }
    uint seg = lo - 1;
    uint i = a;
    ulong hs = 0, ts = 0;
    while (i < b) {
        const uint rs = starts[seg];
        const uint re = (seg + 1 < num_segs) ? starts[seg + 1] : n;
        const uint e  = min(re, b);
        ulong s = 0;
        for (uint j = i; j < e; ++j) s += (ulong)vals[perm[j]];
        if (rs < a)      hs = s;                  // started before this chunk
        else if (re > b) ts = s;                  // started here, continues past
        else             out_sums[seg] = (long)s; // interior: exclusive owner
        i = e; ++seg;
    }
    head_sum[gid] = (long)hs;
    tail_sum[gid] = (long)ts;
    (void)keys;
}

kernel void gb_finalize_i64(
    device const long* keys       [[buffer(0)]],
    device const uint* starts     [[buffer(1)]],
    constant uint&     n          [[buffer(2)]],
    constant uint&     num_segs   [[buffer(3)]],
    device const long* head_sum   [[buffer(4)]],
    device const long* tail_sum   [[buffer(5)]],
    device long*       out_keys   [[buffer(6)]],
    device long*       out_sums   [[buffer(7)]],
    device long*       out_counts [[buffer(8)]],
    constant uint&     with_sums  [[buffer(9)]],
    uint               gid        [[thread_position_in_grid]])
{
    if (gid >= num_segs) return;
    const uint rs = starts[gid];
    const uint re = (gid + 1 < num_segs) ? starts[gid + 1] : n;
    out_keys[gid]   = keys[rs];
    out_counts[gid] = (long)(re - rs);
    if (with_sums != 0u) {
        const uint c0 = rs / GB_CHUNK, c1 = (re - 1u) / GB_CHUNK;
        if (c0 < c1) {
            ulong s = (ulong)tail_sum[c0];
            for (uint t = c0 + 1u; t <= c1; ++t) s += (ulong)head_sum[t];
            out_sums[gid] = (long)s;
        }
    }
}

// dst[i] = src[perm[i]] as raw 64-bit words — used to lay an f64 payload out
// in sorted-key order so the host can stream per-segment double sums
// sequentially (no doubles in MSL; the gather is the random-access part).
kernel void gb_gather_i64(
    device const long* perm [[buffer(0)]],
    device const long* src  [[buffer(1)]],
    device long*       dst  [[buffer(2)]],
    constant uint&     n    [[buffer(3)]],
    uint               gid  [[thread_position_in_grid]])
{
    if (gid < n) dst[gid] = src[perm[gid]];
}

// ---------------------------------------------------------------------------
//  GroupByFilter on the finalized groups (device side, v0.6)
//    HAVING:  per-block survivor counts → host scan → compaction
//    top-k:   8-pass radix select on the aggregate (order-preserving ulong
//             transform of the i64), then compaction of "strictly better than
//             the k-th" plus the first `need_equal` ties
//  cmp: 0 none, 1 >, 2 >=, 3 <, 4 <=. The aggregate is an i64 buffer (sums
//  or counts); f64 sums are filtered on the host (no doubles in MSL).
// ---------------------------------------------------------------------------

inline bool gb_keep(long a, uint cmp, long t) {
    switch (cmp) {
        case 1u: return a >  t;
        case 2u: return a >= t;
        case 3u: return a <  t;
        case 4u: return a <= t;
        default: return true;
    }
}
inline ulong gb_ord(long a) { return (ulong)a ^ 0x8000000000000000ul; }   // order-preserving

kernel void gb_having_counts_i64(
    device const long* agg          [[buffer(0)]],
    constant uint&     num_segs     [[buffer(1)]],
    constant uint&     cmp          [[buffer(2)]],
    constant long&     thr          [[buffer(3)]],
    device uint*       block_counts [[buffer(4)]],
    uint               tid          [[thread_position_in_threadgroup]],
    uint               gid          [[thread_position_in_grid]],
    uint               block_id     [[threadgroup_position_in_grid]])
{
    threadgroup uint shm[BLOCK];
    shm[tid] = (gid < num_segs && gb_keep(agg[gid], cmp, thr)) ? 1u : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) shm[tid] += shm[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) block_counts[block_id] = shm[0];
}

kernel void gb_having_compact_i64(
    device const long* agg           [[buffer(0)]],
    device const long* keys          [[buffer(1)]],
    device const long* sums          [[buffer(2)]],
    device const long* counts        [[buffer(3)]],
    constant uint&     num_segs      [[buffer(4)]],
    constant uint&     cmp           [[buffer(5)]],
    constant long&     thr           [[buffer(6)]],
    device const uint* block_offsets [[buffer(7)]],
    device long*       out_keys      [[buffer(8)]],
    device long*       out_sums      [[buffer(9)]],
    device long*       out_counts    [[buffer(10)]],
    constant uint&     with_sums     [[buffer(11)]],
    uint               gid           [[thread_position_in_grid]],
    uint               block_id      [[threadgroup_position_in_grid]],
    uint               lane          [[thread_index_in_simdgroup]],
    uint               sg            [[simdgroup_index_in_threadgroup]])
{
    threadgroup uint sg_tot[BLOCK];
    const uint f = (gid < num_segs && gb_keep(agg[gid], cmp, thr)) ? 1u : 0u;
    const uint lane_ex = simd_prefix_exclusive_sum(f);
    const uint sg_sum  = simd_sum(f);
    if (lane == 0) sg_tot[sg] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint sg_off = 0u;
    for (uint s = 0; s < sg; ++s) sg_off += sg_tot[s];
    if (f) {
        const uint pos = block_offsets[block_id] + sg_off + lane_ex;
        out_keys[pos]   = keys[gid];
        out_counts[pos] = counts[gid];
        if (with_sums != 0u) out_sums[pos] = sums[gid];
    }
}

// Histogram of byte (ord >> shift) & 255 over candidates that pass cmp and
// whose higher bytes equal `prefix` under `mask`.
kernel void gb_topk_hist_i64(
    device const long*   agg      [[buffer(0)]],
    constant uint&       num_segs [[buffer(1)]],
    constant uint&       cmp      [[buffer(2)]],
    constant long&       thr      [[buffer(3)]],
    constant ulong&      prefix   [[buffer(4)]],
    constant ulong&      mask     [[buffer(5)]],
    constant uint&       shift    [[buffer(6)]],
    device atomic_uint*  hist     [[buffer(7)]],
    uint                 tid      [[thread_position_in_threadgroup]],
    uint                 gid      [[thread_position_in_grid]])
{
    threadgroup atomic_uint h[256];
    atomic_store_explicit(&h[tid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gid < num_segs) {
        const long a = agg[gid];
        if (gb_keep(a, cmp, thr)) {
            const ulong u = gb_ord(a);
            if ((u & mask) == prefix)
                atomic_fetch_add_explicit(&h[(u >> shift) & 255ul], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint v = atomic_load_explicit(&h[tid], memory_order_relaxed);
    if (v != 0u) atomic_fetch_add_explicit(&hist[tid], v, memory_order_relaxed);
}

// Per-block counts of two classes: "better than T" and "equal to T".
kernel void gb_topk_counts_i64(
    device const long* agg           [[buffer(0)]],
    constant uint&     num_segs      [[buffer(1)]],
    constant uint&     cmp           [[buffer(2)]],
    constant long&     thr           [[buffer(3)]],
    constant ulong&    T             [[buffer(4)]],
    constant uint&     desc          [[buffer(5)]],
    device uint*       better_counts [[buffer(6)]],
    device uint*       equal_counts  [[buffer(7)]],
    uint               tid           [[thread_position_in_threadgroup]],
    uint               gid           [[thread_position_in_grid]],
    uint               block_id      [[threadgroup_position_in_grid]])
{
    threadgroup uint sb[BLOCK];
    threadgroup uint se[BLOCK];
    uint b = 0u, e = 0u;
    if (gid < num_segs) {
        const long a = agg[gid];
        if (gb_keep(a, cmp, thr)) {
            const ulong u = gb_ord(a);
            b = (desc != 0u) ? (u > T ? 1u : 0u) : (u < T ? 1u : 0u);
            e = (u == T) ? 1u : 0u;
        }
    }
    sb[tid] = b; se[tid] = e;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) { sb[tid] += sb[tid + s]; se[tid] += se[tid + s]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) { better_counts[block_id] = sb[0]; equal_counts[block_id] = se[0]; }
}

kernel void gb_topk_compact_i64(
    device const long* agg            [[buffer(0)]],
    device const long* keys           [[buffer(1)]],
    device const long* sums           [[buffer(2)]],
    device const long* counts         [[buffer(3)]],
    constant uint&     num_segs       [[buffer(4)]],
    constant uint&     cmp            [[buffer(5)]],
    constant long&     thr            [[buffer(6)]],
    constant ulong&    T              [[buffer(7)]],
    constant uint&     desc           [[buffer(8)]],
    device const uint* better_offsets [[buffer(9)]],
    device const uint* equal_offsets  [[buffer(10)]],
    constant uint&     equal_base     [[buffer(11)]],   // = total "better"
    constant uint&     need_equal     [[buffer(12)]],   // ties to take
    device long*       out_keys       [[buffer(13)]],
    device long*       out_sums       [[buffer(14)]],
    device long*       out_counts     [[buffer(15)]],
    constant uint&     with_sums      [[buffer(16)]],
    uint               gid            [[thread_position_in_grid]],
    uint               block_id       [[threadgroup_position_in_grid]],
    uint               lane           [[thread_index_in_simdgroup]],
    uint               sg             [[simdgroup_index_in_threadgroup]])
{
    threadgroup uint tb[BLOCK];
    threadgroup uint te[BLOCK];
    uint b = 0u, e = 0u;
    if (gid < num_segs) {
        const long a = agg[gid];
        if (gb_keep(a, cmp, thr)) {
            const ulong u = gb_ord(a);
            b = (desc != 0u) ? (u > T ? 1u : 0u) : (u < T ? 1u : 0u);
            e = (u == T) ? 1u : 0u;
        }
    }
    const uint b_ex = simd_prefix_exclusive_sum(b), b_sum = simd_sum(b);
    const uint e_ex = simd_prefix_exclusive_sum(e), e_sum = simd_sum(e);
    if (lane == 0) { tb[sg] = b_sum; te[sg] = e_sum; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint b_off = 0u, e_off = 0u;
    for (uint s = 0; s < sg; ++s) { b_off += tb[s]; e_off += te[s]; }
    uint pos = 0xFFFFFFFFu;
    if (b) pos = better_offsets[block_id] + b_off + b_ex;
    else if (e) {
        const uint r = equal_offsets[block_id] + e_off + e_ex;
        if (r < need_equal) pos = equal_base + r;
    }
    if (pos != 0xFFFFFFFFu) {
        out_keys[pos]   = keys[gid];
        out_counts[pos] = counts[gid];
        if (with_sums != 0u) out_sums[pos] = sums[gid];
    }
}

// ---------------------------------------------------------------------------
//  Exact GROUP BY (v0.7 milestone 3, docs/TRANSPARENT_DESIGN.md §4.1 / §4.2)
//
//  Same sorted-run pipeline as gb_* above (gb_block_counts_i64 /
//  gb_run_starts_i64 give the run starts over the VALID-KEY PREFIX of the
//  sorted cache; NULL-key rows sit in a suffix the host folds into one extra
//  group), but the per-group tuple is the native one:
//    sum   as a 128-bit two's-complement integer (lo ulong, hi long) — never
//          wraps; the limb arithmetic is Sum128 in gpu_backend.hpp;
//    cnt   = count(v): payloads that are not NULL (validity bitmap, DuckDB
//          layout: row i valid iff bit i%64 of word i/64);
//    cstar = count(*) = run length (finalize);
//    mn/mx over non-NULL payloads; 0 when cnt == 0 (SQL surfaces NULL).
//  Chunk partials are 5 longs per 64-element chunk: lo, hi, cnt, mn, mx.
// ---------------------------------------------------------------------------

constant long GBX_LMAX = 0x7FFFFFFFFFFFFFFFl;
constant long GBX_LMIN = (long)0x8000000000000000ul;

struct GbxAcc { ulong lo; long hi; long cnt; long mn; long mx; };

inline GbxAcc gbx_zero() {
    GbxAcc a; a.lo = 0ul; a.hi = 0l; a.cnt = 0l; a.mn = GBX_LMAX; a.mx = GBX_LMIN; return a;
}
inline void gbx_add(thread GbxAcc& a, long v) {
    const ulong u = (ulong)v, old = a.lo;
    a.lo += u;
    a.hi += (v < 0l ? -1l : 0l) + (a.lo < old ? 1l : 0l);
    a.cnt += 1l;
    a.mn = min(a.mn, v);
    a.mx = max(a.mx, v);
}
inline void gbx_merge(thread GbxAcc& a, GbxAcc b) {
    const ulong old = a.lo;
    a.lo += b.lo;
    a.hi += b.hi + (a.lo < old ? 1l : 0l);
    a.cnt += b.cnt;
    a.mn = min(a.mn, b.mn);
    a.mx = max(a.mx, b.mx);
}
inline void gbx_store(device long* p, uint i, GbxAcc a) {
    p[5u * i + 0u] = (long)a.lo; p[5u * i + 1u] = a.hi; p[5u * i + 2u] = a.cnt;
    p[5u * i + 3u] = a.mn;       p[5u * i + 4u] = a.mx;
}
inline GbxAcc gbx_load(device const long* p, uint i) {
    GbxAcc a;
    a.lo = (ulong)p[5u * i + 0u]; a.hi = p[5u * i + 1u]; a.cnt = p[5u * i + 2u];
    a.mn = p[5u * i + 3u];        a.mx = p[5u * i + 4u];
    return a;
}
inline void gbx_out(device long* lo, device long* hi, device long* cnt,
                    device long* mn, device long* mx, uint seg, GbxAcc a) {
    lo[seg] = (long)a.lo; hi[seg] = a.hi; cnt[seg] = a.cnt;
    mn[seg] = a.cnt ? a.mn : 0l;
    mx[seg] = a.cnt ? a.mx : 0l;
}
inline bool gbx_valid(device const ulong* valid, uint has_valid, ulong row) {
    return has_valid == 0u || (((valid[row >> 6] >> (row & 63ul)) & 1ul) != 0ul);
}

kernel void gbx_chunk_i64(
    device const long*  perm      [[buffer(0)]],   // sorted pos -> original index
    device const long*  vals      [[buffer(1)]],   // original order
    device const ulong* valid     [[buffer(2)]],   // payload validity (or a dummy)
    constant uint&      has_valid [[buffer(3)]],
    device const uint*  starts    [[buffer(4)]],
    constant uint&      n         [[buffer(5)]],
    constant uint&      num_segs  [[buffer(6)]],
    device long*        out_lo    [[buffer(7)]],
    device long*        out_hi    [[buffer(8)]],
    device long*        out_cnt   [[buffer(9)]],
    device long*        out_mn    [[buffer(10)]],
    device long*        out_mx    [[buffer(11)]],
    device long*        head      [[buffer(12)]],  // 5 longs per chunk
    device long*        tail      [[buffer(13)]],
    uint                gid       [[thread_position_in_grid]])
{
    const uint a = gid * GB_CHUNK;
    if (a >= n) return;
    const uint b = (n - a < GB_CHUNK) ? n : a + GB_CHUNK;
    uint lo = 0, hi = num_segs;
    while (lo < hi) {
        const uint mid = (lo + hi) >> 1;
        if (starts[mid] <= a) lo = mid + 1; else hi = mid;
    }
    uint seg = lo - 1;
    uint i = a;
    GbxAcc hs = gbx_zero(), ts = gbx_zero();
    while (i < b) {
        const uint rs = starts[seg];
        const uint re = (seg + 1 < num_segs) ? starts[seg + 1] : n;
        const uint e  = min(re, b);
        GbxAcc s = gbx_zero();
        for (uint j = i; j < e; ++j) {
            const ulong row = (ulong)perm[j];
            if (gbx_valid(valid, has_valid, row)) gbx_add(s, vals[row]);
        }
        if (rs < a)      hs = s;
        else if (re > b) ts = s;
        else             gbx_out(out_lo, out_hi, out_cnt, out_mn, out_mx, seg, s);
        i = e; ++seg;
    }
    gbx_store(head, gid, hs);
    gbx_store(tail, gid, ts);
}

// keys + count(*) for every segment; the tuple for boundary-crossing
// segments (tail of the first chunk + heads of the chunks it spans). With
// with_vals == 0 (keys-only form) every segment gets cnt = cstar and zeros.
kernel void gbx_finalize_i64(
    device const long* keys      [[buffer(0)]],
    device const uint* starts    [[buffer(1)]],
    constant uint&     n         [[buffer(2)]],
    constant uint&     num_segs  [[buffer(3)]],
    device const long* head      [[buffer(4)]],
    device const long* tail      [[buffer(5)]],
    device long*       out_keys  [[buffer(6)]],
    device long*       out_cstar [[buffer(7)]],
    device long*       out_lo    [[buffer(8)]],
    device long*       out_hi    [[buffer(9)]],
    device long*       out_cnt   [[buffer(10)]],
    device long*       out_mn    [[buffer(11)]],
    device long*       out_mx    [[buffer(12)]],
    constant uint&     with_vals [[buffer(13)]],
    uint               gid       [[thread_position_in_grid]])
{
    if (gid >= num_segs) return;
    const uint rs = starts[gid];
    const uint re = (gid + 1 < num_segs) ? starts[gid + 1] : n;
    out_keys[gid]  = keys[rs];
    out_cstar[gid] = (long)(re - rs);
    if (with_vals == 0u) {
        out_lo[gid] = 0l; out_hi[gid] = 0l; out_cnt[gid] = (long)(re - rs);
        out_mn[gid] = 0l; out_mx[gid] = 0l;
        return;
    }
    const uint c0 = rs / GB_CHUNK, c1 = (re - 1u) / GB_CHUNK;
    if (c0 < c1) {
        GbxAcc s = gbx_load(tail, c0);
        for (uint t = c0 + 1u; t <= c1; ++t) gbx_merge(s, gbx_load(head, t));
        gbx_out(out_lo, out_hi, out_cnt, out_mn, out_mx, gid, s);
    }
}

// ---- GroupByFilter on the exact tuple ----
//  agg: 0 sum (128-bit), 1 count(v), 2 count(*), 3 min, 4 max.
//  cmp: 0 none, 1 >, 2 >=, 3 <, 4 <=, 5 "aggregate IS NULL" (used to append
//  the NULL-aggregate groups that rank last under top-k).
//  A NULL aggregate (cnt == 0 for sum/min/max) never passes 1..4.
//  Top-k ranks by the 128-bit ordinal (ord_hi, ord_lo): for sum ord_hi is
//  the order-preserving image of hi and ord_lo the raw lo limb; for the
//  int64 aggregates ord_hi is the image of the value and ord_lo is 0.

inline bool gbx_is_null(device const long* cnt, uint agg, uint gid) {
    return (agg == 0u || agg == 3u || agg == 4u) && cnt[gid] == 0l;
}
inline long gbx_agg64(device const long* cnt, device const long* cstar,
                      device const long* mn, device const long* mx, uint agg, uint gid) {
    switch (agg) {
        case 1u: return cnt[gid];
        case 2u: return cstar[gid];
        case 3u: return mn[gid];
        default: return mx[gid];
    }
}
inline bool gbx_keep(device const long* lo, device const long* hi, device const long* cnt,
                     device const long* cstar, device const long* mn, device const long* mx,
                     uint agg, uint cmp, long thr, uint gid) {
    if (cstar[gid] == 0l) return false;   // masked-out group (variant (a)): never emitted
    const bool is_null = gbx_is_null(cnt, agg, gid);
    if (cmp == 7u) return true;        // keep every (non-empty) group
    if (cmp == 5u) return is_null;
    if (cmp == 0u) return !is_null;   // no HAVING: the non-NULL groups are the top-k candidates, NULL ones are appended last by the host
    if (is_null) return false;
    if (agg == 0u) {
        const long  ah = hi[gid], th = (thr < 0l ? -1l : 0l);
        const ulong al = (ulong)lo[gid], tl = (ulong)thr;
        const bool lt = (ah != th) ? (ah < th) : (al < tl);   // a < t
        const bool gt = (ah != th) ? (ah > th) : (al > tl);   // a > t
        switch (cmp) {
            case 1u: return gt;
            case 2u: return !lt;
            case 3u: return lt;
            case 4u: return !gt;
            default: return true;
        }
    }
    return gb_keep(gbx_agg64(cnt, cstar, mn, mx, agg, gid), cmp, thr);
}
inline void gbx_ord(device const long* lo, device const long* hi, device const long* cnt,
                    device const long* cstar, device const long* mn, device const long* mx,
                    uint agg, uint gid, thread ulong& oh, thread ulong& ol) {
    if (agg == 0u) { oh = gb_ord(hi[gid]); ol = (ulong)lo[gid]; }
    else           { oh = gb_ord(gbx_agg64(cnt, cstar, mn, mx, agg, gid)); ol = 0ul; }
}

kernel void gbx_having_counts_i64(
    device const long* lo           [[buffer(0)]],
    device const long* hi           [[buffer(1)]],
    device const long* cnt          [[buffer(2)]],
    device const long* cstar        [[buffer(3)]],
    device const long* mn           [[buffer(4)]],
    device const long* mx           [[buffer(5)]],
    constant uint&     num_segs     [[buffer(6)]],
    constant uint&     agg          [[buffer(7)]],
    constant uint&     cmp          [[buffer(8)]],
    constant long&     thr          [[buffer(9)]],
    device uint*       block_counts [[buffer(10)]],
    uint               tid          [[thread_position_in_threadgroup]],
    uint               gid          [[thread_position_in_grid]],
    uint               block_id     [[threadgroup_position_in_grid]])
{
    threadgroup uint shm[BLOCK];
    shm[tid] = (gid < num_segs && gbx_keep(lo, hi, cnt, cstar, mn, mx, agg, cmp, thr, gid)) ? 1u : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) shm[tid] += shm[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) block_counts[block_id] = shm[0];
}

// Compacts the survivors (8 arrays) to out_* at base + rank, ranks >= limit
// are dropped (used to take only the first `need` NULL-aggregate groups).
kernel void gbx_having_compact_i64(
    device const long*  lo            [[buffer(0)]],
    device const long*  hi            [[buffer(1)]],
    device const long*  cnt           [[buffer(2)]],
    device const long*  cstar         [[buffer(3)]],
    device const long*  mn            [[buffer(4)]],
    device const long*  mx            [[buffer(5)]],
    device const long*  keys          [[buffer(6)]],
    device const uchar* knull         [[buffer(7)]],
    constant uint&      num_segs      [[buffer(8)]],
    constant uint&      agg           [[buffer(9)]],
    constant uint&      cmp           [[buffer(10)]],
    constant long&      thr           [[buffer(11)]],
    device const uint*  block_offsets [[buffer(12)]],
    constant uint&      base          [[buffer(13)]],
    constant uint&      limit         [[buffer(14)]],
    device long*        o_keys        [[buffer(15)]],
    device uchar*       o_knull       [[buffer(16)]],
    device long*        o_lo          [[buffer(17)]],
    device long*        o_hi          [[buffer(18)]],
    device long*        o_cnt         [[buffer(19)]],
    device long*        o_cstar       [[buffer(20)]],
    device long*        o_mn          [[buffer(21)]],
    device long*        o_mx          [[buffer(22)]],
    uint                gid           [[thread_position_in_grid]],
    uint                block_id      [[threadgroup_position_in_grid]],
    uint                lane          [[thread_index_in_simdgroup]],
    uint                sg            [[simdgroup_index_in_threadgroup]])
{
    threadgroup uint sg_tot[BLOCK];
    const uint f = (gid < num_segs && gbx_keep(lo, hi, cnt, cstar, mn, mx, agg, cmp, thr, gid)) ? 1u : 0u;
    const uint lane_ex = simd_prefix_exclusive_sum(f);
    const uint sg_sum  = simd_sum(f);
    if (lane == 0) sg_tot[sg] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint sg_off = 0u;
    for (uint s = 0; s < sg; ++s) sg_off += sg_tot[s];
    if (f) {
        const uint rank = block_offsets[block_id] + sg_off + lane_ex;
        if (rank < limit) {
            const uint pos = base + rank;
            o_keys[pos] = keys[gid];   o_knull[pos] = knull[gid];
            o_lo[pos] = lo[gid];       o_hi[pos] = hi[gid];
            o_cnt[pos] = cnt[gid];     o_cstar[pos] = cstar[gid];
            o_mn[pos] = mn[gid];       o_mx[pos] = mx[gid];
        }
    }
}

// Radix-select histogram over the 128-bit ordinal: `word` 0 = high word,
// 1 = low word; candidates pass cmp and match (prefix_hi, prefix_lo) under
// (mask_hi, mask_lo).
kernel void gbx_topk_hist_i64(
    device const long*  lo        [[buffer(0)]],
    device const long*  hi        [[buffer(1)]],
    device const long*  cnt       [[buffer(2)]],
    device const long*  cstar     [[buffer(3)]],
    device const long*  mn        [[buffer(4)]],
    device const long*  mx        [[buffer(5)]],
    constant uint&      num_segs  [[buffer(6)]],
    constant uint&      agg       [[buffer(7)]],
    constant uint&      cmp       [[buffer(8)]],
    constant long&      thr       [[buffer(9)]],
    constant ulong&     prefix_hi [[buffer(10)]],
    constant ulong&     mask_hi   [[buffer(11)]],
    constant ulong&     prefix_lo [[buffer(12)]],
    constant ulong&     mask_lo   [[buffer(13)]],
    constant uint&      word      [[buffer(14)]],
    constant uint&      shift     [[buffer(15)]],
    device atomic_uint* hist      [[buffer(16)]],
    uint                tid       [[thread_position_in_threadgroup]],
    uint                gid       [[thread_position_in_grid]])
{
    threadgroup atomic_uint h[256];
    atomic_store_explicit(&h[tid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gid < num_segs && gbx_keep(lo, hi, cnt, cstar, mn, mx, agg, cmp, thr, gid)) {
        ulong oh, ol;
        gbx_ord(lo, hi, cnt, cstar, mn, mx, agg, gid, oh, ol);
        if ((oh & mask_hi) == prefix_hi && (ol & mask_lo) == prefix_lo) {
            const ulong w = (word == 0u) ? oh : ol;
            atomic_fetch_add_explicit(&h[(w >> shift) & 255ul], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint v = atomic_load_explicit(&h[tid], memory_order_relaxed);
    if (v != 0u) atomic_fetch_add_explicit(&hist[tid], v, memory_order_relaxed);
}

inline void gbx_classes(ulong oh, ulong ol, ulong Th, ulong Tl, uint desc,
                        thread uint& better, thread uint& equal) {
    const bool lt = (oh != Th) ? (oh < Th) : (ol < Tl);
    const bool gt = (oh != Th) ? (oh > Th) : (ol > Tl);
    better = (desc != 0u) ? (gt ? 1u : 0u) : (lt ? 1u : 0u);
    equal  = (!lt && !gt) ? 1u : 0u;
}

kernel void gbx_topk_counts_i64(
    device const long* lo            [[buffer(0)]],
    device const long* hi            [[buffer(1)]],
    device const long* cnt           [[buffer(2)]],
    device const long* cstar         [[buffer(3)]],
    device const long* mn            [[buffer(4)]],
    device const long* mx            [[buffer(5)]],
    constant uint&     num_segs      [[buffer(6)]],
    constant uint&     agg           [[buffer(7)]],
    constant uint&     cmp           [[buffer(8)]],
    constant long&     thr           [[buffer(9)]],
    constant ulong&    T_hi          [[buffer(10)]],
    constant ulong&    T_lo          [[buffer(11)]],
    constant uint&     desc          [[buffer(12)]],
    device uint*       better_counts [[buffer(13)]],
    device uint*       equal_counts  [[buffer(14)]],
    uint               tid           [[thread_position_in_threadgroup]],
    uint               gid           [[thread_position_in_grid]],
    uint               block_id      [[threadgroup_position_in_grid]])
{
    threadgroup uint sb[BLOCK];
    threadgroup uint se[BLOCK];
    uint b = 0u, e = 0u;
    if (gid < num_segs && gbx_keep(lo, hi, cnt, cstar, mn, mx, agg, cmp, thr, gid)) {
        ulong oh, ol;
        gbx_ord(lo, hi, cnt, cstar, mn, mx, agg, gid, oh, ol);
        gbx_classes(oh, ol, T_hi, T_lo, desc, b, e);
    }
    sb[tid] = b; se[tid] = e;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) { sb[tid] += sb[tid + s]; se[tid] += se[tid + s]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) { better_counts[block_id] = sb[0]; equal_counts[block_id] = se[0]; }
}

kernel void gbx_topk_compact_i64(
    device const long*  lo             [[buffer(0)]],
    device const long*  hi             [[buffer(1)]],
    device const long*  cnt            [[buffer(2)]],
    device const long*  cstar          [[buffer(3)]],
    device const long*  mn             [[buffer(4)]],
    device const long*  mx             [[buffer(5)]],
    device const long*  keys           [[buffer(6)]],
    device const uchar* knull          [[buffer(7)]],
    constant uint&      num_segs       [[buffer(8)]],
    constant uint&      agg            [[buffer(9)]],
    constant uint&      cmp            [[buffer(10)]],
    constant long&      thr            [[buffer(11)]],
    constant ulong&     T_hi           [[buffer(12)]],
    constant ulong&     T_lo           [[buffer(13)]],
    constant uint&      desc           [[buffer(14)]],
    device const uint*  better_offsets [[buffer(15)]],
    device const uint*  equal_offsets  [[buffer(16)]],
    constant uint&      equal_base     [[buffer(17)]],
    constant uint&      need_equal     [[buffer(18)]],
    device long*        o_keys         [[buffer(19)]],
    device uchar*       o_knull        [[buffer(20)]],
    device long*        o_lo           [[buffer(21)]],
    device long*        o_hi           [[buffer(22)]],
    device long*        o_cnt          [[buffer(23)]],
    device long*        o_cstar        [[buffer(24)]],
    device long*        o_mn           [[buffer(25)]],
    device long*        o_mx           [[buffer(26)]],
    uint                gid            [[thread_position_in_grid]],
    uint                block_id       [[threadgroup_position_in_grid]],
    uint                lane           [[thread_index_in_simdgroup]],
    uint                sg             [[simdgroup_index_in_threadgroup]])
{
    threadgroup uint tb[BLOCK];
    threadgroup uint te[BLOCK];
    uint b = 0u, e = 0u;
    if (gid < num_segs && gbx_keep(lo, hi, cnt, cstar, mn, mx, agg, cmp, thr, gid)) {
        ulong oh, ol;
        gbx_ord(lo, hi, cnt, cstar, mn, mx, agg, gid, oh, ol);
        gbx_classes(oh, ol, T_hi, T_lo, desc, b, e);
    }
    const uint b_ex = simd_prefix_exclusive_sum(b), b_sum = simd_sum(b);
    const uint e_ex = simd_prefix_exclusive_sum(e), e_sum = simd_sum(e);
    if (lane == 0) { tb[sg] = b_sum; te[sg] = e_sum; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint b_off = 0u, e_off = 0u;
    for (uint s = 0; s < sg; ++s) { b_off += tb[s]; e_off += te[s]; }
    uint pos = 0xFFFFFFFFu;
    if (b) pos = better_offsets[block_id] + b_off + b_ex;
    else if (e) {
        const uint r = equal_offsets[block_id] + e_off + e_ex;
        if (r < need_equal) pos = equal_base + r;
    }
    if (pos != 0xFFFFFFFFu) {
        o_keys[pos] = keys[gid];   o_knull[pos] = knull[gid];
        o_lo[pos] = lo[gid];       o_hi[pos] = hi[gid];
        o_cnt[pos] = cnt[gid];     o_cstar[pos] = cstar[gid];
        o_mn[pos] = mn[gid];       o_mx[pos] = mx[gid];
    }
}

// ---------------------------------------------------------------------------
//  Exact GROUP BY — WHERE mask (v0.7 §4.6)
//
//  gbx_mask_i64   one pass per predicate over the ORIGINAL rows: pass 0
//                 writes mask[i], later passes AND into it. op: 0 = 1 != 2 <
//                 3 <= 4 > 5 >= 6 is null 7 is not null 8 in. F64 columns
//                 compare under DuckDB's total order via an order-preserving
//                 ulong image (NaN greatest and one value, -0.0 == +0.0).
//                 Rows at or past null_suffix_from are NULL (the key column's
//                 partition; 0xFFFFFFFF for every other column).
//  Variant (a)    gbxm_chunk_i64 / gbxm_finalize_i64: the exact reduce
//                 skipping masked-out rows, count(*) counted (6-long
//                 partials: lo hi cnt cstar mn mx); groups with cstar == 0
//                 are dropped by the compaction that follows.
//  Variant (b)    gbx_sel_counts_i64 / gbx_sel_compact_i64: compact the
//                 sorted keys + permutation to the surviving positions, then
//                 the ordinary gbx_chunk / gbx_finalize run over them.
// ---------------------------------------------------------------------------

inline ulong gbx_f64_key(long bits) {
    ulong u = (ulong)bits;
    const bool nan = (((u >> 52) & 0x7FFul) == 0x7FFul) && ((u & 0xFFFFFFFFFFFFFul) != 0ul);
    if (nan) return ~0ul;
    if ((u & 0x7FFFFFFFFFFFFFFFul) == 0ul) u = 0ul;                 // -0.0 -> +0.0
    return (u & 0x8000000000000000ul) ? ~u : (u | 0x8000000000000000ul);
}
inline bool gbx_cmp_u(uint op, ulong a, ulong b) {
    switch (op) {
        case 0u: return a == b;  case 1u: return a != b;
        case 2u: return a <  b;  case 3u: return a <= b;
        case 4u: return a >  b;  case 5u: return a >= b;
        default: return false;
    }
}
inline bool gbx_cmp_s(uint op, long a, long b) {
    switch (op) {
        case 0u: return a == b;  case 1u: return a != b;
        case 2u: return a <  b;  case 3u: return a <= b;
        case 4u: return a >  b;  case 5u: return a >= b;
        default: return false;
    }
}

kernel void gbx_mask_i64(
    device const long*  col              [[buffer(0)]],
    device const ulong* valid            [[buffer(1)]],
    constant uint&      has_valid        [[buffer(2)]],
    constant uint&      null_suffix_from [[buffer(3)]],
    constant uint&      n                [[buffer(4)]],
    constant uint&      is_f64           [[buffer(5)]],
    constant uint&      op               [[buffer(6)]],
    constant long&      value            [[buffer(7)]],
    device const long*  list             [[buffer(8)]],
    constant uint&      n_list           [[buffer(9)]],
    constant uint&      first            [[buffer(10)]],
    device uchar*       mask             [[buffer(11)]],
    uint                gid              [[thread_position_in_grid]])
{
    if (gid >= n) return;
    if (first == 0u && mask[gid] == 0u) return;
    const bool v = (gid < null_suffix_from) && gbx_valid(valid, has_valid, (ulong)gid);
    bool pass;
    if (op == 6u)      pass = !v;
    else if (op == 7u) pass = v;
    else if (!v)       pass = false;
    else {
        const long raw = col[gid];
        if (is_f64 != 0u) {
            const ulong a = gbx_f64_key(raw);
            if (op == 8u) {
                pass = false;
                for (uint i = 0; i < n_list && !pass; ++i) pass = (a == gbx_f64_key(list[i]));
            } else {
                pass = gbx_cmp_u(op, a, gbx_f64_key(value));
            }
        } else {
            if (op == 8u) {
                pass = false;
                for (uint i = 0; i < n_list && !pass; ++i) pass = (raw == list[i]);
            } else {
                pass = gbx_cmp_s(op, raw, value);
            }
        }
    }
    mask[gid] = pass ? 1u : 0u;
}

// ---- variant (a): masked reduce with count(*) in the tuple ----
struct GbxmAcc { ulong lo; long hi; long cnt; long cstar; long mn; long mx; };
inline GbxmAcc gbxm_zero() {
    GbxmAcc a; a.lo = 0ul; a.hi = 0l; a.cnt = 0l; a.cstar = 0l; a.mn = GBX_LMAX; a.mx = GBX_LMIN; return a;
}
inline void gbxm_add(thread GbxmAcc& a, long v) {
    const ulong u = (ulong)v, old = a.lo;
    a.lo += u;
    a.hi += (v < 0l ? -1l : 0l) + (a.lo < old ? 1l : 0l);
    a.cnt += 1l; a.mn = min(a.mn, v); a.mx = max(a.mx, v);
}
inline void gbxm_merge(thread GbxmAcc& a, GbxmAcc b) {
    const ulong old = a.lo;
    a.lo += b.lo; a.hi += b.hi + (a.lo < old ? 1l : 0l);
    a.cnt += b.cnt; a.cstar += b.cstar; a.mn = min(a.mn, b.mn); a.mx = max(a.mx, b.mx);
}
inline void gbxm_store(device long* p, uint i, GbxmAcc a) {
    p[6u * i + 0u] = (long)a.lo; p[6u * i + 1u] = a.hi; p[6u * i + 2u] = a.cnt;
    p[6u * i + 3u] = a.cstar;    p[6u * i + 4u] = a.mn; p[6u * i + 5u] = a.mx;
}
inline GbxmAcc gbxm_load(device const long* p, uint i) {
    GbxmAcc a;
    a.lo = (ulong)p[6u * i + 0u]; a.hi = p[6u * i + 1u]; a.cnt = p[6u * i + 2u];
    a.cstar = p[6u * i + 3u];     a.mn = p[6u * i + 4u]; a.mx = p[6u * i + 5u];
    return a;
}
inline void gbxm_out(device long* lo, device long* hi, device long* cnt, device long* cstar,
                     device long* mn, device long* mx, uint seg, GbxmAcc a) {
    lo[seg] = (long)a.lo; hi[seg] = a.hi; cnt[seg] = a.cnt; cstar[seg] = a.cstar;
    mn[seg] = a.cnt ? a.mn : 0l;
    mx[seg] = a.cnt ? a.mx : 0l;
}

kernel void gbxm_chunk_i64(
    device const long*  perm      [[buffer(0)]],
    device const long*  vals      [[buffer(1)]],
    device const ulong* valid     [[buffer(2)]],
    constant uint&      has_valid [[buffer(3)]],
    device const uint*  starts    [[buffer(4)]],
    constant uint&      n         [[buffer(5)]],
    constant uint&      num_segs  [[buffer(6)]],
    device const uchar* mask      [[buffer(7)]],   // over ORIGINAL rows
    constant uint&      with_vals [[buffer(8)]],
    device long*        out_lo    [[buffer(9)]],
    device long*        out_hi    [[buffer(10)]],
    device long*        out_cnt   [[buffer(11)]],
    device long*        out_cstar [[buffer(12)]],
    device long*        out_mn    [[buffer(13)]],
    device long*        out_mx    [[buffer(14)]],
    device long*        head      [[buffer(15)]],  // 6 longs per chunk
    device long*        tail      [[buffer(16)]],
    uint                gid       [[thread_position_in_grid]])
{
    const uint a = gid * GB_CHUNK;
    if (a >= n) return;
    const uint b = (n - a < GB_CHUNK) ? n : a + GB_CHUNK;
    uint lo = 0, hi = num_segs;
    while (lo < hi) {
        const uint mid = (lo + hi) >> 1;
        if (starts[mid] <= a) lo = mid + 1; else hi = mid;
    }
    uint seg = lo - 1;
    uint i = a;
    GbxmAcc hs = gbxm_zero(), ts = gbxm_zero();
    while (i < b) {
        const uint rs = starts[seg];
        const uint re = (seg + 1 < num_segs) ? starts[seg + 1] : n;
        const uint e  = min(re, b);
        GbxmAcc s = gbxm_zero();
        for (uint j = i; j < e; ++j) {
            const ulong row = (ulong)perm[j];
            if (mask[row] == 0u) continue;
            s.cstar += 1l;
            if (with_vals != 0u && gbx_valid(valid, has_valid, row)) gbxm_add(s, vals[row]);
        }
        if (with_vals == 0u) s.cnt = s.cstar;
        if (rs < a)      hs = s;
        else if (re > b) ts = s;
        else             gbxm_out(out_lo, out_hi, out_cnt, out_cstar, out_mn, out_mx, seg, s);
        i = e; ++seg;
    }
    gbxm_store(head, gid, hs);
    gbxm_store(tail, gid, ts);
}

kernel void gbxm_finalize_i64(
    device const long* keys      [[buffer(0)]],
    device const uint* starts    [[buffer(1)]],
    constant uint&     n         [[buffer(2)]],
    constant uint&     num_segs  [[buffer(3)]],
    device const long* head      [[buffer(4)]],
    device const long* tail      [[buffer(5)]],
    device long*       out_keys  [[buffer(6)]],
    device long*       out_cstar [[buffer(7)]],
    device long*       out_lo    [[buffer(8)]],
    device long*       out_hi    [[buffer(9)]],
    device long*       out_cnt   [[buffer(10)]],
    device long*       out_mn    [[buffer(11)]],
    device long*       out_mx    [[buffer(12)]],
    uint               gid       [[thread_position_in_grid]])
{
    if (gid >= num_segs) return;
    const uint rs = starts[gid];
    const uint re = (gid + 1 < num_segs) ? starts[gid + 1] : n;
    out_keys[gid] = keys[rs];
    const uint c0 = rs / GB_CHUNK, c1 = (re - 1u) / GB_CHUNK;
    if (c0 < c1) {
        GbxmAcc s = gbxm_load(tail, c0);
        for (uint t = c0 + 1u; t <= c1; ++t) gbxm_merge(s, gbxm_load(head, t));
        gbxm_out(out_lo, out_hi, out_cnt, out_cstar, out_mn, out_mx, gid, s);
    }
}

// ---- variant (b): compact the sorted positions that survive the mask ----
kernel void gbx_sel_counts_i64(
    device const long*  perm         [[buffer(0)]],
    device const uchar* mask         [[buffer(1)]],
    constant uint&      n            [[buffer(2)]],
    device uint*        block_counts [[buffer(3)]],
    uint                tid          [[thread_position_in_threadgroup]],
    uint                gid          [[thread_position_in_grid]],
    uint                block_id     [[threadgroup_position_in_grid]])
{
    threadgroup uint shm[BLOCK];
    shm[tid] = (gid < n && mask[(ulong)perm[gid]] != 0u) ? 1u : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) shm[tid] += shm[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) block_counts[block_id] = shm[0];
}

kernel void gbx_sel_compact_i64(
    device const long*  keys          [[buffer(0)]],
    device const long*  perm          [[buffer(1)]],
    device const uchar* mask          [[buffer(2)]],
    constant uint&      n             [[buffer(3)]],
    device const uint*  block_offsets [[buffer(4)]],
    device long*        o_keys        [[buffer(5)]],
    device long*        o_perm        [[buffer(6)]],
    uint                gid           [[thread_position_in_grid]],
    uint                block_id      [[threadgroup_position_in_grid]],
    uint                lane          [[thread_index_in_simdgroup]],
    uint                sg            [[simdgroup_index_in_threadgroup]])
{
    threadgroup uint sg_tot[BLOCK];
    const uint f = (gid < n && mask[(ulong)perm[gid]] != 0u) ? 1u : 0u;
    const uint lane_ex = simd_prefix_exclusive_sum(f);
    const uint sg_sum  = simd_sum(f);
    if (lane == 0) sg_tot[sg] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint sg_off = 0u;
    for (uint s = 0; s < sg; ++s) sg_off += sg_tot[s];
    if (f) {
        const uint pos = block_offsets[block_id] + sg_off + lane_ex;
        o_keys[pos] = keys[gid];
        o_perm[pos] = perm[gid];
    }
}

// =====================================================================
//  v0.7 §4.8: the materialised key join (gpu_backend.hpp join_materialize)
//
//  Build side: the sorted valid build keys + the permutation back to build
//  rows. jm_unique flags an adjacent equal pair (the build key must be
//  unique). jm_probe binary-searches every valid probe key and classifies
//  the row: 0 no match, 1 match with a valid output key, 2 match with a NULL
//  output key. Class 1 rows fill [0, n1) of the output in probe order, class
//  2 rows the suffix [n1, n1+n2): block counts -> host scan -> jm_pos writes
//  every kept row's destination; jm_gather then moves one lane per dispatch,
//  clearing the destination validity bit of a NULL source cell (the bitmap
//  is DuckDB's uint64 layout, addressed as little-endian uint32 halves so
//  the clear can be atomic).
// =====================================================================
inline bool jm_valid(device const ulong* valid, uint has_valid, uint null_from, ulong row) {
    return row < (ulong)null_from && gbx_valid(valid, has_valid, row);
}

kernel void jm_unique_i64(
    device const long* sorted [[buffer(0)]],
    constant uint&     n      [[buffer(1)]],
    device atomic_uint* flag  [[buffer(2)]],
    uint               gid    [[thread_position_in_grid]])
{
    if (gid + 1u >= n) return;
    if (sorted[gid] == sorted[gid + 1u]) atomic_store_explicit(flag, 1u, memory_order_relaxed);
}

kernel void jm_probe_i64(
    device const long*  pkey         [[buffer(0)]],
    device const ulong* pvalid       [[buffer(1)]],
    constant uint&      p_has_valid  [[buffer(2)]],
    constant uint&      p_null_from  [[buffer(3)]],
    constant uint&      n            [[buffer(4)]],
    device const long*  sorted       [[buffer(5)]],
    device const long*  perm         [[buffer(6)]],
    constant uint&      nb           [[buffer(7)]],
    device const ulong* kvalid       [[buffer(8)]],
    constant uint&      k_has_valid  [[buffer(9)]],
    constant uint&      k_null_from  [[buffer(10)]],
    constant uint&      k_from_build [[buffer(11)]],
    device uint*        match        [[buffer(12)]],
    device uchar*       cls          [[buffer(13)]],
    uint                gid          [[thread_position_in_grid]])
{
    if (gid >= n) return;
    uint  m = 0xFFFFFFFFu;
    uchar c = 0u;
    if (jm_valid(pvalid, p_has_valid, p_null_from, (ulong)gid)) {
        const long k = pkey[gid];
        uint lo = 0u, hi = nb;
        while (lo < hi) {
            const uint mid = lo + ((hi - lo) >> 1);
            if (sorted[mid] < k) lo = mid + 1u; else hi = mid;
        }
        if (lo < nb && sorted[lo] == k) {
            m = (uint)perm[lo];
            const ulong krow = (k_from_build != 0u) ? (ulong)m : (ulong)gid;
            c = jm_valid(kvalid, k_has_valid, k_null_from, krow) ? 1u : 2u;
        }
    }
    match[gid] = m;
    cls[gid]   = c;
}

kernel void jm_counts(
    device const uchar* cls      [[buffer(0)]],
    constant uint&      n        [[buffer(1)]],
    device uint*        counts1  [[buffer(2)]],
    device uint*        counts2  [[buffer(3)]],
    uint                tid      [[thread_position_in_threadgroup]],
    uint                gid      [[thread_position_in_grid]],
    uint                block_id [[threadgroup_position_in_grid]])
{
    threadgroup uint s1[BLOCK];
    threadgroup uint s2[BLOCK];
    const uchar c = (gid < n) ? cls[gid] : (uchar)0u;
    s1[tid] = (c == 1u) ? 1u : 0u;
    s2[tid] = (c == 2u) ? 1u : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) { s1[tid] += s1[tid + s]; s2[tid] += s2[tid + s]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) { counts1[block_id] = s1[0]; counts2[block_id] = s2[0]; }
}

kernel void jm_pos(
    device const uchar* cls      [[buffer(0)]],
    constant uint&      n        [[buffer(1)]],
    device const uint*  off1     [[buffer(2)]],
    device const uint*  off2     [[buffer(3)]],
    constant uint&      n1       [[buffer(4)]],
    device uint*        pos      [[buffer(5)]],
    uint                gid      [[thread_position_in_grid]],
    uint                block_id [[threadgroup_position_in_grid]],
    uint                lane     [[thread_index_in_simdgroup]],
    uint                sg       [[simdgroup_index_in_threadgroup]])
{
    threadgroup uint t1[BLOCK];
    threadgroup uint t2[BLOCK];
    const uchar c = (gid < n) ? cls[gid] : (uchar)0u;
    const uint f1 = (c == 1u) ? 1u : 0u;
    const uint f2 = (c == 2u) ? 1u : 0u;
    const uint ex1 = simd_prefix_exclusive_sum(f1);
    const uint ex2 = simd_prefix_exclusive_sum(f2);
    const uint sum1 = simd_sum(f1);
    const uint sum2 = simd_sum(f2);
    if (lane == 0) { t1[sg] = sum1; t2[sg] = sum2; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint o1 = 0u, o2 = 0u;
    for (uint s = 0; s < sg; ++s) { o1 += t1[s]; o2 += t2[s]; }
    if (f1)      pos[gid] = off1[block_id] + o1 + ex1;
    else if (f2) pos[gid] = n1 + off2[block_id] + o2 + ex2;
}

kernel void jm_gather(
    device const long*  src         [[buffer(0)]],
    device const ulong* svalid      [[buffer(1)]],
    constant uint&      s_has_valid [[buffer(2)]],
    constant uint&      s_null_from [[buffer(3)]],
    constant uint&      from_build  [[buffer(4)]],
    device const uint*  match       [[buffer(5)]],
    device const uchar* cls         [[buffer(6)]],
    device const uint*  pos         [[buffer(7)]],
    constant uint&      n           [[buffer(8)]],
    device long*        dst         [[buffer(9)]],
    device atomic_uint* dvalid      [[buffer(10)]],
    uint                gid         [[thread_position_in_grid]])
{
    if (gid >= n || cls[gid] == 0u) return;
    const ulong srow = (from_build != 0u) ? (ulong)match[gid] : (ulong)gid;
    const uint  d = pos[gid];
    if (jm_valid(svalid, s_has_valid, s_null_from, srow)) {
        dst[d] = src[srow];
    } else {
        dst[d] = 0l;
        atomic_fetch_and_explicit(&dvalid[d >> 5], ~(1u << (d & 31u)), memory_order_relaxed);
    }
}
