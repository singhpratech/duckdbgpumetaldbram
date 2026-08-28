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
