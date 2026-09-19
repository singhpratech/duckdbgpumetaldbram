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

// ---- narrow lane storage (docs/RESIDENT_COLUMNS_DESIGN.md §6, stage C) ----
// An exact I64 lane lives at the narrowest signed width its values fit — 1, 2,
// 4 or 8 bytes, chosen from the upload's min/max. A sorted-key cache keeps its
// key's width and its permutation is u32. Every kernel that reads a lane or a
// sorted key takes that width as a uniform and widens here; the branch is the
// same for every thread of a dispatch, so it costs a scalar compare.
inline long ldw(device const uchar* p, uint w, uint i) {
    switch (w) {
        case 1u: return (long)((device const char*)p)[i];
        case 2u: return (long)((device const short*)p)[i];
        case 4u: return (long)((device const int*)p)[i];
        default: return ((device const long*)p)[i];
    }
}
inline void stw(device uchar* p, uint w, uint i, long v) {
    switch (w) {
        case 1u: ((device char*)p)[i]  = (char)v;  break;
        case 2u: ((device short*)p)[i] = (short)v; break;
        case 4u: ((device int*)p)[i]   = (int)v;   break;
        default: ((device long*)p)[i]  = v;        break;
    }
}

// ===================== int64 SUM =====================

kernel void sum_i64(
    device const uchar* in       [[buffer(0)]],
    device long*        partials [[buffer(1)]],
    constant uint&      n        [[buffer(2)]],
    constant uint&      w        [[buffer(3)]],
    uint                tid      [[thread_position_in_threadgroup]],
    uint                gid      [[thread_position_in_grid]],
    uint                gsize    [[threads_per_grid]],
    uint                block_id [[threadgroup_position_in_grid]])
{
    threadgroup long shm[BLOCK];
    long local = 0;
    for (uint i = gid; i < n; i += gsize) local += ldw(in, w, i);
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
    device const uchar* in       [[buffer(0)]],
    device long*        partials [[buffer(1)]],
    constant uint&      n        [[buffer(2)]],
    constant long&      init     [[buffer(3)]],
    constant uint&      w        [[buffer(4)]],
    uint                tid      [[thread_position_in_threadgroup]],
    uint                gid      [[thread_position_in_grid]],
    uint                gsize    [[threads_per_grid]],
    uint                block_id [[threadgroup_position_in_grid]])
{
    threadgroup long shm[BLOCK];
    long local = init;
    for (uint i = gid; i < n; i += gsize) local = min(local, ldw(in, w, i));
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
    device const uchar* in       [[buffer(0)]],
    device long*        partials [[buffer(1)]],
    constant uint&      n        [[buffer(2)]],
    constant long&      init     [[buffer(3)]],
    constant uint&      w        [[buffer(4)]],
    uint                tid      [[thread_position_in_threadgroup]],
    uint                gid      [[thread_position_in_grid]],
    uint                gsize    [[threads_per_grid]],
    uint                block_id [[threadgroup_position_in_grid]])
{
    threadgroup long shm[BLOCK];
    long local = init;
    for (uint i = gid; i < n; i += gsize) local = max(local, ldw(in, w, i));
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
    device const uchar* in       [[buffer(0)]],
    device long*        partials [[buffer(1)]],   // 4 longs per block
    constant uint&      n        [[buffer(2)]],
    constant uint&      w        [[buffer(3)]],
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
        long x = ldw(in, w, i);
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
    device const uchar* probe_keys   [[buffer(0)]],
    device const uchar* payload      [[buffer(1)]],
    device const uchar* build_sorted [[buffer(2)]],
    device long*        partials     [[buffer(3)]],   // 2 per block: sum, matched
    constant uint&      n_probe      [[buffer(4)]],
    constant uint&      n_build      [[buffer(5)]],
    constant uint&      mode         [[buffer(6)]],
    constant uint&      pk_w         [[buffer(7)]],
    constant uint&      pl_w         [[buffer(8)]],
    constant uint&      bs_w         [[buffer(9)]],
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
        const long k = ldw(probe_keys, pk_w, i);
        // lower_bound
        uint lo = 0, hi = n_build;
        while (lo < hi) {
            const uint mid = (lo + hi) >> 1;
            if (ldw(build_sorted, bs_w, mid) < k) lo = mid + 1; else hi = mid;
        }
        const uint first = lo;
        // upper_bound, resuming from lower_bound
        hi = n_build;
        while (lo < hi) {
            const uint mid = (lo + hi) >> 1;
            if (ldw(build_sorted, bs_w, mid) <= k) lo = mid + 1; else hi = mid;
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
            local_sum += (ulong)c * (ulong)ldw(payload, pl_w, i);
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
    device const uchar* probe_keys   [[buffer(0)]],
    device const uchar* build_sorted [[buffer(1)]],
    device uint*        mult         [[buffer(2)]],
    constant uint&      n_probe      [[buffer(3)]],
    constant uint&      n_build      [[buffer(4)]],
    constant uint&      mode         [[buffer(5)]],
    constant uint&      pk_w         [[buffer(6)]],
    constant uint&      bs_w         [[buffer(7)]],
    uint                gid          [[thread_position_in_grid]],
    uint                gsize        [[threads_per_grid]])
{
    for (uint i = gid; i < n_probe; i += gsize) {
        const long k = ldw(probe_keys, pk_w, i);
        uint lo = 0, hi = n_build;
        while (lo < hi) {
            const uint mid = (lo + hi) >> 1;
            if (ldw(build_sorted, bs_w, mid) < k) lo = mid + 1; else hi = mid;
        }
        const uint first = lo;
        hi = n_build;
        while (lo < hi) {
            const uint mid = (lo + hi) >> 1;
            if (ldw(build_sorted, bs_w, mid) <= k) lo = mid + 1; else hi = mid;
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
    device const uchar* probe_keys   [[buffer(0)]],
    device const uchar* build_sorted [[buffer(1)]],
    device uint*        mcount       [[buffer(2)]],
    device uint*        first        [[buffer(3)]],
    constant uint&      n_probe      [[buffer(4)]],
    constant uint&      n_build      [[buffer(5)]],
    constant uint&      pk_w         [[buffer(6)]],
    constant uint&      bs_w         [[buffer(7)]],
    uint                gid          [[thread_position_in_grid]],
    uint                gsize        [[threads_per_grid]])
{
    for (uint i = gid; i < n_probe; i += gsize) {
        const long k = ldw(probe_keys, pk_w, i);
        uint lo = 0, hi = n_build;
        while (lo < hi) {
            const uint mid = (lo + hi) >> 1;
            if (ldw(build_sorted, bs_w, mid) < k) lo = mid + 1; else hi = mid;
        }
        const uint f = lo;
        hi = n_build;
        while (lo < hi) {
            const uint mid = (lo + hi) >> 1;
            if (ldw(build_sorted, bs_w, mid) <= k) lo = mid + 1; else hi = mid;
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

// The sorted-key cache is narrow (kw) and may start at an element offset
// (koff): a narrow element offset is not a legal MTLBuffer offset, so the
// range is passed as an index rather than bound into the buffer.
kernel void gb_block_counts_i64(
    device const uchar* keys        [[buffer(0)]],
    constant uint&     n            [[buffer(1)]],
    device uint*       block_counts [[buffer(2)]],
    constant uint&     kw           [[buffer(3)]],
    constant uint&     koff         [[buffer(4)]],
    uint               tid          [[thread_position_in_threadgroup]],
    uint               gid          [[thread_position_in_grid]],
    uint               block_id     [[threadgroup_position_in_grid]])
{
    threadgroup uint shm[BLOCK];
    uint f = 0u;
    if (gid < n) f = (gid == 0u || ldw(keys, kw, koff + gid) != ldw(keys, kw, koff + gid - 1u)) ? 1u : 0u;
    shm[tid] = f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) shm[tid] += shm[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) block_counts[block_id] = shm[0];
}

kernel void gb_run_starts_i64(
    device const uchar* keys         [[buffer(0)]],
    constant uint&     n             [[buffer(1)]],
    device const uint* block_offsets [[buffer(2)]],
    device uint*       starts        [[buffer(3)]],
    constant uint&     kw            [[buffer(4)]],
    constant uint&     koff          [[buffer(5)]],
    uint               tid           [[thread_position_in_threadgroup]],
    uint               gid           [[thread_position_in_grid]],
    uint               block_id      [[threadgroup_position_in_grid]],
    uint               lane          [[thread_index_in_simdgroup]],
    uint               sg            [[simdgroup_index_in_threadgroup]],
    uint               sg_size       [[threads_per_simdgroup]])
{
    threadgroup uint sg_tot[BLOCK];
    uint f = 0u;
    if (gid < n) f = (gid == 0u || ldw(keys, kw, koff + gid) != ldw(keys, kw, koff + gid - 1u)) ? 1u : 0u;
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
    device const uchar* keys    [[buffer(0)]],   // sorted (unused: the runs come from `starts`)
    device const uint* perm     [[buffer(1)]],   // sorted pos -> original index
    device const uchar* vals    [[buffer(2)]],   // original order
    device const uint* starts   [[buffer(3)]],
    constant uint&     n        [[buffer(4)]],
    constant uint&     num_segs [[buffer(5)]],
    device long*       out_sums [[buffer(6)]],
    device long*       head_sum [[buffer(7)]],
    device long*       tail_sum [[buffer(8)]],
    constant uint&     vw       [[buffer(9)]],
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
        for (uint j = i; j < e; ++j) s += (ulong)ldw(vals, vw, perm[j]);
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
    device const uchar* keys      [[buffer(0)]],
    device const uint* starts     [[buffer(1)]],
    constant uint&     n          [[buffer(2)]],
    constant uint&     num_segs   [[buffer(3)]],
    device const long* head_sum   [[buffer(4)]],
    device const long* tail_sum   [[buffer(5)]],
    device long*       out_keys   [[buffer(6)]],
    device long*       out_sums   [[buffer(7)]],
    device long*       out_counts [[buffer(8)]],
    constant uint&     with_sums  [[buffer(9)]],
    constant uint&     kw         [[buffer(10)]],
    constant uint&     koff       [[buffer(11)]],
    uint               gid        [[thread_position_in_grid]])
{
    if (gid >= num_segs) return;
    const uint rs = starts[gid];
    const uint re = (gid + 1 < num_segs) ? starts[gid + 1] : n;
    out_keys[gid]   = ldw(keys, kw, koff + rs);
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
    device const uint* perm [[buffer(0)]],
    device const long* src  [[buffer(1)]],   // f64 lane: always 8 bytes wide
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
    device const uint*  perm      [[buffer(0)]],   // sorted pos -> original index
    device const uchar* vals      [[buffer(1)]],   // original order
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
    constant uint&      vw        [[buffer(14)]],
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
            const uint row = perm[j];
            if (gbx_valid(valid, has_valid, (ulong)row)) gbx_add(s, ldw(vals, vw, row));
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
// Second reduction level: block b = the merge of head[b*256 .. b*256+255].
// A group that spans many chunks (few groups over many rows: one thread per
// group used to merge ~rows/64 chunk partials one by one) takes whole blocks
// wherever 256 consecutive chunks lie inside it. The merge is an exact
// 128-bit add / min / max, associative and commutative, so the grouping of
// the partials cannot change a bit of the result.
constant uint GBX_BLOCK = 256;

kernel void gbx_blocks_i64(
    device const long* head    [[buffer(0)]],
    constant uint&     nchunks [[buffer(1)]],
    device long*       blk     [[buffer(2)]],
    uint               gid     [[thread_position_in_grid]])
{
    const uint a = gid * GBX_BLOCK;
    if (a >= nchunks) return;
    const uint b = min(a + GBX_BLOCK, nchunks);
    GbxAcc s = gbx_load(head, a);
    for (uint t = a + 1u; t < b; ++t) gbx_merge(s, gbx_load(head, t));
    gbx_store(blk, gid, s);
}

kernel void gbx_finalize_i64(
    device const uchar* keys     [[buffer(0)]],
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
    device const long* blk       [[buffer(14)]],
    constant uint&     kw        [[buffer(15)]],
    constant uint&     koff      [[buffer(16)]],
    uint               gid       [[thread_position_in_grid]])
{
    if (gid >= num_segs) return;
    const uint rs = starts[gid];
    const uint re = (gid + 1 < num_segs) ? starts[gid + 1] : n;
    out_keys[gid]  = ldw(keys, kw, koff + rs);
    out_cstar[gid] = (long)(re - rs);
    if (with_vals == 0u) {
        out_lo[gid] = 0l; out_hi[gid] = 0l; out_cnt[gid] = (long)(re - rs);
        out_mn[gid] = 0l; out_mx[gid] = 0l;
        return;
    }
    const uint c0 = rs / GB_CHUNK, c1 = (re - 1u) / GB_CHUNK;
    if (c0 < c1) {
        GbxAcc s = gbx_load(tail, c0);
        uint t = c0 + 1u;
        while (t <= c1) {
            if ((t % GBX_BLOCK) == 0u && c1 - t >= GBX_BLOCK - 1u) { gbx_merge(s, gbx_load(blk, t / GBX_BLOCK)); t += GBX_BLOCK; }
            else { gbx_merge(s, gbx_load(head, t)); ++t; }
        }
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
//
//  gbx_mask_i64 is the LEGACY shape, kept because a device that will not build
//  the fused pass still has to answer. What runs by default is the one pass
//  gbx_fused_mask_i64 further down, with gbx_sel_smask_i64 beside it.
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
    device const uchar* col              [[buffer(0)]],
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
    constant uint&      cw               [[buffer(12)]],
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
        const long raw = ldw(col, cw, gid);
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
    device const uint*  perm      [[buffer(0)]],
    device const uchar* vals      [[buffer(1)]],
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
    constant uint&      vw        [[buffer(17)]],
    constant uint&      sorted_mask [[buffer(18)]], // mask is in SORTED order, indexed by j
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
            // The mask reaches this pass one of two ways. In sorted order it
            // is read straight (the gather happened once, in the counting
            // pass); in row order every payload gathers it again.
            if (sorted_mask != 0u) { if (mask[j] == 0u) continue; }
            const uint row = perm[j];
            if (sorted_mask == 0u && mask[row] == 0u) continue;
            s.cstar += 1l;
            if (with_vals != 0u && gbx_valid(valid, has_valid, (ulong)row)) gbxm_add(s, ldw(vals, vw, row));
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

kernel void gbxm_blocks_i64(
    device const long* head    [[buffer(0)]],
    constant uint&     nchunks [[buffer(1)]],
    device long*       blk     [[buffer(2)]],
    uint               gid     [[thread_position_in_grid]])
{
    const uint a = gid * GBX_BLOCK;
    if (a >= nchunks) return;
    const uint b = min(a + GBX_BLOCK, nchunks);
    GbxmAcc s = gbxm_load(head, a);
    for (uint t = a + 1u; t < b; ++t) gbxm_merge(s, gbxm_load(head, t));
    gbxm_store(blk, gid, s);
}

kernel void gbxm_finalize_i64(
    device const uchar* keys     [[buffer(0)]],
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
    device const long* blk       [[buffer(13)]],
    constant uint&     kw        [[buffer(14)]],
    constant uint&     koff      [[buffer(15)]],
    uint               gid       [[thread_position_in_grid]])
{
    if (gid >= num_segs) return;
    const uint rs = starts[gid];
    const uint re = (gid + 1 < num_segs) ? starts[gid + 1] : n;
    out_keys[gid] = ldw(keys, kw, koff + rs);
    const uint c0 = rs / GB_CHUNK, c1 = (re - 1u) / GB_CHUNK;
    if (c0 < c1) {
        GbxmAcc s = gbxm_load(tail, c0);
        uint t = c0 + 1u;
        while (t <= c1) {
            if ((t % GBX_BLOCK) == 0u && c1 - t >= GBX_BLOCK - 1u) { gbxm_merge(s, gbxm_load(blk, t / GBX_BLOCK)); t += GBX_BLOCK; }
            else { gbxm_merge(s, gbxm_load(head, t)); ++t; }
        }
        gbxm_out(out_lo, out_hi, out_cnt, out_cstar, out_mn, out_mx, gid, s);
    }
}

// ---- variant (b): compact the sorted positions that survive the mask ----
kernel void gbx_sel_counts_i64(
    device const uint*  perm         [[buffer(0)]],
    device const uchar* mask         [[buffer(1)]],
    constant uint&      n            [[buffer(2)]],
    device uint*        block_counts [[buffer(3)]],
    uint                tid          [[thread_position_in_threadgroup]],
    uint                gid          [[thread_position_in_grid]],
    uint                block_id     [[threadgroup_position_in_grid]])
{
    threadgroup uint shm[BLOCK];
    shm[tid] = (gid < n && mask[perm[gid]] != 0u) ? 1u : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) shm[tid] += shm[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) block_counts[block_id] = shm[0];
}

kernel void gbx_sel_compact_i64(
    device const uchar* keys          [[buffer(0)]],
    device const uint*  perm          [[buffer(1)]],
    device const uchar* mask          [[buffer(2)]],
    constant uint&      n             [[buffer(3)]],
    device const uint*  block_offsets [[buffer(4)]],
    device uchar*       o_keys        [[buffer(5)]],
    device uint*        o_perm        [[buffer(6)]],
    constant uint&      kw            [[buffer(7)]],
    constant uint&      koff          [[buffer(8)]],
    constant uint&      sorted_mask   [[buffer(9)]],   // mask is in SORTED order
    uint                gid           [[thread_position_in_grid]],
    uint                block_id      [[threadgroup_position_in_grid]],
    uint                lane          [[thread_index_in_simdgroup]],
    uint                sg            [[simdgroup_index_in_threadgroup]])
{
    threadgroup uint sg_tot[BLOCK];
    const uint f = (gid < n && (sorted_mask != 0u ? mask[gid] : mask[perm[gid]]) != 0u) ? 1u : 0u;
    const uint lane_ex = simd_prefix_exclusive_sum(f);
    const uint sg_sum  = simd_sum(f);
    if (lane == 0) sg_tot[sg] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint sg_off = 0u;
    for (uint s = 0; s < sg; ++s) sg_off += sg_tot[s];
    if (f) {
        const uint pos = block_offsets[block_id] + sg_off + lane_ex;
        stw(o_keys, kw, pos, ldw(keys, kw, koff + gid));
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

// ---- stage D1: the row a lane is read at ----------------------------------
// `mode` 0 = read at the row itself (what every caller before stage D does,
// and one comparison of a register), 1 = through an index with no NULL cell,
// 2 = through an index that has a validity bitmap. False means the index cell
// is NULL, and then every lane read through it is NULL — which is what a
// gathered lane holds for the unmatched side of an outer join. The kernels
// below use the cell as a row with no bound of their own; the bound is proved
// on the host, once per column, before any of this runs.
inline bool ix_row(device const uchar* ix, device const ulong* ixv, uint mode, uint w,
                   uint row, thread uint& out) {
    if (mode == 0u) { out = row; return true; }
    if (mode == 2u && !gbx_valid(ixv, 1u, (ulong)row)) return false;
    out = (uint)ldw(ix, w, row);
    return true;
}

// Materialise one lane through an index: dst[i] = src[ix[i]], the gather the
// join used to do per output lane, kept for the paths that still want a lane
// in result order. `dvalid` starts all-ones and a NULL cell clears its bit.
kernel void ix_gather(
    device const uchar* src         [[buffer(0)]],
    device const ulong* svalid      [[buffer(1)]],
    constant uint&      s_has_valid [[buffer(2)]],
    device const uchar* ix          [[buffer(3)]],
    device const ulong* ixvalid     [[buffer(4)]],
    constant uint&      ix_mode     [[buffer(5)]],
    constant uint&      ix_w        [[buffer(6)]],
    constant uint&      n           [[buffer(7)]],
    device uchar*       dst         [[buffer(8)]],
    device atomic_uint* dvalid      [[buffer(9)]],
    constant uint&      w           [[buffer(10)]],
    uint                gid         [[thread_position_in_grid]])
{
    if (gid >= n) return;
    uint srow = gid;
    const bool live = ix_row(ix, ixvalid, ix_mode, ix_w, gid, srow) &&
                      gbx_valid(svalid, s_has_valid, (ulong)srow);
    if (live) {
        stw(dst, w, gid, ldw(src, w, srow));
    } else {
        stw(dst, w, gid, 0l);
        atomic_fetch_and_explicit(&dvalid[gid >> 5], ~(1u << (gid & 31u)), memory_order_relaxed);
    }
}

kernel void jm_unique_i64(
    device const uchar* sorted [[buffer(0)]],
    constant uint&     n      [[buffer(1)]],
    device atomic_uint* flag  [[buffer(2)]],
    constant uint&     sw     [[buffer(3)]],
    uint               gid    [[thread_position_in_grid]])
{
    if (gid + 1u >= n) return;
    if (ldw(sorted, sw, gid) == ldw(sorted, sw, gid + 1u))
        atomic_store_explicit(flag, 1u, memory_order_relaxed);
}

kernel void jm_probe_i64(
    device const uchar* pkey         [[buffer(0)]],
    device const ulong* pvalid       [[buffer(1)]],
    constant uint&      p_has_valid  [[buffer(2)]],
    constant uint&      p_null_from  [[buffer(3)]],
    constant uint&      n            [[buffer(4)]],
    device const uchar* sorted       [[buffer(5)]],
    device const uint*  perm         [[buffer(6)]],
    constant uint&      nb           [[buffer(7)]],
    device const ulong* kvalid       [[buffer(8)]],
    constant uint&      k_has_valid  [[buffer(9)]],
    constant uint&      k_null_from  [[buffer(10)]],
    constant uint&      k_from_build [[buffer(11)]],
    device uint*        match        [[buffer(12)]],
    device uchar*       cls          [[buffer(13)]],
    constant uint&      pw           [[buffer(14)]],
    constant uint&      sw           [[buffer(15)]],
    // ---- stage D1 ----
    // `px`: the probe SET's own index (a chained join, whose probe side is
    // already described by an index vector) — the probe key is read at
    // px[gid], and the row this kernel classifies stays gid, so the caller
    // composes the chain's vectors itself. `kx`: the classifying key lane's
    // own index, read at the row of the lane's side of the join.
    device const uchar* px           [[buffer(16)]],
    device const ulong* pxvalid      [[buffer(17)]],
    constant uint&      px_mode      [[buffer(18)]],
    constant uint&      px_w         [[buffer(19)]],
    device const uchar* kx           [[buffer(20)]],
    device const ulong* kxvalid      [[buffer(21)]],
    constant uint&      kx_mode      [[buffer(22)]],
    constant uint&      kx_w         [[buffer(23)]],
    uint                gid          [[thread_position_in_grid]])
{
    if (gid >= n) return;
    uint  m = 0xFFFFFFFFu;
    uchar c = 0u;
    uint prow = gid;
    const bool p_live = ix_row(px, pxvalid, px_mode, px_w, gid, prow);
    if (p_live && jm_valid(pvalid, p_has_valid, p_null_from, (ulong)prow)) {
        const long k = ldw(pkey, pw, prow);
        uint lo = 0u, hi = nb;
        while (lo < hi) {
            const uint mid = lo + ((hi - lo) >> 1);
            if (ldw(sorted, sw, mid) < k) lo = mid + 1u; else hi = mid;
        }
        if (lo < nb && ldw(sorted, sw, lo) == k) {
            m = perm[lo];
            // The key lane is read on its own side, at its own row, through
            // its own index: the probe side addresses the statement's row gid
            // (the index vector already stands for the chain), the build side
            // the matched build row.
            const uint side = (k_from_build != 0u) ? m : gid;
            uint krow = side;
            const bool k_live = ix_row(kx, kxvalid, kx_mode, kx_w, side, krow);
            c = (k_live && jm_valid(kvalid, k_has_valid, k_null_from, (ulong)krow)) ? 1u : 2u;
        }
    }
    match[gid] = m;
    cls[gid]   = c;
}

// ---- stage D1: the two row vectors the gather would have read -------------
// join_index stops after jm_pos and writes, per kept output row d, the probe
// row that produced it and the build row it matched — narrow, because each is
// bounded by its side's row count. probe_rows is strictly increasing within
// each class and build_rows is never NULL for an inner join, so neither
// carries a validity bitmap.
kernel void jm_rows(
    device const uint*  match [[buffer(0)]],
    device const uchar* cls   [[buffer(1)]],
    device const uint*  pos   [[buffer(2)]],
    constant uint&      n     [[buffer(3)]],
    device uchar*       prow  [[buffer(4)]],
    constant uint&      pw    [[buffer(5)]],
    device uchar*       brow  [[buffer(6)]],
    constant uint&      bw    [[buffer(7)]],
    uint                gid   [[thread_position_in_grid]])
{
    if (gid >= n || cls[gid] == 0u) return;
    const uint d = pos[gid];
    stw(prow, pw, d, (long)gid);
    stw(brow, bw, d, (long)match[gid]);
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

// The gather does not change a lane's min/max, so the output keeps the
// source lane's storage width.
kernel void jm_gather(
    device const uchar* src         [[buffer(0)]],
    device const ulong* svalid      [[buffer(1)]],
    constant uint&      s_has_valid [[buffer(2)]],
    constant uint&      s_null_from [[buffer(3)]],
    constant uint&      from_build  [[buffer(4)]],
    device const uint*  match       [[buffer(5)]],
    device const uchar* cls         [[buffer(6)]],
    device const uint*  pos         [[buffer(7)]],
    constant uint&      n           [[buffer(8)]],
    device uchar*       dst         [[buffer(9)]],
    device atomic_uint* dvalid      [[buffer(10)]],
    constant uint&      w           [[buffer(11)]],
    // stage D1: the lane's own index, read at the row of the lane's side of
    // the join — a lane of a probe set that is itself already indexed.
    device const uchar* lx          [[buffer(12)]],
    device const ulong* lxvalid     [[buffer(13)]],
    constant uint&      lx_mode     [[buffer(14)]],
    constant uint&      lx_w        [[buffer(15)]],
    uint                gid         [[thread_position_in_grid]])
{
    if (gid >= n || cls[gid] == 0u) return;
    const uint side = (from_build != 0u) ? match[gid] : gid;
    uint srow = side;
    const bool live = ix_row(lx, lxvalid, lx_mode, lx_w, side, srow);
    const uint  d = pos[gid];
    if (live && jm_valid(svalid, s_has_valid, s_null_from, (ulong)srow)) {
        stw(dst, w, d, ldw(src, w, srow));
    } else {
        stw(dst, w, d, 0l);
        atomic_fetch_and_explicit(&dvalid[d >> 5], ~(1u << (d & 31u)), memory_order_relaxed);
    }
}

// ===========================================================================
//  v0.7 §4.12 — the global masked aggregate: aggregates without GROUP BY
//
//  ONE pass over the rows in storage order. Each thread walks a grid stride,
//  evaluates the whole WHERE conjunction for its row (gpred_eval below, not
//  one kernel pass per predicate), and folds the surviving row into the
//  payload accumulators it keeps in registers; the threadgroup then reduces
//  them into one partial block and the host merges the blocks. No key, no
//  sort cache, no permutation gather.
//
//  Accumulators are indexed [group * n_pays + payload] and the partials
//  buffer holds one block per (threadgroup, group): {count_star, then
//  n_pays tuples of (lo, hi, cnt, mn, mx)}. n_groups is 1 for the global
//  aggregate; the direct grouped reduce below (gdir_masked_*) is the same
//  kernel with n_groups > 1 and the group id read per row.
// ===========================================================================

constant uint GAGG_MAX_LANES = 12u;   // distinct lanes bound (payloads + predicate columns)
constant uint GAGG_MAX_ACC   = 8u;    // groups * payloads held per thread

// One lane of the row set, as the kernel sees it: its storage, its validity
// bitmap, its stage-C width and whether its cells are IEEE-754 images.
struct GLane {
    device const uchar* data;
    device const ulong* valid;
    uint width;
    uint has_valid;
    uint is_f64;
    // stage D1 (docs/RESIDENT_COLUMNS_DESIGN.md §7): the slot of the index
    // vector this lane is read through, or GL_NO_IDX to read at the row
    // itself. An index vector is bound as an ORDINARY LANE — it has a width
    // and a validity bitmap like any other — so stage D costs no new argument
    // table slot, which matters because the direct reduce already binds all 31
    // of them.
    uint idx_slot;
};
// The per-lane part that travels in a buffer (the pointers are bindings).
// `idx_slot` took the place of the old padding word, so the struct is the same
// size it always was.
struct GLaneMeta { uint width; uint has_valid; uint is_f64; uint idx_slot; };

constant uint GL_NO_IDX = 0xFFFFFFFFu;

// A row no lane has: what gl_row returns when the index cell is NULL, so that
// every lane read through it is NULL (the unmatched side of an outer join).
// The host refuses a column of 2^32-64 rows or more, so this cannot be a real
// row.
constant uint GL_NULL_ROW = 0xFFFFFFFFu;

// The row lane `ln` is read at for statement row `row`: the row itself, or
// the lane's index cell there.
//
// IX is a COMPILE-TIME parameter, not a test in the loop, and that is the
// whole point. `row` is the row loop's induction variable, so the address of
// `lane[row]` and the word of its validity bitmap strength-reduce across the
// loop; a row that comes back from a function does not, and the measurement
// said so — routing the read through a runtime test cost the masked kernels
// 8-15% at SF10 (BENCHMARK.md, 2026-09-19) although no statement was indexed.
// So every kernel that reads a lane is instantiated twice and picks the
// instance once, from a flag that is uniform over the whole dispatch: with
// IX == false the body below compiles away and what is left is, literally,
// the pre-stage-D kernel.
template <bool IX>
inline uint gl_row(thread const GLane* lanes, GLane ln, uint row) {
    if (!IX) return row;
    if (ln.idx_slot == GL_NO_IDX) return row;
    const GLane ix = lanes[ln.idx_slot];
    if (!gbx_valid(ix.valid, ix.has_valid, (ulong)row)) return GL_NULL_ROW;
    return (uint)ldw(ix.data, ix.width, row);
}

// Does any lane of this dispatch name an index? Read from the lane table the
// host wrote, so it is the same for every thread.
inline uint gl_any_idx(device const GLaneMeta* meta, uint n_lanes) {
    uint any = 0u;
    for (uint l = 0; l < n_lanes; ++l) any |= (meta[l].idx_slot != GL_NO_IDX) ? 1u : 0u;
    return any;
}

// One term of the conjunction. `op` is gbx_mask_i64's encoding (0 EQ, 1 NE,
// 2 LT, 3 LE, 4 GT, 5 GE, 6 IsNull, 7 IsNotNull, 8 In); an In list is
// n_list values of `lists` from list_off.
struct GPred { uint lane; uint op; uint list_off; uint n_list; long value; };

// The reusable single-pass evaluation of a WHERE program: program + lane
// table + validity in, one bool out. Semantics are gbx_mask_i64's, term for
// term — a NULL cell fails every comparison and In, IsNull / IsNotNull read
// the validity bit, F64 lanes compare on the total-order image.
template <bool IX>
inline bool gpred_eval(thread const GLane* lanes,
                       device const GPred* prog, uint n_preds,
                       device const long* lists, uint row)
{
    for (uint t = 0; t < n_preds; ++t) {
        const GPred pr = prog[t];
        const GLane ln = lanes[pr.lane];
        // The row this lane is read at — its own, or its index cell's. A NULL
        // index cell makes the lane NULL, so IsNull passes on it and every
        // comparison fails: the same answer a gathered lane gives.
        const uint lr = gl_row<IX>(lanes, ln, row);
        const bool v = (!IX || lr != GL_NULL_ROW) && gbx_valid(ln.valid, ln.has_valid, (ulong)lr);
        bool pass;
        if (pr.op == 6u)      pass = !v;
        else if (pr.op == 7u) pass = v;
        else if (!v)          pass = false;
        else {
            const long raw = ldw(ln.data, ln.width, lr);
            if (ln.is_f64 != 0u) {
                const ulong a = gbx_f64_key(raw);
                if (pr.op == 8u) {
                    pass = false;
                    for (uint i = 0; i < pr.n_list && !pass; ++i)
                        pass = (a == gbx_f64_key(lists[pr.list_off + i]));
                } else {
                    pass = gbx_cmp_u(pr.op, a, gbx_f64_key(pr.value));
                }
            } else {
                if (pr.op == 8u) {
                    pass = false;
                    for (uint i = 0; i < pr.n_list && !pass; ++i)
                        pass = (raw == lists[pr.list_off + i]);
                } else {
                    pass = gbx_cmp_s(pr.op, raw, pr.value);
                }
            }
        }
        if (!pass) return false;
    }
    return true;
}

struct GAggAcc { ulong lo; long hi; long cnt; long mn; long mx; };
inline GAggAcc gagg_zero() {
    GAggAcc a; a.lo = 0ul; a.hi = 0l; a.cnt = 0l; a.mn = GBX_LMAX; a.mx = GBX_LMIN; return a;
}
inline void gagg_add(thread GAggAcc& a, long v) {
    const ulong u = (ulong)v, old = a.lo;
    a.lo += u;
    a.hi += (v < 0l ? -1l : 0l) + (a.lo < old ? 1l : 0l);
    a.cnt += 1l; a.mn = min(a.mn, v); a.mx = max(a.mx, v);
}

struct GAggU { uint n; uint n_preds; uint n_pays; uint n_groups; };

kernel void gagg_masked_i64(
    device const uchar*     d0    [[buffer(0)]],  device const ulong* v0  [[buffer(1)]],
    device const uchar*     d1    [[buffer(2)]],  device const ulong* v1  [[buffer(3)]],
    device const uchar*     d2    [[buffer(4)]],  device const ulong* v2  [[buffer(5)]],
    device const uchar*     d3    [[buffer(6)]],  device const ulong* v3  [[buffer(7)]],
    device const uchar*     d4    [[buffer(8)]],  device const ulong* v4  [[buffer(9)]],
    device const uchar*     d5    [[buffer(10)]], device const ulong* v5  [[buffer(11)]],
    device const uchar*     d6    [[buffer(12)]], device const ulong* v6  [[buffer(13)]],
    device const uchar*     d7    [[buffer(14)]], device const ulong* v7  [[buffer(15)]],
    device const uchar*     d8    [[buffer(16)]], device const ulong* v8  [[buffer(17)]],
    device const uchar*     d9    [[buffer(18)]], device const ulong* v9  [[buffer(19)]],
    device const uchar*     d10   [[buffer(20)]], device const ulong* v10 [[buffer(21)]],
    device const uchar*     d11   [[buffer(22)]], device const ulong* v11 [[buffer(23)]],
    device const GLaneMeta* meta  [[buffer(24)]],   // GAGG_MAX_LANES entries
    device const GPred*     prog  [[buffer(25)]],
    device const long*      lists [[buffer(26)]],
    device const uint*      pay   [[buffer(27)]],   // lane index per payload
    constant GAggU&         u     [[buffer(28)]],
    device long*            out   [[buffer(29)]],   // per (threadgroup, group) block
    uint tid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint ntg  [[threadgroups_per_grid]])
{
    threadgroup long scr[BLOCK * 5];
    GLane lanes[GAGG_MAX_LANES];
    lanes[0].data = d0;  lanes[0].valid = v0;   lanes[1].data = d1;  lanes[1].valid = v1;
    lanes[2].data = d2;  lanes[2].valid = v2;   lanes[3].data = d3;  lanes[3].valid = v3;
    lanes[4].data = d4;  lanes[4].valid = v4;   lanes[5].data = d5;  lanes[5].valid = v5;
    lanes[6].data = d6;  lanes[6].valid = v6;   lanes[7].data = d7;  lanes[7].valid = v7;
    lanes[8].data = d8;  lanes[8].valid = v8;   lanes[9].data = d9;  lanes[9].valid = v9;
    lanes[10].data = d10; lanes[10].valid = v10; lanes[11].data = d11; lanes[11].valid = v11;
    // stage D1: `any_idx` is uniform over the dispatch and picks the
    // instance below. The index SLOT is written into the thread's lane table
    // only when there is one, so a statement that reads no join set sets up
    // exactly the table it set up before stage D.
    uint any_idx = 0u;
    for (uint l = 0; l < GAGG_MAX_LANES; ++l) {
        lanes[l].width = meta[l].width;
        lanes[l].has_valid = meta[l].has_valid;
        lanes[l].is_f64 = meta[l].is_f64;
        any_idx |= (meta[l].idx_slot != GL_NO_IDX) ? 1u : 0u;
    }
    if (any_idx) for (uint l = 0; l < GAGG_MAX_LANES; ++l) lanes[l].idx_slot = meta[l].idx_slot;

    GAggAcc acc[GAGG_MAX_ACC];
    long cstar[GAGG_MAX_ACC];
    for (uint a = 0; a < GAGG_MAX_ACC; ++a) { acc[a] = gagg_zero(); cstar[a] = 0l; }

    // Stage D1: the fold is written once and compiled twice, and the dispatch
    // picks an instance from a flag that is the same for every thread. The
    // indexed instance is only built because a statement over a join set asks
    // for it; the other one is the kernel this always was.
#define GAGG_FOLD(IX)                                                                  \
    for (uint i = tgid * BLOCK + tid; i < u.n; i += gsize) {                           \
        if (!gpred_eval<IX>(lanes, prog, u.n_preds, lists, i)) continue;               \
        const uint g = 0u;             /* one group; a few-group variant reads it */   \
        cstar[g] += 1l;                                                                \
        for (uint p = 0; p < u.n_pays; ++p) {                                          \
            const GLane ln = lanes[pay[p]];                                            \
            const uint lr = gl_row<IX>(lanes, ln, i);                                  \
            if ((IX && lr == GL_NULL_ROW) ||                                           \
                !gbx_valid(ln.valid, ln.has_valid, (ulong)lr)) continue;               \
            gagg_add(acc[g * u.n_pays + p], ldw(ln.data, ln.width, lr));               \
        }                                                                              \
    }
    const uint gsize = BLOCK * ntg;
    if (any_idx) { GAGG_FOLD(true) } else { GAGG_FOLD(false) }
#undef GAGG_FOLD

    // threadgroup reduction, one accumulator at a time through `scr`
    const uint stride = 1u + 5u * u.n_pays;          // longs per (threadgroup, group) block
    for (uint g = 0; g < u.n_groups; ++g) {
        const uint base = (tgid * u.n_groups + g) * stride;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        scr[tid * 5u] = cstar[g];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = BLOCK / 2u; s > 0u; s >>= 1) {
            if (tid < s) scr[tid * 5u] += scr[(tid + s) * 5u];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0u) out[base] = scr[0];
        for (uint p = 0; p < u.n_pays; ++p) {
            const GAggAcc a = acc[g * u.n_pays + p];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            scr[tid * 5u + 0u] = (long)a.lo; scr[tid * 5u + 1u] = a.hi; scr[tid * 5u + 2u] = a.cnt;
            scr[tid * 5u + 3u] = a.mn;       scr[tid * 5u + 4u] = a.mx;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = BLOCK / 2u; s > 0u; s >>= 1) {
                if (tid < s) {
                    const uint x = tid * 5u, y = (tid + s) * 5u;
                    const ulong old = (ulong)scr[x];
                    scr[x] = (long)(old + (ulong)scr[y]);
                    scr[x + 1u] += scr[y + 1u] + (((ulong)scr[x] < old) ? 1l : 0l);
                    scr[x + 2u] += scr[y + 2u];
                    scr[x + 3u] = min(scr[x + 3u], scr[y + 3u]);
                    scr[x + 4u] = max(scr[x + 4u], scr[y + 4u]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tid == 0u) {
                const uint o = base + 1u + 5u * p;
                out[o + 0u] = scr[0]; out[o + 1u] = scr[1]; out[o + 2u] = scr[2];
                out[o + 3u] = scr[3]; out[o + 4u] = scr[4];
            }
        }
    }
}

// ===========================================================================
//  v0.8 — the WHERE stage of the sort path, in one pass (§4.6)
//
//  gbx_mask_i64 is one dispatch per term, and each of them reads a byte of the
//  60 MB mask and writes one back to look at a single lane: five terms walked
//  the mask ten times for 3.3 ms apiece, whatever the lane's width and
//  whatever survived. gpred_eval already evaluates a whole conjunction for one
//  row — the global aggregate and the direct reduce call it — so the mask
//  stage calls it too: one pass, one read per distinct lane, one byte written.
//  Term for term the same answer (a NULL cell fails every comparison and In,
//  IsNull / IsNotNull read the validity bit, an F64 lane compares on the
//  total-order image, narrow lanes widen through ldw).
//
//  Beside it, the mask the REDUCE reads. The sort path visits rows through the
//  permutation, so mask[perm[i]] was gathered once to count the survivors and
//  again for every payload. gbx_sel_smask_i64 keeps that gather's result as a
//  mask in SORTED order, and the masked reduce and the compaction read it
//  sequentially — one random gather per call instead of one per payload plus
//  one.
// ===========================================================================

struct GMaskU { uint n; uint n_preds; uint n_lanes; uint pad; };

kernel void gbx_fused_mask_i64(
    device const uchar*     d0    [[buffer(0)]],  device const ulong* v0  [[buffer(1)]],
    device const uchar*     d1    [[buffer(2)]],  device const ulong* v1  [[buffer(3)]],
    device const uchar*     d2    [[buffer(4)]],  device const ulong* v2  [[buffer(5)]],
    device const uchar*     d3    [[buffer(6)]],  device const ulong* v3  [[buffer(7)]],
    device const uchar*     d4    [[buffer(8)]],  device const ulong* v4  [[buffer(9)]],
    device const uchar*     d5    [[buffer(10)]], device const ulong* v5  [[buffer(11)]],
    device const uchar*     d6    [[buffer(12)]], device const ulong* v6  [[buffer(13)]],
    device const uchar*     d7    [[buffer(14)]], device const ulong* v7  [[buffer(15)]],
    device const uchar*     d8    [[buffer(16)]], device const ulong* v8  [[buffer(17)]],
    device const uchar*     d9    [[buffer(18)]], device const ulong* v9  [[buffer(19)]],
    device const uchar*     d10   [[buffer(20)]], device const ulong* v10 [[buffer(21)]],
    device const uchar*     d11   [[buffer(22)]], device const ulong* v11 [[buffer(23)]],
    device const GLaneMeta* meta  [[buffer(24)]],   // GAGG_MAX_LANES entries
    device const GPred*     prog  [[buffer(25)]],
    device const long*      lists [[buffer(26)]],
    constant GMaskU&        u     [[buffer(27)]],
    device uchar*           mask  [[buffer(28)]],   // one byte per ORIGINAL row
    uint tid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint ntg  [[threadgroups_per_grid]])
{
    GLane lanes[GAGG_MAX_LANES];
    lanes[0].data = d0;  lanes[0].valid = v0;   lanes[1].data = d1;  lanes[1].valid = v1;
    lanes[2].data = d2;  lanes[2].valid = v2;   lanes[3].data = d3;  lanes[3].valid = v3;
    lanes[4].data = d4;  lanes[4].valid = v4;   lanes[5].data = d5;  lanes[5].valid = v5;
    lanes[6].data = d6;  lanes[6].valid = v6;   lanes[7].data = d7;  lanes[7].valid = v7;
    lanes[8].data = d8;  lanes[8].valid = v8;   lanes[9].data = d9;  lanes[9].valid = v9;
    lanes[10].data = d10; lanes[10].valid = v10; lanes[11].data = d11; lanes[11].valid = v11;
    // Only the slots the program names are described; gpred_eval never looks
    // past n_lanes.
    // stage D1: `any_idx` is uniform over the dispatch and picks the
    // instance below. The index SLOT is written into the thread's lane table
    // only when there is one, so a statement that reads no join set sets up
    // exactly the table it set up before stage D.
    uint any_idx = 0u;
    for (uint l = 0; l < u.n_lanes; ++l) {
        lanes[l].width = meta[l].width;
        lanes[l].has_valid = meta[l].has_valid;
        lanes[l].is_f64 = meta[l].is_f64;
        any_idx |= (meta[l].idx_slot != GL_NO_IDX) ? 1u : 0u;
    }
    if (any_idx) for (uint l = 0; l < u.n_lanes; ++l) lanes[l].idx_slot = meta[l].idx_slot;
    // A grid-stride loop, so the lane table above is set up once per thread
    // and not once per row.
    const uint gsize = BLOCK * ntg;
    if (any_idx) {
        for (uint i = tgid * BLOCK + tid; i < u.n; i += gsize)
            mask[i] = gpred_eval<true>(lanes, prog, u.n_preds, lists, i) ? 1u : 0u;
    } else {
        for (uint i = tgid * BLOCK + tid; i < u.n; i += gsize)
            mask[i] = gpred_eval<false>(lanes, prog, u.n_preds, lists, i) ? 1u : 0u;
    }
}

// The survivors of the key range, counted per block of the SORTED order — and
// the gather kept. Block for block the counts gbx_sel_counts_i64 produces, so
// gbx_sel_compact_i64 reads the same offsets after the host's scan.
kernel void gbx_sel_smask_i64(
    device const uint*  perm         [[buffer(0)]],   // bound at the range's first position
    device const uchar* mask         [[buffer(1)]],   // over ORIGINAL rows
    constant uint&      n            [[buffer(2)]],   // rows of the key range
    device uint*        block_counts [[buffer(3)]],
    device uchar*       smask        [[buffer(4)]],   // out: one byte per position of the range
    uint                tid          [[thread_position_in_threadgroup]],
    uint                gid          [[thread_position_in_grid]],
    uint                block_id     [[threadgroup_position_in_grid]])
{
    threadgroup uint shm[BLOCK];
    uint f = 0u;
    if (gid < n) {
        f = (mask[perm[gid]] != 0u) ? 1u : 0u;
        smask[gid] = (uchar)f;
    }
    shm[tid] = f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) shm[tid] += shm[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) block_counts[block_id] = shm[0];
}

// The other shape of the same product: evaluate the WHERE at row = perm[i]
// and never build a row-order mask at all — one random gather per distinct
// predicate lane instead of one sequential read per lane, a 60 MB write and
// the byte gather above. It writes the same smask and the same block counts,
// so it is interchangeable with gbx_sel_smask_i64. Measured slower wherever
// the key range is large (a lane gather costs what a payload gather costs),
// and it cannot serve a NULL-key group, whose rows are not in the
// permutation; kept because it is the honest comparison and one env value
// away (GPUDB_METAL_MASK_PATH=permeval).
kernel void gbx_smask_eval_i64(
    device const uchar*     d0    [[buffer(0)]],  device const ulong* v0  [[buffer(1)]],
    device const uchar*     d1    [[buffer(2)]],  device const ulong* v1  [[buffer(3)]],
    device const uchar*     d2    [[buffer(4)]],  device const ulong* v2  [[buffer(5)]],
    device const uchar*     d3    [[buffer(6)]],  device const ulong* v3  [[buffer(7)]],
    device const uchar*     d4    [[buffer(8)]],  device const ulong* v4  [[buffer(9)]],
    device const uchar*     d5    [[buffer(10)]], device const ulong* v5  [[buffer(11)]],
    device const uchar*     d6    [[buffer(12)]], device const ulong* v6  [[buffer(13)]],
    device const uchar*     d7    [[buffer(14)]], device const ulong* v7  [[buffer(15)]],
    device const uchar*     d8    [[buffer(16)]], device const ulong* v8  [[buffer(17)]],
    device const uchar*     d9    [[buffer(18)]], device const ulong* v9  [[buffer(19)]],
    device const uchar*     d10   [[buffer(20)]], device const ulong* v10 [[buffer(21)]],
    device const uchar*     d11   [[buffer(22)]], device const ulong* v11 [[buffer(23)]],
    device const GLaneMeta* meta  [[buffer(24)]],
    device const GPred*     prog  [[buffer(25)]],
    device const long*      lists [[buffer(26)]],
    constant GMaskU&        u     [[buffer(27)]],
    device const uint*      perm  [[buffer(28)]],   // bound at the range's first position
    device uint*            block_counts [[buffer(29)]],
    device uchar*           smask [[buffer(30)]],
    uint tid      [[thread_position_in_threadgroup]],
    uint gid      [[thread_position_in_grid]],
    uint block_id [[threadgroup_position_in_grid]])
{
    threadgroup uint shm[BLOCK];
    GLane lanes[GAGG_MAX_LANES];
    lanes[0].data = d0;  lanes[0].valid = v0;   lanes[1].data = d1;  lanes[1].valid = v1;
    lanes[2].data = d2;  lanes[2].valid = v2;   lanes[3].data = d3;  lanes[3].valid = v3;
    lanes[4].data = d4;  lanes[4].valid = v4;   lanes[5].data = d5;  lanes[5].valid = v5;
    lanes[6].data = d6;  lanes[6].valid = v6;   lanes[7].data = d7;  lanes[7].valid = v7;
    lanes[8].data = d8;  lanes[8].valid = v8;   lanes[9].data = d9;  lanes[9].valid = v9;
    lanes[10].data = d10; lanes[10].valid = v10; lanes[11].data = d11; lanes[11].valid = v11;
    // stage D1: `any_idx` is uniform over the dispatch and picks the
    // instance below. The index SLOT is written into the thread's lane table
    // only when there is one, so a statement that reads no join set sets up
    // exactly the table it set up before stage D.
    uint any_idx = 0u;
    for (uint l = 0; l < u.n_lanes; ++l) {
        lanes[l].width = meta[l].width;
        lanes[l].has_valid = meta[l].has_valid;
        lanes[l].is_f64 = meta[l].is_f64;
        any_idx |= (meta[l].idx_slot != GL_NO_IDX) ? 1u : 0u;
    }
    if (any_idx) for (uint l = 0; l < u.n_lanes; ++l) lanes[l].idx_slot = meta[l].idx_slot;
    uint f = 0u;
    if (gid < u.n) {
        f = (any_idx ? gpred_eval<true>(lanes, prog, u.n_preds, lists, perm[gid])
                     : gpred_eval<false>(lanes, prog, u.n_preds, lists, perm[gid])) ? 1u : 0u;
        smask[gid] = (uchar)f;
    }
    shm[tid] = f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) shm[tid] += shm[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) block_counts[block_id] = shm[0];
}

// ===========================================================================
//  v0.8 — the direct, row-order grouped reduce (few distinct keys)
//
//  The exact GROUP BY is sort-based: a mask pass per predicate, run starts
//  over the key's sort cache, then a reduce that gathers every payload
//  through the permutation. When the key has few distinct values that whole
//  apparatus is paid to rediscover an answer with four rows in it. This path
//  reads a dense group id per row instead — one pass in storage order, the
//  WHERE evaluated by gpred_eval, every payload folded into acc[gid]: no
//  mask buffer, no run starts, no permutation, no gather.
//
//  The id lane (docs/RESIDENT_COLUMNS_DESIGN.md §7) is derived from the key column's sort cache once
//  (gdir_ids_i64 below) and lives with the column. Ids are the ranks of the
//  distinct valid keys ascending, so iterating them yields the operator's
//  output order; a NULL key takes the reserved id n_valid_groups, which is
//  the last group, exactly where the sort path puts it.
//
//  Accumulators are per thread, GAggAcc[n_groups * n_pays] with count(*) in
//  cstar[n_groups] beside them — the gagg layout with n_groups > 1. They sit
//  in thread space, so their size decides how many groups this path can
//  hold: one kernel is instantiated per capacity bucket and the host picks
//  the smallest that fits (and the sort path answers above the largest).
// ===========================================================================

// The dense group id of a row: 1 byte while the ids fit, else 2.
inline uint ldg(device const uchar* p, uint w, uint i) {
    return w == 1u ? (uint)p[i] : (uint)((device const ushort*)p)[i];
}
inline void stg(device uchar* p, uint w, uint i, uint v) {
    if (w == 1u) p[i] = (uchar)v; else ((device ushort*)p)[i] = (ushort)v;
}

// Every row's id at once: the sorted position's run index (the same
// block-offset + simd-prefix scan gb_run_starts_i64 does) scattered to row
// order through the permutation, and the run's key kept as that group's key.
// Rows with a NULL key are not in the cache and keep the fill value.
kernel void gdir_ids_i64(
    device const uchar* keys          [[buffer(0)]],   // sorted valid keys, width kw
    constant uint&      n             [[buffer(1)]],   // valid rows
    device const uint*  block_offsets [[buffer(2)]],
    device const uint*  perm          [[buffer(3)]],   // sorted pos -> row id
    device uchar*       ids           [[buffer(4)]],   // out: id per row, width gw
    device uchar*       dkeys         [[buffer(5)]],   // out: distinct keys, width kw
    constant uint&      kw            [[buffer(6)]],
    constant uint&      gw            [[buffer(7)]],
    uint                gid           [[thread_position_in_grid]],
    uint                block_id      [[threadgroup_position_in_grid]],
    uint                lane          [[thread_index_in_simdgroup]],
    uint                sg            [[simdgroup_index_in_threadgroup]])
{
    threadgroup uint sg_tot[BLOCK];
    uint f = 0u;
    if (gid < n) f = (gid == 0u || ldw(keys, kw, gid) != ldw(keys, kw, gid - 1u)) ? 1u : 0u;
    const uint lane_ex = simd_prefix_exclusive_sum(f);
    const uint sg_sum  = simd_sum(f);
    if (lane == 0) sg_tot[sg] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint sg_off = 0u;
    for (uint s = 0; s < sg; ++s) sg_off += sg_tot[s];
    if (gid >= n) return;
    // run starts at or before this position, less one == the run's index
    // (position 0 always starts a run, so the count is never zero)
    const uint id = block_offsets[block_id] + sg_off + lane_ex + f - 1u;
    stg(ids, gw, perm[gid], id);
    if (f) stw(dkeys, kw, id, ldw(keys, kw, gid));
}

// The NULL-key rows' id, written before the scatter (only when the key has
// NULLs): the reserved last group.
kernel void gdir_fill_ids(
    device uchar*  ids   [[buffer(0)]],
    constant uint& n     [[buffer(1)]],
    constant uint& gw    [[buffer(2)]],
    constant uint& value [[buffer(3)]],
    uint           i     [[thread_position_in_grid]])
{
    if (i < n) stg(ids, gw, i, value);
}

struct GDirU { uint n; uint n_preds; uint n_pays; uint n_groups; uint gw; uint p0; uint p1; uint p2; };

// One pass, one capacity. CAP bounds n_groups * n_pays (and n_groups, which
// cstar needs); the host never dispatches an instantiation that is too small.
template <uint CAP, bool IX>
void gdir_body(thread const GLane* lanes,
               device const GPred* prog, device const long* lists,
               device const uint* pay, constant GDirU& u,
               device const uchar* ids, device long* out,
               threadgroup long* scr,
               uint tid, uint tgid, uint ntg)
{
    GAggAcc acc[CAP];
    long    cstar[CAP];
    for (uint a = 0; a < CAP; ++a) { acc[a] = gagg_zero(); cstar[a] = 0l; }

    const uint gsize = BLOCK * ntg;
    for (uint i = tgid * BLOCK + tid; i < u.n; i += gsize) {
        if (!gpred_eval<IX>(lanes, prog, u.n_preds, lists, i)) continue;
        const uint g = ldg(ids, u.gw, i);
        cstar[g] += 1l;
        for (uint p = 0; p < u.n_pays; ++p) {
            const GLane ln = lanes[pay[p]];
            const uint lr = gl_row<IX>(lanes, ln, i);
            if ((IX && lr == GL_NULL_ROW) || !gbx_valid(ln.valid, ln.has_valid, (ulong)lr)) continue;
            gagg_add(acc[g * u.n_pays + p], ldw(ln.data, ln.width, lr));
        }
    }

    // threadgroup reduction, one accumulator at a time through `scr`
    const uint stride = 1u + 5u * u.n_pays;
    for (uint g = 0; g < u.n_groups; ++g) {
        const uint base = (tgid * u.n_groups + g) * stride;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        scr[tid * 5u] = cstar[g];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = BLOCK / 2u; s > 0u; s >>= 1) {
            if (tid < s) scr[tid * 5u] += scr[(tid + s) * 5u];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0u) out[base] = scr[0];
        for (uint p = 0; p < u.n_pays; ++p) {
            const GAggAcc a = acc[g * u.n_pays + p];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            scr[tid * 5u + 0u] = (long)a.lo; scr[tid * 5u + 1u] = a.hi; scr[tid * 5u + 2u] = a.cnt;
            scr[tid * 5u + 3u] = a.mn;       scr[tid * 5u + 4u] = a.mx;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = BLOCK / 2u; s > 0u; s >>= 1) {
                if (tid < s) {
                    const uint x = tid * 5u, y = (tid + s) * 5u;
                    const ulong old = (ulong)scr[x];
                    scr[x] = (long)(old + (ulong)scr[y]);
                    scr[x + 1u] += scr[y + 1u] + (((ulong)scr[x] < old) ? 1l : 0l);
                    scr[x + 2u] += scr[y + 2u];
                    scr[x + 3u] = min(scr[x + 3u], scr[y + 3u]);
                    scr[x + 4u] = max(scr[x + 4u], scr[y + 4u]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tid == 0u) {
                const uint o = base + 1u + 5u * p;
                out[o + 0u] = scr[0]; out[o + 1u] = scr[1]; out[o + 2u] = scr[2];
                out[o + 3u] = scr[3]; out[o + 4u] = scr[4];
            }
        }
    }
}

// The lane bindings are gagg_masked_i64's, plus the id lane at 30.
#define GDIR_KERNEL(NAME, CAP)                                                        \
kernel void NAME(                                                                     \
    device const uchar*     d0    [[buffer(0)]],  device const ulong* v0  [[buffer(1)]],  \
    device const uchar*     d1    [[buffer(2)]],  device const ulong* v1  [[buffer(3)]],  \
    device const uchar*     d2    [[buffer(4)]],  device const ulong* v2  [[buffer(5)]],  \
    device const uchar*     d3    [[buffer(6)]],  device const ulong* v3  [[buffer(7)]],  \
    device const uchar*     d4    [[buffer(8)]],  device const ulong* v4  [[buffer(9)]],  \
    device const uchar*     d5    [[buffer(10)]], device const ulong* v5  [[buffer(11)]], \
    device const uchar*     d6    [[buffer(12)]], device const ulong* v6  [[buffer(13)]], \
    device const uchar*     d7    [[buffer(14)]], device const ulong* v7  [[buffer(15)]], \
    device const uchar*     d8    [[buffer(16)]], device const ulong* v8  [[buffer(17)]], \
    device const uchar*     d9    [[buffer(18)]], device const ulong* v9  [[buffer(19)]], \
    device const uchar*     d10   [[buffer(20)]], device const ulong* v10 [[buffer(21)]], \
    device const uchar*     d11   [[buffer(22)]], device const ulong* v11 [[buffer(23)]], \
    device const GLaneMeta* meta  [[buffer(24)]],                                      \
    device const GPred*     prog  [[buffer(25)]],                                      \
    device const long*      lists [[buffer(26)]],                                      \
    device const uint*      pay   [[buffer(27)]],                                      \
    constant GDirU&         u     [[buffer(28)]],                                      \
    device long*            out   [[buffer(29)]],                                      \
    device const uchar*     ids   [[buffer(30)]],                                      \
    uint tid  [[thread_position_in_threadgroup]],                                      \
    uint tgid [[threadgroup_position_in_grid]],                                        \
    uint ntg  [[threadgroups_per_grid]])                                               \
{                                                                                      \
    threadgroup long scr[BLOCK * 5];                                                   \
    GLane lanes[GAGG_MAX_LANES];                                                       \
    lanes[0].data = d0;  lanes[0].valid = v0;   lanes[1].data = d1;  lanes[1].valid = v1;   \
    lanes[2].data = d2;  lanes[2].valid = v2;   lanes[3].data = d3;  lanes[3].valid = v3;   \
    lanes[4].data = d4;  lanes[4].valid = v4;   lanes[5].data = d5;  lanes[5].valid = v5;   \
    lanes[6].data = d6;  lanes[6].valid = v6;   lanes[7].data = d7;  lanes[7].valid = v7;   \
    lanes[8].data = d8;  lanes[8].valid = v8;   lanes[9].data = d9;  lanes[9].valid = v9;   \
    lanes[10].data = d10; lanes[10].valid = v10; lanes[11].data = d11; lanes[11].valid = v11; \
    uint any_idx = 0u;                              /* stage D1, see gl_row */         \
    for (uint l = 0; l < GAGG_MAX_LANES; ++l) {                                        \
        lanes[l].width = meta[l].width;                                                \
        lanes[l].has_valid = meta[l].has_valid;                                        \
        lanes[l].is_f64 = meta[l].is_f64;                                              \
        any_idx |= (meta[l].idx_slot != GL_NO_IDX) ? 1u : 0u;                          \
    }                                                                                  \
    if (any_idx)                                                                       \
        for (uint l = 0; l < GAGG_MAX_LANES; ++l) lanes[l].idx_slot = meta[l].idx_slot;\
    if (any_idx) gdir_body<CAP, true>(lanes, prog, lists, pay, u, ids, out, scr, tid, tgid, ntg); \
    else         gdir_body<CAP, false>(lanes, prog, lists, pay, u, ids, out, scr, tid, tgid, ntg);\
}

GDIR_KERNEL(gdir_masked_8_i64,   8u)
GDIR_KERNEL(gdir_masked_32_i64,  32u)

// ---------------------------------------------------------------------------
//  The same pass with the accumulators in THREADGROUP memory.
//
//  Thread-private accumulators are the fastest thing there is while they stay
//  in a handful of slots and fall off a cliff after that: measured at SF10,
//  4.6 ms at 2 groups x 1 payload, 106 ms at 4 x 5 and 215 ms at 25 x 3 —
//  the array is per thread, so 256 threads times the slots is the working set
//  and no cache holds it. This variant keeps ONE slab per threadgroup and
//  lets the threads share it.
//
//  Apple GPUs have no 64-bit atomics of any kind (device or threadgroup) and
//  no 64-bit simd reductions, so everything here is 32-bit:
//    - count(*) and count(v) are u32 adds (a threadgroup's count cannot
//      exceed the row count, which is below 2^32 by the operator's own cap);
//    - the 128-bit sum is four u32 limbs with the carries propagated by hand:
//      an add returns the old value, so each thread knows whether its own add
//      wrapped and owes 2^32 to the next limb. Negative values are added as
//      their unsigned image and counted, and the count is subtracted at
//      2^64 when the limbs are assembled — which is exact, because the true
//      sum fits in 128 bits;
//    - min and max are u32 atomics over the order-preserving image of a
//      32-bit value, so they are exact for a payload lane stored at 4 bytes
//      or narrower (stage C stores a lane at the narrowest width its values
//      fit, so this is every lane whose values fit int32). A wider lane with
//      min or max asked for takes the sort path.
//
//  Contention is what would kill this with four groups and 256 threads, so
//  the slab is REPLICATED: copy = tid % ncopy, with ncopy a power of two up
//  to 32 chosen by the host from the threadgroup memory a copy costs. At 32
//  copies the 32 lanes of a simdgroup never touch the same word.
// ---------------------------------------------------------------------------

struct GDirSlabU { uint n; uint n_preds; uint n_pays; uint n_groups; uint gw; uint ncopy; uint minmax; uint pad; };

kernel void gdir_slab_i64(
    device const uchar*     d0    [[buffer(0)]],  device const ulong* v0  [[buffer(1)]],
    device const uchar*     d1    [[buffer(2)]],  device const ulong* v1  [[buffer(3)]],
    device const uchar*     d2    [[buffer(4)]],  device const ulong* v2  [[buffer(5)]],
    device const uchar*     d3    [[buffer(6)]],  device const ulong* v3  [[buffer(7)]],
    device const uchar*     d4    [[buffer(8)]],  device const ulong* v4  [[buffer(9)]],
    device const uchar*     d5    [[buffer(10)]], device const ulong* v5  [[buffer(11)]],
    device const uchar*     d6    [[buffer(12)]], device const ulong* v6  [[buffer(13)]],
    device const uchar*     d7    [[buffer(14)]], device const ulong* v7  [[buffer(15)]],
    device const uchar*     d8    [[buffer(16)]], device const ulong* v8  [[buffer(17)]],
    device const uchar*     d9    [[buffer(18)]], device const ulong* v9  [[buffer(19)]],
    device const uchar*     d10   [[buffer(20)]], device const ulong* v10 [[buffer(21)]],
    device const uchar*     d11   [[buffer(22)]], device const ulong* v11 [[buffer(23)]],
    device const GLaneMeta* meta  [[buffer(24)]],
    device const GPred*     prog  [[buffer(25)]],
    device const long*      lists [[buffer(26)]],
    device const uint*      pay   [[buffer(27)]],
    constant GDirSlabU&     u     [[buffer(28)]],
    device long*            out   [[buffer(29)]],
    device const uchar*     ids   [[buffer(30)]],
    threadgroup atomic_uint* slab [[threadgroup(0)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint ntg  [[threadgroups_per_grid]])
{
    GLane lanes[GAGG_MAX_LANES];
    lanes[0].data = d0;  lanes[0].valid = v0;   lanes[1].data = d1;  lanes[1].valid = v1;
    lanes[2].data = d2;  lanes[2].valid = v2;   lanes[3].data = d3;  lanes[3].valid = v3;
    lanes[4].data = d4;  lanes[4].valid = v4;   lanes[5].data = d5;  lanes[5].valid = v5;
    lanes[6].data = d6;  lanes[6].valid = v6;   lanes[7].data = d7;  lanes[7].valid = v7;
    lanes[8].data = d8;  lanes[8].valid = v8;   lanes[9].data = d9;  lanes[9].valid = v9;
    lanes[10].data = d10; lanes[10].valid = v10; lanes[11].data = d11; lanes[11].valid = v11;
    // stage D1: `any_idx` is uniform over the dispatch and picks the
    // instance below. The index SLOT is written into the thread's lane table
    // only when there is one, so a statement that reads no join set sets up
    // exactly the table it set up before stage D.
    uint any_idx = 0u;
    for (uint l = 0; l < GAGG_MAX_LANES; ++l) {
        lanes[l].width = meta[l].width;
        lanes[l].has_valid = meta[l].has_valid;
        lanes[l].is_f64 = meta[l].is_f64;
        any_idx |= (meta[l].idx_slot != GL_NO_IDX) ? 1u : 0u;
    }
    if (any_idx) for (uint l = 0; l < GAGG_MAX_LANES; ++l) lanes[l].idx_slot = meta[l].idx_slot;

    const uint R  = u.ncopy;
    const uint c  = tid & (R - 1u);
    const uint P6 = u.n_groups * u.n_pays * 6u * R;     // sums: 4 limbs, negatives, count(v)
    const uint CB = P6 + u.n_groups * R;                // count(*) per group
    const uint MB = CB;                                 // min / max, when asked for
    const uint total = CB + (u.minmax ? u.n_groups * u.n_pays * 2u * R : 0u);
    for (uint i = tid; i < total; i += BLOCK) {
        uint init = 0u;
        if (u.minmax && i >= MB) init = (((i - MB) / R) & 1u) ? 0u : 0xFFFFFFFFu;   // max, then min
        atomic_store_explicit(&slab[i], init, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage D1: written once, compiled twice, chosen by a flag that is the
    // same for every thread — see gl_row.
#define GDIR_SLAB_FOLD(IX) \
    for (uint i = tgid * BLOCK + tid; i < u.n; i += gsize) {                                            \
        if (!gpred_eval<IX>(lanes, prog, u.n_preds, lists, i)) continue;                                \
        const uint g = ldg(ids, u.gw, i);                                                               \
        atomic_fetch_add_explicit(&slab[P6 + g * R + c], 1u, memory_order_relaxed);                     \
        for (uint p = 0; p < u.n_pays; ++p) {                                                           \
            const GLane ln = lanes[pay[p]];                                                             \
            const uint lr = gl_row<IX>(lanes, ln, i);                                                   \
            if ((IX && lr == GL_NULL_ROW) || !gbx_valid(ln.valid, ln.has_valid, (ulong)lr)) continue;   \
            const long  v  = ldw(ln.data, ln.width, lr);                                                \
            const ulong uv = (ulong)v;                                                                  \
            const uint  s  = g * u.n_pays + p;                                                          \
            const uint  b  = (s * 6u) * R + c;                                                          \
            const uint a0 = (uint)(uv & 0xFFFFFFFFul), a1 = (uint)(uv >> 32);                           \
            uint old = atomic_fetch_add_explicit(&slab[b], a0, memory_order_relaxed);                   \
            uint carry = (old + a0 < old) ? 1u : 0u;                                                    \
            old = atomic_fetch_add_explicit(&slab[b + R], a1, memory_order_relaxed);                    \
            uint carry2 = (old + a1 < old) ? 1u : 0u;                                                   \
            if (carry) {                                                                                \
                old = atomic_fetch_add_explicit(&slab[b + R], 1u, memory_order_relaxed);                \
                if (old + 1u < old) carry2 += 1u;                                                       \
            }                                                                                           \
            if (carry2) {                                                                               \
                old = atomic_fetch_add_explicit(&slab[b + 2u * R], carry2, memory_order_relaxed);       \
                if (old + carry2 < old)                                                                 \
                    atomic_fetch_add_explicit(&slab[b + 3u * R], 1u, memory_order_relaxed);             \
            }                                                                                           \
            if (v < 0l) atomic_fetch_add_explicit(&slab[b + 4u * R], 1u, memory_order_relaxed);         \
            atomic_fetch_add_explicit(&slab[b + 5u * R], 1u, memory_order_relaxed);                     \
            if (u.minmax) {                                                                             \
                const uint uo = (uint)((int)v) ^ 0x80000000u; /*  order-preserving, 32-bit lane */      \
                const uint mb = MB + (s * 2u) * R + c;                                                  \
                atomic_fetch_min_explicit(&slab[mb], uo, memory_order_relaxed);                         \
                atomic_fetch_max_explicit(&slab[mb + R], uo, memory_order_relaxed);                     \
            }                                                                                           \
        }                                                                                               \
    }                                                                                                  
    const uint gsize = BLOCK * ntg;
    if (any_idx) { GDIR_SLAB_FOLD(true) } else { GDIR_SLAB_FOLD(false) }
#undef GDIR_SLAB_FOLD
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Fold the copies into this threadgroup's block; gdir_merge_i64 folds the
    // blocks. One thread per (group, payload), one per group for count(*).
    const uint stride = 1u + 5u * u.n_pays;
    const uint per_g = 1u + u.n_pays;
    for (uint w = tid; w < u.n_groups * per_g; w += BLOCK) {
        const uint g = w / per_g, j = w - g * per_g;
        const uint base = (tgid * u.n_groups + g) * stride;
        if (j == 0u) {
            ulong cs = 0ul;
            for (uint cc = 0; cc < R; ++cc)
                cs += (ulong)atomic_load_explicit(&slab[P6 + g * R + cc], memory_order_relaxed);
            out[base] = (long)cs;
            continue;
        }
        const uint p = j - 1u, s = g * u.n_pays + p, b = (s * 6u) * R;
        ulong lo = 0ul; long hi = 0l, cnt = 0l;
        uint mn = 0xFFFFFFFFu, mx = 0u;
        for (uint cc = 0; cc < R; ++cc) {
            const uint l0 = atomic_load_explicit(&slab[b + 0u * R + cc], memory_order_relaxed);
            const uint l1 = atomic_load_explicit(&slab[b + 1u * R + cc], memory_order_relaxed);
            const uint l2 = atomic_load_explicit(&slab[b + 2u * R + cc], memory_order_relaxed);
            const uint l3 = atomic_load_explicit(&slab[b + 3u * R + cc], memory_order_relaxed);
            const uint ng = atomic_load_explicit(&slab[b + 4u * R + cc], memory_order_relaxed);
            const uint cn = atomic_load_explicit(&slab[b + 5u * R + cc], memory_order_relaxed);
            const ulong plo = ((ulong)l1 << 32) | (ulong)l0;
            const ulong phi = (((ulong)l3 << 32) | (ulong)l2) - (ulong)ng;   // the negatives owe 2^64 each
            const ulong prev = lo;
            lo += plo;
            hi += (long)phi + ((lo < prev) ? 1l : 0l);
            cnt += (long)cn;
            if (u.minmax) {
                mn = min(mn, atomic_load_explicit(&slab[MB + (s * 2u) * R + cc], memory_order_relaxed));
                mx = max(mx, atomic_load_explicit(&slab[MB + (s * 2u + 1u) * R + cc], memory_order_relaxed));
            }
        }
        const uint o = base + 1u + 5u * p;
        out[o + 0u] = (long)lo; out[o + 1u] = hi; out[o + 2u] = cnt;
        out[o + 3u] = (u.minmax && cnt) ? (long)(int)(mn ^ 0x80000000u) : GBX_LMAX;
        out[o + 4u] = (u.minmax && cnt) ? (long)(int)(mx ^ 0x80000000u) : GBX_LMIN;
    }
}

struct GDirMergeU { uint ntg; uint n_groups; uint n_pays; uint pad; };

// The per-(threadgroup, group) blocks folded into one block per group, on
// the device: at 512 groups x 5 payloads the host would otherwise walk
// millions of longs. One thread per (group, payload), one per group for
// count(*).
kernel void gdir_merge_i64(
    device const long*      part [[buffer(0)]],
    constant GDirMergeU&    u    [[buffer(1)]],
    device long*            res  [[buffer(2)]],
    uint                    i    [[thread_position_in_grid]])
{
    const uint per_g = 1u + u.n_pays;
    if (i >= u.n_groups * per_g) return;
    const uint g = i / per_g, j = i - g * per_g;
    const uint stride = 1u + 5u * u.n_pays;
    if (j == 0u) {
        long c = 0l;
        for (uint b = 0; b < u.ntg; ++b) c += part[(b * u.n_groups + g) * stride];
        res[g * stride] = c;
        return;
    }
    const uint p = j - 1u;
    GbxAcc a = gbx_zero();
    for (uint b = 0; b < u.ntg; ++b) {
        const uint o = (b * u.n_groups + g) * stride + 1u + 5u * p;
        GbxAcc t;
        t.lo = (ulong)part[o]; t.hi = part[o + 1u]; t.cnt = part[o + 2u];
        t.mn = part[o + 3u];   t.mx = part[o + 4u];
        gbx_merge(a, t);
    }
    const uint o = g * stride + 1u + 5u * p;
    res[o + 0u] = (long)a.lo; res[o + 1u] = a.hi; res[o + 2u] = a.cnt;
    res[o + 3u] = a.mn;       res[o + 4u] = a.mx;
}
