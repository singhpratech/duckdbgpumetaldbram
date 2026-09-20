// exact_kernel.cu — the device side of the whole v0.7 exact path: the exact
// GROUP BY (§4.1/§4.2), its WHERE mask (§4.6), the global masked aggregate
// (§4.12) and the materialised key join (§4.8).
//
// Every entry point declared in exact_api.h is implemented here, and every one
// of them mirrors a line of the CPU reference (cpu_aggregator.cpp
// exact_impl_indexed). Where the two could drift, the comment says which line
// of the reference the code is reproducing — the exact path's whole point is
// that the answer does not depend on which backend produced it (rule 2).
//
// Three things are worth knowing before reading:
//
//   * The 128-bit sum is a plain {lo, hi} carry add. It is arithmetic mod
//     2^128, hence associative and commutative, so CUB may reduce in any order
//     and any block count and still land on the CPU's bits exactly. This is
//     why the sum needs no determinism flag and no fixed partitioning.
//   * A NULL cell contributes the IDENTITY, never a zero value. Zero would be
//     wrong for min/max (it would drag a min of positive values down to 0) and
//     for count(v). The identity carries mn = INT64_MAX / mx = INT64_MIN, and
//     the finalize step turns an empty payload set into the contract's 0.
//   * The keys-only form (has_vals == 0) reproduces the reference literally,
//     INT64_MAX/INT64_MIN mins and all: the reference sets cnt_v = cnt_star
//     without ever touching mn/mx, so a keys-only group reports those
//     sentinels. The contract calls min/max unspecified there; "unspecified"
//     still has to be the SAME unspecified on both backends, or a cross-engine
//     comparison in the test suite fails for a reason nobody wants to debug.
#include "exact_api.h"

#include "cub_ops.cuh"

#include <cstdint>

// CCCL 3 (CUDA 13) removed cub::CountingInputIterator / TransformInputIterator;
// the thrust ones are the supported spelling and are what the other kernels in
// this directory already use.
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace {

using gpudb_cuda_ops::DevBuf;
using gpudb_cuda_ops::with_temp;
using gpudb_cuda_ops::grid_for;
using gpudb_cuda_ops::kBlock;

using u64 = unsigned long long;
using i64 = std::int64_t;

constexpr i64 kI64Max =  0x7fffffffffffffffLL;
constexpr i64 kI64Min = -0x7fffffffffffffffLL - 1;

// Predicate::Op's underlying values, in declaration order. Kept as named
// constants so a reordering of the enum breaks here loudly rather than
// silently turning every LT into an EQ.
enum : int { kEQ = 0, kNE = 1, kLT = 2, kLE = 3, kGT = 4, kGE = 5,
             kIsNull = 6, kIsNotNull = 7, kIn = 8 };

// ---- narrow lane storage (docs/RESIDENT_COLUMNS_DESIGN.md stage C) ----
// A lane's storage width is backend-private: the interface still says I64,
// and every value read out of one is an i64. `w` is 1, 2, 4 or 8 bytes and is
// the same for every thread of a launch, so the switch is warp-uniform and
// costs a predicted branch rather than divergence.
//
// A uniform was chosen over templating each kernel on the width because
// several kernels read two or three lanes at once (a key, a payload, a
// predicate), and templating would need 4^2 or 4^3 instantiations of each.
__host__ __device__ __forceinline__ i64 ldw(const void* __restrict__ p, int w, std::size_t i) {
    switch (w) {
        case 1: return static_cast<const std::int8_t*>(p)[i];
        case 2: return static_cast<const std::int16_t*>(p)[i];
        case 4: return static_cast<const std::int32_t*>(p)[i];
        default: return static_cast<const i64*>(p)[i];
    }
}
__host__ __device__ __forceinline__ void stw(void* __restrict__ p, int w, std::size_t i, i64 v) {
    switch (w) {
        case 1: static_cast<std::int8_t*>(p)[i]  = static_cast<std::int8_t>(v);  break;
        case 2: static_cast<std::int16_t*>(p)[i] = static_cast<std::int16_t>(v); break;
        case 4: static_cast<std::int32_t*>(p)[i] = static_cast<std::int32_t>(v); break;
        default: static_cast<i64*>(p)[i] = v; break;
    }
}

// Stage C, second half: the sort cache is narrowed too. The sorted keys keep
// the LANE's width (they are the same values, reordered) and the permutation
// holds u32 row ids — the cache already refuses a column above 2^32 rows, so
// eight bytes a row id was always more than it needed. For a 6M-row lane of
// width 2 that is 6 bytes a row instead of 16.
struct LoadKey {
    const void* p;
    int         w;
    __host__ __device__ __forceinline__ i64 operator()(std::size_t i) const {
        return ldw(p, w, i);
    }
};

// Row i is valid iff bit i % 64 of word i / 64 is set; a null bitmap means
// every row is valid (DuckDB's layout, and the CPU reference's `bit` lambda).
__host__ __device__ __forceinline__ bool bit_at(const u64* m, std::size_t i) {
    return !m || ((m[i >> 6] >> (i & 63)) & 1ull);
}

// Signed i64 -> order-preserving u64, for the radix sort (which is unsigned).
__host__ __device__ __forceinline__ u64 key_to_u64(i64 x) {
    return static_cast<u64>(x) ^ (1ull << 63);
}
__host__ __device__ __forceinline__ i64 u64_to_key(u64 u) {
    return static_cast<i64>(u ^ (1ull << 63));
}

// DuckDB's total order on doubles, from the IEEE-754 bits: every NaN is the
// greatest value and equal to any other NaN, -0.0 == 0.0. Bit-for-bit the
// host's f64_total_order_key (predicate_mask.hpp) — written on the bits so it
// needs no isnan() and no fp instruction, which also keeps it exact under
// whatever fast-math flags the TU is built with.
__host__ __device__ __forceinline__ u64 f64_order_key_bits(i64 bits) {
    u64 u = static_cast<u64>(bits);
    const u64 exp = u & 0x7ff0000000000000ull;
    const u64 man = u & 0x000fffffffffffffull;
    if (exp == 0x7ff0000000000000ull && man != 0ull) return ~0ull;   // NaN: one value, greatest
    if ((u << 1) == 0ull) u = 0ull;                                   // -0.0 -> +0.0
    return (u & (1ull << 63)) ? ~u : (u | (1ull << 63));
}

template <class T>
__host__ __device__ __forceinline__ bool dev_cmp(int op, T a, T b) {
    switch (op) {
        case kEQ: return a == b;
        case kNE: return a != b;
        case kLT: return a <  b;
        case kLE: return a <= b;
        case kGT: return a >  b;
        case kGE: return a >= b;
        default:  return false;
    }
}

// ---- the per-group tuple and its monoid ----------------------------------
struct ETup {
    u64 lo;
    i64 hi;
    i64 cnt_v;
    i64 cnt_star;
    i64 mn;
    i64 mx;
};

__host__ __device__ __forceinline__ ETup etup_identity() {
    ETup t;
    t.lo = 0ull; t.hi = 0; t.cnt_v = 0; t.cnt_star = 0;
    t.mn = kI64Max; t.mx = kI64Min;
    return t;
}

// The reduction. `lo` wraps by design; the carry out of it is the +1 into
// `hi`, which is exactly Sum128::add's `(lo < old ? 1 : 0)`.
struct AddExact {
    __host__ __device__ __forceinline__ ETup operator()(const ETup& a, const ETup& b) const {
        ETup r;
        r.lo       = a.lo + b.lo;
        r.hi       = a.hi + b.hi + static_cast<i64>(r.lo < a.lo ? 1 : 0);
        r.cnt_v    = a.cnt_v + b.cnt_v;
        r.cnt_star = a.cnt_star + b.cnt_star;
        r.mn       = a.mn < b.mn ? a.mn : b.mn;
        r.mx       = a.mx > b.mx ? a.mx : b.mx;
        return r;
    }
};

// One row's contribution, sign-extended into 128 bits exactly as
// Sum128::from_i64 does.
__host__ __device__ __forceinline__ ETup etup_of_value(i64 x) {
    ETup t;
    t.lo = static_cast<u64>(x);
    t.hi = x < 0 ? -1 : 0;
    t.cnt_v = 1; t.cnt_star = 1;
    t.mn = x; t.mx = x;
    return t;
}

// Sorted position i -> its tuple. `perm[i]` is the original row, which is what
// the payload and its bitmap are indexed by (the exact path keeps every column
// in input order; only the key's permutation moves).
struct RowTuple {
    const std::uint32_t* perm;
    const void* vals;
    const u64*  vvalid;
    int         has_vals;
    int         vwidth;
    __host__ __device__ __forceinline__ ETup operator()(std::size_t i) const {
        ETup t = etup_identity();
        t.cnt_star = 1;
        if (!has_vals) { t.cnt_v = 1; return t; }   // reference: cnt_v = cnt_star, mn/mx untouched
        const std::size_t row = perm[i];
        if (!bit_at(vvalid, row)) return t;         // NULL payload: counts in count(*) only
        return etup_of_value(ldw(vals, vwidth, row));
    }
};

// The NULL-key group (§4.1): every row whose KEY is NULL and which passes the
// mask, reduced over the whole column in storage order — those rows are not in
// the sort cache at all, since the cache covers the valid keys.
struct NullKeyTuple {
    const u64*           kvalid;
    const unsigned char* mask;
    const void*          vals;
    const u64*           vvalid;
    int                  has_vals;
    int                  vwidth;
    __host__ __device__ __forceinline__ ETup operator()(std::size_t i) const {
        ETup t = etup_identity();
        if (bit_at(kvalid, i)) return t;            // key is present: a different group
        if (mask && !mask[i]) return t;             // failed the WHERE: absent from count(*) too
        t.cnt_star = 1;
        if (!has_vals) { t.cnt_v = 1; return t; }
        if (!bit_at(vvalid, i)) return t;
        return etup_of_value(ldw(vals, vwidth, i));
    }
};

// §4.12: one row's contribution to the global aggregate. No key, no
// permutation — the row index IS the row, which is the whole point of the
// operator: it exists so a keyless aggregate does not pay for a sort it has no
// use for.
struct GlobalTuple {
    const unsigned char* mask;
    const void*          vals;
    const u64*           vvalid;
    int                  vwidth;
    __host__ __device__ __forceinline__ ETup operator()(std::size_t i) const {
        ETup t = etup_identity();
        if (mask && !mask[i]) return t;
        t.cnt_star = 1;
        if (!bit_at(vvalid, i)) return t;   // NULL payload: in count(*) only
        return etup_of_value(ldw(vals, vwidth, i));
    }
};

// Surviving rows when there is no payload to carry the count for us.
struct MaskOne {
    const unsigned char* mask;
    __host__ __device__ __forceinline__ u64 operator()(std::size_t i) const {
        return (!mask || mask[i]) ? 1ull : 0ull;
    }
};

// Stage C: the range of a lane, as a monoid CUB can reduce.
struct MinMax { i64 mn; i64 mx; };
struct MinMaxOp {
    __host__ __device__ __forceinline__ MinMax operator()(const MinMax& a, const MinMax& b) const {
        MinMax r;
        r.mn = a.mn < b.mn ? a.mn : b.mn;
        r.mx = a.mx > b.mx ? a.mx : b.mx;
        return r;
    }
};
struct LoadMinMax {
    const i64* vals;
    __host__ __device__ __forceinline__ MinMax operator()(std::size_t i) const {
        MinMax r; r.mn = vals[i]; r.mx = vals[i]; return r;
    }
};

__global__ void lane_pack_kernel(const i64* __restrict__ src, std::size_t rows,
                                 void* __restrict__ dst, int width) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < rows; i += stride) stw(dst, width, i, src[i]);
}

// Valid bits of word w, with the bits past `rows` masked off — the tail word
// is filled with ones at upload, so counting it whole would under-report NULLs.
struct PopValid {
    const u64*  bits;
    std::size_t rows;
    __host__ __device__ __forceinline__ u64 operator()(std::size_t w) const {
        u64 word = bits[w];
        const std::size_t left = rows - w * 64;
        if (left < 64) word &= (1ull << left) - 1ull;
#ifdef __CUDA_ARCH__
        return static_cast<u64>(__popcll(word));
#else
        u64 c = 0;
        while (word) { word &= word - 1ull; ++c; }
        return c;
#endif
    }
};

struct IsRunStart {
    const void* sorted;
    int         w;
    __host__ __device__ __forceinline__ u64 operator()(std::size_t i) const {
        return (i == 0 || ldw(sorted, w, i) != ldw(sorted, w, i - 1)) ? 1ull : 0ull;
    }
};

// A narrow lane read as the i64 it represents, for the CUB passes that want a
// plain sequence of keys rather than the storage behind them.
struct ReadKey {
    const void* a;
    int         w;
    __host__ __device__ __forceinline__ i64 operator()(std::size_t i) const {
        return ldw(a, w, i);
    }
};

// ---- element-wise kernels -------------------------------------------------

__global__ void fill_valid_kernel(u64* __restrict__ bits, std::size_t words) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < words; i += stride) bits[i] = ~0ull;
}

// A NULL cell stores 0 and clears its bit. The clear is atomic because two
// rows of one span — and two different spans landing at different dst_row —
// can share a 64-bit word of the destination bitmap.
__device__ __forceinline__ void put_cell(bool ok, i64 value, std::size_t d,
                                         void* dst, int width, u64* dst_valid) {
    stw(dst, width, d, ok ? value : 0);
    if (!ok && dst_valid) atomicAnd(&dst_valid[d >> 6], ~(1ull << (d & 63)));
}

__global__ void scatter_lane_kernel(const i64* __restrict__ src, std::size_t rows,
                                    std::size_t n_lanes, std::size_t lane,
                                    const u64* __restrict__ src_valid, std::size_t valid_bit,
                                    void* __restrict__ dst, int dst_width,
                                    u64* __restrict__ dst_valid, std::size_t dst_row) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t j = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         j < rows; j += stride) {
        const bool ok = bit_at(src_valid, valid_bit + j);
        put_cell(ok, src[j * n_lanes + lane], dst_row + j, dst, dst_width, dst_valid);
    }
}

__global__ void scatter_pair_kernel(const i64* __restrict__ kv, std::size_t rows,
                                    const u64* __restrict__ key_valid,
                                    const u64* __restrict__ val_valid, std::size_t valid_bit,
                                    void* __restrict__ keys, int key_width,
                                    void* __restrict__ vals, int val_width,
                                    u64* __restrict__ key_bits, u64* __restrict__ val_bits,
                                    std::size_t dst_row) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t j = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         j < rows; j += stride) {
        const std::size_t d = dst_row + j;
        put_cell(bit_at(key_valid, valid_bit + j), kv[2 * j],     d, keys, key_width, key_bits);
        put_cell(bit_at(val_valid, valid_bit + j), kv[2 * j + 1], d, vals, val_width, val_bits);
    }
}

// The WHERE mask (§4.6) in ONE fused pass: a row reads predicate p only if it
// passed predicates 0..p-1, so a selective leading term spares the rest the
// memory traffic. The CPU reference's `if (!mask[i]) continue;` is the same
// short-circuit, one predicate-pass later.
__global__ void mask_kernel(const gpudb::cuda_exact::DevPred* __restrict__ preds, int n_preds,
                            std::size_t rows, unsigned char* __restrict__ mask) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < rows; i += stride) {
        bool ok = true;
        for (int p = 0; p < n_preds && ok; ++p) {
            const gpudb::cuda_exact::DevPred q = preds[p];
            const bool v = bit_at(q.valid, i);
            if (q.op == kIsNull)    { ok = !v; continue; }
            if (q.op == kIsNotNull) { ok =  v; continue; }
            if (!v) { ok = false; break; }            // NULL fails every comparison and In
            const i64 raw = ldw(q.data, q.width, i);
            if (q.is_f64) {
                const u64 a = f64_order_key_bits(raw);
                if (q.op == kIn) {
                    ok = false;
                    for (int j = 0; j < q.n_list; ++j)
                        if (a == f64_order_key_bits(q.list[j])) { ok = true; break; }
                } else {
                    ok = dev_cmp<u64>(q.op, a, f64_order_key_bits(q.value));
                }
            } else {
                if (q.op == kIn) {
                    ok = false;
                    for (int j = 0; j < q.n_list; ++j)
                        if (raw == q.list[j]) { ok = true; break; }
                } else {
                    ok = dev_cmp<i64>(q.op, raw, q.value);
                }
            }
        }
        mask[i] = ok ? 1u : 0u;
    }
}

__global__ void flags_from_bitmap_kernel(const u64* __restrict__ valid, std::size_t rows,
                                         unsigned char* __restrict__ flags) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < rows; i += stride) flags[i] = bit_at(valid, i) ? 1u : 0u;
}

// flags[i] = mask[perm[i]] — the mask is indexed by ROW, the sorted array by
// position, and this is the one gather that reconciles them. Doing it once
// here is what keeps the row loop free of an indirection per predicate.
__global__ void flags_from_mask_kernel(const std::uint32_t* __restrict__ perm, std::size_t n,
                                       const unsigned char* __restrict__ mask,
                                       unsigned char* __restrict__ flags) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride) flags[i] = mask[perm[i]];
}

__global__ void gather_order_keys_kernel(const void* __restrict__ keys, int kwidth,
                                         const i64* __restrict__ perm,
                                         std::size_t n, u64* __restrict__ out) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride) out[i] = key_to_u64(ldw(keys, kwidth, perm[i]));
}

__global__ void unmap_order_keys_kernel(const u64* __restrict__ in, std::size_t n,
                                        void* __restrict__ out, int w,
                                        const i64* __restrict__ perm_i64,
                                        std::uint32_t* __restrict__ perm_u32) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        stw(out, w, i, u64_to_key(in[i]));
        perm_u32[i] = static_cast<std::uint32_t>(perm_i64[i]);
    }
}

// Keep the k-th selected position of the cache: both arrays move together, so
// selecting POSITIONS once and gathering beats selecting each array.
__global__ void gather_selected_kernel(const void* __restrict__ sorted, int w,
                                       const std::uint32_t* __restrict__ perm,
                                       const std::uint32_t* __restrict__ pick, std::size_t n_sel,
                                       void* __restrict__ sorted_out,
                                       std::uint32_t* __restrict__ perm_out) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t k = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         k < n_sel; k += stride) {
        const std::size_t i = pick[k];
        stw(sorted_out, w, k, ldw(sorted, w, i));
        perm_out[k] = perm[i];
    }
}

// Split the reduced tuples into the six result vectors, applying the
// contract's "a group with counts == 0 has min/max 0" rule.
__global__ void finalize_kernel(const ETup* __restrict__ t, std::size_t n,
                                i64* __restrict__ lo, i64* __restrict__ hi,
                                i64* __restrict__ cnt_v, i64* __restrict__ cnt_star,
                                i64* __restrict__ mn, i64* __restrict__ mx) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        const ETup e = t[i];
        lo[i]       = static_cast<i64>(e.lo);
        hi[i]       = e.hi;
        cnt_v[i]    = e.cnt_v;
        cnt_star[i] = e.cnt_star;
        mn[i]       = e.cnt_v ? e.mn : 0;
        mx[i]       = e.cnt_v ? e.mx : 0;
    }
}

// ---- §4.8: the materialised key join ------------------------------------
// The build side is a sorted array of its VALID keys, so a match is a binary
// search and there is no hash table on the device. The build key is unique
// among its valid cells (the operator's precondition, checked by the caller
// via the run count), so lower_bound landing on an equal cell IS the match.
__device__ __forceinline__ std::size_t dev_lower_bound(const void* __restrict__ a, int w,
                                                       std::size_t n, i64 x) {
    std::size_t lo = 0, hi = n;
    while (lo < hi) {
        const std::size_t mid = lo + ((hi - lo) >> 1);
        if (ldw(a, w, mid) < x) lo = mid + 1; else hi = mid;
    }
    return lo;
}

__global__ void join_probe_kernel(const void* __restrict__ bsorted, int bkey_width,
                                  const std::uint32_t* __restrict__ bperm, std::size_t n_bvalid,
                                  const void* __restrict__ pkeys, int pkey_width,
                                  const u64* __restrict__ pvalid, std::size_t rows_probe,
                                  const u64* __restrict__ keylane_valid, int key_from_build,
                                  std::uint32_t* __restrict__ match,
                                  unsigned char* __restrict__ cls) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < rows_probe; i += stride) {
        match[i] = 0xFFFFFFFFu;
        cls[i]   = 0;
        if (!bit_at(pvalid, i)) continue;              // NULL probe key never matches
        if (!n_bvalid) continue;
        const i64 k = ldw(pkeys, pkey_width, i);
        const std::size_t at = dev_lower_bound(bsorted, bkey_width, n_bvalid, k);
        if (at >= n_bvalid || ldw(bsorted, bkey_width, at) != k) continue;
        const std::size_t brow = bperm[at];
        match[i] = static_cast<std::uint32_t>(brow);
        const std::size_t krow = key_from_build ? brow : i;
        cls[i] = bit_at(keylane_valid, krow) ? 1u : 2u;
    }
}

struct ClsIs {
    const unsigned char* cls;
    unsigned char        want;
    __host__ __device__ __forceinline__ u64 operator()(std::size_t i) const {
        return cls[i] == want ? 1ull : 0ull;
    }
};

// Combine the two exclusive scans into one destination per probe row. Class 1
// fills [0, n1); class 2 fills [n1, n1 + n2). Each keeps probe order, so the
// NULL-key rows land as a suffix of every output column.
__global__ void join_positions_kernel(const unsigned char* __restrict__ cls, std::size_t rows_probe,
                                      const std::uint32_t* __restrict__ scan1,
                                      const std::uint32_t* __restrict__ scan2,
                                      std::size_t n1, std::uint32_t* __restrict__ pos) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < rows_probe; i += stride) {
        const unsigned char c = cls[i];
        pos[i] = c == 1u ? scan1[i]
               : c == 2u ? static_cast<std::uint32_t>(n1) + scan2[i]
                         : 0u;
    }
}

__global__ void join_gather_kernel(const void* __restrict__ src, int src_width,
                                   const u64* __restrict__ src_valid,
                                   int from_build, const std::uint32_t* __restrict__ match,
                                   const unsigned char* __restrict__ cls,
                                   const std::uint32_t* __restrict__ pos, std::size_t rows_probe,
                                   void* __restrict__ dst, int dst_width,
                                   u64* __restrict__ dst_valid) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < rows_probe; i += stride) {
        if (!cls[i]) continue;                          // unmatched probe row: absent (inner join)
        const std::size_t srow = from_build ? static_cast<std::size_t>(match[i]) : i;
        put_cell(bit_at(src_valid, srow), ldw(src, src_width, srow), pos[i], dst, dst_width, dst_valid);
    }
}

// Read one device scalar back, synchronizing the stream (the callers that use
// this need the value to size the next allocation, so there is nothing to
// overlap with).
template <class T>
cudaError_t fetch(const void* d_src, T* h_dst, cudaStream_t s) {
    cudaError_t e = cudaMemcpyAsync(h_dst, d_src, sizeof(T), cudaMemcpyDeviceToHost, s);
    if (e != cudaSuccess) return e;
    return cudaStreamSynchronize(s);
}

// ---- the direct grouped reduce ------------------------------------------
// Why it exists: the sort path reads the payload THROUGH the permutation, one
// random gather per row per payload. On 6M rows that is the whole cost of a
// few-group statement, and it repeats per payload — 3 groups measured 7.6 ms
// and 10,000 groups 8.5 ms over the same column, because the grouping was
// never the expensive part. Reading keys and payload in row order turns those
// gathers into streaming loads.
//
// The accumulators live in shared memory and are REPLICATED. With three
// groups one copy would have every thread in the block updating three
// addresses; `reps` private copies divide that contention, and the block folds
// them together once, over G cells, at the end.
constexpr int kDirectSmemBudget = 32 * 1024;   // shared bytes per block
constexpr int kDirectTupleBytes = 6 * 8;       // lo, hi, cnt_v, cnt_star, mn, mx
constexpr int kDirectMaxGroups  = 256;
constexpr int kDirectMaxReps    = 32;

// Where x sits among the n ascending distinct keys. x IS one of them — it came
// out of this column — so the lower bound is the rank.
__device__ __forceinline__ int dev_rank(const i64* __restrict__ keys, int n, i64 x) {
    int lo = 0, hi = n;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (keys[mid] < x) lo = mid + 1; else hi = mid;
    }
    return lo;
}

__global__ void direct_init_kernel(const i64* __restrict__ distinct, int G,
                                   i64* __restrict__ keys_out,
                                   u64* __restrict__ glo, i64* __restrict__ ghi,
                                   i64* __restrict__ gcv, i64* __restrict__ gcs,
                                   i64* __restrict__ gmn, i64* __restrict__ gmx) {
    for (int g = blockIdx.x * blockDim.x + threadIdx.x; g < G;
         g += gridDim.x * blockDim.x) {
        keys_out[g] = distinct[g];
        glo[g] = 0ull; ghi[g] = 0; gcv[g] = 0; gcs[g] = 0;
        gmn[g] = kI64Max; gmx[g] = kI64Min;
    }
}

__global__ void direct_reduce_kernel(const void* __restrict__ keys, int kw,
                                     const u64* __restrict__ kvalid,
                                     std::size_t rows,
                                     const unsigned char* __restrict__ rowmask,
                                     const void* __restrict__ vals, int vw,
                                     const u64* __restrict__ vvalid, int has_vals,
                                     const i64* __restrict__ distinct, int G, int reps,
                                     u64* __restrict__ glo, i64* __restrict__ ghi,
                                     i64* __restrict__ gcv, i64* __restrict__ gcs,
                                     i64* __restrict__ gmn, i64* __restrict__ gmx) {
    extern __shared__ unsigned char smem[];
    i64* const skeys = reinterpret_cast<i64*>(smem);
    u64* const slo   = reinterpret_cast<u64*>(skeys + G);
    i64* const shi   = reinterpret_cast<i64*>(slo + static_cast<std::size_t>(reps) * G);
    i64* const scv   = shi + static_cast<std::size_t>(reps) * G;
    i64* const scs   = scv + static_cast<std::size_t>(reps) * G;
    i64* const smn   = scs + static_cast<std::size_t>(reps) * G;
    i64* const smx   = smn + static_cast<std::size_t>(reps) * G;

    for (int i = threadIdx.x; i < G; i += blockDim.x) skeys[i] = distinct[i];
    for (int i = threadIdx.x; i < reps * G; i += blockDim.x) {
        slo[i] = 0ull; shi[i] = 0; scv[i] = 0; scs[i] = 0;
        smn[i] = kI64Max; smx[i] = kI64Min;
    }
    __syncthreads();

    const int base = ((reps == 1) ? 0 : static_cast<int>(threadIdx.x & (reps - 1))) * G;

    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < rows; i += stride) {
        if (rowmask && !rowmask[i]) continue;
        if (!bit_at(kvalid, i)) continue;     // NULL keys belong to the null group
        const int g = base + dev_rank(skeys, G, ldw(keys, kw, i));

        atomicAdd(reinterpret_cast<unsigned long long*>(&scs[g]), 1ull);
        if (!has_vals) {
            // Keys-only: the reference sets cnt_v = cnt_star and never touches
            // min/max, so neither does this.
            atomicAdd(reinterpret_cast<unsigned long long*>(&scv[g]), 1ull);
            continue;
        }
        if (!bit_at(vvalid, i)) continue;     // a NULL payload contributes the identity
        const i64 x = ldw(vals, vw, i);
        const u64 old = atomicAdd(&slo[g], static_cast<u64>(x));
        const i64 hi_contrib = (x < 0 ? -1 : 0)
                             + static_cast<i64>((old + static_cast<u64>(x)) < old ? 1 : 0);
        atomicAdd(reinterpret_cast<unsigned long long*>(&shi[g]),
                  static_cast<unsigned long long>(hi_contrib));
        atomicAdd(reinterpret_cast<unsigned long long*>(&scv[g]), 1ull);
        atomicMin(reinterpret_cast<long long*>(&smn[g]), static_cast<long long>(x));
        atomicMax(reinterpret_cast<long long*>(&smx[g]), static_cast<long long>(x));
    }
    __syncthreads();

    // Fold the replicas, then merge once per group. The 128-bit add is the
    // same carry add as AddExact — associative and commutative mod 2^128 — so
    // the order the blocks arrive in changes no bit of the answer.
    for (int g = threadIdx.x; g < G; g += blockDim.x) {
        u64 lo = 0ull; i64 hi = 0, cv = 0, cs = 0, mn = kI64Max, mx = kI64Min;
        for (int r = 0; r < reps; ++r) {
            const int k = r * G + g;
            const u64 prev = lo;
            lo += slo[k];
            hi += shi[k] + static_cast<i64>(lo < prev ? 1 : 0);
            cv += scv[k]; cs += scs[k];
            if (smn[k] < mn) mn = smn[k];
            if (smx[k] > mx) mx = smx[k];
        }
        if (cs == 0) continue;
        const u64 old = atomicAdd(&glo[g], lo);
        atomicAdd(reinterpret_cast<unsigned long long*>(&ghi[g]),
                  static_cast<unsigned long long>(hi + static_cast<i64>((old + lo) < old ? 1 : 0)));
        atomicAdd(reinterpret_cast<unsigned long long*>(&gcv[g]),
                  static_cast<unsigned long long>(cv));
        atomicAdd(reinterpret_cast<unsigned long long*>(&gcs[g]),
                  static_cast<unsigned long long>(cs));
        if (cv) {
            atomicMin(reinterpret_cast<long long*>(&gmn[g]), static_cast<long long>(mn));
            atomicMax(reinterpret_cast<long long*>(&gmx[g]), static_cast<long long>(mx));
        }
    }
}

// The contract's empty-payload case — the same line finalize_kernel applies.
__global__ void direct_finalize_kernel(int G, const i64* __restrict__ gcv,
                                       i64* __restrict__ gmn, i64* __restrict__ gmx) {
    for (int g = blockIdx.x * blockDim.x + threadIdx.x; g < G;
         g += gridDim.x * blockDim.x) {
        if (!gcv[g]) { gmn[g] = 0; gmx[g] = 0; }
    }
}

}  // namespace

extern "C" {

// ---- test hooks (docs: the upload-failure test) ----
// Reserving most of the card is how a test forces an allocation refusal
// deterministically; the alternative is filling it with real sets, which is
// slow and depends on how much else is running.
std::size_t gpudb_cuda_debug_free_bytes() {
    std::size_t freeb = 0, totalb = 0;
    if (cudaMemGetInfo(&freeb, &totalb) != cudaSuccess) return 0;
    return freeb;
}

void* gpudb_cuda_debug_reserve(std::size_t leave_bytes) {
    std::size_t freeb = 0, totalb = 0;
    if (cudaMemGetInfo(&freeb, &totalb) != cudaSuccess) return nullptr;
    std::size_t want = freeb > leave_bytes ? freeb - leave_bytes : 0;
    void* p = nullptr;
    const std::size_t step = 64ull << 20;
    while (want > step && cudaMalloc(&p, want) != cudaSuccess) want -= step;
    cudaGetLastError();                    // the failed attempts are expected
    return want > step ? p : nullptr;
}

void gpudb_cuda_debug_release(void* p) {
    if (p) cudaFree(p);
}

void gpudb_cuda_last_scratch(std::size_t* need_bytes, std::size_t* free_bytes) {
    if (need_bytes) *need_bytes = gpudb_cuda_ops::last_scratch_need();
    if (free_bytes) *free_bytes = gpudb_cuda_ops::last_scratch_free();
}

cudaError_t gpudb_cuda_exact_fill_valid(u64* d_bits, std::size_t rows, cudaStream_t s) {
    const std::size_t words = (rows + 63) / 64;
    if (!words) return cudaSuccess;
    fill_valid_kernel<<<grid_for(words), kBlock, 0, s>>>(d_bits, words);
    return cudaGetLastError();
}

cudaError_t gpudb_cuda_exact_scatter_lane(const i64* d_src, std::size_t rows,
                                          std::size_t n_lanes, std::size_t lane,
                                          const u64* d_src_valid, std::size_t valid_bit,
                                          void* d_dst, int dst_width, u64* d_dst_valid,
                                          std::size_t dst_row, cudaStream_t s) {
    if (!rows) return cudaSuccess;
    scatter_lane_kernel<<<grid_for(rows), kBlock, 0, s>>>(d_src, rows, n_lanes, lane,
                                                          d_src_valid, valid_bit,
                                                          d_dst, dst_width, d_dst_valid, dst_row);
    return cudaGetLastError();
}

cudaError_t gpudb_cuda_exact_scatter_pair(const i64* d_kv, std::size_t rows,
                                          const u64* d_key_valid, const u64* d_val_valid,
                                          std::size_t valid_bit,
                                          void* d_keys, int key_width,
                                          void* d_vals, int val_width,
                                          u64* d_key_bits, u64* d_val_bits,
                                          std::size_t dst_row, cudaStream_t s) {
    if (!rows) return cudaSuccess;
    scatter_pair_kernel<<<grid_for(rows), kBlock, 0, s>>>(d_kv, rows, d_key_valid, d_val_valid,
                                                           valid_bit, d_keys, key_width,
                                                           d_vals, val_width,
                                                           d_key_bits, d_val_bits, dst_row);
    return cudaGetLastError();
}

cudaError_t gpudb_cuda_lane_width(const i64* d_vals, std::size_t rows,
                                  int* h_width, cudaStream_t s) {
    *h_width = 1;
    if (!rows) return cudaSuccess;
    DevBuf out;
    cudaError_t e = out.alloc(sizeof(MinMax));
    if (e != cudaSuccess) return e;
    thrust::counting_iterator<std::size_t> it(0);
    auto vals = thrust::make_transform_iterator(it, LoadMinMax{d_vals});
    MinMax init; init.mn = kI64Max; init.mx = kI64Min;
    e = with_temp([&](void* tmp, std::size_t& b) {
        return cub::DeviceReduce::Reduce(tmp, b, vals, static_cast<MinMax*>(out.p),
                                         static_cast<int>(rows), MinMaxOp(), init, s);
    });
    if (e != cudaSuccess) return e;
    MinMax r{};
    if ((e = fetch(out.p, &r, s)) != cudaSuccess) return e;
    // boundaries inclusive, as the design specifies
    if (r.mn >= -128LL && r.mx <= 127LL)                        *h_width = 1;
    else if (r.mn >= -32768LL && r.mx <= 32767LL)               *h_width = 2;
    else if (r.mn >= -2147483648LL && r.mx <= 2147483647LL)     *h_width = 4;
    else                                                         *h_width = 8;
    return cudaSuccess;
}

cudaError_t gpudb_cuda_lane_pack(const i64* d_src, std::size_t rows,
                                 void* d_dst, int width, cudaStream_t s) {
    if (!rows) return cudaSuccess;
    lane_pack_kernel<<<grid_for(rows), kBlock, 0, s>>>(d_src, rows, d_dst, width);
    return cudaGetLastError();
}

cudaError_t gpudb_cuda_exact_null_count(const u64* d_valid, std::size_t rows,
                                        std::size_t* h_nulls, cudaStream_t s) {
    *h_nulls = 0;
    if (!rows || !d_valid) return cudaSuccess;            // no bitmap = every row valid
    const std::size_t words = (rows + 63) / 64;
    DevBuf out;
    cudaError_t e = out.alloc(sizeof(u64));
    if (e != cudaSuccess) return e;
    thrust::counting_iterator<std::size_t> it(0);
    auto pop = thrust::make_transform_iterator(it, PopValid{d_valid, rows});
    e = gpudb_cuda_ops::sum_u64(pop, words, static_cast<u64*>(out.p), s);
    if (e != cudaSuccess) return e;
    u64 valid = 0;
    if ((e = fetch(out.p, &valid, s)) != cudaSuccess) return e;
    *h_nulls = rows - static_cast<std::size_t>(valid);
    return cudaSuccess;
}

// The sort cache over the VALID rows. Two steps: compact the valid row ids
// into d_perm, then sort (order-key, row id) pairs by the key.
cudaError_t gpudb_cuda_exact_sort(const void* d_keys, int key_width,
                                  const u64* d_valid, std::size_t rows,
                                  void* d_sorted, std::uint32_t* d_perm,
                                  std::size_t* h_n_valid, cudaStream_t s) {
    *h_n_valid = 0;
    if (!rows) return cudaSuccess;
    cudaError_t e;
    std::size_t n_valid = rows;

    // the radix sort carries i64 row ids; they are packed to u32 on the way out
    DevBuf perm64;
    if ((e = perm64.alloc(rows * sizeof(i64))) != cudaSuccess) return e;
    i64* d_perm64 = static_cast<i64*>(perm64.p);
    if (!d_valid) {
        if ((e = gpudb_cuda_ops::sequence(d_perm64, rows, s)) != cudaSuccess) return e;
    } else {
        DevBuf flags, all, num;
        if ((e = flags.alloc(rows)) != cudaSuccess) return e;
        if ((e = all.alloc(rows * sizeof(i64))) != cudaSuccess) return e;
        if ((e = num.alloc(sizeof(int))) != cudaSuccess) return e;
        flags_from_bitmap_kernel<<<grid_for(rows), kBlock, 0, s>>>(
            d_valid, rows, static_cast<unsigned char*>(flags.p));
        if ((e = cudaGetLastError()) != cudaSuccess) return e;
        if ((e = gpudb_cuda_ops::sequence(static_cast<i64*>(all.p), rows, s)) != cudaSuccess) return e;
        e = with_temp([&](void* tmp, std::size_t& b) {
            return cub::DeviceSelect::Flagged(tmp, b, static_cast<const i64*>(all.p),
                                              static_cast<const unsigned char*>(flags.p),
                                              d_perm64, static_cast<int*>(num.p),
                                              static_cast<int>(rows), s);
        });
        if (e != cudaSuccess) return e;
        int n = 0;
        if ((e = fetch(num.p, &n, s)) != cudaSuccess) return e;
        n_valid = static_cast<std::size_t>(n);
    }
    if (!n_valid) return cudaSuccess;

    DevBuf uk;
    if ((e = uk.alloc(n_valid * sizeof(u64))) != cudaSuccess) return e;
    gather_order_keys_kernel<<<grid_for(n_valid), kBlock, 0, s>>>(
        d_keys, key_width, d_perm64, n_valid, static_cast<u64*>(uk.p));
    if ((e = cudaGetLastError()) != cudaSuccess) return e;
    if ((e = gpudb_cuda_ops::sort_pairs_u64(static_cast<u64*>(uk.p), d_perm64, n_valid, s))
        != cudaSuccess) return e;
    unmap_order_keys_kernel<<<grid_for(n_valid), kBlock, 0, s>>>(
        static_cast<const u64*>(uk.p), n_valid, d_sorted, key_width, d_perm64, d_perm);
    if ((e = cudaGetLastError()) != cudaSuccess) return e;
    if ((e = cudaStreamSynchronize(s)) != cudaSuccess) return e;   // uk dies on return
    *h_n_valid = n_valid;
    return cudaSuccess;
}

cudaError_t gpudb_cuda_exact_mask(const gpudb::cuda_exact::DevPred* d_preds, int n_preds,
                                  std::size_t rows, unsigned char* d_mask, cudaStream_t s) {
    if (!rows) return cudaSuccess;
    mask_kernel<<<grid_for(rows), kBlock, 0, s>>>(d_preds, n_preds, rows, d_mask);
    return cudaGetLastError();
}

cudaError_t gpudb_cuda_exact_select_sorted(const void* d_sorted, int key_width,
                                           const std::uint32_t* d_perm,
                                           std::size_t n_valid, const unsigned char* d_mask,
                                           void* d_sorted_out, std::uint32_t* d_perm_out,
                                           std::size_t* h_n_sel, cudaStream_t s) {
    *h_n_sel = 0;
    if (!n_valid) return cudaSuccess;
    cudaError_t e;
    if (!d_mask) {                                        // no WHERE: everything survives
        if ((e = cudaMemcpyAsync(d_sorted_out, d_sorted,
                                 n_valid * static_cast<std::size_t>(key_width),
                                 cudaMemcpyDeviceToDevice, s)) != cudaSuccess) return e;
        if ((e = cudaMemcpyAsync(d_perm_out, d_perm, n_valid * sizeof(std::uint32_t),
                                 cudaMemcpyDeviceToDevice, s)) != cudaSuccess) return e;
        *h_n_sel = n_valid;
        return cudaSuccess;
    }
    DevBuf flags, num;
    if ((e = flags.alloc(n_valid)) != cudaSuccess) return e;
    if ((e = num.alloc(sizeof(int))) != cudaSuccess) return e;
    flags_from_mask_kernel<<<grid_for(n_valid), kBlock, 0, s>>>(
        d_perm, n_valid, d_mask, static_cast<unsigned char*>(flags.p));
    if ((e = cudaGetLastError()) != cudaSuccess) return e;

    // Select the surviving POSITIONS once, then gather both arrays through
    // them. Selecting each array separately would need a Flagged pass per
    // array and, now that the two have different element widths, two
    // differently-typed ones.
    DevBuf pick;
    if ((e = pick.alloc(n_valid * sizeof(std::uint32_t))) != cudaSuccess) return e;
    thrust::counting_iterator<std::uint32_t> pos0(0);
    e = with_temp([&](void* tmp, std::size_t& b) {
        return cub::DeviceSelect::Flagged(tmp, b, pos0,
                                          static_cast<const unsigned char*>(flags.p),
                                          static_cast<std::uint32_t*>(pick.p),
                                          static_cast<int*>(num.p),
                                          static_cast<int>(n_valid), s);
    });
    if (e != cudaSuccess) return e;
    int n = 0;
    if ((e = fetch(num.p, &n, s)) != cudaSuccess) return e;
    if (n > 0) {
        gather_selected_kernel<<<grid_for(static_cast<std::size_t>(n)), kBlock, 0, s>>>(
            d_sorted, key_width, d_perm, static_cast<const std::uint32_t*>(pick.p),
            static_cast<std::size_t>(n), d_sorted_out, d_perm_out);
        if ((e = cudaGetLastError()) != cudaSuccess) return e;
        if ((e = cudaStreamSynchronize(s)) != cudaSuccess) return e;   // pick dies on return
    }
    *h_n_sel = static_cast<std::size_t>(n);
    return cudaSuccess;
}

cudaError_t gpudb_cuda_exact_run_count(const void* d_sorted, int key_width, std::size_t n,
                                       std::size_t* h_runs, cudaStream_t s) {
    *h_runs = 0;
    if (!n) return cudaSuccess;
    DevBuf out;
    cudaError_t e = out.alloc(sizeof(u64));
    if (e != cudaSuccess) return e;
    thrust::counting_iterator<std::size_t> it(0);
    auto starts = thrust::make_transform_iterator(it, IsRunStart{d_sorted, key_width});
    if ((e = gpudb_cuda_ops::sum_u64(starts, n, static_cast<u64*>(out.p), s)) != cudaSuccess) return e;
    u64 runs = 0;
    if ((e = fetch(out.p, &runs, s)) != cudaSuccess) return e;
    *h_runs = static_cast<std::size_t>(runs);
    return cudaSuccess;
}

cudaError_t gpudb_cuda_exact_reduce(const void* d_sorted, int key_width,
                                    const std::uint32_t* d_perm, std::size_t n_sel,
                                    const void* d_vals, int val_width,
                                    const u64* d_vvalid, int has_vals,
                                    i64* d_keys_out, i64* d_lo, i64* d_hi,
                                    i64* d_cnt_v, i64* d_cnt_star, i64* d_mn, i64* d_mx,
                                    std::size_t* h_runs, cudaStream_t s) {
    *h_runs = 0;
    if (!n_sel) return cudaSuccess;
    DevBuf tuples, num;
    cudaError_t e;
    if ((e = tuples.alloc(n_sel * sizeof(ETup))) != cudaSuccess) return e;
    if ((e = num.alloc(sizeof(int))) != cudaSuccess) return e;

    thrust::counting_iterator<std::size_t> it(0);
    auto vals = thrust::make_transform_iterator(
        it, RowTuple{d_perm, d_vals, d_vvalid, has_vals, val_width});

    // One run per distinct key: the keys arrive sorted, so equal keys are
    // adjacent and ReduceByKey's runs ARE the groups.
    thrust::counting_iterator<std::size_t> kit(0);
    auto keys_in = thrust::make_transform_iterator(kit, LoadKey{d_sorted, key_width});
    e = with_temp([&](void* tmp, std::size_t& b) {
        return cub::DeviceReduce::ReduceByKey(tmp, b, keys_in, d_keys_out, vals,
                                              static_cast<ETup*>(tuples.p),
                                              static_cast<int*>(num.p), AddExact(),
                                              static_cast<int>(n_sel), s);
    });
    if (e != cudaSuccess) return e;
    int runs = 0;
    if ((e = fetch(num.p, &runs, s)) != cudaSuccess) return e;
    if (runs > 0) {
        finalize_kernel<<<grid_for(static_cast<std::size_t>(runs)), kBlock, 0, s>>>(
            static_cast<const ETup*>(tuples.p), static_cast<std::size_t>(runs),
            d_lo, d_hi, d_cnt_v, d_cnt_star, d_mn, d_mx);
        if ((e = cudaGetLastError()) != cudaSuccess) return e;
        if ((e = cudaStreamSynchronize(s)) != cudaSuccess) return e;   // tuples dies on return
    }
    *h_runs = static_cast<std::size_t>(runs);
    return cudaSuccess;
}

cudaError_t gpudb_cuda_join_mat_probe(const void* d_bsorted, int bkey_width,
                                      const std::uint32_t* d_bperm, std::size_t n_bvalid,
                                      const void* d_pkeys, int pkey_width, const u64* d_pvalid,
                                      std::size_t rows_probe,
                                      const u64* d_keylane_valid, int key_from_build,
                                      std::uint32_t* d_match, unsigned char* d_cls,
                                      std::size_t* h_n1, std::size_t* h_n2, cudaStream_t s) {
    *h_n1 = 0; *h_n2 = 0;
    if (!rows_probe) return cudaSuccess;
    join_probe_kernel<<<grid_for(rows_probe), kBlock, 0, s>>>(
        d_bsorted, bkey_width, d_bperm, n_bvalid, d_pkeys, pkey_width, d_pvalid, rows_probe,
        d_keylane_valid, key_from_build, d_match, d_cls);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return e;

    DevBuf out;
    if ((e = out.alloc(2 * sizeof(u64))) != cudaSuccess) return e;
    thrust::counting_iterator<std::size_t> it(0);
    auto ones1 = thrust::make_transform_iterator(it, ClsIs{d_cls, 1u});
    auto ones2 = thrust::make_transform_iterator(it, ClsIs{d_cls, 2u});
    if ((e = gpudb_cuda_ops::sum_u64(ones1, rows_probe, static_cast<u64*>(out.p), s))
        != cudaSuccess) return e;
    if ((e = gpudb_cuda_ops::sum_u64(ones2, rows_probe, static_cast<u64*>(out.p) + 1, s))
        != cudaSuccess) return e;
    u64 counts[2] = {0, 0};
    if ((e = cudaMemcpyAsync(counts, out.p, 2 * sizeof(u64), cudaMemcpyDeviceToHost, s))
        != cudaSuccess) return e;
    if ((e = cudaStreamSynchronize(s)) != cudaSuccess) return e;
    *h_n1 = static_cast<std::size_t>(counts[0]);
    *h_n2 = static_cast<std::size_t>(counts[1]);
    return cudaSuccess;
}

cudaError_t gpudb_cuda_join_mat_positions(const unsigned char* d_cls, std::size_t rows_probe,
                                          std::size_t n1, std::uint32_t* d_pos, cudaStream_t s) {
    if (!rows_probe) return cudaSuccess;
    DevBuf s1, s2;
    cudaError_t e;
    if ((e = s1.alloc(rows_probe * sizeof(std::uint32_t))) != cudaSuccess) return e;
    if ((e = s2.alloc(rows_probe * sizeof(std::uint32_t))) != cudaSuccess) return e;
    thrust::counting_iterator<std::size_t> it(0);
    auto ones1 = thrust::make_transform_iterator(it, ClsIs{d_cls, 1u});
    auto ones2 = thrust::make_transform_iterator(it, ClsIs{d_cls, 2u});
    e = with_temp([&](void* tmp, std::size_t& b) {
        return cub::DeviceScan::ExclusiveSum(tmp, b, ones1, static_cast<std::uint32_t*>(s1.p),
                                             static_cast<int>(rows_probe), s);
    });
    if (e != cudaSuccess) return e;
    e = with_temp([&](void* tmp, std::size_t& b) {
        return cub::DeviceScan::ExclusiveSum(tmp, b, ones2, static_cast<std::uint32_t*>(s2.p),
                                             static_cast<int>(rows_probe), s);
    });
    if (e != cudaSuccess) return e;
    join_positions_kernel<<<grid_for(rows_probe), kBlock, 0, s>>>(
        d_cls, rows_probe, static_cast<const std::uint32_t*>(s1.p),
        static_cast<const std::uint32_t*>(s2.p), n1, d_pos);
    if ((e = cudaGetLastError()) != cudaSuccess) return e;
    return cudaStreamSynchronize(s);           // s1 / s2 die on return
}

cudaError_t gpudb_cuda_join_mat_gather(const void* d_src, int src_width,
                                       const u64* d_src_valid, int from_build,
                                       const std::uint32_t* d_match, const unsigned char* d_cls,
                                       const std::uint32_t* d_pos, std::size_t rows_probe,
                                       void* d_dst, int dst_width, u64* d_dst_valid,
                                       cudaStream_t s) {
    if (!rows_probe) return cudaSuccess;
    join_gather_kernel<<<grid_for(rows_probe), kBlock, 0, s>>>(
        d_src, src_width, d_src_valid, from_build, d_match, d_cls, d_pos, rows_probe,
        d_dst, dst_width, d_dst_valid);
    return cudaGetLastError();
}

int gpudb_cuda_exact_direct_max_groups(void) { return kDirectMaxGroups; }

cudaError_t gpudb_cuda_exact_distinct(const void* d_sorted, int key_width, std::size_t n,
                                      i64* d_out, std::size_t cap,
                                      std::size_t* h_groups, cudaStream_t s) {
    *h_groups = 0;
    if (!n) return cudaSuccess;
    // CUB's select takes the item count as an int here; a column past that is
    // far past the group counts this path serves, so say so and let the caller
    // keep the sort path.
    if (n > static_cast<std::size_t>(0x7fffffff)) return cudaErrorInvalidValue;

    // Count first. Unique would otherwise need an output buffer the size of
    // the whole column, which is the allocation this path exists to avoid.
    std::size_t runs = 0;
    cudaError_t e = gpudb_cuda_exact_run_count(d_sorted, key_width, n, &runs, s);
    if (e != cudaSuccess) return e;
    *h_groups = runs;
    if (runs > cap) return cudaSuccess;          // too many groups: nothing written

    DevBuf num;
    if ((e = num.alloc(sizeof(int))) != cudaSuccess) return e;
    thrust::counting_iterator<std::size_t> it(0);
    auto keys = thrust::make_transform_iterator(it, ReadKey{d_sorted, key_width});
    return with_temp([&](void* tmp, std::size_t& bytes) {
        return cub::DeviceSelect::Unique(tmp, bytes, keys, d_out,
                                         static_cast<int*>(num.p),
                                         static_cast<int>(n), s);
    });
}

cudaError_t gpudb_cuda_exact_direct(const void* d_keys, int key_width,
                                    const unsigned long long* d_kvalid,
                                    std::size_t rows, const unsigned char* d_rowmask,
                                    const void* d_vals, int val_width,
                                    const unsigned long long* d_vvalid, int has_vals,
                                    const i64* d_distinct, int n_groups,
                                    i64* d_keys_out, i64* d_lo, i64* d_hi,
                                    i64* d_cnt_v, i64* d_cnt_star,
                                    i64* d_mn, i64* d_mx, cudaStream_t s) {
    if (n_groups <= 0 || n_groups > kDirectMaxGroups) return cudaErrorInvalidValue;

    // Replicas: as many as the shared budget affords, a power of two so the
    // thread's copy is an AND rather than a modulo.
    const int avail = kDirectSmemBudget - n_groups * static_cast<int>(sizeof(i64));
    int reps = avail / (n_groups * kDirectTupleBytes);
    if (reps > kDirectMaxReps) reps = kDirectMaxReps;
    if (reps < 1) reps = 1;
    while (reps & (reps - 1)) reps &= reps - 1;
    const std::size_t smem = static_cast<std::size_t>(n_groups) * sizeof(i64)
                           + static_cast<std::size_t>(reps) * n_groups * kDirectTupleBytes;

    auto* glo = reinterpret_cast<u64*>(d_lo);
    direct_init_kernel<<<grid_for(static_cast<std::size_t>(n_groups)), kBlock, 0, s>>>(
        d_distinct, n_groups, d_keys_out, glo, d_hi, d_cnt_v, d_cnt_star, d_mn, d_mx);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return e;
    if (!rows) return cudaSuccess;

    direct_reduce_kernel<<<grid_for(rows), kBlock, smem, s>>>(
        d_keys, key_width, d_kvalid, rows, d_rowmask,
        d_vals, val_width, d_vvalid, has_vals,
        d_distinct, n_groups, reps,
        glo, d_hi, d_cnt_v, d_cnt_star, d_mn, d_mx);
    if ((e = cudaGetLastError()) != cudaSuccess) return e;

    direct_finalize_kernel<<<grid_for(static_cast<std::size_t>(n_groups)), kBlock, 0, s>>>(
        n_groups, d_cnt_v, d_mn, d_mx);
    return cudaGetLastError();
}

cudaError_t gpudb_cuda_exact_global(const gpudb::cuda_exact::DevPred* d_preds, int n_preds,
                                    std::size_t rows,
                                    const void* const* d_vals, const int* h_widths,
                                    const u64* const* d_vvalid,
                                    int n_pays, gpudb::cuda_exact::ExactTuple* h_out,
                                    std::int64_t* h_count_star, cudaStream_t s) {
    *h_count_star = 0;
    for (int p = 0; p < n_pays; ++p) h_out[p] = gpudb::cuda_exact::ExactTuple{};
    if (!rows) return cudaSuccess;
    cudaError_t e;

    // The mask is evaluated ONCE and every payload reads it, which is the
    // difference between this and calling the aggregate per payload.
    DevBuf maskbuf;
    const unsigned char* mask = nullptr;
    if (n_preds) {
        if ((e = maskbuf.alloc(rows)) != cudaSuccess) return e;
        mask = static_cast<const unsigned char*>(maskbuf.p);
        mask_kernel<<<grid_for(rows), kBlock, 0, s>>>(d_preds, n_preds, rows,
                                                      static_cast<unsigned char*>(maskbuf.p));
        if ((e = cudaGetLastError()) != cudaSuccess) return e;
    }

    DevBuf out;
    if ((e = out.alloc(sizeof(ETup))) != cudaSuccess) return e;
    thrust::counting_iterator<std::size_t> it(0);
    for (int p = 0; p < n_pays; ++p) {
        auto vals = thrust::make_transform_iterator(
            it, GlobalTuple{mask, d_vals[p], d_vvalid[p], h_widths[p]});
        e = with_temp([&](void* tmp, std::size_t& b) {
            return cub::DeviceReduce::Reduce(tmp, b, vals, static_cast<ETup*>(out.p),
                                             static_cast<int>(rows), AddExact(), etup_identity(), s);
        });
        if (e != cudaSuccess) return e;
        ETup t{};
        if ((e = fetch(out.p, &t, s)) != cudaSuccess) return e;
        h_out[p].lo       = t.lo;
        h_out[p].hi       = t.hi;
        h_out[p].cnt_v    = t.cnt_v;
        h_out[p].cnt_star = t.cnt_star;
        h_out[p].mn       = t.cnt_v ? t.mn : 0;
        h_out[p].mx       = t.cnt_v ? t.mx : 0;
    }

    // count(*) is the same number for every payload — a row either survives the
    // mask or it does not, whatever its payload cells hold. So take it from the
    // first payload rather than scanning again; only a payload-less call
    // (predicates alone) has to count the mask itself.
    if (n_pays > 0) {
        *h_count_star = h_out[0].cnt_star;
        return cudaSuccess;
    }
    DevBuf cnt;
    if ((e = cnt.alloc(sizeof(u64))) != cudaSuccess) return e;
    auto ones = thrust::make_transform_iterator(it, MaskOne{mask});
    if ((e = gpudb_cuda_ops::sum_u64(ones, rows, static_cast<u64*>(cnt.p), s)) != cudaSuccess) return e;
    u64 c = 0;
    if ((e = fetch(cnt.p, &c, s)) != cudaSuccess) return e;
    *h_count_star = static_cast<std::int64_t>(c);
    return cudaSuccess;
}

cudaError_t gpudb_cuda_exact_null_group(const u64* d_kvalid, const unsigned char* d_mask,
                                        std::size_t rows, const void* d_vals, int val_width,
                                        const u64* d_vvalid, int has_vals,
                                        gpudb::cuda_exact::ExactTuple* h_out, cudaStream_t s) {
    *h_out = gpudb::cuda_exact::ExactTuple{};
    if (!rows || !d_kvalid) return cudaSuccess;      // no bitmap = no NULL key = no such group
    DevBuf out;
    cudaError_t e = out.alloc(sizeof(ETup));
    if (e != cudaSuccess) return e;

    thrust::counting_iterator<std::size_t> it(0);
    auto vals = thrust::make_transform_iterator(
        it, NullKeyTuple{d_kvalid, d_mask, d_vals, d_vvalid, has_vals, val_width});
    e = with_temp([&](void* tmp, std::size_t& b) {
        return cub::DeviceReduce::Reduce(tmp, b, vals, static_cast<ETup*>(out.p),
                                         static_cast<int>(rows), AddExact(), etup_identity(), s);
    });
    if (e != cudaSuccess) return e;
    ETup t{};
    if ((e = fetch(out.p, &t, s)) != cudaSuccess) return e;

    h_out->lo       = t.lo;
    h_out->hi       = t.hi;
    h_out->cnt_v    = t.cnt_v;
    h_out->cnt_star = t.cnt_star;
    h_out->mn       = t.cnt_v ? t.mn : 0;
    h_out->mx       = t.cnt_v ? t.mx : 0;
    return cudaSuccess;
}

}  // extern "C"
