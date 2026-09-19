// exact_kernel.cu — the device side of the v0.7 exact GROUP BY (§4.1/§4.2).
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
    const i64* perm;
    const i64* vals;
    const u64* vvalid;
    int        has_vals;
    __host__ __device__ __forceinline__ ETup operator()(std::size_t i) const {
        ETup t = etup_identity();
        t.cnt_star = 1;
        if (!has_vals) { t.cnt_v = 1; return t; }   // reference: cnt_v = cnt_star, mn/mx untouched
        const std::size_t row = static_cast<std::size_t>(perm[i]);
        if (!bit_at(vvalid, row)) return t;         // NULL payload: counts in count(*) only
        return etup_of_value(vals[row]);
    }
};

// The NULL-key group (§4.1): every row whose KEY is NULL and which passes the
// mask, reduced over the whole column in storage order — those rows are not in
// the sort cache at all, since the cache covers the valid keys.
struct NullKeyTuple {
    const u64*           kvalid;
    const unsigned char* mask;
    const i64*           vals;
    const u64*           vvalid;
    int                  has_vals;
    __host__ __device__ __forceinline__ ETup operator()(std::size_t i) const {
        ETup t = etup_identity();
        if (bit_at(kvalid, i)) return t;            // key is present: a different group
        if (mask && !mask[i]) return t;             // failed the WHERE: absent from count(*) too
        t.cnt_star = 1;
        if (!has_vals) { t.cnt_v = 1; return t; }
        if (!bit_at(vvalid, i)) return t;
        return etup_of_value(vals[i]);
    }
};

// §4.12: one row's contribution to the global aggregate. No key, no
// permutation — the row index IS the row, which is the whole point of the
// operator: it exists so a keyless aggregate does not pay for a sort it has no
// use for.
struct GlobalTuple {
    const unsigned char* mask;
    const i64*           vals;
    const u64*           vvalid;
    __host__ __device__ __forceinline__ ETup operator()(std::size_t i) const {
        ETup t = etup_identity();
        if (mask && !mask[i]) return t;
        t.cnt_star = 1;
        if (!bit_at(vvalid, i)) return t;   // NULL payload: in count(*) only
        return etup_of_value(vals[i]);
    }
};

// Surviving rows when there is no payload to carry the count for us.
struct MaskOne {
    const unsigned char* mask;
    __host__ __device__ __forceinline__ u64 operator()(std::size_t i) const {
        return (!mask || mask[i]) ? 1ull : 0ull;
    }
};

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
    const i64* sorted;
    __host__ __device__ __forceinline__ u64 operator()(std::size_t i) const {
        return (i == 0 || sorted[i] != sorted[i - 1]) ? 1ull : 0ull;
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
                                         i64* dst, u64* dst_valid) {
    dst[d] = ok ? value : 0;
    if (!ok && dst_valid) atomicAnd(&dst_valid[d >> 6], ~(1ull << (d & 63)));
}

__global__ void scatter_lane_kernel(const i64* __restrict__ src, std::size_t rows,
                                    std::size_t n_lanes, std::size_t lane,
                                    const u64* __restrict__ src_valid, std::size_t valid_bit,
                                    i64* __restrict__ dst, u64* __restrict__ dst_valid,
                                    std::size_t dst_row) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t j = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         j < rows; j += stride) {
        const bool ok = bit_at(src_valid, valid_bit + j);
        put_cell(ok, src[j * n_lanes + lane], dst_row + j, dst, dst_valid);
    }
}

__global__ void scatter_pair_kernel(const i64* __restrict__ kv, std::size_t rows,
                                    const u64* __restrict__ key_valid,
                                    const u64* __restrict__ val_valid, std::size_t valid_bit,
                                    i64* __restrict__ keys, i64* __restrict__ vals,
                                    u64* __restrict__ key_bits, u64* __restrict__ val_bits,
                                    std::size_t dst_row) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t j = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         j < rows; j += stride) {
        const std::size_t d = dst_row + j;
        put_cell(bit_at(key_valid, valid_bit + j), kv[2 * j],     d, keys, key_bits);
        put_cell(bit_at(val_valid, valid_bit + j), kv[2 * j + 1], d, vals, val_bits);
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
            const i64 raw = q.data[i];
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
__global__ void flags_from_mask_kernel(const i64* __restrict__ perm, std::size_t n,
                                       const unsigned char* __restrict__ mask,
                                       unsigned char* __restrict__ flags) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride) flags[i] = mask[perm[i]];
}

__global__ void gather_order_keys_kernel(const i64* __restrict__ keys, const i64* __restrict__ perm,
                                         std::size_t n, u64* __restrict__ out) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride) out[i] = key_to_u64(keys[perm[i]]);
}

__global__ void unmap_order_keys_kernel(const u64* __restrict__ in, std::size_t n,
                                        i64* __restrict__ out) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride) out[i] = u64_to_key(in[i]);
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

// Read one device scalar back, synchronizing the stream (the callers that use
// this need the value to size the next allocation, so there is nothing to
// overlap with).
template <class T>
cudaError_t fetch(const void* d_src, T* h_dst, cudaStream_t s) {
    cudaError_t e = cudaMemcpyAsync(h_dst, d_src, sizeof(T), cudaMemcpyDeviceToHost, s);
    if (e != cudaSuccess) return e;
    return cudaStreamSynchronize(s);
}

}  // namespace

extern "C" {

cudaError_t gpudb_cuda_exact_fill_valid(u64* d_bits, std::size_t rows, cudaStream_t s) {
    const std::size_t words = (rows + 63) / 64;
    if (!words) return cudaSuccess;
    fill_valid_kernel<<<grid_for(words), kBlock, 0, s>>>(d_bits, words);
    return cudaGetLastError();
}

cudaError_t gpudb_cuda_exact_scatter_lane(const i64* d_src, std::size_t rows,
                                          std::size_t n_lanes, std::size_t lane,
                                          const u64* d_src_valid, std::size_t valid_bit,
                                          i64* d_dst, u64* d_dst_valid,
                                          std::size_t dst_row, cudaStream_t s) {
    if (!rows) return cudaSuccess;
    scatter_lane_kernel<<<grid_for(rows), kBlock, 0, s>>>(d_src, rows, n_lanes, lane,
                                                          d_src_valid, valid_bit,
                                                          d_dst, d_dst_valid, dst_row);
    return cudaGetLastError();
}

cudaError_t gpudb_cuda_exact_scatter_pair(const i64* d_kv, std::size_t rows,
                                          const u64* d_key_valid, const u64* d_val_valid,
                                          std::size_t valid_bit,
                                          i64* d_keys, i64* d_vals,
                                          u64* d_key_bits, u64* d_val_bits,
                                          std::size_t dst_row, cudaStream_t s) {
    if (!rows) return cudaSuccess;
    scatter_pair_kernel<<<grid_for(rows), kBlock, 0, s>>>(d_kv, rows, d_key_valid, d_val_valid,
                                                           valid_bit, d_keys, d_vals,
                                                           d_key_bits, d_val_bits, dst_row);
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
cudaError_t gpudb_cuda_exact_sort(const i64* d_keys, const u64* d_valid, std::size_t rows,
                                  i64* d_sorted, i64* d_perm,
                                  std::size_t* h_n_valid, cudaStream_t s) {
    *h_n_valid = 0;
    if (!rows) return cudaSuccess;
    cudaError_t e;
    std::size_t n_valid = rows;

    if (!d_valid) {
        if ((e = gpudb_cuda_ops::sequence(d_perm, rows, s)) != cudaSuccess) return e;
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
                                              d_perm, static_cast<int*>(num.p),
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
        d_keys, d_perm, n_valid, static_cast<u64*>(uk.p));
    if ((e = cudaGetLastError()) != cudaSuccess) return e;
    if ((e = gpudb_cuda_ops::sort_pairs_u64(static_cast<u64*>(uk.p), d_perm, n_valid, s))
        != cudaSuccess) return e;
    unmap_order_keys_kernel<<<grid_for(n_valid), kBlock, 0, s>>>(
        static_cast<const u64*>(uk.p), n_valid, d_sorted);
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

cudaError_t gpudb_cuda_exact_select_sorted(const i64* d_sorted, const i64* d_perm,
                                           std::size_t n_valid, const unsigned char* d_mask,
                                           i64* d_sorted_out, i64* d_perm_out,
                                           std::size_t* h_n_sel, cudaStream_t s) {
    *h_n_sel = 0;
    if (!n_valid) return cudaSuccess;
    cudaError_t e;
    if (!d_mask) {                                        // no WHERE: everything survives
        const std::size_t b = n_valid * sizeof(i64);
        if ((e = cudaMemcpyAsync(d_sorted_out, d_sorted, b, cudaMemcpyDeviceToDevice, s))
            != cudaSuccess) return e;
        if ((e = cudaMemcpyAsync(d_perm_out, d_perm, b, cudaMemcpyDeviceToDevice, s))
            != cudaSuccess) return e;
        *h_n_sel = n_valid;
        return cudaSuccess;
    }
    DevBuf flags, num;
    if ((e = flags.alloc(n_valid)) != cudaSuccess) return e;
    if ((e = num.alloc(sizeof(int))) != cudaSuccess) return e;
    flags_from_mask_kernel<<<grid_for(n_valid), kBlock, 0, s>>>(
        d_perm, n_valid, d_mask, static_cast<unsigned char*>(flags.p));
    if ((e = cudaGetLastError()) != cudaSuccess) return e;

    // Two Flagged passes over one flag array: the sorted keys and the row ids
    // keep the same surviving positions, so the two selections agree.
    e = with_temp([&](void* tmp, std::size_t& b) {
        return cub::DeviceSelect::Flagged(tmp, b, d_sorted,
                                          static_cast<const unsigned char*>(flags.p),
                                          d_sorted_out, static_cast<int*>(num.p),
                                          static_cast<int>(n_valid), s);
    });
    if (e != cudaSuccess) return e;
    e = with_temp([&](void* tmp, std::size_t& b) {
        return cub::DeviceSelect::Flagged(tmp, b, d_perm,
                                          static_cast<const unsigned char*>(flags.p),
                                          d_perm_out, static_cast<int*>(num.p),
                                          static_cast<int>(n_valid), s);
    });
    if (e != cudaSuccess) return e;
    int n = 0;
    if ((e = fetch(num.p, &n, s)) != cudaSuccess) return e;
    *h_n_sel = static_cast<std::size_t>(n);
    return cudaSuccess;
}

cudaError_t gpudb_cuda_exact_run_count(const i64* d_sorted, std::size_t n,
                                       std::size_t* h_runs, cudaStream_t s) {
    *h_runs = 0;
    if (!n) return cudaSuccess;
    DevBuf out;
    cudaError_t e = out.alloc(sizeof(u64));
    if (e != cudaSuccess) return e;
    thrust::counting_iterator<std::size_t> it(0);
    auto starts = thrust::make_transform_iterator(it, IsRunStart{d_sorted});
    if ((e = gpudb_cuda_ops::sum_u64(starts, n, static_cast<u64*>(out.p), s)) != cudaSuccess) return e;
    u64 runs = 0;
    if ((e = fetch(out.p, &runs, s)) != cudaSuccess) return e;
    *h_runs = static_cast<std::size_t>(runs);
    return cudaSuccess;
}

cudaError_t gpudb_cuda_exact_reduce(const i64* d_sorted, const i64* d_perm, std::size_t n_sel,
                                    const i64* d_vals, const u64* d_vvalid, int has_vals,
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
    auto vals = thrust::make_transform_iterator(it, RowTuple{d_perm, d_vals, d_vvalid, has_vals});

    // One run per distinct key: the keys arrive sorted, so equal keys are
    // adjacent and ReduceByKey's runs ARE the groups.
    e = with_temp([&](void* tmp, std::size_t& b) {
        return cub::DeviceReduce::ReduceByKey(tmp, b, d_sorted, d_keys_out, vals,
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

cudaError_t gpudb_cuda_exact_global(const gpudb::cuda_exact::DevPred* d_preds, int n_preds,
                                    std::size_t rows,
                                    const i64* const* d_vals, const u64* const* d_vvalid,
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
        auto vals = thrust::make_transform_iterator(it, GlobalTuple{mask, d_vals[p], d_vvalid[p]});
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
                                        std::size_t rows, const i64* d_vals,
                                        const u64* d_vvalid, int has_vals,
                                        gpudb::cuda_exact::ExactTuple* h_out, cudaStream_t s) {
    *h_out = gpudb::cuda_exact::ExactTuple{};
    if (!rows || !d_kvalid) return cudaSuccess;      // no bitmap = no NULL key = no such group
    DevBuf out;
    cudaError_t e = out.alloc(sizeof(ETup));
    if (e != cudaSuccess) return e;

    thrust::counting_iterator<std::size_t> it(0);
    auto vals = thrust::make_transform_iterator(
        it, NullKeyTuple{d_kvalid, d_mask, d_vals, d_vvalid, has_vals});
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
