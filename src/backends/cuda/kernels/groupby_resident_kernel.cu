// groupby_resident_kernel.cu — resident GROUP BY / top-k device paths (v0.6).
//
// GROUP BY reuses the v0.5 build-side sort cache on the key column (keys
// sorted ascending + original-index permutation). Values are read through
// the permutation and reduced per key run with CUB DeviceReduce::ReduceByKey,
// so output rows come out sorted by key — the cross-backend contract in
// gpu_backend.hpp. i64 sums accumulate in uint64 (wrap-add commutes ->
// bit-exact regardless of reduction order); f64 sums are backend-ordered
// under the tolerance contract.
//
// Every CUB call here runs on EXPLICIT temp storage we allocate and free
// (cub_ops.cuh); Thrust is used only for its allocation-free iterators. So a
// device fault inside any step comes back to the host wrapper as its real
// cudaError_t (-> std::runtime_error -> SQL error) instead of a Thrust
// temporary's destructor throwing on cudaFree and aborting the process.
//
// top-k on an f64 column needs its own sort: doubles are mapped to
// order-preserving uint64 keys (NaN greatest, matching DuckDB's total order)
// so the radix path is used, then the sorted values are gathered back.

#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include "cub_ops.cuh"

namespace {

using u64 = unsigned long long;
using i64 = std::int64_t;
using gpudb_cuda_ops::DevBuf;
using gpudb_cuda_ops::with_temp;

// (value, 1) through the permutation: sorted position i -> original row perm[i].
struct PermValI64 {
    const i64* vals; const i64* perm;
    __host__ __device__ thrust::tuple<u64, u64> operator()(i64 i) const {
        return thrust::make_tuple(static_cast<u64>(vals[perm[i]]), 1ULL);
    }
};
struct PermValF64 {
    const double* vals; const i64* perm;
    __host__ __device__ thrust::tuple<double, u64> operator()(i64 i) const {
        return thrust::make_tuple(vals[perm[i]], 1ULL);
    }
};
struct AddU64Pair {
    __host__ __device__ thrust::tuple<u64, u64> operator()(const thrust::tuple<u64, u64>& a,
                                                           const thrust::tuple<u64, u64>& b) const {
        return thrust::make_tuple(thrust::get<0>(a) + thrust::get<0>(b),
                                  thrust::get<1>(a) + thrust::get<1>(b));
    }
};
struct AddF64Pair {
    __host__ __device__ thrust::tuple<double, u64> operator()(const thrust::tuple<double, u64>& a,
                                                              const thrust::tuple<double, u64>& b) const {
        return thrust::make_tuple(thrust::get<0>(a) + thrust::get<0>(b),
                                  thrust::get<1>(a) + thrust::get<1>(b));
    }
};
struct AddU64 {
    __host__ __device__ u64 operator()(u64 a, u64 b) const { return a + b; }
};

// 1 at every run start of an ascending-sorted key array.
struct RunStart {
    const i64* keys;
    __host__ __device__ u64 operator()(i64 i) const {
        return (i == 0 || keys[i] != keys[i - 1]) ? 1ULL : 0ULL;
    }
};

// Order-preserving map double -> uint64: negatives reversed, positives get the
// top bit, NaN (any sign) maps to the maximum so it sorts last.
struct F64OrderKey {
    __host__ __device__ u64 operator()(double d) const {
        if (d != d) return ~0ULL;
        u64 b;
        memcpy(&b, &d, sizeof(b));
        return (b >> 63) ? ~b : (b | (1ULL << 63));
    }
};

// Order-preserving u64 keys for the aggregate: i64 flips the sign bit; f64
// folds sign-magnitude with EVERY NaN greatest (DuckDB ORDER BY order) and
// -0.0 == 0.0 handled by the tie rule. Inverted keys turn "largest k" into
// "smallest k" so one ascending radix sort serves both directions.
template <typename A> struct OrderKey;
template <> struct OrderKey<i64> {
    __host__ __device__ u64 operator()(i64 v) const { return static_cast<u64>(v) ^ (1ULL << 63); }
};
template <> struct OrderKey<double> {
    __host__ __device__ u64 operator()(double d) const { return F64OrderKey{}(d); }
};
template <typename A> struct InvOrderKey {
    __host__ __device__ u64 operator()(A v) const { return ~OrderKey<A>{}(v); }
};

// Reduce the permuted values per key run. Outputs sized to the run count;
// the run count actually produced lands in *h_runs.
template <typename A, typename Val, typename Add>
cudaError_t reduce_runs(const i64* d_sorted, std::size_t n, Val vals_first, Add add,
                        i64* out_keys, A* out_agg, i64* out_counts,
                        std::size_t* h_runs, cudaStream_t s) {
    DevBuf nr; cudaError_t e = nr.alloc(sizeof(u64));
    if (e != cudaSuccess) return e;
    auto out_vals = thrust::make_zip_iterator(thrust::make_tuple(out_agg, reinterpret_cast<u64*>(out_counts)));
    e = with_temp([&](void* tmp, std::size_t& b) {
        return cub::DeviceReduce::ReduceByKey(tmp, b, d_sorted, out_keys, vals_first, out_vals,
                                              static_cast<u64*>(nr.p), add, n, s);
    });
    if (e != cudaSuccess) return e;
    u64 runs = 0;
    if ((e = cudaMemcpyAsync(&runs, nr.p, sizeof(u64), cudaMemcpyDeviceToHost, s)) != cudaSuccess) return e;
    if ((e = cudaStreamSynchronize(s)) != cudaSuccess) return e;
    *h_runs = static_cast<std::size_t>(runs);
    return cudaSuccess;
}

// Sum of a device-side transform, returned to the host.
template <typename It>
cudaError_t host_sum(It first, std::size_t n, std::size_t* h_out, cudaStream_t s) {
    if (n == 0) { *h_out = 0; return cudaSuccess; }
    DevBuf o; cudaError_t e = o.alloc(sizeof(u64));
    if (e != cudaSuccess) return e;
    if ((e = gpudb_cuda_ops::sum_u64(first, n, static_cast<u64*>(o.p), s)) != cudaSuccess) return e;
    u64 v = 0;
    if ((e = cudaMemcpyAsync(&v, o.p, sizeof(u64), cudaMemcpyDeviceToHost, s)) != cudaSuccess) return e;
    if ((e = cudaStreamSynchronize(s)) != cudaSuccess) return e;
    *h_out = static_cast<std::size_t>(v);
    return cudaSuccess;
}

// Compact `in` (any random-access iterator) where flag(i) into `out`.
template <typename In, typename Flags, typename Out>
cudaError_t select_flagged(In in, Flags flags, Out out, std::size_t n, cudaStream_t s) {
    DevBuf ns; cudaError_t e = ns.alloc(sizeof(u64));
    if (e != cudaSuccess) return e;
    return with_temp([&](void* tmp, std::size_t& b) {
        return cub::DeviceSelect::Flagged(tmp, b, in, flags, out, static_cast<u64*>(ns.p), n, s);
    });
}

// ---- GroupByFilter on the device (v0.6): cmp on the aggregate, then top-k ----
// cmp codes mirror GroupByFilter::Cmp: 0 None, 1 GT, 2 GE, 3 LT, 4 LE.

template <typename A>
struct Keep {
    int cmp; A t;
    __host__ __device__ bool operator()(A a) const {
        switch (cmp) {
            case 1:  return a >  t;
            case 2:  return a >= t;
            case 3:  return a <  t;
            case 4:  return a <= t;
            default: return true;
        }
    }
};
template <typename A>
struct KeepU64 {          // Keep as a 0/1 count over element indices
    const A* agg; Keep<A> k;
    __host__ __device__ u64 operator()(i64 i) const { return k(agg[i]) ? 1ULL : 0ULL; }
};
template <typename A>
struct KeepFlag {         // Keep as a bool flag over element indices
    const A* agg; Keep<A> k;
    __host__ __device__ bool operator()(i64 i) const { return k(agg[i]); }
};

struct KeyLessU64  { const u64* keys; u64 t; __host__ __device__ u64  operator()(i64 i) const { return keys[i] <  t ? 1ULL : 0ULL; } };
struct KeyLessFlag { const u64* keys; u64 t; __host__ __device__ bool operator()(i64 i) const { return keys[i] <  t; } };
struct KeyEqU64    { const u64* keys; u64 t; __host__ __device__ u64  operator()(i64 i) const { return keys[i] == t ? 1ULL : 0ULL; } };
struct KeyEqFlag   { const u64* keys; u64 t; __host__ __device__ bool operator()(i64 i) const { return keys[i] == t; } };

// ---- radix-select for top-k of groups ----
//
// Eight MSB->LSB passes of a 256-bin histogram over the elements matching
// the prefix so far locate the k-th smallest key T; the output is every
// element with key < T plus enough key == T elements to reach k (tie choice
// unspecified, per the contract), finally sorted by key.
__global__ void radix_hist_kernel(const u64* __restrict__ keys, std::size_t n,
                                  u64 prefix, int shift, u64* __restrict__ hist) {
    __shared__ unsigned int sh[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) sh[i] = 0u;
    __syncthreads();
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        const u64 k = keys[i];
        if (shift == 56 || (k >> (shift + 8)) == prefix)
            atomicAdd(&sh[(k >> shift) & 0xFFu], 1u);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 256; i += blockDim.x)
        if (sh[i]) atomicAdd(&hist[i], static_cast<u64>(sh[i]));
}

// Selects the k smallest of `keys` (u64 order keys, n of them) into idx[0..k)
// sorted ascending by key. Requires 0 < k <= n. Uses only its own scratch.
inline cudaError_t radix_select_k(const u64* keys, std::size_t n, std::size_t k,
                                  i64* out_idx, cudaStream_t s) {
    DevBuf hb; cudaError_t e = hb.alloc(256 * sizeof(u64));
    if (e != cudaSuccess) return e;
    auto* d_hist = static_cast<u64*>(hb.p);
    u64 h[256];
    u64 prefix = 0; std::size_t want = k;              // want-th smallest among prefix matches
    const int grid = static_cast<int>(std::min<std::size_t>((n + 255) / 256, 4096));
    for (int pass = 0; pass < 8; ++pass) {
        const int shift = 56 - 8 * pass;
        if ((e = cudaMemsetAsync(d_hist, 0, 256 * sizeof(u64), s)) != cudaSuccess) return e;
        radix_hist_kernel<<<grid, 256, 0, s>>>(keys, n, prefix, shift, d_hist);
        if ((e = cudaGetLastError()) != cudaSuccess) return e;
        if ((e = cudaMemcpyAsync(h, d_hist, sizeof(h), cudaMemcpyDeviceToHost, s)) != cudaSuccess) return e;
        if ((e = cudaStreamSynchronize(s)) != cudaSuccess) return e;
        std::size_t acc = 0; int bin = 255;
        for (int b = 0; b < 256; ++b) { if (acc + h[b] >= want) { bin = b; break; } acc += h[b]; }
        want -= acc;
        prefix = (prefix << 8) | static_cast<u64>(bin);
    }
    const u64 T = prefix;
    auto cnt = thrust::make_counting_iterator<i64>(0);
    // strictly smaller than T: all of them
    std::size_t n_lt = 0;
    if ((e = host_sum(thrust::make_transform_iterator(cnt, KeyLessU64{keys, T}), n, &n_lt, s)) != cudaSuccess) return e;
    if ((e = select_flagged(cnt, thrust::make_transform_iterator(cnt, KeyLessFlag{keys, T}), out_idx, n, s)) != cudaSuccess) return e;
    if (n_lt < k) {
        // ties at T: take the first (k - n_lt) in index order
        const std::size_t need = k - n_lt;
        std::size_t n_eq = 0;
        if ((e = host_sum(thrust::make_transform_iterator(cnt, KeyEqU64{keys, T}), n, &n_eq, s)) != cudaSuccess) return e;
        DevBuf eb; if ((e = eb.alloc(n_eq * sizeof(i64))) != cudaSuccess) return e;
        auto* d_eq = static_cast<i64*>(eb.p);
        if ((e = select_flagged(cnt, thrust::make_transform_iterator(cnt, KeyEqFlag{keys, T}), d_eq, n, s)) != cudaSuccess) return e;
        if ((e = cudaMemcpyAsync(out_idx + n_lt, d_eq, need * sizeof(i64),
                                 cudaMemcpyDeviceToDevice, s)) != cudaSuccess) return e;
    }
    // order the k winners by key (ties by index, unspecified anyway)
    DevBuf kb; if ((e = kb.alloc(k * sizeof(u64))) != cudaSuccess) return e;
    auto* d_k = static_cast<u64*>(kb.p);
    if ((e = gpudb_cuda_ops::gather(out_idx, k, keys, d_k, s)) != cudaSuccess) return e;
    return gpudb_cuda_ops::sort_pairs_u64(d_k, out_idx, k, s);
}

template <typename A>
cudaError_t count_survivors(const A* agg, std::size_t n, int cmp, A t, std::size_t* h_out, cudaStream_t s) {
    if (cmp == 0) { *h_out = n; return cudaSuccess; }
    auto cnt = thrust::make_counting_iterator<i64>(0);
    return host_sum(thrust::make_transform_iterator(cnt, KeepU64<A>{agg, Keep<A>{cmp, t}}), n, h_out, s);
}

// Writes exactly n_out rows: the survivors of cmp (key order) when topk == 0,
// else the first n_out of the survivors sorted by aggregate. cnt may be
// null (count op: the aggregate IS the count). Caller sized the outputs.
template <typename A>
cudaError_t apply_filter(const i64* keys, const A* agg, const i64* cnt, std::size_t n,
                         int cmp, A t, std::size_t topk, bool desc, std::size_t n_out,
                         i64* out_keys, A* out_agg, i64* out_cnt, cudaStream_t s) {
    const bool has_cnt = (cnt != nullptr);
    cudaError_t e;
    DevBuf sk, sa, sc;                       // survivors (only when cmp active and top-k follows)
    const i64* k_in = keys; const A* a_in = agg; const i64* c_in = cnt;
    std::size_t n_surv = n;
    auto ci = thrust::make_counting_iterator<i64>(0);
    if (cmp != 0) {
        if ((e = count_survivors(agg, n, cmp, t, &n_surv, s)) != cudaSuccess) return e;
        i64* dk = (topk == 0) ? out_keys : nullptr;
        A*   da = (topk == 0) ? out_agg  : nullptr;
        i64* dc = (topk == 0) ? out_cnt  : nullptr;
        if (topk != 0) {
            if ((e = sk.alloc(n_surv * sizeof(i64))) != cudaSuccess) return e;
            if ((e = sa.alloc(n_surv * sizeof(A)))   != cudaSuccess) return e;
            if (has_cnt && (e = sc.alloc(n_surv * sizeof(i64))) != cudaSuccess) return e;
            dk = static_cast<i64*>(sk.p); da = static_cast<A*>(sa.p);
            dc = has_cnt ? static_cast<i64*>(sc.p) : nullptr;
        }
        auto flags = thrust::make_transform_iterator(ci, KeepFlag<A>{agg, Keep<A>{cmp, t}});
        if (has_cnt) {
            auto first = thrust::make_zip_iterator(thrust::make_tuple(keys, agg, cnt));
            auto out   = thrust::make_zip_iterator(thrust::make_tuple(dk, da, dc));
            if ((e = select_flagged(first, flags, out, n, s)) != cudaSuccess) return e;
        } else {
            auto first = thrust::make_zip_iterator(thrust::make_tuple(keys, agg));
            auto out   = thrust::make_zip_iterator(thrust::make_tuple(dk, da));
            if ((e = select_flagged(first, flags, out, n, s)) != cudaSuccess) return e;
        }
        if (topk == 0) return cudaGetLastError();
        k_in = dk; a_in = da; c_in = dc;
    } else if (topk == 0) {
        // No filter at all: plain copies (caller normally avoids this path).
        if ((e = cudaMemcpyAsync(out_keys, keys, n * sizeof(i64), cudaMemcpyDeviceToDevice, s)) != cudaSuccess) return e;
        if ((e = cudaMemcpyAsync(out_agg,  agg,  n * sizeof(A),   cudaMemcpyDeviceToDevice, s)) != cudaSuccess) return e;
        if (has_cnt && (e = cudaMemcpyAsync(out_cnt, cnt, n * sizeof(i64), cudaMemcpyDeviceToDevice, s)) != cudaSuccess) return e;
        return cudaGetLastError();
    }
    // top-k over n_surv survivors, on order keys (one total order, NaN greatest).
    DevBuf kb, ix;
    if ((e = kb.alloc(n_surv * sizeof(u64))) != cudaSuccess) return e;
    auto* okeys = static_cast<u64*>(kb.p);
    if (desc) e = gpudb_cuda_ops::transform(a_in, okeys, n_surv, InvOrderKey<A>{}, s);
    else      e = gpudb_cuda_ops::transform(a_in, okeys, n_surv, OrderKey<A>{}, s);
    if (e != cudaSuccess) return e;
    if (n_out * 8 <= n_surv) {
        // Small k: radix-select (8 histogram passes + one compaction).
        if ((e = ix.alloc(n_out * sizeof(i64))) != cudaSuccess) return e;
        i64* idx = static_cast<i64*>(ix.p);
        if ((e = radix_select_k(okeys, n_surv, n_out, idx, s)) != cudaSuccess) return e;
        if ((e = gpudb_cuda_ops::gather(idx, n_out, k_in, out_keys, s)) != cudaSuccess) return e;
        if ((e = gpudb_cuda_ops::gather(idx, n_out, a_in, out_agg, s)) != cudaSuccess) return e;
        if (has_cnt && (e = gpudb_cuda_ops::gather(idx, n_out, c_in, out_cnt, s)) != cudaSuccess) return e;
        return cudaGetLastError();
    }
    // Large k: full radix sort of (order key, index).
    if ((e = ix.alloc(n_surv * sizeof(i64))) != cudaSuccess) return e;
    i64* idx = static_cast<i64*>(ix.p);
    if ((e = gpudb_cuda_ops::sequence(idx, n_surv, s)) != cudaSuccess) return e;
    if ((e = gpudb_cuda_ops::sort_pairs_u64(okeys, idx, n_surv, s)) != cudaSuccess) return e;
    if ((e = gpudb_cuda_ops::gather(idx, n_out, k_in, out_keys, s)) != cudaSuccess) return e;
    if ((e = gpudb_cuda_ops::gather(idx, n_out, a_in, out_agg, s)) != cudaSuccess) return e;
    if (has_cnt && (e = gpudb_cuda_ops::gather(idx, n_out, c_in, out_cnt, s)) != cudaSuccess) return e;
    return cudaGetLastError();
}

} // namespace

extern "C" {

// Number of distinct runs in an ascending-sorted i64 array (0 for n == 0).
// Synchronizes the stream; the count lands on the host.
cudaError_t gpudb_cuda_sorted_run_count(const i64* d_sorted, std::size_t n,
                                        std::size_t* h_runs, cudaStream_t s) {
    auto cnt = thrust::make_counting_iterator<i64>(0);
    return host_sum(thrust::make_transform_iterator(cnt, RunStart{d_sorted}), n, h_runs, s);
}

// SUM(vals) + COUNT(*) per run of d_sorted; outputs sized n_groups exactly.
cudaError_t gpudb_cuda_groupby_sum_i64(const i64* d_sorted, const i64* d_perm,
                                       const i64* d_vals, std::size_t n,
                                       i64* out_keys, i64* out_sums, i64* out_counts,
                                       std::size_t* h_runs, cudaStream_t s) {
    auto first = thrust::make_transform_iterator(thrust::counting_iterator<i64>(0),
                                                 PermValI64{d_vals, d_perm});
    return reduce_runs<u64>(d_sorted, n, first, AddU64Pair{}, out_keys,
                            reinterpret_cast<u64*>(out_sums), out_counts, h_runs, s);
}

cudaError_t gpudb_cuda_groupby_sum_f64(const i64* d_sorted, const i64* d_perm,
                                       const double* d_vals, std::size_t n,
                                       i64* out_keys, double* out_sums, i64* out_counts,
                                       std::size_t* h_runs, cudaStream_t s) {
    auto first = thrust::make_transform_iterator(thrust::counting_iterator<i64>(0),
                                                 PermValF64{d_vals, d_perm});
    return reduce_runs<double>(d_sorted, n, first, AddF64Pair{}, out_keys, out_sums, out_counts, h_runs, s);
}

// COUNT(*) per run of d_sorted (keys only; no permutation needed).
cudaError_t gpudb_cuda_groupby_count(const i64* d_sorted, std::size_t n,
                                     i64* out_keys, i64* out_counts,
                                     std::size_t* h_runs, cudaStream_t s) {
    DevBuf nr; cudaError_t e = nr.alloc(sizeof(u64));
    if (e != cudaSuccess) return e;
    e = with_temp([&](void* tmp, std::size_t& b) {
        return cub::DeviceReduce::ReduceByKey(tmp, b, d_sorted, out_keys,
                                              thrust::constant_iterator<u64>(1ULL),
                                              reinterpret_cast<u64*>(out_counts),
                                              static_cast<u64*>(nr.p), AddU64{}, n, s);
    });
    if (e != cudaSuccess) return e;
    u64 runs = 0;
    if ((e = cudaMemcpyAsync(&runs, nr.p, sizeof(u64), cudaMemcpyDeviceToHost, s)) != cudaSuccess) return e;
    if ((e = cudaStreamSynchronize(s)) != cudaSuccess) return e;
    *h_runs = static_cast<std::size_t>(runs);
    return cudaSuccess;
}

// f64 sort cache: d_sorted <- vals sorted ascending (NaN last), d_perm <- the
// original indices in that order. Radix sort on order-preserving u64 keys
// built in d_sorted itself; the sorted doubles are then gathered over the
// (no longer needed) keys in place, so no extra n-sized buffer is required —
// peak extra memory is the sort's own scratch.
cudaError_t gpudb_cuda_sort_f64_perm(const double* d_vals, double* d_sorted,
                                     i64* d_perm, std::size_t n, cudaStream_t s) {
    auto* k = reinterpret_cast<u64*>(d_sorted);
    cudaError_t e;
    if ((e = gpudb_cuda_ops::transform(d_vals, k, n, F64OrderKey{}, s)) != cudaSuccess) return e;
    if ((e = gpudb_cuda_ops::sequence(d_perm, n, s)) != cudaSuccess) return e;
    if ((e = gpudb_cuda_ops::sort_pairs_u64(k, d_perm, n, s)) != cudaSuccess) return e;
    if ((e = gpudb_cuda_ops::gather(d_perm, n, d_vals, d_sorted, s)) != cudaSuccess) return e;
    return cudaStreamSynchronize(s);
}

cudaError_t gpudb_cuda_groupby_survivors_i64(const i64* agg, std::size_t n, int cmp, i64 t,
                                             std::size_t* h_out, cudaStream_t s) {
    return count_survivors<i64>(agg, n, cmp, t, h_out, s);
}
cudaError_t gpudb_cuda_groupby_survivors_f64(const double* agg, std::size_t n, int cmp, double t,
                                             std::size_t* h_out, cudaStream_t s) {
    return count_survivors<double>(agg, n, cmp, t, h_out, s);
}
cudaError_t gpudb_cuda_groupby_filter_i64(const i64* keys, const i64* agg, const i64* cnt, std::size_t n,
                                          int cmp, i64 t, std::size_t topk, int desc, std::size_t n_out,
                                          i64* out_keys, i64* out_agg, i64* out_cnt, cudaStream_t s) {
    return apply_filter<i64>(keys, agg, cnt, n, cmp, t, topk, desc != 0, n_out, out_keys, out_agg, out_cnt, s);
}
cudaError_t gpudb_cuda_groupby_filter_f64(const i64* keys, const double* agg, const i64* cnt, std::size_t n,
                                          int cmp, double t, std::size_t topk, int desc, std::size_t n_out,
                                          i64* out_keys, double* out_agg, i64* out_cnt, cudaStream_t s) {
    return apply_filter<double>(keys, agg, cnt, n, cmp, t, topk, desc != 0, n_out, out_keys, out_agg, out_cnt, s);
}

} // extern "C"
