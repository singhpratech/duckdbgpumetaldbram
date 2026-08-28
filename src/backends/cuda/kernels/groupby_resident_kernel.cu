// groupby_resident_kernel.cu — resident GROUP BY / top-k device paths (v0.6).
//
// GROUP BY reuses the v0.5 build-side sort cache on the key column (keys
// sorted ascending + original-index permutation). Values are read through
// the permutation and reduced per key run with reduce_by_key (Thrust over
// CUB DeviceReduce::ReduceByKey), so output rows come out sorted by key —
// the cross-backend contract in gpu_backend.hpp. i64 sums accumulate in
// uint64 (wrap-add commutes -> bit-exact regardless of reduction order);
// f64 sums are backend-ordered under the tolerance contract.
//
// top-k on an f64 column needs its own sort: doubles are mapped to
// order-preserving uint64 keys (NaN greatest, matching DuckDB's total order)
// so the radix path is used, then the sorted values are gathered back.

#include <cstdint>
#include <cuda_runtime.h>

#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#include "thrust_errors.cuh"

namespace {

using u64 = unsigned long long;
using i64 = std::int64_t;

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

} // namespace

extern "C" {

// Number of distinct runs in an ascending-sorted i64 array (0 for n == 0).
// Synchronizes the stream; the count lands on the host.
cudaError_t gpudb_cuda_sorted_run_count(const i64* d_sorted, std::size_t n,
                                        std::size_t* h_runs, cudaStream_t s) {
    if (n == 0) { *h_runs = 0; return cudaSuccess; }
    try {
        *h_runs = static_cast<std::size_t>(
            thrust::unique_count(thrust::cuda::par.on(s), d_sorted, d_sorted + n));
    } catch (...) {
        return gpudb_cuda_detail::map_exception();
    }
    return cudaGetLastError();
}

// SUM(vals) + COUNT(*) per run of d_sorted; outputs sized n_groups exactly.
// Writes the run count actually produced to *h_runs (sanity-checked by host).
cudaError_t gpudb_cuda_groupby_sum_i64(const i64* d_sorted, const i64* d_perm,
                                       const i64* d_vals, std::size_t n,
                                       i64* out_keys, i64* out_sums, i64* out_counts,
                                       std::size_t* h_runs, cudaStream_t s) {
    try {
        auto first = thrust::make_transform_iterator(thrust::counting_iterator<i64>(0),
                                                     PermValI64{d_vals, d_perm});
        auto out_vals = thrust::make_zip_iterator(thrust::make_tuple(
            reinterpret_cast<u64*>(out_sums), reinterpret_cast<u64*>(out_counts)));
        auto ends = thrust::reduce_by_key(thrust::cuda::par.on(s),
                                          d_sorted, d_sorted + n, first,
                                          out_keys, out_vals,
                                          thrust::equal_to<i64>(), AddU64Pair());
        *h_runs = static_cast<std::size_t>(ends.first - out_keys);
    } catch (...) {
        return gpudb_cuda_detail::map_exception();
    }
    return cudaGetLastError();
}

cudaError_t gpudb_cuda_groupby_sum_f64(const i64* d_sorted, const i64* d_perm,
                                       const double* d_vals, std::size_t n,
                                       i64* out_keys, double* out_sums, i64* out_counts,
                                       std::size_t* h_runs, cudaStream_t s) {
    try {
        auto first = thrust::make_transform_iterator(thrust::counting_iterator<i64>(0),
                                                     PermValF64{d_vals, d_perm});
        auto out_vals = thrust::make_zip_iterator(thrust::make_tuple(
            out_sums, reinterpret_cast<u64*>(out_counts)));
        auto ends = thrust::reduce_by_key(thrust::cuda::par.on(s),
                                          d_sorted, d_sorted + n, first,
                                          out_keys, out_vals,
                                          thrust::equal_to<i64>(), AddF64Pair());
        *h_runs = static_cast<std::size_t>(ends.first - out_keys);
    } catch (...) {
        return gpudb_cuda_detail::map_exception();
    }
    return cudaGetLastError();
}

// COUNT(*) per run of d_sorted (keys only; no permutation needed).
cudaError_t gpudb_cuda_groupby_count(const i64* d_sorted, std::size_t n,
                                     i64* out_keys, i64* out_counts,
                                     std::size_t* h_runs, cudaStream_t s) {
    try {
        auto ends = thrust::reduce_by_key(thrust::cuda::par.on(s),
                                          d_sorted, d_sorted + n,
                                          thrust::constant_iterator<u64>(1ULL),
                                          out_keys, reinterpret_cast<u64*>(out_counts));
        *h_runs = static_cast<std::size_t>(ends.first - out_keys);
    } catch (...) {
        return gpudb_cuda_detail::map_exception();
    }
    return cudaGetLastError();
}

// f64 sort cache: d_sorted <- vals sorted ascending (NaN last), d_perm <- the
// original indices in that order. Radix sort on order-preserving u64 keys
// built in d_sorted itself; the sorted doubles are then gathered over the
// (no longer needed) keys in place, so no extra n-sized buffer is required —
// peak extra memory is the sort's own scratch.
cudaError_t gpudb_cuda_sort_f64_perm(const double* d_vals, double* d_sorted,
                                     i64* d_perm, std::size_t n, cudaStream_t s) {
    try {
        auto* k = reinterpret_cast<u64*>(d_sorted);
        thrust::transform(thrust::cuda::par.on(s), d_vals, d_vals + n, k, F64OrderKey());
        thrust::sequence(thrust::cuda::par.on(s), d_perm, d_perm + n);
        thrust::sort_by_key(thrust::cuda::par.on(s), k, k + n, d_perm);
        thrust::gather(thrust::cuda::par.on(s), d_perm, d_perm + n, d_vals, d_sorted);
    } catch (...) {
        return gpudb_cuda_detail::map_exception();
    }
    return cudaStreamSynchronize(s);
}

} // extern "C"

// ---- GroupByFilter on the device (v0.6): cmp on the aggregate, then top-k ----
//
// Survivors are selected with copy_if (order preserved, so the key-sorted
// order survives); top-k radix-sorts (aggregate, index) pairs — primitive
// keys and values, so Thrust takes the radix path — and gathers the first k.
// cmp codes mirror GroupByFilter::Cmp: 0 None, 1 GT, 2 GE, 3 LT, 4 LE.

namespace {

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

struct DevBuf {
    void* p = nullptr;
    cudaError_t alloc(std::size_t bytes) { return bytes ? cudaMalloc(&p, bytes) : cudaSuccess; }
    ~DevBuf() { if (p) cudaFree(p); }
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

template <typename A>
std::size_t count_survivors(const A* agg, std::size_t n, int cmp, A t, cudaStream_t s) {
    if (cmp == 0) return n;
    return static_cast<std::size_t>(
        thrust::count_if(thrust::cuda::par.on(s), agg, agg + n, Keep<A>{cmp, t}));
}

// Writes exactly n_out rows: the survivors of cmp (key order) when topk == 0,
// else the first n_out of the survivors sorted by aggregate. cnt may be
// null (count op: the aggregate IS the count). Caller sized the outputs.
template <typename A>
cudaError_t apply_filter(const i64* keys, const A* agg, const i64* cnt, std::size_t n,
                         int cmp, A t, std::size_t topk, bool desc, std::size_t n_out,
                         i64* out_keys, A* out_agg, i64* out_cnt, cudaStream_t s) {
    auto pol = thrust::cuda::par.on(s);
    const bool has_cnt = (cnt != nullptr);
    DevBuf sk, sa, sc;                       // survivors (only when cmp active)
    const i64* k_in = keys; const A* a_in = agg; const i64* c_in = cnt;
    std::size_t n_surv = n;
    try {
        if (cmp != 0) {
            n_surv = count_survivors(agg, n, cmp, t, s);
            // When there is no top-k the survivors ARE the output: select
            // straight into the caller's buffers.
            i64* dk = (topk == 0) ? out_keys : nullptr;
            A*   da = (topk == 0) ? out_agg  : nullptr;
            i64* dc = (topk == 0) ? out_cnt  : nullptr;
            if (topk != 0) {
                cudaError_t e;
                if ((e = sk.alloc(n_surv * sizeof(i64))) != cudaSuccess) return e;
                if ((e = sa.alloc(n_surv * sizeof(A)))   != cudaSuccess) return e;
                if (has_cnt && (e = sc.alloc(n_surv * sizeof(i64))) != cudaSuccess) return e;
                dk = static_cast<i64*>(sk.p); da = static_cast<A*>(sa.p);
                dc = has_cnt ? static_cast<i64*>(sc.p) : nullptr;
            }
            if (has_cnt) {
                auto first = thrust::make_zip_iterator(thrust::make_tuple(keys, agg, cnt));
                auto out   = thrust::make_zip_iterator(thrust::make_tuple(dk, da, dc));
                thrust::copy_if(pol, first, first + n, agg, out, Keep<A>{cmp, t});
            } else {
                auto first = thrust::make_zip_iterator(thrust::make_tuple(keys, agg));
                auto out   = thrust::make_zip_iterator(thrust::make_tuple(dk, da));
                thrust::copy_if(pol, first, first + n, agg, out, Keep<A>{cmp, t});
            }
            if (topk == 0) return cudaGetLastError();
            k_in = dk; a_in = da; c_in = dc;
        } else if (topk == 0) {
            // No filter at all: plain copies (caller normally avoids this path).
            cudaMemcpyAsync(out_keys, keys, n * sizeof(i64), cudaMemcpyDeviceToDevice, s);
            cudaMemcpyAsync(out_agg,  agg,  n * sizeof(A),   cudaMemcpyDeviceToDevice, s);
            if (has_cnt) cudaMemcpyAsync(out_cnt, cnt, n * sizeof(i64), cudaMemcpyDeviceToDevice, s);
            return cudaGetLastError();
        }
        // top-k over n_surv survivors: radix sort of (order key, index) — the
        // order keys give one total order (NaN greatest) for both directions.
        DevBuf kb, ix;
        cudaError_t e;
        if ((e = kb.alloc(n_surv * sizeof(u64))) != cudaSuccess) return e;
        if ((e = ix.alloc(n_surv * sizeof(i64))) != cudaSuccess) return e;
        auto* okeys = static_cast<u64*>(kb.p);
        i64*  idx   = static_cast<i64*>(ix.p);
        if (desc) thrust::transform(pol, a_in, a_in + n_surv, okeys, InvOrderKey<A>{});
        else      thrust::transform(pol, a_in, a_in + n_surv, okeys, OrderKey<A>{});
        thrust::sequence(pol, idx, idx + n_surv);
        thrust::sort_by_key(pol, okeys, okeys + n_surv, idx);
        thrust::gather(pol, idx, idx + n_out, k_in, out_keys);
        thrust::gather(pol, idx, idx + n_out, a_in, out_agg);
        if (has_cnt) thrust::gather(pol, idx, idx + n_out, c_in, out_cnt);
    } catch (...) {
        return gpudb_cuda_detail::map_exception();
    }
    return cudaGetLastError();
}

} // namespace

extern "C" {

cudaError_t gpudb_cuda_groupby_survivors_i64(const i64* agg, std::size_t n, int cmp, i64 t,
                                             std::size_t* h_out, cudaStream_t s) {
    try { *h_out = count_survivors<i64>(agg, n, cmp, t, s); }
    catch (...) { return gpudb_cuda_detail::map_exception(); }
    return cudaGetLastError();
}
cudaError_t gpudb_cuda_groupby_survivors_f64(const double* agg, std::size_t n, int cmp, double t,
                                             std::size_t* h_out, cudaStream_t s) {
    try { *h_out = count_survivors<double>(agg, n, cmp, t, s); }
    catch (...) { return gpudb_cuda_detail::map_exception(); }
    return cudaGetLastError();
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
