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
