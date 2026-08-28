// join_kernel.cu — fused resident equi-join kernels (v0.5).
//
// Algorithm matches the CPU/Metal executable spec: build keys are sorted once
// (cached on the ResidentColumn together with the original-index permutation,
// so the same cache serves the fused sums AND the row-returning join), and
// each probe element binary-searches its multiplicity m in the sorted run.
// Per-JoinKind contribution c = jmult(m, kind); sums accumulate in uint64
// (wrap addition is commutative, so the i64 result bit-matches CPU/Metal
// regardless of reduction order). f64 accumulation order is backend-defined
// per the gpu_backend.hpp tolerance contract — atomicAdd(double) is fine.

#include <cstdint>
#include <cuda_runtime.h>

#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include "cub_ops.cuh"

#include "thrust_errors.cuh"

namespace {

constexpr int BLOCK = 256;

// i64 <-> order-preserving u64 (sign bit flipped), for the radix sort.
struct FlipSign     { __host__ __device__ unsigned long long operator()(std::int64_t v) const { return static_cast<unsigned long long>(v) ^ (1ULL << 63); } };
struct FlipSignBack { __host__ __device__ unsigned long long operator()(unsigned long long k) const { return k ^ (1ULL << 63); } };

using u64 = unsigned long long;

__device__ inline std::size_t lower_bound_i64(const std::int64_t* __restrict__ a,
                                              std::size_t n, std::int64_t key) {
    std::size_t lo = 0, len = n;
    while (len > 0) {
        const std::size_t half = len >> 1;
        const std::size_t mid  = lo + half;
        if (a[mid] < key) { lo = mid + 1; len -= half + 1; }
        else              { len = half; }
    }
    return lo;
}

__device__ inline std::size_t upper_bound_i64(const std::int64_t* __restrict__ a,
                                              std::size_t n, std::int64_t key) {
    std::size_t lo = 0, len = n;
    while (len > 0) {
        const std::size_t half = len >> 1;
        const std::size_t mid  = lo + half;
        if (a[mid] <= key) { lo = mid + 1; len -= half + 1; }
        else               { len = half; }
    }
    return lo;
}

// JoinKind multiplier — must mirror CpuAggregator::join_multiplier exactly.
// kind: 0=INNER, 1=LEFT, 2=SEMI, 3=ANTI (matches JoinKind's underlying values).
__device__ inline u64 jmult(u64 m, int kind) {
    switch (kind) {
        case 0:  return m;
        case 1:  return m ? m : 1ULL;
        case 2:  return m ? 1ULL : 0ULL;
        default: return m ? 0ULL : 1ULL;
    }
}

__device__ inline u64 multiplicity(const std::int64_t* __restrict__ build_sorted,
                                   std::size_t n_build, std::int64_t key) {
    return static_cast<u64>(upper_bound_i64(build_sorted, n_build, key) -
                            lower_bound_i64(build_sorted, n_build, key));
}

__global__ void join_sum_i64_kernel(const std::int64_t* __restrict__ probe,
                                    const std::int64_t* __restrict__ pay,
                                    const std::int64_t* __restrict__ build_sorted,
                                    std::size_t n_probe, std::size_t n_build,
                                    int kind, u64* __restrict__ out_sum,
                                    u64* __restrict__ out_matched) {
    __shared__ u64 s_sum[BLOCK];
    __shared__ u64 s_m[BLOCK];
    u64 sum = 0, matched = 0;
    for (std::size_t i = blockIdx.x * BLOCK + threadIdx.x;
         i < n_probe;
         i += static_cast<std::size_t>(BLOCK) * gridDim.x) {
        const u64 c = jmult(multiplicity(build_sorted, n_build, probe[i]), kind);
        sum     += c * static_cast<u64>(pay[i]);
        matched += c;
    }
    s_sum[threadIdx.x] = sum;
    s_m[threadIdx.x]   = matched;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
            s_m[threadIdx.x]   += s_m[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(out_sum,     s_sum[0]);
        atomicAdd(out_matched, s_m[0]);
    }
}

__global__ void join_sum_f64_kernel(const std::int64_t* __restrict__ probe,
                                    const double* __restrict__ pay,
                                    const std::int64_t* __restrict__ build_sorted,
                                    std::size_t n_probe, std::size_t n_build,
                                    int kind, double* __restrict__ out_sum,
                                    u64* __restrict__ out_matched) {
    __shared__ double s_sum[BLOCK];
    __shared__ u64    s_m[BLOCK];
    double sum = 0.0;
    u64 matched = 0;
    for (std::size_t i = blockIdx.x * BLOCK + threadIdx.x;
         i < n_probe;
         i += static_cast<std::size_t>(BLOCK) * gridDim.x) {
        const u64 c = jmult(multiplicity(build_sorted, n_build, probe[i]), kind);
        if (c) {
            sum     += static_cast<double>(c) * pay[i];
            matched += c;
        }
    }
    s_sum[threadIdx.x] = sum;
    s_m[threadIdx.x]   = matched;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
            s_m[threadIdx.x]   += s_m[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(out_sum,     s_sum[0]);
        atomicAdd(out_matched, s_m[0]);
    }
}

// Row-returning join, pass 1: per-probe output count c (the kind multiplier).
__global__ void join_rows_count_kernel(const std::int64_t* __restrict__ probe,
                                       const std::int64_t* __restrict__ build_sorted,
                                       std::size_t n_probe, std::size_t n_build,
                                       int kind, u64* __restrict__ cnt) {
    for (std::size_t i = blockIdx.x * BLOCK + threadIdx.x;
         i < n_probe;
         i += static_cast<std::size_t>(BLOCK) * gridDim.x) {
        cnt[i] = jmult(multiplicity(build_sorted, n_build, probe[i]), kind);
    }
}

// Row-returning join, pass 2: fill (probe_idx, build_idx) at the scanned
// offsets. perm maps sorted position -> original upload index; build_idx -1
// encodes SQL NULL — same contract as the CPU reference.
__global__ void join_rows_fill_kernel(const std::int64_t* __restrict__ probe,
                                      const std::int64_t* __restrict__ build_sorted,
                                      const std::int64_t* __restrict__ perm,
                                      std::size_t n_probe, std::size_t n_build,
                                      int kind, const u64* __restrict__ offs,
                                      std::int64_t* __restrict__ out_pidx,
                                      std::int64_t* __restrict__ out_bidx) {
    for (std::size_t i = blockIdx.x * BLOCK + threadIdx.x;
         i < n_probe;
         i += static_cast<std::size_t>(BLOCK) * gridDim.x) {
        const std::size_t lo = lower_bound_i64(build_sorted, n_build, probe[i]);
        const std::size_t hi = upper_bound_i64(build_sorted, n_build, probe[i]);
        const u64 m = static_cast<u64>(hi - lo);
        const u64 c = jmult(m, kind);
        if (!c) continue;
        u64 off = offs[i];
        if ((kind == 0 || kind == 1) && m) {
            for (std::size_t j = lo; j < hi; ++j, ++off) {
                out_pidx[off] = static_cast<std::int64_t>(i);
                out_bidx[off] = perm[j];
            }
        } else {
            // LEFT-unmatched, SEMI, ANTI: single row, NULL build side.
            out_pidx[off] = static_cast<std::int64_t>(i);
            out_bidx[off] = -1;
        }
    }
}

} // namespace

extern "C" {

int gpudb_cuda_grid_for(std::size_t n);

// Build the sorted-keys + permutation cache: sorted <- keys (D2D copy),
// perm <- 0..n-1, then sort_by_key. Thrust allocates sort scratch internally,
// so a device-OOM surfaces here — mapped to cudaErrorMemoryAllocation.
cudaError_t gpudb_cuda_join_build_sort(const std::int64_t* d_keys,
                                       std::int64_t* d_sorted,
                                       std::int64_t* d_perm,
                                       std::size_t n, cudaStream_t s) {
    cudaError_t e = cudaMemcpyAsync(d_sorted, d_keys, n * sizeof(std::int64_t),
                                    cudaMemcpyDeviceToDevice, s);
    if (e != cudaSuccess) return e;
    // Sort as order-preserving u64 keys (sign bit flipped) so the radix pass
    // handles negatives; flip back afterwards. Explicit temp storage: a
    // device fault returns its cudaError_t here instead of aborting.
    e = gpudb_cuda_ops::transform(d_sorted, reinterpret_cast<u64*>(d_sorted), n, FlipSign{}, s);
    if (e != cudaSuccess) return e;
    if ((e = gpudb_cuda_ops::sequence(d_perm, n, s)) != cudaSuccess) return e;
    if ((e = gpudb_cuda_ops::sort_pairs_u64(reinterpret_cast<u64*>(d_sorted), d_perm, n, s)) != cudaSuccess) return e;
    e = gpudb_cuda_ops::transform(reinterpret_cast<const u64*>(d_sorted),
                                  reinterpret_cast<u64*>(d_sorted), n, FlipSignBack{}, s);
    if (e != cudaSuccess) return e;
    return cudaStreamSynchronize(s);
}

cudaError_t gpudb_cuda_join_sum_i64(const std::int64_t* d_probe,
                                    const std::int64_t* d_pay,
                                    const std::int64_t* d_build_sorted,
                                    std::size_t n_probe, std::size_t n_build,
                                    int kind, void* d_acc /* [u64 sum, u64 matched] */,
                                    int grid, cudaStream_t s) {
    cudaError_t e = cudaMemsetAsync(d_acc, 0, 2 * sizeof(u64), s);
    if (e != cudaSuccess) return e;
    u64* acc = static_cast<u64*>(d_acc);
    join_sum_i64_kernel<<<grid, BLOCK, 0, s>>>(d_probe, d_pay, d_build_sorted,
                                               n_probe, n_build, kind, acc, acc + 1);
    return cudaGetLastError();
}

cudaError_t gpudb_cuda_join_sum_f64(const std::int64_t* d_probe,
                                    const double* d_pay,
                                    const std::int64_t* d_build_sorted,
                                    std::size_t n_probe, std::size_t n_build,
                                    int kind, void* d_acc /* [f64 sum, u64 matched] */,
                                    int grid, cudaStream_t s) {
    cudaError_t e = cudaMemsetAsync(d_acc, 0, 2 * sizeof(u64), s);
    if (e != cudaSuccess) return e;
    join_sum_f64_kernel<<<grid, BLOCK, 0, s>>>(d_probe, d_pay, d_build_sorted,
                                               n_probe, n_build, kind,
                                               static_cast<double*>(d_acc),
                                               static_cast<u64*>(d_acc) + 1);
    return cudaGetLastError();
}

cudaError_t gpudb_cuda_join_rows_count(const std::int64_t* d_probe,
                                       const std::int64_t* d_build_sorted,
                                       std::size_t n_probe, std::size_t n_build,
                                       int kind, u64* d_cnt,
                                       int grid, cudaStream_t s) {
    join_rows_count_kernel<<<grid, BLOCK, 0, s>>>(d_probe, d_build_sorted,
                                                  n_probe, n_build, kind, d_cnt);
    return cudaGetLastError();
}

// total <- sum(cnt); cnt <- exclusive_scan(cnt) in place. Synchronizes.
cudaError_t gpudb_cuda_join_rows_scan(u64* d_cnt, std::size_t n,
                                      u64* h_total, cudaStream_t s) {
    gpudb_cuda_ops::DevBuf tot;
    cudaError_t e = tot.alloc(sizeof(u64));
    if (e != cudaSuccess) return e;
    if ((e = gpudb_cuda_ops::sum_u64(d_cnt, n, static_cast<u64*>(tot.p), s)) != cudaSuccess) return e;
    e = gpudb_cuda_ops::with_temp([&](void* tmp, std::size_t& b) {
        return cub::DeviceScan::ExclusiveSum(tmp, b, d_cnt, d_cnt, n, s);
    });
    if (e != cudaSuccess) return e;
    if ((e = cudaMemcpyAsync(h_total, tot.p, sizeof(u64), cudaMemcpyDeviceToHost, s)) != cudaSuccess) return e;
    return cudaStreamSynchronize(s);
}

cudaError_t gpudb_cuda_join_rows_fill(const std::int64_t* d_probe,
                                      const std::int64_t* d_build_sorted,
                                      const std::int64_t* d_perm,
                                      std::size_t n_probe, std::size_t n_build,
                                      int kind, const u64* d_offs,
                                      std::int64_t* d_out_pidx,
                                      std::int64_t* d_out_bidx,
                                      int grid, cudaStream_t s) {
    join_rows_fill_kernel<<<grid, BLOCK, 0, s>>>(d_probe, d_build_sorted, d_perm,
                                                 n_probe, n_build, kind, d_offs,
                                                 d_out_pidx, d_out_bidx);
    return cudaGetLastError();
}

} // extern "C"
