// cub_ops.cuh — CUB device primitives on EXPLICIT temp storage, plus the tiny
// element-wise kernels the resident paths need. Nothing here allocates
// behind our back: every buffer is a cudaMalloc we own and free ourselves,
// and a free that fails (sticky device fault) is ignored rather than thrown
// from a destructor — so a fault surfaces to the host wrapper as the real
// cudaError_t instead of std::terminate.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

namespace gpudb_cuda_ops {

using u64 = unsigned long long;
using i64 = std::int64_t;

// How much device memory the last refused allocation wanted, and how much the
// device had. A CUB call's temp storage is knowable BEFORE the launch — the
// size query is the first half of the two-call protocol below — so an
// operator that cannot fit can say what it needed instead of surfacing a bare
// "out of memory" that tells the caller nothing it can act on.
//
// Written only on the failing path, read by the host wrapper when it formats
// the error, so a stale value is never read as a fresh one.
inline std::size_t& last_scratch_need() { static thread_local std::size_t v = 0; return v; }
inline std::size_t& last_scratch_free() { static thread_local std::size_t v = 0; return v; }

struct DevBuf {
    void* p = nullptr;
    std::size_t bytes = 0;
    cudaError_t alloc(std::size_t b) {
        release();
        bytes = b;
        if (!b) return cudaSuccess;
        const cudaError_t e = cudaMalloc(&p, b);
        if (e == cudaErrorMemoryAllocation) {
            // Record what was wanted before the caller unwinds: the size is
            // the only part of an allocation refusal anyone can act on.
            std::size_t freeb = 0, totalb = 0;
            if (cudaMemGetInfo(&freeb, &totalb) != cudaSuccess) freeb = 0;
            last_scratch_need() = b;
            last_scratch_free() = freeb;
            cudaGetLastError();            // handled here, not a sticky fault
        }
        return e;
    }
    void release() { if (p) { (void)cudaFree(p); p = nullptr; } bytes = 0; }
    ~DevBuf() { release(); }
    DevBuf() = default;
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
};

// Run a CUB call twice: size query, then for real on temp storage we own.
template <typename F>
cudaError_t with_temp(F&& call) {
    std::size_t bytes = 0;
    cudaError_t e = call(static_cast<void*>(nullptr), bytes);
    if (e != cudaSuccess) return e;
    DevBuf tmp;
    if ((e = tmp.alloc(bytes ? bytes : 1)) != cudaSuccess) {
        if (e == cudaErrorMemoryAllocation) {
            std::size_t freeb = 0, totalb = 0;
            if (cudaMemGetInfo(&freeb, &totalb) != cudaSuccess) freeb = 0;
            last_scratch_need() = bytes;
            last_scratch_free() = freeb;
            cudaGetLastError();            // clear the sticky flag: this is handled
        }
        return e;
    }
    return call(tmp.p, bytes);
}

constexpr int kBlock = 256;
inline unsigned grid_for(std::size_t n) {
    const std::size_t g = (n + kBlock - 1) / kBlock;
    return static_cast<unsigned>(g > 65535u * 16u ? 65535u * 16u : (g ? g : 1));
}

template <typename T, typename F>
__global__ void transform_kernel(const T* __restrict__ in, u64* __restrict__ out,
                                 std::size_t n, F f) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride) out[i] = f(in[i]);
}
template <typename T, typename F>
cudaError_t transform(const T* in, u64* out, std::size_t n, F f, cudaStream_t s) {
    if (n) transform_kernel<T, F><<<grid_for(n), kBlock, 0, s>>>(in, out, n, f);
    return cudaGetLastError();
}

__global__ inline void sequence_kernel(i64* __restrict__ out, std::size_t n) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride) out[i] = static_cast<i64>(i);
}
inline cudaError_t sequence(i64* out, std::size_t n, cudaStream_t s) {
    if (n) sequence_kernel<<<grid_for(n), kBlock, 0, s>>>(out, n);
    return cudaGetLastError();
}

template <typename T>
__global__ void gather_kernel(const i64* __restrict__ idx, std::size_t n,
                              const T* __restrict__ src, T* __restrict__ dst) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride) dst[i] = src[idx[i]];
}
template <typename T>
cudaError_t gather(const i64* idx, std::size_t n, const T* src, T* dst, cudaStream_t s) {
    if (n) gather_kernel<T><<<grid_for(n), kBlock, 0, s>>>(idx, n, src, dst);
    return cudaGetLastError();
}

// Sum of a transform over [0, n) (used for run counts and survivor counts).
template <typename It>
cudaError_t sum_u64(It first, std::size_t n, u64* d_out, cudaStream_t s) {
    return with_temp([&](void* tmp, std::size_t& b) {
        return cub::DeviceReduce::Sum(tmp, b, first, d_out, n, s);
    });
}

// Ascending radix sort of (u64 key, i64 value) pairs, in place on the
// caller's buffers (an alternate buffer pair is allocated here).
inline cudaError_t sort_pairs_u64(u64* keys, i64* vals, std::size_t n, cudaStream_t s) {
    if (n < 2) return cudaSuccess;
    DevBuf ka, va;
    cudaError_t e;
    if ((e = ka.alloc(n * sizeof(u64))) != cudaSuccess) return e;
    if ((e = va.alloc(n * sizeof(i64))) != cudaSuccess) return e;
    cub::DoubleBuffer<u64> dk(keys, static_cast<u64*>(ka.p));
    cub::DoubleBuffer<i64> dv(vals, static_cast<i64*>(va.p));
    e = with_temp([&](void* tmp, std::size_t& b) {
        return cub::DeviceRadixSort::SortPairs(tmp, b, dk, dv, n, 0, 64, s);
    });
    if (e != cudaSuccess) return e;
    // Results may be in the alternate buffers: copy back if so.
    if (dk.Current() != keys) {
        if ((e = cudaMemcpyAsync(keys, dk.Current(), n * sizeof(u64),
                                 cudaMemcpyDeviceToDevice, s)) != cudaSuccess) return e;
    }
    if (dv.Current() != vals) {
        if ((e = cudaMemcpyAsync(vals, dv.Current(), n * sizeof(i64),
                                 cudaMemcpyDeviceToDevice, s)) != cudaSuccess) return e;
    }
    return cudaStreamSynchronize(s);   // the alternates are freed on return
}

} // namespace gpudb_cuda_ops
