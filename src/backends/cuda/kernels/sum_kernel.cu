// sum_kernel.cu — week-1 CUDA reductions over int64 / float64 columns.
//
// Implementation notes:
// - We use a two-pass reduction: per-block reduction into a small partials
//   array, then a second kernel reduces partials. For week 1 this is
//   simpler than CUB and avoids the dependency. Will swap to CUB later.
// - Grid-stride loop covers arbitrary n.
// - 256 threads per block matches Ada/Hopper warp scheduling sweet spot.

#include <cstdint>
#include <cuda_runtime.h>

namespace gpudb::cuda {

namespace {
    constexpr int BLOCK = 256;
}

// -------- int64 sum --------
__global__ void sum_i64_kernel(const std::int64_t* __restrict__ in,
                               std::int64_t* __restrict__ partials,
                               std::size_t n) {
    __shared__ std::int64_t shm[BLOCK];
    std::int64_t local = 0;
    for (std::size_t i = blockIdx.x * BLOCK + threadIdx.x;
         i < n;
         i += static_cast<std::size_t>(BLOCK) * gridDim.x) {
        local += in[i];
    }
    shm[threadIdx.x] = local;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) shm[threadIdx.x] += shm[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) partials[blockIdx.x] = shm[0];
}

__global__ void sum_partials_i64_kernel(const std::int64_t* __restrict__ partials,
                                        std::int64_t* __restrict__ out,
                                        int n_partials) {
    __shared__ std::int64_t shm[BLOCK];
    std::int64_t local = 0;
    for (int i = threadIdx.x; i < n_partials; i += BLOCK) local += partials[i];
    shm[threadIdx.x] = local;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) shm[threadIdx.x] += shm[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[0] = shm[0];
}

// -------- int64 min --------
__device__ inline std::int64_t i64min(std::int64_t a, std::int64_t b) {
    return a < b ? a : b;
}
__device__ inline std::int64_t i64max(std::int64_t a, std::int64_t b) {
    return a > b ? a : b;
}

__global__ void min_i64_kernel(const std::int64_t* __restrict__ in,
                               std::int64_t* __restrict__ partials,
                               std::size_t n,
                               std::int64_t init) {
    __shared__ std::int64_t shm[BLOCK];
    std::int64_t local = init;
    for (std::size_t i = blockIdx.x * BLOCK + threadIdx.x;
         i < n;
         i += static_cast<std::size_t>(BLOCK) * gridDim.x) {
        local = i64min(local, in[i]);
    }
    shm[threadIdx.x] = local;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) shm[threadIdx.x] = i64min(shm[threadIdx.x], shm[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) partials[blockIdx.x] = shm[0];
}

__global__ void max_i64_kernel(const std::int64_t* __restrict__ in,
                               std::int64_t* __restrict__ partials,
                               std::size_t n,
                               std::int64_t init) {
    __shared__ std::int64_t shm[BLOCK];
    std::int64_t local = init;
    for (std::size_t i = blockIdx.x * BLOCK + threadIdx.x;
         i < n;
         i += static_cast<std::size_t>(BLOCK) * gridDim.x) {
        local = i64max(local, in[i]);
    }
    shm[threadIdx.x] = local;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) shm[threadIdx.x] = i64max(shm[threadIdx.x], shm[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) partials[blockIdx.x] = shm[0];
}

__global__ void min_partials_i64_kernel(const std::int64_t* p, std::int64_t* out, int n) {
    __shared__ std::int64_t shm[BLOCK];
    std::int64_t local = p[0];
    for (int i = threadIdx.x; i < n; i += BLOCK) local = i64min(local, p[i]);
    shm[threadIdx.x] = local;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) shm[threadIdx.x] = i64min(shm[threadIdx.x], shm[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) out[0] = shm[0];
}

__global__ void max_partials_i64_kernel(const std::int64_t* p, std::int64_t* out, int n) {
    __shared__ std::int64_t shm[BLOCK];
    std::int64_t local = p[0];
    for (int i = threadIdx.x; i < n; i += BLOCK) local = i64max(local, p[i]);
    shm[threadIdx.x] = local;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) shm[threadIdx.x] = i64max(shm[threadIdx.x], shm[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) out[0] = shm[0];
}

// -------- f64 sum --------
__global__ void sum_f64_kernel(const double* __restrict__ in,
                               double* __restrict__ partials,
                               std::size_t n) {
    __shared__ double shm[BLOCK];
    double local = 0.0;
    for (std::size_t i = blockIdx.x * BLOCK + threadIdx.x;
         i < n;
         i += static_cast<std::size_t>(BLOCK) * gridDim.x) {
        local += in[i];
    }
    shm[threadIdx.x] = local;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) shm[threadIdx.x] += shm[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) partials[blockIdx.x] = shm[0];
}

__global__ void sum_partials_f64_kernel(const double* p, double* out, int n) {
    __shared__ double shm[BLOCK];
    double local = 0.0;
    for (int i = threadIdx.x; i < n; i += BLOCK) local += p[i];
    shm[threadIdx.x] = local;
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) shm[threadIdx.x] += shm[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[0] = shm[0];
}

// Public launchers with trivial C-linkage signatures.
extern "C" {

int gpudb_cuda_grid_for(std::size_t n) {
    // Small launches: clamp grid for short inputs.
    int g = static_cast<int>((n + BLOCK - 1) / BLOCK);
    if (g < 1) g = 1;
    if (g > 4096) g = 4096;
    return g;
}

cudaError_t gpudb_cuda_sum_i64(const std::int64_t* d_in, std::size_t n,
                               std::int64_t* d_partials, std::int64_t* d_out,
                               int grid, cudaStream_t s) {
    sum_i64_kernel<<<grid, BLOCK, 0, s>>>(d_in, d_partials, n);
    sum_partials_i64_kernel<<<1, BLOCK, 0, s>>>(d_partials, d_out, grid);
    return cudaGetLastError();
}

cudaError_t gpudb_cuda_min_i64(const std::int64_t* d_in, std::size_t n,
                               std::int64_t* d_partials, std::int64_t* d_out,
                               std::int64_t init, int grid, cudaStream_t s) {
    min_i64_kernel<<<grid, BLOCK, 0, s>>>(d_in, d_partials, n, init);
    min_partials_i64_kernel<<<1, BLOCK, 0, s>>>(d_partials, d_out, grid);
    return cudaGetLastError();
}

cudaError_t gpudb_cuda_max_i64(const std::int64_t* d_in, std::size_t n,
                               std::int64_t* d_partials, std::int64_t* d_out,
                               std::int64_t init, int grid, cudaStream_t s) {
    max_i64_kernel<<<grid, BLOCK, 0, s>>>(d_in, d_partials, n, init);
    max_partials_i64_kernel<<<1, BLOCK, 0, s>>>(d_partials, d_out, grid);
    return cudaGetLastError();
}

cudaError_t gpudb_cuda_sum_f64(const double* d_in, std::size_t n,
                               double* d_partials, double* d_out,
                               int grid, cudaStream_t s) {
    sum_f64_kernel<<<grid, BLOCK, 0, s>>>(d_in, d_partials, n);
    sum_partials_f64_kernel<<<1, BLOCK, 0, s>>>(d_partials, d_out, grid);
    return cudaGetLastError();
}

} // extern "C"

} // namespace gpudb::cuda
