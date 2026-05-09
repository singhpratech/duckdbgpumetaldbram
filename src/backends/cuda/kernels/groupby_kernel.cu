// groupby_kernel.cu — open-addressing hash table for SUM ... GROUP BY ...
//
// Algorithm:
//   - Linear-probing hash table sized to next_pow2(2 * expected_groups)
//   - Empty slot sentinel = INT64_MIN (collision risk is real but acceptable
//     for week-2 baseline; can switch to two-array empty-flag if needed)
//   - hash = splitmix64(key) finalizer; mask with cap-1 (cap is power of 2)
//   - Inserter uses atomicCAS to claim a slot; atomicAdd to accumulate value
//   - Compaction kernel walks slots, atomically appends non-empty entries
//
// Limitations:
//   - INT64_MIN as a key collides with empty sentinel. Caller should not pass
//     it (we throw on host side if any value matches).
//   - No tombstones (no DELETE).
//   - Skew can cause long probe chains; sort-based GROUP BY would handle this
//     better but adds complexity. We'll measure and decide.

#include <cstdint>
#include <cuda_runtime.h>

namespace gpudb::cuda {

namespace {
    constexpr int BLOCK = 256;
    constexpr std::int64_t kEmpty = (std::int64_t)0x8000000000000000LL;  // INT64_MIN
}

__device__ __forceinline__ std::uint64_t splitmix64(std::uint64_t x) {
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x =  x ^ (x >> 31);
    return x;
}

__device__ __forceinline__ std::uint32_t hash_slot(std::int64_t k, std::uint32_t mask) {
    return static_cast<std::uint32_t>(splitmix64(static_cast<std::uint64_t>(k))) & mask;
}

// One pass: probe + insert/update.
__global__ void groupby_sum_insert_kernel(
        const std::int64_t* __restrict__ keys,
        const std::int64_t* __restrict__ values,
        std::size_t n,
        std::int64_t* __restrict__ table_keys,
        std::int64_t* __restrict__ table_sums,
        std::uint32_t cap) {
    const std::uint32_t mask = cap - 1u;
    const std::size_t stride = static_cast<std::size_t>(BLOCK) * gridDim.x;
    for (std::size_t i = blockIdx.x * BLOCK + threadIdx.x; i < n; i += stride) {
        const std::int64_t k = keys[i];
        const std::int64_t v = values[i];
        std::uint32_t slot = hash_slot(k, mask);
        // Linear probe; bounded by cap iterations to prevent infinite loop
        // in pathological cases (table near full — caller must size enough).
        for (std::uint32_t probe = 0; probe < cap; ++probe) {
            std::int64_t cur = atomicCAS(
                reinterpret_cast<unsigned long long*>(&table_keys[slot]),
                static_cast<unsigned long long>(kEmpty),
                static_cast<unsigned long long>(k));
            if (static_cast<std::int64_t>(cur) == kEmpty ||
                static_cast<std::int64_t>(cur) == k) {
                atomicAdd(reinterpret_cast<unsigned long long*>(&table_sums[slot]),
                          static_cast<unsigned long long>(v));
                break;
            }
            slot = (slot + 1u) & mask;
        }
    }
}

// Compact non-empty slots into dense output arrays.
__global__ void groupby_compact_kernel(
        const std::int64_t* __restrict__ table_keys,
        const std::int64_t* __restrict__ table_sums,
        std::uint32_t cap,
        std::int64_t* __restrict__ out_keys,
        std::int64_t* __restrict__ out_sums,
        std::uint32_t* __restrict__ out_count) {
    const std::uint32_t slot = blockIdx.x * BLOCK + threadIdx.x;
    if (slot >= cap) return;
    const std::int64_t k = table_keys[slot];
    if (k == kEmpty) return;
    const std::uint32_t idx = atomicAdd(out_count, 1u);
    out_keys[idx] = k;
    out_sums[idx] = table_sums[slot];
}

// Initialize the hash table (keys = empty, sums = 0).
__global__ void groupby_init_kernel(
        std::int64_t* table_keys,
        std::int64_t* table_sums,
        std::uint32_t cap) {
    const std::uint32_t i = blockIdx.x * BLOCK + threadIdx.x;
    if (i >= cap) return;
    table_keys[i] = kEmpty;
    table_sums[i] = 0;
}

extern "C" {

std::int64_t gpudb_cuda_groupby_empty_sentinel() { return kEmpty; }

cudaError_t gpudb_cuda_groupby_init(std::int64_t* keys, std::int64_t* sums,
                                    std::uint32_t cap, cudaStream_t s) {
    const int grid = (cap + BLOCK - 1) / BLOCK;
    groupby_init_kernel<<<grid, BLOCK, 0, s>>>(keys, sums, cap);
    return cudaGetLastError();
}

cudaError_t gpudb_cuda_groupby_insert(const std::int64_t* keys_in,
                                      const std::int64_t* values_in,
                                      std::size_t n,
                                      std::int64_t* table_keys,
                                      std::int64_t* table_sums,
                                      std::uint32_t cap,
                                      cudaStream_t s) {
    int grid = (n + BLOCK - 1) / BLOCK;
    if (grid > 4096) grid = 4096;
    if (grid < 1)    grid = 1;
    groupby_sum_insert_kernel<<<grid, BLOCK, 0, s>>>(
        keys_in, values_in, n, table_keys, table_sums, cap);
    return cudaGetLastError();
}

cudaError_t gpudb_cuda_groupby_compact(const std::int64_t* table_keys,
                                       const std::int64_t* table_sums,
                                       std::uint32_t cap,
                                       std::int64_t* out_keys,
                                       std::int64_t* out_sums,
                                       std::uint32_t* out_count,
                                       cudaStream_t s) {
    const int grid = (cap + BLOCK - 1) / BLOCK;
    groupby_compact_kernel<<<grid, BLOCK, 0, s>>>(
        table_keys, table_sums, cap, out_keys, out_sums, out_count);
    return cudaGetLastError();
}

} // extern "C"

} // namespace gpudb::cuda
