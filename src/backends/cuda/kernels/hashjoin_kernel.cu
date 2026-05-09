// hashjoin_kernel.cu — CUDA inner equi-join on i64 keys.
//
// Algorithm (mirrors the GROUP BY hash table almost exactly):
//
//   BUILD: open-addressing linear-probing hash table. Each slot stores
//     (key: i64, build_row_idx: i64). Empty sentinel for the key field
//     is INT64_MIN. Insertion is atomicCAS on the key word; on success
//     we then write the row index into the parallel slot. Multiple build
//     rows with the SAME key occupy DIFFERENT slots — for a multimap
//     join we need every (key, idx) pair distinct, not just unique keys.
//
//   PROBE: each thread walks one probe row, hashing its key, then linear-
//     probing slots. For every slot whose stored key == probe key, we
//     atomically reserve the next free position in the output buffer
//     via atomicAdd on a single counter, and write (probe_idx, build_idx)
//     there. Probe stops when it hits an empty slot (no more matches
//     possible in this chain).
//
// Notes:
//   - INT64_MIN as a build key collides with the empty sentinel. Caller
//     checks on host and throws.
//   - The output buffer has a fixed pre-allocated capacity. If a probe
//     thread's atomicAdd would exceed capacity, it skips writing — the
//     host detects overflow by comparing the post-kernel counter to
//     capacity and re-runs with a larger buffer (rare; we initial-size
//     to a generous fraction of n_probe).
//   - For probe-side capacity sizing, we assume avg matches per probe
//     <= ~4 — adequate for typical foreign-key joins (e.g., TPC-H
//     lineitem ⋈ orders is exactly 1 match per probe row by FK constraint).
//   - Linear probing is bounded by `cap` iterations per probe to prevent
//     infinite loops in pathological cases (table near full).

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

// Initialize the hash table: every key slot becomes empty.
// Index slots can stay uninitialized — they're only read when the matching
// key slot is non-empty, and we always write them before the key.
__global__ void hashjoin_init_kernel(
        std::int64_t* __restrict__ table_keys,
        std::uint32_t cap) {
    const std::uint32_t i = blockIdx.x * BLOCK + threadIdx.x;
    if (i >= cap) return;
    table_keys[i] = kEmpty;
}

// BUILD: insert each (build_key[i], i) into the table.
// Multiple build rows with the same key occupy distinct slots — that's the
// key difference vs GROUP BY where we'd accumulate. Here every (key, idx)
// must survive distinctly so probe can find all matches.
__global__ void hashjoin_build_kernel(
        const std::int64_t* __restrict__ build_keys,
        std::size_t n_build,
        std::int64_t* __restrict__ table_keys,
        std::int64_t* __restrict__ table_idx,
        std::uint32_t cap) {
    const std::uint32_t mask = cap - 1u;
    const std::size_t stride = static_cast<std::size_t>(BLOCK) * gridDim.x;
    for (std::size_t i = blockIdx.x * BLOCK + threadIdx.x; i < n_build; i += stride) {
        const std::int64_t k = build_keys[i];
        std::uint32_t slot = hash_slot(k, mask);
        // Linear probe; bounded by cap iterations to prevent infinite loop
        // in pathological cases (table near full).
        for (std::uint32_t probe = 0; probe < cap; ++probe) {
            std::int64_t cur = atomicCAS(
                reinterpret_cast<unsigned long long*>(&table_keys[slot]),
                static_cast<unsigned long long>(kEmpty),
                static_cast<unsigned long long>(k));
            if (static_cast<std::int64_t>(cur) == kEmpty) {
                // We claimed this slot. Stash the build row index alongside.
                // Single-threaded write to this slot — no atomic needed.
                table_idx[slot] = static_cast<std::int64_t>(i);
                break;
            }
            // Slot already taken (by us-or-someone-else with this-or-other
            // key). Move on. We do NOT short-circuit on cur == k: distinct
            // build rows with the same key need distinct slots.
            slot = (slot + 1u) & mask;
        }
    }
}

// PROBE: for each probe_keys[i], walk linear-probe chain, atomically
// append (i, build_idx) to output for every match.
__global__ void hashjoin_probe_kernel(
        const std::int64_t* __restrict__ probe_keys,
        std::size_t n_probe,
        const std::int64_t* __restrict__ table_keys,
        const std::int64_t* __restrict__ table_idx,
        std::uint32_t cap,
        std::int64_t* __restrict__ out_probe_idx,
        std::int64_t* __restrict__ out_build_idx,
        std::uint32_t* __restrict__ out_count,
        std::uint32_t out_capacity) {
    const std::uint32_t mask = cap - 1u;
    const std::size_t stride = static_cast<std::size_t>(BLOCK) * gridDim.x;
    for (std::size_t i = blockIdx.x * BLOCK + threadIdx.x; i < n_probe; i += stride) {
        const std::int64_t k = probe_keys[i];
        std::uint32_t slot = hash_slot(k, mask);
        for (std::uint32_t probe = 0; probe < cap; ++probe) {
            const std::int64_t tk = table_keys[slot];
            if (tk == kEmpty) break;            // end of chain
            if (tk == k) {
                const std::uint32_t pos = atomicAdd(out_count, 1u);
                if (pos < out_capacity) {
                    out_probe_idx[pos] = static_cast<std::int64_t>(i);
                    out_build_idx[pos] = table_idx[slot];
                }
                // else: overflow — host re-runs with larger buffer.
                // Continue probing in case more matches in chain.
            }
            slot = (slot + 1u) & mask;
        }
    }
}

extern "C" {

std::int64_t gpudb_cuda_hashjoin_empty_sentinel() { return kEmpty; }

cudaError_t gpudb_cuda_hashjoin_init(std::int64_t* table_keys,
                                     std::uint32_t cap, cudaStream_t s) {
    const int grid = (cap + BLOCK - 1) / BLOCK;
    hashjoin_init_kernel<<<grid, BLOCK, 0, s>>>(table_keys, cap);
    return cudaGetLastError();
}

cudaError_t gpudb_cuda_hashjoin_build(const std::int64_t* build_keys,
                                      std::size_t n_build,
                                      std::int64_t* table_keys,
                                      std::int64_t* table_idx,
                                      std::uint32_t cap,
                                      cudaStream_t s) {
    int grid = (n_build + BLOCK - 1) / BLOCK;
    if (grid > 4096) grid = 4096;
    if (grid < 1)    grid = 1;
    hashjoin_build_kernel<<<grid, BLOCK, 0, s>>>(
        build_keys, n_build, table_keys, table_idx, cap);
    return cudaGetLastError();
}

cudaError_t gpudb_cuda_hashjoin_probe(const std::int64_t* probe_keys,
                                      std::size_t n_probe,
                                      const std::int64_t* table_keys,
                                      const std::int64_t* table_idx,
                                      std::uint32_t cap,
                                      std::int64_t* out_probe_idx,
                                      std::int64_t* out_build_idx,
                                      std::uint32_t* out_count,
                                      std::uint32_t out_capacity,
                                      cudaStream_t s) {
    int grid = (n_probe + BLOCK - 1) / BLOCK;
    if (grid > 4096) grid = 4096;
    if (grid < 1)    grid = 1;
    hashjoin_probe_kernel<<<grid, BLOCK, 0, s>>>(
        probe_keys, n_probe, table_keys, table_idx, cap,
        out_probe_idx, out_build_idx, out_count, out_capacity);
    return cudaGetLastError();
}

} // extern "C"

} // namespace gpudb::cuda
