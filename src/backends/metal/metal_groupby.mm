// metal_groupby.mm — Metal GROUP BY (SUM, i64 keys + i64 values).
//
// Apple-native strategy: bitonic sort on GPU + segment-reduce on host.
// The CUDA-style atomic-CAS hash table doesn't port to Apple Silicon
// (no 64-bit atomic_compare_exchange, no 64-bit atomic_fetch_add on
// `device` storage). Sort-then-reduce sidesteps both.
//
// Pipeline:
//   1. Pad N up to N_pad = next_pow2(N). Pad with (INT64_MAX, 0).
//   2. Stage (key, value) buffers as MTLBuffer (storage mode shared = UMA).
//   3. Run bitonic_step kernel for each (k, j) stage of the sort.
//      Total dispatches = log₂(N_pad)·(log₂(N_pad)+1)/2.
//   4. Read sorted keys/values back via the shared buffer pointer (no D2H
//      memcpy — UMA).
//   5. Single-pass host segment-reduce: emit (unique_key, sum_of_values)
//      for each run of equal keys, stopping at the first INT64_MAX
//      padding sentinel.
//
// kernel_ms is reported as the GPU time of the bitonic sort only (GPUStart
// /GPUEndTime on the command buffer that holds all sort stages). The host
// segment-reduce time is rolled into wall_ms.

#include "gpu_backend.hpp"
#include "metal_kernel_sources.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace gpudb {

namespace {

constexpr NSUInteger kBlock = 256;

[[noreturn]] void metal_throw(const char* what, NSError* err) {
    std::ostringstream os;
    os << "Metal " << what;
    if (err) os << ": " << [[err localizedDescription] UTF8String];
    throw std::runtime_error(os.str());
}

std::uint64_t next_pow2_u64(std::uint64_t v) {
    if (v < 2) return 2;
    --v;
    v |= v >> 1; v |= v >> 2;  v |= v >> 4;
    v |= v >> 8; v |= v >> 16; v |= v >> 32;
    return v + 1;
}

class MetalGroupByAggregator final : public GroupByAggregator {
public:
    MetalGroupByAggregator() {
        @autoreleasepool {
            device_ = MTLCreateSystemDefaultDevice();
            if (!device_) throw std::runtime_error("MTLCreateSystemDefaultDevice returned nil");
            queue_ = [device_ newCommandQueue];
            if (!queue_) throw std::runtime_error("Failed to create Metal command queue");

            NSError* err = nil;
            NSString* src = [NSString stringWithUTF8String:metal::kGroupByKernelSource];
            MTLCompileOptions* opts = [MTLCompileOptions new];
            if (@available(macOS 15.0, *)) {
                [opts setLanguageVersion:MTLLanguageVersion3_2];
            } else {
                [opts setLanguageVersion:MTLLanguageVersion3_1];
            }
            id<MTLLibrary> lib = [device_ newLibraryWithSource:src options:opts error:&err];
            if (!lib) metal_throw("compile groupby.metal", err);

            ps_step_        = make_pso(lib, @"bitonic_step_i64");
            ps_pad_         = make_pso(lib, @"bitonic_pad_i64");
            ps_local_sort_  = make_pso(lib, @"bitonic_local_sort_i64");
            ps_local_merge_ = make_pso(lib, @"bitonic_local_merge_i64");
        }
    }

    Backend backend() const noexcept override { return Backend::METAL; }

    std::string device_name() const override {
        @autoreleasepool {
            std::ostringstream os;
            os << [[device_ name] UTF8String]
               << " (Metal, bitonic sort + host segment-reduce)";
            return os.str();
        }
    }

    GroupByResult groupby_sum_i64(const std::int64_t* keys,
                                  const std::int64_t* values,
                                  std::size_t n,
                                  std::size_t /*expected_groups*/) override {
        GroupByResult r;
        r.input_rows = n;
        if (n == 0) return r;

        // Defensive: INT64_MAX is the sort-padding sentinel. Refuse data
        // containing it so we don't accidentally truncate user keys.
        for (std::size_t i = 0; i < n; ++i) {
            if (keys[i] == INT64_MAX) {
                throw std::runtime_error(
                    "key INT64_MAX clashes with sort padding sentinel; not supported");
            }
        }

        // Pad up to at least CHUNK so the local-sort kernel always has full chunks.
        constexpr std::uint64_t kChunk = 512;
        std::uint64_t n_pad64 = next_pow2_u64(static_cast<std::uint64_t>(n));
        if (n_pad64 < kChunk) n_pad64 = kChunk;
        if (n_pad64 > std::numeric_limits<std::uint32_t>::max()) {
            // Bitonic sort needs power-of-2 length; the kernel takes uint32_t for n.
            // For week-1 cap at 2^31 elements; refuse beyond that.
            throw std::runtime_error("input too large for bitonic_step_i64 uint32 indexing");
        }
        const std::uint32_t n_pad  = static_cast<std::uint32_t>(n_pad64);
        const std::uint32_t n_real = static_cast<std::uint32_t>(n);

        @autoreleasepool {
            const auto t_wall0 = std::chrono::steady_clock::now();

            const std::size_t bytes = n_pad * sizeof(std::int64_t);
            id<MTLBuffer> b_keys   = [device_ newBufferWithLength:bytes
                                                          options:MTLResourceStorageModeShared];
            id<MTLBuffer> b_values = [device_ newBufferWithLength:bytes
                                                          options:MTLResourceStorageModeShared];

            std::memcpy([b_keys   contents], keys,   n * sizeof(std::int64_t));
            std::memcpy([b_values contents], values, n * sizeof(std::int64_t));

            id<MTLCommandBuffer>         cb = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];

            // Pad tail with (INT64_MAX, 0).
            if (n_pad > n_real) {
                const std::uint32_t n_padding = n_pad - n_real;
                [ce setComputePipelineState:ps_pad_];
                [ce setBuffer:b_keys   offset:0 atIndex:0];
                [ce setBuffer:b_values offset:0 atIndex:1];
                [ce setBytes:&n_real length:sizeof(n_real) atIndex:2];
                [ce setBytes:&n_pad  length:sizeof(n_pad)  atIndex:3];
                [ce dispatchThreadgroups:MTLSizeMake((n_padding + kBlock - 1) / kBlock, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];
            }

            // ---- Bitonic sort: two-tier dispatch to amortize launch overhead ----
            //
            // CHUNK = 512 must match the constant in groupby.metal.
            // Tier 1: one local-sort dispatch handles all stages with k ≤ CHUNK.
            // Tier 2: for each k > CHUNK, do cross-block step dispatches for j > CHUNK/2,
            //         then one local-merge dispatch finishes j ≤ CHUNK/2.
            constexpr std::uint32_t CHUNK = 512;

            const NSUInteger n_chunks = n_pad / CHUNK;

            // Tier 1: local sort (handles k = 2 .. CHUNK).
            [ce setComputePipelineState:ps_local_sort_];
            [ce setBuffer:b_keys   offset:0 atIndex:0];
            [ce setBuffer:b_values offset:0 atIndex:1];
            [ce dispatchThreadgroups:MTLSizeMake(n_chunks, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];

            // Tier 2: outer k > CHUNK
            const NSUInteger step_grid = ((n_pad + kBlock - 1) / kBlock);
            for (std::uint32_t k_stage = CHUNK << 1; k_stage <= n_pad; k_stage <<= 1) {
                // Cross-block step for j > CHUNK/2.
                [ce setComputePipelineState:ps_step_];
                [ce setBuffer:b_keys   offset:0 atIndex:0];
                [ce setBuffer:b_values offset:0 atIndex:1];
                [ce setBytes:&n_pad length:sizeof(n_pad) atIndex:2];
                for (std::uint32_t j_step = k_stage >> 1; j_step > (CHUNK >> 1); j_step >>= 1) {
                    [ce setBytes:&k_stage length:sizeof(k_stage) atIndex:3];
                    [ce setBytes:&j_step  length:sizeof(j_step)  atIndex:4];
                    [ce dispatchThreadgroups:MTLSizeMake(step_grid, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];
                }
                // Local merge finishes j = CHUNK/2 .. 1 inside threadgroup memory.
                [ce setComputePipelineState:ps_local_merge_];
                [ce setBuffer:b_keys   offset:0 atIndex:0];
                [ce setBuffer:b_values offset:0 atIndex:1];
                [ce setBytes:&k_stage length:sizeof(k_stage) atIndex:2];
                [ce dispatchThreadgroups:MTLSizeMake(n_chunks, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];
            }

            [ce endEncoding];
            [cb commit];
            [cb waitUntilCompleted];

            const double kernel_ms = ([cb GPUEndTime] - [cb GPUStartTime]) * 1000.0;

            // Host segment-reduce on the sorted (key, value) arrays. Padding
            // (INT64_MAX) sorts to the end, so we stop when we hit it.
            const std::int64_t* sk = static_cast<const std::int64_t*>([b_keys   contents]);
            const std::int64_t* sv = static_cast<const std::int64_t*>([b_values contents]);

            r.keys.reserve(64);
            r.sums.reserve(64);
            std::size_t i = 0;
            while (i < n) {
                const std::int64_t k = sk[i];
                if (k == INT64_MAX) break;
                std::int64_t sum = 0;
                std::size_t j = i;
                while (j < n && sk[j] == k) {
                    sum += sv[j];
                    ++j;
                }
                r.keys.push_back(k);
                r.sums.push_back(sum);
                i = j;
            }

            r.kernel_ms   = kernel_ms;
            r.transfer_ms = 0.0;     // UMA
            r.wall_ms     = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - t_wall0).count();
        }
        return r;
    }

private:
    id<MTLComputePipelineState> make_pso(id<MTLLibrary> lib, NSString* name) {
        @autoreleasepool {
            id<MTLFunction> fn = [lib newFunctionWithName:name];
            if (!fn) {
                std::ostringstream os; os << "no function " << [name UTF8String];
                throw std::runtime_error(os.str());
            }
            NSError* err = nil;
            id<MTLComputePipelineState> pso =
                [device_ newComputePipelineStateWithFunction:fn error:&err];
            if (!pso) metal_throw("newComputePipelineState", err);
            return pso;
        }
    }

    id<MTLDevice>       device_ = nil;
    id<MTLCommandQueue> queue_  = nil;
    id<MTLComputePipelineState> ps_step_        = nil;
    id<MTLComputePipelineState> ps_pad_         = nil;
    id<MTLComputePipelineState> ps_local_sort_  = nil;
    id<MTLComputePipelineState> ps_local_merge_ = nil;
};

} // namespace

std::unique_ptr<GroupByAggregator> make_metal_groupby_aggregator() {
    return std::make_unique<MetalGroupByAggregator>();
}

} // namespace gpudb
