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
#include <array>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

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
            ps_radix_flip_      = make_pso(lib, @"radix_flip_sign_bit");
            ps_radix_histogram_ = make_pso(lib, @"radix_histogram");
            ps_radix_scatter_   = make_pso(lib, @"radix_scatter");
        }
    }

    Backend backend() const noexcept override { return Backend::METAL; }

    std::string device_name() const override {
        @autoreleasepool {
            std::ostringstream os;
            os << [[device_ name] UTF8String]
               << " (Metal, LSD radix sort + host segment-reduce)";
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

        if (n > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("input too large for radix kernel uint32 indexing");
        }
        const std::uint32_t n32 = static_cast<std::uint32_t>(n);

        // Match constants in groupby.metal.
        constexpr std::uint32_t RADIX_BUCKETS        = 256;
        constexpr std::uint32_t RADIX_BLOCK_SIZE     = 256;
        constexpr std::uint32_t RADIX_WORK_PER_BLOCK = 256;
        const std::uint32_t num_blocks = (n32 + RADIX_WORK_PER_BLOCK - 1) / RADIX_WORK_PER_BLOCK;

        @autoreleasepool {
            const auto t_wall0 = std::chrono::steady_clock::now();

            const std::size_t bytes_kv   = static_cast<std::size_t>(n32) * sizeof(std::int64_t);
            const std::size_t bytes_hist = static_cast<std::size_t>(num_blocks)
                                         * RADIX_BUCKETS * sizeof(std::uint32_t);

            // Cached ping-pong buffers.
            if (!b_keys_   || [b_keys_   length] < bytes_kv) {
                b_keys_   = [device_ newBufferWithLength:bytes_kv  options:MTLResourceStorageModeShared];
            }
            if (!b_values_ || [b_values_ length] < bytes_kv) {
                b_values_ = [device_ newBufferWithLength:bytes_kv  options:MTLResourceStorageModeShared];
            }
            if (!b_keys_b_   || [b_keys_b_   length] < bytes_kv) {
                b_keys_b_   = [device_ newBufferWithLength:bytes_kv  options:MTLResourceStorageModeShared];
            }
            if (!b_values_b_ || [b_values_b_ length] < bytes_kv) {
                b_values_b_ = [device_ newBufferWithLength:bytes_kv  options:MTLResourceStorageModeShared];
            }
            if (!b_hist_     || [b_hist_     length] < bytes_hist) {
                b_hist_   = [device_ newBufferWithLength:bytes_hist options:MTLResourceStorageModeShared];
            }
            if (!b_scan_     || [b_scan_     length] < bytes_hist) {
                b_scan_   = [device_ newBufferWithLength:bytes_hist options:MTLResourceStorageModeShared];
            }

            id<MTLBuffer> b_keys_a   = b_keys_;
            id<MTLBuffer> b_values_a = b_values_;
            id<MTLBuffer> b_keys_b   = b_keys_b_;
            id<MTLBuffer> b_values_b = b_values_b_;
            id<MTLBuffer> b_hist     = b_hist_;
            id<MTLBuffer> b_scan     = b_scan_;

            std::memcpy([b_keys_a   contents], keys,   bytes_kv);
            std::memcpy([b_values_a contents], values, bytes_kv);

            // ---- Sign-bit flip on input (signed-radix → unsigned-radix) ----
            {
                id<MTLCommandBuffer>         cb = [queue_ commandBuffer];
                id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
                [ce setComputePipelineState:ps_radix_flip_];
                [ce setBuffer:b_keys_a offset:0 atIndex:0];
                [ce setBytes:&n32 length:sizeof(n32) atIndex:1];
                [ce dispatchThreadgroups:MTLSizeMake((n32 + kBlock - 1) / kBlock, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];
                [ce endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
            }

            double kernel_ms_total = 0.0;

            id<MTLBuffer> in_keys   = b_keys_a,   in_values   = b_values_a;
            id<MTLBuffer> out_keys  = b_keys_b,   out_values  = b_values_b;

            for (std::uint32_t pass = 0; pass < 8; ++pass) {
                const std::uint32_t shift = pass * 8u;

                // ---- 1. Histogram (GPU) ----
                {
                    id<MTLCommandBuffer>         cb = [queue_ commandBuffer];
                    id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
                    [ce setComputePipelineState:ps_radix_histogram_];
                    [ce setBuffer:in_keys offset:0 atIndex:0];
                    [ce setBytes:&n32   length:sizeof(n32)   atIndex:1];
                    [ce setBytes:&shift length:sizeof(shift) atIndex:2];
                    [ce setBuffer:b_hist offset:0 atIndex:3];
                    [ce dispatchThreadgroups:MTLSizeMake(num_blocks, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(RADIX_BLOCK_SIZE, 1, 1)];
                    [ce endEncoding];
                    [cb commit];
                    [cb waitUntilCompleted];
                    kernel_ms_total += ([cb GPUEndTime] - [cb GPUStartTime]) * 1000.0;
                }

                // ---- 2. Scatter offsets (multi-threaded host scan) ----
                // The bucket_total reduction parallelizes cleanly across
                // threads (per-thread private 256-bucket arrays merged at
                // the end). The per-block running sum (step c) is sequential
                // along the block axis but parallelizable across buckets;
                // we keep it cache-friendly (sequential hist+scan walk) and
                // single-threaded — the bucket_total dominates wall at scale.
                {
                    const std::uint32_t* hist_in =
                        static_cast<const std::uint32_t*>([b_hist contents]);
                    std::uint32_t*       scan_out =
                        static_cast<std::uint32_t*>([b_scan contents]);

                    constexpr int kHostThreads = 8;
                    const std::uint64_t total_entries =
                        static_cast<std::uint64_t>(num_blocks) * RADIX_BUCKETS;

                    // (a) bucket totals — parallel, per-thread private arrays
                    std::uint32_t bucket_total[RADIX_BUCKETS] = {0};
                    if (total_entries >= 65536) {
                        std::vector<std::array<std::uint32_t, RADIX_BUCKETS>>
                            per_thread(kHostThreads);
                        for (auto& a : per_thread) a.fill(0);
                        std::vector<std::thread> workers;
                        workers.reserve(kHostThreads);
                        for (int t = 0; t < kHostThreads; ++t) {
                            workers.emplace_back([&, t] {
                                const std::uint64_t lo = total_entries * t / kHostThreads;
                                const std::uint64_t hi = total_entries * (t + 1) / kHostThreads;
                                auto& local = per_thread[t];
                                for (std::uint64_t i = lo; i < hi; ++i) {
                                    local[i & (RADIX_BUCKETS - 1)] += hist_in[i];
                                }
                            });
                        }
                        for (auto& w : workers) w.join();
                        for (int t = 0; t < kHostThreads; ++t) {
                            for (std::uint32_t b = 0; b < RADIX_BUCKETS; ++b) {
                                bucket_total[b] += per_thread[t][b];
                            }
                        }
                    } else {
                        for (std::uint64_t i = 0; i < total_entries; ++i) {
                            bucket_total[i & (RADIX_BUCKETS - 1)] += hist_in[i];
                        }
                    }

                    // (b) bucket offsets (256 elements, trivial)
                    std::uint32_t bucket_offset[RADIX_BUCKETS];
                    std::uint32_t running = 0;
                    for (std::uint32_t b = 0; b < RADIX_BUCKETS; ++b) {
                        bucket_offset[b] = running;
                        running += bucket_total[b];
                    }

                    // (c) per-block-per-bucket offsets — sequential cache-friendly walk
                    std::uint32_t block_running[RADIX_BUCKETS] = {0};
                    for (std::uint32_t bid = 0; bid < num_blocks; ++bid) {
                        const std::uint32_t base = bid * RADIX_BUCKETS;
                        for (std::uint32_t b = 0; b < RADIX_BUCKETS; ++b) {
                            scan_out[base + b]   = bucket_offset[b] + block_running[b];
                            block_running[b]    += hist_in[base + b];
                        }
                    }
                }

                // ---- 3. Scatter (GPU, stable) ----
                {
                    id<MTLCommandBuffer>         cb = [queue_ commandBuffer];
                    id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
                    [ce setComputePipelineState:ps_radix_scatter_];
                    [ce setBuffer:in_keys    offset:0 atIndex:0];
                    [ce setBuffer:in_values  offset:0 atIndex:1];
                    [ce setBytes:&n32   length:sizeof(n32)   atIndex:2];
                    [ce setBytes:&shift length:sizeof(shift) atIndex:3];
                    [ce setBuffer:b_scan     offset:0 atIndex:4];
                    [ce setBuffer:out_keys   offset:0 atIndex:5];
                    [ce setBuffer:out_values offset:0 atIndex:6];
                    [ce dispatchThreadgroups:MTLSizeMake(num_blocks, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(RADIX_BLOCK_SIZE, 1, 1)];
                    [ce endEncoding];
                    [cb commit];
                    [cb waitUntilCompleted];
                    kernel_ms_total += ([cb GPUEndTime] - [cb GPUStartTime]) * 1000.0;
                }

                std::swap(in_keys,   out_keys);
                std::swap(in_values, out_values);
            }

            // ---- Unflip sign on the now-sorted keys ----
            {
                id<MTLCommandBuffer>         cb = [queue_ commandBuffer];
                id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
                [ce setComputePipelineState:ps_radix_flip_];
                [ce setBuffer:in_keys offset:0 atIndex:0];
                [ce setBytes:&n32 length:sizeof(n32) atIndex:1];
                [ce dispatchThreadgroups:MTLSizeMake((n32 + kBlock - 1) / kBlock, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];
                [ce endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
            }

            // ---- Host segment-reduce ----
            const std::int64_t* sk = static_cast<const std::int64_t*>([in_keys   contents]);
            const std::int64_t* sv = static_cast<const std::int64_t*>([in_values contents]);

            r.keys.reserve(64);
            r.sums.reserve(64);
            std::size_t i = 0;
            while (i < n) {
                const std::int64_t k = sk[i];
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

            r.kernel_ms   = kernel_ms_total;
            r.transfer_ms = 0.0;
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
    id<MTLComputePipelineState> ps_step_           = nil;
    id<MTLComputePipelineState> ps_pad_            = nil;
    id<MTLComputePipelineState> ps_local_sort_     = nil;
    id<MTLComputePipelineState> ps_local_merge_    = nil;
    id<MTLComputePipelineState> ps_radix_flip_     = nil;
    id<MTLComputePipelineState> ps_radix_histogram_ = nil;
    id<MTLComputePipelineState> ps_radix_scatter_  = nil;
    // Buffer cache for radix's ping-pong + histogram / scan.
    id<MTLBuffer> b_keys_     = nil;
    id<MTLBuffer> b_values_   = nil;
    id<MTLBuffer> b_keys_b_   = nil;
    id<MTLBuffer> b_values_b_ = nil;
    id<MTLBuffer> b_hist_     = nil;
    id<MTLBuffer> b_scan_     = nil;
};

} // namespace

std::unique_ptr<GroupByAggregator> make_metal_groupby_aggregator() {
    return std::make_unique<MetalGroupByAggregator>();
}

} // namespace gpudb
