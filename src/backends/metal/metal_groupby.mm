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
            ps_radix_bucket_totals_  = make_pso(lib, @"radix_bucket_totals");
            ps_radix_bucket_offsets_ = make_pso(lib, @"radix_bucket_offsets");
            ps_radix_per_bucket_scan_ = make_pso(lib, @"radix_per_bucket_scan");
            ps_radix_minmax_ = make_pso(lib, @"radix_minmax_i64");
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

            // ---- All 8 radix passes + sign-flips in ONE command buffer ----
            // On-device GPU scan eliminates the host-scan bottleneck and lets
            // every dispatch live in the same cb. One commit, one wait —
            // wall converges to kernel time.

            // Bucket-totals + bucket-offsets buffers (small; size lazily).
            const std::size_t bytes_buckets = RADIX_BUCKETS * sizeof(std::uint32_t);
            if (!b_bucket_total_  || [b_bucket_total_  length] < bytes_buckets) {
                b_bucket_total_  = [device_ newBufferWithLength:bytes_buckets
                                                        options:MTLResourceStorageModeShared];
            }
            if (!b_bucket_offset_ || [b_bucket_offset_ length] < bytes_buckets) {
                b_bucket_offset_ = [device_ newBufferWithLength:bytes_buckets
                                                        options:MTLResourceStorageModeShared];
            }
            id<MTLBuffer> b_bucket_total  = b_bucket_total_;
            id<MTLBuffer> b_bucket_offset = b_bucket_offset_;

            id<MTLBuffer> in_keys   = b_keys_a,   in_values   = b_values_a;
            id<MTLBuffer> out_keys  = b_keys_b,   out_values  = b_values_b;

            // ---- Min-max pre-scan to detect "constant byte" passes ----
            // Run a separate cb so we can read the result on the host before
            // building the radix pipeline. The cost is one extra commit/wait
            // (~100 µs) plus a tiny kernel — and it can save 4-7 of the 8
            // radix passes for low-cardinality input.
            constexpr NSUInteger kMinMaxGrid = 256;
            const std::size_t bytes_minmax_partials =
                kMinMaxGrid * 4 * sizeof(std::int32_t);  // per block: 2 ints for min, 2 for max
            if (!b_minmax_partials_ || [b_minmax_partials_ length] < bytes_minmax_partials) {
                b_minmax_partials_ = [device_ newBufferWithLength:bytes_minmax_partials
                                                          options:MTLResourceStorageModeShared];
            }
            // Layout: [min_lo_0, min_hi_0, ..., min_lo_K-1, min_hi_K-1,
            //          max_lo_0, max_hi_0, ..., max_lo_K-1, max_hi_K-1]
            // (K = kMinMaxGrid).
            std::uint8_t active_bytes = 0xFF;  // default: all 8 bytes active
            {
                id<MTLCommandBuffer>         cb_mm = [queue_ commandBuffer];
                id<MTLComputeCommandEncoder> ce_mm = [cb_mm computeCommandEncoder];
                [ce_mm setComputePipelineState:ps_radix_minmax_];
                [ce_mm setBuffer:b_keys_a offset:0 atIndex:0];
                [ce_mm setBytes:&n32 length:sizeof(n32) atIndex:1];
                [ce_mm setBuffer:b_minmax_partials_
                          offset:0
                         atIndex:2];
                [ce_mm setBuffer:b_minmax_partials_
                          offset:kMinMaxGrid * 2 * sizeof(std::int32_t)
                         atIndex:3];
                [ce_mm dispatchThreadgroups:MTLSizeMake(kMinMaxGrid, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [ce_mm endEncoding];
                [cb_mm commit];
                [cb_mm waitUntilCompleted];

                // Host-side final reduction over per-block partials.
                const std::int32_t* parts =
                    static_cast<const std::int32_t*>([b_minmax_partials_ contents]);
                std::int64_t mn = INT64_MAX, mx = INT64_MIN;
                for (NSUInteger b = 0; b < kMinMaxGrid; ++b) {
                    const std::uint64_t mlo = static_cast<std::uint32_t>(parts[b * 2 + 0]);
                    const std::uint64_t mhi = static_cast<std::uint32_t>(parts[b * 2 + 1]);
                    const std::uint64_t Mlo = static_cast<std::uint32_t>(parts[(kMinMaxGrid + b) * 2 + 0]);
                    const std::uint64_t Mhi = static_cast<std::uint32_t>(parts[(kMinMaxGrid + b) * 2 + 1]);
                    std::int64_t bm, bM;
                    const std::uint64_t bmu = (mhi << 32) | mlo;
                    const std::uint64_t bMu = (Mhi << 32) | Mlo;
                    std::memcpy(&bm, &bmu, sizeof(std::int64_t));
                    std::memcpy(&bM, &bMu, sizeof(std::int64_t));
                    if (bm < mn) mn = bm;
                    if (bM > mx) mx = bM;
                }

                // After sign-flip, signed sort == unsigned sort. Compute
                // active byte mask using XOR-with-sign-bit form.
                const std::uint64_t mn_u = static_cast<std::uint64_t>(mn) ^ 0x8000000000000000ULL;
                const std::uint64_t mx_u = static_cast<std::uint64_t>(mx) ^ 0x8000000000000000ULL;
                active_bytes = 0;
                for (std::uint32_t p = 0; p < 8; ++p) {
                    const std::uint64_t shift = p * 8;
                    const std::uint8_t mn_b = static_cast<std::uint8_t>((mn_u >> shift) & 0xFFu);
                    const std::uint8_t mx_b = static_cast<std::uint8_t>((mx_u >> shift) & 0xFFu);
                    if (mn_b != mx_b) active_bytes |= (1u << p);
                }
                // If all keys are equal (no byte varies), still need pass 0
                // so the data lands in the right ping-pong slot.
                if (active_bytes == 0) active_bytes = 0x01;
            }

            id<MTLCommandBuffer>         cb = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];

            // Sign-flip (initial, on b_keys_a)
            [ce setComputePipelineState:ps_radix_flip_];
            [ce setBuffer:b_keys_a offset:0 atIndex:0];
            [ce setBytes:&n32 length:sizeof(n32) atIndex:1];
            [ce dispatchThreadgroups:MTLSizeMake((n32 + kBlock - 1) / kBlock, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];

            for (std::uint32_t pass = 0; pass < 8; ++pass) {
                if (!(active_bytes & (1u << pass))) continue;   // skip constant byte
                const std::uint32_t shift = pass * 8u;

                // 1. Histogram
                [ce setComputePipelineState:ps_radix_histogram_];
                [ce setBuffer:in_keys offset:0 atIndex:0];
                [ce setBytes:&n32   length:sizeof(n32)   atIndex:1];
                [ce setBytes:&shift length:sizeof(shift) atIndex:2];
                [ce setBuffer:b_hist offset:0 atIndex:3];
                [ce dispatchThreadgroups:MTLSizeMake(num_blocks, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(RADIX_BLOCK_SIZE, 1, 1)];

                // 2a. Bucket totals (256 threadgroups, one per bucket)
                [ce setComputePipelineState:ps_radix_bucket_totals_];
                [ce setBuffer:b_hist          offset:0 atIndex:0];
                [ce setBytes:&num_blocks      length:sizeof(num_blocks) atIndex:1];
                [ce setBuffer:b_bucket_total  offset:0 atIndex:2];
                [ce dispatchThreadgroups:MTLSizeMake(RADIX_BUCKETS, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(RADIX_BLOCK_SIZE, 1, 1)];

                // 2b. Bucket offsets (1 threadgroup, 256 threads)
                [ce setComputePipelineState:ps_radix_bucket_offsets_];
                [ce setBuffer:b_bucket_total  offset:0 atIndex:0];
                [ce setBuffer:b_bucket_offset offset:0 atIndex:1];
                [ce dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(RADIX_BLOCK_SIZE, 1, 1)];

                // 2c. Per-bucket per-block scan (256 threadgroups, one per bucket)
                [ce setComputePipelineState:ps_radix_per_bucket_scan_];
                [ce setBuffer:b_hist          offset:0 atIndex:0];
                [ce setBytes:&num_blocks      length:sizeof(num_blocks) atIndex:1];
                [ce setBuffer:b_bucket_offset offset:0 atIndex:2];
                [ce setBuffer:b_scan          offset:0 atIndex:3];
                [ce dispatchThreadgroups:MTLSizeMake(RADIX_BUCKETS, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(RADIX_BLOCK_SIZE, 1, 1)];

                // 3. Scatter (stable)
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

                std::swap(in_keys,   out_keys);
                std::swap(in_values, out_values);
            }

            // Sign-unflip on the now-sorted keys (which are in `in_keys` after
            // 8 swaps, so back to b_keys_a).
            [ce setComputePipelineState:ps_radix_flip_];
            [ce setBuffer:in_keys offset:0 atIndex:0];
            [ce setBytes:&n32 length:sizeof(n32) atIndex:1];
            [ce dispatchThreadgroups:MTLSizeMake((n32 + kBlock - 1) / kBlock, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];

            [ce endEncoding];
            [cb commit];
            [cb waitUntilCompleted];

            const double kernel_ms_total = ([cb GPUEndTime] - [cb GPUStartTime]) * 1000.0;

            // ---- Host segment-reduce (parallel) ----
            // Sorted (key, value) array. Divide into NTHREADS chunks; each
            // thread emits (key, sum) pairs for the runs entirely inside its
            // chunk. Boundary keys (a run that spans the chunk edge) are
            // stitched together in a final single-thread pass over the
            // per-thread results.
            const std::int64_t* sk = static_cast<const std::int64_t*>([in_keys   contents]);
            const std::int64_t* sv = static_cast<const std::int64_t*>([in_values contents]);

            constexpr int kSegThreads = 8;
            if (n >= 16384) {
                std::vector<std::vector<std::int64_t>> tk(kSegThreads), ts(kSegThreads);

                // Align each chunk start to a run boundary so no run is split.
                std::array<std::size_t, kSegThreads + 1> starts;
                starts[0] = 0;
                for (int t = 1; t < kSegThreads; ++t) {
                    std::size_t s = (n / kSegThreads) * t;
                    while (s < n && s > 0 && sk[s] == sk[s - 1]) ++s;
                    starts[t] = s;
                }
                starts[kSegThreads] = n;

                std::vector<std::thread> workers;
                workers.reserve(kSegThreads);
                for (int t = 0; t < kSegThreads; ++t) {
                    workers.emplace_back([&, t] {
                        const std::size_t lo = starts[t];
                        const std::size_t hi = starts[t + 1];
                        std::size_t i = lo;
                        while (i < hi) {
                            const std::int64_t k = sk[i];
                            std::int64_t sum = 0;
                            std::size_t j = i;
                            while (j < hi && sk[j] == k) { sum += sv[j]; ++j; }
                            tk[t].push_back(k);
                            ts[t].push_back(sum);
                            i = j;
                        }
                    });
                }
                for (auto& w : workers) w.join();

                // Concatenate (no boundary stitching needed: starts[] aligned to runs).
                std::size_t total = 0;
                for (int t = 0; t < kSegThreads; ++t) total += tk[t].size();
                r.keys.reserve(total);
                r.sums.reserve(total);
                for (int t = 0; t < kSegThreads; ++t) {
                    r.keys.insert(r.keys.end(), tk[t].begin(), tk[t].end());
                    r.sums.insert(r.sums.end(), ts[t].begin(), ts[t].end());
                }
            } else {
                r.keys.reserve(64);
                r.sums.reserve(64);
                std::size_t i = 0;
                while (i < n) {
                    const std::int64_t k = sk[i];
                    std::int64_t sum = 0;
                    std::size_t j = i;
                    while (j < n && sk[j] == k) { sum += sv[j]; ++j; }
                    r.keys.push_back(k);
                    r.sums.push_back(sum);
                    i = j;
                }
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
    id<MTLComputePipelineState> ps_radix_bucket_totals_ = nil;
    id<MTLComputePipelineState> ps_radix_bucket_offsets_ = nil;
    id<MTLComputePipelineState> ps_radix_per_bucket_scan_ = nil;
    id<MTLComputePipelineState> ps_radix_minmax_ = nil;
    id<MTLBuffer> b_bucket_total_  = nil;
    id<MTLBuffer> b_bucket_offset_ = nil;
    id<MTLBuffer> b_minmax_partials_ = nil;
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
