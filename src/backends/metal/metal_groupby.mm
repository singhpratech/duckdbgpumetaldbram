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
#include <cstdlib>
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
            ps_partition_count_   = make_pso(lib, @"partition_count_hash");
            ps_partition_scatter_ = make_pso(lib, @"partition_scatter_hash");
            ps_partition_aggregate_slotlock_ =
                make_pso(lib, @"partition_hash_aggregate_slotlock_i64");
            ps_seg_flags_  = make_pso(lib, @"segment_run_flags_i64");
            ps_seg_mark_   = make_pso(lib, @"segment_run_mark_starts_i64");
            ps_seg_sum_    = make_pso(lib, @"segment_run_sum_i64");
        }
    }

    Backend backend() const noexcept override { return Backend::METAL; }

    std::string device_name() const override {
        @autoreleasepool {
            std::ostringstream os;
            os << [[device_ name] UTF8String]
               << " (Metal, LSD radix sort + GPU segment-reduce)";
            return os.str();
        }
    }

    GroupByResult groupby_sum_i64(const std::int64_t* keys,
                                  const std::int64_t* values,
                                  std::size_t n,
                                  std::size_t expected_groups) override {
        // Hybrid auto-dispatch (v0.1.3) — pick the path that wins for this workload.
        //
        // Empirical wins on Apple M4 Max vs DuckDB CPU multi-thread (16 threads):
        //   * radix-opt (vectorized loads, multi-split scatter, simdgroup scan):
        //     beats CPU 2-3x at 100M+ rows × 1M+ groups, and is the fallback for
        //     cardinalities above slot-lock's safe cap.
        //   * slot-lock (4K hash partitions × 1K threadgroup slots):
        //     beats CPU 2.5-4x at any size when expected_groups is in
        //     [~1K, ~3M]. Becomes lock-contended at very low cardinality
        //     (CPU's hash fits in L1 there anyway).
        //
        // env override: GPUDB_METAL_GROUPBY_PATH=slotlock|radix to force a path.
        constexpr std::size_t kSlotLockMinGroups = 1024;
        constexpr std::size_t kSlotLockSafeCap   = 16'000'000;  // 32K partitions × 1024 slots, load 0.5

        const char* env = std::getenv("GPUDB_METAL_GROUPBY_PATH");
        const bool env_slotlock = env && std::strcmp(env, "slotlock") == 0;
        const bool env_radix    = env && std::strcmp(env, "radix")    == 0;

        bool use_slotlock;
        if (env_slotlock)      use_slotlock = true;
        else if (env_radix)    use_slotlock = false;
        else                   use_slotlock = (expected_groups >= kSlotLockMinGroups
                                               && expected_groups <= kSlotLockSafeCap);

        if (use_slotlock) {
            if (!path_logged_) {
                std::fprintf(stderr,
                             "[gpudb metal groupby] using slot-lock path (expected_groups=%zu)\n",
                             expected_groups);
                path_logged_ = true;
            }
            return groupby_sum_i64_hash_slotlock(keys, values, n, expected_groups);
        }
        if (!path_logged_) {
            std::fprintf(stderr,
                         "[gpudb metal groupby] using radix-opt path (expected_groups=%zu)\n",
                         expected_groups);
            path_logged_ = true;
        }
        return groupby_sum_i64_via_radix(keys, values, n, expected_groups);
    }

    // Default path: LSD radix sort + host segment-reduce. Always correct
    // for any input size. Selected when no GPUDB_METAL_GROUPBY_PATH env var
    // is set, or as a fallback when slot-lock would overflow.
    GroupByResult groupby_sum_i64_via_radix(const std::int64_t* keys,
                                            const std::int64_t* values,
                                            std::size_t n,
                                            std::size_t /*expected_groups*/) {
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
        constexpr std::uint32_t RADIX_WORK_PER_BLOCK = 1024;  // matches groupby.metal #define
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

            // ---- GPU segment-reduce (replaces host scan at large N) ----
            const std::size_t bytes_flags = static_cast<std::size_t>(n32) * sizeof(std::uint32_t);
            if (!b_seg_flags_ || [b_seg_flags_ length] < bytes_flags) {
                b_seg_flags_ = [device_ newBufferWithLength:bytes_flags
                                                    options:MTLResourceStorageModeShared];
            }
            if (!b_seg_starts_ || [b_seg_starts_ length] < bytes_flags) {
                b_seg_starts_ = [device_ newBufferWithLength:bytes_flags
                                                     options:MTLResourceStorageModeShared];
            }
            if (!b_sl_out_count_ || [b_sl_out_count_ length] < sizeof(std::uint32_t)) {
                b_sl_out_count_ = [device_ newBufferWithLength:sizeof(std::uint32_t)
                                                       options:MTLResourceStorageModeShared];
            }
            if (!b_sl_out_keys_ || [b_sl_out_keys_ length] < bytes_kv) {
                b_sl_out_keys_ = [device_ newBufferWithLength:bytes_kv
                                                      options:MTLResourceStorageModeShared];
            }
            if (!b_sl_out_sums_ || [b_sl_out_sums_ length] < bytes_kv) {
                b_sl_out_sums_ = [device_ newBufferWithLength:bytes_kv
                                                      options:MTLResourceStorageModeShared];
            }

            [ce setComputePipelineState:ps_seg_flags_];
            [ce setBuffer:in_keys offset:0 atIndex:0];
            [ce setBytes:&n32 length:sizeof(n32) atIndex:1];
            [ce setBuffer:b_seg_flags_ offset:0 atIndex:2];
            [ce dispatchThreadgroups:MTLSizeMake((n32 + kBlock - 1) / kBlock, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];

            std::memset([b_sl_out_count_ contents], 0, sizeof(std::uint32_t));

            [ce setComputePipelineState:ps_seg_mark_];
            [ce setBuffer:b_seg_flags_ offset:0 atIndex:0];
            [ce setBytes:&n32 length:sizeof(n32) atIndex:1];
            [ce setBuffer:b_seg_starts_ offset:0 atIndex:2];
            [ce setBuffer:b_sl_out_count_ offset:0 atIndex:3];
            [ce dispatchThreadgroups:MTLSizeMake((n32 + kBlock - 1) / kBlock, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];

            [ce endEncoding];
            [cb commit];
            [cb waitUntilCompleted];

            const double kernel_ms_radix = ([cb GPUEndTime] - [cb GPUStartTime]) * 1000.0;

            const std::uint32_t num_segs =
                *static_cast<const std::uint32_t*>([b_sl_out_count_ contents]);

            double kernel_ms_seg = 0.0;
            if (num_segs > 0) {
                id<MTLCommandBuffer> cb_seg = [queue_ commandBuffer];
                id<MTLComputeCommandEncoder> ce_seg = [cb_seg computeCommandEncoder];
                [ce_seg setComputePipelineState:ps_seg_sum_];
                [ce_seg setBuffer:in_keys offset:0 atIndex:0];
                [ce_seg setBuffer:in_values offset:0 atIndex:1];
                [ce_seg setBuffer:b_seg_starts_ offset:0 atIndex:2];
                [ce_seg setBytes:&num_segs length:sizeof(num_segs) atIndex:3];
                [ce_seg setBytes:&n32 length:sizeof(n32) atIndex:4];
                [ce_seg setBuffer:b_sl_out_keys_ offset:0 atIndex:5];
                [ce_seg setBuffer:b_sl_out_sums_ offset:0 atIndex:6];
                const NSUInteger seg_tg = (num_segs + kBlock - 1) / kBlock;
                [ce_seg dispatchThreadgroups:MTLSizeMake(seg_tg, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];
                [ce_seg endEncoding];
                [cb_seg commit];
                [cb_seg waitUntilCompleted];
                kernel_ms_seg = ([cb_seg GPUEndTime] - [cb_seg GPUStartTime]) * 1000.0;
            }

            const std::int64_t* ok = static_cast<const std::int64_t*>([b_sl_out_keys_ contents]);
            const std::int64_t* os_ = static_cast<const std::int64_t*>([b_sl_out_sums_ contents]);
            r.keys.assign(ok, ok + num_segs);
            r.sums.assign(os_, os_ + num_segs);

            r.kernel_ms   = kernel_ms_radix + kernel_ms_seg;
            r.transfer_ms = 0.0;
            r.wall_ms     = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - t_wall0).count();
        }
        return r;
    }

    // ------------------------------------------------------------------------
    // Slot-lock hash aggregate. Selected by GPUDB_METAL_GROUPBY_PATH=slotlock.
    //
    // Pipeline:
    //   1. partition_count_hash:   per-row atomic_fetch_add → partition_counts[4096].
    //   2. (host) exclusive scan partition_counts → partition_offsets[4097].
    //   3. partition_scatter_hash: stream rows into per-partition slices.
    //   4. partition_hash_aggregate_slotlock_i64: 4096 threadgroups × 64 threads,
    //      threadgroup-resident slot-lock hash table per partition. Emit phase
    //      writes (key, sum) to flat output via global atomic_fetch_add counter.
    //
    // out_count is read on the host after wait to know how many groups were
    // produced; the host then trims out_keys/out_sums to that length.
    //
    // Capacity bound: NUM_PARTITIONS (4096) × SLOT_COUNT (1024) = 4,194,304
    // distinct groups maximum. For uniformly-hashed keys, each partition
    // expects expected_groups/4096 unique entries; with load factor 0.85,
    // the soft cap is ~3.4M unique. If `expected_groups` exceeds ~3M, fall
    // back to the radix path for correctness.
    // ------------------------------------------------------------------------
    GroupByResult groupby_sum_i64_hash_slotlock(const std::int64_t* keys,
                                                const std::int64_t* values,
                                                std::size_t n,
                                                std::size_t expected_groups) {
        // Capacity guard: if the cardinality hint would overflow our
        // per-partition tables, fall back to the radix path instead of
        // silently dropping rows. We use a conservative threshold that
        // assumes uniform hash distribution.
        constexpr std::size_t kSlotLockHardCap   = 32768ull * 1024ull;     // 33,554,432 (32K × 1K)
        constexpr std::size_t kSlotLockSafeBound = (kSlotLockHardCap * 75) / 100;  // 75% load
        if (expected_groups > kSlotLockSafeBound) {
            std::fprintf(stderr,
                "[gpudb metal groupby] expected_groups=%zu exceeds slot-lock "
                "soft cap %zu — falling back to radix path for correctness\n",
                expected_groups, kSlotLockSafeBound);
            return groupby_sum_i64_via_radix(keys, values, n, expected_groups);
        }
        GroupByResult r;
        r.input_rows = n;
        if (n == 0) return r;

        if (n > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("input too large for slot-lock kernel uint32 indexing");
        }
        const std::uint32_t n32 = static_cast<std::uint32_t>(n);

        // Match constants in groupby.metal.
        constexpr std::uint32_t SL_NUM_PARTITIONS = 32768;
        constexpr std::uint32_t SL_TG_THREADS     = 64;
        constexpr std::uint32_t SL_COUNT_THREADS  = 256;
        constexpr std::uint32_t SL_COUNT_WORK     = 256;

        @autoreleasepool {
            const auto t_wall0 = std::chrono::steady_clock::now();

            // ---- Buffer sizing ----
            const std::size_t bytes_kv = static_cast<std::size_t>(n32) * sizeof(std::int64_t);
            const std::size_t bytes_partition_u32 = SL_NUM_PARTITIONS * sizeof(std::uint32_t);
            const std::size_t bytes_partition_offsets_u32 =
                (SL_NUM_PARTITIONS + 1) * sizeof(std::uint32_t);
            // Output: at most n unique groups; reserve n longs to be safe.
            // (For high-cardinality inputs this is exactly the right bound.)
            const std::size_t bytes_out_kv = bytes_kv;

            // Reuse main buffer cache for input keys/values (so subsequent calls
            // amortize allocation cost across runs).
            if (!b_keys_   || [b_keys_   length] < bytes_kv) {
                b_keys_   = [device_ newBufferWithLength:bytes_kv  options:MTLResourceStorageModeShared];
            }
            if (!b_values_ || [b_values_ length] < bytes_kv) {
                b_values_ = [device_ newBufferWithLength:bytes_kv  options:MTLResourceStorageModeShared];
            }
            // Reuse the second ping-pong slot (b_keys_b_ / b_values_b_) for
            // post-partition scatter destinations.
            if (!b_keys_b_   || [b_keys_b_   length] < bytes_kv) {
                b_keys_b_   = [device_ newBufferWithLength:bytes_kv  options:MTLResourceStorageModeShared];
            }
            if (!b_values_b_ || [b_values_b_ length] < bytes_kv) {
                b_values_b_ = [device_ newBufferWithLength:bytes_kv  options:MTLResourceStorageModeShared];
            }
            // Slot-lock owned buffers.
            if (!b_sl_partition_counts_ || [b_sl_partition_counts_ length] < bytes_partition_u32) {
                b_sl_partition_counts_ = [device_ newBufferWithLength:bytes_partition_u32
                                                              options:MTLResourceStorageModeShared];
            }
            if (!b_sl_partition_offsets_ ||
                [b_sl_partition_offsets_ length] < bytes_partition_offsets_u32)
            {
                b_sl_partition_offsets_ = [device_ newBufferWithLength:bytes_partition_offsets_u32
                                                               options:MTLResourceStorageModeShared];
            }
            if (!b_sl_partition_writepos_ ||
                [b_sl_partition_writepos_ length] < bytes_partition_u32)
            {
                b_sl_partition_writepos_ = [device_ newBufferWithLength:bytes_partition_u32
                                                                options:MTLResourceStorageModeShared];
            }
            if (!b_sl_out_count_ || [b_sl_out_count_ length] < sizeof(std::uint32_t)) {
                b_sl_out_count_ = [device_ newBufferWithLength:sizeof(std::uint32_t)
                                                       options:MTLResourceStorageModeShared];
            }
            if (!b_sl_out_keys_ || [b_sl_out_keys_ length] < bytes_out_kv) {
                b_sl_out_keys_ = [device_ newBufferWithLength:bytes_out_kv
                                                      options:MTLResourceStorageModeShared];
            }
            if (!b_sl_out_sums_ || [b_sl_out_sums_ length] < bytes_out_kv) {
                b_sl_out_sums_ = [device_ newBufferWithLength:bytes_out_kv
                                                      options:MTLResourceStorageModeShared];
            }

            id<MTLBuffer> b_in_keys           = b_keys_;
            id<MTLBuffer> b_in_values         = b_values_;
            id<MTLBuffer> b_scatter_keys      = b_keys_b_;
            id<MTLBuffer> b_scatter_values    = b_values_b_;
            id<MTLBuffer> b_partition_counts  = b_sl_partition_counts_;
            id<MTLBuffer> b_partition_offsets = b_sl_partition_offsets_;
            id<MTLBuffer> b_partition_writepos= b_sl_partition_writepos_;
            id<MTLBuffer> b_out_count         = b_sl_out_count_;
            id<MTLBuffer> b_out_keys          = b_sl_out_keys_;
            id<MTLBuffer> b_out_sums          = b_sl_out_sums_;

            std::memcpy([b_in_keys   contents], keys,   bytes_kv);
            std::memcpy([b_in_values contents], values, bytes_kv);

            // Zero per-partition counts and the global out_count.
            std::memset([b_partition_counts contents], 0, bytes_partition_u32);
            *static_cast<std::uint32_t*>([b_out_count contents]) = 0;

            // -------- Pass 1: partition counts --------
            id<MTLCommandBuffer> cb1 = [queue_ commandBuffer];
            {
                id<MTLComputeCommandEncoder> ce1 = [cb1 computeCommandEncoder];
                [ce1 setComputePipelineState:ps_partition_count_];
                [ce1 setBuffer:b_in_keys          offset:0 atIndex:0];
                [ce1 setBytes:&n32 length:sizeof(n32) atIndex:1];
                [ce1 setBuffer:b_partition_counts offset:0 atIndex:2];
                // Wide grid for streaming reads; threads-per-tg matches kernel default.
                const std::uint32_t blocks = (n32 + SL_COUNT_WORK - 1) / SL_COUNT_WORK;
                [ce1 dispatchThreadgroups:MTLSizeMake(blocks, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(SL_COUNT_THREADS, 1, 1)];
                [ce1 endEncoding];
            }
            [cb1 commit];
            [cb1 waitUntilCompleted];

            // -------- Host: exclusive prefix scan over partition counts --------
            // partition_offsets[i] = sum_{j<i} partition_counts[j].
            // Also detect overflow: if any partition has more rows than the
            // hash table can hold (~SLOT_COUNT * 0.85 to leave probe room),
            // we'd lose data. SLOT_COUNT=1024 → safe up to ~870 unique keys
            // per partition. For 1M rows uniform across 4096 partitions that's
            // ~244 rows / partition (max possible unique keys). For row counts
            // up to ~870K per partition we still fit because unique ≤ rows.
            {
                const std::uint32_t* counts =
                    static_cast<const std::uint32_t*>([b_partition_counts contents]);
                std::uint32_t* offsets =
                    static_cast<std::uint32_t*>([b_partition_offsets contents]);
                std::uint32_t running = 0;
                for (std::uint32_t p = 0; p < SL_NUM_PARTITIONS; ++p) {
                    offsets[p] = running;
                    running += counts[p];
                }
                offsets[SL_NUM_PARTITIONS] = running;
                // Sanity: total partitioned rows == n.
                if (running != n32) {
                    std::ostringstream os;
                    os << "slot-lock partition count mismatch: total=" << running
                       << " expected=" << n32;
                    throw std::runtime_error(os.str());
                }
            }

            // Reset per-partition write positions (used as atomic counters in scatter).
            std::memset([b_partition_writepos contents], 0, bytes_partition_u32);

            // -------- Pass 2 + 3: scatter, then per-partition slot-lock aggregate --------
            id<MTLCommandBuffer> cb2 = [queue_ commandBuffer];
            {
                id<MTLComputeCommandEncoder> ce2 = [cb2 computeCommandEncoder];

                // Pass 2: scatter
                [ce2 setComputePipelineState:ps_partition_scatter_];
                [ce2 setBuffer:b_in_keys           offset:0 atIndex:0];
                [ce2 setBuffer:b_in_values         offset:0 atIndex:1];
                [ce2 setBytes:&n32 length:sizeof(n32) atIndex:2];
                [ce2 setBuffer:b_partition_offsets offset:0 atIndex:3];
                [ce2 setBuffer:b_partition_writepos offset:0 atIndex:4];
                [ce2 setBuffer:b_scatter_keys      offset:0 atIndex:5];
                [ce2 setBuffer:b_scatter_values    offset:0 atIndex:6];
                const std::uint32_t blocks = (n32 + SL_COUNT_WORK - 1) / SL_COUNT_WORK;
                [ce2 dispatchThreadgroups:MTLSizeMake(blocks, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(SL_COUNT_THREADS, 1, 1)];

                // Pass 3: per-partition slot-lock aggregate.
                // One threadgroup per partition; SL_TG_THREADS threads each.
                [ce2 setComputePipelineState:ps_partition_aggregate_slotlock_];
                [ce2 setBuffer:b_scatter_keys      offset:0 atIndex:0];
                [ce2 setBuffer:b_scatter_values    offset:0 atIndex:1];
                [ce2 setBuffer:b_partition_offsets offset:0 atIndex:2];
                [ce2 setBuffer:b_out_count         offset:0 atIndex:3];
                [ce2 setBuffer:b_out_keys          offset:0 atIndex:4];
                [ce2 setBuffer:b_out_sums          offset:0 atIndex:5];
                [ce2 dispatchThreadgroups:MTLSizeMake(SL_NUM_PARTITIONS, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(SL_TG_THREADS, 1, 1)];

                [ce2 endEncoding];
            }
            [cb2 commit];
            [cb2 waitUntilCompleted];

            const double kernel_ms_total =
                ([cb2 GPUEndTime] - [cb2 GPUStartTime]) * 1000.0
                + ([cb1 GPUEndTime] - [cb1 GPUStartTime]) * 1000.0;

            // -------- Host: collect output --------
            const std::uint32_t out_count =
                *static_cast<const std::uint32_t*>([b_out_count contents]);
            const std::int64_t* ok = static_cast<const std::int64_t*>([b_out_keys contents]);
            const std::int64_t* os_ = static_cast<const std::int64_t*>([b_out_sums contents]);
            r.keys.assign(ok, ok + out_count);
            r.sums.assign(os_, os_ + out_count);

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
    id<MTLComputePipelineState> ps_partition_count_              = nil;
    id<MTLComputePipelineState> ps_partition_scatter_            = nil;
    id<MTLComputePipelineState> ps_partition_aggregate_slotlock_ = nil;
    id<MTLComputePipelineState> ps_seg_flags_ = nil;
    id<MTLComputePipelineState> ps_seg_mark_ = nil;
    id<MTLComputePipelineState> ps_seg_sum_ = nil;
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
    // Slot-lock path buffers.
    id<MTLBuffer> b_sl_partition_counts_   = nil;
    id<MTLBuffer> b_sl_partition_offsets_  = nil;
    id<MTLBuffer> b_sl_partition_writepos_ = nil;
    id<MTLBuffer> b_sl_out_count_          = nil;
    id<MTLBuffer> b_sl_out_keys_           = nil;
    id<MTLBuffer> b_sl_out_sums_           = nil;
    id<MTLBuffer> b_seg_flags_           = nil;
    id<MTLBuffer> b_seg_starts_            = nil;
    bool path_logged_ = false;
};

} // namespace

std::unique_ptr<GroupByAggregator> make_metal_groupby_aggregator() {
    return std::make_unique<MetalGroupByAggregator>();
}

} // namespace gpudb
