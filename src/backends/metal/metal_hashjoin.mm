// metal_hashjoin.mm — Metal inner equi-join on int64 keys.
//
// Primary path: auto-selected per join size.
//   build ≤ 500k: global slot-lock in one command buffer (lowest sync overhead).
//   build > 500k: partitioned scatter + per-partition TG hash (full radix join).
//   Build scatter is cached when the same build_keys pointer is probed again.
//
// Override with GPUDB_METAL_HASHJOIN_PATH:
//   partition — partitioned scatter + per-partition TG hash join (large builds)
//   partition_scan — partitioned scatter + linear scan probe (debug/fallback)
//   global    — force single global slot-lock table
//   merge     — radix-sort build + binary-search probe

#include "gpu_backend.hpp"
#include "metal_radix_sort.hpp"
#include "metal_kernel_sources.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace gpudb {

namespace {

constexpr NSUInteger kBlock = 256;
constexpr std::int64_t kEmptySentinel = static_cast<std::int64_t>(0x8000000000000000ULL);
constexpr std::uint32_t kMinCap = 1u << 10;
constexpr std::uint32_t kMaxCap = 1u << 28;

// Match groupby.metal slot-lock partition constants.
constexpr std::uint32_t kNumPartitions = 32768;
constexpr std::uint32_t kCountThreads = 256;
constexpr std::uint32_t kCountWork = 256;

[[noreturn]] void metal_throw(const char* what, NSError* err) {
    std::ostringstream os;
    os << "Metal hashjoin " << what;
    if (err) os << ": " << [[err localizedDescription] UTF8String];
    throw std::runtime_error(os.str());
}

std::uint32_t next_pow2(std::uint64_t v) {
    if (v < 2) return 2;
    --v;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    return static_cast<std::uint32_t>(v + 1);
}

std::uint32_t pick_table_capacity(std::size_t n_build) {
    std::uint64_t target = static_cast<std::uint64_t>(n_build) * 2;
    std::uint32_t cap = next_pow2(target);
    if (cap < kMinCap) cap = kMinCap;
    if (cap > kMaxCap) cap = kMaxCap;
    return cap;
}

void ensure_buffer(id<MTLDevice> device, id<MTLBuffer>& buf, std::size_t& cap_bytes,
                   std::size_t need_bytes) {
    if (need_bytes <= cap_bytes) return;
    buf = [device newBufferWithLength:need_bytes options:MTLResourceStorageModeShared];
    cap_bytes = need_bytes;
}

class MetalHashJoinProbe final : public HashJoinProbe {
public:
    MetalHashJoinProbe() {
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

            ps_merge_ = make_pso(device_, lib, @"hashjoin_merge_sorted_i64");
            ps_init_ = make_pso(device_, lib, @"hashjoin_init_table");
            ps_build_ = make_pso(device_, lib, @"hashjoin_build_slotlock_i64");
            ps_probe_global_ = make_pso(device_, lib, @"hashjoin_probe_slotlock_i64");
            ps_partition_count_ = make_pso(device_, lib, @"partition_count_hash");
            ps_partition_scatter_ = make_pso(device_, lib, @"partition_scatter_hash");
            ps_probe_partition_ = make_pso(device_, lib, @"hashjoin_probe_partition_i64");
            ps_partition_hashjoin_ = make_pso(device_, lib, @"partition_hashjoin_i64");
            ps_fill_iota_ = make_pso(device_, lib, @"hashjoin_fill_iota_i64");

            radix_ = std::make_unique<metal_detail::MetalRadixSort>(device_, queue_);
        }
    }

    Backend backend() const noexcept override { return Backend::METAL; }

    std::string device_name() const override {
        @autoreleasepool {
            std::ostringstream os;
            os << [[device_ name] UTF8String] << " (Metal hash join)";
            return os.str();
        }
    }

    JoinResult inner_join_i64(const std::int64_t* build_keys, std::size_t n_build,
                              const std::int64_t* probe_keys, std::size_t n_probe) override {
        const auto t0 = std::chrono::steady_clock::now();

        JoinResult r;
        r.rows_build = n_build;
        r.rows_probe = n_probe;
        if (n_build == 0 || n_probe == 0) {
            r.wall_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
            return r;
        }
        if (n_build > std::numeric_limits<std::uint32_t>::max() ||
            n_probe > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("hash join: row count exceeds uint32 limit");
        }

        for (std::size_t i = 0; i < n_build; ++i) {
            if (build_keys[i] == kEmptySentinel) {
                throw std::runtime_error(
                    "build key INT64_MIN clashes with empty sentinel; not yet supported");
            }
        }

        const auto n_build32 = static_cast<std::uint32_t>(n_build);
        const auto n_probe32 = static_cast<std::uint32_t>(n_probe);

        const char* env = std::getenv("GPUDB_METAL_HASHJOIN_PATH");
        enum class Path { Partition, PartitionScan, Global, Merge };
        Path path;
        if (env) {
            path = Path::Partition;
            if (std::strcmp(env, "global") == 0 || std::strcmp(env, "hash") == 0) path = Path::Global;
            else if (std::strcmp(env, "merge") == 0) path = Path::Merge;
            else if (std::strcmp(env, "partition_scan") == 0) path = Path::PartitionScan;
            else if (std::strcmp(env, "partition") == 0) path = Path::Partition;
        } else {
            // Auto-select: global fused CB wins below ~500k build (one sync, no scatter).
            // Partitioned scatter scales past that (no O(cap) table init, no global CAS).
            const std::uint64_t need = static_cast<std::uint64_t>(n_build) * 2;
            if (need > kMaxCap) {
                path = Path::Merge;
            } else if (n_build <= 500'000) {
                path = Path::Global;
            } else {
                path = Path::Partition;
            }
        }

        if (!path_logged_) {
            const char* name = (path == Path::Partition)     ? "partitioned TG hash"
                                : (path == Path::PartitionScan) ? "partitioned linear scan"
                                : (path == Path::Global)        ? "global slot-lock"
                                                                : "sort-merge";
            std::fprintf(stderr, "[gpudb metal hashjoin] using %s path\n", name);
            path_logged_ = true;
        }

        switch (path) {
        case Path::Partition:
            return join_hash_partitioned(build_keys, n_build32, probe_keys, n_probe32, t0, false);
        case Path::PartitionScan:
            return join_hash_partitioned(build_keys, n_build32, probe_keys, n_probe32, t0, true);
        case Path::Global:
            return join_hash_global(build_keys, n_build32, probe_keys, n_probe32,
                                    pick_table_capacity(n_build), t0);
        case Path::Merge:
            return join_sort_merge(build_keys, n_build32, probe_keys, n_probe32, t0);
        }
        return r;
    }

private:
    static id<MTLComputePipelineState> make_pso(id<MTLDevice> device, id<MTLLibrary> lib,
                                                 NSString* name) {
        NSError* err = nil;
        id<MTLFunction> fn = [lib newFunctionWithName:name];
        if (!fn) {
            std::ostringstream os;
            os << "no function " << [name UTF8String];
            throw std::runtime_error(os.str());
        }
        id<MTLComputePipelineState> ps = [device newComputePipelineStateWithFunction:fn error:&err];
        if (!ps) metal_throw("newComputePipelineState", err);
        return ps;
    }

    static void host_exclusive_scan(const id<MTLBuffer>& counts_buf, id<MTLBuffer>& offsets_buf,
                                    std::size_t& cap_offsets_bytes, id<MTLDevice> device,
                                    std::uint32_t n_partitions, std::uint32_t expected_total) {
        const std::size_t bytes_partition = n_partitions * sizeof(std::uint32_t);
        const std::size_t bytes_offsets = (n_partitions + 1) * sizeof(std::uint32_t);
        ensure_buffer(device, offsets_buf, cap_offsets_bytes, bytes_offsets);
        const std::uint32_t* counts =
            static_cast<const std::uint32_t*>([counts_buf contents]);
        std::uint32_t* offsets = static_cast<std::uint32_t*>([offsets_buf contents]);
        std::uint32_t running = 0;
        for (std::uint32_t p = 0; p < n_partitions; ++p) {
            offsets[p] = running;
            running += counts[p];
        }
        offsets[n_partitions] = running;
        if (expected_total != 0 && running != expected_total) {
            throw std::runtime_error("hashjoin partition count mismatch");
        }
        (void)bytes_partition;
    }

    JoinResult join_hash_partitioned(const std::int64_t* build_keys, std::uint32_t n_build,
                                     const std::int64_t* probe_keys, std::uint32_t n_probe,
                                     std::chrono::steady_clock::time_point t0, bool linear_scan) {
        JoinResult r;
        r.rows_build = n_build;
        r.rows_probe = n_probe;

        constexpr std::uint32_t kTgThreads = 64;

        const std::size_t bytes_build = static_cast<std::size_t>(n_build) * sizeof(std::int64_t);
        const std::size_t bytes_probe = static_cast<std::size_t>(n_probe) * sizeof(std::int64_t);
        const std::size_t bytes_partition = kNumPartitions * sizeof(std::uint32_t);

        const bool build_cache_hit =
            build_scatter_valid_ && cached_build_keys_ == build_keys && cached_n_build_ == n_build;

        @autoreleasepool {
            ensure_buffer(device_, b_build_keys_, cap_build_bytes_, bytes_build);
            ensure_buffer(device_, b_build_idx_, cap_build_idx_bytes_, bytes_build);
            ensure_buffer(device_, b_probe_keys_, cap_probe_bytes_, bytes_probe);
            ensure_buffer(device_, b_probe_idx_, cap_probe_idx_bytes_, bytes_probe);
            ensure_buffer(device_, b_scatter_build_keys_, cap_scatter_build_keys_bytes_, bytes_build);
            ensure_buffer(device_, b_scatter_build_idx_, cap_scatter_build_idx_bytes_, bytes_build);
            ensure_buffer(device_, b_scatter_probe_keys_, cap_scatter_probe_keys_bytes_, bytes_probe);
            ensure_buffer(device_, b_scatter_probe_idx_, cap_scatter_probe_idx_bytes_, bytes_probe);
            ensure_buffer(device_, b_partition_counts_, cap_partition_counts_bytes_, bytes_partition);
            ensure_buffer(device_, b_partition_writepos_, cap_partition_writepos_bytes_, bytes_partition);
            ensure_buffer(device_, b_out_probe_, cap_out_probe_bytes_, bytes_probe);
            ensure_buffer(device_, b_out_build_, cap_out_build_bytes_, bytes_probe);
            if (!b_out_count_ || [b_out_count_ length] < sizeof(std::uint32_t)) {
                b_out_count_ = [device_ newBufferWithLength:sizeof(std::uint32_t)
                                                    options:MTLResourceStorageModeShared];
            }

            std::memcpy([b_probe_keys_ contents], probe_keys, bytes_probe);
            std::memset([b_out_count_ contents], 0, sizeof(std::uint32_t));

            const std::uint32_t blocks_build = (n_build + kCountWork - 1) / kCountWork;
            const std::uint32_t blocks_probe = (n_probe + kCountWork - 1) / kCountWork;

            double kernel_ms = 0.0;

            if (!build_cache_hit) {
                std::memcpy([b_build_keys_ contents], build_keys, bytes_build);
                std::memset([b_partition_counts_ contents], 0, bytes_partition);

                id<MTLCommandBuffer> cb_build01 = [queue_ commandBuffer];
                id<MTLComputeCommandEncoder> ce_build01 = [cb_build01 computeCommandEncoder];
                [ce_build01 setComputePipelineState:ps_fill_iota_];
                [ce_build01 setBuffer:b_build_idx_ offset:0 atIndex:0];
                [ce_build01 setBytes:&n_build length:sizeof(n_build) atIndex:1];
                [ce_build01 dispatchThreadgroups:MTLSizeMake(blocks_build, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(kCountThreads, 1, 1)];

                [ce_build01 setComputePipelineState:ps_partition_count_];
                [ce_build01 setBuffer:b_build_keys_ offset:0 atIndex:0];
                [ce_build01 setBytes:&n_build length:sizeof(n_build) atIndex:1];
                [ce_build01 setBuffer:b_partition_counts_ offset:0 atIndex:2];
                [ce_build01 dispatchThreadgroups:MTLSizeMake(blocks_build, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(kCountThreads, 1, 1)];
                [ce_build01 endEncoding];
                [cb_build01 commit];
                [cb_build01 waitUntilCompleted];
                kernel_ms += ([cb_build01 GPUEndTime] - [cb_build01 GPUStartTime]) * 1000.0;

                host_exclusive_scan(b_partition_counts_, b_build_partition_offsets_,
                                    cap_build_partition_offsets_bytes_, device_, kNumPartitions,
                                    n_build);

                std::memset([b_partition_writepos_ contents], 0, bytes_partition);

                id<MTLCommandBuffer> cb_build2 = [queue_ commandBuffer];
                id<MTLComputeCommandEncoder> ce_build2 = [cb_build2 computeCommandEncoder];
                [ce_build2 setComputePipelineState:ps_partition_scatter_];
                [ce_build2 setBuffer:b_build_keys_ offset:0 atIndex:0];
                [ce_build2 setBuffer:b_build_idx_ offset:0 atIndex:1];
                [ce_build2 setBytes:&n_build length:sizeof(n_build) atIndex:2];
                [ce_build2 setBuffer:b_build_partition_offsets_ offset:0 atIndex:3];
                [ce_build2 setBuffer:b_partition_writepos_ offset:0 atIndex:4];
                [ce_build2 setBuffer:b_scatter_build_keys_ offset:0 atIndex:5];
                [ce_build2 setBuffer:b_scatter_build_idx_ offset:0 atIndex:6];
                [ce_build2 dispatchThreadgroups:MTLSizeMake(blocks_build, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(kCountThreads, 1, 1)];
                [ce_build2 endEncoding];
                [cb_build2 commit];
                [cb_build2 waitUntilCompleted];
                kernel_ms += ([cb_build2 GPUEndTime] - [cb_build2 GPUStartTime]) * 1000.0;

                cached_build_keys_ = build_keys;
                cached_n_build_ = n_build;
                build_scatter_valid_ = true;
            }

            std::memset([b_partition_counts_ contents], 0, bytes_partition);

            id<MTLCommandBuffer> cb_probe01 = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> ce_probe01 = [cb_probe01 computeCommandEncoder];
            [ce_probe01 setComputePipelineState:ps_fill_iota_];
            [ce_probe01 setBuffer:b_probe_idx_ offset:0 atIndex:0];
            [ce_probe01 setBytes:&n_probe length:sizeof(n_probe) atIndex:1];
            [ce_probe01 dispatchThreadgroups:MTLSizeMake(blocks_probe, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(kCountThreads, 1, 1)];

            [ce_probe01 setComputePipelineState:ps_partition_count_];
            [ce_probe01 setBuffer:b_probe_keys_ offset:0 atIndex:0];
            [ce_probe01 setBytes:&n_probe length:sizeof(n_probe) atIndex:1];
            [ce_probe01 setBuffer:b_partition_counts_ offset:0 atIndex:2];
            [ce_probe01 dispatchThreadgroups:MTLSizeMake(blocks_probe, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(kCountThreads, 1, 1)];
            [ce_probe01 endEncoding];
            [cb_probe01 commit];
            [cb_probe01 waitUntilCompleted];
            kernel_ms += ([cb_probe01 GPUEndTime] - [cb_probe01 GPUStartTime]) * 1000.0;

            host_exclusive_scan(b_partition_counts_, b_probe_partition_offsets_,
                                cap_probe_partition_offsets_bytes_, device_, kNumPartitions,
                                n_probe);

            std::memset([b_partition_writepos_ contents], 0, bytes_partition);

            id<MTLCommandBuffer> cb_final = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> ce_final = [cb_final computeCommandEncoder];

            [ce_final setComputePipelineState:ps_partition_scatter_];
            [ce_final setBuffer:b_probe_keys_ offset:0 atIndex:0];
            [ce_final setBuffer:b_probe_idx_ offset:0 atIndex:1];
            [ce_final setBytes:&n_probe length:sizeof(n_probe) atIndex:2];
            [ce_final setBuffer:b_probe_partition_offsets_ offset:0 atIndex:3];
            [ce_final setBuffer:b_partition_writepos_ offset:0 atIndex:4];
            [ce_final setBuffer:b_scatter_probe_keys_ offset:0 atIndex:5];
            [ce_final setBuffer:b_scatter_probe_idx_ offset:0 atIndex:6];
            [ce_final dispatchThreadgroups:MTLSizeMake(blocks_probe, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(kCountThreads, 1, 1)];

            if (linear_scan) {
                [ce_final setComputePipelineState:ps_probe_partition_];
                [ce_final setBuffer:b_probe_keys_ offset:0 atIndex:0];
                [ce_final setBytes:&n_probe length:sizeof(n_probe) atIndex:1];
                [ce_final setBuffer:b_scatter_build_keys_ offset:0 atIndex:2];
                [ce_final setBuffer:b_scatter_build_idx_ offset:0 atIndex:3];
                [ce_final setBuffer:b_build_partition_offsets_ offset:0 atIndex:4];
                [ce_final setBuffer:b_out_probe_ offset:0 atIndex:5];
                [ce_final setBuffer:b_out_build_ offset:0 atIndex:6];
                [ce_final setBuffer:b_out_count_ offset:0 atIndex:7];
                [ce_final dispatchThreadgroups:MTLSizeMake(blocks_probe, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(kCountThreads, 1, 1)];
            } else {
                [ce_final setComputePipelineState:ps_partition_hashjoin_];
                [ce_final setBuffer:b_scatter_build_keys_ offset:0 atIndex:0];
                [ce_final setBuffer:b_scatter_build_idx_ offset:0 atIndex:1];
                [ce_final setBuffer:b_build_partition_offsets_ offset:0 atIndex:2];
                [ce_final setBuffer:b_scatter_probe_keys_ offset:0 atIndex:3];
                [ce_final setBuffer:b_scatter_probe_idx_ offset:0 atIndex:4];
                [ce_final setBuffer:b_probe_partition_offsets_ offset:0 atIndex:5];
                [ce_final setBuffer:b_out_count_ offset:0 atIndex:6];
                [ce_final setBuffer:b_out_probe_ offset:0 atIndex:7];
                [ce_final setBuffer:b_out_build_ offset:0 atIndex:8];
                [ce_final dispatchThreadgroups:MTLSizeMake(kNumPartitions, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(kTgThreads, 1, 1)];
            }

            [ce_final endEncoding];
            [cb_final commit];
            [cb_final waitUntilCompleted];
            kernel_ms += ([cb_final GPUEndTime] - [cb_final GPUStartTime]) * 1000.0;

            const std::uint32_t matched = *static_cast<const std::uint32_t*>([b_out_count_ contents]);
            r.matched = matched;
            r.probe_indices.resize(matched);
            r.build_indices.resize(matched);
            if (matched > 0) {
                std::memcpy(r.probe_indices.data(), [b_out_probe_ contents], matched * sizeof(std::int64_t));
                std::memcpy(r.build_indices.data(), [b_out_build_ contents], matched * sizeof(std::int64_t));
            }
            r.kernel_ms = kernel_ms;
        }

        r.transfer_ms = 0.0;
        r.wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        return r;
    }

    JoinResult join_hash_global(const std::int64_t* build_keys, std::uint32_t n_build,
                                const std::int64_t* probe_keys, std::uint32_t n_probe,
                                std::uint32_t table_cap,
                                std::chrono::steady_clock::time_point t0) {
        JoinResult r;
        r.rows_build = n_build;
        r.rows_probe = n_probe;

        const std::size_t bytes_build = static_cast<std::size_t>(n_build) * sizeof(std::int64_t);
        const std::size_t bytes_probe = static_cast<std::size_t>(n_probe) * sizeof(std::int64_t);
        const std::size_t bytes_table = static_cast<std::size_t>(table_cap) * sizeof(std::int64_t);
        const std::size_t bytes_state = static_cast<std::size_t>(table_cap) * sizeof(std::uint32_t);

        @autoreleasepool {
            ensure_buffer(device_, b_build_keys_, cap_build_bytes_, bytes_build);
            ensure_buffer(device_, b_probe_keys_, cap_probe_bytes_, bytes_probe);
            ensure_buffer(device_, b_table_state_, cap_table_state_bytes_, bytes_state);
            ensure_buffer(device_, b_table_keys_, cap_table_keys_bytes_, bytes_table);
            ensure_buffer(device_, b_table_idx_, cap_table_idx_bytes_, bytes_table);
            ensure_buffer(device_, b_out_probe_, cap_out_probe_bytes_, bytes_probe);
            ensure_buffer(device_, b_out_build_, cap_out_build_bytes_, bytes_probe);
            if (!b_out_count_ || [b_out_count_ length] < sizeof(std::uint32_t)) {
                b_out_count_ = [device_ newBufferWithLength:sizeof(std::uint32_t)
                                                    options:MTLResourceStorageModeShared];
            }

            std::memcpy([b_build_keys_ contents], build_keys, bytes_build);
            std::memcpy([b_probe_keys_ contents], probe_keys, bytes_probe);
            std::memset([b_out_count_ contents], 0, sizeof(std::uint32_t));

            id<MTLCommandBuffer> cb = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];

            const NSUInteger init_tg = (table_cap + kBlock - 1) / kBlock;
            [ce setComputePipelineState:ps_init_];
            [ce setBuffer:b_table_state_ offset:0 atIndex:0];
            [ce setBuffer:b_table_keys_ offset:0 atIndex:1];
            [ce setBytes:&table_cap length:sizeof(table_cap) atIndex:2];
            [ce dispatchThreadgroups:MTLSizeMake(init_tg, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];

            const NSUInteger build_tg = std::min<NSUInteger>((n_build + kBlock - 1) / kBlock, 4096);
            [ce setComputePipelineState:ps_build_];
            [ce setBuffer:b_build_keys_ offset:0 atIndex:0];
            [ce setBytes:&n_build length:sizeof(n_build) atIndex:1];
            [ce setBuffer:b_table_state_ offset:0 atIndex:2];
            [ce setBuffer:b_table_keys_ offset:0 atIndex:3];
            [ce setBuffer:b_table_idx_ offset:0 atIndex:4];
            [ce setBytes:&table_cap length:sizeof(table_cap) atIndex:5];
            [ce dispatchThreadgroups:MTLSizeMake(build_tg, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];

            const NSUInteger probe_tg = std::min<NSUInteger>((n_probe + kBlock - 1) / kBlock, 4096);
            [ce setComputePipelineState:ps_probe_global_];
            [ce setBuffer:b_probe_keys_ offset:0 atIndex:0];
            [ce setBytes:&n_probe length:sizeof(n_probe) atIndex:1];
            [ce setBuffer:b_table_state_ offset:0 atIndex:2];
            [ce setBuffer:b_table_keys_ offset:0 atIndex:3];
            [ce setBuffer:b_table_idx_ offset:0 atIndex:4];
            [ce setBytes:&table_cap length:sizeof(table_cap) atIndex:5];
            [ce setBuffer:b_out_probe_ offset:0 atIndex:6];
            [ce setBuffer:b_out_build_ offset:0 atIndex:7];
            [ce setBuffer:b_out_count_ offset:0 atIndex:8];
            [ce dispatchThreadgroups:MTLSizeMake(probe_tg, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];

            [ce endEncoding];
            [cb commit];
            [cb waitUntilCompleted];

            r.kernel_ms = ([cb GPUEndTime] - [cb GPUStartTime]) * 1000.0;

            const std::uint32_t matched = *static_cast<const std::uint32_t*>([b_out_count_ contents]);
            r.matched = matched;
            r.probe_indices.resize(matched);
            r.build_indices.resize(matched);
            if (matched > 0) {
                std::memcpy(r.probe_indices.data(), [b_out_probe_ contents], matched * sizeof(std::int64_t));
                std::memcpy(r.build_indices.data(), [b_out_build_ contents], matched * sizeof(std::int64_t));
            }
        }

        r.transfer_ms = 0.0;
        r.wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        return r;
    }

    JoinResult join_sort_merge(const std::int64_t* build_keys, std::uint32_t n_build,
                               const std::int64_t* probe_keys, std::uint32_t n_probe,
                               std::chrono::steady_clock::time_point t0) {
        JoinResult r;
        r.rows_build = n_build;
        r.rows_probe = n_probe;

        std::vector<std::int64_t> build_idx(n_build);
        for (std::uint32_t j = 0; j < n_build; ++j) build_idx[j] = static_cast<std::int64_t>(j);

        std::vector<std::int64_t> probe_idx(n_probe);
        for (std::uint32_t i = 0; i < n_probe; ++i) probe_idx[i] = static_cast<std::int64_t>(i);

        double kernel_ms = 0.0;
        const auto sorted =
            radix_->sort_device(build_keys, build_idx.data(), n_build);
        kernel_ms += sorted.kernel_ms;

        const std::size_t bytes_probe = static_cast<std::size_t>(n_probe) * sizeof(std::int64_t);

        @autoreleasepool {
            ensure_buffer(device_, b_probe_keys_, cap_probe_bytes_, bytes_probe);
            ensure_buffer(device_, b_probe_idx_, cap_probe_idx_bytes_, bytes_probe);
            ensure_buffer(device_, b_out_probe_, cap_out_probe_bytes_, bytes_probe);
            ensure_buffer(device_, b_out_build_, cap_out_build_bytes_, bytes_probe);
            if (!b_out_count_ || [b_out_count_ length] < sizeof(std::uint32_t)) {
                b_out_count_ = [device_ newBufferWithLength:sizeof(std::uint32_t)
                                                    options:MTLResourceStorageModeShared];
            }

            std::memcpy([b_probe_keys_ contents], probe_keys, bytes_probe);
            std::memcpy([b_probe_idx_ contents], probe_idx.data(), bytes_probe);
            std::memset([b_out_count_ contents], 0, sizeof(std::uint32_t));

            id<MTLCommandBuffer> cb = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
            [ce setComputePipelineState:ps_merge_];
            [ce setBuffer:b_probe_keys_ offset:0 atIndex:0];
            [ce setBuffer:b_probe_idx_ offset:0 atIndex:1];
            [ce setBuffer:sorted.keys offset:0 atIndex:2];
            [ce setBuffer:sorted.payloads offset:0 atIndex:3];
            [ce setBytes:&n_probe length:sizeof(n_probe) atIndex:4];
            [ce setBytes:&n_build length:sizeof(n_build) atIndex:5];
            [ce setBuffer:b_out_probe_ offset:0 atIndex:6];
            [ce setBuffer:b_out_build_ offset:0 atIndex:7];
            [ce setBuffer:b_out_count_ offset:0 atIndex:8];

            const NSUInteger tg = (n_probe + 255) / 256;
            [ce dispatchThreadgroups:MTLSizeMake(tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [ce endEncoding];
            [cb commit];
            [cb waitUntilCompleted];

            kernel_ms += ([cb GPUEndTime] - [cb GPUStartTime]) * 1000.0;

            const std::uint32_t matched = *static_cast<const std::uint32_t*>([b_out_count_ contents]);
            r.matched = matched;
            r.probe_indices.resize(matched);
            r.build_indices.resize(matched);
            if (matched > 0) {
                std::memcpy(r.probe_indices.data(), [b_out_probe_ contents], matched * sizeof(std::int64_t));
                std::memcpy(r.build_indices.data(), [b_out_build_ contents], matched * sizeof(std::int64_t));
            }
        }

        r.kernel_ms = kernel_ms;
        r.transfer_ms = 0.0;
        r.wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        return r;
    }

    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    id<MTLComputePipelineState> ps_merge_ = nil;
    id<MTLComputePipelineState> ps_init_ = nil;
    id<MTLComputePipelineState> ps_build_ = nil;
    id<MTLComputePipelineState> ps_probe_global_ = nil;
    id<MTLComputePipelineState> ps_partition_count_ = nil;
    id<MTLComputePipelineState> ps_partition_scatter_ = nil;
    id<MTLComputePipelineState> ps_probe_partition_ = nil;
    id<MTLComputePipelineState> ps_partition_hashjoin_ = nil;
    id<MTLComputePipelineState> ps_fill_iota_ = nil;
    std::unique_ptr<metal_detail::MetalRadixSort> radix_;

    id<MTLBuffer> b_build_keys_ = nil;
    id<MTLBuffer> b_build_idx_ = nil;
    id<MTLBuffer> b_probe_keys_ = nil;
    id<MTLBuffer> b_probe_idx_ = nil;
    id<MTLBuffer> b_scatter_build_keys_ = nil;
    id<MTLBuffer> b_scatter_build_idx_ = nil;
    id<MTLBuffer> b_scatter_probe_keys_ = nil;
    id<MTLBuffer> b_scatter_probe_idx_ = nil;
    id<MTLBuffer> b_partition_counts_ = nil;
    id<MTLBuffer> b_build_partition_offsets_ = nil;
    id<MTLBuffer> b_probe_partition_offsets_ = nil;
    id<MTLBuffer> b_partition_writepos_ = nil;
    id<MTLBuffer> b_table_state_ = nil;
    id<MTLBuffer> b_table_keys_ = nil;
    id<MTLBuffer> b_table_idx_ = nil;
    id<MTLBuffer> b_out_probe_ = nil;
    id<MTLBuffer> b_out_build_ = nil;
    id<MTLBuffer> b_out_count_ = nil;
    std::size_t cap_build_bytes_ = 0;
    std::size_t cap_build_idx_bytes_ = 0;
    std::size_t cap_probe_bytes_ = 0;
    std::size_t cap_probe_idx_bytes_ = 0;
    std::size_t cap_scatter_build_keys_bytes_ = 0;
    std::size_t cap_scatter_build_idx_bytes_ = 0;
    std::size_t cap_scatter_probe_keys_bytes_ = 0;
    std::size_t cap_scatter_probe_idx_bytes_ = 0;
    std::size_t cap_partition_counts_bytes_ = 0;
    std::size_t cap_build_partition_offsets_bytes_ = 0;
    std::size_t cap_probe_partition_offsets_bytes_ = 0;
    std::size_t cap_partition_writepos_bytes_ = 0;
    std::size_t cap_table_state_bytes_ = 0;
    std::size_t cap_table_keys_bytes_ = 0;
    std::size_t cap_table_idx_bytes_ = 0;
    std::size_t cap_out_probe_bytes_ = 0;
    std::size_t cap_out_build_bytes_ = 0;
    bool path_logged_ = false;
    const std::int64_t* cached_build_keys_ = nullptr;
    std::uint32_t cached_n_build_ = 0;
    bool build_scatter_valid_ = false;
};

} // namespace

std::unique_ptr<HashJoinProbe> make_metal_hashjoin_probe() {
    return std::make_unique<MetalHashJoinProbe>();
}

} // namespace gpudb
