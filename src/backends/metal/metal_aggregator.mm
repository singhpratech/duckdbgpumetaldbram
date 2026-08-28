// metal_aggregator.mm — Apple Silicon Metal backend (real compute pipelines).
//
// macOS-only. Compiles the .metal kernel sources at runtime via
// MTLDevice newLibraryWithSource:options:error:. The kernel source is
// embedded as a C string by CMake (see metal_kernel_sources.hpp.in).
//
// Strategy:
//   - One MTLBuffer per buffer slot, sized at runtime (resized on growth).
//   - MTLResourceStorageModeShared everywhere — UMA means the GPU reads the
//     same physical pages the CPU wrote. transfer_ms is therefore reported
//     as 0 (the cost is just the memcpy into the shared buffer, which we
//     count as wall, not transfer).
//   - kernel_ms is measured via GPUStartTime/GPUEndTime on the command buffer
//     (Metal exposes these once the buffer has completed).
//   - f64 sum stays on the CPU because Apple Silicon GPUs do not implement
//     IEEE-754 double precision in MSL.

#include "gpu_backend.hpp"
#include "metal_kernel_sources.hpp"
#include "metal_radix_sort.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace gpudb {

namespace {

constexpr NSUInteger kBlock = 256;
constexpr NSUInteger kMaxGrid = 4096;

[[noreturn]] void metal_throw(const char* what, NSError* err) {
    std::ostringstream os;
    os << "Metal " << what;
    if (err) os << ": " << [[err localizedDescription] UTF8String];
    throw std::runtime_error(os.str());
}

NSUInteger pick_grid(std::size_t n) {
    NSUInteger g = (n + kBlock - 1) / kBlock;
    if (g < 1)        g = 1;
    if (g > kMaxGrid) g = kMaxGrid;
    return g;
}

double cb_kernel_ms(id<MTLCommandBuffer> cb) {
    // GPUStart/EndTime are CFAbsoluteTime (seconds). Available after completion.
    const double s = [cb GPUStartTime];
    const double e = [cb GPUEndTime];
    return (e - s) * 1000.0;
}

class MetalAggregator final : public Aggregator {
public:
    MetalAggregator() {
        @autoreleasepool {
            device_ = MTLCreateSystemDefaultDevice();
            if (!device_) throw std::runtime_error("MTLCreateSystemDefaultDevice returned nil");
            queue_ = [device_ newCommandQueue];
            if (!queue_) throw std::runtime_error("Failed to create Metal command queue");

            NSError* err = nil;
            NSString* src = [NSString stringWithUTF8String:metal::kSumKernelSource];
            MTLCompileOptions* opts = [MTLCompileOptions new];
            id<MTLLibrary> lib = [device_ newLibraryWithSource:src options:opts error:&err];
            if (!lib) metal_throw("compile sum.metal", err);

            ps_sum_i64_           = make_pso(lib, @"sum_i64");
            ps_sum_partials_i64_  = make_pso(lib, @"sum_partials_i64");
            ps_min_i64_           = make_pso(lib, @"min_i64");
            ps_min_partials_i64_  = make_pso(lib, @"min_partials_i64");
            ps_max_i64_           = make_pso(lib, @"max_i64");
            ps_max_partials_i64_  = make_pso(lib, @"max_partials_i64");
            ps_agg_all_i64_           = make_pso(lib, @"agg_all_i64");
            ps_agg_all_partials_i64_  = make_pso(lib, @"agg_all_partials_i64");
            ps_join_sum_i64_          = make_pso(lib, @"join_sum_i64");
            ps_join_sum_partials_i64_ = make_pso(lib, @"join_sum_partials_i64");
            ps_join_mult_i64_         = make_pso(lib, @"join_mult_i64");
            ps_join_lookup_i64_       = make_pso(lib, @"join_lookup_i64");
            ps_gb_block_counts_       = make_pso(lib, @"gb_block_counts_i64");
            ps_gb_run_starts_         = make_pso(lib, @"gb_run_starts_i64");
            ps_gb_chunk_sum_          = make_pso(lib, @"gb_chunk_sum_i64");
            ps_gb_finalize_           = make_pso(lib, @"gb_finalize_i64");
            ps_gb_gather_             = make_pso(lib, @"gb_gather_i64");

            partials_buf_ = [device_ newBufferWithLength:(kMaxGrid * sizeof(std::int64_t))
                                                 options:MTLResourceStorageModeShared];
            out_buf_      = [device_ newBufferWithLength:sizeof(std::int64_t)
                                                 options:MTLResourceStorageModeShared];
            // Multi-agg fusion needs 4 longs per block (sum/min/max/count)
            // and 4 longs of output.
            partials_quad_buf_ = [device_ newBufferWithLength:(kMaxGrid * 4 * sizeof(std::int64_t))
                                                      options:MTLResourceStorageModeShared];
            out_quad_buf_      = [device_ newBufferWithLength:(4 * sizeof(std::int64_t))
                                                      options:MTLResourceStorageModeShared];
        }
    }

    ~MetalAggregator() override = default;  // ARC

    Backend backend() const noexcept override { return Backend::METAL; }

    std::string device_name() const override {
        @autoreleasepool {
            NSString* name = [device_ name];
            std::ostringstream os;
            os << [name UTF8String] << " (Metal)";
            return os.str();
        }
    }

    // ---- One-shot: copy host data into a shared MTLBuffer, dispatch, read back ----
    AggResult sum_i64(const std::int64_t* data, std::size_t n) override {
        return run_i64(data, n, ps_sum_i64_, ps_sum_partials_i64_, /*has_init*/false, 0);
    }
    AggResult min_i64(const std::int64_t* data, std::size_t n) override {
        return run_i64(data, n, ps_min_i64_, ps_min_partials_i64_, /*has_init*/true,
                       std::numeric_limits<std::int64_t>::max());
    }
    AggResult max_i64(const std::int64_t* data, std::size_t n) override {
        return run_i64(data, n, ps_max_i64_, ps_max_partials_i64_, /*has_init*/true,
                       std::numeric_limits<std::int64_t>::min());
    }
    AggResult sum_f64(const double* data, std::size_t n) override {
        return host_sum_f64(data, n);
    }

    // ---- Resident column ----
    std::unique_ptr<ResidentColumn> upload_i64(const std::int64_t* d, std::size_t n) override {
        return make_resident(d, n, Dtype::I64, sizeof(std::int64_t));
    }
    std::unique_ptr<ResidentColumn> upload_f64(const double* d, std::size_t n) override {
        return make_resident(d, n, Dtype::F64, sizeof(double));
    }
    AggResult sum_resident_i64(const ResidentColumn& c) override {
        const auto& r = check_i64(c);
        return run_i64_resident(r.buffer(), r.rows(),
                                ps_sum_i64_, ps_sum_partials_i64_, false, 0);
    }
    AggResult min_resident_i64(const ResidentColumn& c) override {
        const auto& r = check_i64(c);
        return run_i64_resident(r.buffer(), r.rows(),
                                ps_min_i64_, ps_min_partials_i64_, true,
                                std::numeric_limits<std::int64_t>::max());
    }
    AggResult max_resident_i64(const ResidentColumn& c) override {
        const auto& r = check_i64(c);
        return run_i64_resident(r.buffer(), r.rows(),
                                ps_max_i64_, ps_max_partials_i64_, true,
                                std::numeric_limits<std::int64_t>::min());
    }
    AggResult sum_resident_f64(const ResidentColumn& c) override {
        const auto& r = check_f64(c);
        return host_sum_f64(static_cast<const double*>([r.buffer() contents]), r.rows());
    }

    AggAllResult agg_all_i64(const std::int64_t* data, std::size_t n) override {
        @autoreleasepool {
            const auto t_wall0 = std::chrono::steady_clock::now();
            if (n == 0) return empty_agg_all(n, t_wall0);
            id<MTLBuffer> in = stage_input(data, n * sizeof(std::int64_t));
            return dispatch_agg_all_i64(in, n, t_wall0);
        }
    }
    AggAllResult agg_all_resident_i64(const ResidentColumn& c) override {
        @autoreleasepool {
            const auto& r = check_i64(c);
            const auto t_wall0 = std::chrono::steady_clock::now();
            if (r.rows() == 0) return empty_agg_all(0, t_wall0);
            return dispatch_agg_all_i64(r.buffer(), r.rows(), t_wall0);
        }
    }

    // Get (or lazily build + cache) the radix-sorted copy of a build-key
    // column. Sort cost is reported through *sort_kernel_ms on the call that
    // pays it; later calls reuse the cache for free.
    id<MTLBuffer> ensure_sorted_cache(const ResidentColumn& build_col,
                                      double* sort_kernel_ms) {
        const auto& bk = check_i64(build_col);
        id<MTLBuffer> sorted = bk.sorted_cache();
        if (sorted) return sorted;
        if (bk.rows() > 0xFFFFFFFFull)
            throw std::runtime_error("resident join: > 2^32 build rows unsupported");
        if (!sorter_)
            sorter_ = std::make_unique<metal_detail::MetalRadixSort>(device_, queue_);
        const std::size_t n = bk.rows();
        const auto* keys = static_cast<const std::int64_t*>([bk.buffer() contents]);
        // Sort (key, original-index) pairs: the payload comes back as the
        // permutation the row join needs.
        std::vector<std::int64_t> idx(n);
        for (std::size_t j = 0; j < n; ++j) idx[j] = static_cast<std::int64_t>(j);
        auto view = sorter_->sort_device(keys, idx.data(),
                                         static_cast<std::uint32_t>(n));
        sorted             = [device_ newBufferWithLength:n * sizeof(std::int64_t)
                                                  options:MTLResourceStorageModeShared];
        id<MTLBuffer> perm = [device_ newBufferWithLength:n * sizeof(std::int64_t)
                                                  options:MTLResourceStorageModeShared];
        if (!sorted || !perm)
            throw std::runtime_error(
                "resident join: device allocation for sorted build cache failed");
        std::memcpy([sorted contents], [view.keys contents],     n * sizeof(std::int64_t));
        std::memcpy([perm contents],   [view.payloads contents], n * sizeof(std::int64_t));
        bk.set_sorted_cache(sorted, perm);
        *sort_kernel_ms += view.kernel_ms;
        return sorted;
    }

    JoinAggResult join_sum_resident_i64(const ResidentColumn& probe_keys,
                                        const ResidentColumn& payload,
                                        const ResidentColumn& build_keys,
                                        JoinKind kind) override {
        @autoreleasepool {
            const auto t_wall0 = std::chrono::steady_clock::now();
            const auto& pk = check_i64(probe_keys);
            const auto& pl = check_i64(payload);
            const auto& bk = check_i64(build_keys);
            if (pk.rows() != pl.rows())
                throw std::runtime_error(
                    "join_sum_resident_i64: probe_keys and payload row counts differ");

            JoinAggResult r{};
            r.rows_probe = pk.rows();
            r.rows_build = bk.rows();
            if (pk.rows() == 0) {
                r.wall_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - t_wall0).count();
                return r;
            }
            // Kernel indices are 32-bit (same convention as every kernel here).
            if (pk.rows() > 0xFFFFFFFFull || bk.rows() > 0xFFFFFFFFull)
                throw std::runtime_error("join_sum_resident_i64: > 2^32 rows unsupported");

            double sort_kernel_ms = 0.0;
            id<MTLBuffer> sorted = ensure_sorted_cache(bk, &sort_kernel_ms);

            const NSUInteger grid = pick_grid(pk.rows());
            const std::uint32_t np32 = static_cast<std::uint32_t>(pk.rows());
            const std::uint32_t nb32 = static_cast<std::uint32_t>(bk.rows());
            const std::uint32_t mode = static_cast<std::uint32_t>(kind);

            id<MTLCommandBuffer>         cb = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];

            // Pass 1: per-threadgroup join+reduce. partials_quad_buf_ holds
            // 4 longs per block; we use the first 2 (sum, matched).
            [ce setComputePipelineState:ps_join_sum_i64_];
            [ce setBuffer:pk.buffer()       offset:0 atIndex:0];
            [ce setBuffer:pl.buffer()       offset:0 atIndex:1];
            [ce setBuffer:sorted            offset:0 atIndex:2];
            [ce setBuffer:partials_quad_buf_ offset:0 atIndex:3];
            [ce setBytes:&np32 length:sizeof(np32) atIndex:4];
            [ce setBytes:&nb32 length:sizeof(nb32) atIndex:5];
            [ce setBytes:&mode length:sizeof(mode) atIndex:6];
            [ce dispatchThreadgroups:MTLSizeMake(grid, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];

            // Pass 2: reduce block partials to (sum, matched).
            const std::uint32_t nblocks = static_cast<std::uint32_t>(grid);
            [ce setComputePipelineState:ps_join_sum_partials_i64_];
            [ce setBuffer:partials_quad_buf_ offset:0 atIndex:0];
            [ce setBuffer:out_quad_buf_      offset:0 atIndex:1];
            [ce setBytes:&nblocks length:sizeof(nblocks) atIndex:2];
            [ce dispatchThreadgroups:MTLSizeMake(1, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];

            [ce endEncoding];
            [cb commit];
            [cb waitUntilCompleted];

            const auto* out = static_cast<const std::int64_t*>([out_quad_buf_ contents]);
            r.sum         = out[0];
            r.matched     = out[1];
            r.kernel_ms   = cb_kernel_ms(cb) + sort_kernel_ms;
            r.transfer_ms = 0.0;  // UMA
            r.wall_ms     = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - t_wall0).count();
            return r;
        }
    }

    // f64 payload: no doubles in MSL, so the probe pass runs as a parallel
    // HOST loop over the UMA buffers — but it still reuses the GPU-built
    // sorted-key cache, so repeated f64 joins skip the sort like i64 ones.
    // Same pattern as sum_resident_f64 (host math on device-held data).
    JoinAggResult join_sum_resident_f64(const ResidentColumn& probe_keys,
                                        const ResidentColumn& payload,
                                        const ResidentColumn& build_keys,
                                        JoinKind kind) override {
        @autoreleasepool {
            const auto t_wall0 = std::chrono::steady_clock::now();
            const auto& pk = check_i64(probe_keys);
            const auto& pl = check_f64(payload);
            const auto& bk = check_i64(build_keys);
            if (pk.rows() != pl.rows())
                throw std::runtime_error(
                    "join_sum_resident_f64: probe_keys and payload row counts differ");

            JoinAggResult r{};
            r.rows_probe = pk.rows();
            r.rows_build = bk.rows();
            if (pk.rows() == 0) {
                r.wall_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - t_wall0).count();
                return r;
            }

            if (pk.rows() > 0xFFFFFFFFull || bk.rows() > 0xFFFFFFFFull)
                throw std::runtime_error("join_sum_resident_f64: > 2^32 rows unsupported");

            double sort_kernel_ms = 0.0;
            id<MTLBuffer> sorted_buf = ensure_sorted_cache(bk, &sort_kernel_ms);
            const std::size_t n_probe = pk.rows();
            const std::size_t n_build = bk.rows();

            // Stage 1 (GPU): per-element contribution counts — the random-
            // access binary searches the GPU is fast at.
            const std::size_t mult_bytes = n_probe * sizeof(std::uint32_t);
            if (!mult_buf_ || [mult_buf_ length] < mult_bytes) {
                mult_buf_ = [device_ newBufferWithLength:mult_bytes
                                                 options:MTLResourceStorageModeShared];
                if (!mult_buf_)
                    throw std::runtime_error(
                        "join_sum_resident_f64: multiplicity buffer allocation failed");
            }
            const NSUInteger grid = pick_grid(n_probe);
            const std::uint32_t np32 = static_cast<std::uint32_t>(n_probe);
            const std::uint32_t nb32 = static_cast<std::uint32_t>(n_build);
            const std::uint32_t mode = static_cast<std::uint32_t>(kind);

            id<MTLCommandBuffer>         cb = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
            [ce setComputePipelineState:ps_join_mult_i64_];
            [ce setBuffer:pk.buffer() offset:0 atIndex:0];
            [ce setBuffer:sorted_buf  offset:0 atIndex:1];
            [ce setBuffer:mult_buf_   offset:0 atIndex:2];
            [ce setBytes:&np32 length:sizeof(np32) atIndex:3];
            [ce setBytes:&nb32 length:sizeof(nb32) atIndex:4];
            [ce setBytes:&mode length:sizeof(mode) atIndex:5];
            [ce dispatchThreadgroups:MTLSizeMake(grid, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];
            [ce endEncoding];
            [cb commit];
            [cb waitUntilCompleted];

            // Stage 2 (host, parallel): one sequential multiply-add stream —
            // no random access, saturates memory bandwidth.
            const auto* mult = static_cast<const std::uint32_t*>([mult_buf_ contents]);
            const auto* pay  = static_cast<const double*>([pl.buffer() contents]);
            const unsigned hw = std::thread::hardware_concurrency();
            const std::size_t workers =
                std::max<std::size_t>(1, std::min<std::size_t>(hw ? hw : 1,
                                                               n_probe / 65536 + 1));
            std::vector<double>       sums(workers, 0.0);
            std::vector<std::int64_t> cnts(workers, 0);
            std::vector<std::thread>  threads;
            const std::size_t per = n_probe / workers;
            for (std::size_t w = 0; w < workers; ++w) {
                const std::size_t begin = w * per;
                const std::size_t end   = (w + 1 == workers) ? n_probe : begin + per;
                threads.emplace_back([&, w, begin, end] {
                    double s = 0.0; std::int64_t c_total = 0;
                    for (std::size_t i = begin; i < end; ++i) {
                        const std::uint32_t c = mult[i];
                        if (c) {
                            s += static_cast<double>(c) * pay[i];
                            c_total += c;
                        }
                    }
                    sums[w] = s; cnts[w] = c_total;
                });
            }
            for (auto& t : threads) t.join();
            double sum = 0.0; std::int64_t matched = 0;
            for (std::size_t w = 0; w < workers; ++w) { sum += sums[w]; matched += cnts[w]; }

            r.sum_f64     = sum;
            r.matched     = matched;
            r.kernel_ms   = cb_kernel_ms(cb) + sort_kernel_ms;
            r.transfer_ms = 0.0;
            r.wall_ms     = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - t_wall0).count();
            return r;
        }
    }

    JoinRowsResult join_rows_resident(const ResidentColumn& probe_keys,
                                      const ResidentColumn& build_keys,
                                      JoinKind kind, std::size_t max_rows) override {
        @autoreleasepool {
            const auto t_wall0 = std::chrono::steady_clock::now();
            const auto& pk = check_i64(probe_keys);
            const auto& bk = check_i64(build_keys);

            JoinRowsResult r{};
            r.rows_probe = pk.rows();
            r.rows_build = bk.rows();
            const std::size_t n_probe = pk.rows();
            const std::size_t n_build = bk.rows();
            if (n_probe == 0) {
                r.wall_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - t_wall0).count();
                return r;
            }
            if (n_probe > 0xFFFFFFFFull || n_build > 0xFFFFFFFFull)
                throw std::runtime_error("join_rows_resident: > 2^32 rows unsupported");

            double sort_kernel_ms = 0.0;
            id<MTLBuffer> sorted_buf = ensure_sorted_cache(bk, &sort_kernel_ms);
            id<MTLBuffer> perm_buf   = bk.perm_cache();

            // Stage 1 (GPU): match count + first sorted position per probe.
            const std::size_t u32_bytes = n_probe * sizeof(std::uint32_t);
            if (!mult_buf_ || [mult_buf_ length] < u32_bytes)
                mult_buf_ = [device_ newBufferWithLength:u32_bytes
                                                 options:MTLResourceStorageModeShared];
            if (!first_buf_ || [first_buf_ length] < u32_bytes)
                first_buf_ = [device_ newBufferWithLength:u32_bytes
                                                  options:MTLResourceStorageModeShared];
            if (!mult_buf_ || !first_buf_)
                throw std::runtime_error("join_rows_resident: scratch allocation failed");

            const NSUInteger grid = pick_grid(n_probe);
            const std::uint32_t np32 = static_cast<std::uint32_t>(n_probe);
            const std::uint32_t nb32 = static_cast<std::uint32_t>(n_build);
            id<MTLCommandBuffer>         cb = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
            [ce setComputePipelineState:ps_join_lookup_i64_];
            [ce setBuffer:pk.buffer() offset:0 atIndex:0];
            [ce setBuffer:sorted_buf  offset:0 atIndex:1];
            [ce setBuffer:mult_buf_   offset:0 atIndex:2];
            [ce setBuffer:first_buf_  offset:0 atIndex:3];
            [ce setBytes:&np32 length:sizeof(np32) atIndex:4];
            [ce setBytes:&nb32 length:sizeof(nb32) atIndex:5];
            [ce dispatchThreadgroups:MTLSizeMake(grid, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];
            [ce endEncoding];
            [cb commit];
            [cb waitUntilCompleted];

            // Stage 2 (host, parallel): chunked prefix-sum of the per-kind
            // output counts, then a parallel fill from the perm cache.
            const auto* mc    = static_cast<const std::uint32_t*>([mult_buf_ contents]);
            const auto* first = static_cast<const std::uint32_t*>([first_buf_ contents]);
            const auto* perm  = static_cast<const std::int64_t*>([perm_buf contents]);

            auto count_of = [&](std::size_t i) -> std::size_t {
                const std::uint32_t m = mc[i];
                switch (kind) {
                    case JoinKind::LEFT:  return m ? m : 1;
                    case JoinKind::SEMI:  return m ? 1 : 0;
                    case JoinKind::ANTI:  return m ? 0 : 1;
                    default:              return m;
                }
            };

            const unsigned hw = std::thread::hardware_concurrency();
            const std::size_t workers =
                std::max<std::size_t>(1, std::min<std::size_t>(hw ? hw : 1,
                                                               n_probe / 65536 + 1));
            const std::size_t per = (n_probe + workers - 1) / workers;
            std::vector<std::size_t> chunk_total(workers, 0);
            {
                std::vector<std::thread> ts;
                for (std::size_t w = 0; w < workers; ++w) {
                    const std::size_t b = w * per, e = std::min(n_probe, b + per);
                    ts.emplace_back([&, w, b, e] {
                        std::size_t s = 0;
                        for (std::size_t i = b; i < e; ++i) s += count_of(i);
                        chunk_total[w] = s;
                    });
                }
                for (auto& t : ts) t.join();
            }
            std::size_t total = 0;
            std::vector<std::size_t> chunk_off(workers, 0);
            for (std::size_t w = 0; w < workers; ++w) { chunk_off[w] = total; total += chunk_total[w]; }
            if (total > max_rows)
                throw std::runtime_error(
                    "join_rows_resident: result has " + std::to_string(total) +
                    " rows, above the cap of " + std::to_string(max_rows) +
                    " (raise GPUDB_JOIN_ROWS_MAX_M if intentional)");

            r.probe_idx.resize(total);
            r.build_idx.resize(total);
            {
                std::vector<std::thread> ts;
                for (std::size_t w = 0; w < workers; ++w) {
                    const std::size_t b = w * per, e = std::min(n_probe, b + per);
                    ts.emplace_back([&, w, b, e] {
                        std::size_t off = chunk_off[w];
                        for (std::size_t i = b; i < e; ++i) {
                            const std::uint32_t m = mc[i];
                            const bool matched = m != 0;
                            if ((kind == JoinKind::INNER || kind == JoinKind::LEFT) && matched) {
                                const std::uint32_t f = first[i];
                                for (std::uint32_t t2 = 0; t2 < m; ++t2) {
                                    r.probe_idx[off] = static_cast<std::int64_t>(i);
                                    r.build_idx[off] = perm[f + t2];
                                    ++off;
                                }
                            } else if (count_of(i)) {
                                r.probe_idx[off] = static_cast<std::int64_t>(i);
                                r.build_idx[off] = -1;
                                ++off;
                            }
                        }
                    });
                }
                for (auto& t : ts) t.join();
            }

            r.kernel_ms   = cb_kernel_ms(cb) + sort_kernel_ms;
            r.transfer_ms = 0.0;
            r.wall_ms     = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - t_wall0).count();
            return r;
        }
    }

    // ---- Resident GROUP BY / top-k (v0.6) ----
    // Rides the join's cached radix sort of the key column (sorted keys +
    // original-index permutation, built once per column). See the kernel
    // pipeline comment in sum.metal. Output sorted by key ascending.
    GroupByResidentResult groupby_sum_resident_i64(const ResidentColumn& keys,
                                                   const ResidentColumn& vals,
                                                   std::size_t max_groups) override {
        @autoreleasepool {
            const auto& k = check_i64(keys);
            const auto& v = check_i64(vals);
            if (k.rows() != v.rows())
                throw std::runtime_error(
                    "groupby_sum_resident_i64: keys and vals row counts differ");
            return groupby_impl(k, &v, GbMode::SumI64, max_groups, "groupby_sum_resident_i64");
        }
    }

    GroupByResidentResult groupby_sum_resident_f64(const ResidentColumn& keys,
                                                   const ResidentColumn& vals,
                                                   std::size_t max_groups) override {
        @autoreleasepool {
            const auto& k = check_i64(keys);
            const auto& v = check_f64(vals);
            if (k.rows() != v.rows())
                throw std::runtime_error(
                    "groupby_sum_resident_f64: keys and vals row counts differ");
            return groupby_impl(k, &v, GbMode::SumF64, max_groups, "groupby_sum_resident_f64");
        }
    }

    GroupByResidentResult groupby_count_resident(const ResidentColumn& keys,
                                                 std::size_t max_groups) override {
        @autoreleasepool {
            const auto& k = check_i64(keys);
            return groupby_impl(k, nullptr, GbMode::Count, max_groups, "groupby_count_resident");
        }
    }

    TopKResult topk_resident(const ResidentColumn& col, std::size_t k,
                             bool descending) override {
        @autoreleasepool {
            const auto t_wall0 = std::chrono::steady_clock::now();
            if (col.backend_tag() != Backend::METAL)
                throw std::runtime_error("ResidentColumn mismatch (Metal)");
            const auto& c = static_cast<const MetalResidentColumn&>(col);
            TopKResult r{};
            r.rows_in = c.rows();
            const std::size_t n = c.rows();
            const std::size_t kk = std::min(k, n);
            if (kk == 0) {
                r.wall_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - t_wall0).count();
                return r;
            }
            if (n > 0xFFFFFFFFull)
                throw std::runtime_error("topk_resident: > 2^32 rows unsupported");

            double sort_kernel_ms = 0.0;
            if (c.dtype() == Dtype::I64) {
                ensure_sorted_cache(c, &sort_kernel_ms);
            } else {
                ensure_sorted_cache_f64(c, &sort_kernel_ms);
            }
            const auto* perm = static_cast<const std::int64_t*>([c.perm_cache() contents]);
            r.idx.resize(kk);
            for (std::size_t i = 0; i < kk; ++i)
                r.idx[i] = descending ? perm[n - 1 - i] : perm[i];
            if (c.dtype() == Dtype::I64) {
                const auto* d = static_cast<const std::int64_t*>([c.buffer() contents]);
                r.values_i64.resize(kk);
                for (std::size_t i = 0; i < kk; ++i) r.values_i64[i] = d[r.idx[i]];
            } else {
                const auto* d = static_cast<const double*>([c.buffer() contents]);
                r.values_f64.resize(kk);
                for (std::size_t i = 0; i < kk; ++i) r.values_f64[i] = d[r.idx[i]];
            }
            r.kernel_ms   = sort_kernel_ms;
            r.transfer_ms = 0.0;
            r.wall_ms     = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - t_wall0).count();
            return r;
        }
    }

private:
    class MetalResidentColumn final : public ResidentColumn {
    public:
        MetalResidentColumn(id<MTLBuffer> buf, std::size_t n, Dtype dt)
            : buf_(buf), rows_(n), dtype_(dt) {}
        Backend     backend_tag() const noexcept override { return Backend::METAL; }
        Dtype       dtype()       const noexcept override { return dtype_; }
        std::size_t rows()        const noexcept override { return rows_; }
        id<MTLBuffer> buffer()    const noexcept { return buf_; }

        // Build-side sorted-key cache for the fused/row joins (see
        // gpu_backend.hpp: backend-private, dies with the column, exempt from
        // the host pool cap). perm holds the sort permutation as ORIGINAL
        // upload indices (i64), aligned with the sorted keys — the row join
        // uses it to emit original build indices. Mutable: built lazily on
        // first join use of a const handle.
        id<MTLBuffer> sorted_cache() const noexcept { return sorted_; }
        id<MTLBuffer> perm_cache()   const noexcept { return perm_; }
        void set_sorted_cache(id<MTLBuffer> keys, id<MTLBuffer> perm) const noexcept {
            sorted_ = keys; perm_ = perm;
        }
    private:
        id<MTLBuffer> buf_;
        std::size_t   rows_;
        Dtype         dtype_;
        mutable id<MTLBuffer> sorted_ = nil;
        mutable id<MTLBuffer> perm_   = nil;
    };

    enum class GbMode { SumI64, SumF64, Count };

    // Radix-sort an F64 column by value through an order-preserving i64
    // key transform (NaN canonicalised and sorted greatest, as native
    // DuckDB orders doubles). Cached on the column like the i64 sort.
    void ensure_sorted_cache_f64(const MetalResidentColumn& c, double* sort_kernel_ms) {
        if (c.sorted_cache()) return;
        if (!sorter_)
            sorter_ = std::make_unique<metal_detail::MetalRadixSort>(device_, queue_);
        const std::size_t n = c.rows();
        const auto* d = static_cast<const double*>([c.buffer() contents]);
        std::vector<std::int64_t> tk(n), idx(n);
        for (std::size_t i = 0; i < n; ++i) {
            double x = d[i];
            std::uint64_t u;
            if (std::isnan(x)) u = 0x7FF8000000000000ull;
            else std::memcpy(&u, &x, sizeof(u));
            std::int64_t key;
            if (static_cast<std::int64_t>(u) < 0)
                key = -static_cast<std::int64_t>(u & 0x7FFFFFFFFFFFFFFFull) - 1;
            else
                key = static_cast<std::int64_t>(u);
            tk[i]  = key;
            idx[i] = static_cast<std::int64_t>(i);
        }
        auto view = sorter_->sort_device(tk.data(), idx.data(), static_cast<std::uint32_t>(n));
        id<MTLBuffer> sorted = [device_ newBufferWithLength:n * sizeof(std::int64_t)
                                                    options:MTLResourceStorageModeShared];
        id<MTLBuffer> perm   = [device_ newBufferWithLength:n * sizeof(std::int64_t)
                                                    options:MTLResourceStorageModeShared];
        if (!sorted || !perm)
            throw std::runtime_error("topk_resident: device allocation for sort cache failed");
        std::memcpy([sorted contents], [view.keys contents],     n * sizeof(std::int64_t));
        std::memcpy([perm contents],   [view.payloads contents], n * sizeof(std::int64_t));
        c.set_sorted_cache(sorted, perm);
        *sort_kernel_ms += view.kernel_ms;
    }

    // Output buffer aliasing a std::vector's storage when it is page-aligned
    // (large vectors are: macOS hands out whole pages) so the GPU writes the
    // result in place; otherwise a shared scratch buffer copied back after.
    struct OutBuf { id<MTLBuffer> buf = nil; bool aliased = false; };
    OutBuf out_for(std::vector<std::int64_t>& vec, std::size_t n) {
        constexpr std::size_t kPage = 16384;
        const std::size_t bytes  = std::max<std::size_t>(1, n * sizeof(std::int64_t));
        const std::size_t padded = ((bytes + kPage - 1) / kPage) * kPage;
        vec.clear();
        vec.reserve(padded / sizeof(std::int64_t));
        vec.resize(n);
        OutBuf o;
        const auto addr = reinterpret_cast<std::uintptr_t>(vec.data());
        if (n > 0 && (addr % kPage) == 0) {
            o.buf = [device_ newBufferWithBytesNoCopy:vec.data() length:padded
                                              options:MTLResourceStorageModeShared
                                          deallocator:nil];
            o.aliased = (o.buf != nil);
        }
        if (!o.aliased)
            o.buf = [device_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!o.buf) throw std::runtime_error("resident group by: output allocation failed");
        return o;
    }
    static void copy_back(const OutBuf& o, std::vector<std::int64_t>& vec) {
        if (!o.aliased && !vec.empty())
            std::memcpy(vec.data(), [o.buf contents], vec.size() * sizeof(std::int64_t));
    }
    id<MTLBuffer> grow(id<MTLBuffer>& b, std::size_t bytes, const char* what) {
        if (!b || [b length] < bytes) {
            b = [device_ newBufferWithLength:std::max<std::size_t>(bytes, 16)
                                     options:MTLResourceStorageModeShared];
            if (!b) throw std::runtime_error(std::string("resident group by: ") + what +
                                             " allocation failed");
        }
        return b;
    }

    GroupByResidentResult groupby_impl(const MetalResidentColumn& k,
                                       const MetalResidentColumn* v,
                                       GbMode mode, std::size_t max_groups,
                                       const char* op) {
        const auto t_wall0 = std::chrono::steady_clock::now();
        GroupByResidentResult r{};
        r.rows_in = k.rows();
        const std::size_t n = k.rows();
        if (n == 0) {
            r.wall_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t_wall0).count();
            return r;
        }
        if (n > 0xFFFFFFFFull)
            throw std::runtime_error(std::string(op) + ": > 2^32 rows unsupported");

        double kernel_ms = 0.0;
        id<MTLBuffer> sorted = ensure_sorted_cache(k, &kernel_ms);
        id<MTLBuffer> perm   = k.perm_cache();
        const std::uint32_t n32 = static_cast<std::uint32_t>(n);
        const std::size_t nblocks = (n + kBlock - 1) / kBlock;
        const std::size_t nchunks = (n + 63) / 64;

        // ---- Stage A: run-start counts per block → host scan → starts ----
        grow(gb_block_buf_, nblocks * sizeof(std::uint32_t), "block counts");
        grow(mult_buf_, n * sizeof(std::uint32_t), "run starts");
        {
            id<MTLCommandBuffer>         cb = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
            [ce setComputePipelineState:ps_gb_block_counts_];
            [ce setBuffer:sorted        offset:0 atIndex:0];
            [ce setBytes:&n32 length:sizeof(n32) atIndex:1];
            [ce setBuffer:gb_block_buf_ offset:0 atIndex:2];
            [ce dispatchThreadgroups:MTLSizeMake(nblocks, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];
            [ce endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            kernel_ms += cb_kernel_ms(cb);
        }
        std::size_t num_segs = 0;
        {
            auto* bc = static_cast<std::uint32_t*>([gb_block_buf_ contents]);
            for (std::size_t b = 0; b < nblocks; ++b) {   // in-place exclusive scan
                const std::uint32_t c = bc[b];
                bc[b] = static_cast<std::uint32_t>(num_segs);
                num_segs += c;
            }
        }
        if (num_segs > max_groups)
            throw std::runtime_error(
                std::string(op) + ": result has " + std::to_string(num_segs) +
                " groups, above the cap of " + std::to_string(max_groups) +
                " (raise GPUDB_GROUPBY_ROWS_MAX_M if intentional)");
        const std::uint32_t ns32 = static_cast<std::uint32_t>(num_segs);

        // ---- Stage B: starts, chunk sums, finalize (one command buffer) ----
        const bool gpu_sums = (mode == GbMode::SumI64);
        OutBuf ok = out_for(r.keys, num_segs);
        OutBuf oc = out_for(r.counts, num_segs);
        OutBuf os;
        if (gpu_sums) {
            os = out_for(r.sums, num_segs);
            grow(gb_head_buf_, nchunks * sizeof(std::int64_t), "head partials");
            grow(gb_tail_buf_, nchunks * sizeof(std::int64_t), "tail partials");
        }
        if (mode == GbMode::SumF64)
            grow(gb_gather_buf_, n * sizeof(std::int64_t), "f64 gather");
        {
            id<MTLCommandBuffer>         cb = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
            [ce setComputePipelineState:ps_gb_run_starts_];
            [ce setBuffer:sorted        offset:0 atIndex:0];
            [ce setBytes:&n32 length:sizeof(n32) atIndex:1];
            [ce setBuffer:gb_block_buf_ offset:0 atIndex:2];
            [ce setBuffer:mult_buf_     offset:0 atIndex:3];
            [ce dispatchThreadgroups:MTLSizeMake(nblocks, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];
            [ce memoryBarrierWithScope:MTLBarrierScopeBuffers];

            if (gpu_sums) {
                [ce setComputePipelineState:ps_gb_chunk_sum_];
                [ce setBuffer:sorted        offset:0 atIndex:0];
                [ce setBuffer:perm          offset:0 atIndex:1];
                [ce setBuffer:v->buffer()   offset:0 atIndex:2];
                [ce setBuffer:mult_buf_     offset:0 atIndex:3];
                [ce setBytes:&n32  length:sizeof(n32)  atIndex:4];
                [ce setBytes:&ns32 length:sizeof(ns32) atIndex:5];
                [ce setBuffer:os.buf        offset:0 atIndex:6];
                [ce setBuffer:gb_head_buf_  offset:0 atIndex:7];
                [ce setBuffer:gb_tail_buf_  offset:0 atIndex:8];
                [ce dispatchThreadgroups:MTLSizeMake((nchunks + kBlock - 1) / kBlock, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];
                [ce memoryBarrierWithScope:MTLBarrierScopeBuffers];
            } else if (mode == GbMode::SumF64) {
                [ce setComputePipelineState:ps_gb_gather_];
                [ce setBuffer:perm           offset:0 atIndex:0];
                [ce setBuffer:v->buffer()    offset:0 atIndex:1];
                [ce setBuffer:gb_gather_buf_ offset:0 atIndex:2];
                [ce setBytes:&n32 length:sizeof(n32) atIndex:3];
                [ce dispatchThreadgroups:MTLSizeMake(nblocks, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];
            }

            const std::uint32_t with_sums = gpu_sums ? 1u : 0u;
            [ce setComputePipelineState:ps_gb_finalize_];
            [ce setBuffer:sorted    offset:0 atIndex:0];
            [ce setBuffer:mult_buf_ offset:0 atIndex:1];
            [ce setBytes:&n32  length:sizeof(n32)  atIndex:2];
            [ce setBytes:&ns32 length:sizeof(ns32) atIndex:3];
            [ce setBuffer:(gpu_sums ? gb_head_buf_ : oc.buf) offset:0 atIndex:4];
            [ce setBuffer:(gpu_sums ? gb_tail_buf_ : oc.buf) offset:0 atIndex:5];
            [ce setBuffer:ok.buf    offset:0 atIndex:6];
            [ce setBuffer:(gpu_sums ? os.buf : oc.buf) offset:0 atIndex:7];
            [ce setBuffer:oc.buf    offset:0 atIndex:8];
            [ce setBytes:&with_sums length:sizeof(with_sums) atIndex:9];
            [ce dispatchThreadgroups:MTLSizeMake((num_segs + kBlock - 1) / kBlock, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];
            [ce endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            kernel_ms += cb_kernel_ms(cb);
        }
        copy_back(ok, r.keys);
        copy_back(oc, r.counts);
        if (gpu_sums) copy_back(os, r.sums);

        // ---- f64: host streams the gathered payload per segment ----
        if (mode == GbMode::SumF64) {
            const auto* g      = static_cast<const double*>([gb_gather_buf_ contents]);
            const auto* starts = static_cast<const std::uint32_t*>([mult_buf_ contents]);
            r.sums_f64.assign(num_segs, 0.0);
            const unsigned hw = std::thread::hardware_concurrency();
            const std::size_t workers =
                std::max<std::size_t>(1, std::min<std::size_t>(hw ? hw : 1,
                                                               std::min(num_segs, n / 65536 + 1)));
            const std::size_t per = (num_segs + workers - 1) / workers;
            std::vector<std::thread> ts;
            for (std::size_t w = 0; w < workers; ++w) {
                const std::size_t s0 = w * per, s1 = std::min(num_segs, s0 + per);
                if (s0 >= s1) break;
                ts.emplace_back([&, s0, s1] {
                    for (std::size_t s = s0; s < s1; ++s) {
                        const std::size_t rs = starts[s];
                        const std::size_t re = (s + 1 < num_segs) ? starts[s + 1] : n;
                        double acc = 0.0;
                        for (std::size_t i = rs; i < re; ++i) acc += g[i];
                        r.sums_f64[s] = acc;
                    }
                });
            }
            for (auto& t : ts) t.join();
        }

        r.kernel_ms   = kernel_ms;
        r.transfer_ms = 0.0;
        r.wall_ms     = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t_wall0).count();
        return r;
    }


    std::unique_ptr<ResidentColumn>
    make_resident(const void* src, std::size_t n, Dtype dt, std::size_t elem) {
        @autoreleasepool {
            const std::size_t bytes = (n == 0) ? 1 : n * elem;
            id<MTLBuffer> buf = [device_ newBufferWithLength:bytes
                                                     options:MTLResourceStorageModeShared];
            if (n > 0) std::memcpy([buf contents], src, n * elem);
            return std::make_unique<MetalResidentColumn>(buf, n, dt);
        }
    }

    static const MetalResidentColumn& check_i64(const ResidentColumn& c) {
        if (c.backend_tag() != Backend::METAL || c.dtype() != Dtype::I64)
            throw std::runtime_error("ResidentColumn mismatch (Metal/i64)");
        return static_cast<const MetalResidentColumn&>(c);
    }
    static const MetalResidentColumn& check_f64(const ResidentColumn& c) {
        if (c.backend_tag() != Backend::METAL || c.dtype() != Dtype::F64)
            throw std::runtime_error("ResidentColumn mismatch (Metal/f64)");
        return static_cast<const MetalResidentColumn&>(c);
    }

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

    AggResult host_sum_f64(const double* data, std::size_t n) {
        const auto t0 = std::chrono::steady_clock::now();
        double acc = 0.0;
        for (std::size_t i = 0; i < n; ++i) acc += data[i];
        AggResult r{};
        r.value_f64 = acc;
        r.rows      = n;
        r.wall_ms   = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0).count();
        return r;
    }

    AggResult run_i64(const std::int64_t* data, std::size_t n,
                      id<MTLComputePipelineState> ps_main,
                      id<MTLComputePipelineState> ps_partials,
                      bool has_init, std::int64_t init) {
        @autoreleasepool {
            const auto t_wall0 = std::chrono::steady_clock::now();
            AggResult r{}; r.rows = n;
            if (n == 0) {
                r.value_i64 = has_init ? init : 0;
                r.wall_ms   = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - t_wall0).count();
                return r;
            }
            id<MTLBuffer> in = stage_input(data, n * sizeof(std::int64_t));
            return dispatch_reduce_i64(in, n, ps_main, ps_partials, has_init, init, t_wall0);
        }
    }

    AggResult run_i64_resident(id<MTLBuffer> in, std::size_t n,
                               id<MTLComputePipelineState> ps_main,
                               id<MTLComputePipelineState> ps_partials,
                               bool has_init, std::int64_t init) {
        @autoreleasepool {
            const auto t_wall0 = std::chrono::steady_clock::now();
            AggResult r{}; r.rows = n;
            if (n == 0) {
                r.value_i64 = has_init ? init : 0;
                r.wall_ms   = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - t_wall0).count();
                return r;
            }
            return dispatch_reduce_i64(in, n, ps_main, ps_partials, has_init, init, t_wall0);
        }
    }

    AggResult dispatch_reduce_i64(id<MTLBuffer> in, std::size_t n,
                                  id<MTLComputePipelineState> ps_main,
                                  id<MTLComputePipelineState> ps_partials,
                                  bool has_init, std::int64_t init,
                                  std::chrono::steady_clock::time_point t_wall0) {
        const NSUInteger grid = pick_grid(n);
        const std::uint32_t n32 = static_cast<std::uint32_t>(n);

        id<MTLCommandBuffer>        cb  = [queue_ commandBuffer];
        id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];

        // ---- Pass 1: per-threadgroup reduction ----
        [ce setComputePipelineState:ps_main];
        [ce setBuffer:in            offset:0 atIndex:0];
        [ce setBuffer:partials_buf_ offset:0 atIndex:1];
        [ce setBytes:&n32 length:sizeof(n32) atIndex:2];
        if (has_init) {
            [ce setBytes:&init length:sizeof(init) atIndex:3];
        }
        [ce dispatchThreadgroups:MTLSizeMake(grid, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];

        // ---- Pass 2: single-threadgroup reduction over partials ----
        const std::uint32_t np = static_cast<std::uint32_t>(grid);
        [ce setComputePipelineState:ps_partials];
        [ce setBuffer:partials_buf_ offset:0 atIndex:0];
        [ce setBuffer:out_buf_      offset:0 atIndex:1];
        [ce setBytes:&np length:sizeof(np) atIndex:2];
        [ce dispatchThreadgroups:MTLSizeMake(1, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];

        [ce endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        AggResult r{};
        r.rows        = n;
        r.value_i64   = *static_cast<const std::int64_t*>([out_buf_ contents]);
        r.kernel_ms   = cb_kernel_ms(cb);
        r.transfer_ms = 0.0;  // UMA
        r.wall_ms     = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t_wall0).count();
        return r;
    }

    // Stage host data into a Metal-readable buffer.
    //
    // FAST PATH (zero-copy): if `src` is page-aligned (16 KiB on Apple Silicon)
    // AND the size is page-multiple, wrap it directly via newBufferWithBytesNoCopy.
    // The GPU reads the same physical pages the CPU wrote — no memcpy. This is
    // the COLD-vs-HOT eraser: at 1B int64 (8 GiB) the memcpy alone costs
    // ~150 ms; eliminating it is the difference between "Metal loses 2× cold"
    // and "Metal wins 5× cold".
    //
    // SLOW PATH: caller-provided pointer isn't aligned. Fall back to allocating
    // a shared MTLBuffer and memcpy'ing.
    id<MTLBuffer> stage_input(const void* src, std::size_t bytes) {
        constexpr std::uintptr_t kPageSize = 16384;  // Apple Silicon
        const std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(src);
        if ((addr % kPageSize) == 0 && bytes >= kPageSize) {
            const std::size_t bytes_padded =
                ((bytes + kPageSize - 1) / kPageSize) * kPageSize;
            // Cache the no-copy buffer if the caller hands us the same pointer
            // again — newBufferWithBytesNoCopy itself takes a few ms at large
            // sizes (presumably because Metal registers the pages with the GPU
            // MMU), so reusing the wrapper across repeated calls eliminates
            // that per-call cost.
            if (zerocopy_src_ == src && zerocopy_bytes_ == bytes_padded
                && zerocopy_buf_ != nil) {
                return zerocopy_buf_;
            }
            zerocopy_buf_ =
                [device_ newBufferWithBytesNoCopy:const_cast<void*>(src)
                                           length:bytes_padded
                                          options:MTLResourceStorageModeShared
                                      deallocator:nil];
            zerocopy_src_   = src;
            zerocopy_bytes_ = bytes_padded;
            return zerocopy_buf_;
        }
        // Slow path: cached shared buffer + memcpy.
        if (!input_buf_ || [input_buf_ length] < bytes) {
            input_buf_ = [device_ newBufferWithLength:bytes
                                              options:MTLResourceStorageModeShared];
        }
        std::memcpy([input_buf_ contents], src, bytes);
        return input_buf_;
    }

    AggAllResult empty_agg_all(std::size_t n,
                               std::chrono::steady_clock::time_point t_wall0) {
        AggAllResult r{};
        r.rows  = n;
        r.count = 0;
        r.sum   = 0;
        r.min   = std::numeric_limits<std::int64_t>::max();
        r.max   = std::numeric_limits<std::int64_t>::min();
        r.wall_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t_wall0).count();
        return r;
    }

    AggAllResult dispatch_agg_all_i64(id<MTLBuffer> in, std::size_t n,
                                      std::chrono::steady_clock::time_point t_wall0) {
        const NSUInteger grid = pick_grid(n);
        const std::uint32_t n32 = static_cast<std::uint32_t>(n);

        id<MTLCommandBuffer>         cb = [queue_ commandBuffer];
        id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];

        // ---- Pass 1: per-threadgroup reduction (4 partials per block) ----
        [ce setComputePipelineState:ps_agg_all_i64_];
        [ce setBuffer:in                 offset:0 atIndex:0];
        [ce setBuffer:partials_quad_buf_ offset:0 atIndex:1];
        [ce setBytes:&n32 length:sizeof(n32) atIndex:2];
        [ce dispatchThreadgroups:MTLSizeMake(grid, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];

        // ---- Pass 2: single-threadgroup reduction over partials ----
        const std::uint32_t np = static_cast<std::uint32_t>(grid);
        [ce setComputePipelineState:ps_agg_all_partials_i64_];
        [ce setBuffer:partials_quad_buf_ offset:0 atIndex:0];
        [ce setBuffer:out_quad_buf_      offset:0 atIndex:1];
        [ce setBytes:&np length:sizeof(np) atIndex:2];
        [ce dispatchThreadgroups:MTLSizeMake(1, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];

        [ce endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        const std::int64_t* out = static_cast<const std::int64_t*>([out_quad_buf_ contents]);
        AggAllResult r{};
        r.rows        = n;
        r.sum         = out[0];
        r.min         = out[1];
        r.max         = out[2];
        r.count       = static_cast<std::size_t>(out[3]);
        r.kernel_ms   = cb_kernel_ms(cb);
        r.transfer_ms = 0.0;  // UMA
        r.wall_ms     = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t_wall0).count();
        return r;
    }

    id<MTLDevice>       device_ = nil;
    id<MTLCommandQueue> queue_  = nil;

    id<MTLComputePipelineState> ps_sum_i64_          = nil;
    id<MTLComputePipelineState> ps_sum_partials_i64_ = nil;
    id<MTLComputePipelineState> ps_min_i64_          = nil;
    id<MTLComputePipelineState> ps_min_partials_i64_ = nil;
    id<MTLComputePipelineState> ps_max_i64_          = nil;
    id<MTLComputePipelineState> ps_max_partials_i64_ = nil;
    id<MTLComputePipelineState> ps_agg_all_i64_          = nil;
    id<MTLComputePipelineState> ps_agg_all_partials_i64_ = nil;
    id<MTLComputePipelineState> ps_join_sum_i64_          = nil;
    id<MTLComputePipelineState> ps_join_sum_partials_i64_ = nil;
    id<MTLComputePipelineState> ps_join_mult_i64_         = nil;
    id<MTLComputePipelineState> ps_join_lookup_i64_       = nil;
    id<MTLComputePipelineState> ps_gb_block_counts_       = nil;
    id<MTLComputePipelineState> ps_gb_run_starts_         = nil;
    id<MTLComputePipelineState> ps_gb_chunk_sum_          = nil;
    id<MTLComputePipelineState> ps_gb_finalize_           = nil;
    id<MTLComputePipelineState> ps_gb_gather_             = nil;

    // Resident GROUP BY scratch (grown on demand): per-256-block run-start
    // counts / offsets (u32), per-64-chunk head and tail partials (i64), and
    // the gathered f64 payload in sorted order.
    id<MTLBuffer> gb_block_buf_  = nil;
    id<MTLBuffer> gb_head_buf_   = nil;
    id<MTLBuffer> gb_tail_buf_   = nil;
    id<MTLBuffer> gb_gather_buf_ = nil;

    // Lazily constructed: only joins pay for the radix-sort pipelines.
    std::unique_ptr<metal_detail::MetalRadixSort> sorter_;

    // Per-probe-element multiplicity scratch for the f64 join path (u32 per
    // row); grown on demand, reused across calls.
    id<MTLBuffer> mult_buf_ = nil;
    // First-match-position scratch for the row-returning join (u32 per row).
    id<MTLBuffer> first_buf_ = nil;

    id<MTLBuffer> input_buf_         = nil;  // grows on demand (slow-path memcpy)
    id<MTLBuffer> partials_buf_      = nil;  // sized for kMaxGrid * sizeof(int64)
    id<MTLBuffer> out_buf_           = nil;  // single int64
    id<MTLBuffer> partials_quad_buf_ = nil;  // 4 * kMaxGrid * sizeof(int64) for agg_all
    id<MTLBuffer> out_quad_buf_      = nil;  // 4 longs (sum/min/max/count)
    // Zero-copy cache (keyed on caller pointer + padded length).
    id<MTLBuffer> zerocopy_buf_   = nil;
    const void*   zerocopy_src_   = nullptr;
    std::size_t   zerocopy_bytes_ = 0;
};

} // namespace

bool metal_runtime_available() noexcept {
    @autoreleasepool {
        id<MTLDevice> d = MTLCreateSystemDefaultDevice();
        return d != nil;
    }
}

std::unique_ptr<Aggregator> make_metal_aggregator() {
    return std::make_unique<MetalAggregator>();
}

} // namespace gpudb
