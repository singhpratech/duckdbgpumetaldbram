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
#include "../groupby_filter.hpp"
#include "metal_kernel_sources.hpp"
#include "metal_radix_sort.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
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

// Shared by the aggregator and every column it uploads: the radix sorter's
// staging buffers are single-use, so one sort runs at a time process-wide
// (mu), and a column that outlives the aggregator can still build its cache.
struct SortCtx {
    id<MTLDevice>       device = nil;
    id<MTLCommandQueue> queue  = nil;
    std::mutex          mu;
    std::unique_ptr<metal_detail::MetalRadixSort> sorter;   // created under mu
    metal_detail::MetalRadixSort& get() {
        if (!sorter) sorter = std::make_unique<metal_detail::MetalRadixSort>(device, queue);
        return *sorter;
    }
};

class MetalAggregator final : public Aggregator {
public:
    MetalAggregator() {
        @autoreleasepool {
            device_ = MTLCreateSystemDefaultDevice();
            if (!device_) throw std::runtime_error("MTLCreateSystemDefaultDevice returned nil");
            queue_ = [device_ newCommandQueue];
            if (!queue_) throw std::runtime_error("Failed to create Metal command queue");
            sort_ctx_ = std::make_shared<SortCtx>();
            sort_ctx_->device = device_;
            sort_ctx_->queue  = queue_;

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
            ps_gb_having_counts_      = make_pso(lib, @"gb_having_counts_i64");
            ps_gb_having_compact_     = make_pso(lib, @"gb_having_compact_i64");
            ps_gb_topk_hist_          = make_pso(lib, @"gb_topk_hist_i64");
            ps_gb_topk_counts_        = make_pso(lib, @"gb_topk_counts_i64");
            ps_gb_topk_compact_       = make_pso(lib, @"gb_topk_compact_i64");
            ps_gbx_chunk_             = make_pso(lib, @"gbx_chunk_i64");
            ps_gbx_finalize_          = make_pso(lib, @"gbx_finalize_i64");
            ps_gbx_having_counts_     = make_pso(lib, @"gbx_having_counts_i64");
            ps_gbx_having_compact_    = make_pso(lib, @"gbx_having_compact_i64");
            ps_gbx_topk_hist_         = make_pso(lib, @"gbx_topk_hist_i64");
            ps_gbx_topk_counts_       = make_pso(lib, @"gbx_topk_counts_i64");
            ps_gbx_topk_compact_      = make_pso(lib, @"gbx_topk_compact_i64");
            ps_gbx_mask_              = make_pso(lib, @"gbx_mask_i64");
            ps_gbxm_chunk_            = make_pso(lib, @"gbxm_chunk_i64");
            ps_gbxm_finalize_         = make_pso(lib, @"gbxm_finalize_i64");
            ps_gbx_sel_counts_        = make_pso(lib, @"gbx_sel_counts_i64");
            ps_gbx_sel_compact_       = make_pso(lib, @"gbx_sel_compact_i64");
            ps_jm_unique_             = make_pso(lib, @"jm_unique_i64");
            ps_jm_probe_              = make_pso(lib, @"jm_probe_i64");
            ps_jm_counts_             = make_pso(lib, @"jm_counts");
            ps_jm_pos_                = make_pso(lib, @"jm_pos");
            ps_jm_gather_             = make_pso(lib, @"jm_gather");

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
    // v0.7 milestone 0b: de-interleave the (key, payload) segments straight
    // into the two shared buffers in one pass — no intermediate host vectors.
    // The payload lane is copied bit-for-bit: for F64 it already holds the
    // IEEE-754 image, and a shared MTLBuffer is plain host memory.
    ResidentPair upload_pair_interleaved(const KvSpan* spans, std::size_t n_spans,
                                         Dtype vdt) override {
        @autoreleasepool {
            std::size_t rows = 0;
            for (std::size_t i = 0; i < n_spans; ++i) rows += spans[i].rows;
            const std::size_t bytes = (rows == 0) ? 1 : rows * sizeof(std::int64_t);
            id<MTLBuffer> kb = [device_ newBufferWithLength:bytes
                                                    options:MTLResourceStorageModeShared];
            id<MTLBuffer> vb = [device_ newBufferWithLength:bytes
                                                    options:MTLResourceStorageModeShared];
            if (!kb || !vb)
                throw std::runtime_error("upload_pair_interleaved: device allocation failed (Metal)");
            auto* k = static_cast<std::int64_t*>([kb contents]);
            auto* v = static_cast<std::int64_t*>([vb contents]);
            std::size_t r = 0;
            for (std::size_t i = 0; i < n_spans; ++i) {
                const std::int64_t* kv = spans[i].kv;
                for (std::size_t j = 0; j < spans[i].rows; ++j, ++r) {
                    k[r] = kv[2 * j];
                    v[r] = kv[2 * j + 1];
                }
            }
            ResidentPair out;
            out.keys = std::make_unique<MetalResidentColumn>(kb, rows, Dtype::I64, sort_ctx_);
            out.vals = std::make_unique<MetalResidentColumn>(vb, rows, vdt, sort_ctx_);
            return out;
        }
    }

    // v0.7 milestone 3 (§4.1): the pair WITH its NULLs. One host pass over
    // the interleaved segments (UMA: the shared buffers are the device
    // memory) partitions NULL-key rows to a suffix of both columns, keeps
    // NULL payloads in place under a validity bitmap, and de-interleaves.
    bool exact_supported() const noexcept override { return true; }

    // ---- v0.7 §4.8: the materialised key join ----
    bool join_supported() const noexcept override { return true; }

    JoinMaterializeResult join_materialize(const ResidentColumn& probe_key,
                                           const ResidentColumn& build_key,
                                           const JoinLane* out, std::size_t n_out) override {
        @autoreleasepool {
            const auto t_wall0 = std::chrono::steady_clock::now();
            const auto& pk = check_i64_nullable(probe_key);
            const auto& bk = check_i64_nullable(build_key);
            if (n_out == 0 || !out) throw std::runtime_error("join_materialize: no output lanes");
            for (std::size_t l = 0; l < n_out; ++l) {
                if (!out[l].col) throw std::runtime_error("join_materialize: output lane without a column");
                if (out[l].col->backend_tag() != Backend::METAL)
                    throw std::runtime_error("ResidentColumn mismatch (Metal join lane)");
                if (out[l].col->rows() != (out[l].from_build ? bk.rows() : pk.rows()))
                    throw std::runtime_error("join_materialize: lane " + std::to_string(l) +
                                             " row count differs from its side of the join");
            }
            if (out[0].col->dtype() != Dtype::I64)
                throw std::runtime_error("join_materialize: the key lane must be I64");
            if (pk.rows() > 0xFFFFFFFFull - 64 || bk.rows() > 0xFFFFFFFFull - 64)
                throw std::runtime_error("join_materialize: > 2^32-64 rows unsupported");

            JoinMaterializeResult r;
            r.rows_probe = pk.rows();
            r.rows_build = bk.rows();
            double kernel_ms = 0.0;
            if (!gbx_dummy_valid_)
                gbx_dummy_valid_ = [device_ newBufferWithLength:8 options:MTLResourceStorageModeShared];

            // ---- build side: sorted valid keys + permutation to build rows ----
            id<MTLBuffer> sorted = nil, perm = nil;
            std::size_t nb = 0;
            if (bk.valid_buffer() == nil) {
                // A key-layout column: the sort cache covers exactly the valid prefix.
                nb = bk.sort_rows();
                if (nb) { bk.build_sort_cache(&kernel_ms); sorted = bk.sorted_cache(); perm = bk.perm_cache(); }
            } else {
                // NULLs under a bitmap: sort the valid cells only.
                const auto* bd = static_cast<const std::int64_t*>([bk.buffer() contents]);
                const auto* bv = static_cast<const std::uint64_t*>([bk.valid_buffer() contents]);
                std::vector<std::int64_t> ks, idx;
                ks.reserve(bk.rows()); idx.reserve(bk.rows());
                for (std::size_t i = 0; i < bk.rows(); ++i)
                    if ((bv[i >> 6] >> (i & 63)) & 1u) { ks.push_back(bd[i]); idx.push_back(static_cast<std::int64_t>(i)); }
                nb = ks.size();
                if (nb) {
                    sorted = [device_ newBufferWithLength:nb * sizeof(std::int64_t) options:MTLResourceStorageModeShared];
                    perm   = [device_ newBufferWithLength:nb * sizeof(std::int64_t) options:MTLResourceStorageModeShared];
                    if (!sorted || !perm) throw std::runtime_error("join_materialize: device allocation failed (Metal)");
                    std::lock_guard<std::mutex> slock(sort_ctx_->mu);
                    auto view = sort_ctx_->get().sort_device(ks.data(), idx.data(), static_cast<std::uint32_t>(nb));
                    std::memcpy([sorted contents], [view.keys contents],     nb * sizeof(std::int64_t));
                    std::memcpy([perm contents],   [view.payloads contents], nb * sizeof(std::int64_t));
                    kernel_ms += view.kernel_ms;
                }
            }

            const std::size_t n = pk.rows();
            std::size_t n1 = 0, n2 = 0;
            const std::size_t nblocks = (n + kBlock - 1) / kBlock;
            const std::uint32_t n32  = static_cast<std::uint32_t>(n);
            const std::uint32_t nb32 = static_cast<std::uint32_t>(nb);
            auto null_from = [](const MetalResidentColumn& c) {
                return c.null_suffix() ? static_cast<std::uint32_t>(c.sort_rows()) : 0xFFFFFFFFu;
            };
            if (n > 0 && nb > 0) {
                grow(jm_flag_, sizeof(std::uint32_t), "join flag");
                grow(jm_match_, n * sizeof(std::uint32_t), "join match");
                grow(jm_cls_, n, "join class");
                grow(gb_block_buf_,  nblocks * sizeof(std::uint32_t), "join block counts");
                grow(gb_block2_buf_, nblocks * sizeof(std::uint32_t), "join block counts");
                *static_cast<std::uint32_t*>([jm_flag_ contents]) = 0u;
                const auto& kc = static_cast<const MetalResidentColumn&>(*out[0].col);
                const std::uint32_t p_has = pk.valid_buffer() != nil ? 1u : 0u;
                const std::uint32_t p_from = null_from(pk);
                const std::uint32_t k_has = kc.valid_buffer() != nil ? 1u : 0u;
                const std::uint32_t k_from = null_from(kc);
                const std::uint32_t k_build = out[0].from_build ? 1u : 0u;
                id<MTLCommandBuffer>         cb = [queue_ commandBuffer];
                id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
                if (nb > 1) {
                    [ce setComputePipelineState:ps_jm_unique_];
                    [ce setBuffer:sorted offset:0 atIndex:0];
                    [ce setBytes:&nb32 length:sizeof(nb32) atIndex:1];
                    [ce setBuffer:jm_flag_ offset:0 atIndex:2];
                    [ce dispatchThreadgroups:MTLSizeMake((nb + kBlock - 1) / kBlock, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];
                }
                [ce setComputePipelineState:ps_jm_probe_];
                [ce setBuffer:pk.buffer() offset:0 atIndex:0];
                [ce setBuffer:(p_has ? pk.valid_buffer() : gbx_dummy_valid_) offset:0 atIndex:1];
                [ce setBytes:&p_has  length:sizeof(p_has)  atIndex:2];
                [ce setBytes:&p_from length:sizeof(p_from) atIndex:3];
                [ce setBytes:&n32    length:sizeof(n32)    atIndex:4];
                [ce setBuffer:sorted offset:0 atIndex:5];
                [ce setBuffer:perm   offset:0 atIndex:6];
                [ce setBytes:&nb32   length:sizeof(nb32)   atIndex:7];
                [ce setBuffer:(k_has ? kc.valid_buffer() : gbx_dummy_valid_) offset:0 atIndex:8];
                [ce setBytes:&k_has   length:sizeof(k_has)   atIndex:9];
                [ce setBytes:&k_from  length:sizeof(k_from)  atIndex:10];
                [ce setBytes:&k_build length:sizeof(k_build) atIndex:11];
                [ce setBuffer:jm_match_ offset:0 atIndex:12];
                [ce setBuffer:jm_cls_   offset:0 atIndex:13];
                [ce dispatchThreadgroups:MTLSizeMake(nblocks, 1, 1) threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];
                [ce memoryBarrierWithScope:MTLBarrierScopeBuffers];
                [ce setComputePipelineState:ps_jm_counts_];
                [ce setBuffer:jm_cls_ offset:0 atIndex:0];
                [ce setBytes:&n32 length:sizeof(n32) atIndex:1];
                [ce setBuffer:gb_block_buf_  offset:0 atIndex:2];
                [ce setBuffer:gb_block2_buf_ offset:0 atIndex:3];
                [ce dispatchThreadgroups:MTLSizeMake(nblocks, 1, 1) threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];
                [ce endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                if ([cb status] == MTLCommandBufferStatusError)
                    throw std::runtime_error("join_materialize: probe command buffer failed (Metal)");
                kernel_ms += cb_kernel_ms(cb);
                if (*static_cast<std::uint32_t*>([jm_flag_ contents]) != 0u)
                    throw std::runtime_error("join_materialize: build key not unique");
                n1 = host_scan_u32(static_cast<std::uint32_t*>([gb_block_buf_ contents]),  nblocks);
                n2 = host_scan_u32(static_cast<std::uint32_t*>([gb_block2_buf_ contents]), nblocks);
            } else if (nb > 1) {
                // No probe rows: uniqueness is still the contract.
                const auto* sk = static_cast<const std::int64_t*>([sorted contents]);
                for (std::size_t i = 0; i + 1 < nb; ++i)
                    if (sk[i] == sk[i + 1]) throw std::runtime_error("join_materialize: build key not unique");
            }
            const std::size_t rows_out = n1 + n2;
            const std::size_t words = (rows_out + 63) / 64;

            // ---- outputs: data + an all-valid bitmap per lane, then gather ----
            std::vector<id<MTLBuffer>> data(n_out, nil), vbits(n_out, nil);
            for (std::size_t l = 0; l < n_out; ++l) {
                data[l]  = [device_ newBufferWithLength:std::max<std::size_t>(16, rows_out * sizeof(std::int64_t))
                                                options:MTLResourceStorageModeShared];
                vbits[l] = [device_ newBufferWithLength:std::max<std::size_t>(8, words * sizeof(std::uint64_t))
                                                options:MTLResourceStorageModeShared];
                if (!data[l] || !vbits[l]) throw std::runtime_error("join_materialize: device allocation failed (Metal)");
                std::memset([vbits[l] contents], 0xFF, [vbits[l] length]);
            }
            if (rows_out > 0) {
                grow(jm_pos_buf_, n * sizeof(std::uint32_t), "join positions");
                const std::uint32_t n1_32 = static_cast<std::uint32_t>(n1);
                id<MTLCommandBuffer>         cb = [queue_ commandBuffer];
                id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
                [ce setComputePipelineState:ps_jm_pos_];
                [ce setBuffer:jm_cls_ offset:0 atIndex:0];
                [ce setBytes:&n32 length:sizeof(n32) atIndex:1];
                [ce setBuffer:gb_block_buf_  offset:0 atIndex:2];
                [ce setBuffer:gb_block2_buf_ offset:0 atIndex:3];
                [ce setBytes:&n1_32 length:sizeof(n1_32) atIndex:4];
                [ce setBuffer:jm_pos_buf_ offset:0 atIndex:5];
                [ce dispatchThreadgroups:MTLSizeMake(nblocks, 1, 1) threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];
                [ce memoryBarrierWithScope:MTLBarrierScopeBuffers];
                for (std::size_t l = 0; l < n_out; ++l) {
                    const auto& sc = static_cast<const MetalResidentColumn&>(*out[l].col);
                    const std::uint32_t s_has = sc.valid_buffer() != nil ? 1u : 0u;
                    const std::uint32_t s_from = null_from(sc);
                    const std::uint32_t fb = out[l].from_build ? 1u : 0u;
                    [ce setComputePipelineState:ps_jm_gather_];
                    [ce setBuffer:sc.buffer() offset:0 atIndex:0];
                    [ce setBuffer:(s_has ? sc.valid_buffer() : gbx_dummy_valid_) offset:0 atIndex:1];
                    [ce setBytes:&s_has  length:sizeof(s_has)  atIndex:2];
                    [ce setBytes:&s_from length:sizeof(s_from) atIndex:3];
                    [ce setBytes:&fb     length:sizeof(fb)     atIndex:4];
                    [ce setBuffer:jm_match_   offset:0 atIndex:5];
                    [ce setBuffer:jm_cls_     offset:0 atIndex:6];
                    [ce setBuffer:jm_pos_buf_ offset:0 atIndex:7];
                    [ce setBytes:&n32 length:sizeof(n32) atIndex:8];
                    [ce setBuffer:data[l]  offset:0 atIndex:9];
                    [ce setBuffer:vbits[l] offset:0 atIndex:10];
                    [ce dispatchThreadgroups:MTLSizeMake(nblocks, 1, 1) threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];
                }
                [ce endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                if ([cb status] == MTLCommandBufferStatusError)
                    throw std::runtime_error("join_materialize: gather command buffer failed (Metal)");
                kernel_ms += cb_kernel_ms(cb);
            }
            r.lanes.reserve(n_out);
            for (std::size_t l = 0; l < n_out; ++l) {
                const auto* w = static_cast<const std::uint64_t*>([vbits[l] contents]);
                std::size_t set = 0;
                for (std::size_t i = 0; i < words; ++i) {
                    std::uint64_t x = w[i];
                    if (i + 1 == words && (rows_out & 63)) x &= (std::uint64_t{1} << (rows_out & 63)) - 1;
                    set += static_cast<std::size_t>(__builtin_popcountll(x));
                }
                const std::size_t nulls = rows_out - set;
                // Lane 0: its NULLs are exactly the suffix (key layout, no bitmap).
                const bool key = l == 0;
                r.lanes.push_back(std::make_unique<MetalResidentColumn>(
                    data[l], rows_out, out[l].col->dtype(), sort_ctx_,
                    /*null_suffix*/ key ? n2 : 0, (key || nulls == 0) ? nil : vbits[l], nulls));
            }
            r.rows_out = rows_out;
            r.null_key_rows = n2;
            r.kernel_ms = kernel_ms;
            r.wall_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t_wall0).count();
            return r;
        }
    }

    ResidentPair upload_pair_exact(const KvSpan* spans, std::size_t n_spans,
                                   Dtype vdt) override {
        if (vdt != Dtype::I64)
            throw std::runtime_error(
                "upload_pair_exact: DOUBLE payloads are not on the exact path (docs/TRANSPARENT_DESIGN.md §4.7)");
        @autoreleasepool {
            std::size_t rows = 0, null_keys = 0;
            auto bit = [](const std::uint64_t* m, std::size_t i) {
                return !m || ((m[i >> 6] >> (i & 63)) & 1u);
            };
            for (std::size_t s = 0; s < n_spans; ++s) {
                rows += spans[s].rows;
                if (spans[s].key_valid)
                    for (std::size_t j = 0; j < spans[s].rows; ++j) null_keys += !bit(spans[s].key_valid, j);
            }
            const std::size_t bytes = (rows == 0) ? 1 : rows * sizeof(std::int64_t);
            const std::size_t words = (rows + 63) / 64;
            id<MTLBuffer> kb = [device_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> vb = [device_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            if (!kb || !vb)
                throw std::runtime_error("upload_pair_exact: device allocation failed (Metal)");
            auto* k = static_cast<std::int64_t*>([kb contents]);
            auto* v = static_cast<std::int64_t*>([vb contents]);
            std::vector<std::uint64_t> vvalid(words, ~std::uint64_t{0});
            std::size_t head = 0, tail = rows - null_keys, null_vals = 0;
            for (std::size_t s = 0; s < n_spans; ++s) {
                const KvSpan& sp = spans[s];
                for (std::size_t j = 0; j < sp.rows; ++j) {
                    const bool kv_ok = bit(sp.key_valid, j);
                    const bool vv_ok = bit(sp.val_valid, j);
                    const std::size_t dst = kv_ok ? head++ : tail++;
                    k[dst] = kv_ok ? sp.kv[2 * j] : 0;
                    v[dst] = vv_ok ? sp.kv[2 * j + 1] : 0;
                    if (!vv_ok) { vvalid[dst >> 6] &= ~(std::uint64_t{1} << (dst & 63)); ++null_vals; }
                }
            }
            id<MTLBuffer> valid = nil;
            if (null_vals) {
                valid = [device_ newBufferWithBytes:vvalid.data()
                                            length:std::max<std::size_t>(8, words * sizeof(std::uint64_t))
                                           options:MTLResourceStorageModeShared];
                if (!valid) throw std::runtime_error("upload_pair_exact: validity allocation failed (Metal)");
            }
            ResidentPair out;
            out.keys = std::make_unique<MetalResidentColumn>(kb, rows, Dtype::I64, sort_ctx_,
                                                             null_keys, nil, null_keys);
            out.vals = std::make_unique<MetalResidentColumn>(vb, rows, Dtype::I64, sort_ctx_,
                                                             0, valid, null_vals);
            return out;
        }
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

    // Get (or build + cache) the radix-sorted copy of a build-key column.
    // The build lives on the column (prepare(), TRANSPARENT_DESIGN.md §5.5);
    // this is the lazy path an operator takes when the column was not
    // prepared. Sort cost is reported through *sort_kernel_ms on the call
    // that pays it; later calls reuse the cache for free.
    id<MTLBuffer> ensure_sorted_cache(const ResidentColumn& build_col,
                                      double* sort_kernel_ms) {
        const auto& bk = check_i64(build_col);
        bk.build_sort_cache(sort_kernel_ms);
        return bk.sorted_cache();
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
                                                   std::size_t max_groups,
                                                   const GroupByFilter& filter) override {
        @autoreleasepool {
            const auto& k = check_i64(keys);
            const auto& v = check_i64(vals);
            if (k.rows() != v.rows())
                throw std::runtime_error(
                    "groupby_sum_resident_i64: keys and vals row counts differ");
            return groupby_impl(k, &v, GbMode::SumI64, max_groups, filter, "groupby_sum_resident_i64");
        }
    }

    GroupByResidentResult groupby_sum_resident_f64(const ResidentColumn& keys,
                                                   const ResidentColumn& vals,
                                                   std::size_t max_groups,
                                                   const GroupByFilter& filter) override {
        @autoreleasepool {
            const auto& k = check_i64(keys);
            const auto& v = check_f64(vals);
            if (k.rows() != v.rows())
                throw std::runtime_error(
                    "groupby_sum_resident_f64: keys and vals row counts differ");
            return groupby_impl(k, &v, GbMode::SumF64, max_groups, filter, "groupby_sum_resident_f64");
        }
    }

    GroupByResidentResult groupby_count_resident(const ResidentColumn& keys,
                                                 std::size_t max_groups,
                                                 const GroupByFilter& filter) override {
        @autoreleasepool {
            const auto& k = check_i64(keys);
            return groupby_impl(k, nullptr, GbMode::Count, max_groups, filter, "groupby_count_resident");
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
        MetalResidentColumn(id<MTLBuffer> buf, std::size_t n, Dtype dt,
                            std::shared_ptr<SortCtx> ctx)
            : buf_(buf), rows_(n), dtype_(dt), ctx_(std::move(ctx)) {}
        // v0.7 §4.1 exact-path column. Key column: `null_suffix` trailing
        // rows have a NULL key (the sort cache covers only the prefix).
        // Payload column: `valid` is the DuckDB-layout validity bitmap (nil
        // when NULL-free) and `nulls` the number of NULL payloads.
        MetalResidentColumn(id<MTLBuffer> buf, std::size_t n, Dtype dt,
                            std::shared_ptr<SortCtx> ctx,
                            std::size_t null_suffix, id<MTLBuffer> valid, std::size_t nulls)
            : buf_(buf), rows_(n), dtype_(dt), ctx_(std::move(ctx)),
              null_suffix_(null_suffix), valid_(valid), nulls_(nulls) {}
        Backend     backend_tag() const noexcept override { return Backend::METAL; }
        Dtype       dtype()       const noexcept override { return dtype_; }
        std::size_t rows()        const noexcept override { return rows_; }
        std::size_t null_count()  const noexcept override { return nulls_; }
        id<MTLBuffer> buffer()    const noexcept { return buf_; }
        id<MTLBuffer> valid_buffer() const noexcept { return valid_; }
        std::size_t null_suffix() const noexcept { return null_suffix_; }
        // Rows the sort cache covers: the valid-key prefix.
        std::size_t sort_rows()   const noexcept { return rows_ - null_suffix_; }

        // ---- v0.7 milestone 0b: readiness (gpu_backend.hpp contract) ----
        // The derived structure is the radix-sorted copy of the column plus
        // the sort permutation as ORIGINAL upload indices (i64), which every
        // GROUP BY, join build side and top-k call needs. prepare() builds it
        // now; an operator builds it lazily on first use otherwise.
        // Idempotent; concurrent calls serialize on cache_mu_ and the loser
        // is a no-op; a failure throws std::runtime_error and leaves the
        // column usable (the next caller retries the build).
        void prepare() override { double ms = 0.0; build_sort_cache(&ms); }
        bool prepared() const noexcept override {
            return sort_rows() == 0 || ready_.load(std::memory_order_acquire);
        }
        std::size_t resident_bytes() const noexcept override {
            const std::size_t base = rows_ * sizeof(std::int64_t);   // i64 and f64: 8 B
            const std::size_t bitmap = valid_ ? [valid_ length] : 0;
            return base + bitmap +
                   (ready_.load(std::memory_order_acquire) ? 2 * sort_rows() * sizeof(std::int64_t) : 0);
        }

        // Sorted copy (i64 keys, or the order-preserving i64 image of f64
        // values with NaN canonicalised greatest, as native DuckDB orders
        // doubles) and the permutation; nil until built. Backend-private,
        // dies with the column, exempt from the host pool cap.
        id<MTLBuffer> sorted_cache() const noexcept {
            return ready_.load(std::memory_order_acquire) ? sorted_ : nil;
        }
        id<MTLBuffer> perm_cache() const noexcept {
            return ready_.load(std::memory_order_acquire) ? perm_ : nil;
        }

        void build_sort_cache(double* sort_kernel_ms) const {
            if (sort_rows() == 0 || ready_.load(std::memory_order_acquire)) return;
            std::lock_guard<std::mutex> lock(cache_mu_);
            if (ready_.load(std::memory_order_relaxed)) return;   // lost the race: built
            if (rows_ > 0xFFFFFFFFull)
                throw std::runtime_error("resident sort cache: > 2^32 rows unsupported (Metal)");
            @autoreleasepool {
                const std::size_t n = sort_rows();   // valid-key prefix (== rows_ for legacy columns)
                std::vector<std::int64_t> tk, idx(n);
                for (std::size_t i = 0; i < n; ++i) idx[i] = static_cast<std::int64_t>(i);
                const std::int64_t* keys = nullptr;
                if (dtype_ == Dtype::I64) {
                    keys = static_cast<const std::int64_t*>([buf_ contents]);
                } else {
                    // Order-preserving i64 image of each double: NaN
                    // canonicalised and sorted greatest, negatives reflected.
                    const auto* d = static_cast<const double*>([buf_ contents]);
                    tk.resize(n);
                    for (std::size_t i = 0; i < n; ++i) {
                        double x = d[i];
                        std::uint64_t u;
                        if (std::isnan(x)) u = 0x7FF8000000000000ull;
                        else std::memcpy(&u, &x, sizeof(u));
                        if (static_cast<std::int64_t>(u) < 0)
                            tk[i] = -static_cast<std::int64_t>(u & 0x7FFFFFFFFFFFFFFFull) - 1;
                        else
                            tk[i] = static_cast<std::int64_t>(u);
                    }
                    keys = tk.data();
                }
                id<MTLBuffer> sorted = [ctx_->device newBufferWithLength:n * sizeof(std::int64_t)
                                                                 options:MTLResourceStorageModeShared];
                id<MTLBuffer> perm   = [ctx_->device newBufferWithLength:n * sizeof(std::int64_t)
                                                                 options:MTLResourceStorageModeShared];
                if (!sorted || !perm)
                    throw std::runtime_error("resident sort cache: device allocation failed (Metal)");
                double ms = 0.0;
                {
                    // One sort at a time: the sorter's staging buffers are single-use.
                    std::lock_guard<std::mutex> slock(ctx_->mu);
                    auto view = ctx_->get().sort_device(keys, idx.data(),
                                                        static_cast<std::uint32_t>(n));
                    std::memcpy([sorted contents], [view.keys contents],     n * sizeof(std::int64_t));
                    std::memcpy([perm contents],   [view.payloads contents], n * sizeof(std::int64_t));
                    ms = view.kernel_ms;
                }
                sorted_ = sorted;
                perm_   = perm;
                ready_.store(true, std::memory_order_release);
                *sort_kernel_ms += ms;
            }
        }
    private:
        id<MTLBuffer> buf_;
        std::size_t   rows_;
        Dtype         dtype_;
        std::shared_ptr<SortCtx> ctx_;
        mutable std::mutex        cache_mu_;      // guards the cache build
        mutable std::atomic<bool> ready_{false};  // release after sorted_/perm_ are set
        mutable id<MTLBuffer> sorted_ = nil;
        mutable id<MTLBuffer> perm_   = nil;
        std::size_t   null_suffix_ = 0;           // key column: trailing NULL-key rows
        id<MTLBuffer> valid_ = nil;               // payload column: validity bitmap (nil = none)
        std::size_t   nulls_ = 0;                 // NULL rows (suffix length, or bitmap zeros)
    };

    enum class GbMode { SumI64, SumF64, Count };

    // F64 sort cache (top-k by value): built on the column, see
    // MetalResidentColumn::build_sort_cache.
    void ensure_sorted_cache_f64(const MetalResidentColumn& c, double* sort_kernel_ms) {
        c.build_sort_cache(sort_kernel_ms);
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
    id<MTLBuffer> grow(__strong id<MTLBuffer>& b, std::size_t bytes, const char* what) {
        if (!b || [b length] < bytes) {
            b = [device_ newBufferWithLength:std::max<std::size_t>(bytes, 16)
                                     options:MTLResourceStorageModeShared];
            if (!b) throw std::runtime_error(std::string("resident group by: ") + what +
                                             " allocation failed");
        }
        return b;
    }

    // ---- Stage C: GroupByFilter on the device (i64 sums or counts) ----
    // HAVING: block counts → host scan → compaction into the result vectors.
    // top-k: 8-pass radix select on the aggregate, then compaction of the
    // "strictly better than the k-th" class plus the first need_equal ties.
    static std::uint32_t cmp_code(GroupByFilter::Cmp c) {
        switch (c) {
            case GroupByFilter::Cmp::GT: return 1u;
            case GroupByFilter::Cmp::GE: return 2u;
            case GroupByFilter::Cmp::LT: return 3u;
            case GroupByFilter::Cmp::LE: return 4u;
            default: return 0u;
        }
    }
    static std::size_t host_scan_u32(std::uint32_t* v, std::size_t n) {
        std::size_t tot = 0;
        for (std::size_t b = 0; b < n; ++b) { const std::uint32_t c = v[b]; v[b] = static_cast<std::uint32_t>(tot); tot += c; }
        return tot;
    }
    static void cap_rows(std::size_t rows, std::size_t max_groups, const char* op) {
        if (rows > max_groups)
            throw std::runtime_error(
                std::string(op) + ": result has " + std::to_string(rows) +
                " rows after the filter, above the cap of " + std::to_string(max_groups) +
                " (raise GPUDB_GROUPBY_ROWS_MAX_M if intentional)");
    }

    void device_filter_i64(GroupByResidentResult& r,
                           id<MTLBuffer> bk, id<MTLBuffer> bs, id<MTLBuffer> bc,
                           bool gpu_sums, std::size_t num_segs,
                           const GroupByFilter& f, std::size_t max_groups,
                           const char* op, double& kernel_ms) {
        r.groups_total = num_segs;
        id<MTLBuffer> agg = gpu_sums ? bs : bc;
        const std::uint32_t ns32  = static_cast<std::uint32_t>(num_segs);
        const std::uint32_t cmp   = cmp_code(f.cmp);
        const std::int64_t  thr   = f.threshold_i64;
        const std::uint32_t with_sums = gpu_sums ? 1u : 0u;
        const std::size_t   nb    = (num_segs + kBlock - 1) / kBlock;
        grow(gb_block_buf_,  nb * sizeof(std::uint32_t), "filter block counts");
        grow(gb_block2_buf_, nb * sizeof(std::uint32_t), "filter block counts (ties)");

        auto run = [&](void (^enc)(id<MTLComputeCommandEncoder>)) {
            id<MTLCommandBuffer>         cb = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
            enc(ce);
            [ce endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            kernel_ms += cb_kernel_ms(cb);
        };
        const MTLSize grid = MTLSizeMake(nb, 1, 1), tg = MTLSizeMake(kBlock, 1, 1);

        // ---- HAVING only ----
        if (f.topk == 0) {
            run(^(id<MTLComputeCommandEncoder> ce) {
                [ce setComputePipelineState:ps_gb_having_counts_];
                [ce setBuffer:agg offset:0 atIndex:0];
                [ce setBytes:&ns32 length:sizeof(ns32) atIndex:1];
                [ce setBytes:&cmp  length:sizeof(cmp)  atIndex:2];
                [ce setBytes:&thr  length:sizeof(thr)  atIndex:3];
                [ce setBuffer:gb_block_buf_ offset:0 atIndex:4];
                [ce dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            });
            const std::size_t surv = host_scan_u32(
                static_cast<std::uint32_t*>([gb_block_buf_ contents]), nb);
            cap_rows(surv, max_groups, op);
            OutBuf ok = out_for(r.keys, surv), oc = out_for(r.counts, surv), os;
            if (gpu_sums) os = out_for(r.sums, surv);
            if (surv > 0) {
                run(^(id<MTLComputeCommandEncoder> ce) {
                    [ce setComputePipelineState:ps_gb_having_compact_];
                    [ce setBuffer:agg offset:0 atIndex:0];
                    [ce setBuffer:bk  offset:0 atIndex:1];
                    [ce setBuffer:(gpu_sums ? bs : bc) offset:0 atIndex:2];
                    [ce setBuffer:bc  offset:0 atIndex:3];
                    [ce setBytes:&ns32 length:sizeof(ns32) atIndex:4];
                    [ce setBytes:&cmp  length:sizeof(cmp)  atIndex:5];
                    [ce setBytes:&thr  length:sizeof(thr)  atIndex:6];
                    [ce setBuffer:gb_block_buf_ offset:0 atIndex:7];
                    [ce setBuffer:ok.buf offset:0 atIndex:8];
                    [ce setBuffer:(gpu_sums ? os.buf : oc.buf) offset:0 atIndex:9];
                    [ce setBuffer:oc.buf offset:0 atIndex:10];
                    [ce setBytes:&with_sums length:sizeof(with_sums) atIndex:11];
                    [ce dispatchThreadgroups:grid threadsPerThreadgroup:tg];
                });
            }
            copy_back(ok, r.keys); copy_back(oc, r.counts);
            if (gpu_sums) copy_back(os, r.sums);
            return;
        }

        // ---- top-k: radix select ----
        grow(gb_hist_buf_, 256 * sizeof(std::uint32_t), "radix-select histogram");
        auto* hist = static_cast<std::uint32_t*>([gb_hist_buf_ contents]);
        const std::uint32_t desc = f.topk_desc ? 1u : 0u;
        std::uint64_t prefix = 0, mask = 0;
        std::size_t remaining = f.topk;     // rank still to satisfy within the prefix
        std::size_t candidates = 0;
        bool take_all = false;
        for (int pass = 0; pass < 8; ++pass) {
            const std::uint32_t shift = static_cast<std::uint32_t>(56 - 8 * pass);
            std::memset(hist, 0, 256 * sizeof(std::uint32_t));
            run(^(id<MTLComputeCommandEncoder> ce) {
                [ce setComputePipelineState:ps_gb_topk_hist_];
                [ce setBuffer:agg offset:0 atIndex:0];
                [ce setBytes:&ns32   length:sizeof(ns32)   atIndex:1];
                [ce setBytes:&cmp    length:sizeof(cmp)    atIndex:2];
                [ce setBytes:&thr    length:sizeof(thr)    atIndex:3];
                [ce setBytes:&prefix length:sizeof(prefix) atIndex:4];
                [ce setBytes:&mask   length:sizeof(mask)   atIndex:5];
                [ce setBytes:&shift  length:sizeof(shift)  atIndex:6];
                [ce setBuffer:gb_hist_buf_ offset:0 atIndex:7];
                [ce dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            });
            if (pass == 0) {
                for (int b = 0; b < 256; ++b) candidates += hist[b];
                if (candidates <= f.topk) { take_all = true; break; }
            }
            int chosen = -1;
            if (desc) {
                for (int b = 255; b >= 0; --b) {
                    if (remaining <= hist[b]) { chosen = b; break; }
                    remaining -= hist[b];
                }
            } else {
                for (int b = 0; b < 256; ++b) {
                    if (remaining <= hist[b]) { chosen = b; break; }
                    remaining -= hist[b];
                }
            }
            if (chosen < 0) throw std::runtime_error(std::string(op) + ": radix select lost the k-th rank (internal)");
            prefix |= static_cast<std::uint64_t>(chosen) << shift;
            mask   |= static_cast<std::uint64_t>(0xFF) << shift;
        }

        std::size_t out_rows = 0, n_better = 0, need_equal = 0;
        std::uint64_t T = prefix;
        if (take_all) {
            // every cmp survivor is in the answer: "equal" class empty, all "better"
            T = desc ? 0ull : ~0ull;   // nothing equals it in practice; better-than-T = everything (u > 0 fails for u==0!)
        }
        // Count the two classes with a T such that class membership is exact.
        // For take_all we instead reuse the HAVING counts (cmp only) to be safe.
        if (take_all) {
            run(^(id<MTLComputeCommandEncoder> ce) {
                [ce setComputePipelineState:ps_gb_having_counts_];
                [ce setBuffer:agg offset:0 atIndex:0];
                [ce setBytes:&ns32 length:sizeof(ns32) atIndex:1];
                [ce setBytes:&cmp  length:sizeof(cmp)  atIndex:2];
                [ce setBytes:&thr  length:sizeof(thr)  atIndex:3];
                [ce setBuffer:gb_block_buf_ offset:0 atIndex:4];
                [ce dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            });
            out_rows = host_scan_u32(static_cast<std::uint32_t*>([gb_block_buf_ contents]), nb);
            cap_rows(out_rows, max_groups, op);
            OutBuf ok = out_for(r.keys, out_rows), oc = out_for(r.counts, out_rows), os;
            if (gpu_sums) os = out_for(r.sums, out_rows);
            if (out_rows > 0) {
                run(^(id<MTLComputeCommandEncoder> ce) {
                    [ce setComputePipelineState:ps_gb_having_compact_];
                    [ce setBuffer:agg offset:0 atIndex:0];
                    [ce setBuffer:bk  offset:0 atIndex:1];
                    [ce setBuffer:(gpu_sums ? bs : bc) offset:0 atIndex:2];
                    [ce setBuffer:bc  offset:0 atIndex:3];
                    [ce setBytes:&ns32 length:sizeof(ns32) atIndex:4];
                    [ce setBytes:&cmp  length:sizeof(cmp)  atIndex:5];
                    [ce setBytes:&thr  length:sizeof(thr)  atIndex:6];
                    [ce setBuffer:gb_block_buf_ offset:0 atIndex:7];
                    [ce setBuffer:ok.buf offset:0 atIndex:8];
                    [ce setBuffer:(gpu_sums ? os.buf : oc.buf) offset:0 atIndex:9];
                    [ce setBuffer:oc.buf offset:0 atIndex:10];
                    [ce setBytes:&with_sums length:sizeof(with_sums) atIndex:11];
                    [ce dispatchThreadgroups:grid threadsPerThreadgroup:tg];
                });
            }
            copy_back(ok, r.keys); copy_back(oc, r.counts);
            if (gpu_sums) copy_back(os, r.sums);
        } else {
            run(^(id<MTLComputeCommandEncoder> ce) {
                [ce setComputePipelineState:ps_gb_topk_counts_];
                [ce setBuffer:agg offset:0 atIndex:0];
                [ce setBytes:&ns32 length:sizeof(ns32) atIndex:1];
                [ce setBytes:&cmp  length:sizeof(cmp)  atIndex:2];
                [ce setBytes:&thr  length:sizeof(thr)  atIndex:3];
                [ce setBytes:&T    length:sizeof(T)    atIndex:4];
                [ce setBytes:&desc length:sizeof(desc) atIndex:5];
                [ce setBuffer:gb_block_buf_  offset:0 atIndex:6];
                [ce setBuffer:gb_block2_buf_ offset:0 atIndex:7];
                [ce dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            });
            n_better = host_scan_u32(static_cast<std::uint32_t*>([gb_block_buf_ contents]), nb);
            const std::size_t n_equal = host_scan_u32(static_cast<std::uint32_t*>([gb_block2_buf_ contents]), nb);
            if (n_better >= f.topk || n_better + n_equal < f.topk)
                throw std::runtime_error(std::string(op) + ": radix select classes inconsistent (internal)");
            need_equal = f.topk - n_better;
            out_rows = f.topk;
            cap_rows(out_rows, max_groups, op);
            const std::uint32_t eb32 = static_cast<std::uint32_t>(n_better);
            const std::uint32_t ne32 = static_cast<std::uint32_t>(need_equal);
            OutBuf ok = out_for(r.keys, out_rows), oc = out_for(r.counts, out_rows), os;
            if (gpu_sums) os = out_for(r.sums, out_rows);
            run(^(id<MTLComputeCommandEncoder> ce) {
                [ce setComputePipelineState:ps_gb_topk_compact_];
                [ce setBuffer:agg offset:0 atIndex:0];
                [ce setBuffer:bk  offset:0 atIndex:1];
                [ce setBuffer:(gpu_sums ? bs : bc) offset:0 atIndex:2];
                [ce setBuffer:bc  offset:0 atIndex:3];
                [ce setBytes:&ns32 length:sizeof(ns32) atIndex:4];
                [ce setBytes:&cmp  length:sizeof(cmp)  atIndex:5];
                [ce setBytes:&thr  length:sizeof(thr)  atIndex:6];
                [ce setBytes:&T    length:sizeof(T)    atIndex:7];
                [ce setBytes:&desc length:sizeof(desc) atIndex:8];
                [ce setBuffer:gb_block_buf_  offset:0 atIndex:9];
                [ce setBuffer:gb_block2_buf_ offset:0 atIndex:10];
                [ce setBytes:&eb32 length:sizeof(eb32) atIndex:11];
                [ce setBytes:&ne32 length:sizeof(ne32) atIndex:12];
                [ce setBuffer:ok.buf offset:0 atIndex:13];
                [ce setBuffer:(gpu_sums ? os.buf : oc.buf) offset:0 atIndex:14];
                [ce setBuffer:oc.buf offset:0 atIndex:15];
                [ce setBytes:&with_sums length:sizeof(with_sums) atIndex:16];
                [ce dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            });
            copy_back(ok, r.keys); copy_back(oc, r.counts);
            if (gpu_sums) copy_back(os, r.sums);
        }

        // Order the k rows by aggregate (desc/asc), ties by key ascending.
        std::vector<std::size_t> idx(r.keys.size());
        for (std::size_t i = 0; i < idx.size(); ++i) idx[i] = i;
        const std::vector<std::int64_t>& a = gpu_sums ? r.sums : r.counts;
        std::sort(idx.begin(), idx.end(), [&](std::size_t x, std::size_t y) {
            if (a[x] != a[y]) return f.topk_desc ? (a[x] > a[y]) : (a[x] < a[y]);
            return r.keys[x] < r.keys[y];
        });
        GroupByResidentResult o{};
        o.keys.reserve(idx.size()); o.counts.reserve(idx.size());
        if (gpu_sums) o.sums.reserve(idx.size());
        for (std::size_t i : idx) {
            o.keys.push_back(r.keys[i]); o.counts.push_back(r.counts[i]);
            if (gpu_sums) o.sums.push_back(r.sums[i]);
        }
        r.keys = std::move(o.keys); r.counts = std::move(o.counts);
        if (gpu_sums) r.sums = std::move(o.sums);
    }

    GroupByResidentResult groupby_impl(const MetalResidentColumn& k,
                                       const MetalResidentColumn* v,
                                       GbMode mode, std::size_t max_groups,
                                       const GroupByFilter& filter, const char* op) {
        const auto t_wall0 = std::chrono::steady_clock::now();
        GroupByResidentResult r{};
        r.rows_in = k.rows();
        const std::size_t n = k.rows();
        if (n == 0) {
            r.wall_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t_wall0).count();
            return r;
        }
        if (n > 0xFFFFFFFFull - 64)
            throw std::runtime_error(std::string(op) + ": > 2^32-64 rows unsupported");

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
        if (!filter.active() && num_segs > max_groups)
            throw std::runtime_error(
                std::string(op) + ": result has " + std::to_string(num_segs) +
                " groups, above the cap of " + std::to_string(max_groups) +
                " (raise GPUDB_GROUPBY_ROWS_MAX_M if intentional)");
        const std::uint32_t ns32 = static_cast<std::uint32_t>(num_segs);

        // ---- Stage B: starts, chunk sums, finalize (one command buffer) ----
        // With a device filter the finalized arrays stay in scratch and only
        // the survivors are written to the result vectors (Stage C).
        const bool gpu_sums   = (mode == GbMode::SumI64);
        const bool dev_filter = filter.active() && mode != GbMode::SumF64;
        OutBuf ok, oc, os;
        id<MTLBuffer> bk = nil, bc = nil, bs = nil;
        if (dev_filter) {
            bk = grow(gb_fkeys_,   num_segs * sizeof(std::int64_t), "filter keys");
            bc = grow(gb_fcounts_, num_segs * sizeof(std::int64_t), "filter counts");
            if (gpu_sums) bs = grow(gb_fsums_, num_segs * sizeof(std::int64_t), "filter sums");
        } else {
            ok = out_for(r.keys, num_segs);
            oc = out_for(r.counts, num_segs);
            if (gpu_sums) os = out_for(r.sums, num_segs);
            bk = ok.buf; bc = oc.buf; bs = os.buf;
        }
        if (gpu_sums) {
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
                [ce setBuffer:bs            offset:0 atIndex:6];
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
            [ce setBuffer:(gpu_sums ? gb_head_buf_ : bc) offset:0 atIndex:4];
            [ce setBuffer:(gpu_sums ? gb_tail_buf_ : bc) offset:0 atIndex:5];
            [ce setBuffer:bk        offset:0 atIndex:6];
            [ce setBuffer:(gpu_sums ? bs : bc) offset:0 atIndex:7];
            [ce setBuffer:bc        offset:0 atIndex:8];
            [ce setBytes:&with_sums length:sizeof(with_sums) atIndex:9];
            [ce dispatchThreadgroups:MTLSizeMake((num_segs + kBlock - 1) / kBlock, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];
            [ce endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            kernel_ms += cb_kernel_ms(cb);
        }
        if (dev_filter) {
            device_filter_i64(r, bk, bs, bc, gpu_sums, num_segs, filter, max_groups, op, kernel_ms);
        } else {
            copy_back(ok, r.keys);
            copy_back(oc, r.counts);
            if (gpu_sums) copy_back(os, r.sums);
            r.groups_total = num_segs;
        }

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

        // f64: the sums were finished on the host, so the filter runs there too
        // (the reference implementation; cap checked on the survivors).
        if (mode == GbMode::SumF64)
            apply_group_filter_host(r, filter, FilterAgg::SumF64, max_groups, op);

        r.kernel_ms   = kernel_ms;
        r.transfer_ms = 0.0;
        r.wall_ms     = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t_wall0).count();
        return r;
    }


    // ---- v0.7 milestone 3: exact GROUP BY (§4.1 / §4.2) ----
    // Same run-start pipeline as groupby_impl over the sorted VALID-KEY
    // prefix; gbx_chunk_i64 / gbx_finalize_i64 produce the native tuple
    // (128-bit sum, count(v), count(*), min, max) under the payload's
    // validity bitmap; the NULL-key suffix is folded into one trailing group
    // on the host (UMA, cost ∝ NULL-key rows); HAVING / top-k run on the
    // device over the tuple (128-bit radix select for the sum) so only the
    // survivors reach the result vectors.
    // grow() for a buffer held in a vector slot (a reference to a vector
    // element cannot be passed as a __strong id& under ARC).
    id<MTLBuffer> grow_slot(std::vector<id<MTLBuffer>>& v, std::size_t i, std::size_t bytes,
                            const char* what) {
        id<MTLBuffer> cur = v[i];
        if (!cur || [cur length] < bytes) {
            cur = [device_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            if (!cur) throw std::runtime_error(std::string("resident group by: allocation failed (") + what + ")");
            v[i] = cur;
        }
        return cur;
    }

    static std::uint32_t agg_code(GroupByFilter::Agg a) {
        switch (a) {
            case GroupByFilter::Agg::Sum:       return 0u;
            case GroupByFilter::Agg::CountV:    return 1u;
            case GroupByFilter::Agg::CountStar: return 2u;
            case GroupByFilter::Agg::Min:       return 3u;
            case GroupByFilter::Agg::Max:       return 4u;
            case GroupByFilter::Agg::Avg:       break;
        }
        throw std::runtime_error("groupby_exact_resident: HAVING / top-k on avg is not on the device path");
    }

    GroupByResidentResult groupby_exact_resident(const ResidentColumn& keys,
                                                 const ResidentColumn* vals,
                                                 std::size_t max_groups,
                                                 const GroupByFilter& filter) override {
        return exact_impl(keys, vals, nullptr, 0, max_groups, filter, "groupby_exact_resident");
    }

    GroupByResidentResult groupby_exact_masked_resident(const ResidentColumn& keys,
                                                        const ResidentColumn* vals,
                                                        const Predicate* preds,
                                                        std::size_t n_preds,
                                                        std::size_t max_groups,
                                                        const GroupByFilter& filter) override {
        return exact_impl(keys, vals, preds, n_preds, max_groups, filter, "groupby_exact_masked_resident");
    }

    // §4.6: multi-lane exact upload — one host pass (UMA) partitions NULL-key
    // rows to a suffix of every lane and de-interleaves into shared buffers;
    // each non-key lane gets its own validity bitmap when it has NULLs.
    std::vector<std::unique_ptr<ResidentColumn>>
    upload_rows_exact(const RowSpan* spans, std::size_t n_spans,
                      const Dtype* dtypes, std::size_t n_lanes) override {
        if (n_lanes == 0) throw std::runtime_error("upload_rows_exact: no lanes");
        if (dtypes[0] != Dtype::I64) throw std::runtime_error("upload_rows_exact: the key lane must be I64");
        if (n_lanes > 1 && dtypes[1] != Dtype::I64)
            throw std::runtime_error(
                "upload_rows_exact: DOUBLE payloads are not on the exact path (docs/TRANSPARENT_DESIGN.md §4.7)");
        @autoreleasepool {
            std::size_t rows = 0, null_keys = 0;
            auto bit = [](const std::uint64_t* m, std::size_t i) {
                return !m || ((m[i >> 6] >> (i & 63)) & 1u);
            };
            auto lane_valid = [](const RowSpan& sp, std::size_t lane) -> const std::uint64_t* {
                return sp.valid ? sp.valid[lane] : nullptr;
            };
            for (std::size_t sidx = 0; sidx < n_spans; ++sidx) {
                if (spans[sidx].n_lanes != n_lanes) throw std::runtime_error("upload_rows_exact: span lane count differs");
                rows += spans[sidx].rows;
                const std::uint64_t* kv = lane_valid(spans[sidx], 0);
                if (kv) for (std::size_t j = 0; j < spans[sidx].rows; ++j) null_keys += !bit(kv, j);
            }
            const std::size_t bytes = (rows == 0) ? 1 : rows * sizeof(std::int64_t);
            const std::size_t words = (rows + 63) / 64;
            std::vector<id<MTLBuffer>> bufs(n_lanes, nil);
            std::vector<std::int64_t*> dst(n_lanes, nullptr);
            for (std::size_t l = 0; l < n_lanes; ++l) {
                bufs[l] = [device_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];
                if (!bufs[l]) throw std::runtime_error("upload_rows_exact: device allocation failed (Metal)");
                dst[l] = static_cast<std::int64_t*>([bufs[l] contents]);
            }
            std::vector<std::vector<std::uint64_t>> valid(n_lanes, std::vector<std::uint64_t>(words, ~std::uint64_t{0}));
            std::vector<std::size_t> nulls(n_lanes, 0);
            std::size_t head = 0, tail = rows - null_keys;
            for (std::size_t sidx = 0; sidx < n_spans; ++sidx) {
                const RowSpan& sp = spans[sidx];
                const std::uint64_t* kv = lane_valid(sp, 0);
                for (std::size_t j = 0; j < sp.rows; ++j) {
                    const bool k_ok = bit(kv, j);
                    const std::size_t d = k_ok ? head++ : tail++;
                    for (std::size_t l = 0; l < n_lanes; ++l) {
                        const bool ok = (l == 0) ? k_ok : bit(lane_valid(sp, l), j);
                        dst[l][d] = ok ? sp.lanes[j * n_lanes + l] : 0;
                        if (!ok) { valid[l][d >> 6] &= ~(std::uint64_t{1} << (d & 63)); ++nulls[l]; }
                    }
                }
            }
            std::vector<std::unique_ptr<ResidentColumn>> out;
            out.reserve(n_lanes);
            for (std::size_t l = 0; l < n_lanes; ++l) {
                id<MTLBuffer> vb = nil;
                if (l != 0 && nulls[l]) {
                    vb = [device_ newBufferWithBytes:valid[l].data()
                                             length:std::max<std::size_t>(8, words * sizeof(std::uint64_t))
                                            options:MTLResourceStorageModeShared];
                    if (!vb) throw std::runtime_error("upload_rows_exact: validity allocation failed (Metal)");
                }
                out.push_back(std::make_unique<MetalResidentColumn>(
                    bufs[l], rows, dtypes[l], sort_ctx_,
                    /*null_suffix*/ l == 0 ? null_keys : 0, vb, nulls[l]));
            }
            return out;
        }
    }

    static std::uint32_t pred_op_code(Predicate::Op op) {
        switch (op) {
            case Predicate::Op::EQ: return 0u;  case Predicate::Op::NE: return 1u;
            case Predicate::Op::LT: return 2u;  case Predicate::Op::LE: return 3u;
            case Predicate::Op::GT: return 4u;  case Predicate::Op::GE: return 5u;
            case Predicate::Op::IsNull: return 6u; case Predicate::Op::IsNotNull: return 7u;
            case Predicate::Op::In: return 8u;
        }
        return 0u;
    }

    // The exact GROUP BY, plain or masked (§4.1, §4.2, §4.6).
    GroupByResidentResult exact_impl(const ResidentColumn& keys, const ResidentColumn* vals,
                                     const Predicate* preds, std::size_t n_preds,
                                     std::size_t max_groups, const GroupByFilter& filter,
                                     const char* op) {
        @autoreleasepool {
            const auto t_wall0 = std::chrono::steady_clock::now();
            const auto& k = check_i64_nullable(keys);
            const MetalResidentColumn* v = vals ? &check_i64_nullable(*vals) : nullptr;
            if (v && v->rows() != k.rows())
                throw std::runtime_error(std::string(op) + ": keys and vals row counts differ");
            GroupByResidentResult r{};
            r.rows_in = k.rows();
            const std::size_t n_total = k.rows();
            if (n_total == 0) {
                r.wall_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - t_wall0).count();
                return r;
            }
            if (n_total > 0xFFFFFFFFull - 64)
                throw std::runtime_error(std::string(op) + ": > 2^32-64 rows unsupported");
            if (filter.active()) (void)agg_code(filter.agg);   // reject avg early
            for (std::size_t p = 0; p < n_preds; ++p) {
                if (!preds[p].col) throw std::runtime_error(std::string(op) + ": predicate without a column");
                if (preds[p].col->backend_tag() != Backend::METAL)
                    throw std::runtime_error("ResidentColumn mismatch (Metal predicate column)");
                if (preds[p].col->rows() != n_total)
                    throw std::runtime_error(std::string(op) + ": predicate column row count differs from the keys");
            }

            const std::size_t n = k.sort_rows();          // valid-key prefix
            double kernel_ms = 0.0;
            id<MTLBuffer> sorted = nil, perm = nil;
            if (n > 0) {
                k.build_sort_cache(&kernel_ms);
                sorted = k.sorted_cache();
                perm   = k.perm_cache();
            }

            // ---- key predicates → a contiguous range of the sorted prefix ----
            // (binary search on the UMA sorted cache, zero per-row cost); the
            // NULL-key group survives only IS NULL or the absence of any key
            // comparison. Everything else goes to the mask.
            std::size_t lo = 0, hi = n;
            bool null_ok = true;
            std::vector<const Predicate*> maskp;
            if (n_preds) {
                const auto* sk = sorted ? static_cast<const std::int64_t*>([sorted contents]) : nullptr;
                auto lower = [&](std::int64_t x) { return sk ? static_cast<std::size_t>(std::lower_bound(sk, sk + n, x) - sk) : 0; };
                auto upper = [&](std::int64_t x) { return sk ? static_cast<std::size_t>(std::upper_bound(sk, sk + n, x) - sk) : 0; };
                for (std::size_t p = 0; p < n_preds; ++p) {
                    const Predicate& pr = preds[p];
                    if (pr.col != &k) { maskp.push_back(&pr); continue; }
                    switch (pr.op) {
                        case Predicate::Op::IsNull:    hi = std::min(hi, lo); break;              // keys: none; NULL group stays
                        case Predicate::Op::IsNotNull: null_ok = false; break;
                        case Predicate::Op::EQ: lo = std::max(lo, lower(pr.value)); hi = std::min(hi, upper(pr.value)); null_ok = false; break;
                        case Predicate::Op::LT: hi = std::min(hi, lower(pr.value)); null_ok = false; break;
                        case Predicate::Op::LE: hi = std::min(hi, upper(pr.value)); null_ok = false; break;
                        case Predicate::Op::GT: lo = std::max(lo, upper(pr.value)); null_ok = false; break;
                        case Predicate::Op::GE: lo = std::max(lo, lower(pr.value)); null_ok = false; break;
                        default: maskp.push_back(&pr); null_ok = false; break;   // NE, In: mask (NULL keys fail them)
                    }
                }
                if (lo > hi) lo = hi;
            }
            const std::size_t range_n = hi - lo;
            const bool masked = !maskp.empty();

            // ---- mask over the ORIGINAL rows (one kernel pass per predicate) and
            //      the survivor count over the key range, in ONE command buffer ----
            const std::size_t sel_nb = (range_n + kBlock - 1) / kBlock;
            if (masked) {
                grow(gbx_mask_buf_, std::max<std::size_t>(1, n_total), "where mask");
                if (!gbx_dummy_valid_)
                    gbx_dummy_valid_ = [device_ newBufferWithLength:8 options:MTLResourceStorageModeShared];
                if (range_n > 0) grow(gb_block_buf_, sel_nb * sizeof(std::uint32_t), "select block counts");
                const std::uint32_t nt32 = static_cast<std::uint32_t>(n_total);
                const std::uint32_t rn32 = static_cast<std::uint32_t>(range_n);
                std::vector<id<MTLBuffer>> lists(maskp.size(), nil);
                for (std::size_t p = 0; p < maskp.size(); ++p) {
                    const Predicate& pr = *maskp[p];
                    if (pr.op == Predicate::Op::In && pr.n_list) {
                        lists[p] = [device_ newBufferWithBytes:pr.list length:pr.n_list * sizeof(std::int64_t)
                                                       options:MTLResourceStorageModeShared];
                        if (!lists[p]) throw std::runtime_error("where mask: IN list allocation failed (Metal)");
                    }
                }
                id<MTLCommandBuffer>         cb = [queue_ commandBuffer];
                id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
                for (std::size_t p = 0; p < maskp.size(); ++p) {
                    const Predicate& pr = *maskp[p];
                    const auto& pc = static_cast<const MetalResidentColumn&>(*pr.col);
                    const bool has_bitmap = pc.valid_buffer() != nil;
                    const std::uint32_t has_valid = has_bitmap ? 1u : 0u;
                    const std::uint32_t null_from = pc.null_suffix()
                        ? static_cast<std::uint32_t>(pc.sort_rows()) : 0xFFFFFFFFu;
                    const std::uint32_t is_f64 = pc.dtype() == Dtype::F64 ? 1u : 0u;
                    const std::uint32_t opc = pred_op_code(pr.op);
                    const std::int64_t  value = pr.value;
                    const std::uint32_t n_list = static_cast<std::uint32_t>(pr.op == Predicate::Op::In ? pr.n_list : 0);
                    const std::uint32_t first = p == 0 ? 1u : 0u;
                    id<MTLBuffer> list = lists[p] ? lists[p] : gbx_dummy_valid_;
                    [ce setComputePipelineState:ps_gbx_mask_];
                    [ce setBuffer:pc.buffer() offset:0 atIndex:0];
                    [ce setBuffer:(has_bitmap ? pc.valid_buffer() : gbx_dummy_valid_) offset:0 atIndex:1];
                    [ce setBytes:&has_valid length:sizeof(has_valid) atIndex:2];
                    [ce setBytes:&null_from length:sizeof(null_from) atIndex:3];
                    [ce setBytes:&nt32      length:sizeof(nt32)      atIndex:4];
                    [ce setBytes:&is_f64    length:sizeof(is_f64)    atIndex:5];
                    [ce setBytes:&opc       length:sizeof(opc)       atIndex:6];
                    [ce setBytes:&value     length:sizeof(value)     atIndex:7];
                    [ce setBuffer:list offset:0 atIndex:8];
                    [ce setBytes:&n_list    length:sizeof(n_list)    atIndex:9];
                    [ce setBytes:&first     length:sizeof(first)     atIndex:10];
                    [ce setBuffer:gbx_mask_buf_ offset:0 atIndex:11];
                    [ce dispatchThreadgroups:MTLSizeMake((n_total + kBlock - 1) / kBlock, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];
                    [ce memoryBarrierWithScope:MTLBarrierScopeBuffers];
                }
                if (range_n > 0) {
                    [ce setComputePipelineState:ps_gbx_sel_counts_];
                    [ce setBuffer:perm offset:lo * sizeof(std::int64_t) atIndex:0];
                    [ce setBuffer:gbx_mask_buf_ offset:0 atIndex:1];
                    [ce setBytes:&rn32 length:sizeof(rn32) atIndex:2];
                    [ce setBuffer:gb_block_buf_ offset:0 atIndex:3];
                    [ce dispatchThreadgroups:MTLSizeMake(sel_nb, 1, 1) threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];
                }
                [ce endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                kernel_ms += cb_kernel_ms(cb);
            }

            // ---- choose the reduce input: the range as is, its compaction (b), or masked (a) ----
            id<MTLBuffer> run_sorted = sorted, run_perm = perm;
            std::size_t run_off = lo, run_n = range_n;
            bool mask_in_reduce = false;
            bool compact_pending = false;     // variant (b): compaction runs in Stage A's command buffer
            std::size_t n_sel_b = 0;
            if (masked && range_n > 0) {
                const std::size_t nb = sel_nb;
                const std::size_t n_sel = host_scan_u32(static_cast<std::uint32_t*>([gb_block_buf_ contents]), nb);
                if (n_sel == 0) {
                    run_n = 0;
                } else if (static_cast<double>(n_sel) < compact_below_ * static_cast<double>(range_n)) {
                    // variant (b): compact sorted keys + perm to the survivors
                    // (encoded below, in Stage A's command buffer)
                    run_sorted = grow(gbx_sel_keys_, n_sel * sizeof(std::int64_t), "compacted keys");
                    run_perm   = grow(gbx_sel_perm_, n_sel * sizeof(std::int64_t), "compacted perm");
                    run_off = 0; run_n = n_sel; compact_pending = true; n_sel_b = n_sel;
                } else {
                    mask_in_reduce = true;               // variant (a)
                }
            }
            const std::size_t soff = run_off * sizeof(std::int64_t);
            const std::uint32_t n32 = static_cast<std::uint32_t>(run_n);
            const std::size_t nblocks = (run_n + kBlock - 1) / kBlock;
            const std::size_t nchunks = (run_n + 63) / 64;

            // ---- Stage A: (variant (b) compaction, then) run starts over the reduce input ----
            std::size_t num_segs = 0;
            if (run_n > 0) {
                // Variant (b): the select offsets live in gb_block_buf_ and are
                // read by the compaction below, so Stage A counts go to a
                // second buffer. (grow() may allocate: take the handle after.)
                id<MTLBuffer> block_a = compact_pending
                    ? grow(gb_block2_buf_, nblocks * sizeof(std::uint32_t), "block counts")
                    : grow(gb_block_buf_,  nblocks * sizeof(std::uint32_t), "block counts");
                grow(mult_buf_, run_n * sizeof(std::uint32_t), "run starts");
                id<MTLCommandBuffer>         cb = [queue_ commandBuffer];
                id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
                if (compact_pending) {
                    const std::uint32_t rn32 = static_cast<std::uint32_t>(range_n);
                    [ce setComputePipelineState:ps_gbx_sel_compact_];
                    [ce setBuffer:sorted offset:lo * sizeof(std::int64_t) atIndex:0];
                    [ce setBuffer:perm   offset:lo * sizeof(std::int64_t) atIndex:1];
                    [ce setBuffer:gbx_mask_buf_ offset:0 atIndex:2];
                    [ce setBytes:&rn32 length:sizeof(rn32) atIndex:3];
                    [ce setBuffer:gb_block_buf_ offset:0 atIndex:4];
                    [ce setBuffer:run_sorted offset:0 atIndex:5];
                    [ce setBuffer:run_perm   offset:0 atIndex:6];
                    [ce dispatchThreadgroups:MTLSizeMake(sel_nb, 1, 1) threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];
                    [ce memoryBarrierWithScope:MTLBarrierScopeBuffers];
                }
                [ce setComputePipelineState:ps_gb_block_counts_];
                [ce setBuffer:run_sorted    offset:soff atIndex:0];
                [ce setBytes:&n32 length:sizeof(n32) atIndex:1];
                [ce setBuffer:block_a offset:0 atIndex:2];
                [ce dispatchThreadgroups:MTLSizeMake(nblocks, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];
                [ce endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                kernel_ms += cb_kernel_ms(cb);
                auto* bc = static_cast<std::uint32_t*>([block_a contents]);
                for (std::size_t b = 0; b < nblocks; ++b) {
                    const std::uint32_t c = bc[b];
                    bc[b] = static_cast<std::uint32_t>(num_segs);
                    num_segs += c;
                }
                if (compact_pending) {
                    // Stage B reads the block offsets from gb_block_buf_
                    std::memcpy([gb_block_buf_ contents], bc, nblocks * sizeof(std::uint32_t));
                    (void)n_sel_b;
                }
            }

            // ---- the NULL-key group: fold the suffix on the host (UMA), under the mask ----
            Sum128 ns; std::int64_t ncnt = 0, ncstar = 0;
            std::int64_t nmn = std::numeric_limits<std::int64_t>::max();
            std::int64_t nmx = std::numeric_limits<std::int64_t>::min();
            if (null_ok && k.null_suffix() > 0) {
                const std::uint8_t* mk = masked ? static_cast<const std::uint8_t*>([gbx_mask_buf_ contents]) : nullptr;
                const auto* vd = v ? static_cast<const std::int64_t*>([v->buffer() contents]) : nullptr;
                const auto* vv = (v && v->valid_buffer())
                    ? static_cast<const std::uint64_t*>([v->valid_buffer() contents]) : nullptr;
                for (std::size_t i = n; i < n_total; ++i) {
                    if (mk && !mk[i]) continue;
                    ++ncstar;
                    if (!v) continue;
                    if (vv && !((vv[i >> 6] >> (i & 63)) & 1u)) continue;
                    const std::int64_t x = vd[i];
                    ns.add(x); ++ncnt; nmn = std::min(nmn, x); nmx = std::max(nmx, x);
                }
                if (!v) ncnt = ncstar;
            }
            const bool null_group = ncstar > 0;
            const std::size_t total = num_segs + (null_group ? 1 : 0);
            // With the mask inside the reduce, groups whose every row was
            // masked out are still counted in num_segs and dropped by the
            // compaction below, so the cap is checked there instead.
            const bool dev_path = filter.active() || mask_in_reduce;
            if (!dev_path && total > max_groups)
                throw std::runtime_error(
                    std::string(op) + ": result has " + std::to_string(total) +
                    " groups, above the cap of " + std::to_string(max_groups) +
                    " (raise GPUDB_GROUPBY_ROWS_MAX_M if intentional)");
            const std::uint32_t ns32 = static_cast<std::uint32_t>(num_segs);

            // ---- finalized tuple arrays: lo hi cnt cstar mn mx keys (+ key_null) ----
            std::vector<OutBuf> o(7);
            std::vector<id<MTLBuffer>> b(8, nil);
            if (dev_path) {
                const std::size_t bytes = std::max<std::size_t>(1, total) * sizeof(std::int64_t);
                for (int i = 0; i < 7; ++i) b[i] = grow_slot(gbx_f_, i, bytes, "exact filter scratch");
                b[7] = grow_slot(gbx_f_, 7, std::max<std::size_t>(1, total), "exact filter key_null");
            } else {
                // Result vectors alias the device output where the caller
                // reads them; the rest land in scratch (GroupByFilter::columns).
                const std::size_t bytes = std::max<std::size_t>(1, total) * sizeof(std::int64_t);
                std::vector<std::int64_t>* vecs[7] = { &r.sums, &r.sums_hi, &r.counts, &r.counts_star,
                                                       &r.mins, &r.maxs, &r.keys };
                const unsigned bits[7] = { 1, 1, 2, 3, 4, 5, 0 };
                for (int i = 0; i < 7; ++i) {
                    if (filter.wants(bits[i])) { o[i] = out_for(*vecs[i], total); b[i] = o[i].buf; }
                    else                       { b[i] = grow_slot(gbx_f_, i, bytes, "exact unread column"); }
                }
            }

            // ---- Stage B: starts, chunk tuples, finalize ----
            if (run_n > 0) {
                const std::uint32_t with_vals = v ? 1u : 0u;
                if (!gbx_dummy_valid_)
                    gbx_dummy_valid_ = [device_ newBufferWithLength:8 options:MTLResourceStorageModeShared];
                const bool has_bitmap = v && v->valid_buffer();
                id<MTLBuffer> valid = has_bitmap ? v->valid_buffer() : gbx_dummy_valid_;
                const std::uint32_t has_valid = has_bitmap ? 1u : 0u;
                id<MTLBuffer> vbuf = v ? v->buffer() : gbx_dummy_valid_;

                id<MTLCommandBuffer>         cb = [queue_ commandBuffer];
                id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
                [ce setComputePipelineState:ps_gb_run_starts_];
                [ce setBuffer:run_sorted    offset:soff atIndex:0];
                [ce setBytes:&n32 length:sizeof(n32) atIndex:1];
                [ce setBuffer:gb_block_buf_ offset:0 atIndex:2];
                [ce setBuffer:mult_buf_     offset:0 atIndex:3];
                [ce dispatchThreadgroups:MTLSizeMake(nblocks, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];
                [ce memoryBarrierWithScope:MTLBarrierScopeBuffers];
                if (mask_in_reduce) {
                    // variant (a): masked reduce, count(*) in the tuple
                    grow(gbxm_head_buf_, nchunks * 6 * sizeof(std::int64_t), "masked head partials");
                    grow(gbxm_tail_buf_, nchunks * 6 * sizeof(std::int64_t), "masked tail partials");
                    [ce setComputePipelineState:ps_gbxm_chunk_];
                    [ce setBuffer:run_perm      offset:soff atIndex:0];
                    [ce setBuffer:vbuf          offset:0 atIndex:1];
                    [ce setBuffer:valid         offset:0 atIndex:2];
                    [ce setBytes:&has_valid length:sizeof(has_valid) atIndex:3];
                    [ce setBuffer:mult_buf_     offset:0 atIndex:4];
                    [ce setBytes:&n32  length:sizeof(n32)  atIndex:5];
                    [ce setBytes:&ns32 length:sizeof(ns32) atIndex:6];
                    [ce setBuffer:gbx_mask_buf_ offset:0 atIndex:7];
                    [ce setBytes:&with_vals length:sizeof(with_vals) atIndex:8];
                    [ce setBuffer:b[0] offset:0 atIndex:9];    // lo
                    [ce setBuffer:b[1] offset:0 atIndex:10];   // hi
                    [ce setBuffer:b[2] offset:0 atIndex:11];   // cnt
                    [ce setBuffer:b[3] offset:0 atIndex:12];   // cstar
                    [ce setBuffer:b[4] offset:0 atIndex:13];   // mn
                    [ce setBuffer:b[5] offset:0 atIndex:14];   // mx
                    [ce setBuffer:gbxm_head_buf_ offset:0 atIndex:15];
                    [ce setBuffer:gbxm_tail_buf_ offset:0 atIndex:16];
                    [ce dispatchThreadgroups:MTLSizeMake((nchunks + kBlock - 1) / kBlock, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];
                    [ce memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    [ce setComputePipelineState:ps_gbxm_finalize_];
                    [ce setBuffer:run_sorted offset:soff atIndex:0];
                    [ce setBuffer:mult_buf_  offset:0 atIndex:1];
                    [ce setBytes:&n32  length:sizeof(n32)  atIndex:2];
                    [ce setBytes:&ns32 length:sizeof(ns32) atIndex:3];
                    [ce setBuffer:gbxm_head_buf_ offset:0 atIndex:4];
                    [ce setBuffer:gbxm_tail_buf_ offset:0 atIndex:5];
                    [ce setBuffer:b[6] offset:0 atIndex:6];    // keys
                    [ce setBuffer:b[3] offset:0 atIndex:7];    // cstar
                    [ce setBuffer:b[0] offset:0 atIndex:8];
                    [ce setBuffer:b[1] offset:0 atIndex:9];
                    [ce setBuffer:b[2] offset:0 atIndex:10];
                    [ce setBuffer:b[4] offset:0 atIndex:11];
                    [ce setBuffer:b[5] offset:0 atIndex:12];
                    [ce dispatchThreadgroups:MTLSizeMake((num_segs + kBlock - 1) / kBlock, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];
                } else {
                    if (v) {
                        grow(gbx_head_buf_, nchunks * 5 * sizeof(std::int64_t), "exact head partials");
                        grow(gbx_tail_buf_, nchunks * 5 * sizeof(std::int64_t), "exact tail partials");
                        [ce setComputePipelineState:ps_gbx_chunk_];
                        [ce setBuffer:run_perm      offset:soff atIndex:0];
                        [ce setBuffer:v->buffer()   offset:0 atIndex:1];
                        [ce setBuffer:valid         offset:0 atIndex:2];
                        [ce setBytes:&has_valid length:sizeof(has_valid) atIndex:3];
                        [ce setBuffer:mult_buf_     offset:0 atIndex:4];
                        [ce setBytes:&n32  length:sizeof(n32)  atIndex:5];
                        [ce setBytes:&ns32 length:sizeof(ns32) atIndex:6];
                        [ce setBuffer:b[0] offset:0 atIndex:7];
                        [ce setBuffer:b[1] offset:0 atIndex:8];
                        [ce setBuffer:b[2] offset:0 atIndex:9];
                        [ce setBuffer:b[4] offset:0 atIndex:10];
                        [ce setBuffer:b[5] offset:0 atIndex:11];
                        [ce setBuffer:gbx_head_buf_ offset:0 atIndex:12];
                        [ce setBuffer:gbx_tail_buf_ offset:0 atIndex:13];
                        [ce dispatchThreadgroups:MTLSizeMake((nchunks + kBlock - 1) / kBlock, 1, 1)
                           threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];
                        [ce memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    }
                    [ce setComputePipelineState:ps_gbx_finalize_];
                    [ce setBuffer:run_sorted offset:soff atIndex:0];
                    [ce setBuffer:mult_buf_  offset:0 atIndex:1];
                    [ce setBytes:&n32  length:sizeof(n32)  atIndex:2];
                    [ce setBytes:&ns32 length:sizeof(ns32) atIndex:3];
                    [ce setBuffer:(v ? gbx_head_buf_ : b[0]) offset:0 atIndex:4];
                    [ce setBuffer:(v ? gbx_tail_buf_ : b[0]) offset:0 atIndex:5];
                    [ce setBuffer:b[6] offset:0 atIndex:6];    // keys
                    [ce setBuffer:b[3] offset:0 atIndex:7];    // cstar
                    [ce setBuffer:b[0] offset:0 atIndex:8];    // lo
                    [ce setBuffer:b[1] offset:0 atIndex:9];    // hi
                    [ce setBuffer:b[2] offset:0 atIndex:10];   // cnt
                    [ce setBuffer:b[4] offset:0 atIndex:11];   // mn
                    [ce setBuffer:b[5] offset:0 atIndex:12];   // mx
                    [ce setBytes:&with_vals length:sizeof(with_vals) atIndex:13];
                    [ce dispatchThreadgroups:MTLSizeMake((num_segs + kBlock - 1) / kBlock, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];
                }
                [ce endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                kernel_ms += cb_kernel_ms(cb);
            }

            if (null_group) {
                static_cast<std::int64_t*>([b[0] contents])[num_segs] = static_cast<std::int64_t>(ns.lo);
                static_cast<std::int64_t*>([b[1] contents])[num_segs] = ns.hi;
                static_cast<std::int64_t*>([b[2] contents])[num_segs] = ncnt;
                static_cast<std::int64_t*>([b[3] contents])[num_segs] = ncstar;
                static_cast<std::int64_t*>([b[4] contents])[num_segs] = ncnt ? nmn : 0;
                static_cast<std::int64_t*>([b[5] contents])[num_segs] = ncnt ? nmx : 0;
                static_cast<std::int64_t*>([b[6] contents])[num_segs] = 0;
            }

            if (dev_path) {
                auto* kn = static_cast<std::uint8_t*>([b[7] contents]);
                std::memset(kn, 0, std::max<std::size_t>(1, total));
                if (null_group) kn[num_segs] = 1;
                device_filter_exact(r, b, total, filter, max_groups, op, kernel_ms);
            } else {
                if (filter.wants(1)) { copy_back(o[0], r.sums);   copy_back(o[1], r.sums_hi); }
                if (filter.wants(2))   copy_back(o[2], r.counts);
                if (filter.wants(3))   copy_back(o[3], r.counts_star);
                if (filter.wants(4))   copy_back(o[4], r.mins);
                if (filter.wants(5))   copy_back(o[5], r.maxs);
                if (filter.wants(0)) {
                    copy_back(o[6], r.keys);
                    r.key_null.assign(total, 0);
                    if (null_group) r.key_null[num_segs] = 1;
                }
                r.groups_total = total;
            }
            r.kernel_ms   = kernel_ms;
            r.transfer_ms = 0.0;
            r.wall_ms     = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - t_wall0).count();
            return r;
        }
    }

    // HAVING / top-k over the finalized exact tuple, on the device. `b` is
    // lo hi cnt cstar mn mx keys key_null(uchar), each `total` long.
    void device_filter_exact(GroupByResidentResult& r, const std::vector<id<MTLBuffer>>& b,
                             std::size_t total,
                             const GroupByFilter& f, std::size_t max_groups,
                             const char* op, double& kernel_ms) {
        r.groups_total = total;
        const std::uint32_t ns32 = static_cast<std::uint32_t>(total);
        const std::uint32_t agg  = f.active() ? agg_code(f.agg) : 0u;
        const std::uint32_t cmp  = f.active() ? cmp_code(f.cmp) : 7u;   // 7 = every non-empty group
        const std::int64_t  thr  = f.threshold_i64;
        const std::uint32_t desc = f.topk_desc ? 1u : 0u;
        const std::size_t   nb   = (total + kBlock - 1) / kBlock;
        grow(gb_block_buf_,  nb * sizeof(std::uint32_t), "filter block counts");
        grow(gb_block2_buf_, nb * sizeof(std::uint32_t), "filter block counts (ties)");
        const MTLSize grid = MTLSizeMake(nb, 1, 1), tg = MTLSizeMake(kBlock, 1, 1);

        auto run = [&](void (^enc)(id<MTLComputeCommandEncoder>)) {
            id<MTLCommandBuffer>         cb = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
            enc(ce);
            [ce endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            kernel_ms += cb_kernel_ms(cb);
        };
        // Per-block survivor counts for a cmp code (5 = "aggregate IS NULL"),
        // scanned in place → offsets; returns the survivor total.
        auto count_pass = [&](std::uint32_t cmpc) -> std::size_t {
            run(^(id<MTLComputeCommandEncoder> ce) {
                [ce setComputePipelineState:ps_gbx_having_counts_];
                for (int i = 0; i < 6; ++i) [ce setBuffer:b[i] offset:0 atIndex:i];
                [ce setBytes:&ns32 length:sizeof(ns32) atIndex:6];
                [ce setBytes:&agg  length:sizeof(agg)  atIndex:7];
                [ce setBytes:&cmpc length:sizeof(cmpc) atIndex:8];
                [ce setBytes:&thr  length:sizeof(thr)  atIndex:9];
                [ce setBuffer:gb_block_buf_ offset:0 atIndex:10];
                [ce dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            });
            return host_scan_u32(static_cast<std::uint32_t*>([gb_block_buf_ contents]), nb);
        };
        std::vector<OutBuf> o(7);
        id<MTLBuffer> kn_out = nil;
        // Columns the caller reads; a top-k result is re-ordered on the host
        // by its aggregate and key, so everything is kept in that case.
        const std::uint32_t want = f.topk ? GroupByFilter::kAllColumns : f.columns;
        auto wants = [&](unsigned bit) { return (want >> bit) & 1u; };
        std::vector<std::int64_t>* vecs[7] = { &r.sums, &r.sums_hi, &r.counts, &r.counts_star,
                                               &r.mins, &r.maxs, &r.keys };
        const unsigned bits[7] = { 1, 1, 2, 3, 4, 5, 0 };
        std::vector<id<MTLBuffer>> unread(7, nil);
        auto alloc_out = [&](std::size_t rows) {
            const std::size_t bytes = std::max<std::size_t>(1, rows) * sizeof(std::int64_t);
            for (int i = 0; i < 7; ++i) {
                if (wants(bits[i])) { o[i] = out_for(*vecs[i], rows); }
                else { o[i] = OutBuf{}; o[i].buf = grow_slot(unread, i, bytes, "exact unread survivor column"); o[i].aliased = true; }
            }
            kn_out = grow(gbx_knull_out_, std::max<std::size_t>(1, rows), "exact key_null out");
        };
        auto compact = [&](std::uint32_t cmpc, std::uint32_t base, std::uint32_t limit) {
            run(^(id<MTLComputeCommandEncoder> ce) {
                [ce setComputePipelineState:ps_gbx_having_compact_];
                for (int i = 0; i < 8; ++i) [ce setBuffer:b[i] offset:0 atIndex:i];
                [ce setBytes:&ns32  length:sizeof(ns32)  atIndex:8];
                [ce setBytes:&agg   length:sizeof(agg)   atIndex:9];
                [ce setBytes:&cmpc  length:sizeof(cmpc)  atIndex:10];
                [ce setBytes:&thr   length:sizeof(thr)   atIndex:11];
                [ce setBuffer:gb_block_buf_ offset:0 atIndex:12];
                [ce setBytes:&base  length:sizeof(base)  atIndex:13];
                [ce setBytes:&limit length:sizeof(limit) atIndex:14];
                [ce setBuffer:o[6].buf offset:0 atIndex:15];   // keys
                [ce setBuffer:kn_out   offset:0 atIndex:16];   // key_null
                [ce setBuffer:o[0].buf offset:0 atIndex:17];   // lo
                [ce setBuffer:o[1].buf offset:0 atIndex:18];   // hi
                [ce setBuffer:o[2].buf offset:0 atIndex:19];   // cnt
                [ce setBuffer:o[3].buf offset:0 atIndex:20];   // cstar
                [ce setBuffer:o[4].buf offset:0 atIndex:21];   // mn
                [ce setBuffer:o[5].buf offset:0 atIndex:22];   // mx
                [ce dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            });
        };
        auto finish = [&](std::size_t rows) {
            for (int i = 0; i < 7; ++i) if (wants(bits[i])) copy_back(o[i], *vecs[i]);
            if (wants(0)) {
                r.key_null.assign(rows, 0);
                if (rows) std::memcpy(r.key_null.data(), [kn_out contents], rows);
            }
        };

        // ---- no filter (masked reduce dropped groups) or HAVING only:
        //      survivors keep key order (compaction is stable) ----
        if (f.topk == 0) {
            const std::size_t surv = count_pass(cmp);
            if (!f.active()) r.groups_total = surv;     // empty groups were never groups
            cap_rows(surv, max_groups, op);
            alloc_out(surv);
            if (surv > 0) compact(cmp, 0u, static_cast<std::uint32_t>(surv));
            finish(surv);
            return;
        }
        // Under a filter over a masked reduce, groups_total must not count
        // the empty (masked-out) groups either.
        r.groups_total = count_pass(7u);

        // ---- top-k ----
        // Candidates pass cmp and have a non-NULL aggregate. Without a HAVING
        // the NULL-aggregate groups rank last and fill the tail when k exceeds
        // the candidates (NULLS LAST in both directions, as native ORDER BY).
        const bool nullable_agg = (f.agg == GroupByFilter::Agg::Sum || f.agg == GroupByFilter::Agg::Min ||
                                   f.agg == GroupByFilter::Agg::Max);
        const std::size_t cand = count_pass(cmp);
        std::size_t out_rows = 0;
        if (cand <= f.topk) {
            std::size_t nulls_avail = 0;
            if (f.cmp == GroupByFilter::Cmp::None && nullable_agg) nulls_avail = count_pass(5u);
            const std::size_t need_null = std::min(f.topk - cand, nulls_avail);
            out_rows = cand + need_null;
            cap_rows(out_rows, max_groups, op);
            alloc_out(out_rows);
            if (cand > 0) {
                (void)count_pass(cmp);              // offsets for this class
                compact(cmp, 0u, static_cast<std::uint32_t>(cand));
            }
            if (need_null > 0) {
                (void)count_pass(5u);
                compact(5u, static_cast<std::uint32_t>(cand), static_cast<std::uint32_t>(need_null));
            }
            finish(out_rows);
        } else {
            // Radix select on the 128-bit ordinal (16 passes for the sum: high
            // word then low word; 8 for the int64 aggregates).
            grow(gb_hist_buf_, 256 * sizeof(std::uint32_t), "radix-select histogram");
            auto* hist = static_cast<std::uint32_t*>([gb_hist_buf_ contents]);
            const int passes = (agg == 0u) ? 16 : 8;
            std::uint64_t prefix_hi = 0, mask_hi = 0, prefix_lo = 0, mask_lo = 0;
            std::size_t remaining = f.topk;
            for (int pass = 0; pass < passes; ++pass) {
                const std::uint32_t word  = pass < 8 ? 0u : 1u;
                const std::uint32_t shift = static_cast<std::uint32_t>(56 - 8 * (pass % 8));
                std::memset(hist, 0, 256 * sizeof(std::uint32_t));
                run(^(id<MTLComputeCommandEncoder> ce) {
                    [ce setComputePipelineState:ps_gbx_topk_hist_];
                    for (int i = 0; i < 6; ++i) [ce setBuffer:b[i] offset:0 atIndex:i];
                    [ce setBytes:&ns32      length:sizeof(ns32)      atIndex:6];
                    [ce setBytes:&agg       length:sizeof(agg)       atIndex:7];
                    [ce setBytes:&cmp       length:sizeof(cmp)       atIndex:8];
                    [ce setBytes:&thr       length:sizeof(thr)       atIndex:9];
                    [ce setBytes:&prefix_hi length:sizeof(prefix_hi) atIndex:10];
                    [ce setBytes:&mask_hi   length:sizeof(mask_hi)   atIndex:11];
                    [ce setBytes:&prefix_lo length:sizeof(prefix_lo) atIndex:12];
                    [ce setBytes:&mask_lo   length:sizeof(mask_lo)   atIndex:13];
                    [ce setBytes:&word      length:sizeof(word)      atIndex:14];
                    [ce setBytes:&shift     length:sizeof(shift)     atIndex:15];
                    [ce setBuffer:gb_hist_buf_ offset:0 atIndex:16];
                    [ce dispatchThreadgroups:grid threadsPerThreadgroup:tg];
                });
                int chosen = -1;
                if (desc) {
                    for (int bkt = 255; bkt >= 0; --bkt) {
                        if (remaining <= hist[bkt]) { chosen = bkt; break; }
                        remaining -= hist[bkt];
                    }
                } else {
                    for (int bkt = 0; bkt < 256; ++bkt) {
                        if (remaining <= hist[bkt]) { chosen = bkt; break; }
                        remaining -= hist[bkt];
                    }
                }
                if (chosen < 0)
                    throw std::runtime_error(std::string(op) + ": radix select lost the k-th rank (internal)");
                if (word == 0u) {
                    prefix_hi |= static_cast<std::uint64_t>(chosen) << shift;
                    mask_hi   |= static_cast<std::uint64_t>(0xFF) << shift;
                } else {
                    prefix_lo |= static_cast<std::uint64_t>(chosen) << shift;
                    mask_lo   |= static_cast<std::uint64_t>(0xFF) << shift;
                }
            }
            const std::uint64_t T_hi = prefix_hi, T_lo = prefix_lo;
            run(^(id<MTLComputeCommandEncoder> ce) {
                [ce setComputePipelineState:ps_gbx_topk_counts_];
                for (int i = 0; i < 6; ++i) [ce setBuffer:b[i] offset:0 atIndex:i];
                [ce setBytes:&ns32 length:sizeof(ns32) atIndex:6];
                [ce setBytes:&agg  length:sizeof(agg)  atIndex:7];
                [ce setBytes:&cmp  length:sizeof(cmp)  atIndex:8];
                [ce setBytes:&thr  length:sizeof(thr)  atIndex:9];
                [ce setBytes:&T_hi length:sizeof(T_hi) atIndex:10];
                [ce setBytes:&T_lo length:sizeof(T_lo) atIndex:11];
                [ce setBytes:&desc length:sizeof(desc) atIndex:12];
                [ce setBuffer:gb_block_buf_  offset:0 atIndex:13];
                [ce setBuffer:gb_block2_buf_ offset:0 atIndex:14];
                [ce dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            });
            const std::size_t n_better = host_scan_u32(static_cast<std::uint32_t*>([gb_block_buf_ contents]), nb);
            const std::size_t n_equal  = host_scan_u32(static_cast<std::uint32_t*>([gb_block2_buf_ contents]), nb);
            if (n_better >= f.topk || n_better + n_equal < f.topk)
                throw std::runtime_error(std::string(op) + ": radix select classes inconsistent (internal)");
            const std::size_t need_equal = f.topk - n_better;
            out_rows = f.topk;
            cap_rows(out_rows, max_groups, op);
            const std::uint32_t eb32 = static_cast<std::uint32_t>(n_better);
            const std::uint32_t ne32 = static_cast<std::uint32_t>(need_equal);
            alloc_out(out_rows);
            run(^(id<MTLComputeCommandEncoder> ce) {
                [ce setComputePipelineState:ps_gbx_topk_compact_];
                for (int i = 0; i < 8; ++i) [ce setBuffer:b[i] offset:0 atIndex:i];
                [ce setBytes:&ns32 length:sizeof(ns32) atIndex:8];
                [ce setBytes:&agg  length:sizeof(agg)  atIndex:9];
                [ce setBytes:&cmp  length:sizeof(cmp)  atIndex:10];
                [ce setBytes:&thr  length:sizeof(thr)  atIndex:11];
                [ce setBytes:&T_hi length:sizeof(T_hi) atIndex:12];
                [ce setBytes:&T_lo length:sizeof(T_lo) atIndex:13];
                [ce setBytes:&desc length:sizeof(desc) atIndex:14];
                [ce setBuffer:gb_block_buf_  offset:0 atIndex:15];
                [ce setBuffer:gb_block2_buf_ offset:0 atIndex:16];
                [ce setBytes:&eb32 length:sizeof(eb32) atIndex:17];
                [ce setBytes:&ne32 length:sizeof(ne32) atIndex:18];
                [ce setBuffer:o[6].buf offset:0 atIndex:19];
                [ce setBuffer:kn_out   offset:0 atIndex:20];
                [ce setBuffer:o[0].buf offset:0 atIndex:21];
                [ce setBuffer:o[1].buf offset:0 atIndex:22];
                [ce setBuffer:o[2].buf offset:0 atIndex:23];
                [ce setBuffer:o[3].buf offset:0 atIndex:24];
                [ce setBuffer:o[4].buf offset:0 atIndex:25];
                [ce setBuffer:o[5].buf offset:0 atIndex:26];
                [ce dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            });
            finish(out_rows);
        }
        // Order the k rows by the aggregate (desc/asc), NULL aggregates last,
        // ties by key ascending — the host reference comparator.
        if (out_rows > 1) {
            GroupByFilter order;
            order.topk = out_rows; order.topk_desc = f.topk_desc; order.agg = f.agg;
            const std::size_t gt = r.groups_total;
            apply_group_filter_host(r, order, FilterAgg::Exact, max_groups, op);
            r.groups_total = gt;
        }
    }

    std::unique_ptr<ResidentColumn>
    make_resident(const void* src, std::size_t n, Dtype dt, std::size_t elem) {
        @autoreleasepool {
            const std::size_t bytes = (n == 0) ? 1 : n * elem;
            id<MTLBuffer> buf = [device_ newBufferWithLength:bytes
                                                     options:MTLResourceStorageModeShared];
            if (n > 0) std::memcpy([buf contents], src, n * elem);
            return std::make_unique<MetalResidentColumn>(buf, n, dt, sort_ctx_);
        }
    }

    // Legacy ops have no NULL semantics: refuse a NULL-bearing column (only
    // upload_pair_exact makes one). groupby_exact_resident uses the
    // _nullable check.
    static const MetalResidentColumn& check_i64(const ResidentColumn& c) {
        const auto& r = check_i64_nullable(c);
        if (r.null_count() != 0)
            throw std::runtime_error(
                "ResidentColumn carries NULL rows (uploaded by gpu_upload_pair_exact) — "
                "only the exact GROUP BY (gpu_groupby_exact_resident) accepts it");
        return r;
    }
    static const MetalResidentColumn& check_i64_nullable(const ResidentColumn& c) {
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
    id<MTLComputePipelineState> ps_gb_having_counts_      = nil;
    id<MTLComputePipelineState> ps_gb_having_compact_     = nil;
    id<MTLComputePipelineState> ps_gb_topk_hist_          = nil;
    id<MTLComputePipelineState> ps_gb_topk_counts_        = nil;
    id<MTLComputePipelineState> ps_gb_topk_compact_       = nil;
    id<MTLComputePipelineState> ps_gbx_chunk_             = nil;
    id<MTLComputePipelineState> ps_gbx_finalize_          = nil;
    id<MTLComputePipelineState> ps_gbx_having_counts_     = nil;
    id<MTLComputePipelineState> ps_gbx_having_compact_    = nil;
    id<MTLComputePipelineState> ps_gbx_topk_hist_         = nil;
    id<MTLComputePipelineState> ps_gbx_topk_counts_       = nil;
    id<MTLComputePipelineState> ps_gbx_topk_compact_      = nil;
    id<MTLComputePipelineState> ps_gbx_mask_              = nil;
    id<MTLComputePipelineState> ps_gbxm_chunk_            = nil;
    id<MTLComputePipelineState> ps_gbxm_finalize_         = nil;
    id<MTLComputePipelineState> ps_gbx_sel_counts_        = nil;
    id<MTLComputePipelineState> ps_gbx_sel_compact_       = nil;
    id<MTLComputePipelineState> ps_jm_unique_             = nil;
    id<MTLComputePipelineState> ps_jm_probe_              = nil;
    id<MTLComputePipelineState> ps_jm_counts_             = nil;
    id<MTLComputePipelineState> ps_jm_pos_                = nil;
    id<MTLComputePipelineState> ps_jm_gather_             = nil;

    // Exact GROUP BY scratch: 5-long chunk partials, the finalized tuple
    // arrays (lo, hi, cnt, cstar, mn, mx, keys, key_null) kept on the device
    // while a filter runs, and an 8-byte dummy bound where a NULL-free
    // payload has no validity bitmap.
    id<MTLBuffer> gbx_head_buf_ = nil, gbx_tail_buf_ = nil;
    std::vector<id<MTLBuffer>> gbx_f_ = std::vector<id<MTLBuffer>>(8, nil);
    id<MTLBuffer> gbx_dummy_valid_ = nil;
    id<MTLBuffer> gbx_knull_out_ = nil;
    // §4.6 WHERE mask scratch: the mask (one byte per original row), IN
    // lists, 6-long masked partials, and the compacted sorted keys /
    // permutation of variant (b).
    id<MTLBuffer> gbx_mask_buf_ = nil, gbx_list_buf_ = nil;
    // join_materialize scratch (match row, class, destination per probe row; the uniqueness flag)
    id<MTLBuffer> jm_match_ = nil, jm_cls_ = nil, jm_pos_buf_ = nil, jm_flag_ = nil;
    id<MTLBuffer> gbxm_head_buf_ = nil, gbxm_tail_buf_ = nil;
    id<MTLBuffer> gbx_sel_keys_ = nil, gbx_sel_perm_ = nil;
    // Variant choice: compact-then-reduce when the surviving fraction of the
    // key range is below this (GPUDB_METAL_MASK_COMPACT_BELOW, default 0.5;
    // the §9.1 selectivity sweep fixes it per backend).
    double compact_below_ = [] {
        double v = 0.5;
        if (const char* e = std::getenv("GPUDB_METAL_MASK_COMPACT_BELOW")) {
            char* end = nullptr;
            const double x = std::strtod(e, &end);
            if (end && end != e && *end == '\0' && x >= 0.0 && x <= 1.0) v = x;
        }
        return v;
    }();

    // Resident GROUP BY scratch (grown on demand): per-256-block run-start
    // counts / offsets (u32), per-64-chunk head and tail partials (i64), and
    // the gathered f64 payload in sorted order.
    id<MTLBuffer> gb_block_buf_  = nil;
    id<MTLBuffer> gb_head_buf_   = nil;
    id<MTLBuffer> gb_tail_buf_   = nil;
    id<MTLBuffer> gb_gather_buf_ = nil;
    // GroupByFilter scratch: the finalized (key, sum, count) arrays stay on
    // the device when a filter is active; per-block counts for the second
    // class ("equal to the k-th") and the 256-bin radix-select histogram.
    id<MTLBuffer> gb_fkeys_      = nil;
    id<MTLBuffer> gb_fsums_      = nil;
    id<MTLBuffer> gb_fcounts_    = nil;
    id<MTLBuffer> gb_block2_buf_ = nil;
    id<MTLBuffer> gb_hist_buf_   = nil;

    // Lazily constructed: only joins pay for the radix-sort pipelines.
    std::shared_ptr<SortCtx> sort_ctx_;

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
