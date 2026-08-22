// metal_radix_sort.mm — LSD radix sort for (int64, int64) pairs on Apple Silicon.

#include "metal_radix_sort.hpp"
#include "metal_kernel_sources.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace gpudb::metal_detail {

namespace {

constexpr NSUInteger kBlock = 256;

[[noreturn]] void metal_throw(const char* what, NSError* err) {
    std::ostringstream os;
    os << "Metal radix " << what;
    if (err) os << ": " << [[err localizedDescription] UTF8String];
    throw std::runtime_error(os.str());
}

} // namespace

MetalRadixSort::MetalRadixSort(id<MTLDevice> device, id<MTLCommandQueue> queue)
    : device_(device), queue_(queue) {
    ensure_library();
}

void MetalRadixSort::ensure_library() {
    if (lib_) return;
    @autoreleasepool {
        NSError* err = nil;
        NSString* src = [NSString stringWithUTF8String:metal::kGroupByKernelSource];
        MTLCompileOptions* opts = [MTLCompileOptions new];
        if (@available(macOS 15.0, *)) {
            [opts setLanguageVersion:MTLLanguageVersion3_2];
        } else {
            [opts setLanguageVersion:MTLLanguageVersion3_1];
        }
        lib_ = [device_ newLibraryWithSource:src options:opts error:&err];
        if (!lib_) metal_throw("compile groupby.metal", err);

        ps_radix_flip_ = make_pso(@"radix_flip_sign_bit");
        ps_radix_histogram_ = make_pso(@"radix_histogram");
        ps_radix_scatter_ = make_pso(@"radix_scatter");
        ps_radix_bucket_totals_ = make_pso(@"radix_bucket_totals");
        ps_radix_bucket_offsets_ = make_pso(@"radix_bucket_offsets");
        ps_radix_per_bucket_scan_ = make_pso(@"radix_per_bucket_scan");
        ps_radix_minmax_ = make_pso(@"radix_minmax_i64");
    }
}

id<MTLComputePipelineState> MetalRadixSort::make_pso(NSString* name) {
    @autoreleasepool {
        id<MTLFunction> fn = [lib_ newFunctionWithName:name];
        if (!fn) {
            std::ostringstream os;
            os << "no function " << [name UTF8String];
            throw std::runtime_error(os.str());
        }
        NSError* err = nil;
        id<MTLComputePipelineState> pso = [device_ newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) metal_throw("newComputePipelineState", err);
        return pso;
    }
}

void MetalRadixSort::ensure_sort_buffers(std::uint32_t n, std::uint32_t num_blocks) {
    constexpr std::uint32_t RADIX_BUCKETS = 256;
    const std::size_t bytes_kv = static_cast<std::size_t>(n) * sizeof(std::int64_t);
    const std::size_t bytes_hist =
        static_cast<std::size_t>(num_blocks) * RADIX_BUCKETS * sizeof(std::uint32_t);
    const std::size_t bytes_buckets = RADIX_BUCKETS * sizeof(std::uint32_t);

    if (!b_keys_a_ || [b_keys_a_ length] < bytes_kv) {
        b_keys_a_ = [device_ newBufferWithLength:bytes_kv options:MTLResourceStorageModeShared];
        b_keys_b_ = [device_ newBufferWithLength:bytes_kv options:MTLResourceStorageModeShared];
        b_vals_a_ = [device_ newBufferWithLength:bytes_kv options:MTLResourceStorageModeShared];
        b_vals_b_ = [device_ newBufferWithLength:bytes_kv options:MTLResourceStorageModeShared];
    }
    if (!b_hist_ || [b_hist_ length] < bytes_hist) {
        b_hist_ = [device_ newBufferWithLength:bytes_hist options:MTLResourceStorageModeShared];
        b_scan_ = [device_ newBufferWithLength:bytes_hist options:MTLResourceStorageModeShared];
    }
    if (!b_bucket_total_ || [b_bucket_total_ length] < bytes_buckets) {
        b_bucket_total_ = [device_ newBufferWithLength:bytes_buckets options:MTLResourceStorageModeShared];
        b_bucket_offset_ = [device_ newBufferWithLength:bytes_buckets options:MTLResourceStorageModeShared];
    }
}

double MetalRadixSort::run_sort_on_staged(std::uint32_t n, id<MTLBuffer>& in_keys,
                                          id<MTLBuffer>& in_vals) {
    constexpr std::uint32_t RADIX_BUCKETS = 256;
    constexpr std::uint32_t RADIX_BLOCK_SIZE = 256;
    constexpr std::uint32_t RADIX_WORK_PER_BLOCK = 1024;
    const std::uint32_t num_blocks = (n + RADIX_WORK_PER_BLOCK - 1) / RADIX_WORK_PER_BLOCK;

    id<MTLBuffer> out_keys = b_keys_b_;
    id<MTLBuffer> out_vals = b_vals_b_;
    if (in_keys != b_keys_a_) {
        out_keys = b_keys_a_;
        out_vals = b_vals_a_;
    }

    constexpr NSUInteger kMinMaxGrid = 256;
    const std::size_t bytes_minmax = kMinMaxGrid * 4 * sizeof(std::int32_t);
    if (!b_minmax_partials_ || [b_minmax_partials_ length] < bytes_minmax) {
        b_minmax_partials_ = [device_ newBufferWithLength:bytes_minmax
                                                  options:MTLResourceStorageModeShared];
    }

    std::uint8_t active_bytes = 0xFF;
    {
        id<MTLCommandBuffer> cb_mm = [queue_ commandBuffer];
        id<MTLComputeCommandEncoder> ce_mm = [cb_mm computeCommandEncoder];
        [ce_mm setComputePipelineState:ps_radix_minmax_];
        [ce_mm setBuffer:in_keys offset:0 atIndex:0];
        [ce_mm setBytes:&n length:sizeof(n) atIndex:1];
        [ce_mm setBuffer:b_minmax_partials_ offset:0 atIndex:2];
        [ce_mm setBuffer:b_minmax_partials_ offset:kMinMaxGrid * 2 * sizeof(std::int32_t) atIndex:3];
        [ce_mm dispatchThreadgroups:MTLSizeMake(kMinMaxGrid, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [ce_mm endEncoding];
        [cb_mm commit];
        [cb_mm waitUntilCompleted];

        const std::int32_t* parts = static_cast<const std::int32_t*>([b_minmax_partials_ contents]);
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
        const std::uint64_t mn_u = static_cast<std::uint64_t>(mn) ^ 0x8000000000000000ULL;
        const std::uint64_t mx_u = static_cast<std::uint64_t>(mx) ^ 0x8000000000000000ULL;
        active_bytes = 0;
        for (std::uint32_t p = 0; p < 8; ++p) {
            const std::uint64_t shift = p * 8;
            const std::uint8_t mn_b = static_cast<std::uint8_t>((mn_u >> shift) & 0xFFu);
            const std::uint8_t mx_b = static_cast<std::uint8_t>((mx_u >> shift) & 0xFFu);
            if (mn_b != mx_b) active_bytes |= (1u << p);
        }
        if (active_bytes == 0) active_bytes = 0x01;
    }

    id<MTLCommandBuffer> cb = [queue_ commandBuffer];
    id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];

    [ce setComputePipelineState:ps_radix_flip_];
    [ce setBuffer:in_keys offset:0 atIndex:0];
    [ce setBytes:&n length:sizeof(n) atIndex:1];
    [ce dispatchThreadgroups:MTLSizeMake((n + kBlock - 1) / kBlock, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];

    for (std::uint32_t pass = 0; pass < 8; ++pass) {
        if (!(active_bytes & (1u << pass))) continue;
        const std::uint32_t shift = pass * 8u;

        [ce setComputePipelineState:ps_radix_histogram_];
        [ce setBuffer:in_keys offset:0 atIndex:0];
        [ce setBytes:&n length:sizeof(n) atIndex:1];
        [ce setBytes:&shift length:sizeof(shift) atIndex:2];
        [ce setBuffer:b_hist_ offset:0 atIndex:3];
        [ce dispatchThreadgroups:MTLSizeMake(num_blocks, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(RADIX_BLOCK_SIZE, 1, 1)];

        [ce setComputePipelineState:ps_radix_bucket_totals_];
        [ce setBuffer:b_hist_ offset:0 atIndex:0];
        [ce setBytes:&num_blocks length:sizeof(num_blocks) atIndex:1];
        [ce setBuffer:b_bucket_total_ offset:0 atIndex:2];
        [ce dispatchThreadgroups:MTLSizeMake(RADIX_BUCKETS, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(RADIX_BLOCK_SIZE, 1, 1)];

        [ce setComputePipelineState:ps_radix_bucket_offsets_];
        [ce setBuffer:b_bucket_total_ offset:0 atIndex:0];
        [ce setBuffer:b_bucket_offset_ offset:0 atIndex:1];
        [ce dispatchThreadgroups:MTLSizeMake(1, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(RADIX_BLOCK_SIZE, 1, 1)];

        [ce setComputePipelineState:ps_radix_per_bucket_scan_];
        [ce setBuffer:b_hist_ offset:0 atIndex:0];
        [ce setBytes:&num_blocks length:sizeof(num_blocks) atIndex:1];
        [ce setBuffer:b_bucket_offset_ offset:0 atIndex:2];
        [ce setBuffer:b_scan_ offset:0 atIndex:3];
        [ce dispatchThreadgroups:MTLSizeMake(RADIX_BUCKETS, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(RADIX_BLOCK_SIZE, 1, 1)];

        [ce setComputePipelineState:ps_radix_scatter_];
        [ce setBuffer:in_keys offset:0 atIndex:0];
        [ce setBuffer:in_vals offset:0 atIndex:1];
        [ce setBytes:&n length:sizeof(n) atIndex:2];
        [ce setBytes:&shift length:sizeof(shift) atIndex:3];
        [ce setBuffer:b_scan_ offset:0 atIndex:4];
        [ce setBuffer:out_keys offset:0 atIndex:5];
        [ce setBuffer:out_vals offset:0 atIndex:6];
        [ce dispatchThreadgroups:MTLSizeMake(num_blocks, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(RADIX_BLOCK_SIZE, 1, 1)];

        std::swap(in_keys, out_keys);
        std::swap(in_vals, out_vals);
    }

    [ce setComputePipelineState:ps_radix_flip_];
    [ce setBuffer:in_keys offset:0 atIndex:0];
    [ce setBytes:&n length:sizeof(n) atIndex:1];
    [ce dispatchThreadgroups:MTLSizeMake((n + kBlock - 1) / kBlock, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];

    [ce endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    return ([cb GPUEndTime] - [cb GPUStartTime]) * 1000.0;
}

MetalRadixSort::DeviceView MetalRadixSort::sort_device(const std::int64_t* keys,
                                                       const std::int64_t* payloads,
                                                       std::uint32_t n) {
    DeviceView view;
    if (n == 0) return view;
    if (n > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("radix sort: n too large for uint32 indexing");
    }

    constexpr std::uint32_t RADIX_WORK_PER_BLOCK = 1024;
    const std::uint32_t num_blocks = (n + RADIX_WORK_PER_BLOCK - 1) / RADIX_WORK_PER_BLOCK;
    const std::size_t bytes_kv = static_cast<std::size_t>(n) * sizeof(std::int64_t);

    @autoreleasepool {
        ensure_sort_buffers(n, num_blocks);
        std::memcpy([b_keys_a_ contents], keys, bytes_kv);
        std::memcpy([b_vals_a_ contents], payloads, bytes_kv);

        id<MTLBuffer> in_keys = b_keys_a_;
        id<MTLBuffer> in_vals = b_vals_a_;
        view.kernel_ms = run_sort_on_staged(n, in_keys, in_vals);
        view.keys = in_keys;
        view.payloads = in_vals;
    }
    return view;
}

double MetalRadixSort::sort_host(const std::int64_t* keys, const std::int64_t* payloads, std::uint32_t n,
                                 std::int64_t* keys_out, std::int64_t* payloads_out) {
    if (n == 0) return 0.0;
    const auto view = sort_device(keys, payloads, n);
    const std::size_t bytes_kv = static_cast<std::size_t>(n) * sizeof(std::int64_t);
    std::memcpy(keys_out, [view.keys contents], bytes_kv);
    std::memcpy(payloads_out, [view.payloads contents], bytes_kv);
    return view.kernel_ms;
}

} // namespace gpudb::metal_detail
