// metal_radix_sort.hpp — shared LSD radix sort for (int64 key, int64 payload) pairs.
// Used by Metal hash-join (and available for future window/join operators).
#pragma once

#import <Metal/Metal.h>

#include <cstddef>
#include <cstdint>

namespace gpudb::metal_detail {

//! Stable LSD radix sort of int64 keys with int64 payloads (UMA buffers).
class MetalRadixSort {
public:
    MetalRadixSort(id<MTLDevice> device, id<MTLCommandQueue> queue);

    //! Sort n pairs. `keys_out` / `payloads_out` may alias `keys` / `payloads`.
    //! Returns GPU kernel time in milliseconds (radix passes only).
    double sort_host(const std::int64_t* keys, const std::int64_t* payloads, std::uint32_t n,
                     std::int64_t* keys_out, std::int64_t* payloads_out);

    //! Sort on GPU; sorted data stays in UMA buffers until the next sort call.
    struct DeviceView {
        id<MTLBuffer> keys = nil;
        id<MTLBuffer> payloads = nil;
        double kernel_ms = 0.0;
    };
    DeviceView sort_device(const std::int64_t* keys, const std::int64_t* payloads, std::uint32_t n);

    //! The resident sort cache at storage width (docs/RESIDENT_COLUMNS_DESIGN.md §6).
    //! `keys` is a lane of `n` values at `in_width` bytes each (1, 2, 4 or 8, signed),
    //! widened into the sorter's staging by the same per-thread copy that stages an
    //! i64 lane; `payloads` is the row id per value (nullptr = the index 0..n-1, which
    //! is generated in place — no host index array). The result is the sorted keys at
    //! `out_width` and the permutation as u32 row ids, two fresh buffers that BELONG TO
    //! THE CALLER. Staging above `keep_bytes` per buffer is released afterwards (a
    //! 300M-row sort would otherwise leave gigabytes parked here, outside any budget).
    struct CacheView {
        id<MTLBuffer> keys = nil;
        id<MTLBuffer> perm = nil;
        double kernel_ms = 0.0;
    };
    CacheView sort_cache(const void* keys, unsigned in_width,
                         const std::int64_t* payloads, std::uint32_t n, unsigned out_width,
                         std::size_t keep_bytes = std::size_t(256) << 20);

private:
    double run_sort_on_staged(std::uint32_t n, __strong id<MTLBuffer>& in_keys, __strong id<MTLBuffer>& in_vals);

    void ensure_sort_buffers(std::uint32_t n, std::uint32_t num_blocks);

    void ensure_library();
    id<MTLComputePipelineState> make_pso(NSString* name);

    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    id<MTLLibrary> lib_ = nil;

    id<MTLComputePipelineState> ps_radix_flip_ = nil;
    id<MTLComputePipelineState> ps_radix_histogram_ = nil;
    id<MTLComputePipelineState> ps_radix_scatter_ = nil;
    id<MTLComputePipelineState> ps_radix_bucket_totals_ = nil;
    id<MTLComputePipelineState> ps_radix_bucket_offsets_ = nil;
    id<MTLComputePipelineState> ps_radix_per_bucket_scan_ = nil;
    id<MTLComputePipelineState> ps_radix_minmax_ = nil;
    id<MTLComputePipelineState> ps_radix_pack_cache_ = nil;

    id<MTLBuffer> b_keys_a_ = nil;
    id<MTLBuffer> b_keys_b_ = nil;
    id<MTLBuffer> b_vals_a_ = nil;
    id<MTLBuffer> b_vals_b_ = nil;
    id<MTLBuffer> b_hist_ = nil;
    id<MTLBuffer> b_scan_ = nil;
    id<MTLBuffer> b_bucket_total_ = nil;
    id<MTLBuffer> b_bucket_offset_ = nil;
    id<MTLBuffer> b_minmax_partials_ = nil;
};

} // namespace gpudb::metal_detail
