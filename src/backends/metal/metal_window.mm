// metal_window.mm — Metal scaffold for window functions.
//
// CURRENT STATE: This file delegates ROW_NUMBER to the CPU reference
// implementation. The MetalWindowAggregator reports backend()==METAL but
// is functionally a thin pass-through. That is a deliberate v1 shape: it
// pins the interface, the factory wiring, and the bench harness so the
// real Metal kernel can land as a focused, reviewable follow-up patch
// without having to also stand up plumbing.
//
// =========================================================================
//  Planned real Metal algorithm (gated on PR #5 — `feat/metal-radix-sort`)
// =========================================================================
//
// ROW_NUMBER over an int64 ASC ordering reduces to "stable sort the input
// rows by key, then write 1..N as ranks back to original positions". On a
// GPU that's two well-known kernels:
//
//   Step 1. Stable radix sort over (key, original_index) pairs.
//
//     PR #5 (`feat/metal-radix-sort`) introduces an LSD radix sort over
//     int64 keys with a sibling payload buffer for groupby_sum. The same
//     kernel (in groupby.metal) accepts a `payload[]` parameter that
//     follows the key permutation. For ROW_NUMBER we reuse it with
//     `payload[i] = i` (the original index, written as int64 so it fits
//     the existing int64-payload kernel without a templated copy).
//
//     The radix sort is stable by construction (each pass scatters in
//     bucket order, preserving relative order within a bucket), which
//     matches the CPU std::stable_sort tie-break exactly.
//
//   Step 2. Scatter rank back to the original positions.
//
//     One trivial kernel:
//
//         kernel void row_number_scatter_i64(
//             device const int64_t* original_index [[buffer(0)]],
//             device int64_t*       output         [[buffer(1)]],
//             uint                  gid            [[thread_position_in_grid]],
//             constant uint&        n              [[buffer(2)]])
//         {
//             if (gid >= n) return;
//             // After radix sort, original_index[p] holds the pre-sort
//             // index of the row that ended up at sorted position p.
//             // ROW_NUMBER is 1-indexed.
//             output[ original_index[gid] ] = (int64_t)(gid + 1);
//         }
//
//     Dispatch: ceil(n/256) threadgroups of 256 threads. Both buffers are
//     MTLResourceStorageModeShared (UMA), so output is host-visible with
//     no D2H copy.
//
// Memory traffic:
//   - Radix sort dominates: 8 passes × (load 2N int64 + store 2N int64) =
//     32N int64 read+write across the 8-bit-bucket passes.
//   - Scatter: N reads + N random writes. Random writes hit the L2 hash
//     not the HBM page line, but for N up to a few hundred million on
//     LPDDR5X-class memory the kernel cost is sub-millisecond.
//
// Expected perf:
//   The same kernel body that gives groupby_sum its win at the
//   1M-rows × 1M-groups sweet spot also wins ROW_NUMBER there, because
//   both are bounded by the sort, not by what comes after. For very low
//   cardinality the sort is overkill (the rank assignment is trivial)
//   and CPU std::stable_sort wins; the planner should keep that in mind.
//
// =========================================================================
//  What unblocks the real implementation
// =========================================================================
//
// PR #5 (`feat/metal-radix-sort`) must merge to main first. That PR adds
// `groupby.metal` kernels for radix histogram + scan + scatter that this
// file will reuse. Once it's in:
//   1. Add a `row_number_scatter_i64` kernel to a new
//      `src/backends/metal/kernels/window.metal` (or extend groupby.metal).
//   2. Replace the `cpu_->row_number_i64(...)` call below with the radix
//      sort + scatter pipeline. Time the GPU command buffer for kernel_ms.
//   3. Update BENCHMARK.md with the Metal vs CPU numbers.
//
// Until then this delegate keeps the public surface honest: backend()
// returns METAL, device_name() says exactly what's happening, and the
// bench will report identical wall times for CPU and Metal — which is
// the truthful state of the world right now.

#include "gpu_backend.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace gpudb {

// Forward decl from cpu_window.cpp; the Metal stub delegates to it.
std::unique_ptr<WindowAggregator> make_cpu_window_aggregator();

namespace {

class MetalWindowAggregator final : public WindowAggregator {
public:
    MetalWindowAggregator() {
        @autoreleasepool {
            device_ = MTLCreateSystemDefaultDevice();
            if (!device_) {
                throw std::runtime_error("MTLCreateSystemDefaultDevice returned nil");
            }
        }
        cpu_ = make_cpu_window_aggregator();
    }

    Backend backend() const noexcept override { return Backend::METAL; }

    std::string device_name() const override {
        @autoreleasepool {
            std::ostringstream os;
            os << [[device_ name] UTF8String]
               << " (Metal scaffold — ROW_NUMBER delegates to CPU; "
                  "real radix-sort + scatter kernel gated on PR #5)";
            return os.str();
        }
    }

    WindowResult row_number_i64(const std::int64_t* keys, std::size_t n) override {
        // Time the call ourselves so wall_ms is consistent with what a
        // future real kernel would report (otherwise we'd inherit the
        // CPU's wall_ms, which is fine but slightly understates wrapper
        // overhead).
        const auto t_wall0 = std::chrono::steady_clock::now();
        WindowResult r = cpu_->row_number_i64(keys, n);
        r.kernel_ms   = 0.0;     // No GPU kernel ran. Honest reporting.
        r.transfer_ms = 0.0;     // UMA on Apple Silicon — would be 0 anyway.
        r.wall_ms     = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t_wall0).count();
        return r;
    }

private:
    id<MTLDevice> device_ = nil;
    std::unique_ptr<WindowAggregator> cpu_;
};

} // namespace

std::unique_ptr<WindowAggregator> make_metal_window_aggregator() {
    return std::make_unique<MetalWindowAggregator>();
}

} // namespace gpudb
