// metal_aggregator.mm — Apple Silicon Metal backend.
//
// macOS-only file. Do NOT include from the Linux build.
//
// Week-1 status: SCAFFOLD ONLY. Returns CPU-equivalent results (host-side
// reduction) so tests pass on macOS today; the actual Metal compute pipeline
// is to be implemented by the macOS Claude Code instance in `feat/metal-*`
// branches. The TODOs below are the implementation plan.

#include "gpu_backend.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <chrono>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace gpudb {

namespace {

// TODO(metal): replace with a real MTLComputePipelineState built from
//   src/backends/metal/kernels/sum.metal once that file exists.
//   Use threadgroup memory + simdgroup reductions (simd_sum, simd_min, simd_max)
//   for the per-tile reduction; second-pass kernel reduces partials.
//
// TODO(metal): zero-copy by wrapping host buffers in MTLBuffer with
//   MTLResourceStorageModeShared (UMA wins are real here).
//
// TODO(metal): instrument with MTLCounterSampleBuffer for kernel ms separate
//   from wall ms; transfer_ms = 0 by definition on UMA.

class MetalAggregator final : public Aggregator {
public:
    MetalAggregator() {
        device_ = MTLCreateSystemDefaultDevice();
        if (!device_) {
            throw std::runtime_error("MTLCreateSystemDefaultDevice returned nil");
        }
        queue_ = [device_ newCommandQueue];
        if (!queue_) {
            throw std::runtime_error("Failed to create Metal command queue");
        }
    }

    ~MetalAggregator() override = default;  // ARC handles release

    Backend backend() const noexcept override { return Backend::METAL; }

    std::string device_name() const override {
        @autoreleasepool {
            NSString* name = [device_ name];
            std::ostringstream os;
            os << [name UTF8String] << " (Metal, scaffold)";
            return os.str();
        }
    }

    AggResult sum_i64(const std::int64_t* data, std::size_t n) override {
        return cpu_fallback_i64(data, n, [](std::int64_t a, std::int64_t b){ return a + b; }, 0);
    }
    AggResult min_i64(const std::int64_t* data, std::size_t n) override {
        return cpu_fallback_i64(data, n,
            [](std::int64_t a, std::int64_t b){ return a < b ? a : b; },
            std::numeric_limits<std::int64_t>::max());
    }
    AggResult max_i64(const std::int64_t* data, std::size_t n) override {
        return cpu_fallback_i64(data, n,
            [](std::int64_t a, std::int64_t b){ return a > b ? a : b; },
            std::numeric_limits<std::int64_t>::min());
    }
    AggResult sum_f64(const double* data, std::size_t n) override {
        const auto t0 = std::chrono::steady_clock::now();
        double acc = 0.0;
        for (std::size_t i = 0; i < n; ++i) acc += data[i];
        const auto t1 = std::chrono::steady_clock::now();
        AggResult r{};
        r.value_f64   = acc;
        r.rows        = n;
        r.wall_ms     = std::chrono::duration<double, std::milli>(t1 - t0).count();
        r.kernel_ms   = 0.0;
        r.transfer_ms = 0.0;  // UMA: no transfer cost
        return r;
    }

private:
    template <typename Op>
    AggResult cpu_fallback_i64(const std::int64_t* data, std::size_t n, Op op, std::int64_t init) {
        const auto t0 = std::chrono::steady_clock::now();
        std::int64_t acc = (n == 0) ? 0 : init;
        for (std::size_t i = 0; i < n; ++i) acc = op(acc, data[i]);
        const auto t1 = std::chrono::steady_clock::now();
        AggResult r{};
        r.value_i64   = acc;
        r.rows        = n;
        r.wall_ms     = std::chrono::duration<double, std::milli>(t1 - t0).count();
        r.kernel_ms   = 0.0;
        r.transfer_ms = 0.0;
        return r;
    }

    id<MTLDevice>       device_ = nil;
    id<MTLCommandQueue> queue_  = nil;
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
