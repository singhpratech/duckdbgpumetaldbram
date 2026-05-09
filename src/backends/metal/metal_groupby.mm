// metal_groupby.mm — GROUP BY on Metal.
//
// STATUS: CPU-delegating fallback (week-1 honest implementation).
//
// Why no real Metal kernel yet:
//   The CUDA path uses an open-addressing hash table whose insertion needs
//   64-bit atomic compare-and-swap. Apple Silicon GPUs (Apple7+/M-series)
//   ship 64-bit atomic_fetch_add/sub/min/max under the int64Atomics feature
//   set, but NOT 64-bit atomic_compare_exchange. The MSL compiler rejects
//   `atomic_compare_exchange_weak_explicit(device atomic_ulong*, ...)`.
//
// Real Metal options for follow-up:
//   1) Sort-then-segment-reduce: parallel radix sort on (key,value) pairs,
//      then a segmented reduction. Standard GPU GROUP BY pattern, no CAS
//      needed. Significant work — likely a separate `feat/metal-groupby-sort`.
//   2) Two-phase claim with 32-bit CAS as slot owner + non-atomic key write
//      ordered by memory fences. Subtle and easy to get wrong; deferred.
//   3) MPSGraph / MLX hash join / groupby (if/when available in the version
//      shipping on the target macOS).
//
// Until one of those lands, MetalGroupByAggregator returns Backend::METAL
// (so callers can dispatch by interface) but performs the reduction on the
// CPU. Performance therefore matches the CPU baseline.

#include "gpu_backend.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <chrono>
#include <sstream>
#include <unordered_map>

namespace gpudb {

namespace {

class MetalGroupByAggregator final : public GroupByAggregator {
public:
    MetalGroupByAggregator() {
        @autoreleasepool {
            device_ = MTLCreateSystemDefaultDevice();
            if (!device_) throw std::runtime_error("MTLCreateSystemDefaultDevice returned nil");
        }
    }

    Backend backend() const noexcept override { return Backend::METAL; }

    std::string device_name() const override {
        @autoreleasepool {
            std::ostringstream os;
            os << [[device_ name] UTF8String]
               << " (Metal — GROUP BY on CPU; needs 64-bit atomic CAS)";
            return os.str();
        }
    }

    GroupByResult groupby_sum_i64(const std::int64_t* keys,
                                  const std::int64_t* values,
                                  std::size_t n,
                                  std::size_t expected_groups) override {
        const auto t0 = std::chrono::steady_clock::now();

        std::unordered_map<std::int64_t, std::int64_t> m;
        if (expected_groups > 0) m.reserve(expected_groups);
        for (std::size_t i = 0; i < n; ++i) m[keys[i]] += values[i];

        GroupByResult r;
        r.input_rows = n;
        r.keys.reserve(m.size());
        r.sums.reserve(m.size());
        for (const auto& [k, v] : m) {
            r.keys.push_back(k);
            r.sums.push_back(v);
        }
        r.kernel_ms   = 0.0;          // no GPU work
        r.transfer_ms = 0.0;          // UMA, and we never touched GPU memory
        r.wall_ms     = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0).count();
        return r;
    }

private:
    id<MTLDevice> device_ = nil;
};

} // namespace

std::unique_ptr<GroupByAggregator> make_metal_groupby_aggregator() {
    return std::make_unique<MetalGroupByAggregator>();
}

} // namespace gpudb
