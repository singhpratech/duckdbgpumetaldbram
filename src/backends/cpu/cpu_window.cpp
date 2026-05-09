// cpu_window.cpp — CPU window-function reference implementation.
//
// v1 scope: ROW_NUMBER() OVER (ORDER BY key ASC).
//
// Algorithm:
//   1. Build a vector of (key, original_index) pairs.
//   2. std::stable_sort by key. Stability gives the tie-break we want
//      (earlier input rows get the smaller rank when keys collide), and
//      matches the GPU plan: a stable radix sort over (key, original_index)
//      pairs leaves equal-keyed rows in their input order.
//   3. Walk the sorted pairs; row at sort position p (0-indexed) gets
//      rank p+1 written back to output[original_index].
//
// Complexity: O(N log N) for the sort; O(N) for the scatter. The whole
// thing is the reference the Metal/CUDA backends must match bit-for-bit,
// so it stays simple and single-threaded — correctness over speed.

#include "gpu_backend.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <utility>
#include <vector>

namespace gpudb {

namespace {

class CpuWindowAggregator final : public WindowAggregator {
public:
    Backend backend() const noexcept override { return Backend::CPU; }

    std::string device_name() const override {
        return "CPU (std::stable_sort reference)";
    }

    WindowResult row_number_i64(const std::int64_t* keys, std::size_t n) override {
        WindowResult r;
        r.rows = n;
        r.output.assign(n, 0);
        if (n == 0) return r;

        const auto t0 = std::chrono::steady_clock::now();

        // (key, original_index) pairs.
        std::vector<std::pair<std::int64_t, std::int64_t>> pairs;
        pairs.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            pairs.emplace_back(keys[i], static_cast<std::int64_t>(i));
        }

        // Stable sort by key. Ties preserve original index order.
        std::stable_sort(pairs.begin(), pairs.end(),
                         [](const auto& a, const auto& b) { return a.first < b.first; });

        // Scatter rank back to original positions. rank is 1-indexed.
        for (std::size_t p = 0; p < n; ++p) {
            const std::int64_t original_index = pairs[p].second;
            r.output[static_cast<std::size_t>(original_index)] =
                static_cast<std::int64_t>(p + 1);
        }

        r.wall_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        r.kernel_ms   = 0.0;
        r.transfer_ms = 0.0;
        return r;
    }
};

} // namespace

std::unique_ptr<WindowAggregator> make_cpu_window_aggregator() {
    return std::make_unique<CpuWindowAggregator>();
}

} // namespace gpudb
