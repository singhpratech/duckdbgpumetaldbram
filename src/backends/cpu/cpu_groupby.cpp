// cpu_groupby.cpp — reference CPU GROUP BY using std::unordered_map.
//
// Single-threaded for week-2 baseline. OpenMP parallelization is non-trivial
// (per-thread maps + final merge) and is left for follow-up.

#include "gpu_backend.hpp"

#include <chrono>
#include <cstring>
#include <unordered_map>

namespace gpudb {
namespace {

class CpuGroupByAggregator final : public GroupByAggregator {
public:
    Backend backend() const noexcept override { return Backend::CPU; }
    std::string device_name() const override { return "CPU (std::unordered_map)"; }

    GroupByResult groupby_sum_i64(const std::int64_t* keys,
                                  const std::int64_t* values,
                                  std::size_t n,
                                  std::size_t expected_groups) override {
        const auto t0 = std::chrono::steady_clock::now();

        std::unordered_map<std::int64_t, std::int64_t> m;
        if (expected_groups > 0) m.reserve(expected_groups);
        for (std::size_t i = 0; i < n; ++i) {
            m[keys[i]] += values[i];
        }

        GroupByResult r;
        r.input_rows = n;
        r.keys.reserve(m.size());
        r.sums.reserve(m.size());
        for (const auto& [k, v] : m) {
            r.keys.push_back(k);
            r.sums.push_back(v);
        }
        r.wall_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
        return r;
    }
};

} // namespace

std::unique_ptr<GroupByAggregator> make_cpu_groupby_aggregator() {
    return std::make_unique<CpuGroupByAggregator>();
}

} // namespace gpudb
