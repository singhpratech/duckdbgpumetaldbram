// cpu_hashjoin.cpp — CPU reference inner equi-join on i64 keys.
//
// Single-threaded baseline using std::unordered_multimap. Build phase
// inserts every (build_key, build_row_idx) pair into the multimap; probe
// phase looks up each probe key, emitting (probe_idx, build_idx) for each
// match. This is intentionally simple — it's the correctness oracle the
// CUDA path is verified against. Performance can be improved with parallel
// partition + per-partition build/probe (similar to the GROUP BY parallel
// path), but at billion-scale we expect CUDA to dominate, so keeping the
// CPU side honest and trivially correct matters more than speed here.
//
// Output ordering: for each probe row i (in original order 0..n_probe-1),
// matches are appended in multimap's bucket order. This is NOT a stable
// canonical order across STL implementations, so callers comparing
// against another backend's output must sort both side-by-side.

#include "gpu_backend.hpp"

#include <chrono>
#include <cstdint>
#include <unordered_map>

namespace gpudb {
namespace {

class CpuHashJoinAggregator final : public HashJoinAggregator {
public:
    Backend backend() const noexcept override { return Backend::CPU; }
    std::string device_name() const override {
        return "CPU (single-threaded std::unordered_multimap)";
    }

    HashJoinResult inner_join_i64(
            const std::int64_t* build_keys, std::size_t n_build,
            const std::int64_t* probe_keys, std::size_t n_probe) override {
        HashJoinResult r;
        r.input_build_rows = n_build;
        r.input_probe_rows = n_probe;

        const auto t0 = std::chrono::steady_clock::now();

        // Build phase: insert (key -> build_row_idx) into a multimap.
        // reserve() is a meaningful win on large builds — without it the
        // multimap rehashes log(n_build) times.
        std::unordered_multimap<std::int64_t, std::int64_t> table;
        table.reserve(n_build);
        for (std::size_t i = 0; i < n_build; ++i) {
            table.emplace(build_keys[i], static_cast<std::int64_t>(i));
        }

        // Probe phase: for each probe key, append every match.
        // We pre-reserve a modest amount to avoid the first few growth steps;
        // the actual output size is unknown a priori.
        r.probe_indices.reserve(n_probe);  // assume ~1 match avg as a starting point
        r.build_indices.reserve(n_probe);
        for (std::size_t i = 0; i < n_probe; ++i) {
            const auto range = table.equal_range(probe_keys[i]);
            for (auto it = range.first; it != range.second; ++it) {
                r.probe_indices.push_back(static_cast<std::int64_t>(i));
                r.build_indices.push_back(it->second);
            }
        }

        r.wall_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        return r;
    }
};

} // namespace

std::unique_ptr<HashJoinAggregator> make_cpu_hashjoin_aggregator() {
    return std::make_unique<CpuHashJoinAggregator>();
}

} // namespace gpudb
