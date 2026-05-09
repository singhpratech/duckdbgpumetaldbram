// cpu_groupby.cpp — CPU GROUP BY using std::unordered_map.
//
// Two implementations:
//   - Single-threaded baseline (kept for clarity/debug, used at very small n)
//   - Parallel: each OpenMP thread builds a private map over its chunk;
//     final merge reduces all maps into one. Standard "private-then-merge"
//     pattern; scales well when groups <<< rows because per-thread maps stay
//     small. For very high cardinality (groups ~ rows), merge cost can
//     dominate — that's the regime where GPU still wins decisively.

#include "gpu_backend.hpp"

#include <chrono>
#include <cstring>
#include <thread>
#include <unordered_map>
#include <vector>

#if GPUDB_HAVE_OPENMP
#include <omp.h>
#endif

namespace gpudb {
namespace {

constexpr std::size_t kSerialThreshold = 1u << 14;  // 16K rows: parallel is overhead

GroupByResult run_serial(const std::int64_t* keys, const std::int64_t* values,
                         std::size_t n, std::size_t expected_groups) {
    const auto t0 = std::chrono::steady_clock::now();
    std::unordered_map<std::int64_t, std::int64_t> m;
    if (expected_groups > 0) m.reserve(expected_groups);
    for (std::size_t i = 0; i < n; ++i) m[keys[i]] += values[i];

    GroupByResult r;
    r.input_rows = n;
    r.keys.reserve(m.size());
    r.sums.reserve(m.size());
    for (const auto& [k, v] : m) { r.keys.push_back(k); r.sums.push_back(v); }
    r.wall_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    return r;
}

#if GPUDB_HAVE_OPENMP
GroupByResult run_parallel(const std::int64_t* keys, const std::int64_t* values,
                           std::size_t n, std::size_t expected_groups) {
    const auto t0 = std::chrono::steady_clock::now();

    const int nthreads = omp_get_max_threads();
    std::vector<std::unordered_map<std::int64_t, std::int64_t>> partials(nthreads);
    if (expected_groups > 0) {
        for (auto& m : partials) m.reserve(expected_groups);
    }

    // Phase 1: each thread aggregates its slice into a private map.
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        auto& local = partials[tid];
        #pragma omp for schedule(static) nowait
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
            local[keys[i]] += values[i];
        }
    }

    // Phase 2: merge into the largest partial (avoids re-hashing the biggest set).
    std::size_t biggest = 0;
    for (int t = 1; t < nthreads; ++t) {
        if (partials[t].size() > partials[biggest].size()) biggest = t;
    }
    auto& dst = partials[biggest];
    for (int t = 0; t < nthreads; ++t) {
        if (t == static_cast<int>(biggest)) continue;
        for (const auto& [k, v] : partials[t]) dst[k] += v;
        partials[t].clear();
    }

    GroupByResult r;
    r.input_rows = n;
    r.keys.reserve(dst.size());
    r.sums.reserve(dst.size());
    for (const auto& [k, v] : dst) { r.keys.push_back(k); r.sums.push_back(v); }
    r.wall_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    return r;
}
#endif

class CpuGroupByAggregator final : public GroupByAggregator {
public:
    Backend backend() const noexcept override { return Backend::CPU; }
    std::string device_name() const override {
#if GPUDB_HAVE_OPENMP
        return "CPU (OpenMP, " + std::to_string(omp_get_max_threads()) +
               " threads, per-thread map + merge)";
#else
        return "CPU (single-threaded std::unordered_map)";
#endif
    }

    GroupByResult groupby_sum_i64(const std::int64_t* keys,
                                  const std::int64_t* values,
                                  std::size_t n,
                                  std::size_t expected_groups) override {
#if GPUDB_HAVE_OPENMP
        // Cardinality-aware switch:
        //
        // The per-thread-map + merge pattern wins decisively at LOW cardinality
        // (groups << rows), where each per-thread map stays small and the merge
        // is cheap. At HIGH cardinality (groups close to rows), per-thread maps
        // approach the full output size, the merge becomes O(threads * groups)
        // hash insertions, and we lose vs single-threaded by 2-5x.
        //
        // Heuristic:
        //   - If caller hints expected_groups <= n / 50 → parallel
        //   - If caller hints higher cardinality → serial
        //   - If unknown (expected_groups == 0) → serial (safer default)
        //
        // For low-cardinality workloads the user wants the parallel speedup;
        // for high-cardinality they should pass the hint (or expect serial).
        // The "right" fix for high-cardinality CPU GROUP BY is hash-partitioning
        // the input or a concurrent hash table — out of scope for week 2.
        const bool low_cardinality = expected_groups > 0
            && expected_groups <= n / 50
            && n >= kSerialThreshold;
        if (low_cardinality && omp_get_max_threads() > 1) {
            return run_parallel(keys, values, n, expected_groups);
        }
#endif
        return run_serial(keys, values, n, expected_groups);
    }
};

} // namespace

std::unique_ptr<GroupByAggregator> make_cpu_groupby_aggregator() {
    return std::make_unique<CpuGroupByAggregator>();
}

} // namespace gpudb
