// cpu_hashjoin.cpp — CPU reference for inner equi-join on int64 keys.
//
// Single-threaded std::unordered_map reference. Mirrors the cpu_groupby.cpp
// style. v1 contract: build side has UNIQUE keys; for each probe[i] we
// emit the FIRST matching build[j]. Probe rows with no match are skipped
// (no left-join semantics here).
//
// Parallelization is intentionally deferred: this is the correctness baseline
// every other backend (CUDA, Metal) is verified against. Adding OpenMP here
// would require either thread-local result vectors + concat (cheap but more
// code) or a concurrent emit (tricky). The bench measures GPU vs this — if
// CPU needs to win on this op, that's a separate PR.
//
// Behavioural notes:
//   * If n_build == 0, every probe row goes unmatched → empty result.
//   * If n_probe == 0, result is empty (zero matched).
//   * Duplicate build keys: only the FIRST seen is kept (stable in the
//     iteration order of the input). Callers that need 1:N joins must
//     pre-deduplicate or wait for v2.

#include "gpu_backend.hpp"

#include <chrono>
#include <cstdint>
#include <string>
#include <unordered_map>

namespace gpudb {
namespace {

class CpuHashJoinProbe final : public HashJoinProbe {
public:
    Backend backend() const noexcept override { return Backend::CPU; }
    std::string device_name() const override {
        return "CPU (single-threaded std::unordered_map hash join)";
    }

    JoinResult inner_join_i64(const std::int64_t* build_keys, std::size_t n_build,
                              const std::int64_t* probe_keys, std::size_t n_probe) override {
        const auto t0 = std::chrono::steady_clock::now();

        JoinResult r;
        r.rows_build = n_build;
        r.rows_probe = n_probe;

        // Build phase: key -> first-seen build-side row index.
        std::unordered_map<std::int64_t, std::int64_t> table;
        table.reserve(n_build);
        for (std::size_t j = 0; j < n_build; ++j) {
            // emplace keeps the FIRST value if the key already exists,
            // matching the v1 "first matching build[j]" contract.
            table.emplace(build_keys[j], static_cast<std::int64_t>(j));
        }

        // Probe phase: emit (probe_idx, build_idx) for each match.
        // Reserve a guess of n_probe (upper bound when every probe matches).
        r.probe_indices.reserve(n_probe);
        r.build_indices.reserve(n_probe);
        for (std::size_t i = 0; i < n_probe; ++i) {
            auto it = table.find(probe_keys[i]);
            if (it == table.end()) continue;       // no match → skip silently
            r.probe_indices.push_back(static_cast<std::int64_t>(i));
            r.build_indices.push_back(it->second);
        }
        r.matched = r.probe_indices.size();

        r.kernel_ms   = 0.0;
        r.transfer_ms = 0.0;
        r.wall_ms     = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0).count();
        return r;
    }
};

} // namespace

std::unique_ptr<HashJoinProbe> make_cpu_hashjoin_probe() {
    return std::make_unique<CpuHashJoinProbe>();
}

} // namespace gpudb
