// gpu_backend.hpp — abstract backend interface
//
// Backends implement this interface. The factory `make_aggregator(Backend)`
// returns the requested backend, or throws if unavailable.
//
// Shared file: any change that breaks the ABI must be coordinated between
// Linux (CUDA) and macOS (Metal) Claude Code instances via PR.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace gpudb {

enum class Backend : std::uint8_t {
    CPU   = 0,
    CUDA  = 1,
    METAL = 2,
};

[[nodiscard]] const char* to_string(Backend b) noexcept;

enum class Dtype : std::uint8_t { I64 = 0, F64 = 1 };

// Returned by Aggregator methods so backends can attach diagnostics
// (kernel time, transfer time, etc.) without C++ exceptions on the hot path.
struct AggResult {
    std::int64_t value_i64;     // result for integer aggregations
    double       value_f64;     // result for floating aggregations
    std::size_t  rows;          // input row count
    double       wall_ms;       // total wall time
    double       kernel_ms;     // GPU kernel time only (0 for CPU)
    double       transfer_ms;   // host<->device transfer time (0 for CPU/Metal-UMA/resident)
};

// Opaque handle to a column resident in backend memory.
// Owns the storage; destruction releases device memory.
// Created by Aggregator::upload_*; must only be used with the SAME aggregator
// that created it (enforced via backend_tag()).
class ResidentColumn {
public:
    virtual ~ResidentColumn() = default;
    [[nodiscard]] virtual Backend     backend_tag() const noexcept = 0;
    [[nodiscard]] virtual Dtype       dtype()       const noexcept = 0;
    [[nodiscard]] virtual std::size_t rows()        const noexcept = 0;
};

// Minimal aggregator surface for week 1.
// Future: GROUP BY, JOIN, WINDOW. Keep this tight; do not add ops casually.
class Aggregator {
public:
    virtual ~Aggregator() = default;

    [[nodiscard]] virtual Backend backend() const noexcept = 0;
    [[nodiscard]] virtual std::string device_name() const = 0;

    // ---- One-shot API (transfer + kernel per call) ----
    virtual AggResult sum_i64(const std::int64_t* data, std::size_t n) = 0;
    virtual AggResult min_i64(const std::int64_t* data, std::size_t n) = 0;
    virtual AggResult max_i64(const std::int64_t* data, std::size_t n) = 0;
    virtual AggResult sum_f64(const double*       data, std::size_t n) = 0;

    // ---- Resident-column API (transfer once, query many times) ----
    // upload_* materializes the data in backend memory and returns a handle.
    // *_resident operate on that handle without re-transfer.
    virtual std::unique_ptr<ResidentColumn>
        upload_i64(const std::int64_t* data, std::size_t n) = 0;
    virtual std::unique_ptr<ResidentColumn>
        upload_f64(const double* data, std::size_t n) = 0;

    virtual AggResult sum_resident_i64(const ResidentColumn&) = 0;
    virtual AggResult min_resident_i64(const ResidentColumn&) = 0;
    virtual AggResult max_resident_i64(const ResidentColumn&) = 0;
    virtual AggResult sum_resident_f64(const ResidentColumn&) = 0;
};

// Factory. Throws std::runtime_error if the requested backend wasn't compiled
// in or no compatible device is present.
std::unique_ptr<Aggregator> make_aggregator(Backend);

// =========================================================================
//  GROUP BY hash aggregate (i64 key, i64 sum value)
// =========================================================================

struct GroupByResult {
    std::vector<std::int64_t> keys;       // unique keys (order may differ across backends)
    std::vector<std::int64_t> sums;       // SUM(value) for each key
    std::size_t input_rows = 0;
    double      wall_ms     = 0.0;
    double      kernel_ms   = 0.0;
    double      transfer_ms = 0.0;
};

class GroupByAggregator {
public:
    virtual ~GroupByAggregator() = default;
    [[nodiscard]] virtual Backend backend() const noexcept = 0;
    [[nodiscard]] virtual std::string device_name() const = 0;

    // SUM(values) GROUP BY keys.
    // expected_groups is an optional hint; 0 means "guess from n".
    virtual GroupByResult groupby_sum_i64(const std::int64_t* keys,
                                          const std::int64_t* values,
                                          std::size_t n,
                                          std::size_t expected_groups = 0) = 0;
};

std::unique_ptr<GroupByAggregator> make_groupby_aggregator(Backend);

// Returns the best backend available at runtime: CUDA if compiled+device,
// else METAL if compiled+device, else CPU.
[[nodiscard]] Backend default_backend() noexcept;

// Returns the list of backends compiled into this build.
[[nodiscard]] std::vector<Backend> available_backends() noexcept;

// =========================================================================
//  Hybrid CPU/GPU planner (GOAL.md item 7)
//
// Picks CPU vs GPU per call based on:
//   - N (input cardinality / row count)
//   - expected_groups (for GROUP BY)
//   - data residency (resident-column path skips PCIe/copy)
//
// The dispatch rule is a SIMPLE deterministic threshold derived from
// BENCHMARK.md numbers — no ML, no auto-tuner. The point is "we know
// our hardware well enough to pick the right backend every time".
//
// Each hybrid call records its decision in `last_decision` so callers
// (the bench, the DuckDB extension, debug tooling) can inspect it.
// =========================================================================

// Why a particular backend was chosen for a single call.
enum class DispatchReason : std::uint8_t {
    // SUM/MIN/MAX reasons
    SmallN_CpuWins        = 0,  // n below GPU launch-overhead break-even
    Cold_BelowGpuBreakeven= 1,  // cold path; n too small for GPU to win
    Cold_AboveGpuBreakeven= 2,  // cold path; n large enough for GPU to win
    Hot_GpuAlwaysWins     = 3,  // resident column → GPU
    GpuUnavailable        = 4,  // no GPU compiled in / no device
    F64_NoGpuDoubles      = 5,  // Metal lacks IEEE-754 doubles, prefer CPU
    // GROUP BY reasons
    GroupBy_LowCard_CpuWins  = 10, // expected_groups < threshold and small n
    GroupBy_HighCard_GpuWins = 11, // expected_groups >= n / 10 → high cardinality regime
    GroupBy_Borderline_GpuTry= 12, // mid regime: try GPU but flag for tuning
    GroupBy_HugeN_CpuWins    = 13, // n above bitonic O(N log²N) crossover
};

[[nodiscard]] const char* to_string(DispatchReason r) noexcept;

struct DispatchDecision {
    Backend        chosen   = Backend::CPU;
    DispatchReason reason   = DispatchReason::SmallN_CpuWins;
    std::size_t    n        = 0;
    std::size_t    expected_groups = 0; // 0 = N/A or unknown
    bool           was_resident    = false;
    bool           borderline      = false; // true if a "future tuning may flip" call
};

// Hybrid scalar aggregator (SUM/MIN/MAX). Wraps CPU + GPU implementations
// and dispatches per call. Falls back to CPU if no GPU is available; in
// that case `chosen == CPU` with reason `GpuUnavailable`.
class HybridAggregator : public Aggregator {
public:
    // Inspect the dispatch decision from the most recent call. Thread-unsafe
    // by design (these aggregators are per-thread; the bench reads it after
    // each call). For SUM-class ops only.
    [[nodiscard]] virtual const DispatchDecision& last_decision() const noexcept = 0;

    // For tests / introspection: the actual GPU backend the hybrid wraps,
    // or CPU if no GPU was available at construction time.
    [[nodiscard]] virtual Backend gpu_backend() const noexcept = 0;
};

class HybridGroupByAggregator : public GroupByAggregator {
public:
    [[nodiscard]] virtual const DispatchDecision& last_decision() const noexcept = 0;
    [[nodiscard]] virtual Backend gpu_backend() const noexcept = 0;
};

// Factories. Each internally constructs a CPU aggregator AND a GPU
// aggregator (GPU = default_backend() if available, else CPU=fallback).
std::unique_ptr<HybridAggregator>        make_hybrid_aggregator();
std::unique_ptr<HybridGroupByAggregator> make_hybrid_groupby_aggregator();

} // namespace gpudb
