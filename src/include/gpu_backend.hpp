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

// Returned by Aggregator::agg_all_i64 — fused SUM + MIN + MAX + COUNT in
// a single pass over the input. The big win is that each int64 is read
// from memory exactly ONCE, halving (or quartering) DRAM traffic vs the
// "call sum/min/max separately" pattern.
struct AggAllResult {
    std::int64_t sum;           // SUM(values)
    std::int64_t min;           // MIN(values), int64 max if rows == 0
    std::int64_t max;           // MAX(values), int64 min if rows == 0
    std::size_t  count;         // number of values read (== rows for non-null)
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

    // ---- Multi-aggregate fusion (SUM + MIN + MAX + COUNT in one pass) ----
    // Each int64 is read from memory ONCE; backends compute all four
    // aggregates simultaneously. Wins over calling sum/min/max separately
    // by 2-3x on memory-bandwidth-bound work, and proportionally more
    // when more aggregates are fused.
    virtual AggAllResult agg_all_i64(const std::int64_t* data, std::size_t n) = 0;
    virtual AggAllResult agg_all_resident_i64(const ResidentColumn&) = 0;
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

// =========================================================================
//  HASH JOIN probe (i64 keys, inner equi-join)
// =========================================================================
//
// v1 contract: inner equi-join on int64 keys, build side has UNIQUE keys
// (no duplicate handling — for each probe[i] we emit the FIRST matching
// build[j], which is unambiguous when build is unique).
//
// Backend implementation strategy varies:
//   - CPU:   std::unordered_map<int64,int64> on the build side.
//   - CUDA:  open-addressing hash table with atomicCAS<int64> (in flight on
//            feat/cuda-hashjoin).
//   - Metal: SORT-MERGE join (Apple Silicon GPUs lack 64-bit atomic CAS, so
//            the CUDA approach cannot be mirrored — see
//            src/backends/metal/metal_hashjoin.mm header for details).
struct JoinResult {
    std::vector<std::int64_t> probe_indices;   // matched probe-side row indices
    std::vector<std::int64_t> build_indices;   // matched build-side row indices
    std::size_t  rows_probe = 0;
    std::size_t  rows_build = 0;
    std::size_t  matched    = 0;
    double       wall_ms     = 0.0;
    double       kernel_ms   = 0.0;
    double       transfer_ms = 0.0;
};

class HashJoinProbe {
public:
    virtual ~HashJoinProbe() = default;
    [[nodiscard]] virtual Backend backend() const noexcept = 0;
    [[nodiscard]] virtual std::string device_name() const = 0;
    // Inner equi-join: for each probe[i], find the FIRST matching build[j].
    // (No build-side duplicate handling for v1.)
    virtual JoinResult inner_join_i64(
        const std::int64_t* build_keys, std::size_t n_build,
        const std::int64_t* probe_keys, std::size_t n_probe) = 0;
};

std::unique_ptr<HashJoinProbe> make_hashjoin_probe(Backend);

// Returns the best backend available at runtime: CUDA if compiled+device,
// else METAL if compiled+device, else CPU.
[[nodiscard]] Backend default_backend() noexcept;

// Returns the list of backends compiled into this build.
[[nodiscard]] std::vector<Backend> available_backends() noexcept;

} // namespace gpudb
