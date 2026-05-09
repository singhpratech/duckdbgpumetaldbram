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

// =========================================================================
//  Hash join (inner equi-join on i64 keys)
// =========================================================================
//
// Build side: small/medium relation whose keys we materialize into a hash
// table. Probe side: large relation whose keys we look up. Output is the
// list of matching (probe_row_idx, build_row_idx) pairs. Inner join only,
// integer-key only — sufficient for the canonical TPC-H Q3/Q5/Q10 join
// shape (lineitem ⋈ orders on l_orderkey = o_orderkey).
//
// Like our GROUP BY, the device side uses an open-addressing hash table
// where each slot holds (key, build_row_idx). Empty sentinel = INT64_MIN
// in the key field — caller must not pass INT64_MIN as a build key.

struct HashJoinResult {
    std::vector<std::int64_t> probe_indices;   // row index in probe input
    std::vector<std::int64_t> build_indices;   // row index in build input
    std::size_t input_probe_rows = 0;
    std::size_t input_build_rows = 0;
    double      wall_ms     = 0.0;
    double      kernel_ms   = 0.0;
    double      transfer_ms = 0.0;
};

class HashJoinAggregator {
public:
    virtual ~HashJoinAggregator() = default;
    [[nodiscard]] virtual Backend backend() const noexcept = 0;
    [[nodiscard]] virtual std::string device_name() const = 0;

    // Inner equi-join on int64 keys. Returns matched (probe_idx, build_idx)
    // pairs. Order is unspecified across backends; sort to compare.
    virtual HashJoinResult inner_join_i64(
        const std::int64_t* build_keys, std::size_t n_build,
        const std::int64_t* probe_keys, std::size_t n_probe) = 0;
};

std::unique_ptr<HashJoinAggregator> make_hashjoin_aggregator(Backend);

// Returns the best backend available at runtime: CUDA if compiled+device,
// else METAL if compiled+device, else CPU.
[[nodiscard]] Backend default_backend() noexcept;

// Returns the list of backends compiled into this build.
[[nodiscard]] std::vector<Backend> available_backends() noexcept;

} // namespace gpudb
