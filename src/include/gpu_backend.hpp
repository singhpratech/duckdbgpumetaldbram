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

// Returns the best backend available at runtime: CUDA if compiled+device,
// else METAL if compiled+device, else CPU.
[[nodiscard]] Backend default_backend() noexcept;

// Returns the list of backends compiled into this build.
[[nodiscard]] std::vector<Backend> available_backends() noexcept;

} // namespace gpudb
