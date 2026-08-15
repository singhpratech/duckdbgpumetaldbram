#include "gpu_backend.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if GPUDB_HAVE_OPENMP
#include <omp.h>
#endif

namespace gpudb {
namespace {

// Portable parallel chunking for builds without OpenMP (Apple clang ships no
// libomp, so the macOS loadable extension otherwise reduces single-threaded).
// Splits [0,n) into one contiguous chunk per worker and combines the partials
// in chunk order — deterministic for a given (n, worker count), which matters
// for f64 sums where the combine order changes the rounding.
template <class Partial, class ChunkFn, class CombineFn>
Partial parallel_chunks(std::size_t n, ChunkFn chunk, CombineFn combine) {
    constexpr std::size_t kMinPerWorker = std::size_t(1) << 19; // 512k elems
    const unsigned hw = std::thread::hardware_concurrency();
    std::size_t workers = std::min<std::size_t>(hw ? hw : 1, n / kMinPerWorker);
    if (workers <= 1) return chunk(std::size_t(0), n);

    std::vector<Partial> parts(workers);
    std::vector<std::thread> threads;
    threads.reserve(workers);
    const std::size_t per = n / workers;
    for (std::size_t w = 0; w < workers; ++w) {
        const std::size_t begin = w * per;
        const std::size_t end   = (w + 1 == workers) ? n : begin + per;
        threads.emplace_back([&parts, &chunk, w, begin, end] {
            parts[w] = chunk(begin, end);
        });
    }
    for (auto& t : threads) t.join();
    Partial acc = parts[0];
    for (std::size_t w = 1; w < workers; ++w) acc = combine(acc, parts[w]);
    return acc;
}

// CPU resident column = a copy of the host data the aggregator owns.
// (Could just hold a const pointer, but a copy matches GPU semantics so
// the bench numbers reflect "queries against backend-owned data".)
class CpuResidentColumn final : public ResidentColumn {
public:
    CpuResidentColumn(const void* src, std::size_t n, Dtype dt)
        : rows_(n), dtype_(dt) {
        const std::size_t elem = (dt == Dtype::I64) ? sizeof(std::int64_t) : sizeof(double);
        buf_.assign(static_cast<const std::byte*>(src),
                    static_cast<const std::byte*>(src) + n * elem);
    }
    Backend     backend_tag() const noexcept override { return Backend::CPU; }
    Dtype       dtype()       const noexcept override { return dtype_; }
    std::size_t rows()        const noexcept override { return rows_; }

    [[nodiscard]] const std::int64_t* as_i64() const {
        return reinterpret_cast<const std::int64_t*>(buf_.data());
    }
    [[nodiscard]] const double* as_f64() const {
        return reinterpret_cast<const double*>(buf_.data());
    }

private:
    std::vector<std::byte> buf_;
    std::size_t rows_;
    Dtype       dtype_;
};

class CpuAggregator final : public Aggregator {
public:
    Backend backend() const noexcept override { return Backend::CPU; }

    std::string device_name() const override {
#if GPUDB_HAVE_OPENMP
        return "CPU (OpenMP, " + std::to_string(omp_get_max_threads()) + " threads)";
#else
        return "CPU (scalar)";
#endif
    }

    AggResult sum_i64(const std::int64_t* data, std::size_t n) override {
        return run_i64(data, n, ReduceKind::Sum, 0);
    }
    AggResult min_i64(const std::int64_t* data, std::size_t n) override {
        return run_i64(data, n, ReduceKind::Min, std::numeric_limits<std::int64_t>::max());
    }
    AggResult max_i64(const std::int64_t* data, std::size_t n) override {
        return run_i64(data, n, ReduceKind::Max, std::numeric_limits<std::int64_t>::min());
    }
    AggResult sum_f64(const double* data, std::size_t n) override {
        return run_f64_sum(data, n);
    }

    std::unique_ptr<ResidentColumn> upload_i64(const std::int64_t* d, std::size_t n) override {
        return std::make_unique<CpuResidentColumn>(d, n, Dtype::I64);
    }
    std::unique_ptr<ResidentColumn> upload_f64(const double* d, std::size_t n) override {
        return std::make_unique<CpuResidentColumn>(d, n, Dtype::F64);
    }

    AggResult sum_resident_i64(const ResidentColumn& c) override {
        const auto& r = check_i64(c);
        return run_i64(r.as_i64(), r.rows(), ReduceKind::Sum, 0);
    }
    AggResult min_resident_i64(const ResidentColumn& c) override {
        const auto& r = check_i64(c);
        return run_i64(r.as_i64(), r.rows(), ReduceKind::Min,
                       std::numeric_limits<std::int64_t>::max());
    }
    AggResult max_resident_i64(const ResidentColumn& c) override {
        const auto& r = check_i64(c);
        return run_i64(r.as_i64(), r.rows(), ReduceKind::Max,
                       std::numeric_limits<std::int64_t>::min());
    }
    AggResult sum_resident_f64(const ResidentColumn& c) override {
        const auto& r = check_f64(c);
        return run_f64_sum(r.as_f64(), r.rows());
    }

    AggAllResult agg_all_i64(const std::int64_t* data, std::size_t n) override {
        return run_agg_all_i64(data, n);
    }
    AggAllResult agg_all_resident_i64(const ResidentColumn& c) override {
        const auto& r = check_i64(c);
        return run_agg_all_i64(r.as_i64(), r.rows());
    }

private:
    enum class ReduceKind { Sum, Min, Max };

    static double elapsed_ms(std::chrono::steady_clock::time_point t0) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }

    AggResult run_i64(const std::int64_t* data, std::size_t n, ReduceKind kind, std::int64_t init) {
        const auto t0 = std::chrono::steady_clock::now();
        std::int64_t v = (n == 0) ? 0 : init;
        switch (kind) {
            case ReduceKind::Sum: {
                // Accumulate in uint64: overflow wraps (two's complement),
                // matching the documented gpu_sum semantics, instead of the
                // signed-overflow UB the old code had.
                std::uint64_t uv = 0;
#if GPUDB_HAVE_OPENMP
                #pragma omp parallel for reduction(+:uv) schedule(static)
                for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
                    uv += static_cast<std::uint64_t>(data[i]);
#else
                uv = parallel_chunks<std::uint64_t>(n,
                    [data](std::size_t b, std::size_t e) {
                        std::uint64_t acc = 0;
                        for (std::size_t i = b; i < e; ++i)
                            acc += static_cast<std::uint64_t>(data[i]);
                        return acc;
                    },
                    [](std::uint64_t a, std::uint64_t b) { return a + b; });
#endif
                v = static_cast<std::int64_t>(uv);
                break;
            }
            case ReduceKind::Min: {
#if GPUDB_HAVE_OPENMP
                #pragma omp parallel for reduction(min:v) schedule(static)
                for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
                    if (data[i] < v) v = data[i];
#else
                if (n > 0) v = parallel_chunks<std::int64_t>(n,
                    [data](std::size_t b, std::size_t e) {
                        std::int64_t m = std::numeric_limits<std::int64_t>::max();
                        for (std::size_t i = b; i < e; ++i) if (data[i] < m) m = data[i];
                        return m;
                    },
                    [](std::int64_t a, std::int64_t b) { return a < b ? a : b; });
#endif
                break;
            }
            case ReduceKind::Max: {
#if GPUDB_HAVE_OPENMP
                #pragma omp parallel for reduction(max:v) schedule(static)
                for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
                    if (data[i] > v) v = data[i];
#else
                if (n > 0) v = parallel_chunks<std::int64_t>(n,
                    [data](std::size_t b, std::size_t e) {
                        std::int64_t m = std::numeric_limits<std::int64_t>::min();
                        for (std::size_t i = b; i < e; ++i) if (data[i] > m) m = data[i];
                        return m;
                    },
                    [](std::int64_t a, std::int64_t b) { return a > b ? a : b; });
#endif
                break;
            }
        }
        AggResult r{};
        r.value_i64 = v; r.rows = n;
        r.wall_ms = elapsed_ms(t0);
        return r;
    }

    // Single-pass sum + min + max + count. The whole point: each cache line
    // is touched once, so on a memory-bandwidth-bound workload this should
    // be ~3x faster than calling sum/min/max separately.
    AggAllResult run_agg_all_i64(const std::int64_t* data, std::size_t n) {
        const auto t0 = std::chrono::steady_clock::now();
        AggAllResult r{};
        r.rows  = n;
        r.count = n;
        if (n == 0) {
            r.sum = 0;
            r.min = std::numeric_limits<std::int64_t>::max();
            r.max = std::numeric_limits<std::int64_t>::min();
            r.wall_ms = elapsed_ms(t0);
            return r;
        }

        // uint64 accumulate: overflow wraps (documented gpu_sum semantics)
        // instead of signed-overflow UB.
        std::uint64_t sum_v = 0;
        std::int64_t min_v = std::numeric_limits<std::int64_t>::max();
        std::int64_t max_v = std::numeric_limits<std::int64_t>::min();
#if GPUDB_HAVE_OPENMP
        #pragma omp parallel for reduction(+:sum_v) reduction(min:min_v) \
                                 reduction(max:max_v) schedule(static)
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
            const std::int64_t x = data[i];
            sum_v += static_cast<std::uint64_t>(x);
            if (x < min_v) min_v = x;
            if (x > max_v) max_v = x;
        }
#else
        for (std::size_t i = 0; i < n; ++i) {
            const std::int64_t x = data[i];
            sum_v += static_cast<std::uint64_t>(x);
            if (x < min_v) min_v = x;
            if (x > max_v) max_v = x;
        }
#endif
        r.sum = static_cast<std::int64_t>(sum_v);
        r.min = min_v;
        r.max = max_v;
        r.wall_ms = elapsed_ms(t0);
        return r;
    }

    AggResult run_f64_sum(const double* data, std::size_t n) {
        const auto t0 = std::chrono::steady_clock::now();
        double acc = 0.0;
#if GPUDB_HAVE_OPENMP
        #pragma omp parallel for reduction(+:acc) schedule(static)
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) acc += data[i];
#else
        acc = parallel_chunks<double>(n,
            [data](std::size_t b, std::size_t e) {
                double a = 0.0;
                for (std::size_t i = b; i < e; ++i) a += data[i];
                return a;
            },
            [](double a, double b) { return a + b; });
#endif
        AggResult r{};
        r.value_f64 = acc; r.rows = n;
        r.wall_ms = elapsed_ms(t0);
        return r;
    }

    static const CpuResidentColumn& check_i64(const ResidentColumn& c) {
        if (c.backend_tag() != Backend::CPU)
            throw std::runtime_error("ResidentColumn from wrong backend");
        if (c.dtype() != Dtype::I64)
            throw std::runtime_error("ResidentColumn dtype mismatch (expected i64)");
        return static_cast<const CpuResidentColumn&>(c);
    }
    static const CpuResidentColumn& check_f64(const ResidentColumn& c) {
        if (c.backend_tag() != Backend::CPU)
            throw std::runtime_error("ResidentColumn from wrong backend");
        if (c.dtype() != Dtype::F64)
            throw std::runtime_error("ResidentColumn dtype mismatch (expected f64)");
        return static_cast<const CpuResidentColumn&>(c);
    }
};

} // namespace

std::unique_ptr<Aggregator> make_cpu_aggregator() {
    return std::make_unique<CpuAggregator>();
}

} // namespace gpudb
