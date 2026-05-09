#include "gpu_backend.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <numeric>
#include <thread>

#if GPUDB_HAVE_OPENMP
#include <omp.h>
#endif

namespace gpudb {
namespace {

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
        const auto t0 = std::chrono::steady_clock::now();
        std::int64_t acc = 0;
#if GPUDB_HAVE_OPENMP
        #pragma omp parallel for reduction(+:acc) schedule(static)
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
            acc += data[i];
        }
#else
        for (std::size_t i = 0; i < n; ++i) acc += data[i];
#endif
        return finish_i64(acc, n, t0);
    }

    AggResult min_i64(const std::int64_t* data, std::size_t n) override {
        const auto t0 = std::chrono::steady_clock::now();
        std::int64_t m = std::numeric_limits<std::int64_t>::max();
        if (n == 0) m = 0;
#if GPUDB_HAVE_OPENMP
        #pragma omp parallel for reduction(min:m) schedule(static)
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
            if (data[i] < m) m = data[i];
        }
#else
        for (std::size_t i = 0; i < n; ++i) if (data[i] < m) m = data[i];
#endif
        return finish_i64(m, n, t0);
    }

    AggResult max_i64(const std::int64_t* data, std::size_t n) override {
        const auto t0 = std::chrono::steady_clock::now();
        std::int64_t m = std::numeric_limits<std::int64_t>::min();
        if (n == 0) m = 0;
#if GPUDB_HAVE_OPENMP
        #pragma omp parallel for reduction(max:m) schedule(static)
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
            if (data[i] > m) m = data[i];
        }
#else
        for (std::size_t i = 0; i < n; ++i) if (data[i] > m) m = data[i];
#endif
        return finish_i64(m, n, t0);
    }

    AggResult sum_f64(const double* data, std::size_t n) override {
        const auto t0 = std::chrono::steady_clock::now();
        double acc = 0.0;
#if GPUDB_HAVE_OPENMP
        #pragma omp parallel for reduction(+:acc) schedule(static)
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
            acc += data[i];
        }
#else
        for (std::size_t i = 0; i < n; ++i) acc += data[i];
#endif
        AggResult r{};
        r.value_f64   = acc;
        r.rows        = n;
        r.wall_ms     = elapsed_ms(t0);
        r.kernel_ms   = 0.0;
        r.transfer_ms = 0.0;
        return r;
    }

private:
    static double elapsed_ms(std::chrono::steady_clock::time_point t0) {
        const auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    static AggResult finish_i64(std::int64_t v, std::size_t n,
                                std::chrono::steady_clock::time_point t0) {
        AggResult r{};
        r.value_i64   = v;
        r.rows        = n;
        r.wall_ms     = elapsed_ms(t0);
        r.kernel_ms   = 0.0;
        r.transfer_ms = 0.0;
        return r;
    }
};

} // namespace

std::unique_ptr<Aggregator> make_cpu_aggregator() {
    return std::make_unique<CpuAggregator>();
}

} // namespace gpudb
