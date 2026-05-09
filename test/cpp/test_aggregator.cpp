// Minimal hand-rolled test runner — avoids pulling in Catch2 for week 1.
// Returns nonzero on failure so `ctest` and CI can pick it up.

#include "gpu_backend.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

int failures = 0;
int total    = 0;

#define EXPECT(cond) do { \
    ++total; \
    if (!(cond)) { \
        ++failures; \
        std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
    } \
} while (0)

#define EXPECT_EQ(a, b) do { \
    ++total; \
    auto _a = (a); auto _b = (b); \
    if (!(_a == _b)) { \
        ++failures; \
        std::fprintf(stderr, "FAIL %s:%d  %s == %s  (got %lld vs %lld)\n", \
                     __FILE__, __LINE__, #a, #b, \
                     static_cast<long long>(_a), static_cast<long long>(_b)); \
    } \
} while (0)

void test_backend(gpudb::Backend b) {
    std::printf("\n--- testing backend: %s ---\n", gpudb::to_string(b));
    std::unique_ptr<gpudb::Aggregator> agg;
    try {
        agg = gpudb::make_aggregator(b);
    } catch (const std::exception& e) {
        std::printf("  skipped (%s)\n", e.what());
        return;
    }
    std::printf("  device: %s\n", agg->device_name().c_str());

    // Empty input
    {
        auto r = agg->sum_i64(nullptr, 0);
        EXPECT_EQ(r.value_i64, 0);
        EXPECT_EQ(r.rows, std::size_t{0});
    }

    // Tiny deterministic
    {
        std::vector<std::int64_t> v{1, 2, 3, 4, 5};
        auto rs = agg->sum_i64(v.data(), v.size());
        auto rn = agg->min_i64(v.data(), v.size());
        auto rx = agg->max_i64(v.data(), v.size());
        EXPECT_EQ(rs.value_i64, 15);
        EXPECT_EQ(rn.value_i64, 1);
        EXPECT_EQ(rx.value_i64, 5);
    }

    // Negative + positive
    {
        std::vector<std::int64_t> v{-100, 50, -25, 75, 0};
        auto rs = agg->sum_i64(v.data(), v.size());
        auto rn = agg->min_i64(v.data(), v.size());
        auto rx = agg->max_i64(v.data(), v.size());
        EXPECT_EQ(rs.value_i64, 0);
        EXPECT_EQ(rn.value_i64, -100);
        EXPECT_EQ(rx.value_i64, 75);
    }

    // Larger random — compare against host-side reference
    {
        std::mt19937_64 rng(0xCAFEBABEULL);
        std::uniform_int_distribution<std::int64_t> dist(-1'000'000, 1'000'000);
        const std::size_t N = 1'000'000;
        std::vector<std::int64_t> v(N);
        for (auto& x : v) x = dist(rng);

        std::int64_t ref_sum = 0, ref_min = v[0], ref_max = v[0];
        for (auto x : v) { ref_sum += x; if (x < ref_min) ref_min = x; if (x > ref_max) ref_max = x; }

        auto rs = agg->sum_i64(v.data(), N);
        auto rn = agg->min_i64(v.data(), N);
        auto rx = agg->max_i64(v.data(), N);
        EXPECT_EQ(rs.value_i64, ref_sum);
        EXPECT_EQ(rn.value_i64, ref_min);
        EXPECT_EQ(rx.value_i64, ref_max);

        std::printf("  N=%zu sum_wall=%.3f ms (kernel=%.3f ms, transfer=%.3f ms)\n",
                    N, rs.wall_ms, rs.kernel_ms, rs.transfer_ms);
    }

    // f64 sum — tolerate FP rounding
    {
        const std::size_t N = 100'000;
        std::vector<double> v(N, 1.5);
        auto r = agg->sum_f64(v.data(), N);
        const double expected = 1.5 * static_cast<double>(N);
        const double err = std::abs(r.value_f64 - expected);
        EXPECT(err < 1e-6 * expected);
    }

    // Multi-agg fusion: sum + min + max + count in one pass.
    // CUDA throws (stub); skip there. CPU + Metal must match the reference.
    if (b != gpudb::Backend::CUDA) {
        // Empty
        {
            auto r = agg->agg_all_i64(nullptr, 0);
            EXPECT_EQ(r.sum, 0);
            EXPECT_EQ(r.count, std::size_t{0});
            EXPECT_EQ(r.rows, std::size_t{0});
        }
        // Tiny deterministic
        {
            std::vector<std::int64_t> v{1, 2, 3, 4, 5};
            auto r = agg->agg_all_i64(v.data(), v.size());
            EXPECT_EQ(r.sum, 15);
            EXPECT_EQ(r.min, 1);
            EXPECT_EQ(r.max, 5);
            EXPECT_EQ(r.count, std::size_t{5});
        }
        // Negative + positive
        {
            std::vector<std::int64_t> v{-100, 50, -25, 75, 0};
            auto r = agg->agg_all_i64(v.data(), v.size());
            EXPECT_EQ(r.sum, 0);
            EXPECT_EQ(r.min, -100);
            EXPECT_EQ(r.max, 75);
            EXPECT_EQ(r.count, std::size_t{5});
        }
        // Larger random — match reference computed per-pass
        {
            std::mt19937_64 rng(0xDEADBEEFULL);
            std::uniform_int_distribution<std::int64_t> dist(-1'000'000, 1'000'000);
            const std::size_t N = 1'000'000;
            std::vector<std::int64_t> v(N);
            for (auto& x : v) x = dist(rng);
            std::int64_t ref_sum = 0, ref_min = v[0], ref_max = v[0];
            for (auto x : v) {
                ref_sum += x;
                if (x < ref_min) ref_min = x;
                if (x > ref_max) ref_max = x;
            }
            auto r = agg->agg_all_i64(v.data(), N);
            EXPECT_EQ(r.sum, ref_sum);
            EXPECT_EQ(r.min, ref_min);
            EXPECT_EQ(r.max, ref_max);
            EXPECT_EQ(r.count, N);

            // Resident path matches too.
            auto col = agg->upload_i64(v.data(), N);
            auto rr = agg->agg_all_resident_i64(*col);
            EXPECT_EQ(rr.sum, ref_sum);
            EXPECT_EQ(rr.min, ref_min);
            EXPECT_EQ(rr.max, ref_max);
            EXPECT_EQ(rr.count, N);
        }
    }
}

} // namespace

int main() {
    std::printf("gpudb test suite\n");
    std::printf("available backends:");
    for (auto b : gpudb::available_backends()) std::printf(" %s", gpudb::to_string(b));
    std::printf("\ndefault backend: %s\n", gpudb::to_string(gpudb::default_backend()));

    test_backend(gpudb::Backend::CPU);

#if GPUDB_HAVE_CUDA
    test_backend(gpudb::Backend::CUDA);
#endif
#if GPUDB_HAVE_METAL
    test_backend(gpudb::Backend::METAL);
#endif

    std::printf("\n%d / %d checks passed\n", total - failures, total);
    return failures == 0 ? 0 : 1;
}
