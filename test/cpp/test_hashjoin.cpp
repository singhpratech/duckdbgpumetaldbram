// Hash join unit tests — every backend verified against CPU reference.

#include "gpu_backend.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <utility>
#include <vector>

namespace {

int failures = 0;
int total = 0;

#define EXPECT(cond)                                                                               \
    do {                                                                                           \
        ++total;                                                                                   \
        if (!(cond)) {                                                                             \
            ++failures;                                                                            \
            std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);                   \
        }                                                                                          \
    } while (0)

#define EXPECT_EQ(a, b)                                                                            \
    do {                                                                                           \
        ++total;                                                                                   \
        auto _a = (a);                                                                             \
        auto _b = (b);                                                                             \
        if (!(_a == _b)) {                                                                         \
            ++failures;                                                                            \
            std::fprintf(stderr, "FAIL %s:%d  %s == %s  (got %lld vs %lld)\n", __FILE__, __LINE__, \
                         #a, #b, static_cast<long long>(_a), static_cast<long long>(_b));        \
        }                                                                                          \
    } while (0)

using Pair = std::pair<std::int64_t, std::int64_t>;

std::vector<Pair> canonical_pairs(const gpudb::JoinResult& r) {
    std::vector<Pair> out;
    out.reserve(r.matched);
    for (std::size_t i = 0; i < r.matched; ++i) {
        out.emplace_back(r.probe_indices[i], r.build_indices[i]);
    }
    std::sort(out.begin(), out.end());
    return out;
}

bool join_equals(const gpudb::JoinResult& a, const gpudb::JoinResult& b) {
    if (a.matched != b.matched) return false;
    return canonical_pairs(a) == canonical_pairs(b);
}

gpudb::JoinResult cpu_reference(const std::int64_t* build_keys, std::size_t n_build,
                                const std::int64_t* probe_keys, std::size_t n_probe) {
    auto cpu = gpudb::make_hashjoin_probe(gpudb::Backend::CPU);
    return cpu->inner_join_i64(build_keys, n_build, probe_keys, n_probe);
}

void expect_join_eq(const gpudb::JoinResult& got, const gpudb::JoinResult& ref, const char* label) {
    if (!join_equals(got, ref)) {
        ++failures;
        ++total;
        std::fprintf(stderr, "FAIL join mismatch: %s (matched %zu vs %zu)\n", label, got.matched,
                     ref.matched);
        return;
    }
    ++total;
}

void test_hashjoin_backend(gpudb::Backend b) {
    std::printf("\n--- testing hash join backend: %s ---\n", gpudb::to_string(b));
    std::unique_ptr<gpudb::HashJoinProbe> hj;
    try {
        hj = gpudb::make_hashjoin_probe(b);
    } catch (const std::exception& e) {
        std::printf("  skipped (%s)\n", e.what());
        return;
    }
    std::printf("  device: %s\n", hj->device_name().c_str());

    // Empty build
    {
        std::vector<std::int64_t> probe{1, 2, 3};
        auto ref = cpu_reference(nullptr, 0, probe.data(), probe.size());
        auto got = hj->inner_join_i64(nullptr, 0, probe.data(), probe.size());
        expect_join_eq(got, ref, "empty build");
        EXPECT_EQ(got.matched, std::size_t{0});
    }

    // Empty probe
    {
        std::vector<std::int64_t> build{10, 20};
        auto ref = cpu_reference(build.data(), build.size(), nullptr, 0);
        auto got = hj->inner_join_i64(build.data(), build.size(), nullptr, 0);
        expect_join_eq(got, ref, "empty probe");
        EXPECT_EQ(got.matched, std::size_t{0});
    }

    // Single match
    {
        std::vector<std::int64_t> build{42};
        std::vector<std::int64_t> probe{99, 42, 7};
        auto ref = cpu_reference(build.data(), build.size(), probe.data(), probe.size());
        auto got = hj->inner_join_i64(build.data(), build.size(), probe.data(), probe.size());
        expect_join_eq(got, ref, "single match");
        EXPECT_EQ(got.matched, std::size_t{1});
        EXPECT_EQ(got.probe_indices[0], std::int64_t{1});
        EXPECT_EQ(got.build_indices[0], std::int64_t{0});
    }

    // No matches
    {
        std::vector<std::int64_t> build{1, 2, 3};
        std::vector<std::int64_t> probe{4, 5, 6};
        auto ref = cpu_reference(build.data(), build.size(), probe.data(), probe.size());
        auto got = hj->inner_join_i64(build.data(), build.size(), probe.data(), probe.size());
        expect_join_eq(got, ref, "no match");
        EXPECT_EQ(got.matched, std::size_t{0});
    }

    // All probe rows match (FK into build domain)
    {
        std::vector<std::int64_t> build{10, 20, 30, 40, 50};
        std::vector<std::int64_t> probe{10, 20, 30, 40, 50};
        auto ref = cpu_reference(build.data(), build.size(), probe.data(), probe.size());
        auto got = hj->inner_join_i64(build.data(), build.size(), probe.data(), probe.size());
        expect_join_eq(got, ref, "all match");
        EXPECT_EQ(got.matched, probe.size());
    }

    // Duplicate build keys: first build row wins (v1 contract)
    {
        std::vector<std::int64_t> build{7, 7, 7};
        std::vector<std::int64_t> probe{7};
        auto ref = cpu_reference(build.data(), build.size(), probe.data(), probe.size());
        auto got = hj->inner_join_i64(build.data(), build.size(), probe.data(), probe.size());
        expect_join_eq(got, ref, "duplicate build keys");
        EXPECT_EQ(got.matched, std::size_t{1});
        EXPECT_EQ(got.build_indices[0], std::int64_t{0});
    }

    // Negative keys
    {
        std::vector<std::int64_t> build{-5, 0, 5};
        std::vector<std::int64_t> probe{-5, 0, 99, 5};
        auto ref = cpu_reference(build.data(), build.size(), probe.data(), probe.size());
        auto got = hj->inner_join_i64(build.data(), build.size(), probe.data(), probe.size());
        expect_join_eq(got, ref, "negative keys");
        EXPECT_EQ(got.matched, std::size_t{3});
    }

    // Medium random FK workload
    {
        const std::size_t n_build = 50'000;
        const std::size_t n_probe = 200'000;
        std::vector<std::int64_t> build(n_build);
        for (std::size_t j = 0; j < n_build; ++j) {
            build[j] = static_cast<std::int64_t>(j * 17 + 3);
        }
        std::mt19937_64 rng(0x4A01BEEFULL);
        std::uniform_int_distribution<std::size_t> pick(0, n_build - 1);
        std::vector<std::int64_t> probe(n_probe);
        for (auto& p : probe) {
            p = build[pick(rng)];
        }
        // Inject some non-matching probes
        probe[0] = -999999;
        probe[1] = 999999999;

        auto ref = cpu_reference(build.data(), n_build, probe.data(), n_probe);
        auto got = hj->inner_join_i64(build.data(), n_build, probe.data(), n_probe);
        expect_join_eq(got, ref, "random 50k×200k");
        EXPECT_EQ(got.matched, ref.matched);
        std::printf("  random 50k×200k: matched=%zu wall=%.3f ms kernel=%.3f ms\n", got.matched,
                    got.wall_ms, got.kernel_ms);
    }

    // GPU backends must execute kernels (not CPU fallback)
    if (b == gpudb::Backend::METAL || b == gpudb::Backend::CUDA) {
        const std::size_t n_build = 10'000;
        const std::size_t n_probe = 50'000;
        std::vector<std::int64_t> build(n_build);
        for (std::size_t j = 0; j < n_build; ++j) build[j] = static_cast<std::int64_t>(j);
        std::vector<std::int64_t> probe(n_probe);
        for (std::size_t i = 0; i < n_probe; ++i) probe[i] = static_cast<std::int64_t>(i % n_build);

        auto got = hj->inner_join_i64(build.data(), n_build, probe.data(), n_probe);
        EXPECT(got.kernel_ms > 0.0);
        std::printf("  gpu kernel check: kernel_ms=%.3f\n", got.kernel_ms);
    }
}

void test_hybrid_hashjoin() {
    std::printf("\n--- testing hybrid hash join ---\n");
    auto ref_cpu = gpudb::make_hashjoin_probe(gpudb::Backend::CPU);
    auto hybrid = gpudb::make_hybrid_hashjoin_probe();
    std::printf("  device: %s\n", hybrid->device_name().c_str());

    const std::size_t n_build = 100'000;
    const std::size_t n_probe = 500'000;
    std::vector<std::int64_t> build(n_build);
    for (std::size_t j = 0; j < n_build; ++j) build[j] = static_cast<std::int64_t>(j * 17 + 3);
    std::vector<std::int64_t> probe(n_probe);
    for (std::size_t i = 0; i < n_probe; ++i) probe[i] = build[i % n_build];

    auto ref = ref_cpu->inner_join_i64(build.data(), n_build, probe.data(), n_probe);
    auto got = hybrid->inner_join_i64(build.data(), n_build, probe.data(), n_probe);
    expect_join_eq(got, ref, "hybrid 100k×500k");
    EXPECT(got.kernel_ms > 0.0);
    std::printf("  hybrid 100k×500k: matched=%zu dispatch=%s wall=%.3f ms\n",
                got.matched, gpudb::to_string(hybrid->last_decision().chosen), got.wall_ms);
}

} // namespace

void test_hashjoin() {
    test_hashjoin_backend(gpudb::Backend::CPU);

#if GPUDB_HAVE_CUDA
    test_hashjoin_backend(gpudb::Backend::CUDA);
#endif
#if GPUDB_HAVE_METAL
    test_hashjoin_backend(gpudb::Backend::METAL);
#endif

    test_hybrid_hashjoin();
}

int test_hashjoin_failures() { return failures; }
int test_hashjoin_total() { return total; }
