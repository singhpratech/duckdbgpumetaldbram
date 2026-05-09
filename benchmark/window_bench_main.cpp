// gpudb-window-bench — ROW_NUMBER() OVER (ORDER BY key ASC).
//
// v1 scaffold: times the CPU reference and the Metal stub (which currently
// delegates to CPU). Verifies correctness against an independent oracle:
// each output[i] should equal i's rank in the stable-sorted key order.
//
// Usage:
//   gpudb-window-bench [--rows N] [--runs K] [--no-verify]
//
// The bench is intentionally simple — no resident-column path, no
// partitioning. The point is to lock the surface for window functions.
// Real Metal speedups land when PR #5 (`feat/metal-radix-sort`) merges
// and metal_window.mm switches from delegate-to-CPU to radix sort + scatter.

#include "gpu_backend.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Args {
    std::size_t rows   = 1'000'000;
    int         runs   = 3;
    bool        verify = true;
};

void usage(const char* a0) {
    std::fprintf(stderr,
        "usage: %s [--rows N] [--runs K] [--no-verify]\n", a0);
}

bool parse(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&](const char* w) -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing arg for %s\n", w); std::exit(1); }
            return argv[++i];
        };
        if      (s == "--rows")        a.rows = std::strtoull(next("--rows").c_str(), nullptr, 10);
        else if (s == "--runs")        a.runs = std::atoi(next("--runs").c_str());
        else if (s == "--no-verify")   a.verify = false;
        else if (s == "-h" || s == "--help") { usage(argv[0]); return false; }
        else { std::fprintf(stderr, "unknown arg: %s\n", s.c_str()); usage(argv[0]); return false; }
    }
    return true;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}
double gibps(std::size_t bytes, double ms) {
    if (ms <= 0.0) return 0.0;
    return (static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0)) / (ms / 1000.0);
}

// Independent oracle for ROW_NUMBER over ORDER BY key ASC: stable-sort the
// (key, original_index) pairs and walk them in sorted order assigning 1..N.
// Lives outside the aggregator so a buggy CPU impl can't validate itself.
std::vector<std::int64_t> oracle_row_number(const std::vector<std::int64_t>& keys) {
    const std::size_t n = keys.size();
    std::vector<std::pair<std::int64_t, std::size_t>> pairs;
    pairs.reserve(n);
    for (std::size_t i = 0; i < n; ++i) pairs.emplace_back(keys[i], i);
    std::stable_sort(pairs.begin(), pairs.end(),
                     [](const auto& a, const auto& b) { return a.first < b.first; });
    std::vector<std::int64_t> out(n, 0);
    for (std::size_t p = 0; p < n; ++p) {
        out[pairs[p].second] = static_cast<std::int64_t>(p + 1);
    }
    return out;
}

bool output_equals(const std::vector<std::int64_t>& a, const std::vector<std::int64_t>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

void run_backend(gpudb::Backend b,
                 const std::vector<std::int64_t>& keys,
                 const std::vector<std::int64_t>& expected,
                 int runs, bool verify) {
    const std::size_t bytes_in  = keys.size() * sizeof(std::int64_t);
    std::printf("\n[%s]\n", gpudb::to_string(b));
    std::unique_ptr<gpudb::WindowAggregator> agg;
    try { agg = gpudb::make_window_aggregator(b); }
    catch (const std::exception& e) { std::printf("  unavailable: %s\n", e.what()); return; }
    std::printf("  device: %s\n", agg->device_name().c_str());

    // First call: verify against the oracle.
    auto first = agg->row_number_i64(keys.data(), keys.size());
    if (verify) {
        if (!output_equals(first.output, expected)) {
            std::fprintf(stderr, "  CORRECTNESS FAIL: ROW_NUMBER mismatch vs oracle\n");
            // Show first divergence to ease debugging.
            for (std::size_t i = 0; i < first.output.size(); ++i) {
                if (first.output[i] != expected[i]) {
                    std::fprintf(stderr,
                        "    first diff at i=%zu: key=%lld got=%lld want=%lld\n",
                        i, static_cast<long long>(keys[i]),
                        static_cast<long long>(first.output[i]),
                        static_cast<long long>(expected[i]));
                    break;
                }
            }
            std::exit(2);
        }
        std::printf("  correctness: OK (%zu rows)\n", first.output.size());
    }

    std::vector<double> wall, kernel;
    for (int i = 0; i < runs; ++i) {
        auto r = agg->row_number_i64(keys.data(), keys.size());
        wall.push_back(r.wall_ms);
        kernel.push_back(r.kernel_ms);
    }
    const double w = median(wall), k = median(kernel);
    std::printf("  median wall=%8.3f ms  kernel=%8.3f ms  "
                "input throughput=%6.2f GiB/s",
                w, k, gibps(bytes_in, w));
    if (k > 0.0) std::printf("  (kernel %6.2f GiB/s)", gibps(bytes_in, k));
    std::printf("\n");
}

} // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parse(argc, argv, a)) return 1;

    std::printf("gpudb-window-bench  rows=%zu runs=%d\n", a.rows, a.runs);
    auto backends = gpudb::available_backends();
    std::printf("backends:");
    for (auto b : backends) std::printf(" %s", gpudb::to_string(b));
    std::printf("\n");

    // Keys: int64 in a moderate range so we get plenty of ties (exercises
    // the stable tie-break path). Seed is fixed for reproducibility.
    std::vector<std::int64_t> keys(a.rows);
    {
        std::mt19937_64 rng(0xCAFED00DULL);
        std::uniform_int_distribution<std::int64_t> kd(0, 1'000'000);
        for (auto& x : keys) x = kd(rng);
    }

    // Oracle: stable-sort once on the host. Used for verification only.
    auto expected = oracle_row_number(keys);
    std::printf("oracle: %zu rows ranked\n", expected.size());

    for (auto b : backends) {
        // Skip backends that don't yet implement WindowAggregator (e.g. CUDA).
        try {
            auto agg = gpudb::make_window_aggregator(b);
            (void)agg;
        } catch (const std::exception& e) {
            std::printf("\n[%s]\n  unavailable: %s\n", gpudb::to_string(b), e.what());
            continue;
        }
        run_backend(b, keys, expected, a.runs, a.verify);
    }

    std::printf("\ndone.\n");
    return 0;
}
