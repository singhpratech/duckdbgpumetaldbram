// gpudb-groupby-bench — SUM(value) GROUP BY key, varying cardinality.
//
// Usage:
//   gpudb-groupby-bench --rows N --groups G [--runs K] [--input-keys F]
//
// Synthetic mode generates int64 keys uniformly drawn from [0, G) and int64
// values uniformly random. We then verify the GPU result against the CPU
// reference by sorting both (k,v) lists.

#include "data_format.hpp"
#include "gpu_backend.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

// --backend selects which backends to bench:
//   all     — every available backend; default.
//   hybrid  — only the hybrid planner; prints the dispatch decision per call.
//   sweep   — bench {hybrid, CPU, GPU} across an N × cardinality grid;
//             produces the data for the BENCHMARK.md "Hybrid planner" section.
enum class BackendSel { All, Hybrid, Sweep };

struct Args {
    std::size_t rows = 10'000'000;
    std::size_t groups = 1024;          // cardinality
    int         runs = 3;
    bool        verify = true;
    std::string input_keys;             // optional: load int64 keys from .gpudb
    BackendSel  bsel = BackendSel::All;
};

void usage(const char* a0) {
    std::fprintf(stderr,
        "usage: %s [--rows N] [--groups G] [--runs K] [--input-keys FILE.gpudb]\n"
        "          [--backend all|hybrid|sweep] [--no-verify]\n", a0);
}

bool parse(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&](const char* w) -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing arg for %s\n", w); std::exit(1); }
            return argv[++i];
        };
        if      (s == "--rows")        a.rows = std::strtoull(next("--rows").c_str(), nullptr, 10);
        else if (s == "--groups")      a.groups = std::strtoull(next("--groups").c_str(), nullptr, 10);
        else if (s == "--runs")        a.runs = std::atoi(next("--runs").c_str());
        else if (s == "--no-verify")   a.verify = false;
        else if (s == "--input-keys")  a.input_keys = next("--input-keys");
        else if (s == "--backend") {
            auto t = next("--backend");
            if      (t == "all")    a.bsel = BackendSel::All;
            else if (t == "hybrid") a.bsel = BackendSel::Hybrid;
            else if (t == "sweep")  a.bsel = BackendSel::Sweep;
            else { std::fprintf(stderr, "bad --backend: %s (use all|hybrid|sweep)\n", t.c_str()); return false; }
        }
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
    return (static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0)) / (ms / 1000.0);
}

void load_int64(const std::string& path, std::vector<std::int64_t>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(1); }
    gpudb::DataHeader h{};
    f.read(reinterpret_cast<char*>(&h), sizeof(h));
    if (std::memcmp(h.magic, gpudb::kMagic, 8) != 0) {
        std::fprintf(stderr, "%s: bad magic\n", path.c_str()); std::exit(1);
    }
    if (h.dtype != 0) {
        std::fprintf(stderr, "%s: expected i64\n", path.c_str()); std::exit(1);
    }
    out.resize(h.count);
    f.read(reinterpret_cast<char*>(out.data()),
           static_cast<std::streamsize>(out.size() * sizeof(std::int64_t)));
}

bool result_equals(gpudb::GroupByResult a, gpudb::GroupByResult b) {
    if (a.keys.size() != b.keys.size()) return false;
    auto pair_lt = [](const std::pair<std::int64_t,std::int64_t>& x,
                      const std::pair<std::int64_t,std::int64_t>& y){
        return x.first < y.first;
    };
    std::vector<std::pair<std::int64_t,std::int64_t>> pa, pb;
    pa.reserve(a.keys.size()); pb.reserve(b.keys.size());
    for (std::size_t i = 0; i < a.keys.size(); ++i) pa.emplace_back(a.keys[i], a.sums[i]);
    for (std::size_t i = 0; i < b.keys.size(); ++i) pb.emplace_back(b.keys[i], b.sums[i]);
    std::sort(pa.begin(), pa.end(), pair_lt);
    std::sort(pb.begin(), pb.end(), pair_lt);
    return pa == pb;
}

// Run the hybrid planner directly. Reports its picked backend + reason
// per call. The first call's median wall is recorded for the headline.
void run_hybrid(const std::vector<std::int64_t>& keys,
                const std::vector<std::int64_t>& values, int runs,
                std::size_t expected_groups,
                const gpudb::GroupByResult* reference, bool verify,
                double* out_wall = nullptr,
                gpudb::DispatchDecision* out_decision = nullptr) {
    std::printf("\n[Hybrid GROUP BY] rows=%zu hint=%zu\n", keys.size(), expected_groups);
    auto agg = gpudb::make_hybrid_groupby_aggregator();
    std::printf("  device: %s\n", agg->device_name().c_str());

    auto first = agg->groupby_sum_i64(keys.data(), values.data(), keys.size(), expected_groups);
    const auto decision = agg->last_decision();
    std::printf("  groups produced: %zu\n", first.keys.size());
    if (reference && verify) {
        if (!result_equals(first, *reference)) {
            std::fprintf(stderr, "  CORRECTNESS FAIL: hybrid result differs from reference\n");
            std::exit(2);
        }
    }
    std::vector<double> wall;
    for (int i = 0; i < runs; ++i) {
        auto r = agg->groupby_sum_i64(keys.data(), values.data(), keys.size(), expected_groups);
        wall.push_back(r.wall_ms);
    }
    const double w = median(wall);
    std::printf("  median wall=%8.3f ms   picked=%s reason=%s%s\n",
                w, gpudb::to_string(decision.chosen),
                gpudb::to_string(decision.reason),
                decision.borderline ? "  [borderline]" : "");
    if (out_wall) *out_wall = w;
    if (out_decision) *out_decision = decision;
}

// Bench a pure backend (CPU or GPU) and return the median wall in ms,
// or NaN on failure / unavailable. Quiet — sweep mode prefers compact rows.
double bench_pure_quiet(gpudb::Backend b, const std::vector<std::int64_t>& keys,
                       const std::vector<std::int64_t>& values, int runs,
                       std::size_t expected_groups) {
    std::unique_ptr<gpudb::GroupByAggregator> agg;
    try { agg = gpudb::make_groupby_aggregator(b); }
    catch (const std::exception&) { return std::numeric_limits<double>::quiet_NaN(); }
    // Warmup
    (void)agg->groupby_sum_i64(keys.data(), values.data(), keys.size(), expected_groups);
    std::vector<double> wall;
    for (int i = 0; i < runs; ++i) {
        auto r = agg->groupby_sum_i64(keys.data(), values.data(), keys.size(), expected_groups);
        wall.push_back(r.wall_ms);
    }
    return median(wall);
}

// N × cardinality sweep: compares hybrid wall to pure-CPU and pure-GPU wall
// at each cell. Prints a compact table showing the dispatch decision and
// who wins at each cell. Used to populate BENCHMARK.md.
void run_sweep(int runs, bool verify) {
    // Conservative grid sized to fit on M4 Max in seconds, not minutes.
    // Each (N, groups) cell does 4 backend runs × `runs` iterations on
    // input arrays of N × 16 bytes, plus a CPU reference for verification.
    const std::vector<std::size_t> ns      = {  100'000ULL,
                                                 1'000'000ULL,
                                                 5'000'000ULL,
                                                10'000'000ULL };
    const std::vector<std::size_t> groups  = {     1024ULL,
                                                  10'000ULL,
                                                 100'000ULL,
                                               1'000'000ULL };

    std::printf("\n=== Hybrid planner v1 sweep ===\n");
    std::printf("Each cell: hybrid wall (ms) | CPU wall | GPU wall | picked | hybrid wins?\n");
    std::printf("%-10s %-10s %12s %12s %12s %-12s %-22s %s\n",
                "N", "groups", "hybrid_ms", "cpu_ms", "gpu_ms", "picked", "reason", "verdict");

    std::size_t hybrid_beats_gpu = 0;
    std::size_t hybrid_beats_cpu = 0;
    std::size_t cells_with_gpu   = 0;
    std::size_t total_cells      = 0;

    for (std::size_t n : ns) {
        for (std::size_t g : groups) {
            ++total_cells;
            // Generate inputs for this cell.
            std::vector<std::int64_t> keys(n), values(n);
            std::mt19937_64 rng(0xC0FFEEULL ^ (n * 1315423911u) ^ g);
            std::uniform_int_distribution<std::int64_t> kd(0, std::max<std::size_t>(1, g) - 1);
            std::uniform_int_distribution<std::int64_t> vd(-1000, 1000);
            for (std::size_t i = 0; i < n; ++i) { keys[i] = kd(rng); values[i] = vd(rng); }

            // CPU reference (also used for correctness verification).
            gpudb::GroupByResult ref;
            {
                auto cpu = gpudb::make_groupby_aggregator(gpudb::Backend::CPU);
                ref = cpu->groupby_sum_i64(keys.data(), values.data(), n, g);
            }

            // Hybrid run (verifies correctness against ref).
            auto hagg = gpudb::make_hybrid_groupby_aggregator();
            auto hres = hagg->groupby_sum_i64(keys.data(), values.data(), n, g);
            if (verify && !result_equals(hres, ref)) {
                std::fprintf(stderr, "  CORRECTNESS FAIL at N=%zu g=%zu (hybrid vs CPU ref)\n", n, g);
                std::exit(2);
            }
            auto decision = hagg->last_decision();
            std::vector<double> hwall;
            for (int i = 0; i < runs; ++i) {
                auto r = hagg->groupby_sum_i64(keys.data(), values.data(), n, g);
                hwall.push_back(r.wall_ms);
            }
            const double h_ms = median(hwall);

            // Pure CPU and pure GPU (the planner's "always-X" comparators).
            const double cpu_ms = bench_pure_quiet(gpudb::Backend::CPU, keys, values, runs, g);

            double gpu_ms = std::numeric_limits<double>::quiet_NaN();
            // Pick the GPU side based on what's available.
            const auto avail = gpudb::available_backends();
            for (auto b : avail) {
                if (b != gpudb::Backend::CPU) {
                    gpu_ms = bench_pure_quiet(b, keys, values, runs, g);
                    break;
                }
            }

            std::string verdict;
            const bool have_gpu = !std::isnan(gpu_ms);
            if (have_gpu) ++cells_with_gpu;
            // 15% tolerance: bench-to-bench jitter on warm caches can swing
            // sub-millisecond runs that much, especially at the smaller cells.
            // We're proving "hybrid picked CORRECTLY", not "hybrid won by
            // microseconds against the same backend".
            constexpr double kTol = 1.15;
            if (have_gpu && h_ms <= gpu_ms * kTol) ++hybrid_beats_gpu;
            if (h_ms <= cpu_ms * kTol)             ++hybrid_beats_cpu;
            if (have_gpu) {
                if      (h_ms <= cpu_ms * kTol && h_ms <= gpu_ms * kTol) verdict = "hybrid_wins";
                else if (h_ms <= cpu_ms * kTol)                          verdict = "ties_cpu_loses_to_gpu";
                else                                                     verdict = "loses";
            } else {
                verdict = (h_ms <= cpu_ms * kTol) ? "hybrid==cpu" : "loses_vs_cpu";
            }

            std::printf("%-10zu %-10zu %12.3f %12.3f %12.3f %-12s %-22s %s%s\n",
                        n, g, h_ms, cpu_ms, gpu_ms,
                        gpudb::to_string(decision.chosen),
                        gpudb::to_string(decision.reason),
                        verdict.c_str(),
                        decision.borderline ? "  [borderline]" : "");
        }
    }

    std::printf("\nSummary across %zu cells (%zu had a GPU):\n", total_cells, cells_with_gpu);
    std::printf("  hybrid <= always-CPU at %zu / %zu cells (%.0f%%)\n",
                hybrid_beats_cpu, total_cells,
                100.0 * static_cast<double>(hybrid_beats_cpu) / static_cast<double>(total_cells));
    if (cells_with_gpu > 0) {
        std::printf("  hybrid <= always-GPU at %zu / %zu cells (%.0f%%)\n",
                    hybrid_beats_gpu, cells_with_gpu,
                    100.0 * static_cast<double>(hybrid_beats_gpu) / static_cast<double>(cells_with_gpu));
    }
}

void run_backend(gpudb::Backend b, const std::vector<std::int64_t>& keys,
                 const std::vector<std::int64_t>& values, int runs,
                 const gpudb::GroupByResult* reference, bool verify) {
    const std::size_t bytes = keys.size() * sizeof(std::int64_t) * 2;  // keys + values
    std::printf("\n[%s]\n", gpudb::to_string(b));
    std::unique_ptr<gpudb::GroupByAggregator> agg;
    try { agg = gpudb::make_groupby_aggregator(b); }
    catch (const std::exception& e) { std::printf("  unavailable: %s\n", e.what()); return; }
    std::printf("  device: %s\n", agg->device_name().c_str());

    // First call: verify. Pass cardinality hint when known (synthetic --groups).
    const std::size_t hint = (reference != nullptr) ? reference->keys.size() : 0;
    auto first = agg->groupby_sum_i64(keys.data(), values.data(), keys.size(), hint);
    std::printf("  groups produced: %zu\n", first.keys.size());
    if (reference && verify) {
        if (!result_equals(first, *reference)) {
            std::fprintf(stderr, "  CORRECTNESS FAIL: result differs from reference\n");
            std::exit(2);
        }
    }

    std::vector<double> wall, kernel, xfer;
    for (int i = 0; i < runs; ++i) {
        auto r = agg->groupby_sum_i64(keys.data(), values.data(), keys.size(), hint);
        wall.push_back(r.wall_ms);
        kernel.push_back(r.kernel_ms);
        xfer.push_back(r.transfer_ms);
    }
    const double w = median(wall), k = median(kernel), x = median(xfer);
    std::printf("  median wall=%8.3f ms  kernel=%8.3f ms  xfer=%8.3f ms  "
                "input throughput=%6.2f GiB/s",
                w, k, x, gibps(bytes, w));
    if (k > 0.0) std::printf("  (kernel %6.2f GiB/s)", gibps(bytes, k));
    std::printf("\n");
}

} // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parse(argc, argv, a)) return 1;

    std::printf("gpudb-groupby-bench  rows=%zu groups=%zu runs=%d backend=%s\n",
                a.rows, a.groups, a.runs,
                a.bsel == BackendSel::Hybrid ? "hybrid" :
                a.bsel == BackendSel::Sweep  ? "sweep"  : "all");
    auto backends = gpudb::available_backends();
    std::printf("backends:");
    for (auto b : backends) std::printf(" %s", gpudb::to_string(b));
    std::printf("\n");

    if (a.bsel == BackendSel::Sweep) {
        // The sweep generates its own data per cell; ignore --rows / --groups.
        run_sweep(a.runs, a.verify);
        std::printf("\ndone.\n");
        return 0;
    }

    std::vector<std::int64_t> keys, values;
    if (!a.input_keys.empty()) {
        load_int64(a.input_keys, keys);
        a.rows = keys.size();
        std::printf("loaded %zu keys from %s\n", keys.size(), a.input_keys.c_str());
    } else {
        std::mt19937_64 rng(0xC0FFEEULL);
        std::uniform_int_distribution<std::int64_t> kd(0, std::max<std::size_t>(1, a.groups) - 1);
        keys.resize(a.rows);
        for (auto& x : keys) x = kd(rng);
    }
    std::mt19937_64 rng2(0xBEEFULL);
    std::uniform_int_distribution<std::int64_t> vd(-1000, 1000);
    values.resize(keys.size());
    for (auto& x : values) x = vd(rng2);

    // Build CPU reference once (always; cheap relative to bench runs).
    gpudb::GroupByResult reference;
    {
        auto cpu = gpudb::make_groupby_aggregator(gpudb::Backend::CPU);
        reference = cpu->groupby_sum_i64(keys.data(), values.data(), keys.size());
        std::printf("cpu reference: %zu groups\n", reference.keys.size());
    }

    if (a.bsel == BackendSel::Hybrid) {
        run_hybrid(keys, values, a.runs, reference.keys.size(), &reference, a.verify);
    } else {
        for (auto b : backends) run_backend(b, keys, values, a.runs, &reference, a.verify);
    }

    std::printf("\ndone.\n");
    return 0;
}
