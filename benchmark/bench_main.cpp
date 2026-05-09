// gpudb-bench — microbenchmark SUM/MIN/MAX across all available backends.
// Modes:
//   --mode cold    one-shot per query (transfer + kernel each call)
//   --mode hot     upload once, then N queries (kernel only)
//   --mode both    print both side-by-side  (default)
//
// Sources:
//   --rows N [--dtype i64|f64]    synthetic random data
//   --input FILE.gpudb            load from .gpudb flat binary

#include "data_format.hpp"
#include "gpu_backend.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

enum class Mode { Cold, Hot, Both };
// --backend selects which set of backends the bench enumerates:
//   all     — every available backend, one report per backend (default).
//   hybrid  — only the hybrid planner; prints which backend it picked per call.
enum class BackendSel { All, Hybrid };

struct Args {
    std::size_t rows = 10'000'000;
    int         runs = 5;
    bool        verify = true;
    gpudb::Dtype dtype = gpudb::Dtype::I64;
    std::string input;
    Mode        mode = Mode::Both;
    BackendSel  bsel = BackendSel::All;
};

void usage(const char* a0) {
    std::fprintf(stderr,
        "usage: %s [--rows N] [--dtype i64|f64] [--input FILE] [--runs K]\n"
        "          [--mode cold|hot|both] [--backend all|hybrid] [--no-verify]\n", a0);
}

bool parse(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&](const char* w) -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing arg for %s\n", w); std::exit(1); }
            return argv[++i];
        };
        if      (s == "--rows")   a.rows = std::strtoull(next("--rows").c_str(), nullptr, 10);
        else if (s == "--runs")   a.runs = std::atoi(next("--runs").c_str());
        else if (s == "--no-verify") a.verify = false;
        else if (s == "--input")  a.input = next("--input");
        else if (s == "--mode") {
            auto t = next("--mode");
            if (t == "cold") a.mode = Mode::Cold;
            else if (t == "hot") a.mode = Mode::Hot;
            else if (t == "both") a.mode = Mode::Both;
            else { std::fprintf(stderr, "bad --mode: %s\n", t.c_str()); return false; }
        }
        else if (s == "--backend") {
            auto t = next("--backend");
            if (t == "all") a.bsel = BackendSel::All;
            else if (t == "hybrid") a.bsel = BackendSel::Hybrid;
            else { std::fprintf(stderr, "bad --backend: %s (use all|hybrid)\n", t.c_str()); return false; }
        }
        else if (s == "--dtype") {
            auto t = next("--dtype");
            if (t == "i64") a.dtype = gpudb::Dtype::I64;
            else if (t == "f64") a.dtype = gpudb::Dtype::F64;
            else { std::fprintf(stderr, "bad --dtype: %s\n", t.c_str()); return false; }
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

template <typename T> T host_sum(const std::vector<T>& v) {
    T acc = T{}; for (auto x : v) acc += x; return acc;
}

template <typename T>
void verify_or_die(const T& got, const T& expected, const char* what) {
    bool ok;
    if constexpr (std::is_floating_point_v<T>) {
        const double err = std::abs(static_cast<double>(got) - static_cast<double>(expected));
        const double tol = 1e-6 * std::max(1.0, std::abs(static_cast<double>(expected)));
        ok = err <= tol;
    } else {
        ok = got == expected;
    }
    if (!ok) {
        std::fprintf(stderr, "  CORRECTNESS FAIL [%s]: got %g, expected %g\n",
                     what, static_cast<double>(got), static_cast<double>(expected));
        std::exit(2);
    }
}

template <typename T>
void bench_backend(gpudb::Backend b, const std::vector<T>& data,
                   int runs, T expected_sum, bool verify, Mode mode) {
    constexpr bool is_i64 = std::is_same_v<T, std::int64_t>;
    const std::size_t bytes = data.size() * sizeof(T);
    const double mib = bytes / (1024.0 * 1024.0);

    std::printf("\n[%s] rows=%zu (%.1f MiB)\n", gpudb::to_string(b), data.size(), mib);
    std::unique_ptr<gpudb::Aggregator> agg;
    try { agg = gpudb::make_aggregator(b); }
    catch (const std::exception& e) { std::printf("  unavailable: %s\n", e.what()); return; }
    std::printf("  device: %s\n", agg->device_name().c_str());

    // ---------- cold mode ----------
    if (mode == Mode::Cold || mode == Mode::Both) {
        std::vector<double> wall, kernel, xfer;
        for (int i = 0; i < runs; ++i) {
            gpudb::AggResult r;
            if constexpr (is_i64) r = agg->sum_i64(data.data(), data.size());
            else                  r = agg->sum_f64(data.data(), data.size());
            wall.push_back(r.wall_ms);
            kernel.push_back(r.kernel_ms);
            xfer.push_back(r.transfer_ms);
            if (i == 0 && verify) {
                if constexpr (is_i64) verify_or_die(r.value_i64, expected_sum, "cold");
                else                  verify_or_die(r.value_f64, expected_sum, "cold");
            }
        }
        std::printf("  COLD  median wall=%8.3f ms  kernel=%8.3f ms  xfer=%8.3f ms  "
                    "throughput=%6.2f GiB/s\n",
                    median(wall), median(kernel), median(xfer),
                    gibps(bytes, median(wall)));
    }

    // ---------- hot mode ----------
    if (mode == Mode::Hot || mode == Mode::Both) {
        std::unique_ptr<gpudb::ResidentColumn> col;
        if constexpr (is_i64) col = agg->upload_i64(data.data(), data.size());
        else                  col = agg->upload_f64(data.data(), data.size());

        // First call: verify correctness against the reference.
        gpudb::AggResult first;
        if constexpr (is_i64) first = agg->sum_resident_i64(*col);
        else                  first = agg->sum_resident_f64(*col);
        if (verify) {
            if constexpr (is_i64) verify_or_die(first.value_i64, expected_sum, "hot");
            else                  verify_or_die(first.value_f64, expected_sum, "hot");
        }

        std::vector<double> wall, kernel;
        for (int i = 0; i < runs; ++i) {
            gpudb::AggResult r;
            if constexpr (is_i64) r = agg->sum_resident_i64(*col);
            else                  r = agg->sum_resident_f64(*col);
            wall.push_back(r.wall_ms);
            kernel.push_back(r.kernel_ms);
        }
        const double w = median(wall), k = median(kernel);
        std::printf("  HOT   median wall=%8.3f ms  kernel=%8.3f ms  xfer=   0.000 ms  "
                    "throughput=%6.2f GiB/s",
                    w, k, gibps(bytes, w));
        if (k > 0.0) std::printf("  (kernel-only %6.2f GiB/s)", gibps(bytes, k));
        std::printf("\n");
    }
}

// Hybrid bench: exercise the hybrid path. Reports the dispatch decision the
// planner made on each first call, plus median wall over `runs`.
template <typename T>
void bench_hybrid(const std::vector<T>& data, int runs, T expected_sum,
                  bool verify, Mode mode) {
    constexpr bool is_i64 = std::is_same_v<T, std::int64_t>;
    const std::size_t bytes = data.size() * sizeof(T);
    const double mib = bytes / (1024.0 * 1024.0);

    std::printf("\n[Hybrid] rows=%zu (%.1f MiB)\n", data.size(), mib);
    auto agg = gpudb::make_hybrid_aggregator();
    std::printf("  device: %s\n", agg->device_name().c_str());

    if (mode == Mode::Cold || mode == Mode::Both) {
        std::vector<double> wall;
        gpudb::DispatchDecision decision{};
        for (int i = 0; i < runs; ++i) {
            gpudb::AggResult r;
            if constexpr (is_i64) r = agg->sum_i64(data.data(), data.size());
            else                  r = agg->sum_f64(data.data(), data.size());
            wall.push_back(r.wall_ms);
            if (i == 0) {
                decision = agg->last_decision();
                if (verify) {
                    if constexpr (is_i64) verify_or_die(r.value_i64, expected_sum, "hybrid cold");
                    else                  verify_or_die(r.value_f64, expected_sum, "hybrid cold");
                }
            }
        }
        std::printf("  COLD  median wall=%8.3f ms  throughput=%6.2f GiB/s   "
                    "picked=%s reason=%s\n",
                    median(wall), gibps(bytes, median(wall)),
                    gpudb::to_string(decision.chosen),
                    gpudb::to_string(decision.reason));
    }

    if (mode == Mode::Hot || mode == Mode::Both) {
        std::unique_ptr<gpudb::ResidentColumn> col;
        if constexpr (is_i64) col = agg->upload_i64(data.data(), data.size());
        else                  col = agg->upload_f64(data.data(), data.size());

        gpudb::AggResult first;
        if constexpr (is_i64) first = agg->sum_resident_i64(*col);
        else                  first = agg->sum_resident_f64(*col);
        if (verify) {
            if constexpr (is_i64) verify_or_die(first.value_i64, expected_sum, "hybrid hot");
            else                  verify_or_die(first.value_f64, expected_sum, "hybrid hot");
        }
        const auto decision = agg->last_decision();

        std::vector<double> wall;
        for (int i = 0; i < runs; ++i) {
            gpudb::AggResult r;
            if constexpr (is_i64) r = agg->sum_resident_i64(*col);
            else                  r = agg->sum_resident_f64(*col);
            wall.push_back(r.wall_ms);
        }
        std::printf("  HOT   median wall=%8.3f ms  throughput=%6.2f GiB/s   "
                    "picked=%s reason=%s\n",
                    median(wall), gibps(bytes, median(wall)),
                    gpudb::to_string(decision.chosen),
                    gpudb::to_string(decision.reason));
    }
}

} // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parse(argc, argv, a)) return 1;

    std::printf("gpudb-bench  mode=%s  runs=%d\n",
                a.mode == Mode::Cold ? "cold" : a.mode == Mode::Hot ? "hot" : "both", a.runs);

    auto backends = gpudb::available_backends();
    std::printf("backends:");
    for (auto b : backends) std::printf(" %s", gpudb::to_string(b));
    std::printf("\n");

    auto run_for_data_i64 = [&](const std::vector<std::int64_t>& data, std::int64_t ref) {
        if (a.bsel == BackendSel::Hybrid) {
            bench_hybrid(data, a.runs, ref, a.verify, a.mode);
        } else {
            for (auto b : backends) bench_backend(b, data, a.runs, ref, a.verify, a.mode);
        }
    };
    auto run_for_data_f64 = [&](const std::vector<double>& data, double ref) {
        if (a.bsel == BackendSel::Hybrid) {
            bench_hybrid(data, a.runs, ref, a.verify, a.mode);
        } else {
            for (auto b : backends) bench_backend(b, data, a.runs, ref, a.verify, a.mode);
        }
    };

    if (!a.input.empty()) {
        std::ifstream f(a.input, std::ios::binary);
        if (!f) { std::fprintf(stderr, "cannot open %s\n", a.input.c_str()); return 1; }
        gpudb::DataHeader h{};
        f.read(reinterpret_cast<char*>(&h), sizeof(h));
        if (std::memcmp(h.magic, gpudb::kMagic, 8) != 0) {
            std::fprintf(stderr, "%s: bad magic\n", a.input.c_str()); return 1;
        }
        std::printf("input: %s  rows=%llu  dtype=%s\n", a.input.c_str(),
                    static_cast<unsigned long long>(h.count),
                    h.dtype == 0 ? "i64" : "f64");
        if (h.dtype == 0) {
            std::vector<std::int64_t> data(h.count);
            f.read(reinterpret_cast<char*>(data.data()),
                   static_cast<std::streamsize>(data.size() * sizeof(std::int64_t)));
            const auto ref = host_sum(data);
            run_for_data_i64(data, ref);
        } else {
            std::vector<double> data(h.count);
            f.read(reinterpret_cast<char*>(data.data()),
                   static_cast<std::streamsize>(data.size() * sizeof(double)));
            const auto ref = host_sum(data);
            run_for_data_f64(data, ref);
        }
    } else {
        std::printf("synthetic: rows=%zu dtype=%s\n",
                    a.rows, a.dtype == gpudb::Dtype::I64 ? "i64" : "f64");
        std::mt19937_64 rng(0xC0FFEEULL);
        if (a.dtype == gpudb::Dtype::I64) {
            std::uniform_int_distribution<std::int64_t> dist(-1'000'000, 1'000'000);
            std::vector<std::int64_t> data(a.rows);
            std::int64_t ref = 0;
            for (auto& x : data) { x = dist(rng); ref += x; }
            run_for_data_i64(data, ref);
        } else {
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            std::vector<double> data(a.rows);
            double ref = 0.0;
            for (auto& x : data) { x = dist(rng); ref += x; }
            run_for_data_f64(data, ref);
        }
    }

    std::printf("\ndone.\n");
    return 0;
}
