// gpudb-bench — microbenchmark SUM/MIN/MAX across all available backends.
// Modes:
//   synthetic:  --rows N [--dtype i64|f64]
//   from file:  --input data/foo.gpudb
//
// Examples:
//   ./gpudb-bench --rows 100000000
//   ./gpudb-bench --input data/sf1_quantity.gpudb --runs 10

#include "data_format.hpp"
#include "gpu_backend.hpp"

#include <algorithm>
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

struct Args {
    std::size_t rows = 10'000'000;
    int         runs = 5;
    bool        verify = true;
    gpudb::Dtype dtype = gpudb::Dtype::I64;
    std::string input;  // if non-empty, load from file
};

void usage(const char* a0) {
    std::fprintf(stderr,
        "usage: %s [--rows N] [--dtype i64|f64] [--input FILE] [--runs K] [--no-verify]\n", a0);
}

bool parse(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing arg for %s\n", what); std::exit(1); }
            return argv[++i];
        };
        if      (s == "--rows")   a.rows = std::strtoull(next("--rows").c_str(), nullptr, 10);
        else if (s == "--runs")   a.runs = std::atoi(next("--runs").c_str());
        else if (s == "--no-verify") a.verify = false;
        else if (s == "--input")  a.input = next("--input");
        else if (s == "--dtype") {
            auto t = next("--dtype");
            if (t == "i64") a.dtype = gpudb::Dtype::I64;
            else if (t == "f64") a.dtype = gpudb::Dtype::F64;
            else { std::fprintf(stderr, "unknown dtype: %s\n", t.c_str()); return false; }
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

template <typename T>
std::vector<T> load_column(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(1); }
    gpudb::DataHeader h{};
    f.read(reinterpret_cast<char*>(&h), sizeof(h));
    if (std::memcmp(h.magic, gpudb::kMagic, 8) != 0) {
        std::fprintf(stderr, "%s: bad magic\n", path.c_str()); std::exit(1);
    }
    std::vector<T> out(h.count);
    f.read(reinterpret_cast<char*>(out.data()),
           static_cast<std::streamsize>(out.size() * sizeof(T)));
    return out;
}

template <typename T>
T host_sum(const std::vector<T>& v) {
    T acc = T{};
    for (auto x : v) acc += x;
    return acc;
}

void bench_one_i64(gpudb::Backend b, const std::vector<std::int64_t>& data,
                   int runs, std::int64_t expected_sum, bool verify) {
    std::printf("\n[%s]\n", gpudb::to_string(b));
    std::unique_ptr<gpudb::Aggregator> agg;
    try { agg = gpudb::make_aggregator(b); }
    catch (const std::exception& e) { std::printf("  unavailable: %s\n", e.what()); return; }
    std::printf("  device: %s\n", agg->device_name().c_str());

    auto warm = agg->sum_i64(data.data(), data.size());
    if (verify && warm.value_i64 != expected_sum) {
        std::fprintf(stderr, "  CORRECTNESS FAIL: got %lld, expected %lld\n",
                     static_cast<long long>(warm.value_i64),
                     static_cast<long long>(expected_sum));
        std::exit(2);
    }

    std::vector<double> wall, kernel, xfer;
    for (int i = 0; i < runs; ++i) {
        auto r = agg->sum_i64(data.data(), data.size());
        wall.push_back(r.wall_ms);
        kernel.push_back(r.kernel_ms);
        xfer.push_back(r.transfer_ms);
    }
    const double w = median(wall), k = median(kernel), x = median(xfer);
    const std::size_t bytes = data.size() * sizeof(std::int64_t);
    std::printf("  SUM i64  median wall=%8.3f ms  kernel=%8.3f ms  xfer=%8.3f ms  "
                "throughput=%6.2f GiB/s\n",
                w, k, x, gibps(bytes, w));
    if (k > 0.0) std::printf("           kernel-only throughput = %6.2f GiB/s\n",
                             gibps(bytes, k));
}

void bench_one_f64(gpudb::Backend b, const std::vector<double>& data,
                   int runs, double expected_sum, bool verify) {
    std::printf("\n[%s]\n", gpudb::to_string(b));
    std::unique_ptr<gpudb::Aggregator> agg;
    try { agg = gpudb::make_aggregator(b); }
    catch (const std::exception& e) { std::printf("  unavailable: %s\n", e.what()); return; }
    std::printf("  device: %s\n", agg->device_name().c_str());

    auto warm = agg->sum_f64(data.data(), data.size());
    if (verify) {
        const double err = std::abs(warm.value_f64 - expected_sum);
        const double tol = 1e-6 * std::max(1.0, std::abs(expected_sum));
        if (err > tol) {
            std::fprintf(stderr, "  CORRECTNESS FAIL: got %g, expected %g (err=%g)\n",
                         warm.value_f64, expected_sum, err);
            std::exit(2);
        }
    }

    std::vector<double> wall, kernel, xfer;
    for (int i = 0; i < runs; ++i) {
        auto r = agg->sum_f64(data.data(), data.size());
        wall.push_back(r.wall_ms);
        kernel.push_back(r.kernel_ms);
        xfer.push_back(r.transfer_ms);
    }
    const double w = median(wall), k = median(kernel), x = median(xfer);
    const std::size_t bytes = data.size() * sizeof(double);
    std::printf("  SUM f64  median wall=%8.3f ms  kernel=%8.3f ms  xfer=%8.3f ms  "
                "throughput=%6.2f GiB/s\n",
                w, k, x, gibps(bytes, w));
    if (k > 0.0) std::printf("           kernel-only throughput = %6.2f GiB/s\n",
                             gibps(bytes, k));
}

} // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parse(argc, argv, a)) return 1;

    std::printf("gpudb-bench\n");

    auto backends = gpudb::available_backends();
    std::printf("backends: ");
    for (auto b : backends) std::printf("%s ", gpudb::to_string(b));
    std::printf("\n");

    if (!a.input.empty()) {
        // Load file; dispatch by header dtype.
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
            for (auto b : backends) bench_one_i64(b, data, a.runs, ref, a.verify);
        } else {
            std::vector<double> data(h.count);
            f.read(reinterpret_cast<char*>(data.data()),
                   static_cast<std::streamsize>(data.size() * sizeof(double)));
            const auto ref = host_sum(data);
            for (auto b : backends) bench_one_f64(b, data, a.runs, ref, a.verify);
        }
    } else {
        std::printf("synthetic: rows=%zu dtype=%s runs=%d\n",
                    a.rows, a.dtype == gpudb::Dtype::I64 ? "i64" : "f64", a.runs);
        std::mt19937_64 rng(0xC0FFEEULL);
        if (a.dtype == gpudb::Dtype::I64) {
            std::uniform_int_distribution<std::int64_t> dist(-1'000'000, 1'000'000);
            std::vector<std::int64_t> data(a.rows);
            std::int64_t ref = 0;
            for (auto& x : data) { x = dist(rng); ref += x; }
            for (auto b : backends) bench_one_i64(b, data, a.runs, ref, a.verify);
        } else {
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            std::vector<double> data(a.rows);
            double ref = 0.0;
            for (auto& x : data) { x = dist(rng); ref += x; }
            for (auto b : backends) bench_one_f64(b, data, a.runs, ref, a.verify);
        }
    }

    std::printf("\ndone.\n");
    return 0;
}
