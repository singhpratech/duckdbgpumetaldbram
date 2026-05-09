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
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

struct Args {
    std::size_t rows = 10'000'000;
    std::size_t groups = 1024;          // cardinality
    int         runs = 3;
    bool        verify = true;
    std::string input_keys;             // optional: load int64 keys from .gpudb
};

void usage(const char* a0) {
    std::fprintf(stderr,
        "usage: %s [--rows N] [--groups G] [--runs K] [--input-keys FILE.gpudb] [--no-verify]\n", a0);
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

void run_backend(gpudb::Backend b, const std::vector<std::int64_t>& keys,
                 const std::vector<std::int64_t>& values, int runs,
                 const gpudb::GroupByResult* reference, bool verify) {
    const std::size_t bytes = keys.size() * sizeof(std::int64_t) * 2;  // keys + values
    std::printf("\n[%s]\n", gpudb::to_string(b));
    std::unique_ptr<gpudb::GroupByAggregator> agg;
    try { agg = gpudb::make_groupby_aggregator(b); }
    catch (const std::exception& e) { std::printf("  unavailable: %s\n", e.what()); return; }
    std::printf("  device: %s\n", agg->device_name().c_str());

    // First call: verify.
    auto first = agg->groupby_sum_i64(keys.data(), values.data(), keys.size());
    std::printf("  groups produced: %zu\n", first.keys.size());
    if (reference && verify) {
        if (!result_equals(first, *reference)) {
            std::fprintf(stderr, "  CORRECTNESS FAIL: result differs from reference\n");
            std::exit(2);
        }
    }

    std::vector<double> wall, kernel, xfer;
    for (int i = 0; i < runs; ++i) {
        auto r = agg->groupby_sum_i64(keys.data(), values.data(), keys.size());
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

    std::printf("gpudb-groupby-bench  rows=%zu groups=%zu runs=%d\n",
                a.rows, a.groups, a.runs);
    auto backends = gpudb::available_backends();
    std::printf("backends:");
    for (auto b : backends) std::printf(" %s", gpudb::to_string(b));
    std::printf("\n");

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

    for (auto b : backends) run_backend(b, keys, values, a.runs, &reference, a.verify);

    std::printf("\ndone.\n");
    return 0;
}
