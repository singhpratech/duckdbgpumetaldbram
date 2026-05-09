// gpudb-hashjoin-bench — inner equi-join on int64 keys, varying scales.
//
// Usage:
//   gpudb-hashjoin-bench [--build-rows N] [--probe-rows M] [--match-prob P]
//                        [--runs K] [--no-verify]
//                        [--build-keys FILE.gpudb] [--probe-keys FILE.gpudb]
//
// Synthetic mode: generates two int64 columns. The build side is a random
// permutation of [0, n_build). The probe side is constructed so that with
// probability `match-prob` the probe key is sampled uniformly from the
// build key set (guaranteed match), and with probability (1 - match-prob)
// it is sampled from a disjoint key range (guaranteed miss). This gives
// us deterministic control over selectivity for benchmarking.
//
// File mode: load build_keys and probe_keys from .gpudb int64 files.
// E.g. for TPC-H lineitem ⋈ orders on l_orderkey = o_orderkey, build is
// the orders.o_orderkey column and probe is lineitem.l_orderkey. (The
// orders column may need to be derived from the lineitem orderkey set
// via DISTINCT — see scripts/gen_tpch.sh patterns.)

#include "data_format.hpp"
#include "gpu_backend.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Args {
    std::size_t build_rows = 10'000'000;
    std::size_t probe_rows = 50'000'000;
    double      match_prob = 1.0;       // probability a probe key matches
    int         runs = 3;
    bool        verify = true;
    std::string build_keys_file;
    std::string probe_keys_file;
};

void usage(const char* a0) {
    std::fprintf(stderr,
        "usage: %s [--build-rows N] [--probe-rows M] [--match-prob P]\n"
        "           [--runs K] [--no-verify]\n"
        "           [--build-keys FILE.gpudb] [--probe-keys FILE.gpudb]\n",
        a0);
}

bool parse(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&](const char* w) -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing arg for %s\n", w); std::exit(1); }
            return argv[++i];
        };
        if      (s == "--build-rows")  a.build_rows = std::strtoull(next("--build-rows").c_str(), nullptr, 10);
        else if (s == "--probe-rows")  a.probe_rows = std::strtoull(next("--probe-rows").c_str(), nullptr, 10);
        else if (s == "--match-prob")  a.match_prob = std::strtod(next("--match-prob").c_str(), nullptr);
        else if (s == "--runs")        a.runs = std::atoi(next("--runs").c_str());
        else if (s == "--no-verify")   a.verify = false;
        else if (s == "--build-keys")  a.build_keys_file = next("--build-keys");
        else if (s == "--probe-keys")  a.probe_keys_file = next("--probe-keys");
        else if (s == "-h" || s == "--help") { usage(argv[0]); return false; }
        else { std::fprintf(stderr, "unknown arg: %s\n", s.c_str()); usage(argv[0]); return false; }
    }
    return true;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

double gibps(std::size_t bytes, double ms) {
    if (ms <= 0.0) return 0.0;
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

// Sort a list of (probe_idx, build_idx) pairs into canonical order.
std::vector<std::pair<std::int64_t, std::int64_t>>
to_sorted_pairs(const gpudb::HashJoinResult& r) {
    std::vector<std::pair<std::int64_t, std::int64_t>> out;
    out.reserve(r.probe_indices.size());
    for (std::size_t i = 0; i < r.probe_indices.size(); ++i) {
        out.emplace_back(r.probe_indices[i], r.build_indices[i]);
    }
    std::sort(out.begin(), out.end());
    return out;
}

bool result_equals(const gpudb::HashJoinResult& a, const gpudb::HashJoinResult& b) {
    if (a.probe_indices.size() != b.probe_indices.size()) return false;
    auto pa = to_sorted_pairs(a);
    auto pb = to_sorted_pairs(b);
    return pa == pb;
}

void run_backend(gpudb::Backend b,
                 const std::vector<std::int64_t>& build_keys,
                 const std::vector<std::int64_t>& probe_keys,
                 int runs,
                 const gpudb::HashJoinResult* reference, bool verify) {
    // Total bytes touched on input: keys for both sides.
    const std::size_t bytes_in = (build_keys.size() + probe_keys.size())
                               * sizeof(std::int64_t);
    std::printf("\n[%s]\n", gpudb::to_string(b));
    std::unique_ptr<gpudb::HashJoinAggregator> agg;
    try { agg = gpudb::make_hashjoin_aggregator(b); }
    catch (const std::exception& e) { std::printf("  unavailable: %s\n", e.what()); return; }
    std::printf("  device: %s\n", agg->device_name().c_str());

    // First call: verify.
    auto first = agg->inner_join_i64(build_keys.data(), build_keys.size(),
                                     probe_keys.data(), probe_keys.size());
    std::printf("  matches: %zu (probe=%zu, build=%zu)\n",
                first.probe_indices.size(),
                first.input_probe_rows, first.input_build_rows);
    if (reference && verify) {
        if (!result_equals(first, *reference)) {
            std::fprintf(stderr,
                "  CORRECTNESS FAIL: result differs from CPU reference\n"
                "    got %zu pairs, expected %zu\n",
                first.probe_indices.size(), reference->probe_indices.size());
            std::exit(2);
        } else {
            std::printf("  verified against CPU reference (sorted pair-set match).\n");
        }
    }

    std::vector<double> wall, kernel, xfer;
    for (int i = 0; i < runs; ++i) {
        auto r = agg->inner_join_i64(build_keys.data(), build_keys.size(),
                                     probe_keys.data(), probe_keys.size());
        wall.push_back(r.wall_ms);
        kernel.push_back(r.kernel_ms);
        xfer.push_back(r.transfer_ms);
    }
    const double w = median(wall), k = median(kernel), x = median(xfer);
    std::printf("  median wall=%9.3f ms  kernel=%9.3f ms  xfer=%9.3f ms  "
                "input throughput=%6.2f GiB/s",
                w, k, x, gibps(bytes_in, w));
    if (k > 0.0) std::printf("  (kernel %6.2f GiB/s)", gibps(bytes_in, k));
    std::printf("\n");
}

void synth_data(std::size_t n_build, std::size_t n_probe, double match_prob,
                std::vector<std::int64_t>& build_keys,
                std::vector<std::int64_t>& probe_keys) {
    // Build: random permutation of [1, n_build] (avoid INT64_MIN; positive
    // is fine). Each build key is unique → 1-to-many later.
    std::mt19937_64 rng(0xC0FFEEULL);
    build_keys.resize(n_build);
    std::iota(build_keys.begin(), build_keys.end(), 1);
    std::shuffle(build_keys.begin(), build_keys.end(), rng);

    // Probe: each row independently:
    //   - with probability match_prob: sample uniform from build_keys
    //     (guaranteed match)
    //   - else: sample from a disjoint range above n_build (guaranteed miss)
    probe_keys.resize(n_probe);
    std::uniform_int_distribution<std::size_t> idx_dist(0, n_build - 1);
    std::uniform_int_distribution<std::int64_t> miss_dist(
        static_cast<std::int64_t>(n_build) + 1,
        static_cast<std::int64_t>(n_build) * 4 + 1);
    std::uniform_real_distribution<double> coin(0.0, 1.0);
    for (auto& k : probe_keys) {
        if (coin(rng) < match_prob) {
            k = build_keys[idx_dist(rng)];
        } else {
            k = miss_dist(rng);
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parse(argc, argv, a)) return 1;

    std::vector<std::int64_t> build_keys, probe_keys;
    if (!a.build_keys_file.empty()) {
        load_int64(a.build_keys_file, build_keys);
        a.build_rows = build_keys.size();
        std::printf("loaded %zu build keys from %s\n",
                    build_keys.size(), a.build_keys_file.c_str());
    }
    if (!a.probe_keys_file.empty()) {
        load_int64(a.probe_keys_file, probe_keys);
        a.probe_rows = probe_keys.size();
        std::printf("loaded %zu probe keys from %s\n",
                    probe_keys.size(), a.probe_keys_file.c_str());
    }

    if (build_keys.empty() || probe_keys.empty()) {
        std::printf("synthetic mode: build=%zu probe=%zu match_prob=%.3f\n",
                    a.build_rows, a.probe_rows, a.match_prob);
        synth_data(a.build_rows, a.probe_rows, a.match_prob,
                   build_keys, probe_keys);
    }

    std::printf("gpudb-hashjoin-bench  build=%zu probe=%zu runs=%d\n",
                build_keys.size(), probe_keys.size(), a.runs);
    auto backends = gpudb::available_backends();
    std::printf("backends:");
    for (auto b : backends) std::printf(" %s", gpudb::to_string(b));
    std::printf("\n");

    // Build CPU reference once.
    gpudb::HashJoinResult reference;
    {
        auto cpu = gpudb::make_hashjoin_aggregator(gpudb::Backend::CPU);
        reference = cpu->inner_join_i64(build_keys.data(), build_keys.size(),
                                        probe_keys.data(), probe_keys.size());
        std::printf("cpu reference: %zu matched pairs (cpu wall=%.2f ms)\n",
                    reference.probe_indices.size(), reference.wall_ms);
    }

    for (auto b : backends) {
        // Skip Metal — it's owned by the macOS instance and not yet
        // implemented on this branch.
        if (b == gpudb::Backend::METAL) continue;
        run_backend(b, build_keys, probe_keys, a.runs, &reference, a.verify);
    }

    std::printf("\ndone.\n");
    return 0;
}
