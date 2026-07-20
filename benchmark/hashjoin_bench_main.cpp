// gpudb-hashjoin-bench — inner equi-join on int64 keys, CPU vs GPU.
//
// Usage:
//   gpudb-hashjoin-bench [--build M] [--probe N] [--runs K] [--skew z]
//                        [--no-verify]
//
// Synthetic mode:
//   - Build side: M unique int64 keys, drawn from a wide domain
//     (kBuildDomainStride * j to keep them sparse and exercise hashing).
//   - Probe side: N int64 keys drawn from the same M-key set, with
//     optional Zipf-like skew controlled by --skew (0 = uniform). At skew
//     > 0 we draw the rank with a power-law-ish bias so a few keys appear
//     often (the realistic FK distribution after a fact-table grows).
//
// Verification:
//   - Run on Backend::CPU once to get the reference JoinResult.
//   - Run on every other available backend; canonicalize the matched-pair
//     vectors via std::sort and compare. Mismatch is exit code 2.

#include "gpu_backend.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

struct Args {
    std::size_t n_build = 1'000'000;
    std::size_t n_probe = 10'000'000;
    int         runs    = 3;
    double      skew    = 0.0;     // 0 = uniform; >0 biases toward small ranks
    bool        verify  = true;
};

void usage(const char* a0) {
    std::fprintf(stderr,
        "usage: %s [--build M] [--probe N] [--runs K] [--skew z] [--no-verify]\n", a0);
}

bool parse(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&](const char* w) -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing arg for %s\n", w); std::exit(1); }
            return argv[++i];
        };
        if      (s == "--build")     a.n_build = std::strtoull(next("--build").c_str(), nullptr, 10);
        else if (s == "--probe")     a.n_probe = std::strtoull(next("--probe").c_str(), nullptr, 10);
        else if (s == "--runs")      a.runs    = std::atoi(next("--runs").c_str());
        else if (s == "--skew")      a.skew    = std::strtod(next("--skew").c_str(), nullptr);
        else if (s == "--no-verify") a.verify  = false;
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

// Compare two JoinResults for set-equality of (probe_idx, build_idx) pairs.
// We canonicalize via sort so that backend-specific output ordering doesn't
// matter — hash backends may emit in hash-table-iteration order, sort-merge
// backends emit in sorted-key order.
bool join_equals(const gpudb::JoinResult& a, const gpudb::JoinResult& b) {
    if (a.matched != b.matched) return false;
    if (a.probe_indices.size() != a.matched || a.build_indices.size() != a.matched) return false;
    if (b.probe_indices.size() != b.matched || b.build_indices.size() != b.matched) return false;

    using Pair = std::pair<std::int64_t, std::int64_t>;
    std::vector<Pair> pa, pb;
    pa.reserve(a.matched); pb.reserve(b.matched);
    for (std::size_t i = 0; i < a.matched; ++i) pa.emplace_back(a.probe_indices[i], a.build_indices[i]);
    for (std::size_t i = 0; i < b.matched; ++i) pb.emplace_back(b.probe_indices[i], b.build_indices[i]);
    std::sort(pa.begin(), pa.end());
    std::sort(pb.begin(), pb.end());
    return pa == pb;
}

void run_backend(gpudb::Backend b,
                 const std::vector<std::int64_t>& build,
                 const std::vector<std::int64_t>& probe,
                 int runs,
                 const gpudb::JoinResult* reference, bool verify) {
    const std::size_t bytes = (build.size() + probe.size()) * sizeof(std::int64_t);
    std::printf("\n[%s]\n", gpudb::to_string(b));
    std::unique_ptr<gpudb::HashJoinProbe> hj;
    try { hj = gpudb::make_hashjoin_probe(b); }
    catch (const std::exception& e) { std::printf("  unavailable: %s\n", e.what()); return; }
    std::printf("  device: %s\n", hj->device_name().c_str());

    auto first = hj->inner_join_i64(build.data(), build.size(),
                                    probe.data(), probe.size());
    std::printf("  matched: %zu / %zu probe rows (%.1f%%)\n",
                first.matched, probe.size(),
                probe.empty() ? 0.0 : 100.0 * static_cast<double>(first.matched) / probe.size());
    if (reference && verify) {
        if (!join_equals(first, *reference)) {
            std::fprintf(stderr, "  CORRECTNESS FAIL: result differs from CPU reference\n");
            std::exit(2);
        }
    }

    std::vector<double> wall, kernel, xfer;
    for (int i = 0; i < runs; ++i) {
        auto r = hj->inner_join_i64(build.data(), build.size(),
                                    probe.data(), probe.size());
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

    std::printf("gpudb-hashjoin-bench  build=%zu probe=%zu runs=%d skew=%.2f\n",
                a.n_build, a.n_probe, a.runs, a.skew);
    auto backends = gpudb::available_backends();
    std::printf("backends:");
    for (auto b : backends) std::printf(" %s", gpudb::to_string(b));
    std::printf("\n");

    // Build side: M unique sparse keys.
    constexpr std::int64_t kBuildDomainStride = 1024;
    std::vector<std::int64_t> build(a.n_build);
    for (std::size_t j = 0; j < a.n_build; ++j) {
        // Stride * (j+1) keeps the keys sparse and avoids the trivial
        // 0..M-1 distribution that hash functions would handle too easily.
        build[j] = kBuildDomainStride * static_cast<std::int64_t>(j + 1);
    }

    // Probe side: N keys drawn from build (so most probes will match), with
    // optional skew. Index into build is computed from a power-law draw when
    // skew > 0, otherwise uniform. We always include a small fraction of
    // probe keys that DO NOT exist in build, to exercise the no-match path.
    std::mt19937_64 rng(0xC0FFEEULL);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    std::vector<std::int64_t> probe(a.n_probe);
    const double inv_skew_norm = (a.skew > 0.0)
        ? 1.0 / (1.0 - std::pow(static_cast<double>(a.n_build), -a.skew))
        : 0.0;
    for (std::size_t i = 0; i < a.n_probe; ++i) {
        // ~5% of probe rows: a key guaranteed not to exist in build.
        if ((rng() & 0x1F) == 0) {
            probe[i] = -1 - static_cast<std::int64_t>(i);
            continue;
        }
        std::size_t rank;
        if (a.skew > 0.0) {
            // Inverse-CDF sample for a Pareto-ish distribution over [1, M].
            const double r = u01(rng);
            const double x = std::pow(1.0 - r * (1.0 - std::pow(static_cast<double>(a.n_build), -a.skew)),
                                      -1.0 / a.skew);
            rank = std::min<std::size_t>(static_cast<std::size_t>(x) - 1, a.n_build - 1);
        } else {
            rank = rng() % a.n_build;
        }
        probe[i] = build[rank];
    }
    (void)inv_skew_norm;  // documentation aid

    // CPU reference (always; cheap relative to bench timing).
    gpudb::JoinResult reference;
    {
        auto cpu = gpudb::make_hashjoin_probe(gpudb::Backend::CPU);
        reference = cpu->inner_join_i64(build.data(), build.size(),
                                        probe.data(), probe.size());
        std::printf("cpu reference: matched %zu / %zu (%.1f%%)\n",
                    reference.matched, probe.size(),
                    probe.empty() ? 0.0 : 100.0 * static_cast<double>(reference.matched) / probe.size());
    }

    for (auto b : backends) run_backend(b, build, probe, a.runs, &reference, a.verify);

    // Hybrid planner dispatch (CPU vs GPU threshold).
    {
        const std::size_t bytes = (build.size() + probe.size()) * sizeof(std::int64_t);
        std::printf("\n[Hybrid]\n");
        try {
            auto hj = gpudb::make_hybrid_hashjoin_probe();
            std::printf("  device: %s\n", hj->device_name().c_str());
            auto first = hj->inner_join_i64(build.data(), build.size(),
                                            probe.data(), probe.size());
            std::printf("  matched: %zu / %zu probe rows (%.1f%%)\n",
                        first.matched, probe.size(),
                        probe.empty() ? 0.0
                                      : 100.0 * static_cast<double>(first.matched) / probe.size());
            std::printf("  dispatch: %s (%s)\n",
                        gpudb::to_string(hj->last_decision().chosen),
                        gpudb::to_string(hj->last_decision().reason));
            if (!join_equals(first, reference)) {
                std::fprintf(stderr, "  CORRECTNESS FAIL: hybrid differs from CPU reference\n");
                std::exit(2);
            }
            std::vector<double> wall, kernel;
            for (int i = 0; i < a.runs; ++i) {
                auto r = hj->inner_join_i64(build.data(), build.size(),
                                            probe.data(), probe.size());
                wall.push_back(r.wall_ms);
                kernel.push_back(r.kernel_ms);
            }
            const double w = median(wall), k = median(kernel);
            std::printf("  median wall=%8.3f ms  kernel=%8.3f ms  throughput=%6.2f GiB/s\n",
                        w, k, gibps(bytes, w));
        } catch (const std::exception& e) {
            std::printf("  unavailable: %s\n", e.what());
        }
    }

    std::printf("\ndone.\n");
    return 0;
}
