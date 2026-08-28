// Minimal hand-rolled test runner — avoids pulling in Catch2 for week 1.
// Returns nonzero on failure so `ctest` and CI can pick it up.

#include "gpu_backend.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

void test_hashjoin();
int test_hashjoin_failures();
int test_hashjoin_total();

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

    // ---- Resident GROUP BY / top-k (v0.6) vs a host reference ----
    // Backends opt in; a "not implemented" throw is reported as SKIP so a
    // backend builds green before its implementation lands.
    {
        std::printf("  resident group by / top-k:\n");
        std::mt19937_64 rng(0x6B0BULL);
        const std::size_t N = 300'007;              // odd, > one chunk
        const std::int64_t K = 4'999;               // dup-heavy keys
        std::vector<std::int64_t> keys(N), vals(N);
        std::vector<double> dv(N);
        std::uniform_int_distribution<std::int64_t> kd(-K, K), vd(-1'000'000, 1'000'000);
        for (std::size_t i = 0; i < N; ++i) {
            keys[i] = kd(rng); vals[i] = vd(rng); dv[i] = static_cast<double>(vals[i]) / 7.0;
        }
        // int64 boundary keys + values that wrap
        keys[0] = std::numeric_limits<std::int64_t>::min(); vals[0] = std::numeric_limits<std::int64_t>::max();
        keys[1] = std::numeric_limits<std::int64_t>::min(); vals[1] = 5;   // wraps
        keys[2] = std::numeric_limits<std::int64_t>::max(); vals[2] = -3;
        std::map<std::int64_t, std::pair<std::uint64_t, std::int64_t>> ref;
        std::map<std::int64_t, double> ref_f;
        for (std::size_t i = 0; i < N; ++i) {
            auto& e = ref[keys[i]];
            e.first += static_cast<std::uint64_t>(vals[i]); e.second += 1;
            ref_f[keys[i]] += dv[i];
        }
        auto kc = agg->upload_i64(keys.data(), N);
        auto vc = agg->upload_i64(vals.data(), N);
        auto fc = agg->upload_f64(dv.data(), N);
        bool implemented = true;
        try {
            auto r = agg->groupby_sum_resident_i64(*kc, *vc, std::size_t(100) * 1000000);
            EXPECT_EQ(r.keys.size(), ref.size());
            EXPECT_EQ(r.rows_in, N);
            bool ok = r.keys.size() == ref.size();
            std::size_t j = 0;
            for (auto it = ref.begin(); ok && it != ref.end(); ++it, ++j) {
                ok = r.keys[j] == it->first &&
                     r.sums[j] == static_cast<std::int64_t>(it->second.first) &&
                     r.counts[j] == it->second.second;
            }
            EXPECT(ok);   // sorted ascending, bit-exact sums, exact counts

            auto c = agg->groupby_count_resident(*kc, std::size_t(100) * 1000000);
            ok = c.keys.size() == ref.size();
            j = 0;
            for (auto it = ref.begin(); ok && it != ref.end(); ++it, ++j)
                ok = c.keys[j] == it->first && c.counts[j] == it->second.second;
            EXPECT(ok);

            auto f = agg->groupby_sum_resident_f64(*kc, *fc, std::size_t(100) * 1000000);
            ok = f.keys.size() == ref_f.size();
            j = 0;
            for (auto it = ref_f.begin(); ok && it != ref_f.end(); ++it, ++j) {
                const double tol = 1e-9 * std::max(1.0, std::abs(it->second));
                ok = f.keys[j] == it->first && std::abs(f.sums_f64[j] - it->second) <= tol;
            }
            EXPECT(ok);

            // cap: must throw naming the count, never truncate
            bool threw = false;
            try { (void)agg->groupby_count_resident(*kc, 10); }
            catch (const std::runtime_error& e) {
                threw = std::string(e.what()).find(std::to_string(ref.size())) != std::string::npos;
            }
            EXPECT(threw);

            // top-k: multiset of values equals the reference's k extremes
            std::vector<std::int64_t> sv(vals);
            std::sort(sv.begin(), sv.end());
            auto t = agg->topk_resident(*vc, 100, /*descending*/true);
            ok = t.values_i64.size() == 100;
            for (std::size_t i = 0; ok && i < 100; ++i)
                ok = t.values_i64[i] == sv[N - 1 - i] && vals[static_cast<std::size_t>(t.idx[i])] == t.values_i64[i];
            EXPECT(ok);
            auto ta = agg->topk_resident(*vc, 5, /*descending*/false);
            ok = ta.values_i64.size() == 5;
            for (std::size_t i = 0; ok && i < 5; ++i) ok = ta.values_i64[i] == sv[i];
            EXPECT(ok);
            auto tf = agg->topk_resident(*fc, 3, /*descending*/true);
            std::vector<double> sdv(dv);
            std::sort(sdv.begin(), sdv.end());
            ok = tf.values_f64.size() == 3;
            for (std::size_t i = 0; ok && i < 3; ++i) ok = tf.values_f64[i] == sdv[N - 1 - i];
            EXPECT(ok);
            auto tk = agg->topk_resident(*vc, N + 10, false);   // k clamps to rows
            EXPECT_EQ(tk.idx.size(), N);

            // Regression: keys whose min and max share a low byte while a
            // key between them does not (0x4146, 0x4E46, 0x4E4F, 0x5246 —
            // TPC-H returnflag/linestatus packed). A radix sort that skips
            // "constant" byte passes based on min/max alone breaks here.
            {
                const std::int64_t kset[4] = {16710, 20038, 20047, 21062};
                const std::size_t M = 50'000;
                std::vector<std::int64_t> mk(M), mv(M);
                std::map<std::int64_t, std::pair<std::uint64_t, std::int64_t>> mref;
                for (std::size_t i = 0; i < M; ++i) {
                    const std::uint64_t h = (static_cast<std::uint64_t>(i) * 2654435761ull) % 100;
                    mk[i] = h < 50 ? kset[2] : h < 51 ? kset[1] : h < 75 ? kset[0] : kset[3];
                    mv[i] = static_cast<std::int64_t>(i);
                    auto& e = mref[mk[i]];
                    e.first += static_cast<std::uint64_t>(mv[i]); e.second += 1;
                }
                auto mkc = agg->upload_i64(mk.data(), M);
                auto mvc = agg->upload_i64(mv.data(), M);
                auto mr = agg->groupby_sum_resident_i64(*mkc, *mvc, std::size_t(100) * 1000000);
                bool mok = mr.keys.size() == mref.size();
                std::size_t mj = 0;
                for (auto it = mref.begin(); mok && it != mref.end(); ++it, ++mj)
                    mok = mr.keys[mj] == it->first &&
                          mr.sums[mj] == static_cast<std::int64_t>(it->second.first) &&
                          mr.counts[mj] == it->second.second;
                EXPECT(mok);
                // and the join over the same build keys (shares the sort cache)
                std::vector<std::int64_t> pk{16710, 20038, 20047, 21062, 1, 20047};
                std::vector<std::int64_t> pv{1, 10, 100, 1000, 7, 100};
                auto pkc = agg->upload_i64(pk.data(), pk.size());
                auto pvc = agg->upload_i64(pv.data(), pv.size());
                std::int64_t jref = 0;
                for (std::size_t i = 0; i < pk.size(); ++i)
                    if (mref.count(pk[i])) jref += pv[i] * mref[pk[i]].second;
                try {
                    auto jr = agg->join_sum_resident_i64(*pkc, *pvc, *mkc, gpudb::JoinKind::INNER);
                    EXPECT_EQ(jr.sum, jref);
                } catch (const std::runtime_error& e) {
                    if (std::string(e.what()).find("not implemented") == std::string::npos) throw;
                }
            }
        } catch (const std::runtime_error& e) {
            if (std::string(e.what()).find("not implemented") != std::string::npos) {
                implemented = false;
                std::printf("    SKIP (%s)\n", e.what());
            } else {
                std::printf("    FAIL: %s\n", e.what());
                ++failures; ++total;
            }
        }
        if (implemented) std::printf("    ok\n");
    }
}

} // namespace

// =====================================================================
//  Hybrid planner tests (GOAL.md item 7)
// =====================================================================
//
// We verify (a) correctness — hybrid produces the same results as the
// pure backends, and (b) the dispatch decision matches our documented
// rule for canonical workloads.

void test_hybrid_aggregator() {
    std::printf("\n--- testing HybridAggregator ---\n");
    auto h = gpudb::make_hybrid_aggregator();
    std::printf("  device: %s\n", h->device_name().c_str());
    std::printf("  gpu_backend: %s\n", gpudb::to_string(h->gpu_backend()));

    // Case 1: tiny N (< 100K) → CPU regardless of GPU availability.
    {
        std::vector<std::int64_t> v(50'000, 7);
        auto r = h->sum_i64(v.data(), v.size());
        EXPECT_EQ(r.value_i64, std::int64_t{350'000});
        const auto& d = h->last_decision();
        EXPECT_EQ(d.chosen, gpudb::Backend::CPU);
        EXPECT(d.reason == gpudb::DispatchReason::SmallN_CpuWins
            || d.reason == gpudb::DispatchReason::GpuUnavailable);
    }

    // Case 2: mid N (1M, cold) → CPU (below the 100M cold-GPU break-even).
    {
        std::mt19937_64 rng(0xCAFEULL);
        std::uniform_int_distribution<std::int64_t> dist(-1'000'000, 1'000'000);
        const std::size_t N = 1'000'000;
        std::vector<std::int64_t> v(N);
        std::int64_t ref = 0;
        for (auto& x : v) { x = dist(rng); ref += x; }
        auto r = h->sum_i64(v.data(), N);
        EXPECT_EQ(r.value_i64, ref);
        const auto& d = h->last_decision();
        EXPECT_EQ(d.chosen, gpudb::Backend::CPU);
        EXPECT(d.reason == gpudb::DispatchReason::Cold_BelowGpuBreakeven
            || d.reason == gpudb::DispatchReason::GpuUnavailable);
    }

    // Case 3: HOT (resident column) on GPU → GPU wins (when available).
    if (h->gpu_backend() != gpudb::Backend::CPU) {
        std::vector<std::int64_t> v(500'000);
        std::int64_t ref = 0;
        for (std::size_t i = 0; i < v.size(); ++i) { v[i] = static_cast<std::int64_t>(i); ref += v[i]; }
        auto col = h->upload_i64(v.data(), v.size());
        auto r = h->sum_resident_i64(*col);
        EXPECT_EQ(r.value_i64, ref);
        const auto& d = h->last_decision();
        EXPECT_EQ(d.chosen, h->gpu_backend());
        EXPECT(d.was_resident);
        EXPECT(d.reason == gpudb::DispatchReason::Hot_GpuAlwaysWins);
    }

    // Case 4: f64 sum → always CPU (no GPU doubles).
    {
        const std::size_t N = 200'000;
        std::vector<double> v(N, 1.5);
        auto r = h->sum_f64(v.data(), N);
        const double err = std::abs(r.value_f64 - 1.5 * static_cast<double>(N));
        EXPECT(err < 1e-6 * 1.5 * N);
        const auto& d = h->last_decision();
        EXPECT_EQ(d.chosen, gpudb::Backend::CPU);
        EXPECT(d.reason == gpudb::DispatchReason::F64_NoGpuDoubles);
    }

    // Case 5: ResidentColumn from a non-hybrid aggregator must be rejected.
    {
        auto cpu = gpudb::make_aggregator(gpudb::Backend::CPU);
        std::vector<std::int64_t> v{1, 2, 3};
        auto col = cpu->upload_i64(v.data(), v.size());
        bool threw = false;
        try { (void)h->sum_resident_i64(*col); }
        catch (const std::exception&) { threw = true; }
        EXPECT(threw);
    }
}

void test_hybrid_groupby() {
    std::printf("\n--- testing HybridGroupByAggregator ---\n");
    auto h = gpudb::make_hybrid_groupby_aggregator();
    std::printf("  device: %s\n", h->device_name().c_str());
    std::printf("  gpu_backend: %s\n", gpudb::to_string(h->gpu_backend()));

    auto build_keys_values = [](std::size_t n, std::size_t groups,
                                std::vector<std::int64_t>& keys,
                                std::vector<std::int64_t>& vals) {
        std::mt19937_64 rng(0xBEEFULL);
        std::uniform_int_distribution<std::int64_t> kd(0, std::max<std::size_t>(1, groups) - 1);
        std::uniform_int_distribution<std::int64_t> vd(-1000, 1000);
        keys.resize(n); vals.resize(n);
        for (std::size_t i = 0; i < n; ++i) { keys[i] = kd(rng); vals[i] = vd(rng); }
    };

    auto verify_against_cpu = [&](const std::vector<std::int64_t>& keys,
                                  const std::vector<std::int64_t>& vals,
                                  std::size_t expected_groups) {
        auto got = h->groupby_sum_i64(keys.data(), vals.data(), keys.size(), expected_groups);
        auto cpu = gpudb::make_groupby_aggregator(gpudb::Backend::CPU);
        auto ref = cpu->groupby_sum_i64(keys.data(), vals.data(), keys.size(), expected_groups);
        EXPECT_EQ(got.keys.size(), ref.keys.size());
        // Sort and compare pairs.
        auto pair_lt = [](const std::pair<std::int64_t,std::int64_t>& a,
                          const std::pair<std::int64_t,std::int64_t>& b){ return a.first < b.first; };
        std::vector<std::pair<std::int64_t,std::int64_t>> pa, pb;
        for (std::size_t i = 0; i < got.keys.size(); ++i) pa.emplace_back(got.keys[i], got.sums[i]);
        for (std::size_t i = 0; i < ref.keys.size(); ++i) pb.emplace_back(ref.keys[i], ref.sums[i]);
        std::sort(pa.begin(), pa.end(), pair_lt);
        std::sort(pb.begin(), pb.end(), pair_lt);
        EXPECT(pa == pb);
        return h->last_decision();
    };

    // Case 1: tiny N (50K) → CPU outright.
    {
        std::vector<std::int64_t> k, v;
        build_keys_values(50'000, 100, k, v);
        auto d = verify_against_cpu(k, v, 100);
        EXPECT_EQ(d.chosen, gpudb::Backend::CPU);
        EXPECT(d.reason == gpudb::DispatchReason::GroupBy_LowCard_CpuWins
            || d.reason == gpudb::DispatchReason::GpuUnavailable);
    }

    // Case 2: low-cardinality + small/mid N (1M rows × 1024 groups) → CPU.
    {
        std::vector<std::int64_t> k, v;
        build_keys_values(1'000'000, 1024, k, v);
        auto d = verify_against_cpu(k, v, 1024);
        EXPECT_EQ(d.chosen, gpudb::Backend::CPU);
        EXPECT(d.reason == gpudb::DispatchReason::GroupBy_LowCard_CpuWins
            || d.reason == gpudb::DispatchReason::GpuUnavailable);
    }

    // Case 3: high-cardinality (1M rows × 1M groups, ratio = 1.0) → GPU.
    if (h->gpu_backend() != gpudb::Backend::CPU) {
        std::vector<std::int64_t> k, v;
        build_keys_values(1'000'000, 1'000'000, k, v);
        auto d = verify_against_cpu(k, v, 1'000'000);
        EXPECT_EQ(d.chosen, h->gpu_backend());
        EXPECT(d.reason == gpudb::DispatchReason::GroupBy_HighCard_GpuWins);
    }

    // Case 4: mid regime (1M rows × 50K groups, ratio = 0.05).
    // The sweep showed CPU wins this cell, so the planner routes CPU but
    // still flags `borderline` so the threshold can be re-tuned later.
    if (h->gpu_backend() != gpudb::Backend::CPU) {
        std::vector<std::int64_t> k, v;
        build_keys_values(1'000'000, 50'000, k, v);
        auto d = verify_against_cpu(k, v, 50'000);
        EXPECT_EQ(d.chosen, gpudb::Backend::CPU);
        EXPECT(d.reason == gpudb::DispatchReason::GroupBy_Borderline_GpuTry);
        EXPECT(d.borderline);
    }

    // Case 5: above-sweet-spot N (5M rows × 100K groups) → CPU
    // (bitonic O(N log²N) loses to hash O(N) on M4 Max past 2M rows).
    if (h->gpu_backend() != gpudb::Backend::CPU) {
        std::vector<std::int64_t> k, v;
        build_keys_values(5'000'000, 100'000, k, v);
        auto r = h->groupby_sum_i64(k.data(), v.data(), k.size(), 100'000);
        EXPECT_EQ(r.input_rows, k.size());
        const auto& d = h->last_decision();
        EXPECT_EQ(d.chosen, gpudb::Backend::CPU);
        EXPECT(d.reason == gpudb::DispatchReason::GroupBy_HugeN_CpuWins);
    }
}

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

    test_hybrid_aggregator();
    test_hybrid_groupby();
    test_hashjoin();

    failures += test_hashjoin_failures();
    total += test_hashjoin_total();

    std::printf("\n%d / %d checks passed\n", total - failures, total);
    return failures == 0 ? 0 : 1;
}
