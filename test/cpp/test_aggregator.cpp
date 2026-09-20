// Minimal hand-rolled test runner — avoids pulling in Catch2 for week 1.
// Returns nonzero on failure so `ctest` and CI can pick it up.

#include "gpu_backend.hpp"
#include "native_avg.hpp"
#include "exact_path_note.hpp"
#include "resident_shed_note.hpp"
#include "../../src/backends/groupby_filter.hpp"
#include "../../src/backends/predicate_mask.hpp"

#include <algorithm>
#if defined(__linux__)
#include <sys/wait.h>
#include <unistd.h>
#endif
#include <cmath>
#include <cstring>
#include <functional>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <set>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <mutex>
#include <atomic>
#include <cfloat>
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
    // Every compiled backend implements it and must match the reference —
    // CUDA included since it grew the fused kernel (it used to throw here).
    {
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

            // GroupByFilter, adversarial cases (CUDA-side additions to the
            // "resident group by filter" block below): threshold equal to a
            // sum for all four comparisons, everything filtered, k > survivors,
            // k == 1, negative thresholds, heavy ties, cmp + top-k, f64 top-k,
            // count op, and the filtered cap wording.
            {
                std::printf("  resident group by filter, adversarial cases:\n");
                using Cmp = gpudb::GroupByFilter::Cmp;
                const std::size_t cap = std::size_t(100) * 1000000;
                auto base_i = agg->groupby_sum_resident_i64(*kc, *vc, cap);
                auto base_f = agg->groupby_sum_resident_f64(*kc, *fc, cap);
                auto base_c = agg->groupby_count_resident(*kc, cap);
                EXPECT_EQ(base_i.groups_total, ref.size());
                const std::int64_t eq_sum = base_i.sums[base_i.sums.size() / 3];
                const double       eq_f   = base_f.sums_f64[base_f.sums_f64.size() / 3];
                const std::int64_t max_sum = *std::max_element(base_i.sums.begin(), base_i.sums.end());
                const gpudb::GroupByFilter filters[] = {
                    {Cmp::GT, eq_sum, eq_f, 0, true}, {Cmp::GE, eq_sum, eq_f, 0, true},
                    {Cmp::LT, eq_sum, eq_f, 0, true}, {Cmp::LE, eq_sum, eq_f, 0, true},
                    {Cmp::GT, max_sum + 1, 1e300, 0, true},              // everything filtered
                    {Cmp::None, 0, 0.0, 10, true}, {Cmp::None, 0, 0.0, 10, false},
                    {Cmp::None, 0, 0.0, 1, true},
                    {Cmp::GT, max_sum - 2, eq_f, 1000000, true},         // k > survivors
                    {Cmp::LT, 0, 0.0, 25, false}, {Cmp::LE, -900000, -2.0e5, 7, true},
                    {Cmp::None, 0, 0.0, 3000, true},                     // ties among small counts
                };
                auto same = [](const gpudb::GroupByResidentResult& a, const gpudb::GroupByResidentResult& b, bool f64) {
                    if (a.keys.size() != b.keys.size() || a.groups_total != b.groups_total) return false;
                    std::vector<std::size_t> ia(a.keys.size()), ib(b.keys.size());
                    for (std::size_t i = 0; i < ia.size(); ++i) { ia[i] = i; ib[i] = i; }
                    std::sort(ia.begin(), ia.end(), [&](std::size_t x, std::size_t y) { return a.keys[x] < a.keys[y]; });
                    std::sort(ib.begin(), ib.end(), [&](std::size_t x, std::size_t y) { return b.keys[x] < b.keys[y]; });
                    for (std::size_t i = 0; i < ia.size(); ++i) {
                        const std::size_t x = ia[i], y = ib[i];
                        if (a.keys[x] != b.keys[y] || a.counts[x] != b.counts[y]) return false;
                        if (f64) {
                            const double tol = 1e-9 * std::max(1.0, std::abs(b.sums_f64[y]));
                            if (std::abs(a.sums_f64[x] - b.sums_f64[y]) > tol) return false;
                        } else if (!a.sums.empty() && a.sums[x] != b.sums[y]) return false;
                    }
                    return true;
                };
                for (const auto& fl : filters) {
                    gpudb::GroupByResidentResult ri = base_i, rf = base_f, rc = base_c;
                    gpudb::apply_group_filter_host(ri, fl, gpudb::FilterAgg::SumI64, cap, "ref");
                    gpudb::apply_group_filter_host(rf, fl, gpudb::FilterAgg::SumF64, cap, "ref");
                    gpudb::apply_group_filter_host(rc, fl, gpudb::FilterAgg::Count,  cap, "ref");
                    auto gi = agg->groupby_sum_resident_i64(*kc, *vc, cap, fl);
                    auto gf = agg->groupby_sum_resident_f64(*kc, *fc, cap, fl);
                    auto gc = agg->groupby_count_resident(*kc, cap, fl);
                    EXPECT(same(gi, ri, false));
                    EXPECT(same(gf, rf, true));
                    EXPECT(same(gc, rc, false));
                    // order contract: key ascending without top-k, by aggregate with it
                    bool ord = true;
                    if (fl.topk == 0) ord = std::is_sorted(gi.keys.begin(), gi.keys.end());
                    else for (std::size_t i = 1; i < gi.sums.size(); ++i)
                        ord = ord && (fl.topk_desc ? gi.sums[i - 1] >= gi.sums[i] : gi.sums[i - 1] <= gi.sums[i]);
                    EXPECT(ord);
                }
                // cap bounds the rows returned, with the filtered wording
                bool threw_f = false;
                try { (void)agg->groupby_sum_resident_i64(*kc, *vc, 3, gpudb::GroupByFilter{Cmp::GE, std::numeric_limits<std::int64_t>::min(), 0.0, 0, true}); }
                catch (const std::runtime_error& e) { threw_f = std::string(e.what()).find("rows after the filter, above the cap of 3") != std::string::npos; }
                EXPECT(threw_f);
                auto small = agg->groupby_sum_resident_i64(*kc, *vc, 9, gpudb::GroupByFilter{Cmp::None, 0, 0.0, 9, true});
                EXPECT_EQ(small.keys.size(), std::size_t(9));
                EXPECT_EQ(small.groups_total, ref.size());

                // f64 NaN / inf rule on THIS backend vs the host reference:
                // keys 1..8 -> +inf, -inf, NaN, -NaN, 10, -10, 0 (5 + -5),
                // NaN from inf + -inf. cmp drops every NaN group; top-k treats
                // every NaN as greatest (DESC first, ASC last).
                {
                    const double qn = std::numeric_limits<double>::quiet_NaN();
                    const double in = std::numeric_limits<double>::infinity();
                    std::vector<std::int64_t> nk = {1, 2, 3, 4, 5, 6, 7, 7, 8, 8};
                    std::vector<double>       nv = {in, -in, qn, -qn, 10.0, -10.0, 5.0, -5.0, in, -in};
                    auto nkc = agg->upload_i64(nk.data(), nk.size());
                    auto nvc = agg->upload_f64(nv.data(), nv.size());
                    const gpudb::GroupByFilter nf[] = {
                        {Cmp::GT, 0, 0.0, 0, true}, {Cmp::LE, 0, 0.0, 0, true}, {Cmp::GE, 0, -in, 0, true},
                        {Cmp::None, 0, 0.0, 3, true}, {Cmp::None, 0, 0.0, 3, false},
                        {Cmp::None, 0, 0.0, 8, true}, {Cmp::GT, 0, -in, 2, true},
                    };
                    auto nbase = agg->groupby_sum_resident_f64(*nkc, *nvc, cap);
                    for (const auto& fl : nf) {
                        gpudb::GroupByResidentResult rr = nbase;
                        gpudb::apply_group_filter_host(rr, fl, gpudb::FilterAgg::SumF64, cap, "ref");
                        auto g = agg->groupby_sum_resident_f64(*nkc, *nvc, cap, fl);
                        // exact row-by-row: NaN==NaN treated as equal, order must match
                        bool ok = g.keys.size() == rr.keys.size();
                        for (std::size_t i = 0; ok && i < g.keys.size(); ++i) {
                            const double x = g.sums_f64[i], y = rr.sums_f64[i];
                            const bool same_val = (std::isnan(x) && std::isnan(y)) || x == y;
                            ok = same_val && (fl.topk == 0 ? g.keys[i] == rr.keys[i] : true);
                        }
                        EXPECT(ok);
                    }
                    auto t3 = agg->groupby_sum_resident_f64(*nkc, *nvc, cap, gpudb::GroupByFilter{Cmp::None, 0, 0.0, 3, true});
                    EXPECT(t3.sums_f64.size() == 3 && std::isnan(t3.sums_f64[0]) && std::isnan(t3.sums_f64[1]) && std::isnan(t3.sums_f64[2]));
                    auto a3 = agg->groupby_sum_resident_f64(*nkc, *nvc, cap, gpudb::GroupByFilter{Cmp::None, 0, 0.0, 3, false});
                    EXPECT(a3.sums_f64.size() == 3 && a3.sums_f64[0] == -in && a3.sums_f64[1] == -10.0 && a3.sums_f64[2] == 0.0);
                }
            }

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

    // ---- GroupByFilter (device-side HAVING / top-k of groups) vs the host reference ----
    {
        std::printf("  resident group by filter (having / top-k):\n");
        using Cmp = gpudb::GroupByFilter::Cmp;
        // small keyed set with deliberate ties in the sums and in the counts
        const std::size_t N = 70'003;
        std::vector<std::int64_t> keys(N), vals(N);
        for (std::size_t i = 0; i < N; ++i) {
            keys[i] = static_cast<std::int64_t>((i * 7919u) % 1000) - 500;   // 1000 keys, -500..499
            vals[i] = static_cast<std::int64_t>(keys[i] % 4);              // sums tie heavily
        }
        keys[N - 1] = std::numeric_limits<std::int64_t>::max(); vals[N - 1] = std::numeric_limits<std::int64_t>::max();
        keys[N - 2] = std::numeric_limits<std::int64_t>::min(); vals[N - 2] = std::numeric_limits<std::int64_t>::min();
        auto kc = agg->upload_i64(keys.data(), N);
        auto vc = agg->upload_i64(vals.data(), N);
        bool implemented = true;
        try {
            const std::size_t cap = std::size_t(100) * 1000000;
            auto full = agg->groupby_sum_resident_i64(*kc, *vc, cap);
            auto fullc = agg->groupby_count_resident(*kc, cap);
            EXPECT_EQ(full.groups_total, full.keys.size());
            std::int64_t smax = full.sums[0], smin = full.sums[0];
            for (auto v : full.sums) { smax = std::max(smax, v); smin = std::min(smin, v); }
            // thresholds: an existing sum (boundary), below min, above max, 0
            const std::int64_t mid = full.sums[full.sums.size() / 2];
            struct Case { Cmp cmp; std::int64_t thr; std::size_t topk; bool desc; };
            std::vector<Case> cases = {
                {Cmp::GT, mid, 0, true}, {Cmp::GE, mid, 0, true}, {Cmp::LT, mid, 0, true}, {Cmp::LE, mid, 0, true},
                {Cmp::GT, smax, 0, true},            // nothing survives
                {Cmp::GE, smin, 0, true},            // everything survives
                {Cmp::None, 0, 1, true}, {Cmp::None, 0, 1, false},
                {Cmp::None, 0, 7, true}, {Cmp::None, 0, 7, false},
                {Cmp::None, 0, 257, true},           // crosses a 256-block
                {Cmp::None, 0, full.keys.size(), true},      // k == groups
                {Cmp::None, 0, full.keys.size() + 5, false}, // k > groups
                {Cmp::GT, mid, 3, true}, {Cmp::LE, mid, 3, false},   // having + topk
                {Cmp::GT, smax, 3, true},            // topk over an empty survivor set
            };
            int idx = 0;
            for (const auto& c : cases) {
                for (int count_mode = 0; count_mode < 2; ++count_mode) {
                    gpudb::GroupByFilter f;
                    f.cmp = c.cmp; f.threshold_i64 = count_mode ? (c.thr == mid ? fullc.counts[fullc.counts.size() / 2] : c.thr == smax ? 1 << 30 : c.thr == smin ? 0 : c.thr) : c.thr;
                    f.topk = c.topk; f.topk_desc = c.desc;
                    gpudb::GroupByResidentResult ref = count_mode ? fullc : full;
                    gpudb::apply_group_filter_host(ref, f, count_mode ? gpudb::FilterAgg::Count : gpudb::FilterAgg::SumI64, cap, "ref");
                    auto got = count_mode ? agg->groupby_count_resident(*kc, cap, f)
                                          : agg->groupby_sum_resident_i64(*kc, *vc, cap, f);
                    const auto& ga = count_mode ? got.counts : got.sums;
                    const auto& ra = count_mode ? ref.counts : ref.sums;
                    bool ok = got.keys.size() == ref.keys.size() && got.groups_total == ref.groups_total && ga == ra;
                    // keys: exact when no tie straddles the k-th rank, else as a set of valid rows
                    bool boundary_tie = false;
                    if (ok && f.topk != 0 && !ra.empty()) {
                        // count survivors with the k-th aggregate in the unfiltered set
                        const auto& base = count_mode ? fullc : full;
                        const auto& ba = count_mode ? base.counts : base.sums;
                        std::size_t eq = 0;
                        for (auto v : ba) eq += (v == ra.back());
                        std::size_t in_out = 0;
                        for (auto v : ra) in_out += (v == ra.back());
                        boundary_tie = eq != in_out;
                    }
                    if (ok && !boundary_tie) ok = got.keys == ref.keys;
                    if (ok && boundary_tie) {
                        std::map<std::int64_t, std::int64_t> m;
                        const auto& base = count_mode ? fullc : full;
                        const auto& ba = count_mode ? base.counts : base.sums;
                        for (std::size_t i = 0; i < base.keys.size(); ++i) m[base.keys[i]] = ba[i];
                        std::set<std::int64_t> seen;
                        for (std::size_t i = 0; ok && i < got.keys.size(); ++i)
                            ok = m.count(got.keys[i]) && m[got.keys[i]] == ga[i] && seen.insert(got.keys[i]).second;
                    }
                    if (!ok) std::printf("    FAIL case %d (count_mode=%d): got %zu rows, ref %zu\n", idx, count_mode, got.keys.size(), ref.keys.size());
                    EXPECT(ok);
                }
                ++idx;
            }
            // cap applies to the survivors: 3 rows allowed, 7 asked
            {
                gpudb::GroupByFilter f; f.topk = 7;
                bool threw = false;
                try { (void)agg->groupby_sum_resident_i64(*kc, *vc, 3, f); }
                catch (const std::runtime_error& e) { threw = std::string(e.what()).find("above the cap") != std::string::npos; }
                EXPECT(threw);
                gpudb::GroupByFilter g; g.topk = 3;
                auto okr = agg->groupby_sum_resident_i64(*kc, *vc, 3, g);   // exactly at the cap: fine
                EXPECT_EQ(okr.keys.size(), std::size_t(3));
            }
            // f64: same contract through the host path
            {
                std::vector<double> dv(N);
                for (std::size_t i = 0; i < N; ++i) dv[i] = static_cast<double>(vals[i]) * 0.5;
                auto fc = agg->upload_f64(dv.data(), N);
                auto fullf = agg->groupby_sum_resident_f64(*kc, *fc, cap);
                gpudb::GroupByFilter f; f.cmp = Cmp::GT; f.threshold_f64 = fullf.sums_f64[fullf.sums_f64.size() / 2]; f.topk = 5; f.topk_desc = true;
                auto ref = fullf;
                gpudb::apply_group_filter_host(ref, f, gpudb::FilterAgg::SumF64, cap, "ref");
                auto got = agg->groupby_sum_resident_f64(*kc, *fc, cap, f);
                EXPECT_EQ(got.keys.size(), ref.keys.size());
                EXPECT(got.sums_f64 == ref.sums_f64);
            }
            // f64 NaN / inf contract: NaN is the greatest for top-k (any sign bit),
            // dropped by every cmp; +inf/-inf compare normally.
            {
                const double inf = std::numeric_limits<double>::infinity();
                const double qnan = std::numeric_limits<double>::quiet_NaN();
                std::vector<std::int64_t> nk{1, 2, 3, 4, 5, 5, 6, 7, 8};
                std::vector<double> nv{inf, -inf, qnan, -qnan, inf, -inf, 10.0, -10.0, 0.0};
                auto nkc = agg->upload_i64(nk.data(), nk.size());
                auto nfc = agg->upload_f64(nv.data(), nv.size());
                gpudb::GroupByFilter d3; d3.topk = 3; d3.topk_desc = true;
                auto top = agg->groupby_sum_resident_f64(*nkc, *nfc, cap, d3);
                bool ok = top.keys.size() == 3;
                for (double v : top.sums_f64) ok = ok && std::isnan(v);   // keys 3, 4, 5 (inf + -inf)
                EXPECT(ok);
                gpudb::GroupByFilter a3; a3.topk = 3; a3.topk_desc = false;
                auto bot = agg->groupby_sum_resident_f64(*nkc, *nfc, cap, a3);
                EXPECT(bot.sums_f64 == std::vector<double>({-inf, -10.0, 0.0}));
                gpudb::GroupByFilter d8; d8.topk = 8; d8.topk_desc = true;
                auto all = agg->groupby_sum_resident_f64(*nkc, *nfc, cap, d8);
                ok = all.keys.size() == 8 && std::isnan(all.sums_f64[0]) && std::isnan(all.sums_f64[1]) &&
                     std::isnan(all.sums_f64[2]) && all.sums_f64[3] == inf && all.sums_f64[4] == 10.0 &&
                     all.sums_f64[5] == 0.0 && all.sums_f64[6] == -10.0 && all.sums_f64[7] == -inf;
                EXPECT(ok);
                gpudb::GroupByFilter ge; ge.cmp = Cmp::GE; ge.threshold_f64 = -inf;   // everything, NaN included
                auto kept = agg->groupby_sum_resident_f64(*nkc, *nfc, cap, ge);
                EXPECT_EQ(kept.keys.size(), std::size_t(8));
                gpudb::GroupByFilter gt0; gt0.cmp = Cmp::GT; gt0.threshold_f64 = 0.0;   // inf, NaN x3, 10 (native HAVING keeps NaN)
                EXPECT_EQ(agg->groupby_sum_resident_f64(*nkc, *nfc, cap, gt0).keys.size(), std::size_t(5));
                gpudb::GroupByFilter len; len.cmp = Cmp::LE; len.threshold_f64 = qnan;  // x <= NaN: everything
                EXPECT_EQ(agg->groupby_sum_resident_f64(*nkc, *nfc, cap, len).keys.size(), std::size_t(8));
                gpudb::GroupByFilter gen; gen.cmp = Cmp::GE; gen.threshold_f64 = qnan;  // x >= NaN: only NaN groups
                EXPECT_EQ(agg->groupby_sum_resident_f64(*nkc, *nfc, cap, gen).keys.size(), std::size_t(3));
                gpudb::GroupByFilter gtn; gtn.cmp = Cmp::GT; gtn.threshold_f64 = qnan;  // x > NaN: nothing
                EXPECT_EQ(agg->groupby_sum_resident_f64(*nkc, *nfc, cap, gtn).keys.size(), std::size_t(0));
                gpudb::GroupByFilter le; le.cmp = Cmp::LE; le.threshold_f64 = 0.0;
                auto low = agg->groupby_sum_resident_f64(*nkc, *nfc, cap, le);
                EXPECT_EQ(low.keys.size(), std::size_t(3));   // -inf, -10, 0
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

    // ---- Exact GROUP BY: the extreme key values ----
    // INT64_MIN is the v0.6 hash paths' empty sentinel, and those refuse it by
    // design (cuda_groupby.cpp / cuda_hashjoin.cpp raise on it). The exact
    // path sorts instead of hashing, so it has no sentinel and must accept
    // both ends of the range — a boundary one code path explicitly rejects is
    // exactly where another can quietly be wrong.
    {
        std::printf("  exact group by, INT64_MIN / INT64_MAX keys:\n");
        constexpr std::int64_t kMin = std::numeric_limits<std::int64_t>::min();
        constexpr std::int64_t kMax = std::numeric_limits<std::int64_t>::max();
        std::vector<std::int64_t> keys{kMin, kMin, kMax, 0, 0};
        std::vector<std::int64_t> vals{10, 20, 30, 40, 50};
        std::vector<std::uint64_t> kvalid(1, ~std::uint64_t{0});
        kvalid[0] &= ~(std::uint64_t{1} << 4);          // last row: NULL key
        gpudb::Aggregator::KvSpan sp{};
        sp.kv = nullptr; sp.rows = keys.size();
        std::vector<std::int64_t> kv(keys.size() * 2);
        for (std::size_t i = 0; i < keys.size(); ++i) { kv[2 * i] = keys[i]; kv[2 * i + 1] = vals[i]; }
        sp.kv = kv.data(); sp.key_valid = kvalid.data(); sp.val_valid = nullptr;
        bool implemented = true;
        try {
            auto pair = agg->upload_pair_exact(&sp, 1, gpudb::Dtype::I64);
            auto r = agg->groupby_exact_resident(*pair.keys, pair.vals.get(), 1000);
            // sorted ascending, NULL key last: INT64_MIN, 0, INT64_MAX, NULL
            EXPECT_EQ(r.keys.size(), std::size_t{4});
            EXPECT_EQ(r.keys[0], kMin);
            EXPECT_EQ(r.sums[0], 30);          // 10 + 20
            EXPECT_EQ(r.counts[0], 2);
            EXPECT_EQ(r.keys[1], std::int64_t{0});
            EXPECT_EQ(r.sums[1], 40);
            EXPECT_EQ(r.keys[2], kMax);
            EXPECT_EQ(r.sums[2], 30);
            EXPECT_EQ(r.key_null[3], std::uint8_t{1});
            EXPECT_EQ(r.sums[3], 50);
        } catch (const std::exception& e) {
            if (std::string(e.what()).find("not implemented") != std::string::npos) {
                implemented = false;
                std::printf("SKIP (%s)\n", e.what());
            } else {
                throw;
            }
        }
        if (implemented) std::printf("    ok\n");
    }

    // ---- Exact GROUP BY (v0.7 milestone 3, §4.1 / §4.2) vs a host reference ----
    // NULL keys form one trailing group; NULL payloads count for count(*)
    // only; the sum is 128-bit and never wraps; an all-NULL-payload group has
    // count(v) == 0. Backends opt in; "not implemented" is reported as SKIP.
    {
        std::printf("  exact group by (NULLs, 128-bit sums):\n");
        using Cmp = gpudb::GroupByFilter::Cmp;
        std::mt19937_64 rng(0xE7ACULL);
        const std::size_t N = 200'003;
        const std::int64_t K = 2'999;
        std::vector<std::int64_t> keys(N), vals(N);
        std::vector<std::uint64_t> kvalid((N + 63) / 64, ~std::uint64_t{0}), vvalid((N + 63) / 64, ~std::uint64_t{0});
        std::uniform_int_distribution<std::int64_t> kd(-K, K);
        std::uniform_int_distribution<int> pct(0, 99);
        // Values span the full int64 range so per-group sums overflow 64 bits.
        std::uniform_int_distribution<std::int64_t> big(std::numeric_limits<std::int64_t>::min(),
                                                        std::numeric_limits<std::int64_t>::max());
        auto clr = [](std::vector<std::uint64_t>& m, std::size_t i) { m[i >> 6] &= ~(std::uint64_t{1} << (i & 63)); };
        for (std::size_t i = 0; i < N; ++i) {
            keys[i] = kd(rng);
            vals[i] = (pct(rng) < 50) ? big(rng) : kd(rng);
            if (pct(rng) < 7)  clr(kvalid, i);          // ~7% NULL keys
            if (pct(rng) < 11) clr(vvalid, i);          // ~11% NULL payloads
        }
        // Key 4000 exists only with NULL payloads -> all-NULL group.
        for (std::size_t i = 0; i < 5; ++i) { keys[i] = 4000; clr(vvalid, i); kvalid[i >> 6] |= std::uint64_t{1} << (i & 63); }
        // Key 4001: a single valid row at each int64 extreme (min/max exact).
        keys[5] = 4001; vals[5] = std::numeric_limits<std::int64_t>::max(); kvalid[0] |= std::uint64_t{1} << 5; vvalid[0] |= std::uint64_t{1} << 5;
        keys[6] = 4001; vals[6] = std::numeric_limits<std::int64_t>::min(); kvalid[0] |= std::uint64_t{1} << 6; vvalid[0] |= std::uint64_t{1} << 6;
        auto isv = [](const std::vector<std::uint64_t>& m, std::size_t i) { return (m[i >> 6] >> (i & 63)) & 1u; };

        struct Ref { gpudb::Sum128 s; std::int64_t cv = 0, cs = 0;
                     std::int64_t mn = std::numeric_limits<std::int64_t>::max();
                     std::int64_t mx = std::numeric_limits<std::int64_t>::min(); };
        std::map<std::int64_t, Ref> ref; Ref ref_null; bool have_null_key = false;
        for (std::size_t i = 0; i < N; ++i) {
            Ref& e = isv(kvalid, i) ? ref[keys[i]] : (have_null_key = true, ref_null);
            ++e.cs;
            if (!isv(vvalid, i)) continue;
            e.s.add(vals[i]); ++e.cv; e.mn = std::min(e.mn, vals[i]); e.mx = std::max(e.mx, vals[i]);
        }
        // Two spans with different bitmap alignment (second starts at row 1000).
        const std::size_t split = 1000;
        std::vector<std::int64_t> kv0(2 * split), kv1(2 * (N - split));
        std::vector<std::uint64_t> kv0k((split + 63) / 64, 0), kv0v((split + 63) / 64, 0),
                                   kv1k((N - split + 63) / 64, 0), kv1v((N - split + 63) / 64, 0);
        for (std::size_t i = 0; i < N; ++i) {
            const bool first = i < split;
            const std::size_t r = first ? i : i - split;
            auto& kv = first ? kv0 : kv1;
            kv[2 * r] = keys[i]; kv[2 * r + 1] = vals[i];
            if (isv(kvalid, i)) (first ? kv0k : kv1k)[r >> 6] |= std::uint64_t{1} << (r & 63);
            if (isv(vvalid, i)) (first ? kv0v : kv1v)[r >> 6] |= std::uint64_t{1} << (r & 63);
        }
        gpudb::Aggregator::KvSpan spans[2];
        spans[0].kv = kv0.data(); spans[0].rows = split;     spans[0].key_valid = kv0k.data(); spans[0].val_valid = kv0v.data();
        spans[1].kv = kv1.data(); spans[1].rows = N - split; spans[1].key_valid = kv1k.data(); spans[1].val_valid = kv1v.data();
        bool implemented = true;
        try {
            auto pair = agg->upload_pair_exact(spans, 2, gpudb::Dtype::I64);
            std::size_t null_keys = 0, null_vals = 0;
            for (std::size_t i = 0; i < N; ++i) { null_keys += !isv(kvalid, i); null_vals += !isv(vvalid, i); }
            EXPECT_EQ(pair.keys->rows(), N);
            EXPECT_EQ(pair.keys->null_count(), null_keys);
            EXPECT_EQ(pair.vals->null_count(), null_vals);

            // The legacy op must refuse a NULL-bearing column, never read it as data.
            bool refused = false;
            try { (void)agg->groupby_sum_resident_i64(*pair.keys, *pair.vals, std::size_t(100) * 1000000); }
            catch (const std::runtime_error&) { refused = true; }
            EXPECT(refused);

            const std::size_t cap = std::size_t(100) * 1000000;
            auto r = agg->groupby_exact_resident(*pair.keys, pair.vals.get(), cap);
            const std::size_t expect_groups = ref.size() + (have_null_key ? 1 : 0);
            EXPECT_EQ(r.keys.size(), expect_groups);
            EXPECT_EQ(r.groups_total, expect_groups);
            EXPECT_EQ(r.rows_in, N);
            bool ok = r.keys.size() == expect_groups && r.key_null.size() == r.keys.size();
            std::size_t j = 0;
            auto same = [&](std::size_t jj, const Ref& e) {
                return r.counts[jj] == e.cv && r.counts_star[jj] == e.cs &&
                       (e.cv == 0 || (static_cast<std::uint64_t>(r.sums[jj]) == e.s.lo && r.sums_hi[jj] == e.s.hi &&
                                      r.mins[jj] == e.mn && r.maxs[jj] == e.mx));
            };
            for (auto it = ref.begin(); ok && it != ref.end(); ++it, ++j)
                ok = r.key_null[j] == 0 && r.keys[j] == it->first && same(j, it->second);
            if (ok && have_null_key) ok = r.key_null[j] == 1 && same(j, ref_null);
            EXPECT(ok);   // sorted ascending, NULL key last, exact 128-bit sums, exact counts/min/max
            // The all-NULL-payload group is present with count(v) == 0.
            {
                auto it = ref.find(4000);
                ok = it != ref.end() && it->second.cv == 0 && it->second.cs == 5;
                EXPECT(ok);
            }
            // Keys-only form: count(*) per key.
            auto c = agg->groupby_exact_resident(*pair.keys, nullptr, cap);
            ok = c.keys.size() == expect_groups;
            j = 0;
            for (auto it = ref.begin(); ok && it != ref.end(); ++it, ++j)
                ok = c.keys[j] == it->first && c.counts_star[j] == it->second.cs;
            EXPECT(ok);

            // Filters against the host reference on the unfiltered result.
            using Agg = gpudb::GroupByFilter::Agg;
            gpudb::GroupByResidentResult base = r;
            struct FC { Cmp cmp; std::int64_t thr; std::size_t topk; bool desc; Agg agg; };
            const std::int64_t mid_cs = base.counts_star[base.counts_star.size() / 2];
            std::vector<FC> fcs = {
                {Cmp::GT, 0, 0, true, Agg::Sum}, {Cmp::LE, 0, 0, true, Agg::Sum},
                {Cmp::GE, mid_cs, 0, true, Agg::CountStar}, {Cmp::LT, mid_cs, 0, true, Agg::CountV},
                {Cmp::GT, 0, 0, true, Agg::Min}, {Cmp::LT, 0, 0, true, Agg::Max},
                {Cmp::None, 0, 5, true, Agg::Sum}, {Cmp::None, 0, 5, false, Agg::Sum},
                {Cmp::None, 0, 9, true, Agg::CountStar}, {Cmp::None, 0, 9, false, Agg::Min},
                {Cmp::None, 0, base.keys.size() + 3, true, Agg::Max},   // k > groups: NULL agg last
                {Cmp::GT, 0, 4, true, Agg::Sum},
            };
            int idx = 0;
            for (const auto& fc : fcs) {
                gpudb::GroupByFilter fl; fl.cmp = fc.cmp; fl.threshold_i64 = fc.thr; fl.topk = fc.topk; fl.topk_desc = fc.desc; fl.agg = fc.agg;
                gpudb::GroupByResidentResult want = base;
                gpudb::apply_group_filter_host(want, fl, gpudb::FilterAgg::Exact, cap, "ref");
                auto got = agg->groupby_exact_resident(*pair.keys, pair.vals.get(), cap, fl);
                ok = got.keys.size() == want.keys.size() && got.groups_total == want.groups_total;
                if (ok && fl.topk == 0) {
                    for (std::size_t q = 0; ok && q < got.keys.size(); ++q)
                        ok = got.keys[q] == want.keys[q] && got.key_null[q] == want.key_null[q] &&
                             got.counts[q] == want.counts[q] && got.counts_star[q] == want.counts_star[q];
                } else if (ok) {
                    // top-k: tie order unspecified -> compare the multiset of (key, key_null, count_star)
                    std::vector<std::tuple<std::int64_t, int, std::int64_t>> a, b;
                    for (std::size_t q = 0; q < got.keys.size(); ++q) {
                        a.emplace_back(got.keys[q], got.key_null[q], got.counts_star[q]);
                        b.emplace_back(want.keys[q], want.key_null[q], want.counts_star[q]);
                    }
                    std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end());
                    ok = a == b;
                }
                if (!ok) std::printf("    FAIL exact filter case %d: got %zu rows, ref %zu\n", idx, got.keys.size(), want.keys.size());
                EXPECT(ok);
                ++idx;
            }
            // Cap: throws naming the count, never truncates.
            bool threw = false;
            try { (void)agg->groupby_exact_resident(*pair.keys, pair.vals.get(), 10); }
            catch (const std::runtime_error& e) {
                threw = std::string(e.what()).find(std::to_string(expect_groups)) != std::string::npos;
            }
            EXPECT(threw);
            // DOUBLE payloads are not on the exact path.
            bool dbl_refused = false;
            try { (void)agg->upload_pair_exact(spans, 2, gpudb::Dtype::F64); }
            catch (const std::runtime_error&) { dbl_refused = true; }
            EXPECT(dbl_refused);
        } catch (const std::runtime_error& e) {
            if (std::string(e.what()).find("not implemented") != std::string::npos) {
                implemented = false;
                std::printf("    SKIP (%s)\n", e.what());
            } else {
                ++failures; ++total;
                std::printf("    FAIL: %s\n", e.what());
            }
        }
        if (implemented) std::printf("    ok\n");
    }

    // ---- Exact GROUP BY with a WHERE mask (v0.7 §4.6) vs a host reference ----
    // upload_rows_exact (key, payload, i64 pred, f64 pred) then conjunctions
    // of predicates incl. NULL cells, IN, IS [NOT] NULL, and the f64 total
    // order (NaN greatest and equal to NaN, -0.0 == 0.0). Backends opt in.
    {
        std::printf("  exact group by with WHERE mask:\n");
        std::mt19937_64 rng(0x3A5EULL);
        const std::size_t N = 120'011;
        std::vector<std::int64_t> lanes(N * 4);
        std::vector<std::uint64_t> valid[4];
        for (auto& m : valid) m.assign((N + 63) / 64, ~std::uint64_t{0});
        auto clr = [](std::vector<std::uint64_t>& m, std::size_t i) { m[i >> 6] &= ~(std::uint64_t{1} << (i & 63)); };
        auto isv = [](const std::vector<std::uint64_t>& m, std::size_t i) { return ((m[i >> 6] >> (i & 63)) & 1u) != 0; };
        std::uniform_int_distribution<std::int64_t> kd(-500, 500), vd(-100000, 100000), ad(0, 40);
        std::uniform_int_distribution<int> pct(0, 99);
        std::uniform_real_distribution<double> dd(-50.0, 50.0);
        for (std::size_t i = 0; i < N; ++i) {
            lanes[4 * i + 0] = kd(rng);
            lanes[4 * i + 1] = vd(rng);
            lanes[4 * i + 2] = ad(rng);
            double d = dd(rng);
            const int r = pct(rng);
            if (r < 2) d = std::numeric_limits<double>::quiet_NaN();
            else if (r < 3) d = -0.0;
            else if (r < 4) d = 0.0;
            else if (r < 5) d = -std::numeric_limits<double>::quiet_NaN();   // sign-bit NaN
            std::memcpy(&lanes[4 * i + 3], &d, sizeof(d));
            if (pct(rng) < 5) clr(valid[0], i);
            if (pct(rng) < 9) clr(valid[1], i);
            if (pct(rng) < 6) clr(valid[2], i);
            if (pct(rng) < 4) clr(valid[3], i);
        }
        const std::uint64_t* vptr[4] = { valid[0].data(), valid[1].data(), valid[2].data(), valid[3].data() };
        gpudb::Aggregator::RowSpan span;
        span.lanes = lanes.data(); span.rows = N; span.n_lanes = 4; span.valid = vptr;
        const gpudb::Dtype dts[4] = { gpudb::Dtype::I64, gpudb::Dtype::I64, gpudb::Dtype::I64, gpudb::Dtype::F64 };
        bool implemented = true;
        try {
            auto cols = agg->upload_rows_exact(&span, 1, dts, 4);
            EXPECT_EQ(cols.size(), std::size_t(4));
            std::size_t nk = 0; for (std::size_t i = 0; i < N; ++i) nk += !isv(valid[0], i);
            EXPECT_EQ(cols[0]->null_count(), nk);
            EXPECT_EQ(cols[3]->rows(), N);

            // Host reference: evaluate the mask on the ORIGINAL rows, then group.
            struct Ref { gpudb::Sum128 s; std::int64_t cv = 0, cs = 0;
                         std::int64_t mn = std::numeric_limits<std::int64_t>::max();
                         std::int64_t mx = std::numeric_limits<std::int64_t>::min(); };
            auto dkey = [](double x) { return gpudb::f64_total_order_key(x); };
            auto run_case = [&](const std::vector<gpudb::Predicate>& ps,
                                const std::function<bool(std::size_t)>& host_mask, int idx) {
                std::map<std::int64_t, Ref> ref; Ref ref_null; bool null_group = false;
                for (std::size_t i = 0; i < N; ++i) {
                    if (!host_mask(i)) continue;
                    Ref& e = isv(valid[0], i) ? ref[lanes[4 * i]] : (null_group = true, ref_null);
                    ++e.cs;
                    if (!isv(valid[1], i)) continue;
                    const std::int64_t x = lanes[4 * i + 1];
                    e.s.add(x); ++e.cv; e.mn = std::min(e.mn, x); e.mx = std::max(e.mx, x);
                }
                auto r = agg->groupby_exact_masked_resident(*cols[0], cols[1].get(), ps.data(), ps.size(),
                                                            std::size_t(100) * 1000000);
                const std::size_t want = ref.size() + (null_group ? 1 : 0);
                bool ok = r.keys.size() == want && r.groups_total == want;
                std::size_t j = 0;
                auto same = [&](std::size_t jj, const Ref& e) {
                    return r.counts[jj] == e.cv && r.counts_star[jj] == e.cs &&
                           (e.cv == 0 || (static_cast<std::uint64_t>(r.sums[jj]) == e.s.lo && r.sums_hi[jj] == e.s.hi &&
                                          r.mins[jj] == e.mn && r.maxs[jj] == e.mx));
                };
                for (auto it = ref.begin(); ok && it != ref.end(); ++it, ++j)
                    ok = r.key_null[j] == 0 && r.keys[j] == it->first && same(j, it->second);
                if (ok && null_group) ok = r.key_null[j] == 1 && same(j, ref_null);
                if (!ok) std::printf("    FAIL mask case %d: got %zu groups, ref %zu\n", idx, r.keys.size(), want);
                EXPECT(ok);
            };
            using Op = gpudb::Predicate::Op;
            auto P = [&](std::size_t lane, Op op, std::int64_t v) { gpudb::Predicate p; p.col = cols[lane].get(); p.op = op; p.value = v; return p; };
            auto PF = [&](Op op, double d) { std::int64_t b; std::memcpy(&b, &d, sizeof(b)); return P(3, op, b); };
            auto A = [&](std::size_t i) { return lanes[4 * i + 2]; };
            auto D = [&](std::size_t i) { double d; std::memcpy(&d, &lanes[4 * i + 3], sizeof(d)); return d; };
            // 0: i64 pred > 20 AND f64 pred <= 12.5 (NULL cells fail)
            run_case({P(2, Op::GT, 20), PF(Op::LE, 12.5)},
                     [&](std::size_t i) { return isv(valid[2], i) && A(i) > 20 && isv(valid[3], i) && dkey(D(i)) <= dkey(12.5); }, 0);
            // 1: f64 > 5 keeps NaN (greatest)
            run_case({PF(Op::GT, 5.0)}, [&](std::size_t i) { return isv(valid[3], i) && dkey(D(i)) > dkey(5.0); }, 1);
            // 2: f64 = 0 matches -0.0 and +0.0
            run_case({PF(Op::EQ, 0.0)}, [&](std::size_t i) { return isv(valid[3], i) && D(i) == 0.0; }, 2);
            // 3: f64 = NaN matches every NaN regardless of sign bit
            run_case({PF(Op::EQ, std::numeric_limits<double>::quiet_NaN())},
                     [&](std::size_t i) { return isv(valid[3], i) && std::isnan(D(i)); }, 3);
            // 4: IN list on i64 pred AND payload IS NOT NULL AND key >= -10 (a key predicate)
            {
                const std::int64_t lst[3] = { 1, 5, 9 };
                gpudb::Predicate in = P(2, Op::In, 0); in.list = lst; in.n_list = 3;
                run_case({in, P(1, Op::IsNotNull, 0), P(0, Op::GE, -10)},
                         [&](std::size_t i) { return isv(valid[2], i) && (A(i) == 1 || A(i) == 5 || A(i) == 9) &&
                                                     isv(valid[1], i) && isv(valid[0], i) && lanes[4 * i] >= -10; }, 4);
            }
            // 5: i64 pred IS NULL (keeps NULL keys too)
            run_case({P(2, Op::IsNull, 0)}, [&](std::size_t i) { return !isv(valid[2], i); }, 5);
            // 6: key IS NULL -> only the NULL-key group
            run_case({P(0, Op::IsNull, 0)}, [&](std::size_t i) { return !isv(valid[0], i); }, 6);
            // 7: nothing passes -> no groups
            run_case({P(2, Op::GT, 1000)}, [&](std::size_t) { return false; }, 7);
            // 8: f64 != NaN (drops every NaN, keeps the rest incl. -0.0)
            run_case({PF(Op::NE, std::numeric_limits<double>::quiet_NaN())},
                     [&](std::size_t i) { return isv(valid[3], i) && !std::isnan(D(i)); }, 8);
            // 9: no predicates == the plain exact op
            {
                auto a = agg->groupby_exact_masked_resident(*cols[0], cols[1].get(), nullptr, 0, std::size_t(100) * 1000000);
                auto b = agg->groupby_exact_resident(*cols[0], cols[1].get(), std::size_t(100) * 1000000);
                EXPECT(a.keys == b.keys && a.sums == b.sums && a.sums_hi == b.sums_hi && a.counts_star == b.counts_star);
            }
            // 10: WHERE + HAVING sum > 0 + top-k desc by count(*): compare to the host filter over the masked result
            {
                std::vector<gpudb::Predicate> ps = { P(2, Op::LT, 30) };
                auto base = agg->groupby_exact_masked_resident(*cols[0], cols[1].get(), ps.data(), ps.size(), std::size_t(100) * 1000000);
                gpudb::GroupByFilter fl; fl.cmp = gpudb::GroupByFilter::Cmp::GT; fl.threshold_i64 = 0; fl.topk = 7; fl.topk_desc = true; fl.agg = gpudb::GroupByFilter::Agg::CountStar;
                gpudb::GroupByResidentResult want = base;
                gpudb::apply_group_filter_host(want, fl, gpudb::FilterAgg::Exact, std::size_t(100) * 1000000, "ref");
                auto got = agg->groupby_exact_masked_resident(*cols[0], cols[1].get(), ps.data(), ps.size(), std::size_t(100) * 1000000, fl);
                std::vector<std::tuple<std::int64_t, int, std::int64_t>> a, b;
                for (std::size_t q = 0; q < got.keys.size(); ++q) a.emplace_back(got.keys[q], got.key_null[q], got.counts_star[q]);
                for (std::size_t q = 0; q < want.keys.size(); ++q) b.emplace_back(want.keys[q], want.key_null[q], want.counts_star[q]);
                std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end());
                EXPECT(a == b && got.groups_total == want.groups_total);
            }
            // Row count mismatch between a predicate column and the keys is an error.
            {
                std::vector<std::int64_t> other(N + 1, 0);
                auto oc = agg->upload_i64(other.data(), N + 1);
                gpudb::Predicate bad; bad.col = oc.get(); bad.op = Op::GT; bad.value = 0;
                bool threw = false;
                try { (void)agg->groupby_exact_masked_resident(*cols[0], cols[1].get(), &bad, 1, std::size_t(100) * 1000000); }
                catch (const std::runtime_error&) { threw = true; }
                EXPECT(threw);
            }
        } catch (const std::runtime_error& e) {
            if (std::string(e.what()).find("not implemented") != std::string::npos) {
                implemented = false;
                std::printf("    SKIP (%s)\n", e.what());
            } else {
                ++failures; ++total;
                std::printf("    FAIL: %s\n", e.what());
            }
        }
        if (implemented) std::printf("    ok\n");
    }

    // ---- Global masked aggregate (v0.7 §4.12) vs the CPU reference ----
    // Aggregates without GROUP BY under a WHERE, one fused pass: several
    // payloads, NULL payload cells and NULL predicate cells, every predicate
    // op including In and IsNull / IsNotNull, an F64 predicate lane, lanes at
    // every narrow-width boundary, a mask that keeps nothing, an empty column,
    // a 128-bit sum that overflows 64 bits, and 2.6M+ rows so the parallel
    // reduction runs many threadgroups. Every case is compared against the
    // CPU reference, limb for limb.
    {
        std::printf("  global masked aggregate:\n");
        using Op = gpudb::Predicate::Op;
        bool implemented = true;
        try {
            std::mt19937_64 rng(0xA11CEULL);
            // lanes: 0 stand-in key (unused), 1 payload i64-wide, 2 i8-wide,
            // 3 i16-wide, 4 i32-wide, 5 one past i32, 6 all NULL, 7 F64
            const std::size_t N = 2'600'011, L = 8;
            std::uniform_int_distribution<int> pct(0, 99);
            std::uniform_int_distribution<std::int64_t>
                wide(std::numeric_limits<std::int64_t>::min() / 64, std::numeric_limits<std::int64_t>::max() / 64),
                d8(-128, 127), d16(-32768, 32767), d32(-2147483648LL, 2147483647LL),
                d33(-2147483649LL, 2147483648LL);
            std::uniform_real_distribution<double> ud(-1.0, 2.0);
            std::vector<std::int64_t> lanes(N * L);
            std::vector<std::vector<std::uint64_t>> vb(L, std::vector<std::uint64_t>((N + 63) / 64, ~std::uint64_t{0}));
            auto clr = [&](std::size_t l, std::size_t i) { vb[l][i >> 6] &= ~(std::uint64_t{1} << (i & 63)); };
            for (std::size_t i = 0; i < N; ++i) {
                lanes[i * L + 0] = static_cast<std::int64_t>(i % 97);
                lanes[i * L + 1] = wide(rng);
                lanes[i * L + 2] = d8(rng);
                lanes[i * L + 3] = d16(rng);
                lanes[i * L + 4] = d32(rng);
                lanes[i * L + 5] = d33(rng);
                lanes[i * L + 6] = 0;
                const double x = ud(rng);
                std::memcpy(&lanes[i * L + 7], &x, sizeof(x));
                if (pct(rng) < 7)  clr(1, i);
                if (pct(rng) < 3)  clr(2, i);
                if (pct(rng) < 11) clr(3, i);
                if (pct(rng) < 2)  clr(7, i);
                clr(6, i);
            }
            // the width boundaries themselves, in rows that stay valid
            const std::int64_t ends[L][2] = {
                {0, 96}, {0, 0}, {-128, 127}, {-32768, 32767},
                {-2147483648LL, 2147483647LL}, {-2147483649LL, 2147483648LL}, {0, 0}, {0, 0} };
            for (std::size_t l = 0; l < L; ++l)
                for (int e = 0; e < 2; ++e) {
                    const std::size_t row = 5 + static_cast<std::size_t>(e);
                    if (l != 6 && l != 7) { lanes[row * L + l] = ends[l][e]; vb[l][row >> 6] |= std::uint64_t{1} << (row & 63); }
                }
            const gpudb::Dtype dts[L] = {gpudb::Dtype::I64, gpudb::Dtype::I64, gpudb::Dtype::I64, gpudb::Dtype::I64,
                                         gpudb::Dtype::I64, gpudb::Dtype::I64, gpudb::Dtype::I64, gpudb::Dtype::F64};
            std::vector<const std::uint64_t*> vp(L);
            for (std::size_t l = 0; l < L; ++l) vp[l] = vb[l].data();
            gpudb::Aggregator::RowSpan sp;
            sp.lanes = lanes.data(); sp.rows = N; sp.n_lanes = L; sp.valid = vp.data();
            auto cols = agg->upload_rows_exact(&sp, 1, dts, L);
            auto ref_agg = gpudb::make_aggregator(gpudb::Backend::CPU);
            auto ref_cols = ref_agg->upload_rows_exact(&sp, 1, dts, L);

            using Tup = std::tuple<std::int64_t, std::vector<std::int64_t>>;
            auto tuple_of = [](const gpudb::GlobalAggResult& r) {
                std::vector<std::int64_t> v;
                for (std::size_t p = 0; p < r.counts.size(); ++p) {
                    v.push_back(r.sums[p]); v.push_back(r.sums_hi[p]); v.push_back(r.counts[p]);
                    v.push_back(r.counts[p] ? r.mins[p] : 0); v.push_back(r.counts[p] ? r.maxs[p] : 0);
                }
                return Tup{r.count_star, v};
            };
            // one case: payload lane indices + a predicate program, both sides
            auto run_case = [&](const char* what, const std::vector<std::size_t>& pay,
                                const std::vector<std::tuple<std::size_t, Op, std::int64_t>>& terms,
                                const std::vector<std::int64_t>& list) {
                std::vector<gpudb::MultiPayload> mp(pay.size()), rmp(pay.size());
                for (std::size_t p = 0; p < pay.size(); ++p) {
                    mp[p].vals = cols[pay[p]].get();
                    rmp[p].vals = ref_cols[pay[p]].get();
                }
                std::vector<gpudb::Predicate> ps(terms.size()), rps(terms.size());
                for (std::size_t q = 0; q < terms.size(); ++q) {
                    ps[q].col = cols[std::get<0>(terms[q])].get();
                    ps[q].op = std::get<1>(terms[q]);
                    ps[q].value = std::get<2>(terms[q]);
                    if (ps[q].op == Op::In) { ps[q].list = list.data(); ps[q].n_list = list.size(); }
                    rps[q] = ps[q];
                    rps[q].col = ref_cols[std::get<0>(terms[q])].get();
                }
                auto got = agg->aggregate_exact_masked(mp.data(), mp.size(), ps.data(), ps.size());
                auto want = ref_agg->aggregate_exact_masked(rmp.data(), rmp.size(), rps.data(), rps.size());
                const bool ok = tuple_of(got) == tuple_of(want) && got.rows_in == N;
                if (!ok) std::printf("    FAIL %s: count_star %lld vs %lld\n", what,
                                     static_cast<long long>(got.count_star), static_cast<long long>(want.count_star));
                EXPECT(ok);
                return got;
            };
            const std::int64_t f_half = [] {
                const double d = 0.5; std::int64_t b; std::memcpy(&b, &d, sizeof(b)); return b;
            }();
            const std::vector<std::int64_t> in_list = {-128, 0, 1, 127, 42};
            run_case("no WHERE, six payloads", {1, 2, 3, 4, 5, 6}, {}, {});
            run_case("count(*) only, no WHERE", {}, {{2, Op::GE, -10}}, {});
            run_case("EQ / NE", {1, 2}, {{2, Op::EQ, 7}}, {});
            run_case("NE", {1, 3}, {{2, Op::NE, 7}}, {});
            run_case("LT / LE / GT / GE", {1, 4}, {{3, Op::LT, 1000}, {3, Op::GE, -1000}, {4, Op::GT, -5}, {2, Op::LE, 100}}, {});
            run_case("In", {1, 2}, {{2, Op::In, 0}}, in_list);
            run_case("IsNull", {1, 3}, {{3, Op::IsNull, 0}}, {});
            run_case("IsNotNull", {1, 3}, {{1, Op::IsNotNull, 0}, {3, Op::IsNotNull, 0}}, {});
            run_case("all NULL lane IsNull", {1}, {{6, Op::IsNull, 0}}, {});
            run_case("F64 lane", {1, 2}, {{7, Op::LT, f_half}}, {});
            run_case("F64 lane IsNull", {1}, {{7, Op::IsNull, 0}}, {});
            run_case("narrow boundaries", {2, 3, 4, 5}, {{5, Op::GE, -2147483649LL}}, {});
            {
                auto none = run_case("mask keeps nothing", {1, 2}, {{2, Op::GT, 1000}}, {});
                EXPECT_EQ(none.count_star, 0);
                EXPECT_EQ(none.counts[0], 0);
            }
            // 128-bit sum: a payload whose total overflows 64 bits
            {
                const std::size_t M = 300'000;
                std::vector<std::int64_t> big(M * 2);
                for (std::size_t i = 0; i < M; ++i) {
                    big[i * 2] = static_cast<std::int64_t>(i);
                    big[i * 2 + 1] = (i % 2) ? std::numeric_limits<std::int64_t>::max() / 2
                                             : std::numeric_limits<std::int64_t>::min() / 2;
                }
                const gpudb::Dtype d2[2] = {gpudb::Dtype::I64, gpudb::Dtype::I64};
                gpudb::Aggregator::RowSpan bs; bs.lanes = big.data(); bs.rows = M; bs.n_lanes = 2;
                auto bc = agg->upload_rows_exact(&bs, 1, d2, 2);
                auto rbc = ref_agg->upload_rows_exact(&bs, 1, d2, 2);
                gpudb::MultiPayload mp{bc[1].get(), gpudb::GroupByFilter::kAllColumns};
                gpudb::MultiPayload rmp{rbc[1].get(), gpudb::GroupByFilter::kAllColumns};
                gpudb::Predicate p; p.col = bc[1].get(); p.op = Op::GT; p.value = 0;
                gpudb::Predicate rp = p; rp.col = rbc[1].get();
                auto got = agg->aggregate_exact_masked(&mp, 1, &p, 1);
                auto want = ref_agg->aggregate_exact_masked(&rmp, 1, &rp, 1);
                EXPECT(tuple_of(got) == tuple_of(want));
                EXPECT(got.sums_hi[0] != 0);           // the sum really left 64 bits
            }
            // an empty column: one result, zeros
            {
                const gpudb::Dtype d2[2] = {gpudb::Dtype::I64, gpudb::Dtype::I64};
                gpudb::Aggregator::RowSpan es; es.lanes = nullptr; es.rows = 0; es.n_lanes = 2;
                auto ec = agg->upload_rows_exact(&es, 1, d2, 2);
                gpudb::MultiPayload mp{ec[1].get(), gpudb::GroupByFilter::kAllColumns};
                auto got = agg->aggregate_exact_masked(&mp, 1, nullptr, 0);
                EXPECT_EQ(got.count_star, 0);
                EXPECT_EQ(got.counts[0], 0);
                EXPECT_EQ(got.rows_in, 0u);
            }
            // row counts must agree, and neither a payload nor a predicate is an error
            {
                std::vector<std::int64_t> other(N + 1, 0);
                auto oc = agg->upload_i64(other.data(), N + 1);
                gpudb::Predicate bad; bad.col = oc.get(); bad.op = Op::GT; bad.value = 0;
                gpudb::MultiPayload mp{cols[1].get(), gpudb::GroupByFilter::kAllColumns};
                bool threw = false;
                try { (void)agg->aggregate_exact_masked(&mp, 1, &bad, 1); }
                catch (const std::runtime_error&) { threw = true; }
                EXPECT(threw);
                threw = false;
                try { (void)agg->aggregate_exact_masked(nullptr, 0, nullptr, 0); }
                catch (const std::runtime_error&) { threw = true; }
                EXPECT(threw);
            }
        } catch (const std::runtime_error& e) {
            if (std::string(e.what()).find("not implemented") != std::string::npos) {
                implemented = false;
                std::printf("    SKIP (%s)\n", e.what());
            } else {
                ++failures; ++total;
                std::printf("    FAIL: %s\n", e.what());
            }
        }
        if (implemented) std::printf("    ok\n");
    }

    // ---- Materialised key join (v0.7 §4.8) vs a host reference ----
    // The join output is an ordinary exact row set, so it is checked through
    // groupby_exact_[masked_]resident over its lanes: key from the probe or
    // the build side, gathered I64 / F64 lanes with NULLs as predicates, a
    // bitmap build key, a chained second join, and the error rules.
    {
        std::printf("  materialised key join:\n");
        using Op = gpudb::Predicate::Op;
        using Opt = std::pair<bool, std::int64_t>;               // (valid, value)
        bool implemented = true;
        try {
            std::mt19937_64 rng(0x701AULL);
            std::uniform_int_distribution<int> pct(0, 99);
            const std::size_t D = 50'003, N = 300'007;
            const std::size_t cap = std::size_t(100) * 1000000;
            auto bits_of = [](double d) { std::int64_t b; std::memcpy(&b, &d, sizeof(b)); return b; };
            auto dbl_of  = [](std::int64_t b) { double d; std::memcpy(&d, &b, sizeof(d)); return d; };
            // build side: unique key (1% NULL), I64 attr (5% NULL), F64 attr (5% NULL)
            std::vector<Opt> bkey(D), battr(D), bf(D);
            {
                std::vector<std::int64_t> ids(D);
                for (std::size_t i = 0; i < D; ++i) ids[i] = static_cast<std::int64_t>(i) * 3 - 1000;
                std::shuffle(ids.begin(), ids.end(), rng);
                std::uniform_int_distribution<std::int64_t> ad(-50, 50);
                std::uniform_real_distribution<double> fd(0.0, 1.0);
                for (std::size_t i = 0; i < D; ++i) {
                    bkey[i]  = {pct(rng) >= 1, ids[i]};
                    battr[i] = {pct(rng) >= 5, ad(rng)};
                    bf[i]    = {pct(rng) >= 5, bits_of(fd(rng))};
                }
            }
            // probe side: group key (3% NULL), payload spanning int64 (3% NULL), fk (5% NULL, 15% dangling)
            std::vector<Opt> pgk(N), pval(N), pfk(N);
            {
                std::uniform_int_distribution<std::int64_t> gd(-2000, 2000);
                std::uniform_int_distribution<std::int64_t> vd(std::numeric_limits<std::int64_t>::min() / 2,
                                                               std::numeric_limits<std::int64_t>::max() / 2);
                std::uniform_int_distribution<std::size_t> pick(0, D - 1);
                for (std::size_t i = 0; i < N; ++i) {
                    pgk[i]  = {pct(rng) >= 3, gd(rng)};
                    pval[i] = {pct(rng) >= 3, vd(rng)};
                    const int p = pct(rng);
                    if (p < 5)       pfk[i] = {false, 0};
                    else if (p < 20) pfk[i] = {true, static_cast<std::int64_t>(pick(rng)) * 3 - 999};   // never a build key
                    else             pfk[i] = {true, bkey[pick(rng)].second};                          // may hit a NULL-key build row's value: no match
                }
            }
            // Upload helper: lanes of Opt columns, lane 0 = key layout.
            auto upload = [&](const std::vector<const std::vector<Opt>*>& cols, const std::vector<gpudb::Dtype>& dts) {
                const std::size_t rows = cols[0]->size(), L = cols.size();
                std::vector<std::int64_t> lanes(rows * L);
                std::vector<std::vector<std::uint64_t>> valid(L, std::vector<std::uint64_t>((rows + 63) / 64, ~std::uint64_t{0}));
                for (std::size_t l = 0; l < L; ++l)
                    for (std::size_t i = 0; i < rows; ++i) {
                        lanes[i * L + l] = (*cols[l])[i].second;
                        if (!(*cols[l])[i].first) valid[l][i >> 6] &= ~(std::uint64_t{1} << (i & 63));
                    }
                std::vector<const std::uint64_t*> vp(L);
                for (std::size_t l = 0; l < L; ++l) vp[l] = valid[l].data();
                gpudb::Aggregator::RowSpan sp; sp.lanes = lanes.data(); sp.rows = rows; sp.n_lanes = L; sp.valid = vp.data();
                return agg->upload_rows_exact(&sp, 1, dts.data(), L);
            };
            using DT = gpudb::Dtype;
            auto bset = upload({&bkey, &battr, &bf}, {DT::I64, DT::I64, DT::F64});           // build key = key layout
            auto bset2 = upload({&battr, &battr, &bkey, &bf}, {DT::I64, DT::I64, DT::I64, DT::F64});   // build key under a bitmap
            auto pset = upload({&pgk, &pval, &pfk}, {DT::I64, DT::I64, DT::I64});

            // Host join: probe row -> build row (or none).
            std::map<std::int64_t, std::size_t> bmap;
            for (std::size_t i = 0; i < D; ++i) if (bkey[i].first) bmap[bkey[i].second] = i;
            std::vector<std::int64_t> hit(N, -1);
            std::size_t matched = 0;
            for (std::size_t i = 0; i < N; ++i) {
                if (!pfk[i].first) continue;
                auto it = bmap.find(pfk[i].second);
                if (it != bmap.end()) { hit[i] = static_cast<std::int64_t>(it->second); ++matched; }
            }
            struct Acc { gpudb::Sum128 s; std::int64_t cnt = 0, cstar = 0, mn = 0, mx = 0; };
            // Reference GROUP BY over joined rows; `keep(i)` is the WHERE.
            auto reference = [&](std::function<Opt(std::size_t)> key, std::function<bool(std::size_t)> keep) {
                std::map<std::pair<int, std::int64_t>, Acc> g;      // (is_null, key): NULL group last
                for (std::size_t i = 0; i < N; ++i) {
                    if (hit[i] < 0 || !keep(i)) continue;
                    const Opt k = key(i);
                    Acc& a = g[{k.first ? 0 : 1, k.first ? k.second : 0}];
                    ++a.cstar;
                    if (pval[i].first) {
                        const std::int64_t v = pval[i].second;
                        if (a.cnt == 0) { a.mn = v; a.mx = v; } else { a.mn = std::min(a.mn, v); a.mx = std::max(a.mx, v); }
                        a.s.add(v); ++a.cnt;
                    }
                }
                return g;
            };
            auto same = [&](const gpudb::GroupByResidentResult& got,
                            const std::map<std::pair<int, std::int64_t>, Acc>& want, const char* what) {
                bool ok = got.keys.size() == want.size();
                std::size_t q = 0;
                for (auto it = want.begin(); ok && it != want.end(); ++it, ++q) {
                    const Acc& a = it->second;
                    ok = got.key_null[q] == it->first.first && (it->first.first || got.keys[q] == it->first.second) &&
                         got.counts[q] == a.cnt && got.counts_star[q] == a.cstar &&
                         (a.cnt == 0 || (static_cast<std::uint64_t>(got.sums[q]) == a.s.lo && got.sums_hi[q] == a.s.hi &&
                                         got.mins[q] == a.mn && got.maxs[q] == a.mx));
                }
                if (!ok) std::printf("    FAIL %s: got %zu groups, ref %zu\n", what, got.keys.size(), want.size());
                return ok;
            };

            // A: key from the probe side; gathered I64 + F64 lanes as predicates.
            for (int variant = 0; variant < 2; ++variant) {
                const auto& bs = variant == 0 ? bset : bset2;
                const gpudb::ResidentColumn& bk = variant == 0 ? *bs[0] : *bs[2];
                const gpudb::ResidentColumn& ba = variant == 0 ? *bs[1] : *bs[0];   // variant 1: a key-layout source lane
                const gpudb::ResidentColumn& bfl = variant == 0 ? *bs[2] : *bs[3];
                gpudb::JoinLane outA[4] = {{pset[0].get(), false}, {pset[1].get(), false}, {&ba, true}, {&bfl, true}};
                auto ja = agg->join_materialize(*pset[2], bk, outA, 4);
                std::size_t null_keys = 0;
                for (std::size_t i = 0; i < N; ++i) null_keys += (hit[i] >= 0 && !pgk[i].first);
                EXPECT_EQ(ja.rows_out, matched);
                EXPECT_EQ(ja.null_key_rows, null_keys);
                EXPECT_EQ(ja.rows_probe, N);
                EXPECT_EQ(ja.rows_build, D);
                EXPECT_EQ(ja.lanes.size(), std::size_t(4));
                EXPECT(ja.lanes[3]->dtype() == DT::F64);
                auto got = agg->groupby_exact_resident(*ja.lanes[0], ja.lanes[1].get(), cap);
                EXPECT(same(got, reference([&](std::size_t i) { return pgk[i]; }, [](std::size_t) { return true; }), "join A plain"));
                gpudb::Predicate pr[2];
                pr[0].col = ja.lanes[2].get(); pr[0].op = Op::GE; pr[0].value = 0;
                pr[1].col = ja.lanes[3].get(); pr[1].op = Op::LT; pr[1].value = bits_of(0.5);
                auto gotw = agg->groupby_exact_masked_resident(*ja.lanes[0], ja.lanes[1].get(), pr, 2, cap);
                EXPECT(same(gotw, reference([&](std::size_t i) { return pgk[i]; },
                    [&](std::size_t i) {
                        const Opt a = battr[hit[i]], f = bf[hit[i]];
                        return a.first && a.second >= 0 && f.first && dbl_of(f.second) < 0.5;
                    }), "join A where"));
                gpudb::Predicate isn; isn.col = ja.lanes[3].get(); isn.op = Op::IsNull;
                auto gotn = agg->groupby_exact_masked_resident(*ja.lanes[0], ja.lanes[1].get(), &isn, 1, cap);
                EXPECT(same(gotn, reference([&](std::size_t i) { return pgk[i]; },
                    [&](std::size_t i) { return !bf[hit[i]].first; }), "join A is null"));
            }
            // B: key from the build side (NULL attr -> the NULL-key suffix).
            gpudb::JoinLane outB[3] = {{bset[1].get(), true}, {pset[1].get(), false}, {pset[0].get(), false}};
            auto jb = agg->join_materialize(*pset[2], *bset[0], outB, 3);
            {
                std::size_t null_keys = 0;
                for (std::size_t i = 0; i < N; ++i) null_keys += (hit[i] >= 0 && !battr[hit[i]].first);
                EXPECT_EQ(jb.rows_out, matched);
                EXPECT_EQ(jb.null_key_rows, null_keys);
                EXPECT(null_keys > 0);
                auto got = agg->groupby_exact_resident(*jb.lanes[0], jb.lanes[1].get(), cap);
                EXPECT(same(got, reference([&](std::size_t i) { return battr[hit[i]]; }, [](std::size_t) { return true; }), "join B plain"));
                gpudb::Predicate p; p.col = jb.lanes[2].get(); p.op = Op::LT; p.value = 0;
                auto gotw = agg->groupby_exact_masked_resident(*jb.lanes[0], jb.lanes[1].get(), &p, 1, cap);
                EXPECT(same(gotw, reference([&](std::size_t i) { return battr[hit[i]]; },
                    [&](std::size_t i) { return pgk[i].first && pgk[i].second < 0; }), "join B where"));
            }
            // C: chain — B's key lane (NULL suffix) probes a second dimension.
            {
                std::vector<Opt> d2k, d2a;
                for (std::int64_t k = -50; k <= 50; ++k) if (k % 7 != 0) { d2k.push_back({true, k}); d2a.push_back({k % 5 != 0, k * 11}); }
                auto d2 = upload({&d2k, &d2a}, {DT::I64, DT::I64});
                gpudb::JoinLane outC[2] = {{d2[1].get(), true}, {jb.lanes[1].get(), false}};
                auto jc = agg->join_materialize(*jb.lanes[0], *d2[0], outC, 2);
                std::map<std::pair<int, std::int64_t>, Acc> want;
                std::size_t rows = 0;
                for (std::size_t i = 0; i < N; ++i) {
                    if (hit[i] < 0 || !battr[hit[i]].first) continue;
                    const std::int64_t k = battr[hit[i]].second;
                    if (k % 7 == 0) continue;
                    ++rows;
                    Acc& a = want[{k % 5 != 0 ? 0 : 1, k % 5 != 0 ? k * 11 : 0}];
                    ++a.cstar;
                    if (pval[i].first) {
                        const std::int64_t v = pval[i].second;
                        if (a.cnt == 0) { a.mn = v; a.mx = v; } else { a.mn = std::min(a.mn, v); a.mx = std::max(a.mx, v); }
                        a.s.add(v); ++a.cnt;
                    }
                }
                EXPECT_EQ(jc.rows_out, rows);
                auto got = agg->groupby_exact_resident(*jc.lanes[0], jc.lanes[1].get(), cap);
                EXPECT(same(got, want, "join C chain"));
            }
            // Empty sides.
            {
                std::vector<Opt> none;
                auto e = upload({&none, &none}, {DT::I64, DT::I64});
                gpudb::JoinLane o1[1] = {{pset[0].get(), false}};
                auto j1 = agg->join_materialize(*pset[2], *e[0], o1, 1);
                EXPECT_EQ(j1.rows_out, std::size_t(0));
                gpudb::JoinLane o2[1] = {{e[1].get(), false}};
                auto j2 = agg->join_materialize(*e[0], *bset[0], o2, 1);
                EXPECT_EQ(j2.rows_out, std::size_t(0));
                auto g0 = agg->groupby_exact_resident(*j1.lanes[0], nullptr, cap);
                EXPECT_EQ(g0.keys.size(), std::size_t(0));
            }
            // Errors: duplicate build key, lane row count, F64 key lane.
            {
                std::vector<Opt> dk = {{true, 5}, {true, 9}, {true, 5}}, da = {{true, 1}, {true, 2}, {true, 3}};
                auto dup = upload({&dk, &da}, {DT::I64, DT::I64});
                gpudb::JoinLane o[1] = {{pset[0].get(), false}};
                bool threw = false;
                try { (void)agg->join_materialize(*pset[2], *dup[0], o, 1); }
                catch (const std::runtime_error& e) { threw = std::string(e.what()).find("not unique") != std::string::npos; }
                EXPECT(threw);
                // duplicates hidden behind NULLs are fine
                std::vector<Opt> nk = {{false, 5}, {true, 9}, {false, 5}};
                auto nul = upload({&da, &da, &nk}, {DT::I64, DT::I64, DT::I64});
                bool ok = true;
                try { (void)agg->join_materialize(*pset[2], *nul[2], o, 1); } catch (const std::runtime_error&) { ok = false; }
                EXPECT(ok);
                gpudb::JoinLane wrong[1] = {{pset[0].get(), true}};       // a probe lane declared as build
                threw = false;
                try { (void)agg->join_materialize(*pset[2], *bset[0], wrong, 1); } catch (const std::runtime_error&) { threw = true; }
                EXPECT(threw);
                gpudb::JoinLane fkey[1] = {{bset[2].get(), true}};        // F64 key lane
                threw = false;
                try { (void)agg->join_materialize(*pset[2], *bset[0], fkey, 1); } catch (const std::runtime_error&) { threw = true; }
                EXPECT(threw);
            }
        } catch (const std::runtime_error& e) {
            if (std::string(e.what()).find("not implemented") != std::string::npos) {
                implemented = false;
                std::printf("    SKIP (%s)\n", e.what());
            } else {
                ++failures; ++total;
                std::printf("    FAIL: %s\n", e.what());
            }
        }
        if (implemented) std::printf("    ok\n");
    }

    // ---- Several payload columns in one GROUP BY (v0.7 §4.9) ----
    // groupby_exact_masked_multi must equal one single-payload call per
    // column, row for row: plain, both WHERE-mask variants (a selective and
    // an unselective mask), a key-range predicate, HAVING and top-k on a
    // payload other than the first, the NULL-key group among the survivors.
    {
        // upload_rows_exact from SEVERAL spans (what an upload session hands over): a backend may
        // copy spans in parallel, so the layout must not depend on it — same groups, sums, counts,
        // mins, maxs and NULL-key group as ONE span of the same rows on the CPU reference. Spans:
        // NULL-free ones (the fast path), ones with NULL keys and NULL payloads, an empty one, and a
        // bitmap that carries stale zero bits past the span's rows (a shared segment does).
        std::printf("  exact upload from many spans:\n");
        try {
            std::mt19937_64 rng(0x5EA9ULL);
            const std::size_t N = 2'600'037, L = 3;
            const std::size_t cap = std::size_t(100) * 1000000;
            std::uniform_int_distribution<int> pct(0, 99);
            std::uniform_int_distribution<std::int64_t> kd(-4000, 4000), vd(-1000000, 1000000), sel(0, 999);
            std::vector<std::int64_t> lanes(N * L);
            for (std::size_t i = 0; i < N; ++i) { lanes[i * L] = kd(rng); lanes[i * L + 1] = vd(rng); lanes[i * L + 2] = sel(rng); }
            const std::size_t cuts[] = {0, 300'001, 300'001, 911'000, 1'048'576, 1'500'003, 1'500'067, 2'200'000, 2'599'999, N};
            const std::size_t n_spans = sizeof(cuts) / sizeof(cuts[0]) - 1;
            std::vector<gpudb::Aggregator::RowSpan> spans(n_spans);
            std::vector<std::vector<std::vector<std::uint64_t>>> sv(n_spans);      // [span][lane] bitmap, row-local
            std::vector<std::vector<const std::uint64_t*>> svp(n_spans, std::vector<const std::uint64_t*>(L, nullptr));
            std::vector<std::vector<std::uint64_t>> whole(L, std::vector<std::uint64_t>((N + 63) / 64, ~std::uint64_t{0}));
            for (std::size_t sidx = 0; sidx < n_spans; ++sidx) {
                const std::size_t a = cuts[sidx], b = cuts[sidx + 1], r = b - a;
                spans[sidx].lanes = lanes.data() + a * L; spans[sidx].rows = r; spans[sidx].n_lanes = L;
                const bool with_nulls = (sidx % 2 == 1) || sidx == 6;           // the others stay NULL-free
                if (with_nulls) {
                    sv[sidx].assign(L, std::vector<std::uint64_t>((r + 63) / 64 + 1, ~std::uint64_t{0}));
                    const int nullp[3] = {sidx == 6 ? 60 : 3, 9, 0};
                    for (std::size_t j = 0; j < r; ++j)
                        for (std::size_t l = 0; l < 2; ++l)
                            if (pct(rng) < nullp[l]) {
                                sv[sidx][l][j >> 6] &= ~(std::uint64_t{1} << (j & 63));
                                whole[l][(a + j) >> 6] &= ~(std::uint64_t{1} << ((a + j) & 63));
                            }
                    // stale zero bits past the span's rows must not be counted as NULL keys
                    for (std::size_t j = r; j < ((r + 63) / 64 + 1) * 64; ++j) sv[sidx][0][j >> 6] &= ~(std::uint64_t{1} << (j & 63));
                    svp[sidx][0] = sv[sidx][0].data(); svp[sidx][1] = sv[sidx][1].data();
                    spans[sidx].valid = svp[sidx].data();
                } else {
                    spans[sidx].valid = (sidx % 4 == 0) ? nullptr : svp[sidx].data();   // no array / an array of null pointers
                }
            }
            const gpudb::Dtype dts[3] = {gpudb::Dtype::I64, gpudb::Dtype::I64, gpudb::Dtype::I64};
            auto cols = agg->upload_rows_exact(spans.data(), n_spans, dts, L);
            auto ref_agg = gpudb::make_aggregator(gpudb::Backend::CPU);
            std::vector<const std::uint64_t*> wp(L);
            for (std::size_t l = 0; l < L; ++l) wp[l] = whole[l].data();
            gpudb::Aggregator::RowSpan one; one.lanes = lanes.data(); one.rows = N; one.n_lanes = L; one.valid = wp.data();
            auto ref_cols = ref_agg->upload_rows_exact(&one, 1, dts, L);
            EXPECT(cols.size() == L && cols[0]->rows() == N && cols[1]->null_count() == ref_cols[1]->null_count());
            for (int with_pred = 0; with_pred < 2; ++with_pred) {
                gpudb::Predicate p, rp;
                p.col = cols[2].get(); p.op = gpudb::Predicate::Op::LT; p.value = 250;
                rp = p; rp.col = ref_cols[2].get();
                auto got = agg->groupby_exact_masked_resident(*cols[0], cols[1].get(), with_pred ? &p : nullptr, with_pred, cap);
                auto want = ref_agg->groupby_exact_masked_resident(*ref_cols[0], ref_cols[1].get(), with_pred ? &rp : nullptr, with_pred, cap);
                using Row = std::tuple<int, std::int64_t, std::uint64_t, std::int64_t, std::int64_t, std::int64_t, std::int64_t, std::int64_t>;
                auto rows_of = [](const decltype(got)& g) {
                    std::vector<Row> out;
                    for (std::size_t i = 0; i < g.keys.size(); ++i)
                        out.emplace_back(g.key_null[i], g.key_null[i] ? 0 : g.keys[i], static_cast<std::uint64_t>(g.sums[i]), g.sums_hi[i],
                                         g.counts[i], g.counts_star[i], g.counts[i] ? g.mins[i] : 0, g.counts[i] ? g.maxs[i] : 0);
                    std::sort(out.begin(), out.end());
                    return out;
                };
                const bool ok = rows_of(got) == rows_of(want) && got.keys.size() > 7000;
                if (!ok) std::printf("    FAIL many spans (%s): %zu groups vs %zu\n", with_pred ? "WHERE" : "plain", got.keys.size(), want.keys.size());
                EXPECT(ok);
            }
        } catch (const std::runtime_error& e) {
            if (std::string(e.what()).find("not implemented") != std::string::npos) std::printf("    SKIP (%s)\n", e.what());
            else { ++failures; ++total; std::printf("    FAIL: %s\n", e.what()); }
        }

        // Stage B primitive: spans that name their destination row and their bitmap bit offset
        // (chunks scanned in any order land at their row-id rank; a chunk that starts mid-segment
        // reads its bitmap from that bit). Placed in shuffled order + offsets == one span.
        std::printf("  exact upload from placed spans:\n");
        try {
            std::mt19937_64 rng(0x9E1ACEULL);
            const std::size_t N = 1'300'011, L = 3;
            const std::size_t cap = std::size_t(100) * 1000000;
            std::uniform_int_distribution<int> pct(0, 99);
            std::uniform_int_distribution<std::int64_t> kd(-3000, 3000), vd(-500000, 500000), sel(0, 99);
            // the "segment": rows in SCAN order, with a bitmap per lane over the whole segment
            std::vector<std::int64_t> lanes(N * L);
            std::vector<std::vector<std::uint64_t>> seg_valid(L, std::vector<std::uint64_t>((N + 63) / 64, ~std::uint64_t{0}));
            for (std::size_t i = 0; i < N; ++i) {
                lanes[i * L] = kd(rng); lanes[i * L + 1] = vd(rng); lanes[i * L + 2] = sel(rng);
                const int nullp[3] = {4, 10, 0};
                for (std::size_t l = 0; l < L; ++l)
                    if (pct(rng) < nullp[l]) seg_valid[l][i >> 6] &= ~(std::uint64_t{1} << (i & 63));
            }
            // chunks of odd sizes, each with a destination = its rank in row-id order, presented shuffled
            std::vector<std::size_t> cuts{0};
            std::uniform_int_distribution<std::size_t> csz(1, 5000);
            while (cuts.back() < N) cuts.push_back(std::min(N, cuts.back() + csz(rng)));
            const std::size_t n_chunks = cuts.size() - 1;
            std::vector<std::size_t> order(n_chunks);
            for (std::size_t c = 0; c < n_chunks; ++c) order[c] = c;
            std::shuffle(order.begin(), order.end(), rng);
            std::vector<gpudb::Aggregator::RowSpan> spans(n_chunks);
            std::vector<std::vector<const std::uint64_t*>> vps(n_chunks, std::vector<const std::uint64_t*>(L));
            for (std::size_t x = 0; x < n_chunks; ++x) {
                const std::size_t c = order[x], a = cuts[c], b = cuts[c + 1];
                spans[x].lanes = lanes.data() + a * L; spans[x].rows = b - a; spans[x].n_lanes = L;
                for (std::size_t l = 0; l < L; ++l) vps[x][l] = seg_valid[l].data();
                spans[x].valid = vps[x].data();
                spans[x].dst_row = a;                 // its rank in row-id order
                spans[x].valid_bit = a;               // its bitmap starts at bit a of the segment's bitmap
            }
            const gpudb::Dtype dts[3] = {gpudb::Dtype::I64, gpudb::Dtype::I64, gpudb::Dtype::I64};
            auto cols = agg->upload_rows_exact(spans.data(), n_chunks, dts, L);
            auto ref_agg = gpudb::make_aggregator(gpudb::Backend::CPU);
            std::vector<const std::uint64_t*> wp(L);
            for (std::size_t l = 0; l < L; ++l) wp[l] = seg_valid[l].data();
            gpudb::Aggregator::RowSpan one; one.lanes = lanes.data(); one.rows = N; one.n_lanes = L; one.valid = wp.data();
            auto ref_cols = ref_agg->upload_rows_exact(&one, 1, dts, L);
            EXPECT(cols.size() == L && cols[0]->rows() == N && cols[0]->null_count() == ref_cols[0]->null_count()
                   && cols[1]->null_count() == ref_cols[1]->null_count());
            gpudb::Predicate p, rp;
            p.col = cols[2].get(); p.op = gpudb::Predicate::Op::GE; p.value = 30;
            rp = p; rp.col = ref_cols[2].get();
            auto got = agg->groupby_exact_masked_resident(*cols[0], cols[1].get(), &p, 1, cap);
            auto want = ref_agg->groupby_exact_masked_resident(*ref_cols[0], ref_cols[1].get(), &rp, 1, cap);
            using Row = std::tuple<int, std::int64_t, std::uint64_t, std::int64_t, std::int64_t, std::int64_t, std::int64_t, std::int64_t>;
            auto rows_of = [](const decltype(got)& g) {
                std::vector<Row> out;
                for (std::size_t i = 0; i < g.keys.size(); ++i)
                    out.emplace_back(g.key_null[i], g.key_null[i] ? 0 : g.keys[i], static_cast<std::uint64_t>(g.sums[i]), g.sums_hi[i],
                                     g.counts[i], g.counts_star[i], g.counts[i] ? g.mins[i] : 0, g.counts[i] ? g.maxs[i] : 0);
                std::sort(out.begin(), out.end());
                return out;
            };
            const bool ok = rows_of(got) == rows_of(want) && got.keys.size() > 5000;
            if (!ok) std::printf("    FAIL placed spans: %zu groups vs %zu\n", got.keys.size(), want.keys.size());
            EXPECT(ok);
            // the join sees row-aligned lanes too: a self join on the key lane against a unique build
            std::vector<std::int64_t> bk(6001), bv(6001);
            for (std::size_t i = 0; i < 6001; ++i) { bk[i] = static_cast<std::int64_t>(i) - 3000; bv[i] = static_cast<std::int64_t>(i) * 7; }
            gpudb::Aggregator::RowSpan bs; bs.lanes = nullptr;
            std::vector<std::int64_t> bl(6001 * 2);
            for (std::size_t i = 0; i < 6001; ++i) { bl[2 * i] = bk[i]; bl[2 * i + 1] = bv[i]; }
            bs.lanes = bl.data(); bs.rows = 6001; bs.n_lanes = 2;
            const gpudb::Dtype d2[2] = {gpudb::Dtype::I64, gpudb::Dtype::I64};
            auto bcols = agg->upload_rows_exact(&bs, 1, d2, 2);
            auto rbcols = ref_agg->upload_rows_exact(&bs, 1, d2, 2);
            gpudb::JoinLane jl[2]; jl[0].col = cols[1].get(); jl[0].from_build = false; jl[1].col = bcols[1].get(); jl[1].from_build = true;
            gpudb::JoinLane rjl[2]; rjl[0].col = ref_cols[1].get(); rjl[0].from_build = false; rjl[1].col = rbcols[1].get(); rjl[1].from_build = true;
            auto jr = agg->join_materialize(*cols[0], *bcols[0], jl, 2);
            auto rjr = ref_agg->join_materialize(*ref_cols[0], *rbcols[0], rjl, 2);
            auto jgot = agg->groupby_exact_resident(*jr.lanes[1], jr.lanes[0].get(), cap);
            auto jwant = ref_agg->groupby_exact_resident(*rjr.lanes[1], rjr.lanes[0].get(), cap);
            const bool jok = jr.rows_out == rjr.rows_out && rows_of(jgot) == rows_of(jwant) && jgot.keys.size() > 5000;
            if (!jok) std::printf("    FAIL placed spans join: %zu vs %zu rows out\n", jr.rows_out, rjr.rows_out);
            EXPECT(jok);
        } catch (const std::runtime_error& e) {
            if (std::string(e.what()).find("not implemented") != std::string::npos) std::printf("    SKIP (%s)\n", e.what());
            else { ++failures; ++total; std::printf("    FAIL: %s\n", e.what()); }
        }

        // ---- narrow lane storage (docs/RESIDENT_COLUMNS_DESIGN.md §6, stage C) ----
        // An exact I64 lane is stored at the narrowest signed width its values
        // fit. Lanes that sit exactly on the I8 / I16 / I32 boundaries and one
        // past each, an all-NULL lane and a 64-bit hash-like lane go through
        // every exact operator — plain, masked, HAVING, top-k, several
        // payloads, a join with narrow lanes on both sides — and must match the
        // CPU reference bit for bit. The widths are read back through
        // resident_bytes(): the reference stores I64, so the expectation is the
        // narrow byte count, not the reference's.
        std::printf("  narrow lane storage:\n");
        try {
            std::mt19937_64 rng(0xC0FFEEULL);
            const std::size_t N = 400'009, L = 8;
            const std::size_t cap = std::size_t(100) * 1000000;
            std::uniform_int_distribution<int> pct(0, 99);
            std::uniform_int_distribution<std::int64_t>
                k16(-32768, 32767), d8(-128, 127), d9(-129, 128), d17(-32769, 32768),
                d32(-2147483648LL, 2147483647LL), d33(-2147483649LL, 2147483648LL);
            std::uniform_int_distribution<std::uint64_t> h64(0, ~std::uint64_t{0});
            std::vector<std::int64_t> lanes(N * L);
            std::vector<std::vector<std::uint64_t>> vb(L, std::vector<std::uint64_t>((N + 63) / 64, ~std::uint64_t{0}));
            auto clr = [&](std::size_t l, std::size_t i) { vb[l][i >> 6] &= ~(std::uint64_t{1} << (i & 63)); };
            for (std::size_t i = 0; i < N; ++i) {
                lanes[i * L + 0] = k16(rng);                                // key, I16 range
                lanes[i * L + 1] = d8(rng);                                 // I8 range
                lanes[i * L + 2] = d9(rng);                                 // one past I8
                lanes[i * L + 3] = d17(rng);                                // one past I16
                lanes[i * L + 4] = d32(rng);                                // I32 range
                lanes[i * L + 5] = d33(rng);                                // one past I32
                lanes[i * L + 6] = 0;                                       // all NULL
                lanes[i * L + 7] = static_cast<std::int64_t>(h64(rng));     // a string-hash lane
                if (pct(rng) < 5) clr(0, i);
                if (pct(rng) < 7) clr(1, i);
                if (pct(rng) < 3) clr(4, i);
                clr(6, i);
            }
            // the boundaries themselves, in two rows that stay valid
            const std::int64_t ends[L][2] = {
                {-32768, 32767}, {-128, 127}, {-129, 128}, {-32769, 32768},
                {-2147483648LL, 2147483647LL}, {-2147483649LL, 2147483648LL}, {0, 0},
                {std::numeric_limits<std::int64_t>::min(), std::numeric_limits<std::int64_t>::max()} };
            for (std::size_t l = 0; l < L; ++l) {
                lanes[0 * L + l] = ends[l][0];
                lanes[1 * L + l] = ends[l][1];
                if (l != 6) vb[l][0] |= 3u;                                 // rows 0 and 1 valid
            }
            const unsigned want_w[L] = {2, 1, 2, 4, 4, 8, 1, 8};
            const gpudb::Dtype dts[L] = {gpudb::Dtype::I64, gpudb::Dtype::I64, gpudb::Dtype::I64, gpudb::Dtype::I64,
                                         gpudb::Dtype::I64, gpudb::Dtype::I64, gpudb::Dtype::I64, gpudb::Dtype::I64};
            const std::uint64_t* vptr[L];
            for (std::size_t l = 0; l < L; ++l) vptr[l] = vb[l].data();
            gpudb::Aggregator::RowSpan sp;
            sp.lanes = lanes.data(); sp.rows = N; sp.n_lanes = L; sp.valid = vptr;
            auto cols = agg->upload_rows_exact(&sp, 1, dts, L);
            auto ref_agg = gpudb::make_aggregator(gpudb::Backend::CPU);
            auto ref = ref_agg->upload_rows_exact(&sp, 1, dts, L);
            EXPECT_EQ(cols.size(), L);
            for (std::size_t l = 0; l < L; ++l) EXPECT_EQ(cols[l]->null_count(), ref[l]->null_count());
            EXPECT_EQ(cols[6]->null_count(), N);                            // the all-NULL lane
            if (agg->backend() == gpudb::Backend::METAL) {
                const std::size_t bm = ((N + 63) / 64) * sizeof(std::uint64_t);
                for (std::size_t l = 0; l < L; ++l) {
                    const std::size_t bits = cols[l]->null_count() ? bm : 0;
                    EXPECT_EQ(cols[l]->resident_bytes(), N * want_w[l] + bits);
                    if (want_w[l] < 8) EXPECT(cols[l]->resident_bytes() < N * 8);
                }
                // the sort cache is narrow too: sorted keys at the key's width
                // plus a u32 row id each
                const std::size_t before = cols[0]->resident_bytes();
                cols[0]->prepare();
                EXPECT_EQ(cols[0]->resident_bytes(),
                          before + (N - cols[0]->null_count()) * (want_w[0] + sizeof(std::uint32_t)));
                // the same lanes without any bitmap take the fast copy path and
                // land at the same widths
                gpudb::Aggregator::RowSpan bare;
                bare.lanes = lanes.data(); bare.rows = N; bare.n_lanes = L; bare.valid = nullptr;
                auto plain = agg->upload_rows_exact(&bare, 1, dts, L);
                for (std::size_t l = 0; l < L; ++l) EXPECT_EQ(plain[l]->resident_bytes(), N * want_w[l]);
            }

            using Row = std::tuple<int, std::int64_t, std::uint64_t, std::int64_t, std::int64_t, std::int64_t, std::int64_t, std::int64_t>;
            auto rows_of = [](const gpudb::GroupByResidentResult& g) {
                std::vector<Row> out;
                for (std::size_t i = 0; i < g.keys.size(); ++i)
                    out.emplace_back(g.key_null[i], g.key_null[i] ? 0 : g.keys[i], static_cast<std::uint64_t>(g.sums[i]), g.sums_hi[i],
                                     g.counts[i], g.counts_star[i], g.counts[i] ? g.mins[i] : 0, g.counts[i] ? g.maxs[i] : 0);
                std::sort(out.begin(), out.end());
                return out;
            };
            // plain: one call per payload lane, every width
            for (std::size_t l = 1; l < L; ++l) {
                auto got  = agg->groupby_exact_resident(*cols[0], cols[l].get(), cap);
                auto want = ref_agg->groupby_exact_resident(*ref[0], ref[l].get(), cap);
                const bool ok = rows_of(got) == rows_of(want) && got.keys.size() > 60000;
                if (!ok) std::printf("    FAIL narrow plain lane %zu: %zu groups vs %zu\n", l, got.keys.size(), want.keys.size());
                EXPECT(ok);
            }
            // masked: predicates on narrow lanes, on the one-past lanes, on the
            // all-NULL lane (IS NULL keeps every row) and an IN over the hash lane
            {
                std::vector<std::int64_t> in_list{lanes[7], lanes[1 * L + 7], lanes[5 * L + 7], 0};
                struct Case { const char* name; std::vector<std::size_t> pc; std::vector<gpudb::Predicate::Op> op; std::vector<std::int64_t> val; };
                const Case cases[] = {
                    {"narrow range", {2, 4},   {gpudb::Predicate::Op::GE, gpudb::Predicate::Op::LT}, {0, 0}},
                    {"one past",     {3, 5},   {gpudb::Predicate::Op::NE, gpudb::Predicate::Op::GT}, {0, -2147483648LL}},
                    {"all null",     {6},      {gpudb::Predicate::Op::IsNull},                       {0}},
                    {"key range",    {0, 1},   {gpudb::Predicate::Op::GE, gpudb::Predicate::Op::LE}, {-1000, 40}},
                    {"selective",    {4},      {gpudb::Predicate::Op::LT},                           {-2147000000LL}},
                };
                for (const auto& c : cases) {
                    std::vector<gpudb::Predicate> p(c.pc.size()), rp(c.pc.size());
                    for (std::size_t i = 0; i < c.pc.size(); ++i) {
                        p[i].col = cols[c.pc[i]].get(); p[i].op = c.op[i]; p[i].value = c.val[i];
                        rp[i] = p[i]; rp[i].col = ref[c.pc[i]].get();
                    }
                    auto got  = agg->groupby_exact_masked_resident(*cols[0], cols[1].get(), p.data(), p.size(), cap);
                    auto want = ref_agg->groupby_exact_masked_resident(*ref[0], ref[1].get(), rp.data(), rp.size(), cap);
                    const bool ok = rows_of(got) == rows_of(want);
                    if (!ok) std::printf("    FAIL narrow mask %s: %zu groups vs %zu\n", c.name, got.keys.size(), want.keys.size());
                    EXPECT(ok);
                }
                gpudb::Predicate p, rp;
                p.col = cols[7].get(); p.op = gpudb::Predicate::Op::In; p.list = in_list.data(); p.n_list = in_list.size();
                rp = p; rp.col = ref[7].get();
                auto got  = agg->groupby_exact_masked_resident(*cols[0], cols[4].get(), &p, 1, cap);
                auto want = ref_agg->groupby_exact_masked_resident(*ref[0], ref[4].get(), &rp, 1, cap);
                EXPECT(rows_of(got) == rows_of(want));
            }
            // HAVING and top-k over a narrow payload
            {
                gpudb::GroupByFilter f;
                f.columns = gpudb::GroupByFilter::kAllColumns;
                f.agg = gpudb::GroupByFilter::Agg::CountStar; f.cmp = gpudb::GroupByFilter::Cmp::GT; f.threshold_i64 = 6;
                auto got  = agg->groupby_exact_resident(*cols[0], cols[2].get(), cap, f);
                auto want = ref_agg->groupby_exact_resident(*ref[0], ref[2].get(), cap, f);
                EXPECT(rows_of(got) == rows_of(want) && !got.keys.empty());
                gpudb::GroupByFilter t;
                t.columns = gpudb::GroupByFilter::kAllColumns;
                t.agg = gpudb::GroupByFilter::Agg::Sum; t.topk = 25; t.topk_desc = true;
                auto gt = agg->groupby_exact_resident(*cols[0], cols[4].get(), cap, t);
                auto wt = ref_agg->groupby_exact_resident(*ref[0], ref[4].get(), cap, t);
                EXPECT(gt.keys == wt.keys && gt.sums == wt.sums && gt.sums_hi == wt.sums_hi
                       && gt.counts == wt.counts && gt.mins == wt.mins && gt.maxs == wt.maxs
                       && gt.keys.size() == 25);
            }
            // several payloads of different widths in one call
            {
                const std::size_t which[3] = {1, 5, 7};
                gpudb::MultiPayload mp[3], rmp[3];
                for (int i = 0; i < 3; ++i) {
                    mp[i].vals = cols[which[i]].get();  mp[i].columns = gpudb::GroupByFilter::kAllColumns;
                    rmp[i].vals = ref[which[i]].get();  rmp[i].columns = gpudb::GroupByFilter::kAllColumns;
                }
                gpudb::GroupByFilter f; f.columns = gpudb::GroupByFilter::kAllColumns;
                auto got  = agg->groupby_exact_masked_multi(*cols[0], mp, 3, 0, nullptr, 0, cap, f);
                auto want = ref_agg->groupby_exact_masked_multi(*ref[0], rmp, 3, 0, nullptr, 0, cap, f);
                bool ok = got.size() == want.size();
                for (std::size_t i = 0; ok && i < got.size(); ++i)
                    ok = got[i].sums == want[i].sums && got[i].sums_hi == want[i].sums_hi
                      && got[i].counts == want[i].counts && got[i].mins == want[i].mins && got[i].maxs == want[i].maxs;
                if (!ok) std::printf("    FAIL narrow multi payload\n");
                EXPECT(ok);
            }
            // join_materialize with narrow lanes on both sides: a unique build
            // key over the probe key's whole range, narrow and wide output lanes
            {
                const std::size_t B = 65536;
                std::vector<std::int64_t> bl(B * 2);
                for (std::size_t i = 0; i < B; ++i) {
                    bl[2 * i]     = static_cast<std::int64_t>(i) - 32768;        // I16 range: width 2
                    bl[2 * i + 1] = static_cast<std::int64_t>(i % 251) - 125;    // I8 range:  width 1
                }
                const gpudb::Dtype d2[2] = {gpudb::Dtype::I64, gpudb::Dtype::I64};
                gpudb::Aggregator::RowSpan bs;
                bs.lanes = bl.data(); bs.rows = B; bs.n_lanes = 2;
                auto bc  = agg->upload_rows_exact(&bs, 1, d2, 2);
                auto rbc = ref_agg->upload_rows_exact(&bs, 1, d2, 2);
                if (agg->backend() == gpudb::Backend::METAL) {
                    EXPECT_EQ(bc[0]->resident_bytes(), B * 2);
                    EXPECT_EQ(bc[1]->resident_bytes(), B * 1);
                }
                gpudb::JoinLane jl[3], rjl[3];
                const std::size_t src[3] = {1, 7, 0};
                const bool from_build[3] = {false, false, true};
                jl[0].col = cols[1].get();  jl[0].from_build = false;           // probe payload, width 1
                jl[1].col = cols[7].get();  jl[1].from_build = false;           // probe hash lane, width 8
                jl[2].col = bc[1].get();    jl[2].from_build = true;            // build payload, width 1
                rjl[0].col = ref[1].get();  rjl[0].from_build = false;
                rjl[1].col = ref[7].get();  rjl[1].from_build = false;
                rjl[2].col = rbc[1].get();  rjl[2].from_build = true;
                (void)src; (void)from_build;
                auto jr  = agg->join_materialize(*cols[0], *bc[0], jl, 3);
                auto rjr = ref_agg->join_materialize(*ref[0], *rbc[0], rjl, 3);
                EXPECT_EQ(jr.rows_out, rjr.rows_out);
                EXPECT_EQ(jr.null_key_rows, rjr.null_key_rows);
                if (agg->backend() == gpudb::Backend::METAL) {
                    // a gather cannot widen a lane: each output keeps its source width
                    const unsigned lane_w[3] = {1, 8, 1};
                    for (std::size_t l = 0; l < 3; ++l) {
                        const std::size_t bits = jr.lanes[l]->null_count()
                            ? ((jr.rows_out + 63) / 64) * sizeof(std::uint64_t) : 0;
                        EXPECT_EQ(jr.lanes[l]->resident_bytes(), jr.rows_out * lane_w[l] + bits);
                    }
                }
                auto jgot  = agg->groupby_exact_resident(*jr.lanes[2], jr.lanes[0].get(), cap);
                auto jwant = ref_agg->groupby_exact_resident(*rjr.lanes[2], rjr.lanes[0].get(), cap);
                EXPECT(rows_of(jgot) == rows_of(jwant) && jgot.keys.size() > 200);
                auto hgot  = agg->groupby_exact_resident(*jr.lanes[0], jr.lanes[1].get(), cap);
                auto hwant = ref_agg->groupby_exact_resident(*rjr.lanes[0], rjr.lanes[1].get(), cap);
                EXPECT(rows_of(hgot) == rows_of(hwant));
            }
            std::printf("    ok\n");
        } catch (const std::runtime_error& e) {
            if (std::string(e.what()).find("not implemented") != std::string::npos) std::printf("    SKIP (%s)\n", e.what());
            else { ++failures; ++total; std::printf("    FAIL: %s\n", e.what()); }
        }

        std::printf("  several payloads in one group by:\n");
        using Op = gpudb::Predicate::Op;
        using Cmp = gpudb::GroupByFilter::Cmp;
        using Agg = gpudb::GroupByFilter::Agg;
        bool implemented = true;
        try {
            std::mt19937_64 rng(0x3A11ULL);
            const std::size_t N = 250'019, L = 5;
            const std::size_t cap = std::size_t(100) * 1000000;
            std::uniform_int_distribution<int> pct(0, 99);
            std::uniform_int_distribution<std::int64_t> kd(-1500, 1500), sel(0, 999);
            std::uniform_int_distribution<std::int64_t> wide(std::numeric_limits<std::int64_t>::min() / 2,
                                                             std::numeric_limits<std::int64_t>::max() / 2);
            std::vector<std::int64_t> lanes(N * L);
            std::vector<std::vector<std::uint64_t>> valid(L, std::vector<std::uint64_t>((N + 63) / 64, ~std::uint64_t{0}));
            for (std::size_t i = 0; i < N; ++i) {
                lanes[i * L + 0] = kd(rng);
                lanes[i * L + 1] = wide(rng);            // payload 0: sums overflow 64 bits
                lanes[i * L + 2] = kd(rng) * 7;          // payload 1
                lanes[i * L + 3] = wide(rng) / 3;        // payload 2
                lanes[i * L + 4] = sel(rng);             // selector
                const int nullp[5] = {2, 4, 30, 1, 0};
                for (std::size_t l = 0; l < L; ++l)
                    if (pct(rng) < nullp[l]) valid[l][i >> 6] &= ~(std::uint64_t{1} << (i & 63));
            }
            std::vector<const std::uint64_t*> vp(L);
            for (std::size_t l = 0; l < L; ++l) vp[l] = valid[l].data();
            gpudb::Aggregator::RowSpan sp; sp.lanes = lanes.data(); sp.rows = N; sp.n_lanes = L; sp.valid = vp.data();
            const gpudb::Dtype dts[5] = {gpudb::Dtype::I64, gpudb::Dtype::I64, gpudb::Dtype::I64, gpudb::Dtype::I64, gpudb::Dtype::I64};
            auto cols = agg->upload_rows_exact(&sp, 1, dts, L);

            struct Case { const char* name; std::vector<gpudb::Predicate> preds; gpudb::GroupByFilter f; std::size_t fp; };
            auto pred = [&](std::size_t lane, Op op, std::int64_t v) { gpudb::Predicate p; p.col = cols[lane].get(); p.op = op; p.value = v; return p; };
            std::vector<Case> cases;
            cases.push_back({"plain", {}, {}, 0});
            cases.push_back({"mask 10% (compacted)", {pred(4, Op::LT, 100)}, {}, 0});
            cases.push_back({"mask 80% (in the reduce)", {pred(4, Op::GE, 200)}, {}, 0});
            cases.push_back({"key range + payload predicate", {pred(0, Op::GE, -200), pred(0, Op::LT, 900), pred(2, Op::IsNotNull, 0)}, {}, 0});
            { gpudb::GroupByFilter f; f.cmp = Cmp::GT; f.threshold_i64 = 0; f.agg = Agg::Sum;
              cases.push_back({"having on payload 1", {pred(4, Op::GE, 200)}, f, 1}); }
            { gpudb::GroupByFilter f; f.cmp = Cmp::GE; f.threshold_i64 = 80; f.agg = Agg::CountStar;
              cases.push_back({"having count(*)", {}, f, 0}); }
            { gpudb::GroupByFilter f; f.topk = 40; f.topk_desc = true; f.agg = Agg::Max;
              cases.push_back({"top-k on payload 2", {pred(4, Op::LT, 600)}, f, 2}); }
            { gpudb::GroupByFilter f; f.topk = 5000; f.topk_desc = false; f.agg = Agg::Sum;
              cases.push_back({"top-k larger than the groups", {}, f, 1}); }
            for (auto& c : cases) {
                gpudb::MultiPayload mp[3];
                for (int p = 0; p < 3; ++p) { mp[p].vals = cols[1 + p].get(); mp[p].columns = (1u << 1) | (1u << 2) | (1u << 4) | (1u << 5); }
                gpudb::GroupByFilter f = c.f; f.columns = 0x9;      // keys + count(*)
                auto got = agg->groupby_exact_masked_multi(*cols[0], mp, 3, c.fp, c.preds.data(), c.preds.size(), cap, f);
                // reference: the filtered payload alone, then every other payload unfiltered, matched by key
                gpudb::GroupByFilter fr = c.f;
                auto prim = agg->groupby_exact_masked_resident(*cols[0], cols[1 + c.fp].get(), c.preds.data(), c.preds.size(), cap, fr);
                bool ok = got.size() == 3 && got[c.fp].keys.size() == prim.keys.size() && got[c.fp].groups_total == prim.groups_total;
                std::map<std::pair<int, std::int64_t>, std::size_t> row_of;      // (is_null, key) -> row in got
                for (std::size_t i = 0; ok && i < got[c.fp].keys.size(); ++i)
                    row_of[{got[c.fp].key_null[i], got[c.fp].key_null[i] ? 0 : got[c.fp].keys[i]}] = i;
                ok = ok && row_of.size() == prim.keys.size();
                for (int p = 0; ok && p < 3; ++p) {
                    auto full = agg->groupby_exact_masked_resident(*cols[0], cols[1 + p].get(), c.preds.data(), c.preds.size(), cap);
                    ok = got[p].sums.size() == prim.keys.size() && got[p].counts.size() == prim.keys.size() &&
                         got[p].mins.size() == prim.keys.size() && got[p].maxs.size() == prim.keys.size();
                    std::size_t seen = 0;
                    for (std::size_t q = 0; ok && q < full.keys.size(); ++q) {
                        auto it = row_of.find({full.key_null[q], full.key_null[q] ? 0 : full.keys[q]});
                        if (it == row_of.end()) continue;               // dropped by the filter
                        const std::size_t i = it->second;
                        ++seen;
                        ok = got[p].counts[i] == full.counts[q] && got[c.fp].counts_star[i] == full.counts_star[q] &&
                             (full.counts[q] == 0 || (got[p].sums[i] == full.sums[q] && got[p].sums_hi[i] == full.sums_hi[q] &&
                                                      got[p].mins[i] == full.mins[q] && got[p].maxs[i] == full.maxs[q]));
                    }
                    ok = ok && seen == prim.keys.size();
                }
                if (!ok) std::printf("    FAIL multi payload case '%s'\n", c.name);
                EXPECT(ok);
            }
            // projection: a payload nobody reads costs nothing and returns nothing
            {
                gpudb::MultiPayload mp[2];
                mp[0].vals = cols[1].get(); mp[0].columns = (1u << 1) | (1u << 2);
                mp[1].vals = cols[2].get(); mp[1].columns = 0;
                gpudb::GroupByFilter f; f.columns = 0x1;
                auto got = agg->groupby_exact_masked_multi(*cols[0], mp, 2, 0, nullptr, 0, cap, f);
                EXPECT(got.size() == 2 && got[1].sums.empty() && got[1].counts.empty() && !got[0].sums.empty() &&
                       !got[0].keys.empty());
            }
            bool threw = false;
            try { gpudb::GroupByFilter f; f.topk = 3; (void)agg->groupby_exact_masked_multi(*cols[0], nullptr, 0, 0, nullptr, 0, cap, f); }
            catch (const std::runtime_error&) { threw = true; }
            EXPECT(threw);
        } catch (const std::runtime_error& e) {
            if (std::string(e.what()).find("not implemented") != std::string::npos) {
                implemented = false;
                std::printf("    SKIP (%s)\n", e.what());
            } else {
                ++failures; ++total;
                std::printf("    FAIL: %s\n", e.what());
            }
        }
        if (implemented) std::printf("    ok\n");
    }

    // ---- Few groups over many rows: groups that span thousands of chunks ----
    // (the second reduction level — blocks of 256 chunk partials — only
    // engages when a group covers >= 256 whole chunks; the other exact tests
    // have short groups). One giant group, a few large ones, many tiny ones
    // and a NULL-key group, sums overflowing 64 bits, NULL payloads, plain
    // and under both mask variants.
    {
        std::printf("  few groups over many rows (block-level reduce):\n");
        using Op = gpudb::Predicate::Op;
        bool implemented = true;
        try {
            std::mt19937_64 rng(0xB10CULL);
            const std::size_t N = 1'300'037, L = 3;
            const std::size_t cap = std::size_t(100) * 1000000;
            std::uniform_int_distribution<int> pct(0, 99);
            std::uniform_int_distribution<std::int64_t> wide(std::numeric_limits<std::int64_t>::min() / 2,
                                                             std::numeric_limits<std::int64_t>::max() / 2);
            std::uniform_int_distribution<std::int64_t> tiny(100, 5000), sel(0, 999);
            std::vector<std::int64_t> lanes(N * L);
            std::vector<std::vector<std::uint64_t>> valid(L, std::vector<std::uint64_t>((N + 63) / 64, ~std::uint64_t{0}));
            for (std::size_t i = 0; i < N; ++i) {
                const int p = pct(rng);
                // 60% key 1 (one giant group), 2 x 12% keys 2..3, 2% tiny keys, 14% NULL — a
                // NULL-key group of ~180K rows, long enough for the threaded host fold
                lanes[i * L + 0] = p < 60 ? 1 : p < 72 ? 2 : p < 84 ? 3 : tiny(rng);
                if (p >= 86) valid[0][i >> 6] &= ~(std::uint64_t{1} << (i & 63));
                lanes[i * L + 1] = wide(rng);
                if (pct(rng) < 7) valid[1][i >> 6] &= ~(std::uint64_t{1} << (i & 63));
                lanes[i * L + 2] = sel(rng);
            }
            std::vector<const std::uint64_t*> vp(L);
            for (std::size_t l = 0; l < L; ++l) vp[l] = valid[l].data();
            gpudb::Aggregator::RowSpan sp; sp.lanes = lanes.data(); sp.rows = N; sp.n_lanes = L; sp.valid = vp.data();
            const gpudb::Dtype dts[3] = {gpudb::Dtype::I64, gpudb::Dtype::I64, gpudb::Dtype::I64};
            auto cols = agg->upload_rows_exact(&sp, 1, dts, L);
            auto bit = [&](std::size_t l, std::size_t i) { return ((valid[l][i >> 6] >> (i & 63)) & 1u) != 0; };
            struct Acc { gpudb::Sum128 s; std::int64_t cnt = 0, cstar = 0, mn = 0, mx = 0; };
            for (int variant = 0; variant < 3; ++variant) {
                // 0: no mask; 1: 90% kept (masked reduce); 2: 15% kept (compacted)
                std::vector<gpudb::Predicate> preds;
                if (variant) { gpudb::Predicate p; p.col = cols[2].get(); p.op = variant == 1 ? Op::GE : Op::LT; p.value = variant == 1 ? 100 : 150; preds.push_back(p); }
                std::map<std::pair<int, std::int64_t>, Acc> want;
                for (std::size_t i = 0; i < N; ++i) {
                    const std::int64_t sv = lanes[i * L + 2];
                    if (variant == 1 && !(sv >= 100)) continue;
                    if (variant == 2 && !(sv < 150)) continue;
                    Acc& a = want[{bit(0, i) ? 0 : 1, bit(0, i) ? lanes[i * L] : 0}];
                    ++a.cstar;
                    if (bit(1, i)) {
                        const std::int64_t v = lanes[i * L + 1];
                        if (a.cnt == 0) { a.mn = v; a.mx = v; } else { a.mn = std::min(a.mn, v); a.mx = std::max(a.mx, v); }
                        a.s.add(v); ++a.cnt;
                    }
                }
                auto got = agg->groupby_exact_masked_resident(*cols[0], cols[1].get(), preds.data(), preds.size(), cap);
                bool ok = got.keys.size() == want.size();
                std::size_t q = 0;
                for (auto it = want.begin(); ok && it != want.end(); ++it, ++q) {
                    const Acc& a = it->second;
                    ok = got.key_null[q] == it->first.first && (it->first.first || got.keys[q] == it->first.second) &&
                         got.counts[q] == a.cnt && got.counts_star[q] == a.cstar &&
                         static_cast<std::uint64_t>(got.sums[q]) == a.s.lo && got.sums_hi[q] == a.s.hi &&
                         got.mins[q] == a.mn && got.maxs[q] == a.mx;
                }
                if (!ok) std::printf("    FAIL few-groups variant %d: got %zu groups, ref %zu\n", variant, got.keys.size(), want.size());
                EXPECT(ok);
            }
        } catch (const std::runtime_error& e) {
            if (std::string(e.what()).find("not implemented") != std::string::npos) {
                implemented = false;
                std::printf("    SKIP (%s)\n", e.what());
            } else {
                ++failures; ++total;
                std::printf("    FAIL: %s\n", e.what());
            }
        }
        if (implemented) std::printf("    ok\n");
    }
}

} // namespace

// =====================================================================
//  Hybrid planner tests — the per-call CPU/GPU dispatch rule
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

#if GPUDB_HAVE_CUDA && defined(__linux__)
extern "C" int gpudb_cuda_debug_fault_inject(void* stream);   // cudaError_t; 0 == success

// ---- an exact upload that cannot fit must leave the device as it found it ----
// The wrapper's memory policy is built on "a failed upload leaves no residue":
// it has to be able to refuse a set, free what it touched, and carry on. This
// forces the refusal deterministically by reserving all but a sliver of the
// card, then checks that free memory comes back — once, and again over
// repeats, because a bounded leak and an unbounded one look the same after a
// single attempt.
extern "C" std::size_t gpudb_cuda_debug_free_bytes();
extern "C" void*       gpudb_cuda_debug_reserve(std::size_t leave_bytes);
extern "C" void        gpudb_cuda_debug_release(void* p);

void test_cuda_failed_upload_leaves_nothing() {
    std::printf("  a refused exact upload frees what it touched:\n");
    // Same skip rule as test_backend(): CUDA can be compiled in on a machine
    // with no device (a build container, a CPU-only box), and there
    // make_aggregator throws. Without this the binary aborts on an uncaught
    // exception instead of finishing the checks that do not need a device.
    std::unique_ptr<gpudb::Aggregator> agg;
    try {
        agg = gpudb::make_aggregator(gpudb::Backend::CUDA);
    } catch (const std::exception& e) {
        std::printf("    SKIP (%s)\n", e.what());
        return;
    }

    // a set far larger than the sliver left free below
    const std::size_t N = 4'000'000;
    std::vector<std::int64_t> lanes(N * 2);
    for (std::size_t i = 0; i < N; ++i) { lanes[2 * i] = static_cast<std::int64_t>(i % 1000);
                                          lanes[2 * i + 1] = static_cast<std::int64_t>(i); }
    gpudb::Aggregator::RowSpan sp{};
    sp.lanes = lanes.data(); sp.rows = N; sp.n_lanes = 2; sp.valid = nullptr;
    const gpudb::Dtype dts[2] = { gpudb::Dtype::I64, gpudb::Dtype::I64 };

    void* hold = gpudb_cuda_debug_reserve(48ull << 20);       // leave ~48 MiB
    if (!hold) {                       // could not corner the device; say so rather than pass
        std::printf("    SKIP (could not reserve the device)\n");
        return;
    }
    const std::size_t free_before = gpudb_cuda_debug_free_bytes();
    int refused = 0;
    for (int attempt = 0; attempt < 4; ++attempt) {
        try {
            auto cols = agg->upload_rows_exact(&sp, 1, dts, 2);
            (void)cols;                                        // it fitted after all
        } catch (const std::exception&) {
            ++refused;
        }
    }
    const std::size_t free_after = gpudb_cuda_debug_free_bytes();
    gpudb_cuda_debug_release(hold);

    EXPECT(refused == 4);                                      // every attempt refused
    // Allow one CUDA allocation granule of slack; a retained set would be
    // megabytes, and a growing one would be four times that.
    const std::size_t slack = 4ull << 20;
    EXPECT(free_after + slack >= free_before);
    if (free_after + slack < free_before) {
        std::printf("    free memory fell from %zu MiB to %zu MiB over %d refusals\n",
                    free_before >> 20, free_after >> 20, refused);
    }
    std::printf("    ok (%d refusals, free %zu -> %zu MiB)\n",
                refused, free_before >> 20, free_after >> 20);
}

// A device fault (illegal address) is sticky: the context is dead for the
// rest of the process. The contract is that it reaches SQL as an error, not
// as an abort — Thrust temporaries used to throw from a destructor on the
// failing cudaFree and terminate the process. The scenario runs in a fresh
// child process (re-exec of this binary; a fork could not reuse the
// parent's CUDA context): it poisons the context, calls a resident op, and
// must see a std::runtime_error naming the fault and exit normally.
int cuda_fault_child() {
    try {
        auto agg = gpudb::make_aggregator(gpudb::Backend::CUDA);
        std::vector<std::int64_t> k = {1, 1, 2, 3};
        auto kc = agg->upload_i64(k.data(), k.size());
        (void)gpudb_cuda_debug_fault_inject(nullptr);
        try {
            (void)agg->groupby_count_resident(*kc, 100);
            return 2;                                                 // no error at all
        } catch (const std::runtime_error& e) {
            return std::string(e.what()).find("illegal memory access") != std::string::npos ? 0 : 1;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "fault child: unexpected: %s\n", e.what());
        return 4;
    }
}

void test_cuda_device_fault_is_an_error() {
    std::printf("--- CUDA device fault surfaces as std::runtime_error (child process) ---\n");
    // No device: the child cannot poison a context it never had, and its rc=4
    // would read as a failure rather than as "not reachable here".
    try {
        (void)gpudb::make_aggregator(gpudb::Backend::CUDA);
    } catch (const std::exception& e) {
        std::printf("  SKIP (%s)\n", e.what());
        return;
    }
    std::fflush(stdout);
    const pid_t pid = fork();
    if (pid == 0) {
        execl("/proc/self/exe", "test_gpudb", "--cuda-fault-child", static_cast<char*>(nullptr));
        std::_Exit(5);                                                // exec failed
    }
    int status = 0;
    (void)waitpid(pid, &status, 0);
    const bool clean_exit = WIFEXITED(status);
    const int  rc         = clean_exit ? WEXITSTATUS(status) : -1;
    EXPECT(clean_exit);           // not SIGABRT / SIGSEGV
    EXPECT_EQ(rc, 0);             // runtime_error naming the illegal access
    std::printf("  child: %s, rc=%d\n", clean_exit ? "exited" : "killed by signal", rc);
}

// ---- the direct grouped reduce answers what the CPU reference answers ------
// A key with few distinct values is grouped on CUDA by a row-order pass over
// replicated shared-memory accumulators instead of the sort path's mask,
// compaction and permuted gather. The choice is backend-private and must not
// be visible in any answer, so every case below runs the SAME operator on the
// CPU backend — the reference the whole exact path is defined against — and
// compares every output vector cell for cell.
//
// The admission rule lives in CudaAggregator::exact_common: take the direct
// path when the COLUMN has 1..gpudb_cuda_exact_direct_max_groups() distinct
// keys (256, a shared-memory budget in exact_kernel.cu), otherwise sort. The
// cases below sit either side of that bound on purpose.
void test_cuda_direct_reduce_matches_reference() {
    std::printf("--- the CUDA direct grouped reduce vs the CPU reference ---\n");
    std::unique_ptr<gpudb::Aggregator> gpu;
    try {
        gpu = gpudb::make_aggregator(gpudb::Backend::CUDA);
    } catch (const std::exception& e) {
        std::printf("  SKIP (%s)\n", e.what());
        return;
    }
    auto cpu = gpudb::make_aggregator(gpudb::Backend::CPU);

    auto same = [](const gpudb::GroupByResidentResult& a,
                   const gpudb::GroupByResidentResult& b) {
        return a.keys == b.keys && a.key_null == b.key_null &&
               a.sums == b.sums && a.sums_hi == b.sums_hi &&
               a.counts == b.counts && a.counts_star == b.counts_star &&
               a.mins == b.mins && a.maxs == b.maxs;
    };

    // One shape, built the same way for both backends. `groups` distinct keys,
    // NULL keys and NULL payloads sprinkled, one group given only NULL
    // payloads, and INT64_MIN/MAX plus values large enough that the 128-bit
    // sum carries out of its low limb.
    struct Case { const char* name; std::size_t groups; std::size_t rows; bool huge; };
    const Case cases[] = {
        {"1 distinct key",    1,   50'000,  true},
        {"2 distinct keys",   2,   50'000,  true},
        {"255 distinct keys", 255, 120'011, false},
        {"256 distinct keys", 256, 120'011, false},   // the last the direct path serves
        {"257 distinct keys", 257, 120'011, false},   // one past it: the sort path, same answer
    };

    for (const auto& c : cases) {
        const std::size_t N = c.rows;
        std::mt19937_64 rng(0xD12EC7ULL + c.groups);
        std::vector<std::int64_t>  kv(2 * N);
        std::vector<std::uint64_t> kvalid((N + 63) / 64, ~std::uint64_t{0});
        std::vector<std::uint64_t> vvalid((N + 63) / 64, ~std::uint64_t{0});
        auto clr = [](std::vector<std::uint64_t>& m, std::size_t i) {
            m[i >> 6] &= ~(std::uint64_t{1} << (i & 63));
        };
        std::uniform_int_distribution<int> pct(0, 99);
        for (std::size_t i = 0; i < N; ++i) {
            const std::int64_t k = static_cast<std::int64_t>(i % c.groups);
            std::int64_t v;
            if (c.huge && pct(rng) < 30) {
                // wide values: the low limb of the 128-bit sum must carry
                v = (pct(rng) & 1) ? std::numeric_limits<std::int64_t>::max()
                                   : std::numeric_limits<std::int64_t>::min();
            } else {
                v = static_cast<std::int64_t>(rng() % 200'001) - 100'000;
            }
            kv[2 * i] = k; kv[2 * i + 1] = v;
            if (pct(rng) < 5) clr(kvalid, i);                 // NULL key -> the null group
            if (k == 0 || pct(rng) < 9) clr(vvalid, i);       // group 0: every payload NULL
        }
        gpudb::Aggregator::KvSpan sp{};
        sp.kv = kv.data(); sp.rows = N;
        sp.key_valid = kvalid.data(); sp.val_valid = vvalid.data();

        auto g = gpu->upload_pair_exact(&sp, 1, gpudb::Dtype::I64);
        auto h = cpu->upload_pair_exact(&sp, 1, gpudb::Dtype::I64);
        const std::size_t cap = std::size_t(100) * 1000000;

        // with a payload: sum / count(v) / count(*) / min / max
        const bool ok_pay = same(gpu->groupby_exact_resident(*g.keys, g.vals.get(), cap),
                                 cpu->groupby_exact_resident(*h.keys, h.vals.get(), cap));
        // keys only: count(*) alone, min/max at their sentinels
        const bool ok_keys = same(gpu->groupby_exact_resident(*g.keys, nullptr, cap),
                                  cpu->groupby_exact_resident(*h.keys, nullptr, cap));
        EXPECT(ok_pay);
        EXPECT(ok_keys);

        // Masks. `keeps nothing` is the one that proves empty groups are
        // dropped rather than emitted with count(*) = 0: the direct path
        // grinds every distinct key of the column, so it produces them and
        // then must throw them away to agree with the reference.
        auto pred = [&](gpudb::Predicate::Op op, std::int64_t value,
                        const gpudb::ResidentColumn& col) {
            gpudb::Predicate p{}; p.col = &col; p.op = op; p.value = value; return p;
        };
        struct M { const char* what; gpudb::Predicate::Op op; std::int64_t value; };
        const M masks[] = {
            {"keeps nothing",     gpudb::Predicate::Op::LT, std::numeric_limits<std::int64_t>::min()},
            {"keeps everything",  gpudb::Predicate::Op::GE, std::numeric_limits<std::int64_t>::min()},
            {"keeps some",        gpudb::Predicate::Op::LT, 0},
        };
        bool ok_masked = true;
        for (const auto& m : masks) {
            const gpudb::Predicate pg = pred(m.op, m.value, *g.vals);
            const gpudb::Predicate ph = pred(m.op, m.value, *h.vals);
            auto rg = gpu->groupby_exact_masked_resident(*g.keys, g.vals.get(), &pg, 1, cap);
            auto rh = cpu->groupby_exact_masked_resident(*h.keys, h.vals.get(), &ph, 1, cap);
            if (!same(rg, rh)) { ok_masked = false; break; }
            // no group may come back empty, whichever path produced it
            for (std::size_t j = 0; j < rg.counts_star.size(); ++j)
                if (rg.counts_star[j] == 0) { ok_masked = false; break; }
        }
        EXPECT(ok_masked);

        // Several payloads in one call (§4.9). The default multi runs one
        // single-payload pass per column, so this is the direct path used
        // repeatedly over one set of keys.
        gpudb::MultiPayload mpg[2]; mpg[0].vals = g.vals.get(); mpg[1].vals = g.vals.get();
        gpudb::MultiPayload mph[2]; mph[0].vals = h.vals.get(); mph[1].vals = h.vals.get();
        auto vg = gpu->groupby_exact_masked_multi(*g.keys, mpg, 2, 0, nullptr, 0, cap);
        auto vh = cpu->groupby_exact_masked_multi(*h.keys, mph, 2, 0, nullptr, 0, cap);
        bool ok_multi = vg.size() == vh.size();
        for (std::size_t p = 0; ok_multi && p < vg.size(); ++p) ok_multi = same(vg[p], vh[p]);
        EXPECT(ok_multi);

        std::printf("    %-18s rows=%zu groups=%zu  pay=%d keys=%d masked=%d multi=%d\n",
                    c.name, N, c.groups, int(ok_pay), int(ok_keys), int(ok_masked), int(ok_multi));
    }

    // Narrow lanes (#165) and the narrow sort cache (#166): the distinct-key
    // list the direct path reads comes out of that cache, so a column stored
    // at 1, 2 and 4 bytes has to give the same answer as one stored at 8.
    {
        const std::size_t N = 60'013;
        bool ok_narrow = true;
        for (const std::int64_t span : {std::int64_t(100),        // fits a byte
                                        std::int64_t(30'000),     // fits two
                                        std::int64_t(2'000'000),  // fits four
                                        std::int64_t(1) << 40}) { // needs eight
            std::mt19937_64 rng(0xBADC0DEULL ^ static_cast<std::uint64_t>(span));
            std::vector<std::int64_t> kv(2 * N);
            for (std::size_t i = 0; i < N; ++i) {
                kv[2 * i]     = static_cast<std::int64_t>(i % 64);      // few groups
                kv[2 * i + 1] = static_cast<std::int64_t>(rng() % static_cast<std::uint64_t>(span))
                              - span / 2;
            }
            gpudb::Aggregator::KvSpan sp{}; sp.kv = kv.data(); sp.rows = N;
            auto g = gpu->upload_pair_exact(&sp, 1, gpudb::Dtype::I64);
            auto h = cpu->upload_pair_exact(&sp, 1, gpudb::Dtype::I64);
            const std::size_t cap = std::size_t(100) * 1000000;
            if (!same(gpu->groupby_exact_resident(*g.keys, g.vals.get(), cap),
                      cpu->groupby_exact_resident(*h.keys, h.vals.get(), cap))) {
                ok_narrow = false; break;
            }
        }
        EXPECT(ok_narrow);
        std::printf("    narrow lanes (1/2/4/8-byte payloads) agree: %d\n", int(ok_narrow));
    }

    // Device memory: the direct path adds ONE device buffer, the distinct-key
    // list (at most 256 * 8 bytes), held in a DeviceOut that frees on every
    // exit including a throw. Its accumulators are shared memory, per block,
    // never an allocation. Repeats of the masked form — the path with the most
    // buffers in flight — must leave free memory where they found it: a
    // bounded leak and an unbounded one look the same after one call, which is
    // what #172 was on Metal.
    {
        const std::size_t N = 200'003;
        std::vector<std::int64_t> kv(2 * N);
        for (std::size_t i = 0; i < N; ++i) {
            kv[2 * i]     = static_cast<std::int64_t>(i % 17);
            kv[2 * i + 1] = static_cast<std::int64_t>(i % 1013) - 500;
        }
        gpudb::Aggregator::KvSpan sp{}; sp.kv = kv.data(); sp.rows = N;
        auto g = gpu->upload_pair_exact(&sp, 1, gpudb::Dtype::I64);
        const std::size_t cap = std::size_t(100) * 1000000;
        gpudb::Predicate p{}; p.col = g.vals.get(); p.op = gpudb::Predicate::Op::LT; p.value = 0;
        (void)gpu->groupby_exact_masked_resident(*g.keys, g.vals.get(), &p, 1, cap);  // warm
        const std::size_t before = gpudb_cuda_debug_free_bytes();
        for (int i = 0; i < 64; ++i)
            (void)gpu->groupby_exact_masked_resident(*g.keys, g.vals.get(), &p, 1, cap);
        const std::size_t after = gpudb_cuda_debug_free_bytes();
        const bool ok_free = after >= before;
        EXPECT(ok_free);
        std::printf("    64 masked direct calls leave free memory at %zu -> %zu MiB: %d\n",
                    before >> 20, after >> 20, int(ok_free));
    }
}
#endif


// ---------------------------------------------------------------------------
// The direct, row-order grouped reduce: a GROUP BY key with few
// distinct values gets a dense group-id lane and the exact operators run one
// row-order pass over it instead of the sort path's mask + run starts +
// gather. The choice is backend-private and changes no answer, which is what
// this block pins: every exact form, on keys either side of the id-lane's
// width and group limits, run with GPUDB_METAL_GROUPBY_EXACT_PATH=direct and
// =sort and compared limb for limb against the CPU reference.
// ---------------------------------------------------------------------------
#if GPUDB_HAVE_METAL
void test_direct_groupby_body();

// Nothing in this block may take the binary down: a GPU that cannot build one
// of the pipelines the exact path needs — including the radix sorter the SORT
// path uses — is a skip, not a crash, and the rest of the suite still has to
// run and report.
void test_direct_groupby() {
    std::printf("\n--- exact GROUP BY: the direct path vs the sort path ---\n");
    try {
        test_direct_groupby_body();
    } catch (const std::exception& e) {
        std::printf("  skipped (%s)\n", e.what());
    }
}

void test_direct_groupby_body() {
    using DT = gpudb::Dtype;
    using Op = gpudb::Predicate::Op;
    const std::size_t cap = std::size_t(100) * 1000000;

    auto ref_agg = gpudb::make_aggregator(gpudb::Backend::CPU);
    auto make_metal = [](const char* path) {
        setenv("GPUDB_METAL_GROUPBY_EXACT_PATH", path, 1);
        auto a = gpudb::make_aggregator(gpudb::Backend::METAL);
        unsetenv("GPUDB_METAL_GROUPBY_EXACT_PATH");
        return a;
    };
    auto direct_agg = make_metal("direct");
    auto sort_agg   = make_metal("sort");

    // Is the direct path available on this GPU at all? A forced-direct probe
    // answers: the backend leaves the algorithm it ran in the path note, and
    // the reason it could not run the other one beside it. A hosted macOS
    // runner's virtualised Apple GPU compiles the kernel library and then
    // refuses to lower these functions, and that must cost the suite nothing
    // but the path assertions.
    std::string unavailable;
    {
        const std::size_t n = 4096, L = 2;
        std::vector<std::int64_t> flat(n * L);
        for (std::size_t i = 0; i < n; ++i) {
            flat[i * L] = static_cast<std::int64_t>(i % 5);
            flat[i * L + 1] = static_cast<std::int64_t>(i);
        }
        const DT dts[2] = {DT::I64, DT::I64};
        gpudb::Aggregator::RowSpan sp;
        sp.lanes = flat.data(); sp.rows = n; sp.n_lanes = L; sp.valid = nullptr;
        try {
            auto c = direct_agg->upload_rows_exact(&sp, 1, dts, L);
            gpudb::GroupByFilter f;
            f.columns = 0x0Fu;
            gpudb::exact_path_note().clear();
            gpudb::exact_path_reason().clear();
            (void)direct_agg->groupby_exact_resident(*c[0], c[1].get(), cap, f);
            if (gpudb::exact_path_note() != "direct")
                unavailable = gpudb::exact_path_reason().empty() ? "no reason given"
                                                                : gpudb::exact_path_reason();
        } catch (const std::exception& e) {
            unavailable = std::string("probe threw: ") + e.what();
        }
    }
    if (!unavailable.empty())
        std::printf("  skipped path assertions (direct path unavailable: %s)\n", unavailable.c_str());
    const bool have_direct = unavailable.empty();

    // Limb-for-limb equality of two results, in order: the group order, the
    // NULL-key group's place and a group the WHERE emptied being absent are
    // all part of the contract.
    // `totals`: also compare groups_total. The sort path counts groups the
    // WHERE emptied in it where the CPU reference and the direct path count
    // only the ones that came back, which predates this work and is a
    // diagnostic count, not an answer — so the sort comparisons leave it out.
    // `cols`: the GroupByFilter column bits the caller asked for. A backend
    // may leave an unwanted vector empty (the reference fills them all), so
    // only what was asked for is compared.
    auto same = [&](const gpudb::GroupByResidentResult& a, const gpudb::GroupByResidentResult& b,
                    const char* what, bool totals = true, std::uint32_t cols = 0x3Fu) {
        auto veq = [&](const auto& x, const auto& y, const char* col) {
            if (x.size() != y.size()) {
                std::printf("    FAIL %s: %s size %zu vs %zu\n", what, col, x.size(), y.size());
                return false;
            }
            for (std::size_t i = 0; i < x.size(); ++i)
                if (x[i] != y[i]) {
                    std::printf("    FAIL %s: %s[%zu] %lld vs %lld\n", what, col, i,
                                static_cast<long long>(x[i]), static_cast<long long>(y[i]));
                    return false;
                }
            return true;
        };
        auto want = [&](unsigned bit) { return (cols >> bit) & 1u; };
        bool ok = true;
        if (want(0)) ok &= veq(a.keys, b.keys, "keys") & veq(a.key_null, b.key_null, "key_null");
        if (want(1)) ok &= veq(a.sums, b.sums, "sums") & veq(a.sums_hi, b.sums_hi, "sums_hi");
        if (want(2)) ok &= veq(a.counts, b.counts, "counts");
        if (want(3)) ok &= veq(a.counts_star, b.counts_star, "counts_star");
        if (want(4)) ok &= veq(a.mins, b.mins, "mins");
        if (want(5)) ok &= veq(a.maxs, b.maxs, "maxs");
        // groups_total is the primary result's; an extra payload carries
        // only its aggregate vectors.
        if (ok && totals && !b.keys.empty() && a.groups_total != b.groups_total) {
            std::printf("    FAIL %s: groups_total %zu vs %zu\n", what, a.groups_total, b.groups_total);
            ok = false;
        }
        return ok;
    };

    // One scenario: N rows, D distinct keys, NULLs in the key / payload /
    // predicate lanes, a second payload, an i64 and an f64 predicate lane.
    // Lane 1 holds values near 2^62 so the sums need both limbs.
    struct Lanes {
        std::vector<std::int64_t> flat;
        std::vector<std::vector<std::uint64_t>> valid;
        std::vector<const std::uint64_t*> vp;
        std::size_t rows = 0, n_lanes = 0;
    };
    auto build = [](std::size_t N, std::size_t D, bool null_keys, bool null_pay, bool null_pred,
                    std::uint64_t seed) {
        const std::size_t L = 5;   // key, payload, payload2, i64 pred, f64 pred
        Lanes x; x.rows = N; x.n_lanes = L;
        x.flat.resize(N * L);
        x.valid.assign(L, std::vector<std::uint64_t>((N + 63) / 64, ~std::uint64_t{0}));
        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<int> pct(0, 99);
        for (std::size_t i = 0; i < N; ++i) {
            const std::size_t g = static_cast<std::size_t>((i * 2654435761ull + (i >> 7)) % D);
            x.flat[i * L + 0] = static_cast<std::int64_t>(g) * 3 - 7;      // keys, not dense
            x.flat[i * L + 1] = (std::int64_t{1} << 62) - static_cast<std::int64_t>(i % 1013) * 7;
            x.flat[i * L + 2] = static_cast<std::int64_t>(i % 251) - 125;  // a narrow lane
            x.flat[i * L + 3] = static_cast<std::int64_t>(i % 1000);
            const double d = static_cast<double>(static_cast<std::int64_t>(i % 997)) / 997.0;
            std::memcpy(&x.flat[i * L + 4], &d, sizeof(double));
            auto clear = [&](std::size_t l) { x.valid[l][i >> 6] &= ~(std::uint64_t{1} << (i & 63)); };
            if (null_keys && pct(rng) < 4)  clear(0);
            if (null_pay  && pct(rng) < 11) { clear(1); clear(2); }
            if (null_pred && pct(rng) < 6)  { clear(3); clear(4); }
        }
        x.vp.resize(L);
        for (std::size_t l = 0; l < L; ++l) x.vp[l] = x.valid[l].data();
        return x;
    };

    struct Case { const char* name; std::size_t rows, distinct; bool nk, np, npr; };
    const Case cases[] = {
        {"1 group",        2'600'037, 1,      true,  true,  true},
        {"2 groups",       2'600'037, 2,      false, true,  true},
        {"7 groups",       2'600'037, 7,      true,  true,  true},
        {"255 groups",     2'600'037, 255,    true,  false, true},   // ids still fit one byte
        {"256 groups",     2'600'037, 256,    true,  true,  true},   // ... with the NULL group: two
        {"257 groups",     2'600'037, 257,    false, true,  true},
        {"512 groups",     2'600'037, 512,    false, false, true},   // the largest bucket
        {"70000 groups",   2'600'037, 70'000, true,  true,  true},   // no id lane: the sort path
    };
    for (const Case& c : cases) {
        std::printf("  %s:\n", c.name);
        try {
            Lanes x = build(c.rows, c.distinct, c.nk, c.np, c.npr, 0xD1EC7ULL + c.distinct);
            const DT dts[5] = {DT::I64, DT::I64, DT::I64, DT::I64, DT::F64};
            gpudb::Aggregator::RowSpan sp;
            sp.lanes = x.flat.data(); sp.rows = x.rows; sp.n_lanes = x.n_lanes; sp.valid = x.vp.data();
            auto dcols = direct_agg->upload_rows_exact(&sp, 1, dts, x.n_lanes);
            auto scols = sort_agg->upload_rows_exact(&sp, 1, dts, x.n_lanes);
            auto rcols = ref_agg->upload_rows_exact(&sp, 1, dts, x.n_lanes);

            double half = 0.5;
            std::int64_t half_bits = 0;
            std::memcpy(&half_bits, &half, sizeof(half_bits));

            // preds: none / one i64 / three terms incl. an f64 lane and a key
            // term / one that keeps nothing at all
            struct PredSet { const char* name; int n; Op ops[3]; int lane[3]; std::int64_t val[3]; };
            const PredSet sets[] = {
                {"plain",     0, {}, {}, {}},
                {"1 term",    1, {Op::LT, Op::EQ, Op::EQ}, {3, 0, 0}, {700, 0, 0}},
                {"3 terms",   3, {Op::GE, Op::LT, Op::GT}, {3, 4, 0},
                              {100, 0 /*filled below*/, -1000000}},
                {"empty",     1, {Op::GT, Op::EQ, Op::EQ}, {3, 0, 0}, {100000, 0, 0}},
                {"key range", 1, {Op::GE, Op::EQ, Op::EQ}, {0, 0, 0}, {-1, 0, 0}},
                {"is null",   1, {Op::IsNull, Op::EQ, Op::EQ}, {3, 0, 0}, {0, 0, 0}},
            };
            for (const PredSet& ps : sets) {
                gpudb::Predicate dp[3], sp2[3], rp[3];
                for (int q = 0; q < ps.n; ++q) {
                    dp[q].op = sp2[q].op = rp[q].op = ps.ops[q];
                    const std::int64_t v = (ps.lane[q] == 4) ? half_bits : ps.val[q];
                    dp[q].value = sp2[q].value = rp[q].value = v;
                    dp[q].col  = dcols[ps.lane[q]].get();
                    sp2[q].col = scols[ps.lane[q]].get();
                    rp[q].col  = rcols[ps.lane[q]].get();
                }
                // Payload 1 holds values near 2^62 (the sums need both
                // limbs) and is stored at 8 bytes; payload 2 is a narrow
                // lane. The direct path keeps min / max as 32-bit atomics,
                // exact for a lane of 4 bytes or fewer, so a wide lane with
                // min / max asked for takes the sort path — run both.
                for (int pv = 1; pv <= 2; ++pv) {
                    const bool wide = (pv == 1);
                    auto run = [&](const std::unique_ptr<gpudb::Aggregator>& a,
                                   const std::vector<std::unique_ptr<gpudb::ResidentColumn>>& col,
                                   gpudb::Predicate* p, const gpudb::GroupByFilter& f) {
                        gpudb::exact_path_note().clear();
                        return a->groupby_exact_masked_resident(*col[0], col[pv].get(), ps.n ? p : nullptr,
                                                                ps.n, cap, f);
                    };
                    // plain, HAVING on count(*), top-k both ways
                    gpudb::GroupByFilter forms[4];
                    forms[1].agg = gpudb::GroupByFilter::Agg::CountStar;
                    forms[1].cmp = gpudb::GroupByFilter::Cmp::GT;
                    forms[1].threshold_i64 = static_cast<std::int64_t>(c.rows / (c.distinct * 2 + 1));
                    forms[2].agg = gpudb::GroupByFilter::Agg::Sum; forms[2].topk = 5; forms[2].topk_desc = true;
                    forms[3].agg = gpudb::GroupByFilter::Agg::Min; forms[3].topk = 3; forms[3].topk_desc = false;
                    const bool needs_mm[4] = {true, true, true, true};   // kAllColumns by default
                    const char* fname[4] = {"plain", "having", "topk desc", "topk asc"};
                    for (int fi = 0; fi < 4; ++fi) {
                        auto got = run(direct_agg, dcols, dp, forms[fi]);
                        const std::string path = gpudb::exact_path_note();
                        auto srt = run(sort_agg, scols, sp2, forms[fi]);
                        auto want = run(ref_agg, rcols, rp, forms[fi]);
                        char what[160];
                        std::snprintf(what, sizeof(what), "%s / %s / %s / pay%d [%s]", c.name, ps.name,
                                      fname[fi], pv, path.c_str());
                        // A wide payload with min / max asked for still takes the
                        // direct path while the thread-private accumulators hold
                        // it (32 slots); above that the slab would have to, and
                        // its min / max are 32-bit.
                        const std::size_t n_groups = c.distinct + (c.nk ? 1 : 0);
                        const bool direct = have_direct && c.distinct <= 512 &&
                                            (!(wide && needs_mm[fi]) || n_groups <= 32);
                        EXPECT(same(got, want, what, /*totals*/direct));
                        EXPECT(same(srt, want, what, /*totals*/false));
                        if (path != (direct ? "direct" : "sort"))
                            std::printf("    FAIL %s: path %s, expected %s\n", what, path.c_str(),
                                        direct ? "direct" : "sort");
                        EXPECT_EQ(path == (direct ? "direct" : "sort"), true);
                    }
                    // the same forms with min / max NOT read: a wide payload
                    // then takes the direct path too
                    if (wide) {
                        gpudb::GroupByFilter f;
                        f.columns = 0x0Fu;    // keys, sums, counts, count(*)
                        auto got = run(direct_agg, dcols, dp, f);
                        const std::string path = gpudb::exact_path_note();
                        auto want = run(ref_agg, rcols, rp, f);
                        char what[160];
                        std::snprintf(what, sizeof(what), "%s / %s / no min-max [%s]", c.name, ps.name,
                                      path.c_str());
                        EXPECT(got.keys == want.keys && got.key_null == want.key_null &&
                               got.sums == want.sums && got.sums_hi == want.sums_hi &&
                               got.counts == want.counts && got.counts_star == want.counts_star);
                        if (!(got.sums == want.sums)) std::printf("    FAIL %s\n", what);
                        EXPECT_EQ(path == (have_direct && c.distinct <= 512 ? "direct" : "sort"), true);
                    }
                }
            }

            // several payloads in one call (§4.9), plain and under a filter
            for (int fi = 0; fi < 2; ++fi) {
                gpudb::GroupByFilter f;
                if (fi) { f.agg = gpudb::GroupByFilter::Agg::CountStar;
                          f.cmp = gpudb::GroupByFilter::Cmp::GE;
                          f.threshold_i64 = static_cast<std::int64_t>(c.rows / (c.distinct * 2 + 1)); }
                gpudb::Predicate dp, sp2, rp;
                dp.op = sp2.op = rp.op = Op::LT;
                dp.value = sp2.value = rp.value = 800;
                dp.col = dcols[3].get(); sp2.col = scols[3].get(); rp.col = rcols[3].get();
                auto multi = [&](const std::unique_ptr<gpudb::Aggregator>& a,
                                 const std::vector<std::unique_ptr<gpudb::ResidentColumn>>& col,
                                 gpudb::Predicate* p) {
                    gpudb::MultiPayload mp[2];
                    mp[0].vals = col[1].get(); mp[0].columns = 0x0Fu;   // wide lane: no min / max
                    mp[1].vals = col[2].get(); mp[1].columns = gpudb::GroupByFilter::kAllColumns;
                    gpudb::exact_path_note().clear();
                    return a->groupby_exact_masked_multi(*col[0], mp, 2, 0, p, 1, cap, f);
                };
                auto got = multi(direct_agg, dcols, &dp);
                const std::string path = gpudb::exact_path_note();
                auto want = multi(ref_agg, rcols, &rp);
                auto srt = multi(sort_agg, scols, &sp2);
                char what[160];
                std::snprintf(what, sizeof(what), "%s / multi payload%s [%s]", c.name,
                              fi ? " having" : "", path.c_str());
                EXPECT_EQ(got.size(), want.size());
                for (std::size_t p = 0; p < got.size() && p < want.size(); ++p) {
                    const std::uint32_t cols = p == 0 ? 0x0Fu : gpudb::GroupByFilter::kAllColumns;
                    EXPECT(same(got[p], want[p], what, /*totals*/path == "direct", cols));
                    EXPECT(same(srt[p], want[p], what, /*totals*/false, cols));
                }
            }

            // count(*) with no payload at all. Compared against the Metal
            // sort path, not the CPU reference: with no payload the CPU
            // reference leaves min / max at the reduce identity where Metal
            // writes 0 (neither is ever read — there is no payload to take a
            // min of), and that difference predates this path.
            {
                gpudb::GroupByFilter f;
                gpudb::exact_path_note().clear();
                auto got = direct_agg->groupby_exact_resident(*dcols[0], nullptr, cap, f);
                auto srt = sort_agg->groupby_exact_resident(*scols[0], nullptr, cap, f);
                auto want = ref_agg->groupby_exact_resident(*rcols[0], nullptr, cap, f);
                char what[128];
                std::snprintf(what, sizeof(what), "%s / count(*) only", c.name);
                EXPECT(same(got, srt, what));
                EXPECT(got.keys == want.keys && got.key_null == want.key_null &&
                       got.counts_star == want.counts_star && got.counts == want.counts);
            }

            // What the id lane costs: prepare() builds it (the wrapper's
            // upload is where the first query's fixed costs belong), so two
            // freshly prepared columns of the same rows — one on each path —
            // differ by exactly the ids at 1 or 2 bytes a row plus the
            // distinct keys, and by nothing above the group limit.
            {
                auto fd = make_metal("direct");
                auto fs = make_metal("sort");
                auto dc = fd->upload_rows_exact(&sp, 1, dts, x.n_lanes);
                auto sc = fs->upload_rows_exact(&sp, 1, dts, x.n_lanes);
                dc[0]->prepare();
                sc[0]->prepare();
                const std::size_t extra = dc[0]->resident_bytes() - sc[0]->resident_bytes();
                if (have_direct && c.distinct <= 512) {
                    const std::size_t null_id = c.nk ? c.distinct : c.distinct - 1;
                    const std::size_t gw = null_id <= 255 ? 1 : 2;
                    const std::size_t want_ids = x.rows * gw;
                    const bool ok = extra >= want_ids && extra <= want_ids + c.distinct * 8;
                    if (!ok) std::printf("    FAIL %s: id lane bytes %zu (expected %zu..%zu)\n",
                                         c.name, extra, want_ids, want_ids + c.distinct * 8);
                    EXPECT(ok);
                } else {
                    EXPECT_EQ(extra, std::size_t(0));
                }
            }
        } catch (const std::exception& e) {
            ++failures; ++total;
            std::printf("    FAIL: %s\n", e.what());
        }
    }

    // An OPTIONAL pipeline's refusal must cost one shape, not the path. A
    // min / max over a payload lane wider than 4 bytes cannot use the slab's
    // 32-bit atomics, so it takes the thread-private kernel; at 9 to 32
    // accumulator slots that is gdir_masked_32_i64. If only that one will not
    // build, this call takes the sort path and every other shape still goes
    // direct. GPUDB_METAL_DIRECT_DISABLE_PSO=masked32 is that device.
    std::printf("  an optional pipeline refused (wide min/max, 9..32 slots):\n");
    try {
        const char* knob = std::getenv("GPUDB_METAL_DIRECT_DISABLE_PSO");
        const std::string mode = knob ? knob : "";
        const std::size_t N = 2'000'003, L = 3;
        std::vector<std::int64_t> flat(N * L);
        std::vector<std::vector<std::uint64_t>> valid(L, std::vector<std::uint64_t>((N + 63) / 64, ~std::uint64_t{0}));
        for (std::size_t i = 0; i < N; ++i) {
            flat[i * L + 0] = static_cast<std::int64_t>(i % 20) * 7 - 3;     // 20 distinct + NULLs = 21 groups
            flat[i * L + 1] = (std::int64_t{1} << 62) - static_cast<std::int64_t>(i % 991);  // 8-byte lane
            flat[i * L + 2] = static_cast<std::int64_t>(i % 127) - 63;       // narrow lane
            if (i % 37 == 0) valid[0][i >> 6] &= ~(std::uint64_t{1} << (i & 63));
        }
        std::vector<const std::uint64_t*> vp{valid[0].data(), valid[1].data(), valid[2].data()};
        const DT dts[3] = {DT::I64, DT::I64, DT::I64};
        gpudb::Aggregator::RowSpan sp;
        sp.lanes = flat.data(); sp.rows = N; sp.n_lanes = L; sp.valid = vp.data();
        auto dc = direct_agg->upload_rows_exact(&sp, 1, dts, L);
        auto rc = ref_agg->upload_rows_exact(&sp, 1, dts, L);
        gpudb::GroupByFilter f;            // kAllColumns: min / max are read
        auto call = [&](std::size_t pay) {
            gpudb::exact_path_note().clear();
            auto got = direct_agg->groupby_exact_resident(*dc[0], dc[pay].get(), cap, f);
            return std::make_pair(got, gpudb::exact_path_note());
        };
        auto wide = call(1), narrow = call(2);
        auto want_wide = ref_agg->groupby_exact_resident(*rc[0], rc[1].get(), cap, f);
        auto want_narrow = ref_agg->groupby_exact_resident(*rc[0], rc[2].get(), cap, f);
        EXPECT(same(wide.first, want_wide, "wide min/max", /*totals*/wide.second == "direct"));
        EXPECT(same(narrow.first, want_narrow, "narrow min/max", /*totals*/narrow.second == "direct"));
        if (have_direct) {
            // the narrow payload never needs the thread kernel, so it is the
            // witness that the path itself is still alive
            if (narrow.second != "direct")
                std::printf("    FAIL narrow min/max took %s, expected direct\n", narrow.second.c_str());
            EXPECT_EQ(narrow.second == "direct", true);
            const char* want_path = (mode == "masked32") ? "sort" : "direct";
            if (wide.second != want_path)
                std::printf("    FAIL wide min/max took %s, expected %s\n",
                            wide.second.c_str(), want_path);
            EXPECT_EQ(wide.second == want_path, true);
        }
    } catch (const std::exception& e) {
        ++failures; ++total;
        std::printf("    FAIL: %s\n", e.what());
    }

    // The capability gate: a device we will not offer the kernel to must not be
    // asked for a single direct pipeline, because on the device that produced
    // this rule one refused build breaks every later build in the process.
    // "Never asked" is not visible in a result, so it is counted.
    if (const char* knob = std::getenv("GPUDB_METAL_DIRECT_DISABLE_PSO");
        knob && std::string(knob) == "unsupported") {
        std::printf("  the capability gate (nothing offered to the device):\n");
        const long before = gpudb::metal_direct_pipeline_requests().load();
        try {
            const std::size_t N = 1'500'001, L = 2;
            std::vector<std::int64_t> flat(N * L);
            for (std::size_t i = 0; i < N; ++i) {
                flat[i * L + 0] = static_cast<std::int64_t>(i % 11);
                flat[i * L + 1] = static_cast<std::int64_t>(i % 733);
            }
            const DT dts[2] = {DT::I64, DT::I64};
            gpudb::Aggregator::RowSpan sp;
            sp.lanes = flat.data(); sp.rows = N; sp.n_lanes = L; sp.valid = nullptr;
            auto dc = direct_agg->upload_rows_exact(&sp, 1, dts, L);
            auto sc = sort_agg->upload_rows_exact(&sp, 1, dts, L);
            auto rc = ref_agg->upload_rows_exact(&sp, 1, dts, L);
            dc[0]->prepare();                      // the prewarm must not ask either
            sc[0]->prepare();
            gpudb::GroupByFilter f;
            f.columns = 0x0Fu;
            gpudb::exact_path_note().clear();
            auto got = direct_agg->groupby_exact_resident(*dc[0], dc[1].get(), cap, f);
            auto want = ref_agg->groupby_exact_resident(*rc[0], rc[1].get(), cap, f);
            const long asked = gpudb::metal_direct_pipeline_requests().load() - before;
            if (asked != 0) std::printf("    FAIL %ld direct pipelines were asked of the device\n", asked);
            EXPECT_EQ(asked, 0L);
            EXPECT_EQ(gpudb::exact_path_note() == "sort", true);
            EXPECT_EQ(dc[0]->resident_bytes() - sc[0]->resident_bytes(), std::size_t(0));
            EXPECT(got.keys == want.keys && got.sums == want.sums &&
                   got.counts_star == want.counts_star);
            std::printf("    %ld pipelines asked, no id lane, answer through sort\n", asked);
        } catch (const std::exception& e) {
            ++failures; ++total;
            std::printf("    FAIL: %s\n", e.what());
        }
    }

    // The admission rule, on both sides of it. `auto` must send a call the
    // sort path's way when there is not enough work per row to pay the
    // group-id lane back, and take the direct path once there is — with the
    // same answer either way. Forced `direct` ignores the rule, which is what
    // the parity block above relies on.
    std::printf("  the work floor (auto):\n");
    try {
        auto auto_agg = [] {
            unsetenv("GPUDB_METAL_GROUPBY_EXACT_PATH");
            return gpudb::make_aggregator(gpudb::Backend::METAL);
        }();
        // one payload, no WHERE: work = rows. 1M rows is below the 6M floor,
        // 7M is above it.
        struct Side { const char* name; std::size_t rows; const char* want; };
        const Side sides[] = {{"below the floor", 1'000'000, "sort"},
                              {"above the floor", 7'000'000, "direct"}};
        for (const Side& sd : sides) {
            const std::size_t L = 2;
            std::vector<std::int64_t> flat(sd.rows * L);
            for (std::size_t i = 0; i < sd.rows; ++i) {
                flat[i * L + 0] = static_cast<std::int64_t>(i % 9);      // 9 groups, above the group floor
                flat[i * L + 1] = static_cast<std::int64_t>(i % 1013) - 500;
            }
            const DT dts[2] = {DT::I64, DT::I64};
            gpudb::Aggregator::RowSpan sp;
            sp.lanes = flat.data(); sp.rows = sd.rows; sp.n_lanes = L; sp.valid = nullptr;
            auto ac = auto_agg->upload_rows_exact(&sp, 1, dts, L);
            auto rc = ref_agg->upload_rows_exact(&sp, 1, dts, L);
            gpudb::GroupByFilter f;
            f.columns = 0x0Fu;                       // keys, sums, counts, count(*)
            gpudb::exact_path_note().clear();
            auto got = auto_agg->groupby_exact_resident(*ac[0], ac[1].get(), cap, f);
            const std::string path = gpudb::exact_path_note();
            auto want = ref_agg->groupby_exact_resident(*rc[0], rc[1].get(), cap, f);
            const char* want_path = have_direct ? sd.want : "sort";
            if (path != want_path)
                std::printf("    FAIL %s: path %s, expected %s\n", sd.name, path.c_str(), want_path);
            EXPECT_EQ(path == want_path, true);
            const bool ok = got.keys == want.keys && got.sums == want.sums &&
                            got.sums_hi == want.sums_hi && got.counts == want.counts &&
                            got.counts_star == want.counts_star;
            if (!ok) std::printf("    FAIL %s: answer differs\n", sd.name);
            EXPECT(ok);
        }
        // two groups never take the direct path, however much work there is
        {
            const std::size_t rows = 7'000'000, L = 2;
            std::vector<std::int64_t> flat(rows * L);
            for (std::size_t i = 0; i < rows; ++i) {
                flat[i * L + 0] = static_cast<std::int64_t>(i & 1);
                flat[i * L + 1] = static_cast<std::int64_t>(i % 977);
            }
            const DT dts[2] = {DT::I64, DT::I64};
            gpudb::Aggregator::RowSpan sp;
            sp.lanes = flat.data(); sp.rows = rows; sp.n_lanes = L; sp.valid = nullptr;
            auto ac = auto_agg->upload_rows_exact(&sp, 1, dts, L);
            auto rc = ref_agg->upload_rows_exact(&sp, 1, dts, L);
            gpudb::GroupByFilter f;
            f.columns = 0x0Fu;
            gpudb::exact_path_note().clear();
            auto got = auto_agg->groupby_exact_resident(*ac[0], ac[1].get(), cap, f);
            const std::string path = gpudb::exact_path_note();
            auto want = ref_agg->groupby_exact_resident(*rc[0], rc[1].get(), cap, f);
            if (path != "sort") std::printf("    FAIL two groups: path %s, expected sort\n", path.c_str());
            EXPECT_EQ(path == "sort", true);
            EXPECT(got.keys == want.keys && got.sums == want.sums && got.counts_star == want.counts_star);
        }
    } catch (const std::exception& e) {
        ++failures; ++total;
        std::printf("    FAIL: %s\n", e.what());
    }

    // Every row masked out, and a key whose every row is NULL.
    std::printf("  degenerate inputs:\n");
    try {
        const std::size_t N = 300'001, L = 2;
        std::vector<std::int64_t> flat(N * L);
        std::vector<std::vector<std::uint64_t>> valid(L, std::vector<std::uint64_t>((N + 63) / 64, ~std::uint64_t{0}));
        for (std::size_t i = 0; i < N; ++i) {
            flat[i * L + 0] = static_cast<std::int64_t>(i % 3);
            flat[i * L + 1] = static_cast<std::int64_t>(i);
            valid[0][i >> 6] &= ~(std::uint64_t{1} << (i & 63));       // every key NULL
        }
        std::vector<const std::uint64_t*> vp{valid[0].data(), valid[1].data()};
        const DT dts[2] = {DT::I64, DT::I64};
        gpudb::Aggregator::RowSpan sp;
        sp.lanes = flat.data(); sp.rows = N; sp.n_lanes = L; sp.valid = vp.data();
        auto dcols = direct_agg->upload_rows_exact(&sp, 1, dts, L);
        auto rcols = ref_agg->upload_rows_exact(&sp, 1, dts, L);
        gpudb::GroupByFilter f;
        gpudb::exact_path_note().clear();
        auto got = direct_agg->groupby_exact_resident(*dcols[0], dcols[1].get(), cap, f);
        auto want = ref_agg->groupby_exact_resident(*rcols[0], rcols[1].get(), cap, f);
        EXPECT(same(got, want, "all keys NULL"));
        gpudb::Predicate dp, rp;
        dp.op = rp.op = Op::GT; dp.value = rp.value = std::int64_t{1} << 40;
        dp.col = dcols[1].get(); rp.col = rcols[1].get();
        auto gotm = direct_agg->groupby_exact_masked_resident(*dcols[0], dcols[1].get(), &dp, 1, cap, f);
        auto wantm = ref_agg->groupby_exact_masked_resident(*rcols[0], rcols[1].get(), &rp, 1, cap, f);
        EXPECT(same(gotm, wantm, "everything masked out"));
    } catch (const std::exception& e) {
        ++failures; ++total;
        std::printf("    FAIL: %s\n", e.what());
    }
}
// A GPU whose compiler refuses the direct path's pipelines must cost the
// operator nothing but the path: no throw out of prepare() or out of any
// exact call, the sort path answers, the answer is the reference's, and the
// reason is kept. GPUDB_METAL_DIRECT_DISABLE_PSO makes every direct pipeline
// refuse, so this runs on a device where the path would otherwise work.
void test_direct_pso_fallback_body();

void test_direct_pso_fallback() {
    std::printf("\n--- exact GROUP BY: the direct path's pipelines refused ---\n");
    try {
        test_direct_pso_fallback_body();
    } catch (const std::exception& e) {
        std::printf("  skipped (%s)\n", e.what());
    }
}

void test_direct_pso_fallback_body() {
    using DT = gpudb::Dtype;
    const std::size_t cap = std::size_t(100) * 1000000;
    auto ref_agg = gpudb::make_aggregator(gpudb::Backend::CPU);

    const std::size_t N = 3'000'011, L = 3;
    std::vector<std::int64_t> flat(N * L);
    std::vector<std::vector<std::uint64_t>> valid(L, std::vector<std::uint64_t>((N + 63) / 64, ~std::uint64_t{0}));
    std::mt19937_64 rng(0xF0FAULL);
    std::uniform_int_distribution<int> pct(0, 99);
    for (std::size_t i = 0; i < N; ++i) {
        flat[i * L + 0] = static_cast<std::int64_t>(i % 9);
        flat[i * L + 1] = (std::int64_t{1} << 61) - static_cast<std::int64_t>(i % 877);
        flat[i * L + 2] = static_cast<std::int64_t>(i % 1000);
        if (pct(rng) < 5) valid[0][i >> 6] &= ~(std::uint64_t{1} << (i & 63));
        if (pct(rng) < 7) valid[1][i >> 6] &= ~(std::uint64_t{1} << (i & 63));
    }
    std::vector<const std::uint64_t*> vp{valid[0].data(), valid[1].data(), valid[2].data()};
    const DT dts[3] = {DT::I64, DT::I64, DT::I64};
    gpudb::Aggregator::RowSpan sp;
    sp.lanes = flat.data(); sp.rows = N; sp.n_lanes = L; sp.valid = vp.data();

    auto rc = ref_agg->upload_rows_exact(&sp, 1, dts, L);
    for (const char* mode : {"auto", "direct"}) {
        setenv("GPUDB_METAL_DIRECT_DISABLE_PSO", "1", 1);
        if (std::string(mode) == "auto") unsetenv("GPUDB_METAL_GROUPBY_EXACT_PATH");
        else setenv("GPUDB_METAL_GROUPBY_EXACT_PATH", "direct", 1);
        try {
            auto agg = gpudb::make_aggregator(gpudb::Backend::METAL);
            auto mc = agg->upload_rows_exact(&sp, 1, dts, L);
            mc[0]->prepare();                       // must not throw either
            gpudb::Predicate mp, rp;
            mp.op = rp.op = gpudb::Predicate::Op::LT;
            mp.value = rp.value = 800;
            mp.col = mc[2].get(); rp.col = rc[2].get();
            gpudb::GroupByFilter f;
            f.columns = 0x0Fu;
            gpudb::exact_path_note().clear();
            gpudb::exact_path_reason().clear();
            auto got = agg->groupby_exact_masked_resident(*mc[0], mc[1].get(), &mp, 1, cap, f);
            const std::string path = gpudb::exact_path_note(), why = gpudb::exact_path_reason();
            auto want = ref_agg->groupby_exact_masked_resident(*rc[0], rc[1].get(), &rp, 1, cap, f);
            if (path != "sort") std::printf("    FAIL %s: path %s, expected sort\n", mode, path.c_str());
            EXPECT_EQ(path == "sort", true);
            if (why.empty()) std::printf("    FAIL %s: no reason recorded\n", mode);
            EXPECT(!why.empty());
            const bool ok = got.keys == want.keys && got.key_null == want.key_null &&
                            got.sums == want.sums && got.sums_hi == want.sums_hi &&
                            got.counts == want.counts && got.counts_star == want.counts_star;
            if (!ok) std::printf("    FAIL %s: answer differs from the reference\n", mode);
            EXPECT(ok);
            // the id lane is never built, so the column costs what it did
            EXPECT_EQ(mc[0]->resident_bytes() > 0, true);
            std::printf("  %s: %zu groups through the sort path (%s)\n", mode, got.keys.size(), why.c_str());
        } catch (const std::exception& e) {
            // A constructor pipeline that will not build means the device is
            // unusable, which this block is not here to judge.
            if (std::string(e.what()).find("Metal pipeline ") == 0)
                std::printf("  %s: skipped (%s)\n", mode, e.what());
            else {
                ++failures; ++total;
                std::printf("    FAIL %s: threw %s\n", mode, e.what());
            }
        }
        unsetenv("GPUDB_METAL_DIRECT_DISABLE_PSO");
        unsetenv("GPUDB_METAL_GROUPBY_EXACT_PATH");
    }
}

#endif  // GPUDB_HAVE_METAL

// ---------------------------------------------------------------------------
// Shedding a column's derived structures (docs/RESIDENT_COLUMNS_DESIGN.md §9).
// Once a key column has a group-id lane and the dispatch rule sends its calls
// to the direct path, the sort cache is dead weight and, for a column that is
// a key and nothing else, so is the key lane: dkeys[gid[row]] reproduces it.
// What this block pins is that neither is ever an answer: every exact form
// stays bit-identical to the CPU reference over a shed column, a call that
// wants what was shed rebuilds it exactly once, and the shapes the rule must
// refuse (a 2-group key, more groups than the id lane holds, an input too
// small, a device without the direct path) never shed at all.
// ---------------------------------------------------------------------------
#if GPUDB_HAVE_METAL
void test_shed_derived_body();

void test_shed_derived() {
    std::printf("\n--- resident columns: shedding the sort cache and the key lane ---\n");
    try {
        test_shed_derived_body();
    } catch (const std::exception& e) {
        std::printf("  skipped (%s)\n", e.what());
    }
}

void test_shed_derived_body() {
    using DT = gpudb::Dtype;
    const std::size_t cap = std::size_t(100) * 1000000;
    // Small enough to upload in a moment, large enough that the dispatch rule
    // admits it once GPUDB_METAL_DIRECT_MIN_WORK says so. The rule is the
    // real one; only the constant it compares against is lowered, exactly as
    // the sweeps do.
    const std::size_t N = 300'003;
    const std::size_t L = 4;                 // key, payload, second payload, i64 predicate

    auto make_metal = [](const char* path, const char* min_work) {
        setenv("GPUDB_METAL_GROUPBY_EXACT_PATH", path, 1);
        if (min_work) setenv("GPUDB_METAL_DIRECT_MIN_WORK", min_work, 1);
        auto a = gpudb::make_aggregator(gpudb::Backend::METAL);
        unsetenv("GPUDB_METAL_GROUPBY_EXACT_PATH");
        unsetenv("GPUDB_METAL_DIRECT_MIN_WORK");
        return a;
    };

    // (key, payload, payload2, predicate) with NULLs in the key and the
    // payload; `distinct` distinct keys, not dense, negative and positive.
    struct Lanes {
        std::vector<std::int64_t> flat;
        std::vector<std::vector<std::uint64_t>> valid;
        std::vector<const std::uint64_t*> vp;
    };
    auto build = [&](std::size_t distinct, bool null_keys) {
        auto x = std::make_shared<Lanes>();
        x->flat.resize(N * L);
        x->valid.assign(L, std::vector<std::uint64_t>((N + 63) / 64, ~std::uint64_t{0}));
        for (std::size_t i = 0; i < N; ++i) {
            const std::size_t g = (i * 2654435761ull + (i >> 5)) % distinct;
            x->flat[i * L + 0] = static_cast<std::int64_t>(g) * 0x100000001LL - 11;
            x->flat[i * L + 1] = (std::int64_t{1} << 61) - static_cast<std::int64_t>(i % 1013) * 3;
            x->flat[i * L + 2] = static_cast<std::int64_t>(i % 251) - 125;
            x->flat[i * L + 3] = static_cast<std::int64_t>(i % 1000);
            if (null_keys && i % 29 == 0) x->valid[0][i >> 6] &= ~(std::uint64_t{1} << (i & 63));
            if (i % 37 == 0)              x->valid[1][i >> 6] &= ~(std::uint64_t{1} << (i & 63));
        }
        for (std::size_t l = 0; l < L; ++l) x->vp.push_back(x->valid[l].data());
        return x;
    };
    const DT dts[L] = {DT::I64, DT::I64, DT::I64, DT::I64};
    auto span_of = [&](const std::shared_ptr<Lanes>& x) {
        gpudb::Aggregator::RowSpan sp;
        sp.lanes = x->flat.data(); sp.rows = N; sp.n_lanes = L; sp.valid = x->vp.data();
        return sp;
    };
    auto eq = [&](const gpudb::GroupByResidentResult& a, const gpudb::GroupByResidentResult& b,
                  const char* what) {
        const bool ok = a.keys == b.keys && a.key_null == b.key_null && a.sums == b.sums &&
                        a.sums_hi == b.sums_hi && a.counts == b.counts &&
                        a.counts_star == b.counts_star && a.mins == b.mins && a.maxs == b.maxs;
        if (!ok) std::printf("    FAIL %s: the answer differs from the CPU reference\n", what);
        EXPECT(ok);
        return ok;
    };

    auto ref_agg = gpudb::make_aggregator(gpudb::Backend::CPU);
    const char* disable = std::getenv("GPUDB_METAL_DIRECT_DISABLE_PSO");
    const std::string disabled = disable ? disable : "";

    // Is the direct path available here at all? (A hosted runner's
    // virtualised GPU refuses its pipelines; this block then asserts the
    // opposite — that nothing sheds — and is just as much of a test.)
    bool have_direct = false;
    {
        auto probe = make_metal("auto", "1000");
        auto x = build(7, false);
        auto sp = span_of(x);
        auto c = probe->upload_rows_exact(&sp, 1, dts, L);
        gpudb::GroupByFilter f;
        f.columns = 0x0Fu;
        gpudb::exact_path_note().clear();
        (void)probe->groupby_exact_resident(*c[0], c[1].get(), cap, f);
        have_direct = gpudb::exact_path_note() == "direct";
    }
    if (!have_direct)
        std::printf("  the direct path is unavailable here: the assertions below are that "
                    "nothing sheds (%s)\n", gpudb::exact_path_reason().c_str());
    // A call that the direct path declines for its own reasons (an optional
    // pipeline that will not build, under GPUDB_METAL_DIRECT_DISABLE_PSO=masked32)
    // sheds nothing either, so every byte assertion below asks the path note
    // what actually ran rather than assuming.

    // ---- what sheds, and what the bytes then are ----
    // The key is 8 bytes wide (values spread past int32) so the numbers are
    // the interesting ones: lane 8, sort cache 8 + 4 per valid row, id lane
    // 1 per row plus the distinct keys.
    {
        auto agg = make_metal("auto", "1000");
        auto x = build(9, true);
        auto sp = span_of(x);
        std::vector<std::unique_ptr<gpudb::ResidentColumn>> cols;
        {
            gpudb::KeyOnlyLanes note(1);          // lane 0 is a key and nothing else
            cols = agg->upload_rows_exact(&sp, 1, dts, L);
        }
        auto rc = ref_agg->upload_rows_exact(&sp, 1, dts, L);
        cols[0]->prepare();
        const std::size_t nulls = cols[0]->null_count();
        const std::size_t valid = N - nulls;
        const std::size_t bitmap = ((N + 63) / 64) * 8;
        // after prepare: lane + bitmap + sort cache + id lane + distinct keys
        const std::size_t want_prepared = N * 8 + bitmap + valid * (8 + 4) + N * 1 + 9 * 8;
        const std::size_t prepared_bytes = cols[0]->resident_bytes();
        if (have_direct && prepared_bytes != want_prepared)
            std::printf("    FAIL prepared bytes %zu, expected %zu\n", prepared_bytes, want_prepared);
        EXPECT_EQ(!have_direct || prepared_bytes == want_prepared, true);

        gpudb::GroupByFilter f;
        f.columns = 0x3Fu;
        gpudb::exact_path_note().clear();
        auto got = agg->groupby_exact_resident(*cols[0], cols[1].get(), cap, f);
        auto want = ref_agg->groupby_exact_resident(*rc[0], rc[1].get(), cap, f);
        eq(got, want, "first call (nothing shed yet)");
        // where the path is unavailable this must be the sort path; where it
        // is available, an optional pipeline may still decline THIS shape
        EXPECT_EQ(have_direct || gpudb::exact_path_note() == "sort", true);

        const bool direct1 = gpudb::exact_path_note() == "direct";
        const std::size_t after = cols[0]->resident_bytes();
        if (direct1) {
            // the sort cache AND the lane are gone: what is left is the
            // bitmap, the id lane and the distinct keys
            const std::size_t want_shed = bitmap + N * 1 + 9 * 8;
            if (after != want_shed)
                std::printf("    FAIL shed bytes %zu, expected %zu (was %zu)\n",
                            after, want_shed, prepared_bytes);
            EXPECT_EQ(after, want_shed);
            EXPECT(after < prepared_bytes / 2);
        } else {
            EXPECT_EQ(after, prepared_bytes);
        }

        // ---- every exact form, over the shed column ----
        gpudb::exact_path_note().clear();
        auto got2 = agg->groupby_exact_resident(*cols[0], cols[1].get(), cap, f);
        eq(got2, want, "sum/count/min/max over a shed column");
        EXPECT_EQ(gpudb::exact_path_note() == (direct1 ? "direct" : "sort"), true);
        {   // a WHERE mask, and a predicate on a lane that is not the key
            gpudb::Predicate p{};
            p.col = cols[3].get(); p.op = gpudb::Predicate::Op::LT; p.value = 640;
            gpudb::Predicate rp = p; rp.col = rc[3].get();
            auto g = agg->groupby_exact_masked_resident(*cols[0], cols[1].get(), &p, 1, cap, f);
            auto w = ref_agg->groupby_exact_masked_resident(*rc[0], rc[1].get(), &rp, 1, cap, f);
            eq(g, w, "masked GROUP BY over a shed column");
        }
        {   // several payloads
            gpudb::MultiPayload mp[2] = {{cols[1].get(), 0x3Fu}, {cols[2].get(), 0x3Fu}};
            gpudb::MultiPayload rmp[2] = {{rc[1].get(), 0x3Fu}, {rc[2].get(), 0x3Fu}};
            auto g = agg->groupby_exact_masked_multi(*cols[0], mp, 2, 0, nullptr, 0, cap, f);
            auto w = ref_agg->groupby_exact_masked_multi(*rc[0], rmp, 2, 0, nullptr, 0, cap, f);
            EXPECT_EQ(g.size(), w.size());
            for (std::size_t p = 0; p < g.size() && p < w.size(); ++p)
                eq(g[p], w[p], "two payloads over a shed column");
        }
        {   // HAVING, which reads the group rows the direct pass wrote
            gpudb::GroupByFilter h;
            h.agg = gpudb::GroupByFilter::Agg::CountStar;
            h.cmp = gpudb::GroupByFilter::Cmp::GT;
            h.threshold_i64 = 1000;
            h.columns = 0x3Fu;
            auto g = agg->groupby_exact_resident(*cols[0], cols[1].get(), cap, h);
            auto w = ref_agg->groupby_exact_resident(*rc[0], rc[1].get(), cap, h);
            eq(g, w, "HAVING over a shed column");
        }
        EXPECT_EQ(cols[0]->resident_bytes(), after);      // still shed: nothing wanted the lane

        // ---- a call that wants what was shed rebuilds it, once ----
        // count(*) with no payload at all is the shape the work rule cannot
        // admit (rows x 0 is 0), so it takes the sort path and needs the
        // cache — and the cache is built out of the lane.
        const long c0 = gpudb::resident_cache_rebuilds().load();
        const long l0 = gpudb::resident_lane_rebuilds().load();
        {
            gpudb::GroupByFilter f0;
            gpudb::exact_path_note().clear();
            auto g = agg->groupby_exact_resident(*cols[0], nullptr, cap, f0);
            auto w = ref_agg->groupby_exact_resident(*rc[0], nullptr, cap, f0);
            const bool ok = g.keys == w.keys && g.key_null == w.key_null &&
                            g.counts_star == w.counts_star;
            if (!ok) std::printf("    FAIL count(*) over a shed column\n");
            EXPECT(ok);
            EXPECT_EQ(gpudb::exact_path_note() == "sort", true);
        }
        auto sort_agg = make_metal("sort", nullptr);
        gpudb::exact_path_note().clear();
        auto srt = sort_agg->groupby_exact_resident(*cols[0], cols[1].get(), cap, f);
        eq(srt, want, "the sort path over a shed column");
        EXPECT_EQ(gpudb::exact_path_note() == "sort", true);
        const long c1 = gpudb::resident_cache_rebuilds().load();
        const long l1 = gpudb::resident_lane_rebuilds().load();
        if (direct1) {
            if (c1 - c0 != 1 || l1 - l0 != 1)
                std::printf("    FAIL rebuilds: cache %ld, lane %ld (expected 1 each)\n",
                            c1 - c0, l1 - l0);
            EXPECT_EQ(c1 - c0, 1L);
            EXPECT_EQ(l1 - l0, 1L);
            // both are back, so the column costs what a prepared one costs
            EXPECT_EQ(cols[0]->resident_bytes(), want_prepared);
        }
        // a further sort-path call rebuilds nothing: the rebuild pinned them
        auto srt2 = sort_agg->groupby_exact_resident(*cols[0], cols[1].get(), cap, f);
        eq(srt2, want, "the sort path again");
        EXPECT_EQ(gpudb::resident_cache_rebuilds().load() - c1, 0L);
        EXPECT_EQ(gpudb::resident_lane_rebuilds().load() - l1, 0L);
        // and a direct call may not shed them again
        gpudb::exact_path_note().clear();
        auto d3 = agg->groupby_exact_resident(*cols[0], cols[1].get(), cap, f);
        eq(d3, want, "direct again, after the pin");
        EXPECT_EQ(!direct1 || cols[0]->resident_bytes() == want_prepared, true);
        std::printf("  8-byte key, %zu rows, 9 groups: %zu -> %zu bytes, rebuilt to %zu\n",
                    N, prepared_bytes, after, cols[0]->resident_bytes());
    }

    // ---- a key that is NOT key-only keeps its lane (a WHERE reads it) ----
    {
        auto agg = make_metal("auto", "1000");
        auto x = build(11, true);
        auto sp = span_of(x);
        auto cols = agg->upload_rows_exact(&sp, 1, dts, L);   // no key-only note
        auto rc = ref_agg->upload_rows_exact(&sp, 1, dts, L);
        cols[0]->prepare();
        gpudb::GroupByFilter f; f.columns = 0x3Fu;
        gpudb::exact_path_note().clear();
        auto got = agg->groupby_exact_resident(*cols[0], cols[1].get(), cap, f);
        auto want = ref_agg->groupby_exact_resident(*rc[0], rc[1].get(), cap, f);
        eq(got, want, "a column that is not key-only");
        const bool direct = gpudb::exact_path_note() == "direct";
        const std::size_t nulls = cols[0]->null_count(), valid = N - nulls;
        const std::size_t bitmap = ((N + 63) / 64) * 8;
        // the cache goes if the call went direct; the lane stays either way
        const std::size_t want_bytes = N * 8 + bitmap + (direct ? 0 : valid * 12) +
                                       (have_direct ? N * 1 + 11 * 8 : 0);
        if (cols[0]->resident_bytes() != want_bytes)
            std::printf("    FAIL not key-only: %zu bytes, expected %zu\n",
                        cols[0]->resident_bytes(), want_bytes);
        EXPECT_EQ(cols[0]->resident_bytes(), want_bytes);
    }

    // ---- a key-only column a call READS as a predicate lane keeps its lane ----
    {
        auto agg = make_metal("auto", "1000");
        auto x = build(9, false);
        auto sp = span_of(x);
        std::vector<std::unique_ptr<gpudb::ResidentColumn>> cols;
        {
            gpudb::KeyOnlyLanes note(1);
            cols = agg->upload_rows_exact(&sp, 1, dts, L);
        }
        auto rc = ref_agg->upload_rows_exact(&sp, 1, dts, L);
        cols[0]->prepare();
        gpudb::GroupByFilter f; f.columns = 0x3Fu;
        gpudb::Predicate p{};                       // WHERE on the GROUP BY key itself
        p.col = cols[0].get(); p.op = gpudb::Predicate::Op::NE; p.value = -11;
        gpudb::Predicate rp = p; rp.col = rc[0].get();
        const long l0 = gpudb::resident_lane_rebuilds().load();
        auto got = agg->groupby_exact_masked_resident(*cols[0], cols[1].get(), &p, 1, cap, f);
        auto want = ref_agg->groupby_exact_masked_resident(*rc[0], rc[1].get(), &rp, 1, cap, f);
        eq(got, want, "a WHERE on the key of a key-only column");
        const std::size_t bitmap = 0;               // no NULL keys in this one
        (void)bitmap;
        // the lane is still there (the call read it), the cache is not
        EXPECT_EQ(cols[0]->resident_bytes() >= N * 8, true);
        // and reading it did not cost a rebuild: it was never shed
        EXPECT_EQ(gpudb::resident_lane_rebuilds().load() - l0, 0L);
    }

    // ---- the shapes that must never shed ----
    // A 2-group key (the sort path's best case), a key with more distinct
    // values than the id lane holds, and an input the work rule does not
    // admit: for each, the column keeps everything it had.
    struct Refusal { const char* name; std::size_t distinct; const char* min_work; };
    const Refusal refusals[] = {
        {"a 2-group key",              2,    "1000"},
        {"more groups than the lane",  4096, "1000"},
        {"an input below the work rule", 9,  nullptr},   // the shipping 6,000,000
    };
    for (const Refusal& r : refusals) {
        auto agg = make_metal("auto", r.min_work);
        auto x = build(r.distinct, false);
        auto sp = span_of(x);
        std::vector<std::unique_ptr<gpudb::ResidentColumn>> cols;
        {
            gpudb::KeyOnlyLanes note(1);
            cols = agg->upload_rows_exact(&sp, 1, dts, L);
        }
        auto rc = ref_agg->upload_rows_exact(&sp, 1, dts, L);
        cols[0]->prepare();
        const std::size_t before = cols[0]->resident_bytes();
        gpudb::GroupByFilter f; f.columns = 0x3Fu;
        auto got = agg->groupby_exact_resident(*cols[0], cols[1].get(), cap, f);
        auto want = ref_agg->groupby_exact_resident(*rc[0], rc[1].get(), cap, f);
        eq(got, want, r.name);
        if (cols[0]->resident_bytes() != before)
            std::printf("    FAIL %s: shed %zu bytes and should have shed none\n",
                        r.name, before - cols[0]->resident_bytes());
        EXPECT_EQ(cols[0]->resident_bytes(), before);
        EXPECT(before >= N * 8);
    }

    // ---- a hash key with NULLs: the lane comes back bit for bit ----
    // The rebuild reads dkeys[gid[row]], so it has to reproduce every valid
    // cell exactly — including keys that are 64-bit hashes with the top bits
    // set, which is what a VARCHAR tuple key looks like.
    {
        auto agg = make_metal("auto", "1000");
        auto x = std::make_shared<Lanes>();
        x->flat.resize(N * L);
        x->valid.assign(L, std::vector<std::uint64_t>((N + 63) / 64, ~std::uint64_t{0}));
        const std::int64_t hashes[6] = {
            static_cast<std::int64_t>(0x9E3779B97F4A7C15ull), static_cast<std::int64_t>(0xC2B2AE3D27D4EB4Full),
            static_cast<std::int64_t>(0x165667B19E3779F9ull), static_cast<std::int64_t>(0x27D4EB2F165667C5ull),
            static_cast<std::int64_t>(0x85EBCA77C2B2AE63ull), static_cast<std::int64_t>(0x0000000100000001ull)};
        for (std::size_t i = 0; i < N; ++i) {
            x->flat[i * L + 0] = hashes[i % 6];
            x->flat[i * L + 1] = static_cast<std::int64_t>(i % 9973) - 5000;
            x->flat[i * L + 2] = static_cast<std::int64_t>(i % 251) - 125;
            x->flat[i * L + 3] = static_cast<std::int64_t>(i % 1000);
            if (i % 23 == 0) x->valid[0][i >> 6] &= ~(std::uint64_t{1} << (i & 63));
        }
        for (std::size_t l = 0; l < L; ++l) x->vp.push_back(x->valid[l].data());
        auto sp = span_of(x);
        std::vector<std::unique_ptr<gpudb::ResidentColumn>> cols;
        {
            gpudb::KeyOnlyLanes note(1);
            cols = agg->upload_rows_exact(&sp, 1, dts, L);
        }
        auto rc = ref_agg->upload_rows_exact(&sp, 1, dts, L);
        cols[0]->prepare();
        gpudb::GroupByFilter f; f.columns = 0x3Fu;
        auto want = ref_agg->groupby_exact_resident(*rc[0], rc[1].get(), cap, f);
        gpudb::exact_path_note().clear();
        auto got = agg->groupby_exact_resident(*cols[0], cols[1].get(), cap, f);
        eq(got, want, "a 64-bit hash key with NULLs");
        const bool shed = cols[0]->resident_bytes() < N * 8;
        EXPECT_EQ(shed, gpudb::exact_path_note() == "direct");
        // force the lane back through the sort path and compare again
        auto sort_agg = make_metal("sort", nullptr);
        auto srt = sort_agg->groupby_exact_resident(*cols[0], cols[1].get(), cap, f);
        eq(srt, want, "a rebuilt hash key lane, through the sort path");
        // the rebuilt lane is the sorted keys' only source, so every distinct
        // cell came back bit for bit; a NULL cell is never read (stage A).
        EXPECT_EQ(srt.keys.size(), want.keys.size());
    }

    // ---- two threads on one shed column, one forcing each path ----
    // The shed, the rebuild and the reads are all guarded; what this asserts
    // is that neither thread ever sees half a structure and that both
    // answers are the reference's.
    {
        auto agg = make_metal("auto", "1000");
        auto sort_agg = make_metal("sort", nullptr);
        auto x = build(13, true);
        auto sp = span_of(x);
        std::vector<std::unique_ptr<gpudb::ResidentColumn>> cols;
        {
            gpudb::KeyOnlyLanes note(1);
            cols = agg->upload_rows_exact(&sp, 1, dts, L);
        }
        auto rc = ref_agg->upload_rows_exact(&sp, 1, dts, L);
        cols[0]->prepare();
        gpudb::GroupByFilter f; f.columns = 0x3Fu;
        auto want = ref_agg->groupby_exact_resident(*rc[0], rc[1].get(), cap, f);
        (void)agg->groupby_exact_resident(*cols[0], cols[1].get(), cap, f);   // sheds
        std::atomic<int> bad{0};
        std::atomic<bool> go{false};
        auto run = [&](gpudb::Aggregator* a, int rounds) {
            while (!go.load()) std::this_thread::yield();
            for (int i = 0; i < rounds; ++i) {
                try {
                    auto g = a->groupby_exact_resident(*cols[0], cols[1].get(), cap, f);
                    if (!(g.keys == want.keys && g.sums == want.sums && g.sums_hi == want.sums_hi &&
                          g.counts == want.counts && g.counts_star == want.counts_star &&
                          g.mins == want.mins && g.maxs == want.maxs))
                        bad.fetch_add(1);
                } catch (const std::exception&) {
                    bad.fetch_add(1);
                }
            }
        };
        // The operators keep per-aggregator scratch, so each thread drives its
        // own aggregator — what they share is the column.
        std::thread t1(run, sort_agg.get(), 6);
        std::thread t2(run, agg.get(), 6);
        go.store(true);
        t1.join();
        t2.join();
        if (bad.load()) std::printf("    FAIL %d concurrent answers differed or threw\n", bad.load());
        EXPECT_EQ(bad.load(), 0);
    }

    // ---- a device without the direct path never sheds ----
    // GPUDB_METAL_DIRECT_DISABLE_PSO=unsupported is the capability gate's
    // device: nothing is offered to it, no id lane is built, and a column
    // therefore keeps its lane and its cache whatever the calls are.
    if (disabled == "unsupported") {
        auto agg = make_metal("auto", "1000");
        auto x = build(9, true);
        auto sp = span_of(x);
        std::vector<std::unique_ptr<gpudb::ResidentColumn>> cols;
        {
            gpudb::KeyOnlyLanes note(1);
            cols = agg->upload_rows_exact(&sp, 1, dts, L);
        }
        auto rc = ref_agg->upload_rows_exact(&sp, 1, dts, L);
        cols[0]->prepare();
        const std::size_t before = cols[0]->resident_bytes();
        gpudb::GroupByFilter f; f.columns = 0x3Fu;
        gpudb::exact_path_note().clear();
        auto got = agg->groupby_exact_resident(*cols[0], cols[1].get(), cap, f);
        auto want = ref_agg->groupby_exact_resident(*rc[0], rc[1].get(), cap, f);
        eq(got, want, "the capability gate's device");
        EXPECT_EQ(gpudb::exact_path_note() == "sort", true);
        EXPECT_EQ(cols[0]->resident_bytes(), before);
        EXPECT(before >= N * 8);
        std::printf("  the direct path is unavailable: %zu bytes kept\n", before);
    }
}

// ---------------------------------------------------------------------------
// The WHERE stage of the sort path, fused and legacy (docs/TRANSPARENT_DESIGN.md
// §4.6). One pass that evaluates the whole conjunction per row must answer what
// the one-pass-per-term loop answers, limb for limb, on every shape the mask
// stage serves: each operator, IN lists, F64 lanes under DuckDB's total order,
// IsNull / IsNotNull, NULL cells in a predicate lane, narrow lanes at their
// width boundaries, nothing surviving and everything surviving, the compaction
// variant and the masked reduce, several payloads, HAVING and top-k, and a
// NULL-key group. A WHERE over more distinct columns than the fused kernel
// binds must fall back to the legacy pass rather than throw, and the note says
// which ran so a silent fallback fails the check.
// ---------------------------------------------------------------------------
void test_fused_mask_body();

void test_fused_mask() {
    std::printf("\n--- exact GROUP BY: the fused WHERE pass vs the legacy one ---\n");
    try {
        test_fused_mask_body();
    } catch (const std::exception& e) {
        std::printf("  skipped (%s)\n", e.what());
    }
}

void test_fused_mask_body() {
    using DT = gpudb::Dtype;
    using Op = gpudb::Predicate::Op;
    const std::size_t cap = std::size_t(100) * 1000000;
    auto ref_agg = gpudb::make_aggregator(gpudb::Backend::CPU);
    // Many-group keys only, so every call takes the sort path and therefore
    // the mask stage; `sort` is forced as well so nothing here depends on the
    // direct path's dispatch rule.
    auto metal = [] {
        setenv("GPUDB_METAL_GROUPBY_EXACT_PATH", "sort", 1);
        auto a = gpudb::make_aggregator(gpudb::Backend::METAL);
        unsetenv("GPUDB_METAL_GROUPBY_EXACT_PATH");
        return a;
    }();

    // Is the fused pass available on this GPU? A device that will not build
    // its pipelines answers everything through the legacy pass, which costs
    // the suite the path assertions and nothing else.
    // The device is read from its name, not from the note under test, so the
    // assertions stay independent of the code they check: the fused pass is
    // offered to Apple7 and later, and never to a virtualised Apple GPU.
    const char* knob = std::getenv("GPUDB_METAL_MASK_DISABLE_PSO");
    const std::string dev = metal->device_name();
    const bool offered = dev.find("Apple7") != std::string::npos &&
                         dev.find("Paravirtual") == std::string::npos;
    if (!offered)
        std::printf("  the fused pass is not offered to this device: every shape must answer "
                    "through the legacy pass (%s)\n", dev.c_str());
    const bool refused = (knob && *knob) || !offered;

    const std::size_t N = 2'600'003, L = 16;
    // Lanes: 0 key (10k distinct, NULLs), 1 key (120k distinct), 2 wide
    // payload, 3 narrow payload, 4..7 the width boundaries 1 / 2 / 4 / 8
    // bytes, 8 an F64 lane (NaN, +-inf, -0.0, NULLs), 9 all NULL, 10 an IN
    // list's lane, 11..15 four more distinct lanes so a WHERE can ask for more
    // than the kernel binds.
    std::vector<std::int64_t> flat(N * L);
    std::vector<std::vector<std::uint64_t>> valid(L, std::vector<std::uint64_t>((N + 63) / 64, ~std::uint64_t{0}));
    const double specials[5] = {std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::infinity(),
                                -std::numeric_limits<double>::infinity(), -0.0, 0.0};
    for (std::size_t i = 0; i < N; ++i) {
        auto clear = [&](std::size_t l) { valid[l][i >> 6] &= ~(std::uint64_t{1} << (i & 63)); };
        flat[i * L + 0] = static_cast<std::int64_t>((i * 2654435761ull) % 10'000) * 3 - 7;
        flat[i * L + 1] = static_cast<std::int64_t>((i * 40503ull) % 120'011);
        flat[i * L + 2] = (std::int64_t{1} << 62) - static_cast<std::int64_t>(i % 1013) * 7;
        flat[i * L + 3] = static_cast<std::int64_t>(i % 251) - 125;
        flat[i * L + 4] = static_cast<std::int64_t>(i % 256) - 128;            // width 1
        flat[i * L + 5] = static_cast<std::int64_t>(i % 65536) - 32768;        // width 2
        flat[i * L + 6] = static_cast<std::int64_t>(i % 4001) * 1'000'000 - 2'000'000'000LL;  // width 4
        flat[i * L + 7] = (std::int64_t{1} << 40) + static_cast<std::int64_t>(i % 7919);      // width 8
        double d = (i % 211 < 5) ? specials[i % 5]
                                 : static_cast<double>(static_cast<std::int64_t>(i % 997)) / 997.0;
        std::memcpy(&flat[i * L + 8], &d, sizeof(double));
        flat[i * L + 9] = 0;
        flat[i * L + 10] = static_cast<std::int64_t>(i % 17) - 8;
        for (std::size_t l = 11; l < L; ++l)
            flat[i * L + l] = static_cast<std::int64_t>((i + l * 13) % 97) - 48;
        if (i % 23 == 0) clear(0);        // a NULL-key group
        if (i % 31 == 0) clear(3);        // NULL payload cells
        if (i % 29 == 0) clear(5);        // NULL cells in a predicate lane
        if (i % 41 == 0) clear(8);
        clear(9);                         // a lane that is NULL everywhere
    }
    std::vector<const std::uint64_t*> vp(L);
    for (std::size_t l = 0; l < L; ++l) vp[l] = valid[l].data();
    DT dts[L];
    for (std::size_t l = 0; l < L; ++l) dts[l] = (l == 8) ? DT::F64 : DT::I64;
    gpudb::Aggregator::RowSpan sp;
    sp.lanes = flat.data(); sp.rows = N; sp.n_lanes = L; sp.valid = vp.data();
    auto mcols = metal->upload_rows_exact(&sp, 1, dts, L);
    auto rcols = ref_agg->upload_rows_exact(&sp, 1, dts, L);

    auto bits = [](double d) { std::int64_t b = 0; std::memcpy(&b, &d, sizeof(b)); return b; };
    const std::vector<std::int64_t> in_list = {-8, 0, 3, 7, 1234};
    const std::vector<std::int64_t> in_f64  = {bits(0.5), bits(std::numeric_limits<double>::infinity()),
                                               bits(-0.0)};

    // Every group's tuple, limb for limb, in order — the group order, the
    // NULL-key group's place and a group the WHERE emptied being absent are
    // all part of the contract.
    auto same = [&](const gpudb::GroupByResidentResult& a, const gpudb::GroupByResidentResult& b,
                    const char* what, std::uint32_t cols = gpudb::GroupByFilter::kAllColumns) {
        bool ok = a.keys == b.keys && a.key_null == b.key_null;
        if (cols & (1u << 1)) ok = ok && a.sums == b.sums && a.sums_hi == b.sums_hi;
        if (cols & (1u << 2)) ok = ok && a.counts == b.counts;
        if (cols & (1u << 3)) ok = ok && a.counts_star == b.counts_star;
        if (cols & (1u << 4)) ok = ok && a.mins == b.mins;
        if (cols & (1u << 5)) ok = ok && a.maxs == b.maxs;
        if (!ok) std::printf("    FAIL %s (%zu vs %zu groups)\n", what, a.keys.size(), b.keys.size());
        return ok;
    };

    // `heavy`: also run HAVING, both top-k directions and the multi-payload
    // call. Every case runs the plain form; the forms differ in what they do
    // with the finished tuple, not in how the mask is built, so four of them
    // carry the whole filter matrix and the rest pin the mask itself.
    struct Term { std::size_t lane; Op op; std::int64_t value; const std::vector<std::int64_t>* list; };
    struct Case { const char* name; bool heavy; std::vector<Term> terms; };
    const std::vector<Case> cases = {
        {"every operator", true, {{4, Op::GE, -120, nullptr}, {4, Op::NE, 0, nullptr},
                            {5, Op::LT, 30000, nullptr}, {5, Op::LE, 29999, nullptr},
                            {6, Op::GT, -2'000'000'000LL, nullptr}, {10, Op::EQ, 3, nullptr}}},
        {"in list",        false, {{10, Op::In, 0, &in_list}}},
        {"in list + term", false, {{10, Op::In, 0, &in_list}, {3, Op::LT, 100, nullptr}}},
        {"f64 total order",false, {{8, Op::GT, bits(0.4), nullptr}}},
        {"f64 in list",    false, {{8, Op::In, 0, &in_f64}}},
        {"f64 vs nan",     false, {{8, Op::LE, bits(std::numeric_limits<double>::quiet_NaN()), nullptr}}},
        {"is null",        false, {{5, Op::IsNull, 0, nullptr}}},
        {"is not null",    false, {{5, Op::IsNotNull, 0, nullptr}, {8, Op::IsNotNull, 0, nullptr}}},
        {"all-null lane",  false, {{9, Op::IsNotNull, 0, nullptr}}},          // 0 survivors
        {"everything",     false, {{7, Op::GE, 0, nullptr}}},                 // 100 % survivors
        {"nothing",        true,  {{6, Op::GT, 1LL << 40, nullptr}}},         // 0 survivors
        {"width 1 edge",   false, {{4, Op::GE, -128, nullptr}, {4, Op::LE, 127, nullptr}}},
        {"width 2 edge",   false, {{5, Op::GE, -32768, nullptr}, {5, Op::LE, 32767, nullptr}}},
        {"width 4 edge",   false, {{6, Op::GE, -2147483648LL, nullptr}, {6, Op::LE, 2147483647LL, nullptr}}},
        {"width 8 lane",   false, {{7, Op::LT, (1LL << 40) + 4000, nullptr}}},
        {"selective (5 %)",true,  {{3, Op::LT, -112, nullptr}}},              // below the compaction bound
        {"half",           true,  {{3, Op::LT, 0, nullptr}}},                 // the masked reduce
        {"key range",      false, {{0, Op::GE, 0, nullptr}, {3, Op::LT, 50, nullptr}}},
        {"key + null",     false, {{0, Op::NE, -7, nullptr}}},
        {"12 lanes",       false, {{3, Op::GE, -125, nullptr}, {4, Op::GE, -128, nullptr},
                            {5, Op::GE, -32768, nullptr}, {6, Op::GE, -2147483648LL, nullptr},
                            {7, Op::GE, 0, nullptr}, {8, Op::IsNotNull, 0, nullptr},
                            {10, Op::GE, -8, nullptr}, {11, Op::GE, -48, nullptr},
                            {12, Op::GE, -48, nullptr}, {13, Op::GE, -48, nullptr},
                            {14, Op::GE, -48, nullptr}, {15, Op::GE, -48, nullptr}}},
        {"13 lanes",       true,  {{2, Op::GE, 0, nullptr}, {3, Op::GE, -125, nullptr},
                            {4, Op::GE, -128, nullptr}, {5, Op::GE, -32768, nullptr},
                            {6, Op::GE, -2147483648LL, nullptr}, {7, Op::GE, 0, nullptr},
                            {8, Op::IsNotNull, 0, nullptr}, {9, Op::IsNull, 0, nullptr},
                            {10, Op::GE, -8, nullptr}, {11, Op::GE, -48, nullptr},
                            {12, Op::GE, -48, nullptr}, {13, Op::GE, -48, nullptr},
                            {14, Op::GE, -48, nullptr}}},
    };

    const char* shapes[2] = {"fused", "legacy"};
    int fused_seen = 0, legacy_seen = 0;
    for (std::size_t key = 0; key < 2; ++key) {     // 10k groups + NULLs, then 120k
        for (const Case& c : cases) {
            auto build = [&](const std::vector<std::unique_ptr<gpudb::ResidentColumn>>& col) {
                std::vector<gpudb::Predicate> ps(c.terms.size());
                for (std::size_t q = 0; q < c.terms.size(); ++q) {
                    ps[q].col = col[c.terms[q].lane].get();
                    ps[q].op = c.terms[q].op;
                    ps[q].value = c.terms[q].value;
                    if (c.terms[q].list) { ps[q].list = c.terms[q].list->data(); ps[q].n_list = c.terms[q].list->size(); }
                }
                return ps;
            };
            auto rp = build(rcols);
            auto mp = build(mcols);
            // The distinct columns the WHERE reads; above what the fused
            // kernel binds it has to fall back rather than throw.
            std::vector<std::size_t> distinct;
            for (const Term& t : c.terms)
                if (std::find(distinct.begin(), distinct.end(), t.lane) == distinct.end())
                    distinct.push_back(t.lane);
            const bool expect_fused = !refused && distinct.size() <= 12;

            gpudb::GroupByFilter forms[4];
            forms[1].agg = gpudb::GroupByFilter::Agg::CountStar;
            forms[1].cmp = gpudb::GroupByFilter::Cmp::GT;
            forms[1].threshold_i64 = 40;
            forms[2].agg = gpudb::GroupByFilter::Agg::Sum; forms[2].topk = 5; forms[2].topk_desc = true;
            forms[3].agg = gpudb::GroupByFilter::Agg::Min; forms[3].topk = 3; forms[3].topk_desc = false;
            const char* fname[4] = {"plain", "having", "topk desc", "topk asc"};
            for (int fi = 0; fi < (c.heavy ? 4 : 1); ++fi) {
                auto want = ref_agg->groupby_exact_masked_resident(
                    *rcols[key], rcols[3].get(), rp.data(), rp.size(), cap, forms[fi]);
                for (const char* sh : shapes) {
                    setenv("GPUDB_METAL_MASK_PATH", sh, 1);
                    gpudb::exact_path_note().clear();
                    gpudb::exact_mask_note().clear();
                    auto got = metal->groupby_exact_masked_resident(
                        *mcols[key], mcols[3].get(), mp.data(), mp.size(), cap, forms[fi]);
                    const std::string note = gpudb::exact_mask_note();
                    char what[200];
                    std::snprintf(what, sizeof(what), "key%zu / %s / %s / %s [%s]", key, c.name,
                                  fname[fi], sh, note.c_str());
                    EXPECT(same(got, want, what));
                    const char* wanted = (std::string(sh) == "fused" && expect_fused) ? "fused" : "legacy";
                    if (note != wanted)
                        std::printf("    FAIL %s: mask stage %s, expected %s\n", what,
                                    note.c_str(), wanted);
                    EXPECT_EQ(note == wanted, true);
                    if (note == "fused") ++fused_seen; else ++legacy_seen;
                }
            }
            // several payloads over one mask (§4.9): the wide lane without
            // min / max, the narrow one with
            for (int fi = 0; c.heavy && fi < 2; ++fi) {
                gpudb::GroupByFilter f;
                if (fi) { f.agg = gpudb::GroupByFilter::Agg::CountStar;
                          f.cmp = gpudb::GroupByFilter::Cmp::GE; f.threshold_i64 = 30; }
                auto multi = [&](const std::unique_ptr<gpudb::Aggregator>& a,
                                 const std::vector<std::unique_ptr<gpudb::ResidentColumn>>& col,
                                 std::vector<gpudb::Predicate>& p) {
                    gpudb::MultiPayload m[3];
                    m[0].vals = col[2].get(); m[0].columns = 0x0Fu;
                    m[1].vals = col[3].get(); m[1].columns = gpudb::GroupByFilter::kAllColumns;
                    m[2].vals = col[6].get(); m[2].columns = gpudb::GroupByFilter::kAllColumns;
                    return a->groupby_exact_masked_multi(*col[key], m, 3, 1, p.data(), p.size(), cap, f);
                };
                auto want = multi(ref_agg, rcols, rp);
                for (const char* sh : shapes) {
                    setenv("GPUDB_METAL_MASK_PATH", sh, 1);
                    gpudb::exact_mask_note().clear();
                    auto got = multi(metal, mcols, mp);
                    const std::string note = gpudb::exact_mask_note();
                    char what[200];
                    std::snprintf(what, sizeof(what), "key%zu / %s / multi%s / %s [%s]", key, c.name,
                                  fi ? " having" : "", sh, note.c_str());
                    EXPECT_EQ(got.size(), want.size());
                    for (std::size_t p = 0; p < got.size() && p < want.size(); ++p)
                        EXPECT(same(got[p], want[p], what, p == 0 ? 0x0Fu : gpudb::GroupByFilter::kAllColumns));
                    const char* wanted = (std::string(sh) == "fused" && expect_fused) ? "fused" : "legacy";
                    EXPECT_EQ(note == wanted, true);
                }
            }
        }
    }
    unsetenv("GPUDB_METAL_MASK_PATH");
    std::printf("  %d calls through the fused pass, %d through the legacy one (rows=%zu)\n",
                fused_seen, legacy_seen, N);
}
#endif  // GPUDB_HAVE_METAL

// ---------------------------------------------------------------------------
// ResidentColumn::prepare() (v0.7 milestone 0b, docs/TRANSPARENT_DESIGN.md
// §5.5/§5.6): ready = uploaded AND prepared; prepare is idempotent, safe to
// call concurrently with itself and with uploads on other threads, and
// changes no answer. Runs through the hybrid aggregator so the CPU-only
// build exercises the defaults (nothing to prepare) and the CUDA build
// exercises the real sort cache.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Stage D1 (docs/RESIDENT_COLUMNS_DESIGN.md §7): a lane read THROUGH an index
// must answer exactly what the gathered copy of that lane answers. The oracle
// is join_materialize itself: the same join is run twice, once materialising
// every lane and once returning the two row vectors, and then every operator
// is asked the same question of both — a grouped reduce keyed on the lane
// under test with a positional payload, which pins not only the lane's values
// but the ROW each value sits at, and a global masked aggregate over the same
// lanes. Covered: identity and scattered indexes, both join sides, the
// 1/2/4/8-byte width boundaries, F64 lanes carrying NaN / ±inf / -0.0, NULL
// cells in the lane, NULL cells in the INDEX (the unmatched side of an outer
// join), a chained three-step join, and the wide-key / DECIMAL shape.
// ---------------------------------------------------------------------------
// The indexed READ on its own, with no join in sight: a scattered index vector
// carrying NULL cells is built on the host, the lane it stands for is gathered
// on the host, both are uploaded, and the operators are asked the same question
// of each. This is what a backend that has the indexed read but not yet the
// indexed join can still be held to, and it covers the width boundaries, an F64
// lane with NaN / ±inf / -0.0, NULL cells in the lane and NULL cells in the
// index.
void test_indexed_reads_only(gpudb::Aggregator& agg,
                             const std::vector<std::unique_ptr<gpudb::ResidentColumn>>& P,
                             const std::vector<std::unique_ptr<gpudb::ResidentColumn>>& Bc,
                             const std::vector<std::int64_t>& pf,
                             const std::vector<std::int64_t>& bfl,
                             std::size_t PL, std::size_t BL,
                             std::size_t F, std::size_t B, std::size_t cap) {
    using DT = gpudb::Dtype;
    using Op = gpudb::Predicate::Op;
    const std::size_t R = 130'003;

    // The index, deliberately scattered and with NULL cells every 11th row.
    std::vector<std::int64_t> ixl(R * 2, 0);
    std::vector<std::uint64_t> ixv((R + 63) / 64, ~std::uint64_t{0});
    std::vector<std::uint64_t> ixv1((R + 63) / 64, ~std::uint64_t{0});
    std::vector<std::size_t> src(R, 0);
    std::vector<bool> inull(R, false);
    for (std::size_t d = 0; d < R; ++d) {
        src[d] = (d * 7919 + 13) % F;
        ixl[d * 2 + 0] = static_cast<std::int64_t>(src[d]);
        ixl[d * 2 + 1] = static_cast<std::int64_t>(d % 997);          // a key, never NULL
        if (d % 11 == 0) { inull[d] = true; ixv[d >> 6] &= ~(std::uint64_t{1} << (d & 63)); }
    }
    const std::uint64_t* ixvp[2] = {ixv.data(), ixv1.data()};
    DT idt[2] = {DT::I64, DT::I64};
    gpudb::Aggregator::RowSpan is;
    is.lanes = ixl.data(); is.rows = R; is.n_lanes = 2; is.valid = ixvp;
    auto IX = agg.upload_rows_exact(&is, 1, idt, 2);

    // A build-side index too, so a small table's gather is covered.
    std::vector<std::int64_t> bxl(R * 2, 0);
    std::vector<std::size_t> bsrc(R, 0);
    for (std::size_t d = 0; d < R; ++d) {
        bsrc[d] = (d * 131 + 7) % B;
        bxl[d * 2 + 0] = static_cast<std::int64_t>(bsrc[d]);
    }
    gpudb::Aggregator::RowSpan bxs;
    bxs.lanes = bxl.data(); bxs.rows = R; bxs.n_lanes = 2; bxs.valid = ixvp;
    auto BX = agg.upload_rows_exact(&bxs, 1, idt, 2);

    auto same = [&](const gpudb::GroupByResidentResult& a, const gpudb::GroupByResidentResult& b,
                    const std::string& what) {
        const bool ok = a.keys == b.keys && a.key_null == b.key_null &&
                        a.sums == b.sums && a.sums_hi == b.sums_hi &&
                        a.counts == b.counts && a.counts_star == b.counts_star &&
                        a.mins == b.mins && a.maxs == b.maxs;
        ++total;
        if (!ok) { ++failures; std::printf("    FAIL %s\n", what.c_str()); }
    };

    // Is the GROUPED form indexed on this backend, or only the row-order
    // global aggregate? Both are legitimate stages of D1; what is not
    // legitimate is answering an indexed call by ignoring the index, so the
    // probe insists that an unavailable form REFUSES.
    bool grouped_ok = true;
    {
        gpudb::MultiPayload probe{IX[1].get(), gpudb::GroupByFilter::kAllColumns, nullptr};
        try {
            agg.groupby_exact_masked_multi_indexed(
                gpudb::IndexedColumn{Bc[1].get(), BX[0].get()}, &probe, 1, 0, nullptr, 0, cap);
        } catch (const std::exception& e) {
            grouped_ok = false;
            std::printf("    the grouped reduce is not indexed on this backend yet (%s); the global "
                        "masked aggregate carries the checks\n", e.what());
        }
    }

    struct C { const char* name; std::size_t lane; bool build; };
    const std::vector<C> cs = {
        {"width 1", 3, false}, {"width 2", 4, false}, {"width 4", 5, false},
        {"width 8", 6, false}, {"f64 nan/inf/-0", 7, false}, {"lane with NULLs", 8, false},
        {"build width 1", 1, true}, {"build wide with NULLs", 2, true}, {"build f64", 3, true},
    };
    for (const auto& c : cs) {
        const std::size_t SL = c.build ? BL : PL;
        const std::vector<std::int64_t>& flat = c.build ? bfl : pf;
        const std::vector<std::size_t>& rows = c.build ? bsrc : src;
        const gpudb::ResidentColumn* srccol = c.build ? Bc[c.lane].get() : P[c.lane].get();
        const std::size_t SR = c.build ? B : F;
        // the host's gather of that lane through the same index
        std::vector<std::int64_t> exp(R * 2, 0);
        std::vector<std::uint64_t> ev0((R + 63) / 64, ~std::uint64_t{0});
        std::vector<std::uint64_t> ev1((R + 63) / 64, ~std::uint64_t{0});
        for (std::size_t d = 0; d < R; ++d) {
            exp[d * 2 + 0] = static_cast<std::int64_t>(d % 997);
            const std::size_t r = rows[d];
            bool cell_null = inull[d] || r >= SR;
            if (!cell_null) {
                // reproduce the source lane's NULL pattern from how it was built
                if (!c.build) cell_null = (c.lane == 8 && r % 37 == 0) || (c.lane == 7 && r % 53 == 0);
                else          cell_null = (c.lane == 2 && r % 17 == 0);
            }
            if (cell_null) { ev1[d >> 6] &= ~(std::uint64_t{1} << (d & 63)); exp[d * 2 + 1] = 0; }
            else exp[d * 2 + 1] = flat[r * SL + c.lane];
        }
        const std::uint64_t* evp[2] = {ev0.data(), ev1.data()};
        DT edt[2] = {DT::I64, srccol->dtype()};
        gpudb::Aggregator::RowSpan es;
        es.lanes = exp.data(); es.rows = R; es.n_lanes = 2; es.valid = evp;
        auto EX = agg.upload_rows_exact(&es, 1, edt, 2);
        const gpudb::ResidentColumn* index = c.build ? BX[0].get() : IX[0].get();

        gpudb::MultiPayload kp{EX[0].get(), gpudb::GroupByFilter::kAllColumns, nullptr};
        gpudb::Predicate pa{}, pb{};
        pa.col = EX[1].get(); pa.op = Op::IsNotNull;
        pb.col = srccol;      pb.op = Op::IsNotNull; pb.index = index;
        if (grouped_ok) {
            // as a KEY
            auto ka = agg.groupby_exact_masked_multi_indexed(
                gpudb::IndexedColumn{EX[1].get(), nullptr}, &kp, 1, 0, nullptr, 0, cap);
            auto kb = agg.groupby_exact_masked_multi_indexed(
                gpudb::IndexedColumn{srccol, index}, &kp, 1, 0, nullptr, 0, cap);
            same(ka[0], kb[0], std::string(c.name) + " as an indexed key");

            // as a PREDICATE lane, on the key of the expected set
            auto wa = agg.groupby_exact_masked_multi_indexed(
                gpudb::IndexedColumn{EX[0].get(), nullptr}, &kp, 1, 0, &pa, 1, cap);
            auto wb = agg.groupby_exact_masked_multi_indexed(
                gpudb::IndexedColumn{EX[0].get(), nullptr}, &kp, 1, 0, &pb, 1, cap);
            same(wa[0], wb[0], std::string(c.name) + " as an indexed predicate");
        }

        if (srccol->dtype() != DT::I64) continue;
        // as a PAYLOAD of the global masked aggregate (§4.12), the pure
        // row-order operator the indexed read was built for
        gpudb::MultiPayload qa{EX[1].get(), gpudb::GroupByFilter::kAllColumns, nullptr};
        gpudb::MultiPayload qb{srccol, gpudb::GroupByFilter::kAllColumns, index};
        auto la = agg.aggregate_exact_masked(&qa, 1, nullptr, 0);
        auto lb = agg.aggregate_exact_masked(&qb, 1, nullptr, 0);
        ++total;
        if (!(la.sums == lb.sums && la.sums_hi == lb.sums_hi && la.counts == lb.counts &&
              la.mins == lb.mins && la.maxs == lb.maxs && la.count_star == lb.count_star)) {
            ++failures;
            std::printf("    FAIL %s as an indexed payload of the global aggregate\n", c.name);
        }
        // and the same under a WHERE, so the mask and the fold both gather
        auto wla = agg.aggregate_exact_masked(&qa, 1, &pa, 1);
        auto wlb = agg.aggregate_exact_masked(&qb, 1, &pb, 1);
        ++total;
        if (!(wla.sums == wlb.sums && wla.counts == wlb.counts && wla.count_star == wlb.count_star)) {
            ++failures;
            std::printf("    FAIL %s as an indexed payload under an indexed WHERE\n", c.name);
        }
    }

    // An out-of-range index cell must be refused, not read.
    {
        std::vector<std::int64_t> bad(64 * 2, 0);
        for (std::size_t d = 0; d < 64; ++d) bad[d * 2] = static_cast<std::int64_t>(B + d);
        DT bdt2[2] = {DT::I64, DT::I64};
        gpudb::Aggregator::RowSpan sp2;
        sp2.lanes = bad.data(); sp2.rows = 64; sp2.n_lanes = 2; sp2.valid = nullptr;
        auto BAD = agg.upload_rows_exact(&sp2, 1, bdt2, 2);
        gpudb::MultiPayload bp{Bc[1].get(), gpudb::GroupByFilter::kAllColumns, BAD[0].get()};
        bool threw = false;
        try { agg.aggregate_exact_masked(&bp, 1, nullptr, 0); }
        catch (const std::exception&) { threw = true; }
        EXPECT(threw);
    }
}

void test_indexed_lanes_backend(gpudb::Backend backend) {
    using DT = gpudb::Dtype;
    using Op = gpudb::Predicate::Op;
    std::unique_ptr<gpudb::Aggregator> agg;
    try {
        agg = gpudb::make_aggregator(backend);
    } catch (const std::exception& e) {
        std::printf("  %s unavailable (%s)\n", gpudb::to_string(backend), e.what());
        return;
    }
    if (!agg->indexed_supported()) {
        std::printf("  %s does not implement stage D1 — the SQL layer keeps the materialised path\n",
                    gpudb::to_string(backend));
        // The contract for such a backend: an index must be REFUSED, never
        // ignored. A silently dropped index is a wrong answer.
        return;
    }
    std::printf("  %s (%s)\n", gpudb::to_string(backend), agg->device_name().c_str());
    const std::size_t cap = std::size_t(100) * 1000000;

    // ---- the two tables ----
    // probe: F rows of a fact table, its join key pointing into dim1.
    // Lane 0 fk1, 1 fk2, 2 pos (the row's own position), 3 width-1, 4 width-2,
    // 5 width-4, 6 width-8, 7 F64 with specials, 8 a lane with NULLs.
    const std::size_t F = 200'003, B = 1'009, PL = 9;
    const double specials[5] = {std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::infinity(),
                                -std::numeric_limits<double>::infinity(), -0.0, 0.0};
    std::vector<std::int64_t> pf(F * PL);
    std::vector<std::vector<std::uint64_t>> pv(PL, std::vector<std::uint64_t>((F + 63) / 64, ~std::uint64_t{0}));
    for (std::size_t i = 0; i < F; ++i) {
        auto clear = [&](std::size_t l) { pv[l][i >> 6] &= ~(std::uint64_t{1} << (i & 63)); };
        pf[i * PL + 0] = static_cast<std::int64_t>((i * 2654435761ull) % (B + 37));   // some miss
        pf[i * PL + 1] = static_cast<std::int64_t>((i * 40503ull) % B);
        pf[i * PL + 2] = static_cast<std::int64_t>(i);
        pf[i * PL + 3] = static_cast<std::int64_t>(i % 256) - 128;
        pf[i * PL + 4] = static_cast<std::int64_t>(i % 65536) - 32768;
        pf[i * PL + 5] = static_cast<std::int64_t>(i % 4001) * 1'000'000 - 2'000'000'000LL;
        pf[i * PL + 6] = (std::int64_t{1} << 40) + static_cast<std::int64_t>(i % 7919);
        double d = (i % 211 < 5) ? specials[i % 5]
                                 : static_cast<double>(static_cast<std::int64_t>(i % 997)) / 997.0;
        std::memcpy(&pf[i * PL + 7], &d, sizeof(double));
        pf[i * PL + 8] = static_cast<std::int64_t>(i % 89) - 44;
        if (i % 37 == 0) clear(8);
        if (i % 53 == 0) clear(7);
        if (i % 101 == 0) clear(0);     // a NULL join key never matches
    }
    // build: lane 0 the unique key, 1 a width-1 lane, 2 a wide lane with
    // NULLs, 3 an F64 lane, 4 a second unique key for the chain.
    const std::size_t BL = 5;
    std::vector<std::int64_t> bfl(B * BL);
    std::vector<std::vector<std::uint64_t>> bv(BL, std::vector<std::uint64_t>((B + 63) / 64, ~std::uint64_t{0}));
    for (std::size_t i = 0; i < B; ++i) {
        bfl[i * BL + 0] = static_cast<std::int64_t>(i);
        bfl[i * BL + 1] = static_cast<std::int64_t>(i % 200) - 100;
        bfl[i * BL + 2] = (std::int64_t{1} << 55) - static_cast<std::int64_t>(i) * 7919;
        double d = (i % 41 < 5) ? specials[i % 5] : static_cast<double>(i) / 13.0;
        std::memcpy(&bfl[i * BL + 3], &d, sizeof(double));
        bfl[i * BL + 4] = static_cast<std::int64_t>(i);
        if (i % 17 == 0) bv[2][i >> 6] &= ~(std::uint64_t{1} << (i & 63));
    }
    std::vector<const std::uint64_t*> pvp(PL), bvp(BL);
    for (std::size_t l = 0; l < PL; ++l) pvp[l] = pv[l].data();
    for (std::size_t l = 0; l < BL; ++l) bvp[l] = bv[l].data();
    DT pdt[PL]; for (std::size_t l = 0; l < PL; ++l) pdt[l] = (l == 7) ? DT::F64 : DT::I64;
    DT bdt[BL]; for (std::size_t l = 0; l < BL; ++l) bdt[l] = (l == 3) ? DT::F64 : DT::I64;
    gpudb::Aggregator::RowSpan ps, bs;
    ps.lanes = pf.data(); ps.rows = F; ps.n_lanes = PL; ps.valid = pvp.data();
    bs.lanes = bfl.data(); bs.rows = B; bs.n_lanes = BL; bs.valid = bvp.data();
    auto P = agg->upload_rows_exact(&ps, 1, pdt, PL);
    auto Bc = agg->upload_rows_exact(&bs, 1, bdt, BL);

    // ---- the same join, materialised and indexed ----
    // Lane 0 is the classifying key lane, as join_materialize requires.
    std::vector<gpudb::JoinLane> lanes = {
        {P[3].get(), false, nullptr},          // key: width-1 probe lane
        {P[2].get(), false, nullptr},          // pos
        {P[4].get(), false, nullptr}, {P[5].get(), false, nullptr},
        {P[6].get(), false, nullptr}, {P[7].get(), false, nullptr},
        {P[8].get(), false, nullptr},
        {Bc[1].get(), true, nullptr}, {Bc[2].get(), true, nullptr}, {Bc[3].get(), true, nullptr},
    };
    gpudb::JoinMaterializeResult jm;
    gpudb::Aggregator::JoinIndexResult ji;
    bool have_join_index = true;
    try {
        jm = agg->join_materialize(*P[0], *Bc[0], lanes.data(), lanes.size());
    } catch (const std::exception& e) {
        std::printf("    FAIL join_materialize: %s\n", e.what());
        ++failures; ++total;
        return;
    }
    try {
        ji = agg->join_index(*P[0], *Bc[0], lanes[0], nullptr, 0, nullptr);
    } catch (const std::exception& e) {
        // A backend may have the indexed READ without the indexed JOIN yet.
        // That is a legitimate half of stage D1 — the SQL layer then builds
        // the set with join_materialize and indexes nothing — but it must
        // REFUSE the call, never answer it wrongly.
        have_join_index = false;
        std::printf("    join_index is not on this backend yet (%s); the indexed reads are checked "
                    "against host-built index vectors below\n", e.what());
    }
    if (!have_join_index) {
        test_indexed_reads_only(*agg, P, Bc, pf, bfl, PL, BL, F, B, cap);
        return;
    }
    EXPECT_EQ(ji.rows_out, jm.rows_out);
    EXPECT_EQ(ji.null_key_rows, jm.null_key_rows);
    EXPECT_EQ(ji.rows_probe, jm.rows_probe);
    EXPECT_EQ(ji.rows_build, jm.rows_build);
    EXPECT(ji.probe_rows && ji.build_rows);
    EXPECT_EQ(ji.probe_rows->rows(), jm.rows_out);
    EXPECT_EQ(ji.build_rows->rows(), jm.rows_out);
    // Some probe rows miss (fk goes past B) and some have a NULL key, so this
    // join is NOT the identity — the scattered case, which is the one that can
    // be slower and the one that must still be right.
    EXPECT(!ji.probe_identity);
    if (!ji.probe_rows || !ji.build_rows) return;

    // Every lane, materialised vs read through its side's index. The key is
    // the lane under test and the payload is the materialised position lane,
    // so two answers agree only if the lane holds the same value at the same
    // OUTPUT ROW, not merely the same multiset of values.
    auto same = [&](const gpudb::GroupByResidentResult& a, const gpudb::GroupByResidentResult& b,
                    const char* what) {
        const bool ok = a.keys == b.keys && a.key_null == b.key_null &&
                        a.sums == b.sums && a.sums_hi == b.sums_hi &&
                        a.counts == b.counts && a.counts_star == b.counts_star &&
                        a.mins == b.mins && a.maxs == b.maxs;
        ++total;
        if (!ok) {
            ++failures;
            std::printf("    FAIL %s (%zu vs %zu groups)\n", what, a.keys.size(), b.keys.size());
        }
        return ok;
    };
    struct LaneCase { const char* name; std::size_t mat; const gpudb::ResidentColumn* src; bool from_build; };
    const std::vector<LaneCase> cases = {
        {"probe width 1 (the key lane)", 0, P[3].get(), false},
        {"probe width 2",                2, P[4].get(), false},
        {"probe width 4",                3, P[5].get(), false},
        {"probe width 8 (wide key)",     4, P[6].get(), false},
        {"probe f64 nan/inf/-0",         5, P[7].get(), false},
        {"probe lane with NULLs",        6, P[8].get(), false},
        {"build width 1",                7, Bc[1].get(), true},
        {"build wide with NULLs",        8, Bc[2].get(), true},
        {"build f64",                    9, Bc[3].get(), true},
    };
    const gpudb::ResidentColumn* pos_mat = jm.lanes[1].get();
    for (const auto& c : cases) {
        const gpudb::ResidentColumn* index = c.from_build ? ji.build_rows.get() : ji.probe_rows.get();
        // grouped: key = the lane, payload = the materialised position
        gpudb::MultiPayload mp_m{pos_mat, gpudb::GroupByFilter::kAllColumns, nullptr};
        gpudb::MultiPayload mp_i{pos_mat, gpudb::GroupByFilter::kAllColumns, nullptr};
        std::vector<gpudb::GroupByResidentResult> ra, rb;
        // A backend may refuse this KEY's dtype — grouping a backend's way on
        // the raw bits of a double is not the same question as grouping on an
        // integer, and Metal refuses it. What stage D1 requires is that the
        // index changes nothing: the indexed call and the materialised call
        // must give the same answer, and a refusal is an answer only if BOTH
        // refuse. (The F64 lane's values are still under test here, as an
        // indexed predicate below and as a materialised join lane above.)
        std::string em, ei;
        try {
            ra = agg->groupby_exact_masked_multi_indexed(
                gpudb::IndexedColumn{jm.lanes[c.mat].get(), nullptr}, &mp_m, 1, 0, nullptr, 0, cap);
        } catch (const std::exception& e) { em = e.what(); }
        try {
            rb = agg->groupby_exact_masked_multi_indexed(
                gpudb::IndexedColumn{c.src, index}, &mp_i, 1, 0, nullptr, 0, cap);
        } catch (const std::exception& e) { ei = e.what(); }
        if (!em.empty() || !ei.empty()) {
            ++total;
            if (em.empty() || ei.empty()) {
                ++failures;
                std::printf("    FAIL %s: one form refused and the other did not (%s | %s)\n",
                            c.name, em.empty() ? "answered" : em.c_str(),
                            ei.empty() ? "answered" : ei.c_str());
            }
        } else {
            same(ra[0], rb[0], c.name);
        }

        // the same lane as an indexed PAYLOAD (f64 lanes are not payloads)
        if (c.src->dtype() == DT::I64) {
            gpudb::MultiPayload qa{jm.lanes[c.mat].get(), gpudb::GroupByFilter::kAllColumns, nullptr};
            gpudb::MultiPayload qb{c.src, gpudb::GroupByFilter::kAllColumns, index};
            auto ga = agg->groupby_exact_masked_multi_indexed(
                gpudb::IndexedColumn{jm.lanes[0].get(), nullptr}, &qa, 1, 0, nullptr, 0, cap);
            auto gb = agg->groupby_exact_masked_multi_indexed(
                gpudb::IndexedColumn{jm.lanes[0].get(), nullptr}, &qb, 1, 0, nullptr, 0, cap);
            same(ga[0], gb[0], (std::string(c.name) + " as a payload").c_str());

            // and as an indexed lane of the GLOBAL masked aggregate (§4.12)
            auto la = agg->aggregate_exact_masked(&qa, 1, nullptr, 0);
            auto lb = agg->aggregate_exact_masked(&qb, 1, nullptr, 0);
            ++total;
            if (!(la.sums == lb.sums && la.sums_hi == lb.sums_hi && la.counts == lb.counts &&
                  la.mins == lb.mins && la.maxs == lb.maxs && la.count_star == lb.count_star)) {
                ++failures;
                std::printf("    FAIL %s in the global aggregate\n", c.name);
            }
        }

        // and as an indexed PREDICATE lane, under a WHERE that keeps a slice
        gpudb::Predicate pa{}, pb{};
        pa.col = jm.lanes[c.mat].get(); pa.op = Op::IsNotNull;
        pb.col = c.src; pb.op = Op::IsNotNull; pb.index = index;
        gpudb::MultiPayload wp{pos_mat, gpudb::GroupByFilter::kAllColumns, nullptr};
        auto wa = agg->groupby_exact_masked_multi_indexed(
            gpudb::IndexedColumn{jm.lanes[0].get(), nullptr}, &wp, 1, 0, &pa, 1, cap);
        auto wb = agg->groupby_exact_masked_multi_indexed(
            gpudb::IndexedColumn{jm.lanes[0].get(), nullptr}, &wp, 1, 0, &pb, 1, cap);
        same(wa[0], wb[0], (std::string(c.name) + " as a predicate").c_str());
    }

    // ---- an index with NULL cells: the unmatched side of an outer join ----
    // invariant 6 — every lane read through a NULL index cell is NULL. The
    // expected column is built on the host and uploaded, so the check does not
    // depend on any other part of stage D.
    {
        const std::size_t R = 40'009;
        std::vector<std::int64_t> ixf(R), expf(R * 2);
        std::vector<std::uint64_t> ixv((R + 63) / 64, ~std::uint64_t{0});
        std::vector<std::uint64_t> ev0((R + 63) / 64, ~std::uint64_t{0});
        std::vector<std::uint64_t> ev1((R + 63) / 64, ~std::uint64_t{0});
        for (std::size_t d = 0; d < R; ++d) {
            const std::size_t srow = (d * 7919) % B;
            ixf[d] = static_cast<std::int64_t>(srow);
            expf[d * 2 + 0] = static_cast<std::int64_t>(d % 1000);            // the key, never NULL
            const bool idx_null = (d % 11 == 0);
            const bool cell_null = ((srow % 17) == 0);                        // build lane 2's NULLs
            if (idx_null) ixv[d >> 6] &= ~(std::uint64_t{1} << (d & 63));
            if (idx_null || cell_null) { ev1[d >> 6] &= ~(std::uint64_t{1} << (d & 63)); expf[d * 2 + 1] = 0; }
            else expf[d * 2 + 1] = bfl[srow * BL + 2];
        }
        std::vector<std::uint64_t> iv2((R + 63) / 64, ~std::uint64_t{0});
        const std::uint64_t* ixvp[2] = {ixv.data(), iv2.data()};
        std::vector<std::int64_t> ix2(R * 2);
        for (std::size_t d = 0; d < R; ++d) { ix2[d * 2] = ixf[d]; ix2[d * 2 + 1] = 0; }
        DT idt[2] = {DT::I64, DT::I64};
        gpudb::Aggregator::RowSpan is;
        is.lanes = ix2.data(); is.rows = R; is.n_lanes = 2; is.valid = ixvp;
        auto IX = agg->upload_rows_exact(&is, 1, idt, 2);
        const std::uint64_t* evp[2] = {ev0.data(), ev1.data()};
        gpudb::Aggregator::RowSpan es;
        es.lanes = expf.data(); es.rows = R; es.n_lanes = 2; es.valid = evp;
        auto EX = agg->upload_rows_exact(&es, 1, idt, 2);

        gpudb::MultiPayload ea{EX[1].get(), gpudb::GroupByFilter::kAllColumns, nullptr};
        gpudb::MultiPayload eb{Bc[2].get(), gpudb::GroupByFilter::kAllColumns, IX[0].get()};
        auto oa = agg->groupby_exact_masked_multi_indexed(
            gpudb::IndexedColumn{EX[0].get(), nullptr}, &ea, 1, 0, nullptr, 0, cap);
        auto ob = agg->groupby_exact_masked_multi_indexed(
            gpudb::IndexedColumn{EX[0].get(), nullptr}, &eb, 1, 0, nullptr, 0, cap);
        same(oa[0], ob[0], "a NULL index cell makes the lane NULL");
        // the same, with the indexed lane as the KEY: a NULL index cell puts
        // the row in the NULL-key group
        gpudb::MultiPayload kp{EX[0].get(), gpudb::GroupByFilter::kAllColumns, nullptr};
        auto ka = agg->groupby_exact_masked_multi_indexed(
            gpudb::IndexedColumn{EX[1].get(), nullptr}, &kp, 1, 0, nullptr, 0, cap);
        auto kb = agg->groupby_exact_masked_multi_indexed(
            gpudb::IndexedColumn{Bc[2].get(), IX[0].get()}, &kp, 1, 0, nullptr, 0, cap);
        same(ka[0], kb[0], "a NULL index cell makes the KEY NULL");
    }

    // ---- a chained three-step join ----
    // Step k's probe key is read through the index step k-1 produced, and the
    // earlier index vectors are composed by handing them to the next step as
    // MATERIALISED lanes: gathering idx_old at probe_rows IS the composition.
    {
        gpudb::JoinLane key1{P[3].get(), false, nullptr};
        auto s1 = agg->join_index(*P[0], *Bc[0], key1, nullptr, 0, nullptr);
        // step 2 probes on the fact table's second fk, read through step 1's
        // probe_rows; it carries step 1's build_rows forward as a lane.
        gpudb::JoinLane carry{s1.build_rows.get(), false, nullptr};
        gpudb::JoinLane key2{P[3].get(), false, s1.probe_rows.get()};
        auto s2 = agg->join_index(*P[1], *Bc[0], key2, &carry, 1, s1.probe_rows.get());
        // step 3 on the dimension's second unique key
        gpudb::JoinLane carry3[2] = {{s2.lanes[0].get(), false, nullptr},
                                     {s2.build_rows.get(), false, nullptr}};
        gpudb::JoinLane key3{P[3].get(), false, nullptr};
        (void)key3;
        // The composed probe index of the chain so far:
        //   chain[d] = probe_rows_1[ probe_rows_2[d] ]  — one gather per step.
        gpudb::JoinLane comp{s1.probe_rows.get(), false, nullptr};
        auto s3 = agg->join_index(*P[1], *Bc[4], gpudb::JoinLane{P[3].get(), false, s2.probe_rows.get()},
                                  &comp, 1, s2.probe_rows.get());
        EXPECT(s3.rows_out > 0);
        EXPECT_EQ(s3.lanes.size(), std::size_t{1});
        // The chain's answer must equal the same three joins materialised.
        std::vector<gpudb::JoinLane> m1 = {{P[3].get(), false, nullptr},
                                           {P[1].get(), false, nullptr},
                                           {P[2].get(), false, nullptr},
                                           {Bc[1].get(), true, nullptr}};
        auto j1 = agg->join_materialize(*P[0], *Bc[0], m1.data(), m1.size());
        std::vector<gpudb::JoinLane> m2 = {{j1.lanes[0].get(), false, nullptr},
                                           {j1.lanes[2].get(), false, nullptr},
                                           {j1.lanes[3].get(), false, nullptr},
                                           {Bc[2].get(), true, nullptr}};
        auto j2 = agg->join_materialize(*j1.lanes[1], *Bc[0], m2.data(), m2.size());
        EXPECT_EQ(s2.rows_out, j2.rows_out);
        // lane for lane: the carried build index against the materialised
        // build lane it stands for
        gpudb::MultiPayload ca{j2.lanes[2].get(), gpudb::GroupByFilter::kAllColumns, nullptr};
        gpudb::MultiPayload cb{Bc[1].get(), gpudb::GroupByFilter::kAllColumns, s2.lanes[0].get()};
        auto xa = agg->groupby_exact_masked_multi_indexed(
            gpudb::IndexedColumn{j2.lanes[0].get(), nullptr}, &ca, 1, 0, nullptr, 0, cap);
        auto xb = agg->groupby_exact_masked_multi_indexed(
            gpudb::IndexedColumn{j2.lanes[0].get(), nullptr}, &cb, 1, 0, nullptr, 0, cap);
        same(xa[0], xb[0], "a chained join's composed index");
    }

    // ---- an index vector the planner got wrong must be refused, not read ----
    {
        const std::size_t R = 64;
        std::vector<std::int64_t> bad(R * 2, 0);
        for (std::size_t d = 0; d < R; ++d) bad[d * 2] = static_cast<std::int64_t>(B + d);  // out of range
        DT bdt2[2] = {DT::I64, DT::I64};
        gpudb::Aggregator::RowSpan sp2;
        sp2.lanes = bad.data(); sp2.rows = R; sp2.n_lanes = 2; sp2.valid = nullptr;
        auto BAD = agg->upload_rows_exact(&sp2, 1, bdt2, 2);
        gpudb::MultiPayload bp{Bc[1].get(), gpudb::GroupByFilter::kAllColumns, BAD[0].get()};
        bool threw = false;
        try {
            agg->groupby_exact_masked_multi_indexed(
                gpudb::IndexedColumn{BAD[1].get(), nullptr}, &bp, 1, 0, nullptr, 0, cap);
        } catch (const std::exception&) { threw = true; }
        EXPECT(threw);
    }
}

void test_indexed_lanes() {
    std::printf("\n--- stage D1: a lane read through an index vs the gathered copy ---\n");
    for (auto b : gpudb::available_backends()) {
        try {
            test_indexed_lanes_backend(b);
        } catch (const std::exception& e) {
            ++total; ++failures;
            std::printf("  FAIL %s: %s\n", gpudb::to_string(b), e.what());
        }
    }
}

// ---------------------------------------------------------------------------
// avg() is finalised the way DuckDB finalises it (src/include/native_avg.hpp).
// No DuckDB needed, so this runs in CI's Linux job, which has no libduckdb and
// therefore never ran the SQL parity tests that caught the bug in the first
// place: the extension derived avg as double(sum) / double(count), which
// reproduces native only where long double IS double. On x86-64 that is one
// ulp out on ~28% of groups (856 of 3002 on the 300k-row set in
// test/sql/gpu_groupby_exact.test).
//
// Two things are asserted. First, native_avg equals an INDEPENDENTLY written
// long double quotient for every case — that is the contract. Second, on a
// platform whose long double is wider than double, at least one crafted case
// must differ from the plain double quotient: without that, a silent
// regression to the double form would still pass, and the test would be
// asserting nothing. Where long double IS double (Apple silicon) the second
// assertion is skipped and said so, because there the two forms agree and
// there is nothing to catch.
// ---------------------------------------------------------------------------
void test_native_avg() {
    std::printf("\n--- avg finalised as DuckDB finalises it ---\n");
    constexpr bool wide = LDBL_MANT_DIG > DBL_MANT_DIG;
    std::printf("  long double mantissa %d bits, double %d bits -> %s\n",
                LDBL_MANT_DIG, DBL_MANT_DIG,
                wide ? "the double form is detectably wrong here"
                     : "the two forms coincide on this platform");

    std::mt19937_64 rng(0xAF6ULL);
    std::size_t differ_from_double = 0, checked = 0, mismatches = 0;
    for (int i = 0; i < 20000; ++i) {
        // Sums that need both limbs, and counts that rarely divide evenly.
        gpudb::Sum128 s;
        s.lo = rng();
        s.hi = static_cast<std::int64_t>(rng() >> 40) - (1 << 23);
        const std::int64_t cnt = static_cast<std::int64_t>(rng() % 100000) + 1;

        // The contract: the same expression DuckDB evaluates, in its type.
        const long double ld_sum = (s.hi == -1)
            ? -static_cast<long double>(~std::uint64_t{0} - s.lo) - 1.0L
            : static_cast<long double>(s.lo) +
              static_cast<long double>(s.hi) * 18446744073709551616.0L;
        const double want = static_cast<double>(ld_sum / static_cast<long double>(cnt));
        const double got  = gpudb::native_avg(s, cnt);
        if (std::memcmp(&want, &got, sizeof(double)) != 0) ++mismatches;
        ++checked;

        const double as_double = s.to_double() / static_cast<double>(cnt);
        if (std::memcmp(&as_double, &got, sizeof(double)) != 0) ++differ_from_double;
    }
    EXPECT(mismatches == 0);   // one check for the whole sweep, not 20000
    std::printf("  %zu cases, %zu mismatches; the double form differs on %zu of them\n",
                checked, mismatches, differ_from_double);
    if (wide) {
        EXPECT(differ_from_double > 0);   // otherwise this test proves nothing
    } else {
        EXPECT(differ_from_double == 0);  // on this platform the two MUST agree
    }

    // The DECIMAL divident is count * 10^scale, formed in the wide type.
    {
        gpudb::Sum128 s; s.lo = 123456789012345678ULL; s.hi = 0;
        const double got = gpudb::native_avg_decimal(s, 7, 2);
        const long double want_ld = static_cast<long double>(s.lo) /
                                    (static_cast<long double>(7) * 100.0L);
        const double want = static_cast<double>(want_ld);
        EXPECT(std::memcmp(&want, &got, sizeof(double)) == 0);
    }
    std::printf("    ok\n");
}

void test_resident_prepare() {
    std::printf("\n--- ResidentColumn::prepare / concurrent upload ---\n");
    auto h = gpudb::make_hybrid_aggregator();
    // The derived-structure expectations below (nothing prepared before
    // prepare(), 3× bytes after) are the CUDA sort cache's; Metal keeps the
    // header defaults until its perm cache moves behind prepare(), and the
    // CPU backend has nothing to derive. The contract every backend must
    // meet — prepared() after prepare(), idempotent, answers unchanged,
    // thread-safe — is checked on all of them.
    const bool on_gpu = h->gpu_backend() == gpudb::Backend::CUDA;
    std::printf("  gpu_backend: %s\n", gpudb::to_string(h->gpu_backend()));

    std::mt19937_64 rng(0x9E7ULL);
    const std::size_t N = 2'000'003;
    std::vector<std::int64_t> keys(N), vals(N);
    std::uniform_int_distribution<std::int64_t> kd(-50'000, 50'000), vd(-1'000, 1'000);
    for (std::size_t i = 0; i < N; ++i) { keys[i] = kd(rng); vals[i] = vd(rng); }
    const std::size_t cap = std::size_t(100) * 1000000;

    // Lazy path (v0.6 behaviour) is the reference.
    auto k_lazy = h->upload_i64(keys.data(), N);
    auto v_lazy = h->upload_i64(vals.data(), N);
    auto ref = h->groupby_sum_resident_i64(*k_lazy, *v_lazy, cap);

    // Eager path: prepared before any operator ran.
    auto k = h->upload_i64(keys.data(), N);
    auto v = h->upload_i64(vals.data(), N);
    if (on_gpu) {
        EXPECT(!k->prepared());                       // CUDA: nothing derived yet
        EXPECT_EQ(k->resident_bytes(), N * 8);
    }
    k->prepare();
    EXPECT(k->prepared());
    if (on_gpu) EXPECT_EQ(k->resident_bytes(), N * 8 * 3);   // + sorted copy + perm
    const std::size_t bytes_after = k->resident_bytes();
    k->prepare();                                      // idempotent
    EXPECT_EQ(k->resident_bytes(), bytes_after);

    auto got = h->groupby_sum_resident_i64(*k, *v, cap);
    EXPECT_EQ(got.keys.size(), ref.keys.size());
    bool same = got.keys == ref.keys && got.sums == ref.sums && got.counts == ref.counts;
    EXPECT(same);

    // Two threads preparing the SAME column race on the per-column lock:
    // exactly one build, both observe prepared, no throw.
    {
        auto c = h->upload_i64(keys.data(), N);
        std::atomic<int> failures{0};
        auto worker = [&] {
            try { c->prepare(); if (!c->prepared()) failures++; }
            catch (const std::exception&) { failures++; }
        };
        std::thread t1(worker), t2(worker), t3(worker);
        t1.join(); t2.join(); t3.join();
        EXPECT_EQ(failures.load(), 0);
        EXPECT_EQ(c->resident_bytes(), bytes_after);
    }

    // Uploads + prepares on four threads while this thread runs operators
    // (serialized, as the extension's device lock does): the operator
    // answers never change and nothing throws. This is the shape of the
    // background upload the wrapper drives on a second connection.
    {
        std::atomic<int> failures{0};
        std::atomic<bool> stop{false};
        const std::size_t M = 400'009;
        std::vector<std::thread> uploaders;
        for (int t = 0; t < 4; ++t) {
            uploaders.emplace_back([&, t] {
                std::mt19937_64 r(0xABC + t);
                std::vector<std::int64_t> kk(M), vv(M);
                std::uniform_int_distribution<std::int64_t> d(-999, 999);
                for (std::size_t i = 0; i < M; ++i) { kk[i] = d(r); vv[i] = d(r); }
                for (int it = 0; it < 6 && !stop.load(); ++it) {
                    try {
                        auto ck = h->upload_i64(kk.data(), M);
                        auto cv = h->upload_i64(vv.data(), M);
                        ck->prepare();
                        if (!ck->prepared()) failures++;
                    } catch (const std::exception& e) {
                        std::printf("    uploader %d: %s\n", t, e.what());
                        failures++;
                    }
                }
            });
        }
        int ops = 0;
        bool ok = true;
        for (int i = 0; i < 40 && ok; ++i) {
            try {
                auto g = h->groupby_sum_resident_i64(*k, *v, cap);
                ok = g.keys == ref.keys && g.sums == ref.sums && g.counts == ref.counts;
                auto tk = h->topk_resident(*v, 10, true);
                ok = ok && tk.idx.size() == 10;
                ++ops;
            } catch (const std::exception& e) {
                std::printf("    operator: %s\n", e.what());
                ok = false;
            }
        }
        stop.store(true);
        for (auto& th : uploaders) th.join();
        EXPECT(ok);
        EXPECT_EQ(failures.load(), 0);
        std::printf("  %d operator calls during 4 concurrent upload+prepare threads: %s\n",
                    ops, ok ? "stable" : "MISMATCH");
    }
    std::printf("  ok\n");
}

int main(int argc, char** argv) {
    // CI reads this through a pipe, where stdout is block-buffered: a crash
    // then loses every line since the last flush and the log points at the
    // wrong place. Line buffering costs nothing here and makes the last line
    // printed the last line that ran.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
#if GPUDB_HAVE_CUDA && defined(__linux__)
    if (argc > 1 && std::string(argv[1]) == "--cuda-fault-child") return cuda_fault_child();
#else
    (void)argc; (void)argv;
#endif
    std::printf("gpudb test suite\n");
    std::printf("available backends:");
    for (auto b : gpudb::available_backends()) std::printf(" %s", gpudb::to_string(b));
    std::printf("\ndefault backend: %s\n", gpudb::to_string(gpudb::default_backend()));

    test_backend(gpudb::Backend::CPU);

#if GPUDB_HAVE_CUDA
    test_backend(gpudb::Backend::CUDA);
    test_cuda_failed_upload_leaves_nothing();
    test_cuda_direct_reduce_matches_reference();
#endif
#if GPUDB_HAVE_METAL
    test_backend(gpudb::Backend::METAL);
    // Small top-k results take a host pass by default; run the block again
    // with the device radix select forced so both paths stay covered.
    std::printf("\n(Metal again, GPUDB_METAL_HOST_FILTER_BELOW=0: device top-k at every size)\n");
    setenv("GPUDB_METAL_HOST_FILTER_BELOW", "0", 1);
    test_backend(gpudb::Backend::METAL);
    unsetenv("GPUDB_METAL_HOST_FILTER_BELOW");
#endif

    test_hybrid_aggregator();
    test_hybrid_groupby();
#if GPUDB_HAVE_METAL
    test_direct_groupby();
    // before test_direct_pso_fallback: that block ends by unsetting
    // GPUDB_METAL_DIRECT_DISABLE_PSO, and this one reads it
    test_shed_derived();
    test_fused_mask();
    test_direct_pso_fallback();
#endif
    test_indexed_lanes();
    test_native_avg();
    test_resident_prepare();
    test_hashjoin();

#if GPUDB_HAVE_CUDA && defined(__linux__)
    test_cuda_device_fault_is_an_error();
#endif

#if GPUDB_HAVE_METAL
    // Regression: the non-resident Metal GROUP BY's radix path (expected
    // groups < 1024) once skipped byte passes whenever min and max agreed
    // on that byte. Keys uniform in [0, 257): min=0 and max=256 share the
    // low byte, every key in between differs there.
    {
        std::printf("\n--- Metal non-resident GROUP BY radix path (keys 0..256) ---\n");
        try {
        const std::size_t N = 200'000;
        std::vector<std::int64_t> k(N), v(N);
        std::map<std::int64_t, std::int64_t> ref;
        for (std::size_t i = 0; i < N; ++i) {
            k[i] = static_cast<std::int64_t>((i * 2654435761ull) % 257);
            v[i] = static_cast<std::int64_t>(i % 1000) - 500;
            ref[k[i]] += v[i];
        }
        auto gb = gpudb::make_groupby_aggregator(gpudb::Backend::METAL);
        auto r = gb->groupby_sum_i64(k.data(), v.data(), N, /*expected_groups*/257);
        EXPECT_EQ(r.keys.size(), ref.size());
        bool ok = r.keys.size() == ref.size();
        for (std::size_t i = 0; ok && i < r.keys.size(); ++i) {
            auto it = ref.find(r.keys[i]);
            ok = it != ref.end() && it->second == r.sums[i];
        }
        EXPECT(ok);
        } catch (const std::exception& e) {
            // A backend whose pipelines will not build is a device we cannot
            // use, not a wrong answer — the hash-join block has skipped on
            // this for as long as it has existed.
            std::printf("  skipped (%s)\n", e.what());
        }
    }
#endif

    failures += test_hashjoin_failures();
    total += test_hashjoin_total();

    std::printf("\n%d / %d checks passed\n", total - failures, total);
    return failures == 0 ? 0 : 1;
}
