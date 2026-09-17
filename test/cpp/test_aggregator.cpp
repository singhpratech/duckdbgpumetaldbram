// Minimal hand-rolled test runner — avoids pulling in Catch2 for week 1.
// Returns nonzero on failure so `ctest` and CI can pick it up.

#include "gpu_backend.hpp"
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

#if GPUDB_HAVE_CUDA && defined(__linux__)
extern "C" int gpudb_cuda_debug_fault_inject(void* stream);   // cudaError_t; 0 == success

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
#endif


// ---------------------------------------------------------------------------
// ResidentColumn::prepare() (v0.7 milestone 0b, docs/TRANSPARENT_DESIGN.md
// §5.5/§5.6): ready = uploaded AND prepared; prepare is idempotent, safe to
// call concurrently with itself and with uploads on other threads, and
// changes no answer. Runs through the hybrid aggregator so the CPU-only
// build exercises the defaults (nothing to prepare) and the CUDA build
// exercises the real sort cache.
// ---------------------------------------------------------------------------
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
    }
#endif

    failures += test_hashjoin_failures();
    total += test_hashjoin_total();

    std::printf("\n%d / %d checks passed\n", total - failures, total);
    return failures == 0 ? 0 : 1;
}
