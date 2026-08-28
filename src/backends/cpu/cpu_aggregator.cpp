#include "gpu_backend.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if GPUDB_HAVE_OPENMP
#include <omp.h>
#endif

namespace gpudb {
namespace {

// Portable parallel chunking for builds without OpenMP (Apple clang ships no
// libomp, so the macOS loadable extension otherwise reduces single-threaded).
// Splits [0,n) into one contiguous chunk per worker and combines the partials
// in chunk order — deterministic for a given (n, worker count), which matters
// for f64 sums where the combine order changes the rounding.
template <class Partial, class ChunkFn, class CombineFn>
Partial parallel_chunks(std::size_t n, ChunkFn chunk, CombineFn combine) {
    constexpr std::size_t kMinPerWorker = std::size_t(1) << 19; // 512k elems
    const unsigned hw = std::thread::hardware_concurrency();
    std::size_t workers = std::min<std::size_t>(hw ? hw : 1, n / kMinPerWorker);
    if (workers <= 1) return chunk(std::size_t(0), n);

    std::vector<Partial> parts(workers);
    std::vector<std::thread> threads;
    threads.reserve(workers);
    const std::size_t per = n / workers;
    for (std::size_t w = 0; w < workers; ++w) {
        const std::size_t begin = w * per;
        const std::size_t end   = (w + 1 == workers) ? n : begin + per;
        threads.emplace_back([&parts, &chunk, w, begin, end] {
            parts[w] = chunk(begin, end);
        });
    }
    for (auto& t : threads) t.join();
    Partial acc = parts[0];
    for (std::size_t w = 1; w < workers; ++w) acc = combine(acc, parts[w]);
    return acc;
}

// CPU resident column = a copy of the host data the aggregator owns.
// (Could just hold a const pointer, but a copy matches GPU semantics so
// the bench numbers reflect "queries against backend-owned data".)
class CpuResidentColumn final : public ResidentColumn {
public:
    CpuResidentColumn(const void* src, std::size_t n, Dtype dt)
        : rows_(n), dtype_(dt) {
        const std::size_t elem = (dt == Dtype::I64) ? sizeof(std::int64_t) : sizeof(double);
        buf_.assign(static_cast<const std::byte*>(src),
                    static_cast<const std::byte*>(src) + n * elem);
    }
    Backend     backend_tag() const noexcept override { return Backend::CPU; }
    Dtype       dtype()       const noexcept override { return dtype_; }
    std::size_t rows()        const noexcept override { return rows_; }

    [[nodiscard]] const std::int64_t* as_i64() const {
        return reinterpret_cast<const std::int64_t*>(buf_.data());
    }
    [[nodiscard]] const double* as_f64() const {
        return reinterpret_cast<const double*>(buf_.data());
    }

private:
    std::vector<std::byte> buf_;
    std::size_t rows_;
    Dtype       dtype_;
};

class CpuAggregator final : public Aggregator {
public:
    Backend backend() const noexcept override { return Backend::CPU; }

    std::string device_name() const override {
#if GPUDB_HAVE_OPENMP
        return "CPU (OpenMP, " + std::to_string(omp_get_max_threads()) + " threads)";
#else
        return "CPU (scalar)";
#endif
    }

    AggResult sum_i64(const std::int64_t* data, std::size_t n) override {
        return run_i64(data, n, ReduceKind::Sum, 0);
    }
    AggResult min_i64(const std::int64_t* data, std::size_t n) override {
        return run_i64(data, n, ReduceKind::Min, std::numeric_limits<std::int64_t>::max());
    }
    AggResult max_i64(const std::int64_t* data, std::size_t n) override {
        return run_i64(data, n, ReduceKind::Max, std::numeric_limits<std::int64_t>::min());
    }
    AggResult sum_f64(const double* data, std::size_t n) override {
        return run_f64_sum(data, n);
    }

    std::unique_ptr<ResidentColumn> upload_i64(const std::int64_t* d, std::size_t n) override {
        return std::make_unique<CpuResidentColumn>(d, n, Dtype::I64);
    }
    std::unique_ptr<ResidentColumn> upload_f64(const double* d, std::size_t n) override {
        return std::make_unique<CpuResidentColumn>(d, n, Dtype::F64);
    }

    AggResult sum_resident_i64(const ResidentColumn& c) override {
        const auto& r = check_i64(c);
        return run_i64(r.as_i64(), r.rows(), ReduceKind::Sum, 0);
    }
    AggResult min_resident_i64(const ResidentColumn& c) override {
        const auto& r = check_i64(c);
        return run_i64(r.as_i64(), r.rows(), ReduceKind::Min,
                       std::numeric_limits<std::int64_t>::max());
    }
    AggResult max_resident_i64(const ResidentColumn& c) override {
        const auto& r = check_i64(c);
        return run_i64(r.as_i64(), r.rows(), ReduceKind::Max,
                       std::numeric_limits<std::int64_t>::min());
    }
    AggResult sum_resident_f64(const ResidentColumn& c) override {
        const auto& r = check_f64(c);
        return run_f64_sum(r.as_f64(), r.rows());
    }

    AggAllResult agg_all_i64(const std::int64_t* data, std::size_t n) override {
        return run_agg_all_i64(data, n);
    }
    AggAllResult agg_all_resident_i64(const ResidentColumn& c) override {
        const auto& r = check_i64(c);
        return run_agg_all_i64(r.as_i64(), r.rows());
    }

    // Same algorithm as the Metal path (sorted build keys + per-probe-element
    // binary search) so the two backends are directly comparable and produce
    // identical results: sum accumulates in uint64 for defined wrap.
    // Per-JoinKind contribution multiplier (see the table in gpu_backend.hpp).
    static std::uint64_t join_multiplier(std::uint64_t m, JoinKind kind) {
        switch (kind) {
            case JoinKind::INNER: return m;
            case JoinKind::LEFT:  return m ? m : 1;
            case JoinKind::SEMI:  return m ? 1 : 0;
            case JoinKind::ANTI:  return m ? 0 : 1;
        }
        return 0;
    }

    JoinAggResult join_sum_resident_i64(const ResidentColumn& probe_keys,
                                        const ResidentColumn& payload,
                                        const ResidentColumn& build_keys,
                                        JoinKind kind) override {
        const auto t0 = std::chrono::steady_clock::now();
        const auto& pk = check_i64(probe_keys);
        const auto& pl = check_i64(payload);
        const auto& bk = check_i64(build_keys);
        if (pk.rows() != pl.rows())
            throw std::runtime_error(
                "join_sum_resident_i64: probe_keys and payload row counts differ");

        JoinAggResult r{};
        r.rows_probe = pk.rows();
        r.rows_build = bk.rows();
        if (pk.rows() == 0 ||
            (bk.rows() == 0 && kind == JoinKind::INNER) ||
            (bk.rows() == 0 && kind == JoinKind::SEMI)) {
            r.wall_ms = elapsed_ms(t0);
            return r;
        }

        std::vector<std::int64_t> sorted(bk.as_i64(), bk.as_i64() + bk.rows());
        std::sort(sorted.begin(), sorted.end());

        const std::int64_t* keys = pk.as_i64();
        const std::int64_t* pay  = pl.as_i64();
        struct Part { std::uint64_t sum = 0; std::int64_t matched = 0; };
        Part total = parallel_chunks<Part>(
            pk.rows(),
            [&](std::size_t begin, std::size_t end) {
                Part p;
                for (std::size_t i = begin; i < end; ++i) {
                    auto [lo, hi] = std::equal_range(sorted.begin(), sorted.end(), keys[i]);
                    const std::uint64_t c =
                        join_multiplier(static_cast<std::uint64_t>(hi - lo), kind);
                    if (c) {
                        p.sum += c * static_cast<std::uint64_t>(pay[i]);
                        p.matched += static_cast<std::int64_t>(c);
                    }
                }
                return p;
            },
            [](Part a, const Part& b) {
                a.sum += b.sum; a.matched += b.matched; return a;
            });

        r.sum     = static_cast<std::int64_t>(total.sum);
        r.matched = total.matched;
        r.wall_ms = elapsed_ms(t0);
        return r;
    }

    JoinAggResult join_sum_resident_f64(const ResidentColumn& probe_keys,
                                        const ResidentColumn& payload,
                                        const ResidentColumn& build_keys,
                                        JoinKind kind) override {
        const auto t0 = std::chrono::steady_clock::now();
        const auto& pk = check_i64(probe_keys);
        const auto& pl = check_f64(payload);
        const auto& bk = check_i64(build_keys);
        if (pk.rows() != pl.rows())
            throw std::runtime_error(
                "join_sum_resident_f64: probe_keys and payload row counts differ");

        JoinAggResult r{};
        r.rows_probe = pk.rows();
        r.rows_build = bk.rows();
        if (pk.rows() == 0) {
            r.wall_ms = elapsed_ms(t0);
            return r;
        }

        std::vector<std::int64_t> sorted(bk.as_i64(), bk.as_i64() + bk.rows());
        std::sort(sorted.begin(), sorted.end());

        const std::int64_t* keys = pk.as_i64();
        const double*       pay  = pl.as_f64();
        struct Part { double sum = 0.0; std::int64_t matched = 0; };
        Part total = parallel_chunks<Part>(
            pk.rows(),
            [&](std::size_t begin, std::size_t end) {
                Part p;
                for (std::size_t i = begin; i < end; ++i) {
                    auto [lo, hi] = std::equal_range(sorted.begin(), sorted.end(), keys[i]);
                    const std::uint64_t c =
                        join_multiplier(static_cast<std::uint64_t>(hi - lo), kind);
                    if (c) {
                        p.sum += static_cast<double>(c) * pay[i];
                        p.matched += static_cast<std::int64_t>(c);
                    }
                }
                return p;
            },
            [](Part a, const Part& b) {
                a.sum += b.sum; a.matched += b.matched; return a;
            });

        r.sum_f64 = total.sum;
        r.matched = total.matched;
        r.wall_ms = elapsed_ms(t0);
        return r;
    }

    JoinRowsResult join_rows_resident(const ResidentColumn& probe_keys,
                                      const ResidentColumn& build_keys,
                                      JoinKind kind, std::size_t max_rows) override {
        const auto t0 = std::chrono::steady_clock::now();
        const auto& pk = check_i64(probe_keys);
        const auto& bk = check_i64(build_keys);

        JoinRowsResult r{};
        r.rows_probe = pk.rows();
        r.rows_build = bk.rows();
        const std::size_t n_probe = pk.rows();
        const std::size_t n_build = bk.rows();
        if (n_probe == 0) { r.wall_ms = elapsed_ms(t0); return r; }

        // Sort (key, original index) pairs so emitted build indices refer to
        // upload order — the same contract as the Metal perm cache.
        std::vector<std::pair<std::int64_t, std::int64_t>> sorted(n_build);
        for (std::size_t j = 0; j < n_build; ++j)
            sorted[j] = { bk.as_i64()[j], static_cast<std::int64_t>(j) };
        std::sort(sorted.begin(), sorted.end());

        const std::int64_t* keys = pk.as_i64();
        auto run_of = [&](std::int64_t k) {
            auto lo = std::lower_bound(sorted.begin(), sorted.end(),
                                       std::make_pair(k, std::numeric_limits<std::int64_t>::min()));
            auto hi = std::upper_bound(sorted.begin(), sorted.end(),
                                       std::make_pair(k, std::numeric_limits<std::int64_t>::max()));
            return std::make_pair(lo, hi);
        };

        // Pass 1: per-row output count (the JoinKind multiplier).
        std::size_t total = 0;
        std::vector<std::uint32_t> cnt(n_probe);
        for (std::size_t i = 0; i < n_probe; ++i) {
            auto [lo, hi] = run_of(keys[i]);
            const std::uint64_t m = static_cast<std::uint64_t>(hi - lo);
            const std::uint64_t c = join_multiplier(m, kind);
            cnt[i] = static_cast<std::uint32_t>(c);
            total += c;
        }
        if (total > max_rows)
            throw std::runtime_error(
                "join_rows_resident: result has " + std::to_string(total) +
                " rows, above the cap of " + std::to_string(max_rows) +
                " (raise GPUDB_JOIN_ROWS_MAX_M if intentional)");

        // Pass 2: fill.
        r.probe_idx.resize(total);
        r.build_idx.resize(total);
        std::size_t off = 0;
        for (std::size_t i = 0; i < n_probe; ++i) {
            if (!cnt[i]) continue;
            auto [lo, hi] = run_of(keys[i]);
            const bool has_match = lo != hi;
            if ((kind == JoinKind::INNER || kind == JoinKind::LEFT) && has_match) {
                for (auto it = lo; it != hi; ++it) {
                    r.probe_idx[off] = static_cast<std::int64_t>(i);
                    r.build_idx[off] = it->second;
                    ++off;
                }
            } else {
                // LEFT-unmatched, SEMI, ANTI: single row, NULL build side.
                r.probe_idx[off] = static_cast<std::int64_t>(i);
                r.build_idx[off] = -1;
                ++off;
            }
        }
        r.wall_ms = elapsed_ms(t0);
        return r;
    }

    // ---- Resident GROUP BY / top-k (v0.6) — the executable reference ----
    // Sort (key, value) pairs, then run-length reduce. Output sorted by key
    // ascending; i64 sums in uint64 wrap arithmetic (bit-exact contract).
    GroupByResidentResult groupby_sum_resident_i64(const ResidentColumn& keys,
                                                   const ResidentColumn& vals,
                                                   std::size_t max_groups) override {
        const auto t0 = std::chrono::steady_clock::now();
        const auto& k = check_i64(keys);
        const auto& v = check_i64(vals);
        if (k.rows() != v.rows())
            throw std::runtime_error(
                "groupby_sum_resident_i64: keys and vals row counts differ");
        GroupByResidentResult r{};
        r.rows_in = k.rows();
        if (k.rows() == 0) { r.wall_ms = elapsed_ms(t0); return r; }

        std::vector<std::pair<std::int64_t, std::int64_t>> p(k.rows());
        for (std::size_t i = 0; i < k.rows(); ++i) p[i] = { k.as_i64()[i], v.as_i64()[i] };
        std::sort(p.begin(), p.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        check_group_cap(count_runs(p), max_groups, "groupby_sum_resident_i64");

        std::size_t i = 0;
        while (i < p.size()) {
            const std::int64_t key = p[i].first;
            std::uint64_t sum = 0; std::int64_t cnt = 0;
            for (; i < p.size() && p[i].first == key; ++i) {
                sum += static_cast<std::uint64_t>(p[i].second); ++cnt;
            }
            r.keys.push_back(key);
            r.sums.push_back(static_cast<std::int64_t>(sum));
            r.counts.push_back(cnt);
        }
        r.wall_ms = elapsed_ms(t0);
        return r;
    }

    GroupByResidentResult groupby_sum_resident_f64(const ResidentColumn& keys,
                                                   const ResidentColumn& vals,
                                                   std::size_t max_groups) override {
        const auto t0 = std::chrono::steady_clock::now();
        const auto& k = check_i64(keys);
        const auto& v = check_f64(vals);
        if (k.rows() != v.rows())
            throw std::runtime_error(
                "groupby_sum_resident_f64: keys and vals row counts differ");
        GroupByResidentResult r{};
        r.rows_in = k.rows();
        if (k.rows() == 0) { r.wall_ms = elapsed_ms(t0); return r; }

        std::vector<std::pair<std::int64_t, double>> p(k.rows());
        for (std::size_t i = 0; i < k.rows(); ++i) p[i] = { k.as_i64()[i], v.as_f64()[i] };
        std::stable_sort(p.begin(), p.end(),
                         [](const auto& a, const auto& b) { return a.first < b.first; });
        check_group_cap(count_runs(p), max_groups, "groupby_sum_resident_f64");

        std::size_t i = 0;
        while (i < p.size()) {
            const std::int64_t key = p[i].first;
            double sum = 0.0; std::int64_t cnt = 0;
            for (; i < p.size() && p[i].first == key; ++i) { sum += p[i].second; ++cnt; }
            r.keys.push_back(key);
            r.sums_f64.push_back(sum);
            r.counts.push_back(cnt);
        }
        r.wall_ms = elapsed_ms(t0);
        return r;
    }

    GroupByResidentResult groupby_count_resident(const ResidentColumn& keys,
                                                 std::size_t max_groups) override {
        const auto t0 = std::chrono::steady_clock::now();
        const auto& k = check_i64(keys);
        GroupByResidentResult r{};
        r.rows_in = k.rows();
        if (k.rows() == 0) { r.wall_ms = elapsed_ms(t0); return r; }

        std::vector<std::int64_t> s(k.as_i64(), k.as_i64() + k.rows());
        std::sort(s.begin(), s.end());
        std::size_t groups = 1;
        for (std::size_t i = 1; i < s.size(); ++i) groups += (s[i] != s[i - 1]);
        check_group_cap(groups, max_groups, "groupby_count_resident");

        std::size_t i = 0;
        while (i < s.size()) {
            const std::int64_t key = s[i];
            std::int64_t cnt = 0;
            for (; i < s.size() && s[i] == key; ++i) ++cnt;
            r.keys.push_back(key);
            r.counts.push_back(cnt);
        }
        r.wall_ms = elapsed_ms(t0);
        return r;
    }

    TopKResult topk_resident(const ResidentColumn& col, std::size_t k,
                             bool descending) override {
        const auto t0 = std::chrono::steady_clock::now();
        if (col.backend_tag() != Backend::CPU)
            throw std::runtime_error("ResidentColumn from wrong backend");
        const auto& c = static_cast<const CpuResidentColumn&>(col);
        TopKResult r{};
        r.rows_in = c.rows();
        const std::size_t n = c.rows();
        const std::size_t kk = std::min(k, n);
        std::vector<std::int64_t> idx(n);
        for (std::size_t i = 0; i < n; ++i) idx[i] = static_cast<std::int64_t>(i);

        if (c.dtype() == Dtype::I64) {
            const std::int64_t* d = c.as_i64();
            auto less = [&](std::int64_t a, std::int64_t b) {
                return descending ? d[a] > d[b] : d[a] < d[b];
            };
            std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(kk),
                              idx.end(), less);
            r.idx.assign(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(kk));
            r.values_i64.reserve(kk);
            for (auto i : r.idx) r.values_i64.push_back(d[i]);
        } else {
            // Total order with NaN greatest (native DuckDB ORDER BY).
            const double* d = c.as_f64();
            auto lt = [](double a, double b) {
                const bool na = std::isnan(a), nb = std::isnan(b);
                if (na || nb) return !na && nb;
                return a < b;
            };
            auto less = [&](std::int64_t a, std::int64_t b) {
                return descending ? lt(d[b], d[a]) : lt(d[a], d[b]);
            };
            std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(kk),
                              idx.end(), less);
            r.idx.assign(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(kk));
            r.values_f64.reserve(kk);
            for (auto i : r.idx) r.values_f64.push_back(d[i]);
        }
        r.wall_ms = elapsed_ms(t0);
        return r;
    }

private:
    enum class ReduceKind { Sum, Min, Max };

    template <class Pairs>
    static std::size_t count_runs(const Pairs& p) {
        std::size_t g = p.empty() ? 0 : 1;
        for (std::size_t i = 1; i < p.size(); ++i) g += (p[i].first != p[i - 1].first);
        return g;
    }

    static void check_group_cap(std::size_t groups, std::size_t max_groups, const char* op) {
        if (groups > max_groups)
            throw std::runtime_error(
                std::string(op) + ": result has " + std::to_string(groups) +
                " groups, above the cap of " + std::to_string(max_groups) +
                " (raise GPUDB_GROUPBY_ROWS_MAX_M if intentional)");
    }

    static double elapsed_ms(std::chrono::steady_clock::time_point t0) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }

    AggResult run_i64(const std::int64_t* data, std::size_t n, ReduceKind kind, std::int64_t init) {
        const auto t0 = std::chrono::steady_clock::now();
        std::int64_t v = (n == 0) ? 0 : init;
        switch (kind) {
            case ReduceKind::Sum: {
                // Accumulate in uint64: overflow wraps (two's complement),
                // matching the documented gpu_sum semantics, instead of the
                // signed-overflow UB the old code had.
                std::uint64_t uv = 0;
#if GPUDB_HAVE_OPENMP
                #pragma omp parallel for reduction(+:uv) schedule(static)
                for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
                    uv += static_cast<std::uint64_t>(data[i]);
#else
                uv = parallel_chunks<std::uint64_t>(n,
                    [data](std::size_t b, std::size_t e) {
                        std::uint64_t acc = 0;
                        for (std::size_t i = b; i < e; ++i)
                            acc += static_cast<std::uint64_t>(data[i]);
                        return acc;
                    },
                    [](std::uint64_t a, std::uint64_t b) { return a + b; });
#endif
                v = static_cast<std::int64_t>(uv);
                break;
            }
            case ReduceKind::Min: {
#if GPUDB_HAVE_OPENMP
                #pragma omp parallel for reduction(min:v) schedule(static)
                for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
                    if (data[i] < v) v = data[i];
#else
                if (n > 0) v = parallel_chunks<std::int64_t>(n,
                    [data](std::size_t b, std::size_t e) {
                        std::int64_t m = std::numeric_limits<std::int64_t>::max();
                        for (std::size_t i = b; i < e; ++i) if (data[i] < m) m = data[i];
                        return m;
                    },
                    [](std::int64_t a, std::int64_t b) { return a < b ? a : b; });
#endif
                break;
            }
            case ReduceKind::Max: {
#if GPUDB_HAVE_OPENMP
                #pragma omp parallel for reduction(max:v) schedule(static)
                for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
                    if (data[i] > v) v = data[i];
#else
                if (n > 0) v = parallel_chunks<std::int64_t>(n,
                    [data](std::size_t b, std::size_t e) {
                        std::int64_t m = std::numeric_limits<std::int64_t>::min();
                        for (std::size_t i = b; i < e; ++i) if (data[i] > m) m = data[i];
                        return m;
                    },
                    [](std::int64_t a, std::int64_t b) { return a > b ? a : b; });
#endif
                break;
            }
        }
        AggResult r{};
        r.value_i64 = v; r.rows = n;
        r.wall_ms = elapsed_ms(t0);
        return r;
    }

    // Single-pass sum + min + max + count. The whole point: each cache line
    // is touched once, so on a memory-bandwidth-bound workload this should
    // be ~3x faster than calling sum/min/max separately.
    AggAllResult run_agg_all_i64(const std::int64_t* data, std::size_t n) {
        const auto t0 = std::chrono::steady_clock::now();
        AggAllResult r{};
        r.rows  = n;
        r.count = n;
        if (n == 0) {
            r.sum = 0;
            r.min = std::numeric_limits<std::int64_t>::max();
            r.max = std::numeric_limits<std::int64_t>::min();
            r.wall_ms = elapsed_ms(t0);
            return r;
        }

        // uint64 accumulate: overflow wraps (documented gpu_sum semantics)
        // instead of signed-overflow UB.
        std::uint64_t sum_v = 0;
        std::int64_t min_v = std::numeric_limits<std::int64_t>::max();
        std::int64_t max_v = std::numeric_limits<std::int64_t>::min();
#if GPUDB_HAVE_OPENMP
        #pragma omp parallel for reduction(+:sum_v) reduction(min:min_v) \
                                 reduction(max:max_v) schedule(static)
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
            const std::int64_t x = data[i];
            sum_v += static_cast<std::uint64_t>(x);
            if (x < min_v) min_v = x;
            if (x > max_v) max_v = x;
        }
#else
        for (std::size_t i = 0; i < n; ++i) {
            const std::int64_t x = data[i];
            sum_v += static_cast<std::uint64_t>(x);
            if (x < min_v) min_v = x;
            if (x > max_v) max_v = x;
        }
#endif
        r.sum = static_cast<std::int64_t>(sum_v);
        r.min = min_v;
        r.max = max_v;
        r.wall_ms = elapsed_ms(t0);
        return r;
    }

    AggResult run_f64_sum(const double* data, std::size_t n) {
        const auto t0 = std::chrono::steady_clock::now();
        double acc = 0.0;
#if GPUDB_HAVE_OPENMP
        #pragma omp parallel for reduction(+:acc) schedule(static)
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) acc += data[i];
#else
        acc = parallel_chunks<double>(n,
            [data](std::size_t b, std::size_t e) {
                double a = 0.0;
                for (std::size_t i = b; i < e; ++i) a += data[i];
                return a;
            },
            [](double a, double b) { return a + b; });
#endif
        AggResult r{};
        r.value_f64 = acc; r.rows = n;
        r.wall_ms = elapsed_ms(t0);
        return r;
    }

    static const CpuResidentColumn& check_i64(const ResidentColumn& c) {
        if (c.backend_tag() != Backend::CPU)
            throw std::runtime_error("ResidentColumn from wrong backend");
        if (c.dtype() != Dtype::I64)
            throw std::runtime_error("ResidentColumn dtype mismatch (expected i64)");
        return static_cast<const CpuResidentColumn&>(c);
    }
    static const CpuResidentColumn& check_f64(const ResidentColumn& c) {
        if (c.backend_tag() != Backend::CPU)
            throw std::runtime_error("ResidentColumn from wrong backend");
        if (c.dtype() != Dtype::F64)
            throw std::runtime_error("ResidentColumn dtype mismatch (expected f64)");
        return static_cast<const CpuResidentColumn&>(c);
    }
};

} // namespace

std::unique_ptr<Aggregator> make_cpu_aggregator() {
    return std::make_unique<CpuAggregator>();
}

} // namespace gpudb
