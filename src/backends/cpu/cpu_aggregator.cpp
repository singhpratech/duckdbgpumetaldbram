#include "gpu_backend.hpp"
#include "../groupby_filter.hpp"
#include "../predicate_mask.hpp"

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
#include <unordered_map>
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
    // §4.1 exact-path column: `valid` is a DuckDB-layout bitmap over the n
    // rows (empty = all valid). Rows stay in input order; a NULL key is a
    // zero bit like a NULL payload (docs/RESIDENT_COLUMNS_DESIGN.md, stage A).
    CpuResidentColumn(std::vector<std::int64_t>&& data, std::vector<std::uint64_t>&& valid,
                      std::size_t null_count, Dtype dt = Dtype::I64)
        : rows_(data.size()), dtype_(dt), valid_(std::move(valid)), nulls_(null_count) {
        buf_.resize(rows_ * sizeof(std::int64_t));
        if (rows_) std::memcpy(buf_.data(), data.data(), buf_.size());
    }
    Backend     backend_tag() const noexcept override { return Backend::CPU; }
    Dtype       dtype()       const noexcept override { return dtype_; }
    std::size_t rows()        const noexcept override { return rows_; }
    std::size_t null_count()  const noexcept override { return nulls_; }
    std::size_t resident_bytes() const noexcept override {
        return rows_ * 8 + valid_.size() * sizeof(std::uint64_t);
    }

    [[nodiscard]] const std::int64_t* as_i64() const {
        return reinterpret_cast<const std::int64_t*>(buf_.data());
    }
    [[nodiscard]] const double* as_f64() const {
        return reinterpret_cast<const double*>(buf_.data());
    }
    // Row i valid? Without a bitmap every row is (a column with NULLs always
    // carries one). With a bitmap, its bit decides.
    [[nodiscard]] bool valid(std::size_t i) const noexcept {
        if (valid_.empty()) return true;
        return ((valid_[i >> 6] >> (i & 63)) & 1u) != 0;
    }

private:
    std::vector<std::byte> buf_;
    std::size_t rows_;
    Dtype       dtype_;
    std::vector<std::uint64_t> valid_;   // empty = no NULLs
    std::size_t nulls_ = 0;
};

class CpuAggregator final : public Aggregator {
public:
    Backend backend() const noexcept override { return Backend::CPU; }
    bool exact_supported() const noexcept override { return true; }   // the reference

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
                                                   std::size_t max_groups,
                                                   const GroupByFilter& filter) override {
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
        if (!filter.active()) check_group_cap(count_runs(p), max_groups, "groupby_sum_resident_i64");

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
        apply_group_filter_host(r, filter, FilterAgg::SumI64, max_groups, "groupby_sum_resident_i64");
        r.wall_ms = elapsed_ms(t0);
        return r;
    }

    GroupByResidentResult groupby_sum_resident_f64(const ResidentColumn& keys,
                                                   const ResidentColumn& vals,
                                                   std::size_t max_groups,
                                                   const GroupByFilter& filter) override {
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
        if (!filter.active()) check_group_cap(count_runs(p), max_groups, "groupby_sum_resident_f64");

        std::size_t i = 0;
        while (i < p.size()) {
            const std::int64_t key = p[i].first;
            double sum = 0.0; std::int64_t cnt = 0;
            for (; i < p.size() && p[i].first == key; ++i) { sum += p[i].second; ++cnt; }
            r.keys.push_back(key);
            r.sums_f64.push_back(sum);
            r.counts.push_back(cnt);
        }
        apply_group_filter_host(r, filter, FilterAgg::SumF64, max_groups, "groupby_sum_resident_f64");
        r.wall_ms = elapsed_ms(t0);
        return r;
    }

    GroupByResidentResult groupby_count_resident(const ResidentColumn& keys,
                                                 std::size_t max_groups,
                                                 const GroupByFilter& filter) override {
        const auto t0 = std::chrono::steady_clock::now();
        const auto& k = check_i64(keys);
        GroupByResidentResult r{};
        r.rows_in = k.rows();
        if (k.rows() == 0) { r.wall_ms = elapsed_ms(t0); return r; }

        std::vector<std::int64_t> s(k.as_i64(), k.as_i64() + k.rows());
        std::sort(s.begin(), s.end());
        std::size_t groups = 1;
        for (std::size_t i = 1; i < s.size(); ++i) groups += (s[i] != s[i - 1]);
        if (!filter.active()) check_group_cap(groups, max_groups, "groupby_count_resident");

        std::size_t i = 0;
        while (i < s.size()) {
            const std::int64_t key = s[i];
            std::int64_t cnt = 0;
            for (; i < s.size() && s[i] == key; ++i) ++cnt;
            r.keys.push_back(key);
            r.counts.push_back(cnt);
        }
        apply_group_filter_host(r, filter, FilterAgg::Count, max_groups, "groupby_count_resident");
        r.wall_ms = elapsed_ms(t0);
        return r;
    }

    // ---- v0.7 milestone 3: exact path (§4.1 / §4.2) — the executable reference ----
    // Rows stay in input order; NULL keys and NULL payloads sit under
    // validity bitmaps (stage A of docs/RESIDENT_COLUMNS_DESIGN.md).
    ResidentPair upload_pair_exact(const KvSpan* spans, std::size_t n_spans,
                                   Dtype vdt) override {
        if (vdt != Dtype::I64)
            throw std::runtime_error(
                "upload_pair_exact: DOUBLE payloads are not on the exact path (docs/TRANSPARENT_DESIGN.md §4.7)");
        std::size_t rows = 0;
        for (std::size_t i = 0; i < n_spans; ++i) rows += spans[i].rows;
        auto bit = [](const std::uint64_t* m, std::size_t i) {
            return !m || ((m[i >> 6] >> (i & 63)) & 1u);
        };
        std::vector<std::int64_t> k(rows), v(rows);
        std::vector<std::uint64_t> kvalid((rows + 63) / 64, 0), vvalid((rows + 63) / 64, 0);
        std::size_t dst = 0, null_keys = 0, null_vals = 0;
        for (std::size_t s = 0; s < n_spans; ++s) {
            const KvSpan& sp = spans[s];
            for (std::size_t j = 0; j < sp.rows; ++j, ++dst) {
                const bool kv_ok = bit(sp.key_valid, j);
                const bool vv_ok = bit(sp.val_valid, j);
                k[dst] = kv_ok ? sp.kv[2 * j] : 0;
                v[dst] = vv_ok ? sp.kv[2 * j + 1] : 0;
                if (kv_ok) kvalid[dst >> 6] |= std::uint64_t{1} << (dst & 63); else ++null_keys;
                if (vv_ok) vvalid[dst >> 6] |= std::uint64_t{1} << (dst & 63); else ++null_vals;
            }
        }
        if (!null_keys) kvalid.clear();               // no bitmap when NULL-free
        if (!null_vals) vvalid.clear();
        ResidentPair out;
        out.keys = std::make_unique<CpuResidentColumn>(std::move(k), std::move(kvalid), null_keys);
        out.vals = std::make_unique<CpuResidentColumn>(std::move(v), std::move(vvalid), null_vals);
        return out;
    }

    GroupByResidentResult groupby_exact_resident(const ResidentColumn& keys,
                                                 const ResidentColumn* vals,
                                                 std::size_t max_groups,
                                                 const GroupByFilter& filter) override {
        return exact_impl(keys, vals, nullptr, 0, max_groups, filter, "groupby_exact_resident");
    }

    // §4.6: multi-lane exact upload. Every lane is written in input order so
    // the columns stay row-aligned; each returned column carries its own
    // validity bitmap when it has NULLs (the key included).
    std::vector<std::unique_ptr<ResidentColumn>>
    upload_rows_exact(const RowSpan* spans, std::size_t n_spans,
                      const Dtype* dtypes, std::size_t n_lanes) override {
        if (n_lanes == 0) throw std::runtime_error("upload_rows_exact: no lanes");
        if (dtypes[0] != Dtype::I64) throw std::runtime_error("upload_rows_exact: the key lane must be I64");
        // (lane 1 may be F64: a store upload orders lanes row id, ints, doubles, strings)
        std::size_t rows = 0, null_keys = 0;
        auto bit = [](const std::uint64_t* m, std::size_t i) {
            return !m || ((m[i >> 6] >> (i & 63)) & 1u);
        };
        auto lane_valid = [](const RowSpan& sp, std::size_t lane) -> const std::uint64_t* {
            return sp.valid ? sp.valid[lane] : nullptr;
        };
        for (std::size_t s = 0; s < n_spans; ++s) {
            if (spans[s].n_lanes != n_lanes) throw std::runtime_error("upload_rows_exact: span lane count differs");
            rows += spans[s].rows;
            const std::uint64_t* kv = lane_valid(spans[s], 0);
            if (kv) for (std::size_t j = 0; j < spans[s].rows; ++j) null_keys += !bit(kv, spans[s].valid_bit + j);
        }
        const std::size_t words = (rows + 63) / 64;
        std::vector<std::vector<std::int64_t>>  data(n_lanes, std::vector<std::int64_t>(rows));
        std::vector<std::vector<std::uint64_t>> valid(n_lanes, std::vector<std::uint64_t>(words, ~std::uint64_t{0}));
        std::vector<std::size_t> nulls(n_lanes, 0);
        (void)null_keys;
        std::size_t next = 0;
        for (std::size_t s = 0; s < n_spans; ++s) {
            const RowSpan& sp = spans[s];
            const std::size_t d0 = sp.dst_row == RowSpan::kNext ? next : sp.dst_row;
            if (d0 + sp.rows > rows) throw std::runtime_error("upload_rows_exact: span destination out of range");
            next = d0 + sp.rows;
            for (std::size_t j = 0; j < sp.rows; ++j) {
                const std::size_t dst = d0 + j;
                for (std::size_t l = 0; l < n_lanes; ++l) {
                    const bool ok = bit(lane_valid(sp, l), sp.valid_bit + j);
                    data[l][dst] = ok ? sp.lanes[j * n_lanes + l] : 0;
                    if (!ok) { valid[l][dst >> 6] &= ~(std::uint64_t{1} << (dst & 63)); ++nulls[l]; }
                }
            }
        }
        std::vector<std::unique_ptr<ResidentColumn>> out;
        out.reserve(n_lanes);
        for (std::size_t l = 0; l < n_lanes; ++l) {
            // a NULL-free lane drops its bitmap
            std::vector<std::uint64_t> vb = (nulls[l] == 0) ? std::vector<std::uint64_t>{} : std::move(valid[l]);
            out.push_back(std::make_unique<CpuResidentColumn>(std::move(data[l]), std::move(vb),
                                                              nulls[l], dtypes[l]));
        }
        return out;
    }

    GroupByResidentResult groupby_exact_masked_resident(const ResidentColumn& keys,
                                                        const ResidentColumn* vals,
                                                        const Predicate* preds,
                                                        std::size_t n_preds,
                                                        std::size_t max_groups,
                                                        const GroupByFilter& filter) override {
        return exact_impl(keys, vals, preds, n_preds, max_groups, filter, "groupby_exact_masked_resident");
    }

    // ---- v0.7 §4.12: the global masked aggregate — reference implementation ----
    // One pass over the rows in storage order: evaluate the whole predicate
    // conjunction for the row, then fold every payload of the surviving row.
    // No key, no sort, no permutation — this is the shape the operator exists
    // for. The accumulator array is indexed [group * n_pays + payload] with
    // one group today, so a later few-group variant reuses the fold unchanged.
    bool global_supported() const noexcept override { return true; }   // the reference

    GlobalAggResult aggregate_exact_masked(const MultiPayload* pays, std::size_t n_pays,
                                           const Predicate* preds, std::size_t n_preds) override {
        static const char* op = "aggregate_exact_masked";
        const auto t0 = std::chrono::steady_clock::now();
        if (n_pays == 0 && n_preds == 0)
            throw std::runtime_error(std::string(op) + ": neither a payload nor a predicate");
        std::vector<const CpuResidentColumn*> pc(n_pays, nullptr);
        std::size_t n = 0;
        bool have_n = false;
        for (std::size_t p = 0; p < n_pays; ++p) {
            if (!pays[p].vals) throw std::runtime_error(std::string(op) + ": payload without a column");
            pc[p] = &check_i64_nullable(*pays[p].vals);
            if (!have_n) { n = pc[p]->rows(); have_n = true; }
            else if (pc[p]->rows() != n)
                throw std::runtime_error(std::string(op) + ": payload row counts differ");
        }
        std::vector<const CpuResidentColumn*> prc(n_preds, nullptr);
        for (std::size_t q = 0; q < n_preds; ++q) {
            if (!preds[q].col) throw std::runtime_error(std::string(op) + ": predicate without a column");
            if (preds[q].col->backend_tag() != Backend::CPU)
                throw std::runtime_error("ResidentColumn from wrong backend");
            prc[q] = static_cast<const CpuResidentColumn*>(preds[q].col);
            if (!have_n) { n = prc[q]->rows(); have_n = true; }
            else if (prc[q]->rows() != n)
                throw std::runtime_error(std::string(op) + ": predicate column row count differs from the payloads");
        }

        GlobalAggResult r{};
        r.rows_in = n;
        r.sums.assign(n_pays, 0); r.sums_hi.assign(n_pays, 0); r.counts.assign(n_pays, 0);
        r.mins.assign(n_pays, 0); r.maxs.assign(n_pays, 0);
        if (n == 0) { r.wall_ms = elapsed_ms(t0); return r; }

        struct Acc {
            Sum128 s;
            std::int64_t cnt = 0;
            std::int64_t mn = std::numeric_limits<std::int64_t>::max();
            std::int64_t mx = std::numeric_limits<std::int64_t>::min();
        };
        struct Part {
            std::vector<Acc> acc;        // [group * n_pays + payload]; one group today
            std::int64_t cstar = 0;
        };
        auto chunk = [&](std::size_t b, std::size_t e) {
            Part part;
            part.acc.assign(n_pays, Acc{});
            for (std::size_t i = b; i < e; ++i) {
                bool keep = true;
                for (std::size_t q = 0; keep && q < n_preds; ++q) {
                    const CpuResidentColumn& c = *prc[q];
                    const std::int64_t* d = c.as_i64();
                    keep = predicate_row(preds[q], c.dtype(), i,
                                         [d](std::size_t row) { return d[row]; },
                                         [&c](std::size_t row) { return c.valid(row); });
                }
                if (!keep) continue;
                ++part.cstar;
                for (std::size_t p = 0; p < n_pays; ++p) {
                    if (!pc[p]->valid(i)) continue;
                    const std::int64_t x = pc[p]->as_i64()[i];
                    Acc& a = part.acc[p];
                    a.s.add(x); ++a.cnt;
                    if (x < a.mn) a.mn = x;
                    if (x > a.mx) a.mx = x;
                }
            }
            return part;
        };
        auto combine = [&](Part a, const Part& b) {
            a.cstar += b.cstar;
            for (std::size_t p = 0; p < a.acc.size(); ++p) {
                Acc& x = a.acc[p];
                const Acc& y = b.acc[p];
                const std::uint64_t old = x.s.lo;
                x.s.lo += y.s.lo;
                x.s.hi += y.s.hi + (x.s.lo < old ? 1 : 0);
                x.cnt += y.cnt;
                if (y.mn < x.mn) x.mn = y.mn;
                if (y.mx > x.mx) x.mx = y.mx;
            }
            return a;
        };
        const Part total = parallel_chunks<Part>(n, chunk, combine);
        r.count_star = total.cstar;
        for (std::size_t p = 0; p < n_pays; ++p) {
            const Acc& a = total.acc[p];
            r.sums[p]    = static_cast<std::int64_t>(a.s.lo);
            r.sums_hi[p] = a.s.hi;
            r.counts[p]  = a.cnt;
            r.mins[p]    = a.cnt ? a.mn : 0;
            r.maxs[p]    = a.cnt ? a.mx : 0;
        }
        r.wall_ms = elapsed_ms(t0);
        return r;
    }

    // ---- v0.7 §4.8: the materialised key join — reference implementation ----
    bool join_supported() const noexcept override { return true; }

    JoinMaterializeResult join_materialize(const ResidentColumn& probe_key,
                                           const ResidentColumn& build_key,
                                           const JoinLane* out, std::size_t n_out) override {
        const auto t0 = std::chrono::steady_clock::now();
        const auto& pk = check_i64_nullable(probe_key);
        const auto& bk = check_i64_nullable(build_key);
        if (n_out == 0 || !out) throw std::runtime_error("join_materialize: no output lanes");
        for (std::size_t l = 0; l < n_out; ++l) {
            if (!out[l].col) throw std::runtime_error("join_materialize: output lane without a column");
            if (out[l].col->backend_tag() != Backend::CPU)
                throw std::runtime_error("ResidentColumn from wrong backend");
            if (out[l].col->rows() != (out[l].from_build ? bk.rows() : pk.rows()))
                throw std::runtime_error("join_materialize: lane " + std::to_string(l) +
                                         " row count differs from its side of the join");
        }
        if (out[0].col->dtype() != Dtype::I64)
            throw std::runtime_error("join_materialize: the key lane must be I64");
        if (pk.rows() > 0xFFFFFFFEull || bk.rows() > 0xFFFFFFFEull)
            throw std::runtime_error("join_materialize: > 2^32-2 rows unsupported");

        JoinMaterializeResult r;
        r.rows_probe = pk.rows();
        r.rows_build = bk.rows();

        // Build side: valid key -> row; a second row for one key is the error.
        std::unordered_map<std::int64_t, std::uint32_t> map;
        map.reserve(bk.rows() * 2);
        const std::int64_t* bd = bk.as_i64();
        for (std::size_t i = 0; i < bk.rows(); ++i) {
            if (!bk.valid(i)) continue;
            if (!map.emplace(bd[i], static_cast<std::uint32_t>(i)).second)
                throw std::runtime_error("join_materialize: build key not unique");
        }

        // Probe: match row and output class (0 none, 1 valid out key, 2 NULL out key).
        const std::size_t n = pk.rows();
        const std::int64_t* pd = pk.as_i64();
        const auto& kc = static_cast<const CpuResidentColumn&>(*out[0].col);
        std::vector<std::uint32_t> match(n, 0xFFFFFFFFu);
        std::vector<std::uint8_t>  cls(n, 0);
        std::size_t n1 = 0, n2 = 0;
        for (std::size_t i = 0; i < n; ++i) {
            if (!pk.valid(i)) continue;
            const auto it = map.find(pd[i]);
            if (it == map.end()) continue;
            match[i] = it->second;
            const std::size_t krow = out[0].from_build ? it->second : i;
            if (kc.valid(krow)) { cls[i] = 1; ++n1; } else { cls[i] = 2; ++n2; }
        }
        const std::size_t rows_out = n1 + n2;
        std::vector<std::uint32_t> pos(n, 0);
        {
            std::size_t a = 0, b = n1;
            for (std::size_t i = 0; i < n; ++i) {
                if (cls[i] == 1) pos[i] = static_cast<std::uint32_t>(a++);
                else if (cls[i] == 2) pos[i] = static_cast<std::uint32_t>(b++);
            }
        }
        const std::size_t words = (rows_out + 63) / 64;
        r.lanes.reserve(n_out);
        for (std::size_t l = 0; l < n_out; ++l) {
            const auto& sc = static_cast<const CpuResidentColumn&>(*out[l].col);
            const std::int64_t* sd = sc.as_i64();          // raw 8-byte cells
            std::vector<std::int64_t>  data(rows_out, 0);
            std::vector<std::uint64_t> valid(words, ~std::uint64_t{0});
            std::size_t nulls = 0;
            for (std::size_t i = 0; i < n; ++i) {
                if (!cls[i]) continue;
                const std::size_t srow = out[l].from_build ? match[i] : i;
                const std::size_t d = pos[i];
                if (sc.valid(srow)) data[d] = sd[srow];
                else { valid[d >> 6] &= ~(std::uint64_t{1} << (d & 63)); ++nulls; }
            }
            std::vector<std::uint64_t> vb = (nulls == 0) ? std::vector<std::uint64_t>{} : std::move(valid);
            r.lanes.push_back(std::make_unique<CpuResidentColumn>(std::move(data), std::move(vb),
                                                                  nulls, sc.dtype()));
        }
        r.rows_out = rows_out;
        r.null_key_rows = n2;
        r.wall_ms = elapsed_ms(t0);
        return r;
    }

    GroupByResidentResult exact_impl(const ResidentColumn& keys, const ResidentColumn* vals,
                                     const Predicate* preds, std::size_t n_preds,
                                     std::size_t max_groups, const GroupByFilter& filter,
                                     const char* op) {
        const auto t0 = std::chrono::steady_clock::now();
        const auto& k = check_i64_nullable(keys);
        const CpuResidentColumn* v = vals ? &check_i64_nullable(*vals) : nullptr;
        if (v && v->rows() != k.rows())
            throw std::runtime_error(std::string(op) + ": keys and vals row counts differ");
        GroupByResidentResult r{};
        r.rows_in = k.rows();
        if (k.rows() == 0) { r.wall_ms = elapsed_ms(t0); return r; }

        const std::size_t n = k.rows();

        // ---- WHERE mask (§4.6): one byte per row, conjunction of preds ----
        std::vector<std::uint8_t> mask;
        if (n_preds) {
            mask.assign(n, 1);
            for (std::size_t p = 0; p < n_preds; ++p) {
                if (!preds[p].col) throw std::runtime_error(std::string(op) + ": predicate without a column");
                if (preds[p].col->backend_tag() != Backend::CPU)
                    throw std::runtime_error("ResidentColumn from wrong backend");
                const auto& pc = static_cast<const CpuResidentColumn&>(*preds[p].col);
                if (pc.rows() != n)
                    throw std::runtime_error(std::string(op) + ": predicate column row count differs from the keys");
                const std::int64_t* pd = pc.as_i64();      // raw 8-byte cells (IEEE bits for F64)
                for (std::size_t i = 0; i < n; ++i) {
                    if (!mask[i]) continue;
                    mask[i] = predicate_row(preds[p], pc.dtype(), i,
                                            [pd](std::size_t row) { return pd[row]; },
                                            [&pc](std::size_t row) { return pc.valid(row); }) ? 1 : 0;
                }
            }
        }
        auto in_mask = [&](std::size_t row) { return mask.empty() || mask[row]; };

        // Sort the surviving valid-key rows by key; the surviving NULL-key rows
        // are one group (any position: the key's bitmap says which rows).
        std::vector<std::size_t> idx, null_idx;
        idx.reserve(n - k.null_count());
        for (std::size_t i = 0; i < n; ++i) {
            if (!in_mask(i)) continue;
            if (k.valid(i)) idx.push_back(i); else null_idx.push_back(i);
        }
        const std::size_t n_sel = idx.size();
        const std::int64_t* kd = k.as_i64();
        std::stable_sort(idx.begin(), idx.end(),
                         [kd](std::size_t a, std::size_t b) { return kd[a] < kd[b]; });
        const std::size_t null_rows = null_idx.size();

        std::size_t groups = 0;
        for (std::size_t i = 0; i < n_sel; ++i)
            groups += (i == 0 || kd[idx[i]] != kd[idx[i - 1]]);
        if (null_rows) ++groups;
        if (!filter.active()) check_group_cap(groups, max_groups, op);

        auto emit = [&](std::int64_t key, bool key_is_null, std::size_t b, std::size_t e,
                        const std::vector<std::size_t>& ix) {
            Sum128 s; std::int64_t cnt_v = 0, cnt_star = 0;
            std::int64_t mn = std::numeric_limits<std::int64_t>::max();
            std::int64_t mx = std::numeric_limits<std::int64_t>::min();
            const std::int64_t* vd = v ? v->as_i64() : nullptr;
            for (std::size_t i = b; i < e; ++i) {
                const std::size_t row = ix[i];               // mask already applied when ix was built
                ++cnt_star;
                if (!v) continue;
                if (!v->valid(row)) continue;
                const std::int64_t x = vd[row];
                s.add(x); ++cnt_v;
                if (x < mn) mn = x;
                if (x > mx) mx = x;
            }
            if (!v) cnt_v = cnt_star;
            r.keys.push_back(key);
            r.key_null.push_back(key_is_null ? 1 : 0);
            r.sums.push_back(static_cast<std::int64_t>(s.lo));
            r.sums_hi.push_back(s.hi);
            r.counts.push_back(cnt_v);
            r.counts_star.push_back(cnt_star);
            r.mins.push_back(cnt_v ? mn : 0);
            r.maxs.push_back(cnt_v ? mx : 0);
        };
        std::size_t i = 0;
        while (i < n_sel) {
            const std::int64_t key = kd[idx[i]];
            std::size_t j = i + 1;
            while (j < n_sel && kd[idx[j]] == key) ++j;
            emit(key, false, i, j, idx);
            i = j;
        }
        if (null_rows) emit(0, true, 0, null_rows, null_idx);
        apply_group_filter_host(r, filter, FilterAgg::Exact, max_groups, op);
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

    // Legacy ops have no NULL semantics: refuse a column that carries NULLs
    // (only upload_pair_exact produces one) instead of reading NULL rows as
    // data. groupby_exact_resident uses check_i64_nullable.
    static const CpuResidentColumn& check_i64(const ResidentColumn& c) {
        const auto& r = check_i64_nullable(c);
        if (r.null_count() != 0)
            throw std::runtime_error(
                "ResidentColumn carries NULL rows (uploaded by gpu_upload_pair_exact) — "
                "only the exact GROUP BY (gpu_groupby_exact_resident) accepts it");
        return r;
    }
    static const CpuResidentColumn& check_i64_nullable(const ResidentColumn& c) {
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
