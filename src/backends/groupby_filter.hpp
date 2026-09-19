// groupby_filter.hpp — host reference implementation of GroupByFilter
// (HAVING on the aggregate, then top-k of groups). f64 uses DuckDB's total
// order for both cmp and top-k: every NaN sum is the greatest value (so
// `> t` / `>= t` keep NaN groups, `< t` / `<= t` drop them, DESC lists them
// first, ASC last); NaN == NaN; -0.0 == 0.0.
// Executable contract for
// every backend; also used where a backend finishes an aggregate on the
// host (Metal f64 sums) and by unit/parity checks.
#pragma once
#include "gpu_backend.hpp"
#include "native_avg.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gpudb {

// Exact = groupby_exact_resident: the aggregate is chosen by GroupByFilter::agg.
enum class FilterAgg { SumI64, SumF64, Count, Exact };

// GroupByFilter reference semantics: cmp on the op's aggregate, then the
// k largest/smallest survivors (ties broken by key ascending here — the
// contract leaves tie order unspecified). Cap applies to rows returned.
// For FilterAgg::Exact a NULL aggregate (count(v) == 0 for sum/min/max/avg)
// never passes cmp and ranks last under top-k in either direction.
inline void apply_group_filter_host(GroupByResidentResult& r, const GroupByFilter& f,
                               FilterAgg agg, std::size_t max_groups, const char* op) {
    r.groups_total = r.keys.size();
    if (!f.active()) return;
    const bool exact = (agg == FilterAgg::Exact);
    using A = GroupByFilter::Agg;
    const bool is_f64 = (agg == FilterAgg::SumF64) || (exact && f.agg == A::Avg);
    // Exact op: is the chosen aggregate NULL for row i?
    auto ex_null = [&](std::size_t i) -> bool {
        if (!exact) return false;
        switch (f.agg) {
            case A::Sum: case A::Min: case A::Max: case A::Avg: return r.counts[i] == 0;
            default: return false;
        }
    };
    auto ex_f64 = [&](std::size_t i) -> double {   // exact op, Avg
        // The SAME expression the emitted avg column uses (native_avg.hpp):
        // a HAVING that compared a differently-rounded average would keep a
        // different set of groups than the values it then returns.
        return native_avg(Sum128{static_cast<std::uint64_t>(r.sums[i]), r.sums_hi[i]},
                          r.counts[i]);
    };
    auto ex_s128 = [&](std::size_t i) -> Sum128 {
        return Sum128{static_cast<std::uint64_t>(r.sums[i]), r.sums_hi[i]};
    };
    auto agg_i64 = [&](std::size_t i) -> std::int64_t {
        if (exact) {
            switch (f.agg) {
                case A::CountV:    return r.counts[i];
                case A::CountStar: return r.counts_star[i];
                case A::Min:       return r.mins[i];
                case A::Max:       return r.maxs[i];
                default:           return r.sums[i];   // Sum handled via ex_s128
            }
        }
        return agg == FilterAgg::SumI64 ? r.sums[i] : r.counts[i];
    };
    auto agg_f64 = [&](std::size_t i) -> double {
        return exact ? ex_f64(i) : r.sums_f64[i];
    };
    auto keep = [&](std::size_t i) -> bool {
        if (f.cmp == GroupByFilter::Cmp::None) return true;
        if (ex_null(i)) return false;               // NULL <cmp> t is not true
        if (exact && f.agg == A::Sum) {
            const Sum128 a = ex_s128(i), t = Sum128::from_i64(f.threshold_i64);
            switch (f.cmp) {
                case GroupByFilter::Cmp::GT: return t < a;
                case GroupByFilter::Cmp::GE: return !(a < t);
                case GroupByFilter::Cmp::LT: return a < t;
                case GroupByFilter::Cmp::LE: return !(t < a);
                default: return true;
            }
        }
        if (is_f64) {
            // DuckDB comparison order, not IEEE: NaN (any sign) is greater
            // than everything and equal to NaN; -0.0 == 0.0. So `sum > t`
            // keeps NaN groups exactly as native HAVING does.
            const double a = agg_f64(i), t = f.threshold_f64;
            auto lt = [](double x, double y) { return !std::isnan(x) && (std::isnan(y) || x < y); };
            switch (f.cmp) {
                case GroupByFilter::Cmp::GT: return lt(t, a);
                case GroupByFilter::Cmp::GE: return !lt(a, t);
                case GroupByFilter::Cmp::LT: return lt(a, t);
                case GroupByFilter::Cmp::LE: return !lt(t, a);
                default: return true;
            }
        }
        const std::int64_t a = agg_i64(i), t = f.threshold_i64;
        switch (f.cmp) {
            case GroupByFilter::Cmp::GT: return a >  t;
            case GroupByFilter::Cmp::GE: return a >= t;
            case GroupByFilter::Cmp::LT: return a <  t;
            case GroupByFilter::Cmp::LE: return a <= t;
            default: return true;
        }
    };
    std::vector<std::size_t> idx;
    idx.reserve(r.keys.size());
    for (std::size_t i = 0; i < r.keys.size(); ++i) if (keep(i)) idx.push_back(i);
    // Tie-break: key ascending, NULL key last (the exact op's output order).
    auto key_before = [&](std::size_t a, std::size_t b) {
        if (exact && !r.key_null.empty() && r.key_null[a] != r.key_null[b]) return r.key_null[a] == 0;
        return r.keys[a] < r.keys[b];
    };
    auto better = [&](std::size_t a, std::size_t b) {   // strict "a ranks before b"
        if (exact) {
            const bool na = ex_null(a), nb = ex_null(b);
            if (na != nb) return nb;                    // NULLS LAST in both directions
            if (na) return key_before(a, b);
            if (f.agg == A::Sum) {
                const Sum128 x = ex_s128(a), y = ex_s128(b);
                if (!(x == y)) return f.topk_desc ? (y < x) : (x < y);
                return key_before(a, b);
            }
        }
        if (is_f64) {
            const double x = agg_f64(a), y = agg_f64(b);
            const bool nx = std::isnan(x), ny = std::isnan(y);
            if (nx != ny) return f.topk_desc ? nx : ny;      // NaN is greatest (DuckDB order)
            if (!nx && x != y) return f.topk_desc ? (x > y) : (x < y);
        } else {
            const std::int64_t x = agg_i64(a), y = agg_i64(b);
            if (x != y) return f.topk_desc ? (x > y) : (x < y);
        }
        return key_before(a, b);
    };
    if (f.topk != 0 && f.topk < idx.size()) {
        std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(f.topk),
                          idx.end(), better);
        idx.resize(f.topk);
    } else if (f.topk != 0) {
        std::sort(idx.begin(), idx.end(), better);
    }
    if (idx.size() > max_groups)
        throw std::runtime_error(
            std::string(op) + ": result has " + std::to_string(idx.size()) +
            " rows after the filter, above the cap of " + std::to_string(max_groups) +
            " (raise GPUDB_GROUPBY_ROWS_MAX_M if intentional)");
    GroupByResidentResult o{};
    o.rows_in = r.rows_in; o.groups_total = r.groups_total;
    o.wall_ms = r.wall_ms; o.kernel_ms = r.kernel_ms; o.transfer_ms = r.transfer_ms;
    o.keys.reserve(idx.size()); o.counts.reserve(idx.size());
    auto gather = [&](const auto& src, auto& dst) {
        if (src.empty()) return;
        dst.reserve(idx.size());
        for (std::size_t i : idx) dst.push_back(src[i]);
    };
    for (std::size_t i : idx) {
        o.keys.push_back(r.keys[i]);
        o.counts.push_back(r.counts[i]);
    }
    gather(r.sums, o.sums);
    gather(r.sums_f64, o.sums_f64);
    gather(r.sums_hi, o.sums_hi);
    gather(r.counts_star, o.counts_star);
    gather(r.mins, o.mins);
    gather(r.maxs, o.maxs);
    gather(r.key_null, o.key_null);
    r = std::move(o);
}

} // namespace gpudb
