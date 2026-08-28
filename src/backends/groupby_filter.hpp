// groupby_filter.hpp — host reference implementation of GroupByFilter
// (HAVING on the aggregate, then top-k of groups). Executable contract for
// every backend; also used where a backend finishes an aggregate on the
// host (Metal f64 sums) and by unit/parity checks.
#pragma once
#include "gpu_backend.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gpudb {

enum class FilterAgg { SumI64, SumF64, Count };

// GroupByFilter reference semantics: cmp on the op's aggregate, then the
// k largest/smallest survivors (ties broken by key ascending here — the
// contract leaves tie order unspecified). Cap applies to rows returned.
inline void apply_group_filter_host(GroupByResidentResult& r, const GroupByFilter& f,
                               FilterAgg agg, std::size_t max_groups, const char* op) {
    r.groups_total = r.keys.size();
    if (!f.active()) return;
    const bool is_f64 = (agg == FilterAgg::SumF64);
    auto agg_i64 = [&](std::size_t i) -> std::int64_t {
        return agg == FilterAgg::SumI64 ? r.sums[i] : r.counts[i];
    };
    auto keep = [&](std::size_t i) -> bool {
        if (f.cmp == GroupByFilter::Cmp::None) return true;
        if (is_f64) {
            const double a = r.sums_f64[i], t = f.threshold_f64;
            switch (f.cmp) {
                case GroupByFilter::Cmp::GT: return a >  t;
                case GroupByFilter::Cmp::GE: return a >= t;
                case GroupByFilter::Cmp::LT: return a <  t;
                case GroupByFilter::Cmp::LE: return a <= t;
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
    if (f.topk != 0 && f.topk < idx.size()) {
        auto better = [&](std::size_t a, std::size_t b) {   // strict "a ranks before b"
            if (is_f64) {
                const double x = r.sums_f64[a], y = r.sums_f64[b];
                if (x != y) return f.topk_desc ? (x > y) : (x < y);
            } else {
                const std::int64_t x = agg_i64(a), y = agg_i64(b);
                if (x != y) return f.topk_desc ? (x > y) : (x < y);
            }
            return r.keys[a] < r.keys[b];
        };
        std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(f.topk),
                          idx.end(), better);
        idx.resize(f.topk);
    } else if (f.topk != 0) {
        auto better = [&](std::size_t a, std::size_t b) {
            if (is_f64) {
                const double x = r.sums_f64[a], y = r.sums_f64[b];
                if (x != y) return f.topk_desc ? (x > y) : (x < y);
            } else {
                const std::int64_t x = agg_i64(a), y = agg_i64(b);
                if (x != y) return f.topk_desc ? (x > y) : (x < y);
            }
            return r.keys[a] < r.keys[b];
        };
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
    if (!r.sums.empty()) o.sums.reserve(idx.size());
    if (!r.sums_f64.empty()) o.sums_f64.reserve(idx.size());
    for (std::size_t i : idx) {
        o.keys.push_back(r.keys[i]);
        o.counts.push_back(r.counts[i]);
        if (!r.sums.empty()) o.sums.push_back(r.sums[i]);
        if (!r.sums_f64.empty()) o.sums_f64.push_back(r.sums_f64[i]);
    }
    r = std::move(o);
}

} // namespace gpudb
