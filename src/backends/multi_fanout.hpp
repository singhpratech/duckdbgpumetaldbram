// multi_fanout.hpp — the payload fan-out shared by every backend that answers
// groupby_exact_masked_multi (v0.7 §4.9) by running ONE single-payload pass per
// column and aligning the passes afterwards.
//
// It was the body of Aggregator::groupby_exact_masked_multi's default. Stage D1
// (docs/RESIDENT_COLUMNS_DESIGN.md §7) needs the same alignment for passes that
// each carry their lane's index vector, so the alignment moved here and the
// caller supplies the pass: `pass(p, GroupByFilter)` runs payload p and returns
// its GroupByResidentResult. Every pass sees the same keys and the same mask,
// hence the same groups in the same order; under a filter the non-primary
// passes are looked up by key.
#pragma once
#include "gpu_backend.hpp"
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpudb {

template <class Pass>
inline std::vector<GroupByResidentResult> multi_payload_fanout(
    const MultiPayload* pays, std::size_t n_pays, std::size_t filter_payload,
    const GroupByFilter& filter, const char* op, Pass pass) {
    if (n_pays == 0 || !pays) throw std::runtime_error(std::string(op) + ": no payload columns");
    const bool filtered = filter.active();
    const std::size_t fp = filtered ? filter_payload : 0;
    if (fp >= n_pays) throw std::runtime_error(std::string(op) + ": filter payload out of range");
    bool others = false;
    for (std::size_t p = 0; p < n_pays; ++p) if (p != fp && pays[p].columns) others = true;
    std::vector<GroupByResidentResult> out(n_pays);
    auto rows_of = [](const GroupByResidentResult& r) {
        return std::max({r.keys.size(), r.sums.size(), r.counts.size(),
                         r.counts_star.size(), r.mins.size(), r.maxs.size()});
    };
    // primary pass: the filtered payload, with the shared columns
    GroupByFilter f0 = filter;
    f0.columns = (pays[fp].columns & ~std::uint32_t{0x9}) | (filter.columns & 0x9u) |
                 ((filtered && others) ? 1u : 0u);
    if (f0.columns == 0) f0.columns = 1u << 3;
    GroupByResidentResult prim = pass(fp, f0);
    const std::size_t rows = rows_of(prim);
    double wall = prim.wall_ms, kernel = prim.kernel_ms;
    for (std::size_t p = 0; p < n_pays; ++p) {
        if (p == fp || !pays[p].columns) continue;
        GroupByFilter fq;
        fq.columns = (pays[p].columns & ~std::uint32_t{0x9}) | (filtered ? 1u : 0u);
        GroupByResidentResult full = pass(p, fq);
        wall += full.wall_ms; kernel += full.kernel_ms;
        if (!filtered) {
            if (rows_of(full) != rows)
                throw std::runtime_error(std::string(op) + ": payload passes disagree on the group count");
            full.keys.clear(); full.key_null.clear(); full.counts_star.clear();
            out[p] = std::move(full);
            continue;
        }
        // survivors -> positions in the full (key-sorted, NULL group last) result
        const bool full_null = !full.key_null.empty() && full.key_null.back();
        const std::size_t n_valid = full.keys.size() - (full_null ? 1 : 0);
        GroupByResidentResult g;
        auto take = [](const std::vector<std::int64_t>& src, std::vector<std::int64_t>& dst, std::size_t at) {
            if (!src.empty()) dst.push_back(src[at]);
        };
        for (std::size_t i = 0; i < rows; ++i) {
            std::size_t at;
            if (!prim.key_null.empty() && prim.key_null[i]) {
                if (!full_null) throw std::runtime_error(std::string(op) + ": NULL-key group missing from a payload pass");
                at = full.keys.size() - 1;
            } else {
                const auto first = full.keys.begin(), last = full.keys.begin() + static_cast<std::ptrdiff_t>(n_valid);
                const auto it = std::lower_bound(first, last, prim.keys[i]);
                if (it == last || *it != prim.keys[i])
                    throw std::runtime_error(std::string(op) + ": a surviving group is missing from a payload pass");
                at = static_cast<std::size_t>(it - first);
            }
            take(full.sums, g.sums, at);   take(full.sums_hi, g.sums_hi, at);
            take(full.counts, g.counts, at);
            take(full.mins, g.mins, at);   take(full.maxs, g.maxs, at);
        }
        out[p] = std::move(g);
    }
    if (!(filter.columns & 1u)) { prim.keys.clear(); prim.key_null.clear(); }
    prim.wall_ms = wall;
    prim.kernel_ms = kernel;
    out[fp] = std::move(prim);
    return out;
}

} // namespace gpudb
