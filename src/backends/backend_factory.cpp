#include "gpu_backend.hpp"
#include <algorithm>
#include "backend_internal.hpp"

#include <stdexcept>

namespace gpudb {

const char* to_string(Backend b) noexcept {
    switch (b) {
        case Backend::CPU:   return "CPU";
        case Backend::CUDA:  return "CUDA";
        case Backend::METAL: return "Metal";
    }
    return "?";
}

// Default for backends that haven't opted into the fused resident join yet
// (CUDA today). Non-pure so adding the op doesn't break their builds.
JoinAggResult Aggregator::join_sum_resident_i64(const ResidentColumn&,
                                                const ResidentColumn&,
                                                const ResidentColumn&,
                                                JoinKind) {
    throw std::runtime_error("join_sum_resident_i64: not implemented on this backend");
}

JoinAggResult Aggregator::join_sum_resident_f64(const ResidentColumn&,
                                                const ResidentColumn&,
                                                const ResidentColumn&,
                                                JoinKind) {
    throw std::runtime_error("join_sum_resident_f64: not implemented on this backend");
}

JoinRowsResult Aggregator::join_rows_resident(const ResidentColumn&,
                                              const ResidentColumn&,
                                              JoinKind, std::size_t) {
    throw std::runtime_error("join_rows_resident: not implemented on this backend");
}

// Same opt-in rule for the v0.6 resident GROUP BY / top-k ops.
GroupByResidentResult Aggregator::groupby_sum_resident_i64(const ResidentColumn&,
                                                           const ResidentColumn&,
                                                           std::size_t, const GroupByFilter&) {
    throw std::runtime_error("groupby_sum_resident_i64: not implemented on this backend");
}

GroupByResidentResult Aggregator::groupby_sum_resident_f64(const ResidentColumn&,
                                                           const ResidentColumn&,
                                                           std::size_t, const GroupByFilter&) {
    throw std::runtime_error("groupby_sum_resident_f64: not implemented on this backend");
}

GroupByResidentResult Aggregator::groupby_count_resident(const ResidentColumn&,
                                                         std::size_t, const GroupByFilter&) {
    throw std::runtime_error("groupby_count_resident: not implemented on this backend");
}

TopKResult Aggregator::topk_resident(const ResidentColumn&, std::size_t, bool) {
    throw std::runtime_error("topk_resident: not implemented on this backend");
}

// v0.7 milestone 3 exact path: same opt-in rule (CPU is the reference).
Aggregator::ResidentPair Aggregator::upload_pair_exact(const KvSpan*, std::size_t, Dtype) {
    throw std::runtime_error("upload_pair_exact: not implemented on this backend");
}

GroupByResidentResult Aggregator::groupby_exact_resident(const ResidentColumn&,
                                                         const ResidentColumn*,
                                                         std::size_t, const GroupByFilter&) {
    throw std::runtime_error("groupby_exact_resident: not implemented on this backend");
}

std::vector<std::unique_ptr<ResidentColumn>>
Aggregator::upload_rows_exact(const RowSpan*, std::size_t, const Dtype*, std::size_t) {
    throw std::runtime_error("upload_rows_exact: not implemented on this backend");
}

GroupByResidentResult Aggregator::groupby_exact_masked_resident(const ResidentColumn& keys,
                                                                const ResidentColumn* vals,
                                                                const Predicate*, std::size_t n_preds,
                                                                std::size_t max_groups,
                                                                const GroupByFilter& filter) {
    if (n_preds == 0) return groupby_exact_resident(keys, vals, max_groups, filter);
    throw std::runtime_error("groupby_exact_masked_resident: not implemented on this backend");
}

std::vector<GroupByResidentResult> Aggregator::groupby_exact_masked_multi(
    const ResidentColumn& keys, const MultiPayload* pays, std::size_t n_pays, std::size_t filter_payload,
    const Predicate* preds, std::size_t n_preds, std::size_t max_groups, const GroupByFilter& filter) {
    if (n_pays == 0 || !pays) throw std::runtime_error("groupby_exact_masked_multi: no payload columns");
    const bool filtered = filter.active();
    const std::size_t fp = filtered ? filter_payload : 0;
    if (fp >= n_pays) throw std::runtime_error("groupby_exact_masked_multi: filter payload out of range");
    bool others = false;
    for (std::size_t p = 0; p < n_pays; ++p) if (p != fp && pays[p].columns) others = true;
    std::vector<GroupByResidentResult> out(n_pays);
    auto rows_of = [](const GroupByResidentResult& r) {
        return std::max({r.keys.size(), r.sums.size(), r.counts.size(), r.counts_star.size(), r.mins.size(), r.maxs.size()});
    };
    // primary pass: the filtered payload, with the shared columns
    GroupByFilter f0 = filter;
    f0.columns = (pays[fp].columns & ~std::uint32_t{0x9}) | (filter.columns & 0x9u) | ((filtered && others) ? 1u : 0u);
    if (f0.columns == 0) f0.columns = 1u << 3;
    GroupByResidentResult prim = groupby_exact_masked_resident(keys, pays[fp].vals, preds, n_preds, max_groups, f0);
    const std::size_t rows = rows_of(prim);
    double wall = prim.wall_ms, kernel = prim.kernel_ms;
    for (std::size_t p = 0; p < n_pays; ++p) {
        if (p == fp || !pays[p].columns) continue;
        GroupByFilter fq;
        fq.columns = (pays[p].columns & ~std::uint32_t{0x9}) | (filtered ? 1u : 0u);
        GroupByResidentResult full = groupby_exact_masked_resident(keys, pays[p].vals, preds, n_preds, max_groups, fq);
        wall += full.wall_ms; kernel += full.kernel_ms;
        if (!filtered) {
            if (rows_of(full) != rows)
                throw std::runtime_error("groupby_exact_masked_multi: payload passes disagree on the group count");
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
                if (!full_null) throw std::runtime_error("groupby_exact_masked_multi: NULL-key group missing from a payload pass");
                at = full.keys.size() - 1;
            } else {
                const auto first = full.keys.begin(), last = full.keys.begin() + static_cast<std::ptrdiff_t>(n_valid);
                const auto it = std::lower_bound(first, last, prim.keys[i]);
                if (it == last || *it != prim.keys[i])
                    throw std::runtime_error("groupby_exact_masked_multi: a surviving group is missing from a payload pass");
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

JoinMaterializeResult Aggregator::join_materialize(const ResidentColumn&, const ResidentColumn&,
                                                   const JoinLane*, std::size_t) {
    throw std::runtime_error("join_materialize: not implemented on this backend");
}

// Per-backend factory forward declarations (impls live in their respective TUs).
// These are declared in `gpudb` so that the hybrid planner TU (which lives
// in the same library) can call them without re-declaring.
// Forward declarations (impls live in their respective TUs)
std::unique_ptr<Aggregator> make_cpu_aggregator();
std::unique_ptr<GroupByAggregator> make_cpu_groupby_aggregator();
std::unique_ptr<WindowAggregator> make_cpu_window_aggregator();
std::unique_ptr<HashJoinProbe> make_cpu_hashjoin_probe();
#if GPUDB_HAVE_CUDA
std::unique_ptr<Aggregator> make_cuda_aggregator();
std::unique_ptr<GroupByAggregator> make_cuda_groupby_aggregator();
std::unique_ptr<HashJoinProbe> make_cuda_hashjoin_probe();
bool cuda_runtime_available() noexcept;
#endif
#if GPUDB_HAVE_METAL
std::unique_ptr<Aggregator> make_metal_aggregator();
std::unique_ptr<GroupByAggregator> make_metal_groupby_aggregator();
std::unique_ptr<WindowAggregator> make_metal_window_aggregator();
std::unique_ptr<HashJoinProbe> make_metal_hashjoin_probe();
bool metal_runtime_available() noexcept;
#endif

std::unique_ptr<Aggregator> make_aggregator(Backend b) {
    switch (b) {
        case Backend::CPU:
            return make_cpu_aggregator();
        case Backend::CUDA:
#if GPUDB_HAVE_CUDA
            return make_cuda_aggregator();
#else
            throw std::runtime_error("CUDA backend not compiled in");
#endif
        case Backend::METAL:
#if GPUDB_HAVE_METAL
            return make_metal_aggregator();
#else
            throw std::runtime_error("Metal backend not compiled in");
#endif
    }
    throw std::runtime_error("Unknown backend");
}

Backend default_backend() noexcept {
#if GPUDB_HAVE_CUDA
    if (cuda_runtime_available()) return Backend::CUDA;
#endif
#if GPUDB_HAVE_METAL
    if (metal_runtime_available()) return Backend::METAL;
#endif
    return Backend::CPU;
}

std::unique_ptr<GroupByAggregator> make_groupby_aggregator(Backend b) {
    switch (b) {
        case Backend::CPU:
            return make_cpu_groupby_aggregator();
        case Backend::CUDA:
#if GPUDB_HAVE_CUDA
            return make_cuda_groupby_aggregator();
#else
            throw std::runtime_error("CUDA backend not compiled in");
#endif
        case Backend::METAL:
#if GPUDB_HAVE_METAL
            return make_metal_groupby_aggregator();
#else
            throw std::runtime_error("Metal backend not compiled in");
#endif
    }
    throw std::runtime_error("Unknown backend");
}

std::unique_ptr<WindowAggregator> make_window_aggregator(Backend b) {
    switch (b) {
        case Backend::CPU:
            return make_cpu_window_aggregator();
        case Backend::CUDA:
#if GPUDB_HAVE_CUDA
            throw std::runtime_error(
                "CUDA window aggregator not implemented yet (Linux lane)");
#else
            throw std::runtime_error("CUDA backend not compiled in");
#endif
        case Backend::METAL:
#if GPUDB_HAVE_METAL
            return make_metal_window_aggregator();
#else
            throw std::runtime_error("Metal backend not compiled in");
#endif
    }
    throw std::runtime_error("Unknown backend");
}

std::unique_ptr<HashJoinProbe> make_hashjoin_probe(Backend b) {
    switch (b) {
        case Backend::CPU:
            return make_cpu_hashjoin_probe();
        case Backend::CUDA:
#if GPUDB_HAVE_CUDA
            return make_cuda_hashjoin_probe();
#else
            throw std::runtime_error("CUDA backend not compiled in");
#endif
        case Backend::METAL:
#if GPUDB_HAVE_METAL
            return make_metal_hashjoin_probe();
#else
            throw std::runtime_error("Metal backend not compiled in");
#endif
    }
    throw std::runtime_error("Unknown backend");
}

std::vector<Backend> available_backends() noexcept {
    std::vector<Backend> v{Backend::CPU};
#if GPUDB_HAVE_CUDA
    if (cuda_runtime_available()) v.push_back(Backend::CUDA);
#endif
#if GPUDB_HAVE_METAL
    if (metal_runtime_available()) v.push_back(Backend::METAL);
#endif
    return v;
}

} // namespace gpudb
