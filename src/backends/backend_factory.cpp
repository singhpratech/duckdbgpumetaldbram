#include "gpu_backend.hpp"
#include <algorithm>
#include "backend_internal.hpp"
#include "multi_fanout.hpp"

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

// v0.7 §4.12: the global masked aggregate — same opt-in rule.
GlobalAggResult Aggregator::aggregate_exact_masked(const MultiPayload*, std::size_t,
                                                   const Predicate*, std::size_t) {
    throw std::runtime_error("aggregate_exact_masked: not implemented on this backend");
}

std::vector<GroupByResidentResult> Aggregator::groupby_exact_masked_multi(
    const ResidentColumn& keys, const MultiPayload* pays, std::size_t n_pays, std::size_t filter_payload,
    const Predicate* preds, std::size_t n_preds, std::size_t max_groups, const GroupByFilter& filter) {
    static const char* op = "groupby_exact_masked_multi";
    // This default fans out to the single-payload op, which has no index
    // parameter: an index passed here would be silently dropped and the answer
    // would be wrong, so refuse instead (a backend with stage D overrides).
    for (std::size_t p = 0; p < n_pays; ++p)
        if (pays[p].index)
            throw std::runtime_error(std::string(op) + ": indexed payloads are not implemented on this backend");
    for (std::size_t q = 0; q < n_preds; ++q)
        if (preds[q].index)
            throw std::runtime_error(std::string(op) + ": indexed predicates are not implemented on this backend");
    return multi_payload_fanout(pays, n_pays, filter_payload, filter, op,
                                [&](std::size_t p, const GroupByFilter& f) {
                                    return groupby_exact_masked_resident(keys, pays[p].vals, preds,
                                                                         n_preds, max_groups, f);
                                });
}

JoinMaterializeResult Aggregator::join_materialize(const ResidentColumn&, const ResidentColumn&,
                                                   const JoinLane*, std::size_t) {
    throw std::runtime_error("join_materialize: not implemented on this backend");
}

// ---- stage D1 defaults: a backend without the mechanism keeps working ----
Aggregator::JoinIndexResult Aggregator::join_index(const ResidentColumn&, const ResidentColumn&,
                                                   const JoinLane&, const JoinLane*, std::size_t,
                                                   const ResidentColumn*) {
    throw std::runtime_error("join_index: not implemented on this backend");
}

// Delegates when nothing is actually indexed, so a caller may always reach for
// the indexed form and a backend without stage D still answers the shapes it
// could answer before. Anything genuinely indexed throws, and the SQL layer
// reads indexed_supported() before it plans one.
std::vector<GroupByResidentResult> Aggregator::groupby_exact_masked_multi_indexed(
    const IndexedColumn& keys, const MultiPayload* pays, std::size_t n_pays,
    std::size_t filter_payload, const Predicate* preds, std::size_t n_preds,
    std::size_t max_groups, const GroupByFilter& filter) {
    static const char* op = "groupby_exact_masked_multi_indexed";
    if (!keys.col) throw std::runtime_error(std::string(op) + ": no key column");
    bool indexed = keys.index != nullptr;
    for (std::size_t p = 0; p < n_pays && !indexed; ++p) indexed = pays[p].index != nullptr;
    for (std::size_t q = 0; q < n_preds && !indexed; ++q) indexed = preds[q].index != nullptr;
    if (indexed)
        throw std::runtime_error(std::string(op) + ": indexed lanes are not implemented on this backend");
    return groupby_exact_masked_multi(*keys.col, pays, n_pays, filter_payload,
                                      preds, n_preds, max_groups, filter);
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
