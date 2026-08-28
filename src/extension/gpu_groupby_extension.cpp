// gpu_groupby_extension.cpp — resident GROUP BY / top-k table functions (v0.6).
//
//   SELECT key, sum, count FROM gpu_groupby_sum_resident('p');
//   SELECT key, sum, count FROM gpu_groupby_sum_resident_f64('pf');
//   SELECT key, count      FROM gpu_groupby_count_resident('p');   -- or a bare column
//   SELECT idx, value      FROM gpu_topk_resident('c', 10, 'desc');     -- BIGINT column
//   SELECT idx, value      FROM gpu_topk_resident_f64('cf', 10, 'desc'); -- DOUBLE column
//   -- HAVING / top-k of groups evaluated ON THE DEVICE: only survivors come back
//   SELECT * FROM gpu_groupby_sum_resident_having('p', '>', 300);        -- HAVING sum > 300
//   SELECT * FROM gpu_groupby_sum_resident_f64_having('pf', '>=', 1e6);
//   SELECT * FROM gpu_groupby_count_resident_having('p', '<', 5);        -- HAVING count(*) < 5
//   SELECT * FROM gpu_groupby_sum_resident_topk('p', 10, 'desc');        -- ORDER BY sum DESC LIMIT 10
//   SELECT * FROM gpu_groupby_sum_resident_f64_topk('pf', 10, 'asc');
//   SELECT * FROM gpu_groupby_count_resident_topk('p', 10, 'desc');      -- ORDER BY count DESC LIMIT 10
//   (cmp is one of '>', '>=', '<', '<='; the cap applies to the rows returned;
//    topk output is ordered by the aggregate, ties in an unspecified order)
//
// The GPU produces the (key, aggregate) rows; everything downstream — WHERE,
// ORDER BY, LIMIT, further joins — is ordinary DuckDB SQL over a small
// result. Same statement-sequencing caveat as gpu_join_rows_resident: the
// uploads must run as earlier statements on the same connection (a table
// function's bind/init cannot be ordered after an aggregate in the same
// statement). Output rows are capped at GPUDB_GROUPBY_ROWS_MAX_M million
// groups (default 100) with a clean error naming the actual count.

#include "gpu_groupby_extension.hpp"
#include "gpu_join_extension.hpp"   // resident registry bridge
#include "gpu_backend.hpp"

#if defined(GPUDB_C_STRUCT_ABI)
DUCKDB_EXTENSION_EXTERN
#endif

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpudb_ext {
namespace {

std::size_t groupby_rows_cap() {
    static const std::size_t cap = [] {
        unsigned long long m = 100;   // 100M groups ≈ 2.4 GB of (key,sum,count)
        if (const char* s = std::getenv("GPUDB_GROUPBY_ROWS_MAX_M")) {
            char* end = nullptr;
            errno = 0;
            const unsigned long long v = std::strtoull(s, &end, 10);
            if (errno == 0 && end && end != s && *end == '\0' && v > 0 &&
                v <= static_cast<unsigned long long>(SIZE_MAX / 1000000)) {
                m = v;
            } else {
                std::fprintf(stderr,
                    "[gpudb] ignoring GPUDB_GROUPBY_ROWS_MAX_M='%s' (not a positive "
                    "integer in range); using %llu\n", s, m);
            }
        }
        return static_cast<std::size_t>(m) * 1000000;
    }();
    return cap;
}

std::string value_to_string(duckdb_value v) {
    char* s = duckdb_get_varchar(v);
    std::string out = s ? s : "";
    if (s) duckdb_free(s);
    return out;
}

// Resolve 'p' -> (p.k, p.v) for pair ops; for keys-only ops accept either a
// pair name (use p.k) or a bare resident column name. Caller holds the mutex.
struct Resolved {
    gpudb::ResidentColumn* keys = nullptr;
    gpudb::ResidentColumn* vals = nullptr;
};

Resolved resolve_pair(const std::string& name, bool need_vals, const char* fn) {
    Resolved r;
    r.keys = find_resident_column(name + ".k");
    r.vals = find_resident_column(name + ".v");
    if (r.keys && (r.vals || !need_vals)) return r;
    if (!need_vals) {
        r.keys = find_resident_column(name);
        if (r.keys) return r;
    }
    throw std::runtime_error(
        std::string(fn) + ": no resident " + (need_vals ? "pair" : "column") +
        " named '" + name + "' — create it with gpu_upload_pair" +
        (need_vals ? "" : " or gpu_upload"));
}

// ---------------------------------------------------------------------------
// GROUP BY table functions
// ---------------------------------------------------------------------------

enum class GbOp : std::uint8_t { SumI64, SumF64, Count };
enum class GbForm : std::uint8_t { Plain, Having, TopK };

struct GbBindData {
    std::string name;
    GbOp op = GbOp::SumI64;
    gpudb::GroupByFilter filter;
};

struct GbInitData {
    gpudb::GroupByResidentResult res;
    GbOp op = GbOp::SumI64;
    std::size_t offset = 0;
};

const char* gb_fn_name(GbOp op, GbForm form = GbForm::Plain) {
    switch (op) {
        case GbOp::SumI64: return form == GbForm::Having ? "gpu_groupby_sum_resident_having"
                                : form == GbForm::TopK  ? "gpu_groupby_sum_resident_topk"
                                                        : "gpu_groupby_sum_resident";
        case GbOp::SumF64: return form == GbForm::Having ? "gpu_groupby_sum_resident_f64_having"
                                : form == GbForm::TopK  ? "gpu_groupby_sum_resident_f64_topk"
                                                        : "gpu_groupby_sum_resident_f64";
        case GbOp::Count:  return form == GbForm::Having ? "gpu_groupby_count_resident_having"
                                : form == GbForm::TopK  ? "gpu_groupby_count_resident_topk"
                                                        : "gpu_groupby_count_resident";
    }
    return "gpu_groupby_resident";
}

// Parse the extra arguments of the _having / _topk forms into a GroupByFilter.
// Returns false after setting a bind error.
template <GbOp OP, GbForm FORM>
bool gb_parse_filter(duckdb_bind_info info, gpudb::GroupByFilter& f) {
    const char* fn = gb_fn_name(OP, FORM);
    if (FORM == GbForm::Plain) return true;
    duckdb_value a1 = duckdb_bind_get_parameter(info, 1);
    duckdb_value a2 = duckdb_bind_get_parameter(info, 2);
    if (!a1 || !a2 || duckdb_is_null_value(a1) || duckdb_is_null_value(a2)) {
        if (a1) duckdb_destroy_value(&a1);
        if (a2) duckdb_destroy_value(&a2);
        duckdb_bind_set_error(info, (std::string(fn) + ": arguments may not be NULL").c_str());
        return false;
    }
    bool ok = true;
    std::string err;
    if (FORM == GbForm::Having) {
        std::string cmp = value_to_string(a1);
        if      (cmp == ">")  f.cmp = gpudb::GroupByFilter::Cmp::GT;
        else if (cmp == ">=") f.cmp = gpudb::GroupByFilter::Cmp::GE;
        else if (cmp == "<")  f.cmp = gpudb::GroupByFilter::Cmp::LT;
        else if (cmp == "<=") f.cmp = gpudb::GroupByFilter::Cmp::LE;
        else { ok = false; err = std::string(fn) + ": comparison must be one of '>', '>=', '<', '<='"; }
        if (OP == GbOp::SumF64) f.threshold_f64 = duckdb_get_double(a2);
        else                    f.threshold_i64 = duckdb_get_int64(a2);
    } else {
        const std::int64_t k = duckdb_get_int64(a1);
        std::string order = value_to_string(a2);
        for (auto& ch : order) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        if (k < 1) { ok = false; err = std::string(fn) + ": k must be >= 1"; }
        else if (order == "desc") f.topk_desc = true;
        else if (order == "asc")  f.topk_desc = false;
        else { ok = false; err = std::string(fn) + ": order must be 'asc' or 'desc'"; }
        if (ok) f.topk = static_cast<std::size_t>(k);
    }
    duckdb_destroy_value(&a1);
    duckdb_destroy_value(&a2);
    if (!ok) duckdb_bind_set_error(info, err.c_str());
    return ok;
}

template <GbOp OP, GbForm FORM>
void gb_bind(duckdb_bind_info info) {
    duckdb_value nv = duckdb_bind_get_parameter(info, 0);
    if (!nv || duckdb_is_null_value(nv)) {
        if (nv) duckdb_destroy_value(&nv);
        duckdb_bind_set_error(info, (std::string(gb_fn_name(OP, FORM)) + ": name may not be NULL").c_str());
        return;
    }
    auto* bind = new GbBindData();
    bind->name = value_to_string(nv);
    bind->op = OP;
    duckdb_destroy_value(&nv);
    if (!gb_parse_filter<OP, FORM>(info, bind->filter)) { delete bind; return; }

    duckdb_logical_type bigint = duckdb_create_logical_type(DUCKDB_TYPE_BIGINT);
    duckdb_logical_type dbl    = duckdb_create_logical_type(DUCKDB_TYPE_DOUBLE);
    duckdb_bind_add_result_column(info, "key", bigint);
    if (OP == GbOp::SumI64) duckdb_bind_add_result_column(info, "sum", bigint);
    if (OP == GbOp::SumF64) duckdb_bind_add_result_column(info, "sum", dbl);
    duckdb_bind_add_result_column(info, "count", bigint);
    duckdb_destroy_logical_type(&bigint);
    duckdb_destroy_logical_type(&dbl);
    duckdb_bind_set_bind_data(info, bind, [](void* p) { delete static_cast<GbBindData*>(p); });
}

void gb_init(duckdb_init_info info) {
    auto* bind = static_cast<GbBindData*>(duckdb_init_get_bind_data(info));
    auto* init = new GbInitData();
    init->op = bind->op;
    const GbForm form = bind->filter.topk != 0 ? GbForm::TopK
                      : bind->filter.cmp != gpudb::GroupByFilter::Cmp::None ? GbForm::Having
                      : GbForm::Plain;
    const char* fn = gb_fn_name(bind->op, form);
    try {
        std::lock_guard<std::mutex> lock(resident_mutex());
        const bool need_vals = bind->op != GbOp::Count;
        Resolved cols = resolve_pair(bind->name, need_vals, fn);
        auto& agg = resident_aggregator();
        const std::size_t cap = groupby_rows_cap();
        switch (bind->op) {
            case GbOp::SumI64:
                if (cols.vals->dtype() != gpudb::Dtype::I64)
                    throw std::runtime_error(std::string(fn) +
                        ": '" + bind->name + ".v' is DOUBLE — use " + gb_fn_name(GbOp::SumF64, form));
                init->res = agg.groupby_sum_resident_i64(*cols.keys, *cols.vals, cap, bind->filter);
                break;
            case GbOp::SumF64:
                if (cols.vals->dtype() != gpudb::Dtype::F64)
                    throw std::runtime_error(std::string(fn) +
                        ": '" + bind->name + ".v' is BIGINT — use " + gb_fn_name(GbOp::SumI64, form));
                init->res = agg.groupby_sum_resident_f64(*cols.keys, *cols.vals, cap, bind->filter);
                break;
            case GbOp::Count:
                if (cols.keys->dtype() != gpudb::Dtype::I64)
                    throw std::runtime_error(std::string(fn) +
                        ": keys must be a BIGINT resident column");
                init->res = agg.groupby_count_resident(*cols.keys, cap, bind->filter);
                break;
        }
        const auto& d = agg.last_decision();
        char buf[320];
        std::snprintf(buf, sizeof(buf),
            "op=%s backend=%s reason=%s rows_in=%zu groups=%zu rows_out=%zu "
            "wall_ms=%.3f kernel_ms=%.3f transfer_ms=%.3f",
            fn + 4 /* strip "gpu_" */, gpudb::to_string(d.chosen),
            gpudb::to_string(d.reason), init->res.rows_in, init->res.groups_total,
            init->res.keys.size(),
            init->res.wall_ms, init->res.kernel_ms, init->res.transfer_ms);
        resident_set_last_stats(buf);
    } catch (const std::exception& e) {
        delete init;
        duckdb_init_set_error(info, e.what());
        return;
    }
    duckdb_init_set_init_data(info, init, [](void* p) { delete static_cast<GbInitData*>(p); });
}

void gb_function(duckdb_function_info info, duckdb_data_chunk output) {
    auto* init = static_cast<GbInitData*>(duckdb_function_get_init_data(info));
    if (!init) return;
    const std::size_t remaining = init->res.keys.size() - init->offset;
    if (remaining == 0) return;

    constexpr idx_t kChunk = 2048;
    const idx_t out_n = static_cast<idx_t>(std::min<std::size_t>(remaining, kChunk));
    const std::size_t off = init->offset;

    auto* key_out = static_cast<std::int64_t*>(
        duckdb_vector_get_data(duckdb_data_chunk_get_vector(output, 0)));
    for (idx_t i = 0; i < out_n; ++i) key_out[i] = init->res.keys[off + i];

    idx_t col = 1;
    if (init->op == GbOp::SumI64) {
        auto* s = static_cast<std::int64_t*>(
            duckdb_vector_get_data(duckdb_data_chunk_get_vector(output, col++)));
        for (idx_t i = 0; i < out_n; ++i) s[i] = init->res.sums[off + i];
    } else if (init->op == GbOp::SumF64) {
        auto* s = static_cast<double*>(
            duckdb_vector_get_data(duckdb_data_chunk_get_vector(output, col++)));
        for (idx_t i = 0; i < out_n; ++i) s[i] = init->res.sums_f64[off + i];
    }
    auto* c = static_cast<std::int64_t*>(
        duckdb_vector_get_data(duckdb_data_chunk_get_vector(output, col)));
    for (idx_t i = 0; i < out_n; ++i) c[i] = init->res.counts[off + i];

    duckdb_data_chunk_set_size(output, out_n);
    init->offset += static_cast<std::size_t>(out_n);
}

// ---------------------------------------------------------------------------
// gpu_topk_resident(name VARCHAR, k BIGINT, order VARCHAR) -> (idx BIGINT, value BIGINT)
// gpu_topk_resident_f64(...)                                 -> (idx BIGINT, value DOUBLE)
// ---------------------------------------------------------------------------

struct TkBindData {
    std::string name;
    std::int64_t k = 0;
    bool descending = false;
    gpudb::Dtype dtype = gpudb::Dtype::I64;
};

struct TkInitData {
    gpudb::TopKResult res;
    gpudb::Dtype dtype = gpudb::Dtype::I64;
    std::size_t offset = 0;
};

// The value column's type is fixed per function (BIGINT / DOUBLE) so bind
// never has to look at the registry — the column may not exist yet when a
// same-statement upload feeds it. The dtype is checked at init.
template <gpudb::Dtype DT>
void tk_bind(duckdb_bind_info info) {
    duckdb_value nv = duckdb_bind_get_parameter(info, 0);
    duckdb_value kv = duckdb_bind_get_parameter(info, 1);
    duckdb_value ov = duckdb_bind_get_parameter(info, 2);
    if (!nv || !kv || !ov || duckdb_is_null_value(nv) || duckdb_is_null_value(kv) ||
        duckdb_is_null_value(ov)) {
        if (nv) duckdb_destroy_value(&nv);
        if (kv) duckdb_destroy_value(&kv);
        if (ov) duckdb_destroy_value(&ov);
        duckdb_bind_set_error(info, "gpu_topk_resident: name, k and order may not be NULL");
        return;
    }
    auto* bind = new TkBindData();
    bind->name = value_to_string(nv);
    bind->k = duckdb_get_int64(kv);
    bind->dtype = DT;
    std::string order = value_to_string(ov);
    duckdb_destroy_value(&nv);
    duckdb_destroy_value(&kv);
    duckdb_destroy_value(&ov);
    for (auto& ch : order) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    if      (order == "asc")  bind->descending = false;
    else if (order == "desc") bind->descending = true;
    else {
        delete bind;
        duckdb_bind_set_error(info, "gpu_topk_resident: order must be 'asc' or 'desc'");
        return;
    }
    if (bind->k < 0) {
        delete bind;
        duckdb_bind_set_error(info, "gpu_topk_resident: k must be >= 0");
        return;
    }

    duckdb_logical_type bigint = duckdb_create_logical_type(DUCKDB_TYPE_BIGINT);
    duckdb_logical_type dbl    = duckdb_create_logical_type(DUCKDB_TYPE_DOUBLE);
    duckdb_bind_add_result_column(info, "idx", bigint);
    duckdb_bind_add_result_column(info, "value", DT == gpudb::Dtype::F64 ? dbl : bigint);
    duckdb_destroy_logical_type(&bigint);
    duckdb_destroy_logical_type(&dbl);
    duckdb_bind_set_bind_data(info, bind, [](void* p) { delete static_cast<TkBindData*>(p); });
}

void tk_init(duckdb_init_info info) {
    auto* bind = static_cast<TkBindData*>(duckdb_init_get_bind_data(info));
    auto* init = new TkInitData();
    init->dtype = bind->dtype;
    try {
        std::lock_guard<std::mutex> lock(resident_mutex());
        // A pair name ranks its payload ('p' -> p.v); otherwise a bare column.
        // Never silently fall back to a pair's key column.
        gpudb::ResidentColumn* target = find_resident_column(bind->name + ".v");
        if (!target) target = find_resident_column(bind->name);
        if (!target)
            throw std::runtime_error(
                "gpu_topk_resident: no resident pair or column named '" + bind->name +
                "' — create it with gpu_upload_pair or gpu_upload");
        if (target->dtype() != bind->dtype)
            throw std::runtime_error(
                std::string(bind->dtype == gpudb::Dtype::F64 ? "gpu_topk_resident_f64" : "gpu_topk_resident") +
                ": '" + bind->name + "' is " +
                (target->dtype() == gpudb::Dtype::F64 ? "DOUBLE — use gpu_topk_resident_f64"
                                                       : "BIGINT — use gpu_topk_resident"));
        auto& agg = resident_aggregator();
        init->res = agg.topk_resident(*target, static_cast<std::size_t>(bind->k),
                                      bind->descending);
        const auto& d = agg.last_decision();
        char buf[320];
        std::snprintf(buf, sizeof(buf),
            "op=topk_resident backend=%s reason=%s rows_in=%zu k=%zu "
            "wall_ms=%.3f kernel_ms=%.3f transfer_ms=%.3f",
            gpudb::to_string(d.chosen), gpudb::to_string(d.reason),
            init->res.rows_in, init->res.idx.size(),
            init->res.wall_ms, init->res.kernel_ms, init->res.transfer_ms);
        resident_set_last_stats(buf);
    } catch (const std::exception& e) {
        delete init;
        duckdb_init_set_error(info, e.what());
        return;
    }
    duckdb_init_set_init_data(info, init, [](void* p) { delete static_cast<TkInitData*>(p); });
}

void tk_function(duckdb_function_info info, duckdb_data_chunk output) {
    auto* init = static_cast<TkInitData*>(duckdb_function_get_init_data(info));
    if (!init) return;
    const std::size_t remaining = init->res.idx.size() - init->offset;
    if (remaining == 0) return;

    constexpr idx_t kChunk = 2048;
    const idx_t out_n = static_cast<idx_t>(std::min<std::size_t>(remaining, kChunk));
    const std::size_t off = init->offset;

    auto* idx_out = static_cast<std::int64_t*>(
        duckdb_vector_get_data(duckdb_data_chunk_get_vector(output, 0)));
    for (idx_t i = 0; i < out_n; ++i) idx_out[i] = init->res.idx[off + i];
    duckdb_vector vv = duckdb_data_chunk_get_vector(output, 1);
    if (init->dtype == gpudb::Dtype::F64) {
        auto* v = static_cast<double*>(duckdb_vector_get_data(vv));
        for (idx_t i = 0; i < out_n; ++i) v[i] = init->res.values_f64[off + i];
    } else {
        auto* v = static_cast<std::int64_t*>(duckdb_vector_get_data(vv));
        for (idx_t i = 0; i < out_n; ++i) v[i] = init->res.values_i64[off + i];
    }
    duckdb_data_chunk_set_size(output, out_n);
    init->offset += static_cast<std::size_t>(out_n);
}

// ---------------------------------------------------------------------------

void register_table_fn(duckdb_connection con, const char* name,
                       const std::vector<duckdb_type>& params,
                       duckdb_table_function_bind_t bind,
                       duckdb_table_function_init_t init,
                       duckdb_table_function_t fn) {
    duckdb_table_function tf = duckdb_create_table_function();
    duckdb_table_function_set_name(tf, name);
    for (duckdb_type t : params) {
        duckdb_logical_type lt = duckdb_create_logical_type(t);
        duckdb_table_function_add_parameter(tf, lt);
        duckdb_destroy_logical_type(&lt);
    }
    duckdb_table_function_set_bind(tf, bind);
    duckdb_table_function_set_init(tf, init);
    duckdb_table_function_set_function(tf, fn);
    if (duckdb_register_table_function(con, tf) == DuckDBError) {
        duckdb_destroy_table_function(&tf);
        throw std::runtime_error(std::string(name) + " registration failed");
    }
    duckdb_destroy_table_function(&tf);
}

} // namespace

void register_gpu_groupby(duckdb_connection con) {
    register_table_fn(con, "gpu_groupby_sum_resident", {DUCKDB_TYPE_VARCHAR},
                      gb_bind<GbOp::SumI64, GbForm::Plain>, gb_init, gb_function);
    register_table_fn(con, "gpu_groupby_sum_resident_f64", {DUCKDB_TYPE_VARCHAR},
                      gb_bind<GbOp::SumF64, GbForm::Plain>, gb_init, gb_function);
    register_table_fn(con, "gpu_groupby_count_resident", {DUCKDB_TYPE_VARCHAR},
                      gb_bind<GbOp::Count, GbForm::Plain>, gb_init, gb_function);
    // HAVING on the aggregate (cmp VARCHAR, threshold) — evaluated on the device
    register_table_fn(con, "gpu_groupby_sum_resident_having",
                      {DUCKDB_TYPE_VARCHAR, DUCKDB_TYPE_VARCHAR, DUCKDB_TYPE_BIGINT},
                      gb_bind<GbOp::SumI64, GbForm::Having>, gb_init, gb_function);
    register_table_fn(con, "gpu_groupby_sum_resident_f64_having",
                      {DUCKDB_TYPE_VARCHAR, DUCKDB_TYPE_VARCHAR, DUCKDB_TYPE_DOUBLE},
                      gb_bind<GbOp::SumF64, GbForm::Having>, gb_init, gb_function);
    register_table_fn(con, "gpu_groupby_count_resident_having",
                      {DUCKDB_TYPE_VARCHAR, DUCKDB_TYPE_VARCHAR, DUCKDB_TYPE_BIGINT},
                      gb_bind<GbOp::Count, GbForm::Having>, gb_init, gb_function);
    // top-k of groups by the aggregate (k BIGINT, order VARCHAR)
    register_table_fn(con, "gpu_groupby_sum_resident_topk",
                      {DUCKDB_TYPE_VARCHAR, DUCKDB_TYPE_BIGINT, DUCKDB_TYPE_VARCHAR},
                      gb_bind<GbOp::SumI64, GbForm::TopK>, gb_init, gb_function);
    register_table_fn(con, "gpu_groupby_sum_resident_f64_topk",
                      {DUCKDB_TYPE_VARCHAR, DUCKDB_TYPE_BIGINT, DUCKDB_TYPE_VARCHAR},
                      gb_bind<GbOp::SumF64, GbForm::TopK>, gb_init, gb_function);
    register_table_fn(con, "gpu_groupby_count_resident_topk",
                      {DUCKDB_TYPE_VARCHAR, DUCKDB_TYPE_BIGINT, DUCKDB_TYPE_VARCHAR},
                      gb_bind<GbOp::Count, GbForm::TopK>, gb_init, gb_function);
    register_table_fn(con, "gpu_topk_resident",
                      {DUCKDB_TYPE_VARCHAR, DUCKDB_TYPE_BIGINT, DUCKDB_TYPE_VARCHAR},
                      tk_bind<gpudb::Dtype::I64>, tk_init, tk_function);
    register_table_fn(con, "gpu_topk_resident_f64",
                      {DUCKDB_TYPE_VARCHAR, DUCKDB_TYPE_BIGINT, DUCKDB_TYPE_VARCHAR},
                      tk_bind<gpudb::Dtype::F64>, tk_init, tk_function);
    std::fprintf(stderr,
        "[gpudb] registered gpu_groupby_{sum,sum_f64,count}_resident[_having|_topk] + gpu_topk_resident[_f64]\n");
}

} // namespace gpudb_ext
