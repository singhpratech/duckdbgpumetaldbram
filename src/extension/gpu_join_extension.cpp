// gpu_join_extension.cpp — gpu_inner_join() table function for DuckDB.
//
// Usage:
//   SELECT probe_idx, build_idx
//   FROM gpu_inner_join([10, 20, 30], [20, 40, 99]);
//
// v1 contract matches HashJoinProbe: unique build keys, first match wins.

#include "gpu_join_extension.hpp"
#include "gpu_backend.hpp"

#if defined(GPUDB_C_STRUCT_ABI)
DUCKDB_EXTENSION_EXTERN
#endif

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpudb_ext {
namespace {

struct JoinBindData {
    std::vector<std::int64_t> build_keys;
    std::vector<std::int64_t> probe_keys;
};

struct JoinInitData {
    std::vector<std::int64_t> probe_indices;
    std::vector<std::int64_t> build_indices;
    std::size_t offset = 0;
};

gpudb::HybridHashJoinProbe& shared_hybrid_join() {
    static std::unique_ptr<gpudb::HybridHashJoinProbe> hj =
        gpudb::make_hybrid_hashjoin_probe();
    return *hj;
}

std::mutex& join_call_mutex() {
    static std::mutex m;
    return m;
}

std::vector<std::int64_t> list_to_i64(duckdb_value list_val) {
    const idx_t n = duckdb_get_list_size(list_val);
    std::vector<std::int64_t> out;
    out.reserve(static_cast<std::size_t>(n));
    for (idx_t i = 0; i < n; ++i) {
        duckdb_value child = duckdb_get_list_child(list_val, i);
        out.push_back(duckdb_get_int64(child));
        duckdb_destroy_value(&child);
    }
    return out;
}

void join_bind(duckdb_bind_info info) {
    if (duckdb_bind_get_parameter_count(info) != 2) {
        duckdb_bind_set_error(info, "gpu_inner_join expects two LIST(BIGINT) arguments");
        return;
    }

    duckdb_value build_list = duckdb_bind_get_parameter(info, 0);
    duckdb_value probe_list = duckdb_bind_get_parameter(info, 1);
    if (!build_list || !probe_list) {
        if (build_list) duckdb_destroy_value(&build_list);
        if (probe_list) duckdb_destroy_value(&probe_list);
        duckdb_bind_set_error(info, "gpu_inner_join: null argument");
        return;
    }

    auto* bind = new JoinBindData();
    try {
        bind->build_keys = list_to_i64(build_list);
        bind->probe_keys = list_to_i64(probe_list);
    } catch (...) {
        delete bind;
        duckdb_destroy_value(&build_list);
        duckdb_destroy_value(&probe_list);
        duckdb_bind_set_error(info, "gpu_inner_join: failed to read LIST arguments");
        return;
    }
    duckdb_destroy_value(&build_list);
    duckdb_destroy_value(&probe_list);

    duckdb_logical_type bigint = duckdb_create_logical_type(DUCKDB_TYPE_BIGINT);
    duckdb_bind_add_result_column(info, "probe_idx", bigint);
    duckdb_bind_add_result_column(info, "build_idx", bigint);
    duckdb_destroy_logical_type(&bigint);

    duckdb_bind_set_bind_data(info, bind, [](void* p) { delete static_cast<JoinBindData*>(p); });
}

void join_init(duckdb_init_info info) {
    auto* bind = static_cast<JoinBindData*>(duckdb_init_get_bind_data(info));
    auto* init = new JoinInitData();
    try {
        std::lock_guard<std::mutex> lock(join_call_mutex());
        auto r = shared_hybrid_join().inner_join_i64(
            bind->build_keys.data(), bind->build_keys.size(),
            bind->probe_keys.data(), bind->probe_keys.size());
        init->probe_indices = std::move(r.probe_indices);
        init->build_indices = std::move(r.build_indices);
    } catch (const std::exception& e) {
        delete init;
        duckdb_init_set_error(info, e.what());
        return;
    }
    duckdb_init_set_init_data(info, init, [](void* p) { delete static_cast<JoinInitData*>(p); });
}

void join_function(duckdb_function_info info, duckdb_data_chunk output) {
    auto* init = static_cast<JoinInitData*>(duckdb_function_get_init_data(info));
    if (!init) return;

    const std::size_t remaining = init->probe_indices.size() - init->offset;
    if (remaining == 0) return;

    constexpr idx_t kChunk = 2048;
    const idx_t out_n = static_cast<idx_t>(std::min<std::size_t>(remaining, kChunk));

    duckdb_vector v_probe = duckdb_data_chunk_get_vector(output, 0);
    duckdb_vector v_build = duckdb_data_chunk_get_vector(output, 1);
    auto* probe_out = static_cast<std::int64_t*>(duckdb_vector_get_data(v_probe));
    auto* build_out = static_cast<std::int64_t*>(duckdb_vector_get_data(v_build));

    for (idx_t i = 0; i < out_n; ++i) {
        probe_out[i] = init->probe_indices[init->offset + static_cast<std::size_t>(i)];
        build_out[i] = init->build_indices[init->offset + static_cast<std::size_t>(i)];
    }

    duckdb_data_chunk_set_size(output, out_n);
    init->offset += static_cast<std::size_t>(out_n);
}

// ---------------------------------------------------------------------------
// gpu_join_rows_resident(probe VARCHAR, build VARCHAR, kind VARCHAR)
// -> (probe_idx BIGINT, build_idx BIGINT)  — the row-returning resident join.
// kind: 'inner' | 'left' | 'semi' | 'anti' (case-insensitive). build_idx is
// NULL for LEFT-unmatched / SEMI / ANTI rows. Output capped at
// GPUDB_JOIN_ROWS_MAX_M million rows (default 100) with a clean error.
// ---------------------------------------------------------------------------

std::size_t join_rows_cap() {
    static const std::size_t cap = [] {
        unsigned long long m = 100;   // 100M rows ≈ 1.6 GB of index pairs
        if (const char* s = std::getenv("GPUDB_JOIN_ROWS_MAX_M")) {
            char* end = nullptr;
            const unsigned long long v = std::strtoull(s, &end, 10);
            if (end && *end == '\0' && v > 0) m = v;
        }
        return static_cast<std::size_t>(m) * 1000000;
    }();
    return cap;
}

struct RowsBindData {
    std::string probe, build;
    gpudb::JoinKind kind = gpudb::JoinKind::INNER;
};

std::string value_to_string(duckdb_value v) {
    char* s = duckdb_get_varchar(v);
    std::string out = s ? s : "";
    if (s) duckdb_free(s);
    return out;
}

void rows_bind(duckdb_bind_info info) {
    duckdb_value pv = duckdb_bind_get_parameter(info, 0);
    duckdb_value bv = duckdb_bind_get_parameter(info, 1);
    duckdb_value kv = duckdb_bind_get_parameter(info, 2);
    if (!pv || !bv || !kv) {
        if (pv) duckdb_destroy_value(&pv);
        if (bv) duckdb_destroy_value(&bv);
        if (kv) duckdb_destroy_value(&kv);
        duckdb_bind_set_error(info, "gpu_join_rows_resident: null argument");
        return;
    }
    auto* bind = new RowsBindData();
    bind->probe = value_to_string(pv);
    bind->build = value_to_string(bv);
    std::string kind = value_to_string(kv);
    duckdb_destroy_value(&pv);
    duckdb_destroy_value(&bv);
    duckdb_destroy_value(&kv);
    for (auto& c : kind) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if      (kind == "inner") bind->kind = gpudb::JoinKind::INNER;
    else if (kind == "left")  bind->kind = gpudb::JoinKind::LEFT;
    else if (kind == "semi")  bind->kind = gpudb::JoinKind::SEMI;
    else if (kind == "anti")  bind->kind = gpudb::JoinKind::ANTI;
    else {
        delete bind;
        duckdb_bind_set_error(info,
            "gpu_join_rows_resident: kind must be 'inner', 'left', 'semi' or "
            "'anti' (right/full compose from these with sides swapped)");
        return;
    }

    duckdb_logical_type bigint = duckdb_create_logical_type(DUCKDB_TYPE_BIGINT);
    duckdb_bind_add_result_column(info, "probe_idx", bigint);
    duckdb_bind_add_result_column(info, "build_idx", bigint);
    duckdb_destroy_logical_type(&bigint);
    duckdb_bind_set_bind_data(info, bind, [](void* p) { delete static_cast<RowsBindData*>(p); });
}

struct RowsInitData {
    gpudb::JoinRowsResult rows;
    std::size_t offset = 0;
};

void rows_init(duckdb_init_info info) {
    auto* bind = static_cast<RowsBindData*>(duckdb_init_get_bind_data(info));
    auto* init = new RowsInitData();
    try {
        std::lock_guard<std::mutex> lock(resident_mutex());
        gpudb::ResidentColumn* probe = find_resident_column(bind->probe);
        gpudb::ResidentColumn* build = find_resident_column(bind->build);
        if (!probe || !build)
            throw std::runtime_error(
                "gpu_join_rows_resident: no resident column named '" +
                (probe ? bind->build : bind->probe) +
                "' — create it with gpu_upload/gpu_upload_pair");
        init->rows = resident_aggregator().join_rows_resident(
            *probe, *build, bind->kind, join_rows_cap());
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "op=join_rows_resident rows_probe=%zu rows_build=%zu out_rows=%zu "
            "wall_ms=%.3f kernel_ms=%.3f transfer_ms=%.3f",
            init->rows.rows_probe, init->rows.rows_build,
            init->rows.probe_idx.size(),
            init->rows.wall_ms, init->rows.kernel_ms, init->rows.transfer_ms);
        resident_set_last_stats(buf);
    } catch (const std::exception& e) {
        delete init;
        duckdb_init_set_error(info, e.what());
        return;
    }
    duckdb_init_set_init_data(info, init, [](void* p) { delete static_cast<RowsInitData*>(p); });
}

void rows_function(duckdb_function_info info, duckdb_data_chunk output) {
    auto* init = static_cast<RowsInitData*>(duckdb_function_get_init_data(info));
    if (!init) return;
    const std::size_t remaining = init->rows.probe_idx.size() - init->offset;
    if (remaining == 0) return;

    constexpr idx_t kChunk = 2048;
    const idx_t out_n = static_cast<idx_t>(std::min<std::size_t>(remaining, kChunk));
    duckdb_vector v_probe = duckdb_data_chunk_get_vector(output, 0);
    duckdb_vector v_build = duckdb_data_chunk_get_vector(output, 1);
    auto* probe_out = static_cast<std::int64_t*>(duckdb_vector_get_data(v_probe));
    auto* build_out = static_cast<std::int64_t*>(duckdb_vector_get_data(v_build));
    duckdb_vector_ensure_validity_writable(v_build);
    uint64_t* build_validity = duckdb_vector_get_validity(v_build);

    for (idx_t i = 0; i < out_n; ++i) {
        const std::size_t s = init->offset + static_cast<std::size_t>(i);
        probe_out[i] = init->rows.probe_idx[s];
        const std::int64_t b = init->rows.build_idx[s];
        if (b < 0) {
            duckdb_validity_set_row_invalid(build_validity, i);
        } else {
            build_out[i] = b;
        }
    }
    duckdb_data_chunk_set_size(output, out_n);
    init->offset += static_cast<std::size_t>(out_n);
}

} // namespace

void register_gpu_join(duckdb_connection con) {
    duckdb_logical_type bigint = duckdb_create_logical_type(DUCKDB_TYPE_BIGINT);
    duckdb_logical_type list_bigint = duckdb_create_list_type(bigint);

    duckdb_table_function fn = duckdb_create_table_function();
    duckdb_table_function_set_name(fn, "gpu_inner_join");
    duckdb_table_function_add_parameter(fn, list_bigint);
    duckdb_table_function_add_parameter(fn, list_bigint);
    duckdb_table_function_set_bind(fn, join_bind);
    duckdb_table_function_set_init(fn, join_init);
    duckdb_table_function_set_function(fn, join_function);

    duckdb_destroy_logical_type(&list_bigint);
    duckdb_destroy_logical_type(&bigint);

    if (duckdb_register_table_function(con, fn) == DuckDBError) {
        duckdb_destroy_table_function(&fn);
        throw std::runtime_error("gpu_inner_join registration failed");
    }
    duckdb_destroy_table_function(&fn);

    duckdb_logical_type varchar = duckdb_create_logical_type(DUCKDB_TYPE_VARCHAR);
    duckdb_table_function rfn = duckdb_create_table_function();
    duckdb_table_function_set_name(rfn, "gpu_join_rows_resident");
    duckdb_table_function_add_parameter(rfn, varchar);   // probe keys name
    duckdb_table_function_add_parameter(rfn, varchar);   // build keys name
    duckdb_table_function_add_parameter(rfn, varchar);   // kind
    duckdb_table_function_set_bind(rfn, rows_bind);
    duckdb_table_function_set_init(rfn, rows_init);
    duckdb_table_function_set_function(rfn, rows_function);
    duckdb_destroy_logical_type(&varchar);
    if (duckdb_register_table_function(con, rfn) == DuckDBError) {
        duckdb_destroy_table_function(&rfn);
        throw std::runtime_error("gpu_join_rows_resident registration failed");
    }
    duckdb_destroy_table_function(&rfn);
    std::fprintf(stderr,
        "[gpudb] registered gpu_inner_join + gpu_join_rows_resident\n");
}

} // namespace gpudb_ext
