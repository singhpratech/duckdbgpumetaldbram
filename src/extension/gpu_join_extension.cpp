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

#include <cstdint>
#include <cstdio>
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
    std::fprintf(stderr, "[gpudb] registered gpu_inner_join (hybrid hash join)\n");
}

} // namespace gpudb_ext
