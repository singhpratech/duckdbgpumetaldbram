// gpu_sum_extension.cpp — register the gpu_sum() aggregate on a DuckDB
// connection.
//
// Public function:
//   void register_gpu_sum(duckdb_connection con);
//
// Used by:
//   - benchmark/sql_demo.cpp (CLI demo: opens a duckdb db, registers, runs SQL)
//   - a future loadable extension wrapper (DuckDB C API _init_c_api entry
//     point) once we package this for `LOAD '...'` in the DuckDB CLI
//
// Strategy (week-2 MVP):
//   - One aggregate state per GROUP. State holds a std::vector<int64_t> that
//     accumulates raw values from each chunk's update() call.
//   - On finalize(), call gpudb::Aggregator::sum_i64 on the full buffer.
//     This means the GPU sees the entire column at once (real acceleration).
//   - For SELECT gpu_sum(x) FROM tbl with no GROUP BY: one state, one GPU
//     reduction at the end. This is the case where the win shows.
//   - For SELECT gpu_sum(x) GROUP BY g: one state per group, each finalized
//     separately. Less ideal — a future operator-replacement pass should
//     batch the per-group reductions; out of scope here.

#include "gpu_sum_extension.hpp"
#include "gpu_backend.hpp"

#include "duckdb.h"

#include <cstdint>
#include <cstdio>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gpudb_ext {

namespace {

gpudb::Aggregator& shared_agg() {
    static std::unique_ptr<gpudb::Aggregator> agg = []() {
        return gpudb::make_aggregator(gpudb::default_backend());
    }();
    return *agg;
}

struct GpuSumState {
    std::vector<std::int64_t> buf;
};

idx_t state_size(duckdb_function_info /*info*/) { return sizeof(GpuSumState); }

void state_init(duckdb_function_info /*info*/, duckdb_aggregate_state state) {
    auto* s = reinterpret_cast<GpuSumState*>(state);
    new (s) GpuSumState();
}

void state_destroy(duckdb_aggregate_state* states, idx_t count) {
    for (idx_t i = 0; i < count; ++i) {
        auto* s = reinterpret_cast<GpuSumState*>(states[i]);
        s->~GpuSumState();
    }
}

void update(duckdb_function_info /*info*/, duckdb_data_chunk input,
            duckdb_aggregate_state* states) {
    duckdb_vector vec = duckdb_data_chunk_get_vector(input, 0);
    const std::int64_t* data = reinterpret_cast<const std::int64_t*>(duckdb_vector_get_data(vec));
    const idx_t n = duckdb_data_chunk_get_size(input);
    if (data == nullptr || n == 0) return;

    // Fast path: SELECT gpu_sum(x) FROM tbl with no GROUP BY.
    // All states[i] alias the same state object, so we can bulk-insert the
    // chunk in one memcpy instead of n push_backs.
    if (n > 1 && states[0] == states[n - 1]) {
        bool all_same = true;
        for (idx_t i = 1; i + 1 < n; ++i) {
            if (states[i] != states[0]) { all_same = false; break; }
        }
        if (all_same) {
            auto* s = reinterpret_cast<GpuSumState*>(states[0]);
            s->buf.insert(s->buf.end(), data, data + n);
            return;
        }
    }
    // Slow path: GROUP BY routes different rows to different states.
    for (idx_t i = 0; i < n; ++i) {
        auto* s = reinterpret_cast<GpuSumState*>(states[i]);
        s->buf.push_back(data[i]);
    }
}

void combine(duckdb_function_info /*info*/, duckdb_aggregate_state* source,
             duckdb_aggregate_state* target, idx_t count) {
    for (idx_t i = 0; i < count; ++i) {
        auto* src = reinterpret_cast<GpuSumState*>(source[i]);
        auto* dst = reinterpret_cast<GpuSumState*>(target[i]);
        if (dst->buf.empty()) {
            dst->buf = std::move(src->buf);
        } else {
            dst->buf.insert(dst->buf.end(), src->buf.begin(), src->buf.end());
        }
    }
}

void finalize(duckdb_function_info /*info*/, duckdb_aggregate_state* source,
              duckdb_vector result, idx_t count, idx_t offset) {
    std::int64_t* out = reinterpret_cast<std::int64_t*>(duckdb_vector_get_data(result));
    auto& agg = shared_agg();
    for (idx_t i = 0; i < count; ++i) {
        auto* s = reinterpret_cast<GpuSumState*>(source[i]);
        if (s->buf.empty()) { out[offset + i] = 0; continue; }
        try {
            auto r = agg.sum_i64(s->buf.data(), s->buf.size());
            out[offset + i] = r.value_i64;
        } catch (const std::exception&) {
            std::int64_t acc = 0;
            for (auto v : s->buf) acc += v;
            out[offset + i] = acc;
        }
    }
}

} // namespace

void register_gpu_sum(duckdb_connection con) {
    duckdb_aggregate_function fn = duckdb_create_aggregate_function();
    duckdb_aggregate_function_set_name(fn, "gpu_sum");

    duckdb_logical_type bigint = duckdb_create_logical_type(DUCKDB_TYPE_BIGINT);
    duckdb_aggregate_function_add_parameter(fn, bigint);
    duckdb_aggregate_function_set_return_type(fn, bigint);
    duckdb_destroy_logical_type(&bigint);

    duckdb_aggregate_function_set_functions(fn,
        state_size, state_init, update, combine, finalize);
    duckdb_aggregate_function_set_destructor(fn, state_destroy);

    duckdb_state st = duckdb_register_aggregate_function(con, fn);
    duckdb_destroy_aggregate_function(&fn);
    if (st == DuckDBError) {
        throw std::runtime_error("gpu_sum registration failed");
    }
    std::fprintf(stderr, "[gpudb] registered gpu_sum  (backend=%s)\n",
                 gpudb::to_string(shared_agg().backend()));
}

} // namespace gpudb_ext
