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
#include <cstdlib>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
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

    // Validity bitmap: 1 bit per row, 64 rows per uint64. NULL rows are skipped
    // (matches DuckDB's SQL semantics for SUM/MIN/MAX). nullptr means no NULLs.
    uint64_t* validity = duckdb_vector_get_validity(vec);

    // Fast path: SELECT gpu_sum(x) FROM tbl with no GROUP BY.
    // All states[i] alias the same state object, so we can bulk-insert the
    // chunk in one memcpy (or filtered copy if there are NULLs).
    if (n > 1 && states[0] == states[n - 1]) {
        bool all_same = true;
        for (idx_t i = 1; i + 1 < n; ++i) {
            if (states[i] != states[0]) { all_same = false; break; }
        }
        if (all_same) {
            auto* s = reinterpret_cast<GpuSumState*>(states[0]);
            if (validity == nullptr) {
                // No NULLs at all — single memcpy.
                s->buf.insert(s->buf.end(), data, data + n);
            } else {
                // Filter out NULL rows.
                s->buf.reserve(s->buf.size() + n);
                for (idx_t i = 0; i < n; ++i) {
                    if (duckdb_validity_row_is_valid(validity, i)) {
                        s->buf.push_back(data[i]);
                    }
                }
            }
            return;
        }
    }
    // Slow path: GROUP BY routes different rows to different states.
    if (validity == nullptr) {
        for (idx_t i = 0; i < n; ++i) {
            auto* s = reinterpret_cast<GpuSumState*>(states[i]);
            s->buf.push_back(data[i]);
        }
    } else {
        for (idx_t i = 0; i < n; ++i) {
            if (!duckdb_validity_row_is_valid(validity, i)) continue;
            auto* s = reinterpret_cast<GpuSumState*>(states[i]);
            s->buf.push_back(data[i]);
        }
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

// Hybrid CPU/GPU planner thresholds — empirically derived from BENCHMARK.md
// (4-column comparison, GROUP BY section). These decide which backend each
// finalize call dispatches to.
//
//   SUM (count == 1):
//     - GPU iff the buffer is large enough to amortize PCIe transfer.
//       Below kSumGpuThreshold rows, CPU wins on streaming SUM (per the
//       1B-row + 200M-row benchmarks).
//   GROUP BY (count > 1):
//     - Per-state CPU iff count < kBatchedGroupByThreshold (kernel-launch
//       overhead would dominate; CPU's parallel hash map already handles
//       low-cardinality fast).
//     - Batched GPU GROUP BY iff count >= kBatchedGroupByThreshold (the
//       regime where atomicCAS hash table beats CPU 5-14x per BENCHMARK).
//
// Toggle via env var GPUDB_FORCE_BACKEND=cpu|gpu|auto (default auto)
// for diagnostic / benchmark runs.
constexpr std::size_t kSumGpuThreshold        = 1'000'000;   // ~8 MiB int64
constexpr idx_t       kBatchedGroupByThreshold = 64;

enum class ForcedBackend { Auto, Cpu, Gpu };

ForcedBackend forced_backend() {
    static ForcedBackend cached = []() {
        const char* e = std::getenv("GPUDB_FORCE_BACKEND");
        if (!e) return ForcedBackend::Auto;
        if (std::string{e} == "cpu") return ForcedBackend::Cpu;
        if (std::string{e} == "gpu") return ForcedBackend::Gpu;
        return ForcedBackend::Auto;
    }();
    return cached;
}

bool should_use_gpu_for_sum(std::size_t n) {
    auto fb = forced_backend();
    if (fb == ForcedBackend::Cpu) return false;
    if (fb == ForcedBackend::Gpu) return true;
    return n >= kSumGpuThreshold;
}

bool should_use_gpu_for_groupby(idx_t state_count) {
    auto fb = forced_backend();
    if (fb == ForcedBackend::Cpu) return false;
    if (fb == ForcedBackend::Gpu) return true;
    return state_count >= kBatchedGroupByThreshold;
}

void finalize(duckdb_function_info /*info*/, duckdb_aggregate_state* source,
              duckdb_vector result, idx_t count, idx_t offset) {
    std::int64_t* out = reinterpret_cast<std::int64_t*>(duckdb_vector_get_data(result));

    // ---- Single-state fast path: ungrouped SELECT gpu_sum(x) FROM tbl ----
    //
    // We keep this path on the GPU unconditionally because:
    //   1. Window functions (OVER ...) reuse the aggregate state and call
    //      finalize repeatedly with count=1 across different per-row windows.
    //      A hybrid CPU/GPU switch here causes incorrect results in window
    //      mode (the per-call state object can be in transient states the
    //      CPU fallback miscounts). Better to always use the same path.
    //   2. For the streaming-SUM case where CPU would win on perf, the user
    //      would just not call gpu_sum() — they'd call sum().
    //
    // The hybrid planning therefore lives in the GROUP BY branch below, where
    // the cardinality threshold actually matters and DuckDB's calling pattern
    // is stable.
    if (count == 1) {
        auto* s = reinterpret_cast<GpuSumState*>(source[0]);
        if (s->buf.empty()) { out[offset] = 0; return; }
        try {
            out[offset] = shared_agg().sum_i64(s->buf.data(), s->buf.size()).value_i64;
        } catch (const std::exception&) {
            std::int64_t acc = 0;
            for (auto v : s->buf) acc += v;
            out[offset] = acc;
        }
        return;
    }

    // ---- Grouped path: count > 1 ----
    // Strategy: if we have many states, batch them all into ONE GPU GROUP BY
    // call (state index = group key). This converts an O(count) sequence of
    // N small kernel launches into a single big launch, which is exactly the
    // regime the GROUP BY kernel is designed for (per BENCHMARK.md, it wins
    // 12-13x over CPU even cold).
    bool used_batched = false;
    if (should_use_gpu_for_groupby(count)) {
        try {
            std::size_t total = 0;
            for (idx_t i = 0; i < count; ++i)
                total += reinterpret_cast<GpuSumState*>(source[i])->buf.size();

            if (total > 0) {
                std::vector<std::int64_t> flat_keys, flat_values;
                flat_keys.reserve(total);
                flat_values.reserve(total);
                for (idx_t i = 0; i < count; ++i) {
                    const auto& buf = reinterpret_cast<GpuSumState*>(source[i])->buf;
                    flat_keys.insert(flat_keys.end(), buf.size(), static_cast<std::int64_t>(i));
                    flat_values.insert(flat_values.end(), buf.begin(), buf.end());
                }

                auto gb = gpudb::make_groupby_aggregator(gpudb::default_backend());
                auto r = gb->groupby_sum_i64(flat_keys.data(), flat_values.data(),
                                              flat_keys.size(), count);

                // Default outputs to 0 for empty groups; then fill those that had data.
                for (idx_t i = 0; i < count; ++i) out[offset + i] = 0;
                for (std::size_t i = 0; i < r.keys.size(); ++i) {
                    const std::int64_t k = r.keys[i];
                    if (k >= 0 && static_cast<idx_t>(k) < count)
                        out[offset + static_cast<idx_t>(k)] = r.sums[i];
                }
                used_batched = true;
            } else {
                for (idx_t i = 0; i < count; ++i) out[offset + i] = 0;
                used_batched = true;
            }
        } catch (const std::exception&) {
            // fall through to per-state path
        }
    }

    if (!used_batched) {
        auto& agg = shared_agg();
        for (idx_t i = 0; i < count; ++i) {
            auto* s = reinterpret_cast<GpuSumState*>(source[i]);
            if (s->buf.empty()) { out[offset + i] = 0; continue; }
            try {
                out[offset + i] = agg.sum_i64(s->buf.data(), s->buf.size()).value_i64;
            } catch (const std::exception&) {
                std::int64_t acc = 0;
                for (auto v : s->buf) acc += v;
                out[offset + i] = acc;
            }
        }
    }
}

// MIN / MAX share state + update + combine with SUM; only finalize differs.
void finalize_min(duckdb_function_info /*info*/, duckdb_aggregate_state* source,
                  duckdb_vector result, idx_t count, idx_t offset) {
    std::int64_t* out = reinterpret_cast<std::int64_t*>(duckdb_vector_get_data(result));
    auto& agg = shared_agg();
    for (idx_t i = 0; i < count; ++i) {
        auto* s = reinterpret_cast<GpuSumState*>(source[i]);
        if (s->buf.empty()) { out[offset + i] = 0; continue; }  // SQL NULL handling: TODO
        try {
            out[offset + i] = agg.min_i64(s->buf.data(), s->buf.size()).value_i64;
        } catch (const std::exception&) {
            std::int64_t v = s->buf[0];
            for (auto x : s->buf) if (x < v) v = x;
            out[offset + i] = v;
        }
    }
}

void finalize_max(duckdb_function_info /*info*/, duckdb_aggregate_state* source,
                  duckdb_vector result, idx_t count, idx_t offset) {
    std::int64_t* out = reinterpret_cast<std::int64_t*>(duckdb_vector_get_data(result));
    auto& agg = shared_agg();
    for (idx_t i = 0; i < count; ++i) {
        auto* s = reinterpret_cast<GpuSumState*>(source[i]);
        if (s->buf.empty()) { out[offset + i] = 0; continue; }
        try {
            out[offset + i] = agg.max_i64(s->buf.data(), s->buf.size()).value_i64;
        } catch (const std::exception&) {
            std::int64_t v = s->buf[0];
            for (auto x : s->buf) if (x > v) v = x;
            out[offset + i] = v;
        }
    }
}

void register_one(duckdb_connection con, const char* name,
                  duckdb_aggregate_finalize_t fin) {
    duckdb_aggregate_function fn = duckdb_create_aggregate_function();
    duckdb_aggregate_function_set_name(fn, name);
    duckdb_logical_type bigint = duckdb_create_logical_type(DUCKDB_TYPE_BIGINT);
    duckdb_aggregate_function_add_parameter(fn, bigint);
    duckdb_aggregate_function_set_return_type(fn, bigint);
    duckdb_destroy_logical_type(&bigint);
    duckdb_aggregate_function_set_functions(fn,
        state_size, state_init, update, combine, fin);
    duckdb_aggregate_function_set_destructor(fn, state_destroy);
    duckdb_state st = duckdb_register_aggregate_function(con, fn);
    duckdb_destroy_aggregate_function(&fn);
    if (st == DuckDBError) {
        throw std::runtime_error(std::string(name) + " registration failed");
    }
}

} // namespace

void register_gpu_sum(duckdb_connection con) {
    register_one(con, "gpu_sum", finalize);
    register_one(con, "gpu_min", finalize_min);
    register_one(con, "gpu_max", finalize_max);
    std::fprintf(stderr, "[gpudb] registered gpu_sum / gpu_min / gpu_max  (backend=%s)\n",
                 gpudb::to_string(shared_agg().backend()));
}

} // namespace gpudb_ext
