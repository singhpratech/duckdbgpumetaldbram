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
//
// Window contract:
//   DuckDB's window operator (e.g. SUM() OVER (PARTITION BY ...)) builds
//   intermediate per-partition aggregate states and then, for each row in the
//   partition, calls combine() repeatedly with the SAME source state into a
//   *different* per-row destination. The source acts as a partial-aggregate
//   "donor" that the operator queries multiple times. That means combine()
//   must NOT mutate the source state — it must read-only-copy from src to
//   dst. See the comment on combine() below for the bug history.
//
// Concurrency model:
//   The Aggregator returned by gpudb::make_aggregator() owns mutable GPU
//   resources (a single CUDA stream + reused d_in/d_partials/d_out device
//   buffers). It is NOT internally thread-safe. DuckDB's window and grouped
//   aggregate operators parallelise across threads, so finalize() can be
//   invoked from multiple threads concurrently for the same aggregate
//   function — and they all share our process-wide shared_agg() singleton.
//   Every call into shared_agg() is therefore guarded by gpu_call_mutex().

#include "gpu_sum_extension.hpp"
#include "gpu_backend.hpp"

#include "duckdb.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
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

// Single global mutex serialising every entry into shared_agg(). The
// Aggregator's reused device buffers (d_in_, d_partials_, d_out_) and CUDA
// stream are shared mutable state; without this lock, two threads finalising
// different partitions race and read each other's results back from d_out_.
std::mutex& gpu_call_mutex() {
    static std::mutex m;
    return m;
}

// Below this threshold we don't even attempt the GPU — kernel-launch overhead
// dwarfs the work, AND going to the GPU would block on gpu_call_mutex() behind
// some other partition's much larger reduction. Window functions in particular
// hand us tiny per-row buffers (1..N values), so this hits constantly there.
constexpr std::size_t kSumGpuMinRows = 4096;

// CPU fallbacks. Used unconditionally for tiny inputs and on GPU exceptions.
inline std::int64_t cpu_sum(const std::int64_t* data, std::size_t n) {
    std::int64_t acc = 0;
    for (std::size_t i = 0; i < n; ++i) acc += data[i];
    return acc;
}
inline std::int64_t cpu_min(const std::int64_t* data, std::size_t n) {
    std::int64_t v = data[0];
    for (std::size_t i = 1; i < n; ++i) if (data[i] < v) v = data[i];
    return v;
}
inline std::int64_t cpu_max(const std::int64_t* data, std::size_t n) {
    std::int64_t v = data[0];
    for (std::size_t i = 1; i < n; ++i) if (data[i] > v) v = data[i];
    return v;
}

// Thread-safe wrappers around the singleton aggregator. For sub-threshold
// inputs they short-circuit to CPU to avoid lock contention on tiny work.
std::int64_t safe_sum_i64(const std::int64_t* data, std::size_t n) {
    if (n == 0) return 0;
    if (n < kSumGpuMinRows) return cpu_sum(data, n);
    try {
        std::lock_guard<std::mutex> lock(gpu_call_mutex());
        return shared_agg().sum_i64(data, n).value_i64;
    } catch (const std::exception&) {
        return cpu_sum(data, n);
    }
}
std::int64_t safe_min_i64(const std::int64_t* data, std::size_t n) {
    if (n == 0) return 0;
    if (n < kSumGpuMinRows) return cpu_min(data, n);
    try {
        std::lock_guard<std::mutex> lock(gpu_call_mutex());
        return shared_agg().min_i64(data, n).value_i64;
    } catch (const std::exception&) {
        return cpu_min(data, n);
    }
}
std::int64_t safe_max_i64(const std::int64_t* data, std::size_t n) {
    if (n == 0) return 0;
    if (n < kSumGpuMinRows) return cpu_max(data, n);
    try {
        std::lock_guard<std::mutex> lock(gpu_call_mutex());
        return shared_agg().max_i64(data, n).value_i64;
    } catch (const std::exception&) {
        return cpu_max(data, n);
    }
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
    // CRITICAL: combine() must NOT mutate the source state.
    //
    // DuckDB's window operator (e.g. SUM() OVER (PARTITION BY g ORDER BY k))
    // builds a partial-aggregate state for each partition and then, for each
    // row's window, calls combine() with that partition state as the SOURCE
    // and a fresh per-row state as the destination. The same source is used
    // many times — it is read-only from combine()'s perspective.
    //
    // The earlier implementation move-assigned src->buf into dst->buf when
    // dst was empty, as an optimisation. That left src empty for every
    // subsequent combine() call against the same source, producing two
    // distinct visible symptoms:
    //   - small inputs (5 rows / 2 partitions): "first row of every
    //     non-first partition is wrong" — intermittent, depending on which
    //     order the parallel partition workers happened to call combine()
    //     and finalize() in.
    //   - larger inputs (any partition that spans more than one update()
    //     chunk, i.e. > ~16 rows): the running sum collapsed to ~0 partway
    //     through every partition, because the per-row windows after the
    //     first stopped seeing the chunk-1 state (it had already been
    //     moved out).
    //
    // The fix is to *copy* src->buf into dst->buf and never mutate src.
    for (idx_t i = 0; i < count; ++i) {
        const auto* src = reinterpret_cast<const GpuSumState*>(source[i]);
        auto* dst = reinterpret_cast<GpuSumState*>(target[i]);
        if (src->buf.empty()) continue;
        dst->buf.insert(dst->buf.end(), src->buf.begin(), src->buf.end());
    }
}

// Hybrid CPU/GPU planner thresholds — empirically derived from BENCHMARK.md
// (4-column comparison, GROUP BY section). These decide which backend each
// finalize call dispatches to.
//
//   GROUP BY (count > 1):
//     - Mid-cardinality (1 < count < kBatchedGroupByThreshold): per-state
//       CPU sum. The grouped finalize hands us count buffers, each typically
//       holding (total_rows / count) values. With ~100k rows per state
//       (e.g. 10M rows / 100 groups), CPU sum is ~50us per state and the
//       whole loop finishes in a few ms. Going to the GPU per-state in this
//       regime would cost a kernel launch + H2D + D2H per state (~100us
//       each) AND serialize on gpu_call_mutex() against any concurrent
//       finalize calls — strictly worse than CPU AND historically a
//       correctness footgun (intermittent wrong sums on busy systems with
//       parallel finalize threads sharing the singleton CudaAggregator).
//       The CPU path is bandwidth-bound and trivially correct, so we use
//       it unconditionally for this regime.
//     - High-cardinality (count >= kBatchedGroupByThreshold): batched
//       single GPU GROUP BY call (state index = group key). One kernel
//       launch instead of `count` launches; per BENCHMARK.md, this beats
//       CPU 5-14x in the regime where atomicCAS hash table wins.
//
// Toggle via env var GPUDB_FORCE_BACKEND=cpu|gpu|auto (default auto)
// for diagnostic / benchmark runs.
constexpr idx_t kBatchedGroupByThreshold = 64;

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

bool should_use_gpu_for_groupby(idx_t state_count) {
    auto fb = forced_backend();
    if (fb == ForcedBackend::Cpu) return false;
    if (fb == ForcedBackend::Gpu) return true;
    return state_count >= kBatchedGroupByThreshold;
}

void finalize(duckdb_function_info /*info*/, duckdb_aggregate_state* source,
              duckdb_vector result, idx_t count, idx_t offset) {
    std::int64_t* out = reinterpret_cast<std::int64_t*>(duckdb_vector_get_data(result));

    // ---- Single-state fast path: ungrouped SELECT gpu_sum(x) FROM tbl,
    //      and per-row finalize calls from window functions without
    //      PARTITION BY. ----
    //
    // safe_sum_i64() short-circuits to CPU under kSumGpuMinRows (so window
    // functions and tiny ungrouped sums avoid the lock and the kernel-launch
    // cost) and serialises shared_agg() calls above that with a mutex (so
    // multi-threaded window finalize calls don't trash the singleton's
    // device buffers).
    if (count == 1) {
        auto* s = reinterpret_cast<GpuSumState*>(source[0]);
        out[offset] = safe_sum_i64(s->buf.data(), s->buf.size());
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
        // Mid-cardinality (1 < count < kBatchedGroupByThreshold) and
        // batched-GPU fallback paths. Sum each state's buffer on the CPU.
        //
        // Why CPU and not safe_sum_i64() (which dispatches to GPU above
        // kSumGpuMinRows)? Three reasons, in order of importance:
        //
        //   1. Correctness. The original implementation looped calling
        //      safe_sum_i64() per state. Each call grabs gpu_call_mutex()
        //      and runs sum_i64 on the singleton CudaAggregator (which
        //      reuses one set of d_in_/d_partials_/d_out_ buffers). When
        //      DuckDB invokes finalize() in parallel from multiple threads
        //      AND the GPU happens to be busy with a long reduction from
        //      another agg, we observed intermittent wrong totals at
        //      mid-cardinality (50–63 groups × ~100k rows/group on the
        //      reproducer in test/sql/gpu_groupby.test:q7). The CPU path
        //      has no shared state and no race surface.
        //   2. Speed. ~100k int64s per state is ~800 KiB — fits in L2 on
        //      modern x86. cpu_sum hits ~30 GB/s easily, finishing one
        //      state in ~25us. Per-state GPU finalize pays a kernel launch
        //      (~5us), H2D + D2H of 800 KiB (~50us each on PCIe gen4),
        //      AND lock contention. CPU wins outright.
        //   3. Predictability. The big GPU win for grouped aggregates
        //      lives in the BATCHED path above (one kernel for ALL groups,
        //      hash-table GROUP BY on device). Mid-cardinality is a "neither
        //      regime is great" zone; CPU is trivially correct and good
        //      enough.
        //
        // count == 1 still goes to safe_sum_i64() (single-state fast path
        // above) so ungrouped SELECT gpu_sum(x) FROM big_tbl still hits the
        // GPU; window functions (which call finalize with count=1 per row)
        // also still benefit when individual partition buffers are big.
        for (idx_t i = 0; i < count; ++i) {
            auto* s = reinterpret_cast<GpuSumState*>(source[i]);
            const auto& buf = s->buf;
            out[offset + i] = buf.empty() ? 0 : cpu_sum(buf.data(), buf.size());
        }
    }
}

// MIN / MAX share state + update + combine with SUM; only finalize differs.
void finalize_min(duckdb_function_info /*info*/, duckdb_aggregate_state* source,
                  duckdb_vector result, idx_t count, idx_t offset) {
    std::int64_t* out = reinterpret_cast<std::int64_t*>(duckdb_vector_get_data(result));
    for (idx_t i = 0; i < count; ++i) {
        auto* s = reinterpret_cast<GpuSumState*>(source[i]);
        out[offset + i] = safe_min_i64(s->buf.data(), s->buf.size());
    }
}

void finalize_max(duckdb_function_info /*info*/, duckdb_aggregate_state* source,
                  duckdb_vector result, idx_t count, idx_t offset) {
    std::int64_t* out = reinterpret_cast<std::int64_t*>(duckdb_vector_get_data(result));
    for (idx_t i = 0; i < count; ++i) {
        auto* s = reinterpret_cast<GpuSumState*>(source[i]);
        out[offset + i] = safe_max_i64(s->buf.data(), s->buf.size());
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
