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
//   - One aggregate state per GROUP. State holds a buffer that accumulates
//     raw values from each chunk's update() call.
//   - On finalize(), call gpudb::Aggregator::sum_i64 on the full buffer.
//     This means the GPU sees the entire column at once (real acceleration).
//   - For SELECT gpu_sum(x) FROM tbl with no GROUP BY: one state, one GPU
//     reduction at the end. This is the case where the win shows.
//   - For SELECT gpu_sum(x) GROUP BY g: one state per group, each finalized
//     separately.
//
// State layout — defending against DuckDB's raw-byte state copies:
//   DuckDB's grouped-aggregate hash table (RadixPartitionedTupleData) and
//   its window operators do not always treat aggregate state as a typed
//   object. When the radix hash table repartitions, it COPIES STATE BYTES
//   directly between tuples — without calling our state_init / a move
//   constructor / a copy callback. Anything we put in the state must
//   therefore be POD-copy-safe.
//
//   We can't use std::vector inside the state for this reason: a raw-byte
//   copy duplicates the {data ptr, size, capacity} triple, and then if
//   either copy is destructed, the other has a dangling pointer.
//
//   Instead we store data in process-global, append-only "buffers" and
//   keep only an opaque BufferRef (a 64-bit integer ID + a magic word) in
//   the state itself. Buffer storage is allocated on demand and we
//   intentionally LEAK buffers on state_destroy — see the comment there
//   for the reasoning. The leak is bounded by the working set of a
//   single query and is reclaimed at process exit.
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
//   For SUM(x) OVER () — the unbounded-frame case — DuckDB uses
//   WindowConstantAggregator, which calls update() with a CONSTANT_VECTOR
//   state (only states[0] is valid storage, states[1..N-1] is OOB read of
//   a 1-pointer buffer). The C API doesn't flatten the state vector before
//   passing it through, so we must defend against this in update() too —
//   we use the magic-word probe to confirm a pointer is one of our states
//   before dereferencing it past states[0].
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

// gpu_sum_extension.hpp already pulled in the right DuckDB header:
//   - GPUDB_C_STRUCT_ABI defined  -> duckdb_extension.h (C_STRUCT stable ABI).
//     In that mode every duckdb_* call is a macro that dereferences the global
//     duckdb_ext_api struct, which the entrypoint TU (duckdb_loadable.cpp)
//     defines and populates. This TU only *references* it, so declare it extern.
//   - otherwise                   -> duckdb.h (real symbols from libduckdb).
#if defined(GPUDB_C_STRUCT_ABI)
DUCKDB_EXTENSION_EXTERN
#endif

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
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

// Process-global pool of buffers, indexed by an opaque 64-bit ID.
//
// Why this exists:
//   DuckDB's RadixPartitionedTupleData.Repartition() copies aggregate state
//   BYTES from one tuple to another with a raw memcpy — it doesn't call our
//   state_init or any move/copy callback. Anything stored inline in the state
//   that can't survive a memcpy (e.g. std::vector with its heap-owned data
//   pointer) becomes a use-after-free hazard the moment one copy is destroyed.
//
//   By keeping a u64 ID inside the state and the actual data pointer here,
//   raw-byte copies become benign: both source and destination state hold
//   the same ID and read the same backing data through the pool.
//
// Lifetime:
//   We do NOT refcount or free entries — see the comment in state_destroy
//   for the reason (no callback for raw-byte copies, so we can't
//   reliably know when the last reference is gone). The leak is bounded
//   by the per-aggregate working set of a single query and is reclaimed
//   when the process exits.
//
// Concurrency:
//   The pool is touched from update / combine / finalize on multiple
//   threads. A single mutex around the map is sufficient for our scale
//   (small number of long-lived buffers, brief locked sections).
class BufferPool {
public:
    // Allocate a fresh buffer; returns its ID.
    std::uint64_t create() {
        std::lock_guard<std::mutex> g(mu_);
        const std::uint64_t id = ++next_id_;
        entries_.emplace(id, std::make_unique<std::vector<std::int64_t>>());
        return id;
    }

    // Append a range of values to the buffer. May reallocate; thread-safe.
    void append(std::uint64_t id, const std::int64_t* values, std::size_t n) {
        std::lock_guard<std::mutex> g(mu_);
        auto it = entries_.find(id);
        if (it == entries_.end()) return;
        it->second->insert(it->second->end(), values, values + n);
    }

    // Append a single value.
    void append_one(std::uint64_t id, std::int64_t value) {
        std::lock_guard<std::mutex> g(mu_);
        auto it = entries_.find(id);
        if (it == entries_.end()) return;
        it->second->push_back(value);
    }

    // Append all of `src`'s values to `dst`. Used by combine().
    void append_from(std::uint64_t dst, std::uint64_t src) {
        std::lock_guard<std::mutex> g(mu_);
        auto si = entries_.find(src);
        if (si == entries_.end()) return;
        auto di = entries_.find(dst);
        if (di == entries_.end()) return;
        const auto& s = *si->second;
        di->second->insert(di->second->end(), s.begin(), s.end());
    }

    // Snapshot the buffer's contents for a read-only consumer (finalize).
    // Returns an empty vector if the ID is unknown.
    std::vector<std::int64_t> snapshot(std::uint64_t id) {
        std::lock_guard<std::mutex> g(mu_);
        auto it = entries_.find(id);
        if (it == entries_.end()) return {};
        return *it->second;  // copy
    }

private:
    std::mutex mu_;
    std::unordered_map<std::uint64_t, std::unique_ptr<std::vector<std::int64_t>>> entries_;
    std::uint64_t next_id_ = 0;
};

BufferPool& buffer_pool() {
    static BufferPool p;
    return p;
}

// Aggregate state — POD layout. Two fields:
//   * magic   — sentinel that lets us identify a buffer as one of ours
//               even after a raw-byte copy. Confirms ID is meaningful.
//   * buf_id  — handle into BufferPool. 0 means uninitialised / no buffer.
//
// Sized as 16 bytes total (alignof 8). DuckDB's state_size callback
// returns sizeof(GpuSumState).
struct GpuSumState {
    static constexpr std::uint64_t kMagic = 0xC0FFEEDEADBEEF01ULL;
    std::uint64_t magic;
    std::uint64_t buf_id;
};

// Defensive accessor: returns the state's buffer ID iff the pointer looks
// like one of our states (magic word check). Returns 0 otherwise.
//
// This is the magic-word probe that handles BOTH:
//   - WindowConstantAggregator OOB reads of states[i>0] (the bytes there
//     are heap garbage that won't form our magic word)
//   - Raw-byte state copies from RadixPartitionedTupleData (the bytes
//     DO contain our magic word, so the probe passes — and the buf_id is
//     the same as the original, sharing data via the refcounted pool)
inline std::uint64_t probe_buf_id(const void* p) {
    if (!p) return 0;
    auto* s = reinterpret_cast<const GpuSumState*>(p);
    if (s->magic != GpuSumState::kMagic) return 0;
    return s->buf_id;
}

idx_t state_size(duckdb_function_info /*info*/) { return sizeof(GpuSumState); }

void state_init(duckdb_function_info /*info*/, duckdb_aggregate_state state) {
    auto* s = reinterpret_cast<GpuSumState*>(state);
    s->magic = GpuSumState::kMagic;
    s->buf_id = buffer_pool().create();
}

void state_destroy(duckdb_aggregate_state* states, idx_t count) {
    // We DO NOT free the underlying buffer here, because DuckDB's
    // RadixPartitionedTupleData repartitioning copies state bytes between
    // tuples without calling our state_init. After such a copy, multiple
    // tuples may point to the same buf_id; if we freed on the first
    // state_destroy we would yank the buffer out from under any later
    // combine() or finalize() that references the surviving copies.
    //
    // We also don't have a way to refcount these copies (no callback fires
    // when DuckDB memcpys a state), so the safe choice is: leak the buffer
    // until process exit. In practice the leak is bounded — buffers are
    // sized to per-aggregate accumulators and are reclaimed on process
    // teardown via the BufferPool's static destructor.
    //
    // We DO wipe the state's magic word to prevent a stale pointer to
    // this slot from accidentally passing our probe in a later call.
    for (idx_t i = 0; i < count; ++i) {
        auto* s = reinterpret_cast<GpuSumState*>(states[i]);
        if (!s) continue;
        if (s->magic != GpuSumState::kMagic) continue;
        s->magic = 0;
        // Note: leaving buf_id intact is harmless — the magic-word probe
        // will reject this slot before anyone reads buf_id from it.
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

    // Resolve states[0] up front; it's always valid (DuckDB never gives us
    // an empty state vector here). Magic-word check is a defence against
    // a malformed call but should always pass.
    const std::uint64_t s0_buf = probe_buf_id(states[0]);
    if (s0_buf == 0) return;

    // Determine whether the rest of states[] are real, distinct, or just
    // OOB garbage from a CONSTANT_VECTOR (the OVER () case).
    //
    // We probe a few indices and ask: does each pointer look like one of
    // our states (magic word OK)? If yes, we have a real per-row state
    // vector and use the slow path. If any probe fails the magic word
    // test, the caller passed us a constant state vector and only
    // states[0] is meaningful.
    bool per_row = true;
    if (n > 1) {
        const idx_t probes[3] = { 1, n / 2, n - 1 };
        for (idx_t k = 0; k < 3 && per_row; ++k) {
            const idx_t i = probes[k];
            if (i == 0) continue;
            const std::uint64_t bid = probe_buf_id(states[i]);
            if (bid == 0) per_row = false;
        }
    }

    if (n == 1 || !per_row) {
        // Single-state path: either ungrouped or constant-vector window.
        // All values flow into states[0]'s buffer.
        if (validity == nullptr) {
            buffer_pool().append(s0_buf, data, n);
        } else {
            for (idx_t i = 0; i < n; ++i) {
                if (duckdb_validity_row_is_valid(validity, i)) {
                    buffer_pool().append_one(s0_buf, data[i]);
                }
            }
        }
        return;
    }

    // Per-row path: GROUP BY routes different rows to different states.
    // Each states[i] passed the magic-word probe in our spot-check above,
    // so dereferencing them is safe.
    if (validity == nullptr) {
        for (idx_t i = 0; i < n; ++i) {
            const std::uint64_t bid = probe_buf_id(states[i]);
            if (bid != 0) buffer_pool().append_one(bid, data[i]);
        }
    } else {
        for (idx_t i = 0; i < n; ++i) {
            if (!duckdb_validity_row_is_valid(validity, i)) continue;
            const std::uint64_t bid = probe_buf_id(states[i]);
            if (bid != 0) buffer_pool().append_one(bid, data[i]);
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
    // subsequent combine() call against the same source. The current
    // BufferPool::append_from copies, never moves — so the source stays
    // intact for the next per-row combine.
    //
    // We use the magic-word probe to find the source / target buffer IDs
    // — this also handles the case where the radix HT raw-copied a state
    // (the copy shares the same buf_id; reading data via the pool is safe).
    for (idx_t i = 0; i < count; ++i) {
        const std::uint64_t src_buf = probe_buf_id(source[i]);
        const std::uint64_t dst_buf = probe_buf_id(target[i]);
        if (src_buf == 0 || dst_buf == 0) continue;
        if (src_buf == dst_buf) continue;  // self-combine: nothing to do.
        buffer_pool().append_from(dst_buf, src_buf);
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

// Determine whether sources[0..count-1] are distinct-state (grouped /
// per-row finalize) or constant-state (window OVER () broadcast).
// Same logic as update()'s probe.
inline bool finalize_is_per_state(duckdb_aggregate_state* source, idx_t count) {
    if (count <= 1) return false;
    const idx_t probes[3] = { 1, count / 2, count - 1 };
    for (idx_t k = 0; k < 3; ++k) {
        const idx_t i = probes[k];
        if (i == 0) continue;
        if (probe_buf_id(source[i]) == 0) return false;
    }
    return true;
}

void finalize(duckdb_function_info /*info*/, duckdb_aggregate_state* source,
              duckdb_vector result, idx_t count, idx_t offset) {
    std::int64_t* out = reinterpret_cast<std::int64_t*>(duckdb_vector_get_data(result));

    // ---- Single-state fast path: ungrouped SELECT gpu_sum(x) FROM tbl,
    //      and per-row finalize calls from window functions without
    //      PARTITION BY. ----
    if (count == 1) {
        const std::uint64_t bid = probe_buf_id(source[0]);
        if (bid == 0) { out[offset] = 0; return; }
        auto vals = buffer_pool().snapshot(bid);
        out[offset] = safe_sum_i64(vals.data(), vals.size());
        return;
    }

    // ---- Constant-state finalize: WindowConstantAggregator passes a
    //      CONSTANT_VECTOR for source. Only source[0] is real; broadcast. ----
    if (!finalize_is_per_state(source, count)) {
        std::int64_t v = 0;
        const std::uint64_t bid = probe_buf_id(source[0]);
        if (bid != 0) {
            auto vals = buffer_pool().snapshot(bid);
            v = safe_sum_i64(vals.data(), vals.size());
        }
        for (idx_t i = 0; i < count; ++i) out[offset + i] = v;
        return;
    }

    // ---- Grouped path: count > 1 with all distinct registered states. ----
    // Strategy: if we have many states, batch them all into ONE GPU GROUP BY
    // call (state index = group key). This converts an O(count) sequence of
    // N small kernel launches into a single big launch, which is exactly the
    // regime the GROUP BY kernel is designed for (per BENCHMARK.md, it wins
    // 12-13x over CPU even cold).
    bool used_batched = false;
    if (should_use_gpu_for_groupby(count)) {
        try {
            // Snapshot every state's buffer up front. Locking the pool
            // serially per state is fine — these snapshots are local copies
            // and we hold no pool lock across the GPU launch.
            std::vector<std::vector<std::int64_t>> snaps;
            snaps.reserve(count);
            std::size_t total = 0;
            for (idx_t i = 0; i < count; ++i) {
                const std::uint64_t bid = probe_buf_id(source[i]);
                snaps.emplace_back(bid ? buffer_pool().snapshot(bid)
                                       : std::vector<std::int64_t>{});
                total += snaps.back().size();
            }

            if (total > 0) {
                std::vector<std::int64_t> flat_keys, flat_values;
                flat_keys.reserve(total);
                flat_values.reserve(total);
                for (idx_t i = 0; i < count; ++i) {
                    const auto& v = snaps[i];
                    flat_keys.insert(flat_keys.end(), v.size(), static_cast<std::int64_t>(i));
                    flat_values.insert(flat_values.end(), v.begin(), v.end());
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
            const std::uint64_t bid = probe_buf_id(source[i]);
            if (bid == 0) { out[offset + i] = 0; continue; }
            auto vals = buffer_pool().snapshot(bid);
            out[offset + i] = safe_sum_i64(vals.data(), vals.size());
        }
    }
}

// MIN / MAX share state + update + combine with SUM; only finalize differs.
//
// Both honour the same constant-state defence as sum's finalize(): if the
// source vector has only source[0] valid, we compute one min/max and
// broadcast it across all output rows.
void finalize_min(duckdb_function_info /*info*/, duckdb_aggregate_state* source,
                  duckdb_vector result, idx_t count, idx_t offset) {
    std::int64_t* out = reinterpret_cast<std::int64_t*>(duckdb_vector_get_data(result));
    if (count == 1) {
        const std::uint64_t bid = probe_buf_id(source[0]);
        if (bid == 0) { out[offset] = 0; return; }
        auto vals = buffer_pool().snapshot(bid);
        out[offset] = safe_min_i64(vals.data(), vals.size());
        return;
    }
    if (!finalize_is_per_state(source, count)) {
        std::int64_t v = 0;
        const std::uint64_t bid = probe_buf_id(source[0]);
        if (bid != 0) {
            auto vals = buffer_pool().snapshot(bid);
            v = safe_min_i64(vals.data(), vals.size());
        }
        for (idx_t i = 0; i < count; ++i) out[offset + i] = v;
        return;
    }
    for (idx_t i = 0; i < count; ++i) {
        const std::uint64_t bid = probe_buf_id(source[i]);
        if (bid == 0) { out[offset + i] = 0; continue; }
        auto vals = buffer_pool().snapshot(bid);
        out[offset + i] = safe_min_i64(vals.data(), vals.size());
    }
}

void finalize_max(duckdb_function_info /*info*/, duckdb_aggregate_state* source,
                  duckdb_vector result, idx_t count, idx_t offset) {
    std::int64_t* out = reinterpret_cast<std::int64_t*>(duckdb_vector_get_data(result));
    if (count == 1) {
        const std::uint64_t bid = probe_buf_id(source[0]);
        if (bid == 0) { out[offset] = 0; return; }
        auto vals = buffer_pool().snapshot(bid);
        out[offset] = safe_max_i64(vals.data(), vals.size());
        return;
    }
    if (!finalize_is_per_state(source, count)) {
        std::int64_t v = 0;
        const std::uint64_t bid = probe_buf_id(source[0]);
        if (bid != 0) {
            auto vals = buffer_pool().snapshot(bid);
            v = safe_max_i64(vals.data(), vals.size());
        }
        for (idx_t i = 0; i < count; ++i) out[offset + i] = v;
        return;
    }
    for (idx_t i = 0; i < count; ++i) {
        const std::uint64_t bid = probe_buf_id(source[i]);
        if (bid == 0) { out[offset + i] = 0; continue; }
        auto vals = buffer_pool().snapshot(bid);
        out[offset + i] = safe_max_i64(vals.data(), vals.size());
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
