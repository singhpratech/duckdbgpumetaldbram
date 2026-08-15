// gpu_resident.cpp — resident-column SQL surface: the path where the GPU
// actually wins.
//
// Public function:
//   void register_gpu_resident(duckdb_connection con);   (called by
//   register_gpu_sum, so both build paths pick these up automatically)
//
// Why this exists (and why the streaming aggregates in gpu_sum_extension.cpp
// deliberately do NOT touch the GPU): a one-shot aggregate over cold data is
// bounded by getting the data to the ALU — DuckDB's scan already streams the
// column through the CPU caches, so copying it over PCIe just to reduce it
// loses (BENCHMARK.md 2026-07-19, 3x-110x). The GPU's win is the HOT path:
// pay the transfer ONCE, keep the column in device memory, and every
// subsequent reduction runs at VRAM bandwidth with zero transfer. That is
// exactly the ResidentColumn API in gpu_backend.hpp; these functions expose
// it to SQL:
//
//   gpu_upload(name VARCHAR, v BIGINT) -> BIGINT     aggregate: buffer the
//   gpu_upload(name VARCHAR, v DOUBLE) -> BIGINT     column, upload once,
//                                                    return rows uploaded
//   gpu_sum_resident(name)     -> BIGINT     reduce the resident column —
//   gpu_min_resident(name)     -> BIGINT     no per-query transfer. i64 only:
//   gpu_max_resident(name)     -> BIGINT     the v1 ABI has no resident f64
//   gpu_sum_resident_f64(name) -> DOUBLE     min/max (see gpu_backend.hpp).
//   gpu_resident_info(name)    -> VARCHAR    dtype/rows/device of a column
//   gpu_last_stats()           -> VARCHAR    dispatch + timing of the last
//                                            resident reduction (proof of
//                                            which backend actually ran)
//   gpu_drop_resident(name)    -> BOOLEAN    free the device memory
//
// Usage shape (single statement — the FROM subquery aggregates first, so the
// upload completes before the projection reads it):
//   SELECT u.n, gpu_sum_resident('l_qty') AS s
//   FROM (SELECT gpu_upload('l_qty', l_quantity::BIGINT) AS n FROM lineitem) u;
// or interactively: upload once, then query the name many times.
// WARNING: the outer query MUST reference the subquery's upload count (u.n
// above) — an unreferenced gpu_upload column is pruned by DuckDB's optimizer
// and the upload silently never runs.
//
// Guardrails (each was a shipped footgun caught by adversarial review):
//   - one gpu_upload state = one name; mixed names in one call error out
//   - NULL names error at upload (read side emits SQL NULL per NULL row)
//   - total buffered bytes capped (default 4 GB, GPUDB_UPLOAD_POOL_MAX_MB):
//     gpu_upload in a running window frame buffers O(n²) and must hit a wall
//     before it OOMs the host
//   - i64 sums wrap on overflow (uint64 accumulate — defined behavior),
//     unlike native sum(BIGINT) which promotes to HUGEINT; see KNOWN_ISSUES
//
// Dispatch goes through HybridAggregator: on a CUDA box the resident
// reductions run on the GPU (DispatchReason::Hot_GpuAlwaysWins); on a
// GPU-less machine everything transparently runs on the CPU backend — same
// SQL, no errors, keeping the LOAD-anywhere guarantee.
//
// The gpu_upload aggregate state deliberately does NOT hold pointers: it is a
// 16-byte POD {magic, buf_id} where buf_id keys a process-global pool. A
// raw-byte state copy (contract #1 in gpu_sum_extension.cpp) duplicates the
// id, not an owning pointer, so the copy still finds the buffer and a double
// destroy is a harmless double-erase. combine() never mutates its source
// (contract #3): it copies the source's buffered values into the target's
// buffer.

#include "gpu_sum_extension.hpp"
#include "gpu_backend.hpp"

#if defined(GPUDB_C_STRUCT_ABI)
DUCKDB_EXTENSION_EXTERN
#endif

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace gpudb_ext {

namespace {

// ---------------------------------------------------------------------------
// Process-global state. Function-local static => constructed on first use,
// never at load time, so LOAD stays cheap and can't throw.
// ---------------------------------------------------------------------------

struct UploadBuf {
    std::string               name;    // registry key, from the VARCHAR arg
    bool                      name_set = false; // '' is a legal explicit name
    gpudb::Dtype              dtype = gpudb::Dtype::I64;
    std::vector<std::int64_t> i64;
    std::vector<double>       f64;
    std::size_t size() const { return dtype == gpudb::Dtype::I64 ? i64.size() : f64.size(); }
    std::size_t bytes() const { return i64.size() * sizeof(std::int64_t)
                                     + f64.size() * sizeof(double); }
};

struct Globals {
    std::mutex mu;
    std::unique_ptr<gpudb::HybridAggregator> agg;                 // lazy
    std::unordered_map<std::string,
        std::unique_ptr<gpudb::ResidentColumn>> registry;         // name -> column
    std::unordered_map<std::uint64_t, UploadBuf> pool;            // buf_id -> buffer
    std::size_t   pool_bytes = 0;   // sum of bytes() over pool, kept exact
    std::uint64_t next_id = 1;
    std::string   last_stats = "no resident reduction has run yet";
};

// Cap on total host memory buffered by in-flight gpu_upload states. Without
// it, gpu_upload inside a running window frame (combine() per output row,
// every state's buffer live until query end) amplifies a KB-scale input into
// tens of GB and OOMs the host — measured n²/2 growth. Default 4 GB,
// override via GPUDB_UPLOAD_POOL_MAX_MB.
std::size_t pool_cap_bytes() {
    static const std::size_t cap = [] {
        unsigned long long mb = 4096;
        if (const char* s = std::getenv("GPUDB_UPLOAD_POOL_MAX_MB")) {
            char* end = nullptr;
            const unsigned long long v = std::strtoull(s, &end, 10);
            if (end && *end == '\0' && v > 0) mb = v;
        }
        return static_cast<std::size_t>(mb) << 20;
    }();
    return cap;
}

constexpr const char* kPoolCapHint =
    " (gpu_upload buffers exceed the pool cap; raise GPUDB_UPLOAD_POOL_MAX_MB"
    " if intentional — note gpu_upload inside a window frame buffers"
    " quadratically and is almost never what you want)";

Globals& g() {
    static Globals instance;
    return instance;
}

// Callers hold g().mu.
gpudb::HybridAggregator& agg_locked() {
    auto& G = g();
    if (!G.agg) G.agg = gpudb::make_hybrid_aggregator();
    return *G.agg;
}

void record_stats_locked(const char* op, const gpudb::AggResult& r) {
    auto& G = g();
    const auto& d = G.agg->last_decision();
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "op=%s backend=%s reason=%s rows=%zu wall_ms=%.3f kernel_ms=%.3f transfer_ms=%.3f",
        op, gpudb::to_string(d.chosen), gpudb::to_string(d.reason),
        r.rows, r.wall_ms, r.kernel_ms, r.transfer_ms);
    G.last_stats = buf;
}

// ---------------------------------------------------------------------------
// gpu_upload aggregate.
// ---------------------------------------------------------------------------

struct UploadState {
    static constexpr std::uint64_t kMagic = 0xB0FFE7C0FFEE0001ULL;
    std::uint64_t magic;
    std::uint64_t buf_id;   // 0 = no buffer yet; keys Globals::pool otherwise
};

idx_t upload_state_size(duckdb_function_info /*info*/) { return sizeof(UploadState); }

void upload_state_init(duckdb_function_info /*info*/, duckdb_aggregate_state state) {
    auto* s = reinterpret_cast<UploadState*>(state);
    s->magic  = UploadState::kMagic;
    s->buf_id = 0;
}

void upload_state_destroy(duckdb_aggregate_state* states, idx_t count) {
    std::lock_guard<std::mutex> lock(g().mu);
    auto& G = g();
    for (idx_t i = 0; i < count; ++i) {
        auto* s = reinterpret_cast<UploadState*>(states[i]);
        if (!s || s->magic != UploadState::kMagic) continue;
        if (s->buf_id != 0) {
            auto it = G.pool.find(s->buf_id);   // double-erase is a no-op
            if (it != G.pool.end()) {
                G.pool_bytes -= it->second.bytes();
                G.pool.erase(it);
            }
        }
        s->magic = 0;
    }
}

inline UploadState* probe_upload_state(void* p) {
    if (!p) return nullptr;
    auto* s = reinterpret_cast<UploadState*>(p);
    if (s->magic != UploadState::kMagic) return nullptr;
    return s;
}

// Get (or create) the pool buffer for a state. Callers hold g().mu.
UploadBuf& state_buf_locked(UploadState* s, gpudb::Dtype dt) {
    auto& G = g();
    if (s->buf_id == 0) {
        s->buf_id = G.next_id++;
        G.pool[s->buf_id].dtype = dt;
    }
    return G.pool[s->buf_id];
}

// Extract the row's VARCHAR cell as a std::string.
inline std::string read_name(duckdb_string_t* names, idx_t row) {
    return std::string(duckdb_string_t_data(&names[row]),
                       duckdb_string_t_length(names[row]));
}

template <class T, gpudb::Dtype DT>
void upload_update_t(duckdb_function_info info, duckdb_data_chunk input,
                     duckdb_aggregate_state* states) {
    duckdb_vector name_vec = duckdb_data_chunk_get_vector(input, 0);
    duckdb_vector val_vec  = duckdb_data_chunk_get_vector(input, 1);
    auto* names = reinterpret_cast<duckdb_string_t*>(duckdb_vector_get_data(name_vec));
    const T* data = reinterpret_cast<const T*>(duckdb_vector_get_data(val_vec));
    const idx_t n = duckdb_data_chunk_get_size(input);
    if (!names || !data || n == 0) return;

    uint64_t* name_validity = duckdb_vector_get_validity(name_vec);
    uint64_t* val_validity  = duckdb_vector_get_validity(val_vec);

    UploadState* s0 = probe_upload_state(states[0]);
    if (!s0) return;

    // Same constant-vector defense as gpu_sum_extension.cpp update_t: probe a
    // few indices; any magic-word miss means only states[0] is real.
    bool per_row = true;
    if (n > 1) {
        const idx_t probes[3] = { 1, n / 2, n - 1 };
        for (idx_t k = 0; k < 3 && per_row; ++k) {
            const idx_t i = probes[k];
            if (i == 0) continue;
            if (probe_upload_state(states[i]) == nullptr) per_row = false;
        }
    }

    std::lock_guard<std::mutex> lock(g().mu);
    auto& G = g();
    // Returns false after setting a function error (aborts the chunk).
    auto append_row = [&](UploadState* s, idx_t i) -> bool {
        // NULL values are skipped (standard aggregate semantics).
        if (val_validity && !duckdb_validity_row_is_valid(val_validity, i)) return true;
        // A NULL name on a contributing row is a user error, not data: the
        // old code silently registered the column under '' instead.
        if (name_validity && !duckdb_validity_row_is_valid(name_validity, i)) {
            duckdb_aggregate_function_set_error(info,
                "gpu_upload: name may not be NULL");
            return false;
        }
        UploadBuf& b = state_buf_locked(s, DT);
        const char*       nm_data = duckdb_string_t_data(&names[i]);
        const std::size_t nm_len  = duckdb_string_t_length(names[i]);
        if (!b.name_set) {
            b.name.assign(nm_data, nm_len);
            b.name_set = true;
        } else if (b.name.size() != nm_len ||
                   std::memcmp(b.name.data(), nm_data, nm_len) != 0) {
            // One state buffer = one column. Mixed names used to silently
            // merge different names' values into the first row's column.
            duckdb_aggregate_function_set_error(info,
                ("gpu_upload: one aggregate received two different names ('" +
                 b.name + "' and '" + std::string(nm_data, nm_len) +
                 "') — use a constant name, or GROUP BY the name column").c_str());
            return false;
        }
        if (G.pool_bytes + sizeof(T) > pool_cap_bytes()) {
            duckdb_aggregate_function_set_error(info,
                (std::string("gpu_upload: out of buffer memory") + kPoolCapHint).c_str());
            return false;
        }
        G.pool_bytes += (DT == gpudb::Dtype::I64) ? sizeof(std::int64_t) : sizeof(double);
        if (DT == gpudb::Dtype::I64) b.i64.push_back(static_cast<std::int64_t>(data[i]));
        else                         b.f64.push_back(static_cast<double>(data[i]));
        return true;
    };

    if (n == 1 || !per_row) {
        for (idx_t i = 0; i < n; ++i) if (!append_row(s0, i)) return;
    } else {
        for (idx_t i = 0; i < n; ++i) {
            UploadState* s = probe_upload_state(states[i]);
            if (s && !append_row(s, i)) return;
        }
    }
}

void upload_combine(duckdb_function_info info, duckdb_aggregate_state* source,
                    duckdb_aggregate_state* target, idx_t count) {
    // Contract #3: source is read-only (window donor states are combined into
    // many targets). We COPY the source buffer's contents into the target's.
    std::lock_guard<std::mutex> lock(g().mu);
    auto& G = g();
    for (idx_t i = 0; i < count; ++i) {
        UploadState* src = probe_upload_state(source[i]);
        UploadState* dst = probe_upload_state(target[i]);
        if (!src || !dst || src == dst || src->buf_id == 0) continue;
        auto it = G.pool.find(src->buf_id);
        if (it == G.pool.end() || it->second.size() == 0) continue;
        const UploadBuf& sb = it->second;
        UploadBuf& db = state_buf_locked(dst, sb.dtype);
        if (!db.name_set && sb.name_set) {
            db.name = sb.name;
            db.name_set = true;
        } else if (db.name_set && sb.name_set && db.name != sb.name) {
            duckdb_aggregate_function_set_error(info,
                ("gpu_upload: combine merged two different names ('" + db.name +
                 "' and '" + sb.name + "')").c_str());
            return;
        }
        const std::size_t delta = sb.bytes();
        if (G.pool_bytes + delta > pool_cap_bytes()) {
            // This is the window-frame quadratic-buffering path: each output
            // row's state receives a copy of its whole frame prefix.
            duckdb_aggregate_function_set_error(info,
                (std::string("gpu_upload: out of buffer memory") + kPoolCapHint).c_str());
            return;
        }
        G.pool_bytes += delta;
        db.i64.insert(db.i64.end(), sb.i64.begin(), sb.i64.end());
        db.f64.insert(db.f64.end(), sb.f64.begin(), sb.f64.end());
    }
}

void upload_finalize(duckdb_function_info info, duckdb_aggregate_state* source,
                     duckdb_vector result, idx_t count, idx_t offset) {
    if (count == 0) return;
    auto* out = reinterpret_cast<std::int64_t*>(duckdb_vector_get_data(result));
    duckdb_vector_ensure_validity_writable(result);
    uint64_t* validity = duckdb_vector_get_validity(result);

    std::lock_guard<std::mutex> lock(g().mu);
    auto& G = g();
    for (idx_t i = 0; i < count; ++i) {
        UploadState* s = probe_upload_state(source[i]);
        auto it = (s && s->buf_id != 0) ? G.pool.find(s->buf_id) : G.pool.end();
        if (it == G.pool.end() || it->second.size() == 0) {
            // No values buffered (empty input / all NULL) -> SQL NULL, no column.
            out[offset + i] = 0;
            duckdb_validity_set_row_invalid(validity, offset + i);
            continue;
        }
        UploadBuf& b = it->second;
        try {
            auto& a = agg_locked();
            std::unique_ptr<gpudb::ResidentColumn> col;
            if (b.dtype == gpudb::Dtype::I64) col = a.upload_i64(b.i64.data(), b.i64.size());
            else                              col = a.upload_f64(b.f64.data(), b.f64.size());
            G.registry[b.name] = std::move(col);  // re-upload under a name replaces
            out[offset + i] = static_cast<std::int64_t>(b.size());
        } catch (const std::exception& e) {
            duckdb_aggregate_function_set_error(info,
                (std::string("gpu_upload failed: ") + e.what()).c_str());
            return;
        }
        // Buffer stays in the pool until state_destroy — finalize must not
        // mutate state, and destroy handles cleanup unconditionally.
    }
}

duckdb_aggregate_function make_upload_fn(duckdb_type value_type,
                                         duckdb_aggregate_update_t update) {
    duckdb_aggregate_function fn = duckdb_create_aggregate_function();
    duckdb_aggregate_function_set_name(fn, "gpu_upload");
    duckdb_logical_type t_name = duckdb_create_logical_type(DUCKDB_TYPE_VARCHAR);
    duckdb_logical_type t_val  = duckdb_create_logical_type(value_type);
    duckdb_logical_type t_ret  = duckdb_create_logical_type(DUCKDB_TYPE_BIGINT);
    duckdb_aggregate_function_add_parameter(fn, t_name);
    duckdb_aggregate_function_add_parameter(fn, t_val);
    duckdb_aggregate_function_set_return_type(fn, t_ret);
    duckdb_destroy_logical_type(&t_name);
    duckdb_destroy_logical_type(&t_val);
    duckdb_destroy_logical_type(&t_ret);
    duckdb_aggregate_function_set_functions(fn,
        upload_state_size, upload_state_init, update, upload_combine, upload_finalize);
    duckdb_aggregate_function_set_destructor(fn, upload_state_destroy);
    duckdb_aggregate_function_set_special_handling(fn);
    return fn;
}

// ---------------------------------------------------------------------------
// Resident scalar functions.
// ---------------------------------------------------------------------------

// Look up a registry column by the VARCHAR cell in input vector 0, row `row`.
// Returns nullptr after setting a function error. Callers hold g().mu and
// have already skipped NULL-name rows (those produce SQL NULL, mirroring
// what DuckDB's default NULL handling does for constant NULL arguments).
gpudb::ResidentColumn* lookup_locked(duckdb_function_info info,
                                     duckdb_string_t* names, idx_t row) {
    std::string name = read_name(names, row);
    auto it = g().registry.find(name);
    if (it == g().registry.end()) {
        duckdb_scalar_function_set_error(info,
            ("no resident column named '" + name +
             "' — create one with gpu_upload('" + name + "', <col>)").c_str());
        return nullptr;
    }
    return it->second.get();
}

// One body for the three BIGINT reductions, parameterized by member call.
using ResidentOpI64 = gpudb::AggResult (gpudb::Aggregator::*)(const gpudb::ResidentColumn&);

template <ResidentOpI64 OP>
void resident_i64_exec(duckdb_function_info info, duckdb_data_chunk input,
                       duckdb_vector output) {
    duckdb_vector name_vec = duckdb_data_chunk_get_vector(input, 0);
    auto* names = reinterpret_cast<duckdb_string_t*>(duckdb_vector_get_data(name_vec));
    uint64_t* validity = duckdb_vector_get_validity(name_vec);
    const idx_t n = duckdb_data_chunk_get_size(input);
    auto* out = reinterpret_cast<std::int64_t*>(duckdb_vector_get_data(output));
    duckdb_vector_ensure_validity_writable(output);
    uint64_t* out_validity = duckdb_vector_get_validity(output);

    std::lock_guard<std::mutex> lock(g().mu);
    for (idx_t i = 0; i < n; ++i) {
        if (validity && !duckdb_validity_row_is_valid(validity, i)) {
            duckdb_validity_set_row_invalid(out_validity, i);
            continue;
        }
        gpudb::ResidentColumn* col = lookup_locked(info, names, i);
        if (!col) return;
        if (col->dtype() != gpudb::Dtype::I64) {
            duckdb_scalar_function_set_error(info,
                "resident column is DOUBLE — use gpu_sum_resident_f64 "
                "(resident f64 min/max are not in the v1 backend ABI)");
            return;
        }
        auto& a = agg_locked();
        gpudb::AggResult r = (a.*OP)(*col);
        record_stats_locked("resident_i64", r);
        out[i] = r.value_i64;
    }
}

void sum_resident_f64_exec(duckdb_function_info info, duckdb_data_chunk input,
                           duckdb_vector output) {
    duckdb_vector name_vec = duckdb_data_chunk_get_vector(input, 0);
    auto* names = reinterpret_cast<duckdb_string_t*>(duckdb_vector_get_data(name_vec));
    uint64_t* validity = duckdb_vector_get_validity(name_vec);
    const idx_t n = duckdb_data_chunk_get_size(input);
    auto* out = reinterpret_cast<double*>(duckdb_vector_get_data(output));
    duckdb_vector_ensure_validity_writable(output);
    uint64_t* out_validity = duckdb_vector_get_validity(output);

    std::lock_guard<std::mutex> lock(g().mu);
    for (idx_t i = 0; i < n; ++i) {
        if (validity && !duckdb_validity_row_is_valid(validity, i)) {
            duckdb_validity_set_row_invalid(out_validity, i);
            continue;
        }
        gpudb::ResidentColumn* col = lookup_locked(info, names, i);
        if (!col) return;
        if (col->dtype() != gpudb::Dtype::F64) {
            duckdb_scalar_function_set_error(info,
                "resident column is BIGINT — use gpu_sum_resident");
            return;
        }
        auto& a = agg_locked();
        gpudb::AggResult r = a.sum_resident_f64(*col);
        record_stats_locked("sum_resident_f64", r);
        out[i] = r.value_f64;
    }
}

void resident_info_exec(duckdb_function_info info, duckdb_data_chunk input,
                        duckdb_vector output) {
    duckdb_vector name_vec = duckdb_data_chunk_get_vector(input, 0);
    auto* names = reinterpret_cast<duckdb_string_t*>(duckdb_vector_get_data(name_vec));
    uint64_t* validity = duckdb_vector_get_validity(name_vec);
    const idx_t n = duckdb_data_chunk_get_size(input);
    duckdb_vector_ensure_validity_writable(output);
    uint64_t* out_validity = duckdb_vector_get_validity(output);

    std::lock_guard<std::mutex> lock(g().mu);
    for (idx_t i = 0; i < n; ++i) {
        if (validity && !duckdb_validity_row_is_valid(validity, i)) {
            duckdb_validity_set_row_invalid(out_validity, i);
            continue;
        }
        gpudb::ResidentColumn* col = lookup_locked(info, names, i);
        if (!col) return;
        char buf[256];
        std::snprintf(buf, sizeof(buf), "dtype=%s rows=%zu device=%s",
            col->dtype() == gpudb::Dtype::I64 ? "I64" : "F64",
            col->rows(), agg_locked().device_name().c_str());
        duckdb_vector_assign_string_element(output, i, buf);
    }
}

void last_stats_exec(duckdb_function_info /*info*/, duckdb_data_chunk input,
                     duckdb_vector output) {
    const idx_t n = duckdb_data_chunk_get_size(input);
    std::lock_guard<std::mutex> lock(g().mu);
    for (idx_t i = 0; i < n; ++i) {
        duckdb_vector_assign_string_element(output, i, g().last_stats.c_str());
    }
}

void drop_resident_exec(duckdb_function_info info, duckdb_data_chunk input,
                        duckdb_vector output) {
    duckdb_vector name_vec = duckdb_data_chunk_get_vector(input, 0);
    auto* names = reinterpret_cast<duckdb_string_t*>(duckdb_vector_get_data(name_vec));
    uint64_t* validity = duckdb_vector_get_validity(name_vec);
    const idx_t n = duckdb_data_chunk_get_size(input);
    auto* out = reinterpret_cast<bool*>(duckdb_vector_get_data(output));
    duckdb_vector_ensure_validity_writable(output);
    uint64_t* out_validity = duckdb_vector_get_validity(output);
    (void)info;

    std::lock_guard<std::mutex> lock(g().mu);
    for (idx_t i = 0; i < n; ++i) {
        if (validity && !duckdb_validity_row_is_valid(validity, i)) {
            duckdb_validity_set_row_invalid(out_validity, i);
            continue;
        }
        out[i] = g().registry.erase(read_name(names, i)) > 0;
    }
}

// Build + register one scalar function. All of these are volatile: their
// result depends on the mutable resident registry, so DuckDB must not
// constant-fold or cache them across calls.
void register_scalar(duckdb_connection con, const char* name,
                     duckdb_scalar_function_t exec,
                     duckdb_type ret, bool takes_name) {
    duckdb_scalar_function fn = duckdb_create_scalar_function();
    duckdb_scalar_function_set_name(fn, name);
    if (takes_name) {
        duckdb_logical_type t_name = duckdb_create_logical_type(DUCKDB_TYPE_VARCHAR);
        duckdb_scalar_function_add_parameter(fn, t_name);
        duckdb_destroy_logical_type(&t_name);
    }
    duckdb_logical_type t_ret = duckdb_create_logical_type(ret);
    duckdb_scalar_function_set_return_type(fn, t_ret);
    duckdb_destroy_logical_type(&t_ret);
    duckdb_scalar_function_set_volatile(fn);
    duckdb_scalar_function_set_function(fn, exec);
    duckdb_state st = duckdb_register_scalar_function(con, fn);
    duckdb_destroy_scalar_function(&fn);
    if (st == DuckDBError) {
        throw std::runtime_error(std::string(name) + " registration failed");
    }
}

} // namespace

void register_gpu_resident(duckdb_connection con) {
    // gpu_upload overload set: (VARCHAR, BIGINT) and (VARCHAR, DOUBLE).
    duckdb_aggregate_function_set set = duckdb_create_aggregate_function_set("gpu_upload");
    duckdb_aggregate_function fn_i64 =
        make_upload_fn(DUCKDB_TYPE_BIGINT, upload_update_t<std::int64_t, gpudb::Dtype::I64>);
    duckdb_aggregate_function fn_f64 =
        make_upload_fn(DUCKDB_TYPE_DOUBLE, upload_update_t<double, gpudb::Dtype::F64>);
    duckdb_state a1 = duckdb_add_aggregate_function_to_set(set, fn_i64);
    duckdb_state a2 = duckdb_add_aggregate_function_to_set(set, fn_f64);
    duckdb_destroy_aggregate_function(&fn_i64);
    duckdb_destroy_aggregate_function(&fn_f64);
    if (a1 == DuckDBError || a2 == DuckDBError) {
        duckdb_destroy_aggregate_function_set(&set);
        throw std::runtime_error("gpu_upload overload set assembly failed");
    }
    duckdb_state st = duckdb_register_aggregate_function_set(con, set);
    duckdb_destroy_aggregate_function_set(&set);
    if (st == DuckDBError) {
        throw std::runtime_error("gpu_upload function set registration failed");
    }

    register_scalar(con, "gpu_sum_resident",
        resident_i64_exec<&gpudb::Aggregator::sum_resident_i64>, DUCKDB_TYPE_BIGINT, true);
    register_scalar(con, "gpu_min_resident",
        resident_i64_exec<&gpudb::Aggregator::min_resident_i64>, DUCKDB_TYPE_BIGINT, true);
    register_scalar(con, "gpu_max_resident",
        resident_i64_exec<&gpudb::Aggregator::max_resident_i64>, DUCKDB_TYPE_BIGINT, true);
    register_scalar(con, "gpu_sum_resident_f64",
        sum_resident_f64_exec, DUCKDB_TYPE_DOUBLE, true);
    register_scalar(con, "gpu_resident_info",
        resident_info_exec, DUCKDB_TYPE_VARCHAR, true);
    register_scalar(con, "gpu_drop_resident",
        drop_resident_exec, DUCKDB_TYPE_BOOLEAN, true);
    register_scalar(con, "gpu_last_stats",
        last_stats_exec, DUCKDB_TYPE_VARCHAR, false);
}

} // namespace gpudb_ext
