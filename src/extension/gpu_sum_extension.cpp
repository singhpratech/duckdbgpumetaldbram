// gpu_sum_extension.cpp — register the gpu_sum / gpu_min / gpu_max aggregates
// on a DuckDB connection, each as an overload SET carrying a (BIGINT)->BIGINT
// and a (DOUBLE)->DOUBLE variant.
//
// Public function:
//   void register_gpu_sum(duckdb_connection con);
//
// Used by:
//   - benchmark/sql_demo.cpp (CLI demo: opens a duckdb db, registers, runs SQL)
//   - the loadable extension wrapper (DuckDB C API _init_c_api entry point in
//     duckdb_loadable.cpp) for `LOAD '...'` in the DuckDB CLI
//
// Design (v0.3.0): STREAMING running accumulators
// -------------------------------------------------
//   Each aggregate state holds a running accumulator (an int64 sum/min/max or
//   a double sum/min/max) plus a non-NULL count. update() folds every incoming
//   value straight into the state; combine() folds one state into another;
//   finalize() reads the accumulator out. There is NO per-state buffer, no
//   process-global pool, and no deferred GPU reduction. This is exactly the
//   algorithmic shape of a native DuckDB aggregate.
//
//   Why we removed the previous "buffer every value, reduce on the GPU at
//   finalize" design: see BENCHMARK.md 2026-07-19 end-to-end section. The
//   buffered path lost 3x-110x to native because it copied the entire column
//   into per-state std::vectors before doing any work; running accumulators
//   match native's shape and never pay that copy. On UMA (Apple Silicon) a
//   value copied out to feed a GPU reduction is a memory pass the CPU already
//   gets for free during the scan — so shipping the column to the GPU only to
//   sum it is pure overhead here. The GPU is deliberately NOT used in this
//   aggregate path anymore; that is the intended v0.3.0 design, not an
//   omission. (The device reductions in src/backends/ remain, exercised by the
//   standalone benchmarks; they are simply not on this SQL aggregate path.)
//
// Two hard-won behavioral contracts survive the rewrite unchanged — they are
// the reason the state is a bare POD guarded by a magic word:
//
//   1. Raw-byte state copies (RadixPartitionedTupleData).
//      DuckDB's grouped-aggregate hash table repartitions by COPYING STATE
//      BYTES from one tuple to another with a raw memcpy — it does NOT call
//      our state_init, nor any move/copy callback. Anything in the state must
//      therefore be POD-copy-safe. An inline accumulator is: a memcpy'd copy
//      of {magic,count,bits} is, in practice, a MOVE — the abandoned original
//      is never combined or finalized again, so both copies holding the same
//      running total causes no double-counting. (This is why we could never
//      use a std::vector inline: a raw-byte copy duplicates its {ptr,size,cap}
//      triple and the second destructor frees a pointer the first still uses.)
//
//   2. WindowConstantAggregator constant-vector states.
//      For SUM(x) OVER () — the unbounded-frame case — DuckDB calls update()
//      with a CONSTANT_VECTOR state array: only states[0] points at real
//      storage; states[1..N-1] are out-of-bounds reads off a 1-pointer buffer
//      (heap garbage). The C API does not flatten the state vector before
//      handing it to us. Our defense is the magic word: we probe a pointer's
//      magic before trusting it, and treat any probe failure past states[0] as
//      "this is a constant vector, only states[0] is mine." Keep this probe
//      strategy exactly — it is the shipped fix.
//
//   3. combine() must NOT mutate the source state.
//      DuckDB's window operator builds a partial-aggregate "donor" state per
//      partition and calls combine() many times with that SAME source into
//      different per-row destinations. If combine() mutated the source, every
//      subsequent per-row combine against it would see corrupted data. We only
//      ever READ from source and fold into target.
//
// NULL semantics (shipped in v0.2.0, PR #44 — must keep passing):
//   SUM/MIN/MAX over zero non-NULL values (empty input, or a group whose every
//   row is NULL) is SQL NULL, not 0. count == 0 is exactly that signal, and
//   finalize marks the output row invalid in that case.
//
// Concurrency:
//   The state is self-contained and touched only through the per-state
//   pointers DuckDB hands each callback; there is no shared mutable singleton
//   on this path, so no global lock is required. DuckDB's own parallelism
//   partitions states across threads.

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

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace gpudb_ext {

namespace {

// ---------------------------------------------------------------------------
// Aggregate state — a 24-byte POD, one layout shared by all six overloads.
//
//   magic  — sentinel proving a pointer is one of our states even after a
//            raw-byte copy (see contract #1/#2 in the file header). Bumped to
//            ...02 for v0.3.0 so a stale slot left by an older loaded build can
//            never accidentally pass the probe.
//   count  — number of non-NULL values folded in so far. count == 0 is the
//            SQL-NULL signal at finalize.
//   bits   — the running accumulator's bit pattern: an int64 (sum/min/max) or
//            a double (sum/min/max), reinterpreted per overload via to_bits /
//            from_bits. Storing bits (not a typed union) keeps the state a
//            single trivially-copyable layout regardless of the overload.
// ---------------------------------------------------------------------------
struct AggState {
    static constexpr std::uint64_t kMagic = 0xC0FFEEDEADBEEF02ULL;  // note: 02, bumped from 01
    std::uint64_t magic;
    std::uint64_t count;   // number of non-NULL values accumulated
    std::uint64_t bits;    // accumulator bit pattern: int64 or double per overload
};

// ---------------------------------------------------------------------------
// Op policy structs — one per (operation, type). Everything below templates
// over these: init() is the fold identity, fold(a, v) folds a value into the
// accumulator. All ops are associative & commutative with their identity, which
// is what lets the streaming single-state / per-state / combine paths be
// correct for MIN/MAX just as they are for SUM.
//
// f64 MIN/MAX use a NaN-aware TOTAL ORDER that matches native DuckDB, where
// NaN sorts GREATEST (the same order as ORDER BY: NaN comes last). A bare
// `v < a` / `v > a` fold is wrong for f64 because IEEE comparisons with NaN are
// always false — a NaN input would never displace the seed, so an all-NaN group
// would leak the fold identity (+inf/-inf), and MAX({nan, x}) would drop the
// NaN that native returns. less_total / greater_total below implement the total
// order instead.
//
// Because NaN is the greatest element, the identity for f64 MIN must be the
// GREATEST element — which is NaN, not +inf (+inf < NaN under this order). f64
// MAX's identity stays -inf (the least element). The identity still never leaks
// into visible output: count == 0 -> SQL NULL (finalize marks the row invalid);
// count > 0 over an all-NaN group -> NaN, which is the CORRECT answer, not a
// leaked seed. Integer MIN/MAX have no NaN and keep the plain comparison.
inline bool less_total(double x, double y)    { return std::isnan(y) ? !std::isnan(x) : (!std::isnan(x) && x < y); }
inline bool greater_total(double x, double y) { return std::isnan(x) ? !std::isnan(y) : (!std::isnan(y) && x > y); }

struct OpSumI64 { using T = std::int64_t; static T init() { return 0; }                                     static T fold(T a, T v) { return a + v; } };
struct OpMinI64 { using T = std::int64_t; static T init() { return std::numeric_limits<std::int64_t>::max(); } static T fold(T a, T v) { return v < a ? v : a; } };
struct OpMaxI64 { using T = std::int64_t; static T init() { return std::numeric_limits<std::int64_t>::min(); } static T fold(T a, T v) { return v > a ? v : a; } };
struct OpSumF64 { using T = double; static T init() { return 0.0; }                                        static T fold(T a, T v) { return a + v; } };
struct OpMinF64 { using T = double; static T init() { return std::numeric_limits<double>::quiet_NaN(); }   static T fold(T a, T v) { return less_total(v, a) ? v : a; } };
struct OpMaxF64 { using T = double; static T init() { return -std::numeric_limits<double>::infinity(); }   static T fold(T a, T v) { return greater_total(v, a) ? v : a; } };

// Bit-pattern helpers. We store the accumulator as a uint64 bit pattern in the
// state and reinterpret it as the op's value type on the way in/out. memcpy is
// the blessed C++17 bit-cast (no std::bit_cast until C++20; a reinterpret_cast
// through a differently-typed pointer would be a strict-aliasing violation).
template <class T> inline std::uint64_t to_bits(T v) { std::uint64_t b; std::memcpy(&b, &v, sizeof(b)); return b; }
template <class T> inline T from_bits(std::uint64_t b) { T v; std::memcpy(&v, &b, sizeof(v)); return v; }

// ---------------------------------------------------------------------------
// Callbacks. Templated over the Op policy but plain (non-member) functions, so
// each instantiation's address is usable as the corresponding C callback.
// ---------------------------------------------------------------------------

// state_size — shared across all overloads (one POD layout).
idx_t state_size(duckdb_function_info /*info*/) { return sizeof(AggState); }

// state_init — stamp the magic word, zero the count, seed the accumulator with
// the op's identity so the very first fold is correct (SUM->0; i64 MIN/MAX ->
// INT64_MAX/INT64_MIN; f64 MIN->NaN and MAX->-inf under the NaN-greatest total
// order — see the op-policy comment above).
template <class OP>
void state_init_t(duckdb_function_info /*info*/, duckdb_aggregate_state state) {
    auto* s = reinterpret_cast<AggState*>(state);
    s->magic = AggState::kMagic;
    s->count = 0;
    s->bits  = to_bits(OP::init());
}

// state_destroy — shared. There is nothing to free now (no pool, no buffer), so
// the ONLY thing we do is wipe the magic word. That matters: DuckDB may later
// reuse or read this slot, and a stale slot that still carried our magic could
// wrongly pass the probe in a subsequent call. Zeroing magic makes any later
// probe of this address fail, exactly as it should.
void state_destroy(duckdb_aggregate_state* states, idx_t count) {
    for (idx_t i = 0; i < count; ++i) {
        auto* s = reinterpret_cast<AggState*>(states[i]);
        if (!s) continue;
        if (s->magic != AggState::kMagic) continue;
        s->magic = 0;
    }
}

// Defensive accessor: returns the state pointer iff it looks like one of ours
// (non-null AND magic matches). Returns nullptr otherwise. This is the single
// probe that handles BOTH the WindowConstantAggregator OOB reads (contract #2)
// and confirms a radix-copied slot (contract #1) is genuinely ours before we
// dereference it.
inline AggState* probe_state(void* p) {
    if (!p) return nullptr;
    auto* s = reinterpret_cast<AggState*>(p);
    if (s->magic != AggState::kMagic) return nullptr;
    return s;
}

template <class OP>
void update_t(duckdb_function_info /*info*/, duckdb_data_chunk input,
              duckdb_aggregate_state* states) {
    using T = typename OP::T;
    duckdb_vector vec = duckdb_data_chunk_get_vector(input, 0);
    const T* data = reinterpret_cast<const T*>(duckdb_vector_get_data(vec));
    const idx_t n = duckdb_data_chunk_get_size(input);
    if (data == nullptr || n == 0) return;

    // Validity bitmap: 1 bit per row, 64 rows per uint64. NULL rows are skipped
    // (matches DuckDB's SQL semantics for SUM/MIN/MAX). nullptr means no NULLs.
    uint64_t* validity = duckdb_vector_get_validity(vec);

    // Resolve states[0] up front; it's always valid (DuckDB never gives us an
    // empty state vector here). The magic-word check is a defense against a
    // malformed call but should always pass.
    AggState* s0 = probe_state(states[0]);
    if (!s0) return;

    // Determine whether the rest of states[] are real, distinct per-row states,
    // or just OOB garbage from a CONSTANT_VECTOR (the OVER () case).
    //
    // We probe a few indices and ask: does each pointer look like one of our
    // states (magic word OK)? If yes, we have a real per-row state vector and
    // use the per-row path. If any probe fails the magic-word test, the caller
    // passed us a constant state vector and only states[0] is meaningful.
    bool per_row = true;
    if (n > 1) {
        const idx_t probes[3] = { 1, n / 2, n - 1 };
        for (idx_t k = 0; k < 3 && per_row; ++k) {
            const idx_t i = probes[k];
            if (i == 0) continue;
            if (probe_state(states[i]) == nullptr) per_row = false;
        }
    }

    if (n == 1 || !per_row) {
        // Single-state path: either ungrouped, or a constant-vector window.
        // Accumulate into a LOCAL running value first, then fold that single
        // partial into states[0] once. Folding a local partial (rather than the
        // state per row) keeps this tight and, crucially, only touches the
        // state's memory once.
        T acc = OP::init();
        std::uint64_t cnt = 0;
        if (validity == nullptr) {
            for (idx_t i = 0; i < n; ++i) { acc = OP::fold(acc, data[i]); ++cnt; }
        } else {
            for (idx_t i = 0; i < n; ++i) {
                if (duckdb_validity_row_is_valid(validity, i)) {
                    acc = OP::fold(acc, data[i]);
                    ++cnt;
                }
            }
        }
        if (cnt > 0) {
            s0->bits = to_bits(OP::fold(from_bits<T>(s0->bits), acc));
            s0->count += cnt;
        }
        return;
    }

    // Per-row path: GROUP BY routes different rows to different states. Each
    // states[i] was spot-checked above, but we still probe every index (cheap)
    // to stay robust and to skip any that don't check out.
    if (validity == nullptr) {
        for (idx_t i = 0; i < n; ++i) {
            AggState* s = probe_state(states[i]);
            if (s) { s->bits = to_bits(OP::fold(from_bits<T>(s->bits), data[i])); s->count++; }
        }
    } else {
        for (idx_t i = 0; i < n; ++i) {
            if (!duckdb_validity_row_is_valid(validity, i)) continue;
            AggState* s = probe_state(states[i]);
            if (s) { s->bits = to_bits(OP::fold(from_bits<T>(s->bits), data[i])); s->count++; }
        }
    }
}

template <class OP>
void combine_t(duckdb_function_info /*info*/, duckdb_aggregate_state* source,
               duckdb_aggregate_state* target, idx_t count) {
    using T = typename OP::T;
    // CRITICAL: combine() must NOT mutate the source state (contract #3).
    //
    // DuckDB's window operator (e.g. SUM() OVER (PARTITION BY g ORDER BY k))
    // builds a partial-aggregate state for each partition and then, for each
    // row's window, calls combine() with that partition state as the SOURCE and
    // a fresh per-row state as the DESTINATION. The same source is used many
    // times — it is read-only from combine()'s perspective. We only ever read
    // from source and fold into target.
    //
    // probe_state handles the radix-copied case too: a raw-byte state copy
    // carries the same magic and the same running total, so folding it is
    // correct.
    for (idx_t i = 0; i < count; ++i) {
        AggState* src = probe_state(source[i]);
        AggState* dst = probe_state(target[i]);
        if (!src || !dst) continue;
        if (src == dst) continue;          // self-combine: nothing to do.
        if (src->count == 0) continue;     // empty donor contributes nothing.
        dst->bits = to_bits(OP::fold(from_bits<T>(dst->bits), from_bits<T>(src->bits)));
        dst->count += src->count;
    }
}

// Determine whether sources[0..count-1] are distinct per-state (grouped /
// per-row finalize) or a constant-state broadcast (window OVER ()). Same probe
// logic as update()'s spot-check.
inline bool finalize_is_per_state(duckdb_aggregate_state* source, idx_t count) {
    if (count <= 1) return false;
    const idx_t probes[3] = { 1, count / 2, count - 1 };
    for (idx_t k = 0; k < 3; ++k) {
        const idx_t i = probes[k];
        if (i == 0) continue;
        if (probe_state(source[i]) == nullptr) return false;
    }
    return true;
}

template <class OP>
void finalize_t(duckdb_function_info /*info*/, duckdb_aggregate_state* source,
                duckdb_vector result, idx_t count, idx_t offset) {
    using T = typename OP::T;
    if (count == 0) return;  // no output rows requested — nothing to write.
    T* out = reinterpret_cast<T*>(duckdb_vector_get_data(result));

    // A group with zero non-NULL values (empty input, or every row NULL) is SQL
    // NULL, not 0 — count == 0 is exactly that signal (v0.2.0 PR #44). Make the
    // output validity mask writable once up front; write() marks NULL rows.
    duckdb_vector_ensure_validity_writable(result);
    uint64_t* validity = duckdb_vector_get_validity(result);

    auto write = [&](idx_t row, const AggState* s) {
        if (!s || s->count == 0) {
            out[row] = T{};
            duckdb_validity_set_row_invalid(validity, row);
        } else {
            out[row] = from_bits<T>(s->bits);
        }
    };

    // Single-state fast path: ungrouped SELECT gpu_*(x) FROM tbl, and per-row
    // finalize calls from window functions without PARTITION BY.
    if (count == 1) {
        write(offset, probe_state(source[0]));
        return;
    }

    // Constant-state broadcast: WindowConstantAggregator passes a
    // CONSTANT_VECTOR for source. Only source[0] is real; broadcast it.
    if (!finalize_is_per_state(source, count)) {
        AggState* s = probe_state(source[0]);
        for (idx_t i = 0; i < count; ++i) write(offset + i, s);
        return;
    }

    // Grouped path: count > 1 with all distinct registered states.
    for (idx_t i = 0; i < count; ++i) write(offset + i, probe_state(source[i]));
}

// ---------------------------------------------------------------------------
// Registration.
// ---------------------------------------------------------------------------

// Build (but do not register) one aggregate overload: (param_type) ->
// param_type, wired to the templated callbacks for OP. The return type mirrors
// the parameter type (BIGINT->BIGINT, DOUBLE->DOUBLE). Caller owns the returned
// handle and must duckdb_destroy_aggregate_function() it.
template <class OP>
duckdb_aggregate_function make_fn(const char* name, duckdb_type param_type) {
    duckdb_aggregate_function fn = duckdb_create_aggregate_function();
    duckdb_aggregate_function_set_name(fn, name);
    duckdb_logical_type t = duckdb_create_logical_type(param_type);
    duckdb_aggregate_function_add_parameter(fn, t);
    duckdb_aggregate_function_set_return_type(fn, t);
    duckdb_destroy_logical_type(&t);
    duckdb_aggregate_function_set_functions(fn,
        state_size, state_init_t<OP>, update_t<OP>, combine_t<OP>, finalize_t<OP>);
    duckdb_aggregate_function_set_destructor(fn, state_destroy);
    // We handle NULL semantics ourselves (empty / all-NULL input -> SQL NULL via
    // the output validity mask in finalize). Special handling tells DuckDB not
    // to short-circuit our aggregate on NULL input; update() already skips NULL
    // rows via the input validity bitmap.
    duckdb_aggregate_function_set_special_handling(fn);
    return fn;
}

// Register one aggregate NAME as a function SET carrying two overloads:
//   NAME(BIGINT) -> BIGINT   (OPI)
//   NAME(DOUBLE) -> DOUBLE   (OPF)
// A function set is the C API's mechanism for same-name overloads; DuckDB's
// binder resolves the overload per call. Smaller integer types (INTEGER /
// SMALLINT / TINYINT) carry no dedicated overload — they widen to the BIGINT
// overload via implicit integer widening (cheaper than the int->double cast, so
// it wins overload resolution).
template <class OPI, class OPF>
void register_set(duckdb_connection con, const char* name) {
    duckdb_aggregate_function_set set = duckdb_create_aggregate_function_set(name);
    duckdb_aggregate_function fn_i64 = make_fn<OPI>(name, DUCKDB_TYPE_BIGINT);
    duckdb_aggregate_function fn_f64 = make_fn<OPF>(name, DUCKDB_TYPE_DOUBLE);
    duckdb_state a1 = duckdb_add_aggregate_function_to_set(set, fn_i64);
    duckdb_state a2 = duckdb_add_aggregate_function_to_set(set, fn_f64);
    duckdb_destroy_aggregate_function(&fn_i64);
    duckdb_destroy_aggregate_function(&fn_f64);
    if (a1 == DuckDBError || a2 == DuckDBError) {
        duckdb_destroy_aggregate_function_set(&set);
        throw std::runtime_error(std::string(name) + " overload set assembly failed");
    }
    duckdb_state st = duckdb_register_aggregate_function_set(con, set);
    duckdb_destroy_aggregate_function_set(&set);
    if (st == DuckDBError) {
        throw std::runtime_error(std::string(name) + " function set registration failed");
    }
}

} // namespace

void register_gpu_sum(duckdb_connection con) {
    register_set<OpSumI64, OpSumF64>(con, "gpu_sum");
    register_set<OpMinI64, OpMinF64>(con, "gpu_min");
    register_set<OpMaxI64, OpMaxF64>(con, "gpu_max");
    std::fprintf(stderr,
        "[gpudb] registered gpu_sum / gpu_min / gpu_max (BIGINT,DOUBLE) streaming aggregates (backend=%s)\n",
        gpudb::to_string(gpudb::default_backend()));
}

} // namespace gpudb_ext
