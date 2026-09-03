#pragma once

// gpu_resident.hpp — the resident-set registry shared by every gpu_* function
// (v0.7 milestone 0b, docs/TRANSPARENT_DESIGN.md §5.3 / §5.5 / §5.6).
//
// One ResidentContext per registration (= per DuckDB database: the loadable
// entrypoint and the embedded CLI each call register_gpu_sum once per
// database). Every function registered against it carries the context as
// its C-API extra_info, so two databases open in one process keep two
// disjoint registries and never cross-serve a resident set.
//
// Locking model — three locks with fixed roles, never held together for long:
//   * registry lock: lookup / insert / erase / invalidate. Short. Held for
//     microseconds; never across device work.
//   * device lock:   one operator call at a time per context. The backend
//     aggregator keeps shared scratch (CUDA: stream, partials, timing events),
//     so two resident operator calls cannot overlap on it. Uploads do NOT
//     take it (they run on the column's own stream) — an upload on one
//     connection never blocks a query on another.
//   * per-column cache lock inside the backend (CUDA sort cache build).
// The gpu_upload aggregate's update callback takes NO lock at all (§5.6): a
// per-thread buffer is appended lock-free; the pool lock is taken once per
// state (buffer creation) and in combine/finalize/destroy.
//
// A set stays alive while any operator call holds a reference to it
// (std::shared_ptr): gpu_drop_resident and re-upload replace the registry
// entry immediately, the device memory goes when the last call finishes.

#if defined(GPUDB_C_STRUCT_ABI)
#include "duckdb_extension.h"
#else
#include "duckdb.h"
#endif

#include "gpu_backend.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

namespace gpudb_ext {

enum class SetState : std::uint8_t {
    Ready,      // uploaded and prepared (sort cache built) — a hit costs no sort
    Uploaded,   // uploaded, derived structures built lazily by the first operator
    Stale       // invalidated; operators refuse it, gpu_assert_rows raises
};
const char* to_string(SetState s) noexcept;

// One registry entry: a bare column (`keys` only) or a (key, payload) pair.
struct ResidentSet {
    std::string   name;             // registry key: the upload name / identity tag
    bool          managed = false;  // origin: tag-shaped name ("gpudb:v1:...") → managed
    // Parsed identity (managed sets only; empty / -1 otherwise).
    std::string   catalog, schema, table, columns, extra;
    std::int64_t  table_oid = -1;

    bool          pair = false;
    std::unique_ptr<gpudb::ResidentColumn> keys;   // bare column, or pair.k
    std::unique_ptr<gpudb::ResidentColumn> vals;   // pair.v (null for a bare column)
    std::size_t   rows = 0;          // rows in the column(s): NULL rows are skipped
    std::size_t   rows_seen = 0;     // rows the upload scan delivered (count(*) of its input)
    std::uint64_t epoch = 0;         // invalidation sequence at registration
    std::int64_t  uploaded_at_us = 0;

    std::atomic<SetState>      state { SetState::Uploaded };
    std::atomic<std::uint64_t> hits { 0 };
    std::atomic<std::int64_t>  last_used_at_us { 0 };
    std::mutex                 stats_mu;
    std::string                last_stats;   // per-set copy of gpu_last_stats()

    std::size_t resident_bytes() const noexcept {
        return (keys ? keys->resident_bytes() : 0) + (vals ? vals->resident_bytes() : 0);
    }
};

class ResidentContext;

std::shared_ptr<ResidentContext> make_resident_context();

// extra_info plumbing: one heap holder per registered function, released by
// DuckDB through resident_extra_info_destroy when the function goes away.
void* resident_extra_info(const std::shared_ptr<ResidentContext>& ctx);
void  resident_extra_info_destroy(void* holder);
ResidentContext& resident_context(void* extra_info);   // from duckdb_*_get_extra_info

// A resolved column plus the reference that keeps its set alive.
struct ResidentRef {
    std::shared_ptr<ResidentSet> set;
    gpudb::ResidentColumn*       col = nullptr;
};

// Registry lookups (registry lock only). Both throw std::runtime_error naming
// `fn` when the set is unknown or stale. Names resolve as:
//   'p'   → the set 'p' (bare column, or the pair's key column for
//           resident_acquire_column)
//   'p.k' → the key column of pair set 'p';  'p.v' → its payload column
// Every successful acquire counts a hit and stamps last_used.
std::shared_ptr<ResidentSet> resident_acquire_set(ResidentContext& ctx,
                                                  const std::string& name, const char* fn);
ResidentRef resident_acquire_column(ResidentContext& ctx,
                                    const std::string& name, const char* fn);

// The device lock (see the header comment) and the context's aggregator.
std::unique_lock<std::mutex> resident_device_lock(ResidentContext& ctx);
gpudb::HybridAggregator&     resident_aggregator(ResidentContext& ctx);

// Record a stats line: the context's gpu_last_stats() and, if given, the set's.
void resident_record_stats(ResidentContext& ctx, ResidentSet* set, const std::string& line);

// Registration entrypoints. register_gpu_resident(con) alone makes a fresh
// context; register_gpu_sum() shares one context across the resident, join
// and group-by function families.
void register_gpu_resident(duckdb_connection con,
                           const std::shared_ptr<ResidentContext>& ctx);

} // namespace gpudb_ext
