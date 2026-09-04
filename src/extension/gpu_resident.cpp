// gpu_resident.cpp — resident-column SQL surface: the path where the GPU
// actually wins.
//
// Public functions:
//   void register_gpu_resident(duckdb_connection con, ctx);   (called by
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
//   gpu_upload_pair(name, k BIGINT, v BIGINT|DOUBLE) -> BIGINT   (key, payload)
//                                                    pair in ONE scan
//   gpu_sum_resident(name)     -> BIGINT     reduce the resident column —
//   gpu_min_resident(name)     -> BIGINT     no per-query transfer. i64 only:
//   gpu_max_resident(name)     -> BIGINT     the v1 ABI has no resident f64
//   gpu_sum_resident_f64(name) -> DOUBLE     min/max (see gpu_backend.hpp).
//   gpu_resident_info(name)    -> VARCHAR    dtype/rows/device/state of a column
//   gpu_residents()            -> TABLE      every resident set with identity,
//                                            origin, state, size, hits, stats
//   gpu_last_stats()           -> VARCHAR    dispatch + timing of the last
//                                            resident operator in this
//                                            database (per-set copy in
//                                            gpu_residents().last_stats)
//   gpu_prepare_resident(name) -> BOOLEAN    build the derived structures now
//   gpu_assert_rows(name, n)   -> BOOLEAN    typed "GPUDB_STALE:" error unless
//                                            n == rows the upload scan saw
//   gpu_invalidate(pattern)    -> BIGINT     mark sets stale (name == pattern
//                                            or name starts with pattern + ':');
//                                            drops open upload sessions too
//   gpu_upload_begin(name)     -> BOOLEAN    open an upload session: every
//                                            gpu_upload[_pair](name, ...) statement
//                                            until finish appends a segment
//   gpu_upload_finish(name)    -> BIGINT     device copy + prepare + publish
//   gpu_upload_abort(name)     -> BOOLEAN    drop the session and its buffers
//   gpu_upload_status(name)    -> VARCHAR    JSON: open, segments, rows, rows_seen,
//                                            bytes, kind, epoch, invalidated
//   gpu_drop_resident(name)    -> BOOLEAN    free the device memory
//
// Usage shape (single statement — the FROM subquery aggregates first, so the
// upload completes before the projection reads it):
//   SELECT u.n, gpu_sum_resident('l_qty') AS s
//   FROM (SELECT gpu_upload('l_qty', l_quantity::BIGINT) AS n FROM lineitem) u;
//
// Set names and origin (docs/TRANSPARENT_DESIGN.md §5.3): a name of the form
//   gpudb:v1:<catalog>:<schema>:<table>:<table_oid>:<col1>[,<col2>...][:<extra>]
// is an IDENTITY TAG; sets created under a tag are `managed` (the wrapper's,
// consumed by the transparent rewrite) and the fields are parsed and shown
// by gpu_residents(). Fields may not contain ':' (columns may not contain
// ','); <extra> is opaque to the extension (the wrapper puts its row count /
// epoch there). Any other name is an `explicit` set. The prefix "gpudb:" is
// reserved: a name that starts with it but does not parse is an error.
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
// The gpu_upload aggregate state deliberately does NOT own anything: it is
// a 24-byte POD {magic, buf_id, buf*} where buf_id keys a process-global
// pool that owns the buffer (a list of fixed 8 MiB segments — see Segment). A raw-byte state copy (contract #1 in
// gpu_sum_extension.cpp) duplicates the id and the pointer, not ownership:
// the copy still finds the buffer, the abandoned original is never touched
// again, and a double destroy is a harmless double-erase (destroy goes
// through the id, never the pointer). combine() never mutates its source
// (contract #3): it copies the source's buffered values into the target's
// buffer.
//
// Threading (v0.7 milestone 0b, §5.6): update() takes NO lock. DuckDB's
// worker pool is shared across connections, so a lock inside the per-thread
// update callback stalls every scan thread of the upload statement and
// starves a native query on another connection (measured 10× on SF10). The
// per-thread buffer is appended lock-free; the pool lock is taken once per
// state (creation) and in combine / finalize / destroy; the pool byte cap is
// an atomic charged once per chunk.

#include "gpu_resident.hpp"
#include "gpu_sum_extension.hpp"
#include "gpu_backend.hpp"

#if defined(GPUDB_C_STRUCT_ABI)
DUCKDB_EXTENSION_EXTERN
#endif

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <climits>
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

#if defined(__linux__)
#include <sys/mman.h>
#endif

namespace gpudb_ext {

const char* to_string(SetState s) noexcept {
    switch (s) {
        case SetState::Ready:    return "ready";
        case SetState::Uploaded: return "uploaded";
        case SetState::Stale:    return "stale";
    }
    return "?";
}

namespace {

// GPUDB_UPLOAD_TRACE=1 prints the phases of every upload finalize to stderr
// (buffer rows, de-interleave / H2D / prepare / publish ms): the residency
// gate uses it to attribute contention to a phase.
bool upload_trace() {
    static const bool on = std::getenv("GPUDB_UPLOAD_TRACE") != nullptr;
    return on;
}
double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

std::int64_t now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

bool starts_with(const std::string& s, const std::string& p) {
    return s.size() >= p.size() && std::memcmp(s.data(), p.data(), p.size()) == 0;
}

// ---------------------------------------------------------------------------
// Identity tags.
// ---------------------------------------------------------------------------

constexpr const char* kTagPrefix = "gpudb:";

struct TagFields {
    std::string catalog, schema, table, columns, extra;
    std::int64_t table_oid = -1;
};

// Returns "" on success, else the error text. Only called for names that
// start with kTagPrefix.
std::string parse_tag(const std::string& name, TagFields& out) {
    std::vector<std::string> f;
    std::size_t pos = 0;
    while (true) {
        const std::size_t c = name.find(':', pos);
        if (c == std::string::npos) { f.push_back(name.substr(pos)); break; }
        f.push_back(name.substr(pos, c - pos));
        pos = c + 1;
        if (f.size() == 7) { f.push_back(name.substr(pos)); break; }   // rest = extra
    }
    const std::string usage =
        " — expected gpudb:v1:<catalog>:<schema>:<table>:<table_oid>:<columns>[:<extra>]";
    if (f.size() < 7)           return "malformed identity tag '" + name + "'" + usage;
    if (f[1] != "v1")           return "unsupported identity tag version '" + f[1] + "' in '" + name + "'" + usage;
    if (f[4].empty())           return "identity tag '" + name + "' has an empty table name" + usage;
    if (f[6].empty())           return "identity tag '" + name + "' has an empty column list" + usage;
    char* end = nullptr;
    errno = 0;
    const long long oid = std::strtoll(f[5].c_str(), &end, 10);
    if (errno != 0 || !end || *end != '\0' || f[5].empty() || oid < 0)
        return "identity tag '" + name + "' has a non-integer table_oid '" + f[5] + "'" + usage;
    out.catalog = f[2]; out.schema = f[3]; out.table = f[4];
    out.table_oid = oid; out.columns = f[6];
    out.extra = f.size() > 7 ? f[7] : std::string();
    return "";
}

// ---------------------------------------------------------------------------
// Host upload pool — process-global (host memory is one budget per process;
// state destroy has no access to a context). Owns the buffers behind
// in-flight gpu_upload states.
// ---------------------------------------------------------------------------

// One append-only block of 8-byte lanes (int64, or the raw bits of a double).
// Fixed size, allocated once, never reallocated: no doubling copies, no
// transient 2× peak, and on Linux a 2 MiB-aligned block that asks for
// transparent huge pages (512× fewer page faults than 4 KiB pages — the
// faults are what a concurrent native query feels, they serialize on the
// process's mmap lock). Segments are shared between states after combine()
// and freed with the last reference.
struct Segment {
    static constexpr std::size_t kLanes = std::size_t(1) << 20;   // 8 MiB
    std::int64_t* data = nullptr;
    std::size_t   n = 0;
    Segment() {
#if defined(__linux__)
        void* p = nullptr;
        if (posix_memalign(&p, std::size_t(2) << 20, kLanes * sizeof(std::int64_t)) != 0 || !p)
            throw std::bad_alloc();
        madvise(p, kLanes * sizeof(std::int64_t), MADV_HUGEPAGE);
        data = static_cast<std::int64_t*>(p);
#else
        data = static_cast<std::int64_t*>(std::malloc(kLanes * sizeof(std::int64_t)));
        if (!data) throw std::bad_alloc();
#endif
    }
    ~Segment() { std::free(data); }
    Segment(const Segment&) = delete;
    Segment& operator=(const Segment&) = delete;
    std::size_t room() const noexcept { return kLanes - n; }
};

// A view of a segment as seen by one buffer: `lanes` is frozen at the time
// the view was taken, so a combine() target never sees lanes a still-open
// source segment gains afterwards (contract #3: the source is read-only
// to us, and it may keep growing for someone else).
struct SegView {
    std::shared_ptr<Segment> seg;
    std::size_t lanes = 0;
};

struct UploadBuf {
    std::string               name;    // registry key, from the VARCHAR arg
    bool                      name_set = false; // '' is a legal explicit name
    bool                      managed = false;
    TagFields                 tag;
    gpudb::Dtype              dtype = gpudb::Dtype::I64;   // lane interpretation
    std::vector<SegView>      views;   // sealed views (own sealed segments + combined sources)
    std::shared_ptr<Segment>  open;    // this state's writable segment (not in views)
    std::size_t               rows_seen = 0;    // rows delivered to update()
    std::uint64_t             seq_at_start = 0; // invalidation seq when named
    std::size_t               charged = 0;      // bytes charged to the pool cap

    std::size_t lanes() const noexcept {
        std::size_t t = open ? open->n : 0;
        for (const auto& v : views) t += v.lanes;
        return t;
    }
    // Lane append; `need` lanes must land in ONE segment (pairs stay whole).
    std::int64_t* reserve_lanes(std::size_t need) {
        if (!open || open->room() < need) {
            if (open) views.push_back(SegView{open, open->n});
            open = std::make_shared<Segment>();
        }
        std::int64_t* dst = open->data + open->n;
        open->n += need;
        return dst;
    }
    // Every view including the open segment, in order.
    std::vector<SegView> all_views() const {
        std::vector<SegView> v = views;
        if (open && open->n) v.push_back(SegView{open, open->n});
        return v;
    }
};

struct Pool {
    std::mutex mu;
    std::unordered_map<std::uint64_t, std::unique_ptr<UploadBuf>> bufs;   // buf_id -> buffer
    std::atomic<std::size_t>   bytes { 0 };    // sum of charged over bufs
    std::atomic<std::uint64_t> next_id { 1 };
};

Pool& pool() {
    static Pool instance;   // function-local static: never constructed at LOAD
    return instance;
}

// Cap on total host memory buffered by in-flight gpu_upload states. Without
// it, gpu_upload inside a running window frame (combine() per output row,
// every state's buffer live until query end) amplifies a KB-scale input into
// tens of GB and OOMs the host — measured n²/2 growth. Default 4 GB,
// override via GPUDB_UPLOAD_POOL_MAX_MB. This is the HOST-side cap; device
// memory is the wrapper's budget (docs/TRANSPARENT_DESIGN.md §5.5).
std::size_t pool_cap_bytes() {
    static const std::size_t cap = [] {
        unsigned long long mb = 4096;
        if (const char* s = std::getenv("GPUDB_UPLOAD_POOL_MAX_MB")) {
            char* end = nullptr;
            errno = 0;
            const unsigned long long v = std::strtoull(s, &end, 10);
            if (errno == 0 && end && end != s && *end == '\0' && v > 0 &&
                v <= (static_cast<unsigned long long>(SIZE_MAX) >> 20)) mb = v;
            else std::fprintf(stderr, "[gpudb] ignoring GPUDB_UPLOAD_POOL_MAX_MB='%s'; using %llu\n", s, mb);
        }
        return static_cast<std::size_t>(mb) << 20;
    }();
    return cap;
}

constexpr const char* kPoolCapHint =
    " (gpu_upload buffers exceed the pool cap; raise GPUDB_UPLOAD_POOL_MAX_MB"
    " if intentional — note gpu_upload inside a window frame buffers"
    " quadratically and is almost never what you want)";

// Pool-cap accounting is per CHUNK, not per row: one reservation of the
// chunk's upper bound before the rows are appended, one release of what was
// not used after. A per-row atomic on one process-wide counter from 20 scan
// threads is a contended cache line ~60M times per SF10 upload — measured as
// the upload starving a native query on another connection 3–7×. The
// per-buffer `charged` field is touched only by the owning thread.
bool pool_reserve(std::size_t n) {
    Pool& P = pool();
    if (P.bytes.load(std::memory_order_relaxed) + n > pool_cap_bytes()) return false;
    P.bytes.fetch_add(n, std::memory_order_relaxed);
    return true;
}
void pool_release(std::size_t n) {
    if (n) pool().bytes.fetch_sub(n, std::memory_order_relaxed);
}

} // namespace

// ---------------------------------------------------------------------------
// ResidentContext — one per database.
// ---------------------------------------------------------------------------

// An open upload session (v0.7 milestone 0c, docs/TRANSPARENT_DESIGN.md
// §5.5): the wrapper uploads a table as a sequence of short statements, one
// per row-id segment, each appending its buffered segments here; nothing
// touches the device until gpu_upload_finish. A statement that fails or is
// interrupted mid-scan never reaches finalize and leaves the session exactly
// as it was (append is one atomic step under the registry lock), so the
// wrapper re-runs that segment and keeps the earlier ones.
struct UploadSession {
    std::string   name;
    bool          managed = false;
    TagFields     tag;
    bool          kind_set = false;                 // fixed by the first segment
    bool          pair = false;
    gpudb::Dtype  vdt = gpudb::Dtype::I64;          // payload (pair) / column (bare) dtype
    std::vector<SegView> views;
    std::size_t   lanes = 0;
    std::size_t   rows_seen = 0;
    std::size_t   segments = 0;
    std::size_t   charged = 0;                      // pool bytes held by the views
    std::uint64_t seq_at_start = 0;
    std::int64_t  started_at_us = 0;
};

class ResidentContext {
public:
    // --- registry ---
    std::mutex registry_mu;
    std::unordered_map<std::string, std::shared_ptr<ResidentSet>> registry;
    std::unordered_map<std::string, std::shared_ptr<UploadSession>> sessions;   // open sessions, by name

    // Invalidation log: (prefix, seq). An upload whose name matches a record
    // with seq > its seq_at_start is discarded at finalize (§5.5 epoch
    // capture). Pruned to one record per prefix, bounded.
    struct Inval { std::string prefix; std::uint64_t seq; };
    std::vector<Inval>         invals;
    std::atomic<std::uint64_t> inval_seq { 0 };

    // --- device ---
    std::mutex device_mu;                              // one operator call at a time
    std::mutex agg_mu;                                 // lazy construction only
    std::unique_ptr<gpudb::HybridAggregator> agg;

    // --- stats ---
    std::mutex  stats_mu;
    std::string last_stats = "no resident reduction has run yet";

    gpudb::HybridAggregator& aggregator() {
        std::lock_guard<std::mutex> lock(agg_mu);
        if (!agg) agg = gpudb::make_hybrid_aggregator();
        return *agg;
    }

    static bool matches(const std::string& name, const std::string& prefix) {
        return name == prefix ||
               (starts_with(name, prefix) && name.size() > prefix.size() &&
                name[prefix.size()] == ':');
    }

    // Caller holds registry_mu.
    bool invalidated_since_locked(const std::string& name, std::uint64_t seq0) const {
        for (const auto& r : invals)
            if (r.seq > seq0 && matches(name, r.prefix)) return true;
        return false;
    }

    std::int64_t invalidate(const std::string& prefix) {
        std::lock_guard<std::mutex> lock(registry_mu);
        const std::uint64_t seq = inval_seq.fetch_add(1, std::memory_order_acq_rel) + 1;
        std::int64_t n = 0;
        for (auto& kv : registry) {
            if (!matches(kv.first, prefix)) continue;
            SetState expect = kv.second->state.load();
            if (expect != SetState::Stale) { kv.second->state.store(SetState::Stale); ++n; }
        }
        for (auto it = sessions.begin(); it != sessions.end();) {
            if (matches(it->first, prefix)) {
                pool_release(it->second->charged);
                it = sessions.erase(it);          // the segments go with the last reference
            } else {
                ++it;
            }
        }
        invals.erase(std::remove_if(invals.begin(), invals.end(),
                                    [&](const Inval& r) { return r.prefix == prefix; }),
                     invals.end());
        if (invals.size() >= 1024) invals.erase(invals.begin());   // oldest first
        invals.push_back(Inval{prefix, seq});
        return n;
    }

    // Lookup with hit accounting. Throws when unknown or stale.
    std::shared_ptr<ResidentSet> acquire(const std::string& name, const char* fn,
                                         bool count_hit = true) {
        std::shared_ptr<ResidentSet> s;
        {
            std::lock_guard<std::mutex> lock(registry_mu);
            auto it = registry.find(name);
            if (it != registry.end()) s = it->second;
        }
        if (!s) {
            throw std::runtime_error(std::string(fn) + ": no resident set named '" + name +
                "' — create it with gpu_upload('" + name + "', <col>) or gpu_upload_pair");
        }
        if (s->state.load(std::memory_order_acquire) == SetState::Stale) {
            throw std::runtime_error(std::string("GPUDB_STALE: ") + fn + ": resident set '" +
                name + "' is stale (invalidated at epoch " + std::to_string(s->epoch) +
                ") — re-upload it");
        }
        if (count_hit) {
            s->hits.fetch_add(1, std::memory_order_relaxed);
            s->last_used_at_us.store(now_us(), std::memory_order_relaxed);
        }
        return s;
    }

    // Column resolution: exact set (bare column) first, then '<set>.k|v'.
    ResidentRef acquire_column(const std::string& name, const char* fn) {
        std::shared_ptr<ResidentSet> s;
        {
            std::lock_guard<std::mutex> lock(registry_mu);
            auto it = registry.find(name);
            if (it != registry.end() && !it->second->pair) s = it->second;
        }
        if (s) {
            ResidentRef r{acquire(name, fn), nullptr};
            r.col = r.set->keys.get();
            return r;
        }
        const std::size_t dot = name.rfind('.');
        if (dot != std::string::npos && dot + 2 == name.size() &&
            (name[dot + 1] == 'k' || name[dot + 1] == 'v')) {
            const std::string base = name.substr(0, dot);
            bool have = false;
            {
                std::lock_guard<std::mutex> lock(registry_mu);
                auto it = registry.find(base);
                have = it != registry.end() && it->second->pair;
            }
            if (have) {
                ResidentRef r{acquire(base, fn), nullptr};
                r.col = name[dot + 1] == 'k' ? r.set->keys.get() : r.set->vals.get();
                return r;
            }
        }
        // A pair name used where a column is expected: hand out the keys.
        {
            std::lock_guard<std::mutex> lock(registry_mu);
            auto it = registry.find(name);
            if (it != registry.end()) s = it->second;
        }
        if (s) {
            ResidentRef r{acquire(name, fn), nullptr};
            r.col = r.set->keys.get();
            return r;
        }
        throw std::runtime_error(std::string(fn) + ": no resident column named '" + name +
            "' — create one with gpu_upload('" + name + "', <col>)");
    }

    // Replace-or-insert. Returns false (and touches nothing) when the upload
    // was invalidated after it started.
    bool publish(std::shared_ptr<ResidentSet> set, std::uint64_t seq_at_start) {
        std::lock_guard<std::mutex> lock(registry_mu);
        if (invalidated_since_locked(set->name, seq_at_start)) return false;
        set->epoch = inval_seq.load(std::memory_order_acquire);
        registry[set->name] = std::move(set);   // old set dies with its last reference
        return true;
    }

    void record_stats(ResidentSet* set, const std::string& line) {
        {
            std::lock_guard<std::mutex> lock(stats_mu);
            last_stats = line;
        }
        if (set) {
            std::lock_guard<std::mutex> lock(set->stats_mu);
            set->last_stats = line;
        }
    }
};

std::shared_ptr<ResidentContext> make_resident_context() {
    return std::make_shared<ResidentContext>();
}

void* resident_extra_info(const std::shared_ptr<ResidentContext>& ctx) {
    return new std::shared_ptr<ResidentContext>(ctx);
}
void resident_extra_info_destroy(void* holder) {
    delete static_cast<std::shared_ptr<ResidentContext>*>(holder);
}
ResidentContext& resident_context(void* extra_info) {
    if (!extra_info) throw std::runtime_error("gpudb: function registered without a resident context");
    return **static_cast<std::shared_ptr<ResidentContext>*>(extra_info);
}

std::shared_ptr<ResidentSet> resident_acquire_set(ResidentContext& ctx,
                                                  const std::string& name, const char* fn) {
    return ctx.acquire(name, fn);
}
ResidentRef resident_acquire_column(ResidentContext& ctx,
                                    const std::string& name, const char* fn) {
    return ctx.acquire_column(name, fn);
}
std::unique_lock<std::mutex> resident_device_lock(ResidentContext& ctx) {
    return std::unique_lock<std::mutex>(ctx.device_mu);
}
gpudb::HybridAggregator& resident_aggregator(ResidentContext& ctx) {
    return ctx.aggregator();
}
void resident_record_stats(ResidentContext& ctx, ResidentSet* set, const std::string& line) {
    ctx.record_stats(set, line);
}

namespace {

ResidentContext& ctx_of(duckdb_function_info info) {
    return resident_context(duckdb_scalar_function_get_extra_info(info));
}
ResidentContext& ctx_of_aggregate(duckdb_function_info info) {
    return resident_context(duckdb_aggregate_function_get_extra_info(info));
}

void record_agg_stats(ResidentContext& ctx, ResidentSet* set, const char* op,
                      const gpudb::AggResult& r) {
    const auto& d = ctx.aggregator().last_decision();
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "op=%s backend=%s reason=%s rows=%zu wall_ms=%.3f kernel_ms=%.3f transfer_ms=%.3f",
        op, gpudb::to_string(d.chosen), gpudb::to_string(d.reason),
        r.rows, r.wall_ms, r.kernel_ms, r.transfer_ms);
    ctx.record_stats(set, buf);
}

// ---------------------------------------------------------------------------
// gpu_upload aggregate.
// ---------------------------------------------------------------------------

struct UploadState {
    static constexpr std::uint64_t kMagic = 0xB0FFE7C0FFEE0002ULL;
    std::uint64_t magic;
    std::uint64_t buf_id;   // 0 = no buffer yet; keys the pool otherwise
    UploadBuf*    buf;      // cached pool entry (owned by the pool, never freed here)
};

idx_t upload_state_size(duckdb_function_info /*info*/) { return sizeof(UploadState); }

void upload_state_init(duckdb_function_info /*info*/, duckdb_aggregate_state state) {
    auto* s = reinterpret_cast<UploadState*>(state);
    s->magic  = UploadState::kMagic;
    s->buf_id = 0;
    s->buf    = nullptr;
}

void upload_state_destroy(duckdb_aggregate_state* states, idx_t count) {
    Pool& P = pool();
    std::lock_guard<std::mutex> lock(P.mu);
    for (idx_t i = 0; i < count; ++i) {
        auto* s = reinterpret_cast<UploadState*>(states[i]);
        if (!s || s->magic != UploadState::kMagic) continue;
        if (s->buf_id != 0) {
            auto it = P.bufs.find(s->buf_id);   // double-erase is a no-op
            if (it != P.bufs.end()) {
                P.bytes.fetch_sub(it->second->charged, std::memory_order_relaxed);
                P.bufs.erase(it);
            }
        }
        s->magic = 0;
        s->buf   = nullptr;
    }
}

inline UploadState* probe_upload_state(void* p) {
    if (!p) return nullptr;
    auto* s = reinterpret_cast<UploadState*>(p);
    if (s->magic != UploadState::kMagic) return nullptr;
    return s;
}

// Get (or create) the pool buffer for a state. Takes the pool lock only on
// creation — once per state, never per chunk.
UploadBuf& state_buf(UploadState* s, gpudb::Dtype dt) {
    if (s->buf) return *s->buf;
    Pool& P = pool();
    auto b = std::make_unique<UploadBuf>();
    b->dtype = dt;
    UploadBuf* raw = b.get();
    const std::uint64_t id = P.next_id.fetch_add(1, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(P.mu);
        P.bufs[id] = std::move(b);
    }
    s->buf_id = id;
    s->buf    = raw;
    return *raw;
}

// Fix the buffer's name on its first contributing row. Parses identity tags
// and captures the invalidation sequence. Returns false after setting an
// error.
bool set_buf_name(duckdb_function_info info, UploadBuf& b, const char* nm, std::size_t len,
                  const char* fn) {
    b.name.assign(nm, len);
    b.name_set = true;
    if (starts_with(b.name, kTagPrefix)) {
        const std::string err = parse_tag(b.name, b.tag);
        if (!err.empty()) {
            duckdb_aggregate_function_set_error(info, (std::string(fn) + ": " + err).c_str());
            return false;
        }
        b.managed = true;
    }
    b.seq_at_start = ctx_of_aggregate(info).inval_seq.load(std::memory_order_acquire);
    return true;
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

    // NO lock here (see the header comment). Each state's buffer is only
    // ever appended by the thread DuckDB hands that state to.
    constexpr std::size_t kElem = sizeof(T);
    const std::size_t reserved = static_cast<std::size_t>(n) * kElem;
    if (!pool_reserve(reserved)) {
        duckdb_aggregate_function_set_error(info,
            (std::string("gpu_upload: out of buffer memory") + kPoolCapHint).c_str());
        return;
    }
    std::size_t used = 0;
    auto append_row = [&](UploadState* s, idx_t i) -> bool {
        UploadBuf& b = state_buf(s, DT);
        ++b.rows_seen;
        // NULL values are skipped (standard aggregate semantics).
        if (val_validity && !duckdb_validity_row_is_valid(val_validity, i)) return true;
        // A NULL name on a contributing row is a user error, not data: the
        // old code silently registered the column under '' instead.
        if (name_validity && !duckdb_validity_row_is_valid(name_validity, i)) {
            duckdb_aggregate_function_set_error(info, "gpu_upload: name may not be NULL");
            return false;
        }
        const char*       nm_data = duckdb_string_t_data(&names[i]);
        const std::size_t nm_len  = duckdb_string_t_length(names[i]);
        if (!b.name_set) {
            if (!set_buf_name(info, b, nm_data, nm_len, "gpu_upload")) return false;
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
        std::int64_t* dst = b.reserve_lanes(1);
        if (DT == gpudb::Dtype::I64) {
            *dst = static_cast<std::int64_t>(data[i]);
        } else {
            const double d = static_cast<double>(data[i]);
            std::memcpy(dst, &d, sizeof(d));     // raw bits in the i64 lane
        }
        b.charged += kElem;
        used += kElem;
        return true;
    };

    bool ok = true;
    if (n == 1 || !per_row) {
        for (idx_t i = 0; i < n && ok; ++i) ok = append_row(s0, i);
    } else {
        for (idx_t i = 0; i < n && ok; ++i) {
            UploadState* s = probe_upload_state(states[i]);
            if (s) ok = append_row(s, i);
        }
    }
    pool_release(reserved - used);
}

void upload_combine(duckdb_function_info info, duckdb_aggregate_state* source,
                    duckdb_aggregate_state* target, idx_t count) {
    // Contract #3: source is read-only (window donor states are combined into
    // many targets). We share the source's segments by reference — no copy,
    // no host peak of 2× the data — with the lane counts frozen now.
    for (idx_t i = 0; i < count; ++i) {
        UploadState* src = probe_upload_state(source[i]);
        UploadState* dst = probe_upload_state(target[i]);
        if (!src || !dst || src == dst || src->buf_id == 0 || !src->buf) continue;
        const UploadBuf& sb = *src->buf;
        UploadBuf& db = state_buf(dst, sb.dtype);
        if (!db.name_set && sb.name_set) {
            db.name = sb.name; db.name_set = true;
            db.managed = sb.managed; db.tag = sb.tag;
            db.seq_at_start = sb.seq_at_start;
        } else if (db.name_set && sb.name_set) {
            if (db.name != sb.name) {
                duckdb_aggregate_function_set_error(info,
                    ("gpu_upload: combine merged two different names ('" + db.name +
                     "' and '" + sb.name + "')").c_str());
                return;
            }
            db.seq_at_start = std::min(db.seq_at_start, sb.seq_at_start);
        }
        const std::size_t delta = sb.lanes() * sizeof(std::int64_t);
        // The cap counts LOGICAL bytes (what a copy would have cost), so the
        // window-frame quadratic-buffering guardrail keeps its meaning: each
        // output row's state referencing its whole frame prefix still adds
        // up and still hits the wall.
        if (delta > 0 && !pool_reserve(delta)) {
            duckdb_aggregate_function_set_error(info,
                (std::string("gpu_upload: out of buffer memory") + kPoolCapHint).c_str());
            return;
        }
        db.charged += delta;
        db.rows_seen += sb.rows_seen;
        for (const auto& v : sb.all_views()) db.views.push_back(v);
    }
}

// Build a set from a finished buffer and publish it. Everything expensive
// (H2D, prepare) runs outside every lock; publish takes the registry lock
// for a map insert. Throws std::runtime_error.
void publish_set(ResidentContext& ctx, UploadBuf& b,
                 std::unique_ptr<gpudb::ResidentColumn> keys,
                 std::unique_ptr<gpudb::ResidentColumn> vals, bool pair, const char* fn) {
    auto set = std::make_shared<ResidentSet>();
    set->name = b.name;
    set->managed = b.managed;
    if (b.managed) {
        set->catalog = b.tag.catalog; set->schema = b.tag.schema; set->table = b.tag.table;
        set->table_oid = b.tag.table_oid; set->columns = b.tag.columns; set->extra = b.tag.extra;
    }
    set->pair = pair;
    set->rows = keys ? keys->rows() : 0;
    set->rows_seen = b.rows_seen;
    set->keys = std::move(keys);
    set->vals = std::move(vals);
    set->uploaded_at_us = now_us();
    // Ready means uploaded AND prepared (§5.5). Managed sets are prepared
    // here so the first hit pays no sort; explicit sets keep the lazy build
    // (gpu_prepare_resident() exists for users who want it eager).
    const auto t_prep = std::chrono::steady_clock::now();
    if (set->managed && set->keys) {
        set->keys->prepare();
    }
    if (upload_trace())
        std::fprintf(stderr, "[gpudb upload] %s '%s': rows=%zu prepare=%.1f ms\n",
                     fn, b.name.c_str(), set->rows, ms_since(t_prep));
    set->state.store(set->keys && set->keys->prepared() ? SetState::Ready : SetState::Uploaded);
    if (!ctx.publish(set, b.seq_at_start)) {
        throw std::runtime_error(std::string("GPUDB_UPLOAD_DISCARDED: ") + fn + ": upload of '" +
            b.name + "' discarded — the set was invalidated while the upload ran; "
            "the buffers were freed, run the upload again");
    }
}

// Extract the row's VARCHAR cell as a std::string.
inline std::string read_name(duckdb_string_t* names, idx_t row) {
    return std::string(duckdb_string_t_data(&names[row]),
                       duckdb_string_t_length(names[row]));
}

// Move a finished buffer onto the device and publish it: bare columns go
// through the contiguous v1 upload (one host concatenation of the segments),
// pairs through upload_pair_interleaved (one H2D per segment, split on the
// device). Returns the rows resident. Throws std::runtime_error.
std::size_t finish_upload(ResidentContext& ctx, UploadBuf& b, bool pair, gpudb::Dtype vdt,
                          const char* fn) {
    auto& a = ctx.aggregator();
    const std::size_t lanes = b.lanes();
    const auto t0 = std::chrono::steady_clock::now();
    if (!pair) {
        std::vector<std::int64_t> flat(lanes);
        std::size_t off = 0;
        for (const auto& v : b.all_views()) {
            std::memcpy(flat.data() + off, v.seg->data, v.lanes * sizeof(std::int64_t));
            off += v.lanes;
        }
        std::unique_ptr<gpudb::ResidentColumn> col;
        if (vdt == gpudb::Dtype::I64) col = a.upload_i64(flat.data(), lanes);
        else                          col = a.upload_f64(reinterpret_cast<const double*>(flat.data()), lanes);
        const std::size_t rows = col->rows();
        publish_set(ctx, b, std::move(col), nullptr, /*pair*/false, fn);
        return rows;
    }
    const std::size_t rows = lanes / 2;
    std::vector<gpudb::Aggregator::KvSpan> spans;
    const auto views = b.all_views();
    spans.reserve(views.size());
    for (const auto& v : views) spans.push_back({v.seg->data, v.lanes / 2});
    gpudb::Aggregator::ResidentPair cols = a.upload_pair_interleaved(spans.data(), spans.size(), vdt);
    if (upload_trace())
        std::fprintf(stderr, "[gpudb upload] %s '%s': rows=%zu segments=%zu buffered=%zu MB "
                     "upload(pair)=%.1f ms\n", fn, b.name.c_str(), rows, spans.size(),
                     (lanes * 8) >> 20, ms_since(t0));
    publish_set(ctx, b, std::move(cols.keys), std::move(cols.vals), /*pair*/true, fn);
    return rows;
}

// If an upload session is open under the buffer's name, append the buffer's
// segments to it (one atomic step) and return true; the statement then
// returns its own row count and the device is not touched. Throws on a
// kind mismatch or when the pool cap would be exceeded.
bool session_append(ResidentContext& ctx, UploadBuf& b, bool pair, gpudb::Dtype vdt,
                    const char* fn) {
    std::lock_guard<std::mutex> lock(ctx.registry_mu);
    auto it = ctx.sessions.find(b.name);
    if (it == ctx.sessions.end()) return false;
    UploadSession& ss = *it->second;
    if (ss.kind_set && (ss.pair != pair || ss.vdt != vdt))
        throw std::runtime_error(std::string(fn) + ": segment kind differs from the open upload "
            "session '" + b.name + "' (all segments must use the same upload function and types)");
    const std::size_t bytes = b.lanes() * sizeof(std::int64_t);
    if (bytes > 0 && !pool_reserve(bytes))
        throw std::runtime_error(std::string(fn) + ": out of buffer memory for the upload "
            "session '" + b.name + "'" + kPoolCapHint);
    ss.kind_set = true; ss.pair = pair; ss.vdt = vdt;
    ss.charged += bytes;
    ss.lanes += b.lanes();
    ss.rows_seen += b.rows_seen;
    ss.segments += 1;
    for (const auto& v : b.all_views()) ss.views.push_back(v);
    return true;
}

void upload_finalize(duckdb_function_info info, duckdb_aggregate_state* source,
                     duckdb_vector result, idx_t count, idx_t offset) {
    if (count == 0) return;
    auto* out = reinterpret_cast<std::int64_t*>(duckdb_vector_get_data(result));
    duckdb_vector_ensure_validity_writable(result);
    uint64_t* validity = duckdb_vector_get_validity(result);

    for (idx_t i = 0; i < count; ++i) {
        UploadState* s = probe_upload_state(source[i]);
        UploadBuf* b = (s && s->buf_id != 0) ? s->buf : nullptr;
        const std::size_t lanes = b ? b->lanes() : 0;
        if (!b || lanes == 0) {
            // No values buffered (empty input / all NULL) -> SQL NULL, no column.
            out[offset + i] = 0;
            duckdb_validity_set_row_invalid(validity, offset + i);
            continue;
        }
        try {
            ResidentContext& ctx = ctx_of_aggregate(info);
            if (session_append(ctx, *b, /*pair*/false, b->dtype, "gpu_upload")) {
                out[offset + i] = static_cast<std::int64_t>(lanes);
                continue;
            }
            out[offset + i] = static_cast<std::int64_t>(
                finish_upload(ctx, *b, /*pair*/false, b->dtype, "gpu_upload"));
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
                                         duckdb_aggregate_update_t update,
                                         const std::shared_ptr<ResidentContext>& ctx) {
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
    duckdb_aggregate_function_set_extra_info(fn, resident_extra_info(ctx),
                                             resident_extra_info_destroy);
    return fn;
}

// ---------------------------------------------------------------------------
// gpu_upload_pair aggregate — upload a (key, payload) column pair in ONE scan.
//
// Two separate gpu_upload calls CANNOT be used as a join's probe keys +
// payload: each upload is an independent aggregate over an independent table
// scan, and DuckDB's parallel scan/combine order differs between them, so
// row i of one column need not correspond to row i of the other. (This is
// invisible to single-column reductions, which are order-insensitive — it
// only bites operations that pair two columns positionally, like the fused
// join.) gpu_upload_pair(name, k, v) buffers the pair interleaved in one
// aggregate state, so whatever order DuckDB delivers, k[i] and v[i] stay
// together. Finalize registers ONE set with two columns, addressed as
// '<name>.k' and '<name>.v'. Rows where either value is NULL are skipped as
// a pair.
// ---------------------------------------------------------------------------

// Payload type V is stored bit-cast into the interleaved i64 buffer (identity
// for BIGINT, raw IEEE-754 bits for DOUBLE); VDT tags which finalize applies.
template <class V, gpudb::Dtype VDT>
void upload_pair_update_t(duckdb_function_info info, duckdb_data_chunk input,
                          duckdb_aggregate_state* states) {
    duckdb_vector name_vec = duckdb_data_chunk_get_vector(input, 0);
    duckdb_vector k_vec    = duckdb_data_chunk_get_vector(input, 1);
    duckdb_vector v_vec    = duckdb_data_chunk_get_vector(input, 2);
    auto* names = reinterpret_cast<duckdb_string_t*>(duckdb_vector_get_data(name_vec));
    const auto* kd = reinterpret_cast<const std::int64_t*>(duckdb_vector_get_data(k_vec));
    const auto* vd = reinterpret_cast<const V*>(duckdb_vector_get_data(v_vec));
    const idx_t n = duckdb_data_chunk_get_size(input);
    if (!names || !kd || !vd || n == 0) return;

    uint64_t* name_validity = duckdb_vector_get_validity(name_vec);
    uint64_t* k_validity    = duckdb_vector_get_validity(k_vec);
    uint64_t* v_validity    = duckdb_vector_get_validity(v_vec);

    UploadState* s0 = probe_upload_state(states[0]);
    if (!s0) return;
    bool per_row = true;
    if (n > 1) {
        const idx_t probes[3] = { 1, n / 2, n - 1 };
        for (idx_t k = 0; k < 3 && per_row; ++k) {
            const idx_t i = probes[k];
            if (i == 0) continue;
            if (probe_upload_state(states[i]) == nullptr) per_row = false;
        }
    }

    constexpr std::size_t kPair = 2 * sizeof(std::int64_t);
    const std::size_t reserved = static_cast<std::size_t>(n) * kPair;
    if (!pool_reserve(reserved)) {
        duckdb_aggregate_function_set_error(info,
            (std::string("gpu_upload_pair: out of buffer memory") + kPoolCapHint).c_str());
        return;
    }
    std::size_t used = 0;
    auto append_row = [&](UploadState* s, idx_t i) -> bool {
        UploadBuf& b = state_buf(s, gpudb::Dtype::I64);
        ++b.rows_seen;
        // Either half NULL -> skip the PAIR (keeps k/v aligned).
        if ((k_validity && !duckdb_validity_row_is_valid(k_validity, i)) ||
            (v_validity && !duckdb_validity_row_is_valid(v_validity, i))) return true;
        if (name_validity && !duckdb_validity_row_is_valid(name_validity, i)) {
            duckdb_aggregate_function_set_error(info, "gpu_upload_pair: name may not be NULL");
            return false;
        }
        const char*       nm_data = duckdb_string_t_data(&names[i]);
        const std::size_t nm_len  = duckdb_string_t_length(names[i]);
        if (!b.name_set) {
            if (!set_buf_name(info, b, nm_data, nm_len, "gpu_upload_pair")) return false;
        } else if (b.name.size() != nm_len ||
                   std::memcmp(b.name.data(), nm_data, nm_len) != 0) {
            duckdb_aggregate_function_set_error(info,
                ("gpu_upload_pair: one aggregate received two different names ('" +
                 b.name + "' and '" + std::string(nm_data, nm_len) +
                 "') — use a constant name").c_str());
            return false;
        }
        static_assert(sizeof(V) == sizeof(std::int64_t), "pair payload width");
        std::int64_t* dst = b.reserve_lanes(2);   // interleaved (k, v) in ONE segment
        dst[0] = kd[i];
        std::memcpy(&dst[1], &vd[i], sizeof(std::int64_t));   // raw payload bits
        b.charged += kPair;
        used += kPair;
        return true;
    };

    bool ok = true;
    if (n == 1 || !per_row) {
        for (idx_t i = 0; i < n && ok; ++i) ok = append_row(s0, i);
    } else {
        for (idx_t i = 0; i < n && ok; ++i) {
            UploadState* s = probe_upload_state(states[i]);
            if (s) ok = append_row(s, i);
        }
    }
    pool_release(reserved - used);
}

template <gpudb::Dtype VDT>
void upload_pair_finalize_t(duckdb_function_info info, duckdb_aggregate_state* source,
                            duckdb_vector result, idx_t count, idx_t offset) {
    if (count == 0) return;
    auto* out = reinterpret_cast<std::int64_t*>(duckdb_vector_get_data(result));
    duckdb_vector_ensure_validity_writable(result);
    uint64_t* validity = duckdb_vector_get_validity(result);

    for (idx_t i = 0; i < count; ++i) {
        UploadState* s = probe_upload_state(source[i]);
        UploadBuf* b = (s && s->buf_id != 0) ? s->buf : nullptr;
        const std::size_t lanes = b ? b->lanes() : 0;
        if (!b || lanes == 0) {
            out[offset + i] = 0;
            duckdb_validity_set_row_invalid(validity, offset + i);
            continue;
        }
        try {
            ResidentContext& ctx = ctx_of_aggregate(info);
            if (session_append(ctx, *b, /*pair*/true, VDT, "gpu_upload_pair")) {
                out[offset + i] = static_cast<std::int64_t>(lanes / 2);
                continue;
            }
            out[offset + i] = static_cast<std::int64_t>(
                finish_upload(ctx, *b, /*pair*/true, VDT, "gpu_upload_pair"));
        } catch (const std::exception& e) {
            duckdb_aggregate_function_set_error(info,
                (std::string("gpu_upload_pair failed: ") + e.what()).c_str());
            return;
        }
    }
}

// ---------------------------------------------------------------------------
// Upload sessions: gpu_upload_begin / gpu_upload_finish / gpu_upload_abort /
// gpu_upload_status.
// ---------------------------------------------------------------------------

// gpu_upload_begin(name) -> BOOLEAN: open (or replace) a session; segment
// statements under this name append to it instead of uploading.
void upload_begin_exec(duckdb_function_info info, duckdb_data_chunk input, duckdb_vector output) {
    duckdb_vector name_vec = duckdb_data_chunk_get_vector(input, 0);
    auto* names = reinterpret_cast<duckdb_string_t*>(duckdb_vector_get_data(name_vec));
    uint64_t* validity = duckdb_vector_get_validity(name_vec);
    const idx_t n = duckdb_data_chunk_get_size(input);
    auto* out = reinterpret_cast<bool*>(duckdb_vector_get_data(output));
    duckdb_vector_ensure_validity_writable(output);
    uint64_t* out_validity = duckdb_vector_get_validity(output);
    ResidentContext& ctx = ctx_of(info);
    for (idx_t i = 0; i < n; ++i) {
        if (validity && !duckdb_validity_row_is_valid(validity, i)) {
            duckdb_validity_set_row_invalid(out_validity, i);
            continue;
        }
        auto ss = std::make_shared<UploadSession>();
        ss->name = read_name(names, i);
        if (starts_with(ss->name, kTagPrefix)) {
            const std::string err = parse_tag(ss->name, ss->tag);
            if (!err.empty()) {
                duckdb_scalar_function_set_error(info, ("gpu_upload_begin: " + err).c_str());
                return;
            }
            ss->managed = true;
        }
        ss->started_at_us = now_us();
        {
            std::lock_guard<std::mutex> lock(ctx.registry_mu);
            ss->seq_at_start = ctx.inval_seq.load(std::memory_order_acquire);
            auto old = ctx.sessions.find(ss->name);
            if (old != ctx.sessions.end()) pool_release(old->second->charged);
            ctx.sessions[ss->name] = ss;
        }
        out[i] = true;
    }
}

// gpu_upload_finish(name) -> BIGINT rows: take the session, move it to the
// device (+ prepare for managed sets), publish it, release its buffers.
void upload_finish_exec(duckdb_function_info info, duckdb_data_chunk input, duckdb_vector output) {
    duckdb_vector name_vec = duckdb_data_chunk_get_vector(input, 0);
    auto* names = reinterpret_cast<duckdb_string_t*>(duckdb_vector_get_data(name_vec));
    uint64_t* validity = duckdb_vector_get_validity(name_vec);
    const idx_t n = duckdb_data_chunk_get_size(input);
    auto* out = reinterpret_cast<std::int64_t*>(duckdb_vector_get_data(output));
    duckdb_vector_ensure_validity_writable(output);
    uint64_t* out_validity = duckdb_vector_get_validity(output);
    ResidentContext& ctx = ctx_of(info);
    for (idx_t i = 0; i < n; ++i) {
        if (validity && !duckdb_validity_row_is_valid(validity, i)) {
            duckdb_validity_set_row_invalid(out_validity, i);
            continue;
        }
        const std::string name = read_name(names, i);
        std::shared_ptr<UploadSession> ss;
        {
            std::lock_guard<std::mutex> lock(ctx.registry_mu);
            auto it = ctx.sessions.find(name);
            if (it != ctx.sessions.end()) { ss = it->second; ctx.sessions.erase(it); }
        }
        if (!ss) {
            duckdb_scalar_function_set_error(info,
                ("GPUDB_UPLOAD_DISCARDED: gpu_upload_finish: no open upload session for '" + name +
                 "' — it was never begun, already finished, or invalidated while open").c_str());
            return;
        }
        try {
            if (ss->lanes == 0)
                throw std::runtime_error("GPUDB_UPLOAD_EMPTY: gpu_upload_finish: the upload session '" +
                                         name + "' received no rows");
            UploadBuf b;
            b.name = ss->name; b.name_set = true; b.managed = ss->managed; b.tag = ss->tag;
            b.dtype = ss->vdt; b.views = ss->views; b.rows_seen = ss->rows_seen;
            b.seq_at_start = ss->seq_at_start;
            const std::size_t rows = finish_upload(ctx, b, ss->pair, ss->vdt, "gpu_upload_finish");
            pool_release(ss->charged);
            ss->charged = 0;
            out[i] = static_cast<std::int64_t>(rows);
        } catch (const std::exception& e) {
            pool_release(ss->charged);
            ss->charged = 0;
            duckdb_scalar_function_set_error(info, e.what());
            return;
        }
    }
}

// gpu_upload_abort(name) -> BOOLEAN: drop an open session and its buffers.
void upload_abort_exec(duckdb_function_info info, duckdb_data_chunk input, duckdb_vector output) {
    duckdb_vector name_vec = duckdb_data_chunk_get_vector(input, 0);
    auto* names = reinterpret_cast<duckdb_string_t*>(duckdb_vector_get_data(name_vec));
    uint64_t* validity = duckdb_vector_get_validity(name_vec);
    const idx_t n = duckdb_data_chunk_get_size(input);
    auto* out = reinterpret_cast<bool*>(duckdb_vector_get_data(output));
    duckdb_vector_ensure_validity_writable(output);
    uint64_t* out_validity = duckdb_vector_get_validity(output);
    ResidentContext& ctx = ctx_of(info);
    for (idx_t i = 0; i < n; ++i) {
        if (validity && !duckdb_validity_row_is_valid(validity, i)) {
            duckdb_validity_set_row_invalid(out_validity, i);
            continue;
        }
        std::shared_ptr<UploadSession> ss;
        {
            std::lock_guard<std::mutex> lock(ctx.registry_mu);
            auto it = ctx.sessions.find(read_name(names, i));
            if (it != ctx.sessions.end()) { ss = it->second; ctx.sessions.erase(it); }
        }
        if (ss) pool_release(ss->charged);
        out[i] = ss != nullptr;
    }
}

// gpu_upload_status(name) -> VARCHAR (JSON): what the wrapper needs after an
// interrupted segment — whether the session is open, how many segments and
// rows it holds, and whether an invalidation since begin will discard it.
void upload_status_exec(duckdb_function_info info, duckdb_data_chunk input, duckdb_vector output) {
    duckdb_vector name_vec = duckdb_data_chunk_get_vector(input, 0);
    auto* names = reinterpret_cast<duckdb_string_t*>(duckdb_vector_get_data(name_vec));
    uint64_t* validity = duckdb_vector_get_validity(name_vec);
    const idx_t n = duckdb_data_chunk_get_size(input);
    duckdb_vector_ensure_validity_writable(output);
    uint64_t* out_validity = duckdb_vector_get_validity(output);
    ResidentContext& ctx = ctx_of(info);
    for (idx_t i = 0; i < n; ++i) {
        if (validity && !duckdb_validity_row_is_valid(validity, i)) {
            duckdb_validity_set_row_invalid(out_validity, i);
            continue;
        }
        const std::string name = read_name(names, i);
        char buf[512];
        {
            std::lock_guard<std::mutex> lock(ctx.registry_mu);
            auto it = ctx.sessions.find(name);
            if (it == ctx.sessions.end()) {
                std::snprintf(buf, sizeof(buf), "{\"open\":false}");
            } else {
                const UploadSession& ss = *it->second;
                std::snprintf(buf, sizeof(buf),
                    "{\"open\":true,\"segments\":%zu,\"rows\":%zu,\"rows_seen\":%zu,"
                    "\"bytes\":%zu,\"kind\":\"%s\",\"epoch\":%llu,\"invalidated\":%s,"
                    "\"started_at_us\":%lld}",
                    ss.segments, ss.pair ? ss.lanes / 2 : ss.lanes, ss.rows_seen, ss.charged,
                    !ss.kind_set ? "unset" : ss.pair ? "pair" : "column",
                    static_cast<unsigned long long>(ss.seq_at_start),
                    ctx.invalidated_since_locked(ss.name, ss.seq_at_start) ? "true" : "false",
                    static_cast<long long>(ss.started_at_us));
            }
        }
        duckdb_vector_assign_string_element(output, i, buf);
    }
}

// ---------------------------------------------------------------------------
// Resident scalar functions.
// ---------------------------------------------------------------------------

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

    ResidentContext& ctx = ctx_of(info);
    for (idx_t i = 0; i < n; ++i) {
        if (validity && !duckdb_validity_row_is_valid(validity, i)) {
            duckdb_validity_set_row_invalid(out_validity, i);
            continue;
        }
        try {
            ResidentRef ref = ctx.acquire_column(read_name(names, i), "gpu_*_resident");
            if (ref.col->dtype() != gpudb::Dtype::I64)
                throw std::runtime_error(
                    "resident column is DOUBLE — use gpu_sum_resident_f64 "
                    "(resident f64 min/max are not in the v1 backend ABI)");
            auto& a = ctx.aggregator();
            auto dev = resident_device_lock(ctx);
            gpudb::AggResult r = (a.*OP)(*ref.col);
            record_agg_stats(ctx, ref.set.get(), "resident_i64", r);
            out[i] = r.value_i64;
        } catch (const std::exception& e) {
            duckdb_scalar_function_set_error(info, e.what());
            return;
        }
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

    ResidentContext& ctx = ctx_of(info);
    for (idx_t i = 0; i < n; ++i) {
        if (validity && !duckdb_validity_row_is_valid(validity, i)) {
            duckdb_validity_set_row_invalid(out_validity, i);
            continue;
        }
        try {
            ResidentRef ref = ctx.acquire_column(read_name(names, i), "gpu_sum_resident_f64");
            if (ref.col->dtype() != gpudb::Dtype::F64)
                throw std::runtime_error("resident column is BIGINT — use gpu_sum_resident");
            auto& a = ctx.aggregator();
            auto dev = resident_device_lock(ctx);
            gpudb::AggResult r = a.sum_resident_f64(*ref.col);
            record_agg_stats(ctx, ref.set.get(), "sum_resident_f64", r);
            out[i] = r.value_f64;
        } catch (const std::exception& e) {
            duckdb_scalar_function_set_error(info, e.what());
            return;
        }
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

    ResidentContext& ctx = ctx_of(info);
    for (idx_t i = 0; i < n; ++i) {
        if (validity && !duckdb_validity_row_is_valid(validity, i)) {
            duckdb_validity_set_row_invalid(out_validity, i);
            continue;
        }
        try {
            ResidentRef ref = ctx.acquire_column(read_name(names, i), "gpu_resident_info");
            char buf[512];
            std::snprintf(buf, sizeof(buf),
                "dtype=%s rows=%zu device=%s set=%s origin=%s state=%s prepared=%s",
                ref.col->dtype() == gpudb::Dtype::I64 ? "I64" : "F64",
                ref.col->rows(), ctx.aggregator().device_name().c_str(),
                ref.set->name.c_str(), ref.set->managed ? "managed" : "explicit",
                to_string(ref.set->state.load()), ref.col->prepared() ? "true" : "false");
            duckdb_vector_assign_string_element(output, i, buf);
        } catch (const std::exception& e) {
            duckdb_scalar_function_set_error(info, e.what());
            return;
        }
    }
}

void last_stats_exec(duckdb_function_info info, duckdb_data_chunk input,
                     duckdb_vector output) {
    const idx_t n = duckdb_data_chunk_get_size(input);
    ResidentContext& ctx = ctx_of(info);
    std::string line;
    {
        std::lock_guard<std::mutex> lock(ctx.stats_mu);
        line = ctx.last_stats;
    }
    for (idx_t i = 0; i < n; ++i) {
        duckdb_vector_assign_string_element(output, i, line.c_str());
    }
}

// gpu_build_info(): which backends are COMPILED into this binary and which
// one dispatch selected at load. Build environments differ (toolchains,
// platforms, CI containers), so this is how CI and users verify what a
// given binary actually carries:
//   compiled=cpu           → built without a GPU toolchain
//   compiled=cpu,cuda      → CUDA backend present (nvcc at build time)
//   compiled=cpu,metal     → Metal backend present (macOS build)
void build_info_exec(duckdb_function_info /*info*/, duckdb_data_chunk input,
                     duckdb_vector output) {
    std::string info = "compiled=cpu";
#if defined(GPUDB_HAVE_CUDA)
    info += ",cuda";
#endif
#if defined(GPUDB_HAVE_METAL)
    info += ",metal";
#endif
    info += " runtime=";
    switch (gpudb::default_backend()) {
        case gpudb::Backend::CUDA:  info += "cuda";  break;
        case gpudb::Backend::METAL: info += "metal"; break;
        default:                    info += "cpu";   break;
    }
    const idx_t n = duckdb_data_chunk_get_size(input);
    for (idx_t i = 0; i < n; ++i) {
        duckdb_vector_assign_string_element(output, i, info.c_str());
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

    ResidentContext& ctx = ctx_of(info);
    for (idx_t i = 0; i < n; ++i) {
        if (validity && !duckdb_validity_row_is_valid(validity, i)) {
            duckdb_validity_set_row_invalid(out_validity, i);
            continue;
        }
        std::string name = read_name(names, i);
        std::shared_ptr<ResidentSet> victim;   // destroyed OUTSIDE the registry lock
        {
            std::lock_guard<std::mutex> lock(ctx.registry_mu);
            auto it = ctx.registry.find(name);
            if (it == ctx.registry.end()) {
                // 'p.k' / 'p.v' name a pair's column: dropping either drops the set.
                const std::size_t dot = name.rfind('.');
                if (dot != std::string::npos && dot + 2 == name.size() &&
                    (name[dot + 1] == 'k' || name[dot + 1] == 'v')) {
                    it = ctx.registry.find(name.substr(0, dot));
                    if (it != ctx.registry.end() && !it->second->pair) it = ctx.registry.end();
                }
            }
            if (it != ctx.registry.end()) {
                victim = std::move(it->second);
                ctx.registry.erase(it);
            }
        }
        out[i] = victim != nullptr;
    }
}

// gpu_prepare_resident(name) -> BOOLEAN: build the derived structures now
// (see ResidentColumn::prepare). 'p' prepares the key column of a pair (what
// GROUP BY / joins need); 'p.v' prepares the payload (what top-k needs).
void prepare_resident_exec(duckdb_function_info info, duckdb_data_chunk input,
                           duckdb_vector output) {
    duckdb_vector name_vec = duckdb_data_chunk_get_vector(input, 0);
    auto* names = reinterpret_cast<duckdb_string_t*>(duckdb_vector_get_data(name_vec));
    uint64_t* validity = duckdb_vector_get_validity(name_vec);
    const idx_t n = duckdb_data_chunk_get_size(input);
    auto* out = reinterpret_cast<bool*>(duckdb_vector_get_data(output));
    duckdb_vector_ensure_validity_writable(output);
    uint64_t* out_validity = duckdb_vector_get_validity(output);

    ResidentContext& ctx = ctx_of(info);
    for (idx_t i = 0; i < n; ++i) {
        if (validity && !duckdb_validity_row_is_valid(validity, i)) {
            duckdb_validity_set_row_invalid(out_validity, i);
            continue;
        }
        try {
            ResidentRef ref = ctx.acquire_column(read_name(names, i), "gpu_prepare_resident");
            ref.col->prepare();   // no device lock: runs on the column's own stream
            if (ref.set->keys.get() == ref.col && ref.set->keys->prepared()) {
                SetState expect = SetState::Uploaded;
                ref.set->state.compare_exchange_strong(expect, SetState::Ready);
            }
            out[i] = ref.col->prepared();
        } catch (const std::exception& e) {
            duckdb_scalar_function_set_error(info, e.what());
            return;
        }
    }
}

// gpu_assert_rows(name, n) -> BOOLEAN. The in-statement staleness guard
// (docs/TRANSPARENT_DESIGN.md §5.4): raises a TYPED error, prefix
// "GPUDB_STALE:", unless the set exists, is not stale, and was uploaded from
// exactly n rows (rows_seen: the count(*) of the upload's input, NULL rows
// included). The wrapper re-runs natively on that prefix and only on it.
void assert_rows_exec(duckdb_function_info info, duckdb_data_chunk input,
                      duckdb_vector output) {
    duckdb_vector name_vec = duckdb_data_chunk_get_vector(input, 0);
    duckdb_vector n_vec    = duckdb_data_chunk_get_vector(input, 1);
    auto* names = reinterpret_cast<duckdb_string_t*>(duckdb_vector_get_data(name_vec));
    auto* counts = reinterpret_cast<const std::int64_t*>(duckdb_vector_get_data(n_vec));
    uint64_t* name_validity = duckdb_vector_get_validity(name_vec);
    uint64_t* n_validity    = duckdb_vector_get_validity(n_vec);
    const idx_t n = duckdb_data_chunk_get_size(input);
    auto* out = reinterpret_cast<bool*>(duckdb_vector_get_data(output));

    ResidentContext& ctx = ctx_of(info);
    for (idx_t i = 0; i < n; ++i) {
        std::string name = (name_validity && !duckdb_validity_row_is_valid(name_validity, i))
                           ? std::string() : read_name(names, i);
        try {
            if (name_validity && !duckdb_validity_row_is_valid(name_validity, i))
                throw std::runtime_error("GPUDB_STALE: gpu_assert_rows: set name is NULL");
            if (n_validity && !duckdb_validity_row_is_valid(n_validity, i))
                throw std::runtime_error("GPUDB_STALE: gpu_assert_rows: row count for '" +
                                         name + "' is NULL");
            std::shared_ptr<ResidentSet> set;
            {
                std::lock_guard<std::mutex> lock(ctx.registry_mu);
                auto it = ctx.registry.find(name);
                if (it != ctx.registry.end()) set = it->second;
            }
            if (!set)
                throw std::runtime_error("GPUDB_STALE: gpu_assert_rows: no resident set named '" +
                                         name + "'");
            if (set->state.load(std::memory_order_acquire) == SetState::Stale)
                throw std::runtime_error("GPUDB_STALE: gpu_assert_rows: resident set '" + name +
                                         "' was invalidated (epoch " + std::to_string(set->epoch) + ")");
            const std::int64_t have = counts[i];
            if (have < 0 || static_cast<std::size_t>(have) != set->rows_seen)
                throw std::runtime_error("GPUDB_STALE: gpu_assert_rows: resident set '" + name +
                    "' is stale: table has " + std::to_string(have) +
                    " rows, the set was uploaded from " + std::to_string(set->rows_seen));
            out[i] = true;
        } catch (const std::exception& e) {
            duckdb_scalar_function_set_error(info, e.what());
            return;
        }
    }
}

// gpu_invalidate(pattern) -> BIGINT: mark stale every set whose name equals
// `pattern` or starts with `pattern` + ':' (a table prefix of the identity
// tag), and record the invalidation so an upload in flight under a matching
// name is discarded when it finishes. Returns the number of sets marked.
void invalidate_exec(duckdb_function_info info, duckdb_data_chunk input,
                     duckdb_vector output) {
    duckdb_vector name_vec = duckdb_data_chunk_get_vector(input, 0);
    auto* names = reinterpret_cast<duckdb_string_t*>(duckdb_vector_get_data(name_vec));
    uint64_t* validity = duckdb_vector_get_validity(name_vec);
    const idx_t n = duckdb_data_chunk_get_size(input);
    auto* out = reinterpret_cast<std::int64_t*>(duckdb_vector_get_data(output));
    duckdb_vector_ensure_validity_writable(output);
    uint64_t* out_validity = duckdb_vector_get_validity(output);

    ResidentContext& ctx = ctx_of(info);
    for (idx_t i = 0; i < n; ++i) {
        if (validity && !duckdb_validity_row_is_valid(validity, i)) {
            duckdb_validity_set_row_invalid(out_validity, i);
            continue;
        }
        out[i] = ctx.invalidate(read_name(names, i));
    }
}

// ---------------------------------------------------------------------------
// gpu_residents() table function.
// ---------------------------------------------------------------------------

struct ResidentsRow {
    std::string name, origin, catalog, schema, table, columns, kind, dtype, state, last_stats, device;
    std::int64_t table_oid = -1;
    std::int64_t rows = 0, rows_seen = 0, bytes = 0, epoch = 0, hits = 0, refs = 0;
    std::int64_t uploaded_at_us = 0, last_used_at_us = 0;
    bool managed = false, prepared = false, has_stats = false;
};

struct ResidentsInit {
    std::vector<ResidentsRow> rows;
    std::size_t offset = 0;
};

void residents_bind(duckdb_bind_info info) {
    auto add = [&](const char* name, duckdb_type t) {
        duckdb_logical_type lt = duckdb_create_logical_type(t);
        duckdb_bind_add_result_column(info, name, lt);
        duckdb_destroy_logical_type(&lt);
    };
    add("name",         DUCKDB_TYPE_VARCHAR);
    add("origin",       DUCKDB_TYPE_VARCHAR);   // 'managed' | 'explicit'
    add("catalog",      DUCKDB_TYPE_VARCHAR);   // identity fields: NULL for explicit sets
    add("schema",       DUCKDB_TYPE_VARCHAR);
    add("table",        DUCKDB_TYPE_VARCHAR);
    add("table_oid",    DUCKDB_TYPE_BIGINT);
    add("columns",      DUCKDB_TYPE_VARCHAR);
    add("kind",         DUCKDB_TYPE_VARCHAR);   // 'pair' | 'column'
    add("dtype",        DUCKDB_TYPE_VARCHAR);   // 'I64' | 'F64' | 'I64/F64' (pair: key/payload)
    add("rows",         DUCKDB_TYPE_BIGINT);    // rows resident (NULLs skipped)
    add("rows_seen",    DUCKDB_TYPE_BIGINT);    // rows the upload scan delivered
    add("bytes",        DUCKDB_TYPE_BIGINT);    // backend memory incl. derived structures
    add("state",        DUCKDB_TYPE_VARCHAR);   // 'ready' | 'uploaded' | 'stale'
    add("prepared",     DUCKDB_TYPE_BOOLEAN);
    add("epoch",        DUCKDB_TYPE_BIGINT);
    add("hits",         DUCKDB_TYPE_BIGINT);
    add("refs",         DUCKDB_TYPE_BIGINT);    // operator calls holding the set right now
    add("device",       DUCKDB_TYPE_VARCHAR);   // backend the column lives on
    add("uploaded_at",  DUCKDB_TYPE_TIMESTAMP);
    add("last_used_at", DUCKDB_TYPE_TIMESTAMP);
    add("last_stats",   DUCKDB_TYPE_VARCHAR);
}

void residents_init(duckdb_init_info info) {
    auto* init = new ResidentsInit();
    try {
        ResidentContext& ctx = resident_context(duckdb_init_get_extra_info(info));
        std::vector<std::shared_ptr<ResidentSet>> sets;
        {
            std::lock_guard<std::mutex> lock(ctx.registry_mu);
            sets.reserve(ctx.registry.size());
            for (auto& kv : ctx.registry) sets.push_back(kv.second);
        }
        std::sort(sets.begin(), sets.end(),
                  [](const auto& a, const auto& b) { return a->name < b->name; });
        for (auto& s : sets) {
            ResidentsRow r;
            r.name = s->name;
            r.managed = s->managed;
            r.origin = s->managed ? "managed" : "explicit";
            r.catalog = s->catalog; r.schema = s->schema; r.table = s->table;
            r.table_oid = s->table_oid; r.columns = s->columns;
            r.kind = s->pair ? "pair" : "column";
            auto dt = [](const gpudb::ResidentColumn* c) {
                return !c ? "" : c->dtype() == gpudb::Dtype::I64 ? "I64" : "F64";
            };
            r.dtype = s->pair ? std::string(dt(s->keys.get())) + "/" + dt(s->vals.get())
                              : std::string(dt(s->keys.get()));
            r.rows = static_cast<std::int64_t>(s->rows);
            r.rows_seen = static_cast<std::int64_t>(s->rows_seen);
            r.bytes = static_cast<std::int64_t>(s->resident_bytes());
            r.state = to_string(s->state.load());
            r.prepared = s->keys ? s->keys->prepared() : false;
            r.epoch = static_cast<std::int64_t>(s->epoch);
            r.hits = static_cast<std::int64_t>(s->hits.load());
            // use_count: registry + our snapshot vector + `s` alias = 2 (the
            // vector element and the registry) — everything above is a call.
            r.refs = static_cast<std::int64_t>(s.use_count()) - 2;
            if (r.refs < 0) r.refs = 0;
            r.device = s->keys ? gpudb::to_string(s->keys->backend_tag()) : "";
            r.uploaded_at_us = s->uploaded_at_us;
            r.last_used_at_us = s->last_used_at_us.load();
            {
                std::lock_guard<std::mutex> lock(s->stats_mu);
                r.last_stats = s->last_stats;
                r.has_stats = !s->last_stats.empty();
            }
            init->rows.push_back(std::move(r));
        }
    } catch (const std::exception& e) {
        delete init;
        duckdb_init_set_error(info, e.what());
        return;
    }
    duckdb_init_set_init_data(info, init, [](void* p) { delete static_cast<ResidentsInit*>(p); });
}

void residents_function(duckdb_function_info info, duckdb_data_chunk output) {
    auto* init = static_cast<ResidentsInit*>(duckdb_function_get_init_data(info));
    if (!init) return;
    const std::size_t remaining = init->rows.size() - init->offset;
    if (remaining == 0) return;
    constexpr idx_t kChunk = 2048;
    const idx_t out_n = static_cast<idx_t>(std::min<std::size_t>(remaining, kChunk));

    auto vec = [&](idx_t c) { return duckdb_data_chunk_get_vector(output, c); };
    auto set_str = [&](idx_t c, idx_t i, const std::string& s, bool valid = true) {
        duckdb_vector v = vec(c);
        if (!valid) {
            duckdb_vector_ensure_validity_writable(v);
            duckdb_validity_set_row_invalid(duckdb_vector_get_validity(v), i);
            return;
        }
        duckdb_vector_assign_string_element_len(v, i, s.data(), s.size());
    };
    auto set_i64 = [&](idx_t c, idx_t i, std::int64_t x, bool valid = true) {
        duckdb_vector v = vec(c);
        if (!valid) {
            duckdb_vector_ensure_validity_writable(v);
            duckdb_validity_set_row_invalid(duckdb_vector_get_validity(v), i);
            return;
        }
        static_cast<std::int64_t*>(duckdb_vector_get_data(v))[i] = x;
    };
    auto set_bool = [&](idx_t c, idx_t i, bool x) {
        static_cast<bool*>(duckdb_vector_get_data(vec(c)))[i] = x;
    };
    auto set_ts = [&](idx_t c, idx_t i, std::int64_t us) {
        duckdb_vector v = vec(c);
        if (us == 0) {
            duckdb_vector_ensure_validity_writable(v);
            duckdb_validity_set_row_invalid(duckdb_vector_get_validity(v), i);
            return;
        }
        static_cast<duckdb_timestamp*>(duckdb_vector_get_data(v))[i].micros = us;
    };

    for (idx_t i = 0; i < out_n; ++i) {
        const ResidentsRow& r = init->rows[init->offset + i];
        set_str(0, i, r.name);
        set_str(1, i, r.origin);
        set_str(2, i, r.catalog, r.managed);
        set_str(3, i, r.schema, r.managed);
        set_str(4, i, r.table, r.managed);
        set_i64(5, i, r.table_oid, r.managed);
        set_str(6, i, r.columns, r.managed);
        set_str(7, i, r.kind);
        set_str(8, i, r.dtype);
        set_i64(9, i, r.rows);
        set_i64(10, i, r.rows_seen);
        set_i64(11, i, r.bytes);
        set_str(12, i, r.state);
        set_bool(13, i, r.prepared);
        set_i64(14, i, r.epoch);
        set_i64(15, i, r.hits);
        set_i64(16, i, r.refs);
        set_str(17, i, r.device);
        set_ts(18, i, r.uploaded_at_us);
        set_ts(19, i, r.last_used_at_us);
        set_str(20, i, r.last_stats, r.has_stats);
    }
    duckdb_data_chunk_set_size(output, out_n);
    init->offset += static_cast<std::size_t>(out_n);
}

// ---------------------------------------------------------------------------
// Registration helpers.
// ---------------------------------------------------------------------------

// Build + register one scalar function. All of these are volatile: their
// result depends on the mutable resident registry, so DuckDB must not
// constant-fold or cache them across calls.
void register_scalar(duckdb_connection con, const char* name,
                     duckdb_scalar_function_t exec, duckdb_type ret,
                     const std::vector<duckdb_type>& params,
                     const std::shared_ptr<ResidentContext>& ctx) {
    duckdb_scalar_function fn = duckdb_create_scalar_function();
    duckdb_scalar_function_set_name(fn, name);
    for (duckdb_type pt : params) {
        duckdb_logical_type t = duckdb_create_logical_type(pt);
        duckdb_scalar_function_add_parameter(fn, t);
        duckdb_destroy_logical_type(&t);
    }
    duckdb_logical_type t_ret = duckdb_create_logical_type(ret);
    duckdb_scalar_function_set_return_type(fn, t_ret);
    duckdb_destroy_logical_type(&t_ret);
    duckdb_scalar_function_set_volatile(fn);
    duckdb_scalar_function_set_function(fn, exec);
    duckdb_scalar_function_set_extra_info(fn, resident_extra_info(ctx), resident_extra_info_destroy);
    duckdb_state st = duckdb_register_scalar_function(con, fn);
    duckdb_destroy_scalar_function(&fn);
    if (st == DuckDBError) {
        throw std::runtime_error(std::string(name) + " registration failed");
    }
}

void register_scalar_names(duckdb_connection con, const char* name,
                           duckdb_scalar_function_t exec, duckdb_type ret, int n_name_params,
                           const std::shared_ptr<ResidentContext>& ctx) {
    std::vector<duckdb_type> params(static_cast<std::size_t>(n_name_params), DUCKDB_TYPE_VARCHAR);
    register_scalar(con, name, exec, ret, params, ctx);
}

} // namespace

// ---------------------------------------------------------------------------
// Fused resident join-aggregate (v0.5).
//   gpu_join_sum_resident(probe_keys, payload, build_keys) -> BIGINT
//     SELECT sum(p.payload) FROM probe p JOIN build b ON p.key = b.key,
//     full build-side multiplicity; NULL when no rows join (SQL SUM semantics).
//   gpu_join_count_resident(probe_keys, build_keys) -> BIGINT
//     the same join's COUNT(*); 0 when no rows join.
// count_only reuses the probe-key column as its own payload — the sum is
// discarded, only `matched` is read.
// ---------------------------------------------------------------------------

namespace {

void record_join_stats(ResidentContext& ctx, ResidentSet* set, const char* op,
                       const gpudb::JoinAggResult& r) {
    const auto& d = ctx.aggregator().last_decision();
    char buf[320];
    std::snprintf(buf, sizeof(buf),
        "op=%s backend=%s reason=%s rows_probe=%zu rows_build=%zu matched=%lld "
        "wall_ms=%.3f kernel_ms=%.3f transfer_ms=%.3f",
        op, gpudb::to_string(d.chosen), gpudb::to_string(d.reason),
        r.rows_probe, r.rows_build, static_cast<long long>(r.matched),
        r.wall_ms, r.kernel_ms, r.transfer_ms);
    ctx.record_stats(set, buf);
}

template <gpudb::JoinKind K, bool COUNT_ONLY>
void join_sum_resident_exec(duckdb_function_info info, duckdb_data_chunk input,
                            duckdb_vector output) {
    constexpr int kArgs = COUNT_ONLY ? 2 : 3;
    duckdb_vector    vecs[3] = {};
    duckdb_string_t* names[3] = {};
    uint64_t*        valid[3] = {};
    for (int a = 0; a < kArgs; ++a) {
        vecs[a]  = duckdb_data_chunk_get_vector(input, a);
        names[a] = reinterpret_cast<duckdb_string_t*>(duckdb_vector_get_data(vecs[a]));
        valid[a] = duckdb_vector_get_validity(vecs[a]);
    }
    const idx_t n = duckdb_data_chunk_get_size(input);
    auto* out = reinterpret_cast<std::int64_t*>(duckdb_vector_get_data(output));
    duckdb_vector_ensure_validity_writable(output);
    uint64_t* out_validity = duckdb_vector_get_validity(output);

    ResidentContext& ctx = ctx_of(info);
    for (idx_t i = 0; i < n; ++i) {
        bool null_arg = false;
        for (int a = 0; a < kArgs; ++a) {
            if (valid[a] && !duckdb_validity_row_is_valid(valid[a], i)) { null_arg = true; break; }
        }
        if (null_arg) {
            duckdb_validity_set_row_invalid(out_validity, i);
            continue;
        }
        try {
            ResidentRef cols[3];
            for (int a = 0; a < kArgs; ++a) {
                cols[a] = ctx.acquire_column(read_name(names[a], i), "gpu_join_*_resident");
                if (cols[a].col->dtype() != gpudb::Dtype::I64)
                    throw std::runtime_error(
                        "gpu_join_*_resident operates on BIGINT resident columns "
                        "(f64 keys/payloads are not in the v0.5 join ABI)");
            }
            gpudb::ResidentColumn* probe   = cols[0].col;
            gpudb::ResidentColumn* payload = COUNT_ONLY ? cols[0].col : cols[1].col;
            gpudb::ResidentColumn* build   = COUNT_ONLY ? cols[1].col : cols[2].col;
            auto& a = ctx.aggregator();
            auto dev = resident_device_lock(ctx);
            gpudb::JoinAggResult r = a.join_sum_resident_i64(*probe, *payload, *build, K);
            record_join_stats(ctx, cols[0].set.get(),
                              COUNT_ONLY ? "join_count_resident" : "join_sum_resident", r);
            if (COUNT_ONLY) {
                out[i] = r.matched;
            } else if (r.matched == 0) {
                duckdb_validity_set_row_invalid(out_validity, i);  // SUM over ∅ = NULL
            } else {
                out[i] = r.sum;
            }
        } catch (const std::exception& e) {
            duckdb_scalar_function_set_error(info, e.what());
            return;
        }
    }
}

// f64-payload flavor: (probe_keys, payload_f64, build_keys) -> DOUBLE.
// Keys must be BIGINT resident columns; payload must be DOUBLE.
template <gpudb::JoinKind K>
void join_sum_resident_f64_exec(duckdb_function_info info, duckdb_data_chunk input,
                                duckdb_vector output) {
    duckdb_vector    vecs[3] = {};
    duckdb_string_t* names[3] = {};
    uint64_t*        valid[3] = {};
    for (int a = 0; a < 3; ++a) {
        vecs[a]  = duckdb_data_chunk_get_vector(input, a);
        names[a] = reinterpret_cast<duckdb_string_t*>(duckdb_vector_get_data(vecs[a]));
        valid[a] = duckdb_vector_get_validity(vecs[a]);
    }
    const idx_t n = duckdb_data_chunk_get_size(input);
    auto* out = reinterpret_cast<double*>(duckdb_vector_get_data(output));
    duckdb_vector_ensure_validity_writable(output);
    uint64_t* out_validity = duckdb_vector_get_validity(output);

    ResidentContext& ctx = ctx_of(info);
    for (idx_t i = 0; i < n; ++i) {
        bool null_arg = false;
        for (int a = 0; a < 3; ++a) {
            if (valid[a] && !duckdb_validity_row_is_valid(valid[a], i)) { null_arg = true; break; }
        }
        if (null_arg) {
            duckdb_validity_set_row_invalid(out_validity, i);
            continue;
        }
        try {
            ResidentRef cols[3];
            for (int a = 0; a < 3; ++a)
                cols[a] = ctx.acquire_column(read_name(names[a], i), "gpu_*join_sum_resident_f64");
            if (cols[0].col->dtype() != gpudb::Dtype::I64 ||
                cols[2].col->dtype() != gpudb::Dtype::I64)
                throw std::runtime_error("gpu_*join_sum_resident_f64: key columns must be BIGINT");
            if (cols[1].col->dtype() != gpudb::Dtype::F64)
                throw std::runtime_error(
                    "gpu_*join_sum_resident_f64: payload must be a DOUBLE resident "
                    "column — for BIGINT payloads use the non-_f64 variant");
            auto& a = ctx.aggregator();
            auto dev = resident_device_lock(ctx);
            gpudb::JoinAggResult r =
                a.join_sum_resident_f64(*cols[0].col, *cols[1].col, *cols[2].col, K);
            record_join_stats(ctx, cols[0].set.get(), "join_sum_resident_f64", r);
            if (r.matched == 0) {
                duckdb_validity_set_row_invalid(out_validity, i);  // SUM over ∅ = NULL
            } else {
                out[i] = r.sum_f64;
            }
        } catch (const std::exception& e) {
            duckdb_scalar_function_set_error(info, e.what());
            return;
        }
    }
}

} // namespace

void register_gpu_resident(duckdb_connection con) {
    register_gpu_resident(con, make_resident_context());
}

void register_gpu_resident(duckdb_connection con,
                           const std::shared_ptr<ResidentContext>& ctx) {
    // gpu_upload overload set: (VARCHAR, BIGINT) and (VARCHAR, DOUBLE).
    duckdb_aggregate_function_set set = duckdb_create_aggregate_function_set("gpu_upload");
    duckdb_aggregate_function fn_i64 =
        make_upload_fn(DUCKDB_TYPE_BIGINT, upload_update_t<std::int64_t, gpudb::Dtype::I64>, ctx);
    duckdb_aggregate_function fn_f64 =
        make_upload_fn(DUCKDB_TYPE_DOUBLE, upload_update_t<double, gpudb::Dtype::F64>, ctx);
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

    // gpu_upload_pair(name, k BIGINT, v BIGINT|DOUBLE) -> BIGINT, registers
    // one set with '<name>.k' and '<name>.v' in guaranteed positional alignment.
    {
        auto make_pair_fn = [&](duckdb_type v_type,
                                duckdb_aggregate_update_t update,
                                duckdb_aggregate_finalize_t finalize) {
            duckdb_aggregate_function pfn = duckdb_create_aggregate_function();
            duckdb_aggregate_function_set_name(pfn, "gpu_upload_pair");
            duckdb_logical_type t_name = duckdb_create_logical_type(DUCKDB_TYPE_VARCHAR);
            duckdb_logical_type t_k    = duckdb_create_logical_type(DUCKDB_TYPE_BIGINT);
            duckdb_logical_type t_v    = duckdb_create_logical_type(v_type);
            duckdb_logical_type t_ret  = duckdb_create_logical_type(DUCKDB_TYPE_BIGINT);
            duckdb_aggregate_function_add_parameter(pfn, t_name);
            duckdb_aggregate_function_add_parameter(pfn, t_k);
            duckdb_aggregate_function_add_parameter(pfn, t_v);
            duckdb_aggregate_function_set_return_type(pfn, t_ret);
            duckdb_destroy_logical_type(&t_name);
            duckdb_destroy_logical_type(&t_k);
            duckdb_destroy_logical_type(&t_v);
            duckdb_destroy_logical_type(&t_ret);
            duckdb_aggregate_function_set_functions(pfn,
                upload_state_size, upload_state_init, update,
                upload_combine, finalize);
            duckdb_aggregate_function_set_destructor(pfn, upload_state_destroy);
            duckdb_aggregate_function_set_special_handling(pfn);
            duckdb_aggregate_function_set_extra_info(pfn, resident_extra_info(ctx),
                                                     resident_extra_info_destroy);
            return pfn;
        };
        duckdb_aggregate_function_set pset =
            duckdb_create_aggregate_function_set("gpu_upload_pair");
        duckdb_aggregate_function p_i64 = make_pair_fn(DUCKDB_TYPE_BIGINT,
            upload_pair_update_t<std::int64_t, gpudb::Dtype::I64>,
            upload_pair_finalize_t<gpudb::Dtype::I64>);
        duckdb_aggregate_function p_f64 = make_pair_fn(DUCKDB_TYPE_DOUBLE,
            upload_pair_update_t<double, gpudb::Dtype::F64>,
            upload_pair_finalize_t<gpudb::Dtype::F64>);
        duckdb_state pa1 = duckdb_add_aggregate_function_to_set(pset, p_i64);
        duckdb_state pa2 = duckdb_add_aggregate_function_to_set(pset, p_f64);
        duckdb_destroy_aggregate_function(&p_i64);
        duckdb_destroy_aggregate_function(&p_f64);
        if (pa1 == DuckDBError || pa2 == DuckDBError) {
            duckdb_destroy_aggregate_function_set(&pset);
            throw std::runtime_error("gpu_upload_pair overload set assembly failed");
        }
        duckdb_state pst = duckdb_register_aggregate_function_set(con, pset);
        duckdb_destroy_aggregate_function_set(&pset);
        if (pst == DuckDBError) {
            throw std::runtime_error("gpu_upload_pair registration failed");
        }
    }

    register_scalar_names(con, "gpu_sum_resident",
        resident_i64_exec<&gpudb::Aggregator::sum_resident_i64>, DUCKDB_TYPE_BIGINT, 1, ctx);
    register_scalar_names(con, "gpu_min_resident",
        resident_i64_exec<&gpudb::Aggregator::min_resident_i64>, DUCKDB_TYPE_BIGINT, 1, ctx);
    register_scalar_names(con, "gpu_max_resident",
        resident_i64_exec<&gpudb::Aggregator::max_resident_i64>, DUCKDB_TYPE_BIGINT, 1, ctx);
    register_scalar_names(con, "gpu_sum_resident_f64",
        sum_resident_f64_exec, DUCKDB_TYPE_DOUBLE, 1, ctx);
    register_scalar_names(con, "gpu_resident_info",
        resident_info_exec, DUCKDB_TYPE_VARCHAR, 1, ctx);
    register_scalar_names(con, "gpu_drop_resident",
        drop_resident_exec, DUCKDB_TYPE_BOOLEAN, 1, ctx);
    register_scalar_names(con, "gpu_prepare_resident",
        prepare_resident_exec, DUCKDB_TYPE_BOOLEAN, 1, ctx);
    register_scalar_names(con, "gpu_invalidate",
        invalidate_exec, DUCKDB_TYPE_BIGINT, 1, ctx);
    register_scalar_names(con, "gpu_upload_begin",
        upload_begin_exec, DUCKDB_TYPE_BOOLEAN, 1, ctx);
    register_scalar_names(con, "gpu_upload_finish",
        upload_finish_exec, DUCKDB_TYPE_BIGINT, 1, ctx);
    register_scalar_names(con, "gpu_upload_abort",
        upload_abort_exec, DUCKDB_TYPE_BOOLEAN, 1, ctx);
    register_scalar_names(con, "gpu_upload_status",
        upload_status_exec, DUCKDB_TYPE_VARCHAR, 1, ctx);
    register_scalar(con, "gpu_assert_rows", assert_rows_exec, DUCKDB_TYPE_BOOLEAN,
                    {DUCKDB_TYPE_VARCHAR, DUCKDB_TYPE_BIGINT}, ctx);
    register_scalar_names(con, "gpu_last_stats",
        last_stats_exec, DUCKDB_TYPE_VARCHAR, 0, ctx);
    register_scalar_names(con, "gpu_build_info",
        build_info_exec, DUCKDB_TYPE_VARCHAR, 0, ctx);

    // gpu_residents() table function.
    {
        duckdb_table_function tf = duckdb_create_table_function();
        duckdb_table_function_set_name(tf, "gpu_residents");
        duckdb_table_function_set_bind(tf, residents_bind);
        duckdb_table_function_set_init(tf, residents_init);
        duckdb_table_function_set_function(tf, residents_function);
        duckdb_table_function_set_extra_info(tf, resident_extra_info(ctx), resident_extra_info_destroy);
        if (duckdb_register_table_function(con, tf) == DuckDBError) {
            duckdb_destroy_table_function(&tf);
            throw std::runtime_error("gpu_residents registration failed");
        }
        duckdb_destroy_table_function(&tf);
    }

    // Fused resident joins, full kind spectrum. RIGHT/FULL OUTER compose:
    // probe-payload sum equals INNER/LEFT respectively, and the extra
    // COUNT(*) term is gpu_anti_join_count_resident with sides swapped.
    using JK = gpudb::JoinKind;
    register_scalar_names(con, "gpu_join_sum_resident",
        join_sum_resident_exec<JK::INNER, false>, DUCKDB_TYPE_BIGINT, 3, ctx);
    register_scalar_names(con, "gpu_join_count_resident",
        join_sum_resident_exec<JK::INNER, true>, DUCKDB_TYPE_BIGINT, 2, ctx);
    register_scalar_names(con, "gpu_join_sum_resident_f64",
        join_sum_resident_f64_exec<JK::INNER>, DUCKDB_TYPE_DOUBLE, 3, ctx);
    register_scalar_names(con, "gpu_left_join_sum_resident",
        join_sum_resident_exec<JK::LEFT, false>, DUCKDB_TYPE_BIGINT, 3, ctx);
    register_scalar_names(con, "gpu_left_join_count_resident",
        join_sum_resident_exec<JK::LEFT, true>, DUCKDB_TYPE_BIGINT, 2, ctx);
    register_scalar_names(con, "gpu_left_join_sum_resident_f64",
        join_sum_resident_f64_exec<JK::LEFT>, DUCKDB_TYPE_DOUBLE, 3, ctx);
    register_scalar_names(con, "gpu_semi_join_sum_resident",
        join_sum_resident_exec<JK::SEMI, false>, DUCKDB_TYPE_BIGINT, 3, ctx);
    register_scalar_names(con, "gpu_semi_join_count_resident",
        join_sum_resident_exec<JK::SEMI, true>, DUCKDB_TYPE_BIGINT, 2, ctx);
    register_scalar_names(con, "gpu_semi_join_sum_resident_f64",
        join_sum_resident_f64_exec<JK::SEMI>, DUCKDB_TYPE_DOUBLE, 3, ctx);
    register_scalar_names(con, "gpu_anti_join_sum_resident",
        join_sum_resident_exec<JK::ANTI, false>, DUCKDB_TYPE_BIGINT, 3, ctx);
    register_scalar_names(con, "gpu_anti_join_count_resident",
        join_sum_resident_exec<JK::ANTI, true>, DUCKDB_TYPE_BIGINT, 2, ctx);
    register_scalar_names(con, "gpu_anti_join_sum_resident_f64",
        join_sum_resident_f64_exec<JK::ANTI>, DUCKDB_TYPE_DOUBLE, 3, ctx);
}

} // namespace gpudb_ext
