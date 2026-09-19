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
//                                                    pair in ONE scan; rows
//                                                    with a NULL half skipped
//   gpu_upload_pair_exact(name, k BIGINT, v BIGINT) -> BIGINT   the pair WITH
//                                                    its NULLs (v0.7 §4.1) for
//                                                    gpu_groupby_exact_resident
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
#include "backend_notes.hpp"
#include "resident_shed_note.hpp"

#if defined(GPUDB_C_STRUCT_ABI)
DUCKDB_EXTENSION_EXTERN
#endif

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cfloat>
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
#include <thread>
#include <unordered_map>
#include <vector>

#include <unistd.h>
#if defined(__linux__)
#include <sys/mman.h>
#endif

namespace gpudb_ext {

// FNV-1a over the bytes, then a 64-bit avalanche (splitmix64 finaliser):
// FNV alone clusters on short similar strings; the finaliser spreads them.
std::uint64_t hash64(const char* p, std::size_t n) noexcept {
    std::uint64_t h = 1469598103934665603ULL;
    for (std::size_t i = 0; i < n; ++i) { h ^= static_cast<unsigned char>(p[i]); h *= 1099511628211ULL; }
    h ^= n;
    h ^= h >> 30; h *= 0xbf58476d1ce4e5b9ULL;
    h ^= h >> 27; h *= 0x94d049bb133111ebULL;
    h ^= h >> 31;
    return h;
}

std::string tuple_component(const char* p, std::size_t n) {
    std::string out = std::to_string(n);
    out.push_back(':');
    out.append(p, n);
    return out;
}

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

// §4.12: the tag of a set only global aggregates read writes lane 0 as '-'.
// There is no GROUP BY key, so nothing is sorted and no key lane is kept.
// An INTERMEDIATE of a chained device join carries the FINAL set's tag with
// '.<step>' appended, so its lane list describes the final set and not it:
// its own lane 0 is whatever the step put there (the next step's probe key),
// and it is never keyless.
bool no_key_tag(const TagFields& t) {
    if (t.extra.find('.') != std::string::npos) return false;
    return t.columns.compare(0, 2, "-,") == 0 || t.columns == "-";
}

// A JOIN RESULT set — `join-<digest>` (materialised on the device) or
// `joinu-<digest>` (DuckDB evaluated the join and the wrapper uploaded its
// rows). Its lane 0 is the statement's GROUP BY key and nothing else, which
// is what lets the backend release that lane once it holds the key's group-id
// lane (docs/RESIDENT_COLUMNS_DESIGN.md §9). An INTERMEDIATE of a chained
// device join is written '<digest>.<step>' and is NOT one: the next step
// probes its lane 0 as a join key.
bool join_result_tag(const TagFields& t) {
    const bool kind = starts_with(t.extra, "join-") || starts_with(t.extra, "joinu-");
    return kind && t.extra.find('.') == std::string::npos;
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

    // Exact uploads (gpu_upload_pair_exact: 2 lanes; gpu_upload_rows_exact:
    // 2 + n_pi + n_pf): one validity bitmap per lane, indexed by row (= lane
    // offset / lanes_per_row); a lane's bitmap is allocated all-ones on its
    // first NULL, so NULL-free segments cost nothing. Never touched by the
    // legacy uploads.
    std::vector<std::vector<std::uint64_t>> lane_valid;
    void mark_lane_null(std::size_t row, std::size_t lane, std::size_t n_lanes) {
        if (lane_valid.empty()) lane_valid.resize(n_lanes);
        auto& m = lane_valid[lane];
        if (m.empty()) m.assign((kLanes / n_lanes + 63) / 64, ~std::uint64_t{0});
        m[row >> 6] &= ~(std::uint64_t{1} << (row & 63));
    }
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
    // gpu_upload_rows_exact: lanes per row (2 + n_pi + n_pf + n_ps), fixed by
    // the first row; 0 for every other upload function.
    std::size_t               lanes_per_row = 0;
    std::size_t               n_pi = 0, n_pf = 0, n_ps = 0;
    bool                      key_str = false;   // key lane holds hash64(tuple text)
    bool                      exact = false;     // filled by an EXACT upload: charged against exact_pool_cap_bytes()
    // Stage B column upload (gpu_upload_columns): lane 0 is the row id. The
    // FAST PATH is a plain scan, whose rows arrive in ascending row-id order
    // inside each update call: every call's rows are remembered as a chunk with
    // the row ids it spans, and finish orders the chunks by first row id and
    // places each at its rank (RowSpan::dst_row / valid_bit) — no data moves.
    // Any other order (a lane expression carrying a correlated subquery is
    // evaluated through a join, which reorders the rows inside a chunk) sets
    // `unordered`, or shows up as chunks overlapping in row-id order; finish
    // then takes the PLACED PATH and gathers the rows into fresh segments at
    // their row-id rank.
    bool                      columns = false;
    struct ChunkRec { std::int64_t first_rowid, last_rowid; std::shared_ptr<Segment> seg; std::size_t row0, rows; };
    std::vector<ChunkRec>     chunks;
    std::uint64_t             chunk_call = 0;      // the update call the open chunk belongs to
    bool                      unordered = false;   // a row id went backwards inside one call
    // hash -> text per string lane: [0] the key (when key_str), then one per
    // s<n> lane. Filled lock-free per state, merged at combine / finish; a
    // second text under one hash is a collision and fails the upload.
    std::vector<std::unordered_map<std::uint64_t, std::string>> dicts;
    void note_string(std::size_t slot, std::uint64_t h, const char* p, std::size_t n, bool& collision) {
        if (dicts.size() <= slot) dicts.resize(slot + 1);
        auto& m = dicts[slot];
        auto it = m.find(h);
        if (it == m.end()) { m.emplace(h, std::string(p, n)); return; }
        if (it->second.size() != n || std::memcmp(it->second.data(), p, n) != 0) collision = true;
    }

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

// The EXACT uploads (gpu_upload_pair_exact / gpu_upload_rows_exact) are the transparent
// path's: the wrapper sizes every set against its device memory budget BEFORE the upload
// (docs/TRANSPARENT_DESIGN.md §5.5), so the host staging of one set is bounded by that
// budget — and a 4 GB wall here kept every set above ~250M row-lanes off the device
// (TPC-H SF50: 0 of 22 queries resident, found 2026-09-17). They get their own cap:
// GPUDB_EXACT_UPLOAD_POOL_MAX_MB, default half of physical memory (never below the
// general cap). The window-frame guardrail keeps its meaning: there is still a wall.
std::size_t exact_pool_cap_bytes() {
    static const std::size_t cap = [] {
        std::size_t c = pool_cap_bytes();
        if (const char* s = std::getenv("GPUDB_EXACT_UPLOAD_POOL_MAX_MB")) {
            char* end = nullptr;
            errno = 0;
            const unsigned long long v = std::strtoull(s, &end, 10);
            if (errno == 0 && end && end != s && *end == '\0' && v > 0 &&
                v <= (static_cast<unsigned long long>(SIZE_MAX) >> 20)) return static_cast<std::size_t>(v) << 20;
            std::fprintf(stderr, "[gpudb] ignoring GPUDB_EXACT_UPLOAD_POOL_MAX_MB='%s'\n", s);
        }
        const long pages = sysconf(_SC_PHYS_PAGES), page = sysconf(_SC_PAGE_SIZE);
        if (pages > 0 && page > 0)
            c = std::max(c, static_cast<std::size_t>(pages) / 2 * static_cast<std::size_t>(page));
        return c;
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
bool pool_reserve(std::size_t n, bool exact = false) {
    Pool& P = pool();
    if (P.bytes.load(std::memory_order_relaxed) + n > (exact ? exact_pool_cap_bytes() : pool_cap_bytes())) return false;
    P.bytes.fetch_add(n, std::memory_order_relaxed);
    return true;
}
void pool_release(std::size_t n) {
    if (n) pool().bytes.fetch_sub(n, std::memory_order_relaxed);
}

// A scoped charge for a temporary buffer: whatever is still held goes back to
// the pool on every exit, the throwing ones included.
struct PoolCharge {
    std::size_t bytes = 0;
    bool take(std::size_t n, bool exact) {
        if (!pool_reserve(n, exact)) return false;
        bytes += n;
        return true;
    }
    void give(std::size_t n) { n = std::min(n, bytes); pool_release(n); bytes -= n; }
    ~PoolCharge() { pool_release(bytes); }
    PoolCharge() = default;
    PoolCharge(const PoolCharge&) = delete;
    PoolCharge& operator=(const PoolCharge&) = delete;
};

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
    bool          exact = false;                    // gpu_upload_pair_exact / gpu_upload_rows_exact segments
    bool          columns = false;                  // gpu_upload_columns segments (stage B)
    std::vector<UploadBuf::ChunkRec> chunks;
    bool          unordered = false;
    std::size_t   lanes_per_row = 0, n_pi = 0, n_pf = 0, n_ps = 0;   // exact: lane layout (fixed by the first segment)
    bool          key_str = false;
    std::vector<std::unordered_map<std::uint64_t, std::string>> dicts;   // string lanes, merged per segment
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
    std::unordered_map<std::string, std::shared_ptr<TableStore>>    stores;     // stage B: by store key

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
        for (auto it = stores.begin(); it != stores.end();) {
            if (matches(it->first, prefix)) { ++n; it = stores.erase(it); }   // the columns go with the last view
            else ++it;
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

    // Caller holds registry_mu. False (and the offending source's name) when
    // a source of `set` is gone, replaced or stale.
    bool deps_current_locked(const ResidentSet& set, std::string& changed, int depth) const {
        for (const auto& d : set.deps) {
            const auto src = d.second.lock();
            const auto it = registry.find(d.first);
            if (!src || it == registry.end() || it->second != src ||
                src->state.load(std::memory_order_acquire) == SetState::Stale ||
                depth > 8 || !deps_current_locked(*src, changed, depth + 1)) {
                if (changed.empty()) changed = d.first;
                return false;
            }
        }
        return true;
    }

    // Lookup with hit accounting. Throws when unknown or stale.
    // Stage B: a plain identity tag with no set of its own is answered by a VIEW
    // over the table's store when every lane it names is resident there. The
    // view is registered under the tag (guards, deps and stats see a set like
    // any other) and shares the store's columns, dictionaries and sort caches.
    // Caller holds registry_mu. Returns null when the store cannot serve it.
    std::shared_ptr<ResidentSet> view_from_store_locked(const std::string& name) {
        if (!starts_with(name, kTagPrefix)) return nullptr;
        TagFields t;
        if (!parse_tag(name, t).empty()) return nullptr;
        // a plain set, a join's base set, or a set only global aggregates read (§4.12)
        const bool global = t.extra == "global";
        if (!t.extra.empty() && t.extra != "join" && !global) return nullptr;
        const std::string key = std::string("gpudb:v1:") + t.catalog + ":" + t.schema + ":" + t.table + ":" + std::to_string(t.table_oid);
        auto st = stores.find(key);
        if (st == stores.end()) return nullptr;
        TableStore& store = *st->second;
        std::vector<std::string> lanes;
        for (std::size_t a = 0, b; a <= t.columns.size(); a = b + 1) {
            b = t.columns.find(',', a);
            if (b == std::string::npos) b = t.columns.size();
            lanes.push_back(t.columns.substr(a, b - a));
        }
        if (lanes.size() < 2) return nullptr;
        auto lane = [&](const std::string& e) -> std::shared_ptr<StoreColumn> {
            auto it = store.cols.find(e);
            return it == store.cols.end() ? nullptr : it->second;
        };
        auto set = std::make_shared<ResidentSet>();
        std::shared_ptr<StoreColumn> k;
        if (global && lanes[0] == "-") {
            // §4.12: no GROUP BY key. The first real lane (the payload, else the
            // first predicate lane) stands in so the set's invariants hold; the
            // global operator never reads it as a key and nothing sorts it.
            set->no_key = true;
            for (std::size_t l = 1; l < lanes.size() && !k; ++l)
                if (lanes[l] != "-") k = lane(lanes[l]);
            if (!k || k->is_str) return nullptr;
        } else {
            k = lane("k#" + lanes[0]);           // a tuple-text key lives under its role prefix
            if (!k) k = lane(lanes[0]);
        }
        if (!k) return nullptr;
        set->keys = k->col; set->key_str = k->is_str; set->key_dict = k->dict;
        set->store_cols.push_back(k);
        if (set->no_key) { set->key_str = false; set->key_dict = nullptr; }
        if (lanes[1] != "-") {
            auto v = lane(lanes[1]);
            if (!v || v->is_str) return nullptr;
            set->vals = v->col;
            set->store_cols.push_back(v);
        } else {
            // no payload (count(*) only): the operators want a payload column to reduce; the
            // key column stands in — the rewrite reads count_star and the keys, nothing of it
            set->vals = set->keys;
        }
        std::vector<std::shared_ptr<StoreColumn>> pi, pf, ps;     // the WHERE program's i<n> / f<n> / s<n> order
        for (std::size_t l = 2; l < lanes.size(); ++l) {
            auto c = lane(lanes[l]);
            if (!c) return nullptr;
            if (c->is_str) ps.push_back(c);
            else if (c->col->dtype() == gpudb::Dtype::F64) pf.push_back(c);
            else pi.push_back(c);
        }
        for (auto* grp : {&pi, &pf, &ps})
            for (auto& c : *grp) {
                set->preds.push_back(c->col);
                set->store_cols.push_back(c);
                if (c->is_str) set->str_dicts.push_back(c->dict);
            }
        set->pred_int = pi.size(); set->pred_dbl = pf.size(); set->pred_str = ps.size();
        set->name = name; set->managed = true; set->pair = true; set->exact = true; set->view = true;
        set->store_key = key;
        set->catalog = t.catalog; set->schema = t.schema; set->table = t.table; set->table_oid = t.table_oid;
        set->columns = t.columns; set->extra = t.extra;
        set->rows = set->rows_seen = store.rows_seen;
        set->epoch = store.epoch;
        set->uploaded_at_us = now_us();
        set->state.store(SetState::Uploaded);
        registry[name] = set;
        return set;
    }
    // A set by name, or a view synthesised for it. Caller holds registry_mu.
    std::shared_ptr<ResidentSet> find_or_view_locked(const std::string& name) {
        auto it = registry.find(name);
        if (it != registry.end()) {
            // a stale view is only a stale name: the store may hold fresh columns again
            if (!(it->second->view && it->second->state.load(std::memory_order_acquire) == SetState::Stale))
                return it->second;
            registry.erase(it);
        }
        return view_from_store_locked(name);
    }
    // Recency for the memory budget's LRU. A view knows which store columns it
    // reads, so this is one relaxed store per lane and needs no lock: it used to
    // re-take the registry lock to walk every column of the store and search the
    // set's lane list for each, per statement, for a number only
    // gpu_store_columns() and the wrapper's budget ever read.
    static void stamp_store_used(const ResidentSet& v, std::int64_t t) {
        for (const auto& c : v.store_cols)
            if (c) c->last_used_at_us.store(t, std::memory_order_relaxed);
    }

    std::shared_ptr<ResidentSet> acquire(const std::string& name, const char* fn,
                                         bool count_hit = true) {
        std::shared_ptr<ResidentSet> s;
        bool fresh_view = false;
        {
            std::lock_guard<std::mutex> lock(registry_mu);
            auto it = registry.find(name);
            if (it != registry.end() &&
                !(it->second->view && it->second->state.load(std::memory_order_acquire) == SetState::Stale)) {
                s = it->second;
            } else {
                if (it != registry.end()) registry.erase(it);       // a stale view: re-synthesise from the store
                s = view_from_store_locked(name);
                fresh_view = s != nullptr;
            }
        }
        if (fresh_view && s->keys) {
            // the sort cache belongs to the (shared) key column: built once, used by every view on it
            // (§4.12: a global set has no key — sorting its lanes would buy nothing)
            if (!s->no_key) s->keys->prepare();
            s->state.store(SetState::Ready);
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
        if (!s->deps.empty()) {
            // A derived (joined) set lives only as long as each source — and,
            // through a chain of joins, each source's sources — is the
            // registry's current, non-stale entry for its name.
            std::lock_guard<std::mutex> lock(registry_mu);
            std::string changed;
            if (!deps_current_locked(*s, changed, 0)) {
                s->state.store(SetState::Stale);
                throw std::runtime_error(std::string("GPUDB_STALE: ") + fn + ": resident set '" +
                    name + "' was joined from '" + changed + "', which changed — materialize it again");
            }
        }
        if (count_hit) {
            const std::int64_t t = now_us();
            s->hits.fetch_add(1, std::memory_order_relaxed);
            s->last_used_at_us.store(t, std::memory_order_relaxed);
            if (s->view) stamp_store_used(*s, t);
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
        // A pair name used where a column is expected: hand out the keys
        // (a plain identity tag may be a view over the table's store, stage B).
        {
            std::lock_guard<std::mutex> lock(registry_mu);
            s = find_or_view_locked(name);
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
        }
        if (db.lanes_per_row == 0 && sb.lanes_per_row != 0) {
            db.lanes_per_row = sb.lanes_per_row; db.n_pi = sb.n_pi; db.n_pf = sb.n_pf; db.n_ps = sb.n_ps;
            db.key_str = sb.key_str;
        } else if (db.lanes_per_row != 0 && sb.lanes_per_row != 0 &&
                   (db.n_pi != sb.n_pi || db.n_pf != sb.n_pf || db.n_ps != sb.n_ps || db.key_str != sb.key_str)) {
            duckdb_aggregate_function_set_error(info,
                "gpu_upload_rows_exact: combine merged two different predicate list lengths — "
                "the lists must have the same length on every row");
            return;
        }
        if (!sb.dicts.empty()) {
            bool collision = false;
            for (std::size_t slot = 0; slot < sb.dicts.size(); ++slot)
                for (const auto& kv : sb.dicts[slot])
                    db.note_string(slot, kv.first, kv.second.data(), kv.second.size(), collision);
            if (collision) {
                duckdb_aggregate_function_set_error(info,
                    "gpu_upload_rows_exact: two different strings hash alike (64-bit collision) — "
                    "this set cannot be resident; the statement runs native");
                return;
            }
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
        if (delta > 0 && !pool_reserve(delta, sb.exact || db.exact)) {
            duckdb_aggregate_function_set_error(info,
                (std::string("gpu_upload: out of buffer memory") + kPoolCapHint).c_str());
            return;
        }
        db.charged += delta;
        db.rows_seen += sb.rows_seen;
        for (const auto& v : sb.all_views()) db.views.push_back(v);
        if (sb.columns) {
            db.columns = true;
            db.unordered = db.unordered || sb.unordered;
            db.chunks.insert(db.chunks.end(), sb.chunks.begin(), sb.chunks.end());
        }
    }
}

// Build a set from a finished buffer and publish it. Everything expensive
// (H2D, prepare) runs outside every lock; publish takes the registry lock
// for a map insert. Throws std::runtime_error.
void publish_set(ResidentContext& ctx, UploadBuf& b,
                 std::unique_ptr<gpudb::ResidentColumn> keys,
                 std::unique_ptr<gpudb::ResidentColumn> vals, bool pair, const char* fn,
                 bool exact = false,
                 std::vector<std::unique_ptr<gpudb::ResidentColumn>> preds = {},
                 std::size_t pred_int = 0, std::size_t pred_dbl = 0, std::size_t pred_str = 0,
                 std::vector<std::pair<std::string, std::weak_ptr<ResidentSet>>> deps = {}) {
    auto set = std::make_shared<ResidentSet>();
    set->name = b.name;
    set->deps = std::move(deps);
    set->managed = b.managed;
    set->exact = exact;
    for (auto& p : preds) set->preds.emplace_back(std::move(p));
    set->pred_int = pred_int;
    set->pred_dbl = pred_dbl;
    set->pred_str = pred_str;
    set->key_str = b.key_str;
    if (b.key_str) set->key_dict = std::make_shared<const ResidentSet::Dict>(
        b.dicts.empty() ? ResidentSet::Dict{} : std::move(b.dicts[0]));
    for (std::size_t i = 0; i < pred_str; ++i) {
        const std::size_t slot = (b.key_str ? 1 : 0) + i;
        set->str_dicts.push_back(std::make_shared<const ResidentSet::Dict>(
            slot < b.dicts.size() ? std::move(b.dicts[slot]) : ResidentSet::Dict{}));
    }
    if (b.managed) {
        set->catalog = b.tag.catalog; set->schema = b.tag.schema; set->table = b.tag.table;
        set->table_oid = b.tag.table_oid; set->columns = b.tag.columns; set->extra = b.tag.extra;
    }
    set->pair = pair;
    set->rows = keys ? keys->rows() : 0;
    set->rows_seen = b.rows_seen;
    set->keys = std::move(keys);
    set->vals = std::move(vals);
    // §4.12: a set only global aggregates read writes its lane 0 as '-'. The
    // single-table form is a view over the store and simply has no key lane;
    // a set over a JOIN is uploaded or materialised lane by lane, so the key
    // slot arrives and is dropped HERE — `keys` points at the payload purely
    // so the set's invariants hold, nothing reads it as a key, and no sort
    // cache is ever built for it.
    if (set->managed && no_key_tag(b.tag) && set->vals) {
        set->no_key = true;
        set->key_str = false;
        set->key_dict = nullptr;
        set->rows = set->vals->rows();
        set->keys = set->vals;
    }
    set->uploaded_at_us = now_us();
    // Ready means uploaded AND prepared (§5.5). Managed sets are prepared
    // here so the first hit pays no sort; explicit sets keep the lazy build
    // (gpu_prepare_resident() exists for users who want it eager).
    const auto t_prep = std::chrono::steady_clock::now();
    if (set->managed && set->keys && !set->no_key) {
        set->keys->prepare();
    }
    if (upload_trace())
        std::fprintf(stderr, "[gpudb upload] %s '%s': rows=%zu prepare=%.1f ms\n",
                     fn, b.name.c_str(), set->rows, ms_since(t_prep));
    set->state.store(set->no_key || (set->keys && set->keys->prepared())
                     ? SetState::Ready : SetState::Uploaded);
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
std::size_t finish_upload_exact(ResidentContext& ctx, UploadBuf& b, const char* fn);

std::size_t finish_upload_columns(ResidentContext& ctx, UploadBuf& b, const char* fn);
std::size_t finish_upload(ResidentContext& ctx, UploadBuf& b, bool pair, gpudb::Dtype vdt,
                          const char* fn) {
    if (b.columns) return finish_upload_columns(ctx, b, fn);            // stage B store upload
    if (b.lanes_per_row != 0) return finish_upload_exact(ctx, b, fn);   // exact (§4.1 / §4.6) buffer
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
                    const char* fn, bool exact = false) {
    std::lock_guard<std::mutex> lock(ctx.registry_mu);
    auto it = ctx.sessions.find(b.name);
    if (it == ctx.sessions.end()) return false;
    UploadSession& ss = *it->second;
    if (ss.kind_set && (ss.pair != pair || ss.vdt != vdt || ss.exact != exact || ss.columns != b.columns ||
                        (exact && (ss.lanes_per_row != b.lanes_per_row || ss.n_pi != b.n_pi || ss.n_pf != b.n_pf ||
                                   ss.n_ps != b.n_ps || ss.key_str != b.key_str))))
        throw std::runtime_error(std::string(fn) + ": segment kind differs from the open upload "
            "session '" + b.name + "' (all segments must use the same upload function, types and "
            "predicate lists)");
    const std::size_t bytes = b.lanes() * sizeof(std::int64_t);
    if (bytes > 0 && !pool_reserve(bytes, exact))
        throw std::runtime_error(std::string(fn) + ": out of buffer memory for the upload "
            "session '" + b.name + "'" + kPoolCapHint);
    ss.kind_set = true; ss.pair = pair; ss.vdt = vdt; ss.exact = exact;
    if (exact) {
        ss.lanes_per_row = b.lanes_per_row; ss.n_pi = b.n_pi; ss.n_pf = b.n_pf; ss.n_ps = b.n_ps; ss.key_str = b.key_str;
        if (ss.dicts.size() < b.dicts.size()) ss.dicts.resize(b.dicts.size());
        for (std::size_t slot = 0; slot < b.dicts.size(); ++slot)
            for (const auto& kv : b.dicts[slot]) {
                auto it = ss.dicts[slot].find(kv.first);
                if (it == ss.dicts[slot].end()) ss.dicts[slot].emplace(kv.first, kv.second);
                else if (it->second != kv.second)
                    throw std::runtime_error(std::string(fn) + ": two different strings hash alike (64-bit collision) — "
                                             "this set cannot be resident; the statement runs native");
            }
    }
    ss.charged += bytes;
    ss.lanes += b.lanes();
    ss.rows_seen += b.rows_seen;
    ss.segments += 1;
    for (const auto& v : b.all_views()) ss.views.push_back(v);
    if (b.columns) {
        ss.columns = true;
        ss.unordered = ss.unordered || b.unordered;
        ss.chunks.insert(ss.chunks.end(), b.chunks.begin(), b.chunks.end());
    }
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
// gpu_upload_pair_exact aggregate (v0.7 milestone 3, §4.1) — the pair WITH
// its NULLs. Same interleaved segments as gpu_upload_pair; a NULL half is
// recorded in the segment's validity bitmaps (its lane holds 0) instead of
// being skipped, so count(*), the NULL-key group and NULL payloads survive
// to the exact GROUP BY. Backends partition NULL keys at upload (§4.1).
// ---------------------------------------------------------------------------

void upload_pair_exact_update(duckdb_function_info info, duckdb_data_chunk input,
                              duckdb_aggregate_state* states) {
    duckdb_vector name_vec = duckdb_data_chunk_get_vector(input, 0);
    duckdb_vector k_vec    = duckdb_data_chunk_get_vector(input, 1);
    duckdb_vector v_vec    = duckdb_data_chunk_get_vector(input, 2);
    auto* names = reinterpret_cast<duckdb_string_t*>(duckdb_vector_get_data(name_vec));
    const auto* kd = reinterpret_cast<const std::int64_t*>(duckdb_vector_get_data(k_vec));
    const auto* vd = reinterpret_cast<const std::int64_t*>(duckdb_vector_get_data(v_vec));
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
    if (!pool_reserve(reserved, /*exact*/true)) {
        duckdb_aggregate_function_set_error(info,
            (std::string("gpu_upload_pair_exact: out of buffer memory") + kPoolCapHint).c_str());
        return;
    }
    std::size_t used = 0;
    auto append_row = [&](UploadState* s, idx_t i) -> bool {
        UploadBuf& b = state_buf(s, gpudb::Dtype::I64);
        b.exact = true;
        ++b.rows_seen;
        if (name_validity && !duckdb_validity_row_is_valid(name_validity, i)) {
            duckdb_aggregate_function_set_error(info, "gpu_upload_pair_exact: name may not be NULL");
            return false;
        }
        const char*       nm_data = duckdb_string_t_data(&names[i]);
        const std::size_t nm_len  = duckdb_string_t_length(names[i]);
        if (!b.name_set) {
            if (!set_buf_name(info, b, nm_data, nm_len, "gpu_upload_pair_exact")) return false;
        } else if (b.name.size() != nm_len ||
                   std::memcmp(b.name.data(), nm_data, nm_len) != 0) {
            duckdb_aggregate_function_set_error(info,
                ("gpu_upload_pair_exact: one aggregate received two different names ('" +
                 b.name + "' and '" + std::string(nm_data, nm_len) +
                 "') — use a constant name").c_str());
            return false;
        }
        const bool k_null = k_validity && !duckdb_validity_row_is_valid(k_validity, i);
        const bool v_null = v_validity && !duckdb_validity_row_is_valid(v_validity, i);
        if (b.lanes_per_row == 0) { b.lanes_per_row = 2; b.n_pi = 0; b.n_pf = 0; }
        std::int64_t* dst = b.reserve_lanes(2);
        const std::size_t row = static_cast<std::size_t>(dst - b.open->data) / 2;
        dst[0] = k_null ? 0 : kd[i];
        dst[1] = v_null ? 0 : vd[i];
        if (k_null) b.open->mark_lane_null(row, 0, 2);
        if (v_null) b.open->mark_lane_null(row, 1, 2);
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

void upload_pair_exact_finalize(duckdb_function_info info, duckdb_aggregate_state* source,
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
            if (session_append(ctx, *b, /*pair*/true, gpudb::Dtype::I64, "gpu_upload_pair_exact", /*exact*/true)) {
                out[offset + i] = static_cast<std::int64_t>(lanes / 2);
                continue;
            }
            out[offset + i] = static_cast<std::int64_t>(finish_upload_exact(ctx, *b, "gpu_upload_pair_exact"));
        } catch (const std::exception& e) {
            duckdb_aggregate_function_set_error(info,
                (std::string("gpu_upload_pair_exact failed: ") + e.what()).c_str());
            return;
        }
    }
}

// ---------------------------------------------------------------------------
// gpu_upload_rows_exact(name, k BIGINT, v BIGINT, pi BIGINT[], pf DOUBLE[])
// (v0.7 §4.6) — an exact pair plus row-aligned predicate columns for the
// device-side WHERE mask. The two lists carry the integer-typed and the
// DOUBLE-typed predicate columns; their lengths are fixed by the first row
// (a different length later is an error, so are NULL lists). Rows are
// buffered as 2 + n_pi + n_pf lanes; NULL cells are recorded per lane in
// the segment's validity bitmaps. The set's columns are addressed by the
// WHERE program as k, v, i<n>, f<n>.
// ---------------------------------------------------------------------------

void upload_rows_exact_update(duckdb_function_info info, duckdb_data_chunk input,
                              duckdb_aggregate_state* states) {
    duckdb_vector name_vec = duckdb_data_chunk_get_vector(input, 0);
    duckdb_vector k_vec    = duckdb_data_chunk_get_vector(input, 1);
    duckdb_vector v_vec    = duckdb_data_chunk_get_vector(input, 2);
    duckdb_vector pi_vec   = duckdb_data_chunk_get_vector(input, 3);
    duckdb_vector pf_vec   = duckdb_data_chunk_get_vector(input, 4);
    auto* names = reinterpret_cast<duckdb_string_t*>(duckdb_vector_get_data(name_vec));
    const auto* kd = reinterpret_cast<const std::int64_t*>(duckdb_vector_get_data(k_vec));
    const auto* vd = reinterpret_cast<const std::int64_t*>(duckdb_vector_get_data(v_vec));
    const auto* pi_ent = reinterpret_cast<const duckdb_list_entry*>(duckdb_vector_get_data(pi_vec));
    const auto* pf_ent = reinterpret_cast<const duckdb_list_entry*>(duckdb_vector_get_data(pf_vec));
    const idx_t n = duckdb_data_chunk_get_size(input);
    if (!names || !kd || !vd || !pi_ent || !pf_ent || n == 0) return;

    duckdb_vector pi_child = duckdb_list_vector_get_child(pi_vec);
    duckdb_vector pf_child = duckdb_list_vector_get_child(pf_vec);
    const auto* pi_data = reinterpret_cast<const std::int64_t*>(duckdb_vector_get_data(pi_child));
    const auto* pf_data = reinterpret_cast<const std::int64_t*>(duckdb_vector_get_data(pf_child));  // raw double bits
    uint64_t* name_validity = duckdb_vector_get_validity(name_vec);
    uint64_t* k_validity    = duckdb_vector_get_validity(k_vec);
    uint64_t* v_validity    = duckdb_vector_get_validity(v_vec);
    uint64_t* pi_validity   = duckdb_vector_get_validity(pi_vec);
    uint64_t* pf_validity   = duckdb_vector_get_validity(pf_vec);
    uint64_t* pic_validity  = duckdb_vector_get_validity(pi_child);
    uint64_t* pfc_validity  = duckdb_vector_get_validity(pf_child);

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
    // Reserve for the widest row in this chunk (lists are usually uniform).
    std::size_t max_lanes = 2;
    for (idx_t i = 0; i < n; ++i) {
        const bool pi_ok = !pi_validity || duckdb_validity_row_is_valid(pi_validity, i);
        const bool pf_ok = !pf_validity || duckdb_validity_row_is_valid(pf_validity, i);
        const std::size_t L = 2 + (pi_ok ? pi_ent[i].length : 0) + (pf_ok ? pf_ent[i].length : 0);
        max_lanes = std::max(max_lanes, L);
    }
    const std::size_t reserved = static_cast<std::size_t>(n) * max_lanes * sizeof(std::int64_t);
    if (!pool_reserve(reserved, /*exact*/true)) {
        duckdb_aggregate_function_set_error(info,
            (std::string("gpu_upload_rows_exact: out of buffer memory") + kPoolCapHint).c_str());
        return;
    }
    std::size_t used = 0;
    auto append_row = [&](UploadState* s, idx_t i) -> bool {
        UploadBuf& b = state_buf(s, gpudb::Dtype::I64);
        b.exact = true;
        ++b.rows_seen;
        if (name_validity && !duckdb_validity_row_is_valid(name_validity, i)) {
            duckdb_aggregate_function_set_error(info, "gpu_upload_rows_exact: name may not be NULL");
            return false;
        }
        if ((pi_validity && !duckdb_validity_row_is_valid(pi_validity, i)) ||
            (pf_validity && !duckdb_validity_row_is_valid(pf_validity, i))) {
            duckdb_aggregate_function_set_error(info,
                "gpu_upload_rows_exact: the predicate lists may not be NULL (use [] for none)");
            return false;
        }
        const char*       nm_data = duckdb_string_t_data(&names[i]);
        const std::size_t nm_len  = duckdb_string_t_length(names[i]);
        if (!b.name_set) {
            if (!set_buf_name(info, b, nm_data, nm_len, "gpu_upload_rows_exact")) return false;
        } else if (b.name.size() != nm_len ||
                   std::memcmp(b.name.data(), nm_data, nm_len) != 0) {
            duckdb_aggregate_function_set_error(info,
                ("gpu_upload_rows_exact: one aggregate received two different names ('" +
                 b.name + "' and '" + std::string(nm_data, nm_len) +
                 "') — use a constant name").c_str());
            return false;
        }
        const std::size_t n_pi = pi_ent[i].length, n_pf = pf_ent[i].length;
        if (b.lanes_per_row == 0) {
            b.n_pi = n_pi; b.n_pf = n_pf; b.lanes_per_row = 2 + n_pi + n_pf;
        } else if (b.n_pi != n_pi || b.n_pf != n_pf) {
            duckdb_aggregate_function_set_error(info,
                ("gpu_upload_rows_exact: predicate list length changed between rows (" +
                 std::to_string(b.n_pi) + "/" + std::to_string(b.n_pf) + " then " +
                 std::to_string(n_pi) + "/" + std::to_string(n_pf) +
                 ") — every row must carry the same columns").c_str());
            return false;
        }
        const std::size_t L = b.lanes_per_row;
        std::int64_t* dst = b.reserve_lanes(L);
        const std::size_t row = static_cast<std::size_t>(dst - b.open->data) / L;
        const bool k_null = k_validity && !duckdb_validity_row_is_valid(k_validity, i);
        const bool v_null = v_validity && !duckdb_validity_row_is_valid(v_validity, i);
        dst[0] = k_null ? 0 : kd[i];
        dst[1] = v_null ? 0 : vd[i];
        if (k_null) b.open->mark_lane_null(row, 0, L);
        if (v_null) b.open->mark_lane_null(row, 1, L);
        for (std::size_t e = 0; e < n_pi; ++e) {
            const idx_t c = pi_ent[i].offset + e;
            const bool nul = pic_validity && !duckdb_validity_row_is_valid(pic_validity, c);
            dst[2 + e] = nul ? 0 : pi_data[c];
            if (nul) b.open->mark_lane_null(row, 2 + e, L);
        }
        for (std::size_t e = 0; e < n_pf; ++e) {
            const idx_t c = pf_ent[i].offset + e;
            const bool nul = pfc_validity && !duckdb_validity_row_is_valid(pfc_validity, c);
            dst[2 + n_pi + e] = nul ? 0 : pf_data[c];
            if (nul) b.open->mark_lane_null(row, 2 + n_pi + e, L);
        }
        b.charged += L * sizeof(std::int64_t);
        used += L * sizeof(std::int64_t);
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
    pool_release(reserved - std::min(reserved, used));
}

// Move an exact buffer (gpu_upload_pair_exact: 2 lanes; gpu_upload_rows_exact:
// 2 + n_pi + n_pf lanes) onto the device through upload_rows_exact and publish
// it as an exact set (key, payload, predicate columns). Returns the rows
// resident (NULLs included). Throws std::runtime_error.
std::size_t finish_upload_exact(ResidentContext& ctx, UploadBuf& b, const char* fn) {
    const std::size_t L = b.lanes_per_row;
    const std::size_t lanes = b.lanes();
    const std::size_t rows = lanes / L;
    const auto t0 = std::chrono::steady_clock::now();
    auto& a = ctx.aggregator();
    std::vector<gpudb::Dtype> dtypes(L, gpudb::Dtype::I64);        // string lanes: I64 hashes
    for (std::size_t e = 0; e < b.n_pf; ++e) dtypes[2 + b.n_pi + e] = gpudb::Dtype::F64;
    const auto views = b.all_views();
    std::vector<gpudb::Aggregator::RowSpan> spans(views.size());
    std::vector<std::vector<const std::uint64_t*>> valid_ptrs(views.size());
    for (std::size_t s2 = 0; s2 < views.size(); ++s2) {
        const auto& v = views[s2];
        spans[s2].lanes = v.seg->data;
        spans[s2].rows = v.lanes / L;
        spans[s2].n_lanes = L;
        valid_ptrs[s2].assign(L, nullptr);
        if (!v.seg->lane_valid.empty())
            for (std::size_t l = 0; l < L && l < v.seg->lane_valid.size(); ++l)
                if (!v.seg->lane_valid[l].empty()) valid_ptrs[s2][l] = v.seg->lane_valid[l].data();
        spans[s2].valid = valid_ptrs[s2].data();
    }
    std::vector<std::unique_ptr<gpudb::ResidentColumn>> cols;
    {
        // Lane 0 of a join result is that statement's GROUP BY key and is
        // read as nothing else, so the backend may release it once it holds
        // the key's group-id lane (§9). A no-key set has no key at all.
        const bool key_only = b.managed && join_result_tag(b.tag) && !no_key_tag(b.tag);
        gpudb::KeyOnlyLanes note(key_only ? std::uint64_t{1} : std::uint64_t{0});
        cols = a.upload_rows_exact(spans.data(), spans.size(), dtypes.data(), L);
    }
    if (cols.size() != L) throw std::runtime_error("upload_rows_exact returned the wrong column count");
    auto kcol = std::move(cols[0]);
    auto vcol = std::move(cols[1]);
    std::vector<std::unique_ptr<gpudb::ResidentColumn>> preds;
    for (std::size_t l = 2; l < L; ++l) preds.push_back(std::move(cols[l]));
    if (upload_trace())
        std::fprintf(stderr, "[gpudb upload] %s '%s': rows=%zu lanes=%zu null_keys=%zu null_vals=%zu "
                     "segments=%zu upload=%.1f ms\n",
                     fn, b.name.c_str(), rows, L, kcol->null_count(), vcol->null_count(),
                     spans.size(), ms_since(t0));
    publish_set(ctx, b, std::move(kcol), std::move(vcol), /*pair*/true, fn,
                /*exact*/true, std::move(preds), b.n_pi, b.n_pf, b.n_ps);
    return rows;
}

// gpu_upload_rows_exact(name, k BIGINT|VARCHAR, v BIGINT, pi BIGINT[], pf DOUBLE[], ps VARCHAR[])
// A VARCHAR key is the wrapper's tuple text, stored as hash64(text) with the
// text kept in the buffer's dictionary; VARCHAR predicate lanes store
// hash64(raw text), each with its own dictionary. Lane order: key, payload,
// pi..., pf..., ps...
template <bool KEY_STR>
void upload_rows_exact_update6(duckdb_function_info info, duckdb_data_chunk input,
                               duckdb_aggregate_state* states) {
    duckdb_vector name_vec = duckdb_data_chunk_get_vector(input, 0);
    duckdb_vector k_vec    = duckdb_data_chunk_get_vector(input, 1);
    duckdb_vector v_vec    = duckdb_data_chunk_get_vector(input, 2);
    duckdb_vector pi_vec   = duckdb_data_chunk_get_vector(input, 3);
    duckdb_vector pf_vec   = duckdb_data_chunk_get_vector(input, 4);
    duckdb_vector ps_vec   = duckdb_data_chunk_get_vector(input, 5);
    auto* names = reinterpret_cast<duckdb_string_t*>(duckdb_vector_get_data(name_vec));
    const auto* kd = reinterpret_cast<const std::int64_t*>(duckdb_vector_get_data(k_vec));
    auto* ks = reinterpret_cast<duckdb_string_t*>(duckdb_vector_get_data(k_vec));
    const auto* vd = reinterpret_cast<const std::int64_t*>(duckdb_vector_get_data(v_vec));
    const auto* pi_ent = reinterpret_cast<const duckdb_list_entry*>(duckdb_vector_get_data(pi_vec));
    const auto* pf_ent = reinterpret_cast<const duckdb_list_entry*>(duckdb_vector_get_data(pf_vec));
    const auto* ps_ent = reinterpret_cast<const duckdb_list_entry*>(duckdb_vector_get_data(ps_vec));
    const idx_t n = duckdb_data_chunk_get_size(input);
    if (!names || (!kd && !ks) || !vd || !pi_ent || !pf_ent || !ps_ent || n == 0) return;

    duckdb_vector pi_child = duckdb_list_vector_get_child(pi_vec);
    duckdb_vector pf_child = duckdb_list_vector_get_child(pf_vec);
    duckdb_vector ps_child = duckdb_list_vector_get_child(ps_vec);
    const auto* pi_data = reinterpret_cast<const std::int64_t*>(duckdb_vector_get_data(pi_child));
    const auto* pf_data = reinterpret_cast<const std::int64_t*>(duckdb_vector_get_data(pf_child));
    auto* ps_data = reinterpret_cast<duckdb_string_t*>(duckdb_vector_get_data(ps_child));
    uint64_t* name_validity = duckdb_vector_get_validity(name_vec);
    uint64_t* k_validity    = duckdb_vector_get_validity(k_vec);
    uint64_t* v_validity    = duckdb_vector_get_validity(v_vec);
    uint64_t* pi_validity   = duckdb_vector_get_validity(pi_vec);
    uint64_t* pf_validity   = duckdb_vector_get_validity(pf_vec);
    uint64_t* ps_validity   = duckdb_vector_get_validity(ps_vec);
    uint64_t* pic_validity  = duckdb_vector_get_validity(pi_child);
    uint64_t* pfc_validity  = duckdb_vector_get_validity(pf_child);
    uint64_t* psc_validity  = duckdb_vector_get_validity(ps_child);

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
    std::size_t max_lanes = 2;
    for (idx_t i = 0; i < n; ++i) {
        const bool pi_ok = !pi_validity || duckdb_validity_row_is_valid(pi_validity, i);
        const bool pf_ok = !pf_validity || duckdb_validity_row_is_valid(pf_validity, i);
        const bool ps_ok = !ps_validity || duckdb_validity_row_is_valid(ps_validity, i);
        const std::size_t L = std::size_t(2) + (pi_ok ? static_cast<std::size_t>(pi_ent[i].length) : 0) +
                              (pf_ok ? static_cast<std::size_t>(pf_ent[i].length) : 0) +
                              (ps_ok ? static_cast<std::size_t>(ps_ent[i].length) : 0);
        max_lanes = std::max(max_lanes, L);
    }
    const std::size_t reserved = static_cast<std::size_t>(n) * max_lanes * sizeof(std::int64_t);
    if (!pool_reserve(reserved, /*exact*/true)) {
        duckdb_aggregate_function_set_error(info,
            (std::string("gpu_upload_rows_exact: out of buffer memory") + kPoolCapHint).c_str());
        return;
    }
    std::size_t used = 0;
    auto append_row = [&](UploadState* s, idx_t i) -> bool {
        UploadBuf& b = state_buf(s, gpudb::Dtype::I64);
        b.exact = true;
        ++b.rows_seen;
        if (name_validity && !duckdb_validity_row_is_valid(name_validity, i)) {
            duckdb_aggregate_function_set_error(info, "gpu_upload_rows_exact: name may not be NULL");
            return false;
        }
        if ((pi_validity && !duckdb_validity_row_is_valid(pi_validity, i)) ||
            (pf_validity && !duckdb_validity_row_is_valid(pf_validity, i)) ||
            (ps_validity && !duckdb_validity_row_is_valid(ps_validity, i))) {
            duckdb_aggregate_function_set_error(info,
                "gpu_upload_rows_exact: the predicate lists may not be NULL (use [] for none)");
            return false;
        }
        const char*       nm_data = duckdb_string_t_data(&names[i]);
        const std::size_t nm_len  = duckdb_string_t_length(names[i]);
        if (!b.name_set) {
            if (!set_buf_name(info, b, nm_data, nm_len, "gpu_upload_rows_exact")) return false;
        } else if (b.name.size() != nm_len ||
                   std::memcmp(b.name.data(), nm_data, nm_len) != 0) {
            duckdb_aggregate_function_set_error(info,
                ("gpu_upload_rows_exact: one aggregate received two different names ('" +
                 b.name + "' and '" + std::string(nm_data, nm_len) +
                 "') — use a constant name").c_str());
            return false;
        }
        const std::size_t n_pi = pi_ent[i].length, n_pf = pf_ent[i].length, n_ps = ps_ent[i].length;
        if (b.lanes_per_row == 0) {
            b.n_pi = n_pi; b.n_pf = n_pf; b.n_ps = n_ps; b.lanes_per_row = 2 + n_pi + n_pf + n_ps;
            b.key_str = KEY_STR;
        } else if (b.n_pi != n_pi || b.n_pf != n_pf || b.n_ps != n_ps) {
            duckdb_aggregate_function_set_error(info,
                "gpu_upload_rows_exact: predicate list length changed between rows — every row must carry the same columns");
            return false;
        }
        const std::size_t L = b.lanes_per_row;
        std::int64_t* dst = b.reserve_lanes(L);
        const std::size_t row = static_cast<std::size_t>(dst - b.open->data) / L;
        const bool k_null = k_validity && !duckdb_validity_row_is_valid(k_validity, i);
        const bool v_null = v_validity && !duckdb_validity_row_is_valid(v_validity, i);
        bool collision = false;
        if (KEY_STR) {
            if (k_null) dst[0] = 0;
            else {
                const char* kp = duckdb_string_t_data(&ks[i]);
                const std::size_t kn = duckdb_string_t_length(ks[i]);
                const std::uint64_t h = hash64(kp, kn);
                dst[0] = static_cast<std::int64_t>(h);
                b.note_string(0, h, kp, kn, collision);
            }
        } else {
            dst[0] = k_null ? 0 : kd[i];
        }
        dst[1] = v_null ? 0 : vd[i];
        if (k_null) b.open->mark_lane_null(row, 0, L);
        if (v_null) b.open->mark_lane_null(row, 1, L);
        for (std::size_t e = 0; e < n_pi; ++e) {
            const idx_t c = pi_ent[i].offset + e;
            const bool nul = pic_validity && !duckdb_validity_row_is_valid(pic_validity, c);
            dst[2 + e] = nul ? 0 : pi_data[c];
            if (nul) b.open->mark_lane_null(row, 2 + e, L);
        }
        for (std::size_t e = 0; e < n_pf; ++e) {
            const idx_t c = pf_ent[i].offset + e;
            const bool nul = pfc_validity && !duckdb_validity_row_is_valid(pfc_validity, c);
            dst[2 + n_pi + e] = nul ? 0 : pf_data[c];
            if (nul) b.open->mark_lane_null(row, 2 + n_pi + e, L);
        }
        for (std::size_t e = 0; e < n_ps; ++e) {
            const idx_t c = ps_ent[i].offset + e;
            const std::size_t lane = 2 + n_pi + n_pf + e;
            const bool nul = psc_validity && !duckdb_validity_row_is_valid(psc_validity, c);
            if (nul) { dst[lane] = 0; b.open->mark_lane_null(row, lane, L); continue; }
            const char* sp = duckdb_string_t_data(&ps_data[c]);
            const std::size_t sn = duckdb_string_t_length(ps_data[c]);
            const std::uint64_t h = hash64(sp, sn);
            dst[lane] = static_cast<std::int64_t>(h);
            b.note_string((KEY_STR ? 1 : 0) + e, h, sp, sn, collision);
        }
        if (collision) {
            duckdb_aggregate_function_set_error(info,
                "gpu_upload_rows_exact: two different strings hash alike (64-bit collision) — "
                "this set cannot be resident; the statement runs native");
            return false;
        }
        b.charged += L * sizeof(std::int64_t);
        used += L * sizeof(std::int64_t);
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
    pool_release(reserved - std::min(reserved, used));
}

void upload_rows_exact_finalize(duckdb_function_info info, duckdb_aggregate_state* source,
                                duckdb_vector result, idx_t count, idx_t offset) {
    if (count == 0) return;
    auto* out = reinterpret_cast<std::int64_t*>(duckdb_vector_get_data(result));
    duckdb_vector_ensure_validity_writable(result);
    uint64_t* validity = duckdb_vector_get_validity(result);

    for (idx_t i = 0; i < count; ++i) {
        UploadState* s = probe_upload_state(source[i]);
        UploadBuf* b = (s && s->buf_id != 0) ? s->buf : nullptr;
        const std::size_t lanes = b ? b->lanes() : 0;
        if (!b || lanes == 0 || b->lanes_per_row == 0) {
            out[offset + i] = 0;
            duckdb_validity_set_row_invalid(validity, offset + i);
            continue;
        }
        try {
            ResidentContext& ctx = ctx_of_aggregate(info);
            if (session_append(ctx, *b, /*pair*/true, gpudb::Dtype::I64, "gpu_upload_rows_exact", /*exact*/true)) {
                out[offset + i] = static_cast<std::int64_t>(lanes / b->lanes_per_row);
                continue;
            }
            out[offset + i] = static_cast<std::int64_t>(finish_upload_exact(ctx, *b, "gpu_upload_rows_exact"));
        } catch (const std::exception& e) {
            duckdb_aggregate_function_set_error(info,
                (std::string("gpu_upload_rows_exact failed: ") + e.what()).c_str());
            return;
        }
    }
}

// ---------------------------------------------------------------------------
// Upload sessions: gpu_upload_begin / gpu_upload_finish / gpu_upload_abort /
// gpu_upload_status.
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// Stage B (docs/RESIDENT_COLUMNS_DESIGN.md): gpu_upload_columns.
//   gpu_upload_columns(tag, rowid BIGINT, ci BIGINT[], cf DOUBLE[], cs VARCHAR[]) -> BIGINT
// The tag names a TABLE STORE and the lanes in upload order:
//   gpudb:v1:<catalog>:<schema>:<table>:<oid>:<ci names>,<cf names>,<cs names>:store
// Lane 0 of the buffer is the row id. Rows of ONE update call of a plain scan
// arrive in ascending row-id order (a DuckDB scan hands a chunk of one row
// group in storage order), so every call's rows are remembered as a chunk with
// the row ids it spans; finish orders the chunks by first row id and places
// each at its rank (RowSpan::dst_row) — the columns end up in row-id order
// without a sort of the data. A scan that delivers rows in another order (a
// join under a correlated subquery) is placed by row-id rank at finish
// instead. Either way every column of the store is row-aligned with every
// other whatever scan produced it.
// ---------------------------------------------------------------------------
std::atomic<std::uint64_t> g_upload_call { 0 };

void upload_columns_update(duckdb_function_info info, duckdb_data_chunk input,
                           duckdb_aggregate_state* states) {
    duckdb_vector name_vec = duckdb_data_chunk_get_vector(input, 0);
    duckdb_vector r_vec    = duckdb_data_chunk_get_vector(input, 1);
    duckdb_vector ci_vec   = duckdb_data_chunk_get_vector(input, 2);
    duckdb_vector cf_vec   = duckdb_data_chunk_get_vector(input, 3);
    duckdb_vector cs_vec   = duckdb_data_chunk_get_vector(input, 4);
    auto* names = reinterpret_cast<duckdb_string_t*>(duckdb_vector_get_data(name_vec));
    const auto* rd = reinterpret_cast<const std::int64_t*>(duckdb_vector_get_data(r_vec));
    const auto* ci_ent = reinterpret_cast<const duckdb_list_entry*>(duckdb_vector_get_data(ci_vec));
    const auto* cf_ent = reinterpret_cast<const duckdb_list_entry*>(duckdb_vector_get_data(cf_vec));
    const auto* cs_ent = reinterpret_cast<const duckdb_list_entry*>(duckdb_vector_get_data(cs_vec));
    const idx_t n = duckdb_data_chunk_get_size(input);
    if (!names || !rd || !ci_ent || !cf_ent || !cs_ent || n == 0) return;
    duckdb_vector ci_child = duckdb_list_vector_get_child(ci_vec);
    duckdb_vector cf_child = duckdb_list_vector_get_child(cf_vec);
    duckdb_vector cs_child = duckdb_list_vector_get_child(cs_vec);
    const auto* ci_data = reinterpret_cast<const std::int64_t*>(duckdb_vector_get_data(ci_child));
    const auto* cf_data = reinterpret_cast<const std::int64_t*>(duckdb_vector_get_data(cf_child));
    auto* cs_data = reinterpret_cast<duckdb_string_t*>(duckdb_vector_get_data(cs_child));
    uint64_t* name_validity = duckdb_vector_get_validity(name_vec);
    uint64_t* r_validity    = duckdb_vector_get_validity(r_vec);
    uint64_t* ci_validity   = duckdb_vector_get_validity(ci_vec);
    uint64_t* cf_validity   = duckdb_vector_get_validity(cf_vec);
    uint64_t* cs_validity   = duckdb_vector_get_validity(cs_vec);
    uint64_t* cic_validity  = duckdb_vector_get_validity(ci_child);
    uint64_t* cfc_validity  = duckdb_vector_get_validity(cf_child);
    uint64_t* csc_validity  = duckdb_vector_get_validity(cs_child);

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
    std::size_t max_lanes = 1;
    for (idx_t i = 0; i < n; ++i) {
        const bool ci_ok = !ci_validity || duckdb_validity_row_is_valid(ci_validity, i);
        const bool cf_ok = !cf_validity || duckdb_validity_row_is_valid(cf_validity, i);
        const bool cs_ok = !cs_validity || duckdb_validity_row_is_valid(cs_validity, i);
        const std::size_t L = 1 + (ci_ok ? static_cast<std::size_t>(ci_ent[i].length) : 0) +
                              (cf_ok ? static_cast<std::size_t>(cf_ent[i].length) : 0) +
                              (cs_ok ? static_cast<std::size_t>(cs_ent[i].length) : 0);
        max_lanes = std::max(max_lanes, L);
    }
    const std::size_t reserved = static_cast<std::size_t>(n) * max_lanes * sizeof(std::int64_t);
    if (!pool_reserve(reserved, /*exact*/true)) {
        duckdb_aggregate_function_set_error(info,
            (std::string("gpu_upload_columns: out of buffer memory") + kPoolCapHint).c_str());
        return;
    }
    const std::uint64_t call = g_upload_call.fetch_add(1, std::memory_order_relaxed) + 1;
    std::size_t used = 0;
    auto append_row = [&](UploadState* s, idx_t i) -> bool {
        UploadBuf& b = state_buf(s, gpudb::Dtype::I64);
        b.exact = true; b.columns = true;
        ++b.rows_seen;
        if (name_validity && !duckdb_validity_row_is_valid(name_validity, i)) {
            duckdb_aggregate_function_set_error(info, "gpu_upload_columns: name may not be NULL");
            return false;
        }
        if (r_validity && !duckdb_validity_row_is_valid(r_validity, i)) {
            duckdb_aggregate_function_set_error(info, "gpu_upload_columns: the row id may not be NULL");
            return false;
        }
        if ((ci_validity && !duckdb_validity_row_is_valid(ci_validity, i)) ||
            (cf_validity && !duckdb_validity_row_is_valid(cf_validity, i)) ||
            (cs_validity && !duckdb_validity_row_is_valid(cs_validity, i))) {
            duckdb_aggregate_function_set_error(info,
                "gpu_upload_columns: the column lists may not be NULL (use [] for none)");
            return false;
        }
        const char*       nm_data = duckdb_string_t_data(&names[i]);
        const std::size_t nm_len  = duckdb_string_t_length(names[i]);
        if (!b.name_set) {
            if (!set_buf_name(info, b, nm_data, nm_len, "gpu_upload_columns")) return false;
            if (!b.managed || b.tag.extra != "store") {
                duckdb_aggregate_function_set_error(info,
                    "gpu_upload_columns: the name must be a store tag (gpudb:v1:<catalog>:<schema>:<table>:<oid>:<lanes>:store)");
                return false;
            }
        } else if (b.name.size() != nm_len || std::memcmp(b.name.data(), nm_data, nm_len) != 0) {
            duckdb_aggregate_function_set_error(info,
                "gpu_upload_columns: one aggregate received two different names — use a constant name");
            return false;
        }
        const std::size_t n_ci = ci_ent[i].length, n_cf = cf_ent[i].length, n_cs = cs_ent[i].length;
        if (b.lanes_per_row == 0) {
            b.n_pi = n_ci; b.n_pf = n_cf; b.n_ps = n_cs; b.lanes_per_row = 1 + n_ci + n_cf + n_cs;
        } else if (b.n_pi != n_ci || b.n_pf != n_cf || b.n_ps != n_cs) {
            duckdb_aggregate_function_set_error(info,
                "gpu_upload_columns: list length changed between rows — every row must carry the same columns");
            return false;
        }
        const std::size_t L = b.lanes_per_row;
        std::int64_t* dst = b.reserve_lanes(L);
        const std::size_t row = static_cast<std::size_t>(dst - b.open->data) / L;
        const std::int64_t rowid = rd[i];
        // chunk bookkeeping: a new update call or a new segment opens a chunk
        if (b.chunks.empty() || b.chunk_call != call || b.chunks.back().seg.get() != b.open.get()) {
            b.chunks.push_back(UploadBuf::ChunkRec{rowid, rowid, b.open, row, 0});
            b.chunk_call = call;
        } else if (rowid <= b.chunks.back().last_rowid) {
            b.unordered = true;
        }
        b.chunks.back().last_rowid = rowid;
        b.chunks.back().rows += 1;
        dst[0] = rowid;
        bool collision = false;
        for (std::size_t e = 0; e < n_ci; ++e) {
            const idx_t c = ci_ent[i].offset + e;
            const bool nul = cic_validity && !duckdb_validity_row_is_valid(cic_validity, c);
            dst[1 + e] = nul ? 0 : ci_data[c];
            if (nul) b.open->mark_lane_null(row, 1 + e, L);
        }
        for (std::size_t e = 0; e < n_cf; ++e) {
            const idx_t c = cf_ent[i].offset + e;
            const bool nul = cfc_validity && !duckdb_validity_row_is_valid(cfc_validity, c);
            dst[1 + n_ci + e] = nul ? 0 : cf_data[c];
            if (nul) b.open->mark_lane_null(row, 1 + n_ci + e, L);
        }
        for (std::size_t e = 0; e < n_cs; ++e) {
            const idx_t c = cs_ent[i].offset + e;
            const std::size_t lane = 1 + n_ci + n_cf + e;
            const bool nul = csc_validity && !duckdb_validity_row_is_valid(csc_validity, c);
            if (nul) { dst[lane] = 0; b.open->mark_lane_null(row, lane, L); continue; }
            const char* sp = duckdb_string_t_data(&cs_data[c]);
            const std::size_t sn = duckdb_string_t_length(cs_data[c]);
            const std::uint64_t h = hash64(sp, sn);
            dst[lane] = static_cast<std::int64_t>(h);
            b.note_string(e, h, sp, sn, collision);
        }
        if (collision) {
            duckdb_aggregate_function_set_error(info,
                "gpu_upload_columns: two different strings hash alike (64-bit collision) — "
                "this column cannot be resident; the statement runs native");
            return false;
        }
        b.charged += L * sizeof(std::int64_t);
        used += L * sizeof(std::int64_t);
        return true;
    };
    bool ok = true;
    if (n == 1 || !per_row) {
        for (idx_t i = 0; i < n && ok; ++i) ok = append_row(s0, i);
    } else {
        for (idx_t i = 0; i < n && ok; ++i) {
            UploadState* st = probe_upload_state(states[i]);
            if (st) ok = append_row(st, i);
        }
    }
    pool_release(reserved - std::min(reserved, used));
}

// The columns of a finished column upload go into the table's store in row-id
// order. A store that exists with another row count (the table changed) is
// replaced and its views marked stale; an upload begun before an invalidation
// of the table is discarded like any set upload.
std::size_t finish_upload_columns(ResidentContext& ctx, UploadBuf& b, const char* fn) {
    const auto t0 = std::chrono::steady_clock::now();
    const std::size_t L = b.lanes_per_row;
    if (L < 2) throw std::runtime_error(std::string(fn) + ": no columns to upload");
    std::vector<std::string> names;
    for (std::size_t a = 0, e; a <= b.tag.columns.size(); a = e + 1) {
        e = b.tag.columns.find(',', a);
        if (e == std::string::npos) e = b.tag.columns.size();
        names.push_back(b.tag.columns.substr(a, e - a));
    }
    if (names.size() != L - 1)
        throw std::runtime_error(std::string(fn) + ": the tag names " + std::to_string(names.size()) +
                                 " lanes but " + std::to_string(L - 1) + " were uploaded");
    std::vector<UploadBuf::ChunkRec> chunks = b.chunks;
    // Fast path unless a chunk's rows came back out of order, or two chunks
    // cover the same row ids (both mean a scan that is not a plain table scan).
    bool placed = b.unordered;
    if (!placed) {
        std::sort(chunks.begin(), chunks.end(),
                  [](const UploadBuf::ChunkRec& x, const UploadBuf::ChunkRec& y) { return x.first_rowid < y.first_rowid; });
        for (std::size_t c = 1; c < chunks.size() && !placed; ++c)
            if (chunks[c].first_rowid <= chunks[c - 1].last_rowid) placed = true;
    }
    std::size_t rows = 0;
    for (const auto& ch : chunks) rows += ch.rows;
    if (rows != b.rows_seen) throw std::runtime_error(std::string(fn) + ": chunk bookkeeping lost rows");

    std::vector<gpudb::Aggregator::RowSpan> spans;
    std::vector<std::vector<const std::uint64_t*>> valid_ptrs;
    std::vector<std::shared_ptr<Segment>> placed_segs;   // alive until upload_rows_exact has read them
    PoolCharge charge;                                   // the placed path's temporaries
    double order_ms = 0.0;
    if (!placed) {
        spans.resize(chunks.size());
        valid_ptrs.resize(chunks.size());
        std::size_t at = 0;
        for (std::size_t c = 0; c < chunks.size(); ++c) {
            const auto& ch = chunks[c];
            spans[c].lanes = ch.seg->data + ch.row0 * L;
            spans[c].rows = ch.rows;
            spans[c].n_lanes = L;
            spans[c].dst_row = at;
            spans[c].valid_bit = ch.row0;
            valid_ptrs[c].assign(L, nullptr);
            if (!ch.seg->lane_valid.empty())
                for (std::size_t l = 0; l < L && l < ch.seg->lane_valid.size(); ++l)
                    if (!ch.seg->lane_valid[l].empty()) valid_ptrs[c][l] = ch.seg->lane_valid[l].data();
            spans[c].valid = valid_ptrs[c].data();
            at += ch.rows;
        }
    } else {
        // ---- placed path ----
        // Rank every row by its row id, then gather the rows into fresh
        // segments in rank order. Peak host memory is 2x the upload: the
        // source segments and the placed copy are both live until
        // upload_rows_exact has read the copy (plus, while the rank is being
        // computed, ~24 bytes per row of index).
        const auto t_ord = std::chrono::steady_clock::now();
        if (rows >= (std::size_t(1) << 32))
            throw std::runtime_error(std::string(fn) + ": " + std::to_string(rows) +
                                     " rows is above the placed path's limit of 2^32 rows");
        // A row's source is (segment index << 32) | row within segment; a chunk
        // lives in exactly one segment and a segment holds at most 2^20 rows.
        std::vector<Segment*> segs;
        std::vector<std::uint64_t> chunk_sid(chunks.size(), 0);
        {
            std::unordered_map<const Segment*, std::uint32_t> seg_id;
            for (std::size_t c = 0; c < chunks.size(); ++c) {
                auto it = seg_id.find(chunks[c].seg.get());
                if (it == seg_id.end()) {
                    it = seg_id.emplace(chunks[c].seg.get(), static_cast<std::uint32_t>(segs.size())).first;
                    segs.push_back(chunks[c].seg.get());
                }
                chunk_sid[c] = std::uint64_t(it->second) << 32;
            }
        }
        std::vector<std::size_t> base(chunks.size() + 1, 0);
        for (std::size_t c = 0; c < chunks.size(); ++c) base[c + 1] = base[c] + chunks[c].rows;
        const unsigned hw = std::thread::hardware_concurrency();
        const std::size_t workers =
            std::max<std::size_t>(1, std::min<std::size_t>(hw ? hw : 1, rows / 65536 + 1));
        struct RowKey { std::int64_t rowid; std::uint64_t src; };
        std::vector<RowKey>        keys;
        // dense: row id - min -> row index + 1, 0 = no row; atomic because two rows with one
        // id (a broken upload) store to the same slot from two threads — relaxed stores cost
        // nothing and keep the race defined, the verify pass then reports the duplicate
        std::unique_ptr<std::atomic<std::uint32_t>[]> pos;
        std::vector<std::uint64_t> order;    // order[rank] = source
        std::size_t idx_bytes = rows * (sizeof(RowKey) + sizeof(std::uint64_t));
        if (!charge.take(idx_bytes, /*exact*/true))
            throw std::runtime_error(std::string(fn) + ": out of buffer memory ordering the rows by row id" +
                                     kPoolCapHint);
        keys.resize(rows);
        order.resize(rows);
        // Pass 1: (row id, source) per row, and the row-id range.
        std::vector<std::int64_t> lo(workers, INT64_MAX), hi(workers, INT64_MIN);
        {
            std::vector<std::thread> ts;
            const std::size_t per = (chunks.size() + workers - 1) / workers;
            for (std::size_t w = 0; w < workers; ++w) {
                const std::size_t cb = std::min(chunks.size(), w * per), ce = std::min(chunks.size(), cb + per);
                ts.emplace_back([&, w, cb, ce] {
                    std::int64_t mn = INT64_MAX, mx = INT64_MIN;
                    for (std::size_t c = cb; c < ce; ++c) {
                        const auto& ch = chunks[c];
                        const std::int64_t* p = ch.seg->data + ch.row0 * L;
                        for (std::size_t r = 0; r < ch.rows; ++r) {
                            const std::int64_t id = p[r * L];
                            keys[base[c] + r] = RowKey{ id, chunk_sid[c] | std::uint64_t(ch.row0 + r) };
                            if (id < mn) mn = id;
                            if (id > mx) mx = id;
                        }
                    }
                    lo[w] = mn; hi[w] = mx;
                });
            }
            for (auto& t : ts) t.join();
        }
        std::int64_t mn = INT64_MAX, mx = INT64_MIN;
        for (std::size_t w = 0; w < workers; ++w) { mn = std::min(mn, lo[w]); mx = std::max(mx, hi[w]); }
        const std::uint64_t span = std::uint64_t(mx) - std::uint64_t(mn) + 1;
        std::vector<char>         dup_hit(workers, 0);
        std::vector<std::int64_t> dup_id(workers, 0);
        if (span <= std::uint64_t(rows) * 2 + 4096) {
            // Dense row ids (a full scan of a table): one slot per id, no sort.
            const std::size_t dense_bytes = std::size_t(span) * sizeof(std::uint32_t);
            if (!charge.take(dense_bytes, /*exact*/true))
                throw std::runtime_error(std::string(fn) + ": out of buffer memory ordering the rows by row id" +
                                         kPoolCapHint);
            idx_bytes += dense_bytes;
            pos.reset(new std::atomic<std::uint32_t>[std::size_t(span)]());   // () zero-initialises
            const std::size_t per = (rows + workers - 1) / workers;
            {
                std::vector<std::thread> ts;
                for (std::size_t w = 0; w < workers; ++w) {
                    const std::size_t g0 = std::min(rows, w * per), g1 = std::min(rows, g0 + per);
                    ts.emplace_back([&, g0, g1] {
                        for (std::size_t g = g0; g < g1; ++g)
                            pos[std::size_t(std::uint64_t(keys[g].rowid) - std::uint64_t(mn))].store(
                                static_cast<std::uint32_t>(g + 1), std::memory_order_relaxed);
                    });
                }
                for (auto& t : ts) t.join();
            }
            // A slot claimed by another row is a duplicate row id (the writes
            // above raced only if two rows share a slot, so this read-only pass
            // is what reports it).
            {
                std::vector<std::thread> ts;
                for (std::size_t w = 0; w < workers; ++w) {
                    const std::size_t g0 = std::min(rows, w * per), g1 = std::min(rows, g0 + per);
                    ts.emplace_back([&, w, g0, g1] {
                        for (std::size_t g = g0; g < g1; ++g)
                            if (pos[std::size_t(std::uint64_t(keys[g].rowid) - std::uint64_t(mn))].load(
                                    std::memory_order_relaxed) != g + 1) {
                                dup_hit[w] = 1; dup_id[w] = keys[g].rowid;
                                return;
                            }
                    });
                }
                for (auto& t : ts) t.join();
            }
            for (std::size_t w = 0; w < workers; ++w)
                if (dup_hit[w])
                    throw std::runtime_error(std::string(fn) + ": duplicate row id " +
                                             std::to_string(dup_id[w]) + " in the upload");
            // Compact the slots into order[rank]: per-block count, prefix, write.
            const std::size_t bper = (std::size_t(span) + workers - 1) / workers;
            std::vector<std::size_t> cnt(workers, 0), off(workers, 0);
            {
                std::vector<std::thread> ts;
                for (std::size_t w = 0; w < workers; ++w) {
                    const std::size_t i0 = std::min(std::size_t(span), w * bper), i1 = std::min(std::size_t(span), i0 + bper);
                    ts.emplace_back([&, w, i0, i1] {
                        std::size_t n_seen = 0;
                        for (std::size_t i = i0; i < i1; ++i) n_seen += pos[i].load(std::memory_order_relaxed) != 0;
                        cnt[w] = n_seen;
                    });
                }
                for (auto& t : ts) t.join();
            }
            std::size_t total = 0;
            for (std::size_t w = 0; w < workers; ++w) { off[w] = total; total += cnt[w]; }
            if (total != rows) throw std::runtime_error(std::string(fn) + ": row-id ranking lost rows");
            {
                std::vector<std::thread> ts;
                for (std::size_t w = 0; w < workers; ++w) {
                    const std::size_t i0 = std::min(std::size_t(span), w * bper), i1 = std::min(std::size_t(span), i0 + bper);
                    ts.emplace_back([&, w, i0, i1] {
                        std::size_t o = off[w];
                        for (std::size_t i = i0; i < i1; ++i)
                            if (const std::uint32_t k = pos[i].load(std::memory_order_relaxed)) order[o++] = keys[k - 1].src;
                    });
                }
                for (auto& t : ts) t.join();
            }
        } else {
            // Sparse row ids (a scan over part of a table): sort the pairs.
            std::sort(keys.begin(), keys.end(),
                      [](const RowKey& x, const RowKey& y) { return x.rowid < y.rowid; });
            for (std::size_t g = 1; g < rows; ++g)
                if (keys[g].rowid == keys[g - 1].rowid)
                    throw std::runtime_error(std::string(fn) + ": duplicate row id " +
                                             std::to_string(keys[g].rowid) + " in the upload");
            for (std::size_t g = 0; g < rows; ++g) order[g] = keys[g].src;
        }
        std::vector<RowKey>().swap(keys);
        pos.reset();
        charge.give(idx_bytes - rows * sizeof(std::uint64_t));   // only order[] is still live
        // Gather into fresh segments, parallel BY DESTINATION SEGMENT: one
        // thread owns whole segments, so lane_valid allocation and the
        // validity bit writes never race.
        const std::size_t rps  = Segment::kLanes / L;            // rows per destination segment
        const std::size_t nseg = (rows + rps - 1) / rps;
        if (!charge.take(nseg * Segment::kLanes * sizeof(std::int64_t), /*exact*/true))
            throw std::runtime_error(std::string(fn) + ": out of buffer memory placing the rows by row id" +
                                     kPoolCapHint);
        placed_segs.resize(nseg);
        for (auto& s : placed_segs) s = std::make_shared<Segment>();
        std::vector<std::vector<std::size_t>> null_lanes(segs.size());
        for (std::size_t s = 0; s < segs.size(); ++s)
            for (std::size_t l = 0; l < L && l < segs[s]->lane_valid.size(); ++l)
                if (!segs[s]->lane_valid[l].empty()) null_lanes[s].push_back(l);
        {
            const std::size_t nw = std::max<std::size_t>(1, std::min(workers, nseg));
            std::vector<std::thread> ts;
            for (std::size_t w = 0; w < nw; ++w) {
                ts.emplace_back([&, w, nw] {
                    for (std::size_t s = w; s < nseg; s += nw) {
                        Segment& d = *placed_segs[s];
                        const std::size_t g0 = s * rps, g1 = std::min(rows, g0 + rps);
                        for (std::size_t g = g0; g < g1; ++g) {
                            const std::uint64_t src = order[g];
                            const std::size_t sid = std::size_t(src >> 32);
                            const std::size_t srow = std::size_t(src & 0xFFFFFFFFu);
                            const Segment& sg = *segs[sid];
                            const std::size_t drow = g - g0;
                            std::memcpy(d.data + drow * L, sg.data + srow * L, L * sizeof(std::int64_t));
                            for (std::size_t l : null_lanes[sid])
                                if (!(sg.lane_valid[l][srow >> 6] & (std::uint64_t{1} << (srow & 63))))
                                    d.mark_lane_null(drow, l, L);
                        }
                        d.n = (g1 - g0) * L;
                    }
                });
            }
            for (auto& t : ts) t.join();
        }
        std::vector<std::uint64_t>().swap(order);
        charge.give(rows * sizeof(std::uint64_t));
        spans.resize(nseg);
        valid_ptrs.resize(nseg);
        std::size_t at = 0;
        for (std::size_t s = 0; s < nseg; ++s) {
            Segment& d = *placed_segs[s];
            spans[s].lanes = d.data;
            spans[s].rows = d.n / L;
            spans[s].n_lanes = L;
            spans[s].dst_row = at;
            spans[s].valid_bit = 0;
            valid_ptrs[s].assign(L, nullptr);
            for (std::size_t l = 0; l < L && l < d.lane_valid.size(); ++l)
                if (!d.lane_valid[l].empty()) valid_ptrs[s][l] = d.lane_valid[l].data();
            spans[s].valid = valid_ptrs[s].data();
            at += spans[s].rows;
        }
        order_ms = ms_since(t_ord);
    }
    std::vector<gpudb::Dtype> dtypes(L, gpudb::Dtype::I64);
    for (std::size_t e = 0; e < b.n_pf; ++e) dtypes[1 + b.n_pi + e] = gpudb::Dtype::F64;
    auto& a = ctx.aggregator();
    std::vector<std::unique_ptr<gpudb::ResidentColumn>> cols;
    {
        // A store lane named 'k#<field>' is the TUPLE TEXT of a GROUP BY key
        // (_rewrite.store_lanes) — the raw column a WHERE or a payload reads
        // lives under its own name, so this lane is only ever read as a key
        // and the backend may release it once the key's group-id lane exists
        // (§9). Lane 0 is the row id and is not published at all.
        std::uint64_t key_only = 0;
        for (std::size_t l = 1; l < L && l < 64; ++l)
            if (starts_with(names[l - 1], "k#")) key_only |= std::uint64_t{1} << l;
        gpudb::KeyOnlyLanes note(key_only);
        cols = a.upload_rows_exact(spans.data(), spans.size(), dtypes.data(), L);
    }
    if (cols.size() != L) throw std::runtime_error(std::string(fn) + ": upload_rows_exact returned the wrong column count");
    const double up_ms = ms_since(t0);
    const std::string key = std::string("gpudb:v1:") + b.tag.catalog + ":" + b.tag.schema + ":" + b.tag.table + ":" +
                            std::to_string(b.tag.table_oid);
    std::size_t published = 0;
    {
        std::lock_guard<std::mutex> lock(ctx.registry_mu);
        if (ctx.invalidated_since_locked(key, b.seq_at_start))
            throw std::runtime_error(std::string("GPUDB_UPLOAD_DISCARDED: ") + fn + ": upload into '" + key +
                "' discarded — the table was invalidated while the upload ran; run the upload again");
        auto it = ctx.stores.find(key);
        std::shared_ptr<TableStore> store;
        if (it != ctx.stores.end() && it->second->rows_seen == b.rows_seen) {
            store = it->second;
        } else {
            if (it != ctx.stores.end()) {
                // another row count: the table changed — every view on the old store is stale
                for (auto& kv : ctx.registry)
                    if (kv.second->view && kv.second->store_key == key) kv.second->state.store(SetState::Stale);
            }
            store = std::make_shared<TableStore>();
            store->key = key; store->catalog = b.tag.catalog; store->schema = b.tag.schema; store->table = b.tag.table;
            store->table_oid = b.tag.table_oid; store->rows_seen = b.rows_seen;
            store->epoch = ctx.inval_seq.load(std::memory_order_acquire);
            ctx.stores[key] = store;
        }
        const std::int64_t now = now_us();
        for (std::size_t l = 1; l < L; ++l) {
            auto sc = std::make_shared<StoreColumn>();
            sc->expr = names[l - 1];
            sc->col = std::move(cols[l]);
            sc->is_str = l >= 1 + b.n_pi + b.n_pf;
            if (sc->is_str) {
                const std::size_t slot = l - (1 + b.n_pi + b.n_pf);
                sc->dict = std::make_shared<const ResidentSet::Dict>(
                    slot < b.dicts.size() ? std::move(b.dicts[slot]) : ResidentSet::Dict{});
            }
            sc->uploaded_at_us = now;
            sc->last_used_at_us.store(now, std::memory_order_relaxed);
            store->cols[sc->expr] = sc;                  // a re-upload of a lane replaces it
            ++published;
        }
    }
    if (upload_trace())
        std::fprintf(stderr, "[gpudb upload] %s '%s': rows=%zu lanes=%zu chunks=%zu placed=%d order=%.1f ms "
                     "upload=%.1f ms (store %s)\n",
                     fn, b.name.c_str(), rows, L - 1, chunks.size(), placed ? 1 : 0, order_ms, up_ms, key.c_str());
    return rows;
}

void upload_columns_finalize(duckdb_function_info info, duckdb_aggregate_state* source,
                             duckdb_vector result, idx_t count, idx_t offset) {
    if (count == 0) return;
    auto* out = reinterpret_cast<std::int64_t*>(duckdb_vector_get_data(result));
    duckdb_vector_ensure_validity_writable(result);
    uint64_t* validity = duckdb_vector_get_validity(result);
    for (idx_t i = 0; i < count; ++i) {
        UploadState* s = probe_upload_state(source[i]);
        UploadBuf* b = (s && s->buf_id != 0) ? s->buf : nullptr;
        const std::size_t lanes = b ? b->lanes() : 0;
        if (!b || lanes == 0 || b->lanes_per_row == 0) {
            out[offset + i] = 0;
            duckdb_validity_set_row_invalid(validity, offset + i);
            continue;
        }
        try {
            ResidentContext& ctx = ctx_of_aggregate(info);
            if (session_append(ctx, *b, /*pair*/true, gpudb::Dtype::I64, "gpu_upload_columns", /*exact*/true)) {
                out[offset + i] = static_cast<std::int64_t>(lanes / b->lanes_per_row);
                continue;
            }
            out[offset + i] = static_cast<std::int64_t>(finish_upload_columns(ctx, *b, "gpu_upload_columns"));
        } catch (const std::exception& e) {
            duckdb_aggregate_function_set_error(info,
                (std::string("gpu_upload_columns failed: ") + e.what()).c_str());
            return;
        }
    }
}

// gpu_drop_column(store, lane) -> BOOLEAN: free one column of a table store
// (the wrapper's memory budget evicts columns, least recently used first).
// Every view that reads the lane is dropped with it.
void drop_column_exec(duckdb_function_info info, duckdb_data_chunk input, duckdb_vector output) {
    duckdb_vector s_vec = duckdb_data_chunk_get_vector(input, 0);
    duckdb_vector c_vec = duckdb_data_chunk_get_vector(input, 1);
    auto* stores = reinterpret_cast<duckdb_string_t*>(duckdb_vector_get_data(s_vec));
    auto* cols = reinterpret_cast<duckdb_string_t*>(duckdb_vector_get_data(c_vec));
    uint64_t* sv = duckdb_vector_get_validity(s_vec);
    uint64_t* cv = duckdb_vector_get_validity(c_vec);
    const idx_t n = duckdb_data_chunk_get_size(input);
    auto* out = reinterpret_cast<bool*>(duckdb_vector_get_data(output));
    ResidentContext& ctx = ctx_of(info);
    for (idx_t i = 0; i < n; ++i) {
        if ((sv && !duckdb_validity_row_is_valid(sv, i)) || (cv && !duckdb_validity_row_is_valid(cv, i))) {
            out[i] = false;
            continue;
        }
        const std::string key = read_name(stores, i), lane = read_name(cols, i);
        std::lock_guard<std::mutex> lock(ctx.registry_mu);
        auto it = ctx.stores.find(key);
        if (it == ctx.stores.end()) { out[i] = false; continue; }
        auto& store = *it->second;
        auto cit = store.cols.find(lane);
        if (cit == store.cols.end()) { out[i] = false; continue; }
        const auto col = cit->second->col;
        store.cols.erase(cit);
        for (auto r = ctx.registry.begin(); r != ctx.registry.end();) {
            const auto& v = r->second;
            const bool uses = v->view && v->store_key == key &&
                (v->keys == col || v->vals == col || std::find(v->preds.begin(), v->preds.end(), col) != v->preds.end());
            if (uses) { v->state.store(SetState::Stale); r = ctx.registry.erase(r); } else ++r;
        }
        if (store.cols.empty()) ctx.stores.erase(it);
        out[i] = true;
    }
}

// gpu_store_columns() -> TABLE: every resident table column (stage B), what it
// costs and when it was last read — what the memory budget accounts for.
struct StoreColumnsRow {
    std::string store, catalog, schema, table, column, dtype;
    std::int64_t table_oid = -1, rows = 0, bytes = 0, epoch = 0, uploaded_at_us = 0, last_used_at_us = 0;
    std::int64_t width = 0;     // storage width of the lane in bytes; 0 = the backend does not say
    bool prepared = false;
};
struct StoreColumnsInit { std::vector<StoreColumnsRow> rows; std::size_t offset = 0; };

void store_columns_bind(duckdb_bind_info info) {
    auto add = [&](const char* name, duckdb_type t) {
        duckdb_logical_type lt = duckdb_create_logical_type(t);
        duckdb_bind_add_result_column(info, name, lt);
        duckdb_destroy_logical_type(&lt);
    };
    add("store",        DUCKDB_TYPE_VARCHAR);
    add("catalog",      DUCKDB_TYPE_VARCHAR);
    add("schema",       DUCKDB_TYPE_VARCHAR);
    add("table",        DUCKDB_TYPE_VARCHAR);
    add("table_oid",    DUCKDB_TYPE_BIGINT);
    add("column",       DUCKDB_TYPE_VARCHAR);
    add("dtype",        DUCKDB_TYPE_VARCHAR);    // 'I64' | 'F64' | 'STR'
    add("rows",         DUCKDB_TYPE_BIGINT);
    add("bytes",        DUCKDB_TYPE_BIGINT);     // backend memory incl. the sort cache
    add("prepared",     DUCKDB_TYPE_BOOLEAN);
    add("epoch",        DUCKDB_TYPE_BIGINT);
    add("uploaded_at",  DUCKDB_TYPE_TIMESTAMP);
    add("last_used_at", DUCKDB_TYPE_TIMESTAMP);
    // Added after the columns above, so a caller that reads this table
    // positionally keeps reading the same values it always did. NULL when the
    // backend leaves no width note (src/include/backend_notes.hpp): an absent
    // answer, not a width of zero.
    add("width",        DUCKDB_TYPE_BIGINT);     // bytes per row of the lane (stage C)
}

void store_columns_init(duckdb_init_info info) {
    auto* init = new StoreColumnsInit();
    try {
        ResidentContext& ctx = resident_context(duckdb_init_get_extra_info(info));
        std::lock_guard<std::mutex> lock(ctx.registry_mu);
        for (const auto& kv : ctx.stores) {
            const TableStore& st = *kv.second;
            for (const auto& ck : st.cols) {
                const StoreColumn& c = *ck.second;
                StoreColumnsRow r;
                r.store = st.key; r.catalog = st.catalog; r.schema = st.schema; r.table = st.table;
                r.table_oid = st.table_oid; r.column = c.expr;
                r.dtype = c.is_str ? "STR" : c.col->dtype() == gpudb::Dtype::F64 ? "F64" : "I64";
                r.rows = static_cast<std::int64_t>(c.col->rows());
                r.bytes = static_cast<std::int64_t>(c.col->resident_bytes());
                r.prepared = c.col->prepared();
                r.width = static_cast<std::int64_t>(gpudb::lane_storage_width(*c.col));
                r.epoch = static_cast<std::int64_t>(st.epoch);
                r.uploaded_at_us = c.uploaded_at_us;
                r.last_used_at_us = c.last_used_at_us.load();
                init->rows.push_back(std::move(r));
            }
        }
        std::sort(init->rows.begin(), init->rows.end(),
                  [](const StoreColumnsRow& a, const StoreColumnsRow& b) {
                      return a.store != b.store ? a.store < b.store : a.column < b.column; });
    } catch (const std::exception& e) {
        delete init;
        duckdb_init_set_error(info, e.what());
        return;
    }
    duckdb_init_set_init_data(info, init, [](void* p) { delete static_cast<StoreColumnsInit*>(p); });
}

void store_columns_function(duckdb_function_info info, duckdb_data_chunk output) {
    auto* init = static_cast<StoreColumnsInit*>(duckdb_function_get_init_data(info));
    if (!init) return;
    const std::size_t remaining = init->rows.size() - init->offset;
    if (remaining == 0) return;
    const idx_t out_n = static_cast<idx_t>(std::min<std::size_t>(remaining, 2048));
    auto vec = [&](idx_t c) { return duckdb_data_chunk_get_vector(output, c); };
    auto set_str = [&](idx_t c, idx_t i, const std::string& v) {
        duckdb_vector_assign_string_element_len(vec(c), i, v.data(), v.size());
    };
    auto set_i64 = [&](idx_t c, idx_t i, std::int64_t x) { static_cast<std::int64_t*>(duckdb_vector_get_data(vec(c)))[i] = x; };
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
        const StoreColumnsRow& r = init->rows[init->offset + i];
        set_str(0, i, r.store); set_str(1, i, r.catalog); set_str(2, i, r.schema); set_str(3, i, r.table);
        set_i64(4, i, r.table_oid); set_str(5, i, r.column); set_str(6, i, r.dtype);
        set_i64(7, i, r.rows); set_i64(8, i, r.bytes);
        static_cast<bool*>(duckdb_vector_get_data(vec(9)))[i] = r.prepared;
        set_i64(10, i, r.epoch); set_ts(11, i, r.uploaded_at_us); set_ts(12, i, r.last_used_at_us);
        if (r.width > 0) {
            set_i64(13, i, r.width);
        } else {
            duckdb_vector v = vec(13);
            duckdb_vector_ensure_validity_writable(v);
            duckdb_validity_set_row_invalid(duckdb_vector_get_validity(v), i);
        }
    }
    duckdb_data_chunk_set_size(output, out_n);
    init->offset += static_cast<std::size_t>(out_n);
}

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
            if (ss->exact) {
                b.lanes_per_row = ss->lanes_per_row; b.n_pi = ss->n_pi; b.n_pf = ss->n_pf; b.n_ps = ss->n_ps;
                b.key_str = ss->key_str; b.dicts = std::move(ss->dicts);
                b.exact = true;
            }
            if (ss->columns) { b.columns = true; b.chunks = std::move(ss->chunks); b.unordered = ss->unordered; }
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
//   join=true|false        → the runtime backend runs join_materialize on its own device (§4.8)
//   exact=true|false       → the runtime backend runs the v0.7 exact GROUP BY
//   global=true|false      → ... and the global masked aggregate (§4.12) on its own device
//   narrow=true|false      → lanes are stored at their narrowest width (docs/RESIDENT_COLUMNS_DESIGN.md
//                            stage C); the wrapper's memory estimate sizes lanes from their type then
//   rebuilds=<c>/<l>       → sort caches and key lanes a resident column shed and had to
//                            rebuild, process-wide (docs/RESIDENT_COLUMNS_DESIGN.md §9).
//                            Zero on a workload whose shapes the shed rule read right
//   device_memory=<bytes>  → what the backend reports for the memory budget (0 = unknown, §5.5)
//                            (NULL-aware, HUGEINT sums, WHERE mask) on its own
//                            device; the wrapper only rewrites when true
//   device='<name>'        → what the GPU calls itself (src/include/backend_notes.hpp).
//                            Absent on a CPU-only build and on a backend that leaves
//                            no note; quoted because the name carries spaces, and last
//                            on the line so a reader can find it either way
void build_info_exec(duckdb_function_info info_, duckdb_data_chunk input,
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
    info += ctx_of(info_).aggregator().exact_supported() ? " exact=true" : " exact=false";
    info += ctx_of(info_).aggregator().join_supported() ? " join=true" : " join=false";
    info += ctx_of(info_).aggregator().global_supported() ? " global=true" : " global=false";
    info += ctx_of(info_).aggregator().narrow_lanes() ? " narrow=true" : " narrow=false";
    info += " device_memory=" + std::to_string(ctx_of(info_).aggregator().device_memory_bytes());
    info += " store=true";
    info += " rebuilds=" + std::to_string(gpudb::resident_cache_rebuilds().load()) +
            "/" + std::to_string(gpudb::resident_lane_rebuilds().load());
    // The backend leaves its device name in backend_notes.hpp when it has
    // one; a CPU-only build leaves none and this clause is simply absent.
    // Quotes and anything that would end the value early are dropped: the
    // name is a label, not a channel.
    std::string device = gpudb::device_name();
    std::string clean;
    for (const char c : device)
        if (c != '\'' && c != '\n' && c != '\r') clean += c;
    if (!clean.empty()) info += " device='" + clean + "'";
    // The mantissa width avg() is finalised in (src/include/native_avg.hpp).
    // 53 means long double IS double here, so the SQL derivation the wrapper
    // uses for a DECIMAL payload — double(unscaled sum) / (count * 10^s) —
    // reproduces native; anything wider means it does not, and the wrapper
    // must decline that shape rather than return a different answer.
    info += " avgf=" + std::to_string(LDBL_MANT_DIG);
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
            if (ref.set->no_key && ref.set->keys.get() == ref.col) {
                // §4.12: a global set has no key; there is nothing to derive
                SetState expect = SetState::Uploaded;
                ref.set->state.compare_exchange_strong(expect, SetState::Ready);
                out[i] = true;
                continue;
            }
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

// gpu_join_materialize(out, probe, probe_lane, build, build_lane, lanes) -> BIGINT
// (v0.7 §4.8). Inner equi-join of two exact sets on the device, published as
// a NEW exact set `out` that every gpu_groupby_exact_resident* form reads
// like an uploaded one. build_lane must be unique among its non-NULL cells
// (a primary / unique key); the call fails with "build key not unique"
// otherwise. `lanes` lists the output columns, each `p.<lane>` (probe set)
// or `b.<lane>` (build set) with <lane> one of k v i<n> f<n> s<n>: the first
// is the new key, the second the new payload (BIGINT), the rest the new
// predicate columns — BIGINT lanes first, then DOUBLE, then VARCHAR, which
// become i<n> / f<n> / s<n> of `out` in that order. String lanes (and a
// string key) carry their dictionaries along. Returns the joined row count.
// `out` is stale as soon as either source set is stale, dropped or replaced.
struct JoinLaneRef {
    const gpudb::ResidentColumn* col = nullptr;
    char kind = 'i';                                                         // i | f | s
    const std::unordered_map<std::uint64_t, std::string>* dict = nullptr;    // s lanes
};
JoinLaneRef join_lane_of(const ResidentSet& set, const std::string& c, const char* fn) {
    JoinLaneRef r;
    auto bad = [&]() {
        return std::runtime_error(std::string(fn) + ": unknown lane '" + c + "' in set '" + set.name +
                                  "' (k, v, i<n>, f<n>, s<n>)");
    };
    if (c == "k") {
        r.col = set.keys.get();
        if (set.key_str) { r.kind = 's'; r.dict = set.key_dict.get(); }
    } else if (c == "v") {
        r.col = set.vals.get();
    } else if (c.size() > 1 && (c[0] == 'i' || c[0] == 'f' || c[0] == 's')) {
        std::size_t idx = 0;
        for (std::size_t q = 1; q < c.size(); ++q) {
            if (!std::isdigit(static_cast<unsigned char>(c[q])) || q > 6) throw bad();
            idx = idx * 10 + static_cast<std::size_t>(c[q] - '0');
        }
        const std::size_t count = c[0] == 'i' ? set.pred_int : c[0] == 'f' ? set.pred_dbl : set.pred_str;
        if (idx >= count) throw bad();
        const std::size_t base = c[0] == 'i' ? 0 : c[0] == 'f' ? set.pred_int : set.pred_int + set.pred_dbl;
        r.col = set.preds[base + idx].get();
        r.kind = c[0];
        if (c[0] == 's') r.dict = set.str_dicts[idx].get();
    } else {
        throw bad();
    }
    if (!r.col) throw bad();
    return r;
}

void join_materialize_exec(duckdb_function_info info, duckdb_data_chunk input, duckdb_vector output) {
    static const char* fn = "gpu_join_materialize";
    const idx_t n = duckdb_data_chunk_get_size(input);
    duckdb_string_t* arg[6];
    uint64_t* val[6];
    for (idx_t a = 0; a < 6; ++a) {
        duckdb_vector v = duckdb_data_chunk_get_vector(input, a);
        arg[a] = reinterpret_cast<duckdb_string_t*>(duckdb_vector_get_data(v));
        val[a] = duckdb_vector_get_validity(v);
    }
    auto* out = reinterpret_cast<std::int64_t*>(duckdb_vector_get_data(output));
    ResidentContext& ctx = ctx_of(info);
    for (idx_t i = 0; i < n; ++i) {
        try {
            for (idx_t a = 0; a < 6; ++a)
                if (val[a] && !duckdb_validity_row_is_valid(val[a], i))
                    throw std::runtime_error(std::string(fn) + ": arguments may not be NULL");
            const std::string out_name = read_name(arg[0], i), probe_name = read_name(arg[1], i),
                              probe_lane = read_name(arg[2], i), build_name = read_name(arg[3], i),
                              build_lane = read_name(arg[4], i), spec = read_name(arg[5], i);
            if (out_name.empty()) throw std::runtime_error(std::string(fn) + ": empty output name");
            if (out_name == probe_name || out_name == build_name)
                throw std::runtime_error(std::string(fn) + ": the output name must differ from both inputs");
            UploadBuf b;                       // carries the name / tag / sequence publish_set reads
            b.name = out_name;
            if (starts_with(b.name, kTagPrefix)) {
                const std::string err = parse_tag(b.name, b.tag);
                if (!err.empty()) throw std::runtime_error(std::string(fn) + ": " + err);
                b.managed = true;
            }
            b.seq_at_start = ctx.inval_seq.load(std::memory_order_acquire);
            const auto t0 = std::chrono::steady_clock::now();
            std::shared_ptr<ResidentSet> ps = ctx.acquire(probe_name, fn);
            std::shared_ptr<ResidentSet> bs = ctx.acquire(build_name, fn);
            if (!ps->exact || !bs->exact)
                throw std::runtime_error(std::string(fn) + ": both inputs must be exact sets "
                                         "(gpu_upload_pair_exact / gpu_upload_rows_exact / gpu_join_materialize)");
            const JoinLaneRef pk = join_lane_of(*ps, probe_lane, fn);
            const JoinLaneRef bk = join_lane_of(*bs, build_lane, fn);
            if (pk.kind != 'i' || bk.kind != 'i')
                throw std::runtime_error(std::string(fn) + ": the join lanes must be BIGINT lanes");

            // lanes spec: "p.k, b.i0, ..."
            std::vector<gpudb::JoinLane> lanes;
            std::vector<JoinLaneRef> refs;
            std::size_t pos = 0;
            while (pos <= spec.size()) {
                std::size_t comma = spec.find(',', pos);
                if (comma == std::string::npos) comma = spec.size();
                std::string tok = spec.substr(pos, comma - pos);
                tok.erase(0, tok.find_first_not_of(" \t"));
                tok.erase(tok.find_last_not_of(" \t") + 1);
                pos = comma + 1;
                if (tok.empty()) { if (comma == spec.size()) break; throw std::runtime_error(std::string(fn) + ": empty lane in '" + spec + "'"); }
                if (tok.size() < 3 || tok[1] != '.' || (tok[0] != 'p' && tok[0] != 'b'))
                    throw std::runtime_error(std::string(fn) + ": lane '" + tok + "' must be p.<lane> or b.<lane>");
                const bool from_build = tok[0] == 'b';
                JoinLaneRef ref = join_lane_of(from_build ? *bs : *ps, tok.substr(2), fn);
                lanes.push_back(gpudb::JoinLane{ref.col, from_build});
                refs.push_back(ref);
            }
            // §4.12: a set only global aggregates read has no key lane at
            // all — its spec starts at the payload, and the published set's
            // key points at that same column.
            const bool nokey = b.managed && no_key_tag(b.tag);
            const std::size_t first_pred = nokey ? 1 : 2;
            if (lanes.size() < first_pred)
                throw std::runtime_error(std::string(fn) + (nokey ? ": lanes needs at least a payload"
                                                                  : ": lanes needs at least a key and a payload"));
            if (lanes.size() > 66)
                throw std::runtime_error(std::string(fn) + ": at most 66 output lanes");
            if (!nokey && refs[0].kind == 'f') throw std::runtime_error(std::string(fn) + ": the key lane may not be a DOUBLE lane");
            if (refs[first_pred - 1].kind != 'i') throw std::runtime_error(std::string(fn) + ": the payload lane must be a BIGINT lane");
            std::size_t n_i = 0, n_f = 0, n_s = 0;
            for (std::size_t l = first_pred; l < refs.size(); ++l) {
                const char k = refs[l].kind;
                if ((k == 'i' && (n_f || n_s)) || (k == 'f' && n_s))
                    throw std::runtime_error(std::string(fn) + ": predicate lanes must list BIGINT lanes first, then DOUBLE, then VARCHAR");
                (k == 'i' ? n_i : k == 'f' ? n_f : n_s)++;
            }

            gpudb::JoinMaterializeResult jr;
            {
                // As for an uploaded join: lane 0 of the RESULT of the last
                // step is the statement's key and nothing else. An
                // intermediate's lane 0 is the next step's probe key, and
                // join_result_tag() says so.
                const bool key_only = b.managed && join_result_tag(b.tag) && !no_key_tag(b.tag);
                gpudb::KeyOnlyLanes note(key_only ? std::uint64_t{1} : std::uint64_t{0});
                auto dev = resident_device_lock(ctx);
                jr = ctx.aggregator().join_materialize(*pk.col, *bk.col, lanes.data(), lanes.size());
            }
            // dictionaries travel with their lanes
            b.key_str = !nokey && refs[0].kind == 's';
            if (b.key_str) b.dicts.push_back(*refs[0].dict);
            for (std::size_t l = first_pred; l < refs.size(); ++l)
                if (refs[l].kind == 's') b.dicts.push_back(*refs[l].dict);
            b.rows_seen = jr.rows_out;
            std::vector<std::unique_ptr<gpudb::ResidentColumn>> preds;
            for (std::size_t l = first_pred; l < jr.lanes.size(); ++l) preds.push_back(std::move(jr.lanes[l]));
            std::vector<std::pair<std::string, std::weak_ptr<ResidentSet>>> deps;
            // A source that is itself a joined set hands down ITS sources:
            // the rows were copied, so the intermediate may be dropped and
            // only the base sets decide staleness.
            for (const auto& src : {std::make_pair(probe_name, ps), std::make_pair(build_name, bs)}) {
                if (src.second->deps.empty()) deps.emplace_back(src.first, src.second);
                else for (const auto& d : src.second->deps) deps.push_back(d);
            }
            publish_set(ctx, b, nokey ? nullptr : std::move(jr.lanes[0]),
                        std::move(jr.lanes[first_pred - 1]), /*pair*/true, fn,
                        /*exact*/true, std::move(preds), n_i, n_f, n_s, std::move(deps));
            char buf[512];
            std::snprintf(buf, sizeof(buf),
                "op=join_materialize set=%s rows_probe=%zu rows_build=%zu rows_out=%zu null_key_rows=%zu "
                "lanes=%zu kernel_ms=%.3f wall_ms=%.3f total_ms=%.3f",
                out_name.c_str(), jr.rows_probe, jr.rows_build, jr.rows_out, jr.null_key_rows,
                lanes.size(), jr.kernel_ms, jr.wall_ms, ms_since(t0));
            ctx.record_stats(nullptr, buf);
            out[i] = static_cast<std::int64_t>(jr.rows_out);
        } catch (const std::exception& e) {
            duckdb_scalar_function_set_error(info, e.what());
            return;
        }
    }
}

// gpu_note_rows(name, n) -> BOOLEAN (v0.7 §4.13). Registers a column-less
// SENTINEL set that only remembers n, the row count of a base table when a
// set built from a JOIN over it was uploaded. gpu_assert_rows(name,
// count(*)) over the table is then the staleness guard for that table, as it
// is for a table's own set. Holds no device memory.
void note_rows_exec(duckdb_function_info info, duckdb_data_chunk input, duckdb_vector output) {
    duckdb_vector name_vec = duckdb_data_chunk_get_vector(input, 0);
    duckdb_vector n_vec    = duckdb_data_chunk_get_vector(input, 1);
    auto* names  = reinterpret_cast<duckdb_string_t*>(duckdb_vector_get_data(name_vec));
    auto* counts = reinterpret_cast<const std::int64_t*>(duckdb_vector_get_data(n_vec));
    uint64_t* name_validity = duckdb_vector_get_validity(name_vec);
    uint64_t* n_validity    = duckdb_vector_get_validity(n_vec);
    const idx_t n = duckdb_data_chunk_get_size(input);
    auto* out = reinterpret_cast<bool*>(duckdb_vector_get_data(output));
    ResidentContext& ctx = ctx_of(info);
    for (idx_t i = 0; i < n; ++i) {
        try {
            if ((name_validity && !duckdb_validity_row_is_valid(name_validity, i)) ||
                (n_validity && !duckdb_validity_row_is_valid(n_validity, i)) || counts[i] < 0)
                throw std::runtime_error("gpu_note_rows: name and a non-negative row count are required");
            auto set = std::make_shared<ResidentSet>();
            set->name = read_name(names, i);
            if (set->name.empty()) throw std::runtime_error("gpu_note_rows: empty name");
            if (starts_with(set->name, kTagPrefix)) {
                TagFields tag;
                const std::string err = parse_tag(set->name, tag);
                if (!err.empty()) throw std::runtime_error("gpu_note_rows: " + err);
                set->managed = true;
                set->catalog = tag.catalog; set->schema = tag.schema; set->table = tag.table;
                set->table_oid = tag.table_oid; set->columns = tag.columns; set->extra = tag.extra;
            }
            set->rows_seen = static_cast<std::size_t>(counts[i]);
            set->uploaded_at_us = now_us();
            set->state.store(SetState::Ready);
            const std::uint64_t seq = ctx.inval_seq.load(std::memory_order_acquire);
            if (!ctx.publish(set, seq))
                throw std::runtime_error("GPUDB_UPLOAD_DISCARDED: gpu_note_rows: '" + read_name(names, i) +
                                         "' was invalidated meanwhile");
            out[i] = true;
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
                set = ctx.find_or_view_locked(name);
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
// gpu_resident_dictionary(tag, n) -> (id BIGINT, c0 .. c<n-1> VARCHAR): the
// key tuples of a VARCHAR-keyed exact set, one row per distinct key, split
// back into their n components ("N" -> NULL). The rewritten statement joins
// it on r.key = d.id to put the strings back.
// ---------------------------------------------------------------------------

struct DictBind { std::string name; std::int64_t n = 1; };
struct DictInit {
    std::vector<std::pair<std::uint64_t, std::string>> rows;
    std::size_t offset = 0;
    std::int64_t n = 1;
};

void dictionary_bind(duckdb_bind_info info) {
    duckdb_value nv = duckdb_bind_get_parameter(info, 0);
    duckdb_value cv = duckdb_bind_get_parameter(info, 1);
    if (!nv || !cv || duckdb_is_null_value(nv) || duckdb_is_null_value(cv)) {
        if (nv) duckdb_destroy_value(&nv);
        if (cv) duckdb_destroy_value(&cv);
        duckdb_bind_set_error(info, "gpu_resident_dictionary: name and n may not be NULL");
        return;
    }
    auto* bind = new DictBind();
    char* nm = duckdb_get_varchar(nv);
    bind->name = nm ? nm : "";
    if (nm) duckdb_free(nm);
    bind->n = duckdb_get_int64(cv);
    duckdb_destroy_value(&nv);
    duckdb_destroy_value(&cv);
    if (bind->n < 1 || bind->n > 8) {
        delete bind;
        duckdb_bind_set_error(info, "gpu_resident_dictionary: n must be between 1 and 8");
        return;
    }
    duckdb_logical_type bigint = duckdb_create_logical_type(DUCKDB_TYPE_BIGINT);
    duckdb_logical_type vc     = duckdb_create_logical_type(DUCKDB_TYPE_VARCHAR);
    duckdb_bind_add_result_column(info, "id", bigint);
    for (std::int64_t i = 0; i < bind->n; ++i)
        duckdb_bind_add_result_column(info, ("c" + std::to_string(i)).c_str(), vc);
    duckdb_destroy_logical_type(&bigint);
    duckdb_destroy_logical_type(&vc);
    duckdb_bind_set_bind_data(info, bind, [](void* p) { delete static_cast<DictBind*>(p); });
}

void dictionary_init(duckdb_init_info info) {
    auto* bind = static_cast<DictBind*>(duckdb_init_get_bind_data(info));
    auto* init = new DictInit();
    init->n = bind->n;
    try {
        ResidentContext& ctx = resident_context(duckdb_init_get_extra_info(info));
        std::shared_ptr<ResidentSet> set = resident_acquire_set(ctx, bind->name, "gpu_resident_dictionary");
        if (!set->exact || !set->key_str)
            throw std::runtime_error("gpu_resident_dictionary: '" + bind->name +
                "' has no string key (upload it with gpu_upload_rows_exact and a VARCHAR key)");
        init->rows.reserve(set->key_dict->size());
        for (const auto& kv : *set->key_dict) init->rows.emplace_back(kv.first, kv.second);
        std::sort(init->rows.begin(), init->rows.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
    } catch (const std::exception& e) {
        delete init;
        duckdb_init_set_error(info, e.what());
        return;
    }
    duckdb_init_set_init_data(info, init, [](void* p) { delete static_cast<DictInit*>(p); });
}

// Splits "<len>:<bytes>..." / "N" into n components; a malformed text (which
// the wrapper never produces) yields NULLs rather than an error.
void split_tuple(const std::string& t, std::int64_t n, std::vector<std::pair<bool, std::string>>& out) {
    out.assign(static_cast<std::size_t>(n), {false, std::string()});
    std::size_t i = 0;
    for (std::int64_t c = 0; c < n && i < t.size(); ++c) {
        if (t[i] == 'N') { ++i; continue; }                  // NULL component
        std::size_t len = 0; bool any = false;
        while (i < t.size() && std::isdigit(static_cast<unsigned char>(t[i]))) { len = len * 10 + (t[i] - '0'); ++i; any = true; }
        if (!any || i >= t.size() || t[i] != ':') return;
        ++i;
        if (i + len > t.size()) return;
        out[static_cast<std::size_t>(c)] = {true, t.substr(i, len)};
        i += len;
    }
}

// gpu_resident_dict_component(name, key BIGINT, i BIGINT) -> VARCHAR: component
// i of ONE key of a string-keyed set (NULL for a NULL component, a NULL key or
// a key the set does not hold). The per-key form of gpu_resident_dictionary:
// after a HAVING or a top-k only a handful of keys are left, and decoding them
// one by one costs microseconds where materialising the whole dictionary (a
// copy and a sort of every key tuple, per statement) cost tens of ms.
void dict_component_exec(duckdb_function_info info, duckdb_data_chunk input, duckdb_vector output) {
    static const char* fn = "gpu_resident_dict_component";
    duckdb_vector name_vec = duckdb_data_chunk_get_vector(input, 0);
    duckdb_vector key_vec  = duckdb_data_chunk_get_vector(input, 1);
    duckdb_vector idx_vec  = duckdb_data_chunk_get_vector(input, 2);
    auto* names = reinterpret_cast<duckdb_string_t*>(duckdb_vector_get_data(name_vec));
    auto* keys  = reinterpret_cast<const std::int64_t*>(duckdb_vector_get_data(key_vec));
    auto* idxs  = reinterpret_cast<const std::int64_t*>(duckdb_vector_get_data(idx_vec));
    uint64_t* nv = duckdb_vector_get_validity(name_vec);
    uint64_t* kv = duckdb_vector_get_validity(key_vec);
    uint64_t* iv = duckdb_vector_get_validity(idx_vec);
    const idx_t n = duckdb_data_chunk_get_size(input);
    duckdb_vector_ensure_validity_writable(output);
    uint64_t* ov = duckdb_vector_get_validity(output);
    ResidentContext& ctx = ctx_of(info);
    std::string cur_name;
    std::shared_ptr<ResidentSet> set;
    std::vector<std::pair<bool, std::string>> parts;
    try {
        for (idx_t r = 0; r < n; ++r) {
            if ((nv && !duckdb_validity_row_is_valid(nv, r)) || (iv && !duckdb_validity_row_is_valid(iv, r)))
                throw std::runtime_error(std::string(fn) + ": name and component index may not be NULL");
            if (kv && !duckdb_validity_row_is_valid(kv, r)) { duckdb_validity_set_row_invalid(ov, r); continue; }
            const std::string name = read_name(names, r);
            if (!set || name != cur_name) {
                set = ctx.acquire(name, fn, /*count_hit*/false);
                if (!set->exact || !set->key_str)
                    throw std::runtime_error(std::string(fn) + ": '" + name + "' has no string key");
                cur_name = name;
            }
            const std::int64_t i = idxs[r];
            if (i < 0 || i >= 8) throw std::runtime_error(std::string(fn) + ": component index must be 0..7");
            const auto it = set->key_dict->find(static_cast<std::uint64_t>(keys[r]));
            if (it == set->key_dict->end()) { duckdb_validity_set_row_invalid(ov, r); continue; }
            split_tuple(it->second, i + 1, parts);
            if (!parts[static_cast<std::size_t>(i)].first) { duckdb_validity_set_row_invalid(ov, r); continue; }
            const std::string& text = parts[static_cast<std::size_t>(i)].second;
            duckdb_vector_assign_string_element_len(output, r, text.data(), text.size());
        }
    } catch (const std::exception& e) {
        duckdb_scalar_function_set_error(info, e.what());
    }
}

void dictionary_function(duckdb_function_info info, duckdb_data_chunk output) {
    auto* init = static_cast<DictInit*>(duckdb_function_get_init_data(info));
    if (!init) return;
    const std::size_t remaining = init->rows.size() - init->offset;
    if (remaining == 0) return;
    constexpr idx_t kChunk = 2048;
    const idx_t out_n = static_cast<idx_t>(std::min<std::size_t>(remaining, kChunk));
    auto* ids = static_cast<std::int64_t*>(duckdb_vector_get_data(duckdb_data_chunk_get_vector(output, 0)));
    std::vector<std::pair<bool, std::string>> parts;
    for (idx_t i = 0; i < out_n; ++i) {
        const auto& row = init->rows[init->offset + i];
        ids[i] = static_cast<std::int64_t>(row.first);
        split_tuple(row.second, init->n, parts);
        for (std::int64_t c = 0; c < init->n; ++c) {
            duckdb_vector v = duckdb_data_chunk_get_vector(output, static_cast<idx_t>(1 + c));
            const auto& pc = parts[static_cast<std::size_t>(c)];
            if (!pc.first) {
                duckdb_vector_ensure_validity_writable(v);
                duckdb_validity_set_row_invalid(duckdb_vector_get_validity(v), i);
            } else {
                duckdb_vector_assign_string_element_len(v, i, pc.second.data(), pc.second.size());
            }
        }
    }
    duckdb_data_chunk_set_size(output, out_n);
    init->offset += static_cast<std::size_t>(out_n);
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
            r.kind = s->view ? "view" : s->pair ? "pair" : "column";
            auto dt = [](const gpudb::ResidentColumn* c) {
                return !c ? "" : c->dtype() == gpudb::Dtype::I64 ? "I64" : "F64";
            };
            r.dtype = s->pair ? std::string(dt(s->keys.get())) + "/" + dt(s->vals.get())
                              : std::string(dt(s->keys.get()));
            r.rows = static_cast<std::int64_t>(s->rows);
            r.rows_seen = static_cast<std::int64_t>(s->rows_seen);
            r.bytes = s->view ? 0 : static_cast<std::int64_t>(s->resident_bytes());   // a view owns nothing: gpu_store_columns() counts
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

        // gpu_upload_pair_exact(name, k BIGINT, v BIGINT) -> BIGINT (v0.7 §4.1)
        duckdb_aggregate_function pex = make_pair_fn(DUCKDB_TYPE_BIGINT,
            upload_pair_exact_update, upload_pair_exact_finalize);
        duckdb_aggregate_function_set_name(pex, "gpu_upload_pair_exact");
        duckdb_state est = duckdb_register_aggregate_function(con, pex);
        duckdb_destroy_aggregate_function(&pex);
        if (est == DuckDBError) {
            throw std::runtime_error("gpu_upload_pair_exact registration failed");
        }

        // gpu_upload_rows_exact overload set (v0.7 §4.6 / §4.5):
        //   (name, k BIGINT,  v BIGINT, pi BIGINT[], pf DOUBLE[])               -> BIGINT
        //   (name, k BIGINT,  v BIGINT, pi BIGINT[], pf DOUBLE[], ps VARCHAR[]) -> BIGINT
        //   (name, k VARCHAR, v BIGINT, pi BIGINT[], pf DOUBLE[], ps VARCHAR[]) -> BIGINT
        {
            auto make_rows_fn = [&](bool key_str, bool with_ps, duckdb_aggregate_update_t update) {
                duckdb_aggregate_function rfn = duckdb_create_aggregate_function();
                duckdb_aggregate_function_set_name(rfn, "gpu_upload_rows_exact");
                duckdb_logical_type t_name = duckdb_create_logical_type(DUCKDB_TYPE_VARCHAR);
                duckdb_logical_type t_k    = duckdb_create_logical_type(key_str ? DUCKDB_TYPE_VARCHAR : DUCKDB_TYPE_BIGINT);
                duckdb_logical_type t_v    = duckdb_create_logical_type(DUCKDB_TYPE_BIGINT);
                duckdb_logical_type t_i    = duckdb_create_logical_type(DUCKDB_TYPE_BIGINT);
                duckdb_logical_type t_d    = duckdb_create_logical_type(DUCKDB_TYPE_DOUBLE);
                duckdb_logical_type t_s    = duckdb_create_logical_type(DUCKDB_TYPE_VARCHAR);
                duckdb_logical_type t_pi   = duckdb_create_list_type(t_i);
                duckdb_logical_type t_pf   = duckdb_create_list_type(t_d);
                duckdb_logical_type t_ps   = duckdb_create_list_type(t_s);
                duckdb_logical_type t_ret  = duckdb_create_logical_type(DUCKDB_TYPE_BIGINT);
                duckdb_aggregate_function_add_parameter(rfn, t_name);
                duckdb_aggregate_function_add_parameter(rfn, t_k);
                duckdb_aggregate_function_add_parameter(rfn, t_v);
                duckdb_aggregate_function_add_parameter(rfn, t_pi);
                duckdb_aggregate_function_add_parameter(rfn, t_pf);
                if (with_ps) duckdb_aggregate_function_add_parameter(rfn, t_ps);
                duckdb_aggregate_function_set_return_type(rfn, t_ret);
                for (auto* t : { &t_name, &t_k, &t_v, &t_i, &t_d, &t_s, &t_pi, &t_pf, &t_ps, &t_ret })
                    duckdb_destroy_logical_type(t);
                duckdb_aggregate_function_set_functions(rfn,
                    upload_state_size, upload_state_init, update,
                    upload_combine, upload_rows_exact_finalize);
                duckdb_aggregate_function_set_destructor(rfn, upload_state_destroy);
                duckdb_aggregate_function_set_special_handling(rfn);
                duckdb_aggregate_function_set_extra_info(rfn, resident_extra_info(ctx),
                                                         resident_extra_info_destroy);
                return rfn;
            };
            duckdb_aggregate_function_set rset = duckdb_create_aggregate_function_set("gpu_upload_rows_exact");
            duckdb_aggregate_function r5  = make_rows_fn(false, false, upload_rows_exact_update);
            duckdb_aggregate_function r6i = make_rows_fn(false, true,  upload_rows_exact_update6<false>);
            duckdb_aggregate_function r6s = make_rows_fn(true,  true,  upload_rows_exact_update6<true>);
            duckdb_state a5 = duckdb_add_aggregate_function_to_set(rset, r5);
            duckdb_state a6 = duckdb_add_aggregate_function_to_set(rset, r6i);
            duckdb_state a7 = duckdb_add_aggregate_function_to_set(rset, r6s);
            duckdb_destroy_aggregate_function(&r5);
            duckdb_destroy_aggregate_function(&r6i);
            duckdb_destroy_aggregate_function(&r6s);
            if (a5 == DuckDBError || a6 == DuckDBError || a7 == DuckDBError) {
                duckdb_destroy_aggregate_function_set(&rset);
                throw std::runtime_error("gpu_upload_rows_exact overload set assembly failed");
            }
            duckdb_state rst = duckdb_register_aggregate_function_set(con, rset);
            duckdb_destroy_aggregate_function_set(&rset);
            if (rst == DuckDBError) {
                throw std::runtime_error("gpu_upload_rows_exact registration failed");
            }
        }
        // gpu_upload_columns(tag, rowid BIGINT, ci BIGINT[], cf DOUBLE[], cs VARCHAR[]) -> BIGINT (stage B)
        {
            duckdb_aggregate_function cfn = duckdb_create_aggregate_function();
            duckdb_aggregate_function_set_name(cfn, "gpu_upload_columns");
            duckdb_logical_type t_name = duckdb_create_logical_type(DUCKDB_TYPE_VARCHAR);
            duckdb_logical_type t_r    = duckdb_create_logical_type(DUCKDB_TYPE_BIGINT);
            duckdb_logical_type t_i    = duckdb_create_logical_type(DUCKDB_TYPE_BIGINT);
            duckdb_logical_type t_d    = duckdb_create_logical_type(DUCKDB_TYPE_DOUBLE);
            duckdb_logical_type t_s    = duckdb_create_logical_type(DUCKDB_TYPE_VARCHAR);
            duckdb_logical_type t_ci   = duckdb_create_list_type(t_i);
            duckdb_logical_type t_cf   = duckdb_create_list_type(t_d);
            duckdb_logical_type t_cs   = duckdb_create_list_type(t_s);
            duckdb_logical_type t_ret  = duckdb_create_logical_type(DUCKDB_TYPE_BIGINT);
            duckdb_aggregate_function_add_parameter(cfn, t_name);
            duckdb_aggregate_function_add_parameter(cfn, t_r);
            duckdb_aggregate_function_add_parameter(cfn, t_ci);
            duckdb_aggregate_function_add_parameter(cfn, t_cf);
            duckdb_aggregate_function_add_parameter(cfn, t_cs);
            duckdb_aggregate_function_set_return_type(cfn, t_ret);
            for (auto* t : { &t_name, &t_r, &t_i, &t_d, &t_s, &t_ci, &t_cf, &t_cs, &t_ret })
                duckdb_destroy_logical_type(t);
            duckdb_aggregate_function_set_functions(cfn, upload_state_size, upload_state_init,
                                                    upload_columns_update, upload_combine, upload_columns_finalize);
            duckdb_aggregate_function_set_destructor(cfn, upload_state_destroy);
            duckdb_aggregate_function_set_special_handling(cfn);
            duckdb_aggregate_function_set_extra_info(cfn, resident_extra_info(ctx), resident_extra_info_destroy);
            duckdb_state cst = duckdb_register_aggregate_function(con, cfn);
            duckdb_destroy_aggregate_function(&cfn);
            if (cst == DuckDBError) throw std::runtime_error("gpu_upload_columns registration failed");
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
    register_scalar_names(con, "gpu_join_materialize",
        join_materialize_exec, DUCKDB_TYPE_BIGINT, 6, ctx);
    register_scalar(con, "gpu_note_rows", note_rows_exec, DUCKDB_TYPE_BOOLEAN,
                    {DUCKDB_TYPE_VARCHAR, DUCKDB_TYPE_BIGINT}, ctx);
    register_scalar(con, "gpu_drop_column", drop_column_exec, DUCKDB_TYPE_BOOLEAN,
                    {DUCKDB_TYPE_VARCHAR, DUCKDB_TYPE_VARCHAR}, ctx);
    register_scalar(con, "gpu_resident_dict_component", dict_component_exec, DUCKDB_TYPE_VARCHAR,
                    {DUCKDB_TYPE_VARCHAR, DUCKDB_TYPE_BIGINT, DUCKDB_TYPE_BIGINT}, ctx);
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

    // gpu_resident_dictionary(tag, n) table function (v0.7 §4.5).
    {
        duckdb_table_function tf = duckdb_create_table_function();
        duckdb_table_function_set_name(tf, "gpu_resident_dictionary");
        duckdb_logical_type vc = duckdb_create_logical_type(DUCKDB_TYPE_VARCHAR);
        duckdb_logical_type bi = duckdb_create_logical_type(DUCKDB_TYPE_BIGINT);
        duckdb_table_function_add_parameter(tf, vc);
        duckdb_table_function_add_parameter(tf, bi);
        duckdb_destroy_logical_type(&vc);
        duckdb_destroy_logical_type(&bi);
        duckdb_table_function_set_bind(tf, dictionary_bind);
        duckdb_table_function_set_init(tf, dictionary_init);
        duckdb_table_function_set_function(tf, dictionary_function);
        duckdb_table_function_set_extra_info(tf, resident_extra_info(ctx), resident_extra_info_destroy);
        if (duckdb_register_table_function(con, tf) == DuckDBError) {
            duckdb_destroy_table_function(&tf);
            throw std::runtime_error("gpu_resident_dictionary registration failed");
        }
        duckdb_destroy_table_function(&tf);
    }

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
    {
        duckdb_table_function tf = duckdb_create_table_function();
        duckdb_table_function_set_name(tf, "gpu_store_columns");
        duckdb_table_function_set_bind(tf, store_columns_bind);
        duckdb_table_function_set_init(tf, store_columns_init);
        duckdb_table_function_set_function(tf, store_columns_function);
        duckdb_table_function_set_extra_info(tf, resident_extra_info(ctx), resident_extra_info_destroy);
        if (duckdb_register_table_function(con, tf) == DuckDBError) {
            duckdb_destroy_table_function(&tf);
            throw std::runtime_error("gpu_store_columns registration failed");
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
