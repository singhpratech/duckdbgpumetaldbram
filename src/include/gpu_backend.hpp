// gpu_backend.hpp — abstract backend interface
//
// Backends implement this interface. The factory `make_aggregator(Backend)`
// returns the requested backend, or throws if unavailable.
//
// Shared file: any change that breaks the ABI must be coordinated between
// Linux (CUDA) and macOS (Metal) Claude Code instances via PR.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace gpudb {

enum class Backend : std::uint8_t {
    CPU   = 0,
    CUDA  = 1,
    METAL = 2,
};

[[nodiscard]] const char* to_string(Backend b) noexcept;

enum class Dtype : std::uint8_t { I64 = 0, F64 = 1 };

// Returned by Aggregator methods so backends can attach diagnostics
// (kernel time, transfer time, etc.) without C++ exceptions on the hot path.
struct AggResult {
    std::int64_t value_i64;     // result for integer aggregations
    double       value_f64;     // result for floating aggregations
    std::size_t  rows;          // input row count
    double       wall_ms;       // total wall time
    double       kernel_ms;     // GPU kernel time only (0 for CPU)
    double       transfer_ms;   // host<->device transfer time (0 for CPU/Metal-UMA/resident)
};

// Returned by Aggregator::agg_all_i64 — fused SUM + MIN + MAX + COUNT in
// a single pass over the input. The big win is that each int64 is read
// from memory exactly ONCE, halving (or quartering) DRAM traffic vs the
// "call sum/min/max separately" pattern.
struct AggAllResult {
    std::int64_t sum;           // SUM(values)
    std::int64_t min;           // MIN(values), int64 max if rows == 0
    std::int64_t max;           // MAX(values), int64 min if rows == 0
    std::size_t  count;         // number of values read (== rows for non-null)
    std::size_t  rows;          // input row count
    double       wall_ms;       // total wall time
    double       kernel_ms;     // GPU kernel time only (0 for CPU)
    double       transfer_ms;   // host<->device transfer time (0 for CPU/Metal-UMA/resident)
};

// Join flavor for the fused resident join-aggregates. With a PROBE-SIDE
// payload, every SQL join type reduces to one formula: each probe row's
// contribution multiplier c is a function of its build-side match count m —
//   INNER c = m          (payload repeated per matching build row)
//   LEFT  c = max(m, 1)  (unmatched probe rows keep their payload once)
//   SEMI  c = (m > 0)    (EXISTS: payload once if any match)
//   ANTI  c = (m == 0)   (NOT EXISTS: payload once if no match)
// sum += c * payload[i]; matched += c. RIGHT and FULL OUTER compose from
// these: their probe-payload sum equals INNER's / LEFT's respectively
// (unmatched build rows contribute NULL payload), and the extra COUNT(*)
// term (# unmatched build rows) is an ANTI count with probe/build swapped.
enum class JoinKind : std::uint8_t { INNER = 0, LEFT = 1, SEMI = 2, ANTI = 3 };

// Returned by Aggregator::join_sum_resident_* — fused equi-join + SUM over
// resident columns. Multiplicity-aware per the JoinKind table above
// (INNER matches SELECT sum(p.payload) FROM probe p JOIN build b
// ON p.key = b.key).
// Cross-backend binding rules (CPU is the executable reference):
//   - matched = TOTAL matched pairs, i.e. the join's COUNT(*) — directly
//     checkable against native.
//   - Both the per-row multiply (m * payload[i]) and the accumulate are
//     performed in uint64 two's-complement arithmetic (signed overflow is UB;
//     unsigned wrap is the defined behavior, same rule as sum_i64).
//   - matched == 0 means "no joined rows": the SQL layer must surface SUM as
//     NULL in that case (sum is 0 here but carries no meaning).
//   - `matched` counts the CONTRIBUTING rows/pairs for the given JoinKind —
//     the result cardinality of that kind's SQL query (for ANTI that is the
//     number of NON-matching probe rows; do not read it as "pairs").
//   - f64 sums are tolerance-checked across backends (relative epsilon,
//     ~1e-9 on warm data), never bit-compared: accumulation order is
//     backend-defined (CUDA atomic/tree reductions are nondeterministic
//     run-to-run; Metal streams sequentially). i64 wrap arithmetic is exact
//     and must bit-match everywhere.
struct JoinAggResult {
    std::int64_t sum;           // Σ payload[i] * multiplicity(probe_keys[i]) (i64 op)
    double       sum_f64;       // same, for the f64-payload op (0.0 otherwise)
    std::int64_t matched;       // Σ multiplicity(probe_keys[i]) (= joined COUNT(*))
    std::size_t  rows_probe;    // probe-side row count
    std::size_t  rows_build;    // build-side row count
    double       wall_ms;       // total wall time
    double       kernel_ms;     // GPU kernel time only (0 for CPU)
    double       transfer_ms;   // host<->device transfer time (0 for CPU/Metal-UMA/resident)
};

// Returned by Aggregator::join_rows_resident — the row-returning join.
// One entry per OUTPUT ROW of the kind's SQL query, in unspecified order:
//   INNER: (probe_idx, build_idx) per matching pair (full multiplicity)
//   LEFT:  matching pairs, plus (probe_idx, -1) for unmatched probe rows
//   SEMI:  (probe_idx, -1) for each probe row with >= 1 match
//   ANTI:  (probe_idx, -1) for each probe row with no match
// build_idx -1 encodes SQL NULL. Indices refer to the ORIGINAL upload order
// of each resident column.
struct JoinRowsResult {
    std::vector<std::int64_t> probe_idx;
    std::vector<std::int64_t> build_idx;   // -1 = NULL (no build match)
    std::size_t rows_probe  = 0;
    std::size_t rows_build  = 0;
    double wall_ms = 0.0, kernel_ms = 0.0, transfer_ms = 0.0;
};

// Returned by Aggregator::groupby_*_resident — one entry per distinct key,
// sorted by key ascending on every backend (so cross-backend parity is a
// plain ordered row diff). Which value vectors are filled depends on the op:
//   groupby_sum_resident_i64: keys, sums, counts
//   groupby_sum_resident_f64: keys, sums_f64, counts
//   groupby_count_resident:   keys, counts
//   groupby_exact_resident:   keys, key_null, sums (LOW limb), sums_hi,
//                             counts (= count(v)), counts_star, mins, maxs
// The exact op (v0.7, docs/TRANSPARENT_DESIGN.md §4.1/§4.2) carries the
// full per-group tuple native GROUP BY needs:
//   - the sum is a 128-bit two's-complement integer: sums[i] holds the low
//     64 bits (as the uint64 bit pattern), sums_hi[i] the high 64 bits —
//     exactly a DuckDB HUGEINT {lower, upper}; it never wraps;
//   - counts[i] is count(v) (payloads that are not NULL), counts_star[i] is
//     count(*) (every row of the group); a group whose payload is entirely
//     NULL has counts[i] == 0 and its sum/min/max are then MEANINGLESS
//     (SQL surfaces NULL) — never read them as the identity;
//   - key_null[i] == 1 marks the NULL-key group (there is at most one, its
//     keys[i] is 0 and meaningless); it sorts after every valid key, as
//     native ORDER BY key (NULLS LAST) does. key_null is sized like keys.
struct GroupByResidentResult {
    std::vector<std::int64_t> keys;
    std::vector<std::int64_t> sums;       // uint64 wrap-add, bit-exact (exact op: low limb)
    std::vector<double>       sums_f64;   // backend-ordered, tolerance-checked
    std::vector<std::int64_t> counts;     // exact op: count(v)
    std::vector<std::int64_t> sums_hi;    // exact op only: high limb of the 128-bit sum
    std::vector<std::int64_t> counts_star;// exact op only: count(*)
    std::vector<std::int64_t> mins;       // exact op only: min(v) over non-NULL payloads
    std::vector<std::int64_t> maxs;       // exact op only: max(v) over non-NULL payloads
    std::vector<std::uint8_t> key_null;   // exact op only: 1 = this row is the NULL-key group
    std::size_t rows_in = 0;
    std::size_t groups_total = 0;         // distinct keys BEFORE any GroupByFilter
    double wall_ms = 0.0, kernel_ms = 0.0, transfer_ms = 0.0;
};

// Returned by Aggregator::aggregate_exact_masked (v0.7 §4.12): the one group
// of an aggregate without GROUP BY. Every per-payload vector holds n_pays
// entries in the caller's payload order; count_star is the shared count of
// surviving rows and is filled once.
struct GlobalAggResult {
    std::vector<std::int64_t> sums;        // low limb of the 128-bit sum, per payload
    std::vector<std::int64_t> sums_hi;     // high limb
    std::vector<std::int64_t> counts;      // count(payload): non-NULL cells among the survivors
    std::vector<std::int64_t> mins;        // 0 where counts == 0
    std::vector<std::int64_t> maxs;
    std::int64_t count_star = 0;           // surviving rows
    std::size_t  rows_in = 0;
    double wall_ms = 0.0, kernel_ms = 0.0;
};

// ---- 128-bit two's-complement sum helpers (§4.2) ----
// The exact op's sum is {lo = low 64 bits as uint64, hi = high 64 bits
// signed} — the DuckDB HUGEINT layout. Every backend accumulates with the
// same arithmetic so limbs bit-match; the CPU reference and the host-side
// filter use these.
struct Sum128 {
    std::uint64_t lo = 0;
    std::int64_t  hi = 0;
    // Add a sign-extended int64.
    void add(std::int64_t v) noexcept {
        const std::uint64_t u = static_cast<std::uint64_t>(v);
        const std::uint64_t old = lo;
        lo += u;
        hi += (v < 0 ? -1 : 0) + (lo < old ? 1 : 0);
    }
    static Sum128 from_i64(std::int64_t v) noexcept {
        return Sum128{static_cast<std::uint64_t>(v), v < 0 ? -1 : 0};
    }
    // The DOUBLE value native DuckDB produces for this HUGEINT — what
    // avg(BIGINT) returns as sum::DOUBLE / count. DuckDB's hugeint -> double
    // cast is NOT correctly rounded: it is `double(lower) + double(upper) *
    // 2^64` (two roundings), with upper == -1 special-cased as
    // `-double(UINT64_MAX - lower) - 1` so small negatives stay exact.
    // Reproduced here so avg matches native to the last bit (verified
    // against native on 300k rows with 128-bit group sums, test/sql/
    // gpu_groupby_exact.test). Do not "fix" this to a correctly rounded
    // conversion: that would differ from native by 1 ulp on large sums.
    double to_double() const noexcept {
        if (hi == -1)
            return -static_cast<double>(~std::uint64_t{0} - lo) - 1.0;
        return static_cast<double>(lo) +
               static_cast<double>(hi) * 18446744073709551616.0;   // 2^64
    }
};
inline bool operator<(const Sum128& a, const Sum128& b) noexcept {
    return a.hi != b.hi ? a.hi < b.hi : a.lo < b.lo;
}
inline bool operator==(const Sum128& a, const Sum128& b) noexcept {
    return a.hi == b.hi && a.lo == b.lo;
}

// Optional device-side post-filter for the resident GROUP BY ops: HAVING on
// the aggregate and/or "the k groups with the largest / smallest aggregate".
// Applied BEFORE any device→host copy, so a 75M-group GROUP BY that keeps
// 3,000 groups moves 3,000 rows. The aggregate compared is the op's own:
// the i64 sum (groupby_sum_resident_i64, threshold_i64), the f64 sum
// (groupby_sum_resident_f64, threshold_f64), the count
// (groupby_count_resident, threshold_i64).
//   cmp   : keep groups whose aggregate satisfies  aggregate <cmp> threshold
//           (None = keep all);
//   topk  : after cmp, keep only the k groups with the largest (topk_desc)
//           or smallest aggregate; 0 = all survivors. k > survivors returns
//           every survivor.
// Output order: sorted by key ASCENDING when topk == 0 (as without a
// filter); when topk > 0, sorted by aggregate (descending iff topk_desc),
// with tie order among equal aggregates UNSPECIFIED and backend-defined —
// cross-backend checks compare the multiset of (key, aggregate) rows.
// f64 comparisons use the backend-ordered sum, so a group whose f64 sum
// sits within rounding of the threshold may be kept by one backend and
// dropped by another (same tolerance caveat as the sums themselves).
// NaN rule (DuckDB's total order, for cmp AND top-k): every NaN sum,
// whatever its sign bit, is the GREATEST value and equal to any other NaN;
// -0.0 and 0.0 are equal. So `> t` / `>= t` keep NaN groups (as native
// HAVING does), `< t` / `<= t` drop them, a NaN threshold with `<=` keeps
// every group and with `>=` keeps only NaN groups; DESC lists NaN groups
// first, ASC last. Metal finishes f64 sums on the host and applies the f64
// filter there (the reference implementation) — the device-side guarantee
// above holds for the i64 sum and count ops on every backend.
// max_groups bounds the rows RETURNED (the survivors), not the number of
// groups; GroupByResidentResult::groups_total reports the unfiltered count.
//
// `agg` is read ONLY by groupby_exact_resident, whose tuple has several
// aggregates: it names the one cmp and topk apply to. Sum compares the full
// 128-bit sum against threshold_i64 sign-extended (a HUGEINT threshold
// outside int64 is not expressible — the SQL layer declines such shapes);
// CountV / CountStar / Min / Max compare the int64; Avg compares the
// exact-sum-over-count double against threshold_f64 (DuckDB total order).
// A group whose aggregate is NULL (count(v) == 0 for sum/min/max/avg) never
// satisfies cmp (SQL three-valued HAVING) and sorts LAST under top-k in both
// directions (NULLS LAST, DuckDB's default). The legacy ops ignore `agg`.
struct GroupByFilter {
    enum class Cmp : std::uint8_t { None, GT, GE, LT, LE };
    enum class Agg : std::uint8_t { Sum, CountV, CountStar, Min, Max, Avg };
    Cmp          cmp = Cmp::None;
    std::int64_t threshold_i64 = 0;
    double       threshold_f64 = 0.0;
    std::size_t  topk = 0;
    bool         topk_desc = true;
    Agg          agg = Agg::Sum;
    // Exact ops only: which result vectors the caller will read, as a bit
    // set — bit 0 keys (+ key_null), 1 sums + sums_hi, 2 counts, 3
    // counts_star, 4 mins, 5 maxs. A backend may leave an unwanted vector
    // EMPTY (it still computes the whole tuple on the device; this only
    // skips allocating and writing host memory nobody reads — at millions of
    // groups that memory costs more than the reduce). Filters read what they
    // need regardless. kAllColumns = every vector filled.
    static constexpr std::uint32_t kAllColumns = 0x3Fu;
    std::uint32_t columns = kAllColumns;
    bool wants(unsigned bit) const { return (columns >> bit) & 1u; }
    bool active() const { return cmp != Cmp::None || topk != 0; }
};

// v0.7 §4.6: one comparison of a resident predicate column against a
// constant, for the device-side WHERE mask of the exact GROUP BY. A
// conjunction is an array of these. `col` is a column of the SAME exact set
// as the keys/payload (row-aligned by upload_rows_exact), I64 or F64:
//   - I64 column: `value` / `list` are int64 constants;
//   - F64 column: `value` / `list` hold the IEEE-754 bits of the double
//     constants, and the comparison uses DuckDB's total order — every NaN is
//     the greatest value and equal to any NaN, -0.0 == 0.0;
//   - a NULL cell fails every comparison and In (SQL three-valued WHERE);
//     IsNull / IsNotNull test the validity bit only; list values are
//     non-NULL (the SQL layer drops a NULL literal from an IN list, whose
//     only effect in a WHERE is "not true").
class ResidentColumn;
struct Predicate {
    enum class Op : std::uint8_t { EQ, NE, LT, LE, GT, GE, IsNull, IsNotNull, In };
    const ResidentColumn* col = nullptr;
    Op                    op = Op::EQ;
    std::int64_t          value = 0;
    const std::int64_t*   list = nullptr;   // In: n_list constants, same encoding as value
    std::size_t           n_list = 0;
};

// ---- v0.7 milestone 5: the materialised key join (docs/TRANSPARENT_DESIGN.md §4.8) ----
// One output lane of Aggregator::join_materialize: a column of the PROBE set
// (row-aligned with probe_key) or of the BUILD set (row-aligned with
// build_key, read through the match).
struct JoinLane {
    const ResidentColumn* col = nullptr;
    bool                  from_build = false;
};
// One payload of Aggregator::groupby_exact_masked_multi (v0.7 §4.9): a BIGINT
// column row-aligned with the keys, and the result vectors the caller reads
// for it (GroupByFilter::columns bits 1 sum, 2 count, 4 min, 5 max).
struct MultiPayload {
    const ResidentColumn* vals = nullptr;
    std::uint32_t         columns = GroupByFilter::kAllColumns;
};

struct JoinMaterializeResult {
    std::vector<std::unique_ptr<ResidentColumn>> lanes;   // n_out columns, lane order
    std::size_t rows_probe = 0;
    std::size_t rows_build = 0;
    std::size_t rows_out = 0;        // probe rows that matched
    std::size_t null_key_rows = 0;   // of those, rows whose lane-0 cell is NULL (the suffix)
    double      wall_ms = 0.0;
    double      kernel_ms = 0.0;
};

// Returned by Aggregator::topk_resident — k rows in the requested order.
// idx is the ORIGINAL upload-order index of each row; values_i64 or
// values_f64 is filled according to the column's dtype.
struct TopKResult {
    std::vector<std::int64_t> idx;
    std::vector<std::int64_t> values_i64;
    std::vector<double>       values_f64;
    std::size_t rows_in = 0;
    double wall_ms = 0.0, kernel_ms = 0.0, transfer_ms = 0.0;
};

// Opaque handle to a column resident in backend memory.
// Owns the storage; destruction releases device memory.
// Created by Aggregator::upload_*; must only be used with the SAME aggregator
// that created it (enforced via backend_tag()).
class ResidentColumn {
public:
    virtual ~ResidentColumn() = default;
    [[nodiscard]] virtual Backend     backend_tag() const noexcept = 0;
    [[nodiscard]] virtual Dtype       dtype()       const noexcept = 0;
    [[nodiscard]] virtual std::size_t rows()        const noexcept = 0;

    // ---- v0.7 milestone 0b: readiness (docs/TRANSPARENT_DESIGN.md §5.5) ----
    // Backends may keep derived structures on a column (CUDA: the sorted-key
    // + permutation cache every GROUP BY / join / top-k call needs). Those
    // are built lazily on first use today, which makes the FIRST query after
    // an upload pay seconds of sorting. prepare() builds them ahead of time,
    // on the backend's own stream/queue, and returns only when they are
    // complete and visible to every other stream (no event handshake is
    // needed by the caller). Contract:
    //   * idempotent and thread-safe: concurrent prepare() calls on one
    //     column serialize on a per-column lock, the second is a no-op;
    //   * throws std::runtime_error on failure (device OOM, fault) and
    //     leaves the column usable — operators then build lazily as before;
    //   * prepared() reports whether the derived structures exist. A column
    //     with nothing to derive (CPU) is prepared from birth.
    // resident_bytes() is the backend memory the column holds INCLUDING any
    // derived structures — the number a device memory budget accounts for.
    // Defaults keep every existing backend building unchanged (CPU has
    // nothing to prepare; Metal overrides when its perm cache moves here).
    virtual void prepare() {}
    [[nodiscard]] virtual bool prepared() const noexcept { return true; }
    [[nodiscard]] virtual std::size_t resident_bytes() const noexcept {
        return rows() * 8;   // i64 and f64 are both 8 bytes wide
    }

    // ---- v0.7 milestone 3: NULL-aware columns (§4.1) ----
    // Number of NULL rows the column carries; rows() INCLUDES them. Columns
    // from upload_i64/upload_f64/upload_pair_interleaved never carry NULLs
    // (0). Exact-path columns (upload_pair_exact / upload_rows_exact) keep
    // their rows in input order; a NULL key or payload is a zero bit in the
    // backend's validity bitmap for that column (docs/RESIDENT_COLUMNS_DESIGN.md,
    // stage A — before it, NULL-key rows were moved to a trailing block). The
    // legacy resident ops refuse a column with null_count() > 0 (they have no
    // NULL semantics).
    [[nodiscard]] virtual std::size_t null_count() const noexcept { return 0; }
};

// Minimal aggregator surface for week 1.
// Future: GROUP BY, JOIN, WINDOW. Keep this tight; do not add ops casually.
class Aggregator {
public:
    virtual ~Aggregator() = default;

    [[nodiscard]] virtual Backend backend() const noexcept = 0;
    [[nodiscard]] virtual std::string device_name() const = 0;

    // ---- One-shot API (transfer + kernel per call) ----
    virtual AggResult sum_i64(const std::int64_t* data, std::size_t n) = 0;
    virtual AggResult min_i64(const std::int64_t* data, std::size_t n) = 0;
    virtual AggResult max_i64(const std::int64_t* data, std::size_t n) = 0;
    virtual AggResult sum_f64(const double*       data, std::size_t n) = 0;

    // ---- Resident-column API (transfer once, query many times) ----
    // upload_* materializes the data in backend memory and returns a handle.
    // *_resident operate on that handle without re-transfer.
    virtual std::unique_ptr<ResidentColumn>
        upload_i64(const std::int64_t* data, std::size_t n) = 0;
    virtual std::unique_ptr<ResidentColumn>
        upload_f64(const double* data, std::size_t n) = 0;

    virtual AggResult sum_resident_i64(const ResidentColumn&) = 0;
    virtual AggResult min_resident_i64(const ResidentColumn&) = 0;
    virtual AggResult max_resident_i64(const ResidentColumn&) = 0;
    virtual AggResult sum_resident_f64(const ResidentColumn&) = 0;

    // ---- v0.7 milestone 0b: pair upload without a host de-interleave ----
    // gpu_upload_pair buffers (key, payload) INTERLEAVED in fixed-size
    // segments — kv[2i] is the key, kv[2i+1] the payload's raw 8 bytes
    // (int64, or the IEEE-754 bits of a double when vdt == F64) — and never
    // concatenates them on the host. Splitting the pair into two columns on
    // the host would materialise two more copies of the data (page-faulted,
    // then read back by the H2D copy): ~1 GB of transient host memory and
    // ~0.5 s of one core on a 60M-row upload. Backends that can split on the
    // device override this (CUDA: one H2D per segment into a staging buffer,
    // one kernel); the default keeps every existing backend correct by
    // concatenating and de-interleaving on the host.
    struct KvSpan {
        const std::int64_t* kv = nullptr;   // 2 * rows lanes
        std::size_t         rows = 0;
        // §4.1 (read ONLY by upload_pair_exact; upload_pair_interleaved
        // ignores them): validity bitmaps for this span, DuckDB's layout —
        // row i is valid iff bit (i % 64) of word i / 64 is set. nullptr =
        // every row of the span is valid. A NULL row's kv lanes hold
        // unspecified bits and must not be read.
        const std::uint64_t* key_valid = nullptr;
        const std::uint64_t* val_valid = nullptr;
    };
    struct ResidentPair {
        std::unique_ptr<ResidentColumn> keys;
        std::unique_ptr<ResidentColumn> vals;
    };
    virtual ResidentPair upload_pair_interleaved(const KvSpan* spans, std::size_t n_spans,
                                                 Dtype vdt) {
        std::size_t rows = 0;
        for (std::size_t i = 0; i < n_spans; ++i) rows += spans[i].rows;
        std::vector<std::int64_t> k(rows);
        ResidentPair out;
        std::size_t r = 0;
        if (vdt == Dtype::I64) {
            std::vector<std::int64_t> v(rows);
            for (std::size_t i = 0; i < n_spans; ++i)
                for (std::size_t j = 0; j < spans[i].rows; ++j, ++r) {
                    k[r] = spans[i].kv[2 * j];
                    v[r] = spans[i].kv[2 * j + 1];
                }
            out.vals = upload_i64(v.data(), rows);
        } else {
            std::vector<double> v(rows);
            for (std::size_t i = 0; i < n_spans; ++i)
                for (std::size_t j = 0; j < spans[i].rows; ++j, ++r) {
                    k[r] = spans[i].kv[2 * j];
                    std::memcpy(&v[r], &spans[i].kv[2 * j + 1], sizeof(double));
                }
            out.vals = upload_f64(v.data(), rows);
        }
        out.keys = upload_i64(k.data(), rows);
        return out;
    }

    // ---- Multi-aggregate fusion (SUM + MIN + MAX + COUNT in one pass) ----
    // Each int64 is read from memory ONCE; backends compute all four
    // aggregates simultaneously. Wins over calling sum/min/max separately
    // by 2-3x on memory-bandwidth-bound work, and proportionally more
    // when more aggregates are fused.
    virtual AggAllResult agg_all_i64(const std::int64_t* data, std::size_t n) = 0;
    virtual AggAllResult agg_all_resident_i64(const ResidentColumn&) = 0;

    // ---- Fused resident join-aggregate (v0.5) ----
    // Inner equi-join with full build-side multiplicity, fused with SUM.
    // payload is probe-side and must have the same row count as probe_keys
    // (throws std::runtime_error otherwise, as do dtype/backend-tag
    // mismatches — same error style as the other resident ops). Resident
    // columns carry no NULLs, so no NULL-key semantics apply here.
    // Backends may cache derived build-side structures (sorted copy, hash
    // table) privately on the ResidentColumn; such caches die with the
    // column (gpu_drop_resident) and do NOT count against the host-side
    // GPUDB_UPLOAD_POOL_MAX_MB cap; a device-OOM while building one must
    // surface as a clean std::runtime_error.
    // Default implementation throws: backends opt in by overriding (CPU and
    // Metal implement it; CUDA pending — keeping this non-pure means the
    // CUDA backend builds unchanged until its implementation lands). The
    // hybrid planner catches the throw and falls back to CPU.
    // NOTE: the `kind = INNER` default argument binds by STATIC type in C++;
    // overrides must not declare a different default (callers always go
    // through this base interface, so the base default is the only one used).
    virtual JoinAggResult join_sum_resident_i64(const ResidentColumn& probe_keys,
                                                const ResidentColumn& payload,
                                                const ResidentColumn& build_keys,
                                                JoinKind kind = JoinKind::INNER);

    // f64-payload flavor: keys are still I64 (join keys are integers in
    // practice); payload is an F64 resident column, result in sum_f64.
    // Multiplicity contributes as m * payload[i] in double arithmetic; the
    // accumulation order is backend-defined, so cross-engine comparisons use
    // a relative tolerance, not equality (same caveat as sum_resident_f64).
    // On Metal this runs as a parallel host pass over UMA buffers (no fp64
    // in MSL); CUDA can run it natively on device.
    virtual JoinAggResult join_sum_resident_f64(const ResidentColumn& probe_keys,
                                                const ResidentColumn& payload,
                                                const ResidentColumn& build_keys,
                                                JoinKind kind = JoinKind::INNER);

    // Row-returning resident join (see JoinRowsResult for per-kind output).
    // max_rows caps the materialized output; exceeding it throws a clean
    // error naming the actual row count (never silently truncates). Same
    // default-throwing / default-arg rules as the fused variants above.
    virtual JoinRowsResult join_rows_resident(const ResidentColumn& probe_keys,
                                              const ResidentColumn& build_keys,
                                              JoinKind kind,
                                              std::size_t max_rows);

    // ---- Resident GROUP BY / top-k (v0.6) ----
    // SELECT key, SUM(val), COUNT(*) FROM pair GROUP BY key, over resident
    // columns (keys I64; vals I64 or F64). Output contract, every backend:
    //   - one row per distinct key, rows sorted by key ASCENDING;
    //   - i64 sums accumulate in uint64 two's-complement (defined wrap,
    //     bit-exact across backends); f64 sums are backend-ordered and
    //     compared under the ~1e-9 relative tolerance; counts are exact;
    //   - keys/vals row counts must match (std::runtime_error otherwise, as
    //     do dtype / backend-tag mismatches);
    //   - the number of rows to return (groups, or the GroupByFilter
    //     survivors) is checked against max_groups BEFORE any device→host
    //     copy; exceeding it throws a clean error naming the actual count
    //     (never truncates).
    // groupby_count_resident is keys-only. `filter` (see GroupByFilter) is
    // applied on the device; the default keeps every group. Backends reuse the join's cached
    // sorted-key + permutation structure where they have one; that cache
    // dies with the column and is outside the upload-pool accounting.
    // Same default-throwing / hybrid-fallback rules as the join ops.
    // Not thread-safe: an Aggregator's resident ops share scratch buffers
    // and lazily build the per-column sort cache, so concurrent calls on
    // one Aggregator (even on different columns) must be serialised by the
    // caller — the SQL extension holds one mutex around every resident op.
    virtual GroupByResidentResult groupby_sum_resident_i64(const ResidentColumn& keys,
                                                           const ResidentColumn& vals,
                                                           std::size_t max_groups,
                                                           const GroupByFilter& filter = GroupByFilter{});
    virtual GroupByResidentResult groupby_sum_resident_f64(const ResidentColumn& keys,
                                                           const ResidentColumn& vals,
                                                           std::size_t max_groups,
                                                           const GroupByFilter& filter = GroupByFilter{});
    virtual GroupByResidentResult groupby_count_resident(const ResidentColumn& keys,
                                                         std::size_t max_groups,
                                                         const GroupByFilter& filter = GroupByFilter{});

    // ---- v0.7 milestone 3: exact GROUP BY (docs/TRANSPARENT_DESIGN.md §4.1, §4.2) ----
    // The pair upload that keeps NULLs. Rows stay in input order in both
    // columns; a NULL key and a NULL payload each sit under a validity bitmap
    // the backend owns (the NULL keys form one group). Both columns report
    // rows() = every input row and null_count() as documented on
    // ResidentColumn. vdt must be I64 (DOUBLE sums are not on the exact
    // path, §4.7; a DOUBLE payload throws). Default throws — backends opt in
    // (CPU is the reference); the hybrid falls back to a CPU upload.
    virtual ResidentPair upload_pair_exact(const KvSpan* spans, std::size_t n_spans, Dtype vdt);

    // SELECT key, sum(v), count(v), count(*), min(v), max(v) FROM pair
    // GROUP BY key with native semantics, over columns from
    // upload_pair_exact (keys.null_count() and vals->null_count() are the
    // only NULL sources; columns without NULLs from any upload also work).
    // vals may be nullptr: keys-only, fills keys/key_null/counts_star (and
    // counts == counts_star). Output contract, every backend:
    //   - one row per distinct key sorted ascending, then the NULL-key group
    //     last if keys carry one (key_null marks it);
    //   - sums/sums_hi is the exact 128-bit sum over non-NULL payloads,
    //     bit-identical across backends; counts/counts_star/mins/maxs exact;
    //     a group with counts[i] == 0 has unspecified sum/min/max;
    //   - `filter` applies per GroupByFilter::agg; cap and error rules as
    //     the legacy ops; not thread-safe, same as the legacy ops.
    // Default throws — same opt-in / hybrid-fallback rule.
    virtual GroupByResidentResult groupby_exact_resident(const ResidentColumn& keys,
                                                         const ResidentColumn* vals,
                                                         std::size_t max_groups,
                                                         const GroupByFilter& filter = GroupByFilter{});

    // ---- v0.7 milestone 3: predicate columns and the WHERE mask (§4.6) ----
    // Row-major upload of an exact set with n_lanes columns per row: lane 0
    // is the key, lane 1 the payload, lanes 2.. the predicate columns. Every
    // lane is 8 bytes (int64, or the IEEE-754 bits of a double when its
    // dtype is F64). `valid` holds one bitmap pointer per lane (nullptr =
    // every row valid; the array itself may be nullptr). Every returned
    // column keeps the input row order, so the columns stay row-aligned and
    // the predicate columns can be read by the key's sort permutation; a NULL
    // in any lane, the key included, is a zero bit in that column's bitmap.
    // Returned in lane order. Default throws.
    struct RowSpan {
        const std::int64_t*        lanes = nullptr;   // n_lanes * rows values
        std::size_t                rows = 0;
        std::size_t                n_lanes = 0;
        const std::uint64_t* const* valid = nullptr;  // n_lanes entries, or nullptr
        // Stage B of docs/RESIDENT_COLUMNS_DESIGN.md: a span may say where its rows
        // land (dst_row; kNext = right after the previous span, the default) and at
        // which bit of each lane's bitmap its first row sits (valid_bit: a span that
        // starts mid-segment). Destinations must not overlap and must cover
        // [0, total rows) exactly; the backend checks nothing beyond bounds.
        static constexpr std::size_t kNext = ~std::size_t{0};
        std::size_t                dst_row = kNext;
        std::size_t                valid_bit = 0;
    };
    virtual std::vector<std::unique_ptr<ResidentColumn>>
        upload_rows_exact(const RowSpan* spans, std::size_t n_spans,
                          const Dtype* dtypes, std::size_t n_lanes);

    // groupby_exact_resident with a WHERE mask: the conjunction of `preds`
    // (n_preds entries, each over a column of the same set) selects the rows
    // that take part. Semantics are native's: a row failing the mask is
    // absent from every aggregate INCLUDING count(*), and a group none of
    // whose rows pass is not emitted (the NULL-key group included).
    // n_preds == 0 is exactly groupby_exact_resident. The output contract,
    // filter and cap rules are those of groupby_exact_resident. Default:
    // delegates when n_preds == 0, throws otherwise (backends opt in).
    virtual GroupByResidentResult groupby_exact_masked_resident(const ResidentColumn& keys,
                                                                const ResidentColumn* vals,
                                                                const Predicate* preds,
                                                                std::size_t n_preds,
                                                                std::size_t max_groups,
                                                                const GroupByFilter& filter = GroupByFilter{});

    // Does this backend run the exact path (upload_rows_exact +
    // groupby_exact_[masked_]resident) on its own device? False means the
    // default-throwing stubs are in place and the hybrid would fall back to
    // the CPU reference — correct, but not the GPU, so the transparent
    // rewrite must not fire on it (rule 1). The hybrid reports its GPU side.
    [[nodiscard]] virtual bool exact_supported() const noexcept { return false; }

    // ---- v0.7 §4.9: several payload columns in one GROUP BY ----
    // groupby_exact_masked_resident over n_pays payload columns of one set:
    // SELECT k, <aggs of pays[0]>, <aggs of pays[1]>, ... WHERE mask GROUP BY k.
    // `filter` (HAVING / top-k) reads payload `filter_payload`. Returns one
    // result per payload, row-aligned: result[p] holds sums / sums_hi /
    // counts / mins / maxs of payload p (as pays[p].columns asks); the
    // shared columns — keys, key_null, counts_star, groups_total, the timing
    // diagnostics — are on result[filter_payload] only (filter.columns bits 0
    // and 3). Semantics per payload are exactly the single-payload op's. The
    // default runs one single-payload pass per column and aligns them (every
    // pass sees the same keys and mask, hence the same groups in the same
    // order; under a filter the others are looked up by key); a backend
    // overrides it to share the mask and the grouping between the payloads.
    virtual std::vector<GroupByResidentResult> groupby_exact_masked_multi(
        const ResidentColumn& keys, const MultiPayload* pays, std::size_t n_pays,
        std::size_t filter_payload, const Predicate* preds, std::size_t n_preds,
        std::size_t max_groups, const GroupByFilter& filter = GroupByFilter{});

    // ---- v0.7 §4.12: the global masked aggregate ----
    // SELECT <aggs of pays[0]>, <aggs of pays[1]>, ..., count(*) FROM set
    // WHERE <preds> — aggregates with no GROUP BY, TPC-H Q6's shape. It is
    // groupby_exact_masked_multi with exactly one group, so every semantic
    // is that operator's: a row failing the conjunction takes part in
    // nothing INCLUDING count(*); a NULL payload cell is skipped by sum /
    // count / min / max and counted by count(*); a NULL cell fails every
    // comparison and In, IsNull / IsNotNull read the validity bit; payload
    // lanes are I64, predicate lanes I64 or F64. The difference is that
    // there is no key: no sort cache is read or built, no permutation is
    // gathered, and the rows are reduced in storage order in ONE pass — the
    // predicate program is evaluated per row inside the reduce.
    //   - pays[p].columns asks for the result vectors of payload p (bit 1
    //     sum + sums_hi, 2 count, 4 min, 5 max); count_star is always filled;
    //   - a payload with count 0 has unspecified sum / min / max (the SQL
    //     layer emits NULL), and zero surviving rows gives count_star 0 and
    //     every count 0 — one result, never no result;
    //   - n_pays may be 0 (count(*) only), n_preds may be 0 (no WHERE); the
    //     row count comes from pays[0] when there is a payload and from
    //     preds[0].col otherwise, and every column must agree on it;
    //   - bit-identical across backends, as the exact GROUP BY.
    // Default throws — backends opt in; global_supported() is the rule-1
    // gate, as exact_supported().
    virtual GlobalAggResult aggregate_exact_masked(const MultiPayload* pays, std::size_t n_pays,
                                                   const Predicate* preds, std::size_t n_preds);
    [[nodiscard]] virtual bool global_supported() const noexcept { return false; }

    // Does this backend store an exact I64 lane at the narrowest signed width
    // its values fit (docs/RESIDENT_COLUMNS_DESIGN.md, stage C)? Storage width
    // is backend-private — the interface keeps saying I64 — but the wrapper's
    // PRE-upload memory estimate (§5.5) has to know whether to charge 8 bytes
    // a lane or size each lane from its DuckDB type. Reported by
    // gpu_build_info() as narrow=true|false.
    [[nodiscard]] virtual bool narrow_lanes() const noexcept { return false; }

    // ---- v0.7 milestone 5: the materialised key join (§4.8) ----
    // Inner equi-join of a PROBE row set against a BUILD row set whose join
    // key is UNIQUE among its valid cells (a primary / unique key: the
    // dimension side of a star or snowflake join). Every probe row then has
    // at most one match, so the join result is a subset of the probe rows
    // with build columns attached, and it is produced as a NEW row set in
    // the layout upload_rows_exact returns: lane 0 is the GROUP BY key (I64),
    // rows whose lane-0 cell is NULL are a suffix of EVERY output column, the
    // valid-key prefix and the suffix each keep probe order. The output
    // columns are ordinary exact columns: groupby_exact_[masked_]resident and
    // every form built on it run over the joined rows unchanged, and an
    // output lane can be the probe key of a further join (snowflake chains).
    //   - NULL join keys on either side never match (SQL equality);
    //   - a probe row without a match is absent (inner join);
    //   - an output cell is NULL iff its source cell is NULL;
    //   - lanes keep their source dtype (F64 lanes travel as raw bits);
    //   - throws std::runtime_error containing "build key not unique" when
    //     two valid build cells are equal — the caller runs native then;
    //   - bit-identical across backends, row for row.
    // probe_key and build_key must be I64. Default throws (backends opt in);
    // join_supported() is the rule-1 gate, as exact_supported().
    virtual JoinMaterializeResult join_materialize(const ResidentColumn& probe_key,
                                                   const ResidentColumn& build_key,
                                                   const JoinLane* out, std::size_t n_out);
    [[nodiscard]] virtual bool join_supported() const noexcept { return false; }
    // Device memory the wrapper's memory budget (docs/TRANSPARENT_DESIGN.md §5.5) may
    // plan against, in bytes: the working-set size on unified memory, total device memory
    // on a discrete GPU; 0 = unknown (the wrapper then uses a conservative default).
    // Reported by gpu_build_info() as device_memory=<bytes>.
    [[nodiscard]] virtual std::size_t device_memory_bytes() const noexcept { return 0; }

    // ORDER BY col [DESC] LIMIT k over a resident column (I64 or F64).
    // Returns the k smallest (descending=false) or largest values with their
    // ORIGINAL upload-order indices; k is clamped to rows(). Values arrive
    // in the requested order. Tie order among equal values is UNSPECIFIED
    // (as SQL ORDER BY without a tiebreaker) and backend-defined; cross-
    // backend checks compare the multiset of values, not idx. F64 ordering
    // is total with NaN sorting greatest (native DuckDB order).
    virtual TopKResult topk_resident(const ResidentColumn& col,
                                     std::size_t k, bool descending);
};

// Factory. Throws std::runtime_error if the requested backend wasn't compiled
// in or no compatible device is present.
std::unique_ptr<Aggregator> make_aggregator(Backend);

// =========================================================================
//  GROUP BY hash aggregate (i64 key, i64 sum value)
// =========================================================================

struct GroupByResult {
    std::vector<std::int64_t> keys;       // unique keys (order may differ across backends)
    std::vector<std::int64_t> sums;       // SUM(value) for each key
    std::size_t input_rows = 0;
    double      wall_ms     = 0.0;
    double      kernel_ms   = 0.0;
    double      transfer_ms = 0.0;
};

class GroupByAggregator {
public:
    virtual ~GroupByAggregator() = default;
    [[nodiscard]] virtual Backend backend() const noexcept = 0;
    [[nodiscard]] virtual std::string device_name() const = 0;

    // SUM(values) GROUP BY keys.
    // expected_groups is an optional hint; 0 means "guess from n".
    virtual GroupByResult groupby_sum_i64(const std::int64_t* keys,
                                          const std::int64_t* values,
                                          std::size_t n,
                                          std::size_t expected_groups = 0) = 0;
};

std::unique_ptr<GroupByAggregator> make_groupby_aggregator(Backend);

// =========================================================================
//  Window functions (v1: ROW_NUMBER() OVER (ORDER BY key ASC))
// =========================================================================
//
// First window-function operator on the project. Sirius (CIDR 2026 GPU OLAP
// paper) explicitly lacks window functions; shipping it differentiates us.
// v1 covers the simplest meaningful case — ROW_NUMBER over a single ascending
// key — to lock the interface shape. Later additions (RANK, DENSE_RANK,
// LAG/LEAD, partitioning, frame clauses) extend this surface without churn.
//
// Output semantics:
//   For input row i, output[i] is the 1-indexed rank that row i would have
//   if the input were sorted by key ASC. Ties are broken by stable order
//   (input position): earlier rows in the input get the smaller rank.
//
// Example:
//   keys   = [40, 10, 30, 10, 20]
//   sorted = [(10,1), (10,3), (20,4), (30,2), (40,0)]   // (key, orig_idx)
//   ranks  =   1       2        3        4        5
//   output[0] = 5  (key=40 → rank 5)
//   output[1] = 1  (key=10, earliest → rank 1)
//   output[2] = 4  (key=30 → rank 4)
//   output[3] = 2  (key=10, later → rank 2)
//   output[4] = 3  (key=20 → rank 3)

struct WindowResult {
    std::vector<std::int64_t> output;     // window function value per input row
    std::size_t  rows = 0;
    double       wall_ms     = 0.0;
    double       kernel_ms   = 0.0;
    double       transfer_ms = 0.0;
};

// =========================================================================
//  HASH JOIN probe (i64 keys, inner equi-join)
// =========================================================================
//
// v1 contract: inner equi-join on int64 keys, build side has UNIQUE keys
// (no duplicate handling — for each probe[i] we emit the FIRST matching
// build[j], which is unambiguous when build is unique).
//
// Backend implementation strategy varies:
//   - CPU:   std::unordered_map<int64,int64> on the build side.
//   - CUDA:  open-addressing hash table with atomicCAS<int64>.
//   - Metal: slot-lock open-addressing hash (32-bit CAS); sort-merge fallback
//            when build exceeds table capacity (see metal_hashjoin.mm).
struct JoinResult {
    std::vector<std::int64_t> probe_indices;   // matched probe-side row indices
    std::vector<std::int64_t> build_indices;   // matched build-side row indices
    std::size_t  rows_probe = 0;
    std::size_t  rows_build = 0;
    std::size_t  matched    = 0;
    double       wall_ms     = 0.0;
    double       kernel_ms   = 0.0;
    double       transfer_ms = 0.0;
};

class WindowAggregator {
public:
    virtual ~WindowAggregator() = default;
    [[nodiscard]] virtual Backend backend() const noexcept = 0;
    [[nodiscard]] virtual std::string device_name() const = 0;
    // ROW_NUMBER() OVER (ORDER BY key ASC).
    // Output row i contains the rank of input row i (1-indexed).
    virtual WindowResult row_number_i64(const std::int64_t* keys, std::size_t n) = 0;
};

std::unique_ptr<WindowAggregator> make_window_aggregator(Backend);
class HashJoinProbe {
public:
    virtual ~HashJoinProbe() = default;
    [[nodiscard]] virtual Backend backend() const noexcept = 0;
    [[nodiscard]] virtual std::string device_name() const = 0;
    // Inner equi-join: for each probe[i], find the FIRST matching build[j].
    // (No build-side duplicate handling for v1.)
    virtual JoinResult inner_join_i64(
        const std::int64_t* build_keys, std::size_t n_build,
        const std::int64_t* probe_keys, std::size_t n_probe) = 0;
};

std::unique_ptr<HashJoinProbe> make_hashjoin_probe(Backend);

// Returns the best backend available at runtime: CUDA if compiled+device,
// else METAL if compiled+device, else CPU.
[[nodiscard]] Backend default_backend() noexcept;

// Returns the list of backends compiled into this build.
[[nodiscard]] std::vector<Backend> available_backends() noexcept;

// =========================================================================
//  Hybrid CPU/GPU planner (GOAL.md item 7)
//
// Picks CPU vs GPU per call based on:
//   - N (input cardinality / row count)
//   - expected_groups (for GROUP BY)
//   - data residency (resident-column path skips PCIe/copy)
//
// The dispatch rule is a SIMPLE deterministic threshold derived from
// BENCHMARK.md numbers — no ML, no auto-tuner. The point is "we know
// our hardware well enough to pick the right backend every time".
//
// Each hybrid call records its decision in `last_decision` so callers
// (the bench, the DuckDB extension, debug tooling) can inspect it.
// =========================================================================

// Why a particular backend was chosen for a single call.
enum class DispatchReason : std::uint8_t {
    // SUM/MIN/MAX reasons
    SmallN_CpuWins        = 0,  // n below GPU launch-overhead break-even
    Cold_BelowGpuBreakeven= 1,  // cold path; n too small for GPU to win
    Cold_AboveGpuBreakeven= 2,  // cold path; n large enough for GPU to win
    Hot_GpuAlwaysWins     = 3,  // resident column → GPU
    GpuUnavailable        = 4,  // no GPU compiled in / no device
    F64_NoGpuDoubles      = 5,  // Metal lacks IEEE-754 doubles, prefer CPU
    Resident_OnCpu        = 6,  // column resides in host memory (f64 on Metal,
                                // or a GPU upload that fell back) → CPU runs it
    // GROUP BY reasons
    GroupBy_LowCard_CpuWins  = 10, // expected_groups < threshold and small n
    GroupBy_HighCard_GpuWins = 11, // expected_groups >= n / 10 → high cardinality regime
    GroupBy_Borderline_GpuTry= 12, // mid regime: try GPU but flag for tuning
    GroupBy_HugeN_CpuWins    = 13, // n above bitonic O(N log²N) crossover
    // Hash join reasons
    Join_SmallN_CpuWins      = 20, // build+probe too small for GPU launch
    Join_LargeProbe_GpuWins  = 21, // probe-heavy FK join → GPU hash path
};

[[nodiscard]] const char* to_string(DispatchReason r) noexcept;

struct DispatchDecision {
    Backend        chosen   = Backend::CPU;
    DispatchReason reason   = DispatchReason::SmallN_CpuWins;
    std::size_t    n        = 0;
    std::size_t    expected_groups = 0; // 0 = N/A or unknown
    bool           was_resident    = false;
    bool           borderline      = false; // true if a "future tuning may flip" call
};

// Hybrid scalar aggregator (SUM/MIN/MAX). Wraps CPU + GPU implementations
// and dispatches per call. Falls back to CPU if no GPU is available; in
// that case `chosen == CPU` with reason `GpuUnavailable`.
class HybridAggregator : public Aggregator {
public:
    // Inspect the dispatch decision from the most recent call. Thread-unsafe
    // by design (these aggregators are per-thread; the bench reads it after
    // each call). For SUM-class ops only.
    [[nodiscard]] virtual const DispatchDecision& last_decision() const noexcept = 0;

    // For tests / introspection: the actual GPU backend the hybrid wraps,
    // or CPU if no GPU was available at construction time.
    [[nodiscard]] virtual Backend gpu_backend() const noexcept = 0;
};

class HybridGroupByAggregator : public GroupByAggregator {
public:
    [[nodiscard]] virtual const DispatchDecision& last_decision() const noexcept = 0;
    [[nodiscard]] virtual Backend gpu_backend() const noexcept = 0;
};

class HybridHashJoinProbe : public HashJoinProbe {
public:
    [[nodiscard]] virtual const DispatchDecision& last_decision() const noexcept = 0;
    [[nodiscard]] virtual Backend gpu_backend() const noexcept = 0;
};

// Factories. Each internally constructs a CPU aggregator AND a GPU
// aggregator (GPU = default_backend() if available, else CPU=fallback).
std::unique_ptr<HybridAggregator>        make_hybrid_aggregator();
std::unique_ptr<HybridGroupByAggregator> make_hybrid_groupby_aggregator();
std::unique_ptr<HybridHashJoinProbe>     make_hybrid_hashjoin_probe();

} // namespace gpudb
