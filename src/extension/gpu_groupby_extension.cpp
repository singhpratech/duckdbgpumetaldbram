// gpu_groupby_extension.cpp — resident GROUP BY / top-k table functions (v0.6).
//
//   SELECT key, sum, count FROM gpu_groupby_sum_resident('p');
//   SELECT key, sum, count FROM gpu_groupby_sum_resident_f64('pf');
//   SELECT key, count      FROM gpu_groupby_count_resident('p');   -- or a bare column
//   SELECT idx, value      FROM gpu_topk_resident('c', 10, 'desc');     -- BIGINT column
//   SELECT idx, value      FROM gpu_topk_resident_f64('cf', 10, 'desc'); -- DOUBLE column
//   -- HAVING / top-k of groups evaluated ON THE DEVICE: only survivors come back
//   SELECT * FROM gpu_groupby_sum_resident_having('p', '>', 300);        -- HAVING sum > 300
//   SELECT * FROM gpu_groupby_sum_resident_f64_having('pf', '>=', 1e6);
//   SELECT * FROM gpu_groupby_count_resident_having('p', '<', 5);        -- HAVING count(*) < 5
//   SELECT * FROM gpu_groupby_sum_resident_topk('p', 10, 'desc');        -- ORDER BY sum DESC LIMIT 10
//   SELECT * FROM gpu_groupby_sum_resident_f64_topk('pf', 10, 'asc');
//   SELECT * FROM gpu_groupby_count_resident_topk('p', 10, 'desc');      -- ORDER BY count DESC LIMIT 10
//   (cmp is one of '>', '>=', '<', '<='; the cap applies to the rows returned;
//    topk output is ordered by the aggregate, ties in an unspecified order)
//
// The GPU produces the (key, aggregate) rows; everything downstream — WHERE,
// ORDER BY, LIMIT, further joins — is ordinary DuckDB SQL over a small
// result. Same statement-sequencing caveat as gpu_join_rows_resident: the
// uploads must run as earlier statements on the same connection (a table
// function's bind/init cannot be ordered after an aggregate in the same
// statement). Output rows are capped at GPUDB_GROUPBY_ROWS_MAX_M million
// groups (default 100) with a clean error naming the actual count.

#include "gpu_groupby_extension.hpp"
#include "gpu_resident.hpp"          // resident registry
#include "gpu_backend.hpp"
#include "exact_path_note.hpp"

#if defined(GPUDB_C_STRUCT_ABI)
DUCKDB_EXTENSION_EXTERN
#endif

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace gpudb_ext {
namespace {

std::size_t groupby_rows_cap() {
    static const std::size_t cap = [] {
        unsigned long long m = 100;   // 100M groups ≈ 2.4 GB of (key,sum,count)
        if (const char* s = std::getenv("GPUDB_GROUPBY_ROWS_MAX_M")) {
            char* end = nullptr;
            errno = 0;
            const unsigned long long v = std::strtoull(s, &end, 10);
            if (errno == 0 && end && end != s && *end == '\0' && v > 0 &&
                v <= static_cast<unsigned long long>(SIZE_MAX / 1000000)) {
                m = v;
            } else {
                std::fprintf(stderr,
                    "[gpudb] ignoring GPUDB_GROUPBY_ROWS_MAX_M='%s' (not a positive "
                    "integer in range); using %llu\n", s, m);
            }
        }
        return static_cast<std::size_t>(m) * 1000000;
    }();
    return cap;
}

std::string value_to_string(duckdb_value v) {
    char* s = duckdb_get_varchar(v);
    std::string out = s ? s : "";
    if (s) duckdb_free(s);
    return out;
}

// Resolve 'p' -> (p.k, p.v) for pair ops; for keys-only ops accept either a
// pair name (use p.k) or a bare resident column name. The returned set
// reference keeps the columns alive for the duration of the call.
struct Resolved {
    std::shared_ptr<ResidentSet> set;
    gpudb::ResidentColumn* keys = nullptr;
    gpudb::ResidentColumn* vals = nullptr;
};

Resolved resolve_pair(ResidentContext& ctx, const std::string& name, bool need_vals,
                      const char* fn) {
    Resolved r;
    r.set = resident_acquire_set(ctx, name, fn);
    if (r.set->pair) {
        r.keys = r.set->keys.get();
        r.vals = r.set->vals.get();
        return r;
    }
    if (!need_vals) {
        r.keys = r.set->keys.get();
        return r;
    }
    throw std::runtime_error(
        std::string(fn) + ": '" + name + "' is a bare resident column, not a pair — "
        "create the pair with gpu_upload_pair");
}

// ---------------------------------------------------------------------------
// GROUP BY table functions
// ---------------------------------------------------------------------------

enum class GbOp : std::uint8_t { SumI64, SumF64, Count };
enum class GbForm : std::uint8_t { Plain, Having, TopK };

struct GbBindData {
    std::string name;
    GbOp op = GbOp::SumI64;
    gpudb::GroupByFilter filter;
};

struct GbInitData {
    gpudb::GroupByResidentResult res;
    GbOp op = GbOp::SumI64;
    std::size_t offset = 0;
};

const char* gb_fn_name(GbOp op, GbForm form = GbForm::Plain) {
    switch (op) {
        case GbOp::SumI64: return form == GbForm::Having ? "gpu_groupby_sum_resident_having"
                                : form == GbForm::TopK  ? "gpu_groupby_sum_resident_topk"
                                                        : "gpu_groupby_sum_resident";
        case GbOp::SumF64: return form == GbForm::Having ? "gpu_groupby_sum_resident_f64_having"
                                : form == GbForm::TopK  ? "gpu_groupby_sum_resident_f64_topk"
                                                        : "gpu_groupby_sum_resident_f64";
        case GbOp::Count:  return form == GbForm::Having ? "gpu_groupby_count_resident_having"
                                : form == GbForm::TopK  ? "gpu_groupby_count_resident_topk"
                                                        : "gpu_groupby_count_resident";
    }
    return "gpu_groupby_resident";
}

// Parse the extra arguments of the _having / _topk forms into a GroupByFilter.
// Returns false after setting a bind error.
template <GbOp OP, GbForm FORM>
bool gb_parse_filter(duckdb_bind_info info, gpudb::GroupByFilter& f) {
    const char* fn = gb_fn_name(OP, FORM);
    if (FORM == GbForm::Plain) return true;
    duckdb_value a1 = duckdb_bind_get_parameter(info, 1);
    duckdb_value a2 = duckdb_bind_get_parameter(info, 2);
    if (!a1 || !a2 || duckdb_is_null_value(a1) || duckdb_is_null_value(a2)) {
        if (a1) duckdb_destroy_value(&a1);
        if (a2) duckdb_destroy_value(&a2);
        duckdb_bind_set_error(info, (std::string(fn) + ": arguments may not be NULL").c_str());
        return false;
    }
    bool ok = true;
    std::string err;
    if (FORM == GbForm::Having) {
        std::string cmp = value_to_string(a1);
        if      (cmp == ">")  f.cmp = gpudb::GroupByFilter::Cmp::GT;
        else if (cmp == ">=") f.cmp = gpudb::GroupByFilter::Cmp::GE;
        else if (cmp == "<")  f.cmp = gpudb::GroupByFilter::Cmp::LT;
        else if (cmp == "<=") f.cmp = gpudb::GroupByFilter::Cmp::LE;
        else { ok = false; err = std::string(fn) + ": comparison must be one of '>', '>=', '<', '<='"; }
        if (OP == GbOp::SumF64) f.threshold_f64 = duckdb_get_double(a2);
        else                    f.threshold_i64 = duckdb_get_int64(a2);
    } else {
        const std::int64_t k = duckdb_get_int64(a1);
        std::string order = value_to_string(a2);
        for (auto& ch : order) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        if (k < 1) { ok = false; err = std::string(fn) + ": k must be >= 1"; }
        else if (order == "desc") f.topk_desc = true;
        else if (order == "asc")  f.topk_desc = false;
        else { ok = false; err = std::string(fn) + ": order must be 'asc' or 'desc'"; }
        if (ok) f.topk = static_cast<std::size_t>(k);
    }
    duckdb_destroy_value(&a1);
    duckdb_destroy_value(&a2);
    if (!ok) duckdb_bind_set_error(info, err.c_str());
    return ok;
}

template <GbOp OP, GbForm FORM>
void gb_bind(duckdb_bind_info info) {
    duckdb_value nv = duckdb_bind_get_parameter(info, 0);
    if (!nv || duckdb_is_null_value(nv)) {
        if (nv) duckdb_destroy_value(&nv);
        duckdb_bind_set_error(info, (std::string(gb_fn_name(OP, FORM)) + ": name may not be NULL").c_str());
        return;
    }
    auto* bind = new GbBindData();
    bind->name = value_to_string(nv);
    bind->op = OP;
    duckdb_destroy_value(&nv);
    if (!gb_parse_filter<OP, FORM>(info, bind->filter)) { delete bind; return; }

    duckdb_logical_type bigint = duckdb_create_logical_type(DUCKDB_TYPE_BIGINT);
    duckdb_logical_type dbl    = duckdb_create_logical_type(DUCKDB_TYPE_DOUBLE);
    duckdb_bind_add_result_column(info, "key", bigint);
    if (OP == GbOp::SumI64) duckdb_bind_add_result_column(info, "sum", bigint);
    if (OP == GbOp::SumF64) duckdb_bind_add_result_column(info, "sum", dbl);
    duckdb_bind_add_result_column(info, "count", bigint);
    duckdb_destroy_logical_type(&bigint);
    duckdb_destroy_logical_type(&dbl);
    duckdb_bind_set_bind_data(info, bind, [](void* p) { delete static_cast<GbBindData*>(p); });
}

void gb_init(duckdb_init_info info) {
    auto* bind = static_cast<GbBindData*>(duckdb_init_get_bind_data(info));
    auto* init = new GbInitData();
    init->op = bind->op;
    const GbForm form = bind->filter.topk != 0 ? GbForm::TopK
                      : bind->filter.cmp != gpudb::GroupByFilter::Cmp::None ? GbForm::Having
                      : GbForm::Plain;
    const char* fn = gb_fn_name(bind->op, form);
    try {
        ResidentContext& ctx = resident_context(duckdb_init_get_extra_info(info));
        const bool need_vals = bind->op != GbOp::Count;
        Resolved cols = resolve_pair(ctx, bind->name, need_vals, fn);
        auto& agg = resident_aggregator(ctx);
        const std::size_t cap = groupby_rows_cap();
        // the lock is the GPU call's: the decision is copied out under it and
        // the stats line formatted after (as in gm_init / gg_init)
        gpudb::DispatchDecision d;
        {
            auto dev = resident_device_lock(ctx);
            switch (bind->op) {
                case GbOp::SumI64:
                    if (cols.vals->dtype() != gpudb::Dtype::I64)
                        throw std::runtime_error(std::string(fn) +
                            ": '" + bind->name + ".v' is DOUBLE — use " + gb_fn_name(GbOp::SumF64, form));
                    init->res = agg.groupby_sum_resident_i64(*cols.keys, *cols.vals, cap, bind->filter);
                    break;
                case GbOp::SumF64:
                    if (cols.vals->dtype() != gpudb::Dtype::F64)
                        throw std::runtime_error(std::string(fn) +
                            ": '" + bind->name + ".v' is BIGINT — use " + gb_fn_name(GbOp::SumI64, form));
                    init->res = agg.groupby_sum_resident_f64(*cols.keys, *cols.vals, cap, bind->filter);
                    break;
                case GbOp::Count:
                    if (cols.keys->dtype() != gpudb::Dtype::I64)
                        throw std::runtime_error(std::string(fn) +
                            ": keys must be a BIGINT resident column");
                    init->res = agg.groupby_count_resident(*cols.keys, cap, bind->filter);
                    break;
            }
            d = agg.last_decision();
        }
        char buf[320];
        std::snprintf(buf, sizeof(buf),
            "op=%s backend=%s reason=%s rows_in=%zu groups=%zu rows_out=%zu "
            "wall_ms=%.3f kernel_ms=%.3f transfer_ms=%.3f",
            fn + 4 /* strip "gpu_" */, gpudb::to_string(d.chosen),
            gpudb::to_string(d.reason), init->res.rows_in, init->res.groups_total,
            init->res.keys.size(),
            init->res.wall_ms, init->res.kernel_ms, init->res.transfer_ms);
        resident_record_stats(ctx, cols.set.get(), buf);
    } catch (const std::exception& e) {
        delete init;
        duckdb_init_set_error(info, e.what());
        return;
    }
    duckdb_init_set_init_data(info, init, [](void* p) { delete static_cast<GbInitData*>(p); });
}

void gb_function(duckdb_function_info info, duckdb_data_chunk output) {
    auto* init = static_cast<GbInitData*>(duckdb_function_get_init_data(info));
    if (!init) return;
    const std::size_t remaining = init->res.keys.size() - init->offset;
    if (remaining == 0) return;

    constexpr idx_t kChunk = 2048;
    const idx_t out_n = static_cast<idx_t>(std::min<std::size_t>(remaining, kChunk));
    const std::size_t off = init->offset;

    auto* key_out = static_cast<std::int64_t*>(
        duckdb_vector_get_data(duckdb_data_chunk_get_vector(output, 0)));
    for (idx_t i = 0; i < out_n; ++i) key_out[i] = init->res.keys[off + i];

    idx_t col = 1;
    if (init->op == GbOp::SumI64) {
        auto* s = static_cast<std::int64_t*>(
            duckdb_vector_get_data(duckdb_data_chunk_get_vector(output, col++)));
        for (idx_t i = 0; i < out_n; ++i) s[i] = init->res.sums[off + i];
    } else if (init->op == GbOp::SumF64) {
        auto* s = static_cast<double*>(
            duckdb_vector_get_data(duckdb_data_chunk_get_vector(output, col++)));
        for (idx_t i = 0; i < out_n; ++i) s[i] = init->res.sums_f64[off + i];
    }
    auto* c = static_cast<std::int64_t*>(
        duckdb_vector_get_data(duckdb_data_chunk_get_vector(output, col)));
    for (idx_t i = 0; i < out_n; ++i) c[i] = init->res.counts[off + i];

    duckdb_data_chunk_set_size(output, out_n);
    init->offset += static_cast<std::size_t>(out_n);
}

// ---------------------------------------------------------------------------
// Exact GROUP BY (v0.7 milestone 3, docs/TRANSPARENT_DESIGN.md §4.1 / §4.2)
//
//   SELECT key, sum, count, count_star, min, max, avg
//   FROM gpu_groupby_exact_resident('p');                          -- p from gpu_upload_pair_exact
//   SELECT * FROM gpu_groupby_exact_resident_having('p', 'sum', '>', 300);
//   SELECT * FROM gpu_groupby_exact_resident_topk('p', 'count_star', 10, 'desc');
//
// Native semantics, column for column what
//   SELECT k, sum(v), count(v), count(*), min(v), max(v), avg(v) FROM t GROUP BY k
// returns: NULL keys form their own group (emitted last, key NULL); NULL
// payloads count for count(*) only; sum is HUGEINT and never wraps; a group
// whose payload is all NULL has NULL sum/min/max/avg and count = 0; avg is
// the exact sum over the count as a DOUBLE. The aggregate named in the
// _having / _topk forms is one of 'sum', 'count', 'count_star', 'min',
// 'max' ('avg' is not accepted there yet: the threshold argument is BIGINT).
// ---------------------------------------------------------------------------

// ---- WHERE program (§4.6) ----
// The _where forms take a VARCHAR program: terms separated by ';', each
//   <col> <op> <constant>       op: = != <> < <= > >=
//   <col> in (c1, c2, ...)
//   <col> is null | <col> is not null
// where <col> is k (key), v (payload), i<n> (n-th BIGINT predicate column of
// gpu_upload_rows_exact), f<n> (n-th DOUBLE one) or s<n> (n-th VARCHAR
// one). Constants are integer literals for k / v / i<n>, decimal or 'nan' /
// 'inf' literals for f<n>, and quoted strings ('it''s') for s<n> and for k
// when the set's key is a VARCHAR tuple (only =, !=, in, is [not] null
// there: string lanes hold hash64() of the text). The program is a
// conjunction; an empty program is no WHERE.
struct WhereTerm {
    std::string col;            // as written
    gpudb::Predicate::Op op = gpudb::Predicate::Op::EQ;
    std::string value;          // literal text (or empty)
    std::vector<std::string> list;
};

struct GxBindData {
    std::string name;
    gpudb::GroupByFilter filter;
    std::vector<WhereTerm> where;
};

std::string trim_copy(const std::string& s) {
    std::size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}
std::string lower_copy(std::string s) {
    for (auto& ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    return s;
}

// Parses the program; returns an error message or "".
std::string parse_where_program(const std::string& prog, std::vector<WhereTerm>& out) {
    out.clear();
    std::size_t pos = 0;
    while (pos <= prog.size()) {
        std::size_t semi = prog.find(';', pos);
        if (semi == std::string::npos) semi = prog.size();
        std::string term = trim_copy(prog.substr(pos, semi - pos));
        pos = semi + 1;
        if (term.empty()) { if (semi == prog.size()) break; else continue; }
        WhereTerm t;
        // column
        std::size_t i = 0;
        while (i < term.size() && !std::isspace(static_cast<unsigned char>(term[i])) &&
               term[i] != '=' && term[i] != '<' && term[i] != '>' && term[i] != '!') ++i;
        t.col = lower_copy(term.substr(0, i));
        std::string rest = trim_copy(term.substr(i));
        std::string lrest = lower_copy(rest);
        // A literal: a quoted string ('' escapes a quote) or a bare token up to
        // the separator. Quoted literals are kept with their quotes so the
        // resolver knows they are strings.
        auto read_literal = [](const std::string& src, std::size_t& pos, std::string& out) -> bool {
            while (pos < src.size() && std::isspace(static_cast<unsigned char>(src[pos]))) ++pos;
            if (pos < src.size() && src[pos] == '\'') {
                std::string v = "'";
                ++pos;
                while (pos < src.size()) {
                    if (src[pos] == '\'') {
                        if (pos + 1 < src.size() && src[pos + 1] == '\'') { v += "''"; pos += 2; continue; }
                        ++pos; v += "'"; out = v; return true;
                    }
                    v += src[pos++];
                }
                return false;                                    // unterminated
            }
            const std::size_t st = pos;
            while (pos < src.size() && src[pos] != ',' && src[pos] != ')') ++pos;
            out = src.substr(st, pos - st);
            while (!out.empty() && std::isspace(static_cast<unsigned char>(out.back()))) out.pop_back();
            return !out.empty();
        };
        if (lrest == "is null")          { t.op = gpudb::Predicate::Op::IsNull; }
        else if (lrest == "is not null") { t.op = gpudb::Predicate::Op::IsNotNull; }
        else if (lrest.rfind("in", 0) == 0 && (lrest.size() == 2 || lrest[2] == ' ' || lrest[2] == '(')) {
            std::string body = trim_copy(rest.substr(2));
            if (body.size() < 2 || body.front() != '(' || body.back() != ')')
                return "WHERE program: 'in' needs a parenthesised list in '" + term + "'";
            body = body.substr(1, body.size() - 2);
            std::size_t p2 = 0;
            while (p2 < body.size()) {
                std::string item;
                if (!read_literal(body, p2, item)) return "WHERE program: bad 'in' list in '" + term + "'";
                t.list.push_back(item);
                while (p2 < body.size() && std::isspace(static_cast<unsigned char>(body[p2]))) ++p2;
                if (p2 < body.size()) { if (body[p2] != ',') return "WHERE program: bad 'in' list in '" + term + "'"; ++p2; }
            }
            if (t.list.empty()) return "WHERE program: empty 'in' list in '" + term + "'";
            t.op = gpudb::Predicate::Op::In;
        } else {
            std::size_t j = 0;
            while (j < rest.size() && (rest[j] == '=' || rest[j] == '<' || rest[j] == '>' || rest[j] == '!')) ++j;
            const std::string op = rest.substr(0, j);
            if      (op == "=")  t.op = gpudb::Predicate::Op::EQ;
            else if (op == "!=" || op == "<>") t.op = gpudb::Predicate::Op::NE;
            else if (op == "<")  t.op = gpudb::Predicate::Op::LT;
            else if (op == "<=") t.op = gpudb::Predicate::Op::LE;
            else if (op == ">")  t.op = gpudb::Predicate::Op::GT;
            else if (op == ">=") t.op = gpudb::Predicate::Op::GE;
            else return "WHERE program: unknown operator in '" + term + "'";
            std::size_t pv = j;
            std::string v;
            if (!read_literal(rest, pv, v)) return "WHERE program: missing constant in '" + term + "'";
            t.value = v;
            while (pv < rest.size() && std::isspace(static_cast<unsigned char>(rest[pv]))) ++pv;
            if (pv != rest.size()) return "WHERE program: trailing text in '" + term + "'";
        }
        if (t.col.empty()) return "WHERE program: missing column in '" + term + "'";
        out.push_back(std::move(t));
        if (semi == prog.size()) break;
    }
    return "";
}

// Resolve a program against a set: fills `preds` (pointing into `set`) and
// the constant storage. Throws std::runtime_error.
struct ResolvedWhere {
    std::vector<gpudb::Predicate> preds;
    std::vector<std::vector<std::int64_t>> lists;   // In constants, one vector per term
};

// The term as the user wrote it — for an error message, and only then: built
// per predicate of every statement it was two heap allocations nobody read.
std::string term_text(const WhereTerm& w) {
    return w.col + (w.value.empty() ? "" : " ... " + w.value);
}

std::int64_t parse_i64_literal(const std::string& s, const WhereTerm& w) {
    errno = 0;
    char* end = nullptr;
    const long long v = std::strtoll(s.c_str(), &end, 10);
    if (errno != 0 || !end || end == s.c_str() || *end != '\0')
        throw std::runtime_error("WHERE program: '" + s + "' is not a BIGINT constant (" + term_text(w) + ")");
    return static_cast<std::int64_t>(v);
}
std::int64_t parse_f64_literal_bits(const std::string& s, const WhereTerm& w) {
    const std::string l = lower_copy(s);
    double d;
    if (l == "nan" || l == "'nan'") d = std::nan("");
    else if (l == "inf" || l == "infinity" || l == "'inf'" || l == "'infinity'") d = INFINITY;
    else if (l == "-inf" || l == "-infinity" || l == "'-inf'" || l == "'-infinity'") d = -INFINITY;
    else {
        errno = 0;
        char* end = nullptr;
        d = std::strtod(s.c_str(), &end);
        if (errno != 0 || !end || end == s.c_str() || *end != '\0')
            throw std::runtime_error("WHERE program: '" + s + "' is not a DOUBLE constant (" + term_text(w) + ")");
    }
    std::int64_t bits;
    std::memcpy(&bits, &d, sizeof(bits));
    return bits;
}

// A quoted program literal ('it''s') -> its text; ok = false when unquoted.
bool unquote_literal(const std::string& s, std::string& out) {
    if (s.size() < 2 || s.front() != '\'' || s.back() != '\'') return false;
    out.clear();
    for (std::size_t i = 1; i + 1 < s.size(); ++i) {
        if (s[i] == '\'' && i + 2 < s.size() && s[i + 1] == '\'') { out.push_back('\''); ++i; }
        else out.push_back(s[i]);
    }
    return true;
}

// The hash a string lane compares against for literal `text`, checked for a
// collision against what the lane actually holds: a different text under the
// same hash means the answer could be wrong, so the statement declines.
std::int64_t string_lane_value(const std::unordered_map<std::uint64_t, std::string>& dict,
                               const std::string& text, const char* fn) {
    const std::uint64_t h = hash64(text.data(), text.size());
    auto it = dict.find(h);
    if (it != dict.end() && it->second != text)
        throw std::runtime_error(std::string(fn) + ": WHERE literal collides with a resident string (64-bit hash) — "
                                 "run this statement natively");
    return static_cast<std::int64_t>(h);
}

ResolvedWhere resolve_where(const ResidentSet& set, const std::vector<WhereTerm>& terms, const char* fn) {
    ResolvedWhere rw;
    rw.lists.resize(terms.size());
    rw.preds.reserve(terms.size());
    for (std::size_t t = 0; t < terms.size(); ++t) {
        const WhereTerm& w = terms[t];
        gpudb::Predicate p;
        p.op = w.op;
        const std::string& c = w.col;
        bool is_f64 = false;
        const std::unordered_map<std::uint64_t, std::string>* sdict = nullptr;   // string lane: its dictionary
        bool key_tuple = false;                                                   // k of a string-keyed set
        if (c == "k") {
            p.col = set.keys.get();
            if (set.key_str) { sdict = set.key_dict.get(); key_tuple = true; }
        }
        else if (c == "v") p.col = set.vals.get();
        else if ((c[0] == 'i' || c[0] == 'f' || c[0] == 's') && c.size() > 1) {
            std::size_t idx = 0;
            for (std::size_t q = 1; q < c.size(); ++q) {
                if (!std::isdigit(static_cast<unsigned char>(c[q])))
                    throw std::runtime_error(std::string(fn) + ": WHERE program: unknown column '" + c + "'");
                idx = idx * 10 + static_cast<std::size_t>(c[q] - '0');
            }
            const char kind = c[0];
            const std::size_t count = kind == 'i' ? set.pred_int : kind == 'f' ? set.pred_dbl : set.pred_str;
            if (idx >= count)
                throw std::runtime_error(std::string(fn) + ": WHERE program: '" + c + "' but the set has " +
                    std::to_string(count) + (kind == 'i' ? " BIGINT" : kind == 'f' ? " DOUBLE" : " VARCHAR") +
                    " predicate column(s)");
            const std::size_t base = kind == 'i' ? 0 : kind == 'f' ? set.pred_int : set.pred_int + set.pred_dbl;
            p.col = set.preds[base + idx].get();
            is_f64 = kind == 'f';
            if (kind == 's') sdict = set.str_dicts[idx].get();
        } else {
            throw std::runtime_error(std::string(fn) + ": WHERE program: unknown column '" + c + "'");
        }
        if (sdict) {
            // string lane: only = != in / is [not] null; literals must be quoted
            using Op = gpudb::Predicate::Op;
            if (p.op != Op::EQ && p.op != Op::NE && p.op != Op::In && p.op != Op::IsNull && p.op != Op::IsNotNull)
                throw std::runtime_error(std::string(fn) + ": WHERE program: only =, !=, in and is [not] null apply to a string column (" + term_text(w) + ")");
            auto lane_value = [&](const std::string& lit) {
                std::string text;
                if (!unquote_literal(lit, text))
                    throw std::runtime_error(std::string(fn) + ": WHERE program: '" + lit + "' must be a quoted string (" + term_text(w) + ")");
                if (key_tuple) text = tuple_component(text.data(), text.size());   // the key holds the 1-tuple text
                return string_lane_value(*sdict, text, fn);
            };
            if (p.op == Op::In) {
                for (const auto& s : w.list) rw.lists[t].push_back(lane_value(s));
                p.list = rw.lists[t].data();
                p.n_list = rw.lists[t].size();
            } else if (p.op == Op::EQ || p.op == Op::NE) {
                p.value = lane_value(w.value);
            }
            rw.preds.push_back(p);
            continue;
        }
        if (p.op == gpudb::Predicate::Op::In) {
            for (const auto& s : w.list)
                rw.lists[t].push_back(is_f64 ? parse_f64_literal_bits(s, w) : parse_i64_literal(s, w));
            p.list = rw.lists[t].data();
            p.n_list = rw.lists[t].size();
        } else if (p.op != gpudb::Predicate::Op::IsNull && p.op != gpudb::Predicate::Op::IsNotNull) {
            p.value = is_f64 ? parse_f64_literal_bits(w.value, w) : parse_i64_literal(w.value, w);
        }
        rw.preds.push_back(p);
    }
    return rw;
}

struct GxInitData {
    gpudb::GroupByResidentResult res;
    std::size_t offset = 0;
    std::size_t rows = 0;      // result rows (some vectors may be left empty, see GroupByFilter::columns)
    std::vector<idx_t> cols;   // projection pushdown: output vector j is source column cols[j]
};

const char* gx_fn_name(GbForm form, bool where = false) {
    if (where)
        return form == GbForm::Having ? "gpu_groupby_exact_resident_where_having"
             : form == GbForm::TopK   ? "gpu_groupby_exact_resident_where_topk"
                                      : "gpu_groupby_exact_resident_where";
    return form == GbForm::Having ? "gpu_groupby_exact_resident_having"
         : form == GbForm::TopK   ? "gpu_groupby_exact_resident_topk"
                                  : "gpu_groupby_exact_resident";
}

bool gx_parse_agg(const std::string& s, gpudb::GroupByFilter::Agg& out) {
    using A = gpudb::GroupByFilter::Agg;
    if (s == "sum")        { out = A::Sum;       return true; }
    if (s == "count")      { out = A::CountV;    return true; }
    if (s == "count_star") { out = A::CountStar; return true; }
    if (s == "min")        { out = A::Min;       return true; }
    if (s == "max")        { out = A::Max;       return true; }
    return false;
}

// _having(name, [where,] agg, cmp, threshold BIGINT); _topk(name, [where,] agg, k, order)
template <GbForm FORM, bool WHERE>
bool gx_parse_filter(duckdb_bind_info info, gpudb::GroupByFilter& f) {
    const char* fn = gx_fn_name(FORM, WHERE);
    if (FORM == GbForm::Plain) return true;
    const idx_t base = WHERE ? 2 : 1;
    duckdb_value a1 = duckdb_bind_get_parameter(info, base);
    duckdb_value a2 = duckdb_bind_get_parameter(info, base + 1);
    duckdb_value a3 = duckdb_bind_get_parameter(info, base + 2);
    auto destroy = [&] {
        if (a1) duckdb_destroy_value(&a1);
        if (a2) duckdb_destroy_value(&a2);
        if (a3) duckdb_destroy_value(&a3);
    };
    if (!a1 || !a2 || !a3 || duckdb_is_null_value(a1) || duckdb_is_null_value(a2) ||
        duckdb_is_null_value(a3)) {
        destroy();
        duckdb_bind_set_error(info, (std::string(fn) + ": arguments may not be NULL").c_str());
        return false;
    }
    std::string err;
    std::string agg = value_to_string(a1);
    for (auto& ch : agg) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    if (!gx_parse_agg(agg, f.agg))
        err = std::string(fn) + ": aggregate must be one of 'sum', 'count', 'count_star', 'min', 'max'";
    if (err.empty() && FORM == GbForm::Having) {
        std::string cmp = value_to_string(a2);
        if      (cmp == ">")  f.cmp = gpudb::GroupByFilter::Cmp::GT;
        else if (cmp == ">=") f.cmp = gpudb::GroupByFilter::Cmp::GE;
        else if (cmp == "<")  f.cmp = gpudb::GroupByFilter::Cmp::LT;
        else if (cmp == "<=") f.cmp = gpudb::GroupByFilter::Cmp::LE;
        else err = std::string(fn) + ": comparison must be one of '>', '>=', '<', '<='";
        f.threshold_i64 = duckdb_get_int64(a3);
    } else if (err.empty()) {
        const std::int64_t k = duckdb_get_int64(a2);
        std::string order = value_to_string(a3);
        for (auto& ch : order) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        if (k < 1) err = std::string(fn) + ": k must be >= 1";
        else if (order == "desc") f.topk_desc = true;
        else if (order == "asc")  f.topk_desc = false;
        else err = std::string(fn) + ": order must be 'asc' or 'desc'";
        if (err.empty()) f.topk = static_cast<std::size_t>(k);
    }
    destroy();
    if (!err.empty()) { duckdb_bind_set_error(info, err.c_str()); return false; }
    return true;
}

template <GbForm FORM, bool WHERE>
void gx_bind(duckdb_bind_info info) {
    duckdb_value nv = duckdb_bind_get_parameter(info, 0);
    if (!nv || duckdb_is_null_value(nv)) {
        if (nv) duckdb_destroy_value(&nv);
        duckdb_bind_set_error(info, (std::string(gx_fn_name(FORM, WHERE)) + ": name may not be NULL").c_str());
        return;
    }
    auto* bind = new GxBindData();
    bind->name = value_to_string(nv);
    duckdb_destroy_value(&nv);
    if (WHERE) {
        duckdb_value wv = duckdb_bind_get_parameter(info, 1);
        if (!wv || duckdb_is_null_value(wv)) {
            if (wv) duckdb_destroy_value(&wv);
            delete bind;
            duckdb_bind_set_error(info, (std::string(gx_fn_name(FORM, WHERE)) +
                ": the WHERE program may not be NULL (use '' for none)").c_str());
            return;
        }
        const std::string prog = value_to_string(wv);
        duckdb_destroy_value(&wv);
        const std::string err = parse_where_program(prog, bind->where);
        if (!err.empty()) {
            delete bind;
            duckdb_bind_set_error(info, (std::string(gx_fn_name(FORM, WHERE)) + ": " + err).c_str());
            return;
        }
    }
    if (!gx_parse_filter<FORM, WHERE>(info, bind->filter)) { delete bind; return; }

    duckdb_logical_type bigint  = duckdb_create_logical_type(DUCKDB_TYPE_BIGINT);
    duckdb_logical_type hugeint = duckdb_create_logical_type(DUCKDB_TYPE_HUGEINT);
    duckdb_logical_type dbl     = duckdb_create_logical_type(DUCKDB_TYPE_DOUBLE);
    duckdb_bind_add_result_column(info, "key",        bigint);
    duckdb_bind_add_result_column(info, "sum",        hugeint);
    duckdb_bind_add_result_column(info, "count",      bigint);
    duckdb_bind_add_result_column(info, "count_star", bigint);
    duckdb_bind_add_result_column(info, "min",        bigint);
    duckdb_bind_add_result_column(info, "max",        bigint);
    duckdb_bind_add_result_column(info, "avg",        dbl);
    duckdb_destroy_logical_type(&bigint);
    duckdb_destroy_logical_type(&hugeint);
    duckdb_destroy_logical_type(&dbl);
    duckdb_bind_set_bind_data(info, bind, [](void* p) { delete static_cast<GxBindData*>(p); });
}

void gx_init(duckdb_init_info info) {
    auto* bind = static_cast<GxBindData*>(duckdb_init_get_bind_data(info));
    auto* init = new GxInitData();
    // Projection pushdown: only the columns the statement reads are filled
    // (a count(*) over the function reads none). Emitting 2M groups x 7
    // columns through DuckDB's chunk pipeline costs more than the reduce.
    std::uint32_t want = 0;
    {
        const idx_t nc = duckdb_init_get_column_count(info);
        init->cols.reserve(nc);
        for (idx_t i = 0; i < nc; ++i) {
            const idx_t c = duckdb_init_get_column_index(info, i);
            init->cols.push_back(c);
            switch (c) {
                case 0: want |= 1u << 0; break;                 // key
                case 1: want |= 1u << 1; break;                 // sum
                case 2: want |= 1u << 2; break;                 // count
                case 3: want |= 1u << 3; break;                 // count_star
                case 4: want |= 1u << 4; break;                 // min
                case 5: want |= 1u << 5; break;                 // max
                case 6: want |= (1u << 1) | (1u << 2); break;   // avg = sum / count
                default: break;
            }
        }
        // sum / min / max / avg NULL-ness is read from count(v)
        if (want & ((1u << 1) | (1u << 4) | (1u << 5))) want |= 1u << 2;
    }
    const GbForm form = bind->filter.topk != 0 ? GbForm::TopK
                      : bind->filter.cmp != gpudb::GroupByFilter::Cmp::None ? GbForm::Having
                      : GbForm::Plain;
    const bool where = !bind->where.empty();
    const char* fn = gx_fn_name(form, where);
    try {
        ResidentContext& ctx = resident_context(duckdb_init_get_extra_info(info));
        std::shared_ptr<ResidentSet> set = resident_acquire_set(ctx, bind->name, fn);
        if (!set->pair || !set->exact)
            throw std::runtime_error(std::string(fn) + ": '" + bind->name +
                "' was not uploaded by gpu_upload_pair_exact / gpu_upload_rows_exact — the exact "
                "GROUP BY needs the pair with its NULLs (gpu_upload_pair drops them)");
        if (set->keys->dtype() != gpudb::Dtype::I64 || set->vals->dtype() != gpudb::Dtype::I64)
            throw std::runtime_error(std::string(fn) + ": '" + bind->name +
                "' must be a BIGINT key / BIGINT payload pair");
        ResolvedWhere rw;
        if (where) rw = resolve_where(*set, bind->where, fn);
        auto& agg = resident_aggregator(ctx);
        const std::size_t cap = groupby_rows_cap();
        gpudb::GroupByFilter filt = bind->filter;
        filt.columns = want;   // only the projected result vectors are materialised
        // The device lock is the GPU call's, not the statement's: what the
        // backend left behind in process-wide state (the decision, the path
        // note) is copied out under it, and the stats line is formatted after
        // it is gone — the way gm_init and gg_init already do it.
        gpudb::DispatchDecision d;
        std::string path;
        {
            auto dev = resident_device_lock(ctx);
            gpudb::exact_path_note().clear();   // the backend names the algorithm it ran
            gpudb::exact_mask_note().clear();   // ... and, when there was one, its WHERE stage
            if (where)
                init->res = agg.groupby_exact_masked_resident(*set->keys, set->vals.get(),
                                                              rw.preds.data(), rw.preds.size(), cap, filt);
            else
                init->res = agg.groupby_exact_resident(*set->keys, set->vals.get(), cap, filt);
            d = agg.last_decision();
            path = gpudb::exact_path_note();
            if (!gpudb::exact_mask_note().empty()) path += "/" + gpudb::exact_mask_note();
        }
        {
            const auto& rr = init->res;
            init->rows = std::max({ rr.keys.size(), rr.sums.size(), rr.counts.size(), rr.counts_star.size(),
                                    rr.mins.size(), rr.maxs.size() });
        }
        char buf[320];
        std::snprintf(buf, sizeof(buf),
            "op=%s backend=%s reason=%s rows_in=%zu groups=%zu rows_out=%zu "
            "wall_ms=%.3f kernel_ms=%.3f transfer_ms=%.3f path=%s",
            fn + 4 /* strip "gpu_" */, gpudb::to_string(d.chosen),
            gpudb::to_string(d.reason), init->res.rows_in, init->res.groups_total,
            init->rows,
            init->res.wall_ms, init->res.kernel_ms, init->res.transfer_ms,
            path.c_str());
        resident_record_stats(ctx, set.get(), buf);
    } catch (const std::exception& e) {
        delete init;
        duckdb_init_set_error(info, e.what());
        return;
    }
    duckdb_init_set_init_data(info, init, [](void* p) { delete static_cast<GxInitData*>(p); });
}

void gx_function(duckdb_function_info info, duckdb_data_chunk output) {
    auto* init = static_cast<GxInitData*>(duckdb_function_get_init_data(info));
    if (!init) return;
    const auto& r = init->res;
    const std::size_t remaining = init->rows - init->offset;
    if (remaining == 0) return;

    constexpr idx_t kChunk = 2048;
    const idx_t out_n = static_cast<idx_t>(std::min<std::size_t>(remaining, kChunk));
    const std::size_t off = init->offset;

    for (idx_t j = 0; j < init->cols.size(); ++j) {
        duckdb_vector vec = duckdb_data_chunk_get_vector(output, j);
        switch (init->cols[j]) {
            case 0: {   // key (NULL for the NULL-key group)
                auto* key = static_cast<std::int64_t*>(duckdb_vector_get_data(vec));
                uint64_t* ok = nullptr;
                for (idx_t i = 0; i < out_n; ++i) {
                    key[i] = r.keys[off + i];
                    if (!r.key_null.empty() && r.key_null[off + i]) {
                        if (!ok) { duckdb_vector_ensure_validity_writable(vec); ok = duckdb_vector_get_validity(vec); }
                        duckdb_validity_set_row_invalid(ok, i);
                    }
                }
                break;
            }
            case 1: {   // sum HUGEINT (NULL when count(v) == 0)
                auto* sum = static_cast<duckdb_hugeint*>(duckdb_vector_get_data(vec));
                uint64_t* ok = nullptr;
                for (idx_t i = 0; i < out_n; ++i) {
                    sum[i].lower = static_cast<std::uint64_t>(r.sums[off + i]);
                    sum[i].upper = r.sums_hi[off + i];
                    if (r.counts[off + i] == 0) {
                        if (!ok) { duckdb_vector_ensure_validity_writable(vec); ok = duckdb_vector_get_validity(vec); }
                        duckdb_validity_set_row_invalid(ok, i);
                    }
                }
                break;
            }
            case 2: {   // count(v)
                auto* c = static_cast<std::int64_t*>(duckdb_vector_get_data(vec));
                for (idx_t i = 0; i < out_n; ++i) c[i] = r.counts[off + i];
                break;
            }
            case 3: {   // count(*)
                auto* c = static_cast<std::int64_t*>(duckdb_vector_get_data(vec));
                for (idx_t i = 0; i < out_n; ++i) c[i] = r.counts_star[off + i];
                break;
            }
            case 4: case 5: {   // min / max (NULL when count(v) == 0)
                const auto& src = init->cols[j] == 4 ? r.mins : r.maxs;
                auto* m = static_cast<std::int64_t*>(duckdb_vector_get_data(vec));
                uint64_t* ok = nullptr;
                for (idx_t i = 0; i < out_n; ++i) {
                    m[i] = src[off + i];
                    if (r.counts[off + i] == 0) {
                        if (!ok) { duckdb_vector_ensure_validity_writable(vec); ok = duckdb_vector_get_validity(vec); }
                        duckdb_validity_set_row_invalid(ok, i);
                    }
                }
                break;
            }
            case 6: {   // avg DOUBLE = native's hugeint->double of the exact sum, over count(v)
                auto* a = static_cast<double*>(duckdb_vector_get_data(vec));
                uint64_t* ok = nullptr;
                for (idx_t i = 0; i < out_n; ++i) {
                    const std::int64_t cnt = r.counts[off + i];
                    if (cnt == 0) {
                        a[i] = 0.0;
                        if (!ok) { duckdb_vector_ensure_validity_writable(vec); ok = duckdb_vector_get_validity(vec); }
                        duckdb_validity_set_row_invalid(ok, i);
                    } else {
                        const gpudb::Sum128 sm{static_cast<std::uint64_t>(r.sums[off + i]), r.sums_hi[off + i]};
                        a[i] = sm.to_double() / static_cast<double>(cnt);
                    }
                }
                break;
            }
            default: break;
        }
    }
    duckdb_data_chunk_set_size(output, out_n);
    init->offset += static_cast<std::size_t>(out_n);
}

// ---------------------------------------------------------------------------
// gpu_groupby_exact_multi(name, program, payloads, filter)  (v0.7 §4.9)
//
// The exact GROUP BY over SEVERAL payload lanes of one exact set in one
// statement: SELECT k, sum(a), min(b), avg(c) ... GROUP BY k.
//   program  : the WHERE program ('' = none), as gpu_groupby_exact_resident_where
//   payloads : 'v, i0, i2' — payload lanes, v or a BIGINT predicate lane; 1..8
//   filter   : '' | 'having <p> <agg> <cmp> <threshold>' | 'topk <p> <agg> <k> <asc|desc>'
//              <p> = index into payloads, <agg> as the single-payload forms
// Columns: key BIGINT, count_star BIGINT, then for payload p:
//   sum<p> HUGEINT, count<p> BIGINT, min<p> BIGINT, max<p> BIGINT, avg<p> DOUBLE
// One resident operator pass per payload the statement reads (projection
// pushdown skips the rest). Every pass sees the same keys and the same WHERE
// mask, so it emits the same groups in the same order (ascending key, the
// NULL-key group last) and the passes are zipped by position; under a filter
// the filtered payload runs first and the others are looked up by key.
// ---------------------------------------------------------------------------
struct GmBindData {
    std::string name;
    std::vector<WhereTerm> where;
    std::vector<std::string> lanes;
    gpudb::GroupByFilter filter;
    std::size_t filter_payload = 0;
};

struct GmInitData {
    std::vector<std::int64_t> keys;
    std::vector<std::uint8_t> key_null;
    std::vector<std::int64_t> counts_star;
    std::vector<gpudb::GroupByResidentResult> pay;   // sums / sums_hi / counts / mins / maxs per payload
    std::size_t rows = 0, offset = 0;
    std::vector<idx_t> cols;
};

void gm_bind(duckdb_bind_info info) {
    static const char* fn = "gpu_groupby_exact_multi";
    std::string a[4];
    for (idx_t i = 0; i < 4; ++i) {
        duckdb_value v = duckdb_bind_get_parameter(info, i);
        const bool null = !v || duckdb_is_null_value(v);
        if (!null) a[i] = value_to_string(v);
        if (v) duckdb_destroy_value(&v);
        if (null) { duckdb_bind_set_error(info, (std::string(fn) + ": arguments may not be NULL").c_str()); return; }
    }
    auto bind = std::make_unique<GmBindData>();
    bind->name = a[0];
    std::string err = parse_where_program(a[1], bind->where);
    // payload lanes
    if (err.empty()) {
        std::size_t pos = 0;
        while (pos <= a[2].size()) {
            std::size_t comma = a[2].find(',', pos);
            if (comma == std::string::npos) comma = a[2].size();
            const std::string tok = lower_copy(trim_copy(a[2].substr(pos, comma - pos)));
            pos = comma + 1;
            bool ok = tok == "v" || (tok.size() > 1 && tok.size() < 6 && tok[0] == 'i');
            for (std::size_t q = 1; ok && tok != "v" && q < tok.size(); ++q) ok = std::isdigit(static_cast<unsigned char>(tok[q])) != 0;
            if (!ok) { err = "payload lane '" + tok + "' must be v or i<n>"; break; }
            bind->lanes.push_back(tok);
            if (comma == a[2].size()) break;
        }
        if (err.empty() && (bind->lanes.empty() || bind->lanes.size() > 8)) err = "1 to 8 payload lanes";
    }
    // filter
    if (err.empty()) {
        std::vector<std::string> w;
        std::size_t pos = 0;
        const std::string f = trim_copy(a[3]);
        while (pos < f.size()) {
            while (pos < f.size() && std::isspace(static_cast<unsigned char>(f[pos]))) ++pos;
            std::size_t e = pos;
            while (e < f.size() && !std::isspace(static_cast<unsigned char>(f[e]))) ++e;
            if (e > pos) w.push_back(f.substr(pos, e - pos));
            pos = e;
        }
        if (!w.empty()) {
            const std::string kind = lower_copy(w[0]);
            auto to_i64 = [&](const std::string& s, std::int64_t& out) {
                errno = 0; char* end = nullptr;
                const long long v = std::strtoll(s.c_str(), &end, 10);
                if (errno != 0 || !end || end == s.c_str() || *end != '\0') return false;
                out = v; return true;
            };
            std::int64_t p = 0, n = 0;
            if (w.size() != 5 || (kind != "having" && kind != "topk")) err = "filter must be '', 'having <p> <agg> <cmp> <threshold>' or 'topk <p> <agg> <k> <asc|desc>'";
            else if (!to_i64(w[1], p) || p < 0 || static_cast<std::size_t>(p) >= bind->lanes.size()) err = "filter payload index out of range";
            else if (!gx_parse_agg(lower_copy(w[2]), bind->filter.agg)) err = "filter aggregate must be one of sum, count, count_star, min, max";
            else if (kind == "having") {
                if      (w[3] == ">")  bind->filter.cmp = gpudb::GroupByFilter::Cmp::GT;
                else if (w[3] == ">=") bind->filter.cmp = gpudb::GroupByFilter::Cmp::GE;
                else if (w[3] == "<")  bind->filter.cmp = gpudb::GroupByFilter::Cmp::LT;
                else if (w[3] == "<=") bind->filter.cmp = gpudb::GroupByFilter::Cmp::LE;
                else err = "filter comparison must be one of > >= < <=";
                if (err.empty() && !to_i64(w[4], n)) err = "filter threshold must be a BIGINT";
                bind->filter.threshold_i64 = n;
            } else {
                const std::string ord = lower_copy(w[4]);
                if (!to_i64(w[3], n) || n < 1) err = "k must be >= 1";
                else if (ord != "asc" && ord != "desc") err = "order must be asc or desc";
                bind->filter.topk = static_cast<std::size_t>(n);
                bind->filter.topk_desc = ord == "desc";
            }
            bind->filter_payload = static_cast<std::size_t>(p);
        }
    }
    if (!err.empty()) { duckdb_bind_set_error(info, (std::string(fn) + ": " + err).c_str()); return; }

    duckdb_logical_type bigint  = duckdb_create_logical_type(DUCKDB_TYPE_BIGINT);
    duckdb_logical_type hugeint = duckdb_create_logical_type(DUCKDB_TYPE_HUGEINT);
    duckdb_logical_type dbl     = duckdb_create_logical_type(DUCKDB_TYPE_DOUBLE);
    duckdb_bind_add_result_column(info, "key", bigint);
    duckdb_bind_add_result_column(info, "count_star", bigint);
    for (std::size_t p = 0; p < bind->lanes.size(); ++p) {
        const std::string s = std::to_string(p);
        duckdb_bind_add_result_column(info, ("sum" + s).c_str(),   hugeint);
        duckdb_bind_add_result_column(info, ("count" + s).c_str(), bigint);
        duckdb_bind_add_result_column(info, ("min" + s).c_str(),   bigint);
        duckdb_bind_add_result_column(info, ("max" + s).c_str(),   bigint);
        duckdb_bind_add_result_column(info, ("avg" + s).c_str(),   dbl);
    }
    duckdb_destroy_logical_type(&bigint);
    duckdb_destroy_logical_type(&hugeint);
    duckdb_destroy_logical_type(&dbl);
    duckdb_bind_set_bind_data(info, bind.release(), [](void* p) { delete static_cast<GmBindData*>(p); });
}

void gm_init(duckdb_init_info info) {
    static const char* fn = "gpu_groupby_exact_multi";
    auto* bind = static_cast<GmBindData*>(duckdb_init_get_bind_data(info));
    auto init = std::make_unique<GmInitData>();
    const std::size_t P = bind->lanes.size();
    // projection -> per-payload GroupByFilter::columns bits (1 sum, 2 count, 4 min, 5 max)
    std::vector<std::uint32_t> want(P, 0);
    bool want_key = false, want_cstar = false;
    const idx_t nc = duckdb_init_get_column_count(info);
    for (idx_t i = 0; i < nc; ++i) {
        const idx_t c = duckdb_init_get_column_index(info, i);
        init->cols.push_back(c);
        if (c == 0) want_key = true;
        else if (c == 1) want_cstar = true;
        else if (c >= 2 && c < 2 + 5 * P) {
            const std::size_t p = (c - 2) / 5;
            switch ((c - 2) % 5) {
                case 0: want[p] |= (1u << 1) | (1u << 2); break;
                case 1: want[p] |= 1u << 2; break;
                case 2: want[p] |= (1u << 4) | (1u << 2); break;
                case 3: want[p] |= (1u << 5) | (1u << 2); break;
                default: want[p] |= (1u << 1) | (1u << 2); break;
            }
        }
    }
    try {
        ResidentContext& ctx = resident_context(duckdb_init_get_extra_info(info));
        std::shared_ptr<ResidentSet> set = resident_acquire_set(ctx, bind->name, fn);
        if (!set->pair || !set->exact)
            throw std::runtime_error(std::string(fn) + ": '" + bind->name + "' is not an exact set");
        std::vector<const gpudb::ResidentColumn*> cols(P, nullptr);
        for (std::size_t p = 0; p < P; ++p) {
            const std::string& l = bind->lanes[p];
            if (l == "v") cols[p] = set->vals.get();
            else {
                const std::size_t idx = static_cast<std::size_t>(std::strtoull(l.c_str() + 1, nullptr, 10));
                if (idx >= set->pred_int)
                    throw std::runtime_error(std::string(fn) + ": payload lane '" + l + "' but the set has " +
                                             std::to_string(set->pred_int) + " BIGINT predicate column(s)");
                cols[p] = set->preds[idx].get();
            }
            if (!cols[p] || cols[p]->dtype() != gpudb::Dtype::I64)
                throw std::runtime_error(std::string(fn) + ": payload lane '" + l + "' is not a BIGINT lane");
        }
        ResolvedWhere rw = resolve_where(*set, bind->where, fn);
        auto& agg = resident_aggregator(ctx);
        const std::size_t cap = groupby_rows_cap();
        const bool filtered = bind->filter.active();
        const std::size_t fp = filtered ? bind->filter_payload : 0;

        std::vector<gpudb::MultiPayload> mp(P);
        for (std::size_t p = 0; p < P; ++p) { mp[p].vals = cols[p]; mp[p].columns = want[p]; }
        gpudb::GroupByFilter f0 = bind->filter;
        f0.columns = (want_cstar ? (1u << 3) : 0u) | (want_key ? 1u : 0u);
        std::size_t passes = 0, rows_in = 0, groups_total = 0;
        double wall = 0.0, kernel = 0.0;
        gpudb::DispatchDecision d;
        std::string path;
        {
            auto dev = resident_device_lock(ctx);
            gpudb::exact_path_note().clear();   // the backend names the algorithm it ran
            gpudb::exact_mask_note().clear();   // ... and, when there was one, its WHERE stage
            init->pay = agg.groupby_exact_masked_multi(*set->keys, mp.data(), P, fp, rw.preds.data(), rw.preds.size(), cap, f0);
            // process-wide state: copied out under the lock that wrote it
            d = agg.last_decision();
            path = gpudb::exact_path_note();
            if (!gpudb::exact_mask_note().empty()) path += "/" + gpudb::exact_mask_note();
        }
        {
            auto& prim = init->pay[fp];
            for (const auto& r : init->pay)
                init->rows = std::max({init->rows, r.keys.size(), r.sums.size(), r.counts.size(), r.counts_star.size(),
                                       r.mins.size(), r.maxs.size()});
            rows_in = prim.rows_in; groups_total = prim.groups_total; wall = prim.wall_ms; kernel = prim.kernel_ms;
            for (std::size_t p = 0; p < P; ++p) if (p == fp || want[p]) ++passes;
            init->keys = std::move(prim.keys);
            init->key_null = std::move(prim.key_null);
            init->counts_star = std::move(prim.counts_star);
        }
        char buf[320];
        std::snprintf(buf, sizeof(buf),
            "op=groupby_exact_multi backend=%s reason=%s rows_in=%zu groups=%zu rows_out=%zu payloads=%zu passes=%zu "
            "wall_ms=%.3f kernel_ms=%.3f transfer_ms=0.000 path=%s",
            gpudb::to_string(d.chosen), gpudb::to_string(d.reason), rows_in, groups_total, init->rows, P, passes,
            wall, kernel, path.c_str());
        resident_record_stats(ctx, set.get(), buf);
    } catch (const std::exception& e) {
        duckdb_init_set_error(info, e.what());
        return;
    }
    duckdb_init_set_init_data(info, init.release(), [](void* p) { delete static_cast<GmInitData*>(p); });
}

void gm_function(duckdb_function_info info, duckdb_data_chunk output) {
    auto* init = static_cast<GmInitData*>(duckdb_function_get_init_data(info));
    if (!init) return;
    const std::size_t remaining = init->rows - init->offset;
    if (remaining == 0) return;
    constexpr idx_t kChunk = 2048;
    const idx_t out_n = static_cast<idx_t>(std::min<std::size_t>(remaining, kChunk));
    const std::size_t off = init->offset;
    auto null_at = [](duckdb_vector vec, uint64_t*& ok, idx_t i) {
        if (!ok) { duckdb_vector_ensure_validity_writable(vec); ok = duckdb_vector_get_validity(vec); }
        duckdb_validity_set_row_invalid(ok, i);
    };
    for (idx_t j = 0; j < init->cols.size(); ++j) {
        duckdb_vector vec = duckdb_data_chunk_get_vector(output, j);
        const idx_t c = init->cols[j];
        uint64_t* ok = nullptr;
        if (c == 0) {
            auto* key = static_cast<std::int64_t*>(duckdb_vector_get_data(vec));
            for (idx_t i = 0; i < out_n; ++i) {
                key[i] = init->keys[off + i];
                if (!init->key_null.empty() && init->key_null[off + i]) null_at(vec, ok, i);
            }
            continue;
        }
        if (c == 1) {
            auto* v = static_cast<std::int64_t*>(duckdb_vector_get_data(vec));
            for (idx_t i = 0; i < out_n; ++i) v[i] = init->counts_star[off + i];
            continue;
        }
        const auto& r = init->pay[(c - 2) / 5];
        switch ((c - 2) % 5) {
            case 0: {
                auto* sum = static_cast<duckdb_hugeint*>(duckdb_vector_get_data(vec));
                for (idx_t i = 0; i < out_n; ++i) {
                    sum[i].lower = static_cast<std::uint64_t>(r.sums[off + i]);
                    sum[i].upper = r.sums_hi[off + i];
                    if (r.counts[off + i] == 0) null_at(vec, ok, i);
                }
                break;
            }
            case 1: {
                auto* v = static_cast<std::int64_t*>(duckdb_vector_get_data(vec));
                for (idx_t i = 0; i < out_n; ++i) v[i] = r.counts[off + i];
                break;
            }
            case 2: case 3: {
                const auto& src = (c - 2) % 5 == 2 ? r.mins : r.maxs;
                auto* v = static_cast<std::int64_t*>(duckdb_vector_get_data(vec));
                for (idx_t i = 0; i < out_n; ++i) {
                    v[i] = src[off + i];
                    if (r.counts[off + i] == 0) null_at(vec, ok, i);
                }
                break;
            }
            default: {
                auto* a = static_cast<double*>(duckdb_vector_get_data(vec));
                for (idx_t i = 0; i < out_n; ++i) {
                    const std::int64_t cnt = r.counts[off + i];
                    if (cnt == 0) { a[i] = 0.0; null_at(vec, ok, i); }
                    else {
                        const gpudb::Sum128 sm{static_cast<std::uint64_t>(r.sums[off + i]), r.sums_hi[off + i]};
                        a[i] = sm.to_double() / static_cast<double>(cnt);
                    }
                }
                break;
            }
        }
    }
    duckdb_data_chunk_set_size(output, out_n);
    init->offset += static_cast<std::size_t>(out_n);
}

// ---------------------------------------------------------------------------
// gpu_agg_exact_global(name, program, payloads)  (v0.7 §4.12)
//
// Aggregates without GROUP BY over an exact set, under a WHERE: TPC-H Q6's
// shape. One fused device pass over the rows — no key, no sort cache, no
// permutation — and exactly ONE result row, whatever the WHERE keeps.
//   program  : the WHERE program ('' = none), as gpu_groupby_exact_resident_where
//   payloads : 'v, i0, i2' — payload lanes, v or a BIGINT predicate lane; '' = count(*) only
// Columns: count_star BIGINT, then for payload p:
//   sum<p> HUGEINT, count<p> BIGINT, min<p> BIGINT, max<p> BIGINT, avg<p> DOUBLE
// Semantics are one group of gpu_groupby_exact_multi: a row failing the mask
// takes part in nothing including count(*); a NULL payload cell counts for
// count(*) only; a payload with count 0 has NULL sum / min / max / avg — which
// is what native returns over an empty input, so the single row IS native's
// answer.
// A set only this function reads carries the tag extra 'global' and no key
// lane ('-'), so nothing sorts its columns.
// ---------------------------------------------------------------------------
struct GgBindData {
    std::string name;
    std::vector<WhereTerm> where;
    std::vector<std::string> lanes;
};

struct GgInitData {
    gpudb::GlobalAggResult res;
    bool done = false;
    std::vector<idx_t> cols;
};

void gg_bind(duckdb_bind_info info) {
    static const char* fn = "gpu_agg_exact_global";
    std::string a[3];
    for (idx_t i = 0; i < 3; ++i) {
        duckdb_value v = duckdb_bind_get_parameter(info, i);
        const bool null = !v || duckdb_is_null_value(v);
        if (!null) a[i] = value_to_string(v);
        if (v) duckdb_destroy_value(&v);
        if (null) { duckdb_bind_set_error(info, (std::string(fn) + ": arguments may not be NULL").c_str()); return; }
    }
    auto bind = std::make_unique<GgBindData>();
    bind->name = a[0];
    std::string err = parse_where_program(a[1], bind->where);
    if (err.empty() && !trim_copy(a[2]).empty()) {
        std::size_t pos = 0;
        while (pos <= a[2].size()) {
            std::size_t comma = a[2].find(',', pos);
            if (comma == std::string::npos) comma = a[2].size();
            const std::string tok = lower_copy(trim_copy(a[2].substr(pos, comma - pos)));
            pos = comma + 1;
            bool ok = tok == "v" || (tok.size() > 1 && tok.size() < 6 && tok[0] == 'i');
            for (std::size_t q = 1; ok && tok != "v" && q < tok.size(); ++q) ok = std::isdigit(static_cast<unsigned char>(tok[q])) != 0;
            if (!ok) { err = "payload lane '" + tok + "' must be v or i<n>"; break; }
            bind->lanes.push_back(tok);
            if (comma == a[2].size()) break;
        }
        if (err.empty() && bind->lanes.size() > 8) err = "at most 8 payload lanes";
    }
    if (!err.empty()) { duckdb_bind_set_error(info, (std::string(fn) + ": " + err).c_str()); return; }

    duckdb_logical_type bigint  = duckdb_create_logical_type(DUCKDB_TYPE_BIGINT);
    duckdb_logical_type hugeint = duckdb_create_logical_type(DUCKDB_TYPE_HUGEINT);
    duckdb_logical_type dbl     = duckdb_create_logical_type(DUCKDB_TYPE_DOUBLE);
    duckdb_bind_add_result_column(info, "count_star", bigint);
    for (std::size_t p = 0; p < bind->lanes.size(); ++p) {
        const std::string s = std::to_string(p);
        duckdb_bind_add_result_column(info, ("sum" + s).c_str(),   hugeint);
        duckdb_bind_add_result_column(info, ("count" + s).c_str(), bigint);
        duckdb_bind_add_result_column(info, ("min" + s).c_str(),   bigint);
        duckdb_bind_add_result_column(info, ("max" + s).c_str(),   bigint);
        duckdb_bind_add_result_column(info, ("avg" + s).c_str(),   dbl);
    }
    duckdb_destroy_logical_type(&bigint);
    duckdb_destroy_logical_type(&hugeint);
    duckdb_destroy_logical_type(&dbl);
    duckdb_bind_set_bind_data(info, bind.release(), [](void* p) { delete static_cast<GgBindData*>(p); });
}

void gg_init(duckdb_init_info info) {
    static const char* fn = "gpu_agg_exact_global";
    auto* bind = static_cast<GgBindData*>(duckdb_init_get_bind_data(info));
    auto init = std::make_unique<GgInitData>();
    const std::size_t P = bind->lanes.size();
    // projection -> per-payload MultiPayload::columns bits (1 sum, 2 count, 4 min, 5 max)
    std::vector<std::uint32_t> want(P, 0);
    const idx_t nc = duckdb_init_get_column_count(info);
    for (idx_t i = 0; i < nc; ++i) {
        const idx_t c = duckdb_init_get_column_index(info, i);
        init->cols.push_back(c);
        if (c == 0) continue;                             // count_star: always computed
        const std::size_t p = (c - 1) / 5;
        if (p >= P) continue;
        switch ((c - 1) % 5) {
            case 0: want[p] |= (1u << 1) | (1u << 2); break;
            case 1: want[p] |= 1u << 2; break;
            case 2: want[p] |= (1u << 4) | (1u << 2); break;
            case 3: want[p] |= (1u << 5) | (1u << 2); break;
            default: want[p] |= (1u << 1) | (1u << 2); break;
        }
    }
    try {
        ResidentContext& ctx = resident_context(duckdb_init_get_extra_info(info));
        std::shared_ptr<ResidentSet> set = resident_acquire_set(ctx, bind->name, fn);
        if (!set->pair || !set->exact)
            throw std::runtime_error(std::string(fn) + ": '" + bind->name + "' is not an exact set");
        std::vector<gpudb::MultiPayload> mp(P);
        for (std::size_t p = 0; p < P; ++p) {
            const std::string& l = bind->lanes[p];
            const gpudb::ResidentColumn* col = nullptr;
            if (l == "v") col = set->vals.get();
            else {
                const std::size_t idx = static_cast<std::size_t>(std::strtoull(l.c_str() + 1, nullptr, 10));
                if (idx >= set->pred_int)
                    throw std::runtime_error(std::string(fn) + ": payload lane '" + l + "' but the set has " +
                                             std::to_string(set->pred_int) + " BIGINT predicate column(s)");
                col = set->preds[idx].get();
            }
            if (!col || col->dtype() != gpudb::Dtype::I64)
                throw std::runtime_error(std::string(fn) + ": payload lane '" + l + "' is not a BIGINT lane");
            mp[p].vals = col;
            // every payload is reduced in the one pass; `columns` only says what is read back
            mp[p].columns = want[p] ? want[p] : gpudb::GroupByFilter::kAllColumns;
        }
        ResolvedWhere rw = resolve_where(*set, bind->where, fn);
        auto& agg = resident_aggregator(ctx);
        gpudb::DispatchDecision d;
        {
            auto dev = resident_device_lock(ctx);
            init->res = agg.aggregate_exact_masked(mp.data(), P, rw.preds.data(), rw.preds.size());
            d = agg.last_decision();   // process-wide state: copied out under the lock that wrote it
        }
        char buf[320];
        std::snprintf(buf, sizeof(buf),
            "op=agg_exact_global backend=%s reason=%s rows_in=%zu groups=1 rows_out=1 payloads=%zu "
            "kept=%lld wall_ms=%.3f kernel_ms=%.3f transfer_ms=0.000",
            gpudb::to_string(d.chosen), gpudb::to_string(d.reason), init->res.rows_in, P,
            static_cast<long long>(init->res.count_star), init->res.wall_ms, init->res.kernel_ms);
        resident_record_stats(ctx, set.get(), buf);
    } catch (const std::exception& e) {
        duckdb_init_set_error(info, e.what());
        return;
    }
    duckdb_init_set_init_data(info, init.release(), [](void* p) { delete static_cast<GgInitData*>(p); });
}

void gg_function(duckdb_function_info info, duckdb_data_chunk output) {
    auto* init = static_cast<GgInitData*>(duckdb_function_get_init_data(info));
    if (!init || init->done) return;
    init->done = true;
    const auto& r = init->res;
    auto null_at = [](duckdb_vector vec) {
        duckdb_vector_ensure_validity_writable(vec);
        duckdb_validity_set_row_invalid(duckdb_vector_get_validity(vec), 0);
    };
    for (idx_t j = 0; j < init->cols.size(); ++j) {
        duckdb_vector vec = duckdb_data_chunk_get_vector(output, j);
        const idx_t c = init->cols[j];
        if (c == 0) {
            *static_cast<std::int64_t*>(duckdb_vector_get_data(vec)) = r.count_star;
            continue;
        }
        const std::size_t p = (c - 1) / 5;
        if (p >= r.counts.size()) { null_at(vec); continue; }
        const std::int64_t cnt = r.counts[p];
        switch ((c - 1) % 5) {
            case 0: {
                auto* sum = static_cast<duckdb_hugeint*>(duckdb_vector_get_data(vec));
                sum[0].lower = static_cast<std::uint64_t>(r.sums[p]);
                sum[0].upper = r.sums_hi[p];
                if (cnt == 0) null_at(vec);
                break;
            }
            case 1:
                *static_cast<std::int64_t*>(duckdb_vector_get_data(vec)) = cnt;
                break;
            case 2: case 3: {
                auto* v = static_cast<std::int64_t*>(duckdb_vector_get_data(vec));
                v[0] = ((c - 1) % 5 == 2) ? r.mins[p] : r.maxs[p];
                if (cnt == 0) null_at(vec);
                break;
            }
            default: {
                auto* a = static_cast<double*>(duckdb_vector_get_data(vec));
                if (cnt == 0) { a[0] = 0.0; null_at(vec); }
                else {
                    const gpudb::Sum128 sm{static_cast<std::uint64_t>(r.sums[p]), r.sums_hi[p]};
                    a[0] = sm.to_double() / static_cast<double>(cnt);
                }
                break;
            }
        }
    }
    duckdb_data_chunk_set_size(output, 1);
}

// ---------------------------------------------------------------------------
// gpu_topk_resident(name VARCHAR, k BIGINT, order VARCHAR) -> (idx BIGINT, value BIGINT)
// gpu_topk_resident_f64(...)                                 -> (idx BIGINT, value DOUBLE)
// ---------------------------------------------------------------------------

struct TkBindData {
    std::string name;
    std::int64_t k = 0;
    bool descending = false;
    gpudb::Dtype dtype = gpudb::Dtype::I64;
};

struct TkInitData {
    gpudb::TopKResult res;
    gpudb::Dtype dtype = gpudb::Dtype::I64;
    std::size_t offset = 0;
};

// The value column's type is fixed per function (BIGINT / DOUBLE) so bind
// never has to look at the registry — the column may not exist yet when a
// same-statement upload feeds it. The dtype is checked at init.
template <gpudb::Dtype DT>
void tk_bind(duckdb_bind_info info) {
    duckdb_value nv = duckdb_bind_get_parameter(info, 0);
    duckdb_value kv = duckdb_bind_get_parameter(info, 1);
    duckdb_value ov = duckdb_bind_get_parameter(info, 2);
    if (!nv || !kv || !ov || duckdb_is_null_value(nv) || duckdb_is_null_value(kv) ||
        duckdb_is_null_value(ov)) {
        if (nv) duckdb_destroy_value(&nv);
        if (kv) duckdb_destroy_value(&kv);
        if (ov) duckdb_destroy_value(&ov);
        duckdb_bind_set_error(info, "gpu_topk_resident: name, k and order may not be NULL");
        return;
    }
    auto* bind = new TkBindData();
    bind->name = value_to_string(nv);
    bind->k = duckdb_get_int64(kv);
    bind->dtype = DT;
    std::string order = value_to_string(ov);
    duckdb_destroy_value(&nv);
    duckdb_destroy_value(&kv);
    duckdb_destroy_value(&ov);
    for (auto& ch : order) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    if      (order == "asc")  bind->descending = false;
    else if (order == "desc") bind->descending = true;
    else {
        delete bind;
        duckdb_bind_set_error(info, "gpu_topk_resident: order must be 'asc' or 'desc'");
        return;
    }
    if (bind->k < 0) {
        delete bind;
        duckdb_bind_set_error(info, "gpu_topk_resident: k must be >= 0");
        return;
    }

    duckdb_logical_type bigint = duckdb_create_logical_type(DUCKDB_TYPE_BIGINT);
    duckdb_logical_type dbl    = duckdb_create_logical_type(DUCKDB_TYPE_DOUBLE);
    duckdb_bind_add_result_column(info, "idx", bigint);
    duckdb_bind_add_result_column(info, "value", DT == gpudb::Dtype::F64 ? dbl : bigint);
    duckdb_destroy_logical_type(&bigint);
    duckdb_destroy_logical_type(&dbl);
    duckdb_bind_set_bind_data(info, bind, [](void* p) { delete static_cast<TkBindData*>(p); });
}

void tk_init(duckdb_init_info info) {
    auto* bind = static_cast<TkBindData*>(duckdb_init_get_bind_data(info));
    auto* init = new TkInitData();
    init->dtype = bind->dtype;
    try {
        ResidentContext& ctx = resident_context(duckdb_init_get_extra_info(info));
        // A pair name ranks its payload ('p' -> p.v); otherwise a bare column.
        // Never silently fall back to a pair's key column.
        std::shared_ptr<ResidentSet> set = resident_acquire_set(ctx, bind->name, "gpu_topk_resident");
        gpudb::ResidentColumn* target = set->pair ? set->vals.get() : set->keys.get();
        if (target->dtype() != bind->dtype)
            throw std::runtime_error(
                std::string(bind->dtype == gpudb::Dtype::F64 ? "gpu_topk_resident_f64" : "gpu_topk_resident") +
                ": '" + bind->name + "' is " +
                (target->dtype() == gpudb::Dtype::F64 ? "DOUBLE — use gpu_topk_resident_f64"
                                                       : "BIGINT — use gpu_topk_resident"));
        auto& agg = resident_aggregator(ctx);
        // the lock is the GPU call's; the stats line is formatted after it
        gpudb::DispatchDecision d;
        {
            auto dev = resident_device_lock(ctx);
            init->res = agg.topk_resident(*target, static_cast<std::size_t>(bind->k),
                                          bind->descending);
            d = agg.last_decision();
        }
        char buf[320];
        std::snprintf(buf, sizeof(buf),
            "op=topk_resident backend=%s reason=%s rows_in=%zu k=%zu "
            "wall_ms=%.3f kernel_ms=%.3f transfer_ms=%.3f",
            gpudb::to_string(d.chosen), gpudb::to_string(d.reason),
            init->res.rows_in, init->res.idx.size(),
            init->res.wall_ms, init->res.kernel_ms, init->res.transfer_ms);
        resident_record_stats(ctx, set.get(), buf);
    } catch (const std::exception& e) {
        delete init;
        duckdb_init_set_error(info, e.what());
        return;
    }
    duckdb_init_set_init_data(info, init, [](void* p) { delete static_cast<TkInitData*>(p); });
}

void tk_function(duckdb_function_info info, duckdb_data_chunk output) {
    auto* init = static_cast<TkInitData*>(duckdb_function_get_init_data(info));
    if (!init) return;
    const std::size_t remaining = init->res.idx.size() - init->offset;
    if (remaining == 0) return;

    constexpr idx_t kChunk = 2048;
    const idx_t out_n = static_cast<idx_t>(std::min<std::size_t>(remaining, kChunk));
    const std::size_t off = init->offset;

    auto* idx_out = static_cast<std::int64_t*>(
        duckdb_vector_get_data(duckdb_data_chunk_get_vector(output, 0)));
    for (idx_t i = 0; i < out_n; ++i) idx_out[i] = init->res.idx[off + i];
    duckdb_vector vv = duckdb_data_chunk_get_vector(output, 1);
    if (init->dtype == gpudb::Dtype::F64) {
        auto* v = static_cast<double*>(duckdb_vector_get_data(vv));
        for (idx_t i = 0; i < out_n; ++i) v[i] = init->res.values_f64[off + i];
    } else {
        auto* v = static_cast<std::int64_t*>(duckdb_vector_get_data(vv));
        for (idx_t i = 0; i < out_n; ++i) v[i] = init->res.values_i64[off + i];
    }
    duckdb_data_chunk_set_size(output, out_n);
    init->offset += static_cast<std::size_t>(out_n);
}

// ---------------------------------------------------------------------------

void register_table_fn(duckdb_connection con, const char* name,
                       const std::vector<duckdb_type>& params,
                       duckdb_table_function_bind_t bind,
                       duckdb_table_function_init_t init,
                       duckdb_table_function_t fn,
                       const std::shared_ptr<ResidentContext>& ctx,
                       bool projection_pushdown = false) {
    duckdb_table_function tf = duckdb_create_table_function();
    duckdb_table_function_set_name(tf, name);
    if (projection_pushdown) duckdb_table_function_supports_projection_pushdown(tf, true);
    for (duckdb_type t : params) {
        duckdb_logical_type lt = duckdb_create_logical_type(t);
        duckdb_table_function_add_parameter(tf, lt);
        duckdb_destroy_logical_type(&lt);
    }
    duckdb_table_function_set_bind(tf, bind);
    duckdb_table_function_set_init(tf, init);
    duckdb_table_function_set_function(tf, fn);
    duckdb_table_function_set_extra_info(tf, resident_extra_info(ctx), resident_extra_info_destroy);
    if (duckdb_register_table_function(con, tf) == DuckDBError) {
        duckdb_destroy_table_function(&tf);
        throw std::runtime_error(std::string(name) + " registration failed");
    }
    duckdb_destroy_table_function(&tf);
}

} // namespace

void register_gpu_groupby(duckdb_connection con, const std::shared_ptr<ResidentContext>& ctx) {
    register_table_fn(con, "gpu_groupby_sum_resident", {DUCKDB_TYPE_VARCHAR},
                      gb_bind<GbOp::SumI64, GbForm::Plain>, gb_init, gb_function, ctx);
    register_table_fn(con, "gpu_groupby_sum_resident_f64", {DUCKDB_TYPE_VARCHAR},
                      gb_bind<GbOp::SumF64, GbForm::Plain>, gb_init, gb_function, ctx);
    register_table_fn(con, "gpu_groupby_count_resident", {DUCKDB_TYPE_VARCHAR},
                      gb_bind<GbOp::Count, GbForm::Plain>, gb_init, gb_function, ctx);
    // HAVING on the aggregate (cmp VARCHAR, threshold) — evaluated on the device
    register_table_fn(con, "gpu_groupby_sum_resident_having",
                      {DUCKDB_TYPE_VARCHAR, DUCKDB_TYPE_VARCHAR, DUCKDB_TYPE_BIGINT},
                      gb_bind<GbOp::SumI64, GbForm::Having>, gb_init, gb_function, ctx);
    register_table_fn(con, "gpu_groupby_sum_resident_f64_having",
                      {DUCKDB_TYPE_VARCHAR, DUCKDB_TYPE_VARCHAR, DUCKDB_TYPE_DOUBLE},
                      gb_bind<GbOp::SumF64, GbForm::Having>, gb_init, gb_function, ctx);
    register_table_fn(con, "gpu_groupby_count_resident_having",
                      {DUCKDB_TYPE_VARCHAR, DUCKDB_TYPE_VARCHAR, DUCKDB_TYPE_BIGINT},
                      gb_bind<GbOp::Count, GbForm::Having>, gb_init, gb_function, ctx);
    // top-k of groups by the aggregate (k BIGINT, order VARCHAR)
    register_table_fn(con, "gpu_groupby_sum_resident_topk",
                      {DUCKDB_TYPE_VARCHAR, DUCKDB_TYPE_BIGINT, DUCKDB_TYPE_VARCHAR},
                      gb_bind<GbOp::SumI64, GbForm::TopK>, gb_init, gb_function, ctx);
    register_table_fn(con, "gpu_groupby_sum_resident_f64_topk",
                      {DUCKDB_TYPE_VARCHAR, DUCKDB_TYPE_BIGINT, DUCKDB_TYPE_VARCHAR},
                      gb_bind<GbOp::SumF64, GbForm::TopK>, gb_init, gb_function, ctx);
    register_table_fn(con, "gpu_groupby_count_resident_topk",
                      {DUCKDB_TYPE_VARCHAR, DUCKDB_TYPE_BIGINT, DUCKDB_TYPE_VARCHAR},
                      gb_bind<GbOp::Count, GbForm::TopK>, gb_init, gb_function, ctx);
    // exact GROUP BY (v0.7 §4): full native tuple, HUGEINT sum, NULL semantics
    register_table_fn(con, "gpu_groupby_exact_resident", {DUCKDB_TYPE_VARCHAR},
                      gx_bind<GbForm::Plain, false>, gx_init, gx_function, ctx, /*projection_pushdown*/true);
    register_table_fn(con, "gpu_groupby_exact_resident_having",
                      {DUCKDB_TYPE_VARCHAR, DUCKDB_TYPE_VARCHAR, DUCKDB_TYPE_VARCHAR, DUCKDB_TYPE_BIGINT},
                      gx_bind<GbForm::Having, false>, gx_init, gx_function, ctx, /*projection_pushdown*/true);
    register_table_fn(con, "gpu_groupby_exact_resident_topk",
                      {DUCKDB_TYPE_VARCHAR, DUCKDB_TYPE_VARCHAR, DUCKDB_TYPE_BIGINT, DUCKDB_TYPE_VARCHAR},
                      gx_bind<GbForm::TopK, false>, gx_init, gx_function, ctx, /*projection_pushdown*/true);
    // ... with a WHERE program over the set's predicate columns (§4.6)
    register_table_fn(con, "gpu_groupby_exact_resident_where", {DUCKDB_TYPE_VARCHAR, DUCKDB_TYPE_VARCHAR},
                      gx_bind<GbForm::Plain, true>, gx_init, gx_function, ctx, /*projection_pushdown*/true);
    register_table_fn(con, "gpu_groupby_exact_resident_where_having",
                      {DUCKDB_TYPE_VARCHAR, DUCKDB_TYPE_VARCHAR, DUCKDB_TYPE_VARCHAR, DUCKDB_TYPE_VARCHAR, DUCKDB_TYPE_BIGINT},
                      gx_bind<GbForm::Having, true>, gx_init, gx_function, ctx, /*projection_pushdown*/true);
    register_table_fn(con, "gpu_groupby_exact_resident_where_topk",
                      {DUCKDB_TYPE_VARCHAR, DUCKDB_TYPE_VARCHAR, DUCKDB_TYPE_VARCHAR, DUCKDB_TYPE_BIGINT, DUCKDB_TYPE_VARCHAR},
                      gx_bind<GbForm::TopK, true>, gx_init, gx_function, ctx, /*projection_pushdown*/true);
    register_table_fn(con, "gpu_groupby_exact_multi",
                      {DUCKDB_TYPE_VARCHAR, DUCKDB_TYPE_VARCHAR, DUCKDB_TYPE_VARCHAR, DUCKDB_TYPE_VARCHAR},
                      gm_bind, gm_init, gm_function, ctx, /*projection_pushdown*/true);
    register_table_fn(con, "gpu_agg_exact_global",
                      {DUCKDB_TYPE_VARCHAR, DUCKDB_TYPE_VARCHAR, DUCKDB_TYPE_VARCHAR},
                      gg_bind, gg_init, gg_function, ctx, /*projection_pushdown*/true);
    register_table_fn(con, "gpu_topk_resident",
                      {DUCKDB_TYPE_VARCHAR, DUCKDB_TYPE_BIGINT, DUCKDB_TYPE_VARCHAR},
                      tk_bind<gpudb::Dtype::I64>, tk_init, tk_function, ctx);
    register_table_fn(con, "gpu_topk_resident_f64",
                      {DUCKDB_TYPE_VARCHAR, DUCKDB_TYPE_BIGINT, DUCKDB_TYPE_VARCHAR},
                      tk_bind<gpudb::Dtype::F64>, tk_init, tk_function, ctx);
    std::fprintf(stderr,
        "[gpudb] registered gpu_groupby_{sum,sum_f64,count,exact}_resident[_having|_topk] + gpu_topk_resident[_f64]\n");
}

} // namespace gpudb_ext
