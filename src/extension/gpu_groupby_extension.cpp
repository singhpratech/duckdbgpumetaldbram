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
        const auto& d = agg.last_decision();
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
// gpu_upload_rows_exact) or f<n> (n-th DOUBLE one). Constants are integer
// literals for k / v / i<n> and decimal or 'nan' / 'inf' literals for f<n>.
// The program is a conjunction; an empty program is no WHERE.
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
        if (lrest == "is null")          { t.op = gpudb::Predicate::Op::IsNull; }
        else if (lrest == "is not null") { t.op = gpudb::Predicate::Op::IsNotNull; }
        else if (lrest.rfind("in", 0) == 0 && (lrest.size() == 2 || lrest[2] == ' ' || lrest[2] == '(')) {
            std::string body = trim_copy(rest.substr(2));
            if (body.size() < 2 || body.front() != '(' || body.back() != ')')
                return "WHERE program: 'in' needs a parenthesised list in '" + term + "'";
            body = body.substr(1, body.size() - 2);
            std::size_t p2 = 0;
            while (p2 <= body.size()) {
                std::size_t c = body.find(',', p2);
                if (c == std::string::npos) c = body.size();
                std::string item = trim_copy(body.substr(p2, c - p2));
                if (!item.empty()) t.list.push_back(item);
                p2 = c + 1;
                if (c == body.size()) break;
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
            t.value = trim_copy(rest.substr(j));
            if (t.value.empty()) return "WHERE program: missing constant in '" + term + "'";
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

std::int64_t parse_i64_literal(const std::string& s, const std::string& term) {
    errno = 0;
    char* end = nullptr;
    const long long v = std::strtoll(s.c_str(), &end, 10);
    if (errno != 0 || !end || end == s.c_str() || *end != '\0')
        throw std::runtime_error("WHERE program: '" + s + "' is not a BIGINT constant (" + term + ")");
    return static_cast<std::int64_t>(v);
}
std::int64_t parse_f64_literal_bits(const std::string& s, const std::string& term) {
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
            throw std::runtime_error("WHERE program: '" + s + "' is not a DOUBLE constant (" + term + ")");
    }
    std::int64_t bits;
    std::memcpy(&bits, &d, sizeof(bits));
    return bits;
}

ResolvedWhere resolve_where(const ResidentSet& set, const std::vector<WhereTerm>& terms, const char* fn) {
    ResolvedWhere rw;
    rw.lists.resize(terms.size());
    for (std::size_t t = 0; t < terms.size(); ++t) {
        const WhereTerm& w = terms[t];
        gpudb::Predicate p;
        p.op = w.op;
        const std::string& c = w.col;
        bool is_f64 = false;
        if (c == "k")      p.col = set.keys.get();
        else if (c == "v") p.col = set.vals.get();
        else if ((c[0] == 'i' || c[0] == 'f') && c.size() > 1) {
            std::size_t idx = 0;
            for (std::size_t q = 1; q < c.size(); ++q) {
                if (!std::isdigit(static_cast<unsigned char>(c[q])))
                    throw std::runtime_error(std::string(fn) + ": WHERE program: unknown column '" + c + "'");
                idx = idx * 10 + static_cast<std::size_t>(c[q] - '0');
            }
            const bool is_i = c[0] == 'i';
            const std::size_t count = is_i ? set.pred_int : set.pred_dbl;
            if (idx >= count)
                throw std::runtime_error(std::string(fn) + ": WHERE program: '" + c + "' but the set has " +
                    std::to_string(count) + (is_i ? " BIGINT" : " DOUBLE") + " predicate column(s)");
            p.col = set.preds[(is_i ? 0 : set.pred_int) + idx].get();
            is_f64 = !is_i;
        } else {
            throw std::runtime_error(std::string(fn) + ": WHERE program: unknown column '" + c + "'");
        }
        const std::string term = w.col + (w.value.empty() ? "" : " ... " + w.value);
        if (p.op == gpudb::Predicate::Op::In) {
            for (const auto& s : w.list)
                rw.lists[t].push_back(is_f64 ? parse_f64_literal_bits(s, term) : parse_i64_literal(s, term));
            p.list = rw.lists[t].data();
            p.n_list = rw.lists[t].size();
        } else if (p.op != gpudb::Predicate::Op::IsNull && p.op != gpudb::Predicate::Op::IsNotNull) {
            p.value = is_f64 ? parse_f64_literal_bits(w.value, term) : parse_i64_literal(w.value, term);
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
        auto dev = resident_device_lock(ctx);
        if (where)
            init->res = agg.groupby_exact_masked_resident(*set->keys, set->vals.get(),
                                                          rw.preds.data(), rw.preds.size(), cap, filt);
        else
            init->res = agg.groupby_exact_resident(*set->keys, set->vals.get(), cap, filt);
        {
            const auto& rr = init->res;
            init->rows = std::max({ rr.keys.size(), rr.sums.size(), rr.counts.size(), rr.counts_star.size(),
                                    rr.mins.size(), rr.maxs.size() });
        }
        const auto& d = agg.last_decision();
        char buf[320];
        std::snprintf(buf, sizeof(buf),
            "op=%s backend=%s reason=%s rows_in=%zu groups=%zu rows_out=%zu "
            "wall_ms=%.3f kernel_ms=%.3f transfer_ms=%.3f",
            fn + 4 /* strip "gpu_" */, gpudb::to_string(d.chosen),
            gpudb::to_string(d.reason), init->res.rows_in, init->res.groups_total,
            init->rows,
            init->res.wall_ms, init->res.kernel_ms, init->res.transfer_ms);
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
        auto dev = resident_device_lock(ctx);
        init->res = agg.topk_resident(*target, static_cast<std::size_t>(bind->k),
                                      bind->descending);
        const auto& d = agg.last_decision();
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
