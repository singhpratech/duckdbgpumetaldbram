// gpu_rewrite.cpp — see gpu_rewrite.hpp.
//
// Shape handled (docs/TRANSPARENT_DESIGN.md §2, first cut = milestone 2):
//
//   SELECT [k [AS a]] [, sum(v) [AS b]] [, count(*) | count(v) [AS c]] ...
//   FROM t                                   -- one base table, no sample / AT
//   GROUP BY k                               -- one integer-family key
//   [HAVING sum(v) | count(*) {> >= < <=} <constant>]
//   [ORDER BY <output> [ASC|DESC]] [LIMIT n]  -- kept; pushed as top-k when
//                                            -- ORDER BY is the aggregate
//
// into
//
//   SELECT CAST("key" AS <type>) AS <name>, CAST(sum AS HUGEINT) AS <name>, ...
//   FROM gpu_groupby_{sum,count}_resident[_having|_topk]('<tag>', ...) r,
//        (SELECT gpu_assert_rows('<tag>', count(*)) AS ok FROM <cat>.<sch>.<t>) gd
//   WHERE gd.ok
//   [ORDER BY ...] [LIMIT n]
//
// Every field of every node on the path is read; anything unexpected means
// "unchanged, reason shape" (§2 "rejected by field, not by node"). Output
// names and types come from the wrapper's DESCRIBE of the original
// statement (context.outputs), so what the user sees is what native shows.
//
// DECIMAL(p<=18, s) payloads: the wrapper uploaded (v * 10^s)::BIGINT, so
// HAVING thresholds are rescaled by 10^s with the exact rounding rule per
// comparison (> floors, >= ceils, < ceils, <= floors) and the sum comes back
// as CAST(sum AS DECIMAL(38-s, 0)) * 10^-s, which DuckDB types DECIMAL(38, s)
// and which equals native bit-for-bit (verified on 1.5.5).

#include "gpu_rewrite.hpp"

#if defined(GPUDB_C_STRUCT_ABI)
DUCKDB_EXTENSION_EXTERN
#endif

#include "nlohmann/json.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpudb_ext {
namespace {

using json = nlohmann::json;

constexpr json::number_unsigned_t kNoLocation = 18446744073709551615ULL;

// A rejection: unwinds the matcher with a reason keyword (+ detail).
struct Reject {
    std::string reason;
    std::string detail;
};
[[noreturn]] void reject(const char* reason, std::string detail = "") {
    throw Reject{reason, std::move(detail)};
}

std::string lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}
bool ieq(const std::string& a, const std::string& b) { return lower(a) == lower(b); }

// ---------------------------------------------------------------------------
// Context (what the wrapper resolved).
// ---------------------------------------------------------------------------

struct ColInfo {
    std::string type;        // DuckDB type name as DESCRIBE prints it
    int scale = 0;           // DECIMAL scale (0 for integers)
    int width = 0;
    bool integer = false;    // integer family
    bool decimal = false;
    bool floating = false;
};

struct Output { std::string name, type; };

struct Context {
    std::string tag;
    std::string catalog, schema, table;
    std::string key_col, val_col;          // from the tag's column list
    ColInfo key, val;
    std::string backend;
    std::int64_t rows = 0;
    bool ready = false;
    std::string default_order;             // "ASC" | "DESC" | ""
    std::vector<Output> outputs;
    std::int64_t min_rows = 0;
    // stats
    bool key_has_null = true, val_has_null = true, have_stats = false;
    double val_abs_max = 0;
};

ColInfo parse_type(std::string t) {
    ColInfo c;
    c.type = t;
    std::string u = t;
    for (auto& ch : u) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    if (u == "BIGINT" || u == "INTEGER" || u == "SMALLINT" || u == "TINYINT" || u == "INT" ||
        u == "INT8" || u == "INT4" || u == "INT2" || u == "INT1" || u == "LONG" || u == "SHORT") {
        c.integer = true;
    } else if (u.rfind("DECIMAL(", 0) == 0 || u.rfind("NUMERIC(", 0) == 0) {
        c.decimal = true;
        if (std::sscanf(u.c_str() + 8, "%d,%d", &c.width, &c.scale) != 2) { c.width = 18; c.scale = 3; }
    } else if (u == "DOUBLE" || u == "FLOAT" || u == "REAL" || u == "FLOAT4" || u == "FLOAT8") {
        c.floating = true;
    }
    return c;
}

Context parse_context(const json& c) {
    Context x;
    if (!c.is_object()) reject("error", "context is not a JSON object");
    x.tag = c.value("tag", "");
    if (c.contains("table") && c["table"].is_object()) {
        const json& t = c["table"];
        x.catalog = t.value("catalog", "");
        x.schema  = t.value("schema", "");
        x.table   = t.value("name", "");
    }
    x.backend = c.value("backend", "");
    x.rows    = c.value("rows", std::int64_t{0});
    x.ready   = c.value("ready", false);
    x.default_order = c.value("default_order", "");
    if (c.contains("outputs") && c["outputs"].is_array()) {
        for (const auto& o : c["outputs"])
            x.outputs.push_back({o.value("name", ""), o.value("type", "")});
    }
    if (c.contains("thresholds") && c["thresholds"].is_object())
        x.min_rows = c["thresholds"].value("min_rows", std::int64_t{0});

    // Key / payload names: the tag's 7th ':'-field ("k,v").
    {
        std::vector<std::string> f;
        std::size_t pos = 0;
        while (f.size() < 7) {
            const std::size_t k = x.tag.find(':', pos);
            if (k == std::string::npos) { f.push_back(x.tag.substr(pos)); break; }
            f.push_back(x.tag.substr(pos, k - pos));
            pos = k + 1;
        }
        if (f.size() >= 7) {
            const std::string& cols = f[6];
            const std::size_t comma = cols.find(',');
            x.key_col = cols.substr(0, comma);
            if (comma != std::string::npos) x.val_col = cols.substr(comma + 1);
            const std::size_t comma2 = x.val_col.find(',');
            if (comma2 != std::string::npos) x.val_col = x.val_col.substr(0, comma2);
        }
    }
    // Column types: {"col": "TYPE"} or {"col": {"type": "TYPE", "scale": s}}.
    auto col_info = [&](const std::string& name) -> ColInfo {
        if (!c.contains("columns") || !c["columns"].is_object()) return ColInfo{};
        for (auto it = c["columns"].begin(); it != c["columns"].end(); ++it) {
            if (!ieq(it.key(), name)) continue;
            if (it.value().is_string()) return parse_type(it.value().get<std::string>());
            if (it.value().is_object()) {
                ColInfo ci = parse_type(it.value().value("type", ""));
                if (it.value().contains("scale")) ci.scale = it.value().value("scale", 0);
                return ci;
            }
        }
        return ColInfo{};
    };
    x.key = col_info(x.key_col);
    x.val = col_info(x.val_col);
    // Stats: {"col": {"has_null": b, "min": n, "max": n}}
    if (c.contains("stats") && c["stats"].is_object()) {
        x.have_stats = true;
        for (auto it = c["stats"].begin(); it != c["stats"].end(); ++it) {
            const json& s = it.value();
            if (!s.is_object()) continue;
            const bool hn = s.value("has_null", true);
            if (ieq(it.key(), x.key_col)) x.key_has_null = hn;
            if (ieq(it.key(), x.val_col)) {
                x.val_has_null = hn;
                double mn = 0, mx = 0;
                if (s.contains("min") && s["min"].is_number()) mn = s["min"].get<double>();
                if (s.contains("max") && s["max"].is_number()) mx = s["max"].get<double>();
                x.val_abs_max = std::max(std::fabs(mn), std::fabs(mx));
            }
        }
    }
    return x;
}

// ---------------------------------------------------------------------------
// Tree readers.
// ---------------------------------------------------------------------------

const json& field(const json& n, const char* k) {
    auto it = n.find(k);
    if (it == n.end()) reject("shape", std::string("missing field ") + k);
    return *it;
}
std::string sfield(const json& n, const char* k) {
    const json& v = field(n, k);
    return v.is_string() ? v.get<std::string>() : std::string();
}
bool is_null(const json& n, const char* k) {
    auto it = n.find(k);
    return it == n.end() || it->is_null();
}

// COLUMN_REF → the bare column name if it refers to base table `t` (alias
// `ta`): ["c"], [t, "c"], [alias, "c"]. Anything else → "".
std::string column_ref(const json& e, const std::string& t, const std::string& ta) {
    if (sfield(e, "class") != "COLUMN_REF") return "";
    const json& names = field(e, "column_names");
    if (!names.is_array() || names.empty() || names.size() > 2) return "";
    if (names.size() == 2) {
        const std::string q = names[0].get<std::string>();
        if (!ieq(q, t) && (ta.empty() || !ieq(q, ta))) return "";
    }
    return names.back().get<std::string>();
}

enum class Agg { None, Sum, Count };

// A plain aggregate over the pair: sum(v), count(*), count(v). Reads every
// field on the FUNCTION node; FILTER / DISTINCT / ORDER BY inside → shape.
Agg aggregate_of(const json& e, const Context& cx, const std::string& t, const std::string& ta) {
    if (sfield(e, "class") != "FUNCTION") return Agg::None;
    if (!is_null(e, "filter")) reject("shape", "FILTER on an aggregate");
    if (field(e, "distinct").get<bool>()) reject("shape", "DISTINCT inside an aggregate");
    const json& ob = field(e, "order_bys");
    if (ob.is_object() && ob.contains("orders") && !ob["orders"].empty())
        reject("shape", "ORDER BY inside an aggregate");
    if (!sfield(e, "schema").empty() && !ieq(sfield(e, "schema"), "main")) reject("shape", "schema-qualified function");
    const std::string fn = lower(sfield(e, "function_name"));
    const json& ch = field(e, "children");
    if (fn == "count_star") {
        if (!ch.empty()) reject("shape", "count_star with arguments");
        return Agg::Count;
    }
    if (fn == "sum" || fn == "count") {
        if (ch.size() != 1) reject("shape", fn + " with " + std::to_string(ch.size()) + " arguments");
        const std::string col = column_ref(ch[0], t, ta);
        if (col.empty() || cx.val_col.empty() || !ieq(col, cx.val_col)) {
            if (fn == "count" && !col.empty() && ieq(col, cx.key_col)) return Agg::Count;  // count(k) == count(*) with no NULL keys
            reject("shape", cx.val_col.empty() ? fn + " over a set that holds only the key column"
                                              : fn + " over a column that is not the resident payload");
        }
        return fn == "sum" ? Agg::Sum : Agg::Count;
    }
    reject("shape", "aggregate " + fn + " is not on the transparent path yet");
}

// ---------------------------------------------------------------------------
// Constants and threshold rescaling.
// ---------------------------------------------------------------------------

struct Threshold { std::int64_t value = 0; };

std::int64_t pow10_i64(int e) {
    std::int64_t p = 1;
    for (int i = 0; i < e; ++i) {
        if (p > std::numeric_limits<std::int64_t>::max() / 10) reject("overflow", "10^" + std::to_string(e));
        p *= 10;
    }
    return p;
}
std::int64_t floor_div(std::int64_t a, std::int64_t b) {   // b > 0
    std::int64_t q = a / b, r = a % b;
    return (r != 0 && a < 0) ? q - 1 : q;
}
std::int64_t ceil_div(std::int64_t a, std::int64_t b) {    // b > 0
    std::int64_t q = a / b, r = a % b;
    return (r != 0 && a > 0) ? q + 1 : q;
}

// cmp: 1 = '>', 2 = '>=', 3 = '<', 4 = '<='. The aggregate is an integer
// (sum of scaled payloads, or a count); an exact rational threshold x maps
// to the integer that keeps the comparison exact.
std::int64_t round_threshold(std::int64_t num, std::int64_t den, int cmp) {
    switch (cmp) {
        case 1: return floor_div(num, den);   // a > x   ⇔ a > floor(x)
        case 2: return ceil_div(num, den);    // a >= x  ⇔ a >= ceil(x)
        case 3: return ceil_div(num, den);    // a < x   ⇔ a < ceil(x)
        default: return floor_div(num, den);  // a <= x  ⇔ a <= floor(x)
    }
}

// Reads a VALUE_CONSTANT and returns the threshold in the aggregate's scale.
Threshold threshold_of(const json& e, int agg_scale, int cmp) {
    if (sfield(e, "class") != "CONSTANT") reject("shape", "HAVING threshold is not a constant");
    const json& v = field(e, "value");
    if (field(v, "is_null").get<bool>()) reject("shape", "NULL HAVING threshold");
    const json& ty = field(v, "type");
    const std::string id = sfield(ty, "id");
    const json& val = field(v, "value");
    Threshold t;
    if (id == "INTEGER" || id == "BIGINT" || id == "SMALLINT" || id == "TINYINT" ||
        id == "UINTEGER" || id == "USMALLINT" || id == "UTINYINT") {
        if (!val.is_number_integer()) reject("shape", "non-integer constant payload");
        const std::int64_t x = val.get<std::int64_t>();
        const std::int64_t m = pow10_i64(agg_scale);
        if (m != 1 && (x > std::numeric_limits<std::int64_t>::max() / m ||
                       x < std::numeric_limits<std::int64_t>::min() / m))
            reject("overflow", "threshold does not fit int64 at the payload scale");
        t.value = x * m;
        return t;
    }
    if (id == "DECIMAL") {
        const json& ti = field(ty, "type_info");
        const int sc = ti.is_object() ? ti.value("scale", 0) : 0;
        if (!val.is_number_integer()) reject("decimal", "DECIMAL constant wider than int64");
        const std::int64_t u = val.get<std::int64_t>();
        if (sc <= agg_scale) {
            const std::int64_t m = pow10_i64(agg_scale - sc);
            if (m != 1 && (u > std::numeric_limits<std::int64_t>::max() / m ||
                           u < std::numeric_limits<std::int64_t>::min() / m))
                reject("overflow", "threshold does not fit int64 at the payload scale");
            t.value = u * m;
        } else {
            t.value = round_threshold(u, pow10_i64(sc - agg_scale), cmp);
        }
        return t;
    }
    if (id == "DOUBLE" || id == "FLOAT") {
        if (!val.is_number()) reject("shape", "non-numeric floating constant");
        const double x = val.get<double>() * std::pow(10.0, agg_scale);
        if (!std::isfinite(x) || std::fabs(x) >= 9007199254740992.0)
            reject("shape", "floating threshold outside the exact integer range");
        const double r = (cmp == 1 || cmp == 4) ? std::floor(x) : std::ceil(x);
        t.value = static_cast<std::int64_t>(r);
        return t;
    }
    reject("shape", "HAVING threshold of type " + id);
}

// ---------------------------------------------------------------------------
// Node builders (field sets exactly as json_serialize_sql emits them).
// ---------------------------------------------------------------------------

json j_colref(const std::string& name, const std::string& alias = "") {
    return json{{"class", "COLUMN_REF"}, {"type", "COLUMN_REF"}, {"alias", alias},
                {"query_location", kNoLocation}, {"column_names", json::array({name})}};
}
json j_colref2(const std::string& q, const std::string& name) {
    return json{{"class", "COLUMN_REF"}, {"type", "COLUMN_REF"}, {"alias", ""},
                {"query_location", kNoLocation}, {"column_names", json::array({q, name})}};
}
json j_type(const std::string& id) {
    return json{{"id", id}, {"type_info", nullptr}};
}
json j_decimal_type(int width, int scale) {
    return json{{"id", "DECIMAL"}, {"type_info", json{{"type", "DECIMAL_TYPE_INFO"}, {"alias", ""},
                {"extension_info", nullptr}, {"width", width}, {"scale", scale}}}};
}
json j_const_varchar(const std::string& s) {
    return json{{"class", "CONSTANT"}, {"type", "VALUE_CONSTANT"}, {"alias", ""},
                {"query_location", kNoLocation},
                {"value", json{{"type", j_type("VARCHAR")}, {"is_null", false}, {"value", s}}}};
}
json j_const_bigint(std::int64_t v) {
    return json{{"class", "CONSTANT"}, {"type", "VALUE_CONSTANT"}, {"alias", ""},
                {"query_location", kNoLocation},
                {"value", json{{"type", j_type("BIGINT")}, {"is_null", false}, {"value", v}}}};
}
json j_const_decimal(int width, int scale, std::int64_t unscaled) {
    return json{{"class", "CONSTANT"}, {"type", "VALUE_CONSTANT"}, {"alias", ""},
                {"query_location", kNoLocation},
                {"value", json{{"type", j_decimal_type(width, scale)}, {"is_null", false}, {"value", unscaled}}}};
}
json j_function(const std::string& name, json children, bool is_operator = false,
                const std::string& alias = "") {
    return json{{"class", "FUNCTION"}, {"type", "FUNCTION"}, {"alias", alias},
                {"query_location", kNoLocation}, {"function_name", name}, {"schema", ""},
                {"children", std::move(children)}, {"filter", nullptr},
                {"order_bys", json{{"type", "ORDER_MODIFIER"}, {"orders", json::array()}}},
                {"distinct", false}, {"is_operator", is_operator}, {"export_state", false},
                {"catalog", ""}};
}
json j_cast(json child, json type, const std::string& alias = "") {
    return json{{"class", "CAST"}, {"type", "OPERATOR_CAST"}, {"alias", alias},
                {"query_location", kNoLocation}, {"child", std::move(child)},
                {"cast_type", std::move(type)}, {"try_cast", false}};
}

// CAST(<col> AS <type string from DESCRIBE>) AS <name>. HUGEINT / integer
// types are a plain cast; DECIMAL(p, s) becomes CAST(col AS DECIMAL(38-s,0)) * 10^-s.
json j_output(const std::string& col, const std::string& type, const std::string& name) {
    ColInfo t = parse_type(type);
    if (t.decimal) {
        if (t.scale < 0 || t.scale > 18) reject("decimal", "output scale " + std::to_string(t.scale));
        if (t.scale == 0) return j_cast(j_colref(col), j_decimal_type(t.width, 0), name);
        json inner = j_cast(j_colref(col), j_decimal_type(38 - t.scale, 0));
        json factor = j_const_decimal(t.scale + 1, t.scale, 1);      // 10^-s
        return j_function("*", json::array({inner, factor}), true, name);
    }
    std::string id = type;
    for (auto& ch : id) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    if (id == "INT") id = "INTEGER";
    if (id == "BIGINT") return j_colref(col, name);   // already BIGINT: no cast
    return j_cast(j_colref(col), j_type(id), name);
}

// ---------------------------------------------------------------------------
// The rewrite.
// ---------------------------------------------------------------------------

struct Result {
    json tree;
    bool rewritten = false;
    std::string reason, detail, form;
};

Result do_rewrite(json tree, const json& ctxj) {
    Result r;
    if (!tree.is_object()) reject("error", "tree is not a JSON object");
    if (tree.value("error", false)) reject("error", "json_serialize_sql reported an error");
    const json& stmts = field(tree, "statements");
    if (!stmts.is_array() || stmts.size() != 1) reject("shape", "not exactly one statement");
    json node = field(stmts[0], "node");
    if (sfield(node, "type") != "SELECT_NODE") reject("shape", "not a SELECT");
    if (stmts[0].contains("named_param_map") && !stmts[0]["named_param_map"].empty())
        reject("shape", "named parameters");

    const Context cx = parse_context(ctxj);
    if (cx.tag.empty() || cx.key_col.empty()) reject("error", "context has no identity tag with a column list");

    // ---- read every field of the select node ----
    if (!is_null(node, "sample") || !is_null(node, "qualify")) reject("shape", "SAMPLE / QUALIFY");
    if (sfield(node, "aggregate_handling") != "STANDARD_HANDLING") reject("shape", "GROUP BY ALL");
    const json& cte = field(node, "cte_map");
    if (cte.is_object() && cte.contains("map") && !cte["map"].empty()) reject("shape", "CTE in scope");
    if (!is_null(node, "where_clause")) reject("shape", "WHERE (predicate mask is a later milestone)");

    const json& from = field(node, "from_table");
    if (sfield(from, "type") != "BASE_TABLE") reject("shape", "FROM is not one base table");
    if (!is_null(from, "sample") || !is_null(from, "at_clause")) reject("shape", "TABLESAMPLE / AT");
    if (from.contains("column_name_alias") && !from["column_name_alias"].empty()) reject("shape", "column aliases on the table");
    const std::string tname = sfield(from, "table_name");
    const std::string talias = sfield(from, "alias");
    if (!ieq(tname, cx.table)) reject("shape", "table '" + tname + "' is not the resolved table");
    if (!sfield(from, "schema_name").empty() && !ieq(sfield(from, "schema_name"), cx.schema))
        reject("shape", "schema does not match the resolved table");
    if (!sfield(from, "catalog_name").empty() && !ieq(sfield(from, "catalog_name"), cx.catalog))
        reject("shape", "catalog does not match the resolved table");

    const json& gexp = field(node, "group_expressions");
    const json& gsets = field(node, "group_sets");
    if (!gexp.is_array() || gexp.size() != 1) reject("shape", "GROUP BY must have exactly one key");
    if (!gsets.is_array() || gsets.size() != 1 || !gsets[0].is_array() || gsets[0].size() != 1)
        reject("shape", "ROLLUP / CUBE / GROUPING SETS");
    if (sfield(gexp[0], "class") == "CONSTANT") reject("shape", "ordinal GROUP BY");
    const std::string gcol = column_ref(gexp[0], tname, talias);
    if (gcol.empty() || !ieq(gcol, cx.key_col)) reject("shape", "GROUP BY key is not the resident key");

    // ---- column types (rule 2 gates) ----
    // A one-column tag (columns == [key]) is a bare key set: count-only
    // shapes over gpu_groupby_count_resident(tag) — no payload, no sum.
    const bool bare = cx.val_col.empty();
    if (!cx.key.integer) reject(cx.key.floating ? "double" : "shape", "key type " + cx.key.type);
    if (!bare) {
        if (cx.val.floating) reject("double", "payload type " + cx.val.type);
        if (cx.val.decimal && cx.val.width > 18) reject("decimal", "payload " + cx.val.type);
        if (!cx.val.integer && !cx.val.decimal) reject("shape", "payload type '" + cx.val.type + "'");
    }
    const int scale = (!bare && cx.val.decimal) ? cx.val.scale : 0;

    // ---- select list ----
    struct Item { enum Kind { Key, Sum, Count } kind; std::string alias; };
    std::vector<Item> items;
    const json& sel = field(node, "select_list");
    if (!sel.is_array() || sel.empty()) reject("shape", "empty select list");
    bool want_sum = false, want_count = false;
    for (const json& e : sel) {
        const std::string cls = sfield(e, "class");
        if (cls == "STAR") reject("shape", "SELECT *");
        if (cls == "WINDOW") reject("shape", "window function");
        if (cls == "PARAMETER") reject("shape", "parameter");
        const std::string alias = sfield(e, "alias");
        if (cls == "COLUMN_REF") {
            const std::string c = column_ref(e, tname, talias);
            if (c.empty() || !ieq(c, cx.key_col)) reject("shape", "column that is not the GROUP BY key");
            items.push_back({Item::Key, alias});
            continue;
        }
        const Agg a = aggregate_of(e, cx, tname, talias);
        if (a == Agg::None) reject("shape", "select-list expression of class " + cls);
        if (a == Agg::Sum && bare) reject("shape", "sum over a set that holds only the key column");
        if (a == Agg::Sum) { want_sum = true; items.push_back({Item::Sum, alias}); }
        else               { want_count = true; items.push_back({Item::Count, alias}); }
    }
    if (cx.outputs.size() != items.size()) reject("error", "context.outputs does not match the select list");

    // ---- HAVING ----
    int cmp = 0;
    Agg having_agg = Agg::None;
    std::int64_t threshold = 0;
    if (!is_null(node, "having")) {
        const json& h = node["having"];
        if (sfield(h, "class") != "COMPARISON") reject("shape", "HAVING is not one comparison");
        const std::string ct = sfield(h, "type");
        bool flip = false;
        const json* aggside = &field(h, "left");
        const json* cside   = &field(h, "right");
        if (sfield(*aggside, "class") == "CONSTANT") { std::swap(aggside, cside); flip = true; }
        if      (ct == "COMPARE_GREATERTHAN")          cmp = flip ? 3 : 1;
        else if (ct == "COMPARE_GREATERTHANOREQUALTO") cmp = flip ? 4 : 2;
        else if (ct == "COMPARE_LESSTHAN")             cmp = flip ? 1 : 3;
        else if (ct == "COMPARE_LESSTHANOREQUALTO")    cmp = flip ? 2 : 4;
        else reject("shape", "HAVING comparison " + ct);
        having_agg = aggregate_of(*aggside, cx, tname, talias);
        if (having_agg == Agg::None) reject("shape", "HAVING is not on an aggregate");
        threshold = threshold_of(*cside, having_agg == Agg::Sum ? scale : 0, cmp).value;
        if (having_agg == Agg::Sum && bare) reject("shape", "HAVING sum over a set that holds only the key column");
        if (having_agg == Agg::Sum) want_sum = true;
    }
    // One device function produces the rows: the SUM form yields
    // (key, sum, count); the COUNT form only (key, count). A HAVING on count
    // while the select needs sum has no device form, so the count predicate
    // goes to the outer WHERE over the (key, sum, count) rows — same rows,
    // names and types as native, the filter just runs on the small result.
    const bool use_sum = want_sum;
    const bool having_on_host = having_agg == Agg::Count && use_sum;
    (void)want_count;

    // ---- modifiers: ORDER BY / LIMIT kept; top-k pushed when possible ----
    const json& mods = field(node, "modifiers");
    const json* order_mod = nullptr;
    const json* limit_mod = nullptr;
    for (const json& m : mods) {
        const std::string mt = sfield(m, "type");
        if      (mt == "ORDER_MODIFIER" && !order_mod) order_mod = &m;
        else if (mt == "LIMIT_MODIFIER" && !limit_mod) limit_mod = &m;
        else reject("shape", "modifier " + mt);
    }

    // Names of the outputs each item maps to (native names from DESCRIBE).
    auto output_name_for = [&](Item::Kind k) -> std::string {
        for (std::size_t i = 0; i < items.size(); ++i) if (items[i].kind == k) return cx.outputs[i].name;
        return "";
    };

    // Remap an ORDER BY expression to the rewritten outputs.
    auto remap = [&](const json& e) -> json {
        const std::string cls = sfield(e, "class");
        if (cls == "COLUMN_REF") {
            const json& names = field(e, "column_names");
            if (names.size() == 1) {
                const std::string n = names[0].get<std::string>();
                for (std::size_t i = 0; i < items.size(); ++i)
                    if (ieq(n, cx.outputs[i].name) || (!items[i].alias.empty() && ieq(n, items[i].alias)))
                        return j_colref(cx.outputs[i].name);
            }
            const std::string c = column_ref(e, tname, talias);
            if (!c.empty() && ieq(c, cx.key_col)) {
                const std::string on = output_name_for(Item::Key);
                if (on.empty()) reject("shape", "ORDER BY the key when the key is not selected");
                return j_colref(on);
            }
            reject("shape", "ORDER BY an expression the rewrite cannot map");
        }
        if (cls == "FUNCTION") {
            const Agg a = aggregate_of(e, cx, tname, talias);
            const std::string on = output_name_for(a == Agg::Sum ? Item::Sum : Item::Count);
            if (on.empty()) reject("shape", "ORDER BY an aggregate that is not selected");
            return j_colref(on);
        }
        if (cls == "CONSTANT") {
            // ORDER BY <ordinal>: the i-th select-list item, 1-based.
            const json& v = field(e, "value");
            if (!v.contains("value") || !v["value"].is_number_integer()) reject("shape", "non-integer ordinal ORDER BY");
            const std::int64_t i = v["value"].get<std::int64_t>();
            if (i < 1 || static_cast<std::size_t>(i) > items.size()) reject("shape", "ordinal ORDER BY out of range");
            return j_colref(cx.outputs[static_cast<std::size_t>(i - 1)].name);
        }
        reject("shape", "ORDER BY expression of class " + cls);
    };

    // Top-k push: ORDER BY <the aggregate> [dir] LIMIT k, no HAVING.
    std::int64_t topk = 0;
    bool topk_desc = false;
    json new_mods = json::array();
    if (order_mod) {
        json om = *order_mod;
        json& orders = om["orders"];
        if (!orders.is_array() || orders.empty()) reject("shape", "empty ORDER BY");
        for (json& o : orders) o["expression"] = remap(o["expression"]);
        if (orders.size() == 1 && limit_mod && having_agg == Agg::None) {
            const json& o = orders[0];
            const json& lim = field(*limit_mod, "limit");
            const bool off_null = is_null(*limit_mod, "offset");
            const std::string ordered = o["expression"]["column_names"][0].get<std::string>();
            const std::string agg_name = output_name_for(use_sum ? Item::Sum : Item::Count);
            std::string dir = sfield(o, "type");
            if (dir == "ORDER_DEFAULT") dir = ieq(cx.default_order, "DESC") ? "DESCENDING" :
                                              ieq(cx.default_order, "ASC")  ? "ASCENDING" : "";
            if (off_null && !agg_name.empty() && ieq(ordered, agg_name) && !dir.empty() &&
                sfield(lim, "class") == "CONSTANT") {
                const json& lv = field(lim, "value");
                if (!field(lv, "is_null").get<bool>() && lv["value"].is_number_integer()) {
                    const std::int64_t k = lv["value"].get<std::int64_t>();
                    if (k > 0) { topk = k; topk_desc = dir == "DESCENDING"; }
                }
            }
        }
        new_mods.push_back(om);
    }
    if (limit_mod) new_mods.push_back(*limit_mod);

    // ---- residency / thresholds / rule-2 gates (after the shape, so the
    //      reason names the first thing the user can act on) ----
    if (ieq(cx.backend, "CPU") || cx.backend.empty()) reject("backend", "backend " + cx.backend);
    if (!cx.have_stats) reject("nulls", "no column statistics in context");
    if (cx.key_has_null || (!bare && cx.val_has_null)) reject("nulls", "key or payload may hold NULLs");
    if (use_sum) {
        // rows * max|v| (already in the payload's scale when DECIMAL) < 2^63
        const double bound = static_cast<double>(cx.rows) * cx.val_abs_max * std::pow(10.0, scale);
        if (bound >= 9.2e18) reject("overflow", "sum may exceed int64");
    }
    if (cx.rows < cx.min_rows) reject("threshold", std::to_string(cx.rows) + " rows below the floor");
    if (!cx.ready) reject("not_resident", "set is not ready");

    // ---- build the replacement ----
    std::string fn = use_sum ? "gpu_groupby_sum_resident" : "gpu_groupby_count_resident";
    json args = json::array({j_const_varchar(cx.tag)});
    if (having_agg != Agg::None && !having_on_host) {
        fn += "_having";
        static const char* ops[] = {"", ">", ">=", "<", "<="};
        args.push_back(j_const_varchar(ops[cmp]));
        args.push_back(j_const_bigint(threshold));
        r.form = "having";
    } else if (having_on_host) {
        r.form = "having";
    } else if (topk > 0) {
        fn += "_topk";
        args.push_back(j_const_bigint(topk));
        args.push_back(j_const_varchar(topk_desc ? "desc" : "asc"));
        r.form = "topk";
    } else {
        r.form = "plain";
    }

    json tf = json{{"type", "TABLE_FUNCTION"}, {"alias", "r"}, {"sample", nullptr},
                   {"query_location", kNoLocation}, {"function", j_function(fn, args)},
                   {"column_name_alias", json::array()}, {"with_ordinality", "WITHOUT_ORDINALITY"}};
    json guard_select = json{{"type", "SELECT_NODE"}, {"modifiers", json::array()},
                             {"cte_map", json{{"map", json::array()}}},
                             {"select_list", json::array({j_function("gpu_assert_rows",
                                 json::array({j_const_varchar(cx.tag), j_function("count_star", json::array())}),
                                 false, "ok")})},
                             {"from_table", json{{"type", "BASE_TABLE"}, {"alias", ""}, {"sample", nullptr},
                                 {"query_location", kNoLocation}, {"schema_name", cx.schema},
                                 {"table_name", cx.table}, {"column_name_alias", json::array()},
                                 {"catalog_name", cx.catalog}, {"at_clause", nullptr}}},
                             {"where_clause", nullptr}, {"group_expressions", json::array()},
                             {"group_sets", json::array()}, {"aggregate_handling", "STANDARD_HANDLING"},
                             {"having", nullptr}, {"sample", nullptr}, {"qualify", nullptr}};
    json guard = json{{"type", "SUBQUERY"}, {"alias", "gd"}, {"sample", nullptr},
                      {"query_location", kNoLocation},
                      {"subquery", json{{"node", guard_select}, {"named_param_map", json::array()}}},
                      {"column_name_alias", json::array()}};
    json new_from = json{{"type", "JOIN"}, {"alias", ""}, {"sample", nullptr},
                         {"query_location", kNoLocation}, {"left", tf}, {"right", guard},
                         {"condition", nullptr}, {"join_type", "INNER"}, {"ref_type", "CROSS"},
                         {"using_columns", json::array()}, {"delim_flipped", false},
                         {"duplicate_eliminated_columns", json::array()}};

    json new_select = json::array();
    for (std::size_t i = 0; i < items.size(); ++i) {
        const Output& o = cx.outputs[i];
        switch (items[i].kind) {
            case Item::Key:   new_select.push_back(j_output("key", o.type, o.name)); break;
            case Item::Sum:   new_select.push_back(j_output("sum", o.type, o.name)); break;
            case Item::Count: new_select.push_back(j_output("count", o.type, o.name)); break;
        }
    }

    node["select_list"] = new_select;
    node["from_table"] = new_from;
    if (having_on_host) {
        static const char* types[] = {"", "COMPARE_GREATERTHAN", "COMPARE_GREATERTHANOREQUALTO",
                                      "COMPARE_LESSTHAN", "COMPARE_LESSTHANOREQUALTO"};
        json pred = json{{"class", "COMPARISON"}, {"type", types[cmp]}, {"alias", ""},
                         {"query_location", kNoLocation}, {"left", j_colref2("r", "count")},
                         {"right", j_const_bigint(threshold)}};
        node["where_clause"] = json{{"class", "CONJUNCTION"}, {"type", "CONJUNCTION_AND"}, {"alias", ""},
                                    {"query_location", kNoLocation},
                                    {"children", json::array({j_colref2("gd", "ok"), pred})}};
    } else {
        node["where_clause"] = j_colref2("gd", "ok");
    }
    node["group_expressions"] = json::array();
    node["group_sets"] = json::array();
    node["having"] = nullptr;
    node["modifiers"] = new_mods;
    tree["statements"][0]["node"] = node;
    r.tree = std::move(tree);
    r.rewritten = true;
    r.reason = "rewritten";
    return r;
}

} // namespace

std::string rewrite_ast(const std::string& tree_json, const std::string& context_json) {
    json tree;
    json ctx;
    Result r;
    try {
        if (tree_json.size() > (std::size_t(16) << 10) * 64)   // 1 MiB of JSON: the wrapper stops at 16 KB of SQL
            throw Reject{"too_long", std::to_string(tree_json.size()) + " bytes"};
        tree = json::parse(tree_json);
        ctx = json::parse(context_json);
        r = do_rewrite(tree, ctx);
    } catch (const Reject& rj) {
        r.tree = tree.is_null() ? json::object() : tree;
        r.rewritten = false;
        r.reason = rj.reason;
        r.detail = rj.detail;
    } catch (const std::exception& e) {
        r.tree = tree.is_null() ? json::object() : tree;
        r.rewritten = false;
        r.reason = "error";
        r.detail = e.what();
    }
    json info = json{{"rewritten", r.rewritten}, {"reason", r.reason}};
    if (!r.detail.empty()) info["detail"] = r.detail;
    if (r.rewritten) {
        info["form"] = r.form;
        info["set"] = ctx.value("tag", "");
    }
    if (!r.tree.is_object()) r.tree = json::object();
    r.tree["gpudb"] = info;
    return r.tree.dump();
}

namespace {

void rewrite_exec(duckdb_function_info info, duckdb_data_chunk input, duckdb_vector output) {
    duckdb_vector tv = duckdb_data_chunk_get_vector(input, 0);
    duckdb_vector cv = duckdb_data_chunk_get_vector(input, 1);
    auto* trees = reinterpret_cast<duckdb_string_t*>(duckdb_vector_get_data(tv));
    auto* ctxs  = reinterpret_cast<duckdb_string_t*>(duckdb_vector_get_data(cv));
    uint64_t* tval = duckdb_vector_get_validity(tv);
    uint64_t* cval = duckdb_vector_get_validity(cv);
    const idx_t n = duckdb_data_chunk_get_size(input);
    duckdb_vector_ensure_validity_writable(output);
    uint64_t* oval = duckdb_vector_get_validity(output);
    for (idx_t i = 0; i < n; ++i) {
        if ((tval && !duckdb_validity_row_is_valid(tval, i)) ||
            (cval && !duckdb_validity_row_is_valid(cval, i))) {
            duckdb_validity_set_row_invalid(oval, i);
            continue;
        }
        const std::string tree(duckdb_string_t_data(&trees[i]), duckdb_string_t_length(trees[i]));
        const std::string ctx(duckdb_string_t_data(&ctxs[i]), duckdb_string_t_length(ctxs[i]));
        const std::string out = rewrite_ast(tree, ctx);
        duckdb_vector_assign_string_element_len(output, i, out.data(), out.size());
    }
    (void)info;
}

} // namespace

void register_gpu_rewrite(duckdb_connection con) {
    duckdb_scalar_function fn = duckdb_create_scalar_function();
    duckdb_scalar_function_set_name(fn, "gpu_rewrite_ast");
    duckdb_logical_type vc = duckdb_create_logical_type(DUCKDB_TYPE_VARCHAR);
    duckdb_scalar_function_add_parameter(fn, vc);
    duckdb_scalar_function_add_parameter(fn, vc);
    duckdb_scalar_function_set_return_type(fn, vc);
    duckdb_destroy_logical_type(&vc);
    duckdb_scalar_function_set_function(fn, rewrite_exec);
    // Deliberately NOT volatile: pure function of its two arguments, so
    // DuckDB may constant-fold and cache it.
    duckdb_state st = duckdb_register_scalar_function(con, fn);
    duckdb_destroy_scalar_function(&fn);
    if (st == DuckDBError) throw std::runtime_error("gpu_rewrite_ast registration failed");
}

} // namespace gpudb_ext
