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
#include <functional>
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
    bool date = false;       // DATE: resident as days since 1970-01-01 (BIGINT lane)
    bool timestamp = false;  // TIMESTAMP: resident as microseconds since the epoch (BIGINT lane)
    bool temporal() const { return date || timestamp; }
};

struct Output { std::string name, type; };

struct PredCol { std::string name; ColInfo info; std::string lane; };   // lane: "i<n>" / "f<n>"
// One component of the GROUP BY key. A single key is resident as is; two or
// three (§4.4) are packed into one BIGINT as a mixed-radix number: component
// i occupies slot (v - min + 1), 0 meaning NULL, with radix range = max - min
// + 2 and stride = product of the later ranges.
struct KeyPart { std::string name; ColInfo info; std::int64_t min = 0; std::int64_t range = 0; std::int64_t stride = 1; };

struct Context {
    std::string tag;
    std::string catalog, schema, table;
    std::string key_col, val_col;          // from the tag's column list
    ColInfo key, val;
    bool exact = false;                    // the set was uploaded by gpu_upload_rows_exact (v0.7 §4)
    std::vector<PredCol> preds;            // tag columns after key/payload, in order (exact sets)
    std::vector<KeyPart> keys;             // 1..3 key components (packed when > 1)
    bool packed() const { return keys.size() > 1; }
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
    } else if (u == "DATE") {
        c.date = true;
    } else if (u == "TIMESTAMP" || u == "DATETIME" || u == "TIMESTAMP WITHOUT TIME ZONE") {
        c.timestamp = true;
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
    x.exact   = c.value("exact", false);
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
            // "k[,v[,p1,p2,...]]" — an exact set's payload slot may be "-" (count-only shapes)
            std::vector<std::string> cols;
            const std::string& cl = f[6];
            std::size_t p0 = 0;
            while (true) {
                const std::size_t k = cl.find(',', p0);
                cols.push_back(cl.substr(p0, k == std::string::npos ? std::string::npos : k - p0));
                if (k == std::string::npos) break;
                p0 = k + 1;
            }
            if (!cols.empty()) x.key_col = cols[0];
            if (cols.size() >= 2 && cols[1] != "-") x.val_col = cols[1];
            for (std::size_t i = 2; i < cols.size(); ++i) x.preds.push_back(PredCol{cols[i], ColInfo{}, ""});
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
    {
        int ni = 0, nf = 0;
        for (auto& pc : x.preds) {
            pc.info = col_info(pc.name);
            if (pc.info.floating) pc.lane = "f" + std::to_string(nf++);
            else                  pc.lane = "i" + std::to_string(ni++);   // integer family and DECIMAL
        }
    }
    // Key components: the tag's key field is "a" or "a+b[+c]"; "keys" in the
    // context carries each component's integer image bounds for packing.
    {
        std::vector<std::string> names;
        std::size_t p0 = 0;
        while (true) {
            const std::size_t k = x.key_col.find('+', p0);
            names.push_back(x.key_col.substr(p0, k == std::string::npos ? std::string::npos : k - p0));
            if (k == std::string::npos) break;
            p0 = k + 1;
        }
        for (const auto& n : names) { KeyPart kp; kp.name = n; kp.info = col_info(n); x.keys.push_back(kp); }
        if (c.contains("keys") && c["keys"].is_array()) {
            const json& ks = c["keys"];
            if (ks.size() != x.keys.size()) reject("error", "context.keys does not match the tag's key field");
            for (std::size_t i = 0; i < ks.size(); ++i) {
                const json& k = ks[i];
                if (k.is_object()) {
                    if (k.contains("type") && k["type"].is_string()) x.keys[i].info = parse_type(k["type"].get<std::string>());
                    if (k.contains("min") && k["min"].is_number_integer() && k.contains("max") && k["max"].is_number_integer()) {
                        x.keys[i].min = k["min"].get<std::int64_t>();
                        const std::int64_t mx = k["max"].get<std::int64_t>();
                        if (mx < x.keys[i].min) reject("error", "context.keys max < min");
                        // range = max - min + 2 (one slot for NULL), checked against int64
                        const std::uint64_t span = static_cast<std::uint64_t>(mx) - static_cast<std::uint64_t>(x.keys[i].min);
                        if (span > (std::uint64_t{1} << 62)) reject("shape", "packed key component range too wide");
                        x.keys[i].range = static_cast<std::int64_t>(span) + 2;
                    }
                }
            }
        }
        if (x.keys.size() > 3) reject("shape", "more than three GROUP BY keys");
        if (x.keys.size() > 1) {
            // strides from the right; the product must fit in int64
            std::int64_t stride = 1;
            for (std::size_t i = x.keys.size(); i-- > 0;) {
                if (x.keys[i].range < 2) reject("error", "context.keys has no bounds for a packed key component");
                x.keys[i].stride = stride;
                if (stride > std::numeric_limits<std::int64_t>::max() / x.keys[i].range)
                    reject("shape", "packed key does not fit in 64 bits");
                stride *= x.keys[i].range;
            }
        }
        x.key = x.keys.empty() ? ColInfo{} : x.keys[0].info;
    }
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

// ---- temporal constants (DATE 'yyyy-mm-dd' serialises as CAST('yyyy-mm-dd' AS DATE)) ----
// Days since 1970-01-01 for a proleptic Gregorian civil date (H. Hinnant).
std::int64_t days_from_civil(std::int64_t y, unsigned m, unsigned d) {
    y -= m <= 2;
    const std::int64_t era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = static_cast<unsigned>(y - era * 400);
    const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + static_cast<std::int64_t>(doe) - 719468;
}

// Parses "yyyy-mm-dd" (optionally "[-]yyyy-mm-dd"); ok = false when not that shape.
std::int64_t parse_date_days(const std::string& s, bool& ok) {
    ok = false;
    std::size_t i = 0;
    bool neg = false;
    if (i < s.size() && s[i] == '-') { neg = true; ++i; }
    auto num = [&](std::size_t min_digits, std::int64_t& out) {
        std::size_t st = i; out = 0;
        while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) { out = out * 10 + (s[i] - '0'); ++i; }
        return i - st >= min_digits;
    };
    std::int64_t y, m, d;
    if (!num(1, y) || i >= s.size() || s[i] != '-') return 0; ++i;
    if (!num(1, m) || i >= s.size() || s[i] != '-') return 0; ++i;
    if (!num(1, d)) return 0;
    while (i < s.size() && s[i] == ' ') ++i;
    if (i != s.size()) return 0;
    if (m < 1 || m > 12 || d < 1 || d > 31) return 0;
    ok = true;
    return days_from_civil(neg ? -y : y, static_cast<unsigned>(m), static_cast<unsigned>(d));
}

// Parses "yyyy-mm-dd[ T]hh:mm[:ss[.ffffff]]" (or a bare date) to microseconds since
// the epoch; ok = false for anything else (time zones, named specials).
std::int64_t parse_timestamp_us(const std::string& s, bool& ok) {
    ok = false;
    const std::size_t sep = s.find_first_of(" T");
    const std::string ds = sep == std::string::npos ? s : s.substr(0, sep);
    bool dok = false;
    const std::int64_t days = parse_date_days(ds, dok);
    if (!dok) return 0;
    std::int64_t us = 0;
    if (sep != std::string::npos) {
        const std::string ts = s.substr(sep + 1);
        std::size_t i = 0;
        auto num2 = [&](std::int64_t& out) {
            std::size_t st = i; out = 0;
            while (i < ts.size() && std::isdigit(static_cast<unsigned char>(ts[i]))) { out = out * 10 + (ts[i] - '0'); ++i; }
            return i > st;
        };
        std::int64_t hh = 0, mm = 0, ss = 0;
        if (!num2(hh) || i >= ts.size() || ts[i] != ':') return 0; ++i;
        if (!num2(mm)) return 0;
        if (i < ts.size() && ts[i] == ':') { ++i; if (!num2(ss)) return 0; }
        std::int64_t frac = 0;
        if (i < ts.size() && ts[i] == '.') {
            ++i; int digits = 0;
            while (i < ts.size() && std::isdigit(static_cast<unsigned char>(ts[i]))) {
                if (digits < 6) { frac = frac * 10 + (ts[i] - '0'); ++digits; }
                else if (ts[i] != '0') return 0;     // finer than microseconds: not representable
                ++i;
            }
            while (digits < 6) { frac *= 10; ++digits; }
        }
        if (i != ts.size()) return 0;                 // time zone or trailing text: declined
        if (hh > 23 || mm > 59 || ss > 59) return 0;
        us = ((hh * 60 + mm) * 60 + ss) * 1000000 + frac;
    }
    ok = true;
    return days * 86400000000LL + us;
}

// If `e` is CAST(<VARCHAR constant> AS DATE|TIMESTAMP), returns the integer
// image the resident lane holds (days / microseconds) and sets kind to the
// ColInfo flag; otherwise returns false.
bool temporal_const(const json& e, ColInfo& kind, std::int64_t& value) {
    if (sfield(e, "class") != "CAST") return false;
    const json& ch = field(e, "child");
    if (sfield(ch, "class") != "CONSTANT") return false;
    const json& v = field(ch, "value");
    if (field(v, "is_null").get<bool>()) reject("shape", "NULL temporal constant");
    if (sfield(field(v, "type"), "id") != "VARCHAR") return false;
    const std::string target = sfield(field(e, "cast_type"), "id");
    const std::string text = field(v, "value").get<std::string>();
    bool ok = false;
    kind = ColInfo{};
    if (target == "DATE") { kind.date = true; value = parse_date_days(text, ok); }
    else if (target == "TIMESTAMP") { kind.timestamp = true; value = parse_timestamp_us(text, ok); }
    else return false;
    if (!ok) reject("shape", "temporal constant '" + text + "' is not a plain literal");
    return true;
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

// ---------------------------------------------------------------------------
// The exact path (v0.7 §4): sets from gpu_upload_rows_exact, functions
// gpu_groupby_exact_resident[_where][_having|_topk], native NULL semantics,
// HUGEINT sums, the full aggregate set and a WHERE program.
// ---------------------------------------------------------------------------

enum class XAgg { None, Sum, CountV, CountStar, Min, Max, Avg };

const char* xagg_col(XAgg a) {
    switch (a) {
        case XAgg::Sum: return "sum";        case XAgg::CountV: return "count";
        case XAgg::CountStar: return "count_star"; case XAgg::Min: return "min";
        case XAgg::Max: return "max";        case XAgg::Avg: return "avg";
        default: return "";
    }
}

// The aggregate a FUNCTION node computes over the exact set, or None.
XAgg xagg_of(const json& e, const Context& cx, const std::string& t, const std::string& ta) {
    if (sfield(e, "class") != "FUNCTION") return XAgg::None;
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
        return XAgg::CountStar;
    }
    if (fn == "count" || fn == "sum" || fn == "min" || fn == "max" || fn == "avg") {
        if (ch.size() != 1) reject("shape", fn + " with " + std::to_string(ch.size()) + " arguments");
        const std::string col = column_ref(ch[0], t, ta);
        if (col.empty()) reject("shape", fn + " over an expression");
        if (cx.val_col.empty() || !ieq(col, cx.val_col))
            reject("shape", cx.val_col.empty() ? fn + " over a set that holds only the key column"
                                              : fn + " over a column that is not the resident payload");
        if (fn == "count") return XAgg::CountV;
        if (fn == "sum")   return XAgg::Sum;
        if (fn == "min")   return XAgg::Min;
        if (fn == "max")   return XAgg::Max;
        return XAgg::Avg;
    }
    reject("shape", "aggregate " + fn + " is not on the transparent path yet");
}

// Native output type of an aggregate over this set (what DESCRIBE reports).
std::string xagg_native_type(XAgg a, const Context& cx) {
    switch (a) {
        case XAgg::Sum:       return cx.val.decimal ? "DECIMAL(38," + std::to_string(cx.val.scale) + ")" : "HUGEINT";
        case XAgg::CountV: case XAgg::CountStar: return "BIGINT";
        case XAgg::Min: case XAgg::Max: return cx.val.type;
        case XAgg::Avg:       return "DOUBLE";
        default: return "";
    }
}

// Output expression for a column of the exact function, typed exactly as
// native: integers and HUGEINT by cast; DECIMAL(p, s) through the exact
// scaled multiply, then CAST to DECIMAL(p, s) when p != 38.
// A BIGINT-valued expression typed as `type` (integer family / DATE / TIMESTAMP).
json j_typed_expr(json expr, const std::string& type, const std::string& name) {
    ColInfo t = parse_type(type);
    if (t.date) {
        json epoch = j_cast(j_const_varchar("1970-01-01"), j_type("DATE"));
        return j_function("+", json::array({epoch, j_cast(std::move(expr), j_type("INTEGER"))}), true, name);
    }
    if (t.timestamp) return j_function("make_timestamp", json::array({std::move(expr)}), false, name);
    std::string id = type;
    for (auto& ch : id) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    if (id == "INT") id = "INTEGER";
    json out = j_cast(std::move(expr), j_type(id), name);
    return out;
}

// Component i of a packed key, back to its column: nullif((key // stride) % range, 0) - 1 + min.
json j_key_component(const Context& cx, std::size_t i, const std::string& type, const std::string& name) {
    const KeyPart& kp = cx.keys[i];
    json slot = j_colref("key");
    if (kp.stride > 1) slot = j_function("//", json::array({slot, j_const_bigint(kp.stride)}), true);
    if (i > 0)         slot = j_function("%",  json::array({slot, j_const_bigint(kp.range)}), true);
    json v = j_function("nullif", json::array({slot, j_const_bigint(0)}));
    v = j_function("-", json::array({v, j_const_bigint(1)}), true);
    v = j_function("+", json::array({v, j_const_bigint(kp.min)}), true);
    return j_typed_expr(std::move(v), type, name);
}

json j_output_exact(const std::string& col, const std::string& type, const std::string& name) {
    ColInfo t = parse_type(type);
    if (t.date) {
        // DATE '1970-01-01' + CAST(col AS INTEGER)  (days since the epoch)
        json epoch = j_cast(j_const_varchar("1970-01-01"), j_type("DATE"));
        return j_function("+", json::array({epoch, j_cast(j_colref(col), j_type("INTEGER"))}), true, name);
    }
    if (t.timestamp) return j_function("make_timestamp", json::array({j_colref(col)}), false, name);
    if (t.decimal && t.width != 38 && t.scale > 0) {
        json inner = j_output(col, "DECIMAL(38," + std::to_string(t.scale) + ")", "");
        return j_cast(std::move(inner), j_decimal_type(t.width, t.scale), name);
    }
    return j_output(col, type, name);
}

// A constant as an exact int64 at `scale`; ok = false when not representable
// (a fractional constant against an integer / coarser DECIMAL column).
std::int64_t const_exact_scaled(const json& e, int scale, bool& ok) {
    ok = true;
    if (sfield(e, "class") != "CONSTANT") reject("shape", "not a constant");
    const json& v = field(e, "value");
    if (field(v, "is_null").get<bool>()) reject("shape", "NULL constant");
    const std::string id = sfield(field(v, "type"), "id");
    const json& val = field(v, "value");
    const std::int64_t m = pow10_i64(scale);
    auto scale_up = [&](std::int64_t x) {
        if (m != 1 && (x > std::numeric_limits<std::int64_t>::max() / m ||
                       x < std::numeric_limits<std::int64_t>::min() / m))
            reject("overflow", "constant does not fit int64 at the column's scale");
        return x * m;
    };
    if (id == "INTEGER" || id == "BIGINT" || id == "SMALLINT" || id == "TINYINT" ||
        id == "UINTEGER" || id == "USMALLINT" || id == "UTINYINT") {
        if (!val.is_number_integer()) reject("shape", "non-integer constant payload");
        return scale_up(val.get<std::int64_t>());
    }
    if (id == "DECIMAL") {
        const json& ti = field(field(v, "type"), "type_info");
        const int sc = ti.is_object() ? ti.value("scale", 0) : 0;
        if (!val.is_number_integer()) reject("decimal", "DECIMAL constant wider than int64");
        const std::int64_t u = val.get<std::int64_t>();
        if (sc <= scale) {
            const std::int64_t mm = pow10_i64(scale - sc);
            if (mm != 1 && (u > std::numeric_limits<std::int64_t>::max() / mm ||
                            u < std::numeric_limits<std::int64_t>::min() / mm))
                reject("overflow", "constant does not fit int64 at the column's scale");
            return u * mm;
        }
        const std::int64_t d = pow10_i64(sc - scale);
        if (u % d != 0) { ok = false; return 0; }
        return u / d;
    }
    if (id == "DOUBLE" || id == "FLOAT") {
        if (!val.is_number()) reject("shape", "non-numeric floating constant");
        const double x = val.get<double>() * std::pow(10.0, scale);
        if (!std::isfinite(x) || std::fabs(x) >= 9007199254740992.0)
            reject("shape", "floating constant outside the exact integer range");
        if (x != std::floor(x)) { ok = false; return 0; }
        return static_cast<std::int64_t>(x);
    }
    reject("shape", "constant of type " + id);
}

// A constant as the double DuckDB compares a DOUBLE column against.
double const_as_double(const json& e) {
    if (sfield(e, "class") != "CONSTANT") reject("shape", "not a constant");
    const json& v = field(e, "value");
    if (field(v, "is_null").get<bool>()) reject("shape", "NULL constant");
    const std::string id = sfield(field(v, "type"), "id");
    const json& val = field(v, "value");
    if (id == "DOUBLE" || id == "FLOAT") { if (!val.is_number()) reject("shape", "non-numeric constant"); return val.get<double>(); }
    if (id == "DECIMAL") {
        const json& ti = field(field(v, "type"), "type_info");
        const int sc = ti.is_object() ? ti.value("scale", 0) : 0;
        if (!val.is_number_integer()) reject("decimal", "DECIMAL constant wider than int64");
        return static_cast<double>(val.get<std::int64_t>()) / std::pow(10.0, sc);
    }
    if (val.is_number_integer()) return static_cast<double>(val.get<std::int64_t>());
    reject("shape", "constant of type " + id);
}

std::string fmt_double(double d) {
    if (std::isnan(d)) return "nan";
    if (std::isinf(d)) return d > 0 ? "inf" : "-inf";
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.17g", d);
    return buf;
}

// WHERE → the program of gpu_groupby_exact_resident_where: a conjunction of
// <lane> <op> <const>, <lane> in (...), <lane> is [not] null over k, v and
// the set's predicate lanes. Anything else declines the statement (shape).
struct WhereOut { std::string program; };

WhereOut where_program(const json& w, const Context& cx, const std::string& t, const std::string& ta) {
    WhereOut out;
    std::vector<std::string> terms;
    // lane + column info for a COLUMN_REF, or reject
    auto lane_of = [&](const json& e, ColInfo& info) -> std::string {
        const std::string c = column_ref(e, t, ta);
        if (c.empty()) reject("shape", "WHERE on an expression");
        for (const auto& pc : cx.preds) if (ieq(c, pc.name)) { info = pc.info; return pc.lane; }
        if (!cx.packed() && ieq(c, cx.key_col)) { info = cx.key; return "k"; }
        if (!cx.val_col.empty() && ieq(c, cx.val_col)) { info = cx.val; return "v"; }
        reject("shape", "WHERE column '" + c + "' is not in the resident set");
    };
    auto const_text = [&](const json& c, const ColInfo& info, int cmp, bool& drop_term, bool& always_false) -> std::string {
        // cmp: 1 > 2 >= 3 < 4 <= 5 = 6 <>
        drop_term = false; always_false = false;
        if (info.temporal()) {
            ColInfo ck; std::int64_t v = 0;
            if (!temporal_const(c, ck, v) || ck.date != info.date || ck.timestamp != info.timestamp)
                reject("shape", "WHERE on a " + std::string(info.date ? "DATE" : "TIMESTAMP") +
                                " column against a constant that is not a plain literal of that type");
            return std::to_string(v);
        }
        if (info.floating) return fmt_double(const_as_double(c));
        if (!info.integer && !info.decimal) reject("shape", "WHERE on a column of type " + info.type);
        const int scale = info.decimal ? info.scale : 0;
        if (cmp >= 1 && cmp <= 4) return std::to_string(threshold_of(c, scale, cmp).value);
        bool ok = true;
        const std::int64_t x = const_exact_scaled(c, scale, ok);
        if (!ok) { if (cmp == 5) always_false = true; else drop_term = true; return ""; }   // = never true; <> always true
        return std::to_string(x);
    };
    std::function<void(const json&)> walk = [&](const json& e) {
        const std::string cls = sfield(e, "class");
        const std::string ty  = sfield(e, "type");
        if (cls == "CONJUNCTION") {
            if (ty != "CONJUNCTION_AND") reject("shape", "OR in WHERE");
            for (const json& ch : field(e, "children")) walk(ch);
            return;
        }
        if (cls == "COMPARISON") {
            const json* col = &field(e, "left");
            const json* cst = &field(e, "right");
            bool flip = false;
            if (sfield(*col, "class") == "CONSTANT") { std::swap(col, cst); flip = true; }
            int cmp = 0;
            if      (ty == "COMPARE_GREATERTHAN")          cmp = flip ? 3 : 1;
            else if (ty == "COMPARE_GREATERTHANOREQUALTO") cmp = flip ? 4 : 2;
            else if (ty == "COMPARE_LESSTHAN")             cmp = flip ? 1 : 3;
            else if (ty == "COMPARE_LESSTHANOREQUALTO")    cmp = flip ? 2 : 4;
            else if (ty == "COMPARE_EQUAL")                cmp = 5;
            else if (ty == "COMPARE_NOTEQUAL")             cmp = 6;
            else reject("shape", "WHERE comparison " + ty);
            ColInfo info;
            const std::string lane = lane_of(*col, info);
            bool drop = false, never = false;
            const std::string c = const_text(*cst, info, cmp, drop, never);
            if (never) reject("shape", "WHERE equality against a constant the column cannot hold");
            if (drop) return;
            static const char* ops[] = {"", ">", ">=", "<", "<=", "=", "!="};
            terms.push_back(lane + " " + ops[cmp] + " " + c);
            return;
        }
        if (cls == "OPERATOR" && (ty == "OPERATOR_IS_NULL" || ty == "OPERATOR_IS_NOT_NULL")) {
            const json& ch = field(e, "children");
            if (ch.size() != 1) reject("shape", "IS NULL arity");
            ColInfo info;
            terms.push_back(lane_of(ch[0], info) + (ty == "OPERATOR_IS_NULL" ? " is null" : " is not null"));
            return;
        }
        if (cls == "OPERATOR" && ty == "COMPARE_IN") {
            const json& ch = field(e, "children");
            if (ch.size() < 2) reject("shape", "IN without values");
            ColInfo info;
            const std::string lane = lane_of(ch[0], info);
            std::vector<std::string> vals;
            for (std::size_t i = 1; i < ch.size(); ++i) {
                if (info.temporal()) {
                    bool drop = false, never = false;
                    vals.push_back(const_text(ch[i], info, 5, drop, never));
                    continue;
                }
                if (sfield(ch[i], "class") != "CONSTANT") reject("shape", "IN over a non-constant");
                const json& v = field(ch[i], "value");
                if (field(v, "is_null").get<bool>()) continue;    // x IN (.., NULL) is never TRUE for that element
                bool drop = false, never = false;
                const std::string c = const_text(ch[i], info, 5, drop, never);
                if (never) continue;                                // not representable: matches nothing
                vals.push_back(c);
            }
            if (vals.empty()) reject("shape", "IN list with no representable value");
            std::string term = lane + " in (";
            for (std::size_t i = 0; i < vals.size(); ++i) term += (i ? ", " : "") + vals[i];
            terms.push_back(term + ")");
            return;
        }
        if (cls == "BETWEEN") {
            if (ty != "COMPARE_BETWEEN") reject("shape", "WHERE " + ty);
            ColInfo info;
            const std::string lane = lane_of(field(e, "input"), info);
            bool d1 = false, n1 = false, d2 = false, n2 = false;
            const std::string lo = const_text(field(e, "lower"), info, 2, d1, n1);
            const std::string hi = const_text(field(e, "upper"), info, 4, d2, n2);
            terms.push_back(lane + " >= " + lo);
            terms.push_back(lane + " <= " + hi);
            return;
        }
        reject("shape", "WHERE expression of class " + cls);
    };
    walk(w);
    for (std::size_t i = 0; i < terms.size(); ++i) out.program += (i ? "; " : "") + terms[i];
    return out;
}

Result do_rewrite_exact(json tree, json node, const Context& cx,
                        const std::string& tname, const std::string& talias) {
    Result r;
    // ---- column types (rule 2 gates) ----
    const bool bare = cx.val_col.empty();
    for (const auto& kp : cx.keys)
        if (!kp.info.integer && !kp.info.temporal()) reject(kp.info.floating ? "double" : "shape", "key type " + kp.info.type);
    if (!bare) {
        if (cx.val.floating) reject("double", "payload type " + cx.val.type);
        if (cx.val.decimal && cx.val.width > 18) reject("decimal", "payload " + cx.val.type);
        if (!cx.val.integer && !cx.val.decimal) reject("shape", "payload type '" + cx.val.type + "'");
    }
    const int scale = (!bare && cx.val.decimal) ? cx.val.scale : 0;

    // ---- WHERE → program ----
    WhereOut where;
    if (!is_null(node, "where_clause")) where = where_program(node["where_clause"], cx, tname, talias);

    // ---- select list ----
    struct Item { bool key; XAgg agg; std::string alias; std::size_t key_index; };
    std::vector<Item> items;
    auto key_index_of = [&](const std::string& c) -> int {
        for (std::size_t i = 0; i < cx.keys.size(); ++i) if (ieq(c, cx.keys[i].name)) return static_cast<int>(i);
        return -1;
    };
    const json& sel = field(node, "select_list");
    if (!sel.is_array() || sel.empty()) reject("shape", "empty select list");
    for (const json& e : sel) {
        const std::string cls = sfield(e, "class");
        if (cls == "STAR") reject("shape", "SELECT *");
        if (cls == "WINDOW") reject("shape", "window function");
        if (cls == "PARAMETER") reject("shape", "parameter");
        const std::string alias = sfield(e, "alias");
        if (cls == "COLUMN_REF") {
            const std::string c = column_ref(e, tname, talias);
            const int ki = c.empty() ? -1 : key_index_of(c);
            if (ki < 0) reject("shape", "column that is not a GROUP BY key");
            items.push_back({true, XAgg::None, alias, static_cast<std::size_t>(ki)});
            continue;
        }
        const XAgg a = xagg_of(e, cx, tname, talias);
        if (a == XAgg::None) reject("shape", "select-list expression of class " + cls);
        if (a == XAgg::Avg && cx.val.decimal) reject("decimal", "avg over a DECIMAL payload is not on the exact path");
        items.push_back({false, a, alias, 0});
    }
    if (cx.outputs.size() != items.size()) reject("error", "context.outputs does not match the select list");

    // ---- HAVING: one comparison; device form for sum/count/count_star/min/max
    //      with > >= < <=; anything else (=, <>, avg) is exact on the host
    //      over the small result ----
    int cmp = 0;
    XAgg having_agg = XAgg::None;
    std::int64_t threshold = 0;
    bool having_dev = false;
    json host_pred;
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
        else if (ct == "COMPARE_EQUAL")                cmp = 5;
        else if (ct == "COMPARE_NOTEQUAL")             cmp = 6;
        else reject("shape", "HAVING comparison " + ct);
        having_agg = xagg_of(*aggside, cx, tname, talias);
        if (having_agg == XAgg::None) reject("shape", "HAVING is not on an aggregate");
        if (having_agg == XAgg::Avg && cx.val.decimal) reject("decimal", "HAVING avg over a DECIMAL payload");
        if (sfield(*cside, "class") != "CONSTANT") reject("shape", "HAVING threshold is not a constant");
        if (cmp <= 4 && having_agg != XAgg::Avg) {
            const int tscale = (having_agg == XAgg::Sum || having_agg == XAgg::Min || having_agg == XAgg::Max) ? scale : 0;
            threshold = threshold_of(*cside, tscale, cmp).value;
            having_dev = true;
        } else {
            // native-typed aggregate expression <op> the user's constant, in the outer WHERE
            static const char* types[] = {"", "COMPARE_GREATERTHAN", "COMPARE_GREATERTHANOREQUALTO",
                                          "COMPARE_LESSTHAN", "COMPARE_LESSTHANOREQUALTO",
                                          "COMPARE_EQUAL", "COMPARE_NOTEQUAL"};
            json lhs = j_output_exact(xagg_col(having_agg), xagg_native_type(having_agg, cx), "");
            host_pred = json{{"class", "COMPARISON"}, {"type", types[cmp]}, {"alias", ""},
                             {"query_location", kNoLocation}, {"left", lhs}, {"right", *cside}};
        }
    }

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
    auto output_name_for_key = [&](std::size_t ki) -> std::string {
        for (std::size_t i = 0; i < items.size(); ++i) if (items[i].key && items[i].key_index == ki) return cx.outputs[i].name;
        return "";
    };
    auto output_name_for_agg = [&](XAgg a) -> std::string {
        for (std::size_t i = 0; i < items.size(); ++i) if (!items[i].key && items[i].agg == a) return cx.outputs[i].name;
        return "";
    };
    XAgg order_agg = XAgg::None;   // the aggregate a single ORDER BY names (for the top-k push)
    auto remap = [&](const json& e) -> json {
        const std::string cls = sfield(e, "class");
        if (cls == "COLUMN_REF") {
            const json& names = field(e, "column_names");
            if (names.size() == 1) {
                const std::string n = names[0].get<std::string>();
                for (std::size_t i = 0; i < items.size(); ++i)
                    if (ieq(n, cx.outputs[i].name) || (!items[i].alias.empty() && ieq(n, items[i].alias))) {
                        if (!items[i].key) order_agg = items[i].agg;
                        return j_colref(cx.outputs[i].name);
                    }
            }
            const std::string c = column_ref(e, tname, talias);
            const int ki = c.empty() ? -1 : key_index_of(c);
            if (ki >= 0) {
                const std::string on = output_name_for_key(static_cast<std::size_t>(ki));
                if (on.empty()) reject("shape", "ORDER BY a key that is not selected");
                return j_colref(on);
            }
            reject("shape", "ORDER BY an expression the rewrite cannot map");
        }
        if (cls == "FUNCTION") {
            const XAgg a = xagg_of(e, cx, tname, talias);
            const std::string on = output_name_for_agg(a);
            if (on.empty()) reject("shape", "ORDER BY an aggregate that is not selected");
            order_agg = a;
            return j_colref(on);
        }
        if (cls == "CONSTANT") {
            const json& v = field(e, "value");
            if (!v.contains("value") || !v["value"].is_number_integer()) reject("shape", "non-integer ordinal ORDER BY");
            const std::int64_t i = v["value"].get<std::int64_t>();
            if (i < 1 || static_cast<std::size_t>(i) > items.size()) reject("shape", "ordinal ORDER BY out of range");
            if (!items[static_cast<std::size_t>(i - 1)].key) order_agg = items[static_cast<std::size_t>(i - 1)].agg;
            return j_colref(cx.outputs[static_cast<std::size_t>(i - 1)].name);
        }
        reject("shape", "ORDER BY expression of class " + cls);
    };

    std::int64_t topk = 0;
    bool topk_desc = false;
    json new_mods = json::array();
    if (order_mod) {
        json om = *order_mod;
        json& orders = om["orders"];
        if (!orders.is_array() || orders.empty()) reject("shape", "empty ORDER BY");
        for (json& o : orders) { order_agg = XAgg::None; o["expression"] = remap(o["expression"]); }
        if (orders.size() == 1 && limit_mod && having_agg == XAgg::None &&
            order_agg != XAgg::None && order_agg != XAgg::Avg) {
            const json& o = orders[0];
            const json& lim = field(*limit_mod, "limit");
            const bool off_null = is_null(*limit_mod, "offset");
            std::string dir = sfield(o, "type");
            if (dir == "ORDER_DEFAULT") dir = ieq(cx.default_order, "DESC") ? "DESCENDING" :
                                              ieq(cx.default_order, "ASC")  ? "ASCENDING" : "";
            if (off_null && !dir.empty() && sfield(lim, "class") == "CONSTANT") {
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

    // ---- residency / thresholds (exactness needs no stats gate) ----
    if (ieq(cx.backend, "CPU") || cx.backend.empty()) reject("backend", "backend " + cx.backend);
    if (cx.rows < cx.min_rows) reject("threshold", std::to_string(cx.rows) + " rows below the floor");
    if (!cx.ready) reject("not_resident", "set is not ready");

    // ---- build the replacement ----
    std::string fn = "gpu_groupby_exact_resident";
    json args = json::array({j_const_varchar(cx.tag)});
    if (!where.program.empty()) { fn += "_where"; args.push_back(j_const_varchar(where.program)); }
    if (having_dev) {
        fn += "_having";
        static const char* ops[] = {"", ">", ">=", "<", "<="};
        args.push_back(j_const_varchar(xagg_col(having_agg)));
        args.push_back(j_const_varchar(ops[cmp]));
        args.push_back(j_const_bigint(threshold));
        r.form = "having";
    } else if (having_agg != XAgg::None) {
        r.form = "having";
    } else if (topk > 0) {
        fn += "_topk";
        args.push_back(j_const_varchar(xagg_col(order_agg)));
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
        if (items[i].key) {
            if (cx.packed()) new_select.push_back(j_key_component(cx, items[i].key_index, o.type, o.name));
            else             new_select.push_back(j_output_exact("key", o.type, o.name));
        } else {
            new_select.push_back(j_output_exact(xagg_col(items[i].agg), o.type, o.name));
        }
    }

    node["select_list"] = new_select;
    node["from_table"] = new_from;
    if (!host_pred.is_null()) {
        node["where_clause"] = json{{"class", "CONJUNCTION"}, {"type", "CONJUNCTION_AND"}, {"alias", ""},
                                    {"query_location", kNoLocation},
                                    {"children", json::array({j_colref2("gd", "ok"), host_pred})}};
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
    if (!gexp.is_array() || gexp.empty() || gexp.size() > 3) reject("shape", "GROUP BY must have one to three keys");
    if (!gsets.is_array() || gsets.size() != 1 || !gsets[0].is_array() || gsets[0].size() != gexp.size())
        reject("shape", "ROLLUP / CUBE / GROUPING SETS");
    for (std::size_t i = 0; i < gexp.size(); ++i) {
        if (!gsets[0][i].is_number_integer() || gsets[0][i].get<std::int64_t>() != static_cast<std::int64_t>(i))
            reject("shape", "GROUPING SETS");
        if (sfield(gexp[i], "class") == "CONSTANT") reject("shape", "ordinal GROUP BY");
    }
    if (gexp.size() != cx.keys.size()) reject("shape", "GROUP BY key count is not the resident set's");
    for (std::size_t i = 0; i < gexp.size(); ++i) {
        const std::string gcol = column_ref(gexp[i], tname, talias);
        if (gcol.empty() || !ieq(gcol, cx.keys[i].name)) reject("shape", "GROUP BY key is not the resident key");
    }
    if (cx.packed() && !cx.exact) reject("shape", "packed keys need an exact set");

    if (cx.exact) return do_rewrite_exact(std::move(tree), std::move(node), cx, tname, talias);
    if (!is_null(node, "where_clause")) reject("shape", "WHERE needs an exact set (gpu_upload_rows_exact)");

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
