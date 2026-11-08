// gpu_functions.cpp — gpu_functions(): what every gpudb SQL function does,
// readable from SQL.
//
//   SELECT function_name, description FROM gpu_functions();
//
// Why a table function of our own: the loadable extension is built against
// DuckDB's stable C API (C_STRUCT ABI), and that API has no way to attach a
// description, an example or parameter names to a function — a function gets
// a name and parameter TYPES, nothing else, so duckdb_functions() lists every
// gpudb overload with parameters col0, col1, ... and an empty description.
// This table carries what duckdb_functions() cannot: real parameter names, a
// one-sentence description, one runnable example and a category or two, one
// row per overload, in columns named and typed like duckdb_functions()'s so
// the two join on (function_name, function_type, parameter_types).
//
// This file is the single source of truth for that documentation. Two checks
// keep it honest:
//   - test/sql/gpu_functions.test compares (function_name, function_type,
//     parameter_types, return_type) here with duckdb_functions() WHERE
//     function_name LIKE 'gpu\_%', both directions: a function registered
//     without an entry here, or an entry for a function that is not
//     registered, fails the suite;
//   - scripts/gpu_functions_check.py runs every `example` in a fresh database
//     after LOAD and fails if any of them errors.
//
// An example is ONE string that DuckDB can run as given (duckdb_query, the
// CLI's -c, Python's execute all take several ';'-separated statements and
// return the last result). Functions that read a resident set need the
// upload as an EARLIER statement — a table function binds before an
// aggregate in the same statement runs — so their example starts with it.

#include "gpu_functions.hpp"

#if defined(GPUDB_C_STRUCT_ABI)
DUCKDB_EXTENSION_EXTERN
#endif

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpudb_ext {
namespace {

// Setup statements shared by several examples. Each ends in "; ".
const std::string kUploadC  = "SELECT gpu_upload('c', x) FROM range(1000) t(x); ";
const std::string kUploadCf = "SELECT gpu_upload('cf', x::DOUBLE / 4) FROM range(1000) t(x); ";
const std::string kPair     = "SELECT gpu_upload_pair('p', x % 10, x) FROM range(1000) t(x); ";
const std::string kPairF    = "SELECT gpu_upload_pair('pf', x % 10, x::DOUBLE / 4) FROM range(1000) t(x); ";
const std::string kExact    = "SELECT gpu_upload_pair_exact('e', CASE WHEN x % 7 = 0 THEN NULL ELSE x % 10 END, x) "
                              "FROM range(1000) t(x); ";
const std::string kRows     = "SELECT gpu_upload_rows_exact('w', x % 10, x, [x % 3, x % 7], [x::DOUBLE / 1000]) "
                              "FROM range(1000) t(x); ";
const std::string kJoin     = "SELECT gpu_upload_pair('l', x % 100, x) FROM range(1000) t(x); "
                              "SELECT gpu_upload('o', x) FROM range(50) t(x); ";
const std::string kJoinF    = "SELECT gpu_upload_pair('lf', x % 100, x::DOUBLE / 4) FROM range(1000) t(x); "
                              "SELECT gpu_upload('o', x) FROM range(50) t(x); ";
const std::string kStore    = "CREATE TABLE t AS SELECT (x % 10)::BIGINT AS k, x AS v FROM range(1000) r(x); "
                              "SET VARIABLE st = 'gpudb:v1:' || current_database() || ':main:t:' || "
                              "(SELECT table_oid FROM duckdb_tables() WHERE database_name = current_database() "
                              "AND table_name = 't') || ':k,v:store'; "
                              "SELECT gpu_upload_columns(getvariable('st'), rowid, [k, v], []::DOUBLE[], []::VARCHAR[]) "
                              "FROM t; ";
const std::string kDict     = "SELECT gpu_upload_rows_exact('ws', s, v, []::BIGINT[], []::DOUBLE[], []::VARCHAR[]) "
                              "FROM (VALUES ('3:abc', 1), ('2:xy', 2), ('N', 3)) t(s, v); ";

// The WHERE-program grammar, shared by the descriptions that take one.
const std::string kProgram  = "a WHERE program of ';'-separated terms over the set's lanes k, v, i<n>, f<n> and "
                              "s<n> ('<lane> <op> <constant>', '<lane> in (...)', '<lane> is [not] null'; "
                              "'' for none)";

std::vector<FunctionDoc> build_docs() {
    const std::string S = "scalar", A = "aggregate", T = "table";
    const std::string BI = "BIGINT", DB = "DOUBLE", VC = "VARCHAR", BO = "BOOLEAN", HI = "HUGEINT";
    std::vector<FunctionDoc> d;
    auto add = [&](std::string name, std::string type, std::vector<std::string> params,
                   std::vector<std::string> types, std::string ret, std::string desc,
                   std::string example, std::vector<std::string> cats) {
        d.push_back(FunctionDoc{std::move(name), std::move(type), std::move(params), std::move(types),
                                std::move(ret), std::move(desc), std::move(example), std::move(cats)});
    };

    // ---- streaming aggregates (gpu_sum_extension.cpp) ----------------------
    add("gpu_sum", A, {"value"}, {BI}, BI,
        "Sum of a BIGINT column as a streaming aggregate with a running accumulator on the CPU, usable in any "
        "query shape (GROUP BY, windows, FILTER); NULL when no row is non-NULL, and it wraps on int64 overflow "
        "where native sum() returns HUGEINT.",
        "SELECT gpu_sum(x) FROM range(1000) t(x)", {"aggregate", "streaming"});
    add("gpu_sum", A, {"value"}, {DB}, DB,
        "Sum of a DOUBLE column as a streaming aggregate with a running accumulator on the CPU, usable in any "
        "query shape; NULL when no row is non-NULL.",
        "SELECT gpu_sum(x::DOUBLE) FROM range(1000) t(x)", {"aggregate", "streaming"});
    add("gpu_min", A, {"value"}, {BI}, BI,
        "Minimum of a BIGINT column as a streaming aggregate on the CPU; NULL when no row is non-NULL.",
        "SELECT gpu_min(x) FROM range(1000) t(x)", {"aggregate", "streaming"});
    add("gpu_min", A, {"value"}, {DB}, DB,
        "Minimum of a DOUBLE column as a streaming aggregate on the CPU; NULL when no row is non-NULL.",
        "SELECT gpu_min(x::DOUBLE) FROM range(1000) t(x)", {"aggregate", "streaming"});
    add("gpu_max", A, {"value"}, {BI}, BI,
        "Maximum of a BIGINT column as a streaming aggregate on the CPU; NULL when no row is non-NULL.",
        "SELECT gpu_max(x) FROM range(1000) t(x)", {"aggregate", "streaming"});
    add("gpu_max", A, {"value"}, {DB}, DB,
        "Maximum of a DOUBLE column as a streaming aggregate on the CPU; NULL when no row is non-NULL.",
        "SELECT gpu_max(x::DOUBLE) FROM range(1000) t(x)", {"aggregate", "streaming"});

    // ---- uploads (gpu_resident.cpp) -----------------------------------------
    add("gpu_upload", A, {"name", "v"}, {VC, BI}, BI,
        "Uploads a BIGINT column, NULLs skipped, as the resident column `name` (replacing any set of that name) "
        "and returns the number of rows uploaded; the gpu_*_resident functions then read it without another "
        "transfer.",
        "SELECT gpu_upload('c', x) FROM range(1000) t(x)", {"resident", "upload"});
    add("gpu_upload", A, {"name", "v"}, {VC, DB}, BI,
        "Uploads a DOUBLE column, NULLs skipped, as the resident column `name` (replacing any set of that name) "
        "and returns the number of rows uploaded.",
        "SELECT gpu_upload('cf', x::DOUBLE / 4) FROM range(1000) t(x)", {"resident", "upload"});
    add("gpu_upload_pair", A, {"name", "k", "v"}, {VC, BI, BI}, BI,
        "Uploads a BIGINT key and a BIGINT payload in one scan as the resident pair `name` (its columns are "
        "'name.k' and 'name.v'), skipping rows where either is NULL; returns the number of rows uploaded.",
        "SELECT gpu_upload_pair('p', x % 10, x) FROM range(1000) t(x)", {"resident", "upload"});
    add("gpu_upload_pair", A, {"name", "k", "v"}, {VC, BI, DB}, BI,
        "Uploads a BIGINT key and a DOUBLE payload in one scan as the resident pair `name` (columns 'name.k' "
        "and 'name.v'), skipping rows where either is NULL; returns the number of rows uploaded.",
        "SELECT gpu_upload_pair('pf', x % 10, x::DOUBLE / 4) FROM range(1000) t(x)", {"resident", "upload"});
    add("gpu_upload_pair_exact", A, {"name", "k", "v"}, {VC, BI, BI}, BI,
        "Uploads a BIGINT key and payload as the exact set `name`, keeping NULLs in both, for the "
        "gpu_groupby_exact_resident functions; returns the number of rows uploaded.",
        "SELECT gpu_upload_pair_exact('e', CASE WHEN x % 7 = 0 THEN NULL ELSE x % 10 END, x) FROM range(1000) t(x)",
        {"resident", "upload", "exact"});
    add("gpu_upload_rows_exact", A, {"name", "k", "v", "pi", "pf"}, {VC, BI, BI, "BIGINT[]", "DOUBLE[]"}, BI,
        "Uploads an exact set `name`: a BIGINT key, a BIGINT payload and row-aligned predicate columns, the "
        "list pi becoming lanes i0, i1, ... and pf lanes f0, f1, ... (list lengths fixed by the first row), "
        "NULLs kept; returns the number of rows uploaded.",
        "SELECT gpu_upload_rows_exact('w', x % 10, x, [x % 3, x % 7], [x::DOUBLE / 1000]) FROM range(1000) t(x)",
        {"resident", "upload", "exact"});
    add("gpu_upload_rows_exact", A, {"name", "k", "v", "pi", "pf", "ps"},
        {VC, BI, BI, "BIGINT[]", "DOUBLE[]", "VARCHAR[]"}, BI,
        "As the five-argument form, plus VARCHAR predicate columns in ps that become lanes s0, s1, ... "
        "(stored as 64-bit hashes of the text, each with its own dictionary); returns the number of rows "
        "uploaded.",
        "SELECT gpu_upload_rows_exact('w', x % 10, x, [x % 3], [x::DOUBLE / 1000], "
        "[CASE WHEN x % 2 = 0 THEN 'a' ELSE 'b' END]) FROM range(1000) t(x)",
        {"resident", "upload", "exact"});
    add("gpu_upload_rows_exact", A, {"name", "k", "v", "pi", "pf", "ps"},
        {VC, VC, BI, "BIGINT[]", "DOUBLE[]", "VARCHAR[]"}, BI,
        "As the six-argument form with a VARCHAR key: k is the key tuple as text, each component written "
        "'<byte length>:<text>' and a NULL one as 'N', stored hashed and decoded by gpu_resident_dictionary; "
        "returns the number of rows uploaded.",
        "SELECT gpu_upload_rows_exact('ws', s, v, []::BIGINT[], []::DOUBLE[], []::VARCHAR[]) "
        "FROM (VALUES ('3:abc', 1), ('2:xy', 2), ('N', 3)) t(s, v)",
        {"resident", "upload", "exact"});
    add("gpu_upload_columns", A, {"tag", "rowid", "ci", "cf", "cs"},
        {VC, BI, "BIGINT[]", "DOUBLE[]", "VARCHAR[]"}, BI,
        "Uploads columns of one table into its per-table store, named by an identity tag "
        "'gpudb:v1:<catalog>:<schema>:<table>:<table_oid>:<columns>:store', placed in rowid order; ci, cf and "
        "cs carry the BIGINT, DOUBLE and VARCHAR columns; returns the number of rows uploaded.",
        kStore + "SELECT \"column\", dtype, rows FROM gpu_store_columns()", {"resident", "upload", "store"});

    // ---- reductions over a resident column ----------------------------------
    add("gpu_sum_resident", S, {"name"}, {VC}, BI,
        "Sum of a resident BIGINT column (a gpu_upload name or 'pair.v') with no transfer at query time; "
        "wraps on int64 overflow.",
        kUploadC + "SELECT gpu_sum_resident('c')", {"resident", "aggregate"});
    add("gpu_min_resident", S, {"name"}, {VC}, BI,
        "Minimum of a resident BIGINT column with no transfer at query time.",
        kUploadC + "SELECT gpu_min_resident('c')", {"resident", "aggregate"});
    add("gpu_max_resident", S, {"name"}, {VC}, BI,
        "Maximum of a resident BIGINT column with no transfer at query time.",
        kUploadC + "SELECT gpu_max_resident('c')", {"resident", "aggregate"});
    add("gpu_sum_resident_f64", S, {"name"}, {VC}, DB,
        "Sum of a resident DOUBLE column with no transfer at query time.",
        kUploadCf + "SELECT gpu_sum_resident_f64('cf')", {"resident", "aggregate"});

    // ---- registry, lifecycle, diagnostics -----------------------------------
    add("gpu_resident_info", S, {"name"}, {VC}, VC,
        "One line describing a resident column or set: dtype, rows, device, set, origin, state and whether it "
        "is prepared.",
        kUploadC + "SELECT gpu_resident_info('c')", {"resident", "diagnostics"});
    add("gpu_drop_resident", S, {"name"}, {VC}, BO,
        "Drops the resident set `name` and frees its memory; false when there is no such set.",
        kUploadC + "SELECT gpu_drop_resident('c')", {"resident"});
    add("gpu_prepare_resident", S, {"name"}, {VC}, BO,
        "Builds a resident set's derived structures (the sort cache a GROUP BY uses) now rather than on first "
        "use; returns whether the set is prepared.",
        kPair + "SELECT gpu_prepare_resident('p')", {"resident"});
    add("gpu_invalidate", S, {"pattern"}, {VC}, BI,
        "Marks stale every resident set whose name equals `pattern` or starts with `pattern` followed by ':' "
        "(a table prefix of an identity tag), so that an upload in flight under a matching name is discarded "
        "when it finishes; returns the number of sets marked.",
        kUploadC + "SELECT gpu_invalidate('c')", {"resident"});
    add("gpu_assert_rows", S, {"name", "n"}, {VC, BI}, BO,
        "Staleness guard: true when n equals the row count the upload of set `name` saw, otherwise an error "
        "whose message starts with 'GPUDB_STALE:'.",
        "CREATE TABLE t AS SELECT x FROM range(1000) r(x); SELECT gpu_upload('t_x', x) FROM t; "
        "SELECT gpu_assert_rows('t_x', (SELECT count(*) FROM t))",
        {"resident", "guard"});
    add("gpu_note_rows", S, {"name", "n"}, {VC, BI}, BO,
        "Registers a set with no columns that only remembers a base table's row count n, so gpu_assert_rows "
        "can guard a set built from a join over that table; holds no device memory.",
        "SELECT gpu_note_rows('t_rows', count(*)) FROM range(1000)", {"resident", "guard"});
    add("gpu_last_stats", S, {}, {}, VC,
        "Dispatch decision and timing of the last resident operator in this database: op, backend, reason, "
        "row counts, wall_ms, kernel_ms and transfer_ms.",
        kUploadC + "SELECT gpu_sum_resident('c'); SELECT gpu_last_stats()", {"diagnostics"});
    add("gpu_build_info", S, {}, {}, VC,
        "Which backends this binary was compiled with and which one runs, with capability flags such as exact, "
        "join, global and store, and the device memory.",
        "SELECT gpu_build_info()", {"diagnostics"});
    add("gpu_residents", T, {}, {}, "",
        "Every resident set in this database, one row each: name, origin, identity fields, kind, dtype, rows, "
        "bytes, state, prepared, hits, device, upload and last-use times, and last_stats.",
        kUploadC + "SELECT name, kind, rows, state FROM gpu_residents()", {"resident", "diagnostics"});

    // ---- upload sessions ------------------------------------------------------
    add("gpu_upload_begin", S, {"name"}, {VC}, BO,
        "Opens an upload session for `name`: until gpu_upload_finish(name), each upload statement for that "
        "name appends a segment and returns its own row count instead of publishing the set.",
        "SELECT gpu_upload_begin('s'); SELECT gpu_upload('s', x) FROM range(500) t(x); "
        "SELECT gpu_upload('s', x) FROM range(500, 1000) t(x); SELECT gpu_upload_finish('s')",
        {"resident", "upload", "session"});
    add("gpu_upload_finish", S, {"name"}, {VC}, BI,
        "Closes the upload session for `name`: copies its segments to the device, prepares and publishes the "
        "set, and returns its row count.",
        "SELECT gpu_upload_begin('s'); SELECT gpu_upload('s', x) FROM range(1000) t(x); "
        "SELECT gpu_upload_finish('s')",
        {"resident", "upload", "session"});
    add("gpu_upload_abort", S, {"name"}, {VC}, BO,
        "Drops the open upload session for `name` and its buffers; false when no session is open.",
        "SELECT gpu_upload_begin('s'); SELECT gpu_upload('s', x) FROM range(1000) t(x); "
        "SELECT gpu_upload_abort('s')",
        {"resident", "upload", "session"});
    add("gpu_upload_status", S, {"name"}, {VC}, VC,
        "The upload session for `name` as JSON: open, segments, rows, rows_seen, bytes, kind, epoch, "
        "invalidated and started_at_us.",
        "SELECT gpu_upload_begin('s'); SELECT gpu_upload('s', x) FROM range(1000) t(x); "
        "SELECT gpu_upload_status('s')",
        {"resident", "upload", "session"});

    // ---- per-table column store ---------------------------------------------
    add("gpu_store_columns", T, {}, {}, "",
        "Every column held in a per-table store, one row each: store, catalog, schema, table, table_oid, "
        "column, dtype, rows, bytes, prepared, epoch, upload and last-use times, width and on_gpu.",
        kStore + "SELECT \"column\", dtype, rows, bytes FROM gpu_store_columns()", {"resident", "store", "diagnostics"});
    add("gpu_drop_column", S, {"store", "lane"}, {VC, VC}, BO,
        "Frees one column of a per-table store (store and column as gpu_store_columns() reports them) and "
        "drops every view that reads it; false when there is no such column.",
        kStore + "SELECT gpu_drop_column(store, \"column\") FROM gpu_store_columns() WHERE \"column\" = 'v'",
        {"resident", "store"});

    // ---- VARCHAR-keyed exact sets -----------------------------------------------
    add("gpu_resident_dictionary", T, {"name", "n"}, {VC, BI}, "",
        "The key tuples of a VARCHAR-keyed exact set, one row per distinct key: id (the resident key value) "
        "and its n components c0 .. c<n-1> (n from 1 to 8; a NULL component is NULL).",
        kDict + "SELECT * FROM gpu_resident_dictionary('ws', 1)", {"resident", "exact", "dictionary"});
    add("gpu_resident_dict_component", S, {"name", "key", "i"}, {VC, BI, BI}, VC,
        "Component i (from 0) of one key of a VARCHAR-keyed exact set; NULL for a NULL component, a NULL key "
        "or a key the set does not hold.",
        kDict + "SELECT id, gpu_resident_dict_component('ws', id, 0) FROM gpu_resident_dictionary('ws', 1)",
        {"resident", "exact", "dictionary"});
    add("gpu_avg_decimal", S, {"sum", "count", "scale"}, {HI, BI, BI}, DB,
        "avg() over a DECIMAL column finalised as native DuckDB does it: the unscaled 128-bit sum divided, in "
        "long double, by count * 10^scale; NULL when count is 0.",
        "SELECT gpu_avg_decimal(12345::HUGEINT, 10, 2)", {"exact", "aggregate"});

    // ---- fused resident joins ---------------------------------------------------
    // The native statement each one answers, as scripts/join_parity_check.sh
    // checks it: p = (probe_keys, payload), b = build_keys.
    struct JoinKind { const char* prefix; const char* from; };
    const JoinKind kinds[] = {
        {"gpu_join",      "FROM probe p JOIN build b ON p.k = b.k"},
        {"gpu_left_join", "FROM probe p LEFT JOIN build b ON p.k = b.k"},
        {"gpu_semi_join", "FROM probe p WHERE EXISTS (SELECT 1 FROM build b WHERE b.k = p.k)"},
        {"gpu_anti_join", "FROM probe p WHERE NOT EXISTS (SELECT 1 FROM build b WHERE b.k = p.k)"},
    };
    for (const JoinKind& k : kinds) {
        const std::string p = k.prefix, from = k.from;
        add(p + "_sum_resident", S, {"probe_keys", "payload", "build_keys"}, {VC, VC, VC}, BI,
            "The answer of SELECT sum(p.v) " + from + ", where probe_keys, payload and build_keys name resident "
            "BIGINT columns (p.k, p.v, b.k), computed in one pass without materialising the join.",
            kJoin + "SELECT " + p + "_sum_resident('l.k', 'l.v', 'o')", {"resident", "join"});
        add(p + "_count_resident", S, {"probe_keys", "build_keys"}, {VC, VC}, BI,
            "The answer of SELECT count(*) " + from + ", where probe_keys and build_keys name resident BIGINT "
            "columns (p.k, b.k), computed without materialising the join.",
            kJoin + "SELECT " + p + "_count_resident('l.k', 'o')", {"resident", "join"});
        add(p + "_sum_resident_f64", S, {"probe_keys", "payload", "build_keys"}, {VC, VC, VC}, DB,
            "The answer of SELECT sum(p.v) " + from + " with a resident DOUBLE payload (p.v) and resident BIGINT "
            "keys (p.k, b.k), computed in one pass without materialising the join.",
            kJoinF + "SELECT " + p + "_sum_resident_f64('lf.k', 'lf.v', 'o')", {"resident", "join"});
    }
    add("gpu_inner_join", T, {"build_keys", "probe_keys"}, {"BIGINT[]", "BIGINT[]"}, "",
        "Inner equi-join of two BIGINT lists, returning (probe_idx, build_idx) as 0-based positions; build keys "
        "are expected unique, and for a duplicated one the first match wins.",
        "SELECT * FROM gpu_inner_join([10, 20, 30], [20, 40, 99])", {"join"});
    add("gpu_join_rows_resident", T, {"probe", "build", "kind"}, {VC, VC, VC}, "",
        "The rows of a join of two resident BIGINT key columns as (probe_idx, build_idx), kind 'inner', "
        "'left', 'semi' or 'anti'; build_idx is NULL for left-unmatched, semi and anti rows.",
        "SELECT gpu_upload('pk', x % 20) FROM range(100) t(x); SELECT gpu_upload('bk', x) FROM range(10) t(x); "
        "SELECT * FROM gpu_join_rows_resident('pk', 'bk', 'inner')",
        {"resident", "join"});
    add("gpu_join_materialize", S, {"out", "probe", "probe_lane", "build", "build_lane", "lanes"},
        {VC, VC, VC, VC, VC, VC}, BI,
        "Inner equi-joins two exact sets on probe_lane = build_lane (unique on the build side) and publishes "
        "the result as the exact set `out` with the columns listed in lanes ('p.<lane>' or 'b.<lane>': key, "
        "payload, then predicate lanes); returns the joined row count.",
        "SELECT gpu_upload_rows_exact('f', x, x % 100, [x % 10], []) FROM range(1000) t(x); "
        "SELECT gpu_upload_rows_exact('d', x, x * 2, [], []) FROM range(10) t(x); "
        "SELECT gpu_join_materialize('j', 'f', 'i0', 'd', 'k', 'p.k, p.v, b.v')",
        {"resident", "join", "exact"});

    // ---- resident GROUP BY (gpu_groupby_extension.cpp) ---------------------------
    struct Gb { const char* fn; const char* agg; const char* setup_name; const std::string* setup;
                const char* what; const char* cols; std::string thr_type; const char* thr_example; };
    const Gb gbs[] = {
        {"gpu_groupby_sum_resident",     "sum",   "p",  &kPair,  "BIGINT payload",
         "(key, sum, count)", BI, "49500"},
        {"gpu_groupby_sum_resident_f64", "sum",   "pf", &kPairF, "DOUBLE payload",
         "(key, sum DOUBLE, count)", DB, "12400.5"},
        {"gpu_groupby_count_resident",   "count", "p",  &kPair,  "pair or a bare resident column",
         "(key, count)", BI, "50"},
    };
    for (const Gb& g : gbs) {
        const std::string fn = g.fn, agg = g.agg, nm = g.setup_name;
        add(fn, T, {"name"}, {VC}, "",
            "GROUP BY key over a resident " + std::string(g.what) + ": " + g.cols + " per group, ascending key; "
            "rows with a NULL key or payload were skipped at upload.",
            *g.setup + "SELECT * FROM " + fn + "('" + nm + "')", {"resident", "groupby"});
        add(fn + "_having", T, {"name", "cmp", "threshold"}, {VC, VC, g.thr_type}, "",
            "As " + fn + ", keeping only groups where " + agg + " <cmp> threshold (cmp one of '>', '>=', '<', "
            "'<='), filtered on the device so only those rows come back.",
            *g.setup + "SELECT * FROM " + fn + "_having('" + nm + "', '>', " + g.thr_example + ")",
            {"resident", "groupby"});
        add(fn + "_topk", T, {"name", "k", "order"}, {VC, BI, VC}, "",
            "As " + fn + ", returning only the k groups with the largest ('desc') or smallest ('asc') " + agg +
            ", ordered by it, chosen on the device; ties in an unspecified order.",
            *g.setup + "SELECT * FROM " + fn + "_topk('" + nm + "', 3, 'desc')", {"resident", "groupby", "topk"});
    }
    add("gpu_groupby_exact_resident", T, {"name"}, {VC}, "",
        "Exact GROUP BY key over an exact set (gpu_upload_pair_exact, gpu_upload_rows_exact, "
        "gpu_join_materialize): key, sum HUGEINT, count, count_star, min, max, avg per group with native "
        "NULL semantics, the NULL-key group last.",
        kExact + "SELECT * FROM gpu_groupby_exact_resident('e')", {"resident", "groupby", "exact"});
    add("gpu_groupby_exact_resident_having", T, {"name", "agg", "cmp", "threshold"}, {VC, VC, VC, BI}, "",
        "As gpu_groupby_exact_resident, keeping only groups where agg <cmp> threshold (agg one of 'sum', "
        "'count', 'count_star', 'min', 'max'; cmp one of '>', '>=', '<', '<='), filtered on the device.",
        kExact + "SELECT * FROM gpu_groupby_exact_resident_having('e', 'sum', '>', 45000)",
        {"resident", "groupby", "exact"});
    add("gpu_groupby_exact_resident_topk", T, {"name", "agg", "k", "order"}, {VC, VC, BI, VC}, "",
        "As gpu_groupby_exact_resident, returning only the k groups with the largest ('desc') or smallest "
        "('asc') agg ('sum', 'count', 'count_star', 'min' or 'max'), chosen on the device.",
        kExact + "SELECT * FROM gpu_groupby_exact_resident_topk('e', 'count_star', 3, 'desc')",
        {"resident", "groupby", "exact", "topk"});
    add("gpu_groupby_exact_resident_where", T, {"name", "program"}, {VC, VC}, "",
        "As gpu_groupby_exact_resident over only the rows that pass " + kProgram + ".",
        kRows + "SELECT * FROM gpu_groupby_exact_resident_where('w', 'i0 = 1; f0 < 0.5')",
        {"resident", "groupby", "exact", "where"});
    add("gpu_groupby_exact_resident_where_having", T, {"name", "program", "agg", "cmp", "threshold"},
        {VC, VC, VC, VC, BI}, "",
        "As gpu_groupby_exact_resident_having over only the rows that pass the WHERE program "
        "(as in gpu_groupby_exact_resident_where).",
        kRows + "SELECT * FROM gpu_groupby_exact_resident_where_having('w', 'i0 = 1', 'sum', '>', 15000)",
        {"resident", "groupby", "exact", "where"});
    add("gpu_groupby_exact_resident_where_topk", T, {"name", "program", "agg", "k", "order"},
        {VC, VC, VC, BI, VC}, "",
        "As gpu_groupby_exact_resident_topk over only the rows that pass the WHERE program "
        "(as in gpu_groupby_exact_resident_where).",
        kRows + "SELECT * FROM gpu_groupby_exact_resident_where_topk('w', 'i0 = 1', 'sum', 3, 'desc')",
        {"resident", "groupby", "exact", "where", "topk"});
    add("gpu_groupby_exact_multi", T, {"name", "program", "payloads", "filter"}, {VC, VC, VC, VC}, "",
        "Exact GROUP BY key over 1 to 8 payload lanes of one exact set ('v, i0, ...') under an optional WHERE "
        "program, with filter '' or 'having <p> <agg> <cmp> <threshold>' or 'topk <p> <agg> <k> <asc|desc>'; "
        "columns key, count_star, then sum<p>, count<p>, min<p>, max<p>, avg<p> per payload.",
        kRows + "SELECT * FROM gpu_groupby_exact_multi('w', 'i0 = 1', 'v, i1', 'topk 0 sum 3 desc')",
        {"resident", "groupby", "exact"});
    add("gpu_agg_exact_global", T, {"name", "program", "payloads"}, {VC, VC, VC}, "",
        "Aggregates with no GROUP BY over an exact set under an optional WHERE program: exactly one row with "
        "count_star, then sum<p>, count<p>, min<p>, max<p>, avg<p> per payload lane ('' for count(*) only).",
        kRows + "SELECT * FROM gpu_agg_exact_global('w', 'f0 < 0.5', 'v, i0')", {"resident", "exact", "aggregate"});
    add("gpu_topk_resident", T, {"name", "k", "order"}, {VC, BI, VC}, "",
        "The k largest ('desc') or smallest ('asc') values of a resident BIGINT column as (idx, value), idx "
        "being the row's position in the column.",
        kUploadC + "SELECT * FROM gpu_topk_resident('c', 3, 'desc')", {"resident", "topk"});
    add("gpu_topk_resident_f64", T, {"name", "k", "order"}, {VC, BI, VC}, "",
        "The k largest ('desc') or smallest ('asc') values of a resident DOUBLE column as (idx, value), idx "
        "being the row's position in the column.",
        kUploadCf + "SELECT * FROM gpu_topk_resident_f64('cf', 3, 'asc')", {"resident", "topk"});

    // ---- the statement rewriter (gpu_rewrite.cpp) ---------------------------
    add("gpu_rewrite_ast", S, {"tree", "context"}, {VC, VC}, VC,
        "The pure statement rewriter behind the Python wrapper's transparent path: takes json_serialize_sql() "
        "output and a JSON context and returns the rewritten or unchanged tree with a top-level 'gpudb' key "
        "giving the decision and its reason.",
        "SELECT gpu_rewrite_ast(json_serialize_sql('SELECT 42'), "
        "'{\"tag\": \"gpudb:v1:memory:main:t:1:k,v\"}')::JSON->'gpudb' AS decision",
        {"rewrite"});

    // ---- this table ----------------------------------------------------------
    add("gpu_functions", T, {}, {}, "",
        "Every gpudb function overload with its parameter names and types, return type, a one-sentence "
        "description, a runnable example and categories; joins duckdb_functions() on (function_name, "
        "function_type, parameter_types).",
        "SELECT function_name, description FROM gpu_functions()", {"diagnostics"});
    return d;
}

// ---- the table function --------------------------------------------------------

struct FnInit { std::size_t offset = 0; };

void fn_bind(duckdb_bind_info info) {
    duckdb_logical_type vc   = duckdb_create_logical_type(DUCKDB_TYPE_VARCHAR);
    duckdb_logical_type list = duckdb_create_list_type(vc);
    duckdb_bind_add_result_column(info, "function_name",   vc);
    duckdb_bind_add_result_column(info, "function_type",   vc);
    duckdb_bind_add_result_column(info, "parameters",      list);
    duckdb_bind_add_result_column(info, "parameter_types", list);
    duckdb_bind_add_result_column(info, "return_type",     vc);
    duckdb_bind_add_result_column(info, "description",     vc);
    duckdb_bind_add_result_column(info, "example",         vc);
    duckdb_bind_add_result_column(info, "categories",      list);
    duckdb_destroy_logical_type(&list);
    duckdb_destroy_logical_type(&vc);
}

void fn_init(duckdb_init_info info) {
    duckdb_init_set_init_data(info, new FnInit(), [](void* p) { delete static_cast<FnInit*>(p); });
}

void set_str(duckdb_vector v, idx_t i, const std::string& s) {
    duckdb_vector_assign_string_element_len(v, i, s.data(), s.size());
}

void set_null(duckdb_vector v, idx_t i) {
    duckdb_vector_ensure_validity_writable(v);
    duckdb_validity_set_row_invalid(duckdb_vector_get_validity(v), i);
}

// Fill LIST(VARCHAR) column `col` for rows [first, first + n) of the docs.
void set_lists(duckdb_vector v, const std::vector<FunctionDoc>& docs, std::size_t first, idx_t n,
               const std::vector<std::string> FunctionDoc::*member) {
    idx_t total = 0;
    for (idx_t i = 0; i < n; ++i) total += (docs[first + i].*member).size();
    if (duckdb_list_vector_reserve(v, total) == DuckDBError ||
        duckdb_list_vector_set_size(v, total) == DuckDBError) {
        throw std::runtime_error("gpu_functions: list vector allocation failed");
    }
    duckdb_vector child = duckdb_list_vector_get_child(v);
    auto* entries = static_cast<duckdb_list_entry*>(duckdb_vector_get_data(v));
    idx_t at = 0;
    for (idx_t i = 0; i < n; ++i) {
        const auto& items = docs[first + i].*member;
        entries[i].offset = at;
        entries[i].length = items.size();
        for (const auto& s : items) set_str(child, at++, s);
    }
}

void fn_function(duckdb_function_info info, duckdb_data_chunk output) {
    auto* init = static_cast<FnInit*>(duckdb_function_get_init_data(info));
    const auto& docs = function_docs();
    if (!init || init->offset >= docs.size()) {
        duckdb_data_chunk_set_size(output, 0);
        return;
    }
    const idx_t n = static_cast<idx_t>(std::min<std::size_t>(docs.size() - init->offset, duckdb_vector_size()));
    const std::size_t first = init->offset;
    auto vec = [&](idx_t c) { return duckdb_data_chunk_get_vector(output, c); };
    try {
        for (idx_t i = 0; i < n; ++i) {
            const FunctionDoc& f = docs[first + i];
            set_str(vec(0), i, f.name);
            set_str(vec(1), i, f.type);
            if (f.return_type.empty()) set_null(vec(4), i); else set_str(vec(4), i, f.return_type);
            set_str(vec(5), i, f.description);
            set_str(vec(6), i, f.example);
        }
        set_lists(vec(2), docs, first, n, &FunctionDoc::parameters);
        set_lists(vec(3), docs, first, n, &FunctionDoc::parameter_types);
        set_lists(vec(7), docs, first, n, &FunctionDoc::categories);
    } catch (const std::exception& e) {
        duckdb_function_set_error(info, e.what());
        return;
    }
    duckdb_data_chunk_set_size(output, n);
    init->offset += static_cast<std::size_t>(n);
}

} // namespace

const std::vector<FunctionDoc>& function_docs() {
    static const std::vector<FunctionDoc> docs = build_docs();
    return docs;
}

void register_gpu_functions(duckdb_connection con) {
    duckdb_table_function tf = duckdb_create_table_function();
    duckdb_table_function_set_name(tf, "gpu_functions");
    duckdb_table_function_set_bind(tf, fn_bind);
    duckdb_table_function_set_init(tf, fn_init);
    duckdb_table_function_set_function(tf, fn_function);
    if (duckdb_register_table_function(con, tf) == DuckDBError) {
        duckdb_destroy_table_function(&tf);
        throw std::runtime_error("gpu_functions registration failed");
    }
    duckdb_destroy_table_function(&tf);
}

} // namespace gpudb_ext
