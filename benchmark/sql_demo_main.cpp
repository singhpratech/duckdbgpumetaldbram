// gpudb-sql — small CLI that opens a DuckDB connection, registers gpu_sum,
// then runs queries from --sql ... or stdin. Compares gpu_sum vs native sum.
//
// Examples:
//   gpudb-sql --sql "SELECT gpu_sum(range::BIGINT) FROM range(100000000);"
//   gpudb-sql --sql "SELECT gpu_sum(l_orderkey), sum(l_orderkey)
//                    FROM read_parquet('data/tpch_sf1/lineitem_orderkey.parquet');"
//   echo "SELECT gpu_sum(x) FROM tbl;" | gpudb-sql --db /tmp/x.duckdb

#include "gpu_sum_extension.hpp"
#include "duckdb.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>

namespace {

void usage(const char* a0) {
    std::fprintf(stderr,
        "usage: %s [--db FILE] [--sql QUERY] [--multi]\n"
        "  --db FILE    use FILE as the DuckDB database (default: in-memory)\n"
        "  --sql QUERY  run this single SQL statement (default: read stdin)\n"
        "  --multi      split input on ';' and run statements sequentially in\n"
        "               ONE connection (resident columns persist between\n"
        "  --multi-last same as --multi but print only the LAST statement's result\n"
        "               statements); prints each result + per-statement time\n", a0);
}

void die(const char* what) {
    std::fprintf(stderr, "error: %s\n", what);
    std::exit(1);
}

void print_result(duckdb_result& r) {
    const idx_t cols = duckdb_column_count(&r);
    const idx_t rows = duckdb_row_count(&r);
    for (idx_t c = 0; c < cols; ++c) {
        if (c) std::fputs("\t", stdout);
        std::fputs(duckdb_column_name(&r, c), stdout);
    }
    std::fputc('\n', stdout);
    for (idx_t i = 0; i < rows; ++i) {
        for (idx_t c = 0; c < cols; ++c) {
            if (c) std::fputs("\t", stdout);
            char* s = duckdb_value_varchar(&r, c, i);
            std::fputs(s ? s : "NULL", stdout);
            duckdb_free(s);
        }
        std::fputc('\n', stdout);
    }
}

} // namespace

int main(int argc, char** argv) {
    std::string db_path = ":memory:";
    std::string sql;

    bool multi = false;
    bool multi_last = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--db" && i + 1 < argc) {
            db_path = argv[++i];
        } else if (a == "--sql" && i + 1 < argc) {
            sql = argv[++i];
        } else if (a == "--multi") {
            multi = true;
        } else if (a == "--multi-last") {
            multi = true; multi_last = true;
        } else if (a == "-h" || a == "--help") {
            usage(argv[0]); return 0;
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", a.c_str());
            usage(argv[0]); return 1;
        }
    }
    if (sql.empty()) {
        std::stringstream ss;
        ss << std::cin.rdbuf();
        sql = ss.str();
    }
    if (sql.empty()) die("no SQL provided");

    duckdb_database db;
    if (duckdb_open(db_path.c_str(), &db) == DuckDBError) die("duckdb_open failed");
    duckdb_connection con;
    if (duckdb_connect(db, &con) == DuckDBError) die("duckdb_connect failed");

    try {
        gpudb_ext::register_gpu_sum(con);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "registration failed: %s\n", e.what());
        duckdb_disconnect(&con); duckdb_close(&db); return 2;
    }

    if (multi) {
        // Naive ';' split — good enough for test scripts (no ';' inside
        // string literals). Statements run on the SAME connection so
        // resident columns persist between them.
        std::size_t pos = 0;
        int stmt_no = 0;
        int rc = 0;
        while (pos < sql.size()) {
            std::size_t semi = sql.find(';', pos);
            std::string stmt = sql.substr(pos, semi == std::string::npos
                                                   ? std::string::npos : semi - pos);
            pos = (semi == std::string::npos) ? sql.size() : semi + 1;
            // Skip empty/whitespace-only fragments.
            const std::size_t nonws = stmt.find_first_not_of(" \t\r\n");
            if (nonws == std::string::npos) continue;
            ++stmt_no;
            const auto s0 = std::chrono::steady_clock::now();
            duckdb_result sr;
            if (duckdb_query(con, stmt.c_str(), &sr) == DuckDBError) {
                std::fprintf(stderr, "statement %d failed: %s\n",
                             stmt_no, duckdb_result_error(&sr));
                duckdb_destroy_result(&sr);
                rc = 3;
                break;
            }
            const auto s1 = std::chrono::steady_clock::now();
            const bool is_last = sql.find_first_not_of(" \t\r\n;", pos) == std::string::npos;
            if (!multi_last || is_last) print_result(sr);
            duckdb_destroy_result(&sr);
            std::fprintf(stderr, "[gpudb-sql] stmt %d elapsed %.3f ms\n", stmt_no,
                std::chrono::duration<double, std::milli>(s1 - s0).count());
        }
        duckdb_disconnect(&con);
        duckdb_close(&db);
        return rc;
    }

    const auto t0 = std::chrono::steady_clock::now();
    duckdb_result r;
    if (duckdb_query(con, sql.c_str(), &r) == DuckDBError) {
        std::fprintf(stderr, "query failed: %s\n", duckdb_result_error(&r));
        duckdb_destroy_result(&r); duckdb_disconnect(&con); duckdb_close(&db); return 3;
    }
    const auto t1 = std::chrono::steady_clock::now();
    print_result(r);
    duckdb_destroy_result(&r);

    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::fprintf(stderr, "[gpudb-sql] elapsed %.3f ms\n", ms);

    duckdb_disconnect(&con);
    duckdb_close(&db);
    return 0;
}
