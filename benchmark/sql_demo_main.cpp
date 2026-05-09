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
        "usage: %s [--db FILE] [--sql QUERY]\n"
        "  --db FILE    use FILE as the DuckDB database (default: in-memory)\n"
        "  --sql QUERY  run this single SQL statement (default: read stdin)\n", a0);
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

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--db" && i + 1 < argc) {
            db_path = argv[++i];
        } else if (a == "--sql" && i + 1 < argc) {
            sql = argv[++i];
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
