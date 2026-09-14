// concurrent_upload_bench.cpp — the residency gate row
// (docs/TRANSPARENT_DESIGN.md §5.6 / §9.3): a native statement on ONE
// connection while a resident upload (+ prepare) runs on ANOTHER connection
// of the same database, in one process, on the same DuckDB worker pool.
//
// Reports the native statement's min / median / p99 / max alone (baseline)
// and during the upload, and the ratio baseline / concurrent at p99 — the
// number rule 1 needs to be ≥ 1.0 ("the first query pays nothing") and the
// number min-of-N hides entirely (it reported 80 ms during an 800 ms
// window). Every timing is statement-vs-statement on one clock.
//
//   gpudb-concurrent-bench --db data/tpch_bench/tpch_sf10.duckdb \
//       --native "SELECT l_orderkey, sum(l_quantity) FROM lineitem GROUP BY 1 HAVING sum(l_quantity) > 300" \
//       --upload "SELECT gpu_upload_pair('gpudb:v1:memory:main:lineitem:1:l_orderkey,l_quantity', l_orderkey, l_quantity::BIGINT) FROM lineitem" \
//       [--control "SELECT len(list(l_orderkey)), len(list(l_quantity)) FROM lineitem"] \
//       [--iters 30] [--upload-iters 3] [--rounds 2] [--label q18] [--verbose]
//
// Output: one line per phase plus a final `gate:` line the residency gate
// script parses. Exit code 0 always — the pass/fail decision is the
// script's (it compares against the threshold it prints).

#include "gpu_sum_extension.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

[[noreturn]] void die(const char* what) {
    std::fprintf(stderr, "gpudb-concurrent-bench: %s\n", what);
    std::exit(2);
}

double run_ms(duckdb_connection con, const std::string& sql, const char* what) {
    const auto t0 = std::chrono::steady_clock::now();
    duckdb_result r;
    if (duckdb_query(con, sql.c_str(), &r) == DuckDBError) {
        std::fprintf(stderr, "%s failed: %s\n", what, duckdb_result_error(&r));
        duckdb_destroy_result(&r);
        std::exit(3);
    }
    duckdb_destroy_result(&r);
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

struct Stats { double min = 0, med = 0, p99 = 0, max = 0; std::size_t n = 0; };

Stats stats_of(std::vector<double> v) {
    Stats s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    s.n = v.size();
    s.min = v.front();
    s.max = v.back();
    s.med = v[v.size() / 2];
    const std::size_t i99 = static_cast<std::size_t>(std::min<double>(
        static_cast<double>(v.size() - 1), std::ceil(0.99 * static_cast<double>(v.size())) - 1));
    s.p99 = v[i99];
    return s;
}

} // namespace

int main(int argc, char** argv) {
    std::string db_path, native, upload, control, label = "native";
    int iters = 30, upload_iters = 3, rounds = 2;
    bool verbose = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) die("missing argument value");
            return argv[++i];
        };
        if      (a == "--db")           db_path = next();
        else if (a == "--native")       native = next();
        else if (a == "--upload")       upload = next();
        else if (a == "--control")      control = next();
        else if (a == "--rounds")       rounds = std::atoi(next().c_str());
        else if (a == "--label")        label = next();
        else if (a == "--iters")        iters = std::atoi(next().c_str());
        else if (a == "--upload-iters") upload_iters = std::atoi(next().c_str());
        else if (a == "--verbose")      verbose = true;
        else die("unknown argument (see the header of concurrent_upload_bench.cpp)");
    }
    if (db_path.empty() || native.empty() || upload.empty()) die("--db, --native and --upload are required");
    if (iters < 3) iters = 3;

    duckdb_database db;
    if (duckdb_open(db_path.c_str(), &db) == DuckDBError) die("duckdb_open failed");
    duckdb_connection con_native, con_upload;
    if (duckdb_connect(db, &con_native) == DuckDBError) die("duckdb_connect (native) failed");
    if (duckdb_connect(db, &con_upload) == DuckDBError) die("duckdb_connect (upload) failed");
    try {
        gpudb_ext::register_gpu_sum(con_native);   // one registry for the database
    } catch (const std::exception& e) {
        std::fprintf(stderr, "registration failed: %s\n", e.what());
        return 2;
    }

    // Warm both statements once (page cache, DuckDB buffer pool, CUDA
    // context) so neither phase pays first-touch costs.
    (void)run_ms(con_native, native, "native warm-up");
    const double upload_warm_ms = run_ms(con_upload, upload, "upload warm-up");
    std::printf("[%s] upload statement alone: %.1f ms\n", label.c_str(), upload_warm_ms);

    // Phase 1 — baseline: native alone.
    std::vector<double> base;
    for (int i = 0; i < iters; ++i) base.push_back(run_ms(con_native, native, "native"));
    const Stats b = stats_of(base);
    std::printf("[%s] baseline  n=%zu min=%.1f med=%.1f p99=%.1f max=%.1f ms\n",
                label.c_str(), b.n, b.min, b.med, b.p99, b.max);

    // Phase 2 — concurrent: the second connection runs a competitor statement
    // back-to-back on its own thread; every native run recorded overlapped a
    // competitor in flight (we stop as soon as the competitor thread is done
    // and drop any run that started after it finished). With --control, the
    // control statement (a native buffering aggregate over the same columns)
    // and the upload alternate for `rounds` rounds inside ONE process against
    // ONE baseline, so the two are compared on the same clock and the same
    // machine state — separate invocations drift by 10–50% on a shared box.
    struct Phase { std::vector<double> native, competitor; };
    auto run_phase = [&](const std::string& sql, const char* what, Phase& acc) {
        std::atomic<bool> running{true};
        const auto phase0 = std::chrono::steady_clock::now();
        auto since = [&] { return std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - phase0).count(); };
        if (verbose) {
            const auto wall = std::chrono::system_clock::now();
            const std::time_t tt = std::chrono::system_clock::to_time_t(wall);
            char buf[32];
            std::strftime(buf, sizeof(buf), "%H:%M:%S", std::localtime(&tt));
            const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(wall.time_since_epoch()).count() % 1000;
            std::printf("  [%s] phase start wall-clock %s.%03lld\n", what, buf, static_cast<long long>(ms));
        }
        std::thread competitor([&] {
            for (int i = 0; i < upload_iters; ++i) {
                const double t = since();
                acc.competitor.push_back(run_ms(con_upload, sql, what));
                if (verbose) std::printf("  %s #%d  t=%.0f..%.0f ms\n", what, i, t, since());
            }
            running.store(false);
        });
        while (running.load()) {
            const double t = since();
            const double ms = run_ms(con_native, native, "native (concurrent)");
            if (running.load()) {
                acc.native.push_back(ms);
                if (verbose) std::printf("  native t=%.0f ms: %.1f ms\n", t, ms);
            }
        }
        competitor.join();
    };

    Phase up, ctl;
    if (!control.empty()) (void)run_ms(con_upload, control, "control warm-up");
    for (int r = 0; r < std::max(1, rounds); ++r) {
        if (!control.empty()) run_phase(control, "control", ctl);
        run_phase(upload, "upload", up);
    }

    auto report = [&](const char* what, const Phase& ph) {
        const Stats c = stats_of(ph.native);
        const Stats u = stats_of(ph.competitor);
        std::printf("[%s/%s] competitor statement: n=%zu min=%.1f med=%.1f max=%.1f ms\n",
                    label.c_str(), what, u.n, u.min, u.med, u.max);
        std::printf("[%s/%s] native concurrent n=%zu min=%.1f med=%.1f p99=%.1f max=%.1f ms\n",
                    label.c_str(), what, c.n, c.min, c.med, c.p99, c.max);
        if (c.n == 0)
            std::printf("gate: %s/%s no concurrent sample (competitor too short — raise --upload-iters)\n",
                        label.c_str(), what);
        else
            std::printf("gate: %s/%s p99 %.1f -> %.1f ms  ratio(baseline/concurrent) med=%.2fx p99=%.2fx max=%.2fx\n",
                        label.c_str(), what, b.p99, c.p99, b.med / c.med, b.p99 / c.p99, b.max / c.max);
        return c;
    };
    if (!control.empty()) {
        const Stats cc = report("control", ctl);
        const Stats cu = report("upload", up);
        if (cc.n && cu.n)
            std::printf("gate: %s upload-vs-control concurrent med %.1f vs %.1f ms (%.2fx)  p99 %.1f vs %.1f ms (%.2fx)\n",
                        label.c_str(), cu.med, cc.med, cc.med / cu.med, cu.p99, cc.p99, cc.p99 / cu.p99);
    } else {
        (void)report("upload", up);
    }

    duckdb_disconnect(&con_upload);
    duckdb_disconnect(&con_native);
    duckdb_close(&db);
    return 0;
}
