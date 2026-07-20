#!/usr/bin/env bash
# local_check.sh — single command that runs everything we'd run in CI.
#
# Usage:
#   ./scripts/local_check.sh                  # full sweep
#   ./scripts/local_check.sh --no-cuda        # CPU-only path (mirrors CI)
#   ./scripts/local_check.sh --no-bench       # skip benchmarks (fast)
#   ./scripts/local_check.sh --no-extension   # skip DuckDB ext + gpudb-sql
#
# Exit 0 = all green. Exit nonzero = something failed.
#
# Use this BEFORE pushing while the GitHub Actions budget is paused.

set -euo pipefail
cd "$(dirname "$0")/.."

# Pick up CUDA toolkit if installed (no-op when CPU-only).
if [ -f scripts/env.sh ]; then
    # shellcheck disable=SC1091
    . scripts/env.sh >/dev/null 2>&1 || true
fi

NO_CUDA=0
NO_BENCH=0
NO_EXT=0
for arg in "$@"; do
    case "$arg" in
        --no-cuda)      NO_CUDA=1 ;;
        --no-bench)     NO_BENCH=1 ;;
        --no-extension) NO_EXT=1 ;;
        -h|--help)
            sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "unknown arg: $arg"; exit 2 ;;
    esac
done

CMAKE_FLAGS=()
if [ "$NO_CUDA" = "1" ]; then
    CMAKE_FLAGS+=( -DGPUDB_ENABLE_CUDA=OFF )
fi
if [ "$NO_EXT" = "0" ]; then
    if [ -f third_party/duckdb-libs/duckdb.h ]; then
        CMAKE_FLAGS+=( -DGPUDB_BUILD_EXT=ON )
    else
        echo "==> third_party/duckdb-libs missing; skipping extension build."
        echo "    (run ./scripts/get_duckdb_libs.sh to enable; ~140 MB download)"
        NO_EXT=1
    fi
fi

green () { printf "\033[32m%s\033[0m\n" "$*"; }
red   () { printf "\033[31m%s\033[0m\n" "$*"; }
hr    () { printf -- "\n--- %s ---\n" "$*"; }

hr "configure"
rm -rf build-linux
cmake -S . -B build-linux -DCMAKE_BUILD_TYPE=Release "${CMAKE_FLAGS[@]}" \
    > /tmp/gpudb-cmake.log 2>&1 || {
    red "configure failed (see /tmp/gpudb-cmake.log)"
    tail -20 /tmp/gpudb-cmake.log
    exit 1
}
green "configure OK"

hr "build"
cmake --build build-linux -j > /tmp/gpudb-build.log 2>&1 || {
    red "build failed (see /tmp/gpudb-build.log)"
    tail -30 /tmp/gpudb-build.log
    exit 1
}
green "build OK"

hr "unit tests"
if ! ./build-linux/test/test_gpudb; then
    red "tests failed"
    exit 1
fi
green "tests OK"

if [ "$NO_BENCH" = "0" ]; then
    hr "smoke bench (1M rows i64 SUM)"
    ./build-linux/bin/gpudb-bench --rows 1000000 --runs 2 --mode both
    green "bench smoke OK"

    hr "smoke groupby (1M rows × 1K groups)"
    ./build-linux/bin/gpudb-groupby-bench --rows 1000000 --groups 1000 --runs 2
    green "groupby smoke OK"

    hr "smoke hashjoin (100k build × 500k probe)"
    ./build-linux/bin/gpudb-hashjoin-bench --build 100000 --probe 500000 --runs 2
    green "hashjoin smoke OK"
fi

if [ "$NO_EXT" = "0" ]; then
    hr "DuckDB extension demo (gpu_sum on synthetic + correctness vs native sum)"
    ./build-linux/bin/gpudb-sql --sql \
        "SELECT gpu_sum(range::BIGINT) AS gpu, sum(range::BIGINT) AS native FROM range(10);"
    if [ -f data/tpch_sf1/lineitem_orderkey.parquet ]; then
        ./build-linux/bin/gpudb-sql --sql \
            "SELECT gpu_sum(v) AS gpu, sum(v) AS native FROM read_parquet('data/tpch_sf1/lineitem_orderkey.parquet');"
    fi
    green "extension OK"

    hr "SQL test suite (test/sql/*.test) — gpu_sum, gpu_min/max, GROUP BY, window"
    if ! ./scripts/run_sql_tests.sh; then
        red "SQL test suite reported FAIL (beyond the documented expected_fails)"
        exit 1
    fi
    green "SQL test suite OK"
fi

hr "summary"
green "ALL CHECKS PASSED"
echo
echo "Artifacts:"
find build-linux -maxdepth 4 -type f \
    \( -name 'gpudb-bench' -o -name 'gpudb-groupby-bench' -o -name 'gpudb-sql' \
       -o -name 'gpudb-gen' -o -name 'gpudb-csv2bin' \
       -o -name 'libgpudb*' -o -name 'libgpudb_duckdb*' \
       -o -name 'gpudb_duckdb.duckdb_extension' \
       -o -name 'test_gpudb' \) 2>/dev/null | sed 's/^/  /'
