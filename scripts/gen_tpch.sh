#!/usr/bin/env bash
# gen_tpch.sh — generate TPC-H data using the DuckDB CLI's tpch extension,
# then export numeric columns to our flat .gpudb format for benchmarking.
#
# TPC-H is a TPC-published industry standard benchmark for analytical query
# processing. dbgen reference: https://www.tpc.org/tpch/
# DuckDB's tpch extension reproduces it under the MIT license:
#   https://duckdb.org/docs/extensions/tpch.html
#
# Usage: SF=1 ./scripts/gen_tpch.sh   # SF1 ≈ 1 GB lineitem
#        SF=10 ./scripts/gen_tpch.sh
#
# Requires: curl, the duckdb CLI (auto-downloaded into ./.tools if missing)

set -euo pipefail
cd "$(dirname "$0")/.."

SF="${SF:-1}"
DATA_DIR="data/tpch_sf${SF}"
mkdir -p "$DATA_DIR" .tools

# 1. Get DuckDB CLI if not present
DUCKDB="${DUCKDB:-./.tools/duckdb}"
if [[ ! -x "$DUCKDB" ]]; then
    OS="$(uname -s)"; ARCH="$(uname -m)"
    case "$OS-$ARCH" in
        Linux-x86_64)  asset="duckdb_cli-linux-amd64.zip" ;;
        Linux-aarch64) asset="duckdb_cli-linux-arm64.zip" ;;
        Darwin-arm64)  asset="duckdb_cli-osx-universal.zip" ;;
        Darwin-x86_64) asset="duckdb_cli-osx-universal.zip" ;;
        *) echo "unsupported platform: $OS-$ARCH"; exit 1 ;;
    esac
    echo "==> downloading DuckDB CLI ($asset)"
    curl -fsSL -o .tools/duckdb.zip \
        "https://github.com/duckdb/duckdb/releases/latest/download/$asset"
    (cd .tools && unzip -o duckdb.zip && rm duckdb.zip)
    chmod +x .tools/duckdb
fi

DBFILE="$DATA_DIR/tpch.duckdb"
echo "==> generating TPC-H SF=$SF into $DBFILE (this can take a while at high SF)"
"$DUCKDB" "$DBFILE" <<SQL
INSTALL tpch;
LOAD tpch;
CALL dbgen(sf = ${SF});
-- export the columns we care about for SUM/MIN/MAX benchmarks
COPY (SELECT l_quantity::DOUBLE      AS v FROM lineitem) TO '${DATA_DIR}/lineitem_quantity.parquet'      (FORMAT PARQUET);
COPY (SELECT l_extendedprice::DOUBLE AS v FROM lineitem) TO '${DATA_DIR}/lineitem_extendedprice.parquet' (FORMAT PARQUET);
COPY (SELECT l_orderkey::BIGINT      AS v FROM lineitem) TO '${DATA_DIR}/lineitem_orderkey.parquet'      (FORMAT PARQUET);
SQL

# 2. Convert each column to .gpudb flat binary
convert_col () {
    local pq="$1" out="$2" dtype="$3"
    echo "==> $pq -> $out  (dtype=$dtype)"
    "$DUCKDB" -c "
        COPY (SELECT v FROM read_parquet('${pq}'))
        TO '${out}.tmp.csv' (HEADER false, FORMAT CSV);
    "
    # Final conversion via the project's gpudb-gen-from-csv helper (built tool).
    # If the helper isn't built yet, leave the CSV — user can run the helper later.
    if [[ -x build-linux/bin/gpudb-csv2bin || -x build-macos/bin/gpudb-csv2bin || -x build/bin/gpudb-csv2bin ]]; then
        local tool
        for cand in build-linux/bin/gpudb-csv2bin build-macos/bin/gpudb-csv2bin build/bin/gpudb-csv2bin; do
            [[ -x "$cand" ]] && tool="$cand" && break
        done
        "$tool" --in "${out}.tmp.csv" --out "$out" --dtype "$dtype"
        rm -f "${out}.tmp.csv"
    else
        echo "  (gpudb-csv2bin not built yet — leaving CSV for now)"
    fi
}

convert_col "$DATA_DIR/lineitem_quantity.parquet"      "$DATA_DIR/lineitem_quantity.gpudb"      f64
convert_col "$DATA_DIR/lineitem_extendedprice.parquet" "$DATA_DIR/lineitem_extendedprice.gpudb" f64
convert_col "$DATA_DIR/lineitem_orderkey.parquet"      "$DATA_DIR/lineitem_orderkey.gpudb"      i64

echo
echo "Done. Files in $DATA_DIR:"
ls -lh "$DATA_DIR"
echo
echo "Bench: ./build-linux/bin/gpudb-bench --input $DATA_DIR/lineitem_orderkey.gpudb --runs 10"
