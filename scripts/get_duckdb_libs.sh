#!/usr/bin/env bash
# get_duckdb_libs.sh — fetch DuckDB pre-built libduckdb.so + headers into
# third_party/duckdb-libs/. Required for building the DuckDB extension
# (-DGPUDB_BUILD_EXT=ON). Downloads ~40 MB, extracts to ~140 MB.
#
# Why pre-built and not a submodule? DuckDB's full source clone + build is
# multi-GB and minutes per compile. For our purposes (registering aggregate
# functions via the C API in a loadable .so) the pre-built distribution is
# enough.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

DEST="third_party/duckdb-libs"
mkdir -p "$DEST"

OS="$(uname -s)"; ARCH="$(uname -m)"
case "$OS-$ARCH" in
    Linux-x86_64)  asset="libduckdb-linux-amd64.zip"  ;;
    Linux-aarch64) asset="libduckdb-linux-arm64.zip"  ;;
    Darwin-arm64)  asset="libduckdb-osx-universal.zip" ;;
    Darwin-x86_64) asset="libduckdb-osx-universal.zip" ;;
    *) echo "unsupported platform: $OS-$ARCH" >&2; exit 1 ;;
esac

if [ -f "$DEST/duckdb.h" ] && [ -f "$DEST/libduckdb.so" -o -f "$DEST/libduckdb.dylib" ]; then
    echo "==> $DEST already populated; nothing to do"
    ls -lh "$DEST"
    exit 0
fi

echo "==> downloading $asset"
curl -fsSL -o "$DEST/$asset" \
    "https://github.com/duckdb/duckdb/releases/latest/download/$asset"

echo "==> extracting"
(cd "$DEST" && unzip -o "$asset" && rm "$asset")

echo "==> done"
ls -lh "$DEST"
