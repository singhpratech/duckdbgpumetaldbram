#!/usr/bin/env bash
# get_duckdb_libs.sh — fetch DuckDB pre-built libduckdb.so + headers into
# third_party/duckdb-libs/. Required for building the DuckDB extension
# (-DGPUDB_BUILD_EXT=ON). Downloads ~40 MB, extracts to ~140 MB.
#
# Why pre-built and not a submodule? DuckDB's full source clone + build is
# multi-GB and minutes per compile. For our purposes (registering aggregate
# functions via the C API in a loadable .so) the pre-built distribution is
# enough.
#
# Version: PINNED, not "latest". These libs back `gpudb-sql`, which is what
# ./scripts/run_sql_tests.sh runs the SQL suite through — so the DuckDB build
# under the suite must be the same one on every machine and in CI, or the
# expected answers in test/sql/*.test drift with whatever DuckDB shipped that
# morning. The default tracks the version the extension is published for and
# gated against: the community registry serves gpudb for DuckDB v1.5.5, which
# is also the hard leg of .github/workflows/duckdb-compat.yml and the floor
# stated in README.md. (Independent of the extension ABI, which is the
# vendored C_STRUCT header set in third_party/duckdb_capi/, TARGET_DUCKDB_VERSION
# = v1.2.0 in the root Makefile.)
#
# Usage:
#   ./scripts/get_duckdb_libs.sh                    # the pinned default below
#   DUCKDB_VERSION=v1.4.5 ./scripts/get_duckdb_libs.sh
#   DUCKDB_VERSION=latest ./scripts/get_duckdb_libs.sh   # whatever is newest
#   FORCE=1 ./scripts/get_duckdb_libs.sh            # re-fetch over an existing tree

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

DUCKDB_VERSION="${DUCKDB_VERSION:-v1.5.5}"

# `--print-version` prints the tag this script would fetch and exits. It exists
# so that a caller which must install something ELSE at the same version reads
# the pin from here instead of repeating it: CI installs the pip `duckdb`
# module at exactly this version to run python/tests/test_wrapper.py, and a
# second copy of the number would drift the day this default moves.
if [ "${1:-}" = "--print-version" ]; then
    echo "$DUCKDB_VERSION"
    exit 0
fi

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

if [ "${FORCE:-0}" != "1" ] &&
   [ -f "$DEST/duckdb.h" ] && { [ -f "$DEST/libduckdb.so" ] || [ -f "$DEST/libduckdb.dylib" ]; }; then
    echo "==> $DEST already populated; nothing to do"
    echo "    (FORCE=1 re-fetches; a tree fetched earlier can predate the pin)"
    have="$(sed -n 's/^#define DUCKDB_VERSION "\(.*\)"/\1/p' "$DEST/duckdb.hpp" 2>/dev/null | head -1 || true)"
    if [ -n "$have" ]; then
        echo "    on disk: $have   (pin: $DUCKDB_VERSION)"
    fi
    ls -lh "$DEST"
    exit 0
fi

# `latest` keeps the old floating behaviour, on request only.
if [ "$DUCKDB_VERSION" = "latest" ]; then
    url="https://github.com/duckdb/duckdb/releases/latest/download/$asset"
else
    url="https://github.com/duckdb/duckdb/releases/download/$DUCKDB_VERSION/$asset"
fi

echo "==> fetching $asset ($DUCKDB_VERSION)"
# GitHub's release CDN resets a connection now and then, and CI has failed on
# `curl: (35) Recv failure: Connection reset by peer`. --retry alone only retries
# transient HTTP codes and timeouts, so --retry-all-errors is what covers a reset
# mid-transfer (curl >= 7.71; ubuntu-24.04 ships 8.5, macOS 15 ships 8.7).
curl -fsSL --retry 5 --retry-delay 3 --retry-all-errors -o "$DEST/$asset" "$url"

echo "==> extracting"
(cd "$DEST" && unzip -o "$asset" && rm "$asset")

echo "==> done"
sed -n 's/^#define DUCKDB_VERSION "\(.*\)"/    version: \1/p' "$DEST/duckdb.hpp" 2>/dev/null | head -1
ls -lh "$DEST"
