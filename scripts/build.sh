#!/usr/bin/env bash
# build.sh — convenience wrapper. Detects platform; sets sensible defaults.
set -euo pipefail

cd "$(dirname "$0")/.."

# Pick up CUDA toolkit even in non-interactive shells (~/.bashrc isn't read).
if [ -f "$(dirname "$0")/env.sh" ]; then
    # shellcheck disable=SC1091
    . "$(dirname "$0")/env.sh" >/dev/null 2>&1 || true
fi

OS="$(uname -s)"
case "$OS" in
    Linux*)
        BUILD_DIR="${BUILD_DIR:-build-linux}"
        ;;
    Darwin*)
        BUILD_DIR="${BUILD_DIR:-build-macos}"
        ;;
    *)
        BUILD_DIR="${BUILD_DIR:-build}"
        ;;
esac

CMAKE_ARGS=(
    -S .
    -B "$BUILD_DIR"
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE:-Release}"
)

# Allow user to disable CUDA/Metal explicitly:
if [[ "${GPUDB_NO_CUDA:-}" == "1" ]]; then
    CMAKE_ARGS+=(-DGPUDB_ENABLE_CUDA=OFF)
fi
if [[ "${GPUDB_NO_METAL:-}" == "1" ]]; then
    CMAKE_ARGS+=(-DGPUDB_ENABLE_METAL=OFF)
fi

echo "==> configure ($OS) into $BUILD_DIR"
cmake "${CMAKE_ARGS[@]}"

echo "==> build"
cmake --build "$BUILD_DIR" -j

# Append DuckDB extension metadata footer so the .duckdb_extension is
# loadable from the duckdb CLI via `LOAD '<path>'`. Skipped if the build
# didn't produce the .so (e.g. -DGPUDB_BUILD_EXT=OFF).
LOADABLE_SO="$BUILD_DIR/src/extension/libgpudb_duckdb.so"
LOADABLE_DYLIB="$BUILD_DIR/src/extension/libgpudb_duckdb.dylib"
RAW_EXT=""
if [ -f "$LOADABLE_SO" ];   then RAW_EXT="$LOADABLE_SO";   fi
if [ -f "$LOADABLE_DYLIB" ]; then RAW_EXT="$LOADABLE_DYLIB"; fi

if [ -n "$RAW_EXT" ] && [ -f "$(dirname "$0")/build_helpers/append_extension_metadata.py" ]; then
    case "$OS" in
        Linux*)  PLATFORM="linux_amd64" ;;
        Darwin*) PLATFORM="osx_arm64"   ;;
        *)       PLATFORM="" ;;
    esac
    DUCKDB_C_API_VERSION="${DUCKDB_C_API_VERSION:-v1.2.0}"
    EXT_VERSION="${EXT_VERSION:-v0.1.1}"
    OUT_FILE="$BUILD_DIR/src/extension/gpudb.${PLATFORM}.duckdb_extension"
    if [ -n "$PLATFORM" ]; then
        echo "==> packaging loadable extension ($PLATFORM, ABI=$DUCKDB_C_API_VERSION, ext=$EXT_VERSION)"
        python3 "$(dirname "$0")/build_helpers/append_extension_metadata.py" \
            --library-file "$RAW_EXT" \
            --extension-name gpudb \
            --duckdb-platform "$PLATFORM" \
            --duckdb-version "$DUCKDB_C_API_VERSION" \
            --extension-version "$EXT_VERSION" \
            --out-file "$OUT_FILE" >/dev/null
        echo "    -> $OUT_FILE"
    fi
fi

echo
echo "Built artifacts:"
find "$BUILD_DIR" -maxdepth 4 -type f \( -name 'gpudb-bench' -o -name 'test_gpudb' -o -name 'libgpudb*' -o -name 'gpudb*.duckdb_extension' \) 2>/dev/null | sed 's/^/  /'
echo
echo "Run tests:    ./$BUILD_DIR/test/test_gpudb"
echo "Run bench:    ./$BUILD_DIR/bin/gpudb-bench --rows 10000000"
echo "Load in CLI:  duckdb -unsigned -c \"LOAD '$(pwd)/$BUILD_DIR/src/extension/gpudb.<platform>.duckdb_extension'; SELECT gpu_sum(range::BIGINT) FROM range(1000);\""
