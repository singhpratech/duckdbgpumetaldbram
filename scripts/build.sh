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

echo
echo "Built artifacts:"
find "$BUILD_DIR" -maxdepth 4 -type f \( -name 'gpudb-bench' -o -name 'test_gpudb' -o -name 'libgpudb*' \) 2>/dev/null | sed 's/^/  /'
echo
echo "Run tests:    ./$BUILD_DIR/test/test_gpudb"
echo "Run bench:    ./$BUILD_DIR/bin/gpudb-bench --rows 10000000"
