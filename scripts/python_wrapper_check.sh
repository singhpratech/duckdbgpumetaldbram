#!/usr/bin/env bash
# python_wrapper_check.sh — run the gpudb Python wrapper tests against the
# built extension (build-macos or build-linux; or GPUDB_EXTENSION_PATH).
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 python/tests/test_wrapper.py "$@"
