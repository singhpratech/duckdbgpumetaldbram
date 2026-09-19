#!/usr/bin/env bash
# python_wrapper_check.sh — run the gpudb Python wrapper tests against the
# built extension (build-macos or build-linux; or GPUDB_EXTENSION_PATH), then
# the tests for the `gpudb` shell that sits on top of the wrapper.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 python/tests/test_wrapper.py "$@"
python3 python/tests/test_shell.py
