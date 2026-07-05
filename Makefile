# Root Makefile for the DuckDB community-extensions CI.
#
# This is the entrypoint the community-extensions pipeline drives (reusable
# workflow duckdb/extension-ci-tools .../_extension_distribution.yml). It mirrors
# the canonical C-API extension template (duckdb/extension-template-c) by
# including the two c_api_extensions makefiles, which provide:
#   set_duckdb_version / configure_ci / configure / release / test_release ...
#
# It builds ONLY the loadable extension (gpudb.duckdb_extension) via the stable
# C_STRUCT ABI — no libduckdb link, no DuckDB source/submodule. For the local
# developer flow (unit tests, benchmarks, the embedded gpudb-sql CLI) use
# ./scripts/build.sh instead; the two paths are independent.

.PHONY: clean clean_all

PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Main extension configuration
EXTENSION_NAME=gpudb
EXTENSION_CANONICAL=gpudb

# Stable C API (C_STRUCT). 0 => binaries are forward-compatible with any DuckDB
# whose C API >= TARGET_DUCKDB_VERSION (the CI builds against DuckDB v1.5.2).
USE_UNSTABLE_C_API=0

# The C API version we target. v1.2.0 is the stable-ABI baseline; the vendored
# headers in third_party/duckdb_capi/ match it, and this is the value stamped
# into the extension metadata footer (FIELD3 duckdb_version).
TARGET_DUCKDB_VERSION=v1.2.0

# CMake flags for the loadable-extension build path only.
#   - GPUDB_BUILD_EXT=ON    : build the loadable extension target
#   - TESTS/BENCH=OFF       : CI only wants the extension artifact
#   - ENABLE_CUDA=OFF       : no CUDA toolchain wired into CI yet (phase 2);
#                             Metal stays auto-on for APPLE (osx_arm64 job)
CMAKE_EXTRA_BUILD_FLAGS=-DGPUDB_BUILD_EXT=ON -DGPUDB_BUILD_TESTS=OFF -DGPUDB_BUILD_BENCH=OFF -DGPUDB_ENABLE_CUDA=OFF

all: configure release

# Include makefiles from extension-ci-tools (checked out as a submodule locally;
# the CI checks it out into ./extension-ci-tools itself).
include extension-ci-tools/makefiles/c_api_extensions/base.Makefile
include extension-ci-tools/makefiles/c_api_extensions/c_cpp.Makefile

# Redirect the sqllogictest runner to test/sqllogic/ so it does NOT pick up the
# repo's custom-format test/sql/*.test suite (run by scripts/run_sql_tests.sh,
# a different, incompatible format). base.Makefile hard-codes `--test-dir
# test/sql`; overriding TEST_RUNNER_BASE (used by TEST_RUNNER_DEBUG/RELEASE)
# points sqllogictest at our isolated, valid-format directory instead.
override TEST_RUNNER_BASE = $(TEST_RUNNER) --test-dir test/sqllogic $(EXTRA_EXTENSIONS_PARAM)

configure: venv platform extension_version

debug: build_extension_library_debug build_extension_with_metadata_debug
release: build_extension_library_release build_extension_with_metadata_release

test: test_release
test_debug: test_extension_debug
test_release: test_extension_release

clean: clean_build clean_cmake
clean_all: clean clean_configure
