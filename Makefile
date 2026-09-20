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

.PHONY: clean clean_all check_git_checkout

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

# CUDA: auto-detected via nvcc, x86_64 only. NOTE the registry does NOT
# currently provide nvcc: community-extensions pins extension-ci-tools to a
# branch (v1.5-variegata) with no CUDA toolchain support, so its containers
# never have nvcc and this stays OFF there. The "cuda" extra_toolchains token
# exists only on extension-ci-tools main (unreleased); when the registry's pin
# moves and description.yml lists cuda in requires_toolchains, this auto-flips
# ON with no further changes here.
#
# The uname gate exists because a future cuda-capable ci-tools would also
# install the sbsa toolkit on linux_arm64 — an aarch64 CUDA build we have
# never validated. Lift the gate once sbsa is actually tested.
#
# GPUDB_CUDA_STATIC_RUNTIME=ON is the load-safety requirement, not an
# optimization: the shipped .so must have NO dynamic dependency on
# libcudart.so/libcuda.so (verify with ldd) so it LOADs on GPU-less machines;
# cudart_static dlopens the driver at runtime and backend_factory falls back
# to CPU when no device/driver is present. CUDA archs come from the CMake
# default (75;80;86;89;90).
NVCC := $(shell command -v nvcc 2>/dev/null)
UNAME_M := $(shell uname -m)
ifeq ($(NVCC),)
GPUDB_CUDA_FLAGS=-DGPUDB_ENABLE_CUDA=OFF
else ifneq ($(UNAME_M),x86_64)
GPUDB_CUDA_FLAGS=-DGPUDB_ENABLE_CUDA=OFF
else
GPUDB_CUDA_FLAGS=-DGPUDB_ENABLE_CUDA=ON -DGPUDB_CUDA_STATIC_RUNTIME=ON
endif

# CMake flags for the loadable-extension build path only.
#   - GPUDB_BUILD_EXT=ON    : build the loadable extension target
#   - TESTS/BENCH=OFF       : CI only wants the extension artifact
CMAKE_EXTRA_BUILD_FLAGS=-DGPUDB_BUILD_EXT=ON -DGPUDB_BUILD_TESTS=OFF -DGPUDB_BUILD_BENCH=OFF $(GPUDB_CUDA_FLAGS)

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

# This build path needs a git checkout, not just the sources. The version
# stamped into the extension metadata footer comes from `git describe` (the
# `extension_version` target in base.Makefile). Built from an unpacked tarball
# that command fails, its error text is written into the footer instead of a
# version, and the resulting binary cannot be loaded at all — DuckDB reports
# something like "Unknown ABI type for extension: 't filesystem boundary
# (GIT_DISCO'". Nothing further along the pipeline notices, so stop here and
# say why. In a git checkout this target does nothing.
check_git_checkout:
	@git -C "$(PROJ_DIR)" rev-parse --git-dir >/dev/null 2>&1 || { \
	    echo "Makefile: $(PROJ_DIR) is not a git checkout (or git is not installed)."; \
	    echo "  This build path stamps the extension version from \`git describe\`;"; \
	    echo "  without a repository the metadata footer would hold git's error text"; \
	    echo "  and the extension would not load. Build from a \`git clone\` of the"; \
	    echo "  repository, or use ./scripts/build.sh for the local developer flow."; \
	    exit 1; \
	}

configure: check_git_checkout venv platform extension_version

# ... and the same guard directly on the target that reads git, so it holds
# however the pipeline reaches it.
extension_version: check_git_checkout

debug: build_extension_library_debug build_extension_with_metadata_debug
release: build_extension_library_release build_extension_with_metadata_release

test: test_release
test_debug: test_extension_debug
test_release: test_extension_release

clean: clean_build clean_cmake
clean_all: clean clean_configure
