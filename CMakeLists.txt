cmake_minimum_required(VERSION 3.20)

# -------------------------------------------------------------------
# macOS deployment target — the minimum macOS the produced binaries
# declare (Mach-O LC_BUILD_VERSION minos). It must be set BEFORE
# project(), because that is when the compiler is probed and the value
# baked into the toolchain.
#
# Left unset, a binary inherits the floor of whatever SDK built it, and
# claims to need the macOS it was compiled on. That is how the
# extension the community registry serves ended up at minos 15.0 while
# DuckDB's own core extensions ship at 11.0 — extension-ci-tools sets no
# deployment target of its own, so the value is whatever the CI runner
# happened to have. For a pip wheel that floor is not a detail: the
# platform tag is read from the binary, and it decides where the wheel
# can be installed at all.
#
# 15.0 is the floor of the binary the registry has been serving, so it
# is the one floor this project has evidence for.
#
# It is not the lowest that COMPILES: 14.0 also builds with no
# unguarded-availability warnings. But the shader-compile paths choose
# their Metal language version at run time — 3.2 on macOS 15 or newer,
# 3.1 below — and that 3.1 branch has never executed anywhere, because
# every machine this has run on is 15 or newer. A target of 14.0 would
# ship that branch on the compiler's word alone. Lower still (11.0) does
# not even compile cleanly: MTLLanguageVersion3_1 (macOS 14.0) in the
# shader-compile paths and MTLGPUFamilyMetal3 (macOS 13.0) in the
# aggregator are then reached unguarded, and guarding them would need a
# fallback language version this code does not have.
#
# So: 15.0 is claimed because a 15.0 binary has shipped and run. 14.0 is
# not claimed, and nothing here was tested on a macOS older than the one
# it was built on — no such machine is available.
if(APPLE AND NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET
         AND NOT DEFINED ENV{MACOSX_DEPLOYMENT_TARGET})
    set(CMAKE_OSX_DEPLOYMENT_TARGET "15.0" CACHE STRING
        "Minimum macOS version the produced binaries declare")
endif()

project(gpudb
    VERSION 0.7.0
    DESCRIPTION "GPU-accelerated operators for DuckDB (Metal + CUDA)"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# -------------------------------------------------------------------
# Build options
# -------------------------------------------------------------------
option(GPUDB_ENABLE_CUDA   "Enable CUDA backend (auto-detected)"  ON)
option(GPUDB_ENABLE_METAL  "Enable Metal backend (auto on macOS)" ON)
option(GPUDB_BUILD_TESTS   "Build unit tests"                     ON)
option(GPUDB_BUILD_BENCH   "Build CLI benchmark tool"             ON)
option(GPUDB_BUILD_EXT     "Build DuckDB extension wrapper"       OFF)
option(GPUDB_REQUIRE_CUDA  "Fail configure if the CUDA toolkit is not found (no silent CPU fallback)" OFF)
# Static cudart => the produced binaries have NO dynamic dependency on
# libcudart.so/libcuda.so, so a CUDA-enabled loadable extension still LOADs on
# GPU-less/driver-less machines (cudart_static dlopens the driver at runtime;
# backend_factory falls back to CPU when that fails). The community-CI Makefile
# turns this ON; the local ./scripts/build.sh flow keeps the shared runtime.
option(GPUDB_CUDA_STATIC_RUNTIME "Link the CUDA runtime statically" OFF)

# -------------------------------------------------------------------
# Compiler flags
# -------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Apply strict flags only to C++ host code; nvcc's generated stub files
# trip -Wpedantic on its own internal output, so scope by language.
add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
    $<$<COMPILE_LANGUAGE:OBJCXX>:-Wall>
    $<$<COMPILE_LANGUAGE:OBJCXX>:-Wextra>
    # Metal host code relies on ARC to release MTLBuffer / pipeline objects
    # (resident columns, sort caches, scratch); without it every device
    # allocation leaks for the life of the process.
    $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Debug>:-g>
)

# -------------------------------------------------------------------
# CUDA detection
# -------------------------------------------------------------------
set(GPUDB_HAS_CUDA OFF)
if(GPUDB_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        # RTX 4090 is sm_89 (Ada). Sirius requires sm_70+.
        # CUDA 13 dropped sm_70 (Volta) — start at sm_75 (Turing).
        # MUST be set BEFORE enable_language(CUDA): enable_language initializes
        # CMAKE_CUDA_ARCHITECTURES to nvcc's single default arch, after which a
        # `if(NOT ...)` guard never fires (that bug shipped sm_75-only fatbins).
        # Overridable via -DCMAKE_CUDA_ARCHITECTURES or the CUDAARCHS env var.
        # CUDAARCHS must be treated as unset when DEFINED-BUT-EMPTY: the
        # extension-ci-tools pipeline injects an empty CUDAARCHS into the build
        # container, and CMake ignores an empty value — a `NOT DEFINED` guard
        # would then skip our default and ship a single-arch fatbin again.
        if(NOT CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
            set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")
        endif()
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        # Need the runtime library + headers for the .cpp host wrapper too.
        find_package(CUDAToolkit REQUIRED)
        if(GPUDB_CUDA_STATIC_RUNTIME)
            set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
        endif()
        set(GPUDB_HAS_CUDA ON)
        message(STATUS "CUDA enabled (compiler: ${CMAKE_CUDA_COMPILER}, archs: ${CMAKE_CUDA_ARCHITECTURES}, toolkit: ${CUDAToolkit_VERSION}, static runtime: ${GPUDB_CUDA_STATIC_RUNTIME})")
    else()
        if(GPUDB_REQUIRE_CUDA)
            message(FATAL_ERROR "GPUDB_REQUIRE_CUDA=ON but no CUDA compiler was found. Put nvcc on PATH (e.g. export PATH=/usr/local/cuda/bin:$PATH) or set CUDACXX=/path/to/nvcc, or build without the flag to get the CPU-only fallback.")
        endif()
        message(STATUS "CUDA toolkit not found — CUDA backend disabled")
    endif()
endif()

# -------------------------------------------------------------------
# Metal detection
# -------------------------------------------------------------------
set(GPUDB_HAS_METAL OFF)
if(GPUDB_ENABLE_METAL AND APPLE)
    set(GPUDB_HAS_METAL ON)
    enable_language(OBJCXX)
    set(CMAKE_OBJCXX_STANDARD 17)
    message(STATUS "Metal enabled (macOS)")
endif()

# -------------------------------------------------------------------
# Library
# -------------------------------------------------------------------
add_subdirectory(src)

# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------
if(GPUDB_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

# -------------------------------------------------------------------
# DuckDB extension (loadable .so)
# -------------------------------------------------------------------
# Added BEFORE benchmark so the optional gpudb_ext target (embedded CLI helper)
# is known when benchmark/CMakeLists.txt decides whether to build gpudb-sql.
if(GPUDB_BUILD_EXT)
    add_subdirectory(src/extension)
endif()

# -------------------------------------------------------------------
# Benchmarks
# -------------------------------------------------------------------
if(GPUDB_BUILD_BENCH)
    add_subdirectory(benchmark)
endif()

# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------
message(STATUS "")
message(STATUS "gpudb configuration:")
message(STATUS "  Version:       ${PROJECT_VERSION}")
message(STATUS "  Build type:    ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA backend:  ${GPUDB_HAS_CUDA}")
message(STATUS "  Metal backend: ${GPUDB_HAS_METAL}")
message(STATUS "  Tests:         ${GPUDB_BUILD_TESTS}")
message(STATUS "  Benchmarks:    ${GPUDB_BUILD_BENCH}")
message(STATUS "  DuckDB ext:    ${GPUDB_BUILD_EXT}")
message(STATUS "")
