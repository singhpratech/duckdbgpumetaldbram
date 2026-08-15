cmake_minimum_required(VERSION 3.20)

project(gpudb
    VERSION 0.4.0
    DESCRIPTION "GPU-accelerated operators for DuckDB (CUDA + Metal)"
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
