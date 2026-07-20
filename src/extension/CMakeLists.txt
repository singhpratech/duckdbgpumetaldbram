# DuckDB extension targets. Built only when -DGPUDB_BUILD_EXT=ON.
#
# TWO INDEPENDENT PATHS:
#
#   1. Loadable extension  (gpudb_duckdb → libgpudb.<dylib|so>)
#      The artifact users LOAD from the DuckDB CLI. Built against ONLY the
#      vendored C API headers (third_party/duckdb_capi/) using the stable
#      C_STRUCT ABI, so it links NO libduckdb and requires NO pre-built libs.
#      This is what the community-extensions CI builds (`make release`).
#
#   2. Embedded CLI helper (gpudb_ext static + benchmark/gpudb-sql)
#      For in-process use / the SQL demo. This one genuinely links libduckdb,
#      so it requires third_party/duckdb-libs/ (./scripts/get_duckdb_libs.sh).
#      If those libs are absent it is silently skipped (STATUS, not FATAL) so
#      the CI build — which only wants the loadable lib — never trips on it.

# ---------------------------------------------------------------------------
# 1. Loadable extension — C_STRUCT ABI, no libduckdb
# ---------------------------------------------------------------------------
set(DUCKDB_CAPI_DIR "${CMAKE_SOURCE_DIR}/third_party/duckdb_capi")

if(NOT EXISTS "${DUCKDB_CAPI_DIR}/duckdb_extension.h")
    message(FATAL_ERROR
        "Vendored DuckDB C API headers missing at ${DUCKDB_CAPI_DIR}.\n"
        "These are committed to the repo (third_party/duckdb_capi/). If absent, "
        "restore them from the DuckDB v1.2.0 C API headers.")
endif()

add_library(gpudb_duckdb SHARED
    duckdb_loadable.cpp
    gpu_sum_extension.cpp
    gpu_join_extension.cpp)

target_include_directories(gpudb_duckdb PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${DUCKDB_CAPI_DIR})

# The core compute library (CPU / Metal / CUDA backends). It has NO DuckDB
# dependency, so linking it introduces no undefined duckdb_* symbols.
target_link_libraries(gpudb_duckdb PRIVATE gpudb)

target_compile_definitions(gpudb_duckdb PRIVATE
    GPUDB_C_STRUCT_ABI=1
    DUCKDB_EXTENSION_NAME=gpudb)

# Match the template: hide everything except the exported entrypoint.
set_target_properties(gpudb_duckdb PROPERTIES
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)

# Output naming / location.
#   - CI path (extension-ci-tools Makefile passes -DEXTENSION_NAME): emit
#     lib<EXTENSION_NAME>.<suffix> at the top-level build dir, exactly where the
#     makefile's build_extension_library_release copies from
#     (cmake_build/release/lib<name>.<suffix>).
#   - Local path (scripts/build.sh, no EXTENSION_NAME): keep the historical
#     name/location src/extension/libgpudb_duckdb.<suffix>.
if(DEFINED EXTENSION_NAME)
    set_target_properties(gpudb_duckdb PROPERTIES
        PREFIX "lib"
        OUTPUT_NAME "${EXTENSION_NAME}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
else()
    set_target_properties(gpudb_duckdb PROPERTIES
        PREFIX "lib"
        OUTPUT_NAME "gpudb_duckdb")
endif()

# ---- Local-only: package a loadable .duckdb_extension via the metadata footer.
# Skipped entirely under the CI path (EXTENSION_NAME defined): there the
# extension-ci-tools makefile appends the footer with its own script, and we
# must not fight it. This POST_BUILD keeps `./scripts/build.sh` and a bare
# `cmake --build` producing a directly-loadable artifact locally.
if(NOT DEFINED EXTENSION_NAME)
    if(APPLE)
        if(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "arm64" OR CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
            set(DUCKDB_PLATFORM "osx_arm64")
        else()
            set(DUCKDB_PLATFORM "osx_amd64")
        endif()
    elseif(UNIX)
        if(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "aarch64" OR CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "arm64")
            set(DUCKDB_PLATFORM "linux_arm64")
        else()
            set(DUCKDB_PLATFORM "linux_amd64")
        endif()
    else()
        set(DUCKDB_PLATFORM "unknown")
    endif()

    set(GPUDB_EXT_VERSION "v${PROJECT_VERSION}")
    set(GPUDB_DUCKDB_C_API_VERSION "v1.2.0")

    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(NOT Python3_Interpreter_FOUND)
        set(Python3_EXECUTABLE python3)
    endif()

    set(GPUDB_HELPER "${CMAKE_SOURCE_DIR}/scripts/build_helpers/append_extension_metadata.py")
    set(GPUDB_LOADABLE_OUT "$<TARGET_FILE_DIR:gpudb_duckdb>/gpudb.${DUCKDB_PLATFORM}.duckdb_extension")

    if(EXISTS "${GPUDB_HELPER}" AND NOT "${DUCKDB_PLATFORM}" STREQUAL "unknown")
        add_custom_command(TARGET gpudb_duckdb POST_BUILD
            COMMAND ${Python3_EXECUTABLE} ${GPUDB_HELPER}
                --library-file       $<TARGET_FILE:gpudb_duckdb>
                --extension-name     gpudb
                --duckdb-platform    ${DUCKDB_PLATFORM}
                --duckdb-version     ${GPUDB_DUCKDB_C_API_VERSION}
                --extension-version  ${GPUDB_EXT_VERSION}
                --abi-type           C_STRUCT
                --out-file           ${GPUDB_LOADABLE_OUT}
            COMMENT "Packaging gpudb.${DUCKDB_PLATFORM}.duckdb_extension (loadable via duckdb -unsigned -c \"LOAD '...';\")")
    endif()
endif()

# ---------------------------------------------------------------------------
# 2. Embedded CLI helper — links real libduckdb (optional, degrades gracefully)
# ---------------------------------------------------------------------------
set(DUCKDB_LIBS_DIR "${CMAKE_SOURCE_DIR}/third_party/duckdb-libs")

if(EXISTS "${DUCKDB_LIBS_DIR}/duckdb.h")
    # Compiled WITHOUT GPUDB_C_STRUCT_ABI: uses real duckdb.h + real libduckdb.
    add_library(gpudb_ext STATIC gpu_sum_extension.cpp)
    target_include_directories(gpudb_ext PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${DUCKDB_LIBS_DIR})
    target_link_libraries(gpudb_ext PUBLIC
        gpudb
        "${DUCKDB_LIBS_DIR}/libduckdb${CMAKE_SHARED_LIBRARY_SUFFIX}")
else()
    message(STATUS
        "third_party/duckdb-libs not found — skipping embedded gpudb_ext / gpudb-sql "
        "(loadable extension still builds). Run ./scripts/get_duckdb_libs.sh to enable.")
endif()
