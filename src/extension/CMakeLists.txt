# DuckDB extension targets. Built only when -DGPUDB_BUILD_EXT=ON.
#
# Three artifacts:
#   - gpudb_ext (static)         : gpudb_ext::register_gpu_sum(con) for in-process use
#                                  (linked by benchmark/sql_demo_main.cpp)
#   - libgpudb_duckdb (.so)      : DuckDB loadable extension
#                                  LOAD '/path/to/libgpudb_duckdb.so' from the DuckDB CLI
#                                  exposes gpu_sum / gpu_min / gpu_max in SQL
#   - libgpudb_duckdb.duckdb_extension (alias .so with the right name suffix)

set(DUCKDB_LIBS_DIR "${CMAKE_SOURCE_DIR}/third_party/duckdb-libs")

if(NOT EXISTS "${DUCKDB_LIBS_DIR}/duckdb.h")
    message(FATAL_ERROR
        "DuckDB pre-built libs missing at ${DUCKDB_LIBS_DIR}.\n"
        "Run: ./scripts/get_duckdb_libs.sh")
endif()

# ---- Static lib: in-process registration helper ----
add_library(gpudb_ext STATIC gpu_sum_extension.cpp)
target_include_directories(gpudb_ext PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${DUCKDB_LIBS_DIR}
)
target_link_libraries(gpudb_ext PUBLIC
    gpudb
    "${DUCKDB_LIBS_DIR}/libduckdb${CMAKE_SHARED_LIBRARY_SUFFIX}"
)

# ---- Loadable .so: what users actually LOAD from DuckDB ----
add_library(gpudb_duckdb SHARED duckdb_loadable.cpp)
target_link_libraries(gpudb_duckdb PRIVATE gpudb_ext)
set_target_properties(gpudb_duckdb PROPERTIES
    PREFIX "lib"
    OUTPUT_NAME "gpudb_duckdb"
    BUILD_RPATH "${DUCKDB_LIBS_DIR}"
    INSTALL_RPATH "${DUCKDB_LIBS_DIR}"
)

# Produce a real, LOADABLE .duckdb_extension by appending DuckDB's required
# metadata footer to the .so. Without this, DuckDB CLI rejects the file as
# "The metadata at the end of the file is invalid".
#
# Implementation: call scripts/build_helpers/append_extension_metadata.py
# (the upstream DuckDB extension-template helper, vendored at PR #36) so
# both `cmake --build` and `./scripts/build.sh` produce the same loadable
# artifact at gpudb.<platform>.duckdb_extension.

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
