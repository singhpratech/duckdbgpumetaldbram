# DuckDB extension code: registers gpu_sum() on a DuckDB connection.
# Built only when -DGPUDB_BUILD_EXT=ON.
#
# Two artifacts:
#   - libgpudb_ext (static): exposes gpudb_ext::register_gpu_sum(con).
#     Used by benchmark/sql_demo.cpp and by the future loadable .so wrapper.
#   - libduckdb.so is linked at build time via -L/-l so we get the symbols.

set(DUCKDB_LIBS_DIR "${CMAKE_SOURCE_DIR}/third_party/duckdb-libs")

if(NOT EXISTS "${DUCKDB_LIBS_DIR}/duckdb.h")
    message(FATAL_ERROR
        "DuckDB pre-built libs missing at ${DUCKDB_LIBS_DIR}.\n"
        "Run: ./scripts/get_duckdb_libs.sh")
endif()

add_library(gpudb_ext STATIC gpu_sum_extension.cpp)
target_include_directories(gpudb_ext PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${DUCKDB_LIBS_DIR}
)
target_link_libraries(gpudb_ext PUBLIC
    gpudb
    "${DUCKDB_LIBS_DIR}/libduckdb.so"
)
