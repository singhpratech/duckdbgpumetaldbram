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
    "${DUCKDB_LIBS_DIR}/libduckdb.so"
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

# Also produce the same .so under the .duckdb_extension suffix DuckDB recognizes,
# so users can do LOAD 'libgpudb_duckdb.duckdb_extension' or just LOAD '...so'.
add_custom_command(TARGET gpudb_duckdb POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:gpudb_duckdb>
        $<TARGET_FILE_DIR:gpudb_duckdb>/gpudb_duckdb.duckdb_extension
    COMMENT "Aliasing .so as .duckdb_extension")
