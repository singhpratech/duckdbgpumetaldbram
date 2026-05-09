add_executable(gpudb-bench bench_main.cpp)
target_link_libraries(gpudb-bench PRIVATE gpudb)
set_target_properties(gpudb-bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

add_executable(gpudb-gen gen_main.cpp)
target_link_libraries(gpudb-gen PRIVATE gpudb)
set_target_properties(gpudb-gen PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

add_executable(gpudb-csv2bin csv2bin_main.cpp)
target_link_libraries(gpudb-csv2bin PRIVATE gpudb)
set_target_properties(gpudb-csv2bin PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

add_executable(gpudb-groupby-bench groupby_bench_main.cpp)
target_link_libraries(gpudb-groupby-bench PRIVATE gpudb)
set_target_properties(gpudb-groupby-bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

add_executable(gpudb-hashjoin-bench hashjoin_bench_main.cpp)
target_link_libraries(gpudb-hashjoin-bench PRIVATE gpudb)
set_target_properties(gpudb-hashjoin-bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
