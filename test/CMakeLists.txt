add_executable(test_gpudb cpp/test_aggregator.cpp)
target_link_libraries(test_gpudb PRIVATE gpudb)
add_test(NAME test_gpudb COMMAND test_gpudb)
