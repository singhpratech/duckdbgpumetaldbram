add_executable(test_gpudb cpp/test_aggregator.cpp cpp/test_hashjoin.cpp)
find_package(Threads REQUIRED)
target_link_libraries(test_gpudb PRIVATE gpudb Threads::Threads)
add_test(NAME test_gpudb COMMAND test_gpudb)
