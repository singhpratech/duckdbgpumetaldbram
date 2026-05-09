# gpudb core library: backend abstraction + CPU + (optionally) CUDA + Metal

set(GPUDB_SOURCES
    backends/cpu/cpu_aggregator.cpp
    backends/cpu/cpu_groupby.cpp
    backends/cpu/cpu_hashjoin.cpp
    backends/backend_factory.cpp
)

if(GPUDB_HAS_CUDA)
    list(APPEND GPUDB_SOURCES
        backends/cuda/cuda_aggregator.cpp
        backends/cuda/cuda_groupby.cpp
        backends/cuda/cuda_hashjoin.cpp
        backends/cuda/kernels/sum_kernel.cu
        backends/cuda/kernels/groupby_kernel.cu
        backends/cuda/kernels/hashjoin_kernel.cu
    )
endif()

if(GPUDB_HAS_METAL)
    list(APPEND GPUDB_SOURCES
        backends/metal/metal_aggregator.mm
        backends/metal/metal_groupby.mm
    )

    # Embed .metal source as C strings so the Objective-C++ TUs can compile
    # them at runtime via MTLDevice newLibraryWithSource:options:error:.
    include(${CMAKE_SOURCE_DIR}/cmake/embed_metal_sources.cmake)
    set(GPUDB_METAL_GENERATED_HEADER
        ${CMAKE_CURRENT_BINARY_DIR}/generated/metal_kernel_sources.hpp)
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/generated)
    gpudb_embed_metal_sources(${GPUDB_METAL_GENERATED_HEADER}
        kSumKernelSource     ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/kernels/sum.metal
        kGroupByKernelSource ${CMAKE_CURRENT_SOURCE_DIR}/backends/metal/kernels/groupby.metal
    )
endif()

add_library(gpudb STATIC ${GPUDB_SOURCES})

target_include_directories(gpudb
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

if(GPUDB_HAS_METAL)
    target_include_directories(gpudb PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}/generated)
endif()

# OpenMP for CPU backend (optional, falls back to scalar if not found)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(gpudb PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(gpudb PUBLIC GPUDB_HAVE_OPENMP=1)
endif()

# CUDA flags
if(GPUDB_HAS_CUDA)
    target_compile_definitions(gpudb PUBLIC GPUDB_HAVE_CUDA=1)
    # Separable compilation OFF: keeps the static lib self-contained so that
    # downstream executables (test_gpudb, gpudb-bench, gpudb-sql, ...) don't
    # need their own device-link step. If we ever introduce cross-TU __device__
    # calls, flip this to ON and add CUDA_RESOLVE_DEVICE_SYMBOLS to consumers.
    set_target_properties(gpudb PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
    # cudart is needed by the .cpp host wrapper (cuda_runtime.h, cudaMalloc, ...)
    target_link_libraries(gpudb PUBLIC CUDA::cudart)
    # Suppress diagnostics that nvcc's generated stubs trip on, only for .cu files.
    target_compile_options(gpudb PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-pedantic>
    )
endif()

# Metal frameworks
if(GPUDB_HAS_METAL)
    target_compile_definitions(gpudb PUBLIC GPUDB_HAVE_METAL=1)
    target_link_libraries(gpudb PUBLIC
        "-framework Foundation"
        "-framework Metal"
        "-framework MetalPerformanceShaders"
    )
endif()
