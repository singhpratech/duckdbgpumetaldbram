# gpudb core library: backend abstraction + CPU + (optionally) CUDA + Metal

set(GPUDB_SOURCES
    backends/cpu/cpu_aggregator.cpp
    backends/backend_factory.cpp
)

if(GPUDB_HAS_CUDA)
    list(APPEND GPUDB_SOURCES
        backends/cuda/cuda_aggregator.cpp
        backends/cuda/kernels/sum_kernel.cu
    )
endif()

if(GPUDB_HAS_METAL)
    list(APPEND GPUDB_SOURCES
        backends/metal/metal_aggregator.mm
    )
endif()

add_library(gpudb STATIC ${GPUDB_SOURCES})

target_include_directories(gpudb
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# OpenMP for CPU backend (optional, falls back to scalar if not found)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(gpudb PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(gpudb PUBLIC GPUDB_HAVE_OPENMP=1)
endif()

# CUDA flags
if(GPUDB_HAS_CUDA)
    target_compile_definitions(gpudb PUBLIC GPUDB_HAVE_CUDA=1)
    set_target_properties(gpudb PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
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
