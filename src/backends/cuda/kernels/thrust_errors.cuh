// thrust_errors.cuh — map exceptions thrown inside Thrust/CUB calls to the
// cudaError_t the launchers return, so the host wrapper reports the real
// cause (launch failure, illegal address, bad stream, ...) instead of
// labelling every failure "out of memory".
#pragma once

#include <cuda_runtime.h>
#include <new>
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>

namespace gpudb_cuda_detail {

// Call as: `catch (...) { return gpudb_cuda_detail::map_exception(); }`
inline cudaError_t map_exception() noexcept {
    try {
        throw;
    } catch (const thrust::system_error& e) {
        if (e.code().category() == thrust::cuda_category())
            return static_cast<cudaError_t>(e.code().value());
        return cudaErrorUnknown;
    } catch (const std::bad_alloc&) {
        return cudaErrorMemoryAllocation;
    } catch (...) {
        return cudaErrorUnknown;
    }
}

} // namespace gpudb_cuda_detail
