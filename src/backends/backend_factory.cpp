#include "gpu_backend.hpp"

#include <stdexcept>

namespace gpudb {

const char* to_string(Backend b) noexcept {
    switch (b) {
        case Backend::CPU:   return "CPU";
        case Backend::CUDA:  return "CUDA";
        case Backend::METAL: return "Metal";
    }
    return "?";
}

// Forward declarations (impls live in their respective TUs)
std::unique_ptr<Aggregator> make_cpu_aggregator();
std::unique_ptr<GroupByAggregator> make_cpu_groupby_aggregator();
std::unique_ptr<HashJoinAggregator> make_cpu_hashjoin_aggregator();
#if GPUDB_HAVE_CUDA
std::unique_ptr<Aggregator> make_cuda_aggregator();
std::unique_ptr<GroupByAggregator> make_cuda_groupby_aggregator();
std::unique_ptr<HashJoinAggregator> make_cuda_hashjoin_aggregator();
bool cuda_runtime_available() noexcept;
#endif
#if GPUDB_HAVE_METAL
std::unique_ptr<Aggregator> make_metal_aggregator();
std::unique_ptr<GroupByAggregator> make_metal_groupby_aggregator();
bool metal_runtime_available() noexcept;
#endif

std::unique_ptr<Aggregator> make_aggregator(Backend b) {
    switch (b) {
        case Backend::CPU:
            return make_cpu_aggregator();
        case Backend::CUDA:
#if GPUDB_HAVE_CUDA
            return make_cuda_aggregator();
#else
            throw std::runtime_error("CUDA backend not compiled in");
#endif
        case Backend::METAL:
#if GPUDB_HAVE_METAL
            return make_metal_aggregator();
#else
            throw std::runtime_error("Metal backend not compiled in");
#endif
    }
    throw std::runtime_error("Unknown backend");
}

Backend default_backend() noexcept {
#if GPUDB_HAVE_CUDA
    if (cuda_runtime_available()) return Backend::CUDA;
#endif
#if GPUDB_HAVE_METAL
    if (metal_runtime_available()) return Backend::METAL;
#endif
    return Backend::CPU;
}

std::unique_ptr<GroupByAggregator> make_groupby_aggregator(Backend b) {
    switch (b) {
        case Backend::CPU:
            return make_cpu_groupby_aggregator();
        case Backend::CUDA:
#if GPUDB_HAVE_CUDA
            return make_cuda_groupby_aggregator();
#else
            throw std::runtime_error("CUDA backend not compiled in");
#endif
        case Backend::METAL:
#if GPUDB_HAVE_METAL
            return make_metal_groupby_aggregator();
#else
            throw std::runtime_error("Metal backend not compiled in");
#endif
    }
    throw std::runtime_error("Unknown backend");
}

std::unique_ptr<HashJoinAggregator> make_hashjoin_aggregator(Backend b) {
    switch (b) {
        case Backend::CPU:
            return make_cpu_hashjoin_aggregator();
        case Backend::CUDA:
#if GPUDB_HAVE_CUDA
            return make_cuda_hashjoin_aggregator();
#else
            throw std::runtime_error("CUDA backend not compiled in");
#endif
        case Backend::METAL:
#if GPUDB_HAVE_METAL
            // No Metal hash-join yet — Mac instance owns Metal backend.
            throw std::runtime_error("Metal hash-join not implemented yet");
#else
            throw std::runtime_error("Metal backend not compiled in");
#endif
    }
    throw std::runtime_error("Unknown backend");
}

std::vector<Backend> available_backends() noexcept {
    std::vector<Backend> v{Backend::CPU};
#if GPUDB_HAVE_CUDA
    if (cuda_runtime_available()) v.push_back(Backend::CUDA);
#endif
#if GPUDB_HAVE_METAL
    if (metal_runtime_available()) v.push_back(Backend::METAL);
#endif
    return v;
}

} // namespace gpudb
