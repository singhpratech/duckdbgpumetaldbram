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
std::unique_ptr<HashJoinProbe> make_cpu_hashjoin_probe();
#if GPUDB_HAVE_CUDA
std::unique_ptr<Aggregator> make_cuda_aggregator();
std::unique_ptr<GroupByAggregator> make_cuda_groupby_aggregator();
bool cuda_runtime_available() noexcept;
#endif
#if GPUDB_HAVE_METAL
std::unique_ptr<Aggregator> make_metal_aggregator();
std::unique_ptr<GroupByAggregator> make_metal_groupby_aggregator();
std::unique_ptr<HashJoinProbe> make_metal_hashjoin_probe();
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

std::unique_ptr<HashJoinProbe> make_hashjoin_probe(Backend b) {
    switch (b) {
        case Backend::CPU:
            return make_cpu_hashjoin_probe();
        case Backend::CUDA:
#if GPUDB_HAVE_CUDA
            // Will be filled in by the Linux Claude when feat/cuda-hashjoin
            // lands on main. The interface and CPU/Metal implementations
            // exist already so the swap is a single TU change.
            throw std::runtime_error(
                "CUDA hash-join not implemented yet — coming on feat/cuda-hashjoin");
#else
            throw std::runtime_error("CUDA backend not compiled in");
#endif
        case Backend::METAL:
#if GPUDB_HAVE_METAL
            return make_metal_hashjoin_probe();
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
