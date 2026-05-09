// backend_internal.hpp — internal factory declarations shared between
// backend_factory.cpp and hybrid_planner.cpp.
//
// Not part of the public API; not installed; only TUs inside libgpudb
// should include it.
#pragma once

#include "gpu_backend.hpp"

namespace gpudb {

// CPU factories live in src/backends/cpu/.
std::unique_ptr<Aggregator>        make_cpu_aggregator();
std::unique_ptr<GroupByAggregator> make_cpu_groupby_aggregator();

} // namespace gpudb
