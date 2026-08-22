#pragma once

#if defined(GPUDB_C_STRUCT_ABI)
#include "duckdb_extension.h"
#else
#include "duckdb.h"
#endif

#include <mutex>
#include <string>

namespace gpudb {
class ResidentColumn;
class HybridAggregator;
}

namespace gpudb_ext {

// Register gpu_inner_join(build_keys LIST(BIGINT), probe_keys LIST(BIGINT))
// returning (probe_idx BIGINT, build_idx BIGINT), and
// gpu_join_rows_resident(probe VARCHAR, build VARCHAR, kind VARCHAR)
// returning the same shape from resident columns (build_idx NULL for
// LEFT-unmatched / SEMI / ANTI rows).
void register_gpu_join(duckdb_connection con);

// Bridge into gpu_resident.cpp's registry (implemented there). All access
// must hold resident_mutex(). find_resident_column returns nullptr if the
// name is unknown; the pointer is owned by the registry and is only valid
// while the mutex is held or the column is not dropped.
std::mutex& resident_mutex();
gpudb::ResidentColumn* find_resident_column(const std::string& name);
gpudb::HybridAggregator& resident_aggregator();
void resident_set_last_stats(const std::string& s);

} // namespace gpudb_ext
