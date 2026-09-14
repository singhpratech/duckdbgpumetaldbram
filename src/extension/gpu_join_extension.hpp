#pragma once

#if defined(GPUDB_C_STRUCT_ABI)
#include "duckdb_extension.h"
#else
#include "duckdb.h"
#endif

#include <memory>

namespace gpudb_ext {

class ResidentContext;

// Register gpu_inner_join(build_keys LIST(BIGINT), probe_keys LIST(BIGINT))
// returning (probe_idx BIGINT, build_idx BIGINT), and
// gpu_join_rows_resident(probe VARCHAR, build VARCHAR, kind VARCHAR)
// returning the same shape from resident columns (build_idx NULL for
// LEFT-unmatched / SEMI / ANTI rows).
// `ctx` is the resident registry the functions read (gpu_resident.hpp);
// register_gpu_sum shares one context across all gpu_* families.
void register_gpu_join(duckdb_connection con, const std::shared_ptr<ResidentContext>& ctx);

} // namespace gpudb_ext
