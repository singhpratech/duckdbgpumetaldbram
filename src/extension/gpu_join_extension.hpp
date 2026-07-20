#pragma once

#if defined(GPUDB_C_STRUCT_ABI)
#include "duckdb_extension.h"
#else
#include "duckdb.h"
#endif

namespace gpudb_ext {

// Register gpu_inner_join(build_keys LIST(BIGINT), probe_keys LIST(BIGINT))
// returning (probe_idx BIGINT, build_idx BIGINT).
void register_gpu_join(duckdb_connection con);

} // namespace gpudb_ext
