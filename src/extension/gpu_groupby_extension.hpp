#pragma once

#if defined(GPUDB_C_STRUCT_ABI)
#include "duckdb_extension.h"
#else
#include "duckdb.h"
#endif

#include <memory>

namespace gpudb_ext {

class ResidentContext;

// Register the v0.6 resident GROUP BY / top-k table functions:
//   gpu_groupby_sum_resident(name)      -> (key BIGINT, sum BIGINT, count BIGINT)
//   gpu_groupby_sum_resident_f64(name)  -> (key BIGINT, sum DOUBLE, count BIGINT)
//   gpu_groupby_count_resident(name)    -> (key BIGINT, count BIGINT)
//   gpu_topk_resident(name, k, order)   -> (idx BIGINT, value BIGINT)
//   gpu_topk_resident_f64(name, k, order) -> (idx BIGINT, value DOUBLE)
// `name` is a gpu_upload_pair name ('p' resolves p.k / p.v) or, for the
// count / top-k functions, a plain gpu_upload column name. Rows come out
// sorted by key ascending (group by) or in the requested order (top-k).
// Called by register_gpu_sum. Implementation: gpu_groupby_extension.cpp.
void register_gpu_groupby(duckdb_connection con, const std::shared_ptr<ResidentContext>& ctx);

} // namespace gpudb_ext
