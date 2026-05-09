#pragma once

#include "duckdb.h"

namespace gpudb_ext {

// Register gpu_sum(BIGINT) -> BIGINT on the given open DuckDB connection.
// Throws std::runtime_error if registration fails.
void register_gpu_sum(duckdb_connection con);

} // namespace gpudb_ext
