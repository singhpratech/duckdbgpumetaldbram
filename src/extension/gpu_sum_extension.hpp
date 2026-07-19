#pragma once

// gpu_sum_extension.hpp — public registration entrypoint, shared by the two
// build paths:
//
//   * Loadable extension (libgpudb → gpudb.duckdb_extension): compiled with
//     -DGPUDB_C_STRUCT_ABI. Pulls in duckdb_extension.h, which redirects every
//     duckdb_* call through the C_STRUCT function-pointer struct — so the
//     shared object links NO libduckdb.
//   * Embedded CLI / benchmark (gpudb_ext static lib + gpudb-sql): compiled
//     WITHOUT that define. Uses the real duckdb.h and links a real libduckdb.
#if defined(GPUDB_C_STRUCT_ABI)
#include "duckdb_extension.h"
#else
#include "duckdb.h"
#endif

namespace gpudb_ext {

// Register gpu_sum / gpu_min / gpu_max on the given open DuckDB connection.
//   gpu_sum: overload set — (BIGINT) -> BIGINT and (DOUBLE) -> DOUBLE.
//   gpu_min / gpu_max: (BIGINT) -> BIGINT only (v0.2.0; DOUBLE deferred).
// Smaller integer types (INTEGER/SMALLINT/TINYINT) resolve to the BIGINT
// overloads via DuckDB's implicit integer widening. Throws std::runtime_error
// if registration fails.
void register_gpu_sum(duckdb_connection con);

} // namespace gpudb_ext
