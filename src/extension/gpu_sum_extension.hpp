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
// Each is an overload SET carrying two variants:
//   (BIGINT) -> BIGINT   and   (DOUBLE) -> DOUBLE.
// Smaller integer types (INTEGER/SMALLINT/TINYINT) carry no dedicated overload
// and widen to the BIGINT variant via DuckDB's implicit integer widening.
// v0.3.0 streaming implementation: each aggregate keeps a running accumulator
// (a running sum/min/max plus a non-NULL count) rather than buffering values —
// the same algorithmic shape as a native DuckDB aggregate. Throws
// std::runtime_error if registration fails.
void register_gpu_sum(duckdb_connection con);

} // namespace gpudb_ext
