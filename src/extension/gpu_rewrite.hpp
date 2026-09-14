#pragma once

// gpu_rewrite.hpp — the statement rewriter behind the v0.7 transparent path
// (docs/TRANSPARENT_DESIGN.md §3, §6).
//
//   gpu_rewrite_ast(tree VARCHAR, context VARCHAR) -> VARCHAR
//
// `tree` is what DuckDB's own parser produced (`json_serialize_sql(sql)`);
// `context` is a JSON object the client wrapper built from the catalog and
// the resident registry. The function is PURE — no registry lookup, no
// shared state, no recording — so DuckDB may evaluate it in parallel per
// vector and cache it per template. It returns the rewritten tree (same
// shape `json_deserialize_sql` / `json_execute_serialized_sql` accept) or
// the input unchanged, and always adds a top-level key
//
//   "gpudb": {"rewritten": bool, "reason": "<keyword>", "form": "plain|having|topk", "set": "<tag>"}
//
// which DuckDB's deserializer ignores and the wrapper reads for
// last_rewrite(). Reason keywords: rewritten | shape | not_resident |
// threshold | backend | double | nulls | overflow | decimal | collation |
// too_long | error.
//
// The pure core is exposed so unit tests can drive it without DuckDB.

#if defined(GPUDB_C_STRUCT_ABI)
#include "duckdb_extension.h"
#else
#include "duckdb.h"
#endif

#include <string>

namespace gpudb_ext {

// Pure: rewrite `tree_json` under `context_json`. Never throws; every
// failure is reported through the "gpudb" key (reason "error" with the
// message in "detail").
std::string rewrite_ast(const std::string& tree_json, const std::string& context_json);

void register_gpu_rewrite(duckdb_connection con);

} // namespace gpudb_ext
