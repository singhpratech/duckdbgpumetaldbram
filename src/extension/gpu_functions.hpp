#pragma once

#if defined(GPUDB_C_STRUCT_ABI)
#include "duckdb_extension.h"
#else
#include "duckdb.h"
#endif

#include <string>
#include <vector>

namespace gpudb_ext {

// One overload of one gpudb SQL function, as gpu_functions() reports it.
// The columns line up with duckdb_functions(): function_type is spelled
// 'scalar' | 'aggregate' | 'table', parameter_types are DuckDB's own type
// names ('BIGINT[]' for a list), and return_type is empty (reported as NULL)
// for a table function.
struct FunctionDoc {
    std::string name;
    std::string type;
    std::vector<std::string> parameters;
    std::vector<std::string> parameter_types;
    std::string return_type;
    std::string description;
    std::string example;
    std::vector<std::string> categories;
};

// The documentation table: one entry per registered overload. This is the
// single place a gpudb function is described. test/sql/gpu_functions.test
// fails when the set of (name, type, parameter types, return type) here
// differs from what duckdb_functions() reports for the gpu_ functions, and
// scripts/gpu_functions_check.py runs every example.
const std::vector<FunctionDoc>& function_docs();

// Registers the gpu_functions() table function, which returns function_docs().
void register_gpu_functions(duckdb_connection con);

} // namespace gpudb_ext
