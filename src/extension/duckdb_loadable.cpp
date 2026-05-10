// duckdb_loadable.cpp — DuckDB C API loadable-extension entry point.
//
// LOADABLE: yes. The .duckdb_extension file is produced by appending the
// 528-byte metadata footer (scripts/build_helpers/append_extension_metadata.py)
// to the libgpudb_duckdb.so. Resulting file works with `LOAD '/path/...'`
// from the DuckDB CLI when launched with `-unsigned`.
//
// DuckDB looks up two C symbols by extension-name convention:
//   <name>_init_c_api   — registers the extension's functions
//   <name>_version      — DuckDB-version compatibility stamp
// The extension name in description.yml is `gpudb`, so the symbols are
// gpudb_init_c_api and gpudb_version (NOT gpudb_duckdb_*).

#include "gpu_sum_extension.hpp"
#include "duckdb.h"

#include <cstdio>
#include <exception>

extern "C" {

// Standard C API extension entry. DuckDB calls this once on LOAD.
DUCKDB_EXTENSION_API bool gpudb_init_c_api(
        duckdb_extension_info info,
        const struct duckdb_extension_access *access) {
    if (!access || !access->get_database) {
        return false;
    }
    duckdb_database *db_ptr = access->get_database(info);
    if (!db_ptr || !*db_ptr) {
        if (access->set_error) {
            access->set_error(info, "gpudb extension: failed to retrieve database handle");
        }
        return false;
    }
    duckdb_connection con;
    if (duckdb_connect(*db_ptr, &con) == DuckDBError) {
        if (access->set_error) {
            access->set_error(info, "gpudb extension: duckdb_connect failed");
        }
        return false;
    }
    try {
        gpudb_ext::register_gpu_sum(con);   // registers gpu_sum, gpu_min, gpu_max
    } catch (const std::exception &e) {
        if (access->set_error) {
            access->set_error(info, e.what());
        }
        duckdb_disconnect(&con);
        return false;
    }
    duckdb_disconnect(&con);
    return true;
}

// Required version stamp for unsigned-LOAD compatibility checks.
DUCKDB_EXTENSION_API const char *gpudb_version() {
    return duckdb_library_version();
}

} // extern "C"
