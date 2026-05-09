// duckdb_loadable.cpp — DuckDB C API loadable-extension entry point.
//
// Status: the entry point function compiles and is exported from the .so,
// BUT direct `LOAD '...'` from the DuckDB CLI requires a metadata footer
// (296 bytes: signature + length + 8x32-byte fields) appended to the .so.
// That packaging is normally done by the official DuckDB extension-template
// build pipeline. We ship the entry point + .duckdb_extension alias so it's
// ready for that template; the next PR will add the footer-append step.
//
// In the meantime, the working demo is the `gpudb-sql` CLI (benchmark/
// sql_demo_main.cpp) which embeds DuckDB in-process and registers the
// same functions via gpudb_ext::register_gpu_sum. The PUBLIC-facing way
// (INSTALL FROM community + LOAD) lights up once the metadata footer is
// added and we submit to https://github.com/duckdb/community-extensions.

#include "gpu_sum_extension.hpp"
#include "duckdb.h"

#include <cstdio>
#include <exception>

extern "C" {

// Standard C API extension entry. DuckDB calls this once on LOAD.
DUCKDB_EXTENSION_API bool gpudb_duckdb_init_c_api(
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
DUCKDB_EXTENSION_API const char *gpudb_duckdb_version() {
    return duckdb_library_version();
}

} // extern "C"
