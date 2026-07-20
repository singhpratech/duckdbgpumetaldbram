// duckdb_loadable.cpp — DuckDB C API loadable-extension entry point.
//
// Built ONLY into the loadable extension (libgpudb → gpudb.duckdb_extension),
// always with -DGPUDB_C_STRUCT_ABI and -DDUCKDB_EXTENSION_NAME=gpudb.
//
// Uses the stable C_STRUCT ABI via duckdb_extension.h:
//   * DUCKDB_EXTENSION_ENTRYPOINT generates the exported C symbol
//     `gpudb_init_c_api`, defines and populates the global `duckdb_ext_api`
//     function-pointer struct (from access->get_api), opens a connection, and
//     forwards it to the body below. All duckdb_* calls (here and in
//     gpu_sum_extension.cpp) route through that struct, so the shared object
//     links NO libduckdb — there are no undefined duckdb_* symbols.
//   * The extension metadata footer (ABI type = C_STRUCT, duckdb_version =
//     v1.2.0) is appended after the build (by extension-ci-tools in CI, or by
//     scripts/build.sh / the CMake POST_BUILD step locally). No `_version`
//     symbol is needed under the C_STRUCT ABI.
//
// The entrypoint macro requires DUCKDB_EXTENSION_NAME to be defined; CMake sets
// it on this target.

#include "duckdb_extension.h"

#include "gpu_sum_extension.hpp"
#include "gpu_join_extension.hpp"

#include <exception>

DUCKDB_EXTENSION_ENTRYPOINT(duckdb_connection connection,
                            duckdb_extension_info info,
                            struct duckdb_extension_access *access) {
    try {
        // Registers gpu_sum / gpu_min / gpu_max / gpu_inner_join on the auto-opened connection.
        gpudb_ext::register_gpu_sum(connection);
        gpudb_ext::register_gpu_join(connection);
    } catch (const std::exception &e) {
        if (access && access->set_error) {
            access->set_error(info, e.what());
        }
        return false;
    } catch (...) {
        if (access && access->set_error) {
            access->set_error(info, "gpudb extension: unknown error during registration");
        }
        return false;
    }
    return true;
}
