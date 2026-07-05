# DuckDB C API Headers (vendored)

These are the DuckDB **C API** headers used to build the `gpudb` *loadable*
extension (`libgpudb.<so|dylib>` -> `gpudb.duckdb_extension`).

- `duckdb.h`           — DuckDB C API type/function declarations.
- `duckdb_extension.h` — the **C_STRUCT (stable) ABI** glue. It redirects every
  `duckdb_*` call through a function-pointer struct (`duckdb_ext_api`) that
  DuckDB hands to the extension at load time. Because of this indirection the
  loadable extension links **no** `libduckdb` — there are no undefined
  `duckdb_*` symbols in the shared object, so it loads into any host DuckDB.

They match `TARGET_DUCKDB_VERSION` (the C API version) declared in the repo
root `Makefile` — currently **v1.2.0**, the stable C API baseline. An extension
built against stable C API v1.2.0 loads into any DuckDB >= 1.2.0; the
community-extensions CI builds & tests it against DuckDB v1.5.2.

To refresh to a new `TARGET_DUCKDB_VERSION`, fetch `duckdb.h` and
`duckdb_extension.h` from `duckdb/duckdb` at that tag (the
`make update_duckdb_headers` target in extension-ci-tools automates this for
the template's default `duckdb_capi/` layout). Never use headers from a DuckDB
release *newer* than `TARGET_DUCKDB_VERSION`: the stable-ABI struct only
exposes functions present at that version.
