# macOS build validation report — DuckDB extension (`feat/ext-macos-validate`)

Validation pass for the `gpu_sum` / `gpu_min` / `gpu_max` DuckDB extension on
Apple Silicon. Performed against:

- `origin/main` @ `2e8f56d` (which contains the merged extension commit
  `545cc67  feat(ext): SQL→GPU through DuckDB — gpu_sum() aggregate`)
- `origin/feat/ext-duckdb-stub` @ `91ff525` (latest extension iteration with
  `gpu_min` / `gpu_max`, NULL handling, C-API loadable entry, footer-append
  script)

Test machine: MacBook Pro, Apple M4 Max, macOS 26.4.1 (Darwin 25.4.0 arm64),
Apple clang 21.0.0, CMake 4.3.2, DuckDB CLI v1.5.2 (Variegata), DuckDB
pre-built libs from upstream `libduckdb-osx-universal.zip` (universal2,
ships `libduckdb.dylib`).

## Two build entrypoints

There are now **two independent ways** to build this project, for two different
purposes. They do not interfere with each other:

1. **`./scripts/build.sh`** — the local developer flow. Builds everything: the
   `gpudb` core lib, unit tests (`test_gpudb`), benchmarks, the embedded
   `gpudb-sql` CLI, and the loadable extension. Requires `third_party/duckdb-libs/`
   (run `./scripts/get_duckdb_libs.sh` first) for the `gpudb-sql` path; the
   loadable extension itself does not. Custom SQL suite: `./scripts/run_sql_tests.sh`.

2. **`make release`** (root `Makefile`) — the **community-extensions CI** flow.
   Mirrors `duckdb/extension-template-c`: includes the `extension-ci-tools`
   `c_api_extensions` makefiles and builds ONLY the loadable extension via the
   stable **C_STRUCT ABI** (no `libduckdb` link, no DuckDB submodule). Targets:
   `make set_duckdb_version` (no-op) → `make configure_ci` → `make release` →
   `make test_release` (SQLLogicTest suite in `test/sqllogic/`). The final
   artifact lands at `build/release/gpudb.duckdb_extension` (staged copy at
   `build/release/extension/gpudb/gpudb.duckdb_extension`). CUDA is off in this
   path (phase 2); Metal stays auto-on for the `osx_arm64` job.

The vendored DuckDB C API headers used by path 2 live in
`third_party/duckdb_capi/` (committed); `extension-ci-tools` is a git submodule.

## TL;DR — status matrix

| Path                                                         | Status on macOS  |
|--------------------------------------------------------------|------------------|
| `cmake -DGPUDB_BUILD_EXT=ON` configure                       | works            |
| Build static `gpudb_ext` + executable `gpudb-sql`            | works (after one-line fix below) |
| Build loadable `libgpudb_duckdb.dylib` + `.duckdb_extension` alias | works (after the same fix) |
| In-process `gpu_sum`/`gpu_min`/`gpu_max` via `gpudb-sql` (Metal backend) | works |
| GROUP BY w/ `gpu_sum` (in-process)                           | works            |
| NULL-skipping (validity bitmap)                              | works            |
| `LOAD '<path>.duckdb_extension'` from DuckDB CLI             | fails, same as Linux (metadata-footer reject) |

The extension is functionally correct on Apple Silicon when consumed
in-process (the `gpudb-sql` demo path the Linux Claude has been using).
The CLI `LOAD` path is broken on **both** platforms — not a macOS-specific
gap. It's blocked behind the documented DuckDB extension-template
integration listed in `scripts/append_extension_footer.py`.

## Issues found

### 1. Hardcoded `libduckdb.so` in `src/extension/CMakeLists.txt:27`

The static `gpudb_ext` target unconditionally links
`${DUCKDB_LIBS_DIR}/libduckdb.so`:

```cmake
target_link_libraries(gpudb_ext PUBLIC
    gpudb
    "${DUCKDB_LIBS_DIR}/libduckdb.so"
)
```

On macOS the DuckDB universal release ships `libduckdb.dylib` (verified
from `libduckdb-osx-universal.zip`), so the link command produces:

```
make[2]: *** No rule to make target
'/.../third_party/duckdb-libs/libduckdb.so',
needed by 'src/extension/libgpudb_duckdb.dylib'.  Stop.
```

This breaks both `libgpudb_duckdb` (the loadable) and `gpudb-sql` (which
transitively links `gpudb_ext`).

**Suggested fix (one line, applied on this branch):**

```cmake
target_link_libraries(gpudb_ext PUBLIC
    gpudb
    "${DUCKDB_LIBS_DIR}/libduckdb${CMAKE_SHARED_LIBRARY_SUFFIX}"
)
```

`CMAKE_SHARED_LIBRARY_SUFFIX` resolves to `.so` on Linux and `.dylib` on
macOS, matching what `scripts/get_duckdb_libs.sh` already extracts on each
platform (the script handles both suffixes correctly — only the CMake
side was Linux-specific). Same fix is needed on `feat/ext-duckdb-stub`.

After this change:

```
$ cmake --build build-macos -j
...
[ 95%] Built target gpudb-sql
[100%] Built target gpudb_duckdb
```

Both `build-macos/src/extension/libgpudb_duckdb.dylib` and the
`gpudb_duckdb.duckdb_extension` alias build successfully.

### 2. `LOAD '...duckdb_extension'` rejected by DuckDB CLI (NOT macOS-specific)

```
$ ./.tools/duckdb -unsigned -c "LOAD '.../gpudb_duckdb.duckdb_extension'; ..."
Invalid Input Error: Failed to load '...', The file is not a DuckDB
extension. The metadata at the end of the file is invalid
```

Without a metadata footer the bare `.dylib` is also refused
(`DuckDB extensions are files ending with '.duckdb_extension'`), and the
manual footer produced by `scripts/append_extension_footer.py` from
`feat/ext-duckdb-stub` is likewise rejected — the script's own header
documents this:

> STATUS (2026-05-09): the manual format below is REJECTED by DuckDB
> 1.4.x ("The metadata at the end of the file is invalid"). The exact
> field semantics have shifted in recent DuckDB versions and reverse-
> engineering them is fragile.

I confirmed the same rejection on macOS against DuckDB CLI v1.5.2
using `--platform osx_arm64`, so this is **not** a macOS-specific
shortcoming — it's the documented limit on both platforms. The two
working paths the script itself recommends:

1. Submit to <https://github.com/duckdb/community-extensions> (their CI
   builds correctly footered + signed `.duckdb_extension` files for all
   platforms; users then `INSTALL gpudb FROM community; LOAD gpudb;`).
2. Adopt the official extension-template
   (<https://github.com/duckdb/extension-template>) and let its CMake
   macros generate the footer; users then `SET
   allow_unsigned_extensions=true; LOAD '...';`.

### 3. Comment / variable-name drift now that the artifact is `.dylib`

Cosmetic only: `src/extension/CMakeLists.txt:7,30` and the header
comment of `duckdb_loadable.cpp` say "loadable .so" / "Loadable .so". On
macOS the artifact is `.dylib`. Worth a one-line refresh next time the
file is touched, but not a build blocker. Not modifying here per
ownership rules.

## What works on Apple Silicon today (validation evidence)

```bash
# After the one-line fix in src/extension/CMakeLists.txt
$ ./scripts/get_duckdb_libs.sh
$ cmake -S . -B build-macos -DGPUDB_BUILD_EXT=ON
$ cmake --build build-macos -j
```

Tests (24/24 pass, both CPU and Metal backends):

```
$ ./build-macos/test/test_gpudb
gpudb test suite
available backends: CPU Metal
default backend: Metal
...
24 / 24 checks passed
```

Ungrouped `gpu_sum` (Metal backend, dispatched via real DuckDB
aggregate-function registration through the C API):

```
$ ./build-macos/bin/gpudb-sql --sql \
  "SELECT gpu_sum(range::BIGINT) AS gpu_total, sum(range::BIGINT) AS native_total
   FROM range(1000000);"
[gpudb] registered gpu_sum / gpu_min / gpu_max  (backend=Metal)
gpu_total       native_total
499999500000    499999500000
```

`gpu_min` / `gpu_max` on 10M rows:

```
$ ./build-macos/bin/gpudb-sql --sql \
  "SELECT gpu_sum(range::BIGINT), gpu_min(range::BIGINT), gpu_max(range::BIGINT)
   FROM range(10000000);"
gpu_sum(...)    gpu_min(...)    gpu_max(...)
49999995000000  0               9999999
```

GROUP BY (5 groups, 100 rows — exercises the per-state finalize path):

```
$ ./build-macos/bin/gpudb-sql --sql \
  "CREATE TABLE t AS SELECT range%5 AS g, range::BIGINT AS v FROM range(100);
   SELECT g, gpu_sum(v), sum(v) FROM t GROUP BY g ORDER BY g;"
g  gpu_sum(v)  sum(v)
0  950         950
1  970         970
2  990         990
3  1010        1010
4  1030        1030
```

NULL handling (validity bitmap, `feat/ext-duckdb-stub` commit `31537e2`):

```
$ ./build-macos/bin/gpudb-sql --sql \
  "SELECT gpu_sum(x) FROM
   (VALUES (1::BIGINT),(NULL::BIGINT),(3::BIGINT),(NULL::BIGINT),(4::BIGINT))
   AS t(x);"
gpu_sum(x)
8
```

(NULLs skipped → `1+3+4=8`, matches DuckDB SQL semantics for `SUM`.)

Mach-O sanity check on the loadable:

```
$ file build-macos/src/extension/libgpudb_duckdb.dylib
build-macos/src/extension/libgpudb_duckdb.dylib: Mach-O 64-bit dynamically
  linked shared library arm64
$ otool -L build-macos/src/extension/libgpudb_duckdb.dylib
  ... Foundation.framework ...
  ... Metal.framework ...
  ... MetalPerformanceShaders.framework ...
  @rpath/libduckdb.dylib
  /usr/lib/libc++.1.dylib
```

Clean arm64 dylib with the expected Metal stack and DuckDB rpath link.

## Repro commands for future macOS validators

```bash
# 1. Pull and build
git checkout main && git pull --rebase
./scripts/get_duckdb_libs.sh                # ~140 MB extracted to third_party/
cmake -S . -B build-macos -DGPUDB_BUILD_EXT=ON
cmake --build build-macos -j

# 2. Unit tests
./build-macos/test/test_gpudb

# 3. End-to-end SQL via in-process registration
./build-macos/bin/gpudb-sql --sql \
  "SELECT gpu_sum(range::BIGINT) FROM range(1000000);"

# 4. (When the upstream LOAD path is fixed via extension-template
#     or community-extensions submission)
#    EXT=$PWD/build-macos/src/extension/gpudb_duckdb.duckdb_extension
#    ./.tools/duckdb -unsigned -c "LOAD '$EXT'; SELECT gpu_sum(...) ..."
```

## Suggested follow-ups for the Linux Claude

1. Apply the `CMAKE_SHARED_LIBRARY_SUFFIX` fix to
   `src/extension/CMakeLists.txt` on `feat/ext-duckdb-stub` so it stays
   in sync with main once both branches converge. (Already applied to
   main via this PR.)
2. When integrating the official DuckDB extension-template (the path
   `scripts/append_extension_footer.py` recommends), build `.duckdb_extension`
   artifacts for both `linux_amd64` and `osx_arm64` so the
   community-extensions submission ships both platforms day one — the
   Apple-Silicon angle is the project's stated unique differentiator
   (`GOAL.md:18`).
3. A small refresh of the `.so` wording in `src/extension/CMakeLists.txt`
   and the `duckdb_loadable.cpp` header comment to read "loadable
   shared library" or "loadable .so/.dylib" would prevent future
   readers from assuming Linux-only.
