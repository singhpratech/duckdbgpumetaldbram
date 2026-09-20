# Environment variables gpudb honours

Every name below is read by the code — `getenv` in `src/`, `os.environ` in
`python/`, or the shell scripts where that is said. **None of them changes a
query's answer.** They change which code path runs, how much host or device
memory it may use, or what it prints. If you find one that changes an answer,
that is a bug worth an issue.

Anything not on this list is read by nothing in `src/` or `python/`. In
particular `GPUDB_FORCE_BACKEND` is
**not** honoured: it routed machinery deleted in v0.3.0 and setting it does
nothing (`test/sql/gpu_force_backend.test` only pins that it does not crash).

## The Python wrapper and the `gpudb` shell

| Variable | Default | What it does |
|---|---|---|
| `GPUDB_EXTENSION_PATH` | unset | Path to the `.duckdb_extension` to load. Second in the lookup order, after an explicit `gpudb.connect(extension=…)` and before a `build-macos/` / `build-linux/` directory beside a checkout and DuckDB's own installed copy. |
| `GPUDB_MEMORY_BUDGET_MB` | see below | Device memory, in MB, that resident sets may use. `0` removes the cap. An explicit `memory_budget=` argument or `--memory-budget` wins over it. Unset: half of device memory on CUDA, a quarter of host memory on unified memory (Apple silicon), clamped to what the backend reports. |
| `GPUDB_UPLOAD_QUIET_MS` | `400` | The idle stretch a background upload's two uninterruptible steps — the device copy and the sort cache — ask the connection for before they start, on a machine whose cores are contended. `0` turns the back-off off and takes the old path. |
| `GPUDB_UPLOAD_QUIET_MAX_S` | `20` | How long such a step waits for that window. The window asked for decays to the ordinary idle threshold across it, so the step takes the best gap the connection offers and runs unconditionally at the end. This is the bound on how long readiness can be delayed, per step. |
| `GPUDB_RESIDENCY_TRACE` | unset | Anything but empty or `0` prints one stderr line per segment attempt and per quiet wait: how long the session waited for its window, how long the scan then ran, how many cores it got, and whether it landed. |
| `NO_COLOR` | unset | Any non-empty value turns off the shell's colour ([no-color.org](https://no-color.org)). Colour also needs a terminal. |
| `TERM` | — | The shell uses a 256-colour accent when it contains `256color`, and basic cyan otherwise. |

## The extension: host-side caps and tracing

| Variable | Default | What it does |
|---|---|---|
| `GPUDB_UPLOAD_POOL_MAX_MB` | `4096` | Cap, in MB, on the **host** memory in-flight `gpu_upload` states may buffer. It exists because `gpu_upload` inside a window frame grows quadratically. Over the cap you get a clean error, never a truncated set. |
| `GPUDB_EXACT_UPLOAD_POOL_MAX_MB` | the larger of the cap above and half of physical memory | The same cap for the exact uploads the transparent path uses, which the wrapper has already sized against its device budget. |
| `GPUDB_UPLOAD_TRACE` | unset | Set to anything to print each upload's phases (buffer, de-interleave, host-to-device, prepare, publish) on stderr. |
| `GPUDB_GROUPBY_ROWS_MAX_M` | `100` | Millions of group rows a resident `GROUP BY` table function may return. Above it, a clean error. |
| `GPUDB_JOIN_ROWS_MAX_M` | `100` | Millions of row pairs `gpu_join_rows_resident` may emit. Above it, a clean error. |

A bad value for any of the four caps prints one line on stderr naming the
variable and the default it is falling back to, and the run continues.

## The CUDA backend

| Variable | Default | What it does |
|---|---|---|
| `GPUDB_CUDA_EXACT` | **on** | The v0.7 exact operators are on by default on CUDA, which is what lets the transparent path rewrite plain SQL on NVIDIA hardware. Set it to `0` to turn them off: `exact_supported()`, and with it `global_supported()`, `join_supported()` and the placement of exact-upload columns on the device, all go false and every statement stays on DuckDB. Read once per process. The switch exists because a resident column is single-homed, so turning the path off is the only way to put a set back behind the CPU reference without a rebuild. |

## The Metal backend: path selection and sweeps

These pick between algorithms that produce the same rows. They exist so the
measurements in [BENCHMARK.md](../BENCHMARK.md) can be reproduced and so the
fallback paths can be exercised on hardware where the fast one works.

| Variable | Default | What it does |
|---|---|---|
| `GPUDB_METAL_GROUPBY_EXACT_PATH` | `auto` | `direct \| sort \| auto` — which algorithm answers an exact `GROUP BY`. `direct` still falls back to the sort path for calls it cannot express. |
| `GPUDB_METAL_DIRECT_KERNEL` | `auto` | `thread \| slab \| auto` — the accumulator shape the direct grouped reduce uses. |
| `GPUDB_METAL_DIRECT_MIN_WORK` | `6000000` | Rows × (payloads + `WHERE` terms) the direct pass must exceed before building its group-id lane pays for itself. |
| `GPUDB_METAL_DIRECT_MIN_GROUPS` | `3` | Fewest groups the direct pass is ever taken at. |
| `GPUDB_METAL_DIRECT_MAX_GROUPS` | `512` | Measured crossover above which the sort path wins. Clamped to a compiled-in ceiling. |
| `GPUDB_METAL_DIRECT_THREAD_SLOTS` | `0` | At or below this many (group × payload) accumulator slots, the thread-private kernel is used instead of the slab. |
| `GPUDB_METAL_DIRECT_DISABLE_PSO` | unset | Makes chosen direct pipelines refuse to build, so the fallback can be tested. `1` / `all`, or one of `slab`, `masked32`, `masked8`, `ids`, `fill`, `merge`, or a literal Metal function name. |
| `GPUDB_METAL_MASK_PATH` | `auto` | `fused \| legacy \| permeval \| auto` — which `WHERE` pass the sort path runs. Re-read on every call, so one process can interleave the shapes for a measurement. |
| `GPUDB_METAL_MASK_DISABLE_PSO` | unset | Makes the fused mask pipelines refuse to build, to exercise the fallback to the legacy pass. |
| `GPUDB_METAL_MASK_COMPACT_BELOW` | `0.5` | Surviving fraction below which a masked reduce compacts before reducing instead of reducing in place. |
| `GPUDB_METAL_HOST_FILTER_BELOW` | `65536` | Group count at or below which top-k runs on the host rather than the device (`0` = always on the device). `HAVING` stays on the device at every size. |
| `GPUDB_METAL_PREWARM` | on | `0` makes `prepare()` build only the sort cache, so the first-call gap prewarming removes can be measured. |
| `GPUDB_METAL_GROUPBY_PATH` | `auto` | `slotlock \| radix` for the standalone v0.6 `GROUP BY` operator. |
| `GPUDB_METAL_HASHJOIN_PATH` | `auto` | `hash \| merge \| partition_scan \| partition` for the hash join. Note: setting it to anything unrecognised selects the partitioned path rather than restoring `auto`. |
| `GPUDB_METAL_TRACE_EXACT` | unset | Per-stage GPU times for the exact path on stderr, plus one line per resident-column rebuild. |
| `GPUDB_METAL_UPLOAD_REFUSE_MB` | unset | A test hook: the backend refuses any exact upload larger than this many MiB, as a full device would. It is the only way to reach the device-refusal paths on a machine with memory to spare, and it is what `scripts/budget_gate.py --refuse-mb` sets for its child process. Read once at load; unset, the branch costs one `getenv` per upload and changes nothing. |

## Build and test scripts only

Read by the shell scripts and CMake, never by the extension at run time.

| Variable | Default | What it does |
|---|---|---|
| `BUILD_DIR` | `build-macos` / `build-linux` | Build directory for `scripts/build.sh` and `scripts/run_sql_tests.sh`. |
| `GPUDB_NO_CUDA` | unset | `1` configures with `-DGPUDB_ENABLE_CUDA=OFF`. |
| `GPUDB_REQUIRE_CUDA` | unset | `1` makes configure fail loudly instead of silently falling back to CPU. |
| `GPUDB_NO_METAL` | unset | `1` configures with `-DGPUDB_ENABLE_METAL=OFF`. |
| `DUCKDB_C_API_VERSION` | `v1.2.0` | The C API version the loadable extension is packaged against. |
| `DUCKDB_VERSION` | `v1.5.5` | Which DuckDB release `scripts/get_duckdb_libs.sh` fetches (`latest` is accepted). |
| `CUDAARCHS` | `75;80;86;89;90` | Standard CMake variable; when unset **or set but empty** the project falls back to that list. |
| `GPUDB_SQL_TIMEOUT_SECS` | `30` | Per-query timeout in the SQL suite. |
| `BUILD_TYPE` | `Release` | `CMAKE_BUILD_TYPE` for `scripts/build.sh`. |
| `EXT_VERSION` | the project version | The version stamped into the packaged extension's metadata footer. |
| `SF` | `1` | TPC-H scale factor for `scripts/gen_tpch.sh`. |
| `DUCKDB` | `./.tools/duckdb` | Which DuckDB CLI `scripts/gen_tpch.sh` generates the data with. |
| `FORCE` | `0` | `1` makes `scripts/get_duckdb_libs.sh` re-fetch over an existing `third_party/duckdb-libs/`. |
| `RUNTIME_BACKEND` | read from `gpu_build_info()` | Which backend `scripts/run_sql_tests.sh` reports and gates backend-specific cases on; falls back to `cpu`. |

`GPUDB_CUDA_STATIC_RUNTIME` is a CMake option, not an environment variable:
pass `-DGPUDB_CUDA_STATIC_RUNTIME=ON` (which is what the root `Makefile`
does).
