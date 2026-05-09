# Release readiness — DuckDB Community Extensions submission

This document tracks what's needed to ship `gpudb_duckdb` to
[duckdb/community-extensions](https://github.com/duckdb/community-extensions),
the official catalog. It's the Goal-MD-item-9 punch list.

Last validated: 2026-05-09 on Apple M4 Max + macOS 15.x (Darwin 25.4.0).

---

## ✅ What's working today

### Library (libgpudb)
- ✅ Builds clean on macOS with `cmake -S . -B build-macos -DCMAKE_BUILD_TYPE=Release` (no CUDA on Macs)
- ✅ `test_gpudb` 24/24 checks pass on CPU + Metal
- ✅ `gpudb-bench` runs on synthetic + TPC-H SF1 inputs
- ✅ `gpudb-groupby-bench` runs at 1M → 1B rows
- ✅ Apple Silicon GPU path: SUM/MIN/MAX (i64), SUM (f64 host fallback), GROUP BY (radix sort, on PR #5)
- ✅ M4 Max bandwidth utilization: 92% of LPDDR5X peak on SUM kernel

### DuckDB extension (`gpudb_duckdb`)
- ✅ Static `libgpudb_ext.a` builds on macOS
- ✅ Pre-built DuckDB libs auto-fetched via `./scripts/get_duckdb_libs.sh`
- ✅ `gpu_sum`, `gpu_min`, `gpu_max` aggregates registered via DuckDB C API (per Linux Claude's PR #1)

### Benchmarks
- ✅ TPC-H SF1 generated and benched on Metal (`./scripts/gen_tpch.sh`)
- ✅ BENCHMARK.md: comprehensive matrix across rows × cardinality × backends
- ✅ Headline win documented: Metal SUM 1B HOT 5.28× over single-thread CPU, GROUP BY peak 4.89× at 500M × 1M groups, TPC-H GROUP BY ties CUDA wall

### Docs
- ✅ `GOAL.md` — project commitments, ownership table
- ✅ `BENCHMARK.md` — append-only reproducible numbers
- ✅ `docs/BENCHMARK_PLAN.md` — comparison framework (operators × scales × scenarios × backends)
- ✅ `docs/ARCHITECTURE.md`, `docs/DATASETS.md`, `docs/DEVELOPMENT.md`
- ✅ `CLAUDE.md` — coordination rules for parallel Claude Code instances

---

## 🚧 Blockers for DuckDB Community Extensions submission

### 1. macOS extension build is broken (`.dylib` vs `.so` hardcoded)

**File:** `src/extension/CMakeLists.txt:27`

```cmake
target_link_libraries(gpudb_ext PUBLIC
    gpudb
    "${DUCKDB_LIBS_DIR}/libduckdb.so"   # ← hardcoded; macOS has libduckdb.dylib
)
```

**Symptom on macOS:**
```
make[2]: *** No rule to make target '.../third_party/duckdb-libs/libduckdb.so', needed by 'src/extension/libgpudb_duckdb.dylib'.  Stop.
```

**Fix:** use a CMake variable that selects the right suffix:
```cmake
target_link_libraries(gpudb_ext PUBLIC
    gpudb
    "${DUCKDB_LIBS_DIR}/libduckdb${CMAKE_SHARED_LIBRARY_SUFFIX}"
)
```

This is being addressed by `feat/ext-macos-validate` (in flight). Owner: Linux Claude per CLAUDE.md ownership of `src/extension/`.

### 2. CI workflow is disabled

**File:** `.github/workflows/ci.yml.disabled` (per main)

Need to re-enable before publishing. Currently waiting for repo to go public. The `./scripts/local_check.sh` mirrors what CI would run.

### 3. CI doesn't cover macOS

`local_check.sh` hardcodes `build-linux/`. The script needs an `--macos` mode or auto-detect via `uname -s`. Same set of steps, different build dir + extension suffix.

### 4. SQL test coverage

`test/sql/gpu_sum.test` exists for `gpu_sum`. Need similar `.test` files for:
- `gpu_min` ← (Linux Claude added)
- `gpu_max` ← (Linux Claude added)
- NULL handling regression test
- TPC-H Q1 / Q3 SQL (uses gpu_sum on lineitem)

### 5. `description.yml` for DuckDB Community Extensions

Each community extension needs a `description.yml` with:
- name, description, version
- maintainers
- repo URL
- license
- supported platforms

Not yet present in the repo. Template available at https://github.com/duckdb/community-extensions/tree/main/extensions

### 6. Multi-platform CI matrix

DuckDB Community Extensions requires builds for:
- linux_amd64
- linux_arm64
- osx_amd64
- osx_arm64
- windows_amd64

Our local builds cover Linux x86_64 and macOS arm64. Need CI matrix to cover the rest.

---

## 📋 Open PRs that affect release readiness

| PR | Branch | What it adds | Merge order |
|---|---|---|---|
| #5 | `feat/metal-groupby-radix-gpu` | Metal GROUP BY radix sort + GPU scan + min-max (peak 4.89×) | merge first |
| (in flight) | `feat/metal-multiagg` | `agg_all_i64` operator (sum+min+max+count one pass) | parallel agent — under review |
| (in flight) | `feat/ext-macos-validate` | macOS extension build validation + fixes | **MUST land for macOS in submission** |
| (in flight) | `feat/metal-hashjoin-scaffold` | Hash-join interface + CPU + Metal stub | optional for v1 |

---

## ✅ Validation steps (run before each release)

On Linux:
```
./scripts/get_duckdb_libs.sh
./scripts/local_check.sh                     # full sweep, CPU+CUDA
```

On macOS (currently — needs build-dir fix in local_check.sh):
```
export PATH="/Users/aiexplore369/Library/Python/3.9/bin:$PATH"  # cmake from pip
./scripts/get_duckdb_libs.sh                  # downloads libduckdb.dylib
cmake -S . -B build-macos -DCMAKE_BUILD_TYPE=Release -DGPUDB_BUILD_EXT=ON
cmake --build build-macos -j
./build-macos/test/test_gpudb                  # 24/24
./build-macos/bin/gpudb-bench --rows 1000000 --runs 2 --mode both
./build-macos/bin/gpudb-groupby-bench --rows 1000000 --groups 1000 --runs 2
./build-macos/bin/gpudb-sql --sql \
   "SELECT gpu_sum(range::BIGINT) AS gpu, sum(range::BIGINT) AS native FROM range(10);"
```

For TPC-H validation:
```
SF=1 ./scripts/gen_tpch.sh                   # ~1 min
./build-macos/bin/gpudb-bench --input data/tpch_sf1/lineitem_orderkey.gpudb --runs 7
./build-macos/bin/gpudb-groupby-bench --input-keys data/tpch_sf1/lineitem_orderkey.gpudb --runs 5
```

---

## 🎯 Submission timeline

When all items in "Blockers" above are resolved:
1. Push final main branch with version tag (e.g., `v0.1.0`)
2. Generate `description.yml`
3. Open PR to `duckdb/community-extensions` with extension descriptor
4. DuckDB CI builds for all platforms
5. Maintainer review
6. Merged → installable via `INSTALL gpudb FROM community;`

ETA: subject to (a) macOS extension build fix landing, (b) CI matrix being set up, (c) any reviewer feedback from DuckDB maintainers.
