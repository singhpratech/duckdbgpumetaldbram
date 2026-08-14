# Instructions for Claude Code working in this repo

This repository is developed by **multiple Claude Code instances in parallel**. Two flavors of parallelism are supported:

1. **Cross-machine**: a Linux instance (NVIDIA RTX 4090, CUDA backend) and a macOS instance (Apple Silicon, Metal backend), each on its own clone.
2. **Same-machine**: multiple Claude Code instances on ONE box, each running in a separate **git worktree** with its own branch.

Treat these instructions as load-bearing.

## At the START of every conversation

```bash
./scripts/sync.sh check          # see what every other instance is working on
./scripts/sync.sh whoami         # confirm your instance name (= worktree dir name)
git fetch origin && git status   # confirm you're not stale
```

Then, before doing real work:
```bash
./scripts/sync.sh status "<one sentence: what you are about to do>"
```

When you finish a unit of work or get blocked:
```bash
./scripts/sync.sh status --done
# or
./scripts/sync.sh status --blocked-on <peer-name> "waiting on X"
```

## Same-machine parallelism via worktrees

The Linux box can run multiple Claude Code instances simultaneously without conflict if each instance is launched from its own git worktree:

```bash
./scripts/new_worktree.sh feat/cuda-groupby      # creates ../worktrees/feat-cuda-groupby
cd ../worktrees/feat-cuda-groupby
# launch a fresh Claude Code instance from THIS directory
```

Each worktree:
- Has its own working tree and its own checked-out branch
- Shares the underlying `.git` (so commits from one are visible to another after `git fetch`/`git pull`)
- Has its own `build-linux/` directory (gitignored) — no build conflicts
- Is identified to `.sync/` by its directory basename (so `worktrees/feat-cuda-groupby/` writes to `.sync/feat-cuda-groupby.md`)

To merge work between worktrees: push a feature branch, the other instance pulls it.

## First: detect which machine you are on

```bash
uname -s   # Linux | Darwin
```

Your platform determines which directories you may modify:

| Path | Linux Claude | macOS Claude |
|---|---|---|
| `src/backends/cuda/**`, `src/include/cuda/**` | OWN | DO NOT MODIFY |
| `src/backends/metal/**`, `src/include/metal/**` | DO NOT MODIFY | OWN |
| `src/backends/cpu/**` | shared (PR + review) | shared (PR + review) |
| `src/include/gpu_backend.hpp` (the abstract interface) | shared | shared |
| `src/operators/**` | shared | shared |
| `src/extension/**` (DuckDB wrapper) | shared | shared |
| `CMakeLists.txt`, `src/CMakeLists.txt` | shared | shared |
| `test/cpp/**`, `benchmark/**` | shared | shared |
| `docs/**`, `README.md` | shared | shared |
| `.github/workflows/**` | shared | shared |

"Shared" means: only modify on a feature branch, push, open a PR. Never push directly to `main`.

## Branching protocol

Branch names use these prefixes — they tell the other instance what platform a branch belongs to:
- `feat/cuda-*`     — Linux Claude only
- `feat/metal-*`    — macOS Claude only
- `feat/core-*`     — either, but coordinate via PR
- `feat/ext-*`      — DuckDB extension work, either
- `chore/*`, `docs/*`, `ci/*` — either

**Always** start by:
```bash
git fetch origin
git pull --rebase origin main
git checkout -b feat/<your-platform>-<short-name>
```

**Always** finish by:
```bash
git push -u origin <branch>
gh pr create --fill --base main
```

Do not commit directly to `main`. Do not force-push to `main`. Do not merge your own PR without the user explicitly approving.

## Per-platform rules

### If `uname -s` returns `Linux` (CUDA dev box, RTX 4090)
- You may write `.cu`, `.cuh` files, run `nvcc`, run CUDA programs.
- You **must not** write `.metal`, `.mm` files, attempt `xcodebuild`, or modify `src/backends/metal/`.
- If `nvcc` is missing, ask the user to install `nvidia-cuda-toolkit` (sudo); do not attempt sudo yourself.
- Build directory: `build-linux/` (gitignored).

### If `uname -s` returns `Darwin` (Apple Silicon dev box)
- You may write `.metal`, `.mm` files, use `xcrun metal`, MPS, MPSGraph, MLX.
- You **must not** write `.cu`/`.cuh`, attempt `nvcc`, or modify `src/backends/cuda/`.
- Build directory: `build-macos/` (gitignored).

## Build, test, benchmark

```bash
./scripts/build.sh                   # configure + build into build-linux/ (or build-macos/);
                                     # auto-enables the DuckDB extension if third_party/duckdb-libs/
                                     # is present, and packages gpudb.<platform>.duckdb_extension
./scripts/local_check.sh             # everything CI runs: configure → build → unit tests →
                                     # smoke benchmarks → SQL suite. Run BEFORE pushing.
./scripts/local_check.sh --no-bench  # fast variant (skip benchmarks)

./build-linux/test/test_gpudb        # unit tests (CPU + CUDA backends; single binary, no per-test filter)
./scripts/run_sql_tests.sh           # full SQL suite (test/sql/*.test)
./scripts/run_sql_tests.sh gpu_sum   # single SQL test file by basename
```

- Unit tests are one hand-rolled binary (`test/cpp/test_aggregator.cpp`) — no gtest, no filter flag. To run "one test", use the SQL suite's per-file selection instead.
- `test/sql/*.test` format: plain SQL separated by `;`, with `-- expect:` lines after each query; other directives are `-- env: K=V`, `-- requires_file: <path>`, `-- expected_fail: <reason>`. See the header of `scripts/run_sql_tests.sh`. `test/sqllogic/` is a SEPARATE suite in DuckDB sqllogictest format (incompatible with `test/sql/`), run only by the community-CI `make test` path.
- Embedded-CLI/extension prerequisite: `./scripts/get_duckdb_libs.sh` fetches pre-built libduckdb + headers into `third_party/duckdb-libs/` (that's what flips `build.sh` into building the extension and `gpudb-sql`).
- TPC-H data for benchmarks/SQL tests: `SF=1 ./scripts/gen_tpch.sh` → `data/tpch_sf1/`.
- On Linux, `scripts/env.sh` puts `/usr/local/cuda/bin` on PATH; `build.sh` and `local_check.sh` source it automatically.
- Benchmark CLIs land in `build-*/bin/`: `gpudb-sql` (embedded DuckDB + gpu_* functions), `gpudb-bench`, `gpudb-groupby-bench`, `gpudb-window-bench`, `gpudb-hashjoin-bench`.

### The second build path: root Makefile (community-extensions CI)

The root `Makefile` is the entrypoint the DuckDB community-extensions pipeline drives (`make configure && make release`, sqllogictests via `make test`). It builds ONLY the loadable `gpudb.duckdb_extension` against the stable C_STRUCT ABI — no libduckdb link, no `duckdb-libs` needed — and requires the `extension-ci-tools` submodule (`git submodule update --init`). It is fully independent from `./scripts/build.sh`; day-to-day development uses the scripts, not the Makefile.

## CMake flags reference
- `-DGPUDB_ENABLE_CUDA=ON/OFF`  (default ON; auto-disabled if no nvcc)
- `-DGPUDB_ENABLE_METAL=ON/OFF` (default ON on macOS; ignored on Linux)
- `-DGPUDB_BUILD_TESTS=ON/OFF`  (default ON)
- `-DGPUDB_BUILD_BENCH=ON/OFF`  (default ON)
- `-DGPUDB_BUILD_EXT=ON/OFF`    (default OFF, but `build.sh` auto-enables it when `third_party/duckdb-libs/duckdb.h` exists)
- `-DCMAKE_CUDA_ARCHITECTURES`  (default `75;80;86;89;90`; RTX 4090 = sm_89)

## Architecture

Everything hangs off one abstract interface: **`src/include/gpu_backend.hpp`**. It defines per-operator interfaces (`Aggregator` for SUM/MIN/MAX + fused `agg_all`, `GroupByAggregator`, `WindowAggregator`, `HashJoinProbe`), a `ResidentColumn` opaque handle (upload once, query many times without re-transfer), result structs carrying timing diagnostics (`wall_ms` / `kernel_ms` / `transfer_ms`), and `make_*` factories keyed on `Backend {CPU, CUDA, METAL}`. This file is the cross-platform ABI — changes must be coordinated between the Linux and macOS instances via PR.

Layers, top to bottom:

1. **`src/extension/`** — DuckDB integration, with TWO independent targets (see the comment atop `src/extension/CMakeLists.txt`):
   - *Loadable extension* (`gpudb_duckdb`): what users `LOAD` in the DuckDB CLI. Registers `gpu_sum`/`gpu_min`/`gpu_max` (BIGINT + DOUBLE) as streaming aggregates. Built against ONLY the vendored C API headers in `third_party/duckdb_capi/` (stable C_STRUCT ABI) — links no libduckdb.
   - *Embedded CLI helper* (`gpudb_ext` + `gpudb-sql`): genuinely links libduckdb from `third_party/duckdb-libs/`; silently skipped if those libs are absent.
   - Note: `GPUDB_FORCE_BACKEND` is NOT currently honored — `test/sql/gpu_force_backend.test` only pins that setting it doesn't crash. The one real env override is `GPUDB_METAL_GROUPBY_PATH` (Metal GROUP BY algorithm selection).
2. **`src/backends/backend_factory.cpp`** — runtime dispatch: `default_backend()` picks CUDA if compiled + device present, else Metal, else CPU.
3. **`src/backends/hybrid_planner.cpp`** — `HybridAggregator` / `HybridGroupByAggregator` wrap a CPU and a GPU implementation and pick per call using deterministic thresholds (row count, expected groups, residency) derived from BENCHMARK.md; each call records a `DispatchDecision` (inspectable via `last_decision()`) explaining why.
4. **Backend implementations** — `src/backends/{cpu,cuda,metal}/`, one file per operator (`*_aggregator`, `*_groupby`, `*_hashjoin`, `*_window`). CUDA kernels in `cuda/kernels/*.cu`; Metal shaders in `metal/kernels/*.metal` hosted by `.mm` files. Backends differ algorithmically where hardware demands it (e.g. Metal lacks 64-bit atomic CAS, so its hash join is sort-merge while CUDA's is open-addressing atomicCAS; Metal GROUP BY auto-dispatches slot-lock vs radix-sort paths).

The DuckDB dependency is deliberately NOT a submodule of DuckDB source: the loadable extension uses committed vendored headers (`third_party/duckdb_capi/`), and the embedded CLI uses pre-built binaries fetched by script (`third_party/duckdb-libs/`, gitignored).

## Conflict-avoidance checklist before every commit
1. `./scripts/sync.sh check` — confirm no peer is mid-edit on the files you touched.
2. `git fetch origin && git status` — confirm you're not behind `main`.
3. Your changed files match your platform's allowed paths in the table above.
4. If you touched a "shared" file, the change is minimal and unambiguous, and you mentioned it in `.sync/<you>.md`.
5. Tests pass: `./scripts/build.sh && ./build-*/test/test_gpudb`.
6. Commit message uses [Conventional Commits](https://www.conventionalcommits.org/): `feat(cuda): ...`, `fix(metal): ...`, `chore(ci): ...`.
7. After commit: `./scripts/sync.sh status --done` (or describe next step).

## What lives outside this repo (do not look for it)
The user's planning notes, business strategy, and the deep research briefing live in `~/Documents/gpubasedpostrgress/.private/` on the Linux machine — **outside** the git repo. Do not try to read them from inside this repo, and never commit anything from that directory.

## When in doubt
Ask the user. Do not invent. Do not push to `main`. Do not modify the other platform's backend.
