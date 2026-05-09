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

## CMake flags reference
- `-DGPUDB_ENABLE_CUDA=ON/OFF`  (default ON; auto-disabled if no nvcc)
- `-DGPUDB_ENABLE_METAL=ON/OFF` (default ON on macOS; ignored on Linux)
- `-DGPUDB_BUILD_TESTS=ON/OFF`
- `-DGPUDB_BUILD_BENCH=ON/OFF`
- `-DGPUDB_BUILD_EXT=ON/OFF`    (DuckDB extension; OFF until DuckDB submodule wired in)

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
