# Instructions for Claude Code working in this repo

This repository is developed by **two Claude Code instances in parallel** — one on a Linux machine (NVIDIA RTX 4090, CUDA backend) and one on a macOS machine (Apple Silicon, Metal backend). Treat these instructions as load-bearing.

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
1. `git fetch origin && git status` — confirm you're not behind `main`.
2. Your changed files match your platform's allowed paths in the table above.
3. If you touched a "shared" file, the change is minimal and unambiguous.
4. Tests pass: `cmake --build <build> --target test_gpudb && ./<build>/test/test_gpudb`.
5. Commit message uses [Conventional Commits](https://www.conventionalcommits.org/): `feat(cuda): ...`, `fix(metal): ...`, `chore(ci): ...`.

## What lives outside this repo (do not look for it)
The user's planning notes, business strategy, and the deep research briefing live in `~/Documents/gpubasedpostrgress/.private/` on the Linux machine — **outside** the git repo. Do not try to read them from inside this repo, and never commit anything from that directory.

## When in doubt
Ask the user. Do not invent. Do not push to `main`. Do not modify the other platform's backend.
