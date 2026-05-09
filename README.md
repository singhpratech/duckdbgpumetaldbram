# duckdbgpumetaldb

GPU-accelerated analytical operators for DuckDB, with backends for **NVIDIA CUDA** and **Apple Metal**.

Status: pre-alpha, week 1 of development. Not for production use.

## What this is
A C++ library (`libgpudb`) and a DuckDB extension that offload selected analytical operators (initially: aggregations) to a GPU backend. Two backends are supported:

- **CUDA** — discrete NVIDIA GPUs on Linux/Windows
- **Metal** — Apple Silicon GPUs (unified memory architecture)
- **CPU fallback** — used when no GPU backend is available

The design is a backend abstraction layer so the same operator dispatch can target either GPU. Inspired by Sirius (UW-Madison + NVIDIA, CIDR 2026), but with first-class Apple Silicon support — there is currently no published Metal-based SQL execution engine.

## Build

### Linux (CPU + CUDA)
```bash
sudo apt install -y build-essential cmake nvidia-cuda-toolkit  # one-time
./scripts/build.sh
./build/bin/gpudb-bench --rows 10000000
```

If CUDA toolkit is not installed, the build will succeed with CPU backend only.

### macOS (CPU + Metal)
```bash
brew install cmake
./scripts/build.sh
./build/bin/gpudb-bench --rows 10000000
```

Metal backend is enabled automatically on macOS.

## Running tests
```bash
./build/test/test_gpudb
```

## Project layout
```
src/include/         abstract backend interface
src/backends/cpu/    CPU baseline (scalar + OpenMP)
src/backends/cuda/   CUDA backend (Linux/Windows)
src/backends/metal/  Metal backend (macOS)
src/operators/       operator-level dispatch
src/extension/       DuckDB extension wrapper
test/                Catch2 unit tests + DuckDB SQL tests
benchmark/           microbenchmarks and TPC-H harness
```

## License
Apache-2.0. See [LICENSE](LICENSE).
