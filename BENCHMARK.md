# Benchmark log

Append-only. Reproducible runs only — include hardware, CUDA toolkit, build flags.

## 2026-05-09 — first CUDA vs CPU SUM, RTX 4090 Laptop

**Hardware**
- CPU: 20-thread x86_64 (Linux Mint 22.3 / Ubuntu 24.04 base)
- GPU: NVIDIA GeForce RTX 4090 Laptop, sm_89, 15.6 GiB VRAM
- Driver 580.142, CUDA Toolkit 13.0.88
- Memory: DDR5 (CPU baseline saturates ~54 GiB/s — close to dual-channel theoretical)
- PCIe: Gen4 x16 (~32 GiB/s theoretical, observed effective ≈ 10 GiB/s on these copies)

**Build**
- gcc 13.3.0, CMake 3.x, Release, OpenMP enabled
- Two-pass tree reduction, 256 threads/block, grid-stride load
- No CUB/Thrust dependency yet (hand-rolled kernel)

**Workload**
- 100,000,000 int64 elements (762 MiB), uniform random in [-1e6, +1e6]
- Operation: SUM
- 5 runs, median reported

**Result**
| Backend | wall (ms) | kernel (ms) | transfer H2D (ms) | wall throughput | kernel throughput |
|---|---:|---:|---:|---:|---:|
| CPU (20 thread OpenMP) | 13.76 | n/a | n/a | 54.16 GiB/s | — |
| CUDA — total | 80.63 | 1.44 | 79.17 | 9.24 GiB/s | — |
| CUDA — kernel only | — | **1.44** | — | — | **517.29 GiB/s** |

**Interpretation**
- **The PCIe wall is real** on this hardware. 98 % of CUDA wall time is host→device transfer.
- **The GPU kernel is 9.5× faster** than the CPU once data is resident.
- This reproduces the Shanbhag/Madden/Yu (Crystal, SIGMOD 2020) finding that GPU-as-coprocessor with cold data underperforms CPU; only GPU-resident workloads win.

**What this tells us about the project direction**
- Single-shot SUM on cold data is the wrong workload to optimize.
- Real wins require: (a) data already resident in VRAM (multiple queries per load), (b) NVLink/GH200 unified memory, or (c) Apple Silicon UMA (no transfer).
- Next step: implement a workload that amortizes transfer — e.g. multiple aggregations on the same column, or a hash group-by where the GPU stays "hot" across many queries.

## 2026-05-09 — CPU baseline, smaller workload (sanity check)

50M int64 SUM, 5 runs:
- CPU 6.72 ms → 55.46 GiB/s
- CUDA: not yet run (was CPU-only build at the time)

10M int64 SUM from disk file (`data/synth_10M_i64.gpudb`), 5 runs:
- CPU 1.43 ms → 51.95 GiB/s

## How to reproduce

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
cd ~/Documents/gpubasedpostrgress/duckdbgpumetaldb
./scripts/build.sh
./build-linux/bin/gpudb-bench --rows 100000000 --runs 5
```
