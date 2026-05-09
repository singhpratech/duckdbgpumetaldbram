# duckdbgpumetaldb
- branch: feat/metal-real-kernels
- working on: Metal real compute pipelines done — sum/min/max_i64 hitting GPU (220 GiB/s kernel-only on M4 Max). f64 stays on CPU (no MSL double). GroupBy is honest CPU fallback because Apple GPUs lack 64-bit atomic CAS.
- status: in_progress
- blocked on: nothing
- last update: 2026-05-09T18:57:49Z
