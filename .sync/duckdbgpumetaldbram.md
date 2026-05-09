# duckdbgpumetaldbram
- branch: feat/metal-groupby-radix-gpu
- working on: Metal GROUP BY peak 4.89x at 500M × 1M groups, wins every workload >= 10M rows, TPC-H 2.08x. Min-max pre-scan + GPU on-device scan are the unlocks.
- status: in_progress
- blocked on: nothing
- last update: 2026-05-09T21:20:54Z
