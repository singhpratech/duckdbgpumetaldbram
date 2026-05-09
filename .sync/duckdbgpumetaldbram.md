# duckdbgpumetaldbram
- branch: feat/metal-groupby-sort
- working on: TPC-H SF1 + 1B-row sweep landed on PR #3. Honest numbers across the scale: Metal wins 2.1x at 1M sweet spot, CPU wins 1.5-2.2x at 10M-1B. Bitonic O(N log^2 N) is the bottleneck; GPU-resident radix sort is next.
- status: in_progress
- blocked on: nothing
- last update: 2026-05-09T20:27:32Z
