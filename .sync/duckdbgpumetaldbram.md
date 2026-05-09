# duckdbgpumetaldbram
- branch: feat/metal-groupby-sort
- working on: Metal GROUP BY: bitonic sort + host segment-reduce. Correct on all cardinalities; perf is dispatch-overhead-bound (loses to CPU 3-16x). Architecture win first, perf follow-up next.
- status: in_progress
- blocked on: nothing
- last update: 2026-05-09T19:48:41Z
