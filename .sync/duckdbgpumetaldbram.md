# duckdbgpumetaldbram
- branch: feat/metal-groupby-sort
- working on: Metal GROUP BY now WINS over CPU at 1M rows x 1M groups (10.6ms vs 21.4ms = 2.0x). Two-tier bitonic dispatch (local sort + cross-block step + local merge) cut dispatches from 210 to 78. Tests 24/24 pass. PR #3 ready for review.
- status: in_progress
- blocked on: nothing
- last update: 2026-05-09T19:57:07Z
