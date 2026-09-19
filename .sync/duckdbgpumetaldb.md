# duckdbgpumetaldb
- branch: feat/cuda-exact-path
- working on: PR #152 open: CUDA exact GROUP BY on the device (gate off by default behind GPUDB_CUDA_EXACT=1). Touches shared hybrid_planner.cpp (placement now follows exact_supported()), gpu_resident.cpp (comment only), src/CMakeLists.txt, CLAUDE.md. Unit 652/652, SQL 223/1 unchanged by default. Next: aggregate_exact_masked + global_supported().
- status: in_progress
- blocked on: nothing
- last update: 2026-09-19T22:35:28Z
