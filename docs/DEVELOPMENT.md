# Development guide

Two machines, no GitHub required (yet).

## Quick start (Linux, this machine)

```bash
cd ~/Documents/gpubasedpostrgress/duckdbgpumetaldb
./scripts/build.sh
./build-linux/test/test_gpudb
./build-linux/bin/gpudb-bench --rows 50000000 --runs 5
```

If `nvcc` is missing the build is CPU-only: everything compiles and the tests run, with the CUDA backend absent. `scripts/install_cuda_ubuntu.sh` installs the toolkit.

## Quick start (macOS)

```bash
cd ~/dev/duckdbgpumetaldb     # wherever you place it
./scripts/build.sh
./build-macos/test/test_gpudb
./build-macos/bin/gpudb-bench --rows 50000000 --runs 5
```

Metal backend is auto-enabled on macOS. Xcode Command Line Tools are required (`xcode-select --install`).

## Moving the repo to your Mac without GitHub

Pick whichever fits — both keep the repo private to you and the two machines.

### Option A: rsync over SSH
On the Mac, after enabling Remote Login in System Settings:
```bash
# from Linux:
rsync -avz --exclude build-linux --exclude data --exclude .tools \
    ~/Documents/gpubasedpostrgress/duckdbgpumetaldb/ \
    you@your-mac.local:~/dev/duckdbgpumetaldb/
```

### Option B: tarball + AirDrop / USB
```bash
# from Linux:
cd ~/Documents/gpubasedpostrgress
tar --exclude='duckdbgpumetaldb/build-linux' \
    --exclude='duckdbgpumetaldb/data' \
    --exclude='duckdbgpumetaldb/.tools' \
    -czf duckdbgpumetaldb.tgz duckdbgpumetaldb/
# AirDrop or scp duckdbgpumetaldb.tgz to the Mac, then:
#   tar -xzf duckdbgpumetaldb.tgz && cd duckdbgpumetaldb && ./scripts/build.sh
```

Both options preserve the local git history. On the Mac side:
```bash
cd duckdbgpumetaldb
git status              # should be clean
git log --oneline       # should match Linux history
```

## Two-machine git workflow without a remote

To share commits between the two machines without GitHub, either of these works:

### Direct SSH push between machines
```bash
# from Mac, pull from Linux box (one-time):
git remote add linux ssh://you@linux-box/home/you/Documents/gpubasedpostrgress/duckdbgpumetaldb
git pull linux main

# subsequent pulls:
git pull linux main
```

This requires SSH access between the boxes but never touches GitHub.

### USB / AirDrop bundle
```bash
# Linux side (sender):
git bundle create /tmp/sync.bundle main
# move sync.bundle to Mac via AirDrop/USB
# Mac side (receiver):
git fetch /Volumes/USB/sync.bundle main:linux-main
git merge linux-main      # or rebase, your call
```

## When you're ready to push to GitHub

The repo `singhpratech/duckdbgpumetaldbram` already exists and is private. To start using it:

```bash
cd ~/Documents/gpubasedpostrgress/duckdbgpumetaldb
git remote add origin https://github.com/singhpratech/duckdbgpumetaldbram.git
git push -u origin main
```

That's it. Until you run that command nothing leaves your machines.

## Branch / commit conventions
Short version:
- Never commit to `main` directly (well — first commit excepted). Use `feat/cuda-*`, `feat/metal-*`, `feat/core-*`, `chore/*`.
- Conventional commit messages: `feat(cuda): add radix sort kernel`, `fix(cpu): handle empty input`.
- Test before committing: `./scripts/build.sh && ./build-*/test/test_gpudb`.

## The performance gate (`scripts/perf_gate.py`)

The release gate has two halves. `scripts/release_gate.py` asks whether a
candidate returns what DuckDB returns; `scripts/perf_gate.py` asks whether it
is slower than the released baseline, slower than DuckDB alone where it runs
on the device, or runs fewer statements on the device than the baseline did.
It compares two Python wrappers (and, optionally, two extension binaries)
statement by statement over a fixed workload: TPC-H's 22 queries at SF1 and
SF10 through `execute()` and through `sql()`, one statement per shape the
wrapper distinguishes over generated tables of 2M and 20M rows, statements
meant to stay on DuckDB (below the row floor, an unrewritten shape, over the
memory budget, during a background upload, `SELECT 1`, an `INSERT`, a fresh
`con.cursor()`), and optionally the fourteen OS Open TOID statements.

```bash
# quick: TPC-H SF1 and the 2M-row tables (generates SF1 if missing); 2 min for one
# candidate, 5 min for four, on an M4 Max
python -I scripts/perf_gate.py --quick --baseline installed --candidate python
# full: SF1 + SF10, 2M + 20M rows, fresh-process figures; 26-28 min for two candidates
python -I scripts/perf_gate.py --absolute --baseline /path/to/released/python \
    --candidate python --ext build-macos/src/extension/gpudb.duckdb_extension
# a candidate extension binary against the released one
python -I scripts/perf_gate.py --baseline B/python --baseline-ext old.duckdb_extension \
    --candidate python --candidate-ext new.duckdb_extension
```

The interpreter that runs the script supplies DuckDB; run it once per DuckDB
version that matters. Every side is a process of its own, and every role
(native DuckDB, the baseline, each candidate) is two processes, interleaved
one run each per round in a carryover-balanced order. A verdict needs both
candidate processes to differ from both baseline processes: on Apple Silicon
one process of identical code can run 20% slower than its twins for a whole
run, which says where the scheduler put it, not what it ran. Tolerances are
derived per run from the baseline's own A/A pair (the 90th percentile of
their distance per duration bucket) and from the samples' spread, never a
fixed percentage; the report prints that floor.

What fails a run (exit 1): `SLOWER_THAN_NATIVE` (the candidate runs the
statement on the device and DuckDB alone is faster), `SLOWER_THAN_BASELINE`
(on the same route as the baseline, slower), `COVERAGE_LOST` (the baseline
ran it on the device in every process, the candidate in none, and not
because its own timing said a marginal win was not one) and
`CANDIDATE_ERROR`. `BASELINE_SLOWER_THAN_NATIVE` is the same rule breach when
the baseline has it too; it fails the run under `--absolute`. Reported but
not failing: first execution on a fresh connection, time to resident in
background mode, peak host and device memory, statements during a background
upload. Informational: `ROUTE_CHANGED` (form, renderer or decline reason),
`COVERAGE_GAINED`, `COVERAGE_UNSTABLE` and `ROUTE_UNSTABLE` (processes of
the same wrapper disagree), `ROWS_DIFFER`, `BIMODAL` (the samples sit in two
modes; compared on the fast one), `NOISY`.

A release needs a clean full run with `--absolute` on a quiet Apple Silicon
machine and on the NVIDIA machine. Quiet is checked: before starting, the
load average and the CPU share of every other process; during the run, the
share again around every statement. A full run holds the machine's
benchmark lock — `.bench.lock` in the directory that holds the main checkout,
or `GPUDB_BENCH_LOCK` — for its duration, and every run refuses to start
while it exists. `--allow-noisy` runs anyway and labels
every figure provisional; such a run cannot clear a release. `--quick` is
what a CPU-only CI runner can afford: there the rewriter declines everything
(reason `backend`), so it measures the wrapper's overhead on the declined
path and nothing about the device.

A loss that is deliberate — a statement that had to leave the device to be
correct — is recorded, never hidden:

```bash
python scripts/perf_gate.py --accept 'tpch.sf1.q15.*' --verdict COVERAGE_LOST \
    --reason "<why the statement has to stay on DuckDB, and what would bring it back>"
```

That appends to `scripts/perf_gate_accepted.json` (commit it with the change
that needs it). An accepted loss no longer fails a run, and every report
lists every entry in that file, matched or not. `--rejudge <result.json>`
applies the file (and the current verdict rules) to a run already measured,
without measuring again. Results land in
`--out` (default `data/perf_gate/runs/<time>/`): `result.json` with every
sample, and `report.md`, whose tables can be pasted into `BENCHMARK.md`.
