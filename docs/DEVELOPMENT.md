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

## The release gate (`scripts/release_gate.py`)

Rule 2 of `docs/TRANSPARENT_DESIGN.md` §0, checked on statements nobody wrote by
hand: a statement the wrapper answers on the device returns exactly what DuckDB
returns, and a valid statement never raises a gpudb error. The wrapper suite
(`python/tests/test_wrapper.py`) and the TPC-H checks cover the statements
someone thought of, on clean data; the gate generates its statements and its
data to be hostile.

**What it does.** From one seed it builds a fact table whose columns cover every
type the wrapper reads (and several it declines) with NULLs, duplicates, each
type's extreme values, `infinity` / `-infinity`, NaN, -0.0, empty strings,
strings that differ only in case or by a non-ASCII letter and values that
collide after rounding, keys with 1, 2, 7, ~300 (skewed), 512 and thousands of
groups, a unique and a non-unique join dimension and a view; a CTE and derived
tables appear in the statements. It then generates statements as the product of
forms (GROUP BY, global aggregate, HAVING, top-k, DISTINCT, count(DISTINCT),
expressions over aggregates, nested / derived / CTE / view, joins, UNION ALL,
an IN subquery) x keys (each type, computed, multi-column, packed) x aggregates
x WHERE terms (every operator; literals the column can and cannot hold) x
ORDER BY / LIMIT (directions, NULL orders, ordinals, ALL, OFFSET, ties at the
limit) x name hazards (aliases equal to columns, to aggregate names, to the
names the rewrite itself uses): every PAIR of dimension values in at least one
statement, then random samples, then the full WHERE literal matrix (column x
literal x operator). Each statement runs three times through gpudb — first
execution (drawn by the extension's renderer), a re-run with a different literal
set (drawn by the Python renderer from the cached decision), the first set
again — and each execution is compared with native DuckDB for its own literals:
rows (in order where the ORDER BY is total, up to ties where it is not, as a
multiset without one), column names, column types, or the error class.

**Modes** (`--modes`, all by default): `eager` (test settings: eager residency,
`floor_rows=0`, `thresholds=False`), `lazy` (the `sql()` relation path, and the
rows also compared as DuckDB's own text, which is the only way to see an
infinite TIMESTAMP that a Python fetch turns into `datetime.max`), `background`
(the shipped residency), `write-same` / `write-sibling` / `write-raw` (an
insert, an in-place update or a delete between executions, through the
statement's connection, a sibling `cursor()` or a raw DuckDB cursor), `settings`
(collation, default order and null order, integer division, time zone, IEEE
float ops, set before the statement and between its executions through each of
those handles and `SET GLOBAL`), `warm` (one connection, decisions kept across
statements), `cursor` (each statement on a fresh `con.cursor()`), `shipped`
(`gpudb.connect()` defaults over a 1.1M-row table, waiting for the background
upload). Native mirrors the wrapper's handles, so a setting or a write reaches
the expected answer exactly where plain DuckDB makes it reach.

**Running it.**

```bash
python3 scripts/release_gate.py --quick                 # 5,332 statement-mode tasks, ~3 min on an M4 Max
python3 scripts/release_gate.py --full                  # 32,335 tasks, ~27 min: the release run
python3 scripts/release_gate.py --quick --baseline scripts/release_gate_baseline.json
python3 scripts/release_gate.py --wrapper ../other/python   # gate another checkout's wrapper
python3 scripts/release_gate.py --list                  # task counts only
```

It needs the extension (a local build or `GPUDB_EXTENSION_PATH`) and runs
under whichever DuckDB the interpreter has: run it once per supported DuckDB
version. The report gives, per mode, statements, executions, how many were
answered on the device at least once, declined (and by which reason),
native-ambiguous (native itself does not determine the answer: order among
ties, rows at a LIMIT that cuts a tie; compared as sets up to the tie and never
counted as a device pass), both-error, DIFFERENT and RAISED, then the device
coverage per form, so a gate that passes because everything declined is
visible. `--out results.jsonl` keeps one record per statement and mode.

**Replaying a failure.** Every failure prints a stable id, its mode, the SQL of
the failing execution, both results, where they part and `last_rewrite()`.
`--replay <id>` runs that statement alone, in that mode, with the same data;
`--only <text>` runs every statement whose SQL contains the text. Use the same
`--seed`, `--rows` and budget as the run that printed the id.

Samples are nested: a `--quick` run's statements are a subset of a `--full`
run's (same seed), so one baseline written from `--full` runs serves both.

**The baseline.** `--baseline scripts/release_gate_baseline.json` turns "no
failure" into "no NEW failure": each entry is a known-open failure keyed by its
id, with its defect area, the DuckDB versions and backends it fails on, and the
area's fix (a pull request or branch). The run fails on a failure not listed for
this version and backend, and on a listed entry that ran and PASSED (the fix
merged: remove the entry). `--write-baseline FILE` records the current run into
FILE (adding this version to failing entries, dropping it from passing ones,
removing entries left with no version); review the diff before committing it.
Entries from the timing-dependent modes (`background`, `shipped`, `settings`,
where whether the device answers depends on an upload finishing in time) are
marked `flaky` and only reported when they pass. The baseline shrinks to empty as the
fixes merge; it is never a place to park a new failure.

**A release requires a clean `--full` run** — exit 0 with an empty baseline —
on Apple Silicon (Metal) and on NVIDIA (CUDA), under every DuckDB version the
release supports.

**On a CPU-only runner** the wrapper declines every statement (reason
`backend`, rule 1), so a plain run proves only that declining is exact. Pass
`--cpu-reference` to let the transparent path lower onto the extension's CPU
reference operators: every rewrite, renderer, lane, mask, join lowering and
fallback runs and is compared exactly as on a device; what it cannot cover is
device code (kernels, device memory, the backend's own upload path) and the
`shipped` mode (thresholds are measured per device), which it skips.
