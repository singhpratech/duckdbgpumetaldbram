# gpudb in CI — copy-paste recipes

GitHub's `macos-14` and `macos-15` hosted runners are **Apple Silicon (M1/M2)
machines**. That means the GPU path of `gpudb` — the Metal backend — runs in
plain, free GitHub Actions with zero setup. As far as we know, gpudb is the
only DuckDB extension that does anything special on those runners.

All recipes below install from the community registry (`INSTALL gpudb FROM
community;`), so they work on any DuckDB ≥ 1.5.5 with no extra flags. If a
runner has no usable GPU, gpudb falls back to CPU cleanly — same SQL surface,
same results — so none of these jobs can break just because the GPU is
missing.

---

## 0. The parity gate — a drop-in correctness check for your data

If you already aggregate data with DuckDB in CI, this workflow adds an
independent second opinion: every aggregate is computed twice — once by
DuckDB's native engine, once by gpudb's separately-implemented backend
(Metal on macOS runners, CPU elsewhere) — and the job fails if they ever
disagree. Two independent implementations agreeing on your real data is a
stronger check than either alone, and on `macos-15` the second opinion runs
on the GPU for free.

Copy, point `FILES` at your data, done:

```yaml
name: aggregate-parity-gate
on:
  pull_request:
  schedule:
    - cron: "23 5 * * *"   # nightly, against fresh data

jobs:
  parity:
    runs-on: macos-15       # Apple Silicon: gpudb's second opinion runs on Metal
    env:
      FILES: "data/*.parquet"   # <-- your data here
    steps:
      - uses: actions/checkout@v4
      - name: Install DuckDB CLI
        run: brew install duckdb
      - name: Aggregate parity gate (native vs gpudb)
        run: |
          duckdb -c "
          INSTALL gpudb FROM community; LOAD gpudb;
          SELECT CASE
            WHEN gpu_sum(x) = sum(x)
             AND gpu_min(x) = min(x)
             AND gpu_max(x) = max(x)
            THEN 'parity ok: ' || count(*) || ' rows'
            ELSE error('native and gpudb disagree — investigate before merging')
          END
          FROM (SELECT amount::BIGINT AS x FROM read_parquet(getenv('FILES'))) t;
          "
```

Adapt the inner `SELECT` to the columns you actually gate on; add one
`CASE` block per column. For `DOUBLE` columns compare with a tolerance
(`abs(gpu_sum(d) - sum(d)) < 1e-6 * greatest(abs(sum(d)), 1)`) — float
summation order differs between engines. Read the
[semantics section](#semantics-worth-knowing-before-you-gate-ci-on-gpudb)
below before gating on columns near int64 range.

## 1. GitHub Actions on Apple Silicon (the fun one)

Data checks with the Metal backend on a free hosted runner:

```yaml
name: data-checks
on: [push, schedule]

jobs:
  gpu-data-checks:
    runs-on: macos-15          # Apple Silicon (M-series) hosted runner
    steps:
      - uses: actions/checkout@v4

      - name: Install DuckDB CLI
        run: brew install duckdb

      - name: GPU-accelerated checks
        run: |
          duckdb -c "
            INSTALL gpudb FROM community;
            LOAD gpudb;
            -- example: validate a dataset with GPU aggregates
            SELECT
              gpu_sum(amount::BIGINT)  AS total,
              gpu_min(amount::BIGINT)  AS lo,
              gpu_max(amount::BIGINT)  AS hi
            FROM read_parquet('data/*.parquet');
          "
```

Useful patterns on top of this skeleton:

- **Parity assertion** — run `gpu_sum` and native `sum` side by side and fail
  the job if they differ. Cheap correctness gate for your own data:

  ```sql
  SELECT CASE
    WHEN gpu_sum(x::BIGINT) = sum(x::BIGINT) THEN 'ok'
    ELSE error('gpu/native mismatch')
  END FROM read_parquet('data/*.parquet') AS t(x);
  ```

- **Scheduled runs** — add a `schedule:` trigger (cron) and the job re-runs
  nightly against fresh data. This is how most production DuckDB extension
  usage actually looks: unattended, on a timer.

  ```yaml
  on:
    schedule:
      - cron: "17 4 * * *"   # nightly 04:17 UTC
  ```

Note on virtualized Metal: hosted macOS runners expose the GPU through
Apple's paravirtualized Metal device. The Metal backend initializes and runs
there; if a particular runner image ever doesn't expose it, gpudb logs the
fallback and the job still passes on CPU.

## 2. GitHub Actions on Linux (correctness fallback)

The community `linux_amd64` binary is built without the CUDA toolchain
(registry CI has no GPUs), so on `ubuntu-latest` gpudb runs its CPU fallback.
Still useful as a portability/correctness check — and it exercises the exact
binary your Linux users get:

```yaml
  cpu-fallback-checks:
    runs-on: ubuntu-latest
    steps:
      - name: Install DuckDB CLI
        run: |
          curl -sL https://github.com/duckdb/duckdb/releases/latest/download/duckdb_cli-linux-amd64.zip -o duckdb.zip
          unzip duckdb.zip && sudo mv duckdb /usr/local/bin/
      - name: gpudb smoke
        run: |
          duckdb -c "
            INSTALL gpudb FROM community;
            LOAD gpudb;
            SELECT gpu_sum(value::BIGINT) FROM range(1000000) AS t(value);
          "
```

For the real CUDA backend on Linux, build from source on a machine with
`nvcc` (see recipe 4).

## 3. Docker

Bake the extension into an image so containers don't re-download at runtime
— or install at container start if you always want the latest registry build:

```dockerfile
FROM ubuntu:24.04
RUN apt-get update && apt-get install -y curl unzip \
 && curl -sL https://github.com/duckdb/duckdb/releases/latest/download/duckdb_cli-linux-amd64.zip -o /tmp/duckdb.zip \
 && unzip /tmp/duckdb.zip -d /usr/local/bin && rm /tmp/duckdb.zip
# pre-install gpudb into the image's extension directory
RUN duckdb -c "INSTALL gpudb FROM community;"
ENTRYPOINT ["duckdb"]
```

```bash
docker build -t duckdb-gpudb .
docker run --rm duckdb-gpudb -c "LOAD gpudb; SELECT gpu_sum(value::BIGINT) FROM range(1e6::BIGINT) t(value);"
```

## 4. Self-hosted CUDA runner (full GPU on Linux)

If you have a self-hosted runner with an NVIDIA GPU (sm_75+, Turing or newer), build once and
cache the artifact:

```yaml
  cuda-build-and-check:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v4
        with:
          repository: singhpratech/duckdbgpumetaldbram
          submodules: recursive
      - name: Build with CUDA
        run: ./scripts/build.sh          # auto-detects nvcc
      - name: GPU checks
        run: |
          ./build-linux/test/test_gpudb
```

## 5. Colab / Jupyter

There is a ready-made notebook at
[`examples/gpudb_quickstart.ipynb`](../examples/gpudb_quickstart.ipynb) —
one-click quick start on Google Colab, including an optional
build-from-source section that compiles the CUDA backend on Colab's free GPU
runtime.

---

## Semantics worth knowing before you gate CI on gpudb

- Empty input and all-NULL groups return SQL `NULL`, matching native DuckDB.
- `DOUBLE` min/max use DuckDB's NaN-aware total order (NaN sorts greatest).
- `gpu_sum(BIGINT)` wraps on int64 overflow where native `sum()` promotes to
  `HUGEINT` — don't gate on columns that can exceed int64 range.
- `HUGEINT`/`DECIMAL`/`FLOAT` arguments cast to the `DOUBLE` overload; use
  native aggregates where exactness beyond 2^53 matters.

Full details: [KNOWN_ISSUES.md](../KNOWN_ISSUES.md) and the
[community extension page](https://duckdb.org/community_extensions/extensions/gpudb).
