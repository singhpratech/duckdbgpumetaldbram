# Datasets

This project benchmarks aggregations over numeric columns. Three data sources are supported.

## 1. Synthetic (no dependencies)

The `gpudb-gen` tool generates random columns in the project's flat `.gpudb` binary format.

```bash
mkdir -p data
./build-linux/bin/gpudb-gen --out data/synth_100M_i64.gpudb --rows 100000000 --dtype i64
./build-linux/bin/gpudb-gen --out data/synth_100M_f64.gpudb --rows 100000000 --dtype f64
./build-linux/bin/gpudb-bench --input data/synth_100M_i64.gpudb --runs 10
```

Deterministic given a `--seed`. Use this for fast smoke tests and microbenchmarks.

## 2. TPC-H (industry standard, license-friendly)

[TPC-H](https://www.tpc.org/tpch/) is the canonical analytical query benchmark. We use [DuckDB's `tpch` extension](https://duckdb.org/docs/extensions/tpch.html) to generate it (MIT-licensed reproduction of dbgen).

```bash
SF=1 ./scripts/gen_tpch.sh    # ≈ 1 GB lineitem
SF=10 ./scripts/gen_tpch.sh   # ≈ 10 GB lineitem
```

The script downloads the DuckDB CLI into `./.tools/`, generates SF data into `./data/tpch_sf<N>/`, and exports three columns to `.gpudb` format for the benchmark:

| Column | Type | Notes |
|---|---|---|
| `lineitem.l_quantity` | f64 | Small integers as doubles |
| `lineitem.l_extendedprice` | f64 | Decimal-as-double |
| `lineitem.l_orderkey` | i64 | Sequential IDs |

Then:
```bash
./build-linux/bin/gpudb-bench --input data/tpch_sf1/lineitem_orderkey.gpudb --runs 10
```

## 3. ClickBench (real-world, bigger)

[ClickBench](https://benchmark.clickhouse.com/) is the modern analytical leaderboard, based on real Yandex.Metrica web traffic. The `hits.parquet` file is ~14 GB.

```bash
mkdir -p data/clickbench
curl -fsSL -o data/clickbench/hits.parquet \
    https://datasets.clickhouse.com/hits_compatible/hits.parquet
```

ClickBench is mostly used at the SQL level (we'll wire this up once the DuckDB extension is in). For week-1 microbenchmarks of single-column SUM, TPC-H lineitem is sufficient.

## Format reference: `.gpudb` flat binary

```
[8 bytes] magic "GPUDB001"
[4 bytes] uint32_t dtype  (0 = int64, 1 = float64)
[4 bytes] uint32_t flags  (reserved)
[8 bytes] uint64_t count
[count * sizeof(elem)] raw little-endian elements
```

Defined in [`src/include/data_format.hpp`](../src/include/data_format.hpp). Trivial to mmap or stream.

## Why not Parquet directly (yet)

Parquet readers (Apache Arrow C++) add ~100 MB of build dependencies and complicate the CUDA/Metal build matrix. For week 1 we ship the trivial format and convert via DuckDB CLI. Once we wire the DuckDB extension (`-DGPUDB_BUILD_EXT=ON`), Parquet ingest will come for free via DuckDB's reader.
