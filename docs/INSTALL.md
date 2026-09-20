# Installing gpudb

Everything needed to get the extension into DuckDB and the wrapper onto your
`PATH`, on Apple Silicon Metal or NVIDIA CUDA — the two install routes, the
platform detail, troubleshooting, upgrading and what to do when GPU memory is
full.
[← back to the README](../README.md) · [the `gpudb` shell](USING_THE_SHELL.md) ·
[the Python way](USING_PYTHON.md)

## Installing, in full

Two routes, both of which end with the extension inside DuckDB and the wrapper
on your `PATH`. Installing one piece without the other is the single most
common way to end up with a shell that works but never uses the GPU.

**Route 1 — the registry extension plus the pip wrapper.**

1. Install the extension into DuckDB, from any DuckDB ≥ 1.5.5 client:
   ```sql
   INSTALL gpudb FROM community;
   LOAD gpudb;
   ```
   It is signed; no flags.
2. Install the wrapper:
   ```bash
   pip install duckdb-gpudb          # installs the `gpudb` command and the gpudb module
   ```

   The `duckdb` module `pip` pulls in (or the one you already have) must be a
   version the registry publishes a gpudb build for — **Supported versions**,
   at the end of this section, says which and how to pin it.

**Route 2 — from a checkout.** Three steps; the first is the one that is easy
to miss, because `scripts/build.sh` only builds the loadable extension when
DuckDB's own headers are present:

1. Fetch pre-built libduckdb + headers into `third_party/duckdb-libs/`:
   ```bash
   ./scripts/get_duckdb_libs.sh
   ```
2. Build:
   ```bash
   ./scripts/build.sh
   # → build-macos/src/extension/gpudb.osx_arm64.duckdb_extension
   #   (or build-linux/…/gpudb.linux_amd64.duckdb_extension)
   ```
3. Install the wrapper from the same checkout:
   ```bash
   pip install -e python/
   ```

Step 3 is also how the wrapper finds the extension you just built: it walks up
from its own `gpudb/` directory and looks for a `build-macos/` or
`build-linux/` beside the checkout. If you installed the wrapper from
somewhere else, point it at the file:

```bash
export GPUDB_EXTENSION_PATH=/path/to/build-macos/src/extension/gpudb.osx_arm64.duckdb_extension
```

**The lookup order**, whichever route you took: an explicit
`gpudb.connect(extension="…")`, then `GPUDB_EXTENSION_PATH`, then a
`build-macos/` / `build-linux/` beside a source checkout, then whatever
DuckDB itself has installed. If nothing usable is found — or what is found is
older than the client — the banner's `transparent:` line says so,
`con.extension_note` carries the same sentence, and every statement simply
runs on DuckDB.

**Supported versions.** Python ≥ 3.9 and the `duckdb` module ≥ 1.4 — the
wrapper's own requirements, and they are *not* the same as route 1's. The
registry builds gpudb separately for each DuckDB version from 1.5.5 on, so a
`duckdb` module that satisfies `pip` can still be a version the registry has
nothing to install for: on DuckDB 1.4.5, `INSTALL gpudb FROM community` is a
404. Pin it with `pip install "duckdb==1.5.5"` if you are taking route 1. A
binary from the releases page needs only DuckDB ≥ 1.2, because the loadable
extension is built against the stable C API v1.2.0.

macOS: **macOS 14 or later on Apple silicon** (Metal shading language 3.1;
3.2 is used on macOS 15 and later). The binary asks the OS at run time and
compiles its shaders as MSL 3.2 where that is available, MSL 3.1 below
(`src/backends/metal/metal_groupby.mm`). Building needs the macOS 15 SDK,
which is where `MTLLanguageVersion3_2` comes from (CI builds on `macos-15`).
Linux with CUDA: see the [CUDA
requirements](#cuda-requirements-build-from-source-on-linux) table — the short
version is that a binary built with CUDA 13 needs an R580+ driver, and one
built with CUDA 12.x reaches the GPU on R525+.

## Platforms and install

| | Metal (Apple Silicon) | CUDA (NVIDIA) | No GPU |
|---|---|---|---|
| Plain SQL on the GPU, through the shell or `gpudb.connect()` | yes | opt-in, see below | no — everything runs on DuckDB |
| Explicit `gpu_*` functions | yes | yes | yes, on the CPU backend, same answers |
| From the community registry | yes | a registry Linux binary may report `compiled=cpu`; `SELECT gpu_build_info();` is what answers this for whichever binary is in front of you | yes |

On NVIDIA hardware every operator the transparent path needs is implemented —
exact `GROUP BY`, the `WHERE` mask, the global aggregate and the materialised
join. On an RTX 4090 Laptop with the path enabled the unit suite is
**750 / 750**, the SQL suite **224 passing, 0 failing**, and TPC-H at SF1 is
**17 of 22 queries on the device, 0 rows differing from native** — the same
coverage and the same five declines as Metal at that scale factor
([BENCHMARK.md](../BENCHMARK.md), *the transparent path on CUDA*).

It is **opt-in** in this release: set `GPUDB_CUDA_EXACT=1` to let the CUDA
backend answer plain SQL. One thing makes it opt-in, and it is not the
kernels. Every threshold the wrapper decides with was measured on Metal, and
the gate that measures them (`scripts/transparent_gate.py`) has not been run on
CUDA hardware. Rule 1 says never slower, and an unswept table is not
evidence for it — TPC-H Q1 straddling 1.0× on that box is exactly the kind of
row a sweep is for. Without the flag a CUDA machine gets the explicit `gpu_*`
functions and leaves plain SQL to DuckDB — correct, with no speed-up.

`SELECT gpu_build_info();` tells any binary apart: `compiled=` lists the
backends it was built with and `runtime=` the one it chose. The install routes
are in [Quick start](../README.md#quick-start): the community registry, a release
binary, or a build from source. The Python wrapper and the `gpudb` shell come
from `pip install duckdb-gpudb`.


## What a given binary carries

Works in any DuckDB ≥ 1.5.5 client (CLI, Python, etc.), signed, no flags
needed — that is the range the registry builds gpudb for. The registry binary
carries the **full Metal backend on Apple Silicon**. A registry Linux binary
may report `compiled=cpu`, in which case every `gpu_*` function works and
returns the same results, with `gpu_last_stats()` saying `backend=CPU`.
**`SELECT gpu_build_info();` is the answer for whichever binary is in front of
you**: `compiled=` lists what it was built with and `runtime=` the backend it
chose. For the CUDA backend take the release binary (Option B; statically
linked CUDA runtime, needs only a driver) or build from source with `nvcc`.
The v0.7.0 build carries all 65 `gpu_*` functions: the
streaming aggregates, the full resident-column surface (`gpu_upload`,
`gpu_sum_resident`, `gpu_residents`, `gpu_build_info`, …), the GPU join
functions (`gpu_upload_pair`, `gpu_join_*_resident`, `gpu_join_rows_resident`,
`gpu_inner_join`), the resident GROUP BY / top-k table functions
(`gpu_groupby_*_resident`, `gpu_topk_resident`) and the exact family the
transparent path uses (`gpu_upload_*_exact`, `gpu_groupby_exact_*`,
`gpu_agg_exact_global`, `gpu_rewrite_ast`). Installed an earlier version?
`UPDATE EXTENSIONS;` pulls the latest.

The loadable extension is built against the stable C API v1.2.0, so a release
binary loads in any DuckDB ≥ 1.2; the community install above is what needs
≥ 1.5.5, because the registry builds gpudb separately for each DuckDB version
from 1.5.5 on.
Release binaries track the latest tag. `LOAD` needs `-unsigned` here because
release-page binaries are unsigned — the community install above does not.

`SELECT gpu_build_info();` on the binary in front of you:

```sql
SELECT gpu_build_info();
-- compiled=cpu,metal runtime=metal exact=true join=true global=true narrow=true
--   device_memory=55662788608 store=true rebuilds=0/0 device='Apple M4 Max'
--   avgf=53
-- (avgf is the mantissa bits of the host's long double, which is what native
--  finalises an avg in; 53 on arm64, 64 on x86-64)
```

## CUDA requirements (build from source on Linux)

| | Supported | Notes |
|---|---|---|
| **CUDA Toolkit** | **13.0** (verified: 13.0.88 / CUB 3.0.1, all benchmarks) | Runtime API plus the CUB that ships with the toolkit (`DeviceReduce`, `DeviceSelect`, `DeviceRadixSort`, `DeviceScan` on explicit temp storage; Thrust only for iterators; no cooperative groups). 64-bit item counts need **CUB ≥ 2.1, i.e. CUDA 12.2 or newer**; older 12.x narrows counts to 32-bit (fine below 2^31 rows) and 11.x is not supported. Only 13.0 is tested by us — if you build on 12.x, please open an issue with your `nvcc --version` either way. C++17 host + device. |
| **NVIDIA driver** | **580.x** (verified) | Any driver that supports your toolkit (NVIDIA's minimum for 13.0 is R580; for 12.x, R525+). Runtime linking: `-DGPUDB_CUDA_STATIC_RUNTIME=ON` (what the registry build in the root `Makefile` uses; off by default in `scripts/build.sh`) bakes `cudart` into the extension, so the only runtime dependency is `libcuda.so` from the driver — and the extension still loads on machines with no GPU/driver, falling back to CPU. |
| **GPUs** | **sm_75 – sm_90**: Turing (T4, RTX 20xx), Ampere (A100, RTX 30xx), Ada (RTX 40xx, L4/L40), Hopper (H100) | Default fatbin: `75;80;86;89;90`, each with SASS + PTX. Newer parts (Blackwell / RTX 50xx, sm_100+) load via PTX JIT from `compute_90` — should work; not measured here. **Volta (sm_70) and older are not supported**: CUDA 13 dropped them from `nvcc`. Override with `-DCMAKE_CUDA_ARCHITECTURES=...` or `CUDAARCHS=...` (the Colab notebook builds `CUDAARCHS=75` for its T4). |

Verified configuration: RTX 4090 Laptop (sm_89, 16 GB), CUDA 13.0.88, driver
580.x, Linux — every CUDA number in this README and BENCHMARK.md comes from
that box. `SELECT gpu_build_info();` reports whether any given binary was
compiled with CUDA and which backend it picked at runtime.

## Troubleshooting

| What you see | What it is |
|---|---|
| `backend: none — the extension is not loaded` in the banner, and `transparent: a plain DuckDB shell — every statement goes straight to DuckDB` under it | No extension this connection can load. `con.extension_note` (and `last_rewrite()["detail"]`) spells it out: *install it with `INSTALL gpudb FROM community` run on the same DuckDB version as this client's `duckdb` module, or point `GPUDB_EXTENSION_PATH` at a built one.* The registry builds gpudb separately for each DuckDB version and installs it under that version's own directory, so an `INSTALL` run from a different version leaves nothing this one will find. |
| `transparent: off — the loaded gpudb extension is older than this client: it does not provide …` | DuckDB has an older gpudb installed. The message ends with the advice that works: *update it with `FORCE INSTALL gpudb FROM community;` (or `UPDATE EXTENSIONS;`), then start a new session.* A plain `INSTALL` does nothing when a copy is already installed — it keeps the file it finds — and neither form reaches a process that has already loaded the old one, which is why it ends in a new session. |
| `IO Error: Extension "…" could not be loaded because its signature is either missing or invalid` | A locally built binary. Start DuckDB with `-unsigned`, or from Python pass `config={"allow_unsigned_extensions": "true"}`. The `gpudb` shell already does this for a build it found itself. |
| `transparent: not on this build` | The extension loaded but has no exact operators — a CPU-only build, or a CUDA build without `GPUDB_CUDA_EXACT=1`. |
| Every statement says `DuckDB (threshold: …)` | Working as intended: your tables or your shapes are below the measured bounds. `.gpu` names the bound. |
| `Catalog Error: … gpu_sum does not exist` | The extension is installed but not loaded in *this* session. `LOAD gpudb;`. |

## Upgrade, uninstall, and turning it off

```bash
pip install -U duckdb-gpudb                  # the wrapper
pip uninstall duckdb-gpudb                   # ... and remove it
```
```sql
FORCE INSTALL gpudb FROM community;          -- replace an installed extension
UPDATE EXTENSIONS;                           -- or bring every extension up to date
```
DuckDB keeps installed extensions under `~/.duckdb/extensions/`; deleting
gpudb's directory there uninstalls it.

To turn the GPU path off without uninstalling anything: `.gpu off` in the
shell (`.gpu on` puts it back), `--no-gpu` to start that way, or
`con.transparent = False` from Python. All three leave a plain DuckDB session
that answers exactly as it did before.

## When GPU memory is full

The budget is a hard admission bound, not a target. A set whose estimated size
would put the total over it is refused **before** the upload, its statements
keep running on DuckDB, and `last_rewrite()["reason"]` is `memory` with the
arithmetic in `["error"]` — *"900 MiB resident + about 300 MiB needed > 1024
MiB, and this set is worth 0.012 ms/s (12.9 per GiB) against 0.030 ms/s for
&lt;the set that would have to go&gt;"*.

What is kept under pressure is decided by measured value per byte, not by
recency. A set's value is the DuckDB time it has saved, decayed over about
five minutes; divided by its bytes that is the density admission compares.
Eviction only runs when the candidate is worth at least 25% more than what it
would displace, a set younger than 60 seconds that has not earned anything yet
is protected unless the candidate is twice the median density, and a set in
use by a running operator, a set another resident set was derived from, and
anything you uploaded by hand are never evicted at all. A refused set is
retried after 30 seconds — and because statements that ran on DuckDB still
feed the value model, a set that was refused once can be admitted later on its
own merit.

## Reporting a bug

[github.com/singhpratech/duckdbgpumetaldbram/issues](https://github.com/singhpratech/duckdbgpumetaldbram/issues).
The two most useful things to paste are `SELECT gpu_build_info();` and, for a
statement that went the wrong way, the whole of `.gpu`. A statement that
returns *different rows* from DuckDB is the bug we most want to hear about.
