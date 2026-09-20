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
requirements](../README.md#cuda-requirements-build-from-source-on-linux) table — the short
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
**730 / 730**, the SQL suite **224 passing, 0 failing**, and TPC-H at SF1 is
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

## Troubleshooting

| What you see | What it is |
|---|---|
| `backend: none — the extension is not loaded` | No extension found. Run `INSTALL gpudb FROM community; LOAD gpudb;` in DuckDB, or set `GPUDB_EXTENSION_PATH`. |
| `transparent: off — the loaded gpudb extension is older than this client: it does not provide …` | DuckDB has an older gpudb installed. `INSTALL` alone will not replace it — use `FORCE INSTALL gpudb FROM community;` (or `UPDATE EXTENSIONS;`), then `LOAD gpudb;`. |
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
