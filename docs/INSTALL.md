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

**Route 1 — `pip`, which on a supported platform is the whole install.**

```bash
pip install duckdb-gpudb          # the `gpudb` command, the gpudb module, and the extension
```

On **Apple Silicon, macOS 15 or later** and on **x86-64 Linux with glibc 2.34
or newer** (Ubuntu 22.04 and later) `pip` installs a platform wheel that carries
the v0.7.0 extension binary inside the package, in `gpudb/_ext/`. Nothing else
is needed: no `INSTALL`, no build, no `GPUDB_EXTENSION_PATH`. One binary, and it
has been shown loading under both DuckDB 1.4.5 and 1.5.5 from a clean install —
which is what building against the stable C API buys.

On any other platform `pip` installs the `py3-none-any` wheel, which carries no
binary, and the extension has to come from DuckDB itself:

```sql
INSTALL gpudb FROM community;     -- in any DuckDB ≥ 1.5.5 client
LOAD gpudb;
```

It is signed; no flags. That is also the route for any DuckDB client that wants
the explicit `gpu_*` functions without the wrapper. The `duckdb` module `pip`
pulls in (or the one you already have) must then be a version the registry
publishes a gpudb build for — **Supported versions**, at the end of this
section, says which and how to pin it — and `FORCE INSTALL gpudb FROM
community;` or `UPDATE EXTENSIONS;` is what replaces an already-installed copy
with the newest the registry has.

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

**The lookup order**, whichever route you took:

1. an explicit `gpudb.connect(extension="…")`,
2. `GPUDB_EXTENSION_PATH`,
3. a `build-macos/` / `build-linux/` beside a source checkout,
4. the copy bundled in the installed package (`gpudb/_ext/`, present only in a
   platform wheel),
5. whatever DuckDB itself has installed — `LOAD gpudb`.

A checkout's own build comes *before* the bundled copy on purpose: someone who
has just built the extension is testing that binary, and an installed wheel
shadowing it silently would be the worst kind of wrong. `LOAD gpudb` is last
because it is the only entry that can hand back something older than the
client. If nothing usable is found — or what is found is older than the client
— the banner's `transparent:` line says so, `con.extension_note` carries the
same sentence, and every statement simply runs on DuckDB.

**Supported versions.** Python ≥ 3.9 and the `duckdb` module ≥ 1.4 — the
wrapper's own requirements, and they are *not* the same as the registry's. A
bundled or downloaded binary needs only DuckDB ≥ 1.2, because the loadable
extension is built against the stable C API v1.2.0; the wheel's copy has been
run under both 1.4.5 and 1.5.5. The registry, by contrast, builds gpudb
separately for each DuckDB version from 1.5.5 on, so a `duckdb` module that
satisfies `pip` can still be a version the registry has nothing to install for:
on DuckDB 1.4.5, `INSTALL gpudb FROM community` is a 404. Pin it with `pip
install "duckdb==1.5.5"` if the extension is coming from the registry.

**macOS: 15.0 or later, on Apple silicon.** That is the floor the shipped
binaries declare (Mach-O `minos 15.0`), and it is the floor of the binary the
community registry has been serving, so it is the one this project has evidence
for: a 15.0 binary has shipped and has run. A lower target compiles — 14.0
builds with no unguarded-availability warnings — but the shader-compile paths
pick their Metal language version at run time (MSL 3.2 on macOS 15 and later,
MSL 3.1 below, `src/backends/metal/metal_groupby.mm`) and that 3.1 branch has
never executed anywhere, because every machine gpudb has run on is 15 or newer.
Nothing here was tested on a macOS older than the one that built it. Building
needs the macOS 15 SDK, which is where `MTLLanguageVersion3_2` comes from (CI
builds on `macos-15`).

**Linux: x86-64, glibc 2.34 or newer** — the release binary and the platform
wheel are built on Ubuntu 22.04, so Ubuntu 20.04 and other older userlands are
not supported. The extension links libstdc++ and libgcc statically, so it
carries no `GLIBCXX_`/`CXXABI_` symbol-version floor from the build machine, and
the CUDA runtime is linked statically too — the only driver-side dependency is
`libcuda.so`. For the GPU it needs an **NVIDIA driver R525 or newer** and a card
in the **sm_75 – sm_90** range (see the [CUDA
requirements](#cuda-requirements-build-from-source-on-linux) table; a binary
built with CUDA 13 needs R580+, one built with CUDA 12.x reaches the GPU on
R525+). On a machine with no NVIDIA driver at all it still loads and falls back
to the CPU backend cleanly. An extension installed from the community registry
on Linux is a **CPU-only build** — it reports `compiled=cpu` — so for the CUDA
backend take the release binary or build from source.

**Linux also needs `libgomp.so.1` at load time.** The extension links OpenMP
dynamically, so the shared library has to be on the machine:

```bash
apt install libgomp1     # Debian / Ubuntu
dnf install libgomp      # Fedora / RHEL
```

It is already present on most desktop, CI and notebook images. In a minimal
container it is not, and `LOAD` then fails with `libgomp.so.1: cannot open
shared object file`. This applies to both the community-extensions build and the
binary on the GitHub release page. The Python wheel bundles its own copy and
needs nothing installed.

## Platforms and install

| | Metal (Apple Silicon) | CUDA (NVIDIA) | No GPU |
|---|---|---|---|
| Plain SQL on the GPU, through the shell or `gpudb.connect()` | yes | yes, on by default — from a CUDA build, see below | no — everything runs on DuckDB |
| Explicit `gpu_*` functions | yes | yes | yes, on the CPU backend, same answers |
| From the community registry | yes | a registry Linux binary may report `compiled=cpu`; `SELECT gpu_build_info();` is what answers this for whichever binary is in front of you | yes |

On NVIDIA hardware every operator the transparent path needs is implemented —
exact `GROUP BY`, the `WHERE` mask, the global aggregate and the materialised
join — and the path is **on by default**. On an RTX 4090 Laptop the unit suite
is **752 / 752**, the SQL suite **225 passing, 0 failing**, the wrapper suite
**1258 passing, 0 failing**, and TPC-H at SF1 is **17 of 22 queries on the
device, 0 rows differing from native** — the same coverage and the same five
declines as Metal at that scale factor ([BENCHMARK.md](../BENCHMARK.md), *the
CUDA exact path on by default*).

What turned it on was a measurement: the full gate on that box, at the
wrapper's own memory budget, ran **1630 cells with 0 slower than native and 0
differing**, minimum ratio 1.07×. Two caveats are worth carrying. The
thresholds the wrapper decides with are the Metal-measured ones, **verified on
one CUDA machine** rather than measured for every GPU. And TPC-H Q1 sits at
parity on that box — it has measured either side of 1.0× across runs of the
same build — which is the case the per-process measured rule exists to settle.
`GPUDB_CUDA_EXACT=0` turns the path off without a rebuild, leaving a CUDA
machine the explicit `gpu_*` functions and plain SQL on DuckDB: correct, with
no speed-up.

**This needs a binary that carries CUDA.** A Linux binary installed from the
community registry may report `compiled=cpu` and have no CUDA in it at all, and
no environment variable changes that — take the release binary (Option B) or
build from source. `SELECT gpu_build_info();` is what tells them apart.

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
| `libgomp.so.1: cannot open shared object file` on Linux | The OpenMP runtime the extension links dynamically is not on the machine — usually a minimal container. `apt install libgomp1` (Debian/Ubuntu) or `dnf install libgomp` (Fedora/RHEL). The Python wheel bundles its own copy and never hits this. |
| `IO Error: Extension "…" could not be loaded because its signature is either missing or invalid` | A locally built binary. Start DuckDB with `-unsigned`, or from Python pass `config={"allow_unsigned_extensions": "true"}`. The `gpudb` shell already does this for a build it found itself. |
| `transparent: not on this build` | The extension loaded but has no exact operators — a CPU-only build (a registry Linux binary can be one), or a CUDA build started with `GPUDB_CUDA_EXACT=0`. |
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

The budget is a hard admission bound, not a target, and what it is compared
with is the **physical** total: every store column counted once, plus whatever
a set holds of its own (`con.memory()["bytes"]`). A set whose estimated size
would put that total over the budget is refused **before** the upload, its
statements keep running on DuckDB, and `last_rewrite()["reason"]` is `memory`
with the arithmetic in `["detail"]` (and verbatim in `["error"]`) — *"900 MiB
resident + about 300 MiB needed &gt; 1024 MiB, and this set is worth 0.012 ms/s
(12.9 per GiB) against 0.030 ms/s for &lt;the set that would have to go&gt;"*.

**A refusal is remembered.** Asking again costs an upload and gets the same
answer, so it is not asked again until something that could change it changes:
the data, the budget, the resident population, or the anti-thrash window
lapsing. Fifty executions of a statement whose set cannot fit are one upload
attempt, not fifty.

**An upload the device itself refuses fails cleanly.** The set is not quietly
placed in host memory — that would be slower than plain DuckDB and would leave
a table's columns split across two backends — so the statement is answered by
DuckDB and the set is refused like any other.

**Working memory is not resident**, so no budget over resident bytes can see a
reduce's scratch or a sort's temporaries. A statement that runs out of it is
answered by DuckDB, its sets are refused rather than retried, and where the
backend reports how much it wanted against how much was free, that shortfall is
held back as headroom from then on.

What is kept under pressure is decided by measured value per byte, not by
recency. A set's value is the DuckDB time it has saved, decayed over about
five minutes; divided by its bytes that is the density admission compares.
Eviction only runs when the candidate is worth at least 25% more than what it
would displace, a set younger than 60 seconds that has not earned anything yet
is protected unless the candidate is twice the median density, and a set in
use by a running operator, a set another resident set was derived from, and
anything you uploaded by hand are never evicted at all. Because statements that
ran on DuckDB still feed the value model, a set refused once can be admitted
later on its own merit.

## Reporting a bug

[github.com/singhpratech/duckdbgpumetaldbram/issues](https://github.com/singhpratech/duckdbgpumetaldbram/issues).
The two most useful things to paste are `SELECT gpu_build_info();` and, for a
statement that went the wrong way, the whole of `.gpu`. A statement that
returns *different rows* from DuckDB is the bug we most want to hear about.
