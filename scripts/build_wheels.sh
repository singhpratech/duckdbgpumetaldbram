#!/usr/bin/env bash
# build_wheels.sh — build the pip distributions for the python/ wrapper.
#
#   scripts/build_wheels.sh <built .duckdb_extension> [platform tag]
#
# Produces three files in python/dist/:
#
#   1. duckdb_gpudb-<v>-py3-none-<tag>.whl   — carries the extension binary in
#      gpudb/_ext/, so `pip install duckdb-gpudb` on that platform is enough to
#      get the GPU path. The wrapper prefers a source checkout's own fresh
#      build over this copy (connection.py::_find_extension). On Linux the same
#      directory also carries libgomp.so.1 and its licence, because the
#      extension links OpenMP dynamically and a minimal userland has no copy.
#   2. duckdb_gpudb-<v>-py3-none-any.whl     — no binary. The wrapper then falls
#      back to `LOAD gpudb`, the copy DuckDB installed from the community
#      registry, and says so when there is none.
#   3. duckdb_gpudb-<v>.tar.gz               — sdist, no binary.
#
# Why bundle at all: the package and the extension are released on different
# clocks. The package is published before the registry serves the matching
# extension build, and a registry build older than the client is declined by
# the catalogue check in _probe_extension — so without a bundled copy a
# pip-only install would run every statement on DuckDB.
#
# The platform tag:
#   macOS  — derived from the binary itself (LC_BUILD_VERSION `minos` via
#            `otool -l`, floored at 11.0) and its architecture (`lipo -archs`).
#            Pass a tag explicitly to override.
#   Linux  — REQUIRED as the second argument. The manylinux/glibc floor of a
#            binary depends on the toolchain it was built with; this script
#            does not guess it.
#
# Nothing is uploaded here. `twine check` runs when twine is available.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
PY_DIR="$ROOT/python"
EXT_DIR="$PY_DIR/gpudb/_ext"
DIST="$PY_DIR/dist"

die() { echo "build_wheels.sh: $*" >&2; exit 1; }

[ $# -ge 1 ] || die "usage: scripts/build_wheels.sh <built .duckdb_extension> [platform tag]"

BIN="$1"
TAG="${2:-}"
[ -f "$BIN" ] || die "no such file: $BIN"
case "$BIN" in
    *.duckdb_extension) ;;
    *) die "not an extension binary (expected *.duckdb_extension): $BIN" ;;
esac
BIN="$(cd "$(dirname "$BIN")" && pwd)/$(basename "$BIN")"

# The interpreter that runs the build. It needs `build`, `setuptools` >= 61 and
# `wheel` (the build runs with --no-isolation, so nothing is fetched and the
# versions used are the ones you can see). Override with PYTHON=/path/to/python.
PYTHON="${PYTHON:-python3}"
command -v "$PYTHON" >/dev/null || die "no $PYTHON on PATH"
"$PYTHON" -c "import build, setuptools, wheel" 2>/dev/null \
    || die "$PYTHON is missing build/setuptools/wheel (pip install build setuptools wheel)"

# ---------------------------------------------------------------- platform tag
if [ -z "$TAG" ]; then
    case "$(uname -s)" in
        Darwin)
            # The minimum macOS the binary itself declares. A wheel may not
            # claim to install on anything older than what the loader will
            # accept, so the floor is the larger of 11.0 and this value.
            MINOS="$(otool -l "$BIN" | awk '/LC_BUILD_VERSION/{f=1} f&&/ minos /{print $2; exit}')"
            [ -n "$MINOS" ] || die "could not read LC_BUILD_VERSION minos from $BIN"
            ARCH="$(lipo -archs "$BIN" | tr -d ' ')"
            case "$ARCH" in
                arm64|x86_64) ;;
                *) die "unexpected architecture '$ARCH' in $BIN (pass a tag explicitly)" ;;
            esac
            TAG="$("$PYTHON" - "$MINOS" "$ARCH" <<'PY'
import sys
minos, arch = sys.argv[1], sys.argv[2]
parts = [int(p) for p in minos.split(".")] + [0, 0]
major, minor = parts[0], parts[1]
if (major, minor) < (11, 0):      # the floor this project publishes at
    major, minor = 11, 0
if major >= 11:                   # macOS 11+ wheel tags use <major>_0
    minor = 0
print("macosx_%d_%d_%s" % (major, minor, arch))
PY
)"
            echo "==> macOS minimum from the binary: $MINOS ($ARCH) -> $TAG"
            ;;
        *)
            die "on $(uname -s) the platform tag must be given explicitly, e.g.
       scripts/build_wheels.sh <binary> manylinux_2_28_x86_64
     (the manylinux/glibc floor depends on the build toolchain; this script does not guess)"
            ;;
    esac
fi

echo "==> binary:       $BIN"
echo "==> platform tag: $TAG"

# ------------------------------------------------- Linux: the OpenMP companion
# The extension links OpenMP dynamically, so it NEEDs libgomp.so.1, and a
# minimal userland has no copy — on a bare ubuntu:22.04 the LOAD fails with
# "libgomp.so.1: cannot open shared object file". The wheel therefore carries
# one, which the extension finds through the $ORIGIN runpath it is linked with
# (src/extension/CMakeLists.txt). A process that already has libgomp mapped
# keeps using that one: the sonames match, so nothing is loaded twice.
#
# The copy has to come from the machine that BUILT the extension, not from
# whichever machine runs this script. libgomp carries its own glibc floor
# (2.38 on a 24.04 host, 2.34 in the 22.04 release container), and bundling the
# wrong one raises the wheel's real floor above its tag without changing the
# tag — a wheel that installs cleanly and then cannot load. So: taken from
# beside the binary, and checked against the tag below.
LIBGOMP=""
LIBGOMP_LICENSE=""
if [ "$(uname -s)" = "Linux" ]; then
    BIN_DIR="$(dirname "$BIN")"
    LIBGOMP="${GPUDB_LIBGOMP:-$BIN_DIR/libgomp.so.1}"
    LIBGOMP_LICENSE="${GPUDB_LIBGOMP_LICENSE:-$BIN_DIR/LICENSE.libgomp}"
    [ -f "$LIBGOMP" ] || die "no libgomp.so.1 beside the binary ($BIN_DIR).
     The Linux wheel bundles the libgomp of the build environment. Copy it out
     of the same image the extension was built in, with its licence:
       docker run --rm -v \"$BIN_DIR\":/out <build image> bash -c \\
         'cp \"\$(readlink -f /usr/lib/x86_64-linux-gnu/libgomp.so.1)\" /out/libgomp.so.1; \\
          cp /usr/share/doc/libgomp1/copyright /out/LICENSE.libgomp'
     or point GPUDB_LIBGOMP / GPUDB_LIBGOMP_LICENSE at them."
    [ -f "$LIBGOMP_LICENSE" ] || die "no LICENSE.libgomp beside the binary ($BIN_DIR).
     libgomp is GPL with the GCC Runtime Library Exception; the wheel
     redistributes the binary, so it carries the licence next to it."
    echo "==> libgomp:      $LIBGOMP"
    echo "==> its licence:  $LIBGOMP_LICENSE"
fi

# ------------------------------------------------------------- the glibc floor
# A manylinux tag promises the glibc of the oldest userland the wheel may
# install on. Nothing in the packaging enforces that, so check it here against
# what the ELF objects actually ask for.
if [ "$(uname -s)" = "Linux" ]; then
    command -v objdump >/dev/null || die "objdump not found (install binutils);
     it is what checks the glibc floor of the binaries against $TAG."
    TAG_GLIBC=""
    case "$TAG" in
        manylinux_*_*_*) TAG_GLIBC="$(echo "$TAG" | sed -E 's/^manylinux_([0-9]+)_([0-9]+)_.*/\1.\2/')" ;;
    esac
    if [ -z "$TAG_GLIBC" ]; then
        echo "==> $TAG names no glibc version; skipping the floor check"
    else
        check_floor() {                        # <file> <label>
            local floor version
            floor="$(objdump -T "$1" 2>/dev/null | grep -oE 'GLIBC_[0-9]+\.[0-9]+' | sort -u -V | tail -n1)"
            [ -n "$floor" ] || die "could not read a GLIBC symbol version from $1"
            version="${floor#GLIBC_}"
            if [ "$(printf '%s\n%s\n' "$version" "$TAG_GLIBC" | sort -V | tail -n1)" != "$TAG_GLIBC" ]; then
                die "$2 needs $floor, but $TAG promises glibc $TAG_GLIBC.
     The wheel would install on userlands it cannot load on. Build in an older
     image, or tag the wheel manylinux_${version%%.*}_${version#*.}_x86_64."
            fi
            echo "==> $2: glibc floor $floor (tag allows $TAG_GLIBC)"
        }
        check_floor "$BIN" "extension"
        check_floor "$LIBGOMP" "libgomp"
    fi
fi

clean() {
    rm -rf "$PY_DIR/build" "$PY_DIR"/*.egg-info
}
rm -rf "$DIST" "$EXT_DIR"
clean

# ------------------------------------------------------- 1. the platform wheel
mkdir -p "$EXT_DIR"
cp "$BIN" "$EXT_DIR/"
if [ -n "$LIBGOMP" ]; then
    cp "$LIBGOMP" "$EXT_DIR/libgomp.so.1"
    cp "$LIBGOMP_LICENSE" "$EXT_DIR/LICENSE.libgomp"
fi
echo "==> building the platform wheel (binary bundled)"
( cd "$PY_DIR" && GPUDB_WHEEL_PLAT="$TAG" "$PYTHON" -m build --wheel --no-isolation --outdir dist )
clean

[ "$(ls "$DIST"/*.whl | wc -l | tr -d ' ')" = "1" ] || die "expected exactly one platform wheel in $DIST"
PLAT_WHEEL="$(ls "$DIST"/*.whl)"

# ------------------------------------- 2. + 3. the pure wheel and the sdist
rm -rf "$EXT_DIR"
echo "==> building the py3-none-any wheel and the sdist (no binary)"
( cd "$PY_DIR" && "$PYTHON" -m build --no-isolation --outdir dist )
clean

# ---------------------------------------------------------------- verification
ANY_WHEEL="$(ls "$DIST"/*-py3-none-any.whl)"
SDIST="$(ls "$DIST"/*.tar.gz)"

count_ext_whl() { unzip -l "$1" | grep -c '\.duckdb_extension$' || true; }
count_ext_tar() { tar -tzf "$1" | grep -c '\.duckdb_extension$' || true; }

n="$(count_ext_whl "$PLAT_WHEEL")"
[ "$n" = "1" ] || die "the platform wheel carries $n extension binaries, expected exactly 1: $PLAT_WHEEL"
case "$(basename "$PLAT_WHEEL")" in
    *-py3-none-"$TAG".whl) ;;
    *) die "the platform wheel is not tagged py3-none-$TAG: $(basename "$PLAT_WHEEL")" ;;
esac

n="$(count_ext_whl "$ANY_WHEEL")"
[ "$n" = "0" ] || die "the py3-none-any wheel carries $n extension binaries, expected none: $ANY_WHEEL"

n="$(count_ext_tar "$SDIST")"
[ "$n" = "0" ] || die "the sdist carries $n extension binaries, expected none: $SDIST"

# The Linux platform wheel carries exactly one libgomp and its licence; the
# portable distributions carry neither.
if [ -n "$LIBGOMP" ]; then
    count_gomp_whl() { unzip -l "$1" | grep -c '/libgomp\.so\.1$' || true; }

    n="$(count_gomp_whl "$PLAT_WHEEL")"
    [ "$n" = "1" ] || die "the platform wheel carries $n copies of libgomp.so.1, expected exactly 1: $PLAT_WHEEL"

    n="$(unzip -l "$PLAT_WHEEL" | grep -c '/LICENSE\.libgomp$' || true)"
    [ "$n" = "1" ] || die "the platform wheel carries $n libgomp licence files, expected exactly 1: $PLAT_WHEEL"

    n="$(count_gomp_whl "$ANY_WHEEL")"
    [ "$n" = "0" ] || die "the py3-none-any wheel carries $n copies of libgomp.so.1, expected none: $ANY_WHEEL"

    n="$(tar -tzf "$SDIST" | grep -c '/libgomp\.so\.1$' || true)"
    [ "$n" = "0" ] || die "the sdist carries $n copies of libgomp.so.1, expected none: $SDIST"
fi

echo
echo "==> $(basename "$PLAT_WHEEL")"
unzip -l "$PLAT_WHEEL"
echo "==> $(basename "$ANY_WHEEL")"
unzip -l "$ANY_WHEEL"
echo "==> $(basename "$SDIST")"
tar -tzf "$SDIST"

if command -v twine >/dev/null; then
    echo
    echo "==> twine check"
    twine check "$DIST"/*
else
    echo
    echo "==> twine not on PATH; skipping metadata check"
fi

echo
echo "==> done. Nothing was uploaded. Artifacts:"
ls -l "$DIST"
