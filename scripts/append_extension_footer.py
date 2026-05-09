#!/usr/bin/env python3
"""
append_extension_footer.py — best-effort attempt to append a DuckDB
extension metadata footer.

⚠️ STATUS (2026-05-09): the manual format below is REJECTED by DuckDB
1.4.x ("The metadata at the end of the file is invalid"). The exact
field semantics have shifted in recent DuckDB versions and reverse-
engineering them is fragile.

THE OFFICIAL PATHS THAT WORK:
  1. Submit to https://github.com/duckdb/community-extensions — their
     CI builds .duckdb_extension files for all platforms with the right
     metadata + signing. After merge: INSTALL gpudb FROM community;
  2. Clone https://github.com/duckdb/extension-template, adapt our build
     to use its CMake macros which generate the footer correctly.
     Then: SET allow_unsigned_extensions=true at startup; LOAD '...';

This script is kept as a reference for the FORMAT we attempted, in case
someone wants to dig further or DuckDB documents the format publicly.

NOT FOR PRODUCTION USE.

Usage:
    python3 scripts/append_extension_footer.py \\
        --in build-linux/src/extension/libgpudb_duckdb.so \\
        --out build-linux/src/extension/gpudb.duckdb_extension \\
        --name gpudb \\
        --extension-version 0.1.0 \\
        --duckdb-version 1.4.0 \\
        --platform linux_amd64

Format reference (DuckDB extension format v1):
    The footer appended to the .so consists of:
      - 256 bytes signature  (zeros for unsigned extensions)
      - 8 fields x 32 bytes  (256 bytes of metadata)
    Total appended: 512 bytes.

    Field layout (each field is exactly 32 bytes, NUL-padded):
      Field 0: format version magic (literal "4")
      Field 1: platform (e.g. "linux_amd64", "osx_arm64")
      Field 2: duckdb version (e.g. "v1.4.0")
      Field 3: extension version (e.g. "v0.1.0")
      Field 4: signature mode (zeros for unsigned)
      Field 5: extension ABI type ("C_STRUCTS" for C API extensions)
      Field 6: reserved
      Field 7: extension name (e.g. "gpudb")

NOTE: The exact field semantics have shifted between DuckDB versions.
This script reflects DuckDB 1.4.x. If LOAD fails with a metadata error
on a newer DuckDB, switch to the official extension-template build.
"""
import argparse
import shutil
import struct
import sys
from pathlib import Path

FIELD_SIZE = 32
NUM_FIELDS = 8
SIGNATURE_SIZE = 256
FORMAT_VERSION = b"4"


def pad_field(value: bytes) -> bytes:
    if len(value) > FIELD_SIZE:
        raise ValueError(f"field value too long ({len(value)} > {FIELD_SIZE}): {value!r}")
    return value + b"\0" * (FIELD_SIZE - len(value))


def build_metadata(name: str, ext_version: str, duckdb_version: str,
                   platform: str, abi_type: str = "C_STRUCTS") -> bytes:
    fields = [
        FORMAT_VERSION,
        platform.encode("ascii"),
        duckdb_version.encode("ascii"),
        ext_version.encode("ascii"),
        b"\0",                       # signature mode = unsigned
        abi_type.encode("ascii"),
        b"\0",                       # reserved
        name.encode("ascii"),
    ]
    if len(fields) != NUM_FIELDS:
        raise RuntimeError("internal: wrong field count")
    return b"".join(pad_field(f) for f in fields)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="src", required=True, help="input .so")
    p.add_argument("--out", dest="dst", required=True, help="output .duckdb_extension")
    p.add_argument("--name", required=True, help="extension name (e.g. gpudb)")
    p.add_argument("--extension-version", default="v0.1.0")
    p.add_argument("--duckdb-version", default="v1.4.0")
    p.add_argument("--platform", default="linux_amd64")
    p.add_argument("--abi-type", default="C_STRUCTS")
    args = p.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.exists():
        print(f"input not found: {src}", file=sys.stderr)
        return 1

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

    metadata = build_metadata(
        name=args.name,
        ext_version=args.extension_version,
        duckdb_version=args.duckdb_version,
        platform=args.platform,
        abi_type=args.abi_type,
    )
    if len(metadata) != FIELD_SIZE * NUM_FIELDS:
        print(f"internal: bad metadata length {len(metadata)}", file=sys.stderr)
        return 1

    signature = b"\0" * SIGNATURE_SIZE

    with dst.open("ab") as f:
        f.write(metadata)
        f.write(signature)

    print(f"wrote {dst}  ({src.stat().st_size} + {len(metadata)+len(signature)} footer = "
          f"{dst.stat().st_size} bytes)")
    print(f"  name={args.name}  ext_version={args.extension_version}  "
          f"duckdb_version={args.duckdb_version}  platform={args.platform}")
    print()
    print("Test with:")
    print(f"  duckdb -unsigned -c \"LOAD '{dst.resolve()}'; "
          "SELECT gpu_sum(range::BIGINT) FROM range(10);\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
