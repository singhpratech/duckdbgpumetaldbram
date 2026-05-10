#!/usr/bin/env python3
"""
append_extension_footer.py — append the DuckDB extension metadata footer
so the .so/.dylib loads via DuckDB's `LOAD '<path>.duckdb_extension'`.

Verified against the stock DuckDB CLI v1.5.2 on osx_arm64 — produces a
file that loads end-to-end and runs `gpu_sum / gpu_min / gpu_max`.

Usage (C API extension — what we build):
    python3 scripts/append_extension_footer.py \\
        --in build-macos/src/extension/libgpudb_duckdb.dylib \\
        --out build-macos/src/extension/gpudb_duckdb.duckdb_extension \\
        --extension-version v0.1.1 \\
        --c-api-version v1.2.0 \\
        --platform osx_arm64

Then load (DuckDB CLI must be invoked with -unsigned for community-built
extensions, or the user can SET allow_unsigned_extensions=true):
    duckdb -unsigned -c "LOAD '/abs/path/gpudb_duckdb.duckdb_extension'; \\
                         SELECT gpu_sum(range::BIGINT) FROM range(10);"

Footer format (reverse-engineered against DuckDB v1.5.2; matches the
behavior of duckdb/main/extension/extension_load.cpp::ParseExtensionMetaData):

    Last 512 bytes of the file are the footer:
      - 256 bytes  metadata block: 8 fields x 32 bytes, NUL-padded ASCII
          Field 0:  zeros (reserved)
          Field 1:  zeros (reserved)
          Field 2:  zeros (reserved)
          Field 3:  ABI type
                    "C_STRUCT" -> entrypoint <basename>_init_c_api  (what we use)
                    "C_STRUCT_UNSTABLE" -> same entrypoint, no version check
                    "CPP" -> entrypoint <basename>_duckdb_cpp_init  (DuckDB-internal)
                    empty -> CPP
          Field 4:  extension version  (informational; e.g. "v0.1.1")
          Field 5:  semantics depends on Field 3:
                    C_STRUCT -> DuckDB C API version (must satisfy the loader's
                                acceptable range, currently "v1.x.y" with x<=2)
                    CPP      -> DuckDB version (must equal the loader's version)
          Field 6:  platform           ("osx_arm64", "linux_amd64", ...)
          Field 7:  format magic       ("4" for the current footer format)
      - 256 bytes  signature block: zeros for unsigned extensions

The 512-byte footer is appended directly to the .so/.dylib body — no
intermediate magic, no padding, no separator.

The basename used to derive the entrypoint name is the .duckdb_extension
file's basename (without the .duckdb_extension suffix). Our shared lib
exports `gpudb_duckdb_init_c_api`, so the file MUST be named
`gpudb_duckdb.duckdb_extension`.
"""
import argparse
import shutil
import sys
from pathlib import Path

FIELD_SIZE = 32
NUM_FIELDS = 8
METADATA_SIZE = FIELD_SIZE * NUM_FIELDS   # 256
SIGNATURE_SIZE = 256
FOOTER_SIZE = METADATA_SIZE + SIGNATURE_SIZE   # 512
FORMAT_VERSION_MAGIC = b"4"


def pad_field(value: bytes) -> bytes:
    if len(value) > FIELD_SIZE:
        raise ValueError(f"field value too long ({len(value)} > {FIELD_SIZE}): {value!r}")
    return value + b"\0" * (FIELD_SIZE - len(value))


def build_metadata(ext_version: str, version_field5: str,
                   platform: str, abi_type: str = "C_STRUCT") -> bytes:
    fields = [
        b"\0",                                   # 0: reserved
        b"\0",                                   # 1: reserved
        b"\0",                                   # 2: reserved
        abi_type.encode("ascii"),                # 3: ABI type
        ext_version.encode("ascii"),             # 4: extension version
        version_field5.encode("ascii"),          # 5: C API version (C_STRUCT) or DuckDB version (CPP)
        platform.encode("ascii"),                # 6: platform
        FORMAT_VERSION_MAGIC,                    # 7: format magic
    ]
    if len(fields) != NUM_FIELDS:
        raise RuntimeError("internal: wrong field count")
    return b"".join(pad_field(f) for f in fields)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="src", required=True, help="input .so/.dylib")
    p.add_argument("--out", dest="dst", required=True,
                   help="output path; basename (minus .duckdb_extension) determines entrypoint name")
    p.add_argument("--extension-version", required=True, help="e.g. v0.1.1")
    p.add_argument("--c-api-version", help="DuckDB C API version, e.g. v1.2.0 (used when --abi-type=C_STRUCT)")
    p.add_argument("--duckdb-version", help="DuckDB version, e.g. v1.5.2 (used when --abi-type=CPP)")
    p.add_argument("--platform", required=True, help="osx_arm64 | osx_amd64 | linux_amd64 | linux_arm64")
    p.add_argument("--abi-type", default="C_STRUCT",
                   choices=["C_STRUCT", "C_STRUCT_UNSTABLE", "CPP"],
                   help="default C_STRUCT (matches gpudb_duckdb_init_c_api entrypoint)")
    args = p.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.exists():
        print(f"input not found: {src}", file=sys.stderr)
        return 1

    if args.abi_type in ("C_STRUCT", "C_STRUCT_UNSTABLE"):
        version_field5 = args.c_api_version or "v1.2.0"
    else:
        version_field5 = args.duckdb_version
        if not version_field5:
            print("--duckdb-version is required when --abi-type=CPP", file=sys.stderr)
            return 1

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

    metadata = build_metadata(
        ext_version=args.extension_version,
        version_field5=version_field5,
        platform=args.platform,
        abi_type=args.abi_type,
    )
    if len(metadata) != METADATA_SIZE:
        print(f"internal: bad metadata length {len(metadata)}", file=sys.stderr)
        return 1

    signature = b"\0" * SIGNATURE_SIZE

    with dst.open("ab") as f:
        f.write(metadata)
        f.write(signature)

    print(f"wrote {dst}  ({src.stat().st_size} + {FOOTER_SIZE} footer = "
          f"{dst.stat().st_size} bytes)")
    print(f"  abi={args.abi_type}  ext_version={args.extension_version}  "
          f"field5={version_field5}  platform={args.platform}")
    print()
    print("Test with:")
    print(f"  duckdb -unsigned -c \"LOAD '{dst.resolve()}'; "
          "SELECT gpu_sum(range::BIGINT) FROM range(10);\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
