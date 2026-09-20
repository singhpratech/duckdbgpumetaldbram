"""Packaging shim. pyproject.toml holds the metadata (setup.cfg mirrors it for
setuptools older than 61); this file exists only so that a wheel which carries
the extension binary gets an honest tag.

`gpudb/_ext/` is empty in a source checkout and in the sdist.
`scripts/build_wheels.sh` copies one built `.duckdb_extension` into it, builds
the platform wheel, removes it again, and then builds the pure
`py3-none-any` wheel and the sdist with no binary in them.

When that directory holds a binary the distribution is marked impure and the
wheel is tagged `py3-none-<platform>`:

  * `<platform>` is real — the file is Mach-O or ELF for one architecture and
    one minimum OS, and a wheel that carries it must not install anywhere else.
    The tag comes from GPUDB_WHEEL_PLAT (set by the build script, which reads
    the binary's own minimum OS version) or from `--plat-name`.
  * `py3`/`none` are also real — there is no Python C extension inside, so the
    same file works on every CPython 3.x. Left to itself setuptools would stamp
    the interpreter and ABI of whichever Python ran the build, which would be a
    narrower claim than the contents justify.
"""
import glob
import os

from setuptools import setup

_HERE = os.path.dirname(os.path.abspath(__file__))
_BUNDLED = sorted(glob.glob(os.path.join(_HERE, "gpudb", "_ext", "*.duckdb_extension")))

cmdclass = {}

if _BUNDLED:
    try:                                     # setuptools >= 70.1 vendors the command
        from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
    except ImportError:                      # older setuptools: the wheel package owns it
        from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):         # noqa: N801  (distutils command naming)
        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = False        # carries a platform binary

        def get_tag(self):
            _py, _abi, plat = _bdist_wheel.get_tag(self)
            return "py3", "none", os.environ.get("GPUDB_WHEEL_PLAT") or plat

    cmdclass["bdist_wheel"] = bdist_wheel

setup(cmdclass=cmdclass)
