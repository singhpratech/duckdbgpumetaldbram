"""`python -m gpudb` — the same shell as the `gpudb` console script."""
import sys

from ._shell import main

if __name__ == "__main__":
    sys.exit(main())
