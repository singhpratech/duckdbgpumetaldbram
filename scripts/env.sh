# env.sh — source this in your shell to put CUDA on PATH.
#   source ./scripts/env.sh
#
# Or add the two exports to ~/.bashrc once.

if [ -d /usr/local/cuda/bin ]; then
    export PATH=/usr/local/cuda/bin:$PATH
fi
if [ -d /usr/local/cuda/lib64 ]; then
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
fi

if command -v nvcc >/dev/null 2>&1; then
    echo "CUDA on PATH: $(which nvcc)  ($(nvcc --version | tail -1))"
else
    echo "nvcc not found"
fi
