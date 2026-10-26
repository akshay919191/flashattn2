#!/usr/bin/env bash
set -euo pipefail

echo "=== Installing system dependencies ==="
sudo apt install -y gcc-11 g++-11 cuda-nvcc-11-8 cuda-libraries-dev-11-8

echo "=== Setting environment variables ==="
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="$CUDA_HOME/bin:$PATH"
export CUDAHOSTCXX=/usr/bin/g++-11
export TORCH_CUDA_ARCH_LIST="8.6"

echo "=== Initializing Conda ==="
if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but was not found on PATH" >&2
  exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "=== Creating conda environment ==="
conda env create -f environment.yml
conda activate akku

echo "=== Installing Python dependencies ==="
python -m pip install -r requirements.txt

echo "=== Building CUDA extension ==="
python -m pip install -e . --no-build-isolation

echo "=== Done! Activate with: conda activate akku ==="
