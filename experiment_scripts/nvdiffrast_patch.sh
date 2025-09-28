#!/usr/bin/env bash
set -euo pipefail

# 0) Optional: upgrade PyTorch to nightly with CUDA 13.x (supports compute 12.0)
# pip install -U --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130

# 1) Install CUDA Toolkit 13.0 (nvcc) on Ubuntu 24.04+
source /etc/os-release
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/${ID}${VERSION_ID/./}/x86_64/cuda-keyring_1.1-1_all.deb -o /tmp/cuda-keyring.deb || \
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/${ID}${VERSION_CODENAME}/x86_64/cuda-keyring_1.1-1_all.deb -o /tmp/cuda-keyring.deb
sudo dpkg -i /tmp/cuda-keyring.deb
sudo apt-get update -y
sudo apt-get install -y cuda-toolkit-13-0

# 2) Set CUDA and arch env for build
export CUDA_HOME=/usr/local/cuda-13.0
export CUDA_PATH=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST=12.0

# 3) Patch nvdiffrast to cache the loaded module object (avoids import-by-name failure)
python - <<'PY'
import sys, importlib.util, pathlib, io
import site, re

# Locate installed ops.py
candidates = []
for p in site.getsitepackages() + [site.getusersitepackages()]:
    ops = pathlib.Path(p) / 'nvdiffrast' / 'torch' / 'ops.py'
    if ops.exists():
        candidates.append(ops)
if not candidates:
    print("nvdiffrast not found in site-packages", file=sys.stderr)
    sys.exit(1)
ops_path = candidates[0]

src = ops_path.read_text()
# Prefix the load call with 'mod =' and assign cached plugin from the module, not import-by-name.
src = re.sub(
    r"(\s*)torch\.utils\.cpp_extension\.load\(",
    r"\1mod = torch.utils.cpp_extension.load(",
    src,
    count=1,
)
src = src.replace(
    "_cached_plugin[gl] = importlib.import_module(plugin_name)",
    "_cached_plugin[gl] = mod",
)

ops_path.write_text(src)
print(f"Patched: {ops_path}")
PY

# 4) Force-compile the CUDA plugin (sm_120) and verify context creation
python - <<'PY'
import os, shutil, torch
import nvdiffrast.torch as dr

print("nvcc:", shutil.which("nvcc"))
ctx = dr.RasterizeCudaContext()  # triggers build/load
print("RasterizeCudaContext OK")
PY

# 5) Smoke test rasterization on GPU
python - <<'PY'
import torch, nvdiffrast.torch as dr
ctx = dr.RasterizeCudaContext()
pos = torch.randn(1, 3, 4, device='cuda')
tri = torch.tensor([[0,1,2]], dtype=torch.int32, device='cuda')
rast, _ = dr.rasterize(ctx, pos, tri, [64,64])
print("rasterize ran:", rast.shape)
PY
