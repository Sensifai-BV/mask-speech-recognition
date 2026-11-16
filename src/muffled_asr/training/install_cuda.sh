#!/usr/bin/env bash
# Install the NeMo ASR training stack on Linux + NVIDIA CUDA.
#
# Training needs CUDA: NeMo's default transducer loss is warprnnt_numba, a
# Numba CUDA kernel. The pure-PyTorch fallback is a nested Python loop over the
# RNN-T lattice that NVIDIA itself marks "for debugging purposes only", so
# there is no practical CPU or Metal path for a 600M model.
#
#   ./install_cuda.sh [venv-dir]
#
# Override the torch build for your card if needed:
#   TORCH_CUDA=cu126 ./install_cuda.sh        # RTX 30/40 series
#   TORCH_CUDA=cu128 ./install_cuda.sh        # RTX 50 series (Blackwell), default

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${1:-.venv-train}"
TORCH_CUDA="${TORCH_CUDA:-cu128}"

echo "==> Checking for an NVIDIA GPU"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found. This script needs an NVIDIA GPU with drivers installed." >&2
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo
echo "==> System packages"
# libsndfile is what soundfile binds to; ffmpeg decodes anything that is not WAV.
if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y libsndfile1 ffmpeg build-essential python3-venv python3-dev
elif command -v dnf >/dev/null 2>&1; then
    sudo dnf install -y libsndfile ffmpeg gcc gcc-c++ python3-devel
else
    echo "WARNING: unknown package manager — install libsndfile, ffmpeg and a C toolchain yourself." >&2
fi

echo
echo "==> Virtualenv at ${VENV_DIR}"
python3 -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip wheel

echo
echo "==> PyTorch (${TORCH_CUDA})"
pip install torch torchaudio --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"

echo
echo "==> Build prerequisites for NeMo"
# NeMo compiles Cython extensions during install, so these must land first and
# cannot simply be listed alongside it in requirements.txt.
pip install Cython packaging

echo
echo "==> Training dependencies"
pip install -r "${HERE}/requirements.txt"

echo
echo "==> Verifying"
python - <<'PYCHECK'
import torch

print(f"torch            {torch.__version__}")
print(f"cuda available   {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device           {torch.cuda.get_device_name(0)}")
    print(f"capability       {'.'.join(map(str, torch.cuda.get_device_capability(0)))}")
    print(f"bf16 supported   {torch.cuda.is_bf16_supported()}")
else:
    raise SystemExit("ERROR: torch cannot see the GPU — check driver and CUDA build match.")

# The transducer loss is the thing that actually decides whether training works.
from nemo.collections.asr.losses.rnnt import RNNT_LOSS_RESOLVER

numba = RNNT_LOSS_RESOLVER["warprnnt_numba"]
print(f"warprnnt_numba   {'available' if numba.is_available else 'MISSING'}")
if not numba.is_available:
    raise SystemExit(f"ERROR: {numba.installation_msg}")

import nemo
print(f"nemo             {nemo.__version__}")
PYCHECK

echo
echo "Done. Activate with:  source ${VENV_DIR}/bin/activate"
