#!/usr/bin/env bash
# One command. Trains encoder adapters on the masked + noisy LibriSpeech corpus
# and leaves you with the best checkpoint by held-out WER.
#
#     ./train.sh
#
# Everything is preset for this project and this machine (RTX 4060 8 GB,
# i7-13650HX, 32 GB RAM). Override with environment variables if you need to:
#
#     STEPS=5000 ./train.sh
#     DATA=datasets/masked/other-corpus ./train.sh
#
# Runs from the project root regardless of where you invoke it from, and every
# path below is relative to that root.

set -euo pipefail

cd "${MSR_ROOT:-$(dirname "${BASH_SOURCE[0]}")/../../..}"

DATA="${DATA:-datasets/masked/librispeech-welding}"
NEMO_MODEL="${NEMO_MODEL:-models/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo}"
EXP_DIR="${EXP_DIR:-experiments}"
STEPS="${STEPS:-3000}"
GPU_MEM_FRAC="${GPU_MEM_FRAC:-0.92}"

step() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
die()  { printf '\nERROR: %s\n' "$1" >&2; exit 1; }

step "1/4  Environment"
[[ -d "${DATA}" ]] || die "dataset not found: ${DATA}"
[[ -f "${NEMO_MODEL}" ]] || die "base checkpoint not found: ${NEMO_MODEL}"
[[ -f nemo/examples/asr/asr_adapters/train_asr_adapter.py ]] || die "NeMo checkout not found at ./nemo"

if [[ -f .venv-train/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv-train/bin/activate
elif python3 -c "import nemo.collections.asr" >/dev/null 2>&1; then
    echo "  using the active environment (NeMo present)"
else
    die "no training environment. Run ./src/muffled_asr/training/install_cuda.sh first."
fi

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found; this needs a CUDA GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/  /'
python3 -c "
import sys, torch
if not torch.cuda.is_available():
    sys.exit('torch cannot see the GPU -- driver and CUDA build may not match')
print(f'  torch {torch.__version__}, bf16 {torch.cuda.is_bf16_supported()}')
from nemo.collections.asr.losses.rnnt import RNNT_LOSS_RESOLVER
cfg = RNNT_LOSS_RESOLVER['warprnnt_numba']
if not cfg.is_available:
    sys.exit(cfg.installation_msg)
print('  warprnnt_numba available')
" || die "environment check failed"

# Every Hydra override must use `++`, which sets a key whether or not NeMo's
# config already has it. A copy of run_adapter.sh predating that change fails
# minutes later with an opaque "not in struct" error, so catch it here.
if ! grep -q "save_best_model" src/muffled_asr/training/conf/adapter_masked.yaml \
   || ! grep -q "CKPT_EVERY \* ACCUM" src/muffled_asr/training/run_adapter.sh; then
    die "src/muffled_asr/training/run_adapter.sh is out of date on this machine.
       Re-sync it:  git pull (or rsync this repository's src/ from the machine that has it)"
fi
echo "  scripts up to date"

step "2/4  Manifests"
# --fix-paths rebases absolute paths written on another machine; a no-op when
# the dataset was built here.
for split in train dev; do
    python3 src/muffled_asr/training/preflight.py "${DATA}/${split}.json" --fix-paths >/dev/null \
        || die "${split}.json is not usable -- see the message above"
done
# test.json is not used for training, but eval_adapter.py reads it afterwards.
# Repair it now, so a dataset copied from another machine is fully usable here.
if [[ -f "${DATA}/test.json" ]]; then
    python3 src/muffled_asr/training/preflight.py "${DATA}/test.json" --fix-paths >/dev/null \
        || echo "  WARNING: test.json has unusable entries; eval_adapter.py will report them"
fi

# `find` on a missing directory exits non-zero, and under `set -e` that ends
# the script with no message at all. Guard every one of them.
LAST_CKPT=""
[[ -d "${EXP_DIR}" ]] && LAST_CKPT="$(find "${EXP_DIR}" -name '*-last.ckpt' 2>/dev/null | head -1 || true)"
if [[ -n "${LAST_CKPT}" ]]; then
    echo "  found ${LAST_CKPT}"
    echo "  training will RESUME from it. Use FRESH=1 to start over instead."
fi

step "3/4  Training (${STEPS} steps)"
echo "  this is the long part; run under tmux if you have not already"
printf '\n===== run started %s  (STEPS=%s) =====\n' "$(date '+%F %T')" "${STEPS}" >> adapter.log

RESUME_ARGS=()
[[ -n "${FRESH:-}" ]] && RESUME_ARGS+=(--fresh)

./src/muffled_asr/training/run_adapter.sh \
    --data "${DATA}" \
    --nemo "${NEMO_MODEL}" \
    --exp-dir "${EXP_DIR}" \
    --steps "${STEPS}" \
    --gpu-mem-frac "${GPU_MEM_FRAC}" \
    "${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}" \
    2>&1 | tee -a adapter.log

step "4/4  Result"
echo "  Training finished normally: reached STEPS=${STEPS}."
echo "  (\"Trainer.fit stopped: max_steps reached\" in the log is the normal ending, not an error.)"
echo
python3 src/muffled_asr/training/report.py adapter.log || true

CKPT_DIR=""
[[ -d "${EXP_DIR}" ]] && CKPT_DIR="$(find "${EXP_DIR}" -type d -name checkpoints 2>/dev/null | tail -1 || true)"
if [[ -n "${CKPT_DIR}" ]]; then
    echo
    echo "  checkpoints:"
    find "${CKPT_DIR}" -name '*.ckpt' 2>/dev/null | sed 's/^/    /' || true
    find "${EXP_DIR}" -name 'adapters.pt' 2>/dev/null | head -1 | sed 's/^/  adapter weights: /' || true
    find "${CKPT_DIR}" -name '*.nemo' 2>/dev/null | head -1 | sed 's/^/  exported model:  /' || true
fi
cat <<SUMMARY

  Log:    ./adapter.log
  Report: python3 src/muffled_asr/training/report.py     (works during training too)

  Next: score it on the 15 speakers it has never seen (runs in NeMo, here):
    python3 src/muffled_asr/training/eval_adapter.py
SUMMARY
