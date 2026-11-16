#!/usr/bin/env bash
# Fine-tune nemotron-speech-streaming-en-0.6b on a masked-speech manifest.
#
#   ./run_finetune.sh --train manifest.json --smoke        # validate setup, 5 steps
#   ./run_finetune.sh --train manifest.json                # real run
#
# Validation is optional. Given --val (or a dev.json sitting beside the training
# manifest) the best epoch is chosen by val_wer. Without it every epoch is kept
# and the exported .nemo is the last one.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${MSR_ROOT:-$(cd "${HERE}/../../.." && pwd)}"

TRAIN_MANIFEST=""
VAL_MANIFEST=""
NO_VAL=false
SKIP_MISSING=false
NEMO_MODEL="${PROJECT_ROOT}/models/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo"
EXP_DIR="${PROJECT_ROOT}/experiments"
VENV_DIR="${PROJECT_ROOT}/.venv-train"
EPOCHS=10
BATCH_SIZE=4
ACCUM=4
LR=1e-5
FREEZE_ENCODER=false
SMOKE=false
GPU_MEM_FRAC=""
WORKERS=""
KEEP_EPOCHS=""
SAVE_NEMO_EACH_EPOCH=false
LOW_PRIORITY=false

usage() {
    cat <<USAGE
Fine-tune nemotron-speech-streaming-en-0.6b on a masked-speech manifest.

  ./run_finetune.sh --train manifest.json --smoke     # validate setup, 5 steps
  ./run_finetune.sh --train manifest.json             # real run

Options:
  --data DIR          dataset dir holding train.json and dev.json
  --train PATH        training manifest — the model learns from this
  --val PATH          validation manifest — scored each epoch, never trained on,
                      decides the best epoch [defaults to dev.json beside --train]
  --no-val            train without validation; keep every epoch, no selection
  --skip-missing      drop entries whose audio is absent instead of stopping
  --nemo PATH         checkpoint to fine-tune [${NEMO_MODEL}]
  --exp-dir PATH      where epoch checkpoints and logs go [${EXP_DIR}]
  --venv PATH         virtualenv from install_cuda.sh [${VENV_DIR}]
  --epochs N          max epochs [${EPOCHS}]
  --batch N           per-step batch size [${BATCH_SIZE}]
  --accum N           gradient accumulation [${ACCUM}]
  --lr F              learning rate [${LR}]
  --gpu-mem-frac F    cap VRAM use at this fraction, e.g. 0.8, leaving the rest
                      for the desktop on a laptop GPU
  --workers N         dataloader worker processes [8; use 2 on a laptop]
  --keep-epochs N     how many epoch checkpoints to keep [all; each is GBs]
  --save-nemo-each-epoch
                      re-export the full .nemo every epoch. Off by default: it
                      serializes 600M params twice per epoch and stalls a laptop
  --low-priority      run under nice/ionice so the desktop stays responsive
  --freeze-encoder    train decoder and joint only — required under ~12 GB VRAM
  --smoke             5 steps at batch size 1, then exit
USAGE
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data)           TRAIN_MANIFEST="$2/train.json"; VAL_MANIFEST="$2/dev.json"; shift 2 ;;
        --train)          TRAIN_MANIFEST="$2"; shift 2 ;;
        --val)            VAL_MANIFEST="$2";   shift 2 ;;
        --no-val)         NO_VAL=true;         shift ;;
        --skip-missing)   SKIP_MISSING=true;   shift ;;
        --nemo)           NEMO_MODEL="$2";     shift 2 ;;
        --exp-dir)        EXP_DIR="$2";        shift 2 ;;
        --venv)           VENV_DIR="$2";       shift 2 ;;
        --epochs)         EPOCHS="$2";         shift 2 ;;
        --batch)          BATCH_SIZE="$2";     shift 2 ;;
        --accum)          ACCUM="$2";          shift 2 ;;
        --lr)             LR="$2";             shift 2 ;;
        --gpu-mem-frac)   GPU_MEM_FRAC="$2";   shift 2 ;;
        --workers)        WORKERS="$2";        shift 2 ;;
        --keep-epochs)    KEEP_EPOCHS="$2";    shift 2 ;;
        --save-nemo-each-epoch) SAVE_NEMO_EACH_EPOCH=true; shift ;;
        --low-priority)   LOW_PRIORITY=true;   shift ;;
        --freeze-encoder) FREEZE_ENCODER=true; shift ;;
        --smoke)          SMOKE=true;          shift ;;
        -h|--help)        usage 0 ;;
        *) echo "unknown option: $1" >&2; usage 1 ;;
    esac
done

[[ -n "${TRAIN_MANIFEST}" ]] || { echo "ERROR: --train or --data is required" >&2; usage 1; }
[[ -f "${TRAIN_MANIFEST}" ]] || { echo "ERROR: not found: ${TRAIN_MANIFEST}" >&2; exit 1; }
[[ -f "${NEMO_MODEL}" ]] || { echo "ERROR: not found: ${NEMO_MODEL}" >&2; exit 1; }

if [[ "${NO_VAL}" == true ]]; then
    VAL_MANIFEST=""
elif [[ -z "${VAL_MANIFEST}" ]]; then
    CANDIDATE="$(dirname "${TRAIN_MANIFEST}")/dev.json"
    [[ -f "${CANDIDATE}" ]] && VAL_MANIFEST="${CANDIDATE}"
fi
[[ -z "${VAL_MANIFEST}" || -f "${VAL_MANIFEST}" ]] || { echo "ERROR: not found: ${VAL_MANIFEST}" >&2; exit 1; }

# Environment: prefer the installer's venv, fall back to whatever is active.
if [[ -f "${VENV_DIR}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
elif python3 -c "import nemo.collections.asr" >/dev/null 2>&1; then
    echo "No venv at ${VENV_DIR}; using the active environment, which has NeMo."
else
    echo "ERROR: no training environment found." >&2
    echo "       ${VENV_DIR} does not exist and NeMo is not importable here." >&2
    echo "       Run ./src/muffled_asr/training/install_cuda.sh, or pass --venv /path/to/venv" >&2
    exit 1
fi

# Verify every entry before committing to a multi-hour job. Prints the manifest
# to use, which may be a filtered copy under --skip-missing.
PREFLIGHT_ARGS=()
[[ "${SKIP_MISSING}" == true ]] && PREFLIGHT_ARGS+=(--skip-missing)
TRAIN_MANIFEST="$(python3 "${HERE}/preflight.py" "${TRAIN_MANIFEST}" "${PREFLIGHT_ARGS[@]+"${PREFLIGHT_ARGS[@]}"}")"
if [[ -n "${VAL_MANIFEST}" ]]; then
    VAL_MANIFEST="$(python3 "${HERE}/preflight.py" "${VAL_MANIFEST}" "${PREFLIGHT_ARGS[@]+"${PREFLIGHT_ARGS[@]}"}")"
fi

# Without validation nothing can rank an epoch, so keep them all.
VAL_ARGS=(++model.validation_ds.manifest_filepath="${VAL_MANIFEST:-null}")
if [[ -z "${VAL_MANIFEST}" ]]; then
    VAL_ARGS+=(
        ++trainer.limit_val_batches=0
        ++trainer.num_sanity_val_steps=0
        # NeMo does `if "val" in params.monitor`, so monitor cannot be null.
        # train_loss is logged by the RNNT training_step and contains no "val".
        ++exp_manager.checkpoint_callback_params.monitor=train_loss
        ++exp_manager.checkpoint_callback_params.save_top_k=-1
        # Without a val loop, checkpoints must be written at train-epoch end;
        # NeMo defaults to saving at validation end, which never happens here.
        ++exp_manager.checkpoint_callback_params.save_on_train_epoch_end=true
        ++exp_manager.create_early_stopping_callback=false
    )
fi

# Long runs fragment the allocator: plenty of free VRAM in total, none of it
# contiguous enough for the next block. This lets CUDA grow segments instead.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"

TUNING_ARGS=()
[[ -n "${GPU_MEM_FRAC}" ]] && TUNING_ARGS+=(++gpu_mem_fraction="${GPU_MEM_FRAC}")
[[ -n "${WORKERS}" ]] && TUNING_ARGS+=(++model.train_ds.num_workers="${WORKERS}")
[[ -n "${KEEP_EPOCHS}" ]] && TUNING_ARGS+=(++exp_manager.checkpoint_callback_params.save_top_k="${KEEP_EPOCHS}")
TUNING_ARGS+=(++exp_manager.checkpoint_callback_params.always_save_nemo="${SAVE_NEMO_EACH_EPOCH}")
# save_last writes a second full checkpoint at every checkpoint event, and
# save_weights_only drops the optimizer state from the file. Together they cut
# the per-epoch disk burst that stalls a laptop.
TUNING_ARGS+=(++exp_manager.checkpoint_callback_params.save_weights_only=true)
TUNING_ARGS+=(++exp_manager.checkpoint_callback_params.save_last=false)

EXTRA_ARGS=()
if [[ "${SMOKE}" == true ]]; then
    echo
    echo "=== SMOKE TEST: 5 steps, batch size 1 ==="
    EXTRA_ARGS+=(
        trainer.max_steps=5
        trainer.max_epochs=1
        model.train_ds.batch_size=1
        model.validation_ds.batch_size=1
        trainer.accumulate_grad_batches=1
        exp_manager.create_checkpoint_callback=false
        exp_manager.create_early_stopping_callback=false
        exp_manager.resume_if_exists=false
    )
    [[ -n "${VAL_MANIFEST}" ]] && EXTRA_ARGS+=(trainer.val_check_interval=5)
else
    echo
    echo "=== Fine-tuning ==="
    echo "  checkpoint  ${NEMO_MODEL}"
    echo "  train       ${TRAIN_MANIFEST}"
    if [[ -n "${VAL_MANIFEST}" ]]; then
        echo "  val         ${VAL_MANIFEST}  (best epoch chosen by val_wer)"
    else
        echo "  val         none — every epoch kept, final .nemo is the LAST epoch"
    fi
    echo "  epochs ->   ${EXP_DIR}/nemotron_masked_finetune/<timestamp>/checkpoints/"
    echo "  effective batch $((BATCH_SIZE * ACCUM))   lr ${LR}   max ${EPOCHS} epochs"
    echo "  encoder     $([[ "${FREEZE_ENCODER}" == true ]] && echo frozen || echo trainable)"
    echo
    FREE_GIB="$(df -BG --output=avail "${EXP_DIR%/*}" 2>/dev/null | tail -1 | tr -dc '0-9')"
    NEEDED_GIB=$(( (EPOCHS + 1) * 3 ))
    echo "  disk        ${FREE_GIB:-?} GiB free; ~${NEEDED_GIB} GiB needed for ${EPOCHS} epoch checkpoints"
    if [[ -n "${FREE_GIB}" && "${FREE_GIB}" -lt "${NEEDED_GIB}" ]]; then
        echo "  WARNING: may run out of disk. Use --keep-epochs 1 to keep only the best."
    fi
    echo
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    VRAM_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)"
    if [[ "${VRAM_MIB}" -lt 12000 && "${FREEZE_ENCODER}" != true ]]; then
        echo
        echo "WARNING: ${VRAM_MIB} MiB of VRAM. A full fine-tune needs ~9.6 GB of AdamW"
        echo "         state alone (600M params x 4 bytes for weights, grads, two moments)"
        echo "         before any activations. Expect an OOM. Retry with:"
        echo "           --freeze-encoder --batch 1 --accum 16"
    fi
    echo
fi

LAUNCH=(python "${HERE}/finetune_masked.py")
if [[ "${LOW_PRIORITY}" == true ]]; then
    command -v ionice >/dev/null 2>&1 && LAUNCH=(ionice -c2 -n7 "${LAUNCH[@]}")
    LAUNCH=(nice -n 10 "${LAUNCH[@]}")
    echo "Running at low CPU/IO priority."
fi

exec "${LAUNCH[@]}" \
    --config-path "${HERE}/conf" \
    --config-name finetune_masked \
    init_from_nemo_model="${NEMO_MODEL}" \
    model.train_ds.manifest_filepath="${TRAIN_MANIFEST}" \
    model.train_ds.batch_size="${BATCH_SIZE}" \
    model.optim.lr="${LR}" \
    model.freeze_encoder="${FREEZE_ENCODER}" \
    trainer.max_epochs="${EPOCHS}" \
    trainer.accumulate_grad_batches="${ACCUM}" \
    exp_manager.exp_dir="${EXP_DIR}" \
    "${VAL_ARGS[@]}" \
    "${TUNING_ARGS[@]+"${TUNING_ARGS[@]}"}" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
