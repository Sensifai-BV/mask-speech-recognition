#!/usr/bin/env bash
# Train encoder adapters on masked + noisy speech.
#
#   ./run_adapter.sh --data <dataset-dir> --smoke     # validate setup
#   ./run_adapter.sh --data <dataset-dir>             # real run
#
# Adapters, not a fine-tune, and specifically on the ENCODER. Noise and channel
# robustness live in the encoder's early layers; the decoder and joint model
# language. The earlier `--freeze-encoder` run trained the language half to fix
# an acoustic problem, which is why it learned nothing useful and drifted.
#
# Adapters insert a small bottleneck module inside each encoder layer. The base
# weights stay frozen, so the optimizer state is tiny -- which is what makes
# encoder adaptation possible on 8 GB, where a full fine-tune cannot fit.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${MSR_ROOT:-$(cd "${HERE}/../../.." && pwd)}"
NEMO_ROOT="${PROJECT_ROOT}/nemo"

DATA_DIR=""
TRAIN_MANIFEST=""
VAL_MANIFEST=""
NEMO_MODEL="${PROJECT_ROOT}/models/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo"
EXP_DIR="${PROJECT_ROOT}/experiments"
VENV_DIR="${PROJECT_ROOT}/.venv-train"
ADAPTER_NAME="welding"
ADAPTER_DIM=32          # NeMo: 16-32 is generally enough
BATCH_SIZE=2
ACCUM=8
LR=1e-3                 # adapters are fresh random modules, so higher than a fine-tune
MAX_STEPS=3000
GPU_MEM_FRAC=""
WORKERS=2                # conservative host-RAM/CPU default for the laptop
VAL_WORKERS=0            # avoid a second worker pool during validation
CPU_THREADS=2
PIN_MEMORY=false
CKPT_EVERY=500           # optimizer steps between validations (3000 steps -> 6 checks)
LOW_PRIORITY=true        # checkpoint bursts otherwise starve the desktop of IO
RESUME=true              # pick up from the last checkpoint if one exists
SMOKE=false
MEM_MAX="auto"           # host-RAM cap for the whole training process tree (cgroup); auto = 60 % of RAM
CPU_QUOTA=""             # e.g. 4 -> at most 4 cores (CPUQuota=400%); empty = no cap

usage() {
    cat <<USAGE
Train encoder adapters on masked + noisy speech.

  ./run_adapter.sh --data <dataset-dir> --smoke
  ./run_adapter.sh --data <dataset-dir>

Options:
  --data DIR          dataset dir with train.json and dev.json (required)
  --nemo PATH         base checkpoint [${NEMO_MODEL}]
  --name NAME         adapter name, must be unique per model [${ADAPTER_NAME}]
  --dim N             adapter bottleneck width [${ADAPTER_DIM}]
  --steps N           max optimizer steps [${MAX_STEPS}]
  --batch N           per-step batch size [${BATCH_SIZE}]
  --accum N           gradient accumulation [${ACCUM}]
  --lr F              learning rate [${LR}]
  --workers N         dataloader workers [${WORKERS}]
  --val-workers N     validation workers [${VAL_WORKERS}]
  --cpu-threads N     CPU threads per process [${CPU_THREADS}]
  --pin-memory        enable pinned host buffers (default is disabled)
  --ckpt-every N      steps between checkpoint writes [${CKPT_EVERY}]
  --normal-priority   do not lower CPU/IO priority (default is lowered)
  --fresh             ignore existing checkpoints and start over
  --resume            resume an existing run (the default)
  --gpu-mem-frac F    cap VRAM share, e.g. 0.8, leaving room for the display
  --exp-dir PATH      [${EXP_DIR}]
  --venv PATH         [${VENV_DIR}]
  --mem-max SIZE      host-RAM cap, e.g. 12G; 'auto' = 60 % of RAM, 'off' = none [${MEM_MAX}]
                      runs training in its own systemd cgroup: memory above 90 % of the cap is
                      reclaimed/throttled, and only the training job (never the desktop) is
                      killed at the cap -- rerun with --resume
  --cpu-quota N       at most N CPU cores for training + dataloader workers [no cap]
  --smoke             5 steps at batch size 1, then exit
USAGE
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data)         DATA_DIR="$2";      shift 2 ;;
        --nemo)         NEMO_MODEL="$2";    shift 2 ;;
        --name)         ADAPTER_NAME="$2";  shift 2 ;;
        --dim)          ADAPTER_DIM="$2";   shift 2 ;;
        --steps)        MAX_STEPS="$2";     shift 2 ;;
        --batch)        BATCH_SIZE="$2";    shift 2 ;;
        --accum)        ACCUM="$2";         shift 2 ;;
        --lr)           LR="$2";            shift 2 ;;
        --workers)      WORKERS="$2";       shift 2 ;;
        --val-workers)  VAL_WORKERS="$2";   shift 2 ;;
        --cpu-threads)  CPU_THREADS="$2";   shift 2 ;;
        --pin-memory)   PIN_MEMORY=true;    shift ;;
        --ckpt-every)   CKPT_EVERY="$2";    shift 2 ;;
        --normal-priority) LOW_PRIORITY=false; shift ;;
        --fresh)        RESUME=false;       shift ;;
        --resume)       RESUME=true;        shift ;;
        --gpu-mem-frac) GPU_MEM_FRAC="$2";  shift 2 ;;
        --exp-dir)      EXP_DIR="$2";       shift 2 ;;
        --venv)         VENV_DIR="$2";      shift 2 ;;
        --smoke)        SMOKE=true;         shift ;;
        --mem-max)      MEM_MAX="$2";       shift 2 ;;
        --cpu-quota)    CPU_QUOTA="$2";     shift 2 ;;
        -h|--help)      usage 0 ;;
        *) echo "unknown option: $1" >&2; usage 1 ;;
    esac
done

for value in "${WORKERS}" "${VAL_WORKERS}"; do
    [[ "$value" =~ ^[0-9]+$ ]] || { echo "ERROR: worker counts must be nonnegative integers" >&2; exit 1; }
done
for value in "${CPU_THREADS}" "${BATCH_SIZE}" "${ACCUM}" "${CKPT_EVERY}" "${MAX_STEPS}"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: threads, batch, accum, checkpoint interval and steps must be positive integers" >&2; exit 1; }
done

# Apply before any Python process imports numerical libraries, including the
# environment check and preflight. Worker processes inherit these limits.
export OMP_NUM_THREADS="${CPU_THREADS}"
export MKL_NUM_THREADS="${CPU_THREADS}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS}"
export VECLIB_MAXIMUM_THREADS="${CPU_THREADS}"
export TOKENIZERS_PARALLELISM=false
# glibc otherwise keeps up to 8 x cores malloc arenas per process; with variable-length
# audio in several dataloader workers that shows up as steadily growing host RAM.
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

[[ -n "${DATA_DIR}" ]] || { echo "ERROR: --data is required" >&2; usage 1; }
TRAIN_MANIFEST="${DATA_DIR}/train.json"
VAL_MANIFEST="${DATA_DIR}/dev.json"

for f in "${TRAIN_MANIFEST}" "${VAL_MANIFEST}" "${NEMO_MODEL}"; do
    [[ -f "$f" ]] || { echo "ERROR: not found: $f" >&2; exit 1; }
done

# This checkpoint was trained with the Lhotse dataloader, and its stored config
# carries `num_buckets: null` and `bucket_buffer_size: null`. LhotseDataLoadingConfig
# declares both as plain `int`, so merging that config raises
#   ValidationError: Incompatible value 'None' for field of type 'int'
# NeMo's adapter config is written for the legacy dataloader anyway (its
# bucketing_strategy / shuffle_n keys are legacy-only), so we select that path
# and also supply ints for the two fields, in case a NeMo version validates them
# regardless of which loader runs.

ADAPTER_SCRIPT="${HERE}/adapter_masked.py"
ADAPTER_CONF="${HERE}/conf"
[[ -f "${ADAPTER_SCRIPT}" ]] || { echo "ERROR: ${ADAPTER_SCRIPT} missing -- re-sync src/" >&2; exit 1; }

if [[ -f "${VENV_DIR}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
elif python3 -c "import nemo.collections.asr" >/dev/null 2>&1; then
    echo "No venv at ${VENV_DIR}; using the active environment, which has NeMo."
else
    echo "ERROR: no training environment found. Run ./src/muffled_asr/training/install_cuda.sh" >&2
    exit 1
fi

python3 "${HERE}/preflight.py" "${TRAIN_MANIFEST}" >/dev/null || exit 1
python3 "${HERE}/preflight.py" "${VAL_MANIFEST}" >/dev/null || exit 1

# The adapter's input width is the encoder's d_model, read from the checkpoint
# rather than hardcoded -- a wrong value fails with an opaque shape error.
IN_FEATURES="$(python3 - "${PROJECT_ROOT}" <<'PYDIM'
import json, sys
from pathlib import Path
cfg = Path(sys.argv[1]) / "models/nemotron-speech-streaming-en-0.6b/config.json"
print(json.loads(cfg.read_text())["encoder_config"]["hidden_size"] if cfg.exists() else 1024)
PYDIM
)"

# Lightning counts val_check_interval in *training batches*, not optimizer
# steps. With accumulation 8, "1000" meant every 125 steps: 23 validations and
# as many checkpoint writes in a 3000-step run. The multiplication below is
# what makes --ckpt-every mean optimizer steps.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Checkpointing is what stalls this machine, but resume needs some of what the
# cheapest settings remove, and on a machine that has already crashed mid-run,
# losing 2000 steps costs more than an extra write:
#   save_last=true             the "-last" checkpoint resume_if_exists looks for
#   save_weights_only=false    without optimizer state, a resume restarts the
#                              optimizer and scheduler from zero
#   save_last_n_optim_states=-1 disables distributed optimizer-state pruning;
#                              adapter optimizer state is small
#   save_top_k=1               one best checkpoint on disk, not five
#   always_save_nemo=false     NeMo's adapter config defaults this to TRUE, which
#                              re-tars the whole 600M model at every event
# async_save is not usable here: it needs AsyncFinalizableCheckpointIO and
# raises ValueError on a single-GPU strategy.

EXTRA=()
if [[ "${SMOKE}" == true ]]; then
    echo "=== SMOKE TEST: 5 steps, batch size 1 ==="
    EXTRA+=(trainer.max_steps=5 model.train_ds.batch_size=1 model.validation_ds.batch_size=1
            trainer.accumulate_grad_batches=1 exp_manager.create_checkpoint_callback=false)
else
    echo "=== Adapter training ==="
    echo "  base        ${NEMO_MODEL}"
    echo "  train       ${TRAIN_MANIFEST}"
    echo "  val         ${VAL_MANIFEST}  (best step chosen by val_wer)"
    echo "  adapter     ${ADAPTER_NAME}, linear, dim ${ADAPTER_DIM}, in_features ${IN_FEATURES}, on encoder"
    echo "  effective batch $((BATCH_SIZE * ACCUM))   lr ${LR}   max ${MAX_STEPS} steps"
    echo "  CPU threads ${CPU_THREADS}   workers train=${WORKERS}, val=${VAL_WORKERS}   pin_memory=${PIN_MEMORY}"
    echo "  output ->   ${EXP_DIR}/nemotron_welding_adapter/"
    echo
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
    echo
fi
[[ -n "${GPU_MEM_FRAC}" ]] && EXTRA+=("gpu_mem_fraction=${GPU_MEM_FRAC}")

LAUNCH=(python "${ADAPTER_SCRIPT}")
if [[ "${LOW_PRIORITY}" == true ]]; then
    command -v ionice >/dev/null 2>&1 && LAUNCH=(ionice -c2 -n7 "${LAUNCH[@]}")
    LAUNCH=(nice -n 10 "${LAUNCH[@]}")
fi

# Hard resource limits via a transient systemd scope (cgroup v2). The whole process
# tree -- trainer, dataloader workers, and the page cache of checkpoint writes -- is
# charged to it. Falls back to unlimited (with a warning) if systemd-run is unusable.
to_bytes() {  # 12G / 512M / 12000000000 -> bytes
    local v; v="$(printf %s "$1" | tr a-z A-Z)"
    case "${v}" in
        *G) echo $(( ${v%G} * 1024 * 1024 * 1024 )) ;;
        *M) echo $(( ${v%M} * 1024 * 1024 )) ;;
        *) echo "${v}" ;;
    esac
}
if [[ "${MEM_MAX}" == auto ]]; then
    total_kb="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)"
    MEM_MAX=$(( total_kb * 1024 * 6 / 10 ))
    (( MEM_MAX > 0 )) || MEM_MAX=off
fi
SCOPE_PROPS=()
if [[ "${MEM_MAX}" != off && -n "${MEM_MAX}" ]]; then
    mem_bytes="$(to_bytes "${MEM_MAX}")"
    [[ "${mem_bytes}" =~ ^[0-9]+$ ]] || { echo "ERROR: --mem-max must look like 12G, 12000M or 'off'" >&2; exit 1; }
    SCOPE_PROPS+=(-p "MemoryHigh=$(( mem_bytes * 9 / 10 ))" -p "MemoryMax=${mem_bytes}" -p "MemorySwapMax=0")
fi
if [[ -n "${CPU_QUOTA}" ]]; then
    [[ "${CPU_QUOTA}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: --cpu-quota must be a positive integer (cores)" >&2; exit 1; }
    SCOPE_PROPS+=(-p "CPUQuota=$(( CPU_QUOTA * 100 ))%")
fi
if (( ${#SCOPE_PROPS[@]} > 0 )); then
    if command -v systemd-run >/dev/null 2>&1 && systemd-run --user --scope --quiet "${SCOPE_PROPS[@]}" true 2>/dev/null; then
        echo "  limits      ${SCOPE_PROPS[*]}"
        LAUNCH=(systemd-run --user --scope --quiet --collect --unit="adapter-${ADAPTER_NAME}-$$" "${SCOPE_PROPS[@]}" -- "${LAUNCH[@]}")
    else
        echo "WARNING: systemd-run --user --scope is not available; running WITHOUT memory/CPU limits." >&2
        echo "         (threads=${CPU_THREADS}, workers=${WORKERS} still apply)" >&2
    fi
fi

exec "${LAUNCH[@]}" \
    --config-path="${ADAPTER_CONF}" \
    --config-name="adapter_masked" \
    init_from_nemo_model="${NEMO_MODEL}" \
    model.adapter.name="${ADAPTER_NAME}" \
    model.adapter.dim="${ADAPTER_DIM}" \
    model.adapter.in_features="${IN_FEATURES}" \
    model.train_ds.manifest_filepath="${TRAIN_MANIFEST}" \
    model.train_ds.batch_size="${BATCH_SIZE}" \
    model.train_ds.num_workers="${WORKERS}" \
    model.train_ds.pin_memory="${PIN_MEMORY}" \
    model.validation_ds.manifest_filepath="${VAL_MANIFEST}" \
    model.validation_ds.batch_size="${BATCH_SIZE}" \
    model.validation_ds.num_workers="${VAL_WORKERS}" \
    model.validation_ds.pin_memory="${PIN_MEMORY}" \
    model.optim.lr="${LR}" \
    trainer.max_steps="${MAX_STEPS}" \
    trainer.accumulate_grad_batches="${ACCUM}" \
    trainer.val_check_interval="$((CKPT_EVERY * ACCUM))" \
    exp_manager.exp_dir="${EXP_DIR}" \
    ++exp_manager.resume_if_exists="${RESUME}" \
    "${EXTRA[@]+"${EXTRA[@]}"}"
