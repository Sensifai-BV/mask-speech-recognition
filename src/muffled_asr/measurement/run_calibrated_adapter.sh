#!/usr/bin/env bash
# Fresh adapter run for the Cox-measured covering distribution.
# Run on the NVIDIA training machine after building calibrated manifests.
set -euo pipefail
ROOT="${MSR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
ARM="${1:-measured}"
case "${ARM}" in
    measured)
        DATA="${ROOT}/datasets/calibrated/librispeech-welding-bounded_pca"
        NAME="cox_measured_bounded_pca"
        ;;
    empirical)
        DATA="${ROOT}/datasets/calibrated/librispeech-welding-empirical"
        NAME="cox_measured_empirical"
        ;;
    noise-control)
        DATA="${ROOT}/datasets/calibrated/librispeech-noise-control"
        NAME="cox_matched_noise_control"
        ;;
    # Literature-based workshop SNRs (build_realistic_training_set.py); full_visor held out.
    realistic-measured)
        DATA="${ROOT}/datasets/realistic/bounded_pca"
        NAME="realistic_bounded_pca"
        ;;
    realistic-empirical)
        DATA="${ROOT}/datasets/realistic/empirical"
        NAME="realistic_empirical"
        ;;
    realistic-noise-control)
        DATA="${ROOT}/datasets/realistic/noise_control"
        NAME="realistic_noise_control"
        ;;
    *) echo "usage: $0 [measured|empirical|noise-control|realistic-measured|realistic-empirical|realistic-noise-control] [run_adapter.sh options, e.g. --smoke or --resume]" >&2; exit 2 ;;
esac
if [[ $# -gt 0 ]]; then shift; fi
BASE="${ROOT}/models/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo"
EXP_DIR="${ROOT}/experiments/${NAME}"
TRAIN_PYTHON="${ROOT}/.venv-train/bin/python"
[[ -x "${TRAIN_PYTHON}" ]] || TRAIN_PYTHON="python3"

# Manifests may have been built on the Mac and copied with the dataset. Repair
# their stored absolute paths against this checkout before the strict runner
# preflight checks every audio file.
"${TRAIN_PYTHON}" "${ROOT}/src/muffled_asr/training/preflight.py" "${DATA}/train.json" --fix-paths >/dev/null
"${TRAIN_PYTHON}" "${ROOT}/src/muffled_asr/training/preflight.py" "${DATA}/dev.json" --fix-paths >/dev/null

# New matched experiments start fresh; --resume supplied below overrides this
# for recovery. Forward resource options so the wrapper cannot hide them.
exec "${ROOT}/src/muffled_asr/training/run_adapter.sh" --data "${DATA}" --nemo "${BASE}" \
    --name "${NAME}" --dim 32 --steps 3000 --batch 2 --accum 8 \
    --lr 3e-4 --ckpt-every 500 --fresh --exp-dir "${EXP_DIR}" "$@"
