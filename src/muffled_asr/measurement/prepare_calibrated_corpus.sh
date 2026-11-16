#!/usr/bin/env bash
set -euo pipefail
ROOT="${MSR_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
WELDING_NOISE_ROOT="${WELDING_NOISE_ROOT:-$ROOT/datasets/original/intel_robotic_welding_dataset}"
MODE="${1:-bounded_pca}"
case "$MODE" in
  bounded_pca|empirical) ;;
  *) echo "usage: $0 [bounded_pca|empirical]" >&2; exit 2 ;;
esac
ARCHIVE="$ROOT/datasets/librispeech-pc/train-clean-100.tar.gz"
[[ -s "$ARCHIVE" ]] || { echo "Missing LibriSpeech archive: $ARCHIVE" >&2; exit 1; }
mkdir -p "$ROOT/datasets/librispeech-pc"
tar -xzf "$ARCHIVE" -C "$ROOT/datasets/librispeech-pc"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON="python3"
"$PYTHON" "$ROOT/src/muffled_asr/measurement/build_calibrated_training_set.py" \
  "$ROOT/datasets/librispeech-pc/train-clean-100.json" \
  "$ROOT/datasets/librispeech-pc/LibriSpeech/train-clean-100" \
  "$WELDING_NOISE_ROOT" \
  "$ROOT/datasets/calibrated/librispeech-welding-$MODE" \
  --mode "$MODE" \
  --noise-control-out "$ROOT/datasets/calibrated/librispeech-noise-control"
