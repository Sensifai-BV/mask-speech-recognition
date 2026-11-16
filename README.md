# mask-speech-recognition

Streaming speech recognition for **muffled speech**: talkers behind face masks, transparent masks and shields, or welding helmets, often in loud workshop noise.

The code base does four things around NVIDIA's cache-aware streaming recogniser [`nemotron-speech-streaming-en-0.6b`](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b) (FastConformer RNN-T, 80 / 160 / 560 / 1120 ms latency modes):

1. **Model coverings.** It provides parametric presets, and measured responses from Cox et al. (2022) with empirical and bounded-PCA samplers.
2. **Build corpora.** It builds speaker-disjoint training and test corpora with literature-based workshop noise levels and A-weighted SNR.
3. **Adapt the recogniser.** It trains encoder adapters in NeMo on a CUDA machine and runs them on a Mac through 🤗 Transformers.
4. **Analyse coverings.** It asks how much of a covering's damage is a linear filter, using the oracle-filter study with reading-to-reading and recognition-fitted controls.

---

## Installation

Python ≥ 3.11.

```bash
git clone <this repo> && cd mask-speech-recognition
python -m venv .venv && source .venv/bin/activate
pip install -e ".[infer,audio]"        # analysis + Mac/CPU inference + microphone tools
```

`pip install -e .` alone is enough for corpus building and scoring. That covers numpy, scipy, soundfile, pandas, jiwer and matplotlib.

**Training** needs Linux + an NVIDIA GPU, because NeMo's RNN-T loss is a Numba CUDA kernel:

```bash
src/muffled_asr/training/install_cuda.sh .venv-train      # torch for your CUDA, Cython, NeMo, Lightning (see training/README.md)
git clone https://github.com/NVIDIA/NeMo nemo             # train.sh / run_adapter.sh expect ./nemo
```

**Model weights**: put the HF files and the `.nemo` of `nvidia/nemotron-speech-streaming-en-0.6b` in `models/nemotron-speech-streaming-en-0.6b/`. The inference package fetches the HF files there if they are missing. Its preprocessor uses 128 mel bands, a 25 ms window and a 10 ms stride at 16 kHz.

## Quick start

```bash
M=src/muffled_asr

# 1. Hear a covering
python $M/simulation/mask_sim.py speech.flac --mask surgical
python $M/simulation/mask_sim.py speech.flac --mask welding_helmet --noise weld.flac --snr 5

# 2. Measured covering responses (Cox et al. 2022)
python $M/measurement/download_cox.py                 # Figshare 19174235, MD5-verified
python $M/measurement/build_cox_bank.py               # -> datasets/calibrated/response_bank.json

# 3. A paired corpus: clean / covering / noise / covering+noise, speaker- and noise-session-disjoint
python $M/measurement/build_realistic_training_set.py datasets/librispeech-pc/train-clean-100.json \
    datasets/librispeech-pc/LibriSpeech/train-clean-100 "$WELDING_NOISE_ROOT" datasets/realistic

# 4. Train an encoder adapter (Linux + CUDA)
src/muffled_asr/training/run_adapter.sh --data datasets/realistic/bounded_pca --smoke
src/muffled_asr/training/run_adapter.sh --data datasets/realistic/bounded_pca --mem-max auto
python $M/training/export_adapter.py --exp-dir experiments/<run> --out models/<name>

# 5. Score on real masked talkers (Mac or CPU)
python $M/measurement/build_cox_human_eval.py --channel far
python $M/evaluation/eval_hf.py --manifests datasets/calibrated/cox-human-eval-far/test.json \
    --models base models/<name> --lookahead 0 1 6 13 --out results/human_quiet

# 6. Live transcription with an adapter
python -m muffled_asr.inference --help
```

## Components

### Covering simulation (`simulation/mask_sim.py`)
- **Fabric masks** (surgical, KN95, N95) are pure insertion-loss shelves taken from Corey, Jones & Singer (JASA 2020).
- **Enclosures** (shield, respirator, welding helmet) add four effects: a steep low-pass past a 1.5–3 kHz knee, low-frequency occlusion boost, cavity modes at c/2L, and short cavity reverberation. Their parameters are physically motivated **estimates**, not measurements.
- `mix_noise` adds noise after the covering.

### Measured coverings (`measurement/`)
- `build_cox_bank.py` aligns each mannikin impulse response separately, applies a 12 ms window, and stores covering-relative magnitude on 31 points from 250 to 7600 Hz.
- `CoveringBank.sample(split, mode, rng)` draws a design, then either one measured refit (`empirical`) or a bounded rank-≤2 PCA perturbation within the design (`bounded_pca`). It returns a minimum-phase FIR and a JSON recipe. The held-out designs (cloth v2 TPU A–D) are never used for fitting.
- `acoustic_levels.py` defines the A-weighted active speech level, the noise Leq, and a Lombard talker-level model (+0.6 dB per dB above 48 dBA, ceiling 82 dBA). Its workshop scenes are shop ambient U(55, 70) dBA and arc welding N(88, 4.2) dBA, clipped to 79–98.
- SNR convention: the reference is the **uncovered** talker level, so a covering keeps the level it removes.

### Training (`training/`)
`adapter_masked.py` trains one NeMo `LinearAdapter` after each of the 24 encoder layers (dim 32; 1.62 M trainable parameters) with the base frozen. It sets `slim_checkpoints` so checkpoints hold only trainable tensors and optimiser state, which avoids host-RAM exhaustion at save time. `run_adapter.sh` adds resource caps: `--mem-max SIZE|auto|off` and `--cpu-quota N`, via `systemd-run --user --scope`. `export_adapter.py` converts the best checkpoint into an adapter directory of about 6 MB that `inference/adapter.py` attaches to the Transformers model with forward hooks. See [training/README.md](src/muffled_asr/training/README.md) for card sizing and epochs.

### Evaluation (`evaluation/`)
`eval_hf.py` decodes NeMo manifests with any set of models and look-ahead settings and writes `hyps_<model>_<manifest>_look<k>.jsonl` plus `scores.csv`. Lookahead tokens map to latency as 0 → 80 ms, 1 → 160 ms, 6 → 560 ms and 13 → 1120 ms. It writes `*.partial.jsonl` as it goes and resumes after interruptions. Its decodes match NeMo's logged WER (e.g. 3.04 vs 3.06).

### Oracle-filter study (`oracle_filter/`)
The study asks whether a covering acts on the recogniser like a linear filter. For each talker × covering pair it builds four signals: **U** unmasked, **M** masked, **F** = oracle forward filter applied to U, and **R** = oracle inverse filter applied to M. It reports:
- gap = WER(M) − WER(U);
- R_fwd = (WER(M) − WER(F)) / gap, the share of the gap the filter does *not* explain;
- R_inv = (WER(R) − WER(U)) / gap.

CIs come from a talker-cluster bootstrap (1,000 resamples, seed 0). The controls are:
- the Cox-R reading-to-reading floor (`coxr_floor.py`);
- a recognition-fitted filter (`recfit.py`);
- a fixed-absolute-noise variant (`gen_abs.py`);
- the listening check of flagged reference runs (`listening_kit.py`, `apply_listening.py`).

Scoring references and normalisation are defined once, in `oracle_filter/common.py`.

### Robot commands (`slu/artaban_mic_test.py`)
This records spoken commands on the laptop microphone, optionally with held-out noise played through the speakers, and scores four metrics per model: strict WER, exact match, fuzzy wake-word detection, and intent (nearest command). Held-out noise comes from the test split of the corpus built by `measurement/build_calibrated_training_set.py` (its `provenance.json`; set `NOISE_PROVENANCE` to use another).

## Results so far (tables are written to `results/`; table names below)

| Result | Value | Table |
|---|---|---|
| Real coverings (Cox, 4 talkers, 1 m), gap at 0 / 5 / 10 dB (Table I SNR rule) | 2.96 / 2.86 / 1.63 pts | `oracle_filter/table_P0-1.csv` |
| Surgical mask (Geng, 28 talkers), gap at 0 / −10 dB | 4.13 / 2.24 pts | `table_P0-1_geng_lowsnr.csv` |
| Cox-R: masked gap above the reading floor at 0 / −5 / −10 dB | +4.51 / +3.81 / +1.66 pts (CI excludes 0 at −5 and −10 dB) | `coxr_masked_vs_null_pooled.csv` |
| Same, non-filter residual above the floor | not significant at any SNR | same |
| Adapters, Cox quiet 1 m test (corrected refs): base / noise-ctl / bounded-PCA / empirical | 1.69 / 1.08 / 1.05 / 0.93 % WER | `followup/table3_rescored_overall.csv` |
| Adapters, Cox 1 m in welding noise −10…10 dB | 15.34 / 14.19 / 14.13 / 14.17 % WER | same |

The noise-only adapter accounts for most of the adapter gain. All adapter results are single-seed.

## Citing the data and methods
- T. J. Cox *et al.*, "Improving the measurement and acoustic performance of transparent face masks and shields," *JASA* 151, 2931–2944 (2022). Data: Figshare 19174235.
- Geng *et al.*, PLOS ONE 2023, doi:10.1371/journal.pone.0283724 (masked/unmasked "North Wind and the Sun" recordings).
- S. Hu *et al.*, "On-the-fly Routing for Zero-shot MoE Speaker Adaptation of Speech Foundation Models for Dysarthric Speech Recognition," Interspeech 2025.
- L. Lugosch *et al.*, "Speech Model Pre-training for End-to-End Spoken Language Understanding," Interspeech 2019.
- R. M. Corey, U. Jones, A. C. Singer, "Acoustic effects of medical, cloth, and transparent face masks on speech signals," *JASA* 148(4), 2020.

Licences: the ESC-50 knock clips used by `data/noises.py` are CC BY-NC 3.0 (non-commercial). The code in this repository is released under the Apache License 2.0 (see `LICENSE`).
