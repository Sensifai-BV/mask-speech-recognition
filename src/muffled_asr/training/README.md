# Fine-tuning on a Linux + NVIDIA RTX box

Training needs CUDA. NeMo's default transducer loss is `warprnnt_numba`, a Numba
CUDA kernel; the pure-PyTorch fallback is a nested Python loop over the RNN-T
lattice that NVIDIA marks *"for debugging purposes only"*. There is no practical
CPU or Apple-Metal path for a 600M-parameter model.

The Mac stays useful for everything else — dataset generation, manifest checks,
inference and WER scoring all run there.

## Three commands

```bash
./install_cuda.sh                                  # ~10 min, one time
./run_finetune.sh --train train.json --val dev.json --smoke    # ~1 min
./run_finetune.sh --train train.json --val dev.json            # the real run
```

**Always run `--smoke` first.** Five steps at batch size 1. It exercises manifest
parsing, checkpoint restore, tokenizer compatibility and tensor shapes — which is
where first runs actually fail — for the price of a minute instead of discovering
it three hours into a job.

## How many epochs

`max_epochs: 10` is a **ceiling, not a target**. Early stopping watches `val_wer`
with `patience: 3`, so a run that stops improving ends on its own. Expect most
runs to stop between epochs 3 and 6.

| Training data | Setting | Why |
| --- | --- | --- |
| under ~10 h | `--freeze-encoder --epochs 5` | Too little data to move 600M parameters without overfitting. Decoder + joint only. |
| 10–100 h | default, `--epochs 10` | Full fine-tune, early stopping decides the real number. |
| over 100 h | `--epochs 5` | Each epoch is already many thousands of steps. |

What matters more than the epoch count is `val_wer` per epoch. If it rises for
three checks in a row, the run stops and the best three checkpoints are kept.

## Sizing for your card

The memory hog in a transducer is the joint's `[B, T, U, V]` tensor. At batch 8,
16 s of audio and a 1025-token vocabulary that is several GB by itself, and it is
the usual cause of an OOM. `finetune_masked.py` turns on the fused loss path,
which splits it into `fused_batch_size` chunks and frees each after use.

A full fine-tune also carries a fixed ~9.6 GB of AdamW state (600M parameters ×
4 bytes for each of weights, gradients, and two moments).

| VRAM | Card | Suggested |
| --- | --- | --- |
| 16 GB | 4060 Ti / 4080 | `--freeze-encoder --batch 2`, or `fused_batch_size: 2` |
| 24 GB | 3090 / 4090 | defaults (`--batch 4 --accum 4`) |
| 32 GB | 5090 | `--batch 8 --accum 2` |
| 48 GB+ | A6000 / A100 | `--batch 16 --accum 1` |

If it still OOMs, in order: drop `fused_batch_size` to 1, drop `--batch` to 1,
lower `max_duration` in the config, then `--freeze-encoder`.

On cards older than Ampere, change `precision` to `16-mixed` — bf16 needs
compute capability 8.0+.

## Deliberate departures from NeMo's stock config

| Setting | Stock | Here | Reason |
| --- | --- | --- | --- |
| `optim.lr` | 1e-4 | **1e-5** | Stock is for adapting a generic base. At 1e-4 a 600M model on a small set forgets English within a few hundred steps. |
| `warmup_steps` | 5000 | **1000** | 5000 would be most of a short run. |
| `precision` | 32 | **bf16-mixed** | Roughly halves memory and time on Ampere+. |
| `gradient_clip_val` | 0.0 | **1.0** | Transducer gradients spike; 0.0 disables clipping entirely. |
| `max_duration` | 20.0 | **16.0** | Trades a few long clips for VRAM headroom. |
| `batch_size` | 16 | **4 × accum 4** | Same effective batch, fits 24 GB. |

## Two things that are load-bearing

**The encoder config is never touched.** The four runtime chunk sizes
(80/160/560/1120 ms) come from the encoder's `att_context_size`. Override any of
it and the model collapses to one latency operating point — the `--latency-ms`
switch in the GUI stops meaning anything.

**`model.joint.*` cannot be set from YAML.** NeMo's own fine-tune script applies
only `train_ds`, `validation_ds`, `test_ds`, `tokenizer`, `char_labels`, `optim`
and `spec_augment` to the restored model. A `model.joint:` block would be parsed
by Hydra and silently ignored. That is the entire reason `finetune_masked.py`
exists rather than calling NeMo's script directly.

## Do not train on LJSpeech alone

LJSpeech is **one speaker**. Fine-tuning 600M parameters on a single voice for
several epochs produces a model that is excellent at that voice and worse at
everyone else. Use it to prove the pipeline runs, then train on a speaker-diverse
mix — masked LibriSpeech (251 speakers in train-clean-100) plus 10–30% unmasked
speech to hold general accuracy.

## Afterwards

`always_save_nemo: true` writes a `.nemo` beside the Lightning checkpoints under
`experiments/nemotron_masked_finetune/`. From there:

```bash
# back to Transformers, for the macOS GUI app
python transformers/models/nemotron_asr_streaming/convert_nemotron_asr_streaming_to_hf.py \
    --nemo_file <tuned>.nemo --output_dir models/nemotron-speech-streaming-en-0.6b

# or to quantized GGUF, for NeMo-Speech.cpp
python convert_model.py <tuned>.nemo --outfile tuned.gguf --outtype q8_0
```

Score the result with `mask_eval.py` **at all four chunk sizes**. A fine-tune can
improve 1120 ms while leaving 80 ms flat or worse, and a single-configuration
check would not show it.

## Untested here

These were written on macOS and are syntax-checked only (`bash -n`, `shellcheck
-S error`, `ast.parse`, YAML parse). Nothing in them has been executed against a
real GPU — the `--smoke` run is the first real test.
