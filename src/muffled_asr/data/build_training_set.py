"""Build a masked + noisy training corpus from LibriSpeech.

Everything here is a response to something measured earlier in this project:

SPEAKER DIVERSITY. A fine-tune on LJSpeech's single speaker made held-out
speakers 0.5-0.9 WER *worse* at every chunk size, on clean audio as well as
masked. LibriSpeech train-clean-100 has 251 speakers, and speaker-disjoint
splits mean the dev/test numbers measure generalisation rather than recall.

NOISE, NOT JUST THE MASK. The helmet filter alone costs the base model about
0.4 WER -- too little to learn from. Welding noise at 0-5 dB SNR costs 7-9 WER
at low latency. The noise is the problem worth training against, so it is mixed
in at randomised SNR rather than left out.

CLEAN RETENTION. A fixed share of utterances is left untouched, so the model
keeps its general accuracy instead of drifting toward one narrow condition.

VARIED CONDITIONS. Mask preset and SNR are randomised per utterance. Train on a
single fixed filter and one noise clip and the model learns that recording, not
robustness.

DISJOINT NOISE. Each noise file is split by time: training draws from the first
70%, dev and test from separate later spans. Sharing seconds of a noise
recording between train and test inflates the score with no real improvement.

    python src/muffled_asr/data/build_training_set.py \
        <librispeech-root> <noise-dir> <out-dir> --pc-manifest <librispeech-pc.json>
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf


from muffled_asr.simulation.mask_sim import SAMPLE_RATE, apply_mask, load_16k_mono, mix_noise  # noqa: E402

# Enclosure presets, sampled per utterance. Varying the filter stops the model
# from fitting one exact transfer function.
MASK_PRESETS = ["welding_helmet", "welding_helmet", "respirator", "shield"]

# Fraction of each noise recording reserved per split, so no second of noise is
# heard by more than one split.
NOISE_SPANS = {"train": (0.00, 0.70), "dev": (0.70, 0.85), "test": (0.85, 1.00)}


def read_transcripts(root: Path, pc_manifest: Path | None) -> dict[str, str]:
    if pc_manifest:
        texts = {}
        for line in pc_manifest.read_text().splitlines():
            if line.strip():
                entry = json.loads(line)
                texts[Path(entry["audio_filepath"]).stem] = entry["text"]
        return texts

    texts = {}
    for transcript in root.rglob("*.trans.txt"):
        for line in transcript.read_text().splitlines():
            if line.strip():
                utterance_id, _, text = line.partition(" ")
                texts[utterance_id] = text
    return texts


def load_noise_bank(noise_dir: Path, include: list[str] | None) -> list[tuple[str, np.ndarray]]:
    """Load noise, optionally restricted to paths containing one of `include`.

    Noise has to match the deployment. A bank dominated by drones and football
    crowds teaches robustness to the wrong thing, and every utterance spent on
    an irrelevant background is one not spent on the floor you actually run on.
    """
    bank = []
    for path in sorted(noise_dir.rglob("*")):
        if path.suffix.lower() not in {".wav", ".flac", ".mp3", ".ogg"}:
            continue
        if include and not any(token.lower() in str(path).lower() for token in include):
            continue
        try:
            bank.append((path.stem, load_16k_mono(path)))
        except SystemExit:
            continue  # below 16 kHz: cannot carry the band the mask acts on
    if not bank:
        raise SystemExit(f"no usable noise found under {noise_dir} matching {include}")
    return bank


def noise_span(noise: np.ndarray, split: str) -> np.ndarray:
    low, high = NOISE_SPANS[split]
    start, end = int(len(noise) * low), int(len(noise) * high)
    return noise[start:end] if end - start > SAMPLE_RATE else noise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("librispeech", type=Path, help="LibriSpeech root (speaker dirs)")
    parser.add_argument("noise_dir", type=Path, help="directory of noise recordings, searched recursively")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--noise-include", nargs="+",
                        help="only use noise whose path contains one of these, "
                             "e.g. welding warehouse motor")
    parser.add_argument("--pc-manifest", type=Path, help="LibriSpeech-PC manifest (punctuated labels)")
    parser.add_argument("--allow-uppercase", action="store_true", help="accept stock labels, losing PnC")
    parser.add_argument("--clean-frac", type=float, default=0.25, help="share left unmasked and unnoised")
    parser.add_argument("--snr-range", type=float, nargs=2, default=[0.0, 20.0])
    parser.add_argument("--dev-speakers", type=int, default=15)
    parser.add_argument("--test-speakers", type=int, default=15)
    parser.add_argument("--max-duration", type=float, default=16.0)
    parser.add_argument("--limit", type=int, help="stop after N utterances (pilot runs)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.pc_manifest and not args.allow_uppercase:
        raise SystemExit(
            "Stock LibriSpeech transcripts are uppercase and unpunctuated, and this\n"
            "checkpoint emits punctuation and capitalization natively -- training on\n"
            "them strips that capability out within a few hundred steps.\n\n"
            "Pass --pc-manifest <LibriSpeech-PC manifest> (openslr.org/145, 25 MB of\n"
            "manifests only, reuses the audio you already have), or --allow-uppercase\n"
            "to override deliberately."
        )

    texts = read_transcripts(args.librispeech, args.pc_manifest)
    noise_bank = load_noise_bank(args.noise_dir, args.noise_include)
    print(f"{len(texts)} transcripts, {len(noise_bank)} noise recordings: "
          f"{', '.join(n for n, _ in noise_bank[:6])}{' ...' if len(noise_bank) > 6 else ''}")

    speakers = sorted(d.name for d in args.librispeech.iterdir() if d.is_dir())
    rng = random.Random(args.seed)
    shuffled = speakers[:]
    rng.shuffle(shuffled)
    test = set(shuffled[: args.test_speakers])
    dev = set(shuffled[args.test_speakers : args.test_speakers + args.dev_speakers])
    print(f"{len(speakers)} speakers -> {len(speakers) - len(dev) - len(test)} train, "
          f"{len(dev)} dev, {len(test)} test (disjoint)\n")

    manifests: dict[str, list[dict]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    skipped = 0

    clips = sorted(args.librispeech.rglob("*.flac"))
    for index, clip in enumerate(clips):
        if args.limit and index >= args.limit:
            break
        text = texts.get(clip.stem)
        if text is None:
            skipped += 1
            continue

        audio = load_16k_mono(clip)
        duration = len(audio) / SAMPLE_RATE
        if duration > args.max_duration or duration < 0.5:
            skipped += 1
            continue

        speaker = clip.relative_to(args.librispeech).parts[0]
        split = "test" if speaker in test else "dev" if speaker in dev else "train"

        # Seed per utterance so a rebuild reproduces the same corpus exactly.
        local = random.Random(f"{args.seed}:{clip.stem}")
        if local.random() < args.clean_frac:
            condition = "clean"
        else:
            audio = apply_mask(audio, local.choice(MASK_PRESETS))
            snr = local.uniform(*args.snr_range)
            name, noise = noise_bank[local.randrange(len(noise_bank))]
            audio = mix_noise(audio, noise_span(noise, split), snr,
                              np.random.default_rng(local.randrange(2**32)))
            condition = f"masked+{name}@{snr:.0f}dB"
        counts["clean" if condition == "clean" else "masked"] += 1

        destination = args.out_dir / split / speaker / f"{clip.stem}.flac"
        destination.parent.mkdir(parents=True, exist_ok=True)
        peak = np.abs(audio).max()
        if peak > 0.99:
            audio = audio * (0.99 / peak)
        sf.write(destination, audio.astype(np.float32), SAMPLE_RATE)

        manifests[split].append({
            "audio_filepath": str(destination.resolve()),
            "duration": round(duration, 3),
            "text": text,
            "condition": condition,
        })
        if (index + 1) % 2000 == 0:
            print(f"  {index + 1}/{len(clips)} clips...", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print()
    for split in ("train", "dev", "test"):
        path = args.out_dir / f"{split}.json"
        rows = manifests[split]
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        hours = sum(r["duration"] for r in rows) / 3600
        holders = len({Path(r["audio_filepath"]).parent.name for r in rows})
        print(f"{path}  {len(rows):>6} utts  {hours:6.2f} h  {holders:>3} speakers")

    total = counts["clean"] + counts["masked"]
    if total:
        print(f"\nclean {counts['clean']} ({100 * counts['clean'] / total:.0f}%), "
              f"masked+noisy {counts['masked']} ({100 * counts['masked'] / total:.0f}%)")
    if skipped:
        print(f"skipped {skipped} (no transcript, or outside {0.5}-{args.max_duration}s)")


if __name__ == "__main__":
    main()
