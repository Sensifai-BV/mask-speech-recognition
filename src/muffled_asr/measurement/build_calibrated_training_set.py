"""Build paired clean/noisy speech with measured covering responses.

The four conditions form a small factorial mix: clean, covering-only,
noise-only, and covering-plus-noise. Whole speech speakers and whole welding
recording sessions are split before augmentation. The held-out Cox design family
is used only for the synthetic test split; the response sampler never fits it.

Example:
    python src/muffled_asr/measurement/build_calibrated_training_set.py \
        datasets/librispeech-pc/train-clean-100.json \
        datasets/librispeech-pc/LibriSpeech/train-clean-100 \
        /path/to/intel_robotic_welding_dataset \
        datasets/calibrated/librispeech-welding
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from muffled_asr.paths import ROOT
from muffled_asr.simulation.mask_sim import SAMPLE_RATE, load_16k_mono, mix_noise  # noqa: E402
from muffled_asr.measurement.covering import CoveringBank  # noqa: E402

CONDITIONS = ("clean", "covering-only", "noise-only", "covering+noise")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pc_manifest", type=Path)
    parser.add_argument("speech_root", type=Path, help="extracted train-clean-100 directory")
    parser.add_argument("noise_root", type=Path, help="Intel welding audio collection")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--noise-control-out", type=Path,
                        help="optional paired control with measured coverings removed")
    parser.add_argument("--bank", type=Path, default=Path("datasets/calibrated/response_bank.json"))
    parser.add_argument("--mode", choices=("bounded_pca", "empirical"), default="bounded_pca")
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--dev-speakers", type=int, default=15)
    parser.add_argument("--test-speakers", type=int, default=15)
    parser.add_argument("--snr", type=float, nargs=2, default=(0.0, 20.0))
    parser.add_argument("--limit", type=int, help="small pipeline pilot")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.pc_manifest.read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit("Speech manifest is empty")

    # LibriSpeech-PC paths normally begin at train-clean-100/; accept either
    # the corpus parent or the subset directory as speech_root.
    def speech_path(row: dict) -> Path:
        candidate = args.speech_root / row["audio_filepath"]
        if candidate.is_file():
            return candidate
        return args.speech_root / Path(row["audio_filepath"]).name

    # Resolve robustly from the canonical speaker/chapter/file suffix, since
    # manifests can contain absolute paths from another machine.
    for row in rows:
        parts = Path(row["audio_filepath"]).parts
        if len(parts) >= 3:
            candidate = args.speech_root.joinpath(*parts[-3:])
            if candidate.is_file():
                row["_audio"] = str(candidate)
                continue
        candidate = speech_path(row)
        if candidate.is_file():
            row["_audio"] = str(candidate)
        else:
            raise SystemExit(f"Missing source audio: {candidate}")

    speakers = sorted({Path(r["_audio"]).parent.parent.name for r in rows})
    rng = random.Random(args.seed)
    rng.shuffle(speakers)
    n_test, n_dev = args.test_speakers, args.dev_speakers
    if len(speakers) <= n_test + n_dev:
        raise SystemExit("Need more LibriSpeech speakers than requested dev and test speakers")
    split_of = {s: "test" for s in speakers[:n_test]}
    split_of.update({s: "dev" for s in speakers[n_test:n_test+n_dev]})
    split_of.update({s: "train" for s in speakers[n_test+n_dev:]})

    noise_paths = sorted(args.noise_root.rglob("*.flac"))
    if not noise_paths:
        raise SystemExit(f"No FLAC welding recordings under {args.noise_root}")
    noise_groups = sorted({p.relative_to(args.noise_root).parts[0] for p in noise_paths})
    rng.shuffle(noise_groups)
    n_noise_test = max(1, round(.15 * len(noise_groups)))
    n_noise_dev = max(1, round(.15 * len(noise_groups)))
    noise_split = {g: "test" for g in noise_groups[:n_noise_test]}
    noise_split.update({g: "dev" for g in noise_groups[n_noise_test:n_noise_test+n_noise_dev]})
    noise_split.update({g: "train" for g in noise_groups[n_noise_test+n_noise_dev:]})
    noise_by_split: dict[str, list[Path]] = defaultdict(list)
    for path in noise_paths:
        noise_by_split[noise_split[path.relative_to(args.noise_root).parts[0]]].append(path)
    if any(not noise_by_split[s] for s in ("train", "dev", "test")):
        raise SystemExit("Noise groups did not populate each split")

    bank = CoveringBank(args.bank)
    manifests: dict[str, list[dict]] = defaultdict(list)
    counts: Counter = Counter()
    for i, row in enumerate(rows):
        if args.limit is not None and i >= args.limit:
            break
        source = Path(row["_audio"])
        speaker = source.parent.parent.name
        split = split_of[speaker]
        audio = load_16k_mono(source)
        if not .5 <= len(audio) / SAMPLE_RATE <= 16:
            continue

        # Seed by utterance and global seed so each build is reproducible.
        def rng_for(kind: str) -> np.random.Generator:
            seed_bytes = hashlib.sha256(f"{args.seed}:{source.stem}:{kind}".encode()).digest()[:8]
            return np.random.default_rng(int.from_bytes(seed_bytes, "little"))

        condition = CONDITIONS[int(rng_for("condition").integers(len(CONDITIONS)))]
        treatment = audio
        recipe: dict = {"condition": condition, "noise": None, "covering": None}
        if condition in ("covering-only", "covering+noise"):
            response_split = "train" if split == "train" else ("dev" if split == "dev" else "test")
            gain, filter_recipe = bank.sample(response_split, args.mode, rng_for("covering"))
            treatment = bank.apply(treatment, gain)
            recipe["covering"] = filter_recipe
        control = audio.copy()
        if condition in ("noise-only", "covering+noise"):
            noises = noise_by_split[split]
            noise_rng = rng_for("noise")
            noise_path = noises[int(noise_rng.integers(len(noises)))]
            noise = load_16k_mono(noise_path)
            snr = float(noise_rng.uniform(*args.snr))
            noise_seed = int(noise_rng.integers(0, 2**32))
            treatment = mix_noise(treatment, noise, snr, np.random.default_rng(noise_seed))
            control = mix_noise(control, noise, snr, np.random.default_rng(noise_seed))
            recipe["noise"] = {"file": str(noise_path.relative_to(args.noise_root)), "snr_db": snr}

        peak = float(np.abs(treatment).max(initial=0))
        if peak > .99:
            treatment = treatment * (.99 / peak)
        out = args.out_dir / split / speaker / f"{source.stem}.flac"
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out, treatment.astype(np.float32), SAMPLE_RATE)
        treatment_row = {
            "audio_filepath": str(out.resolve()),
            "duration": round(len(treatment) / SAMPLE_RATE, 3),
            "text": row["text"],
            "speaker": speaker,
            "source_id": source.stem,
            "condition": condition,
            "recipe": recipe,
        }
        manifests[split].append(treatment_row)
        counts[(split, condition)] += 1

        if args.noise_control_out:
            control_condition = "noise-only" if recipe["noise"] else "clean"
            control_peak = float(np.abs(control).max(initial=0))
            if control_peak > .99:
                control = control * (.99 / control_peak)
            control_path = args.noise_control_out / split / speaker / f"{source.stem}.flac"
            control_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(control_path, control.astype(np.float32), SAMPLE_RATE)
            control_row = dict(treatment_row)
            control_row.update({"audio_filepath": str(control_path.resolve()),
                                "condition": control_condition})
            control_row["recipe"] = {**recipe, "condition": control_condition,
                                      "covering_removed": recipe["covering"] is not None}
            manifests[f"control_{split}"].append(control_row)
        if (i + 1) % 1000 == 0:
            print(f"processed {i+1}/{len(rows)}", flush=True)

    for split in ("train", "dev", "test"):
        out_manifest = args.out_dir / f"{split}.json"
        out_manifest.parent.mkdir(parents=True, exist_ok=True)
        out_manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in manifests[split]))
        hours = sum(r["duration"] for r in manifests[split]) / 3600
        print(f"{split}: {len(manifests[split])} utterances, {hours:.2f} h, "
              f"{dict((c, counts[(split,c)]) for c in CONDITIONS)} -> {out_manifest}")

    if args.noise_control_out:
        for split in ("train", "dev", "test"):
            rows = manifests[f"control_{split}"]
            out_manifest = args.noise_control_out / f"{split}.json"
            out_manifest.parent.mkdir(parents=True, exist_ok=True)
            out_manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
            print(f"noise control {split}: {len(rows)} utterances -> {out_manifest}")

    provenance = {
        "seed": args.seed, "response_bank": str(args.bank), "response_mode": args.mode,
        "held_out_response_family": bank.data["held_out_test_family"],
        "speech_corpus": "LibriSpeech train-clean-100 with LibriSpeech-PC transcripts",
        "speech_license": "CC BY 4.0",
        "speech_manifest": str(args.pc_manifest.resolve()),
        "speech_root": str(args.speech_root.resolve()),
        "noise_root": str(args.noise_root.resolve()),
        "speech_speakers": {s: split for s, split in split_of.items()},
        "noise_groups": noise_split,
        "conditions": CONDITIONS,
        "snr_range_db": args.snr,
    }
    (args.out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    if args.noise_control_out:
        (args.noise_control_out / "provenance.json").write_text(json.dumps(
            {**provenance, "arm": "matched-noise-control", "covering_applied": False}, indent=2))


if __name__ == "__main__":
    main()
