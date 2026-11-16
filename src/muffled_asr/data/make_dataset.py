"""Turn a clean LibriSpeech tree into a masked corpus with NeMo manifests.

The mask is a filter, so the audio can be manufactured while the transcripts
carry over untouched. That is the whole economics of this approach: 100 hours of
labelled masked speech for the cost of a disk write, no recording session.

Splits are speaker-disjoint. Sharing a speaker between train and test inflates
ASR results badly, and LibriSpeech makes the mistake easy to commit because the
speaker id is only ever a directory name.

Transcripts: stock LibriSpeech is ALL CAPS with no punctuation, and this
checkpoint emits punctuation and capitalization natively. Training on stock
labels destroys that within a few hundred steps, so it is refused unless you
pass --allow-uppercase deliberately. Point --pc-manifest at LibriSpeech-PC
(openslr.org/145 — manifests only, 25 MB, reuses the audio you already have) to
get properly cased and punctuated labels instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf

from muffled_asr.simulation.mask_sim import MASKS, SAMPLE_RATE, apply_mask, load_16k_mono


def read_stock_transcripts(root: Path) -> dict[str, str]:
    """LibriSpeech ships one *.trans.txt per chapter: "<utt-id> TEXT"."""
    texts: dict[str, str] = {}
    for transcript in root.rglob("*.trans.txt"):
        for line in transcript.read_text().splitlines():
            if line:
                utterance_id, _, text = line.partition(" ")
                texts[utterance_id] = text
    return texts


def read_pc_manifest(path: Path) -> dict[str, str]:
    """LibriSpeech-PC manifest lines, keyed by utterance id from the audio path."""
    texts: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        texts[Path(entry["audio_filepath"]).stem] = entry["text"]
    return texts


def split_for(speaker: str, dev: set[str], test: set[str]) -> str:
    if speaker in test:
        return "test"
    return "dev" if speaker in dev else "train"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("librispeech", type=Path, help="LibriSpeech root (speaker dirs)")
    parser.add_argument("out_dir", type=Path, help="where masked audio and manifests go")
    parser.add_argument("--mask", default="welding_helmet", choices=sorted(MASKS))
    parser.add_argument("--pc-manifest", type=Path, help="LibriSpeech-PC manifest (punctuated labels)")
    parser.add_argument("--allow-uppercase", action="store_true", help="accept stock labels, losing PnC")
    parser.add_argument("--dev-speakers", type=int, default=5)
    parser.add_argument("--test-speakers", type=int, default=5)
    parser.add_argument("--limit", type=int, help="stop after this many utterances (for a pilot run)")
    parser.add_argument("--max-duration", type=float, default=20.0, help="skip longer clips, as NeMo would")
    args = parser.parse_args()

    if args.pc_manifest:
        texts = read_pc_manifest(args.pc_manifest)
    elif args.allow_uppercase:
        texts = read_stock_transcripts(args.librispeech)
        print("WARNING: stock LibriSpeech labels — training on these removes punctuation\n"
              "         and capitalization from the model. See --pc-manifest.\n")
    else:
        raise SystemExit(
            "Refusing to use stock LibriSpeech transcripts: they are uppercase and\n"
            "unpunctuated, and fine-tuning on them strips PnC from the model.\n"
            "Pass --pc-manifest <LibriSpeech-PC manifest>, or --allow-uppercase to\n"
            "override deliberately."
        )

    speakers = sorted(d.name for d in args.librispeech.iterdir() if d.is_dir())
    test = set(speakers[: args.test_speakers])
    dev = set(speakers[args.test_speakers : args.test_speakers + args.dev_speakers])
    print(f"{len(speakers)} speakers -> {len(dev)} dev, {len(test)} test, rest train")
    print(f"mask: {args.mask} — {MASKS[args.mask].note}\n")

    manifests = {name: [] for name in ("train", "dev", "test")}
    skipped_long = skipped_unlabelled = 0

    clips = sorted(args.librispeech.rglob("*.flac"))
    for index, clip in enumerate(clips):
        if args.limit and index >= args.limit:
            break
        text = texts.get(clip.stem)
        if text is None:  # LibriSpeech-PC drops utterances it could not restore
            skipped_unlabelled += 1
            continue

        audio = load_16k_mono(clip)
        duration = len(audio) / SAMPLE_RATE
        if duration > args.max_duration:
            skipped_long += 1
            continue

        speaker = clip.relative_to(args.librispeech).parts[0]
        split = split_for(speaker, dev, test)
        destination = args.out_dir / split / speaker / f"{clip.stem}.flac"
        destination.parent.mkdir(parents=True, exist_ok=True)
        sf.write(destination, apply_mask(audio, args.mask), SAMPLE_RATE)

        manifests[split].append(
            {"audio_filepath": str(destination.resolve()), "duration": round(duration, 3), "text": text}
        )
        if (index + 1) % 500 == 0:
            print(f"  {index + 1} clips...")

    for split, entries in manifests.items():
        path = args.out_dir / f"{split}.json"
        path.write_text("".join(json.dumps(e) + "\n" for e in entries))
        hours = sum(e["duration"] for e in entries) / 3600
        print(f"{path}  {len(entries):>6} utterances  {hours:.2f} h")

    if skipped_long:
        print(f"\nskipped {skipped_long} clips over {args.max_duration}s")
    if skipped_unlabelled:
        print(f"skipped {skipped_unlabelled} clips with no transcript")


if __name__ == "__main__":
    main()
