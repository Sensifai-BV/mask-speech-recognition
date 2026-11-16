"""Apply a simulated mask to a directory of clean speech.

Writes 16 kHz mono WAVs plus a NeMo manifest:

    {"audio_filepath": "/abs/path/LJ001-0001.wav", "duration": 9.66, "text": "Printing, in the only..."}

Pass --transcripts to fill the text field from the corpus's own ground truth;
without it the field is left empty for manual entry. Two layouts are understood:

    a file       LJSpeech `metadata.csv` — "<id>|raw|normalized", pipe delimited.
                 Also any "<id> <text>" or "<id>|<text>" list.
    a directory  LibriSpeech — every *.trans.txt below it, "<id> TEXT" per line.

For LJSpeech the *normalized* column is used, because it writes numbers the way
they are actually spoken ("1455" -> "fourteen fifty-five") and that is what the
model has to emit. Pass --raw-text for the verbatim column instead.

Output keeps the input's directory layout, so whatever grouping the source uses
(speaker, session, take) survives into the generated set and can still be used
to build speaker-disjoint splits later.

Usage:

    python src/muffled_asr/data/generate_masked_wavs.py <clean-dir> <out-dir>
    python src/muffled_asr/data/generate_masked_wavs.py datasets/original/LJSpeech-1.1/wavs out \
        --transcripts datasets/original/LJSpeech-1.1/metadata.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf


from muffled_asr.simulation.mask_sim import MASKS, SAMPLE_RATE, apply_mask, load_16k_mono  # noqa: E402

AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus"}


def read_transcripts(path: Path, raw_text: bool) -> dict[str, str]:
    """Utterance id -> text, from a metadata file or a LibriSpeech tree."""
    texts: dict[str, str] = {}

    if path.is_dir():
        for transcript in sorted(path.rglob("*.trans.txt")):
            for line in transcript.read_text().splitlines():
                if line.strip():
                    utterance_id, _, text = line.partition(" ")
                    texts[utterance_id] = text.strip()
        if not texts:
            raise SystemExit(f"no *.trans.txt found under {path}")
        return texts

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if "|" in line:
            fields = line.split("|")
            utterance_id = fields[0]
            # LJSpeech ships raw then normalized; prefer the last column.
            text = fields[1] if raw_text and len(fields) > 2 else fields[-1]
        else:
            utterance_id, _, text = line.partition(" ")
        texts[utterance_id.strip()] = text.strip()

    if not texts:
        raise SystemExit(f"no transcripts parsed from {path}")
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("clean_dir", type=Path, help="directory of clean speech, searched recursively")
    parser.add_argument("out_dir", type=Path, help="where the masked WAVs and manifest go")
    parser.add_argument("--mask", default="welding_helmet", choices=sorted(MASKS))
    parser.add_argument("--transcripts", type=Path, help="metadata file, or a dir of LibriSpeech *.trans.txt")
    parser.add_argument("--raw-text", action="store_true", help="LJSpeech: verbatim column, not normalized")
    parser.add_argument("--manifest", type=Path, help="manifest path (default: <out-dir>/manifest.json)")
    args = parser.parse_args()

    clips = sorted(p for p in args.clean_dir.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES)
    if not clips:
        raise SystemExit(f"no audio found under {args.clean_dir}")

    texts = read_transcripts(args.transcripts, args.raw_text) if args.transcripts else {}
    manifest_path = args.manifest or args.out_dir / "manifest.json"
    print(f"{len(clips)} files   mask: {args.mask} — {MASKS[args.mask].note}")
    if texts:
        print(f"{len(texts)} transcripts from {args.transcripts}")

    entries = []
    skipped = []
    missing_text = []
    for clip in clips:
        try:
            audio = load_16k_mono(clip)
        except SystemExit as reason:  # below 16 kHz: the mask band is not in the file
            skipped.append((clip, str(reason)))
            continue

        relative = clip.relative_to(args.clean_dir).with_suffix(".wav")
        destination = args.out_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        sf.write(destination, apply_mask(audio, args.mask), SAMPLE_RATE, subtype="PCM_16")

        text = texts.get(clip.stem, "")
        if texts and not text:
            missing_text.append(clip.stem)

        entries.append(
            {
                "audio_filepath": str(destination.resolve()),
                "duration": round(len(audio) / SAMPLE_RATE, 3),
                "text": text,
            }
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries))

    hours = sum(e["duration"] for e in entries) / 3600
    print(f"\n{args.out_dir}   {len(entries)} wavs, {hours:.2f} h")
    if texts:
        print(f"{manifest_path}   {len(entries) - len(missing_text)} of {len(entries)} entries have text")
    else:
        print(f"{manifest_path}   text fields empty — fill them in before training")
    for clip, reason in skipped:
        print(f"  skipped {clip.name}: {reason}")
    if missing_text:
        print(f"  no transcript for {len(missing_text)}: {', '.join(missing_text[:5])}"
              + (" ..." if len(missing_text) > 5 else ""))


if __name__ == "__main__":
    main()
