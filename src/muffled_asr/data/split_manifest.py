"""Split a manifest into train/dev/test, and repair relative audio paths.

Two jobs, both prerequisites for training:

PATHS. NeMo resolves `audio_filepath` as given. A manifest holding bare
filenames only works if the trainer happens to run in that directory, which it
will not on a GPU box. Paths are rewritten absolute against --audio-root
(default: the manifest's own directory) and every file is checked to exist.

GROUPING. Utterances are assigned to splits in whole groups, never individually.
LJSpeech is a single speaker, so speaker-disjoint splits are impossible — but
neighbouring clips come from the same passage and recording session, so a random
split puts near-identical context on both sides and makes val_wer look better
than the model is. Grouping by book (LJ001, LJ002, ...) is the honest option
available. For LibriSpeech use --group-by parent, which groups by speaker
directory and gives a genuinely speaker-disjoint split.

    python src/muffled_asr/data/split_manifest.py datasets/masked/ljspeech/manifest.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def group_key(entry: dict, mode: str) -> str:
    path = Path(entry["audio_filepath"])
    if mode == "parent":
        return path.parent.name
    if mode == "prefix":
        return path.stem.split("-")[0]
    return path.stem  # "none": every utterance its own group


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--audio-root", type=Path, help="base for relative paths [manifest's directory]")
    parser.add_argument("--out-dir", type=Path, help="where the splits go [manifest's directory]")
    parser.add_argument("--group-by", default="prefix", choices=["prefix", "parent", "none"])
    parser.add_argument("--dev-frac", type=float, default=0.05)
    parser.add_argument("--test-frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    audio_root = args.audio_root or args.manifest.parent
    out_dir = args.out_dir or args.manifest.parent

    entries = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    if not entries:
        raise SystemExit(f"{args.manifest} is empty")

    missing = []
    for entry in entries:
        path = Path(entry["audio_filepath"])
        if not path.is_absolute():
            path = audio_root / path
        path = path.resolve()
        if not path.exists():
            missing.append(str(path))
        entry["audio_filepath"] = str(path)
    if missing:
        raise SystemExit(
            f"{len(missing)} audio files not found, e.g. {missing[0]}\n"
            f"Pass --audio-root pointing at the directory that holds them."
        )

    blank = sum(1 for e in entries if not e.get("text", "").strip())
    if blank:
        raise SystemExit(f"{blank} entries have empty text — fill them in before splitting")

    groups: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        groups[group_key(entry, args.group_by)].append(entry)

    names = sorted(groups)
    random.Random(args.seed).shuffle(names)

    # Assign whole groups until each split has its share of total utterances.
    targets = {"test": args.test_frac * len(entries), "dev": args.dev_frac * len(entries)}
    splits: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    for name in names:
        for split in ("test", "dev"):
            if len(splits[split]) < targets[split]:
                splits[split].extend(groups[name])
                break
        else:
            splits["train"].extend(groups[name])

    print(f"{len(entries)} utterances in {len(groups)} groups (--group-by {args.group_by})\n")
    for split, rows in splits.items():
        path = out_dir / f"{split}.json"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        hours = sum(r["duration"] for r in rows) / 3600
        held = len({group_key(r, args.group_by) for r in rows})
        print(f"{path}  {len(rows):>6} utts  {hours:6.2f} h  {held:>3} groups")


if __name__ == "__main__":
    main()
