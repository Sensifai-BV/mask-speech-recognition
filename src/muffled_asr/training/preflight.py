"""Check a manifest is trainable, and say precisely what is wrong if it is not.

NeMo opens `audio_filepath` exactly as written and fails per-file deep inside
the dataloader, hours in. This checks every entry up front instead.

Prints the manifest to train on to stdout; diagnostics go to stderr.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def log(message: str = "") -> None:
    print(message, file=sys.stderr)


def rebase_paths(rows: list[dict], root: Path) -> int:
    """Rewrite audio paths that do not exist here onto `root`, in place.

    A manifest built on another machine carries that machine's absolute paths.
    Supports <split>/<speaker>/<file> (synthetic data) and <split>/<file>
    (Cox human recordings). Returns how many were repaired.
    """
    repaired = 0
    for row in rows:
        if os.path.exists(row["audio_filepath"]):
            continue
        parts = Path(row["audio_filepath"]).parts
        for depth in (3, 2):
            candidate = root.joinpath(*parts[-depth:])
            if candidate.is_file():
                row["audio_filepath"] = str(candidate.resolve())
                repaired += 1
                break
    return repaired


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--skip-missing", action="store_true", help="drop unusable entries and continue")
    parser.add_argument("--fix-paths", action="store_true",
                        help="rewrite audio_filepath against the manifest's own directory")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    if not rows:
        log(f"ERROR: {args.manifest} is empty")
        raise SystemExit(1)

    if args.fix_paths:
        repaired = rebase_paths(rows, args.manifest.parent)
        if repaired:
            args.manifest.write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
            )
            log(f"{args.manifest}: rebased {repaired} paths onto {args.manifest.parent}")

    missing = [r for r in rows if not os.path.exists(r["audio_filepath"])]
    blank = [r for r in rows if not r.get("text", "").strip()]
    good = [r for r in rows if os.path.exists(r["audio_filepath"]) and r.get("text", "").strip()]

    hours = sum(r["duration"] for r in good) / 3600
    log(f"{args.manifest}: {len(rows)} entries, {len(good)} usable ({hours:.2f} h)")

    if missing:
        pct = 100 * len(missing) / len(rows)
        log(f"  {len(missing)} missing audio files ({pct:.1f}%)")
        for row in missing[:3]:
            log(f"    {row['audio_filepath']}")
        # A handful means a broken generation run; most of them means the audio
        # was never copied to this machine, and skipping would hide that.
        log("  -> incomplete copy or interrupted generation." if pct > 20 else
            "  -> a few clips failed to generate.")
    if blank:
        log(f"  {len(blank)} entries with empty text")

    if (missing or blank) and not args.skip_missing:
        log()
        log("Nothing was changed. Either fix the source data, or rerun with --skip-missing")
        log(f"to train on the {len(good)} usable entries.")
        raise SystemExit(1)

    if not good:
        log("ERROR: no usable entries remain")
        raise SystemExit(1)

    if missing or blank:
        filtered = args.manifest.with_suffix(".usable.json")
        filtered.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in good))
        log(f"  -> wrote {filtered} with {len(good)} entries")
        print(filtered)
    else:
        print(args.manifest)


if __name__ == "__main__":
    main()
