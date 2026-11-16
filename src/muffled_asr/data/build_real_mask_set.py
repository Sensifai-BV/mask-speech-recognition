"""Build a training corpus from real recordings of speech through face masks.

Everything else in this project simulates the mask. These two PLOS ONE
supplementary sets are real microphone recordings of people wearing one:

NWS (pone.0283724, S1-S4). 30 talkers read "The North Wind and the Sun" twice,
masked and unmasked, in English (E) and Mandarin (M), 48 kHz, with a forced-
aligned word tier. Only English is used: the model is English-only. The shipped
alignments are unreliable (five readings contain retakes the aligner ignored),
so each reading is instead cut at its own pauses into pieces of at most 12 s.
The base model transcribes each piece only to find where it sits in the known
passage; the label is that span of the canonical punctuated text. Pieces that
cannot be placed unambiguously (retakes, missed edge words) are dropped.

HINT (journal.pone.0246842, S4-S5). 2 talkers x 5 masks (none, surgical, N95,
two cloth) x 20 HINT sentences, pre-mixed with six-talker babble at +3 and
+13 dB, 22.05 kHz. The archive carries no transcripts; labels come from
datasets/original/hint_lists_3_19.tsv.

SPLITS. NWS is speaker-disjoint (stratified by sex). HINT has only two talkers,
so it is sentence-disjoint instead -- otherwise every test sentence would have
been heard in training. NWS speakers all read the same passage, so its dev/test
WER is optimistic once the model has trained on that text: use HINT, or the
masked test sets elsewhere, as the honest number.

Output follows build_training_set.py: 16 kHz mono FLAC under
<out>/<split>/<speaker>/, and NeMo manifests <out>/{train,dev,test}.json.

    python src/muffled_asr/data/build_real_mask_set.py datasets/original datasets/masked/real-masks
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf


from muffled_asr.simulation.mask_sim import SAMPLE_RATE, load_16k_mono  # noqa: E402

LOOKAHEAD_1120MS = 13  # most accurate chunk size: placement only needs the words right

PASSAGE = (
    "One day the North Wind and the Sun were disputing which was the stronger, when a traveler "
    "came along wrapped in a warm cloak. They agreed that the one who first succeeded in making "
    "the traveler take his cloak off should be considered stronger than the other. Then the North "
    "Wind blew as hard as he could, but the more he blew the more closely did the traveler fold "
    "his cloak around him; and at last the North Wind gave up the attempt. Then the Sun shined out "
    "warmly, and immediately the traveler took off his cloak. And so the North Wind was obliged to "
    "confess that the Sun was the stronger of the two."
)
# Spellings the model uses for words the passage spells differently.
SPELLING = {"traveller": "traveler", "shone": "shined"}

HINT_NAME = re.compile(r"List\d+_(?P<mask>.+)_(?P<list>\d\d)_(?P<sentence>\d\d)_(?P<talker>jt|ct)_SNR(?P<snr>\d+)\.wav")

FRAME = SAMPLE_RATE // 50  # 20 ms
MIN_PAUSE = 0.2  # quiet run long enough to cut at, seconds
EDGE_PAD = 0.25  # silence kept either side of a clip, seconds
MAX_CLIP = 12.0  # pieces longer than this are split again at their longest pause
MAX_DURATION = 16.0  # trainer's max_duration
# A piece is kept only if the base model's words match the passage span it was
# placed on this well. It rejects retakes and misplacements, not hard audio: the
# base model makes ~5% WER on correctly placed pieces, masked or not.
MAX_PLACEMENT_WER = 0.4


def words(text: str) -> list[str]:
    return [SPELLING.get(w, w) for w in re.sub(r"[^a-z' ]", " ", text.lower()).split()]


def edits(ref: list[str], hyp: list[str]) -> list[tuple[int | None, int | None]]:
    """Levenshtein alignment of two word lists, as (ref index, hyp index) pairs;
    None on one side marks an insertion or deletion."""
    n, m = len(ref), len(hyp)
    cost = np.zeros((n + 1, m + 1), dtype=np.int32)
    cost[:, 0], cost[0, :] = np.arange(n + 1), np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i, j] = min(cost[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]), cost[i - 1, j] + 1, cost[i, j - 1] + 1)
    pairs, i, j = [], n, m
    while i or j:
        if i and j and cost[i, j] == cost[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            i, j = i - 1, j - 1
            pairs.append((i, j))
        elif i and cost[i, j] == cost[i - 1, j] + 1:
            i -= 1
            pairs.append((i, None))
        else:
            j -= 1
            pairs.append((None, j))
    return pairs[::-1]


def speech_pieces(audio: np.ndarray) -> list[tuple[int, int]]:
    """Cut a long reading at its pauses into pieces of at most MAX_CLIP seconds.

    Always splits at the longest pause available, so cuts land between
    sentences and phrases, never inside a word.
    """
    frames = audio[: len(audio) // FRAME * FRAME].reshape(-1, FRAME)
    level = 10 * np.log10(np.mean(frames**2, axis=1) + 1e-10)
    floor, peak = np.percentile(level, [10, 95])
    quiet = level < floor + 0.25 * (peak - floor)
    loud = np.flatnonzero(~quiet)
    if not len(loud):
        return []

    pauses, run = [], None
    for index, is_quiet in enumerate(quiet):
        if is_quiet and run is None:
            run = index
        elif not is_quiet and run is not None:
            if (index - run) * FRAME >= MIN_PAUSE * SAMPLE_RATE:
                pauses.append((run, index))
            run = None

    def split(lo: int, hi: int) -> list[tuple[int, int]]:
        if (hi - lo) * FRAME <= MAX_CLIP * SAMPLE_RATE:
            return [(lo, hi)]
        margin = SAMPLE_RATE // FRAME  # never leave a piece under a second
        inside = [(a, b) for a, b in pauses if a > lo + margin and b < hi - margin]
        if not inside:
            return [(lo, hi)]
        a, b = max(inside, key=lambda p: p[1] - p[0])
        return split(lo, (a + b) // 2) + split((a + b) // 2, hi)

    pad = int(EDGE_PAD * SAMPLE_RATE / FRAME)
    pieces = []
    for lo, hi in split(loud[0], loud[-1] + 1):
        speech = loud[(loud >= lo) & (loud < hi)]
        if len(speech):
            pieces.append((max(speech[0] - pad, 0) * FRAME, min(speech[-1] + 1 + pad, len(quiet)) * FRAME))
    return pieces


def place(passage: list[str], hyps: list[list[str]]) -> list[tuple[int, int] | None]:
    """Where in the passage each piece's words fall, as an inclusive word span.

    One global alignment of all pieces against the passage, so every passage word
    is claimed by at most one piece. A piece whose edge word was missed by the
    model is ambiguous (that word could belong to either neighbour) and gets None.
    """
    flat = [(c, w) for c, ws in enumerate(hyps) for w in ws]
    owner: list[int | None] = [None] * len(passage)
    matched = [0] * len(hyps)
    for r, h in edits(passage, [w for _, w in flat]):
        if r is not None and h is not None:
            owner[r] = flat[h][0]
            matched[flat[h][0]] += passage[r] == flat[h][1]

    spans: list[tuple[int, int] | None] = []
    for c, hyp in enumerate(hyps):
        claimed = [r for r, o in enumerate(owner) if o == c]
        if not hyp or not claimed or matched[c] < 0.6 * len(hyp):
            spans.append(None)
            continue
        lo, hi = claimed[0], claimed[-1]
        contiguous = all(owner[r] in (c, None) for r in range(lo, hi + 1))
        clear_edges = (lo == 0 or owner[lo - 1] is not None) and (hi == len(passage) - 1 or owner[hi + 1] is not None)
        spans.append((lo, hi) if contiguous and clear_edges else None)
    return spans


def nws_clips(root: Path, rng: random.Random, dev: int, test: int, transcribe):
    readings = sorted(root.rglob("*_E_*.wav"))
    if not readings:
        raise SystemExit(f"no English NWS recordings under {root}")
    speakers = sorted({"_".join(r.name.split("_")[:2]) for r in readings})  # S01_F
    split_of = {}
    for sex in ("F", "M"):
        group = [s for s in speakers if s.endswith(sex)]
        rng.shuffle(group)
        n_test, n_dev = round(test * len(group) / len(speakers)), round(dev * len(group) / len(speakers))
        for i, speaker in enumerate(group):
            split_of[speaker] = "test" if i < n_test else "dev" if i < n_test + n_dev else "train"

    tokens = PASSAGE.split()
    passage = words(PASSAGE)
    assert len(tokens) == len(passage)
    for reading in readings:
        speaker, sex, _, mask, _ = reading.stem.split("_")
        audio = load_16k_mono(reading)
        pieces = speech_pieces(audio)
        hyps = [words(transcribe(audio[a:b])) for a, b in pieces]
        kept = 0
        for index, ((a, b), hyp, span) in enumerate(zip(pieces, hyps, place(passage, hyps))):
            if span is None:
                continue
            lo, hi = span
            ref = passage[lo : hi + 1]
            errors = sum(r is None or h is None or ref[r] != hyp[h] for r, h in edits(ref, hyp))
            if errors > MAX_PLACEMENT_WER * len(ref):
                continue
            kept += 1
            yield {
                "split": split_of[f"{speaker}_{sex}"],
                "speaker": f"nws-{speaker}_{sex}",
                "id": f"{reading.stem}_p{index + 1:02d}",
                "audio": audio[a:b],
                "text": " ".join(tokens[lo : hi + 1]),  # passage casing, as LibriSpeech-PC
                "condition": "real-mask" if mask == "MSK" else "clean",
            }
        print(f"  {reading.stem}: kept {kept}/{len(pieces)} pieces", flush=True)


def extract(archive_dir: Path, work: Path) -> None:
    for archive in sorted(archive_dir.iterdir()):
        if archive.name.startswith("journal.pone.0246842"):
            with zipfile.ZipFile(archive) as z:
                z.extractall(work / "hint", [n for n in z.namelist() if n.endswith(".wav") and "__MACOSX" not in n])
        elif archive.suffix == ".rar" and archive.name.startswith("pone.0283724"):
            tool = shutil.which("unrar") or shutil.which("bsdtar")
            if not tool:
                raise SystemExit("extracting .rar needs `unrar` or `bsdtar` on PATH (brew install rar)")
            (work / "nws").mkdir(parents=True, exist_ok=True)
            cmd = ([tool, "x", "-o+", "-idq", str(archive), f"{work / 'nws'}/"] if tool.endswith("unrar")
                   else [tool, "-xf", str(archive), "-C", str(work / "nws")])
            subprocess.run(cmd, check=True)


def hint_clips(root: Path, labels: dict[str, str], rng: random.Random, dev: int, test: int):
    files = sorted(root.rglob("*.wav"))
    parsed = [(f, HINT_NAME.fullmatch(f.name)) for f in files]
    if not files or any(m is None for _, m in parsed):
        raise SystemExit(f"unexpected HINT file names under {root}")
    sentences = sorted({f"{m['list']}_{m['sentence']}" for _, m in parsed})
    missing = [s for s in sentences if s not in labels]
    if missing:
        raise SystemExit(f"no label for HINT sentences {missing}")
    rng.shuffle(sentences)
    split_of = {s: "test" if i < test else "dev" if i < test + dev else "train" for i, s in enumerate(sentences)}

    for path, m in parsed:
        key = f"{m['list']}_{m['sentence']}"
        mask = "unmasked" if m["mask"] == "no_mask" else m["mask"]
        yield {
            "split": split_of[key],
            "speaker": f"hint-{m['talker']}",
            "id": f"{key}_{m['mask']}_{m['talker']}_snr{m['snr']}",
            "audio": load_16k_mono(path),  # 22.05 kHz: a plain downsample
            "text": labels[key],
            "condition": f"{mask}+babble@{m['snr']}dB",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("archives", type=Path, help="directory holding the PLOS ONE supplementary archives")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--hint-labels", type=Path, help="[<archives>/hint_lists_3_19.tsv]")
    parser.add_argument("--nws-dev-speakers", type=int, default=5)
    parser.add_argument("--nws-test-speakers", type=int, default=5)
    parser.add_argument("--hint-dev-sentences", type=int, default=3)
    parser.add_argument("--hint-test-sentences", type=int, default=3)
    parser.add_argument("--model", type=Path, default=Path("models/nemotron-speech-streaming-en-0.6b"),
                        help="places NWS pieces in the passage; labels never come from it")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from compare_models import transcribe
    from muffled_asr.inference.asr import load_model, pick_device

    processor, model = load_model(args.model, pick_device())
    processor.set_num_lookahead_tokens(LOOKAHEAD_1120MS)

    label_file = args.hint_labels or args.archives / "hint_lists_3_19.tsv"
    labels = dict(
        line.split("\t", 1) for line in label_file.read_text().splitlines() if line.strip() and not line.startswith("#")
    )

    manifests: dict[str, list[dict]] = defaultdict(list)
    counts: Counter = Counter()
    too_long = 0
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        print(f"extracting {args.archives} ...", flush=True)
        extract(args.archives, work)
        rng = random.Random(args.seed)
        sources = (
            ("nws", nws_clips(work / "nws", rng, args.nws_dev_speakers, args.nws_test_speakers,
                                     lambda audio: transcribe(processor, model, audio))),
            ("hint", hint_clips(work / "hint", labels, rng, args.hint_dev_sentences, args.hint_test_sentences)),
        )
        for source, clips in sources:
            for clip in clips:
                audio = clip["audio"]
                duration = len(audio) / SAMPLE_RATE
                if duration > MAX_DURATION:
                    too_long += 1
                    continue
                peak = np.abs(audio).max()
                if peak > 0.99:
                    audio = audio * (0.99 / peak)
                destination = args.out_dir / clip["split"] / clip["speaker"] / f"{clip['id']}.flac"
                destination.parent.mkdir(parents=True, exist_ok=True)
                sf.write(destination, audio.astype(np.float32), SAMPLE_RATE)
                manifests[clip["split"]].append({
                    "audio_filepath": str(destination.resolve()),
                    "duration": round(duration, 3),
                    "text": clip["text"],
                    "condition": clip["condition"],
                    "source": source,
                })
                counts[(source, clip["split"])] += 1

    print()
    for split in ("train", "dev", "test"):
        rows = manifests[split]
        (args.out_dir / f"{split}.json").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        hours = sum(r["duration"] for r in rows) / 3600
        print(f"{split:>5}: {len(rows):4d} clips, {hours:.2f} h  "
              f"(nws {counts[('nws', split)]}, hint {counts[('hint', split)]})  -> {args.out_dir / split}.json")
    if too_long:
        print(f"skipped {too_long} clips over {MAX_DURATION:.0f} s")


if __name__ == "__main__":
    main()
