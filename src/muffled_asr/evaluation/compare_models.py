"""Score base vs tuned on held-out speakers with Transformers, per chunk size.

The Mac-side twin of `src/muffled_asr/training/eval_adapter.py`: same manifest, same text
normalisation, same clean / masked+noisy split, so its numbers can be laid next
to the NeMo run. Matching numbers mean the Transformers port of the adapter is
faithful; the GUI then runs exactly what was measured.

    python src/muffled_asr/evaluation/compare_models.py                       # full test set
    python src/muffled_asr/evaluation/compare_models.py --limit 300 --latencies 80
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from preflight import rebase_paths  # noqa: E402

from muffled_asr.simulation.mask_sim import SAMPLE_RATE, load_16k_mono  # noqa: E402
from muffled_asr.inference.asr import load_model, pick_device  # noqa: E402

LOOKAHEADS = {80: 0, 160: 1, 560: 6, 1120: 13}

# RNN-T generation ends at the audio, not at max_length; the default only warns.
warnings.filterwarnings("ignore", message="Using the model-agnostic default `max_length`")


def normalize(text: str) -> list[str]:
    return re.sub(r"[^a-z' ]", " ", text.lower()).split()


def edits(ref: list[str], hyp: list[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(prev[j - 1] if r == h else 1 + min(prev[j - 1], prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def transcribe(processor, model, audio) -> str:
    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    inputs = inputs.to(model.device, dtype=model.dtype)
    with torch.inference_mode():
        output = model.generate(**inputs, return_dict_in_generate=True)
    text = processor.decode(output.sequences, skip_special_tokens=True)
    return text[0] if isinstance(text, list) else text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--manifest", type=Path, default=Path("datasets/masked/librispeech-welding/test.json"))
    parser.add_argument("--base", type=Path, default=Path("models/nemotron-speech-streaming-en-0.6b"))
    parser.add_argument("--tuned", type=Path, default=Path("models/nemotron-welding"))
    parser.add_argument("--latencies", type=int, nargs="+", default=list(LOOKAHEADS), choices=list(LOOKAHEADS))
    parser.add_argument("--limit", type=int, help="first N rows, as eval_adapter.py --limit")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    rebase_paths(rows, args.manifest.parent)
    rows = rows[: args.limit] if args.limit else rows
    missing = [r["audio_filepath"] for r in rows if not os.path.exists(r["audio_filepath"])]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(rows)} audio files not found, e.g.\n  {missing[0]}")

    audio = [load_16k_mono(Path(r["audio_filepath"])) for r in rows]
    refs = [normalize(r["text"]) for r in rows]
    conditions = sorted({r.get("condition", "clean") for r in rows})
    groups = {condition: [i for i, r in enumerate(rows)
                          if r.get("condition", "clean") == condition]
              for condition in conditions}
    groups["all"] = list(range(len(rows)))
    device = pick_device()
    summary = ", ".join(f"{condition} {len(groups[condition])}"
                         for condition in conditions)
    print(f"{len(rows)} held-out utterances: {summary}, on {device}\n")

    results: dict[str, dict[int, dict[str, float]]] = {}
    for label, model_dir in (("base", args.base), ("tuned", args.tuned)):
        print(f"=== {label}: {model_dir} ===")
        processor, model = load_model(model_dir, device)
        results[label] = {}
        for ms in args.latencies:
            processor.set_num_lookahead_tokens(LOOKAHEADS[ms])
            hyps = [normalize(transcribe(processor, model, a)) for a in audio]
            results[label][ms] = {
                group: 100 * sum(edits(refs[i], hyps[i]) for i in idx) / max(sum(len(refs[i]) for i in idx), 1)
                for group, idx in groups.items()
            }
            r = results[label][ms]
            breakdown = "   ".join(f"{condition} {r[condition]:5.2f}%"
                                    for condition in conditions)
            print(f"  {ms:>5} ms   all {r['all']:5.2f}%   {breakdown}", flush=True)
        del model

    print("\nWER delta, tuned - base (negative = tuned better)\n")
    print(f"{'chunk':>8} {'all':>9} " + " ".join(f"{c:>18}" for c in conditions))
    for ms in args.latencies:
        d = {g: results["tuned"][ms][g] - results["base"][ms][g] for g in groups}
        print(f"{ms:>6}ms {d['all']:>+9.2f} "
              + " ".join(f"{d[c]:>+18.2f}" for c in conditions))


if __name__ == "__main__":
    main()
