"""Measure what a simulated mask costs the model, per streaming latency setting.

Answers the only question that decides whether a fine-tune is worth doing: how
much accuracy does the mask actually take away, and at which chunk sizes?
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForRNNT, AutoProcessor

from muffled_asr.simulation.mask_sim import MASKS, SAMPLE_RATE, apply_mask, load_16k_mono, mix_noise
from muffled_asr.inference.asr import ensure_model, pick_device

# {lookahead tokens: chunk milliseconds}, per the model card.
LOOKAHEADS = {0: 80, 1: 160, 6: 560, 13: 1120}


def normalize(text: str) -> list[str]:
    """Upper-case, strip punctuation — LibriSpeech references carry neither case
    nor punctuation, so both sides have to be flattened before comparison."""
    return re.sub(r"[^A-Z' ]", " ", text.upper()).split()


def wer(reference: list[str], hypothesis: list[str]) -> float:
    previous = list(range(len(hypothesis) + 1))
    for i, ref_word in enumerate(reference, 1):
        current = [i]
        for j, hyp_word in enumerate(hypothesis, 1):
            current.append(
                previous[j - 1] if ref_word == hyp_word
                else 1 + min(previous[j - 1], previous[j], current[j - 1])
            )
        previous = current
    return previous[-1] / max(len(reference), 1)


def read_references(chapter: Path) -> dict[str, str]:
    (transcript,) = chapter.glob("*.trans.txt")
    lines = (line.split(" ", 1) for line in transcript.read_text().splitlines() if line)
    return {utterance_id: text for utterance_id, text in lines}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("chapter", type=Path, help="a LibriSpeech chapter directory")
    parser.add_argument("--limit", type=int, default=10, help="utterances to score")
    parser.add_argument("--masks", nargs="+", default=["surgical", "n95", "respirator", "welding_helmet"])
    parser.add_argument("--noise", type=Path, help="noise recording to mix in after the mask")
    parser.add_argument("--snr", type=float, nargs="+", default=[10.0, 5.0, 0.0])
    args = parser.parse_args()

    references = read_references(args.chapter)
    clips = sorted(args.chapter.glob("*.flac"))[: args.limit]

    device = pick_device()
    path = ensure_model()
    processor = AutoProcessor.from_pretrained(path)
    model = AutoModelForRNNT.from_pretrained(path).to(device).eval()
    print(f"{len(clips)} utterances on {device}\n")

    audio = {"clean": [load_16k_mono(c) for c in clips]}
    for mask in args.masks:
        audio[mask] = [apply_mask(a, mask) for a in audio["clean"]]

    if args.noise:
        noise = load_16k_mono(args.noise)
        for condition in list(audio):
            for snr in args.snr:
                rng = np.random.default_rng(0)  # same noise crops across conditions
                audio[f"{condition} @{snr:g}dB"] = [mix_noise(a, noise, snr, rng) for a in audio[condition]]
    truth = [normalize(references[c.stem]) for c in clips]

    header = "condition".ljust(24) + "".join(f"{ms:>9}ms" for ms in LOOKAHEADS.values())
    print(header)
    print("-" * len(header))

    baseline: dict[int, float] = {}
    for condition, waveforms in audio.items():
        cells = []
        for lookahead in LOOKAHEADS:
            processor.set_num_lookahead_tokens(lookahead)
            errors = []
            for waveform, reference in zip(waveforms, truth):
                inputs = processor(waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt")
                inputs = inputs.to(model.device, dtype=model.dtype)
                with torch.no_grad():
                    output = model.generate(**inputs, return_dict_in_generate=True)
                text = processor.decode(output.sequences, skip_special_tokens=True)
                if isinstance(text, list):  # batch of one
                    text = text[0]
                errors.append(wer(reference, normalize(text)))
            rate = 100 * float(np.mean(errors))
            if condition == "clean":
                baseline[lookahead] = rate
                cells.append(f"{rate:>10.1f}")
            else:
                cells.append(f"{rate:>6.1f}{rate - baseline[lookahead]:+.1f}")
        print(condition.ljust(24) + "".join(cells))

    print("\nWER %, and the delta from clean. Columns are the four chunk sizes.")


if __name__ == "__main__":
    main()
