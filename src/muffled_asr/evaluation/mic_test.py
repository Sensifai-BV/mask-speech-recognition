"""Record yourself reading a known sentence, score both checkpoints on it.

The held-out evaluation uses simulated masks and noise, so it cannot capture
your microphone, your room, your actual PPE, or the way people change how they
talk behind a mask. This closes that gap — your voice, your mask, a sentence
whose transcript is known, and a WER for each model.

    python src/muffled_asr/evaluation/mic_test.py                 # 10 prompts, 8 s each
    python src/muffled_asr/evaluation/mic_test.py --prompts 20

Do a pass without the mask, then a pass with it, and compare the four numbers.
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch


from muffled_asr.inference.asr import load_model, pick_device  # noqa: E402

SAMPLE_RATE = 16000

# RNN-T generation ends at the audio, not at max_length; the default only warns.
warnings.filterwarnings("ignore", message="Using the model-agnostic default `max_length`")

# Harvard sentences: phonetically balanced, and fricative-rich, which is the
# band a mask attenuates most.
PROMPTS = [
    "The birch canoe slid on the smooth planks.",
    "Glue the sheet to the dark blue background.",
    "It's easy to tell the depth of a well.",
    "Four hours of steady work faced us.",
    "The source of the huge river is the clear spring.",
    "These days a chicken leg is a rare dish.",
    "Rice is often served in round bowls.",
    "The juice of lemons makes fine punch.",
    "The box was thrown beside the parked truck.",
    "The hogs were fed chopped corn and garbage.",
    "A large size in stockings is hard to sell.",
    "The boy was there when the sun rose.",
    "A rod is used to catch pink salmon.",
    "Kick the ball straight and follow through.",
    "Help the woman get back to her feet.",
    "A pot of tea helps to pass the evening.",
    "Smoky fires lack flame and heat.",
    "The soft cushion broke the man's fall.",
    "The salt breeze came across from the sea.",
    "The girl at the booth sold fifty bonds.",
]


def normalize(text: str) -> list[str]:
    return re.sub(r"[^a-z' ]", " ", text.lower()).split()


def wer(reference: list[str], hypothesis: list[str]) -> tuple[int, int]:
    previous = list(range(len(hypothesis) + 1))
    for i, ref_word in enumerate(reference, 1):
        current = [i]
        for j, hyp_word in enumerate(hypothesis, 1):
            current.append(
                previous[j - 1] if ref_word == hyp_word
                else 1 + min(previous[j - 1], previous[j], current[j - 1])
            )
        previous = current
    return previous[-1], len(reference)


def clipped_edges(audio: np.ndarray) -> list[str]:
    """Edges where speech runs into the recording window, i.e. words were lost.
    A cut-off sentence scores both models on words neither of them heard."""
    edge = int(0.3 * SAMPLE_RATE)
    peak = np.abs(audio).max() + 1e-9
    loud = lambda part: 20 * np.log10(np.abs(part).max() / peak + 1e-9) > -12  # noqa: E731
    return [name for name, part in (("start", audio[:edge]), ("end", audio[-edge:])) if loud(part)]


def record(seconds: float, device: int | None) -> np.ndarray:
    while True:
        input("    press Enter, then read the sentence aloud... ")
        audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                       channels=1, dtype="float32", device=device)
        sd.wait()
        audio = audio.squeeze()

        level = 20 * np.log10(max(np.abs(audio).max(), 1e-9))
        if level < -40:
            print(f"    WARNING: peak {level:.0f} dBFS — nearly silent. Check microphone permission.")
            return audio
        clipped = clipped_edges(audio)
        if not clipped:
            return audio
        # Loud background noise also trips this, so the take can be kept.
        answer = input(f"    speech seems cut off at the {' and '.join(clipped)} (start right after "
                       f"Enter; --seconds for more time). Enter = record again, k = keep: ")
        if answer.strip().lower() == "k":
            return audio


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base", default="models/nemotron-speech-streaming-en-0.6b")
    parser.add_argument("--tuned", default="models/nemotron-welding")
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--prompts", type=int, default=10, help=f"1-{len(PROMPTS)}")
    parser.add_argument("--input-device", type=int)
    parser.add_argument("--save-dir", type=Path, default=Path("examples/mic_test"))
    args = parser.parse_args()

    prompts = PROMPTS[: args.prompts]
    args.save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {len(prompts)} sentences, {args.seconds:g}s each.\n")
    recordings = []
    for index, prompt in enumerate(prompts):
        print(f"[{index + 1}/{len(prompts)}]  \"{prompt}\"")
        audio = record(args.seconds, args.input_device)
        path = args.save_dir / f"prompt{index:02d}.wav"
        sf.write(path, audio, SAMPLE_RATE, subtype="PCM_16")
        recordings.append((prompt, audio))
        print()

    device = pick_device()
    print(f"Scoring on {device}. Audio kept in {args.save_dir}/\n")

    results = {}
    for label, model_path in [("base", args.base), ("tuned", args.tuned)]:
        processor, model = load_model(model_path, device)
        errors = words = 0
        print(f"=== {label} ===")
        for prompt, audio in recordings:
            inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            inputs = inputs.to(model.device, dtype=model.dtype)
            with torch.no_grad():
                output = model.generate(**inputs, return_dict_in_generate=True)
            text = processor.decode(output.sequences, skip_special_tokens=True)
            if isinstance(text, list):
                text = text[0]
            e, w = wer(normalize(prompt), normalize(text))
            errors += e
            words += w
            print(f"  {e}/{w} errors  {text.strip()}")
        results[label] = 100.0 * errors / max(words, 1)
        print(f"  WER {results[label]:.1f}%\n")
        del model

    delta = results["tuned"] - results["base"]
    print(f"base {results['base']:.1f}%   tuned {results['tuned']:.1f}%   delta {delta:+.1f}")
    print("negative = the fine-tune helped on your own voice.")


if __name__ == "__main__":
    main()
