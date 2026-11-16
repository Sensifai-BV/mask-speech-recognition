"""Make a clean-transcript evaluation manifest from Cox's four human talkers.

The Cox recordings contain one take per participant and covering condition,
as 48 kHz stereo: index 0 = microphone taped to the cheek, index 1 = microphone
at 1 m (the file description lists cheek first, then 1 m; the paper analysed
only the 1 m mic). Verified on all 40 files: index 1 arrives 2.3-2.8 ms after
index 0, i.e. ~0.8-0.95 m further away. Channel LEVEL does not identify the
mics: the cheek mic is quieter mask-off and 5-16 dB louder mask-on.

The default is the 1 m channel, which matches the far-field response bank used
for training. Every file is checked by cross-correlation lag so a swapped file
fails loudly instead of silently mixing microphones.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, correlate, resample_poly, sosfiltfilt

CHANNELS = {"far": 1, "cheek": 0}


def far_lag_ms(stereo: np.ndarray, rate: int) -> float:
    """Delay of index 1 behind index 0 (ms), from band-passed cross-correlation."""
    sos = butter(4, [300, 4000], btype="band", fs=rate, output="sos")
    x = sosfiltfilt(sos, stereo[: rate * 20], axis=0)
    c = correlate(x[:, 1], x[:, 0], mode="full", method="fft")
    lags = np.arange(-len(x) + 1, len(x))
    keep = np.abs(lags) <= int(0.02 * rate)
    return float(lags[keep][np.argmax(np.abs(c[keep]))] / rate * 1000)

TEXT = (
    "When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow. "
    "The rainbow is a division of white light into many beautiful colors. "
    "These take the shape of a long round arch, with its path high above, and its two ends "
    "apparently beyond the horizon. There is, according to legend, a boiling pot of gold at one end. "
    "People look, but no one ever finds it. When a man looks for something beyond his reach, "
    "his friends say he is looking for the pot of gold at the end of the rainbow."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("datasets/original/cox-2022/human"))
    parser.add_argument("--channel", choices=sorted(CHANNELS), default="far",
                        help="far = 1 m mic (index 1, default); cheek = taped mic (index 0)")
    parser.add_argument("--out", type=Path,
                        help="default datasets/calibrated/cox-human-eval-<channel>")
    args = parser.parse_args()
    args.out = args.out or Path(f"datasets/calibrated/cox-human-eval-{args.channel}")

    rows = []
    for path in sorted(args.source.glob("*_mask-*_P0[1-4]_trim.wav")):
        stem = path.stem
        design, mask_state, participant, _ = stem.split("_")
        # A-D are the held-out TPU-window constructions. E is plain cloth v2
        # and is reserved as a diagnostic development condition.
        split = "test" if design in "ABCD" else "dev"
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        lag = far_lag_ms(audio, rate)
        if not 1.5 <= lag <= 4.0:
            raise SystemExit(f"{path.name}: index 1 lags index 0 by {lag:.2f} ms; "
                             "expected ~2.3-2.8 ms (1 m vs cheek). Inspect channel order.")
        audio = audio[:, CHANNELS[args.channel]]
        if rate != 16000:
            from math import gcd
            factor = gcd(rate, 16000)
            audio = resample_poly(audio, 16000 // factor, rate // factor).astype(np.float32)
        destination = args.out / split / f"{design}_{mask_state}_{participant}.wav"
        destination.parent.mkdir(parents=True, exist_ok=True)
        sf.write(destination, audio, 16000)
        rows.append({
            "audio_filepath": str(destination.resolve()),
            "duration": round(len(audio) / 16000, 3),
            "text": TEXT,
            "speaker": participant,
            "condition": "quiet-unmasked" if mask_state == "mask-off" else f"quiet-covering-{design}",
            "design": design,
            "source": "Cox2022-human",
            "channel": args.channel,
            "channel_index": CHANNELS[args.channel],
            "far_lag_ms": round(lag, 3),
        })

    if not rows:
        raise SystemExit(f"No Cox human recordings found under {args.source}")
    for split in ("dev", "test"):
        selected = [row for row in rows if ("/dev/" in row["audio_filepath"] if split == "dev"
                                             else "/test/" in row["audio_filepath"])]
        path = args.out / f"{split}.json"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected))
        print(f"{split}: {len(selected)} recordings -> {path}")


if __name__ == "__main__":
    main()
