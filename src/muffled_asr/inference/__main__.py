"""Entry point: `python -m muffled_asr.inference`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m muffled_asr.inference",
        description="Real-time speech-to-text GUI using nvidia/nemotron-speech-streaming-en-0.6b.",
    )
    parser.add_argument(
        "--list-devices", action="store_true", help="list available input devices and exit"
    )
    parser.add_argument(
        "--input-device", type=int, default=None, help="input device index (default: system default)"
    )
    parser.add_argument(
        "--torch-device",
        default=None,
        choices=["mps", "cpu"],
        help="inference device (default: mps when available)",
    )
    parser.add_argument(
        "--latency-ms",
        type=int,
        default=560,
        choices=[80, 160, 560, 1120],
        help="streaming chunk latency; lower is more responsive but costs more compute",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="where the weights live (default: <project>/models/nemotron-speech-streaming-en-0.6b)",
    )
    args = parser.parse_args()

    from .stream import list_input_devices

    if args.list_devices:
        print(list_input_devices())
        return 0

    from .app import App
    from .asr import MODEL_DIR, pick_device

    lookahead = {80: 0, 160: 1, 560: 6, 1120: 13}[args.latency_ms]
    App(
        args.input_device,
        pick_device(args.torch_device),
        lookahead,
        args.model_dir or MODEL_DIR,
    ).mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
