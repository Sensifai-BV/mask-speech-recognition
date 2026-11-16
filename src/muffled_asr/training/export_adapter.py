"""Export the best adapter checkpoint for the Transformers model on the Mac.

Writes an adapter directory (see src/muffled_asr/inference/adapter.py): the adapter tensors
from the lowest-val_wer checkpoint -- the one eval_adapter.py scores -- renamed
to the Transformers encoder layout, plus a pointer to the base model. About
6.5 MB; the frozen base weights are not copied.

    python3 src/muffled_asr/training/export_adapter.py
    python3 src/muffled_asr/training/export_adapter.py --ckpt path/to/other.ckpt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report import best_checkpoint_on_disk  # noqa: E402

# encoder.layers.<i>.adapter_layer.<name>.module.<j>.<param>: NeMo's pre-norm
# LinearAdapter is Sequential(LayerNorm, Linear, SiLU, Linear).
KEY = re.compile(r"^encoder\.layers\.(\d+)\.adapter_layer\.([^.]+)\.module\.(\d+)\.(weight|bias)$")
PARTS = {("0", "weight"): "norm.weight", ("0", "bias"): "norm.bias", ("1", "weight"): "down.weight",
         ("3", "weight"): "up.weight"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ckpt", type=Path, help="checkpoint to export [best val_wer under --exp-dir]")
    parser.add_argument("--exp-dir", type=Path, default=Path("experiments"))
    parser.add_argument("--out", type=Path, default=Path("models/nemotron-welding"))
    parser.add_argument("--base-model", default="../nemotron-speech-streaming-en-0.6b",
                        help="base Transformers model, relative to --out")
    args = parser.parse_args()

    ckpt = args.ckpt
    if ckpt is None:
        best = best_checkpoint_on_disk(args.exp_dir)
        if best is None:
            raise SystemExit(f"no val_wer checkpoint under {args.exp_dir} (searched **/checkpoints/*.ckpt)")
        ckpt = best[1]
    print(f"reading {ckpt}")
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]

    tensors, names = {}, set()
    for key, value in state.items():
        if "adapter" not in key:
            continue
        match = KEY.match(key)
        part = match and PARTS.get((match[3], match[4]))
        if not part:
            raise SystemExit(f"unexpected adapter tensor {key}: only pre-norm LinearAdapters export")
        names.add(match[2])
        tensors[f"encoder.layers.{match[1]}.adapter.{part}"] = value.float().contiguous()
    if len(names) != 1:
        raise SystemExit(f"expected one adapter, found {sorted(names) or 'none'} in {ckpt.name}")
    layers = len(tensors) // len(PARTS)
    if len(tensors) != layers * len(PARTS):
        raise SystemExit(f"{len(tensors)} adapter tensors do not form whole layers")

    args.out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.out / "adapter.safetensors")
    config = {"base_model": args.base_model, "adapter": names.pop(), "checkpoint": ckpt.name}
    (args.out / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
    size = (args.out / "adapter.safetensors").stat().st_size / 1e6
    print(f"wrote {args.out}/: {len(tensors)} tensors over {layers} layers ({size:.1f} MB)")


if __name__ == "__main__":
    main()
