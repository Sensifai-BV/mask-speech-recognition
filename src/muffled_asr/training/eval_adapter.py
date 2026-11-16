"""Score base vs adapter on held-out speakers, per chunk size and condition.

Runs inside NeMo on the training machine. export_adapter.py hands the same
checkpoint to the Transformers side, where compare_models.py is this script's
twin; the two should agree.

    python src/muffled_asr/training/eval_adapter.py                 # defaults below
    python src/muffled_asr/training/eval_adapter.py --limit 300     # quicker pass

By default it loads the BEST checkpoint (lowest val_wer in its filename) into the
adapter model, not the final weights. It reports WER separately for the clean
utterances and the masked+noisy ones, because an adapter that helps on noise but
hurts clean speech is a regression you would not see in a single number.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preflight import rebase_paths  # noqa: E402

from nemo.collections.asr.models import ASRModel

# right-context frames for the four trained chunk sizes: 80/160/560/1120 ms
LATENCIES = {80: [70, 0], 160: [70, 1], 560: [70, 6], 1120: [70, 13]}


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


def best_checkpoint(ckpt_dir: Path) -> Path | None:
    scored = []
    for path in ckpt_dir.glob("*.ckpt"):
        if "-last" in path.name:
            continue
        match = re.search(r"val_wer=([\d.]+)", path.name)
        if match:
            scored.append((float(match.group(1).rstrip(".")), path))
    return min(scored)[1] if scored else None


def load(nemo: Path, ckpt: Path | None, has_adapter: bool) -> ASRModel:
    model = ASRModel.restore_from(str(nemo), map_location="cpu")
    if ckpt is not None:
        state = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        adapter_keys = [k for k in state if "adapter" in k]
        if not adapter_keys:
            raise SystemExit(f"{ckpt.name} contains no adapter weights")
        print(f"  loaded {ckpt.name}: {len(adapter_keys)} adapter tensors "
              f"({len(missing)} missing, {len(unexpected)} unexpected keys)")
    if has_adapter:
        if not model.is_adapter_available():
            raise SystemExit(f"{nemo} has no adapter -- is this the right .nemo?")
        model.set_enabled_adapters(enabled=True)
    model.wer.log_prediction = False
    # CUDA on the training machine; CPU elsewhere (NeMo's RNNT decoding is not
    # reliable on MPS). The Mac path for exported adapters is src/muffled_asr/evaluation/eval_hf.py.
    return model.eval().to("cuda" if torch.cuda.is_available() else "cpu")


def transcribe(model: ASRModel, paths: list[str], batch: int) -> list[str]:
    out = model.transcribe(paths, batch_size=batch, verbose=False)
    if isinstance(out, tuple):  # some versions return (best, all)
        out = out[0]
    return [h.text if hasattr(h, "text") else str(h) for h in out]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--manifest", type=Path, default=Path("datasets/masked/librispeech-welding/test.json"))
    parser.add_argument("--base", type=Path,
                        default=Path("models/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo"))
    parser.add_argument("--adapter", type=Path,
                        default=Path("experiments/nemotron_welding_adapter/checkpoints/nemotron_welding_adapter.nemo"))
    parser.add_argument("--ckpt", type=Path, help="checkpoint to load [best by val_wer]")
    parser.add_argument("--final", action="store_true", help="use the adapter .nemo as-is, not the best ckpt")
    parser.add_argument("--latencies", type=int, nargs="+", default=list(LATENCIES))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]

    # Resolve every audio path before loading a 600M model, not after. A
    # manifest built on another machine otherwise fails on its first file,
    # minutes in, from deep inside the Lhotse dataloader.
    repaired = rebase_paths(rows, args.manifest.parent)
    if repaired:
        args.manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        print(f"rebased {repaired} audio paths in {args.manifest} onto this machine")
    missing = [r["audio_filepath"] for r in rows if not os.path.exists(r["audio_filepath"])]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(rows)} audio files not found, e.g.\n  {missing[0]}")

    rows = rows[: args.limit] if args.limit else rows
    paths = [r["audio_filepath"] for r in rows]
    refs = [normalize(r["text"]) for r in rows]
    conditions = sorted({r.get("condition", "clean") for r in rows})
    groups = {condition: [i for i, r in enumerate(rows)
                          if r.get("condition", "clean") == condition]
              for condition in conditions}
    groups["all"] = list(range(len(rows)))
    summary = ", ".join(f"{condition} {len(groups[condition])}"
                         for condition in conditions)
    print(f"{len(rows)} held-out utterances: {summary}\n")

    ckpt = None
    if not args.final:
        ckpt = args.ckpt or best_checkpoint(args.adapter.parent)

    results: dict[str, dict[int, dict[str, float]]] = {}
    for label, nemo, use_ckpt in (("base", args.base, None), ("adapter", args.adapter, ckpt)):
        print(f"=== {label} ===")
        model = load(nemo, use_ckpt, has_adapter=label == "adapter")
        results[label] = {}
        for ms in args.latencies:
            model.encoder.set_default_att_context_size(LATENCIES[ms])
            hyps = [normalize(h) for h in transcribe(model, paths, args.batch)]
            results[label][ms] = {}
            for group, idx in groups.items():
                errors = sum(edits(refs[i], hyps[i]) for i in idx)
                words = sum(len(refs[i]) for i in idx)
                results[label][ms][group] = 100 * errors / max(words, 1)
            breakdown = "   ".join(f"{condition} {results[label][ms][condition]:5.2f}%"
                                    for condition in conditions)
            print(f"  {ms:>5} ms   all {results[label][ms]['all']:5.2f}%   {breakdown}")
        del model
        torch.cuda.empty_cache()

    print("\nWER delta, adapter - base (negative = adapter better)\n")
    print(f"{'chunk':>8} {'all':>9} " + " ".join(f"{c:>18}" for c in conditions))
    for ms in args.latencies:
        d = {g: results["adapter"][ms][g] - results["base"][ms][g] for g in groups}
        print(f"{ms:>6}ms {d['all']:>+9.2f} "
              + " ".join(f"{d[c]:>+18.2f}" for c in conditions))


if __name__ == "__main__":
    main()
