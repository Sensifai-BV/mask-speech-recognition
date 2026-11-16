"""Summarise a training log: every validation point, and which one won.

The raw NeMo log buries the numbers that matter among thousands of progress-bar
updates. This pulls out the checkpoint decisions, which is where Lightning
records the monitored metric at each validation.

    python src/muffled_asr/training/report.py adapter.log
    watch -n 30 'python src/muffled_asr/training/report.py adapter.log'   # while training
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# Lightning: "Epoch 0, global step 819: 'val_wer' reached 0.08120 (best 0.08120), saving model to '...'"
#            "Epoch 1, global step 1638: 'val_wer' was not in top 1"
REACHED = re.compile(
    r"Epoch (\d+), global step (\d+): '(\w+)' reached ([\d.]+) \(best ([\d.]+)\)"
)
SKIPPED = re.compile(r"Epoch (\d+), global step (\d+): '(\w+)' was not in top")
# NeMo echoes the running loss in the progress bar.
STEP_TIME = re.compile(r"train_step_timing in s=([\d.]+)")
THROUGHPUT = re.compile(r"(\d+)/(\d+) \[[\d:]+<([\d:]+), *([\d.]+)it/s")


def best_checkpoint_on_disk(exp_dir: Path) -> tuple[float, Path] | None:
    """Lowest val_wer encoded in a checkpoint filename, ignoring -last.

    Searches any depth below exp_dir, so both layouts work:
      <exp_dir>/<date>/checkpoints/                     (--exp-dir .../nemotron_welding_adapter)
      <exp_dir>/nemotron_welding_adapter/<date>/checkpoints/   (--exp-dir experiments/<run>)
    Dated runs of one experiment (e.g. a crash + resume) are pooled. Checkpoints from
    different experiments are never mixed: that raises and lists them.
    """
    scored, roots = [], set()
    for path in exp_dir.glob("**/checkpoints/*.ckpt"):
        if "-last" in path.name:
            continue
        match = re.search(r"val_wer=([\d.]+?)(?:-|\.ckpt)", path.name)
        if match:
            scored.append((float(match.group(1)), path))
            roots.add(path.parent.parent.parent)  # experiment dir above <date>/checkpoints
    if len(roots) > 1:
        listing = "\n  ".join(str(r) for r in sorted(roots))
        raise SystemExit(f"checkpoints from {len(roots)} experiments under {exp_dir}; pass one of:\n  {listing}")
    return min(scored) if scored else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("log", type=Path, nargs="?", default=Path("adapter.log"))
    parser.add_argument("--exp-dir", type=Path, default=Path("experiments"),
                        help="where to look for checkpoints when the log has no best")
    args = parser.parse_args()

    if not args.log.exists():
        raise SystemExit(f"{args.log} not found")
    text = args.log.read_text(errors="ignore")

    rows = []
    for line in text.splitlines():
        if (m := REACHED.search(line)):
            epoch, step, metric, value, best = m.groups()
            rows.append((int(epoch), int(step), metric, float(value), float(best), True))
        elif (m := SKIPPED.search(line)):
            epoch, step, metric = m.groups()
            rows.append((int(epoch), int(step), metric, None, None, False))

    if not rows:
        print(f"{args.log}: no validation points yet.")
        steps = THROUGHPUT.findall(text)
        if steps:
            done, total, eta, rate = steps[-1]
            print(f"  progress {done}/{total} batches, {float(rate):.2f} it/s, eta {eta}")
        return

    metric = rows[-1][2]
    best_value = min((r[3] for r in rows if r[3] is not None), default=None)

    print(f"{args.log}  —  monitoring {metric} (lower is better)\n")
    print(f"{'epoch':>6} {'step':>8} {'value':>10}   ")
    print("-" * 34)
    for epoch, step, _, value, _, saved in rows:
        if value is None:
            print(f"{epoch:>6} {step:>8} {'—':>10}   not improved")
        else:
            mark = "  <-- best" if value == best_value else ""
            print(f"{epoch:>6} {step:>8} {value:>10.4f}   {'saved' if saved else ''}{mark}")

    # Judge the trend from every validation, not only the saved ones.
    print()
    if best_value is None:
        # Nothing in this log improved. That is normal after a resume: the best
        # was set by an earlier run, whose lines this log may not contain. Never
        # call a run of "not improved" rows an improvement.
        print(f"  {metric} did not improve in any of the {len(rows)} checks in this log.")
        prior = best_checkpoint_on_disk(args.exp_dir)
        if prior:
            value, path = prior
            print(f"  Best overall is still {value:.4f} from an earlier run: {path.name}")
        print("  More steps are unlikely to help. Evaluate the best checkpoint instead.")
    else:
        best_index = max(i for i, r in enumerate(rows) if r[3] == best_value)
        since_best = len(rows) - 1 - best_index
        best_step = rows[best_index][1]
        if since_best == 0:
            print(f"  {metric} improved at the latest check (step {best_step}) — more steps may still help.")
        elif since_best < 3:
            print(f"  {metric} best at step {best_step}, {since_best} check(s) since without improvement.")
            print("  Too early to call; watch the next check.")
        else:
            print(f"  {metric} has plateaued: best at step {best_step}, then {since_best} checks with no improvement.")
            print("  More steps are unlikely to help. Evaluate the best checkpoint instead.")

    timings = STEP_TIME.findall(text)
    if timings:
        recent = [float(t) for t in timings[-200:]]
        print(f"\n  step time {sum(recent) / len(recent):.3f}s (last {len(recent)} steps)")
    throughput = THROUGHPUT.findall(text)
    if throughput:
        done, total, eta, rate = throughput[-1]
        print(f"  progress  {done}/{total} batches, {float(rate):.2f} it/s, eta {eta}")


if __name__ == "__main__":
    main()
