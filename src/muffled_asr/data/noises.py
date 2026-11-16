"""Dry (source-side) noise signals for mixing: welding, knock, babble.

All functions return mono float32 at 16 kHz plus a JSON-able recipe. Spatialisation (RIR
convolution) and SNR scaling happen in build_mc_dataset.py.

- welding : Intel robotic-welding recordings, split 80/20 by top-level session folder.
- knock   : impulsive knocking. Single knock events are cut from ESC-50 `door_wood_knock`
            (folds 1-4 train, fold 5 test; CC BY-NC 3.0, Piczak 2015) or synthesised as modal
            impacts (wood: 150-900 Hz, short decay; metal/hammer-on-steel: inharmonic
            0.8-6 kHz, long decay). Events are placed in "knock-knock" groups of 2-4 strikes.
- babble  : 3-8 competing LibriSpeech talkers from the SAME split, never the target speaker.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import hilbert, resample_poly

FS = 16000


def _fade(x: np.ndarray, n_in: int = 32, n_out: int = 400) -> np.ndarray:
    x = x.copy(); n_out = min(n_out, len(x)); n_in = min(n_in, len(x))
    x[:n_in] *= np.linspace(0, 1, n_in); x[-n_out:] *= np.linspace(1, 0, n_out)
    return x


# ----------------------------------------------------------------------------- welding
def welding_crop(path: Path, n: int, rng: np.random.Generator, load) -> tuple[np.ndarray, dict]:
    x = load(path)
    if len(x) < n: x = np.tile(x, int(np.ceil(n / len(x))) + 1)
    off = int(rng.integers(0, len(x) - n + 1))
    return x[off:off + n].astype(np.float32), {"offset": off}


# ----------------------------------------------------------------------------- knock
def esc_knock_events(esc_dir: Path, folds: set[int]) -> list[tuple[str, np.ndarray]]:
    """Cut single knock strikes (5 ms pre, <=250 ms post onset) from ESC-50 door_wood_knock clips."""
    import csv
    events = []
    for row in csv.DictReader(open(esc_dir / "esc50.csv")):
        if row["category"] != "door_wood_knock" or int(row["fold"]) not in folds: continue
        x, sr = sf.read(esc_dir / row["filename"], dtype="float32")
        if x.ndim > 1: x = x.mean(1)
        x = resample_poly(x, 160, 441).astype(np.float32) if sr == 44100 else x
        env = np.abs(hilbert(x)); k = 80; env = np.convolve(env, np.ones(k) / k, "same")
        thr = env.max() * 10 ** (-20 / 20); mind = int(0.08 * FS); last = -mind
        for i in range(1, len(env) - 1):
            if env[i] > thr and env[i] >= env[i - 1] and env[i] >= env[i + 1] and i - last >= mind:
                a, b = max(0, i - int(0.005 * FS)), min(len(x), i + int(0.25 * FS))
                seg = x[a:b]
                if len(seg) > int(0.05 * FS): events.append((row["filename"], _fade(seg)))
                last = i
    return events


def synth_impact(rng: np.random.Generator, kind: str) -> np.ndarray:
    if kind == "wood":
        f0 = rng.uniform(150, 900); ratios = np.array([1, 2.1, 3.4, 5.2, 7.9]); tau = rng.uniform(0.012, 0.06, 5)
        dur = 0.25
    else:  # metal: hammer on steel plate / pipe
        f0 = rng.uniform(800, 2500); ratios = np.sort(np.r_[1, rng.uniform(1.3, 4.0, 6)]); tau = rng.uniform(0.08, 0.45, 7)
        dur = 0.8
    t = np.arange(int(dur * FS)) / FS
    amps = 10 ** (-rng.uniform(0, 18, len(ratios)) / 20)
    f = np.minimum(f0 * ratios, 7500)
    y = sum(a * np.exp(-t / ta) * np.sin(2 * np.pi * fi * t + rng.uniform(0, 2 * np.pi)) for a, fi, ta in zip(amps, f, tau))
    click = rng.standard_normal(len(t)) * np.exp(-t / 0.002) * 0.5  # contact transient
    y = (y + click).astype(np.float32)
    return _fade(y / (np.abs(y).max() + 1e-9))


def knock_track(n: int, rng: np.random.Generator, events: list[tuple[str, np.ndarray]],
                p_recorded: float = 0.5) -> tuple[np.ndarray, dict]:
    """Groups of 2-4 strikes 0.12-0.35 s apart, groups separated by 0.4-2.5 s of silence."""
    u = rng.random()
    kind = "recorded" if u < p_recorded else ("wood_synth" if u < p_recorded + (1 - p_recorded) / 2 else "metal_synth")
    y = np.zeros(n, np.float32); t = int(rng.uniform(0.0, 1.0) * FS); files = set(); n_strikes = 0
    proto = None if kind == "recorded" else synth_impact(rng, kind.split("_")[0])
    while t < n:
        for _ in range(int(rng.integers(2, 5))):
            if t >= n: break
            if kind == "recorded":
                name, ev = events[int(rng.integers(len(events)))]; files.add(name)
            else:  # same object struck repeatedly: small pitch/decay jitter around the prototype
                ev = proto if rng.random() < 0.5 else synth_impact(rng, kind.split("_")[0])
            g = 10 ** (rng.uniform(-4, 4) / 20); m = min(len(ev), n - t)
            y[t:t + m] += g * ev[:m]; n_strikes += 1
            t += int(rng.uniform(0.12, 0.35) * FS)
        t += int(rng.uniform(0.4, 2.5) * FS)
    if n_strikes == 0:  # very short utterance: force one strike
        ev = events[0][1] if kind == "recorded" else proto; m = min(len(ev), n); y[:m] += ev[:m]; n_strikes = 1
    return y, {"kind": kind, "n_strikes": n_strikes, "esc_files": sorted(files)}


# ----------------------------------------------------------------------------- babble
def babble_talker(n: int, rng: np.random.Generator, utt_paths: list[str], load) -> tuple[np.ndarray, list[str]]:
    """One competing talker: concatenated utterances (50-300 ms gaps) cropped to n samples."""
    parts, used, total = [], [], 0
    order = rng.permutation(len(utt_paths))
    i = 0
    while total < n + FS:
        pth = utt_paths[order[i % len(order)]]; i += 1
        x = load(Path(pth)); used.append(Path(pth).stem)
        x = x / (np.sqrt(np.mean(x ** 2)) + 1e-9)
        gap = np.zeros(int(rng.uniform(0.05, 0.3) * FS), np.float32)
        parts += [x, gap]; total += len(x) + len(gap)
        if i > 50: break
    y = np.concatenate(parts)
    if len(y) < n: y = np.tile(y, int(np.ceil(n / len(y))) + 1)
    off = int(rng.integers(0, len(y) - n + 1))
    return y[off:off + n].astype(np.float32), used
