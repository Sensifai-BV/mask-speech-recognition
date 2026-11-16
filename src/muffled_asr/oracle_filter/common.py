"""Shared P0 conventions (see Experiments P0-P4, section 1)."""
from __future__ import annotations
import re, struct
from pathlib import Path
import numpy as np
from scipy.signal import butter, correlate, resample_poly, sosfiltfilt

from muffled_asr.paths import ROOT
P0 = ROOT / "datasets/p0"
RAINBOW = ("When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow. "
    "The rainbow is a division of white light into many beautiful colors. These take the shape of a long "
    "round arch, with its path high above, and its two ends apparently beyond the horizon. There is, "
    "according to legend, a boiling pot of gold at one end. People look, but no one ever finds it. When a "
    "man looks for something beyond his reach, his friends say he is looking for the pot of gold at the end "
    "of the rainbow.")


def norm_text(s: str) -> str:
    return " ".join(re.sub(r"[^a-z' ]", " ", s.lower()).split())


def to16k(x: np.ndarray, rate: int) -> np.ndarray:
    if rate != 48000: raise ValueError(f"expected 48 kHz, got {rate}")
    return resample_poly(x, 1, 3).astype(np.float32)


def peak_norm(x: np.ndarray, dbfs: float = -3.0) -> np.ndarray:
    return (x * (10 ** (dbfs / 20) / (np.abs(x).max() + 1e-12))).astype(np.float32)


def lag_ms(stereo: np.ndarray, rate: int) -> float:
    """Delay of index 1 behind index 0 (ms). Cox-H: ~2.3-2.8 ms => index 0 cheek, index 1 at 1 m."""
    sos = butter(4, [300, 4000], btype="band", fs=rate, output="sos")
    x = sosfiltfilt(sos, stereo[: rate * 20], axis=0)
    c = correlate(x[:, 1], x[:, 0], mode="full", method="fft"); L = np.arange(-len(x) + 1, len(x))
    k = np.abs(L) <= int(0.02 * rate)
    return float(L[k][np.argmax(np.abs(c[k]))] / rate * 1000)


def read_binary_textgrid(path: Path) -> dict[str, list[tuple[float, float, str]]]:
    """Praat 'ooBinaryFile' TextGrid -> {tier: [(start, end, text), ...]} (interval tiers only)."""
    b = Path(path).read_bytes(); p = 0
    def take(n):
        nonlocal p; v = b[p:p + n]; p += n; return v
    def s8():  # 1-byte length string (class names)
        n = take(1)[0]; return take(n).decode()
    def s16():  # 2-byte length string; 0xFFFF => UTF-16 follows
        n = struct.unpack(">H", take(2))[0]
        if n == 0xFFFF:
            n = struct.unpack(">H", take(2))[0]; return take(2 * n).decode("utf-16-be")
        return take(n).decode("latin-1")
    d = lambda: struct.unpack(">d", take(8))[0]; i32 = lambda: struct.unpack(">i", take(4))[0]
    if take(12) != b"ooBinaryFile": raise ValueError(f"{path}: not a binary TextGrid")
    if s8() != "TextGrid": raise ValueError(f"{path}: not a TextGrid")
    d(); d(); take(1); tiers = {}
    for _ in range(i32()):
        cls = s8(); name = s16(); d(); d(); n = i32()
        if cls == "IntervalTier":
            tiers[name] = [(d(), d(), s16()) for _ in range(n)]
        elif cls == "TextTier":
            [(d(), s16()) for _ in range(n)]
        else:
            raise ValueError(f"{path}: unknown tier class {cls}")
    return tiers


# ---- Scoring (audit 2026-10-18; applied identically to every system) ----
SPELLING = {"traveller": "traveler", "travellers": "travelers", "colours": "colors", "colour": "color"}
RAINBOW_READ = RAINBOW.replace("they act as a prism", "they act like a prism")  # version read by all 4 talkers (38/40 files)
TALKER_REF = {("coxh", "P04"): RAINBOW_READ.replace("at the end of the rainbow", "at the end of a rainbow")}
GENG_EXCLUDE = {"S02", "S29"}  # restarted/repeated passage portions (40-46 insertions); sensitivity only


def score_norm(s: str) -> str:
    return " ".join(SPELLING.get(w, w) for w in norm_text(s).split())


def reference(dataset: str, speaker: str, default: str) -> str:
    if dataset == "coxh":
        return score_norm(TALKER_REF.get((dataset, speaker), RAINBOW_READ))
    return score_norm(default)
