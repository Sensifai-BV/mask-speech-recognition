"""P0 section 3.3: build U/M/F/R/Fx/Fir/Ri x {quiet,0,5,10,20 dB} (+ pre-filter SNR, +3/+9 dB thresholds).

Primary convention: SNR on active-frame power of the signal being mixed (post-filter), signal peak-normalised
to -3 dBFS after filtering and before noise. Pre-filter convention (U/F/M/R): noise gain from U's active power;
F = h*U and M keep their physical level relative to U (rebuilt from the 48 kHz originals with U's factor).
Writes datasets/p0/conds/<block>/... and datasets/p0/manifests/<block>.jsonl.
"""
import csv, hashlib, json, os, sys
from pathlib import Path
import numpy as np, soundfile as sf
from scipy.signal import fftconvolve, stft
sys.path.insert(0, str(Path(__file__).parent))
from common import P0, ROOT, peak_norm, to16k

SNRS = [None, 0, 5, 10, 20]
# Optional extension runs (e.g. P0_SNRS="-10,-5" P0_SUFFIX=_lowsnr): same seeds/recipe, separate output block.
if os.environ.get("P0_SNRS"): SNRS = [int(v) for v in os.environ["P0_SNRS"].split(",")]
SUFFIX = os.environ.get("P0_SUFFIX", "")
NOISE = json.loads((P0 / "noise_w_test.json").read_text())
FIL = np.load(P0 / "filters.npz"); META = json.loads((P0 / "filters_meta.json").read_text())


def active_power(x, thr=6.0):
    _, _, Z = stft(x, fs=16000, window="hann", nperseg=512, noverlap=352, boundary=None, padded=False)
    P = np.abs(Z) ** 2; e = 10 * np.log10(P.sum(0) + 1e-20); act = e > np.median(e) + thr
    # time-domain power of active frames: mean |Z|^2 scales with power; ratio to noise uses same estimator
    return float(P[:, act].sum(0).mean())


def noise_for(pid, snr, n):
    s = int.from_bytes(hashlib.sha256(f"{pid}|{snr}".encode()).digest()[:8], "little"); rng = np.random.default_rng(s)
    ex = sf.read(NOISE[int(rng.integers(len(NOISE)))]["path"], dtype="float32")[0]
    off = int(rng.integers(len(ex))); ex = np.roll(ex, -off)
    return np.tile(ex, int(np.ceil(n / len(ex))))[:n]


def mix(sig, pid, snr, ref_power=None):
    if snr is None: return sig, None
    n = noise_for(pid, snr, len(sig)); ps = active_power(sig) if ref_power is None else ref_power
    pn = active_power_noise(n); y = sig + n * np.sqrt(ps / (pn * 10 ** (snr / 10)))
    pk = np.abs(y).max(); return (y / pk * 0.99 if pk > 0.99 else y).astype(np.float32), float(snr)


def active_power_noise(n):
    _, _, Z = stft(n, fs=16000, window="hann", nperseg=512, noverlap=352, boundary=None, padded=False)
    return float((np.abs(Z) ** 2).sum(0).mean())


def filt(x, h): return fftconvolve(x, h)[: len(x)].astype(np.float32)


def orig16(row):  # unnormalised 16 kHz original (for pre-filter convention)
    x, sr = sf.read(ROOT / ("datasets/original/geng2023/" if row["dataset"] == "geng" else "datasets/original/cox-2022/human/") / row["source"], dtype="float32", always_2d=True)
    return to16k(x[:, {"cheek": 0, "far": 1, "mono": 0}[row["channel"]]], sr)


def main():
    blocks = sys.argv[1:] or ["geng", "coxh_cheek", "coxh_far"]
    man = list(csv.DictReader(open(P0 / "speech_manifest.csv"))); pairs = {}
    for r in man: pairs.setdefault(f'{r["dataset"]}|{r["channel"]}|{r["speaker_id"]}|{r["mask_type"]}', {})["U" if r["condition"] == "mask_off" else "M"] = r
    for block in blocks:
        ds, ch = (block, "mono") if block == "geng" else ("coxh", block.split("_")[1])
        rows = []; outdir = P0 / "conds" / (block + SUFFIX)
        for pid, pr in sorted(pairs.items()):
            if not pid.startswith(f"{ds}|{ch}|"): continue
            U = sf.read(pr["U"]["path"], dtype="float32")[0]; M = sf.read(pr["M"]["path"], dtype="float32")[0]
            hf, hi = FIL[f"{pid}|fwd"], FIL[f"{pid}|inv"]; fx = FIL[f'{META[pid]["fx_partner"]}|fwd']
            sigs = {"U": U, "M": M, "F": peak_norm(filt(U, hf)), "R": peak_norm(filt(M, hi)), "Fx": peak_norm(filt(U, fx)), "Ri": peak_norm(filt(U, hi))}
            if ds == "coxh": sigs["Fir"] = peak_norm(filt(U, FIL[f'mannikin|{ch}|{pid.split("|")[3]}|fwd']))
            if ch == "far": sigs = {k: sigs[k] for k in ("U", "M", "F", "R")}  # robustness channel: core conditions
            jobs = [(c, s, snr, "post", 6) for c, s in sigs.items() for snr in SNRS]
            if ch != "far":
                for thr in (3, 9):
                    jobs += [("F", peak_norm(filt(U, FIL[f"{pid}@thr{thr}|fwd"])), snr, "post", thr) for snr in SNRS]
                    jobs += [("R", peak_norm(filt(M, FIL[f"{pid}@thr{thr}|inv"])), snr, "post", thr) for snr in SNRS]
                u0, m0 = orig16(pr["U"]), orig16(pr["M"]); k = 10 ** (-3 / 20) / np.abs(u0).max()
                u0, m0 = u0 * k, m0 * k; pu = active_power(u0)
                pre = {"U": u0, "M": m0, "F": filt(u0, hf), "R": filt(m0, hi)}
                jobs += [(c, s, snr, "pre", 6, pu) for c, s in pre.items() for snr in SNRS if snr is not None and c != "U"]
            for j in jobs:
                cond, s, snr, conv, thr = j[:5]; y, _ = mix(s, pid, snr, j[5] if len(j) > 5 else None)
                tag = f"{cond}_thr{thr}_{conv}_" + ("quiet" if snr is None else f"snr{snr:+d}")
                out = outdir / pid.replace("|", "_") / f"{tag}.flac"; out.parent.mkdir(parents=True, exist_ok=True)
                sf.write(out, y, 16000, subtype="PCM_16")
                rows.append({"audio_filepath": str(out), "text": pr["U"]["ref_scoring"], "dataset": ds, "channel": ch,
                             "speaker": pr["U"]["speaker_id"], "mask": pr["U"]["mask_type"], "pair_id": pid, "condition": cond,
                             "snr_db": snr, "filter": {"U": "none", "M": "real covering", "F": "oracle fwd", "R": "oracle inv on M",
                             "Fx": f"oracle fwd of {META[pid]['fx_partner']}", "Fir": "mannikin IR", "Ri": "oracle inv on U"}[cond],
                             "snr_convention": conv, "threshold_db": thr, "primary": pr["U"]["primary"] == "True",
                             "duration": round(len(y) / 16000, 3)})
        mf = P0 / "manifests" / f"{block}{SUFFIX}.jsonl"; mf.parent.mkdir(exist_ok=True)
        mf.write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(block, len(rows), "files", round(sum(r["duration"] for r in rows) / 3600, 2), "h", flush=True)


if __name__ == "__main__": main()
