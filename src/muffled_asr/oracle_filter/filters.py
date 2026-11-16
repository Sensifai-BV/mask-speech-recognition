"""P0 section 3.2: per-pair oracle filters (forward/inverse), Fx partners, Fir from mannikin IRs."""
import csv, json, sys
from pathlib import Path
import numpy as np, soundfile as sf
from scipy.interpolate import CubicSpline
from scipy.signal import stft
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "measurement"))
from common import P0, ROOT
from cox_tables import measurement_table

FS, NFFT, NTAPS = 16000, 512, 1025
BANDS = np.array([100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300.])
FFT_F = np.fft.rfftfreq(NFFT, 1 / FS)


def ltas(x, thr_db=6.0):
    _, _, Z = stft(x, fs=FS, window="hann", nperseg=512, noverlap=512 - 160, nfft=NFFT, boundary=None, padded=False)
    P = np.abs(Z) ** 2; e = 10 * np.log10(P.sum(0) + 1e-20); act = e > np.median(e) + thr_db
    return P[:, act].mean(1), float(act.mean())


def band_db(Pm, Pu, f=FFT_F):
    g = []
    for c in BANDS:
        s = (f >= c / 2 ** (1 / 6)) & (f < c * 2 ** (1 / 6))
        g.append(10 * np.log10(Pm[s].sum() / Pu[s].sum()))
    return np.array(g)


def dense_gain(band_gain, n=16384):
    f = np.fft.rfftfreq(n, 1 / FS); lf = np.log(np.clip(f, BANDS[0], BANDS[-1]))
    return CubicSpline(np.log(BANDS), band_gain)(lf)  # held constant outside 100-6300 Hz via clip


def minphase_fir(gain_db, n=16384):
    """Real-cepstrum minimum-phase FIR (NTAPS) with magnitude 10^(gain/20) on the rfft grid of n."""
    logmag = np.log(10 ** (gain_db / 20)); full = np.r_[logmag, logmag[-2:0:-1]]
    c = np.fft.ifft(full).real; fold = np.zeros(n); fold[0] = c[0]; fold[1:n // 2] = 2 * c[1:n // 2]; fold[n // 2] = c[n // 2]
    h = np.fft.ifft(np.exp(np.fft.fft(fold))).real[:NTAPS]
    return h * np.r_[np.ones(NTAPS - 64), np.hanning(128)[64:]]  # short tail taper against truncation ripple


def fwd_inv(band_gain):
    g = dense_gain(band_gain)
    return minphase_fir(np.clip(g, -25, 10)), minphase_fir(np.clip(-g, -10, 25))


def mannikin_band(design_file, mic):
    def pw(ir, rate, win_ms=12.):
        pk = int(np.argmax(np.abs(ir))); s = max(0, pk - round(rate * .002)); e = min(len(ir), pk + round(rate * win_ms / 1000))
        seg = ir[s:e] * np.r_[np.linspace(0, 1, 48), np.ones(e - s - 96), np.linspace(1, 0, 48)]
        return np.abs(np.fft.rfft(seg, n=32768)) ** 2
    J = ROOT / "datasets/original/cox-2022/june"
    on = [pw(a, r) for n, a, r in measurement_table(J / design_file) if f"_{mic}_" in n]
    off = [pw(a, r) for n, a, r in measurement_table(J / f"no-mask_{mic}-mic_ir_data.mat")]
    f = np.fft.rfftfreq(32768, 1 / 48000)
    return band_db(np.mean(on, 0), np.mean(off, 0), f), len(on), len(off)


def main():
    man = list(csv.DictReader(open(P0 / "speech_manifest.csv")))
    pairs = {}
    for r in man:
        key = (r["dataset"], r["channel"], r["speaker_id"], r["mask_type"])
        pairs.setdefault(key, {})["U" if r["condition"] == "mask_off" else "M"] = r
    out, gains, meta = {}, [], {}
    for (ds, ch, spk, mask), pr in sorted(pairs.items()):
        pid = f"{ds}|{ch}|{spk}|{mask}"
        u = sf.read(pr["U"]["path"], dtype="float32")[0]; m = sf.read(pr["M"]["path"], dtype="float32")[0]
        for thr in (6.0, 3.0, 9.0):
            Pu, au = ltas(u, thr); Pm, am = ltas(m, thr); g = band_db(Pm, Pu)
            hf, hi = fwd_inv(g); tag = "" if thr == 6.0 else f"@thr{int(thr)}"
            out[f"{pid}{tag}|fwd"] = hf; out[f"{pid}{tag}|inv"] = hi
            gains += [{"pair_id": pid, "dataset": ds, "channel": ch, "speaker": spk, "mask": mask, "threshold_db": thr,
                       "band_hz": b, "gain_db": round(float(x), 3)} for b, x in zip(BANDS, g)]
            if thr == 6.0: meta[pid] = {"active_frac_U": round(au, 3), "active_frac_M": round(am, 3), "primary": pr["U"]["primary"]}
    # Fx partner: next talker (cyclic, sorted) with the same dataset/channel/mask
    for pid in list(meta):
        ds, ch, spk, mask = pid.split("|")
        mates = sorted(p for p in meta if p.split("|")[0] == ds and p.split("|")[1] == ch and p.split("|")[3] == mask)
        meta[pid]["fx_partner"] = mates[(mates.index(pid) + 1) % len(mates)]
    # Fir: Cox-H mannikin, cheek -> near IRs, far -> far IRs
    files = {"A": "A-TSE_ir_data.mat", "B": "B-TUF_ir_data.mat", "C": "C-TFR_ir_data.mat", "D": "D-SWO_ir_data.mat", "E": "E-FAB_ir_data.mat"}
    for d, fn in files.items():
        for ch, mic in (("cheek", "near"), ("far", "far")):
            g, n_on, n_off = mannikin_band(fn, mic); out[f"mannikin|{ch}|{d}|fwd"] = fwd_inv(g)[0]
            gains += [{"pair_id": f"mannikin|{ch}|{d}", "dataset": "mannikin", "channel": ch, "speaker": "", "mask": d,
                       "threshold_db": "", "band_hz": b, "gain_db": round(float(x), 3)} for b, x in zip(BANDS, g)]
            meta[f"mannikin|{ch}|{d}"] = {"refits_on": n_on, "refits_off": n_off}
    np.savez(P0 / "filters.npz", **out)
    with open(P0 / "band_gains.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, list(gains[0])); w.writeheader(); w.writerows(gains)
    (P0 / "filters_meta.json").write_text(json.dumps(meta, indent=1))
    # realisation check: FIR magnitude vs target at band centres
    from scipy.signal import freqz
    err = []
    for pid in [p for p in meta if not p.startswith("mannikin")]:
        g = np.array([x["gain_db"] for x in gains if x["pair_id"] == pid and x["threshold_db"] == 6.0])
        _, H = freqz(out[f"{pid}|fwd"], worN=BANDS, fs=FS); err.append(20 * np.log10(np.abs(H)) - np.clip(g, -25, 10))
    err = np.abs(np.array(err))
    af = np.array([[v["active_frac_U"], v["active_frac_M"]] for k, v in meta.items() if "active_frac_U" in v])
    print({"pairs": len([p for p in meta if not p.startswith('mannikin')]), "filters": len(out),
           "fir_err_db_max": round(float(err.max()), 2), "fir_err_db_mean": round(float(err.mean()), 3),
           "active_frac_range": [round(float(af.min()), 2), round(float(af.max()), 2)]})


if __name__ == "__main__": main()
