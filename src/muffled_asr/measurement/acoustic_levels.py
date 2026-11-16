"""Literature-based level model for speech in welding workshops (A-weighted, dB).

Sources (values used, not copied text):
- Talker level vs noise (Lombard): Pearsons, Bennett & Fidell (1977), EPA-600/1-77-025:
  speech level rises ~0.6 dB per dB of noise once noise exceeds ~48 dBA, up to a ceiling;
  normal effort at 1 m ~55-58 dBA; shouting up to ~89 dBA (anechoic, short).
- Arc-welding noise: welder dosimetry 79-98 dBA, mean 88.9, SD 4.2 (Occupational health
  concerns in the welding industry, PubMed 11141606); pulsed MIG 84-97 dB at 0.5 m.
- Low-noise processes / shop background: manual TIG <= 64 dBA vs 57 dBA background at
  0.55 m; Ar+CO2 semi-automatic up to 96 dBA (Paton Welding J. 2015, no. 9).
"""
from __future__ import annotations
import numpy as np

SCENES = {  # name: (weight, sampler)
    "shop_ambient": (0.3, lambda r: r.uniform(55.0, 70.0)),
    "arc_welding": (0.7, lambda r: float(np.clip(r.normal(88.0, 4.2), 79.0, 98.0))),
}
LOMBARD_SLOPE, LOMBARD_ONSET, QUIET_LEVEL, CEILING, TALKER_SD = 0.6, 48.0, 55.0, 82.0, 3.0


def sample_levels(rng, distance=(0.5, 1.5)):
    names = list(SCENES); w = np.array([SCENES[n][0] for n in names]); w /= w.sum()
    scene = names[int(rng.choice(len(names), p=w))]
    noise = float(SCENES[scene][1](rng))
    talker_1m = QUIET_LEVEL + LOMBARD_SLOPE * max(0.0, noise - LOMBARD_ONSET) + rng.normal(0, TALKER_SD)
    talker_1m = float(np.clip(talker_1m, 50.0, CEILING))
    d = float(rng.uniform(*distance))
    speech_at_mic = talker_1m - 20 * np.log10(d)
    return {"scene": scene, "noise_dba": noise, "talker_dba_1m": talker_1m, "distance_m": d,
            "speech_dba_mic": speech_at_mic, "snr_a_db": speech_at_mic - noise}


def a_weight_db(f):
    f = np.maximum(np.asarray(f, float), 1e-3); f2 = f * f
    ra = 12194**2 * f2**2 / ((f2 + 20.6**2) * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12194**2))
    return 20 * np.log10(ra) + 2.0


def _frames_a(x, rate=16000, n=512, hop=160):
    if len(x) < n: x = np.pad(x, (0, n - len(x)))
    idx = np.arange(0, len(x) - n + 1, hop)[:, None] + np.arange(n)
    X = np.fft.rfft(x[idx] * np.hanning(n), axis=1)
    wa = 10 ** (a_weight_db(np.fft.rfftfreq(n, 1 / rate)) / 10)
    return (np.abs(X) ** 2 * wa).sum(1)


def active_level_a(x, rate=16000):
    """A-weighted active speech power (dB, arbitrary ref): frames above max(p10+6 dB, peak-45 dB)."""
    e = 10 * np.log10(_frames_a(x, rate) + 1e-20)
    act = e > max(np.percentile(e, 10) + 6, e.max() - 45)
    return float(10 * np.log10(np.mean(10 ** (e[act] / 10)))), float(act.mean())


def leq_a(x, rate=16000):
    return float(10 * np.log10(np.mean(_frames_a(x, rate)) + 1e-20))
