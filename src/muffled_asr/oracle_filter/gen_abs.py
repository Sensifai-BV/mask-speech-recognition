"""Exp 3: fixed absolute noise level (scenes of 75 / 85 / 95 dBA), Cox-H, cheek and 1 m channels.

    python src/muffled_asr/oracle_filter/gen_abs.py          # -> datasets/p0/conds/coxh_abs/, datasets/p0/manifests/coxh_abs.jsonl
    python src/muffled_asr/evaluation/eval_hf.py --manifests datasets/p0/manifests/coxh_abs.jsonl --models base --lookahead 6 --out results/p0
    python src/muffled_asr/oracle_filter/score_abs.py

Physical convention ("abs"): one fixed microphone gain per pair and scene.
  U = mask-off recording, M = mask-on recording, both from the 48 kHz originals with ONE common scale factor, so M keeps
      its physical level relative to U (assumes Cox kept the recording gain fixed between mask-on and mask-off takes).
  F = oracle forward filter (filters.npz, P0 shape) applied to U, then scaled to M's A-weighted active level.
  R = oracle inverse filter applied to M, then scaled to U's A-weighted active level.
  Noise (welding test excerpts, seeded per pair and scene) is set ONCE per pair/scene from U:
      A-weighted active level of U  -  A-weighted Leq of noise  =  L_talker(channel, scene) - N_scene
  and the same noise sample and gain are added to U, M, F, R. Any clipping guard scales the four mixes together.
Equalised convention ("eq", the P0 post-filter rule at the same SNRs, for comparison): each of U, M, F, R is mixed at
  the scene's U-SNR measured on its OWN A-weighted active level; same noise sample.

Scene -> U-SNR (src/muffled_asr/measurement/acoustic_levels.py talker model): L_1m = min(55 + 0.6*max(0, N-48), 82) dBA.
  1 m:   SNR = L_1m - N                     -> -3.8 / -7.8 / -13.0 dB
  cheek: SNR = L_1m + P - N, P = proximity gain of the cheek mic over 1 m, ASSUMED: P = 20 dB (primary), 14 dB (sens.)
"""
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np, soundfile as sf
sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).resolve().parents[1] / "measurement")]
from common import P0
from gen_conditions import FIL, NOISE, filt, orig16
from acoustic_levels import active_level_a, leq_a

SCENES = (75, 85, 95); PROX = {"cheek": (20, 14), "far": (0,)}
BLOCK = "coxh_abs"


def talker_1m(n): return min(55 + 0.6 * max(0.0, n - 48), 82.0)


def noise(pid, scene, n):
    s = int.from_bytes(hashlib.sha256(f"{pid}|abs{scene}".encode()).digest()[:8], "little"); rng = np.random.default_rng(s)
    ex = sf.read(NOISE[int(rng.integers(len(NOISE)))]["path"], dtype="float32")[0]
    ex = np.roll(ex, -int(rng.integers(len(ex)))); return np.tile(ex, int(np.ceil(n / len(ex))))[:n]


A = lambda x: active_level_a(x)[0]
db_gain = lambda x, target: x * 10 ** ((target - A(x)) / 20)


def main():
    man = list(csv.DictReader(open(P0 / "speech_manifest.csv", newline=""))); pairs = {}
    for r in man:
        if r["dataset"] == "coxh":
            pairs.setdefault(f'coxh|{r["channel"]}|{r["speaker_id"]}|{r["mask_type"]}', {})["U" if r["condition"] == "mask_off" else "M"] = r
    rows, levels = [], []; outdir = P0 / "conds" / BLOCK
    for pid, pr in sorted(pairs.items()):
        ch = pid.split("|")[1]
        u0, m0 = orig16(pr["U"]), orig16(pr["M"])
        k = 10 ** (-26 / 20) / 10 ** (A(u0) / 20)  # U at a fixed digital active level; M keeps its physical offset
        u0, m0 = (u0 * k).astype(np.float32), (m0 * k).astype(np.float32)
        Lu, Lm = A(u0), A(m0)
        sig = {"U": u0, "M": m0, "F": db_gain(filt(u0, FIL[f"{pid}|fwd"]), Lm), "R": db_gain(filt(m0, FIL[f"{pid}|inv"]), Lu)}
        levels.append({"pair_id": pid, "M_minus_U_dBA": round(Lm - Lu, 2)})
        for scene in SCENES:
            for P in PROX[ch]:
                snr = talker_1m(scene) + P - scene
                mixes = {}
                for conv in ("abs", "eq"):
                    for c, s in sig.items():
                        nz = noise(pid, scene, len(s)); ref = Lu if conv == "abs" else A(s)
                        mixes[(conv, c)] = s + nz * 10 ** ((ref - snr - leq_a(nz)) / 20)
                for conv in ("abs", "eq"):
                    pk = max(np.abs(mixes[(conv, c)]).max() for c in sig); g = 0.99 / pk if pk > 0.99 else 1.0
                    for c in sig:
                        y = (mixes[(conv, c)] * g).astype(np.float32)
                        f = outdir / pid.replace("|", "_") / f"{c}_{conv}_N{scene}_P{P}.flac"; f.parent.mkdir(parents=True, exist_ok=True)
                        sf.write(f, y, 16000, subtype="PCM_16")
                        rows.append({"audio_filepath": str(f), "text": pr["U"]["ref_scoring"], "dataset": "coxh", "channel": ch,
                                     "speaker": pr["U"]["speaker_id"], "mask": pr["U"]["mask_type"], "pair_id": pid, "condition": c,
                                     "snr_db": scene, "scene_dba": scene, "prox_db": P, "u_snr_db": round(snr, 2), "conv": conv,
                                     "snr_convention": "post", "threshold_db": 6, "primary": pr["U"]["primary"] == "True",
                                     "clip_gain_db": round(20 * np.log10(g), 2), "duration": round(len(y) / 16000, 3)})
        print(pid, f"M-U {Lm - Lu:+.1f} dBA", flush=True)
    (P0 / "manifests" / f"{BLOCK}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (P0 / "coxh_abs_levels.json").write_text(json.dumps(levels, indent=1))
    print(len(rows), "files", round(sum(r["duration"] for r in rows) / 3600, 2), "h")


if __name__ == "__main__": main()
