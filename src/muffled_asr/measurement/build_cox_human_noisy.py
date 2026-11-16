"""Real masked/unmasked Cox speech (1 m mic) + held-out welding noise at fixed SNRs.

SNR reference = A-weighted active level of the SAME talker's mask-off recording in the
same design session, so a real covering keeps whatever level it removes (pre-covering
convention). The mask-on and mask-off files of a pair get the identical noise crop and
gain. Noise comes only from the test welding folders of the corpus split.

    python src/muffled_asr/measurement/build_cox_human_noisy.py --snr -10 -5 0 5 10
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np, soundfile as sf
from muffled_asr.paths import ROOT
from muffled_asr.simulation.mask_sim import load_16k_mono  # noqa: E402
from muffled_asr.measurement import acoustic_levels as AL  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--human", type=Path, default=Path("datasets/calibrated/cox-human-eval-far"))
    p.add_argument("--provenance", type=Path, default=Path("datasets/calibrated/librispeech-noise-control/provenance.json"))
    p.add_argument("--noise-root", type=Path, default=Path(__import__("os").environ.get("WELDING_NOISE_ROOT", "datasets/original/intel_robotic_welding_dataset")))
    p.add_argument("--snr", type=float, nargs="+", default=[-10, -5, 0, 5, 10])
    p.add_argument("--out", type=Path, default=Path("datasets/realistic/cox-human-noisy-far"))
    a = p.parse_args()
    prov = json.loads(a.provenance.read_text())
    test_groups = {g for g, s in prov["noise_groups"].items() if s == "test"}
    noises = sorted(q for q in a.noise_root.rglob("*.flac") if q.relative_to(a.noise_root).parts[0] in test_groups)
    rows = [json.loads(l) for f in ("test.json", "dev.json") for l in (a.human / f).read_text().splitlines() if l.strip()]
    by = {(r["design"], r["speaker"], r["condition"] == "quiet-unmasked"): r for r in rows}
    out_rows = {"test": [], "dev": []}
    for (design, spk, unmasked), r in sorted(by.items()):
        if unmasked: continue
        off_row = by[(design, spk, True)]
        ref_level = AL.active_level_a(sf.read(off_row["audio_filepath"], dtype="float32")[0])[0]
        split = "test" if design in "ABCD" else "dev"
        for snr in a.snr:
            seed = int.from_bytes(hashlib.sha256(f"{design}:{spk}:{snr}".encode()).digest()[:8], "little")
            rng = np.random.default_rng(seed); npath = noises[int(rng.integers(len(noises)))]
            noise = load_16k_mono(npath)
            for row in (r, off_row):
                x = sf.read(row["audio_filepath"], dtype="float32")[0]
                n = noise if len(noise) >= len(x) else np.tile(noise, int(np.ceil(len(x) / len(noise))))
                o = int(np.random.default_rng(seed + 1).integers(0, len(n) - len(x) + 1)); n = n[o:o + len(x)]
                g = 10 ** ((ref_level - snr - AL.leq_a(n)) / 20); y = x + g * n
                y = y * (10 ** (-22 / 20) / (np.sqrt(np.mean(y ** 2)) + 1e-12)); y = y / max(1, np.abs(y).max() / .99)
                state = "mask-off" if row is off_row else "mask-on"
                dest = a.out / split / f"snr{int(snr):+d}" / f"{design}_{state}_{spk}.wav"; dest.parent.mkdir(parents=True, exist_ok=True)
                sf.write(dest, y.astype(np.float32), 16000)
                out_rows[split].append({**row, "audio_filepath": str(dest.resolve()), "snr_db": snr,
                                        "noise": str(npath.relative_to(a.noise_root)),
                                        "snr_convention": "A-weighted, ref = same talker mask-off active level"})
    for s, v in out_rows.items():
        (a.out / f"{s}.json").write_text("".join(json.dumps(x) + "\n" for x in v)); print(s, len(v))


if __name__ == "__main__": main()
