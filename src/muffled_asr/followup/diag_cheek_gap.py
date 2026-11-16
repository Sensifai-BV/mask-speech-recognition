"""Task 6.2 diagnostic: why the Cox-H cheek gap is ~6 pts in Table I (gen_conditions, 10 dB) but ~1 pt in the
fixed-noise table's 'equalised' rows (gen_abs, 95 dBA / P=14, U-SNR 1.0 dB) at a similar WER_U (6.4 vs 7.1 %).

For every Cox-H pair and channel it re-derives, without writing audio, the noise gain each generator uses for U and M,
and measures the resulting mixes with BOTH level estimators:
  unweighted = gen_conditions.active_power (STFT frames > median + 6 dB) vs mean noise power   (Table I rule)
  A-weighted = acoustic_levels.active_level_a (frames > max(p10+6, peak-45)) vs leq_a          (gen_abs rule)
Output: results/followup/diag_cheek_gap.csv and a printed per-channel summary.
    python src/muffled_asr/followup/diag_cheek_gap.py
"""
import csv, sys
from pathlib import Path
import numpy as np, pandas as pd, soundfile as sf
from muffled_asr.paths import ROOT as R
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("oracle_filter", "measurement")]
from common import P0
from gen_conditions import active_power, active_power_noise, noise_for
from gen_abs import noise as noise_abs
from acoustic_levels import active_level_a, leq_a

OUT = R / "results/followup"; OUT.mkdir(parents=True, exist_ok=True)
db = lambda p: 10 * np.log10(p)
man = list(csv.DictReader(open(P0 / "speech_manifest.csv", newline=""))); pairs = {}
for r in man:
    if r["dataset"] == "coxh":
        pairs.setdefault(f'coxh|{r["channel"]}|{r["speaker_id"]}|{r["mask_type"]}', {})["U" if r["condition"] == "mask_off" else "M"] = r
rows = []
for pid, pr in sorted(pairs.items()):
    ch, spk, mask = pid.split("|")[1:]
    U = sf.read(pr["U"]["path"], dtype="float32")[0]; M = sf.read(pr["M"]["path"], dtype="float32")[0]
    n_t1 = noise_for(pid, 10, max(len(U), len(M)))           # Table I noise draw (10 dB)
    n_ab = noise_abs(pid, 95, max(len(U), len(M)))           # fixed-noise draw (95 dBA scene)
    rec = {"pair_id": pid, "channel": ch, "speaker": spk, "mask": mask}
    for c, s in (("U", U), ("M", M)):
        rec[f"uw_active_{c}"] = db(active_power(s)); rec[f"A_active_{c}"] = active_level_a(s)[0]
        rec[f"A_activefrac_{c}"] = active_level_a(s)[1]
    rec["noise_t1_uw"] = db(active_power_noise(n_t1)); rec["noise_t1_A"] = leq_a(n_t1)
    rec["noise_abs_uw"] = db(active_power_noise(n_ab)); rec["noise_abs_A"] = leq_a(n_ab)
    # A-weighted SNR of Table I's 10 dB mixes (noise scaled so the unweighted SNR is 10 dB for each signal)
    for c in "UM":
        g_db = rec[f"uw_active_{c}"] - 10 - rec["noise_t1_uw"]           # noise gain applied by gen_conditions.mix
        rec[f"T1_snrA_{c}"] = rec[f"A_active_{c}"] - (rec["noise_t1_A"] + g_db)
    # M - U level differences under the two estimators (positive = M louder)
    rec["M_minus_U_uw"] = rec["uw_active_M"] - rec["uw_active_U"]; rec["M_minus_U_A"] = rec["A_active_M"] - rec["A_active_U"]
    rows.append(rec)
d = pd.DataFrame(rows); d.to_csv(OUT / "diag_cheek_gap.csv", index=False)
pd.set_option("display.width", 200)
s = d.groupby("channel")[["T1_snrA_U", "T1_snrA_M", "M_minus_U_uw", "M_minus_U_A", "A_activefrac_U", "A_activefrac_M"]].agg(["mean", "min", "max"]).round(2)
print(s.T.to_string())
d["T1_A_snr_drop_M"] = d.T1_snrA_U - d.T1_snrA_M
print("\nTable I 10 dB: A-weighted SNR of M below that of U (dB), by mask (cheek):")
print(d[d.channel == "cheek"].groupby("mask").T1_A_snr_drop_M.agg(["mean", "min", "max"]).round(2).to_string())
print("\nnoise uw-A offset, Table I draws: ", (d.noise_t1_uw - d.noise_t1_A).describe()[["mean", "min", "max"]].round(2).to_dict())
