"""P0 extension: Geng at -10/-5 dB (block geng_lowsnr), scored with score.ratios unchanged and joined to the original Geng rows."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from score import OUT, load, ratios, wer

def main():
    lo, hi = load("geng_lowsnr"), load("geng")
    d = pd.concat([lo, hi], ignore_index=True)
    t1 = []
    for name, prim in (("Geng (surgical, 28 talkers)", True), ("Geng all 30 (sensitivity)", False)):
        g = d[d.primary] if prim else d
        for snr in (99, -10, -5, 0, 5, 10, 20):
            t1.append({"dataset": name, "snr": "quiet" if snr == 99 else snr, **ratios(g, snr)})
    T1 = pd.DataFrame(t1); T1.to_csv(OUT / "table_P0-1_geng_lowsnr.csv", index=False)
    t3 = []; g = lo[lo.primary]
    for snr in (-10, -5):
        s = g[(g.snr == snr) & (g.threshold_db == 6) & (g.snr_convention == "post")]
        row = {"snr": snr, **{f"WER_{c}": round(wer(s[s.condition == c]), 2) for c in ("U", "F", "M", "R", "Fx", "Ri")}}
        pre = ratios(g, snr, conv="pre"); row.update(pre_WER_M=pre["WER_M"], pre_WER_F=pre["WER_F"], pre_gap=pre["gap"], pre_R_fwd=pre["R_fwd"], pre_R_fwd_CI=pre["R_fwd_CI"])
        row["R_fwd_with_Fx"] = ratios(g, snr, cond_fwd="Fx")["R_fwd"]
        for thr in (3, 9): row[f"R_fwd_thr{thr}"] = ratios(g, snr, thr_fwd=thr)["R_fwd"]
        t3.append(row)
    pd.DataFrame(t3).to_csv(OUT / "table_P0-3_geng_lowsnr.csv", index=False)
    t2 = []
    for snr in (-10, -5, 0):
        s = d[d.primary & (d.snr == snr) & (d.threshold_db == 6) & (d.snr_convention == "post")]
        for spk, x in s.groupby("speaker"):
            w = {c: wer(x[x.condition == c]) for c in ("U", "M", "F", "R")}
            t2.append({"snr": snr, "speaker": spk, **{f"WER_{k}": round(v, 2) for k, v in w.items()}, "gap": round(w["M"] - w["U"], 2), "M_minus_F": round(w["M"] - w["F"], 2)})
    T2 = pd.DataFrame(t2); T2.to_csv(OUT / "table_P0-2_geng_lowsnr.csv", index=False)
    pd.set_option("display.width", 250); print(T1.drop(columns=["n_talkers"]).to_string()); print(pd.DataFrame(t3).to_string())
    for snr, x in T2.groupby("snr"):
        print(snr, "per-talker corr(gap, M-F)=%.2f  talkers gap>0: %d/%d" % (np.corrcoef(x.gap, x.M_minus_F)[0, 1], (x.gap > 0).sum(), len(x)))

if __name__ == "__main__": main()
