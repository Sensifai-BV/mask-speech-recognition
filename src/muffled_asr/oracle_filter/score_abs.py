"""Exp 3 scoring: Table I rows for the fixed-absolute-noise convention (block coxh_abs, see gen_abs.py).

Uses score.load (so listening-check overrides apply) and score.ratios unchanged (4-talker bootstrap, seed 0), per
channel x proximity assumption x convention (abs = physical level, eq = equalised at the same SNR) x scene.
-> results/p0/tables/table_P0-abs.csv
"""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from score import OUT, load, ratios


def main():
    d = load("coxh_abs")
    if d is None: sys.exit("no hyps for coxh_abs yet")
    rows = []
    for (ch, P, conv), g in d.groupby(["channel", "prox_db", "conv"]):
        for scene, s in g.groupby("scene_dba"):
            r = ratios(s, scene)
            rows.append({"channel": {"cheek": "cheek", "far": "1 m"}[ch], "proximity_db": P if ch == "cheek" else None,
                         "convention": {"abs": "fixed noise (physical level)", "eq": "equalised (post-filter rule)"}[conv],
                         "scene_dBA": scene, "U_SNR_dB": float(s.u_snr_db.iloc[0]), **r})
    T = pd.DataFrame(rows).sort_values(["channel", "proximity_db", "scene_dBA", "convention"])
    T.to_csv(OUT / "table_P0-abs.csv", index=False)
    pd.set_option("display.width", 250); print(T.drop(columns=["n_talkers"]).to_string(index=False))


if __name__ == "__main__": main()
