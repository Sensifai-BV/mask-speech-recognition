"""Task 6.1: re-score the Cox-H adapter evaluations (paper Table III) with the corrected P0 references.

The adapter hyps in results/human_quiet and results/human_noisy were written by src/muffled_asr/evaluation/eval_hf.py on
14 Oct 2026, i.e. before the 18 Oct scoring audit. Their stored references (`ref_norm`) are the textbook Rainbow
passage ('they act AS a prism'; P04 '... at the end of THE rainbow') with eval_hf's plain normalisation.
Corrected = src/muffled_asr/oracle_filter/common.reference('coxh', speaker) + score_norm on both sides (identical to every P0 table),
plus score.apply_reference_overrides-equivalent edits from the listening check if any exist.

    python src/muffled_asr/followup/rescore_adapters.py
-> results/followup/table3_rescored_by_condition.csv   (model x set x SNR x unmasked/masked, old vs corrected)
   results/followup/table3_rescored_overall.csv        (model x set, old vs corrected; old must equal scores.csv)
   results/followup/table3_rescored_utts.csv           (per utterance)
"""
import json, sys
from pathlib import Path
import jiwer, pandas as pd
from muffled_asr.paths import ROOT as R
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "oracle_filter"))
from common import RAINBOW, reference, score_norm

OUT = R / "results/followup"; OUT.mkdir(parents=True, exist_ok=True)
ov = json.loads((R / "datasets/p0/reference_overrides.json").read_text()).get("references", {})
if any(k.startswith("coxh|") for k in ov):
    sys.exit("listening-check overrides for Cox-H now exist: map them onto these rows before re-scoring (not implemented)")
rows = []
for f in sorted(list((R / "results/human_quiet").glob("hyps_*.jsonl")) + list((R / "results/human_noisy").glob("hyps_*.jsonl"))):
    model, man = f.stem[len("hyps_"):].rsplit("_look", 1)[0].split("_", 1)
    for l in open(f):
        r = json.loads(l)
        new_ref = reference("coxh", r["speaker"], RAINBOW); hyp = score_norm(r["pred_text"])
        o = jiwer.process_words(new_ref, hyp)
        rows.append({"model": model, "set": man, "speaker": r["speaker"], "design": r["design"],
                     "masked": "masked" if "covering" in r["condition"] else "unmasked", "snr": r.get("snr_db", "quiet"),
                     "E_old": r["S"] + r["D"] + r["I"], "N_old": r["n_ref"],
                     "E_new": o.substitutions + o.deletions + o.insertions, "N_new": len(new_ref.split()),
                     "ref_old": r["ref_norm"], "ref_new": new_ref, "hyp": r["pred_text"]})
d = pd.DataFrame(rows); d["snr"] = d.snr.fillna("quiet").astype(str)
d.drop(columns=["ref_old", "ref_new"]).to_csv(OUT / "table3_rescored_utts.csv", index=False)
w = lambda g: pd.Series({"WER_old": 100 * g.E_old.sum() / g.N_old.sum(), "WER_new": 100 * g.E_new.sum() / g.N_new.sum(), "n_utts": len(g)})
ov_ = d.groupby(["model", "set"]).apply(w).round(3).reset_index(); ov_["delta"] = (ov_.WER_new - ov_.WER_old).round(3)
ov_.to_csv(OUT / "table3_rescored_overall.csv", index=False)
bc = d.groupby(["set", "model", "snr", "masked"]).apply(w).round(2).reset_index()
g = bc.pivot_table(index=["set", "model", "snr"], columns="masked", values=["WER_old", "WER_new"])
g.columns = [f"{a}_{b}" for a, b in g.columns]; g = g.reset_index()
if "WER_old_masked" in g:
    g["gap_old"] = (g.WER_old_masked - g.WER_old_unmasked).round(2); g["gap_new"] = (g.WER_new_masked - g.WER_new_unmasked).round(2)
g.to_csv(OUT / "table3_rescored_by_condition.csv", index=False)
pd.set_option("display.width", 220); print(ov_.to_string(index=False)); print(g[g.set.str.contains("noisy")].to_string(index=False))
print("refs differing from old, by speaker:", d[d.ref_old != d.ref_new].groupby("speaker").size().to_dict())
