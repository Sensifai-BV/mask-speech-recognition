"""P0 sections 3.5-3.7: WER, gap, R_fwd, R_inv with paired talker bootstrap (1,000, seed 0)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, jiwer
sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, score_norm

RES = ROOT / "results/p0"; OUT = RES / "tables"; OUT.mkdir(parents=True, exist_ok=True)


def load(block, look=6):
    f = RES / f"hyps_base_manifests-{block}_look{look}.jsonl"
    if not f.exists(): return None
    d = pd.DataFrame([json.loads(l) for l in open(f)])
    d = apply_reference_overrides(d)
    errs = [jiwer.process_words(score_norm(r), score_norm(h)) for r, h in zip(d.text, d.pred_text)]
    d["E"] = [o.substitutions + o.deletions + o.insertions for o in errs]; d["N"] = [len(score_norm(r).split()) for r in d.text]
    d["snr"] = d.snr_db.fillna(99).astype(int); d["look"] = look
    return d


OVERRIDES = ROOT / "datasets/p0/reference_overrides.json"  # written by apply_listening.py from the listening check
U_SOURCED = {"U", "F", "Fx", "Ri", "Fir"}  # conditions derived from the mask-off recording; M and R come from mask-on


def apply_reference_overrides(d):
    """Replace the scoring reference of rows whose SOURCE recording was judged a reading departure by ear.
    No file (or an empty one) => d unchanged. Keyed by '<dataset>|<channel>|<speaker>|<mask>|<mask_off|mask_on>'."""
    if not OVERRIDES.exists() or "pair_id" not in d: return d
    ov = json.loads(OVERRIDES.read_text()).get("references", {})
    if not ov: return d
    side = d.condition.map(lambda c: "mask_off" if c in U_SOURCED else "mask_on")
    key = d.pair_id + "|" + side
    d = d.copy(); hit = key.isin(ov); d.loc[hit, "text"] = key[hit].map(ov)
    return d


def wer(d): return 100 * d.E.sum() / d.N.sum()


def ratios(d, snr, conv="post", thr_fwd=6, cond_fwd="F", B=1000):
    """Return dict of pooled WERs, gap and ratios with talker-bootstrap CIs for one dataset x SNR."""
    base = d[(d.snr == snr)]
    pick = lambda c, conv_=conv, thr=6: base[(base.condition == c) & (base.snr_convention == (conv_ if c != "U" else "post")) & (base.threshold_db == thr)]
    U, M, F, Rr = pick("U"), pick("M"), pick(cond_fwd, conv, thr_fwd), pick("R", conv, thr_fwd)
    if snr == 99: U, M, F, Rr = (pick(c, "post", thr_fwd if c in "FR" else 6) for c in ("U", "M", cond_fwd, "R"))
    agg = {k: v.groupby("speaker")[["E", "N"]].sum() for k, v in dict(U=U, M=M, F=F, R=Rr).items() if len(v)}
    spk = np.array(sorted(agg["U"].index)); rng = np.random.default_rng(0)
    def stat(s):
        w = {k: 100 * a.loc[s].E.sum() / a.loc[s].N.sum() for k, a in agg.items()}
        gap = w["M"] - w["U"]
        return w, gap, (w["M"] - w.get("F", np.nan)) / gap if gap else np.nan, (w.get("R", np.nan) - w["U"]) / gap if gap else np.nan
    w, gap, rf, ri = stat(spk); bs = np.array([stat(rng.choice(spk, len(spk)))[1:] for _ in range(B)])
    ok = gap >= 2
    q = lambda i: np.nanpercentile(bs[:, i], [2.5, 97.5]).round(2).tolist()
    return {"n_talkers": len(spk), **{f"WER_{k}": round(v, 2) for k, v in w.items()}, "gap": round(gap, 2), "gap_CI": q(0),
            "R_fwd": round(rf, 3) if ok else None, "R_fwd_CI": q(1) if ok else None,
            "R_inv": round(ri, 3) if ok else None, "R_inv_CI": q(2) if ok else None}


def main():
    blocks = {"Geng (surgical, 28 talkers)": ("geng", True), "Geng all 30 (sensitivity)": ("geng", False),
              "Cox-H cheek (A-E, 4 talkers)": ("coxh_cheek", True), "Cox-H 1 m (robustness)": ("coxh_far", True)}
    data = {b: load(b) for b in ("geng", "coxh_cheek", "coxh_far")}
    t1 = []
    for name, (b, prim) in blocks.items():
        d = data[b]
        if d is None: continue
        d = d[d.primary] if prim else d
        for snr in (99, 0, 5, 10, 20):
            t1.append({"dataset": name, "snr": "quiet" if snr == 99 else snr, **ratios(d, snr)})
    T1 = pd.DataFrame(t1); T1.to_csv(OUT / "table_P0-1.csv", index=False)
    # P0-2: per talker at 10 dB, Geng primary
    t2 = []
    if data["geng"] is not None:
        g = data["geng"]; g = g[g.primary & (g.snr == 10) & (g.threshold_db == 6) & (g.snr_convention == "post")]
        for s, d in g.groupby("speaker"):
            w = {c: wer(d[d.condition == c]) for c in ("U", "M", "F", "R")}
            t2.append({"speaker": s, **{f"WER_{k}": round(v, 2) for k, v in w.items()}, "gap": round(w["M"] - w["U"], 2), "M_minus_F": round(w["M"] - w["F"], 2)})
    pd.DataFrame(t2).to_csv(OUT / "table_P0-2.csv", index=False)
    # P0-3: controls and conventions at 5 and 10 dB
    t3 = []
    for b, name in (("geng", "Geng"), ("coxh_cheek", "Cox-H cheek")):
        d = data[b]
        if d is None: continue
        d = d[d.primary]
        for snr in (5, 10):
            s = d[(d.snr == snr) & (d.threshold_db == 6) & (d.snr_convention == "post")]
            row = {"dataset": name, "snr": snr, **{f"WER_{c}": round(wer(s[s.condition == c]), 2) for c in ("U", "F", "M", "Fx", "Fir", "Ri") if (s.condition == c).any()}}
            pre = ratios(d, snr, conv="pre"); row.update({"pre_WER_M": pre["WER_M"], "pre_WER_F": pre["WER_F"], "pre_gap": pre["gap"], "pre_R_fwd": pre["R_fwd"], "pre_R_fwd_CI": pre["R_fwd_CI"]})
            fx = ratios(d, snr, cond_fwd="Fx"); row.update({"R_fwd_with_Fx": fx["R_fwd"]})
            if (s.condition == "Fir").any(): row["R_fwd_with_Fir"] = ratios(d, snr, cond_fwd="Fir")["R_fwd"]
            for thr in (3, 9): row[f"R_fwd_thr{thr}"] = ratios(d, snr, thr_fwd=thr)["R_fwd"]
            t3.append(row)
    pd.DataFrame(t3).to_csv(OUT / "table_P0-3.csv", index=False)
    # context
    tc = []
    for look, ms in ((0, 80), (6, 560), (13, 1120)):
        for b, name in (("geng", "Geng"), ("coxh_cheek", "Cox-H cheek")):
            d = load("context_subset", look) if look != 6 else data[b]
            if d is None: continue
            d = d[(d.dataset == ("geng" if b == "geng" else "coxh")) & d.primary]
            for snr in (5, 10): tc.append({"dataset": name, "context_ms": ms, "snr": snr, **ratios(d, snr)})
    pd.DataFrame(tc).to_csv(OUT / "table_P0-context.csv", index=False)
    g = T1[(T1.dataset == "Geng (surgical, 28 talkers)") & T1.snr.isin([5, 10])]
    rf = g.R_fwd.astype(float).mean() if len(g) == 2 and g.R_fwd.notna().all() else None
    rule = None if rf is None else ("R_fwd >= 0.30: filter paradigm substantially bounded -> P1, P2, P3" if rf >= .30 else
            "0.15-0.30: filter explains most, not all -> P1-P3 with modest claims" if rf >= .15 else "< 0.15: covering is a filter for ASR -> stop after P1")
    (OUT / "decision.json").write_text(json.dumps({"Geng_R_fwd_mean_5_10dB": rf, "rule": rule, "gaps_5_10": g.gap.tolist()}, indent=1))
    print(T1.to_string()); print("decision:", rf, rule)


if __name__ == "__main__": main()
