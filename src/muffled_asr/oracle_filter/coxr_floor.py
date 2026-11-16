"""Reading-to-reading floor on Cox-R (plan step 1): is the far-field residual more than two different readings?

    python src/muffled_asr/oracle_filter/coxr_floor.py prep      # cut 6 x 6 readings -> datasets/p0/data16k/coxr/  (segments.json)
    python src/muffled_asr/oracle_filter/coxr_floor.py gen       # 45 null + 108 masked pairs x U/M/F/R x {quiet, 0, -5, -10 dB}
    python src/muffled_asr/evaluation/eval_hf.py --manifests datasets/p0/manifests/coxr.jsonl --models base --lookahead 6 --out results/p0
    python src/muffled_asr/oracle_filter/coxr_floor.py score     # tables -> results/p0/tables/coxr_*.csv, figure, decision

Data: Cox 2022 repeated_reading_rainbow_passage.zip (one talker, mono, 48 kHz, six readings per file; mic position not
stated in the dataset description). Files: '<mask> off' and '<mask> on' for cloth v1 + acetate insert, printed cloth,
surgical. SHA256: datasets/SHA256SUMS.

Pairs (all P0 machinery unchanged: P0 LTAS oracle filter at the 6 dB active threshold, min-phase FIR fwd/inv,
peak_norm, welding test noise, post-filter SNR convention, seed per pair id and SNR):
  null    U = reading i, M' = reading j (i < j, time order) of the SAME mask-off file: 15 pairs x 3 files = 45.
  masked  U = reading i of the mask-off file, M = reading j of the matching mask-on file: 6 x 6 = 36 per mask = 108.
Metrics per pair set and SNR (pooled over pairs, as P0 pools over talkers):
  gap = WER_M - WER_U, residual = WER_M - WER_F, R_fwd = residual / gap and R_inv = (WER_R - WER_U) / gap (gap >= 2).
CIs: 2,000-sample bootstrap over READINGS (resample the 6 U and 6 M readings, keep the pairs they induce), which
respects that the 15 / 36 pairs share readings; a pair-level bootstrap is reported as well.
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from common import P0, RAINBOW, RAINBOW_READ, ROOT, peak_norm, score_norm, to16k  # noqa: E402

SRC = ROOT / "datasets/original/cox-2022/repeated_reading"
SEG = P0 / "coxr/segments.json"
D16 = P0 / "data16k/coxr"
MASKS = {  # family -> (off file stem, on file stem)
    "cloth_v1_acetate": ("Cloth v1 with acetate insert mask off", "Cloth v1 with acetate insert mask on"),
    "printed_cloth": ("PrintedClothMaskOff", "PrintedClothMaskOn"),
    "surgical": ("SurgicalMaskOff", "SurgicalMaskOn"),
}
SNRS = [None, 0, -5, -10]
TABLES = ROOT / "results/p0/tables"


def reading_path(stem: str, k: int) -> Path:
    return D16 / f"{stem.replace(' ', '_')}_r{k}.wav"


def prep() -> None:
    seg = json.loads(SEG.read_text()); D16.mkdir(parents=True, exist_ok=True)
    for stem, rs in seg.items():
        x, sr = sf.read(SRC / f"{stem}.wav", dtype="float32")
        for r in rs:
            y = peak_norm(to16k(x[int(r["start_s"] * sr):int(r["end_s"] * sr)], sr))
            sf.write(reading_path(stem, r["reading"]), y, 16000)
    print(f"{sum(len(v) for v in seg.values())} readings -> {D16}")


def pairs() -> list[dict]:
    out = []
    for fam, (off, on) in MASKS.items():
        for i, j in combinations(range(1, 7), 2):
            out.append({"pair_id": f"coxr|{fam}|null|{i}{j}", "kind": "null", "mask": fam, "u_file": off, "u_read": i, "m_file": off, "m_read": j})
        for i in range(1, 7):
            for j in range(1, 7):
                out.append({"pair_id": f"coxr|{fam}|masked|{i}{j}", "kind": "masked", "mask": fam, "u_file": off, "u_read": i, "m_file": on, "m_read": j})
    return out


def gen() -> None:
    from filters import band_db, fwd_inv, ltas
    from gen_conditions import filt, mix
    rows, gains = [], []; outdir = P0 / "conds/coxr"
    for p in pairs():
        U = sf.read(reading_path(p["u_file"], p["u_read"]), dtype="float32")[0]
        M = sf.read(reading_path(p["m_file"], p["m_read"]), dtype="float32")[0]
        Pu, au = ltas(U); Pm, am = ltas(M); g = band_db(Pm, Pu); hf, hi = fwd_inv(g)
        gains.append({**p, "band_gain_db": [round(float(v), 3) for v in g], "active_frac_U": round(au, 3), "active_frac_M": round(am, 3)})
        sigs = {"U": U, "M": M, "F": peak_norm(filt(U, hf)), "R": peak_norm(filt(M, hi))}
        for cond, s in sigs.items():
            for snr in SNRS:
                y, _ = mix(s, p["pair_id"], snr)
                tag = f"{cond}_" + ("quiet" if snr is None else f"snr{snr:+d}")
                f = outdir / p["pair_id"].replace("|", "_") / f"{tag}.flac"; f.parent.mkdir(parents=True, exist_ok=True)
                sf.write(f, y, 16000, subtype="PCM_16")
                rows.append({"audio_filepath": str(f), "text": RAINBOW, "dataset": "coxr", "speaker": "coxr_talker", **p,
                             "condition": cond, "snr_db": snr, "duration": round(len(y) / 16000, 3)})
    (P0 / "manifests").mkdir(exist_ok=True)
    (P0 / "manifests/coxr.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (P0 / "coxr/pair_gains.json").write_text(json.dumps({"bands_hz": [100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250,
                                                         1600, 2000, 2500, 3150, 4000, 5000, 6300], "pairs": gains}, indent=1))
    print(f"{len(pairs())} pairs, {len(rows)} files, {sum(r['duration'] for r in rows) / 3600:.2f} h -> manifests/coxr.jsonl")


# ----------------------------------------------------------------------------- scoring
def load_scored():
    import jiwer
    import pandas as pd
    d = pd.DataFrame([json.loads(l) for l in open(ROOT / "results/p0/hyps_base_manifests-coxr_look6.jsonl")])
    # which passage version did this talker read? decide on the quiet unmasked readings, apply to every row
    q = d[(d.condition == "U") & d.snr_db.isna()].drop_duplicates(["u_file", "u_read"])
    hyp = [score_norm(h) for h in q.pred_text]
    w = {name: jiwer.wer([score_norm(t)] * len(hyp), hyp) for name, t in (("as a prism", RAINBOW), ("like a prism", RAINBOW_READ))}
    ref = score_norm(RAINBOW if w["as a prism"] <= w["like a prism"] else RAINBOW_READ)
    o = [jiwer.process_words(ref, score_norm(h)) for h in d.pred_text]
    d["E"] = [x.substitutions + x.deletions + x.insertions for x in o]; d["N"] = len(ref.split())
    d["snr"] = d.snr_db.fillna(99).astype(int)
    return d, {"reference_version": min(w, key=w.get), "quiet_U_wer_pct": {k: round(100 * v, 2) for k, v in w.items()}}


def pair_table(d):
    """One row per pair and SNR with WER of U, M, F, R (each pair has exactly one file per condition)."""
    t = d.pivot_table(index=["pair_id", "kind", "mask", "u_read", "m_read", "snr"], columns="condition", values="E", aggfunc="sum").reset_index()
    n = d.N.iloc[0]
    for c in "UMFR": t[c] = 100 * t[c] / n
    t["gap"] = t.M - t.U; t["residual"] = t.M - t.F
    return t


def pooled(t):
    gap = t.gap.mean(); res = t.residual.mean(); ri = (t.R - t.U).mean()
    return {"WER_U": t.U.mean(), "WER_M": t.M.mean(), "WER_F": t.F.mean(), "WER_R": t.R.mean(), "gap": gap, "residual": res,
            "R_fwd": res / gap if gap >= 2 else np.nan, "R_inv": ri / gap if gap >= 2 else np.nan}


def boot(t, B=2000, seed=0, level="reading"):
    rng = np.random.default_rng(seed); out = []
    for _ in range(B):
        if level == "pair":
            s = t.iloc[rng.integers(0, len(t), len(t))]
        else:  # resample U readings and M readings, keep induced pairs (multiplicity preserved)
            ui = rng.integers(1, 7, 6); mi = rng.integers(1, 7, 6)
            cu = np.bincount(ui, minlength=7); cm = np.bincount(mi, minlength=7)
            wts = cu[t.u_read.values] * cm[t.m_read.values]
            if wts.sum() == 0: continue
            s = t.loc[np.repeat(t.index.values, wts)]
        out.append(pooled(s))
    return out


def ci(vals):
    v = np.array([x for x in vals if np.isfinite(x)])
    return [round(float(np.percentile(v, 2.5)), 2), round(float(np.percentile(v, 97.5)), 2)] if len(v) > 50 else None


def score() -> None:
    import pandas as pd
    d, refinfo = load_scored(); t = pair_table(d); TABLES.mkdir(parents=True, exist_ok=True); t.to_csv(TABLES / "coxr_pairs.csv", index=False)
    rows = []
    for snr in (99, 0, -5, -10):
        for mask in list(MASKS) + ["all"]:
            for kind in ("null", "masked"):
                s = t[(t.snr == snr) & (t.kind == kind) & ((t["mask"] == mask) if mask != "all" else True)]
                p = pooled(s)
                if mask == "all":
                    bs_r = []
                    rng = np.random.default_rng(1)
                    fam = {m: s[s["mask"] == m] for m in MASKS}
                    for b in range(2000):
                        parts = []
                        for m, sm in fam.items():
                            ui = rng.integers(1, 7, 6); mi = rng.integers(1, 7, 6)
                            w = np.bincount(ui, minlength=7)[sm.u_read.values] * np.bincount(mi, minlength=7)[sm.m_read.values]
                            parts.append(sm.loc[np.repeat(sm.index.values, w)])
                        bs_r.append(pooled(pd.concat(parts)))
                else:
                    bs_r = boot(s)
                bs_p = boot(s, level="pair")
                rows.append({"snr": "quiet" if snr == 99 else snr, "mask": mask, "kind": kind, "n_pairs": len(s),
                             **{k: round(float(v), 3) for k, v in p.items()},
                             "gap_CI_reading": ci([b["gap"] for b in bs_r]), "residual_CI_reading": ci([b["residual"] for b in bs_r]),
                             "R_fwd_CI_reading": ci([b["R_fwd"] for b in bs_r]) if np.isfinite(p["R_fwd"]) else None,
                             "gap_CI_pair": ci([b["gap"] for b in bs_p]), "residual_CI_pair": ci([b["residual"] for b in bs_p])})
    T = pd.DataFrame(rows); T.to_csv(TABLES / "coxr_floor.csv", index=False)

    # masked minus null, per SNR and mask: difference of gaps and of residuals (independent reading bootstraps)
    cmp = []
    for snr in (99, 0, -5, -10):
        for mask in MASKS:
            a = t[(t.snr == snr) & (t.kind == "masked") & (t["mask"] == mask)]; b = t[(t.snr == snr) & (t.kind == "null") & (t["mask"] == mask)]
            ba, bb = boot(a, seed=2), boot(b, seed=3); n = min(len(ba), len(bb))
            dg = [ba[i]["gap"] - bb[i]["gap"] for i in range(n)]; dr = [ba[i]["residual"] - bb[i]["residual"] for i in range(n)]
            dabs = [ba[i]["gap"] - abs(bb[i]["gap"]) for i in range(n)]
            pa, pb = pooled(a), pooled(b)
            cmp.append({"snr": "quiet" if snr == 99 else snr, "mask": mask, "gap_masked": round(pa["gap"], 2), "gap_null": round(pb["gap"], 2),
                        "gap_diff": round(pa["gap"] - pb["gap"], 2), "gap_diff_CI": ci(dg),
                        "null_abs_gap_p95": round(float(np.percentile(np.abs(b.gap), 95)), 2),
                        "residual_masked": round(pa["residual"], 2), "residual_null": round(pb["residual"], 2),
                        "residual_diff": round(pa["residual"] - pb["residual"], 2), "residual_diff_CI": ci(dr),
                        "P_gap_masked_le_null": round(float(np.mean(np.array(dg) <= 0)), 4)})
    C = pd.DataFrame(cmp); C.to_csv(TABLES / "coxr_masked_vs_null.csv", index=False)
    m10 = C[C.snr == -10]
    decision = {"reference": refinfo, "rule": "at -10 dB the masked-vs-unmasked gap exceeds the unmasked-vs-unmasked gap (reading bootstrap)",
                "per_mask_at_-10dB": {r["mask"]: {"gap_diff": r["gap_diff"], "CI": r["gap_diff_CI"], "exceeds": bool(r["gap_diff_CI"] and r["gap_diff_CI"][0] > 0)}
                                      for _, r in m10.iterrows()}}
    (TABLES / "coxr_decision.json").write_text(json.dumps(decision, indent=1))
    print(json.dumps(refinfo)); print(T.to_string()); print(C.to_string()); print(json.dumps(decision, indent=1))


if __name__ == "__main__":
    {"prep": prep, "gen": gen, "score": score}[sys.argv[1]]()
