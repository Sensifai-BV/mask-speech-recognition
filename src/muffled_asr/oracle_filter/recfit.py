"""P0 experiment 2: recognition-fitted LTI filter (resumable pipeline).

    NUMBA_CACHE_DIR=$PWD/.numba_cache python src/muffled_asr/oracle_filter/recfit.py [split|search|eval|report|all]

F'_talker = minphase FIR of (spectral-oracle band gains g_t + shared correction Delta(b)), same clipping/spline as
filters.fwd_inv's forward path. Delta is fitted by coordinate descent on 14 fit talkers at 0 dB (loss = |pooled
WER(F')-WER(M)| + mean per-talker word edit distance F'<->M hypotheses / len(M hyp) in %), evaluated on the 14
held-out talkers at 0 and -10 dB. Every decode is cached (results/p0/recfit/decode_cache.jsonl) and every search
candidate is logged (search_log.jsonl), so a crash / disk drop only needs a relaunch. No pandas needed.
"""
import csv, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np, soundfile as sf, jiwer
sys.path.insert(0, str(Path(__file__).parent))
from common import P0, ROOT, peak_norm, score_norm
import filters as FL
from gen_conditions import filt, mix

RES = ROOT / "results/p0"; OUT = RES / "recfit"; OUT.mkdir(parents=True, exist_ok=True)
SPLIT, LOG, CACHE, DFINAL = OUT / "split.json", OUT / "search_log.jsonl", OUT / "decode_cache.jsonl", OUT / "delta_final.json"
STEP, DMAX, MAX_SWEEPS, LOOK = 2.0, 10.0, 2, 6
NB = len(FL.BANDS)
BASE = ROOT / "models/nemotron-speech-streaming-en-0.6b"


# ---------------- data ----------------
def load_hyps(snr):
    f = RES / ("hyps_base_manifests-geng_look6.jsonl" if snr >= 0 else "hyps_base_manifests-geng_lowsnr_look6.jsonl")
    d = {}
    for l in open(f):
        r = json.loads(l)
        if r["primary"] and r["snr_db"] == snr and r["snr_convention"] == "post" and r["threshold_db"] == 6 and r["condition"] in ("U", "M", "F"):
            d[(r["condition"], r["speaker"])] = r
    return d


def err(ref, hyp):
    o = jiwer.process_words(score_norm(ref), score_norm(hyp)) if score_norm(hyp) else None
    n = len(score_norm(ref).split())
    return (n if o is None else o.substitutions + o.deletions + o.insertions), n


def pooled(pairs):  # list of (E, N)
    return 100 * sum(e for e, _ in pairs) / sum(n for _, n in pairs)


def speech_manifest():
    return {f'{r["dataset"]}|{r["channel"]}|{r["speaker_id"]}|{r["mask_type"]}|{r["condition"]}': r
            for r in csv.DictReader(open(P0 / "speech_manifest.csv")) if r["dataset"] == "geng"}


def gains():
    g = {}
    for r in csv.DictReader(open(P0 / "band_gains.csv")):
        if r["threshold_db"] and float(r["threshold_db"]) == 6 and r["dataset"] == "geng":
            g.setdefault(r["pair_id"], {})[float(r["band_hz"])] = float(r["gain_db"])
    return {k: np.array([v[b] for b in FL.BANDS]) for k, v in g.items()}


GAINS = None; SM = None; UCACHE = {}


def setup():
    global GAINS, SM
    if GAINS is None: GAINS = gains(); SM = speech_manifest()


def pid_of(spk): return f"geng|mono|{spk}|surgical"


def U_audio(spk):
    setup()
    if spk not in UCACHE: UCACHE[spk] = sf.read(SM[pid_of(spk) + "|mask_off"]["path"], dtype="float32")[0]
    return UCACHE[spk]


def fprime_fir(spk, delta): return FL.fwd_inv(GAINS[pid_of(spk)] + np.asarray(delta, float))[0]


def make_signal(spk, delta, snr):
    """Exactly gen_conditions' F recipe with the shifted filter: peak_norm(filt(U,h)) then mix(sig, pid, snr)."""
    y, _ = mix(peak_norm(filt(U_audio(spk), fprime_fir(spk, delta))), pid_of(spk), snr)
    return y


def q16(y): return np.clip(np.round(y * 32768), -32768, 32767).astype(np.float32) / 32768  # = PCM_16 FLAC round trip


# ---------------- decoder ----------------
class Decoder:
    def __init__(self):
        import torch
        from transformers import AutoModelForRNNT, AutoProcessor
        self.torch = torch; self.dev = "mps" if torch.backends.mps.is_available() else "cpu"
        self.proc = AutoProcessor.from_pretrained(BASE); self.model = AutoModelForRNNT.from_pretrained(BASE).to(self.dev).eval()
        self.proc.set_num_lookahead_tokens(LOOK)
        self.cache = {}
        if CACHE.exists():
            for l in open(CACHE):
                try: r = json.loads(l); self.cache[r["key"]] = r["hyp"]
                except Exception: pass  # torn last line after a disk drop
        self.fh = open(CACHE, "a"); self.ndec = 0
        print("decoder on", self.dev, "cache entries", len(self.cache), flush=True)

    def decode(self, key, make):
        if key in self.cache: return self.cache[key]
        x = make(); torch = self.torch
        inp = self.proc(x, sampling_rate=16000, return_tensors="pt").to(self.dev, dtype=self.model.dtype)
        with torch.inference_mode(): o = self.model.generate(**inp)
        seq = o.sequences if hasattr(o, "sequences") else o
        import re
        hyp = " ".join(re.sub(r"[^a-z' ]", " ", self.proc.batch_decode(seq, skip_special_tokens=True)[0].lower()).split())
        del inp, o, seq
        if self.dev == "mps": torch.mps.empty_cache()
        self.cache[key] = hyp; self.ndec += 1
        self.fh.write(json.dumps({"key": key, "hyp": hyp}) + "\n"); self.fh.flush()
        return hyp


def ckey(spk, delta, snr): return hashlib.sha1(f"{spk}|{snr}|{[round(float(v), 3) for v in delta]}".encode()).hexdigest()


# ---------------- split ----------------
def do_split():
    if SPLIT.exists(): return json.loads(SPLIT.read_text())
    U, M = load_hyps(0), load_hyps(0)
    spks = sorted({s for _, s in U})
    gap = {}
    for s in spks:
        gap[s] = pooled([err(U[("M", s)]["text"], U[("M", s)]["pred_text"])]) - pooled([err(U[("U", s)]["text"], U[("U", s)]["pred_text"])])
    order = sorted(spks, key=lambda s: (gap[s], s)); rng = np.random.default_rng(0); fit, held = [], []
    for i in range(0, len(order), 2):  # consecutive pairs in gap order, random member -> fit
        a, b = order[i], order[i + 1]
        if rng.random() < .5: a, b = b, a
        fit.append(a); held.append(b)
    sp = {"seed": 0, "n_fit": len(fit), "n_held": len(held), "fit": sorted(fit), "held": sorted(held),
          "gap0dB_per_talker": {s: round(gap[s], 2) for s in order},
          "mean_gap_fit": round(float(np.mean([gap[s] for s in fit])), 2), "mean_gap_held": round(float(np.mean([gap[s] for s in held])), 2)}
    SPLIT.write_text(json.dumps(sp, indent=1)); return sp


# ---------------- search ----------------
def fit_loss(dec, fit, delta, Mh):
    setup(); E = []; ed = []
    for s in fit:
        h = dec.decode(ckey(s, delta, 0), lambda: q16(make_signal(s, delta, 0)))
        E.append(err(Mh[s]["text"], h))
        mh = Mh[s]["pred_text"]; o = jiwer.process_words(mh, h) if h else None
        ed.append(100 * ((len(mh.split())) if o is None else o.substitutions + o.deletions + o.insertions) / max(1, len(mh.split())))
    wF = pooled(E); wM = pooled([err(Mh[s]["text"], Mh[s]["pred_text"]) for s in fit])
    t1, t2 = abs(wF - wM), float(np.mean(ed))
    return {"loss": t1 + t2, "term_wer": t1, "term_edit": t2, "WER_Fp": wF, "WER_M": wM}


def do_search(dec):
    sp = do_split(); fit = sp["fit"]; Mh = {s: r for (c, s), r in load_hyps(0).items() if c == "M"}
    logged = {}
    if LOG.exists():
        for l in open(LOG):
            try: r = json.loads(l); logged[tuple(r["delta"])] = r
            except Exception: pass
    state = {"n": 0}

    def ev(delta):
        k = tuple(float(v) for v in delta)
        if k in logged: return logged[k]["loss"]
        t0 = time.time(); r = fit_loss(dec, fit, delta, Mh)
        rec = {"i": len(logged), "delta": list(k), **{a: round(b, 4) for a, b in r.items()}, "sec": round(time.time() - t0, 1)}
        logged[k] = rec; open(LOG, "a").write(json.dumps(rec) + "\n"); print(rec["i"], "loss", rec["loss"], "WERF'", rec["WER_Fp"], "WERM", rec["WER_M"], f'{rec["sec"]}s', flush=True)
        return r["loss"]

    cur = np.zeros(NB); best = ev(cur); hist = []
    for sweep in range(MAX_SWEEPS):
        start = best
        for b in range(NB):
            for sgn in (+1, -1):
                moved = False
                while True:
                    cand = cur.copy(); cand[b] += sgn * STEP
                    if abs(cand[b]) > DMAX + 1e-9: break
                    l = ev(cand)
                    if l < best - 1e-9: cur, best, moved = cand, l, True
                    else: break
                if moved: break  # improved in this direction; skip the opposite one
            hist.append({"sweep": sweep, "band_hz": float(FL.BANDS[b]), "delta": cur.tolist(), "loss": best})
        print(f"sweep {sweep} done: loss {start:.3f} -> {best:.3f}", flush=True)
        if best >= start - 1e-9: break
    l0 = logged[tuple([0.0] * NB)]
    DFINAL.write_text(json.dumps({"bands_hz": FL.BANDS.tolist(), "delta_db": cur.tolist(), "loss_final": best, "loss_delta0": l0["loss"],
                                  "WER_Fp_fit_final": logged[tuple(float(v) for v in cur)]["WER_Fp"], "WER_M_fit": l0["WER_M"],
                                  "sweeps_run": sweep + 1, "n_candidates": len(logged), "step_db": STEP, "max_abs_db": DMAX}, indent=1))
    return cur


# ---------------- evaluation ----------------
def do_eval(dec, delta):
    sp = do_split(); outdir = P0 / "conds/geng_recfit"; rows = []; hyp = {}
    SM_ = speech_manifest()
    for snr in (0, -10):
        for s in sp["fit"] + sp["held"]:
            pid = pid_of(s); out = outdir / pid.replace("|", "_") / f"Frec_thr6_post_snr{snr:+d}.flac"
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True); sf.write(str(out) + ".tmp.flac", make_signal(s, delta, snr), 16000, subtype="PCM_16")
                os.replace(str(out) + ".tmp.flac", out)
            ref = SM_[pid + "|mask_off"]["ref_scoring"]; x = sf.read(out, dtype="float32")[0]
            hyp[(snr, s)] = dec.decode("file|" + hashlib.sha1(f"{out}|{[round(float(v), 3) for v in delta]}".encode()).hexdigest(), lambda: x)
            rows.append({"audio_filepath": str(out), "text": ref, "dataset": "geng", "channel": "mono", "speaker": s, "mask": "surgical", "pair_id": pid,
                         "condition": "Frec", "snr_db": snr, "filter": "recognition-fitted fwd", "snr_convention": "post", "threshold_db": 6,
                         "primary": True, "duration": round(len(x) / 16000, 3), "split": "fit" if s in sp["fit"] else "held"})
    mf = P0 / "manifests/geng_recfit.jsonl"; mf.write_text("".join(json.dumps(r) + "\n" for r in rows))
    hf = RES / "hyps_base_manifests-geng_recfit_look6.jsonl"
    with open(hf, "w") as fh:
        for r in rows:
            h = hyp[(r["snr_db"], r["speaker"])]; ref = score_norm(r["text"]); o = jiwer.process_words(ref, h) if h else None
            fh.write(json.dumps({**r, "pred_text": h, "ref_norm": ref, "n_ref": len(ref.split()), "S": o.substitutions if o else 0,
                                 "D": o.deletions if o else len(ref.split()), "I": o.insertions if o else 0}) + "\n")
    return hyp


def boot(speakers, per, B=1000):
    """per[cond][spk]=(E,N). Returns point estimates and 95% talker-bootstrap CI (seed 0)."""
    spk = np.array(sorted(speakers)); rng = np.random.default_rng(0)

    def stat(s):
        w = {c: 100 * sum(per[c][x][0] for x in s) / sum(per[c][x][1] for x in s) for c in per}
        gap = w["M"] - w["U"]
        return [w["U"], w["M"], w["F"], w["Fp"], gap, (w["M"] - w["F"]) / gap if gap else np.nan, (w["M"] - w["Fp"]) / gap if gap else np.nan]
    pt = stat(spk); bs = np.array([stat(rng.choice(spk, len(spk))) for _ in range(B)])
    ci = np.nanpercentile(bs, [2.5, 97.5], axis=0)
    return pt, ci


def ltas_diag(delta, spks):
    setup(); res = []
    for s in spks:
        u = U_audio(s); g = GAINS[pid_of(s)]; Pu = FL.ltas(u)[0]
        fo = filt(u, FL.fwd_inv(g)[0]); fp = filt(u, fprime_fir(s, delta))  # un-normalised: level carried by the filter
        do, dp = FL.band_db(FL.ltas(fo)[0], Pu), FL.band_db(FL.ltas(fp)[0], Pu)
        res.append({"speaker": s, "oracle_abs": np.abs(do - g).mean(), "recfit_abs": np.abs(dp - g).mean(),
                    "oracle_shape": np.abs((do - do.mean()) - (g - g.mean())).mean(), "recfit_shape": np.abs((dp - dp.mean()) - (g - g.mean())).mean(),
                    "recfit_vs_oracle_shift_db": float(np.mean(dp - do)), "Mspec_level_db": float(g.mean())})
    return res


def do_report(dec, delta):
    sp = do_split(); hyp = do_eval(dec, delta); rows = []
    for snr in (0, -10):
        H = load_hyps(snr)
        for half in ("held", "fit"):
            sps = sp[half]; per = {c: {} for c in ("U", "M", "F", "Fp")}
            Mh = {}
            for s in sps:
                for c in ("U", "M", "F"): per[c][s] = err(H[(c, s)]["text"], H[(c, s)]["pred_text"])
                per["Fp"][s] = err(H[("M", s)]["text"], hyp[(snr, s)])
            pt, ci = boot(sps, per)
            names = ["WER_U", "WER_M", "WER_F_spectral", "WER_Fp_recfit", "gap", "R_fwd_spectral", "R_fwd_recfit"]
            ed = lambda c: float(np.mean([100 * (lambda o: o.substitutions + o.deletions + o.insertions)(jiwer.process_words(H[("M", s)]["pred_text"], c(s))) / len(H[("M", s)]["pred_text"].split()) for s in sps]))
            row = {"split": half, "in_sample": half == "fit", "snr_db": snr, "n_talkers": len(sps)}
            for i, n in enumerate(names): row[n] = round(float(pt[i]), 3); row[n + "_lo"] = round(float(ci[0, i]), 3); row[n + "_hi"] = round(float(ci[1, i]), 3)
            row["edit_to_M_hyp_spectral_pct"] = round(ed(lambda s: H[("F", s)]["pred_text"]), 2); row["edit_to_M_hyp_recfit_pct"] = round(ed(lambda s: hyp[(snr, s)]), 2)
            row["edit_to_U_hyp_recfit_pct"] = round(float(np.mean([100 * (lambda o: o.substitutions + o.deletions + o.insertions)(jiwer.process_words(H[("U", s)]["pred_text"], hyp[(snr, s)])) / len(H[("U", s)]["pred_text"].split()) for s in sps])), 2)
            rows.append(row)
    with open(OUT / "recfit_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
    dg = ltas_diag(delta, sp["fit"] + sp["held"])
    summ = {h: {k: round(float(np.mean([d[k] for d in dg if (d["speaker"] in sp[h])])), 3) for k in dg[0] if k != "speaker"} for h in ("fit", "held")}
    (OUT / "ltas_diagnostic.json").write_text(json.dumps({"per_talker": dg, "mean": summ, "delta_db": list(map(float, delta))}, indent=1, default=float))
    for r in rows: print({k: r[k] for k in ("split", "snr_db", "WER_U", "WER_M", "WER_F_spectral", "WER_Fp_recfit", "gap", "R_fwd_spectral", "R_fwd_recfit")})
    print(summ)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3.6)); ax.plot(FL.BANDS, delta, "o-"); ax.axhline(0, color="k", lw=.5); ax.set_xscale("log")
        ax.set_xlabel("Third-octave band centre (Hz)"); ax.set_ylabel("Shared correction Δ(b) (dB)"); ax.set_title("Recognition-fitted correction to the spectral oracle")
        ax.set_xticks([100, 200, 500, 1000, 2000, 5000]); ax.set_xticklabels(["100", "200", "500", "1k", "2k", "5k"]); fig.tight_layout(); fig.savefig(OUT / "recfit_delta.png", dpi=200)
    except ImportError: print("matplotlib missing: figure skipped")


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage == "split": print(do_split()); return
    dec = Decoder()
    delta = do_search(dec) if stage in ("search", "all") else np.array(json.loads(DFINAL.read_text())["delta_db"])
    if stage in ("eval", "report", "all"): do_report(dec, delta)


if __name__ == "__main__": main()
