"""P0 preparation: 16 kHz copies, speech manifest, Geng word intervals, Noise-W test excerpts.

Originals are never modified. Cox-R is absent (Figshare download blocked); add
datasets/original/cox-2022/repeated_reading_rainbow_passage.zip and extend this script.
"""
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np, soundfile as sf
sys.path.insert(0, str(Path(__file__).parent))
from common import P0, ROOT, RAINBOW, lag_ms, norm_text, peak_norm, read_binary_textgrid, to16k
from muffled_asr.simulation.mask_sim import load_16k_mono

D16 = P0 / "data16k"; D16.mkdir(parents=True, exist_ok=True)
rows, chan_log, words = [], [], []
# Cox-H: 4 talkers x designs A-E x on/off; cheek (index 0) primary, 1 m (index 1) robustness
for f in sorted((ROOT / "datasets/original/cox-2022/human").glob("*_P0[1-4]_trim.wav")):
    design, state, spk, _ = f.stem.split("_"); x, sr = sf.read(f, dtype="float32", always_2d=True)
    lag = lag_ms(x, sr)
    if not 1.5 <= lag <= 4.0: raise SystemExit(f"{f.name}: lag {lag:.2f} ms; channel order unclear")
    chan_log.append({"file": f.name, "lag_ms_index1_behind_index0": round(lag, 3), "cheek": 0, "far_1m": 1})
    for ch, label in ((0, "cheek"), (1, "far")):
        out = D16 / "coxh" / label / f"{design}_{state}_{spk}.wav"; out.parent.mkdir(parents=True, exist_ok=True)
        y = peak_norm(to16k(x[:, ch], sr)); sf.write(out, y, 16000)
        rows.append({"dataset": "coxh", "channel": label, "speaker_id": spk, "mask_type": design,
                     "condition": "mask_on" if state == "mask-on" else "mask_off", "path": str(out),
                     "duration": round(len(y) / 16000, 3), "text": norm_text(RAINBOW), "source": f.name})
# Geng: 30 talkers, surgical mask on/off, North Wind; TextGrid word tier = reference
for f in sorted((ROOT / "datasets/original/geng2023").glob("S??_?_E_*_01.wav")):
    spk, sex, _, state, _ = f.stem.split("_"); x, sr = sf.read(f, dtype="float32")
    out = D16 / "geng" / f"{f.stem}.wav"; out.parent.mkdir(parents=True, exist_ok=True)
    y = peak_norm(to16k(x, sr)); sf.write(out, y, 16000)
    tg = read_binary_textgrid(f.with_suffix(".TextGrid"))
    w = [(a, b, t) for a, b, t in tg["words"] if t.strip()]
    words += [{"file": f.stem, "speaker": spk, "condition": state, "start": round(a, 4), "end": round(b, 4), "word": t} for a, b, t in w]
    rows.append({"dataset": "geng", "channel": "mono", "speaker_id": spk, "sex": sex, "mask_type": "surgical",
                 "condition": "mask_on" if state == "MSK" else "mask_off", "path": str(out),
                 "duration": round(len(y) / 16000, 3), "text": norm_text(" ".join(t for *_, t in w)),
                 "speech_start": round(w[0][0], 3), "speech_end": round(w[-1][1], 3), "source": f.name})
keys = sorted({k for r in rows for k in r})
with open(P0 / "speech_manifest.csv", "w", newline="") as fh:
    wr = csv.DictWriter(fh, keys); wr.writeheader(); wr.writerows(rows)
with open(P0 / "geng_word_intervals.csv", "w", newline="") as fh:
    wr = csv.DictWriter(fh, list(words[0])); wr.writeheader(); wr.writerows(words)
(P0 / "coxh_channel_log.json").write_text(json.dumps({"rule": "index 1 lags index 0 by 2.3-2.8 ms => index 0 = cheek, index 1 = 1 m (level is NOT a valid rule: cheek is quieter mask-off, louder mask-on)", "files": chan_log}, indent=1))
# Noise-W: 30 s excerpts from test folders only (same split as Study 2)
prov = json.loads((ROOT / "datasets/calibrated/librispeech-noise-control/provenance.json").read_text())
nroot = Path(prov["noise_root"]); test = sorted(g for g, s in prov["noise_groups"].items() if s == "test")
ex = []
for g in test:
    for q in sorted((nroot / g).rglob("*.flac")):
        n = load_16k_mono(q)
        if len(n) < 30 * 16000: continue
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(str(q.relative_to(nroot)).encode()).digest()[:8], "little"))
        o = int(rng.integers(0, len(n) - 30 * 16000 + 1)); out = D16 / "noise_w_test" / f"{g}__{q.stem}.wav"
        out.parent.mkdir(parents=True, exist_ok=True); sf.write(out, n[o:o + 30 * 16000].astype(np.float32), 16000)
        ex.append({"path": str(out), "folder": g, "source": str(q.relative_to(nroot)), "offset": o}); break  # one excerpt per folder
(P0 / "noise_w_test.json").write_text(json.dumps(ex, indent=1))
print({"coxh_files": sum(r["dataset"] == "coxh" for r in rows), "geng_files": sum(r["dataset"] == "geng" for r in rows),
       "geng_words": len(words), "noise_excerpts": len(ex), "noise_test_folders": len(test)})
