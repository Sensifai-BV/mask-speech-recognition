"""Listening check of the 17 flagged reference runs (Exp 4): cut clips + decisions sheet.

    python src/muffled_asr/oracle_filter/listening_kit.py [reference_audit.csv]     (default: datasets/p0/reference_audit.csv)

Writes datasets/p0/listening/clips/NN_<file>_<channel>.wav (original sample rate, run +/- 3 s context, -1 dBFS peak)
and datasets/p0/listening/decisions.csv. Fill in `verdict` for every row with one of
    ref_error   - the talker really said something else; give the words actually spoken in `spoken_words`
    asr_error   - the talker read the reference; the hypothesis is a recognition error (no change)
    unclear     - cannot decide (treated as asr_error, i.e. no change, and listed in the paper)
then run src/muffled_asr/oracle_filter/apply_listening.py.

Clip timing: Geng from the TextGrid word intervals (datasets/p0/geng_word_intervals.csv; word index = ref_pos,
re-anchored to the nearest matching word within +/-6 words). TextGrid end times are wrong in >= 5 files, so each row
also gives clip start/end in the full file for scrubbing. Cox from the proportional position of ref_pos in the
passage (no alignment exists); the clip is widened to +/- 5 s for that reason.
"""
import csv, sys
from pathlib import Path
import numpy as np, soundfile as sf
sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, P0, RAINBOW_READ, score_norm

GENG_REF = next(r["ref_scoring"] for r in csv.DictReader(open(P0 / "speech_manifest.csv", newline="")) if r["dataset"] == "geng")
OUT = P0 / "listening"


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else P0 / "reference_audit.csv"
    rows = [r for r in csv.DictReader(open(src, newline="")) if (r.get("flag") or "").startswith("LISTEN")]
    words = {}
    for w in csv.DictReader(open(P0 / "geng_word_intervals.csv", newline="")):
        words.setdefault(w["file"], []).append((float(w["start"]), float(w["end"]), w["word"].lower()))
    (OUT / "clips").mkdir(parents=True, exist_ok=True); sheet = []
    for k, r in enumerate(rows, 1):
        pos = int(float(r["ref_pos"])); nref = len((r["ref_words"] or "").split()); nhyp = len(r["hyp_words"].split())
        if r["dataset"] == "geng":
            path = ROOT / "datasets/original/geng2023" / f"{r['file']}.wav"; x, sr = sf.read(path, dtype="float32", always_2d=True)
            x = x[:, 0]; ref = GENG_REF.split(); tg = words[r["file"]]
            want = ref[min(pos, len(ref) - 1)]
            cand = [i for i in range(max(0, pos - 6), min(len(tg), pos + 7)) if score_norm(tg[i][2]) == want]
            i0 = min(cand, key=lambda i: abs(i - pos)) if cand else min(pos, len(tg) - 1)
            t0 = tg[max(0, i0 - 1)][0]; t1 = tg[min(len(tg) - 1, i0 + max(nref, 1))][1]
            t1 = max(t1, t0 + 0.4 * max(nhyp, nref))  # insertions: cover roughly the inserted words
            ctx, how = 3.0, "TextGrid"
        else:
            path = ROOT / "datasets/original/cox-2022/human" / f"{r['file']}_trim.wav"
            x, sr = sf.read(path, dtype="float32", always_2d=True); x = x[:, {"cheek": 0, "far": 1}[r["channel"]]]
            n = len(score_norm(RAINBOW_READ).split()); dur = len(x) / sr
            t0 = dur * pos / n; t1 = t0 + 0.4 * max(nhyp, nref); ctx, how = 5.0, "proportional"
        a, b = max(0.0, t0 - ctx), min(len(x) / sr, t1 + ctx)
        y = x[int(a * sr):int(b * sr)]; y = y * (10 ** (-1 / 20) / (np.abs(y).max() + 1e-12))
        name = f"{k:02d}_{r['file']}_{r['channel']}.wav"; sf.write(OUT / "clips" / name, y, sr)
        sheet.append({"id": k, "clip": name, "dataset": r["dataset"], "file": r["file"], "channel": r["channel"],
                      "type": r["type"], "ref_pos": pos, "ref_words": r["ref_words"], "hyp_words": r["hyp_words"],
                      "clip_start_s": round(a, 2), "clip_end_s": round(b, 2), "timing": how,
                      "verdict": "", "spoken_words": "", "notes": ""})
    with open(OUT / "decisions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(sheet[0])); w.writeheader(); w.writerows(sheet)
    print(f"{len(sheet)} clips -> {OUT}")


if __name__ == "__main__": main()
