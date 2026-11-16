"""Apply the listening-check verdicts (datasets/p0/listening/decisions.csv) and rescore every P0 table.

    python src/muffled_asr/oracle_filter/apply_listening.py            # write overrides, rescore, print Table I diff
    python src/muffled_asr/oracle_filter/apply_listening.py --dry-run  # show the corrected references only

verdict = ref_error  -> the source recording's reference is edited: the `ref_words` run at `ref_pos` (0-based word
                        index into that file's scoring reference) is replaced by `spoken_words`; for insertion rows
                        (`ref_words` empty) `spoken_words` is inserted at `ref_pos`.
verdict = asr_error / unclear / empty -> no change.
The edit applies to every condition built from that recording (mask-off -> U, F, Fx, Ri, Fir; mask-on -> M, R), via
score.apply_reference_overrides, so every scorer that uses score.load (score.py, score_lowsnr.py, recfit, the _abs
block) picks it up. Tables are snapshotted to results/p0/tables/pre_listening/ before rescoring and diffed after.
"""
import csv, json, shutil, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import P0, ROOT, score_norm
from score import OVERRIDES

import os
SHEET = Path(os.environ.get("P0_LISTEN_SHEET", P0 / "listening/decisions.csv")); TAB = ROOT / "results/p0/tables"
SCORERS = ["src/muffled_asr/oracle_filter/score.py", "src/muffled_asr/oracle_filter/score_lowsnr.py", "src/muffled_asr/oracle_filter/score_abs.py"]


def build():
    man = list(csv.DictReader(open(P0 / "speech_manifest.csv", newline="")))
    dec = [r for r in csv.DictReader(open(SHEET, newline=""))]
    bad = [r["id"] for r in dec if r["verdict"].strip() not in ("", "ref_error", "asr_error", "unclear")]
    if bad: sys.exit(f"unknown verdict in rows {bad}; use ref_error / asr_error / unclear")
    edits = {}
    for r in dec:
        if r["verdict"].strip() != "ref_error": continue
        if not r["spoken_words"].strip() and r["ref_words"].strip() == "": sys.exit(f"row {r['id']}: ref_error insertion needs spoken_words")
        stem = r["file"] if r["dataset"] == "geng" else r["file"] + "_trim"
        hits = [m for m in man if Path(m["source"]).stem == stem and m["channel"] == r["channel"]]
        if len(hits) != 1: sys.exit(f"row {r['id']}: {len(hits)} manifest rows for {stem}/{r['channel']}")
        m = hits[0]; key = f'{m["dataset"]}|{m["channel"]}|{m["speaker_id"]}|{m["mask_type"]}|{m["condition"]}'
        edits.setdefault(key, {"ref": m["ref_scoring"], "ops": []})["ops"].append(
            (int(r["ref_pos"]), len(score_norm(r["ref_words"]).split()), score_norm(r["spoken_words"]).split(), r["id"]))
    refs, log = {}, []
    for key, e in edits.items():
        w = score_norm(e["ref"]).split()
        for pos, n, new, rid in sorted(e["ops"], reverse=True):
            log.append({"row": rid, "key": key, "old": " ".join(w[pos:pos + n]), "new": " ".join(new)}); w[pos:pos + n] = new
        refs[key] = " ".join(w)
    # Cox cheek and 1 m rows are the same recording: their verdicts must agree
    cox = {}
    for r in dec:
        if r["dataset"] == "coxh": cox.setdefault((r["file"], r["ref_pos"]), set()).add((r["verdict"].strip(), score_norm(r["spoken_words"])))
    for f, v in cox.items():
        if len(v) > 1: sys.exit(f"{f}: cheek and 1 m verdicts differ for the same recording")
    return refs, log, dec


def main():
    refs, log, dec = build()
    n = {v: sum(r["verdict"].strip() == v for r in dec) for v in ("ref_error", "asr_error", "unclear", "")}
    print(f"{len(dec)} rows: {n['ref_error']} ref_error, {n['asr_error']} asr_error, {n['unclear']} unclear, {n['']} not yet decided")
    for l in log: print(f"  row {l['row']:>2} {l['key']}: '{l['old']}' -> '{l['new']}'")
    if "--dry-run" in sys.argv: return
    snap = TAB / "pre_listening"; snap.mkdir(exist_ok=True)
    for f in TAB.glob("*.csv"): shutil.copy2(f, snap / f.name)
    OVERRIDES.write_text(json.dumps({"source": str(SHEET), "edits": log, "references": refs}, indent=1))
    for s in SCORERS:
        if (ROOT / s).exists(): subprocess.run([sys.executable, s], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    import pandas as pd
    changed = 0
    for f in sorted(snap.glob("table_P0-*.csv")):
        new = TAB / f.name
        if not new.exists(): continue
        a, b = pd.read_csv(f).astype(str), pd.read_csv(new).astype(str)
        if a.shape != b.shape: print(f"{f.name}: shape {a.shape} -> {b.shape}"); changed += 1; continue
        diff = (a != b)
        for i, j in zip(*diff.values.nonzero()):
            changed += 1; print(f"{f.name} row {i} {a.columns[j]}: {a.iat[i, j]} -> {b.iat[i, j]}")
    print("no table changed" if not changed else f"{changed} cells changed")


if __name__ == "__main__": main()
