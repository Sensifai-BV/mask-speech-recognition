"""Score base + exported adapters on NeMo-style manifests with the Transformers model (MPS/CPU).

    python src/muffled_asr/evaluation/eval_hf.py --manifests a.json b.json --models base models/cox-empirical \
        --lookahead 6 --out results/human_quiet

Writes <out>/hyps_<model>_<manifest-tag>_look<k>.jsonl and appends pooled WER rows to <out>/scores.csv.
WER normalisation: lowercase, keep a-z, apostrophe, space (same as eval_adapter.py).
"""
import argparse, csv, json, re, sys, time
from pathlib import Path
import numpy as np, soundfile as sf, torch, jiwer
from muffled_asr.paths import ROOT
from transformers import AutoModelForRNNT, AutoProcessor
from muffled_asr.inference import adapter as ad

BASE = ROOT / "models/nemotron-speech-streaming-en-0.6b"
norm = lambda s: " ".join(re.sub(r"[^a-z' ]", " ", s.lower()).split())

def load(model_dir, dev):
    proc = AutoProcessor.from_pretrained(BASE); m = AutoModelForRNNT.from_pretrained(BASE)
    if model_dir != "base": ad.attach(m, ROOT / model_dir)
    return proc, m.to(dev).eval()

def main():
    p = argparse.ArgumentParser(); p.add_argument("--manifests", nargs="+", type=Path, required=True)
    p.add_argument("--models", nargs="+", default=["base"]); p.add_argument("--lookahead", type=int, nargs="+", default=[6])
    p.add_argument("--out", type=Path, required=True); a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    for md in a.models:
        proc, model = load(md, dev); tag_m = Path(md).name
        for look in a.lookahead:
            proc.set_num_lookahead_tokens(look)
            for man in a.manifests:
                rows = [json.loads(l) for l in man.read_text().splitlines() if l.strip()]
                tag = f"{man.parent.name}-{man.stem}"; out = []; t0 = time.time()
                part = a.out / f"hyps_{tag_m}_{tag}_look{look}.partial.jsonl"  # resume after a crash
                done = {json.loads(l)["audio_filepath"]: json.loads(l) for l in part.read_text().splitlines()} if part.exists() else {}
                fp = open(part, "a")
                for r in rows:
                    if r["audio_filepath"] in done:
                        out.append(done[r["audio_filepath"]]); continue
                    x, sr = sf.read(r["audio_filepath"], dtype="float32")
                    inp = proc(x, sampling_rate=sr, return_tensors="pt").to(dev, dtype=model.dtype)
                    with torch.inference_mode(): o = model.generate(**inp)
                    seq = o.sequences if hasattr(o, "sequences") else o
                    hyp = norm(proc.batch_decode(seq, skip_special_tokens=True)[0]); ref = norm(r["text"])
                    w = jiwer.process_words(ref, hyp)
                    out.append({**r, "pred_text": hyp, "ref_norm": ref, "n_ref": len(ref.split()),
                                "S": w.substitutions, "D": w.deletions, "I": w.insertions})
                    fp.write(json.dumps(out[-1]) + "\n"); fp.flush()
                    del inp, o, seq
                    if dev == "mps": torch.mps.empty_cache()  # varying lengths otherwise grow the cache until the OS kills us
                fp.close()
                (a.out / f"hyps_{tag_m}_{tag}_look{look}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
                e = sum(r["S"] + r["D"] + r["I"] for r in out); n = sum(r["n_ref"] for r in out)
                new = not (a.out / "scores.csv").exists()
                with open(a.out / "scores.csv", "a", newline="") as f:
                    wtr = csv.writer(f); new and wtr.writerow(["model", "manifest", "lookahead", "n_utts", "wer_pct"])
                    wtr.writerow([tag_m, tag, look, len(out), round(100 * e / n, 3)])
                print(f"{tag_m:18s} {tag:34s} look{look} WER {100*e/n:5.2f}%  ({time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__": main()
