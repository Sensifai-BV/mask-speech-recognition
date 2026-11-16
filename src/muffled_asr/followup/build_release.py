"""Tasks 2 + 5: stage the coauthor / GitHub release package (no new decoding).

    python src/muffled_asr/followup/build_release.py [name]
-> release/<name>/ (folder) and release/<name>.zip   (default name p0_followup)

Copies transcripts (every ASR output), condition lists (manifests), filters, WER tables, and the scripts that build
conditions, compute WER and the bootstrap CIs. Absolute paths of this machine inside JSON/JSONL files are rewritten
relative to the repository root. Audio (speech corpora, welding noise) is NOT included; see README section 'Not included'.
The recognition-fitted filter search decodes (results/p0/recfit/decode_cache.jsonl, keyed by sha1) are also decoded
into a readable table: transcripts/recfit/recfit_search_transcripts.jsonl.
"""
import hashlib, json, shutil, sys, zipfile
from pathlib import Path
from muffled_asr.paths import ROOT as R
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "oracle_filter"))
DST = R / "release" / (sys.argv[1] if len(sys.argv) > 1 else "p0_followup"); ABS = str(R) + "/"

FILES = {  # destination subdir -> list of repo-relative globs
 "transcripts/main": ["results/p0/hyps_base_manifests-geng_look6.jsonl", "results/p0/hyps_base_manifests-geng_lowsnr_look6.jsonl",
                      "results/p0/hyps_base_manifests-coxh_cheek_look6.jsonl", "results/p0/hyps_base_manifests-coxh_far_look6.jsonl",
                      "results/p0/hyps_base_manifests-context_subset_look0.jsonl", "results/p0/hyps_base_manifests-context_subset_look13.jsonl",
                      "results/p0/hyps_base_manifests-coxh_abs_look6.jsonl", "results/p0/quiet/hyps_*.jsonl"],
 "transcripts/coxr": ["results/p0/hyps_base_manifests-coxr_look6.jsonl"],
 "transcripts/recfit": ["results/p0/hyps_base_manifests-geng_recfit_look6.jsonl", "results/p0/recfit/decode_cache.jsonl",
                        "results/p0/recfit/search_log.jsonl", "results/p0/recfit/split.json", "results/p0/recfit/delta_final.json"],
 "transcripts/adapters": ["results/human_quiet/hyps_*.jsonl", "results/human_noisy/hyps_*.jsonl"],
 "conditions": ["datasets/p0/manifests/*.jsonl", "datasets/p0/speech_manifest.csv", "datasets/p0/coxr/segments.json",
                "datasets/p0/coxr/pair_gains.json", "datasets/p0/noise_w_test.json", "datasets/p0/geng_crop_log.csv",
                "datasets/p0/coxh_channel_log.json", "datasets/p0/coxh_abs_levels.json", "datasets/p0/reference_audit.csv",
                "datasets/p0/reference_overrides.json", "datasets/p0/listening/decisions.csv"],
 "filters": ["datasets/p0/filters.npz", "datasets/p0/filters_meta.json", "datasets/p0/band_gains.csv"],
 "tables": ["results/p0/tables/*.csv", "results/p0/tables/*.json", "results/p0/tables/*.txt", "results/p0/recfit/recfit_results.csv",
            "results/p0/recfit/ltas_diagnostic.json", "results/p0/scores.csv", "results/p0/quiet/scores.csv",
            "results/human_quiet/scores.csv", "results/human_noisy/scores.csv", "results/followup/*.csv", "results/followup/coauthor_answers.md"],
 "tables/pre_listening": ["results/p0/tables/pre_listening/*.csv"],
 "code/oracle_filter": ["src/muffled_asr/oracle_filter/*.py"], "code/followup": ["src/muffled_asr/followup/*.py"],
 "code/evaluation": ["src/muffled_asr/evaluation/eval_hf.py"],
 "code/measurement": ["src/muffled_asr/measurement/acoustic_levels.py", "src/muffled_asr/measurement/build_cox_human_eval.py",
                         "src/muffled_asr/measurement/build_cox_human_noisy.py"],
 "code/simulation": ["src/muffled_asr/simulation/mask_sim.py"],
 "model_config": ["models/nemotron-speech-streaming-en-0.6b/config.json", "models/nemotron-speech-streaming-en-0.6b/processor_config.json"],
}


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix in (".json", ".jsonl", ".csv", ".md", ".txt"):
        dst.write_text(src.read_text().replace(ABS, ""))
    else: shutil.copy2(src, dst)


def recfit_table(out):
    from recfit import ckey
    sp = json.loads((R / "results/p0/recfit/split.json").read_text())
    cache = {}
    for l in open(R / "results/p0/recfit/decode_cache.jsonl"):
        try: r = json.loads(l); cache[r["key"]] = r["hyp"]
        except Exception: pass
    rows, used = [], set()
    for l in open(R / "results/p0/recfit/search_log.jsonl"):
        c = json.loads(l)
        for s in sp["fit"]:
            k = ckey(s, c["delta"], 0)
            if k in cache: rows.append({"stage": "search", "candidate": c["i"], "speaker": s, "snr_db": 0, "delta_db": c["delta"], "pred_text": cache[k]}); used.add(k)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return len(rows), len(cache) - len(used)


def main():
    if DST.exists(): sys.exit(f"{DST} exists; move it aside first (this script never deletes)")
    n = 0
    for sub, globs in FILES.items():
        for g in globs:
            hits = sorted(R.glob(g))
            if not hits: print("missing:", g)
            for f in hits:
                if "partial" in f.name: continue
                name = f.name if f.name != "scores.csv" else f"scores_{f.parent.relative_to(R).as_posix().replace('/', '_')}.csv"
                if (DST / sub / name).exists(): sys.exit(f"name collision: {sub}/{name}")
                copy(f, DST / sub / name); n += 1
    k, rest = recfit_table(DST / "transcripts/recfit/recfit_search_transcripts.jsonl")
    print(f"recfit: {k} search decodes mapped to (candidate, speaker); {rest} cache entries are the held-out/eval decodes "
          "(in hyps_base_manifests-geng_recfit_look6.jsonl)")
    shutil.copy2(R / "results/followup/README_release.md", DST / "README.md")
    inv = []
    for f in sorted(p for p in DST.rglob("*") if p.is_file()):
        b = f.read_bytes(); inv.append(f"{hashlib.sha256(b).hexdigest()}  {f.relative_to(DST)}")
    (DST / "SHA256SUMS").write_text("\n".join(inv) + "\n")
    z = DST.with_suffix(".zip")
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(p for p in DST.rglob("*") if p.is_file()): zf.write(f, Path(DST.name) / f.relative_to(DST))
    print(n, "files copied;", len(inv), "in package;", z, round(z.stat().st_size / 1e6, 1), "MB")


if __name__ == "__main__": main()
