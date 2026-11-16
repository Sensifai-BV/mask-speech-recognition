"""Rebuild the measured-covering corpora with literature-based workshop SNRs.

Differences from build_calibrated_training_set.py (which it otherwise mirrors: same seed,
speaker split, noise-folder split and condition assignment per utterance):
1. SNR is A-weighted active speech level of the UNCOVERED utterance at the microphone vs
   A-weighted noise Leq (acoustic_levels.py: scene noise level, Lombard talker level,
   distance). The covering keeps its insertion loss (pre-covering reference).
2. All three arms (empirical, bounded_pca, noise_control) are written in one pass with
   identical noise crops and gains, so they are paired by construction.
3. --holdout-designs moves whole training designs (default: full_visor) to the test split,
   giving a held-out shield in the synthetic test.
4. Output level: the final mix is set to a random RMS of -26..-18 dBFS, then peak-limited.

    python src/muffled_asr/measurement/build_realistic_training_set.py \
        datasets/librispeech-pc/train-clean-100.json datasets/librispeech-pc/LibriSpeech/train-clean-100 \
        $WELDING_NOISE_ROOT datasets/realistic
"""
from __future__ import annotations
import argparse, hashlib, json, random, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, soundfile as sf

from muffled_asr.paths import ROOT
from muffled_asr.simulation.mask_sim import SAMPLE_RATE, load_16k_mono  # noqa: E402
from muffled_asr.measurement.covering import CoveringBank  # noqa: E402
from muffled_asr.measurement import acoustic_levels as AL  # noqa: E402

CONDITIONS = ("clean", "covering-only", "noise-only", "covering+noise")
ARMS = ("empirical", "bounded_pca", "noise_control")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("pc_manifest", type=Path); p.add_argument("speech_root", type=Path)
    p.add_argument("noise_root", type=Path); p.add_argument("out_root", type=Path)
    p.add_argument("--bank", type=Path, default=Path("datasets/calibrated/response_bank.json"))
    p.add_argument("--seed", type=int, default=20261007)
    p.add_argument("--dev-speakers", type=int, default=15); p.add_argument("--test-speakers", type=int, default=15)
    p.add_argument("--holdout-designs", nargs="*", default=["full_visor"])
    p.add_argument("--distance", type=float, nargs=2, default=(0.5, 1.5))
    p.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    p.add_argument("--limit", type=int)
    a = p.parse_args()

    rows = [json.loads(l) for l in a.pc_manifest.read_text().splitlines() if l.strip()]
    for r in rows:
        parts = Path(r["audio_filepath"]).parts
        c = a.speech_root.joinpath(*parts[-3:])
        if not c.is_file(): raise SystemExit(f"Missing source audio: {c}")
        r["_audio"] = str(c)
    speakers = sorted({Path(r["_audio"]).parent.parent.name for r in rows})
    rng = random.Random(a.seed); rng.shuffle(speakers)
    nt, nd = a.test_speakers, a.dev_speakers
    split_of = {s: "test" for s in speakers[:nt]}
    split_of.update({s: "dev" for s in speakers[nt:nt + nd]}); split_of.update({s: "train" for s in speakers[nt + nd:]})

    noise_paths = sorted(a.noise_root.rglob("*.flac"))
    groups = sorted({q.relative_to(a.noise_root).parts[0] for q in noise_paths}); rng.shuffle(groups)
    k = max(1, round(.15 * len(groups)))
    nsplit = {g: "test" for g in groups[:k]}; nsplit.update({g: "dev" for g in groups[k:2 * k]})
    nsplit.update({g: "train" for g in groups[2 * k:]})
    noise_by = defaultdict(list)
    for q in noise_paths: noise_by[nsplit[q.relative_to(a.noise_root).parts[0]]].append(q)

    bank = CoveringBank(a.bank)
    for d in a.holdout_designs:  # whole design -> test; sampler never sees it in training
        moved = bank.groups.pop(("train", d), None)
        if moved is None: raise SystemExit(f"--holdout-designs: no training design {d}")
        bank.groups[("test", d)] = moved

    man = defaultdict(list); counts = Counter(); t0 = time.time(); n_done = 0
    for i, row in enumerate(rows):
        if a.limit is not None and n_done >= a.limit: break
        src = Path(row["_audio"]); spk = src.parent.parent.name; split = split_of[spk]
        if split not in a.splits: continue
        audio = load_16k_mono(src)
        if not .5 <= len(audio) / SAMPLE_RATE <= 16: continue

        def rng_for(kind):
            b = hashlib.sha256(f"{a.seed}:{src.stem}:{kind}".encode()).digest()[:8]
            return np.random.default_rng(int.from_bytes(b, "little"))

        cond = CONDITIONS[int(rng_for("condition").integers(len(CONDITIONS)))]
        covered = {"noise_control": audio}; cov_recipe = {}
        if cond in ("covering-only", "covering+noise"):
            for mode in ("empirical", "bounded_pca"):
                g, rec = bank.sample(split, mode, rng_for("covering")); covered[mode] = bank.apply(audio, g); cov_recipe[mode] = rec
        else:
            covered["empirical"] = covered["bounded_pca"] = audio
        speech_ref, active_frac = AL.active_level_a(audio)
        noise_rec, noise_sig = None, None
        if cond in ("noise-only", "covering+noise"):
            nr = rng_for("noise"); lv = AL.sample_levels(nr, tuple(a.distance))
            npath = noise_by[split][int(nr.integers(len(noise_by[split])))]
            noise = load_16k_mono(npath)
            if len(noise) < len(audio): noise = np.tile(noise, int(np.ceil(len(audio) / len(noise))))
            off = int(nr.integers(0, len(noise) - len(audio) + 1)); noise = noise[off:off + len(audio)]
            nl = AL.leq_a(noise)
            gain = 10 ** ((speech_ref - lv["snr_a_db"] - nl) / 20)  # SNR set on the uncovered speech
            noise_sig = (gain * noise).astype(np.float32)
            noise_rec = {"file": str(npath.relative_to(a.noise_root)), "offset": off, **{k_: round(v, 3) if isinstance(v, float) else v for k_, v in lv.items()}}
        target = rng_for("level").uniform(-26, -18)
        for arm in ARMS:
            y = covered[arm] + (noise_sig if noise_sig is not None else 0)
            arm_cond = cond if arm != "noise_control" else ("noise-only" if noise_sig is not None else "clean")
            post = None
            if noise_sig is not None and arm != "noise_control" and cond == "covering+noise":
                post = round(AL.active_level_a(covered[arm])[0] - AL.leq_a(noise_sig), 3)
            rms = np.sqrt(np.mean(y ** 2)) + 1e-12; y = y * (10 ** (target / 20) / rms)
            pk = np.abs(y).max()
            if pk > .99: y = y * (.99 / pk)
            out = a.out_root / arm / split / spk / f"{src.stem}.flac"; out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(out, y.astype(np.float32), SAMPLE_RATE)
            man[(arm, split)].append({"audio_filepath": str(out.resolve()), "duration": round(len(y) / SAMPLE_RATE, 3),
                "text": row["text"], "speaker": spk, "source_id": src.stem, "condition": arm_cond,
                "recipe": {"condition": arm_cond, "covering": cov_recipe.get(arm), "noise": noise_rec,
                           "covering_removed": arm == "noise_control" and bool(cov_recipe),
                           "snr_convention": "A-weighted active level of uncovered speech vs noise Leq",
                           "snr_post_covering_db": post, "speech_active_fraction": round(active_frac, 3),
                           "output_rms_dbfs": round(target, 2)}})
        counts[(split, cond)] += 1; n_done += 1
        if n_done % 1000 == 0: print(f"{n_done} utts {time.time()-t0:.0f}s", flush=True)

    for arm in ARMS:
        for s in a.splits:
            m = man[(arm, s)]; f = a.out_root / arm / f"{s}.json"; f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in m))
        (a.out_root / arm / "provenance.json").write_text(json.dumps({
            "seed": a.seed, "arm": arm, "bank": str(a.bank), "holdout_designs_to_test": a.holdout_designs,
            "test_designs": sorted(d for s_, d in bank.groups if s_ == "test"),
            "train_designs": sorted(d for s_, d in bank.groups if s_ == "train"),
            "level_model": {"scenes": {k_: v[0] for k_, v in AL.SCENES.items()}, "lombard_slope": AL.LOMBARD_SLOPE,
                            "lombard_onset_dba": AL.LOMBARD_ONSET, "quiet_talker_dba_1m": AL.QUIET_LEVEL,
                            "ceiling_dba": AL.CEILING, "talker_sd_db": AL.TALKER_SD, "distance_m": a.distance},
            "speech_speakers": split_of, "noise_groups": nsplit, "noise_root": str(a.noise_root)}, indent=2))
    print({f"{s}/{c}": n for (s, c), n in sorted(counts.items())}, f"{time.time()-t0:.0f}s")


if __name__ == "__main__": main()
