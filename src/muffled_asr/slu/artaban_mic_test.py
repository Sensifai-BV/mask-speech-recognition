"""Real-microphone test with Artaban robot commands: record on the MacBook, score base vs adapters.

    python src/muffled_asr/slu/artaban_mic_test.py commands                 # list the 30 commands
    python src/muffled_asr/slu/artaban_mic_test.py devices                  # list audio devices
    python src/muffled_asr/slu/artaban_mic_test.py record --session s1 --condition quiet
    python src/muffled_asr/slu/artaban_mic_test.py record --session s1 --condition welding-speaker \
        --play-noise welding --noise-level-dbfs -18                             # noise from the speakers
    python src/muffled_asr/slu/artaban_mic_test.py score --session s1 \
        --models base models/<adapter> --add-noise welding knock babble --snr 0 5 10

record  Each command is shown; press Enter, say it; recording stops after --seconds (default 4).
        After each take: Enter = keep, r = record again, p = play back. Takes are appended to
        datasets/real_mic/<session>/manifest.json (NeMo format), so a session can be resumed and
        several conditions (quiet, mask, noise from speakers, distance) collected over time.
        --play-noise plays held-out TEST-split noise (welding / knock / babble) through the
        speakers during each take: real acoustic mixing in your room.
score   Decodes every take with each model (Transformers Nemotron, 560 ms = lookahead 6) and
        reports, per model and condition:
          WER     strict word error rate vs the command text
          exact   whole command transcribed exactly (after normalisation)
          wake    "artaban" detected (fuzzy: 1-3 adjacent words within 2 letter edits,
                  e.g. "art a ban", "arta ban")
          intent  nearest command in the list (word edit distance, wake word normalised) is
                  the one spoken: what the robot would execute
        --add-noise/--snr also mix held-out test noise into your real recordings at A-weighted
        SNRs (A-weighted active speech level vs noise Leq), for controlled noise levels.
        Paired comparison vs the first model (base): relative WER change in %, bootstrap 95 % CI
        over takes. Decodes are cached, so adding takes or models only decodes what is new.
Writes results/real_mic/<session>/{hyps_*.jsonl, scores.csv, report.md, wer_by_model.png}.
Held-out noise = the TEST split recorded in a corpus provenance.json written by
measurement/build_calibrated_training_set.py (default datasets/calibrated/librispeech-noise-control/provenance.json,
override with NOISE_PROVENANCE): welding session folders and LibriSpeech speakers never used in training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from muffled_asr.paths import ROOT
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".numba_cache"))
warnings.filterwarnings("ignore", message="Using the model-agnostic default `max_length`")
SR = 16000
WAKE = "artaban"
WELDING_ROOT = Path(os.environ.get("WELDING_ROOT", os.environ.get("WELDING_NOISE_ROOT", str(ROOT / "datasets/original/intel_robotic_welding_dataset"))))

COMMANDS = [  # (id, text, category)
    ("wake", "hey artaban", "wake"),
    ("wake2", "artaban", "wake"),
    ("come", "artaban come here", "motion"),
    ("follow", "artaban follow me", "motion"),
    ("stop", "artaban stop", "safety"),
    ("estop", "artaban emergency stop", "safety"),
    ("wait", "artaban wait here", "motion"),
    ("back", "artaban go back", "motion"),
    ("left", "artaban turn left", "motion"),
    ("right", "artaban turn right", "motion"),
    ("fwd2", "artaban move forward two meters", "motion"),
    ("slow", "artaban slow down", "motion"),
    ("base", "artaban return to base", "motion"),
    ("charge", "artaban go to the charging station", "motion"),
    ("weld_on", "artaban start welding", "tool"),
    ("weld_off", "artaban stop welding", "tool"),
    ("grip_open", "artaban open the gripper", "tool"),
    ("grip_close", "artaban close the gripper", "tool"),
    ("pick", "artaban pick up the hammer", "tool"),
    ("toolbox", "artaban bring me the toolbox", "tool"),
    ("hold", "artaban hold the plate", "tool"),
    ("lift", "artaban lift the arm", "tool"),
    ("lower", "artaban lower the arm", "tool"),
    ("battery", "artaban what is your battery level", "status"),
    ("status", "artaban report status", "status"),
    ("lights_on", "artaban lights on", "status"),
    ("lights_off", "artaban lights off", "status"),
    ("cancel", "artaban cancel that", "safety"),
    ("yes", "artaban yes confirm", "dialog"),
    ("no", "artaban no", "dialog"),
]


# ----------------------------------------------------------------------------- text metrics
def norm_words(s: str) -> list[str]:
    return re.sub(r"[^a-z' ]", " ", s.lower()).split()


def lev(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] if x == y else 1 + min(prev[j - 1], prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def wake_normalise(words: list[str], max_edits: int = 2) -> tuple[list[str], bool]:
    """Replace the best 1-3 word span that spells ~'artaban' by 'artaban'."""
    best = None
    for n in (1, 2, 3):
        for i in range(len(words) - n + 1):
            d = lev("".join(words[i:i + n]).replace("'", ""), WAKE)
            if d <= max_edits and (best is None or d < best[0]):
                best = (d, i, n)
    if best is None:
        return words, False
    _, i, n = best
    return words[:i] + [WAKE] + words[i + n:], True


def intent_of(words: list[str]) -> str:
    """Nearest command by CHARACTER edit distance on the wake-normalised text, so inflections
    ("stopped building" ~ "stop welding") do not lose to a shorter command on a word-count tie."""
    w = " ".join(wake_normalise(words)[0])
    return min(COMMANDS, key=lambda c: (lev(" ".join(norm_words(c[1])), w) / max(len(c[1]), 1), -len(c[1])))[0]


def strip_name(r: list[str], h: list[str]) -> tuple[list[str], list[str]]:
    """Command words only: drop 'hey'/'artaban' from the reference and the best leading 0-3
    hypothesis words (the name, however it was spelled)."""
    rc = [w for w in r if w not in ("hey", WAKE)]
    lead = 1 + (r[:1] == ["hey"])
    best = min(range(0, lead + 3), key=lambda k: (lev(rc, h[k:]), k))
    return rc, h[best:]


def score_row(ref: str, hyp: str, cmd_id: str) -> dict:
    r, h = norm_words(ref), norm_words(hyp)
    hn, wake = wake_normalise(h); it = intent_of(h); rc, hc = strip_name(r, h)
    return {"n_ref": len(r), "errors": lev(r, h), "errors_wake_norm": lev(r, hn),
            "n_cmd": len(rc), "errors_cmd": lev(rc, hc) if rc else 0, "exact": int(r == h), "wake": int(wake),
            "intent": it, "intent_ok": int(it == cmd_id), "empty": int(not h)}


# ----------------------------------------------------------------------------- noise (held-out test split)
class TestNoise:
    """Held-out TEST-split noise sources of the adapter training corpus (never seen in training)."""

    def __init__(self):
        from muffled_asr.simulation.mask_sim import load_16k_mono
        from muffled_asr.data import noises as NZ
        self.load, self.NZ = load_16k_mono, NZ
        pf = Path(os.environ.get("NOISE_PROVENANCE", ROOT / "datasets/calibrated/librispeech-noise-control/provenance.json"))
        if not pf.exists(): raise SystemExit(f"{pf} not found: build the corpus with measurement/build_calibrated_training_set.py or set NOISE_PROVENANCE")
        prov = json.loads(pf.read_text())
        groups = sorted(g for g, s in prov["noise_groups"].items() if s == "test")
        self.welding = sorted(p for g in groups for p in (WELDING_ROOT / g).rglob("*.flac"))
        if not self.welding: raise SystemExit(f"no test-split welding files under {WELDING_ROOT} (set WELDING_ROOT)")
        self.knock = NZ.esc_knock_events(ROOT / "datasets/original/esc50-knock", {5})
        lib = Path(prov.get("speech_root", ROOT / "datasets/librispeech-pc/LibriSpeech/train-clean-100"))
        if not lib.exists(): lib = ROOT / "datasets/librispeech-pc/LibriSpeech/train-clean-100"
        self.babble = defaultdict(list)
        for s, sp in prov["speech_speakers"].items():
            if sp == "test":
                self.babble[s] = sorted(str(p) for p in (lib / s).rglob("*.flac"))
        self.babble_spk = sorted(s for s in self.babble if self.babble[s])

    def make(self, kind: str, n: int, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
        if kind == "welding":
            f = self.welding[int(rng.integers(len(self.welding)))]
            x, rec = self.NZ.welding_crop(f, n, rng, self.load)
            return x, {"file": str(f.relative_to(WELDING_ROOT)), **rec}
        if kind == "knock":
            return self.NZ.knock_track(n, rng, self.knock)
        k = int(rng.integers(3, 9)); spk = rng.choice(self.babble_spk, size=k, replace=False)
        y = np.zeros(n, np.float32)
        for s in spk:
            x, _ = self.NZ.babble_talker(n, rng, self.babble[s], self.load)
            y += x / (np.sqrt(np.mean(x ** 2)) + 1e-9) * 10 ** (rng.uniform(-3, 3) / 20)
        return y, {"n_talkers": k, "speakers": [str(s) for s in spk]}


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr: float) -> np.ndarray:
    from muffled_asr.measurement import acoustic_levels as AL
    s_lev, _ = AL.active_level_a(speech)
    y = speech + 10 ** ((s_lev - snr - AL.leq_a(noise)) / 20) * noise
    pk = float(np.abs(y).max())
    return (y * (0.99 / pk) if pk > 0.99 else y).astype(np.float32)


# ----------------------------------------------------------------------------- record
class NoisePlayer:
    """Loops a noise buffer on its own OUTPUT stream (no duplex stream: on macOS the built-in mic
    and speakers are separate devices, and a combined play+record stream can stay silent or stall).
    Runs at the output device's native rate; `active` gates the sound, so the device stays open."""

    def __init__(self, noise16k: np.ndarray, rms_dbfs: float, device=None):
        import sounddevice as sd
        from scipy.signal import resample_poly
        info = sd.query_devices(device, "output")
        self.name, self.rate = info["name"], int(info["default_samplerate"])
        g = np.gcd(self.rate, SR)
        x = resample_poly(noise16k, self.rate // g, SR // g).astype(np.float32)
        self.x = x / (np.sqrt(np.mean(x ** 2)) + 1e-12)  # unit RMS; gain applied in the callback
        self.set_level(rms_dbfs); self.pos = 0; self.active = False
        self.stream = sd.OutputStream(samplerate=self.rate, channels=1, dtype="float32", device=device,
                                      callback=self._cb, blocksize=0)
        self.stream.start()

    def set_level(self, rms_dbfs: float):
        self.level = rms_dbfs; self.gain = 10 ** (rms_dbfs / 20)

    def _cb(self, out, frames, t, status):
        if not self.active:
            out.fill(0); return
        idx = (self.pos + np.arange(frames)) % len(self.x); self.pos = int((self.pos + frames) % len(self.x))
        out[:, 0] = np.clip(self.x[idx] * self.gain, -0.98, 0.98)

    def close(self):
        self.active = False; self.stream.stop(); self.stream.close()


def record_take(sd, n: int, device, timeout_s: float) -> np.ndarray | None:
    """sd.rec with a timeout: sd.wait() has none and blocks Ctrl-C if the device stalls."""
    x = sd.rec(n, samplerate=SR, channels=1, dtype="float32", device=device)
    t0 = time.time()
    while sd.get_stream().active:
        if time.time() - t0 > timeout_s:
            sd.stop(); return None
        time.sleep(0.05)
    return x.squeeze().astype(np.float32)


def cmd_record(a) -> None:
    import sounddevice as sd
    sess = a.data_root / a.session; out_dir = sess / a.condition; out_dir.mkdir(parents=True, exist_ok=True)
    man_p = sess / "manifest.json"
    rows = [json.loads(l) for l in man_p.read_text().splitlines() if l.strip()] if man_p.exists() else []
    have = set() if a.overwrite else {(r["condition"], r["take"], r["command_id"]) for r in rows}
    cmds = [c for c in COMMANDS if not a.only or c[0] in a.only]
    print(f"input : {sd.query_devices(a.input_device, 'input')['name']}")
    player = None
    if a.play_noise:
        noise, _ = TestNoise().make(a.play_noise, 60 * SR, np.random.default_rng(7))  # 60 s loop
        player = NoisePlayer(noise, a.noise_level_dbfs, a.output_device)
        print(f"output: {player.name} @ {player.rate} Hz -- held-out {a.play_noise} noise, {player.level:g} dBFS RMS")
        print("Set the Mac volume to a fixed level (e.g. 50 %) and keep it for the whole session.")
        while True:  # level check: the user must actually hear it
            player.active = True; time.sleep(2.0); player.active = False
            ans = input(f"   Heard the noise at {player.level:g} dBFS? Enter = yes, + / - = 6 dB louder / softer, "
                        f"t = play again: ").strip()
            if ans == "+": player.set_level(min(player.level + 6, -3))
            elif ans == "-": player.set_level(player.level - 6)
            elif ans == "": break
        player.active = a.noise_continuous
    print(f"Session '{a.session}', condition '{a.condition}': {len(cmds)} commands x {a.takes} take(s), {a.seconds:g} s each.")
    print("Speak naturally at your usual distance from the MacBook. Ctrl-C stops; rerun the same command to resume.\n")
    n = int(a.seconds * SR)
    try:
        for take in range(1, a.takes + 1):
            for i, (cid, text, cat) in enumerate(cmds, 1):
                if (a.condition, take, cid) in have:
                    continue
                while True:
                    input(f"[take {take}  {i}/{len(cmds)}]  say:  \"{text}\"   (Enter, then speak) ")
                    if player:
                        player.active = True; time.sleep(0.3)  # noise already running when you start talking
                    print("   recording...", end="", flush=True)
                    x = record_take(sd, n, a.input_device, a.seconds + 3)
                    if player and not a.noise_continuous:
                        player.active = False
                    if x is None:
                        print("  ERROR: the input device stalled; try again (or pick --input-device from `devices`)."); continue
                    pk = float(np.abs(x).max()); peak = 20 * np.log10(max(pk, 1e-9)); edge = int(0.25 * SR)
                    rms = 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)
                    warn = []
                    if peak < -40: warn.append("almost silent (mic permission / --input-device?)")
                    if peak > -1: warn.append("clipping: speak softer, lower the noise or move away")
                    if not player and 20 * np.log10(np.abs(x[-edge:]).max() / (pk + 1e-9) + 1e-9) > -12:
                        warn.append("speech may be cut off at the end (use --seconds 5)")
                    print(f" peak {peak:.0f} dBFS, rms {rms:.0f} dBFS" + (f"   WARNING: {'; '.join(warn)}" if warn else ""))
                    ans = input("   Enter = keep, r = redo, p = play: ").strip().lower()
                    while ans == "p":
                        sd.play(x, SR); sd.wait(); ans = input("   Enter = keep, r = redo, p = play: ").strip().lower()
                    if ans != "r":
                        break
                f = out_dir / f"take{take}_{cid}.wav"; sf.write(f, x, SR, subtype="PCM_16")
                row = {"audio_filepath": str(f.resolve()), "duration": round(len(x) / SR, 3), "text": text,
                       "command_id": cid, "category": cat, "condition": a.condition, "take": take,
                       "played_noise": a.play_noise, "noise_level_dbfs_rms": player.level if player else None,
                       "output_device": player.name if player else None, "peak_dbfs": round(float(peak), 1),
                       "rms_dbfs": round(float(rms), 1), "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S")}
                with open(man_p, "a") as fp:
                    fp.write(json.dumps(row) + "\n")
    finally:
        if player: player.close()
    print(f"\nSaved in {sess}. Score with:\n  python src/muffled_asr/slu/artaban_mic_test.py score --session {a.session}")


# ----------------------------------------------------------------------------- score
def decode_all(model_dir: str, items: list[tuple[str, np.ndarray]], lookahead: int) -> dict[str, str]:
    import torch
    from muffled_asr.inference.asr import load_model, pick_device
    dev = pick_device()
    path = ROOT / "models/nemotron-speech-streaming-en-0.6b" if model_dir == "base" else ROOT / model_dir
    proc, model = load_model(path, dev); proc.set_num_lookahead_tokens(lookahead); out = {}
    for key, x in items:
        inp = proc(x, sampling_rate=SR, return_tensors="pt").to(dev, dtype=model.dtype)
        with torch.inference_mode():
            o = model.generate(**inp)
        seq = o.sequences if hasattr(o, "sequences") else o
        out[key] = proc.batch_decode(seq, skip_special_tokens=True)[0].strip()
        if dev.type == "mps": torch.mps.empty_cache()
    del model
    return out


def cmd_score(a) -> None:
    sess = a.data_root / a.session; res = a.results_root / a.session
    res.mkdir(parents=True, exist_ok=True)
    man_p = sess / "manifest.json"
    if not man_p.exists():
        have = sorted(d.name for d in a.data_root.glob("*") if (d / "manifest.json").exists()) if a.data_root.exists() else []
        raise SystemExit(f"No recordings for session '{a.session}' ({man_p} does not exist).\n"
                         f"Record first:\n  python src/muffled_asr/slu/artaban_mic_test.py record --session {a.session} --condition quiet --takes 2\n"
                         f"Sessions with recordings: {', '.join(have) if have else 'none yet'}")
    rows = [json.loads(l) for l in man_p.read_text().splitlines() if l.strip()]
    rows = list({(r["condition"], r["take"], r["command_id"]): r for r in rows}.values())  # latest take wins
    if a.conditions: rows = [r for r in rows if r["condition"] in a.conditions]
    if not rows: raise SystemExit("no recordings for this session / condition")
    items, meta = [], {}
    for r in rows:
        x, _ = sf.read(r["audio_filepath"], dtype="float32", always_2d=True); x = x.mean(1)
        base_key = f"{r['condition']}|{r['take']}|{r['command_id']}"
        items.append((base_key + "|clean", x)); meta[base_key + "|clean"] = {**r, "variant": r["condition"], "added_noise": None, "snr_db": None}
        if a.add_noise:
            tn = getattr(cmd_score, "_tn", None) or TestNoise(); cmd_score._tn = tn
            for kind in a.add_noise:
                for snr in a.snr:
                    seed = int.from_bytes(hashlib.sha256(f"{base_key}:{kind}:{snr}".encode()).digest()[:8], "little")
                    nz, _ = tn.make(kind, len(x), np.random.default_rng(seed))
                    k = f"{base_key}|{kind}{snr:+g}"; y = mix_at_snr(x, nz, snr); items.append((k, y))
                    meta[k] = {**r, "variant": f"{r['condition']} + {kind} {snr:+g} dB", "added_noise": kind, "snr_db": snr}
                    if a.save_mixes:
                        p = res / "mixes" / f"{k.replace('|', '_')}.wav"; p.parent.mkdir(parents=True, exist_ok=True)
                        sf.write(p, y, SR, subtype="PCM_16")
    print(f"{len(rows)} recordings -> {len(items)} decodes per model, models: {', '.join(a.models)}")
    all_rows = []
    for m in a.models:
        tag = Path(m).name; hyp_p = res / f"hyps_{tag}_look{a.lookahead}.jsonl"
        cache = {}
        if hyp_p.exists():
            for l in hyp_p.read_text().splitlines():
                d = json.loads(l); cache[d["key"]] = d
        todo = [(k, x) for k, x in items if k not in cache or cache[k].get("recorded_at") != meta[k].get("recorded_at")]
        t0 = time.time(); hyps = decode_all(m, todo, a.lookahead) if todo else {}
        for k, h in hyps.items():
            cache[k] = {"key": k, "model": tag, **meta[k], "pred_text": h, **score_row(meta[k]["text"], h, meta[k]["command_id"])}
        for k, _ in items:  # metrics are recomputed from the cached transcript
            cache[k].update(meta[k]); cache[k].update(score_row(meta[k]["text"], cache[k]["pred_text"], meta[k]["command_id"]))
        hyp_p.write_text("".join(json.dumps(cache[k]) + "\n" for k, _ in items))
        all_rows += [cache[k] for k, _ in items]
        print(f"  {tag:14s} decoded {len(todo)} new in {time.time()-t0:.0f}s")
    report(all_rows, [Path(m).name for m in a.models], res, a.lookahead)


def agg(rs):
    n = sum(r["n_ref"] for r in rs)
    nc = sum(r["n_cmd"] for r in rs)
    return {"n": len(rs), "wer": 100 * sum(r["errors"] for r in rs) / max(n, 1),
            "wer_wake_norm": 100 * sum(r["errors_wake_norm"] for r in rs) / max(n, 1),
            "cmd_wer": 100 * sum(r["errors_cmd"] for r in rs) / max(nc, 1), "empty": 100 * np.mean([r["empty"] for r in rs]),
            "exact": 100 * np.mean([r["exact"] for r in rs]),
            "wake": 100 * np.mean([r["wake"] for r in rs]), "intent": 100 * np.mean([r["intent_ok"] for r in rs])}


def paired_rel(a_rows, b_rows, n_boot=4000, seed=0):
    """Relative WER change of a vs b in % (negative = a has fewer errors); bootstrap CI over takes."""
    A = {r["key"]: r for r in a_rows}; B = {r["key"]: r for r in b_rows}; keys = sorted(set(A) & set(B))
    ea = np.array([A[k]["errors"] for k in keys], float); eb = np.array([B[k]["errors"] for k in keys], float)
    if eb.sum() == 0: return None
    idx = np.random.default_rng(seed).integers(0, len(keys), (n_boot, len(keys)))
    sb = eb[idx].sum(1); ok = sb > 0
    rel = 100 * (ea[idx].sum(1)[ok] - sb[ok]) / sb[ok]
    return 100 * (ea.sum() - eb.sum()) / eb.sum(), float(np.percentile(rel, 2.5)), float(np.percentile(rel, 97.5))


def report(rows, tags, res: Path, look: int) -> None:
    variants = list(dict.fromkeys(r["variant"] for r in rows))
    by = defaultdict(list)
    for r in rows:
        by[(r["model"], r["variant"])].append(r); by[(r["model"], "ALL")].append(r)
    ref = tags[0]; table = []
    for v in variants + ["ALL"]:
        for t in tags:
            s = agg(by[(t, v)]); rel = paired_rel(by[(t, v)], by[(ref, v)]) if t != ref else None
            table.append({"condition": v, "model": t, "n": s["n"], "wer_pct": round(s["wer"], 2),
                          f"rel_wer_vs_{ref}_pct": round(rel[0], 1) if rel else "", "rel_ci95": f"[{rel[1]:+.1f}, {rel[2]:+.1f}]" if rel else "",
                          "wer_wake_norm_pct": round(s["wer_wake_norm"], 2), "cmd_wer_pct": round(s["cmd_wer"], 2),
                          "empty_pct": round(s["empty"], 1), "exact_pct": round(s["exact"], 1), "wake_pct": round(s["wake"], 1), "intent_pct": round(s["intent"], 1)})
    with open(res / "scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
    L = [f"# Real-microphone Artaban command test — session {res.name}", "",
         f"Lookahead {look} (6 = 560 ms). Strict WER; exact = whole command right; wake = 'artaban' found",
         "(fuzzy, <=2 letter edits); intent = nearest command in the list is the spoken one.",
         f"rel. WER = relative change vs {ref} on the same takes (negative = fewer errors); 95 % CI = bootstrap over takes.", "",
         "Command WER = words after the name (name span removed, however it was spelled); empty = no words decoded.", "",
         f"| condition | model | n | WER % | rel. WER vs {ref} | command WER % | wake % | intent % | empty % |", "|---|---|---|---|---|---|---|---|---|"]
    for t in table:
        rel = f"{t[f'rel_wer_vs_{ref}_pct']:+.1f}% {t['rel_ci95']}" if t["rel_ci95"] else "—"
        L.append(f"| {t['condition']} | {t['model']} | {t['n']} | {t['wer_pct']:.1f} | {rel} | {t['cmd_wer_pct']:.1f} | {t['wake_pct']:.0f} | {t['intent_pct']:.0f} | {t['empty_pct']:.0f} |")
    L += ["", "## Most frequent mistakes", ""]
    for t in tags:
        bad = defaultdict(int)
        for r in by[(t, "ALL")]:
            if not r["exact"]: bad[(r["text"], " ".join(norm_words(r["pred_text"])))] += 1
        top = sorted(bad.items(), key=lambda kv: -kv[1])[:8]
        L.append(f"- **{t}**: " + ("; ".join(f"\"{a}\" → \"{b}\" ×{n}" for (a, b), n in top) if top else "none"))
    (res / "report.md").write_text("\n".join(L) + "\n")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
        fig, ax = plt.subplots(figsize=(max(4.5, 0.9 * len(variants) + 2.5), 3.0), layout="constrained")
        pal = ["#4D4D4D", "#0072B2", "#D55E00", "#009E73", "#CC79A7"]; w = 0.8 / len(tags); x = np.arange(len(variants))
        for i, t in enumerate(tags):
            ax.bar(x + (i - (len(tags) - 1) / 2) * w, [agg(by[(t, v)])["wer"] for v in variants], w, color=pal[i % len(pal)], label=t)
        ax.set_xticks(x, variants, rotation=25, ha="right"); ax.set_ylabel("WER (%) — lower is better")
        ax.set_title(f"Real MacBook recordings of Artaban commands (n = {len(by[(ref, variants[0])])} takes per condition)", loc="left")
        ax.legend(frameon=False)
        fig.savefig(res / "wer_by_model.png", dpi=200); plt.close(fig)
    except ImportError:
        pass
    print("\n".join(L[6:]))
    print(f"\nreport -> {res / 'report.md'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("commands", help="list the command set"); sub.add_parser("devices", help="list audio devices")
    r = sub.add_parser("record", help="record commands from the microphone")
    r.add_argument("--session", required=True)
    r.add_argument("--condition", default="quiet", help="free label, e.g. quiet, mask, far-2m, welding-speaker")
    r.add_argument("--takes", type=int, default=1); r.add_argument("--seconds", type=float, default=4.0)
    r.add_argument("--only", nargs="*", help="command ids to record (default: all)")
    r.add_argument("--input-device", type=int)
    r.add_argument("--play-noise", choices=["welding", "knock", "babble"])
    r.add_argument("--output-device", type=int, help="speaker for --play-noise (see `devices`)")
    r.add_argument("--noise-level-dbfs", type=float, default=-18.0, help="RMS level of the played noise (dBFS); checked by ear at start")
    r.add_argument("--noise-gain-db", dest="noise_level_dbfs", type=float, help=argparse.SUPPRESS)  # old name
    r.add_argument("--noise-continuous", action="store_true", help="keep the noise playing between takes")
    r.add_argument("--overwrite", action="store_true", help="re-record takes that already exist for this condition")
    s = sub.add_parser("score", help="score base vs adapters")
    s.add_argument("--session", required=True); s.add_argument("--conditions", nargs="*")
    s.add_argument("--models", nargs="+", default=["base"], help="first = reference; adapters as models/<name>")
    s.add_argument("--lookahead", type=int, default=6)
    s.add_argument("--add-noise", nargs="*", choices=["welding", "knock", "babble"], default=[])
    s.add_argument("--snr", type=float, nargs="+", default=[0.0, 5.0, 10.0])
    s.add_argument("--save-mixes", action="store_true", help="write the noise-mixed wavs to results/.../mixes/")
    for sp in (r, s):
        sp.add_argument("--data-root", type=Path, default=ROOT / "datasets/real_mic")
        sp.add_argument("--results-root", type=Path, default=ROOT / "results/real_mic")
    a = p.parse_args()
    if a.cmd == "commands":
        for cid, text, cat in COMMANDS: print(f"{cid:11s} {cat:7s} {text}")
    elif a.cmd == "devices":
        import sounddevice as sd; print(sd.query_devices())
    elif a.cmd == "record":
        cmd_record(a)
    else:
        cmd_score(a)


if __name__ == "__main__":
    main()
