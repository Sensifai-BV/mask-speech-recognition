"""Write reports/coxr-floor/P1_step1_reading_floor.md from the coxr_floor.py tables (numbers are not typed)."""
import ast, json
from pathlib import Path
import pandas as pd

from muffled_asr.paths import ROOT; TB = ROOT / "results/p0/tables"; OUT = ROOT / "reports/coxr-floor"
rd = lambda f: pd.read_csv(TB / f, keep_default_na=False, na_values=[""])
F, P, C, Q = rd("coxr_floor.csv"), rd("coxr_masked_vs_null_pooled.csv"), rd("coxr_masked_vs_null.csv"), rd("coxr_gap_residual_corr.csv")
seg = json.loads((ROOT / "datasets/p0/coxr/segments.json").read_text()); dec = json.loads((TB / "coxr_decision.json").read_text())
L = lambda v: ast.literal_eval(v) if isinstance(v, str) and v.startswith("[") else v
fm = lambda x, d=2: "—" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x:+.{d}f}" if isinstance(x, (int, float)) else str(x)
ci = lambda v: "—" if v in (None, "", "None") or (isinstance(v, float) and pd.isna(v)) else "[{:.2f}, {:.2f}]".format(*L(v))
snrs = ["quiet", "0", "-5", "-10"]; S = lambda s: "quiet" if s == "quiet" else f"{int(s)} dB"
allr = F[F["mask"] == "all"]; get = lambda kind, snr: allr[(allr.kind == kind) & (allr.snr.astype(str) == snr)].iloc[0]
p = {str(r.snr): r for _, r in P.iterrows()}; q = {(int(r.snr), r.kind): r for _, r in Q.iterrows()}
asr = [r["asr_wer_pct"] for v in seg.values() for r in v]; durs = [r["dur_s"] for v in seg.values() for r in v]
m10, m5, m0 = get("masked", "-10"), get("masked", "-5"), get("masked", "0")

md = f"""# P1 Step 1: Reading-to-Reading Floor (Cox-R)

Does the masked-minus-unmasked difference exceed what two different readings of the same passage produce?

Technical report · 21 October 2026 · step 1 of the P1 plan; uses the P0 oracle-filter recipe unchanged

- **Model** nvidia/nemotron-speech-streaming-en-0.6b, frozen; greedy RNN-T; 560 ms context
- **Data** Cox et al. 2022 repeated-reading set (`repeated_reading_rainbow_passage.zip`): one talker, six readings per file, mask off and on for cloth v1 + acetate insert, printed cloth and surgical; checksums in `datasets/SHA256SUMS`
- **Audio** 153 pairs × U/M/F/R × quiet, 0, −5, −10 dB = 2,448 files, 15.7 h; welding noise from the held-out P0 test excerpts

## Summary

- **The mask effect is larger than the reading floor when the three masks are pooled.** At −10 dB the masked-vs-unmasked gap is {m10.gap:.2f} points; two unmasked readings differ by {get('null','-10').gap:.2f}. Difference {fm(p['-10'].gap_diff)} points, 95 % CI {ci(p['-10'].gap_diff_CI)} (reading bootstrap, stratified by mask). At −5 dB {fm(p['-5'].gap_diff)} {ci(p['-5'].gap_diff_CI)}; at 0 dB {fm(p['0'].gap_diff)} {ci(p['0'].gap_diff_CI)}. In quiet there is no gap ({fm(p['quiet'].gap_diff)}).
- **No single mask clears the floor.** At −10 dB the per-mask differences are {", ".join(f"{k.replace('_',' ')} {fm(v['gap_diff'])} {ci(str(v['CI']))}" for k, v in dec['per_mask_at_-10dB'].items())}; all three intervals include zero.
- **The residual that the oracle filter leaves is NOT above the floor.** Residual = WER(M) − WER(F), masked minus null: {fm(p['0'].residual_diff)} {ci(p['0'].residual_diff_CI)} at 0 dB, {fm(p['-5'].residual_diff)} {ci(p['-5'].residual_diff_CI)} at −5 dB, {fm(p['-10'].residual_diff)} {ci(p['-10'].residual_diff_CI)} at −10 dB. For this talker, what the linear filter does not explain is indistinguishable from reading-to-reading variability.
- **The filter explains about half of this talker's mask gap.** Pooled R_fwd = {m0.R_fwd:.2f} {ci(m0.R_fwd_CI_reading)} at 0 dB, {m5.R_fwd:.2f} {ci(m5.R_fwd_CI_reading)} at −5 dB, {m10.R_fwd:.2f} {ci(m10.R_fwd_CI_reading)} at −10 dB — lower than P0's Geng surgical estimates (0.68 at 0 dB, 0.75 and 1.26 at −5 and −10 dB) and its Cox 1 m estimates (0.84–1.02), where the filter explained little or none of the gap.
- **A high per-talker correlation between gap and residual is not evidence of production.** Between two UNMASKED readings, r(gap, residual) across pairs is already {q[(0,'null')].r_gap_residual:.2f} / {q[(-5,'null')].r_gap_residual:.2f} / {q[(-10,'null')].r_gap_residual:.2f} at 0 / −5 / −10 dB — the size of Geng's 0.80–0.96. When F ≈ U the residual is the gap minus a small term, so the two correlate whatever their cause.
- **One reading pair is too noisy to show a mask effect.** The gap of a single unmasked pair has SD {q[(0,'null')].sd_gap:.1f} / {q[(-5,'null')].sd_gap:.1f} / {q[(-10,'null')].sd_gap:.1f} points at 0 / −5 / −10 dB, comparable to the pooled masked gap itself. A per-talker gap from one mask-on and one mask-off reading (Geng, Cox-H) should be expected to carry reading noise of this size; only pooled estimates can show the mask effect.
- **Floor for Table II (last row) and Table IV:** unmasked-vs-unmasked gap {get('null','0').gap:+.2f} {ci(get('null','0').gap_CI_reading)} / {get('null','-5').gap:+.2f} {ci(get('null','-5').gap_CI_reading)} / {get('null','-10').gap:+.2f} {ci(get('null','-10').gap_CI_reading)} points and residual {get('null','0').residual:+.2f} / {get('null','-5').residual:+.2f} / {get('null','-10').residual:+.2f} points at 0 / −5 / −10 dB (45 pairs, one talker).

## 1  Data and segmentation

The six files are mono, 48 kHz, 24-bit, 143–148 s each. The dataset description does not say which microphone was used. Energy after word offsets decays by 15 dB in 60–80 ms, like the Cox-H 1 m channel (60–85 ms) and unlike the cheek channel (15–40 ms), so the recordings are most likely far-field; this is an indirect cue, not a specification.

**Segmentation (deviation from the plan).** The planned rule — split at silences ≥ 1 s below −40 dBFS — gives 3–9 interior gaps per file instead of 5: some gaps between readings are shorter than 1 s and some pauses inside a reading are longer. The readings were therefore cut at the longest pause (≥ 0.3 s below −40 dBFS) within ±5 s of where one sixth of the speech span predicts each boundary, with 150 ms of padding. Every one of the 36 segments was transcribed by the base model and checked: all start with "when the sunlight", end in "of the rainbow", last {min(durs):.1f}–{max(durs):.1f} s and score {min(asr):.1f}–{max(asr):.1f} % WER in quiet. Boundaries are in `datasets/p0/coxr/segments.json` for the listening check. The talker reads the "like a prism" version of the passage (quiet unmasked WER {dec['reference']['quiet_U_wer_pct']['like a prism']} % vs {dec['reference']['quiet_U_wer_pct']['as a prism']} % against the other version), which is used as the reference.

## 2  Method

All P0 machinery is reused without change (`src/muffled_asr/oracle_filter/filters.py`, `gen_conditions.py`): per-pair LTAS oracle filter at the 6 dB activity threshold, minimum-phase forward and inverse FIRs, peak normalisation, welding test noise with the post-filter SNR convention, and one noise excerpt per pair and SNR shared by U, M, F and R.

| Set | U | M | Pairs |
|---|---|---|---|
| Null (floor) | reading i of a mask-off file | a later reading j > i of the same file | 15 per file × 3 = 45 |
| Masked | reading i of the mask-off file | reading j of the matching mask-on file | 36 per mask × 3 = 108 |

Gap Δ = WER(M) − WER(U); residual = WER(M) − WER(F); R_fwd = residual / Δ and R_inv = (WER(R) − WER(U)) / Δ, defined when Δ ≥ 2 points. WERs are pooled over pairs. Pairs share readings, so confidence intervals resample the six U readings and the six M readings (2,000–4,000 resamples, stratified by mask when masks are pooled) and keep the pairs they induce; a pair-level bootstrap gives similar or narrower intervals and is in `coxr_floor.csv`.

## 3  Results

![Reading floor]({'figures/fig1_reading_floor.png'})

*Figure 1. (a) Gap and (b) residual pooled over the three masks, for two unmasked readings (grey, the floor) and masked vs unmasked readings (orange), with 95 % reading-bootstrap CIs. (c) Gap of every individual pair. One talker; noise = held-out welding excerpts.*

### 3.1  Pooled over masks

| SNR | Set | U | M | F | R | Δ | Δ CI | Residual | Residual CI | R_fwd | R_inv |
|---|---|---|---|---|---|---|---|---|---|---|---|
"""
for s in snrs:
    for k in ("null", "masked"):
        r = get(k, s)
        md += f"| {S(s)} | {k} | {r.WER_U:.2f} | {r.WER_M:.2f} | {r.WER_F:.2f} | {r.WER_R:.2f} | {r.gap:+.2f} | {ci(r.gap_CI_reading)} | {r.residual:+.2f} | {ci(r.residual_CI_reading)} | {fm(r.R_fwd)} | {fm(r.R_inv)} |\n"
md += """
### 3.2  Masked minus null (the test of step 1)

| SNR | Δ masked | Δ null | Difference | 95 % CI | P(diff ≤ 0) | Residual masked | Residual null | Difference | 95 % CI |
|---|---|---|---|---|---|---|---|---|---|
"""
for s in snrs:
    r = p[s]
    md += f"| {S(s)} | {r.gap_masked:+.2f} | {r.gap_null:+.2f} | {r.gap_diff:+.2f} | {ci(r.gap_diff_CI)} | {r.P_diff_le_0:.3f} | {r.residual_masked:+.2f} | {r.residual_null:+.2f} | {r.residual_diff:+.2f} | {ci(r.residual_diff_CI)} |\n"
md += """
### 3.3  Per mask

| SNR | Mask | Δ masked | Δ null | Difference | 95 % CI | Residual difference | 95 % CI |
|---|---|---|---|---|---|---|---|
"""
for _, r in C[C.snr.astype(str) != "quiet"].iterrows():
    md += f"| {S(str(r.snr))} | {r['mask'].replace('_', ' ')} | {r.gap_masked:+.2f} | {r.gap_null:+.2f} | {r.gap_diff:+.2f} | {ci(r.gap_diff_CI)} | {r.residual_diff:+.2f} | {ci(r.residual_diff_CI)} |\n"
md += """
### 3.4  Correlation of gap and residual across pairs

| SNR | Set | Pairs | r(Δ, residual) | SD of Δ (points) | SD of F − U | mean F − U |
|---|---|---|---|---|---|---|
"""
for _, r in Q.iterrows():
    md += f"| {int(r.snr)} dB | {r.kind} | {r.n} | {r.r_gap_residual:.2f} | {r.sd_gap:.2f} | {r.sd_F_minus_U:.2f} | {r.mean_F_minus_U:+.2f} |\n"
md += f"""
## 4  What this means for the plan

1. **Step 1 rule.** Pooled over the three coverings, the masked gap at −10 dB exceeds the unmasked-vs-unmasked gap ({fm(p['-10'].gap_diff)} points, CI {ci(p['-10'].gap_diff_CI)}). It does not for any single covering. The mask effect on WER is real for this talker, but only detectable in aggregate.
2. **The residual question.** For this talker the oracle filter explains roughly half of the gap, and what it leaves is within the reading floor at every SNR. On this evidence the far-field residual is not shown to be production; it is consistent with reading-to-reading variability. This is one talker with cloth and surgical masks, so it does not refute a production effect in Geng — it shows that a per-talker residual of a few points is expected from reading variability alone.
3. **For P1.** Phone-class effects must be tested against this floor and pooled across talkers; per-talker gap/residual correlations should not be cited as evidence. The per-phone floor (the residual envelope r_k between two unmasked readings) is not computed yet: it needs the P1 phone alignment (MFA) and the P1 residual-envelope definition, which are not in the repository.

## 5  Limitations

- One talker, three cloth/surgical coverings, quiet readings; a floor, not a replication.
- Microphone position not documented (decay suggests far-field).
- At −10 dB all conditions are near 90 % WER, which compresses every difference; −5 dB is the less compressed low-SNR point.
- The null includes noise-realisation variability across pairs (each pair has its own noise excerpt, as in P0), so it bounds reading plus noise variability, which is the right comparison for P0's single-pair-per-talker design.
- Segmentation used a duration prior instead of the planned 1 s rule (Section 1); the 17 flagged P0 reference runs and these 36 boundaries are still unverified by ear.

## Files

- Pipeline: `src/muffled_asr/oracle_filter/coxr_floor.py` (prep, gen, score), report: `src/muffled_asr/oracle_filter/coxr_report.py`
- Segments: `datasets/p0/coxr/segments.json`; readings `datasets/p0/data16k/coxr/`; per-pair filters `datasets/p0/coxr/pair_gains.json`
- Conditions: `datasets/p0/conds/coxr/`, manifest `datasets/p0/manifests/coxr.jsonl`; transcripts `results/p0/hyps_base_manifests-coxr_look6.jsonl`
- Tables: `results/p0/tables/coxr_floor.csv`, `coxr_masked_vs_null.csv`, `coxr_masked_vs_null_pooled.csv`, `coxr_gap_residual_corr.csv`, `coxr_pairs.csv`, `coxr_decision.json`
"""
OUT.mkdir(parents=True, exist_ok=True); (OUT / "P1_step1_reading_floor.md").write_text(md)
print(md[:3000])
