"""Simulate a face mask or enclosed PPE on clean speech.

Masked training data can be manufactured from any clean 16 kHz corpus, which is
what makes this worth doing: the transcripts come along unchanged, and one
recording session's worth of measurement turns 100 hours of LibriSpeech into 100
hours of masked LibriSpeech.

Two physically different things get called "a mask", and they need different
models:

LIGHT FABRIC (surgical, KN95, N95)
    Essentially just insertion loss: a few dB of attenuation above 1 kHz and
    nothing else. Values are measured, from Corey, Jones & Singer, "Acoustic
    effects of medical, cloth, and transparent face masks on speech signals",
    JASA 148(4) 2020. A pure magnitude shelf is a fair model here.

ENCLOSURES (face shield, respirator, welding helmet)
    A shelf is *not* a fair model, and does not sound muffled. Four things
    happen that a shelf leaves out, and together they are the percept:

      1. A steep low-pass, not a gentle shelf. The enclosure stops behaving like
         a thin resistive layer and starts behaving like a wall, so the high end
         falls off at 12-24 dB/octave past a knee around 1.5-3 kHz.
      2. Occlusion. Trapped air in front of the mouth boosts the low end, which
         is what makes the voice sound boxed in rather than merely quiet.
      3. Cavity resonance. The enclosed volume has modes at c/2L and its
         multiples, giving peaks and notches — the "talking into a bucket"
         coloration.
      4. Cavity reverberation. Dense early reflections off hard plastic inside a
         very small space, decaying in tens of milliseconds.

    Enclosure parameters are ESTIMATES: physically motivated (mode frequencies
    follow from the cavity dimension) but not measured off real gear. They give
    you realistic-sounding training data; they do not give you a calibrated
    model of your specific helmet. Measure yours if the numbers matter.

Not modelled: speakers change how they talk behind PPE, and PPE arrives with a
different microphone in a louder room. Those are separate terms — see mix_noise.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, firwin2, resample_poly, sosfilt

SAMPLE_RATE = 16000
SPEED_OF_SOUND = 343.0


@dataclass(frozen=True)
class Mask:
    """A mask as a signal chain. Light masks use `shelf` alone."""

    note: str
    # (frequency Hz, attenuation dB), interpolated on a log-frequency axis.
    shelf: list[tuple[float, float]] = field(default_factory=list)
    # Steep roll-off past the knee. Order is per-octave slope / 6.
    lowpass_hz: float | None = None
    lowpass_order: int = 2
    # Occlusion: low-shelf boost from trapped air in front of the mouth.
    lf_boost_db: float = 0.0
    lf_boost_hz: float = 500.0
    # Cavity modes as peaking EQs: (frequency Hz, gain dB, Q).
    resonances: list[tuple[float, float, float]] = field(default_factory=list)
    # Enclosure geometry. Modes land at c/2L and multiples.
    cavity_m: float | None = None
    cavity_rt60_ms: float = 20.0
    cavity_wet: float = 0.4


MASKS: dict[str, Mask] = {
    "surgical": Mask(
        note="measured (JASA 2020)",
        shelf=[(0, 0.0), (500, 0.0), (1000, 0.5), (2000, 2.5), (4000, 4.0), (6000, 3.5), (8000, 3.0)],
    ),
    "kn95": Mask(
        note="measured (JASA 2020)",
        shelf=[(0, 0.0), (500, 0.0), (1000, 0.5), (2000, 2.8), (4000, 4.2), (6000, 3.8), (8000, 3.4)],
    ),
    "n95": Mask(
        note="measured (JASA 2020)",
        shelf=[(0, 0.0), (500, 0.0), (1000, 1.0), (2000, 3.5), (4000, 6.0), (6000, 5.5), (8000, 5.0)],
    ),
    "shield": Mask(
        note="estimate — rigid plastic visor, open at the bottom",
        shelf=[(0, 0.0), (500, 0.0), (1000, 1.5), (2000, 4.0), (4000, 7.0), (8000, 7.0)],
        lowpass_hz=3200.0, lowpass_order=2,
        lf_boost_db=3.0, lf_boost_hz=400.0,
        resonances=[(900, 5.0, 2.0), (2500, -8.0, 2.0)],
        cavity_m=0.12, cavity_rt60_ms=12.0, cavity_wet=0.30,
    ),
    "respirator": Mask(
        note="estimate — half-face elastomeric, sealed to the face",
        shelf=[(0, 0.0), (400, 0.0), (1000, 2.0), (2000, 5.0), (4000, 9.0), (8000, 10.0)],
        lowpass_hz=2600.0, lowpass_order=2,
        lf_boost_db=3.5, lf_boost_hz=500.0,
        resonances=[(600, 4.0, 1.5), (1800, -5.0, 2.0)],
        cavity_m=0.08, cavity_rt60_ms=10.0, cavity_wet=0.40,
    ),
    "welding_helmet": Mask(
        note="estimate — full head enclosure, large hard cavity",
        shelf=[(0, 0.0), (300, 0.0), (1000, 2.5), (2000, 6.0), (4000, 10.0), (8000, 12.0)],
        lowpass_hz=2200.0, lowpass_order=2,
        lf_boost_db=4.0, lf_boost_hz=500.0,
        resonances=[(450, 4.0, 1.2), (1500, -6.0, 2.0), (2800, -5.0, 3.0)],
        cavity_m=0.22, cavity_rt60_ms=25.0, cavity_wet=0.50,
    ),
}

SHELF_TAPS = 513  # linear phase, ~32 ms at 16 kHz


def _shelf_fir(breakpoints: list[tuple[float, float]], sample_rate: int) -> np.ndarray:
    nyquist = sample_rate / 2
    freqs_hz = np.array([f for f, _ in breakpoints], dtype=float)
    atten_db = np.array([a for _, a in breakpoints], dtype=float)

    grid = np.concatenate([[0.0], np.geomspace(20.0, nyquist, 512)])
    interpolated = np.interp(
        np.log10(np.maximum(grid, 20.0)),
        np.log10(np.maximum(freqs_hz, 20.0)),
        atten_db,
    )
    interpolated[grid < freqs_hz[1]] = atten_db[0]

    gain = 10.0 ** (-interpolated / 20.0)
    gain[-1] = gain[-2]
    return firwin2(SHELF_TAPS, grid / nyquist, gain)


def _peaking_sos(freq_hz: float, gain_db: float, q: float, sample_rate: int) -> np.ndarray:
    """RBJ peaking EQ biquad, as a second-order section."""
    amplitude = 10.0 ** (gain_db / 40.0)
    omega = 2.0 * np.pi * freq_hz / sample_rate
    alpha = np.sin(omega) / (2.0 * q)
    cos_omega = np.cos(omega)

    b = [1 + alpha * amplitude, -2 * cos_omega, 1 - alpha * amplitude]
    a = [1 + alpha / amplitude, -2 * cos_omega, 1 - alpha / amplitude]
    return np.array([[*b, *a]]) / a[0]


def _low_shelf_sos(freq_hz: float, gain_db: float, sample_rate: int) -> np.ndarray:
    """RBJ low-shelf biquad — the occlusion boost."""
    amplitude = 10.0 ** (gain_db / 40.0)
    omega = 2.0 * np.pi * freq_hz / sample_rate
    cos_omega, sin_omega = np.cos(omega), np.sin(omega)
    alpha = sin_omega / 2.0 * np.sqrt(2.0)
    shared = 2.0 * np.sqrt(amplitude) * alpha

    b = [
        amplitude * ((amplitude + 1) - (amplitude - 1) * cos_omega + shared),
        2 * amplitude * ((amplitude - 1) - (amplitude + 1) * cos_omega),
        amplitude * ((amplitude + 1) - (amplitude - 1) * cos_omega - shared),
    ]
    a = [
        (amplitude + 1) + (amplitude - 1) * cos_omega + shared,
        -2 * ((amplitude - 1) + (amplitude + 1) * cos_omega),
        (amplitude + 1) + (amplitude - 1) * cos_omega - shared,
    ]
    return np.array([[*b, *a]]) / a[0]


def cavity_ir(length_m: float, rt60_ms: float, sample_rate: int, num_modes: int = 8) -> np.ndarray:
    """Impulse response of a small hard enclosure.

    Sum of damped sinusoids at the axial modes c/2L, 2c/2L, ... over a decaying
    noise floor for the dense late part. The direct path stays at sample 0 so
    convolution does not shift the signal — alignment with the transcript and
    with the unmasked file has to survive.
    """
    length = max(1, int(sample_rate * rt60_ms / 1000.0))
    t = np.arange(length) / sample_rate
    decay = np.exp(-6.9 * t / (rt60_ms / 1000.0))

    fundamental = SPEED_OF_SOUND / (2.0 * length_m)
    rng = np.random.default_rng(0)  # fixed: the enclosure is one fixed object
    response = rng.standard_normal(length) * decay * 0.3
    for mode in range(1, num_modes + 1):
        freq = fundamental * mode * (1.0 + 0.02 * rng.standard_normal())
        if freq >= sample_rate / 2:
            break
        response += np.sin(2.0 * np.pi * freq * t) * decay / mode

    response /= np.abs(response).max()
    response[0] += 1.0  # direct path
    return response / np.sqrt(np.sum(response**2))


def apply_mask(audio: np.ndarray, preset: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Run the full chain. Output stays sample-aligned with the input."""
    try:
        mask = MASKS[preset]
    except KeyError:
        raise SystemExit(f"unknown mask {preset!r}; choose from {', '.join(MASKS)}")

    out = audio.astype(np.float64)

    if mask.shelf:
        taps = _shelf_fir(mask.shelf, sample_rate)
        delay = (len(taps) - 1) // 2
        padded = np.concatenate([out, np.zeros(delay)])
        out = np.convolve(padded, taps, mode="full")[delay : delay + len(audio)]

    sections = []
    if mask.lf_boost_db:
        sections.append(_low_shelf_sos(mask.lf_boost_hz, mask.lf_boost_db, sample_rate))
    for freq_hz, gain_db, q in mask.resonances:
        if freq_hz < sample_rate / 2:
            sections.append(_peaking_sos(freq_hz, gain_db, q, sample_rate))
    if mask.lowpass_hz:
        sections.append(
            butter(mask.lowpass_order, mask.lowpass_hz, btype="low", fs=sample_rate, output="sos")
        )
    if sections:
        # Causal, not zero-phase. Zero-phase filtering applies the response
        # twice and so doubles every gain in dB; it is also wrong physically,
        # since a real enclosure is minimum-phase. The few ms of group delay
        # this adds is realistic and far below one streaming chunk.
        out = sosfilt(np.concatenate(sections), out)

    if mask.cavity_m:
        wet = np.convolve(out, cavity_ir(mask.cavity_m, mask.cavity_rt60_ms, sample_rate))[: len(out)]
        out = (1.0 - mask.cavity_wet) * out + mask.cavity_wet * wet

    peak = np.abs(out).max()
    if peak > 0.99:  # resonances can add gain; keep it inside the PCM range
        out *= 0.99 / peak
    return out.astype(np.float32)


def mix_noise(audio: np.ndarray, noise: np.ndarray, snr_db: float, rng=None) -> np.ndarray:
    """Add `noise` at the requested SNR, looping or cropping it to fit.

    The crop offset is random so one long recording yields many distinct
    backgrounds instead of the same few seconds over and over.
    """
    rng = rng or np.random.default_rng()
    if len(noise) < len(audio):
        noise = np.tile(noise, int(np.ceil(len(audio) / len(noise))))
    offset = int(rng.integers(0, len(noise) - len(audio) + 1))
    noise = noise[offset : offset + len(audio)]

    speech_power = np.mean(audio**2)
    noise_power = np.mean(noise**2)
    if noise_power <= 0 or speech_power <= 0:
        return audio
    scale = np.sqrt(speech_power / (noise_power * 10.0 ** (snr_db / 10.0)))
    return audio + scale * noise


def load_16k_mono(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        if sample_rate < SAMPLE_RATE:
            raise SystemExit(
                f"{path.name} is {sample_rate} Hz. Upsampling cannot recreate the "
                f"1-8 kHz content a mask acts on, so the result would be meaningless."
            )
        audio = resample_poly(audio, SAMPLE_RATE, sample_rate).astype("float32")
    return audio


def octave_band_delta(clean: np.ndarray, masked: np.ndarray) -> list[tuple[str, float]]:
    """Measured band-by-band change — proof the chain did what the preset claims."""
    spec_clean = np.abs(np.fft.rfft(clean)) ** 2
    spec_masked = np.abs(np.fft.rfft(masked)) ** 2
    freqs = np.fft.rfftfreq(len(clean), 1 / SAMPLE_RATE)

    rows = []
    for low, high in [(125, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]:
        band = (freqs >= low) & (freqs < high)
        energy_clean, energy_masked = spec_clean[band].sum(), spec_masked[band].sum()
        if energy_clean > 0 and energy_masked > 0:
            rows.append((f"{low}-{high} Hz", 10 * np.log10(energy_masked / energy_clean)))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("inputs", nargs="+", type=Path, help="clean speech files (flac/wav)")
    parser.add_argument("--mask", default="welding_helmet", choices=sorted(MASKS))
    parser.add_argument("--noise", type=Path, help="noise recording to mix in after the mask")
    parser.add_argument("--snr", type=float, default=5.0)
    parser.add_argument("--out-dir", type=Path, default=Path("examples/mask_sim"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    noise = load_16k_mono(args.noise) if args.noise else None
    print(f"mask: {args.mask} — {MASKS[args.mask].note}  ->  {args.out_dir}/\n")

    for path in args.inputs:
        clean = load_16k_mono(path)
        masked = apply_mask(clean, args.mask)
        suffix = args.mask
        if noise is not None:
            masked = mix_noise(masked, noise, args.snr)
            suffix = f"{args.mask}.snr{args.snr:g}"

        sf.write(args.out_dir / f"{path.stem}.clean.wav", clean, SAMPLE_RATE, subtype="PCM_16")
        sf.write(args.out_dir / f"{path.stem}.{suffix}.wav", masked, SAMPLE_RATE, subtype="PCM_16")

        print(f"{path.stem}  ({len(clean) / SAMPLE_RATE:.1f}s)")
        for band, delta in octave_band_delta(clean, masked):
            bar = ("#" if delta < 0 else "+") * min(60, int(round(abs(delta) * 2)))
            print(f"    {band:>12}  {delta:+6.1f} dB  {bar}")
        print()


if __name__ == "__main__":
    main()
