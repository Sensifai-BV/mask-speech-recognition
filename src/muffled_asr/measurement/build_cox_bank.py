"""Extract covering-relative, smoothed magnitudes from verified Cox archives."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import zipfile
import numpy as np
from cox_tables import measurement_table

FREQUENCIES = np.geomspace(250., 7600., 31)

def relative_curve(masked, bare, rate, window_ms=12.):
    # Acquisition delays vary by ~20 ms in June. Align each arrival separately
    # before magnitude estimation; measured relative phase is not retained.
    # This suppresses late room energy, not all early room reflections.
    def power(ir):
        peak = int(np.argmax(np.abs(ir)))
        start = max(0, peak-round(rate*.002))
        end = min(len(ir), peak+round(rate*window_ms/1000))
        width = end-start; taper = np.ones(width)
        edge = min(round(rate*.001), width//4)
        taper[:edge] = np.linspace(0, 1, edge)
        taper[-edge:] = np.linspace(1, 0, edge)
        return np.abs(np.fft.rfft(ir[start:end]*taper, n=32768))**2
    base_power = np.mean([power(ir) for ir in bare], axis=0) if np.ndim(bare)==2 else power(bare)
    mask_power = power(masked)
    frequency = np.fft.rfftfreq(32768, 1/rate)
    floor = np.max(base_power)*1e-8
    gain = []
    for center in FREQUENCIES:
        select = (frequency >= center/2**(1/6)) & (frequency < center*2**(1/6))
        gain.append(10*np.log10((mask_power[select].sum()+floor)/(base_power[select].sum()+floor)))
    gain = np.asarray(gain)
    if not np.isfinite(gain).all() or gain.min() < -40 or gain.max() > 25:
        raise ValueError('Implausible relative gain; inspect the acquisition pair')
    return gain.tolist()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('datasets/original/cox-2022'))
    p.add_argument('--out', type=Path, default=Path('datasets/calibrated/response_bank.json'))
    p.add_argument('--window-ms', type=float, default=12.)
    args = p.parse_args()
    records = []
    for month in ['march', 'june', 'august']:
        folder = args.source/month
        folder.mkdir(exist_ok=True)
        with zipfile.ZipFile(args.source/f'measurements_from_{month}_2021.zip') as z:
            for member in z.namelist():
                if Path(member).name == member and member.endswith('.mat'):
                    if not (folder/member).exists(): z.extract(member, folder)
    def append(path, design, split, masked, bare):
        name, ir, rate = masked; bare_name, bare_ir, bare_rate = bare
        if rate != bare_rate: raise ValueError('Sample-rate mismatch')
        records.append({'id': f'{path.stem}:{name}', 'design': design, 'split': split,
                        'source_file': path.name, 'source_sha256': hashes[path.name],
                        'masked_row': name, 'unmasked_row': bare_name,
                        'sample_rate_original': rate,
                        'gain_db': relative_curve(ir, bare_ir, rate, args.window_ms)})
    paths = list((args.source/'march').glob('*.mat')) + list((args.source/'june').glob('*.mat')) + list((args.source/'august').glob('*.mat'))
    hashes = {}
    for path in paths:
        with path.open('rb') as f: hashes[path.name] = hashlib.file_digest(f, 'sha256').hexdigest()
    for path in sorted((args.source/'march').glob('*.mat')):
        rows = measurement_table(path)
        code = path.stem.split('_')[-1]
        design = 'cloth_v1' if code.startswith('PCBL') else code
        if len(rows) != 12: raise ValueError(f'{path}: expected six paired refits')
        for off, on in zip(rows[::2], rows[1::2]):
            if not re.search(r'mask_?0', off[0]) or not re.search(r'mask_?1', on[0]):
                raise ValueError(f'{path}: unrecognized pair {off[0]}, {on[0]}')
            append(path, design, 'train', on, off)
    reference = measurement_table(args.source/'june/no-mask_far-mic_ir_data.mat')
    # June's unmasked recordings form a separate baseline acquisition series.
    # Average spectral power across six repeats; averaging delayed waveforms
    # would cancel energy and create artificial attenuation/amplification.
    reference_mean = ('mean_power_far_baseline', np.asarray([r[1] for r in reference]), reference[0][2])
    for path in sorted((args.source/'june').glob('[A-E]-*.mat')):
        split = 'dev' if path.name.startswith('E-') else 'test'
        design = 'cloth_v2_plain' if split == 'dev' else 'cloth_v2_TPU_'+path.stem[0]
        for row in measurement_table(path):
            if '_far_' in row[0]: append(path, design, split, row, reference_mean)
    # The cling-film visor has an ambiguously named baseline and is excluded.
    # Cloth-v2 cling-film is also excluded from this pilot.
    for name, design in [('measured_ir_data_full_visor.mat','full_visor'),
                         ('measured_ir_data_visor_TPU_insert.mat','TPU_visor')]:
        path = args.source/'august'/name; rows = measurement_table(path)
        if rows[0][0] != 'No_visor': raise ValueError('Unknown visor baseline')
        for row in rows[1:]: append(path, design, 'train', row, rows[0])
    bank = {'source_doi': '10.17866/rd.salford.19174235.v1', 'license': 'CC BY 4.0',
            'frequencies_hz': FREQUENCIES.tolist(), 'window_ms': args.window_ms,
            'method': 'individually arrival-aligned windows; third-octave power ratio; minimum-phase synthesis',
            'low_frequency_policy': 'constant gain below 250 Hz; uncalibrated extrapolation',
            'high_frequency_policy': 'constant gain above 7600 Hz to Nyquist',
            'train_designs': sorted({r['design'] for r in records if r['split']=='train'}),
            'held_out_test_family': 'cloth v2 with TPU window, all A-D constructions',
            'records': records}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(bank, indent=2))
    print(args.out, dict(Counter(r['split'] for r in records)))

if __name__ == '__main__': main()
