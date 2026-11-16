"""Empirical and bounded low-rank sampling of covering magnitude responses."""
import json
from pathlib import Path
import numpy as np
from scipy.signal import firwin2, minimum_phase, fftconvolve

class CoveringBank:
    def __init__(self, path):
        self.data = json.loads(Path(path).read_text())
        self.frequency = np.asarray(self.data['frequencies_hz'])
        self.groups = {}
        for row in self.data['records']:
            self.groups.setdefault((row['split'], row['design']), []).append(row)

    def sample(self, split, mode, rng):
        designs = sorted(d for s,d in self.groups if s==split)
        if not designs: raise ValueError(f'No filters for {split}')
        design = designs[int(rng.integers(len(designs)))]; rows = self.groups[(split,design)]
        if mode == 'empirical' or split != 'train':
            row = rows[int(rng.integers(len(rows)))]; gain = np.array(row['gain_db'])
            recipe = {'mode':'empirical', 'design':design, 'response_id':row['id']}
        else:
            x = np.asarray([r['gain_db'] for r in rows]); mean = x.mean(axis=0)
            _, singular, axes = np.linalg.svd(x-mean, full_matrices=False)
            rank = min(2, len(rows)-1)
            z = np.clip(rng.standard_normal(rank), -2., 2.)
            gain = mean + (z*singular[:rank]/np.sqrt(len(rows)-1)) @ axes[:rank]
            # Bounds and covariance are fitted within this design only.
            gain = np.clip(gain, x.min(axis=0), x.max(axis=0))
            recipe = {'mode':'bounded_pca', 'design':design, 'rank':rank, 'coefficients':z.tolist()}
        recipe['gain_db'] = gain.tolist()
        return gain, recipe

    def fir(self, gain, rate=16000):
        frequencies = np.r_[0., self.frequency, rate/2]
        amplitude = 10**(np.r_[gain[0],gain,gain[-1]]/20)
        linear = firwin2(1025, frequencies, amplitude, fs=rate)
        return minimum_phase(linear, method='homomorphic', half=False)

    def apply(self, audio, gain):
        return fftconvolve(audio, self.fir(gain), mode='full')[:len(audio)].astype(np.float32)
