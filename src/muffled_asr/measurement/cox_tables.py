"""Read the specific MCOS tables in Cox's MATLAB v5 measurement archives.

SciPy cannot expose table columns directly. Read the embedded MAT subsystem,
then locate columns by their names and validate all dimensions. This is a
dataset-specific reader, not a general MATLAB class deserializer.
"""
import io
import numpy as np
from scipy.io import loadmat
from scipy.io.matlab._mio5 import MatFile5Reader

def measurement_table(path):
    mat = loadmat(path)
    workspace = mat.get('__function_workspace__')
    if workspace is None:
        raise ValueError(f'{path}: no MATLAB object subsystem')
    raw = workspace.tobytes()
    header = b'MATLAB 5.0 MAT-file'.ljust(116, b' ') + b'\0'*8 + raw[:4]
    reader = MatFile5Reader(io.BytesIO(header+raw[8:]), squeeze_me=True, struct_as_record=False)
    reader.initialize_read(); reader.read_file_header()
    var_header, _ = reader.read_var_header()
    wrapper = reader.read_var_array(var_header).MCOS
    field = '_ObjectMetadata' if '_ObjectMetadata' in wrapper.dtype.names else 'arr'
    cells = wrapper[0][field].ravel()
    def string_array(x):
        return isinstance(x, np.ndarray) and x.dtype == object and all(isinstance(v, str) for v in x.flat)
    names = next((x.ravel().tolist() for x in cells if string_array(x) and 'ImpulseResponse' in x), None)
    if names is None:
        raise ValueError(f'{path}: no named impulse-response column')
    candidates = [x.ravel() for x in cells if isinstance(x, np.ndarray) and x.dtype == object and x.size == len(names)]
    columns = next((x for x in candidates if isinstance(x[names.index('ImpulseResponse')], np.ndarray)
                    and x[names.index('ImpulseResponse')].dtype == object
                    and hasattr(x[names.index('ImpulseResponse')].flat[0], 'Amplitude')), None)
    if columns is None:
        raise ValueError(f'{path}: could not resolve table data')
    irs = columns[names.index('ImpulseResponse')].ravel()
    rows = next((x.ravel().tolist() for x in cells if string_array(x) and x.size == len(irs)
                 and any(k in v.lower() for v in x.flat for k in ('mask', 'visor', 'far', 'near'))), None)
    if rows is None:
        raise ValueError(f'{path}: missing row names')
    rates = np.atleast_1d(columns[names.index('SampleRate')]).ravel()
    if len(rates) != len(irs):
        raise ValueError(f'{path}: rate/IR count mismatch')
    result = []
    for name, ir, rate in zip(rows, irs, rates):
        amplitude = np.asarray(ir.Amplitude, dtype=float).reshape(-1)
        time = np.asarray(ir.Time, dtype=float).reshape(-1)
        if not np.isfinite(amplitude).all() or len(amplitude) != len(time):
            raise ValueError(f'{path}/{name}: invalid IR')
        if not np.isclose(np.median(np.diff(time)), 1/float(rate)):
            raise ValueError(f'{path}/{name}: sample rate conflicts with time axis')
        result.append((name, amplitude, int(rate)))
    return result
