"""VEER'S DRIVER -- produces every decomposition the study needs.

Run order:
    python run_decompositions.py --task validate      (10 minutes, do this first)
    python run_decompositions.py --task onetime       (~2 min per series)
    python run_decompositions.py --task walkforward   (~25 min per series)

Everything here is NumPy and SciPy on CPU. No GPU is used or needed.
Outputs land in ../data/ as .npz plus a _meta.json for each run.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from decomp import decompose, iceemdan, emd_imf, vmd          # noqa: E402
from walkforward import protocol_one_time, protocol_walk_forward, save  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), '..', 'data')
TICKERS = {'SPX': '^GSPC', 'SSEC': '000001.SS'}


def fetch(ticker, period='10y'):
    import yfinance as yf
    d = yf.download(ticker, period=period, interval='1d', progress=False)
    return (d['High'].dropna().values.ravel().astype(float),
            d['Low'].dropna().values.ravel().astype(float))


def get_series(name, which='High'):
    os.makedirs(OUT, exist_ok=True)
    cache = os.path.join(OUT, 'series_%s_%s.npy' % (name, which))
    if os.path.exists(cache):
        return np.load(cache)
    hi, lo = fetch(TICKERS[name])
    np.save(os.path.join(OUT, 'series_%s_High.npy' % name), hi)
    np.save(os.path.join(OUT, 'series_%s_Low.npy' % name), lo)
    return hi if which == 'High' else lo


# ------------------------------------------------------------------ TASK 1
def task_validate():
    """Three checks. Report all three numbers in your results document."""
    print('=' * 68)
    print('VALIDATION -- confirms the decomposition is sound before the long runs')
    print('=' * 68)
    res = {}

    print('\n[1] Does plain EMD reconstruct? (should be ~1e-12 or smaller)')
    rng = np.random.default_rng(0)
    x = 100 + np.cumsum(rng.standard_normal(600) * 0.5)
    e = emd_imf(x)
    res['emd_recon_err'] = float(np.max(np.abs(e.sum(axis=0) - x)))
    print('    max reconstruction error: %.3e' % res['emd_recon_err'])

    print('\n[2] Does the corrected ICEEMDAN reconstruct? (should be ~1e-12)')
    im = iceemdan(x, num_realizations=20, seed=0)
    res['iceemdan_recon_err'] = float(np.max(np.abs(im.sum(axis=1) - x)))
    res['iceemdan_n_imfs'] = int(im.shape[1])
    print('    IMFs: %d, max reconstruction error: %.3e'
          % (res['iceemdan_n_imfs'], res['iceemdan_recon_err']))

    print('\n[3] ORIGINAL (uncorrected) averaging, at the submission settings.')
    print('    The paper reports reconstruction error below 0.01%% of the price')
    print('    range. This measures the same quantity with the original code.')
    s = get_series('SPX', 'High')
    orig = _iceemdan_original(s[:1200], num_realizations=50, seed=0)
    err = float(np.max(np.abs(orig.sum(axis=1) - s[:1200])))
    rng_ = float(s[:1200].max() - s[:1200].min())
    res['original_recon_err'] = err
    res['original_recon_pct'] = 100.0 * err / rng_
    print('    max error %.4f  =  %.4f%% of price range' % (err, res['original_recon_pct']))
    print('    (report this number: it says whether the paper\'s 0.01%% claim holds)')

    with open(os.path.join(OUT, 'validation.json'), 'w') as f:
        json.dump(res, f, indent=1)
    print('\nsaved ../data/validation.json')


def _iceemdan_original(signal, noise_std=0.2, num_realizations=50, seed=None):
    """The submission's averaging, kept verbatim for the comparison in check 3."""
    rng = np.random.default_rng(seed)
    signal = np.asarray(signal, dtype=float).ravel()
    N = len(signal)
    max_imfs = 0
    for _ in range(5):
        max_imfs = max(max_imfs, emd_imf(signal + rng.normal(0, noise_std, N)).shape[0])
    max_imfs = max(max_imfs, 3)
    accum = np.zeros((max_imfs, N))
    count = np.zeros(max_imfs)
    for _ in range(num_realizations):
        imfs_r = emd_imf(signal + rng.normal(0, noise_std, N))
        k = min(imfs_r.shape[0], max_imfs)
        accum[:k] += imfs_r[:k]
        count[:k] += 1
    count = np.where(count > 0, count, 1)
    return (accum / count[:, None]).T[:, ::-1]


# ------------------------------------------------------------------ TASK 2
def task_onetime():
    for name in ('SPX', 'SSEC'):
        for which in ('High', 'Low'):
            s = get_series(name, which)
            tag = 'onetime_%s_%s' % (name, which)
            print('\n=== %s  (%d observations) ===' % (tag, len(s)))
            t0 = time.time()
            d = protocol_one_time(s, num_realizations=50, seed=0)
            path, meta = save(OUT, tag, d)
            print('  components %d | K=%d alpha=%.1f | recon %.4f%% of range'
                  % (meta['n_components'], meta['K'], meta['alpha'],
                     meta['diag']['recon_pct_of_range']))
            print('  train windows %d, test windows %d | %.0f s | %s (%.1f MB)'
                  % (meta['n_train_windows'], meta['n_test_windows'],
                     time.time() - t0, os.path.basename(path), meta['file_mb']))


# ------------------------------------------------------------------ TASK 3
def task_walkforward(nreal, stride):
    for name in ('SPX', 'SSEC'):
        for which in ('High', 'Low'):
            s = get_series(name, which)
            tag = 'walkforward_%s_%s' % (name, which)
            print('\n=== %s  (%d observations) ===' % (tag, len(s)))
            print('    one decomposition per test point, %d realisations, stride %d'
                  % (nreal, stride))
            d = protocol_walk_forward(s, num_realizations=nreal, seed=0,
                                      stride=stride)
            path, meta = save(OUT, tag, d)
            print('  %d decompositions in %.0f s (%.1f s each)'
                  % (meta['n_decompositions'], meta['seconds'],
                     meta['seconds'] / max(meta['n_decompositions'], 1)))
            print('  components %d | train %d, test %d | %s (%.1f MB)'
                  % (meta['n_components'], meta['n_train_windows'],
                     meta['n_test_windows'], os.path.basename(path), meta['file_mb']))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True,
                    choices=['validate', 'onetime', 'walkforward'])
    ap.add_argument('--realizations', type=int, default=20)
    ap.add_argument('--stride', type=int, default=1)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.task == 'validate':
        task_validate()
    elif a.task == 'onetime':
        task_onetime()
    else:
        task_walkforward(a.realizations, a.stride)
