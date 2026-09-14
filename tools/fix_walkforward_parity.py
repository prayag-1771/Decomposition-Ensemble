"""Correct the walk-forward windows that end one day early on alternate days.

decompose() trims its input to an even length because VMD needs one, and it
trims from the END. Walk-forward features for test day t come from
decompose(series[:t]), so whenever t is odd the newest price x_{t-1} is
dropped, the 30-day window ends at t-2, and the model is in effect asked for a
two-step forecast. Component targets for even t, read from
decompose(series[:t+1]), are affected in the same way.

The fix decomposes series[t % 2 : t] instead: the OLDEST price is dropped
rather than the newest, the length is always even, and the last row is always
t-1. Nothing at or after t is used, so the protocol stays strictly causal.

Blocks for even t are unchanged by the fix (same slice, same seed), so only
odd-t decompositions are recomputed:
    odd t      -> new feature window for test day t
    odd t + 1  -> new component target for test day t (t even)
Before patching, three even blocks per series are recomputed and compared with
the delivered rows, to confirm this environment reproduces Veer's numbers; if
it does not, --all recomputes every block so the file is internally
consistent.

    python tools/fix_walkforward_parity.py --check     reproduction check only
    python tools/fix_walkforward_parity.py             patch odd blocks
    python tools/fix_walkforward_parity.py --all       recompute every block
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse                                                  # noqa: E402
import json                                                      # noqa: E402
import shutil                                                    # noqa: E402
import sys                                                       # noqa: E402
import time                                                      # noqa: E402
from concurrent.futures import ProcessPoolExecutor               # noqa: E402

import numpy as np                                               # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
DATA = os.path.join(ROOT, 'data')
BACKUP = os.path.join(DATA, '_pre_parity_fix')
VEER = os.path.join(ROOT, '_veer_incoming', 'veer_deliverable', 'shared')
SERIES = ('SPX_High', 'SPX_Low', 'SSEC_High', 'SSEC_Low')
WINDOW = 30
SEED = 0


def block(x, t, K, alpha, C, nreal):
    """Last WINDOW rows of the decomposition of the history before t."""
    if VEER not in sys.path:
        sys.path.insert(0, VEER)
    from decomp import decompose
    from walkforward import fit_components
    sub, _, _, _ = decompose(x[t % 2:t], num_realizations=nreal, K=K, alpha=alpha, seed=SEED)
    return fit_components(sub, C)[-WINDOW:, :]


def lag_counts(X, x, idx):
    """How many rows line up with a window ending at t-1 (lag 0) or t-2 (lag 1)."""
    e0 = np.stack([np.abs(X[i].sum(1) - x[t - WINDOW:t]).max() for i, t in enumerate(idx)])
    e1 = np.stack([np.abs(X[i].sum(1) - x[t - WINDOW - 1:t - 1]).max() for i, t in enumerate(idx)])
    return int((e0 <= e1).sum()), int((e1 < e0).sum()), float((e0 / x[idx]).max() * 100)


def run(s, mode, nreal):
    x = np.load(os.path.join(DATA, 'series_%s.npy' % s)).astype(float).ravel()
    tag = 'walkforward_' + s
    zpath = os.path.join(DATA, tag + '.npz')
    mpath = os.path.join(DATA, tag + '_meta.json')
    with np.load(zpath) as zf:
        z = {k: zf[k] for k in zf.files}
    meta = json.load(open(mpath))
    if 'parity_fix' in meta and mode != 'check':
        return '%s: already fixed on %s' % (s, meta['parity_fix']['date'])
    C, K, alpha = int(z['n_components'][0]), int(meta['K']), float(meta['alpha'])
    idx = z['test_index'].astype(int)
    X = np.stack([z['Xte_%d' % c][:, :, 0] for c in range(C)], axis=2).astype(float)
    Y = np.stack([z['yte_%d' % c] for c in range(C)], axis=1).astype(float)
    before = lag_counts(X, x, idx)

    even = [i for i, t in enumerate(idx) if t % 2 == 0]
    rep = []
    for i in (even[0], even[len(even) // 2], even[-1]):
        b = block(x, int(idx[i]), K, alpha, C, nreal)
        rep.append(float(np.abs(b - X[i]).max() / x[idx[i]]))
    report = ('%s: C=%d K=%d alpha=%g | rows at lag0/lag1 before: %d/%d | '
              'even-block reproduction, max |diff| / price: %s'
              % (s, C, K, alpha, before[0], before[1], ', '.join('%.1e' % r for r in rep)))
    if mode == 'check':
        return report
    if mode == 'odd' and max(rep) > 1e-5:
        return report + '\n  NOT PATCHED: this environment does not reproduce the delivered rows; use --all'

    t0, done = time.time(), 0
    jobs = [(i, int(t)) for i, t in enumerate(idx) if mode == 'all' or t % 2 == 1 or (t + 1) % 2 == 1]
    for i, t in jobs:
        if mode == 'all' or t % 2 == 1:
            X[i] = block(x, t, K, alpha, C, nreal)
        if mode == 'all' or (t + 1) % 2 == 1:
            Y[i] = block(x, t + 1, K, alpha, C, nreal)[-1]
        done += 1
        if done % 50 == 0:
            el = time.time() - t0
            print('  %s %d/%d rows, %.0f s, eta %.0f s'
                  % (s, done, len(jobs), el, el / done * (len(jobs) - done)), flush=True)

    after = lag_counts(X, x, idx)
    tgt_err = float((np.abs(Y.sum(1) - x[idx]) / x[idx]).max() * 100)
    if after[1] != 0:
        return report + '\n  NOT SAVED: %d rows still end at t-2' % after[1]

    os.makedirs(BACKUP, exist_ok=True)
    for p in (zpath, mpath):
        dst = os.path.join(BACKUP, os.path.basename(p))
        if not os.path.exists(dst):
            shutil.copy2(p, dst)
    for c in range(C):
        z['Xte_%d' % c] = X[:, :, c:c + 1].astype(np.float32)
        z['yte_%d' % c] = Y[:, c].astype(np.float32)
    tmp = zpath[:-4] + '.tmp.npz'
    np.savez_compressed(tmp, **z)
    os.replace(tmp, zpath)
    meta['parity_fix'] = {
        'date': time.strftime('%Y-%m-%d %H:%M'), 'mode': mode, 'num_realizations': nreal,
        'rows_recomputed': len(jobs), 'seconds': round(time.time() - t0, 1),
        'lag0_rows_before': before[0], 'lag1_rows_before': before[1],
        'lag0_rows_after': after[0], 'lag1_rows_after': after[1],
        'window_recon_max_pct': after[2], 'target_recon_max_pct': tgt_err,
        'even_block_reproduction_max_rel_diff': max(rep),
        'note': 'history slice series[t % 2 : t] so the window always ends at t-1'}
    tmpm = mpath + '.tmp'
    with open(tmpm, 'w') as f:
        json.dump(meta, f, indent=1, default=float)
    os.replace(tmpm, mpath)
    return (report + '\n  PATCHED %d rows in %.0f s | lag0/lag1 after: %d/%d | window recon max %.3f%% | '
            'target recon max %.3f%%' % (len(jobs), time.time() - t0, after[0], after[1], after[2], tgt_err))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--realizations', type=int, default=20)
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--only', default=None)
    a = ap.parse_args()
    mode = 'check' if a.check else ('all' if a.all else 'odd')
    todo = [s for s in SERIES if not a.only or a.only in s]
    with ProcessPoolExecutor(max_workers=min(a.workers, len(todo))) as pool:
        for msg in pool.map(run, todo, [mode] * len(todo), [a.realizations] * len(todo)):
            print(msg, flush=True)
