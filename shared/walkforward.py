"""Build the two decomposition protocols that the paper compares.

PROTOCOL A -- one-time (what the original submission does, and what most of
this literature does). Decompose the ENTIRE series once, then split. Every
component value at a test timestamp was computed using data from the whole
series, including later points, so future information is present in the
training features. This is the protocol under test, not a protocol we endorse.

PROTOCOL B -- walk-forward (leakage-free). The training components come from
decomposing only the training portion. For each test index t the series is
re-decomposed on history[0:t], so nothing at or after t is ever seen. This
costs one decomposition per test point, which is why it is parallelised.

Both protocols emit the same file format, so the training code downstream does
not need to know which one produced its inputs.
"""
import json
import os
import sys
import time

import numpy as np

from decomp import decompose

WINDOW = 30


# --------------------------------------------------------------- windowing
def make_windows(subseqs, idx_from, idx_to, window=WINDOW):
    """Per-component sliding windows. X[c] has shape (n, window, 1)."""
    C = subseqs.shape[1]
    X = [[] for _ in range(C)]
    y = [[] for _ in range(C)]
    for t in range(idx_from, idx_to):
        for c in range(C):
            X[c].append(subseqs[t - window:t, c:c + 1])
            y[c].append(subseqs[t, c])
    return ([np.asarray(x, dtype=np.float32) for x in X],
            [np.asarray(v, dtype=np.float32) for v in y])


# ------------------------------------------------------------- protocol A
def protocol_one_time(series, train_frac=0.8, num_realizations=50, seed=0):
    subseqs, K, alpha, diag = decompose(series, num_realizations, seed=seed)
    n = subseqs.shape[0]
    train_end = WINDOW + int(train_frac * (n - WINDOW))
    Xtr, ytr = make_windows(subseqs, WINDOW, train_end)
    Xte, yte = make_windows(subseqs, train_end, n)
    return {'subseqs_train': subseqs[:train_end], 'X_train': Xtr, 'y_train': ytr,
            'X_test': Xte, 'y_test': yte, 'target_test': series[train_end:n],
            'target_train': series[WINDOW:train_end],
            'K': K, 'alpha': alpha, 'diag': diag, 'train_end': train_end, 'n': n}


# ------------------------------------------------------------- protocol B
def _one_step(series, t, window, nreal, K, alpha, seed):
    """Decompose history[0:t] and return the last `window` rows per component."""
    sub, _, _, _ = decompose(series[:t], num_realizations=nreal,
                             K=K, alpha=alpha, seed=seed)
    if sub.shape[0] < window:
        return None
    return sub[-window:, :].astype(np.float32)


def protocol_walk_forward(series, train_frac=0.8, num_realizations=20,
                          seed=0, stride=1, progress_every=25, **_ignored):
    """Leakage-free features. One decomposition per test point.

    Runs as a plain loop. One decomposition costs well under a second once K
    and alpha are fixed, so a few hundred of them finish in minutes and the
    multiprocessing this used to carry was not worth its portability problems.

    stride > 1 refits only every `stride` test points and reuses the most recent
    refit in between. stride=1 is the strict protocol; raise it only if the
    strict run will not finish, and report the value used.
    """
    series = np.asarray(series, dtype=float).ravel()
    n_full = len(series)
    train_end = WINDOW + int(train_frac * (n_full - WINDOW))

    # training features: decompose the training portion only. The PSO search for
    # K and alpha happens here, on training data alone, and the values are then
    # held fixed so that no test point can influence a hyperparameter.
    sub_tr, K, alpha, diag = decompose(series[:train_end],
                                       num_realizations=num_realizations, seed=seed)
    n_tr = sub_tr.shape[0]
    Xtr, ytr = make_windows(sub_tr, WINDOW, n_tr)
    C = sub_tr.shape[1]

    test_idx = list(range(train_end, n_full))
    refit_at = test_idx[::stride]
    t0 = time.time()
    blocks = []
    for i, t in enumerate(refit_at):
        blocks.append(_one_step(series, t, WINDOW, num_realizations, K, alpha, seed))
        if progress_every and (i + 1) % progress_every == 0:
            el = time.time() - t0
            print('      %d/%d decompositions | %.1f s elapsed | %.2f s each | '
                  'eta %.0f s' % (i + 1, len(refit_at), el, el / (i + 1),
                                  el / (i + 1) * (len(refit_at) - i - 1)),
                  flush=True)
    elapsed = time.time() - t0

    wins, targets = [], []
    for j, t in enumerate(test_idx):
        b = blocks[j // stride]
        if b is None or b.shape[1] != C:
            continue
        wins.append(b)
        targets.append(series[t])
    W = np.stack(wins)                      # (n_test, window, C)
    Xte = [W[:, :, c:c + 1].astype(np.float32) for c in range(C)]
    yte = [W[:, -1, c].astype(np.float32) for c in range(C)]

    return {'X_train': Xtr, 'y_train': ytr, 'X_test': Xte, 'y_test': yte,
            'target_test': np.asarray(targets), 'target_train': series[WINDOW:n_tr],
            'K': K, 'alpha': alpha, 'diag': diag, 'train_end': train_end,
            'n': n_full, 'n_decompositions': len(refit_at),
            'seconds': elapsed, 'stride': stride}


# ------------------------------------------------------------------- saving
def save(out_dir, tag, d):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, tag + '.npz')
    payload = {'target_test': d['target_test'], 'target_train': d['target_train']}
    for c, (x, y) in enumerate(zip(d['X_train'], d['y_train'])):
        payload['Xtr_%d' % c] = x
        payload['ytr_%d' % c] = y
    for c, (x, y) in enumerate(zip(d['X_test'], d['y_test'])):
        payload['Xte_%d' % c] = x
        payload['yte_%d' % c] = y
    payload['n_components'] = np.array([len(d['X_train'])])
    np.savez_compressed(path, **payload)

    meta = {k: v for k, v in d.items()
            if k not in ('X_train', 'y_train', 'X_test', 'y_test',
                         'target_test', 'target_train', 'subseqs_train')}
    meta['n_components'] = len(d['X_train'])
    meta['n_train_windows'] = int(d['X_train'][0].shape[0])
    meta['n_test_windows'] = int(d['X_test'][0].shape[0])
    meta['file_mb'] = round(os.path.getsize(path) / 1e6, 2)
    with open(os.path.join(out_dir, tag + '_meta.json'), 'w') as f:
        json.dump(meta, f, indent=1, default=float)
    return path, meta
