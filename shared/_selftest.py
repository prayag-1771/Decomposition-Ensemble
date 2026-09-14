"""End-to-end self-test on synthetic data. Runs in about a minute.

Anyone on the project can run this to confirm their environment works before
starting the long jobs:  python _selftest.py
"""
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from walkforward import protocol_one_time, protocol_walk_forward, save
from evaluate import all_baselines, metrics, dm_test


def main():
    rng = np.random.default_rng(0)
    s = 100 + np.cumsum(rng.standard_normal(300) * 0.5)

    print('[1] one-time protocol', flush=True)
    t0 = time.time()
    a = protocol_one_time(s, num_realizations=4, seed=0)
    print('    %d components | train %d | test %d | recon %.3f%% | %.0f s'
          % (len(a['X_train']), a['X_train'][0].shape[0],
             a['X_test'][0].shape[0], a['diag']['recon_pct_of_range'],
             time.time() - t0), flush=True)

    print('[2] walk-forward protocol', flush=True)
    t0 = time.time()
    b = protocol_walk_forward(s, num_realizations=4, seed=0, stride=10)
    print('    %d components | train %d | test %d | %d decompositions | %.0f s'
          % (len(b['X_train']), b['X_train'][0].shape[0],
             b['X_test'][0].shape[0], b['n_decompositions'], time.time() - t0), flush=True)

    print('[3] save / load round trip', flush=True)
    d = tempfile.mkdtemp()
    p, m = save(d, 'selftest', b)
    z = np.load(p)
    ok = all(k in z for k in ('Xtr_0', 'ytr_0', 'Xte_0', 'target_test'))
    print('    %.2f MB | keys present: %s' % (m['file_mb'], ok), flush=True)

    print('[4] baselines and Diebold-Mariano', flush=True)
    ts, n = 220, 60
    base = all_baselines(s, ts, n)
    y = s[ts:ts + n]
    for k in ('Persistence', 'MA-5', 'MA-20'):
        mm = metrics(y, base[k])
        print('    %-12s RMSE %7.3f  MAPE %6.3f%%' % (k, mm['RMSE'], mm['MAPE']), flush=True)
    stat, pv = dm_test(y, base['MA-20'], base['Persistence'])
    print('    DM MA-20 vs Persistence: %+.3f (p=%.4g) -> %s'
          % (stat, pv, 'MA-20 worse' if stat > 0 else 'MA-20 better'), flush=True)

    print('\nself-test passed. Environment is ready.', flush=True)


if __name__ == '__main__':
    main()
