"""Check the decomposition after the corrections. Run: python test_decomp.py"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decomp import emd_imf, iceemdan, vmd, pso_vmd, decompose

fail = []


def check(cond, msg, detail=''):
    print(('  ok    ' if cond else '  FAIL  ') + msg + ('  ' + detail if detail else ''))
    if not cond:
        fail.append(msg)


def zc(x):
    x = x - x.mean()
    return int(np.sum(np.diff(np.sign(x)) != 0))


rng = np.random.default_rng(0)
# a price-like series: level in the thousands, so the noise-scaling fix matters
n = 400
price = 5000 + np.cumsum(rng.normal(0, 8, n)) + 40 * np.sin(2 * np.pi * np.arange(n) / 60)

print('=' * 66)
print('EMD core')
print('=' * 66)
e = emd_imf(price)
check(np.max(np.abs(e.sum(axis=0) - price)) < 1e-9,
      'IMFs sum back to the signal', '%.2e' % np.max(np.abs(e.sum(axis=0) - price)))

print()
print('=' * 66)
print('ICEEMDAN')
print('=' * 66)
t0 = time.time()
im = iceemdan(price, num_realizations=20, seed=0)
el = time.time() - t0
check(im.shape[0] == n, 'returns one row per sample', '%s' % (im.shape,))
err = np.max(np.abs(im.sum(axis=1) - price))
check(err < 1e-9, 'components sum back exactly', '%.2e' % err)
cross = [zc(im[:, c]) for c in range(im.shape[1])]
check(cross[0] == max(cross), 'column 0 is the highest-frequency component',
      'zero-crossings %s' % cross)
check(cross[-1] == min(cross), 'last column is the trend',
      'trend crossings %d' % cross[-1])

# the noise must actually do something now
a = iceemdan(price, num_realizations=6, seed=1)
b = iceemdan(price, num_realizations=6, seed=2)
spread = float(np.max(np.abs(a - b)))
check(spread > 1e-6, 'the ensemble noise is large enough to matter',
      'seed-to-seed spread %.3f' % spread)
print('        %d components, %.1f s for 20 realisations' % (im.shape[1], el))

print()
print('=' * 66)
print('VMD')
print('=' * 66)
for N in (400, 399):          # 399 is genuinely odd; price is only 400 long
    m = vmd(price[:N], 500.0, 4)
    check(m.shape == (4, N), 'length preserved for N=%d' % N, str(m.shape))
m = vmd(price, 500.0, 5)
rel = np.max(np.abs(m.sum(axis=0) - price)) / (price.max() - price.min())
check(rel < 0.25, 'modes roughly reconstruct (VMD is lossy)', 'rel err %.3f' % rel)

print()
print('=' * 66)
print('PSO')
print('=' * 66)
t0 = time.time()
K, alpha = pso_vmd(im[:, 0], pop=6, iters=3, seed=0)
check(3 <= K <= 10 and 100 <= alpha <= 5000, 'returns in-bounds (K, alpha)',
      'K=%d alpha=%.0f in %.1f s' % (K, alpha, time.time() - t0))

print()
print('=' * 66)
print('decompose()')
print('=' * 66)
for N in (400, 399):
    sub, K, alpha, diag = decompose(price[:N], num_realizations=6, seed=0)
    check(sub.shape[0] == N, 'row count equals input length for N=%d' % N,
          '%s' % (sub.shape,))
    check(diag['recon_pct_of_range'] < 5.0,
          'reconstruction within 5%% of range for N=%d' % N,
          '%.4f%%' % diag['recon_pct_of_range'])

print()
print('=' * 66)
print('%d failure(s)' % len(fail))
for f in fail:
    print('  - ' + f)
sys.exit(1 if fail else 0)
