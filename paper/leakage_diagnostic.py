"""How much information about tomorrow's price change sits in today's model
inputs under each protocol, measured without any neural network.

For a test date t, every component input window ends with that component's
value at day t-1. Under one-time decomposition the value was computed from the
whole series, x_t included; under walk-forward decomposition it was computed
from the history before t alone. If the one-time inputs carry the answer, even
a three-variable linear regression can read tomorrow's change out of them.

Two measures on the dates both protocols test:
  corr   Pearson correlation between the next-day change x_t - x_{t-1} and
         the high-frequency group (the VMD modes) at t-1.
  skill  out-of-sample skill of a linear regression of the next-day change on
         three input features, relative to predicting no change (persistence):
         1 - SSE_regression / SSE_persistence. Fitted on one half of the dates
         and scored on the other, both ways round, errors pooled.

Every input window is first checked to end at t-1: its components must add
back to the prices x_{t-30..t-1}, and the script stops if any row instead lines
up with the window one day earlier.

    python leakage_diagnostic.py   -> leakage_diagnostic.json, figs/fig_mechanism.png
"""
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
SERIES = ('SPX_High', 'SPX_Low', 'SSEC_High', 'SSEC_Low')
WINDOW = 30


def load(tag):
    z = np.load(os.path.join(DATA, tag + '.npz'))
    C = int(z['n_components'][0])
    idx = z['test_index'].astype(int)
    W = np.stack([z['Xte_%d' % c][:, :, 0] for c in range(C)], axis=2)    # (n, 30, C)
    meta = json.load(open(os.path.join(DATA, tag + '_meta.json')))
    return idx, W, C, meta


def aligned(W, x, idx):
    """Rows whose summed window matches x[t-30:t] better than x[t-31:t-1]."""
    s = W.sum(axis=2)
    e0 = np.array([np.abs(s[i] - x[t - WINDOW:t]).max() for i, t in enumerate(idx)])
    e1 = np.array([np.abs(s[i] - x[t - WINDOW - 1:t - 1]).max() for i, t in enumerate(idx)])
    return e0 <= e1, float((e0 / x[idx]).max() * 100)


def features(last, prev, K):
    """VMD modes (first K columns) as one group, the remaining oscillatory
    components as another, residual (last column) excluded."""
    hf, hf_prev = last[:, :K].sum(1), prev[:, :K].sum(1)
    mid, mid_prev = last[:, K:-1].sum(1), prev[:, K:-1].sum(1)
    return hf, np.column_stack([hf, hf - hf_prev, mid - mid_prev])


def skill(F, y):
    """Two-fold out-of-sample skill against a zero-change forecast."""
    n = len(y)
    halves = (np.arange(n // 2), np.arange(n // 2, n))
    sse_fit = sse_rw = 0.0
    for fit, score in (halves, halves[::-1]):
        A = np.column_stack([F[fit], np.ones(len(fit))])
        coef, *_ = np.linalg.lstsq(A, y[fit], rcond=None)
        pred = np.column_stack([F[score], np.ones(len(score))]) @ coef
        sse_fit += float(np.sum((y[score] - pred) ** 2))
        sse_rw += float(np.sum(y[score] ** 2))
    return 1.0 - sse_fit / sse_rw


out, keep = {}, {}
print('%-10s %-12s %4s %6s %9s %9s %8s %11s'
      % ('series', 'protocol', 'C', 'dates', 'corr', 'p', 'skill', 'window err%'))
for s in SERIES:
    x = np.load(os.path.join(DATA, 'series_%s.npy' % s)).astype(float)
    loaded = {p: load('%s_%s' % (p, s)) for p in ('onetime', 'walkforward')}
    common = np.intersect1d(loaded['onetime'][0], loaded['walkforward'][0])
    dx = x[common] - x[common - 1]
    out[s] = {'common_dates': int(len(common))}
    for p, (idx, W, C, meta) in loaded.items():
        ok, werr = aligned(W, x, idx)
        if not ok.all():
            sys.exit('%s %s: %d input windows end at t-2; run tools/fix_walkforward_parity.py'
                     % (s, p, int((~ok).sum())))
        K = int(meta['K'])
        rows = np.searchsorted(idx, common)
        assert np.all(idx[rows] == common)
        L, P = W[rows, -1, :], W[rows, -2, :]
        hf, F = features(L, P, K)
        r, pv = stats.pearsonr(hf, dx)
        sk = skill(F, dx)
        keep[(s, p)] = (hf, dx)
        out[s][p] = {'C': C, 'K': K, 'corr_hf_nextchange': float(r), 'p': float(pv),
                     'oos_skill_vs_persistence': float(sk), 'window_recon_max_pct': werr}
        print('%-10s %-12s %4d %6d %+9.3f %9.2g %+8.3f %11.3f'
              % (s, p, C, len(common), r, pv, sk, werr))
json.dump(out, open(os.path.join(HERE, 'leakage_diagnostic.json'), 'w'), indent=1)
print('\nwrote leakage_diagnostic.json')

LABEL = {'SPX_High': r'S\&P~500 high', 'SPX_Low': r'S\&P~500 low',
         'SSEC_High': 'SSEC high', 'SSEC_Low': 'SSEC low'}


def num(v, fmt):
    return ('$' + fmt % v + '$').replace('-', '{-}')


rows = []
for s in SERIES:
    for i, p in enumerate(('onetime', 'walkforward')):
        d = out[s][p]
        pv = '$<0.001$' if d['p'] < 0.001 else '%.2f' % d['p']
        rows.append('%s & %s & %s & %s & %s \\\\' % (
            LABEL[s] if i == 0 else '', 'full series' if p == 'onetime' else 'walk forward',
            num(d['corr_hf_nextchange'], '%.3f'), pv, num(d['oos_skill_vs_persistence'], '%.3f')))
table = '\n'.join([
    '\\begin{table}[ht]',
    '\\caption{Information about the next day\'s price change in the model inputs, '
    'measured without any network. $r$ is the correlation between the high frequency '
    'group at day $t-1$ and the change from $t-1$ to $t$; skill is the out of sample '
    'reduction in squared error that a linear regression on three input features '
    'achieves over persistence.}',
    '\\label{tab:diagnostic}', '\\centering',
    '\\begin{tabular}{@{}llrrr@{}}', '\\toprule',
    'Series & Protocol & $r$ & $p$ & Skill \\\\', '\\midrule'] + rows
    + ['\\botrule', '\\end{tabular}', '\\end{table}', ''])
open(os.path.join(HERE, 'tab_diagnostic.tex'), 'w', encoding='utf-8', newline='\n').write(table)
print('wrote tab_diagnostic.tex')

# PNG at 300 dpi: PDF export needs a fontTools extension that this machine's
# application control policy blocks.
import matplotlib                                          # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                            # noqa: E402

os.makedirs(os.path.join(HERE, 'figs'), exist_ok=True)
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'savefig.bbox': 'tight',
                     'savefig.dpi': 300})
fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.4), sharey=True)
for ax, p, col, nm in zip(axes, ('onetime', 'walkforward'), ('#3b6ea8', '#c8553d'),
                          ('Full series decomposition', 'Walk forward decomposition')):
    hf, dx = keep[('SPX_High', p)]
    ax.scatter(hf, dx, s=4, alpha=0.55, color=col, linewidths=0)
    ax.axhline(0, color='0.7', lw=0.6)
    ax.axvline(0, color='0.7', lw=0.6)
    ax.set_title('%s, $r = %.2f$' % (nm, out['SPX_High'][p]['corr_hf_nextchange']))
    ax.set_xlabel('high frequency group at day $t-1$')
axes[0].set_ylabel('price change from $t-1$ to $t$')
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'figs', 'fig_mechanism.png'))
plt.close(fig)
print('wrote figs/fig_mechanism.png')
