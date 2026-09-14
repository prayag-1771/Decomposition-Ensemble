"""Results for the paper, from a finished training run.

Every comparison is made on the dates that both protocols and both heads
test, so row i of every forecast refers to the same trading day.

Writes, next to this script unless --out says otherwise:
  results_numbers.json   every number quoted in the text
  tab_accuracy.tex       persistence and the four model forecasts per series
  tab_baselines.tex      RMSE of the five benchmarks per series
  tab_leakage.tex        leakage ratio and MDM test, walk forward against full series
  tab_dm.tex             each model against persistence
  tab_heads.tex          TCN against TRM under each protocol
  figs/fig_rho.png       leakage ratio per series and head
  figs/fig_trace.png     forecasts against the price over part of the test period

    python analyze.py                      the real run
    python analyze.py --root ../_smoke     the smoke fixture
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--root', default=os.path.join(HERE, '..'))
ap.add_argument('--out', default=HERE)
a = ap.parse_args()
sys.path.insert(0, os.path.join(a.root, 'shared'))
from evaluate import all_baselines, metrics, dm_test      # noqa: E402

SERIES = ('SPX_High', 'SPX_Low', 'SSEC_High', 'SSEC_Low')
PROTOCOLS = ('onetime', 'walkforward')
HEADS = ('tcn', 'trm')
BASES = ('Persistence', 'Drift', 'MA-5', 'MA-20', 'EWMA-10')
LABEL = {'SPX_High': r'S\&P~500 high', 'SPX_Low': r'S\&P~500 low',
         'SSEC_High': 'SSEC high', 'SSEC_Low': 'SSEC low'}
PLAIN = {'SPX_High': 'S&P 500 high', 'SPX_Low': 'S&P 500 low',
         'SSEC_High': 'SSEC high', 'SSEC_Low': 'SSEC low'}
HNAME = {'tcn': 'TCN', 'trm': 'TRM'}
PNAME = {'onetime': 'full series', 'walkforward': 'walk forward'}
ALPHA = 0.05


def load_pred(tag, head):
    p = os.path.join(a.root, 'results', 'pred_%s_%s.npz' % (tag, head))
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return (z['test_index'].astype(int), z['y_true'].astype(float),
            z['y_pred'].astype(float))


def pfmt(p):
    return '$<0.001$' if p < 0.001 else '%.3f' % p


def sfmt(v):
    return ('$%+.2f$' % v).replace('+', '{+}').replace('-', '{-}')


R, F = {}, {}
for s in SERIES:
    got = {(p, h): load_pred('%s_%s' % (p, s), h) for p in PROTOCOLS for h in HEADS}
    if any(v is None for v in got.values()):
        print('skip %s: predictions incomplete' % s)
        continue
    x = np.load(os.path.join(a.root, 'data', 'series_%s.npy' % s)).astype(float)
    common = None
    for idx, _, _ in got.values():
        common = idx if common is None else np.intersect1d(common, idx)
    if not np.all(np.diff(common) == 1):
        sys.exit('%s: common test dates are not contiguous' % s)
    y = x[common]
    fc = {}
    for (p, h), (idx, yt, yp) in got.items():
        k = np.searchsorted(idx, common)
        if not (np.all(idx[k] == common) and np.allclose(yt[k], y)):
            sys.exit('%s %s %s: stored prices disagree with the series' % (s, p, h))
        fc[(p, h)] = yp[k]
    base = all_baselines(x, int(common[0]), len(common))
    r = {'dates': int(len(common)), 'first_index': int(common[0]),
         'baselines': {b: metrics(y, base[b]) for b in BASES},
         'models': {}, 'leakage': {}, 'heads': {}}
    for (p, h), f in fc.items():
        m = metrics(y, f)
        m['DM_vs_persistence'], m['p_vs_persistence'] = dm_test(y, f, base['Persistence'])
        r['models']['%s_%s' % (p, h)] = m
    for h in HEADS:
        ot, wf = r['models']['onetime_' + h], r['models']['walkforward_' + h]
        d, pv = dm_test(y, fc[('walkforward', h)], fc[('onetime', h)])
        r['leakage'][h] = {'rmse_onetime': ot['RMSE'], 'rmse_walkforward': wf['RMSE'],
                           'mape_onetime': ot['MAPE'], 'mape_walkforward': wf['MAPE'],
                           'rho': wf['RMSE'] / ot['RMSE'], 'DM_wf_vs_ot': d, 'p': pv}
    for p in PROTOCOLS:
        d, pv = dm_test(y, fc[(p, 'tcn')], fc[(p, 'trm')])
        r['heads'][p] = {'DM_tcn_vs_trm': d, 'p': pv}
        # Input persistence: forecast every component with the last value of
        # its own input window and sum. Under either protocol this is x_{t-1}
        # up to the decomposition's reconstruction error, so it shows what a
        # network would score by copying its inputs, and how much the trained
        # networks add or lose relative to that.
        z = np.load(os.path.join(a.root, 'data', '%s_%s.npz' % (p, s)))
        idx = z['test_index'].astype(int)
        k = np.searchsorted(idx, common)
        last = np.sum([z['Xte_%d' % c][k, -1, 0] for c in range(int(z['n_components'][0]))], axis=0)
        r['input_persistence_' + p] = metrics(y, last.astype(float))
    R[s], F[s] = r, (common, y, base['Persistence'], fc)

if not R:
    sys.exit('no complete series to analyse')

cases = [(s, h) for s in R for h in HEADS]
rho = np.array([R[s]['leakage'][h]['rho'] for s, h in cases])
summary = {
    'series_analysed': list(R),
    'rho_min': float(rho.min()), 'rho_max': float(rho.max()),
    'rho_geomean': float(np.exp(np.mean(np.log(rho)))),
    'wf_significantly_worse': sum(1 for s, h in cases
                                  if R[s]['leakage'][h]['DM_wf_vs_ot'] > 0
                                  and R[s]['leakage'][h]['p'] < ALPHA),
    'n_cases': len(cases),
}
for p in PROTOCOLS:
    ms = [R[s]['models']['%s_%s' % (p, h)] for s, h in cases]
    summary['beats_persistence_' + p] = sum(1 for m in ms if m['DM_vs_persistence'] < 0
                                            and m['p_vs_persistence'] < ALPHA)
    summary['worse_than_persistence_' + p] = sum(1 for m in ms if m['DM_vs_persistence'] > 0
                                                 and m['p_vs_persistence'] < ALPHA)
    summary['mape_range_' + p] = [min(m['MAPE'] for m in ms), max(m['MAPE'] for m in ms)]
    summary['r2_range_' + p] = [min(m['R2'] for m in ms), max(m['R2'] for m in ms)]
summary['persistence_mape_range'] = [min(R[s]['baselines']['Persistence']['MAPE'] for s in R),
                                     max(R[s]['baselines']['Persistence']['MAPE'] for s in R)]
json.dump({'summary': summary, 'series': R},
          open(os.path.join(a.out, 'results_numbers.json'), 'w'), indent=1)


def write(name, caption, label, spec, header, rows):
    body = ['\\begin{table}[ht]', '\\caption{%s}' % caption, '\\label{%s}' % label,
            '\\centering', '\\begin{tabular}{@{}%s@{}}' % spec, '\\toprule', header + ' \\\\',
            '\\midrule']
    for row in rows:
        body.append('\\midrule' if row == 'MID' else ' & '.join(row) + ' \\\\')
    body += ['\\botrule', '\\end{tabular}', '\\end{table}', '']
    open(os.path.join(a.out, name), 'w', encoding='utf-8', newline='\n').write('\n'.join(body))


rows = []
for s in R:
    if rows:
        rows.append('MID')
    items = [('Persistence', R[s]['baselines']['Persistence'])]
    items += [('%s, %s' % (HNAME[h], PNAME[p]), R[s]['models']['%s_%s' % (p, h)])
              for h in HEADS for p in PROTOCOLS]
    for i, (nm, m) in enumerate(items):
        rows.append([LABEL[s] if i == 0 else '', nm, '%.2f' % m['RMSE'], '%.2f' % m['MAE'],
                     '%.3f' % m['MAPE'], '%.4f' % m['R2']])
write('tab_accuracy.tex',
      'Out of sample accuracy on the dates tested by both protocols. Persistence '
      'forecasts each price with the previous day\'s value.',
      'tab:accuracy', 'llrrrr', 'Series & Forecast & RMSE & MAE & MAPE (\\%) & $R^2$', rows)

rows = [[b] + ['%.2f' % R[s]['baselines'][b]['RMSE'] for s in R] for b in BASES]
write('tab_baselines.tex',
      'RMSE of the benchmark forecasts on the common test dates.',
      'tab:baselines', 'l' + 'r' * len(R), 'Benchmark & ' + ' & '.join(LABEL[s] for s in R), rows)

rows = []
for s in R:
    for i, h in enumerate(HEADS):
        L = R[s]['leakage'][h]
        rows.append([LABEL[s] if i == 0 else '', HNAME[h], '%.2f' % L['rmse_onetime'],
                     '%.2f' % L['rmse_walkforward'], '%.2f' % L['rho'],
                     sfmt(L['DM_wf_vs_ot']), pfmt(L['p'])])
write('tab_leakage.tex',
      'Effect of the decomposition protocol. $\\rho$ is the walk forward RMSE (protocol B) '
      'divided by the full series RMSE (protocol A); a positive MDM statistic means the walk forward forecast '
      'has the larger squared error.',
      'tab:leakage', 'llrrrrr',
      'Series & Head & RMSE$_{\\mathrm{A}}$ & RMSE$_{\\mathrm{B}}$ & $\\rho$ & MDM & $p$', rows)

rows = []
for s in R:
    for i, (h, p) in enumerate((h, p) for h in HEADS for p in PROTOCOLS):
        m = R[s]['models']['%s_%s' % (p, h)]
        rows.append([LABEL[s] if i == 0 else '', '%s, %s' % (HNAME[h], PNAME[p]),
                     sfmt(m['DM_vs_persistence']), pfmt(m['p_vs_persistence'])])
write('tab_dm.tex',
      'Each model against persistence. A negative MDM statistic means the model has '
      'the smaller squared error.',
      'tab:dm', 'llrr', 'Series & Forecast & MDM & $p$', rows)

rows = [[LABEL[s]] + sum(([sfmt(R[s]['heads'][p]['DM_tcn_vs_trm']), pfmt(R[s]['heads'][p]['p'])]
                          for p in PROTOCOLS), []) for s in R]
write('tab_heads.tex',
      'TCN head against TRM head. A positive MDM statistic means the TCN head has the '
      'larger squared error.',
      'tab:heads', 'lrrrr', 'Series & MDM, full series & $p$ & MDM, walk forward & $p$', rows)

# ---------------------------------------------------------------- figures
import matplotlib                                          # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                            # noqa: E402

# PNG at 300 dpi: PDF export needs a fontTools extension that this machine's
# application control policy blocks.
os.makedirs(os.path.join(a.out, 'figs'), exist_ok=True)
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'legend.fontsize': 7,
                     'savefig.bbox': 'tight', 'savefig.dpi': 300})

fig, ax = plt.subplots(figsize=(5.2, 2.4))
xs = np.arange(len(R))
for j, (h, col) in enumerate(zip(HEADS, ('#3b6ea8', '#c8553d'))):
    vals = [R[s]['leakage'][h]['rho'] for s in R]
    bars = ax.bar(xs + (j - 0.5) * 0.36, vals, 0.36, label=HNAME[h] + ' head', color=col)
    for b, s in zip(bars, R):
        if R[s]['leakage'][h]['p'] < ALPHA:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), '*', ha='center', va='bottom')
ax.axhline(1.0, color='k', lw=0.8)
ax.set_xticks(xs)
ax.set_xticklabels([PLAIN[s] for s in R])
ax.set_ylabel(r'leakage ratio $\rho$')
ax.legend(frameon=False)
fig.savefig(os.path.join(a.out, 'figs', 'fig_rho.png'))
plt.close(fig)

show = [s for s in ('SPX_High', 'SSEC_High') if s in F]
fig, axes = plt.subplots(len(show), 1, figsize=(5.2, 2.1 * len(show)), squeeze=False)
for ax, s in zip(axes[:, 0], show):
    common, y, pers, fc = F[s]
    n = min(100, len(common))
    t = np.arange(n)
    ax.plot(t, y[:n], color='k', lw=1.1, label='price')
    ax.plot(t, pers[:n], color='0.6', lw=0.9, ls='--', label='persistence')
    ax.plot(t, fc[('onetime', 'tcn')][:n], color='#3b6ea8', lw=0.9, label='TCN, full series')
    ax.plot(t, fc[('walkforward', 'tcn')][:n], color='#c8553d', lw=0.9, label='TCN, walk forward')
    ax.set_title(PLAIN[s])
    ax.set_xlabel('test day')
axes[0, 0].legend(frameon=False, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(a.out, 'figs', 'fig_trace.png'))
plt.close(fig)

print(json.dumps(summary, indent=1))
print('wrote tables and figures to %s' % a.out)
