"""AYUSH'S DRIVER -- baselines, metrics, significance tests and figures.

Run order:
    python run_evaluation.py --task baselines    (minutes, do this first)
    python run_evaluation.py --task compare      (after model predictions arrive)
    python run_evaluation.py --task figures

Task 1 needs nothing from anybody else, so start there immediately.
Everything here is CPU only.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
from evaluate import (all_baselines, metrics, dm_test, comparison_table,  # noqa: E402
                      bias_correct)

DATA = os.path.join(os.path.dirname(__file__), '..', 'data')
RES = os.path.join(os.path.dirname(__file__), '..', 'results')
WINDOW = 30
TICKERS = {'SPX': '^GSPC', 'SSEC': '000001.SS'}


def get_series(name, which='High'):
    cache = os.path.join(DATA, 'series_%s_%s.npy' % (name, which))
    if os.path.exists(cache):
        return np.load(cache)
    import yfinance as yf
    d = yf.download(TICKERS[name], period='10y', interval='1d', progress=False)
    os.makedirs(DATA, exist_ok=True)
    hi = d['High'].dropna().values.ravel().astype(float)
    lo = d['Low'].dropna().values.ravel().astype(float)
    np.save(os.path.join(DATA, 'series_%s_High.npy' % name), hi)
    np.save(os.path.join(DATA, 'series_%s_Low.npy' % name), lo)
    return hi if which == 'High' else lo


# ------------------------------------------------------------------ TASK 1
def task_baselines():
    """Baseline performance on every series. Independent of the deep models."""
    os.makedirs(RES, exist_ok=True)
    out = {}
    for name in ('SPX', 'SSEC'):
        for which in ('High', 'Low'):
            s = get_series(name, which)
            n_full = len(s)
            train_end = WINDOW + int(0.8 * (n_full - WINDOW))
            n_test = n_full - train_end
            y = s[train_end:]
            base = all_baselines(s, train_end, n_test)
            key = '%s_%s' % (name, which)
            out[key] = {'n_obs': int(n_full), 'n_test': int(n_test),
                        'train_end': int(train_end), 'baselines': {}}
            print('\n=== %s  (%d obs, %d test) ===' % (key, n_full, n_test))
            print('  %-14s %9s %9s %9s %9s' % ('baseline', 'RMSE', 'MAE', 'MAPE%', 'R2'))
            for bname, bp in base.items():
                m = metrics(y, bp)
                out[key]['baselines'][bname] = m
                print('  %-14s %9.3f %9.3f %9.4f %9.5f'
                      % (bname, m['RMSE'], m['MAE'], m['MAPE'], m['R2']))
            # every baseline against persistence
            print('  DM vs Persistence (positive = worse than persistence):')
            for bname, bp in base.items():
                if bname == 'Persistence':
                    continue
                stat, p = dm_test(y, bp, base['Persistence'])
                out[key]['baselines'][bname]['DM_vs_persistence'] = stat
                out[key]['baselines'][bname]['p_vs_persistence'] = p
                print('    %-12s DM=%+8.3f  p=%.4g  %s'
                      % (bname, stat, p, 'SIGNIFICANT' if abs(stat) > 1.96 else 'ns'))
            np.save(os.path.join(RES, 'baselines_%s.npy' % key),
                    np.vstack([base[k] for k in sorted(base)]))
            out[key]['baseline_order'] = sorted(base)
    with open(os.path.join(RES, 'baselines.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print('\nsaved ../results/baselines.json')


# ------------------------------------------------------------------ TASK 2
def task_compare():
    """Full model-vs-baseline matrix, for every prediction file present."""
    os.makedirs(RES, exist_ok=True)
    preds = [f for f in os.listdir(RES) if f.startswith('pred_') and f.endswith('.npz')]
    if not preds:
        print('No pred_*.npz files in ../results/. Waiting on the training runs.')
        return
    allrows, summary, dates = [], {}, {}
    for f in sorted(preds):
        tag = f[5:-4]                       # pred_<protocol>_<index>_<which>_<head>.npz
        d = np.load(os.path.join(RES, f))
        y = d['y_true']
        parts = tag.split('_')
        name, which = parts[1], parts[2]
        s = get_series(name, which)
        # Which dates were tested. Prefer the recorded test_index: Protocol A
        # stops one short of the series tail (VMD trims to an even length), so
        # inferring the start from the row count is off by one for every
        # Protocol A file.
        if 'test_index' in d.files:
            idx = np.asarray(d['test_index'], dtype=int)
            if len(idx) != len(y) or not np.all(np.diff(idx) == 1):
                raise SystemExit('\ntest_index in %s is not a contiguous run of '
                                 '%d dates; cannot build baselines.\n' % (f, len(y)))
            train_end = int(idx[0])
        else:
            train_end = len(s) - len(y)
        # The baselines are built from the series in ../data/ but scored
        # against y_true from the prediction file. If those came from two
        # different downloads the 10-year window has slid, the baselines are
        # scored against the wrong dates, and every DM test is meaningless --
        # silently, and usually in the model's favour. Refuse to continue.
        tol = 1e-6 * max(1.0, float(np.abs(y).max()))
        seg = s[train_end:train_end + len(y)]
        if (train_end < WINDOW or len(seg) != len(y)
                or not np.allclose(seg, y, rtol=0, atol=tol)):
            raise SystemExit(
                '\nDATA MISMATCH in %s\n'
                '  The series in ../data/series_%s_%s.npy does not end with the\n'
                '  test targets stored in the prediction file, so the baselines\n'
                '  would be scored against different dates from the model.\n'
                '  Fix: use the exact data/series_*.npy files the models were\n'
                '  trained on. Delete your own downloaded copies from ../data/,\n'
                '  put the shared ones there, then rerun this task.\n'
                % (f, name, which))
        base = all_baselines(s, train_end, len(y))

        m = metrics(y, d['y_pred'])
        summary[tag] = m
        dates[tag] = (np.arange(train_end, train_end + len(y)), y,
                      np.asarray(d['y_pred'], dtype=float))
        print('\n=== %s ===' % tag)
        print('  model    RMSE %9.3f  MAE %9.3f  MAPE %7.4f%%  R2 %8.5f'
              % (m['RMSE'], m['MAE'], m['MAPE'], m['R2']))
        for bname, bp in base.items():
            bm = metrics(y, bp)
            stat, p = dm_test(y, d['y_pred'], bp)
            verdict = ('model worse' if stat > 0 else 'model better')
            sig = 'SIGNIFICANT' if abs(stat) > 1.96 else 'ns'
            print('  vs %-12s RMSE %9.3f | DM %+8.3f p=%.4g  %s, %s'
                  % (bname, bm['RMSE'], stat, p, verdict, sig))
        allrows += comparison_table(y, {tag: d['y_pred']}, base)
    with open(os.path.join(RES, 'comparison.json'), 'w') as f:
        json.dump({'metrics': summary, 'dm_rows': allrows}, f, indent=1)
    print('\nsaved ../results/comparison.json')

    # Leakage quantification: same index+head, one-time vs walk-forward. The
    # two protocols split at different points, so they test overlapping but
    # non-identical dates. Compare them on the dates BOTH tested; row i of one
    # file and row i of the other are different days.
    print('\n=== leakage inflation (one-time vs walk-forward, common dates) ===')
    infl = {}
    for tag in summary:
        if not tag.startswith('onetime'):
            continue
        wf = 'walkforward' + tag[len('onetime'):]
        if wf not in dates:
            continue
        ia, ya, pa = dates[tag]
        ib, yb, pb = dates[wf]
        common, ka, kb = np.intersect1d(ia, ib, return_indices=True)
        if len(common) == 0:
            continue
        if not np.allclose(ya[ka], yb[kb]):
            raise SystemExit('\nprotocols disagree on the price at shared dates '
                             'for %s; they were built from different data.\n' % tag)
        a = metrics(ya[ka], pa[ka])
        b = metrics(yb[kb], pb[kb])
        key = tag[len('onetime_'):]
        infl[key] = {
            'common_dates': int(len(common)),
            'onetime_only': int(len(ia) - len(common)),
            'walkforward_only': int(len(ib) - len(common)),
            'onetime_RMSE': a['RMSE'], 'walkforward_RMSE': b['RMSE'],
            'inflation_ratio': b['RMSE'] / a['RMSE'] if a['RMSE'] else None,
            'onetime_MAPE': a['MAPE'], 'walkforward_MAPE': b['MAPE']}
        print('  %-24s %d common dates | one-time RMSE %8.3f -> walk-forward %8.3f  (x%.2f)'
              % (key, len(common), a['RMSE'], b['RMSE'], b['RMSE'] / a['RMSE']))
    if infl:
        with open(os.path.join(RES, 'leakage_inflation.json'), 'w') as f:
            json.dump(infl, f, indent=1)
        print('  saved ../results/leakage_inflation.json')


# ------------------------------------------------------------------ TASK 3
def task_figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(RES, exist_ok=True)
    plt.rcParams.update({'font.size': 9, 'figure.dpi': 200, 'savefig.bbox': 'tight'})

    bj = os.path.join(RES, 'baselines.json')
    if not os.path.exists(bj):
        print('run --task baselines first'); return
    B = json.load(open(bj))

    fig, axes = plt.subplots(1, len(B), figsize=(3.2 * len(B), 2.8), squeeze=False)
    for ax, (key, d) in zip(axes[0], sorted(B.items())):
        names = list(d['baselines'])
        vals = [d['baselines'][n]['RMSE'] for n in names]
        ax.bar(range(len(names)), vals, color='0.6')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
        ax.set_title(key, fontsize=9)
        ax.set_ylabel('RMSE')
    fig.tight_layout(); fig.savefig(os.path.join(RES, 'fig_baselines.png')); plt.close(fig)
    print('saved ../results/fig_baselines.png')

    cj = os.path.join(RES, 'comparison.json')
    if os.path.exists(cj):
        C = json.load(open(cj))['metrics']
        tags = sorted(C)
        fig, ax = plt.subplots(figsize=(max(4, 0.7 * len(tags)), 2.9))
        ax.bar(range(len(tags)), [C[t]['RMSE'] for t in tags], color='tab:blue')
        for key, d in sorted(B.items()):
            ax.axhline(d['baselines']['Persistence']['RMSE'], ls='--', lw=1,
                       color='tab:red')
        ax.set_xticks(range(len(tags)))
        ax.set_xticklabels(tags, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('RMSE')
        ax.set_title('Models against the persistence baseline (dashed)', fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(RES, 'fig_models.png')); plt.close(fig)
        print('saved ../results/fig_models.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True,
                    choices=['baselines', 'compare', 'figures'])
    a = ap.parse_args()
    {'baselines': task_baselines, 'compare': task_compare,
     'figures': task_figures}[a.task]()
