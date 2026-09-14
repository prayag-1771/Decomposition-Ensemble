"""Every number quoted in the prose, recomputed from the result files.

Each entry pairs the value as written in the text with the value the files
give, at the precision used in the text. Any disagreement fails the run, so a
corrected result cannot leave a stale number behind in a sentence.

    python check_numbers.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, 'results_numbers.json')))
S, X = R['summary'], R['series']
D = json.load(open(os.path.join(HERE, 'leakage_diagnostic.json')))
B = json.load(open(os.path.join(HERE, 'boundary_amplitude.json')))
SER = ['SPX_High', 'SPX_Low', 'SSEC_High', 'SSEC_Low']
HEADS = ['tcn', 'trm']

checks = []


def claim(text, written, computed, nd=1):
    ok = np.allclose(np.round(np.asarray(written, float), nd), np.round(np.asarray(computed, float), nd))
    checks.append((ok, text, written, np.round(np.asarray(computed, float), nd + 1).tolist()))


def rng(vals):
    return [min(vals), max(vals)]


ot = [X[s]['models']['onetime_' + h] for s in SER for h in HEADS]
wf = [X[s]['models']['walkforward_' + h] for s in SER for h in HEADS]
pe = [X[s]['baselines']['Persistence'] for s in SER]

claim('full series RMSE 5.0 to 10.7', [5.0, 10.7], rng([m['RMSE'] for m in ot]))
claim('full series MAPE 0.066 to 0.134', [0.066, 0.134], rng([m['MAPE'] for m in ot]), 3)
claim('full series R2 0.9992 to 0.9999', [0.9992, 0.9999], rng([m['R2'] for m in ot]), 4)
claim('walk forward RMSE 42.5 to 68.5', [42.5, 68.5], rng([m['RMSE'] for m in wf]))
claim('walk forward MAPE 0.72 to 0.83', [0.72, 0.83], rng([m['MAPE'] for m in wf]), 2)
claim('walk forward R2 0.981 to 0.991', [0.981, 0.991], rng([m['R2'] for m in wf]), 3)
claim('persistence RMSE 36.2 to 57.9', [36.2, 57.9], rng([m['RMSE'] for m in pe]))
claim('persistence MAPE 0.52 to 0.67', [0.52, 0.67], rng([m['MAPE'] for m in pe]), 2)
claim('abstract MAPE 0.07 to 0.13', [0.07, 0.13], rng([m['MAPE'] for m in ot]), 2)

rho = [X[s]['leakage'][h]['rho'] for s in SER for h in HEADS]
claim('rho 4.9 to 10.4', [4.9, 10.4], rng(rho))
claim('rho geometric mean 7.2', 7.2, S['rho_geomean'])
claim('MDM walk forward vs full series +4.5 to +12.4', [4.5, 12.4],
      rng([X[s]['leakage'][h]['DM_wf_vs_ot'] for s in SER for h in HEADS]))
claim('rho TCN 6.4 to 10.4', [6.4, 10.4], rng([X[s]['leakage']['tcn']['rho'] for s in SER]))
claim('rho TRM 4.9 to 8.3', [4.9, 8.3], rng([X[s]['leakage']['trm']['rho'] for s in SER]))
claim('all eight walk forward significantly worse', 8, S['wf_significantly_worse'], 0)

drift = [abs(X[s]['baselines']['Drift']['RMSE'] / X[s]['baselines']['Persistence']['RMSE'] - 1) * 100 for s in SER]
checks.append((max(drift) <= 0.3, 'drift within 0.3% of persistence', 0.3, round(max(drift), 2)))
for b, lo, hi in (('MA-5', 1.5, 1.6), ('EWMA-10', 1.8, 1.9), ('MA-20', 2.7, 2.9)):
    claim('%s %.1f to %.1f times persistence' % (b, lo, hi), [lo, hi],
          rng([X[s]['baselines'][b]['RMSE'] / X[s]['baselines']['Persistence']['RMSE'] for s in SER]))

claim('MDM full series vs persistence -4.5 to -10.1', [-10.1, -4.5], rng([m['DM_vs_persistence'] for m in ot]))
claim('MDM walk forward vs persistence +3.5 to +7.4', [3.5, 7.4], rng([m['DM_vs_persistence'] for m in wf]))
claim('full series beat persistence 8 of 8', 8, S['beats_persistence_onetime'], 0)
claim('walk forward worse than persistence 8 of 8', 8, S['worse_than_persistence_walkforward'], 0)
worse = [(X[s]['models']['walkforward_' + h]['RMSE'] / X[s]['baselines']['Persistence']['RMSE'] - 1) * 100
         for s in SER for h in HEADS]
claim('walk forward 13% to 30% worse than persistence', [13, 30], rng(worse), 0)

claim('heads full series MDM -2.9 to -6.1', [-6.1, -2.9], rng([X[s]['heads']['onetime']['DM_tcn_vs_trm'] for s in SER]))
checks.append((max(X[s]['heads']['onetime']['p'] for s in SER) < 0.01, 'heads full series p < 0.01', 0.01,
               round(max(X[s]['heads']['onetime']['p'] for s in SER), 4)))
claim('heads walk forward p 0.56, 0.52, 0.97 on three series', [0.56, 0.52, 0.97],
      [X[s]['heads']['walkforward']['p'] for s in ('SPX_High', 'SSEC_High', 'SSEC_Low')], 2)
claim('heads walk forward S&P 500 low MDM +4.2', 4.2, X['SPX_Low']['heads']['walkforward']['DM_tcn_vs_trm'])
diff = [abs(X[s]['models']['walkforward_tcn']['RMSE'] / X[s]['models']['walkforward_trm']['RMSE'] - 1) * 100
        for s in ('SPX_High', 'SSEC_High', 'SSEC_Low')]
checks.append((max(diff) < 0.7, 'walk forward head RMSE differ < 0.7% on three series', 0.7, round(max(diff), 2)))

ip = [abs(X[s]['input_persistence_' + p]['RMSE'] / X[s]['baselines']['Persistence']['RMSE'] - 1) * 100
      for s in SER for p in ('onetime', 'walkforward')]
checks.append((max(ip) <= 0.7, 'input persistence within 0.7% of persistence', 0.7, round(max(ip), 2)))
lost = [(X[s]['models']['walkforward_' + h]['RMSE'] / X[s]['input_persistence_walkforward']['RMSE'] - 1) * 100
        for s in SER for h in HEADS]
claim('networks lost 12% to 29% relative to input persistence', [12, 29], rng(lost), 0)

claim('diagnostic r full series -0.58 to -0.53', [-0.58, -0.53], rng([D[s]['onetime']['corr_hf_nextchange'] for s in SER]), 2)
checks.append((max(D[s]['onetime']['p'] for s in SER) < 1e-35, 'diagnostic p < 1e-35', '1e-35',
               '%.1e' % max(D[s]['onetime']['p'] for s in SER)))
claim('diagnostic skill 51% to 60%', [51, 60], rng([100 * D[s]['onetime']['oos_skill_vs_persistence'] for s in SER]), 0)
claim('diagnostic r walk forward 0.00 to 0.10', [0.00, 0.10], rng([D[s]['walkforward']['corr_hf_nextchange'] for s in SER]), 2)
claim('diagnostic walk forward skill -0.9% to -3.0%', [-3.0, -0.9],
      rng([100 * D[s]['walkforward']['oos_skill_vs_persistence'] for s in SER]), 1)
claim('SSEC high walk forward r = 0.10, p = 0.03', [0.10, 0.03],
      [D['SSEC_High']['walkforward']['corr_hf_nextchange'], D['SSEC_High']['walkforward']['p']], 2)
claim('boundary amplitude ratio 2.5 to 3.3', [2.5, 3.3], rng([B[s]['ratio'] for s in SER]))
claim('common dates 496 and 478', [496, 478], [X['SPX_High']['dates'], X['SSEC_High']['dates']], 0)

bad = [c for c in checks if not c[0]]
for ok, text, written, computed in checks:
    print('%s  %-58s written %-28s computed %s' % ('ok ' if ok else 'BAD', text, written, computed))
print('\n%d checks, %d disagree' % (len(checks), len(bad)))
sys.exit(1 if bad else 0)
