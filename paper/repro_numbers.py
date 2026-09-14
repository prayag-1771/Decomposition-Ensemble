"""Counts for the reproducibility subsection, read from the saved files
rather than worked out by hand.

    python repro_numbers.py   -> repro_numbers.json (and a printout)
"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
DATA = os.path.join(ROOT, 'data')
RES = os.path.join(ROOT, 'results')
SERIES = ('SPX_High', 'SPX_Low', 'SSEC_High', 'SSEC_Low')

out = {'series': {}, 'observations': {}, 'walkforward_decompositions': {},
       'components': {}, 'test_days': {}}
n_networks = 0
for s in SERIES:
    x = np.load(os.path.join(DATA, 'series_%s.npy' % s))
    out['observations'][s] = int(len(x))
    for p in ('onetime', 'walkforward'):
        tag = '%s_%s' % (p, s)
        meta = json.load(open(os.path.join(DATA, tag + '_meta.json')))
        z = np.load(os.path.join(DATA, tag + '.npz'))
        C = int(z['n_components'][0])
        out['components'][tag] = C
        out['test_days'][tag] = int(len(z['test_index']))
        if p == 'walkforward':
            out['walkforward_decompositions'][s] = int(meta['n_decompositions'])
        for head in ('tcn', 'trm'):
            parts = glob.glob(os.path.join(RES, 'parts', '%s__%s__c*.npz' % (tag, head)))
            assert len(parts) == C, (tag, head, len(parts), C)
            n_networks += len(parts)
            pm = json.load(open(os.path.join(RES, 'pred_%s_%s_meta.json' % (tag, head))))
            assert len(pm['component_losses']) == C
out['networks_trained'] = n_networks
out['prediction_files'] = len(glob.glob(os.path.join(RES, 'pred_*.npz')))
out['walkforward_decompositions_total'] = sum(out['walkforward_decompositions'].values())
rn = json.load(open(os.path.join(HERE, 'results_numbers.json')))['series']
out['common_dates'] = {s: rn[s]['dates'] for s in SERIES}
out['seconds_training'] = sum(json.load(open(f))['seconds']
                              for f in glob.glob(os.path.join(RES, 'pred_*_meta.json')))
ld = json.load(open(os.path.join(HERE, 'leakage_diagnostic.json')))
out['diagnostic_dates'] = {s: ld[s]['common_dates'] for s in SERIES}
json.dump(out, open(os.path.join(HERE, 'repro_numbers.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
