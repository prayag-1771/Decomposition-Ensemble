"""Check the last-value scaling on the components that exposed the problem,
before committing the GPU to a full run.

Each unit is trained with the scaling now in run_training.train_part, and its
test forecasts are scored in component units against three references:
    persistence  the component's own last input value
    minmax       the archived forecast made with the source model's min-max
                 scaling (results/minmax_scaling_run/parts), where available
    std ratio    standard deviation of the forecast over that of the truth; a
                 value near zero means the network collapsed to a constant

    python validate_scaling.py            5 units, 5 processes, full budget
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
UNITS = [('onetime_SPX_High', 'tcn', 0), ('onetime_SPX_High', 'trm', 0),
         ('onetime_SPX_High', 'tcn', 14), ('onetime_SPX_High', 'trm', 14),
         ('onetime_SSEC_High', 'trm', 10)]


def one(unit):
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(HERE, '..', 'shared'))
    import run_training as rt
    from models import train_component
    tag, head, c = unit
    Xtr, ytr, Xte = rt.load(tag)[:3]
    t0 = time.time()
    sXtr, sytr, sXte, unscale = rt.scale_component(Xtr[c], ytr[c], Xte[c])
    p_te, p_tr, info = train_component(sXtr, sytr, sXte, head=head, device='cuda',
                                       epochs=150, seed=c, fused=True)
    p_te, _ = unscale(p_te, p_tr)
    z = np.load(os.path.join(rt.DATA, tag + '.npz'))
    truth = z['yte_%d' % c].astype(float)
    last = z['Xte_%d' % c][:, -1, 0].astype(float)
    n = min(len(truth), len(p_te))
    rmse = lambda a: float(np.sqrt(np.mean((a[:n] - truth[:n]) ** 2)))
    old = os.path.join(rt.RES, 'minmax_scaling_run', 'parts', '%s__%s__c%02d.npz' % (tag, head, c))
    old_rmse = rmse(np.load(old)['p_te']) if os.path.exists(old) else float('nan')
    return ('%-18s %s c%02d | %3d epochs %4.0f s | RMSE new %8.2f  minmax %8.2f  persistence %8.2f'
            ' | std ratio %.2f' % (tag, head, c, info['epochs_run'], time.time() - t0, rmse(p_te),
                                   old_rmse, rmse(last), float(np.std(p_te[:n]) / np.std(truth[:n]))))


if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=len(UNITS)) as pool:
        for line in pool.map(one, UNITS):
            print(line, flush=True)
