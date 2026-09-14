"""Component-level units of work, shared by the trainer and the scheduler.

A unit is one component model: (dataset tag, head, component index). Its
prediction is saved on its own in results/parts/, and a dataset's prediction
file is assembled once every one of its components exists. Kept free of torch
so the scheduler can use it without loading CUDA.
"""
import json
import os

import numpy as np

SERIES = ('SPX_High', 'SPX_Low', 'SSEC_High', 'SSEC_Low')
PROTOCOLS = ('onetime', 'walkforward')
HEADS = ('tcn', 'trm')

_n = {}


def part_path(res, tag, head, c):
    return os.path.join(res, 'parts', '%s__%s__c%02d.npz' % (tag, head, c))


def pred_path(res, tag, head):
    return os.path.join(res, 'pred_%s_%s.npz' % (tag, head))


def n_components(data, tag):
    zp = os.path.join(data, tag + '.npz')
    if zp not in _n:
        with np.load(zp) as z:
            _n[zp] = int(z['n_components'][0])
    return _n[zp]


def ready(data, tag):
    """Whether a dataset may be trained on.

    The delivered walk-forward files ended the input window at t-2 on every
    odd test day; tools/fix_walkforward_parity.py corrects them and records
    'parity_fix' in the meta file. A walk-forward file whose meta lacks that
    record is not trained on. Files without any meta (the smoke fixture) are
    built correctly and need no record.
    """
    if not tag.startswith('walkforward'):
        return True
    mp = os.path.join(data, tag + '_meta.json')
    if not os.path.exists(mp):
        return True
    try:
        with open(mp) as f:
            return 'parity_fix' in json.load(f)
    except (OSError, ValueError):
        return False


def units(data, res, only=None, include_done=False):
    """Component models still to train (every one of them with include_done).

    Ordered series by series, with both protocols of one head together, so
    the first one-time versus walk-forward comparison exists early in a long
    run instead of only at its end. Datasets that are not ready are left out.
    """
    out = []
    for s in SERIES:
        for head in HEADS:
            for p in PROTOCOLS:
                tag = '%s_%s' % (p, s)
                if only and only not in tag:
                    continue
                if not os.path.exists(os.path.join(data, tag + '.npz')):
                    continue
                if not ready(data, tag):
                    continue
                assembled = os.path.exists(pred_path(res, tag, head))
                for c in range(n_components(data, tag)):
                    saved = assembled or os.path.exists(part_path(res, tag, head, c))
                    if include_done or not saved:
                        out.append((tag, head, c))
    return out
