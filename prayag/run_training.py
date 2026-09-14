"""PRAYAG'S DRIVER -- trains both heads on whichever decompositions exist.

    python run_training.py --list
    python run_training.py --tag onetime_SPX_High --head tcn   one dataset, serially
    python run_training.py --all                               everything, serially
    python run_training.py --worker                            one GPU worker (run_all starts several)
    python run_training.py --assemble                          prediction files from saved components

The unit of work is one component model. Each is saved to
../results/parts/<tag>__<head>__c<k>.npz as soon as it finishes, and a
dataset's prediction file ../results/pred_<tag>_<head>.npz, which Ayush's
compare task reads, is assembled once all of its components exist. A component
is trained by the same function with the same seed however the work is split,
so the split decides where a model is trained, not how.
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'shared'))
from models import train_component                       # noqa: E402
from evaluate import metrics, bias_correct               # noqa: E402
import parts                                             # noqa: E402

DATA = os.path.join(HERE, '..', 'data')
RES = os.path.join(HERE, '..', 'results')
SCALING = 'last_value'

_loaded = {}


def load(tag):
    if tag not in _loaded:
        z = np.load(os.path.join(DATA, tag + '.npz'))
        C = int(z['n_components'][0])
        Xtr = [z['Xtr_%d' % c] for c in range(C)]
        ytr = [z['ytr_%d' % c] for c in range(C)]
        Xte = [z['Xte_%d' % c] for c in range(C)]
        # test_index records exactly which dates were tested. The two protocols
        # split at different points and Protocol A stops short of the series
        # tail, so the evaluation cannot infer the dates from the row count;
        # carry it.
        tidx = z['test_index'] if 'test_index' in z.files else None
        _loaded[tag] = (Xtr, ytr, Xte, z['target_test'], z['target_train'], C, tidx)
    return _loaded[tag]


def scale_component(Xtr_c, ytr_c, Xte_c):
    """Express every window relative to its own last value.

    The source model min-max scales each component on its training rows. That
    cannot represent a test level outside the training range, and the
    components that carry the index trend leave it: on the S&P 500 the trend
    component lies entirely above its training maximum for the whole test
    period, and networks trained on [0, 1] under-forecast it by hundreds of
    points (measured: TCN bias -145, TRM bias -776 on that one component).

    Subtracting each window's last value makes the input independent of the
    level, and the network forecasts the change to the next value instead of
    the value itself. For oscillatory components a 30-day window spans several
    cycles, so the phase and amplitude the level carried are still visible in
    the window's shape. Both scales are fitted on training windows only, and
    the transform is the same for both heads and both protocols.

    Returns scaled arrays and a function mapping network outputs back to
    component values: (test values, training values).
    """
    Xtr_c = np.asarray(Xtr_c, dtype=np.float64)
    ytr_c = np.asarray(ytr_c, dtype=np.float64).ravel()
    Xte_c = np.asarray(Xte_c, dtype=np.float64)
    last_tr, last_te = Xtr_c[:, -1, 0], Xte_c[:, -1, 0]
    dev_tr = Xtr_c - last_tr[:, None, None]
    s_in = float(dev_tr.std()) or 1.0
    s_out = float((ytr_c - last_tr).std()) or 1.0
    sXtr = (dev_tr / s_in).astype(np.float32)
    sytr = ((ytr_c - last_tr) / s_out).astype(np.float32)
    sXte = ((Xte_c - last_te[:, None, None]) / s_in).astype(np.float32)

    def unscale(p_te, p_tr):
        return (last_te + np.asarray(p_te, dtype=np.float64) * s_out,
                last_tr + np.asarray(p_tr, dtype=np.float64) * s_out)
    return sXtr, sytr, sXte, unscale


def train_part(tag, head, c, device, epochs, seed=0, fused=False, amp=False):
    """Train component c of one dataset with one head and save it by itself."""
    Xtr, ytr, Xte = load(tag)[:3]
    t0 = time.time()
    sXtr, sytr, sXte, unscale = scale_component(Xtr[c], ytr[c], Xte[c])

    on_cuda = device == 'cuda' and torch.cuda.is_available()
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()
    p_te, p_tr, info = train_component(
        sXtr, sytr, sXte, head=head, device=device,
        epochs=epochs, seed=seed + c, fused=fused, amp=amp)
    peak = torch.cuda.max_memory_reserved() / 2 ** 20 if on_cuda else 0.0
    secs = time.time() - t0
    p_te, p_tr = unscale(p_te, p_tr)

    out = parts.part_path(RES, tag, head, c)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out[:-4] + '.tmp%d.npz' % os.getpid()
    np.savez(tmp, p_te=p_te, p_tr=p_tr,
             final_loss=info['final_loss'], epochs_run=info['epochs_run'],
             seconds=secs, peak_mb=peak, scaling=SCALING,
             epochs=epochs, seed=seed, fused=bool(fused), amp=bool(amp))
    os.replace(tmp, out)        # a half-written part is never taken for a finished one
    if on_cuda:
        # Hand cached blocks back to the driver. Otherwise every worker keeps
        # the high-water mark of its largest component, and five workers on the
        # 4 GB card outgrow it after their first component: Windows then pages
        # GPU memory to system RAM and training all but stops (seen: twelve
        # minutes without a single component finishing).
        torch.cuda.empty_cache()
    return info, peak, secs


def assemble(tag, head, bias):
    """Sum a dataset's saved components into its prediction file."""
    Xtr, _, _, tgt_te, tgt_tr, C, tidx = load(tag)
    paths = [parts.part_path(RES, tag, head, c) for c in range(C)]
    missing = [c for c, p in enumerate(paths) if not os.path.exists(p)]
    if missing:
        print('  %s %s: components %s not trained yet' % (tag, head, missing))
        return None
    zs = [np.load(p) for p in paths]
    setting = {k: [z[k].item() for z in zs] for k in ('epochs', 'seed', 'fused', 'amp')}
    setting['scaling'] = [str(z['scaling']) if 'scaling' in z.files else 'minmax' for z in zs]
    mixed = [k for k, v in setting.items() if len(set(v)) > 1]
    if mixed:
        sys.exit('%s %s: components were trained with different %s; delete its '
                 'parts and train them again' % (tag, head, mixed))

    preds_te = np.zeros((len(tgt_te), C), dtype=float)
    preds_tr = np.zeros((Xtr[0].shape[0], C), dtype=float)
    for c, z in enumerate(zs):
        n = min(len(z['p_te']), preds_te.shape[0])
        preds_te[:n, c] = z['p_te'][:n]
        preds_tr[:len(z['p_tr']), c] = z['p_tr']
    y_pred = preds_te.sum(axis=1)
    y_true = np.asarray(tgt_te, dtype=float)[:len(y_pred)]
    y_pred = y_pred[:len(y_true)]

    print('\n=== %s | head=%s | %d components ===' % (tag, head, C))
    if bias:
        # fitted on the last 10% of TRAIN, applied identically to both heads
        k = max(10, int(0.1 * preds_tr.shape[0]))
        val_pred = preds_tr[-k:].sum(axis=1)
        val_true = np.asarray(tgt_tr, dtype=float)[-k:]
        y_pred, a, b = bias_correct(val_pred, val_true, y_pred)
        print('  bias correction fitted on train tail: a=%.4f b=%.4f' % (a, b))

    m = metrics(y_true, y_pred)
    print('  RMSE %9.3f | MAE %9.3f | MAPE %7.4f%% | R2 %8.5f'
          % (m['RMSE'], m['MAE'], m['MAPE'], m['R2']))

    out = parts.pred_path(RES, tag, head)
    extra = {}
    if tidx is not None:
        extra['test_index'] = np.asarray(tidx)[:len(y_true)]
    np.savez_compressed(out, y_true=y_true, y_pred=y_pred,
                        components=preds_te[:len(y_true)], **extra)
    secs = [float(z['seconds']) for z in zs]
    with open(out.replace('.npz', '_meta.json'), 'w') as f:
        json.dump({'tag': tag, 'head': head, 'metrics': m,
                   'component_losses': [float(z['final_loss']) for z in zs],
                   'component_epochs': [int(z['epochs_run']) for z in zs],
                   'component_seconds': [round(s, 1) for s in secs],
                   'peak_gpu_mb': round(max(float(z['peak_mb']) for z in zs), 1),
                   'bias_corrected': bool(bias), 'scaling': setting['scaling'][0],
                   'epochs': setting['epochs'][0], 'seed': setting['seed'][0],
                   'fused_adam': bool(setting['fused'][0]),
                   'amp': bool(setting['amp'][0]),
                   'seconds': round(sum(secs), 1)}, f, indent=1)
    print('  wrote %s' % os.path.basename(out))
    return m


def run(tag, head, device, epochs, bias, seed=0, fused=False, amp=False):
    """One dataset, every component in this process, then assemble."""
    C = load(tag)[5]
    print('\n=== %s | head=%s | %d components ===' % (tag, head, C))
    t0 = time.time()
    for c in range(C):
        if os.path.exists(parts.part_path(RES, tag, head, c)):
            print('  component %2d/%d  already saved' % (c + 1, C))
            continue
        info, _, _ = train_part(tag, head, c, device, epochs, seed, fused, amp)
        print('  component %2d/%d  final loss %.6f  (%d epochs)'
              % (c + 1, C, info['final_loss'], info['epochs_run']))
    m = assemble(tag, head, bias)
    print('  %.0f s total' % (time.time() - t0))
    return m


def _claim(tag, head, c):
    """Reserve one unit for this process atomically; None if another has it."""
    done = parts.part_path(RES, tag, head, c)
    lock = done[:-4] + '.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    if os.path.exists(done):        # another worker finished it since we looked
        os.remove(lock)
        return None
    return lock


def worker(device, epochs, seed=0, fused=False, amp=False, only=None):
    """Claim and train free units until none is left for this process."""
    os.makedirs(os.path.join(RES, 'parts'), exist_ok=True)
    failed = set()
    while True:
        unit = lock = None
        for u in parts.units(DATA, RES, only):
            if u in failed:
                continue
            lock = _claim(*u)
            if lock:
                unit = u
                break
        if unit is None:
            return 1 if failed else 0
        tag, head, c = unit
        try:
            info, peak, secs = train_part(tag, head, c, device, epochs, seed, fused, amp)
            print('%s  %-22s %s c%02d  loss %.6f  %3d epochs  %4.0f s  peak %4.0f MB'
                  % (time.strftime('%H:%M:%S'), tag, head, c, info['final_loss'],
                     info['epochs_run'], secs, peak), flush=True)
        except Exception:
            # typically running out of GPU memory; leave the unit to the
            # scheduler's one-worker retry rather than taking the worker down
            failed.add(unit)
            print('%s  FAILED %s %s c%02d' % (time.strftime('%H:%M:%S'), tag, head, c),
                  flush=True)
            traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            try:
                os.remove(lock)
            except OSError:
                pass


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag')
    ap.add_argument('--head', choices=list(parts.HEADS))
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--worker', action='store_true',
                    help='claim and train free components until none is left')
    ap.add_argument('--assemble', action='store_true',
                    help='build prediction files from saved components')
    ap.add_argument('--only', default=None, help='substring filter on the tag (worker)')
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--bias', action='store_true',
                    help='apply the OLS bias correction to BOTH heads')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--fused', action='store_true', help='single-kernel Adam (CUDA)')
    ap.add_argument('--amp', action='store_true', help='float16 autocast with loss scaling')
    a = ap.parse_args()

    tags = sorted(f[:-4] for f in os.listdir(DATA)
                  if f.endswith('.npz') and not f.startswith('series'))
    if a.list:
        print('decomposition files available in ../data/:')
        for t in tags:
            print('   ', t)
        sys.exit()

    if a.assemble:
        built = 0
        for t in tags:
            for h in parts.HEADS:
                if (a.tag and t != a.tag) or (a.head and h != a.head):
                    continue
                if not a.tag and os.path.exists(parts.pred_path(RES, t, h)):
                    continue
                if assemble(t, h, a.bias) is not None:
                    built += 1
        print('\nassembled %d prediction file(s)' % built)
        sys.exit()

    # Never fall back to the CPU silently. On this machine a CPU-only torch
    # build once left the GPU idle for a whole session; a run that is 3-4x
    # slower without saying so is worse than one that refuses to start.
    if a.device == 'cuda' and not torch.cuda.is_available():
        sys.exit('CUDA requested but not available (torch %s). Install the CUDA '
                 'build of PyTorch, or pass --device cpu deliberately.'
                 % torch.__version__)
    print('device: %s | torch %s | %s'
          % (a.device, torch.__version__,
             torch.cuda.get_device_name(0) if a.device == 'cuda' else 'cpu'), flush=True)
    if a.worker:
        sys.exit(worker(a.device, a.epochs, fused=a.fused, amp=a.amp, only=a.only))
    if a.all:
        for t in tags:
            for h in parts.HEADS:
                run(t, h, a.device, a.epochs, a.bias, fused=a.fused, amp=a.amp)
    else:
        run(a.tag, a.head, a.device, a.epochs, a.bias, fused=a.fused, amp=a.amp)
