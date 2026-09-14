"""Everything downstream of Veer, in one resumable command.

    python run_all.py --check     verify Veer's files before committing hours
    python run_all.py --smoke     whole chain on synthetic data, a few minutes
    python run_all.py             the real run (resumable; safe to re-launch)

Stages: train every component of every dataset with both heads, then the
baselines, the model-vs-baseline comparison, the leakage ratio and the
figures. Each component model is saved the moment it finishes, so an
interrupted run resumes where it stopped rather than starting over.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
RES = os.path.join(HERE, 'results')
sys.path.insert(0, os.path.join(HERE, 'shared'))
import parts                                             # noqa: E402

SERIES = list(parts.SERIES)
PROTOCOLS = list(parts.PROTOCOLS)
HEADS = list(parts.HEADS)


def tags():
    return ['%s_%s' % (p, s) for p in PROTOCOLS for s in SERIES]


# ------------------------------------------------------------------- checking
def check(verbose=True):
    """Veer's files must be present, carry test_index, and agree with the
    price series at every test date. Anything else and the whole run is
    wasted, so it is worth thirty seconds up front."""
    problems, info = [], []
    for s in SERIES:
        sp = os.path.join(DATA, 'series_%s.npy' % s)
        have_series = os.path.exists(sp)
        if not have_series:
            problems.append('missing data/series_%s.npy' % s)
        series = np.load(sp) if have_series else None
        for p in PROTOCOLS:
            zp = os.path.join(DATA, '%s_%s.npz' % (p, s))
            if not os.path.exists(zp):
                problems.append('missing data/%s_%s.npz' % (p, s))
                continue
            if series is None:
                continue          # already reported; cannot cross-check yet
            z = np.load(zp)
            need = ['n_components', 'target_test', 'target_train']
            miss = [k for k in need if k not in z.files]
            if miss:
                problems.append('%s_%s.npz lacks %s' % (p, s, miss))
                continue
            C = int(z['n_components'][0])
            nte = len(z['target_test'])
            if 'test_index' not in z.files:
                problems.append('%s_%s.npz has no test_index (ask Veer to run '
                                'recover_test_indices.py)' % (p, s))
            else:
                idx = z['test_index'].astype(int)
                if len(idx) != nte:
                    problems.append('%s_%s.npz: test_index %d vs target_test %d'
                                    % (p, s, len(idx), nte))
                elif idx.max() >= len(series):
                    problems.append('%s_%s.npz: test_index runs past the series' % (p, s))
                elif not np.allclose(series[idx], z['target_test']):
                    problems.append('%s_%s.npz does not match series_%s.npy '
                                    '(different download)' % (p, s, s))
                elif not np.all(np.diff(idx) == 1):
                    problems.append('%s_%s.npz test_index is not contiguous' % (p, s))
            shapes_ok = all('Xtr_%d' % c in z.files and 'Xte_%d' % c in z.files
                            for c in range(C))
            if not shapes_ok:
                problems.append('%s_%s.npz is missing component arrays' % (p, s))
            info.append((p + '_' + s, C, int(z['Xtr_0'].shape[0]) if shapes_ok else 0, nte))
    if verbose:
        print('=' * 72)
        print("CHECKING VEER'S FILES")
        print('=' * 72)
        if info:
            print('  %-24s %6s %8s %8s' % ('dataset', 'comps', 'train', 'test'))
            for t, C, ntr, nte in info:
                print('  %-24s %6d %8d %8d' % (t, C, ntr, nte))
        print()
        if problems:
            for p in problems:
                print('  PROBLEM: ' + p)
        else:
            total = sum(c for _, c, _, _ in info)
            print('  all files present and consistent')
            print('  %d components x %d heads = %d models to train'
                  % (total, len(HEADS), total * len(HEADS)))
    return problems, info


# -------------------------------------------------------------------- smoke
def smoke():
    """Build tiny Veer-shaped files and push them through the whole chain."""
    import shutil
    sm = os.path.join(HERE, '_smoke')
    shutil.rmtree(sm, ignore_errors=True)
    os.makedirs(os.path.join(sm, 'data'))
    os.makedirs(os.path.join(sm, 'results'))
    # the drivers resolve data/ and results/ relative to their own location,
    # so the sandbox needs its own copy of the code
    for d in ('shared', 'prayag', 'ayush'):
        shutil.copytree(os.path.join(HERE, d), os.path.join(sm, d),
                        ignore=shutil.ignore_patterns('__pycache__'))
    # Components that sum exactly to the price, as real decompositions do: a
    # trend carrying the price level plus two oscillations near zero. A correct
    # pipeline then forecasts near the price; a broken rescale or summation
    # forecasts near zero, which is ~100% error and cannot be missed.
    rng = np.random.default_rng(0)
    W, C, nte, n = 30, 3, 40, 260
    t = np.arange(n)
    for s in SERIES:
        trend = 5000 + 0.5 * t + np.cumsum(rng.normal(0, 3, n))
        slow = 30 * np.sin(2 * np.pi * t / 40)
        fast = 10 * np.sin(2 * np.pi * t / 9) + rng.normal(0, 2, n)
        comps = np.stack([fast, slow, trend], axis=1)
        series = comps.sum(axis=1)
        np.save(os.path.join(sm, 'data', 'series_%s.npy' % s), series)
        if s != SERIES[0]:
            continue        # a series for every index, so baselines need no download
        # Protocol A stops one short of the tail, as Veer's files do
        for p, train_end in (('onetime', n - nte - 1), ('walkforward', n - nte)):
            idx = np.arange(train_end, train_end + nte)
            pay = {'n_components': np.array([C]),
                   'target_test': series[idx],
                   'target_train': series[W:train_end],
                   'test_index': idx}
            for c in range(C):
                col = comps[:, c]
                pay['Xtr_%d' % c] = np.stack([col[i - W:i, None] for i in range(W, train_end)]).astype(np.float32)
                pay['ytr_%d' % c] = col[W:train_end].astype(np.float32)
                pay['Xte_%d' % c] = np.stack([col[k - W:k, None] for k in idx]).astype(np.float32)
                pay['yte_%d' % c] = col[idx].astype(np.float32)
            np.savez_compressed(os.path.join(sm, 'data', '%s_%s.npz' % (p, s)), **pay)
    print('smoke fixtures in %s' % sm)
    return sm


def smoke_verdict(root):
    """Pass on sane numbers, not merely on the absence of a crash."""
    res = os.path.join(root, 'results')
    C = json.load(open(os.path.join(res, 'comparison.json')))['metrics']
    L = json.load(open(os.path.join(res, 'leakage_inflation.json')))
    print()
    print('=' * 72)
    print('SMOKE VERDICT')
    print('=' * 72)
    ok = True
    for k, m in sorted(C.items()):
        good = m['MAPE'] < 10.0
        ok = ok and good
        print('  %-28s MAPE %8.3f%%   %s'
              % (k, m['MAPE'], 'ok' if good else 'TOO HIGH: rescale or sum is broken'))
    for k, v in sorted(L.items()):
        good = v['common_dates'] == 39
        ok = ok and good
        print('  %-28s %d common dates (expect 39)   %s'
              % (k, v['common_dates'], 'ok' if good else 'WRONG'))
    carried = all('test_index' in np.load(os.path.join(res, 'pred_%s.npz' % k)).files
                  for k in C)
    ok = ok and carried
    print('  test_index in every prediction file: %s' % ('yes' if carried else 'NO'))
    print()
    print('  SMOKE %s' % ('PASS' if ok else 'FAIL'))
    return 0 if ok else 1


# -------------------------------------------------------------------- stages
def _gpu():
    """(used MB, total MB, utilisation %) from nvidia-smi, or None."""
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10).stdout
        used, total, util = (float(v) for v in out.strip().splitlines()[0].split(','))
        return used, total, util
    except Exception:
        return None


def _run_workers(root, n, epochs, device, only, fused, amp, poll):
    """Start n workers on the shared queue and report progress until all exit."""
    data = os.path.join(root, 'data')
    res = os.path.join(root, 'results')
    cmd = ([sys.executable, os.path.join(root, 'prayag', 'run_training.py'),
            '--worker', '--epochs', str(epochs), '--device', device]
           + (['--only', only] if only else [])
           + (['--fused'] if fused else []) + (['--amp'] if amp else []))
    env = dict(os.environ)
    # several processes share the cores; one thread each avoids them fighting
    # over the same ones
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    start = len(parts.units(data, res, only))
    procs = []
    for i in range(n):
        fh = open(os.path.join(res, 'logs', 'worker_%d.log' % i), 'a', encoding='utf-8')
        procs.append((subprocess.Popen(cmd, cwd=os.path.join(root, 'prayag'), env=env,
                                       stdout=fh, stderr=subprocess.STDOUT), fh))
    t0, shown, warned = time.time(), -1, False
    while True:
        running = any(p.poll() is None for p, _ in procs)
        left = len(parts.units(data, res, only))
        done = start - left
        if done != shown or not running:
            shown = done
            el = time.time() - t0
            g = _gpu()
            print('  [%3d/%d] %6.1f min elapsed, eta %s%s'
                  % (done, start, el / 60,
                     '%5.1f min' % (el / done * left / 60) if done else '    ?',
                     '   GPU %d%%, %d of %d MB' % (g[2], g[0], g[1]) if g else ''),
                  flush=True)
            # Past ~90% of the card, Windows pages GPU memory to system RAM
            # and every worker slows to a crawl.
            if g and g[0] > 0.9 * g[1] and not warned:
                warned = True
                print('  WARNING: GPU memory nearly full; expect paging and a sharp '
                      'slowdown. Re-run with fewer --workers.', flush=True)
        if not running:
            break
        time.sleep(poll)
    for p, fh in procs:
        fh.close()
    return max(p.returncode for p, _ in procs)


def train_all(epochs, device, only=None, root=HERE, bias=False, workers=5,
              fused=False, amp=False, poll=20):
    """Train every component model not already saved, then assemble the
    prediction files.

    The unit of work is one component model, not a whole dataset. `workers`
    long-lived processes share one queue of (dataset, head, component) units
    and each claims the next free unit, so no GPU slot waits while another
    worker still has a long dataset ahead of it, and an interrupted run loses
    at most the component each worker was on. Every unit is trained by the
    same function with the same per-component seed as a serial run; the split
    decides where a model is trained, not how.

    Workers hand their cached GPU memory back after every component. The
    earlier dataset-level scheduler did not, and its five workers outgrew the
    4 GB card after their first component: Windows paged GPU memory to system
    RAM and no component finished for twelve minutes.
    """
    data = os.path.join(root, 'data')
    res = os.path.join(root, 'results')
    logs = os.path.join(res, 'logs')
    pdir = os.path.join(res, 'parts')
    os.makedirs(logs, exist_ok=True)
    os.makedirs(pdir, exist_ok=True)
    # a killed run leaves its claims and half-written files behind
    for f in os.listdir(pdir):
        if f.endswith('.lock') or '.tmp' in f:
            os.remove(os.path.join(pdir, f))

    waiting = [t for t in tags() if (not only or only in t)
               and os.path.exists(os.path.join(data, t + '.npz')) and not parts.ready(data, t)]
    if waiting:
        print('  not trained yet, walk-forward windows still uncorrected: %s' % ', '.join(waiting),
              flush=True)
    total = len(parts.units(data, res, only, include_done=True))
    if not total:
        print('  no decomposition files found')
        return 1
    left = parts.units(data, res, only)
    if left:
        print('  %d component models, %d already saved; %d GPU workers; logs in %s'
              % (total, total - len(left), workers, logs), flush=True)
        _run_workers(root, max(1, workers), epochs, device, only, fused, amp, poll)
        left = parts.units(data, res, only)
        if left and workers > 1:
            print('\n  %d component(s) did not finish; retrying with one worker'
                  % len(left), flush=True)
            _run_workers(root, 1, epochs, device, only, fused, amp, poll)
            left = parts.units(data, res, only)
        if left:
            print('\n  %d component(s) still failing; the reason is in %s'
                  % (len(left), logs))
            for u in left[:10]:
                print('    %s %s c%02d' % u)
            return 1
    else:
        print('  every component model already saved')

    r = subprocess.run([sys.executable, os.path.join(root, 'prayag', 'run_training.py'),
                        '--assemble'] + (['--bias'] if bias else []),
                       cwd=os.path.join(root, 'prayag'))
    return r.returncode


def evaluate(root=HERE):
    for task in ('baselines', 'compare', 'figures'):
        print('\n--- evaluation: %s ---' % task, flush=True)
        r = subprocess.run([sys.executable, 'run_evaluation.py', '--task', task],
                           cwd=os.path.join(root, 'ayush'))
        if r.returncode != 0:
            return r.returncode
    return 0


def summarise(root=HERE):
    res = os.path.join(root, 'results')
    print()
    print('=' * 72)
    print('RESULTS')
    print('=' * 72)
    cj = os.path.join(res, 'comparison.json')
    if os.path.exists(cj):
        C = json.load(open(cj))['metrics']
        print('  %-34s %10s %10s' % ('model', 'RMSE', 'MAPE%'))
        for k in sorted(C):
            print('  %-34s %10.3f %10.4f' % (k, C[k]['RMSE'], C[k]['MAPE']))
    lj = os.path.join(res, 'leakage_inflation.json')
    if os.path.exists(lj):
        L = json.load(open(lj))
        print()
        print('  leakage inflation (walk-forward RMSE / one-time RMSE):')
        for k in sorted(L):
            print('    %-28s x%.2f   on %d common dates'
                  % (k, L[k]['inflation_ratio'], L[k]['common_dates']))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--only', default=None, help='substring filter on the tag')
    ap.add_argument('--bias', action='store_true',
                    help='OLS bias correction, applied to BOTH heads or neither')
    ap.add_argument('--workers', type=int, default=5,
                    help='GPU workers sharing the card (measured on the GTX 1650, '
                         'fused TRM: 3 -> 2.0x, 4 -> 2.4x, 5 -> 2.6x, 6 -> 2.5x)')
    ap.add_argument('--fused', action='store_true', help='single-kernel Adam (CUDA)')
    ap.add_argument('--amp', action='store_true', help='float16 autocast with loss scaling')
    a = ap.parse_args()

    if a.check:
        problems, _ = check()
        sys.exit(1 if problems else 0)

    root = HERE
    if a.smoke:
        root = smoke()
        a.epochs = min(a.epochs, 6)

    if not a.smoke:
        problems, _ = check()
        if problems:
            print('\nfix the above before running; nothing was trained.')
            sys.exit(1)

    print()
    print('=' * 72)
    print('TRAINING')
    print('=' * 72)
    rc = train_all(a.epochs, a.device, a.only, root, a.bias, a.workers,
                   a.fused, a.amp, poll=5 if a.smoke else 20)
    if rc:
        sys.exit(rc)
    rc = evaluate(root)
    if rc:
        sys.exit(rc)
    summarise(root)
    if a.smoke:
        sys.exit(smoke_verdict(root))
