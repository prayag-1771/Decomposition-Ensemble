"""Which training configuration finishes the job soonest without hurting it.

    python speed_options.py

A. Seconds per epoch and final loss for: baseline, fused Adam, fused Adam +
   float16 autocast. Loss is reported so a faster option that trains worse is
   visible, not just its speed.
B. Throughput with 1 to 6 trainings sharing the GPU, each limited to one CPU
   thread (with several processes, extra CPU threads per process only fight
   over the same cores).
C. Estimated wall time for the whole job under each option.
"""
import os
import subprocess
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'shared'))
from models import train_component                                     # noqa: E402

N_COMPONENTS = 131
TRAIN_ROWS, TEST_ROWS, WINDOW = 1985, 497, 30
EPOCHS_SEEN = {'tcn': 93, 'trm': 140}
EPOCHS = 15
CONFIGS = {'baseline': dict(fused=False, amp=False),
           'fused': dict(fused=True, amp=False),
           'fused+amp': dict(fused=True, amp=True)}


def synth(seed=0):
    rng = np.random.default_rng(seed)
    n = TRAIN_ROWS + TEST_ROWS + WINDOW + 1
    t = np.arange(n)
    s = 5000 + 0.4 * t + 60 * np.sin(2 * np.pi * t / 90) + rng.normal(0, 8, n)
    X = np.stack([s[i - WINDOW:i, None] for i in range(WINDOW, n)]).astype(np.float32)
    y = s[WINDOW:].astype(np.float32)
    lo, hi = float(X.min()), float(X.max())
    X, y = (X - lo) / (hi - lo), (y - lo) / (hi - lo)
    return X[:TRAIN_ROWS], y[:TRAIN_ROWS], X[TRAIN_ROWS:TRAIN_ROWS + TEST_ROWS]


def run(head, cfg, epochs=EPOCHS):
    Xtr, ytr, Xte = synth()
    torch.cuda.synchronize()
    t0 = time.time()
    _, _, info = train_component(Xtr, ytr, Xte, head=head, device='cuda',
                                 epochs=epochs, patience=10 ** 9, seed=0,
                                 **CONFIGS[cfg])
    torch.cuda.synchronize()
    return (time.time() - t0) / epochs, info['best_loss']


if len(sys.argv) > 1 and sys.argv[1] == '--worker':
    torch.set_num_threads(1)
    cfg, head, epochs = sys.argv[2], sys.argv[3], int(sys.argv[4])
    torch.cuda.reset_peak_memory_stats()
    sec, _ = run(head, cfg, epochs)
    print('WORKER %.4f %.0f' % (sec, torch.cuda.max_memory_allocated() / 1e6), flush=True)
    sys.exit(0)

if not torch.cuda.is_available():
    sys.exit('no CUDA device')

print('=' * 74)
print('A. One training at a time: seconds per epoch, and loss after %d epochs' % EPOCHS)
print('=' * 74)
single = {}
print('  %-10s %-5s %12s %14s' % ('config', 'head', 's / epoch', 'best loss'))
for cfg in CONFIGS:
    for head in ('tcn', 'trm'):
        sec, loss = run(head, cfg)
        single[(cfg, head)] = sec
        print('  %-10s %-5s %10.3f s %14.6f' % (cfg, head, sec, loss))

best_cfg = min(CONFIGS, key=lambda c: single[(c, 'tcn')] * EPOCHS_SEEN['tcn']
               + single[(c, 'trm')] * EPOCHS_SEEN['trm'])
print('\n  fastest single-process config: %s' % best_cfg)

print()
print('=' * 74)
print('B. Sharing the GPU (%s, TRM, 1 CPU thread per process)' % best_cfg)
print('=' * 74)
print('  %-9s %14s %10s %12s %10s' % ('parallel', 's/epoch each', 'VRAM MB', 'throughput', 'status'))
speed = {}
base = None
for k in (1, 2, 3, 4, 5, 6):
    procs = [subprocess.Popen([sys.executable, os.path.abspath(__file__), '--worker',
                               best_cfg, 'trm', str(EPOCHS)],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for _ in range(k)]
    secs, mems, fail = [], [], None
    for p in procs:
        out, err = p.communicate()
        line = [l for l in out.splitlines() if l.startswith('WORKER')]
        if p.returncode != 0 or not line:
            fail = 'out of memory' if 'out of memory' in err.lower() else 'failed'
        else:
            secs.append(float(line[0].split()[1]))
            mems.append(float(line[0].split()[2]))
    if fail:
        print('  %-9d %14s %10s %12s %10s' % (k, '-', '-', '-', fail))
        break
    each = max(secs)
    base = base or each
    speed[k] = k * base / each
    print('  %-9d %12.3f s %10.0f %11.2fx %10s' % (k, each, max(mems), speed[k], 'ok'))

print()
print('=' * 74)
print('C. Whole job: %d components x 2 heads' % N_COMPONENTS)
print('=' * 74)
for cfg in CONFIGS:
    serial = sum(N_COMPONENTS * EPOCHS_SEEN[h] * single[(cfg, h)] for h in ('tcn', 'trm'))
    print('  %-10s one at a time: %5.1f h' % (cfg, serial / 3600))
bk = max(speed, key=speed.get)
serial = sum(N_COMPONENTS * EPOCHS_SEEN[h] * single[(best_cfg, h)] for h in ('tcn', 'trm'))
print('  %-10s %d in parallel: %5.1f h   <- recommended' % (best_cfg, bk, serial / speed[bk] / 3600))
