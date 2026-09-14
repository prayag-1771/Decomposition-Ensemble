"""Measure the GPU half before committing to it.

    python benchmark.py

1. EMA bias correction, tested exactly rather than by eye.
2. Per-epoch time for each head, on the GPU and on the CPU, at the exact
   shape of Veer's files.
3. Throughput with several trainings sharing the GPU at once.
4. An estimate for the whole job from those measurements.
"""
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shared'))
from models import train_component, Forecaster, EMA          # noqa: E402

# component counts from Veer's delivered files
N_COMPONENTS = 131
TRAIN_ROWS, TEST_ROWS, WINDOW = 1985, 497, 30
# epochs each head ran before early stopping in the first benchmark; real
# components will differ, this only scales the estimate
EPOCHS_SEEN = {'tcn': 93, 'trm': 140}
TIMED_EPOCHS = 12


def synth(ntr=TRAIN_ROWS, nte=TEST_ROWS, seed=0):
    rng = np.random.default_rng(seed)
    n = ntr + nte + WINDOW + 1
    t = np.arange(n)
    s = 5000 + 0.4 * t + 60 * np.sin(2 * np.pi * t / 90) + rng.normal(0, 8, n)
    X = np.stack([s[i - WINDOW:i, None] for i in range(WINDOW, n)]).astype(np.float32)
    y = s[WINDOW:].astype(np.float32)
    lo, hi = float(X.min()), float(X.max())
    X, y = (X - lo) / (hi - lo), (y - lo) / (hi - lo)
    return X[:ntr], y[:ntr], X[ntr:ntr + nte]


def timed_epochs(head, device, epochs=TIMED_EPOCHS):
    Xtr, ytr, Xte = synth()
    if device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    train_component(Xtr, ytr, Xte, head=head, device=device, epochs=epochs,
                    patience=10 ** 9, seed=0)
    if device == 'cuda':
        torch.cuda.synchronize()
    return (time.time() - t0) / epochs


# ------------------------------------------------------------- worker mode
if len(sys.argv) > 1 and sys.argv[1] == '--worker':
    head, epochs = sys.argv[2], int(sys.argv[3])
    t = timed_epochs(head, 'cuda', epochs)
    print('WORKER_SEC_PER_EPOCH %.4f' % t, flush=True)
    sys.exit(0)

print('=' * 72)
print('ENVIRONMENT')
print('=' * 72)
print('  torch          : %s' % torch.__version__)
print('  cuda available : %s' % torch.cuda.is_available())
if torch.cuda.is_available():
    print('  device         : %s  (%.1f GB)'
          % (torch.cuda.get_device_name(0),
             torch.cuda.get_device_properties(0).total_memory / 1e9))
print('  cpu threads    : %d' % torch.get_num_threads())

# ------------------------------------------------------------- 1. EMA test
print()
print('=' * 72)
print('1. EMA bias correction')
print('=' * 72)
m = Forecaster('trm')
init_mean = torch.cat([p.detach().flatten() for p in m.parameters()]).abs().mean().item()
ema = EMA(m, 0.999)
steps = 300
with torch.no_grad():
    for _ in range(steps):
        for p in m.parameters():
            p.fill_(1.0)                       # every weight visited is exactly 1
        ema.update(m)
ema.apply(m)
vals = torch.cat([p.detach().flatten() for p in m.parameters()])
err = (vals - 1.0).abs().max().item()
leftover = 0.999 ** steps
print('  every visited weight is 1.0; the average must therefore be 1.0')
print('  corrected EMA   : max |average - 1| = %.2e   %s'
      % (err, 'PASS' if err < 1e-4 else 'FAIL'))
print('  uncorrected EMA : would still carry %.0f%% of the random initial weights'
      % (100 * leftover))

# ---------------------------------------------------------- 2. per-epoch time
print()
print('=' * 72)
print('2. Seconds per epoch at the real shape (%d train rows)' % TRAIN_ROWS)
print('=' * 72)
per_epoch = {}
devices = ['cuda', 'cpu'] if torch.cuda.is_available() else ['cpu']
print('  %-6s %10s %10s' % ('head', 'GPU', 'CPU'))
for head in ('tcn', 'trm'):
    row = {}
    for dev in devices:
        row[dev] = timed_epochs(head, dev, TIMED_EPOCHS if dev == 'cuda' else 3)
    per_epoch[head] = row
    print('  %-6s %10s %10s'
          % (head, '%.3f s' % row['cuda'] if 'cuda' in row else '-',
             '%.3f s' % row['cpu']))
for head in per_epoch:
    r = per_epoch[head]
    if 'cuda' in r:
        faster = 'GPU' if r['cuda'] < r['cpu'] else 'CPU'
        print('  %s: %s is faster by %.1fx'
              % (head, faster, max(r['cuda'], r['cpu']) / min(r['cuda'], r['cpu'])))

# ---------------------------------------------------- 3. sharing the GPU
speedup = {1: 1.0}
if torch.cuda.is_available():
    print()
    print('=' * 72)
    print('3. Several trainings sharing the GPU (TRM, %d epochs each)' % TIMED_EPOCHS)
    print('=' * 72)
    here = os.path.abspath(__file__)
    base = None
    print('  %-9s %16s %12s %10s' % ('parallel', 's/epoch each', 'throughput', 'status'))
    for k in (1, 2, 3, 4):
        procs = [subprocess.Popen([sys.executable, here, '--worker', 'trm',
                                   str(TIMED_EPOCHS)],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True) for _ in range(k)]
        times, failed = [], False
        for p in procs:
            out, errtxt = p.communicate()
            line = [l for l in out.splitlines() if l.startswith('WORKER_SEC_PER_EPOCH')]
            if p.returncode != 0 or not line:
                failed = True
                reason = 'out of memory' if 'out of memory' in errtxt.lower() else 'failed'
            else:
                times.append(float(line[0].split()[1]))
        if failed:
            print('  %-9d %16s %12s %10s' % (k, '-', '-', reason))
            break
        t_each = max(times)
        if base is None:
            base = t_each
        speedup[k] = k * base / t_each
        print('  %-9d %14.3f s %11.2fx %10s' % (k, t_each, speedup[k], 'ok'))

# ------------------------------------------------------------ 4. estimate
print()
print('=' * 72)
print('4. Estimate for the whole job: %d components x 2 heads' % N_COMPONENTS)
print('=' * 72)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
serial = sum(N_COMPONENTS * EPOCHS_SEEN[h] * min(per_epoch[h].values())
             for h in ('tcn', 'trm'))
print('  one at a time, fastest device per head : %.1f h' % (serial / 3600))
best_k = max(speedup, key=lambda k: speedup[k])
if best_k > 1:
    print('  %d in parallel on the GPU              : %.1f h  (%.2fx throughput)'
          % (best_k, serial / speedup[best_k] / 3600, speedup[best_k]))
print()
print('  Epoch counts come from synthetic components; real ones will stop')
print('  earlier or later, so read this as the order of magnitude.')
