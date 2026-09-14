"""The two prediction heads, sharing one encoder.

Both models use the same BiLSTM + self-attention encoder and differ only in the
head, which is what makes the comparison controlled. The original submission
did not: TCN was trained for 50 epochs at lr 1e-3 with no early stopping, no
gradient clipping and no EMA, while TRM got 150 epochs at lr 5e-4 with all
three plus a post-hoc OLS bias correction applied to TRM alone. A head-to-head
under different training budgets and different post-processing is not a
controlled comparison, so `train_component` gives both arms identical
treatment and the bias correction is applied to both or neither.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class SelfAttention(nn.Module):
    def __init__(self, d_model, nhead=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        out, _ = self.attn(x, x, x)
        return self.norm(x + out)


class Encoder(nn.Module):
    """BiLSTM + SAM. Shared by both heads."""

    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers,
                              bidirectional=True, batch_first=True,
                              dropout=0.1 if num_layers > 1 else 0.0)
        self.sam = SelfAttention(2 * hidden_size)

    def forward(self, x):
        out, _ = self.bilstm(x)
        return self.sam(out)                       # (B, W, 2H)


class TCNHead(nn.Module):
    """Dilated causal convolution, k=3, dilation=2, 2H -> H, residual."""

    def __init__(self, hidden_size=64, k=3, dilation=2, p=0.1):
        super().__init__()
        H = hidden_size
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(2 * H, H, kernel_size=k, dilation=dilation)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p)
        self.res = nn.Linear(2 * H, H)
        self.fc = nn.Linear(H, 1)

    def forward(self, enc):
        z = enc.transpose(1, 2)                    # (B, 2H, W)
        z = nn.functional.pad(z, (self.pad, 0))    # causal: pad left only
        z = self.drop(self.act(self.conv(z)))
        z = z.transpose(1, 2)                      # (B, W, H)
        z = z + self.res(enc)                      # residual
        return self.fc(z[:, -1, :])


class TRMHead(nn.Module):
    """Recursive latent refinement, following Jolicoeur-Martineau (2025)."""

    def __init__(self, hidden_size=64, n_rec=6, p=0.1):
        super().__init__()
        H = hidden_size
        self.proj = nn.Linear(2 * H, H)
        self.n_rec = n_rec
        self.latent_net = nn.Sequential(
            nn.Linear(H, 2 * H), nn.SiLU(), nn.Dropout(p),
            nn.Linear(2 * H, H), nn.LayerNorm(H))
        self.answer_net = nn.Sequential(
            nn.Linear(H, 2 * H), nn.SiLU(), nn.Dropout(p),
            nn.Linear(2 * H, H), nn.LayerNorm(H))
        self.fc = nn.Linear(H, 1)

    def _cell(self, x_enc, y, z):
        for _ in range(self.n_rec):
            z = self.latent_net(x_enc + y + z) + z
        y = self.answer_net(y + z) + y
        return y, z

    def forward(self, enc, T=3):
        x_enc = self.proj(enc[:, -1, :])
        B, H = x_enc.size(0), x_enc.size(1)
        y = torch.zeros(B, H, device=x_enc.device)
        z = torch.zeros(B, H, device=x_enc.device)
        if T > 1:
            with torch.no_grad():
                for _ in range(T - 1):
                    y, z = self._cell(x_enc, y, z)
            y, z = y.detach(), z.detach()
        y, _ = self._cell(x_enc, y, z)
        return self.fc(y)


class Forecaster(nn.Module):
    def __init__(self, head='tcn', hidden_size=64, n_rec=6):
        super().__init__()
        self.enc = Encoder(1, hidden_size)
        self.head_name = head
        self.head = (TCNHead(hidden_size) if head == 'tcn'
                     else TRMHead(hidden_size, n_rec))

    def forward(self, x):
        return self.head(self.enc(x))


class EMA:
    """Exponential moving average of the weights, with bias correction.

    Seeding the shadow with the model's own initial weights, as the original
    submission does, leaves a weight of decay**steps on the random
    initialisation. At decay 0.999 a component that early-stops after ~900
    steps is still 39% random weights when the average is applied. Components
    stop at different times, so the contamination differs from component to
    component and from head to head, which is precisely what a controlled
    comparison cannot tolerate.

    Starting from zero and dividing by (1 - decay**steps) at apply time gives a
    true weighted average of the weights actually visited, with no dependence
    on how long the component happened to train.
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.steps = 0
        sd = model.state_dict()
        self.float_keys = [k for k, v in sd.items() if v.dtype.is_floating_point]
        self.other_keys = [k for k, v in sd.items() if not v.dtype.is_floating_point]
        self.shadow = {k: torch.zeros_like(sd[k]) for k in self.float_keys}
        for k in self.other_keys:
            self.shadow[k] = sd[k].detach().clone()

    def update(self, model):
        self.steps += 1
        sd = model.state_dict()
        shadow = [self.shadow[k] for k in self.float_keys]
        live = [sd[k].detach() for k in self.float_keys]
        # shadow <- decay * shadow + (1 - decay) * live, as one grouped kernel
        # call rather than two kernel launches per tensor
        try:
            torch._foreach_lerp_(shadow, live, 1.0 - self.decay)
        except (AttributeError, RuntimeError, TypeError):
            for s, v in zip(shadow, live):
                s.lerp_(v, 1.0 - self.decay)
        for k in self.other_keys:
            self.shadow[k] = sd[k].detach().clone()

    def apply(self, model):
        if self.steps == 0:
            return
        corr = 1.0 - self.decay ** self.steps
        state = {k: self.shadow[k] / corr for k in self.float_keys}
        for k in self.other_keys:
            state[k] = self.shadow[k]
        model.load_state_dict(state)


def _predict(model, X, dev, chunk=512):
    out = []
    for i in range(0, len(X), chunk):
        xb = torch.as_tensor(np.asarray(X[i:i + chunk]), dtype=torch.float32,
                             device=dev)
        out.append(model(xb).squeeze(-1).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def train_component(Xtr, ytr, Xte, head='tcn', device='cuda', epochs=150,
                    patience=20, lr=5e-4, batch=64, clip=1.0, ema_decay=0.999,
                    seed=0, verbose=False, fused=False, amp=False):
    """Train one component model. Identical settings for both heads.

    fused: single-kernel Adam on CUDA. Same update rule, fewer kernel launches.
    amp:   float16 autocast for the forward pass with loss scaling. The original
           submission trained this way; it changes numerics slightly, identically
           for both heads and both protocols.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device if torch.cuda.is_available() or device == 'cpu' else 'cpu')
    on_cuda = dev.type == 'cuda'
    if on_cuda:
        torch.backends.cudnn.benchmark = True
    use_amp = bool(amp and on_cuda)

    model = Forecaster(head).to(dev)
    ema = EMA(model, ema_decay)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5,
                     fused=bool(fused and on_cuda))
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5, min_lr=1e-6)
    crit = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # The whole training set lives on the device and each minibatch is an index
    # slice of a fresh permutation: the same sampling as a shuffled DataLoader,
    # partial last batch included, seeded per component. Measured on the GTX
    # 1650 this is no faster than a DataLoader (the recurrent encoder dominates
    # each step); it is kept because the sampling is explicit and reproducible.
    # The running loss stays on the device and is read once per epoch.
    X = torch.as_tensor(np.asarray(Xtr), dtype=torch.float32, device=dev)
    Y = torch.as_tensor(np.asarray(ytr), dtype=torch.float32, device=dev)
    n = X.shape[0]
    n_batches = max((n + batch - 1) // batch, 1)
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)

    best, bad, hist = float('inf'), 0, []
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev, generator=gen)
        tot = torch.zeros((), device=dev)
        for b in range(n_batches):
            sel = perm[b * batch:(b + 1) * batch]
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                out = model(X[sel]).squeeze(-1)
            loss = crit(out.float(), Y[sel])
            # with amp off every scaler call below is a no-op and this is a
            # plain backward / clip / step
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(opt)
            scaler.update()
            ema.update(model)
            tot += loss.detach()
        avg = float(tot) / n_batches
        hist.append(avg)
        sched.step(avg)
        if avg < best - 1e-6:
            best, bad = avg, 0
        else:
            bad += 1
        if verbose and ep % 25 == 0:
            print('    epoch %3d  loss %.6f' % (ep, avg))
        if bad >= patience:
            break

    ema.apply(model)
    model.eval()
    with torch.no_grad():
        pred = _predict(model, Xte, dev)
        pred_tr = _predict(model, Xtr, dev)
    return pred, pred_tr, {'final_loss': float(hist[-1]), 'best_loss': float(best),
                           'epochs_run': len(hist)}
