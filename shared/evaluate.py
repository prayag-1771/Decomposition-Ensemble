"""Baselines, error metrics and the modified Diebold-Mariano test.

The point of this module is that the deep models must be scored against
baselines that are genuinely hard to beat on daily equity data. Persistence
(tomorrow equals today) is the important one: daily index levels are close to a
random walk, so persistence is strong, and a decomposition pipeline that cannot
beat it has not demonstrated anything. The original submission did include
persistence, and lost to it; that result is the reason this study exists.
"""
import numpy as np
from scipy import stats


# ------------------------------------------------------------------ baselines
def persistence(series, test_start, n):
    """Tomorrow equals today."""
    return np.asarray([series[test_start - 1 + i] for i in range(n)], dtype=float)


def moving_average(series, test_start, n, k):
    return np.asarray([series[test_start - k + i: test_start + i].mean()
                       for i in range(n)], dtype=float)


def ewma(series, test_start, n, span=10):
    a = 2.0 / (span + 1.0)
    out = np.zeros(n)
    prev = float(np.mean(series[max(0, test_start - span):test_start]))
    for t in range(n):
        prev = a * series[test_start + t - 1] + (1 - a) * prev
        out[t] = prev
    return out


def drift(series, test_start, n):
    """Random walk with drift estimated on the training portion."""
    d = (series[test_start - 1] - series[0]) / max(test_start - 1, 1)
    return np.asarray([series[test_start - 1 + i] + d for i in range(n)], dtype=float)


def all_baselines(series, test_start, n):
    return {
        'Persistence': persistence(series, test_start, n),
        'Drift': drift(series, test_start, n),
        'MA-5': moving_average(series, test_start, n, 5),
        'MA-20': moving_average(series, test_start, n, 20),
        'EWMA-10': ewma(series, test_start, n, 10),
    }


# -------------------------------------------------------------------- metrics
def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        'RMSE': float(np.sqrt(np.mean(err ** 2))),
        'MAE': float(np.mean(np.abs(err))),
        'MAPE': float(np.mean(np.abs(err / np.maximum(np.abs(y_true), 1e-8))) * 100),
        'R2': float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan'),
    }


# ------------------------------------------------- modified Diebold-Mariano
def dm_test(y_true, pred_a, pred_b, h=1, power=2):
    """Harvey-Leybourne-Newbold modified DM test.

    Returns (statistic, p_value). The loss differential is
    d_t = |e_a|^power - |e_b|^power, so a POSITIVE statistic means model A has
    the larger loss, i.e. A is worse than B. Uses a Student t reference
    distribution with T-1 degrees of freedom, per Harvey et al. (1997).
    """
    y_true = np.asarray(y_true, dtype=float)
    e1 = y_true - np.asarray(pred_a, dtype=float)
    e2 = y_true - np.asarray(pred_b, dtype=float)
    d = np.abs(e1) ** power - np.abs(e2) ** power
    T = len(d)
    d_bar = float(np.mean(d))

    gamma = [float(np.mean((d[k:] - d_bar) * (d[:T - k] - d_bar))) for k in range(h)]
    V = (gamma[0] + 2.0 * sum(gamma[1:])) / T
    if V <= 0:
        return float('nan'), float('nan')

    dm = d_bar / np.sqrt(V)
    correction = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_mod = dm * correction
    p = 2 * (1 - stats.t.cdf(abs(dm_mod), df=T - 1))
    return float(dm_mod), float(p)


def comparison_table(y_true, model_preds, baseline_preds, h=1):
    """Every model against every baseline. Returns a list of row dicts."""
    rows = []
    for mname, mpred in model_preds.items():
        for bname, bpred in baseline_preds.items():
            stat, p = dm_test(y_true, mpred, bpred, h=h)
            rows.append({'model': mname, 'baseline': bname,
                         'DM': stat, 'p': p,
                         'significant_5pct': bool(abs(stat) > 1.96) if stat == stat else False,
                         'model_worse': bool(stat > 0) if stat == stat else None})
    return rows


def bias_correct(pred_val, true_val, pred_test):
    """OLS affine correction fitted on a validation split.

    The original submission applied this to the TRM arm only. If it is used at
    all it must be applied to both arms, so this returns the fitted (a, b) for
    the caller to apply symmetrically.
    """
    A = np.vstack([np.asarray(pred_val, dtype=float), np.ones(len(pred_val))]).T
    coef, *_ = np.linalg.lstsq(A, np.asarray(true_val, dtype=float), rcond=None)
    a, b = float(coef[0]), float(coef[1])
    return a * np.asarray(pred_test, dtype=float) + b, a, b
