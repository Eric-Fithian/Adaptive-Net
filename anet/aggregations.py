from __future__ import annotations

import math
from typing import Callable, Dict, List
import numpy as np


# ============================== #
#         Edge-case policy       #
# ============================== #
# Unless noted otherwise below:
# - Empty input -> 0.0
# - Insufficient length for a statistic (e.g., n<2 for std/var/corr, n<3 for second-diff, etc.) -> 0.0
# - Zero-variance / degenerate cases -> return 0.0 (except kurtosis where a degenerate series returns -3.0)
# - Fixed hyperparameters baked in (no kwargs): EMA half-lives, Fano blocks/k_sigma, etc.


# ----------------- helpers (private) ----------------- #

def _to_np(xs: List[float]) -> np.ndarray:
    # Accepts list[float]; casts to float64 vector without copying if already ndarray of same dtype.
    arr = np.asarray(xs, dtype=np.float64)
    return arr

def _n(xs: np.ndarray) -> int:
    return xs.shape[0]

def _mean(xs: np.ndarray) -> float:
    return float(xs.mean()) if xs.size else 0.0

def _variance_pop(xs: np.ndarray, mean: float | None = None) -> float:
    n = _n(xs)
    if n == 0:
        return 0.0
    m = _mean(xs) if mean is None else mean
    return float(((xs - m) ** 2).mean())

def _std_pop(xs: np.ndarray, mean: float | None = None) -> float:
    v = _variance_pop(xs, mean)
    return float(math.sqrt(v))

def _percentile(xs: np.ndarray, q: float) -> float:
    if xs.size == 0:
        return 0.0
    return float(np.percentile(xs, q))

def _pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    n = _n(x)
    if n != _n(y) or n < 2:
        return 0.0
    mx, my = float(x.mean()), float(y.mean())
    dx, dy = x - mx, y - my
    den = float(np.linalg.norm(dx) * np.linalg.norm(dy))
    if den == 0.0:
        return 0.0
    num = float((dx * dy).sum())
    return num / den

def _ranks_average_ties(values: np.ndarray) -> np.ndarray:
    # Average ranks with ties; 0-based average rank (consistent with original code)
    if values.size == 0:
        return np.array([], dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty_like(values, dtype=np.float64)
    i = 0
    while i < values.size:
        j = i
        # Walk tie-group in sorted order
        while (j + 1 < values.size) and (values[order[j + 1]] == values[order[i]]):
            j += 1
        avg_rank = (i + j) / 2.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    return ranks

def _ema_half_life(xs: np.ndarray, half_life: float) -> np.ndarray:
    if xs.size == 0:
        return xs
    alpha = 1.0 - 2.0 ** (-1.0 / float(half_life))
    out = np.empty_like(xs)
    out[0] = xs[0]
    for i in range(1, xs.size):
        out[i] = alpha * xs[i] + (1.0 - alpha) * out[i - 1]
    return out

def _split_blocks(n: int, blocks: int) -> List[slice]:
    base = n // blocks
    rem = n % blocks
    slices: List[slice] = []
    start = 0
    for b in range(blocks):
        extra = 1 if b < rem else 0
        end = start + base + extra
        slices.append(slice(start, end))
        start = end
    return slices


# ----------------- core aggregators (public API) ----------------- #
# Each accepts: values: list[float] -> float

def agg_mean(values: List[float]) -> float:
    xs = _to_np(values)
    return _mean(xs)

def agg_std(values: List[float]) -> float:
    xs = _to_np(values)
    return _std_pop(xs) if _n(xs) >= 2 else 0.0

def agg_slope_ols(values: List[float]) -> float:
    xs = _to_np(values)
    n = _n(xs)
    if n <= 1:
        return 0.0
    t = np.arange(n, dtype=np.float64)
    mt, mx = float(t.mean()), float(xs.mean())
    denom = float(((t - mt) ** 2).sum())
    if denom == 0.0:
        return 0.0
    num = float(((t - mt) * (xs - mx)).sum())
    return num / denom

# A) Load level
def agg_median(values: List[float]) -> float:
    xs = _to_np(values)
    return float(np.median(xs)) if xs.size else 0.0

def agg_ema_mean_hl(values: List[float]) -> float:
    # fixed half-life = 10.0
    xs = _to_np(values)
    if xs.size == 0:
        return 0.0
    ema = _ema_half_life(xs, half_life=10.0)
    return float(ema[-1])

# B) Volatility
def agg_var(values: List[float]) -> float:
    xs = _to_np(values)
    return _variance_pop(xs)

def agg_mad(values: List[float]) -> float:
    xs = _to_np(values)
    if xs.size == 0:
        return 0.0
    med = float(np.median(xs))
    return float(np.median(np.abs(xs - med)))

def agg_iqr(values: List[float]) -> float:
    xs = _to_np(values)
    if xs.size == 0:
        return 0.0
    return _percentile(xs, 75.0) - _percentile(xs, 25.0)

# C) Extremes / tails
def agg_min(values: List[float]) -> float:
    xs = _to_np(values)
    return float(xs.min()) if xs.size else 0.0

def agg_max(values: List[float]) -> float:
    xs = _to_np(values)
    return float(xs.max()) if xs.size else 0.0

def agg_range(values: List[float]) -> float:
    xs = _to_np(values)
    return float(xs.max() - xs.min()) if xs.size else 0.0

def agg_q10(values: List[float]) -> float:
    xs = _to_np(values)
    return _percentile(xs, 10.0)

def agg_q90(values: List[float]) -> float:
    xs = _to_np(values)
    return _percentile(xs, 90.0)

def agg_tail_exceed_p90_rate(values: List[float]) -> float:
    xs = _to_np(values)
    n = _n(xs)
    if n == 0:
        return 0.0
    q90 = _percentile(xs, 90.0)
    return float((xs > q90).sum() / n)

# D) Trend / drift
def agg_trend_tstat(values: List[float]) -> float:
    xs = _to_np(values)
    n = _n(xs)
    if n < 3:
        return 0.0
    t = np.arange(n, dtype=np.float64)
    mt, mx = float(t.mean()), float(xs.mean())
    sxx = float(((t - mt) ** 2).sum())
    if sxx == 0.0:
        return 0.0
    beta = float(((t - mt) * (xs - mx)).sum() / sxx)
    a = mx - beta * mt
    resid = xs - (a + beta * t)
    s2 = float((resid @ resid) / (n - 2))
    if s2 < 0.0:
        return 0.0
    se_beta = math.sqrt(s2 / sxx) if sxx > 0.0 else 0.0
    if se_beta == 0.0:
        return 0.0
    return float(beta / se_beta)

def agg_delta_last_first(values: List[float]) -> float:
    xs = _to_np(values)
    return float(xs[-1] - xs[0]) if xs.size else 0.0

def agg_pct_change_last_first(values: List[float]) -> float:
    xs = _to_np(values)
    if xs.size == 0:
        return 0.0
    eps = 1e-12
    denom = xs[0] if abs(xs[0]) > eps else (eps if xs[0] >= 0 else -eps)
    return float(xs[-1] / denom - 1.0)

def agg_spearman_rho_time(values: List[float]) -> float:
    xs = _to_np(values)
    n = _n(xs)
    if n < 2:
        return 0.0
    rx = _ranks_average_ties(xs)
    rt = np.arange(n, dtype=np.float64)  # already ranks 0..n-1
    return _pearsonr(rx, rt)

# E) Inertia / oscillation
def agg_autocorr_lag1(values: List[float]) -> float:
    xs = _to_np(values)
    if _n(xs) < 2:
        return 0.0
    return _pearsonr(xs[:-1], xs[1:])

def agg_vol_cluster_acf(values: List[float]) -> float:
    xs = _to_np(values)
    n = _n(xs)
    if n < 2:
        return 0.0
    y = (xs - xs.mean()) ** 2
    return _pearsonr(y[:-1], y[1:])

def agg_zero_cross_rate(values: List[float]) -> float:
    xs = _to_np(values)
    n = _n(xs)
    if n < 2:
        return 0.0
    sgn = np.sign(xs)  # -1, 0, +1
    a, b = sgn[:-1], sgn[1:]
    mask = (a != 0) & (b != 0)
    denom = int(mask.sum())
    if denom == 0:
        return 0.0
    flips = int((a[mask] != b[mask]).sum())
    return float(flips / denom)

def agg_sign_persistence(values: List[float]) -> float:
    xs = _to_np(values)
    n = _n(xs)
    if n == 0:
        return 0.0
    sgn = np.sign(xs)
    max_run = 0
    run = 0
    prev = 0
    for s in sgn.tolist():
        if s == 0:
            run = 0
            prev = 0
            continue
        if s == prev:
            run += 1
        else:
            run = 1
        prev = s
        if run > max_run:
            max_run = run
    return float(max_run / n)

# F) Curvature / acceleration
def agg_mean_second_diff(values: List[float]) -> float:
    xs = _to_np(values)
    if _n(xs) < 3:
        return 0.0
    diffs2 = xs[2:] - 2.0 * xs[1:-1] + xs[:-2]
    return float(diffs2.mean())

def agg_abs_mean_second_diff(values: List[float]) -> float:
    xs = _to_np(values)
    if _n(xs) < 3:
        return 0.0
    diffs2 = np.abs(xs[2:] - 2.0 * xs[1:-1] + xs[:-2])
    return float(diffs2.mean())

# G) Energy / effort
def agg_l1norm(values: List[float]) -> float:
    xs = _to_np(values)
    return float(np.abs(xs).sum())

def agg_l2norm(values: List[float]) -> float:
    xs = _to_np(values)
    return float(np.linalg.norm(xs))

# H) Regime shift
def agg_half_diff_mean(values: List[float]) -> float:
    xs = _to_np(values)
    n = _n(xs)
    if n == 0:
        return 0.0
    half = n // 2
    early = xs[:half]
    late = xs[half:]
    if early.size == 0 or late.size == 0:
        return 0.0
    return float(late.mean() - early.mean())

def agg_ema_crossover_rate(values: List[float]) -> float:
    # fixed short HL = 5.0, long HL = 20.0
    xs = _to_np(values)
    n = _n(xs)
    if n < 2:
        return 0.0
    s = _ema_half_life(xs, 5.0)
    l = _ema_half_life(xs, 20.0)
    diff = s - l
    sgn = np.sign(diff)
    a, b = sgn[:-1], sgn[1:]
    mask = (a != 0) & (b != 0)
    denom = int(mask.sum())
    if denom == 0:
        return 0.0
    crossings = int((a[mask] != b[mask]).sum())
    return float(crossings / denom)

# I) Spectral / periodicity
def agg_dom_freq_idx(values: List[float]) -> float:
    xs = _to_np(values)
    n = _n(xs)
    if n < 2:
        return 0.0
    x = xs - xs.mean()
    spec = np.fft.rfft(x)
    power = (spec.real ** 2 + spec.imag ** 2)
    if power.size <= 1:
        return 0.0
    # Exclude DC at index 0
    idx = int(np.argmax(power[1:]) + 1)
    return float(idx)

def agg_spectral_entropy(values: List[float]) -> float:
    xs = _to_np(values)
    n = _n(xs)
    if n < 2:
        return 0.0
    x = xs - xs.mean()
    spec = np.fft.rfft(x)
    power = (spec.real ** 2 + spec.imag ** 2)
    power = power[1:]  # drop DC
    total = float(power.sum())
    if total <= 0.0 or power.size == 0:
        return 0.0
    p = power / total
    # Avoid log(0) by clamping to 1 for zero bins → contributes 0 to entropy
    p = np.where(p > 0, p, 1.0)
    ent = float(-(p * np.log(p)).sum())
    return ent

def agg_low_high_bandpower_ratio(values: List[float]) -> float:
    xs = _to_np(values)
    n = _n(xs)
    if n < 2:
        return 0.0
    x = xs - xs.mean()
    power = np.abs(np.fft.rfft(x)) ** 2
    power = power[1:]  # exclude DC
    m = power.size
    if m == 0:
        return 0.0
    cutoff_fraction = 0.25
    K = max(1, int(m * cutoff_fraction))
    low = float(power[:K].sum())
    high = float(power[K:].sum()) if K < m else 0.0
    if high == 0.0:
        return 0.0
    return float(low / high)

# J) Robust exceedance counts
def agg_above_1_sigma_rate(values: List[float]) -> float:
    xs = _to_np(values)
    n = _n(xs)
    if n == 0:
        return 0.0
    m = _mean(xs)
    s = _std_pop(xs, m)
    thr = m + 1.0 * s
    return float((xs > thr).sum() / n)

def agg_above_2_sigma_rate(values: List[float]) -> float:
    xs = _to_np(values)
    n = _n(xs)
    if n == 0:
        return 0.0
    m = _mean(xs)
    s = _std_pop(xs, m)
    thr = m + 2.0 * s
    return float((xs > thr).sum() / n)

def agg_fano_factor(values: List[float]) -> float:
    # Fixed blocks=4, k_sigma=1.0
    xs = _to_np(values)
    n = _n(xs)
    if n == 0:
        return 0.0
    m = _mean(xs)
    s = _std_pop(xs, m)
    thr = m + 1.0 * s
    counts: List[float] = []
    for sl in _split_blocks(n, 4):
        if sl.start == sl.stop:
            continue
        c = float((xs[sl] > thr).sum())
        counts.append(c)
    if not counts:
        return 0.0
    cxs = np.asarray(counts, dtype=np.float64)
    mu = float(cxs.mean())
    if mu == 0.0:
        return 0.0
    var = float(((cxs - mu) ** 2).mean())
    return float(var / mu)

# Additional (standard)
def agg_skew(values: List[float]) -> float:
    xs = _to_np(values)
    n = _n(xs)
    if n < 3:
        return 0.0
    m = _mean(xs)
    s = _std_pop(xs, m)
    if s == 0.0:
        return 0.0
    m3 = float(((xs - m) ** 3).mean())
    return float(m3 / (s ** 3))

def agg_kurtosis(values: List[float]) -> float:
    # Excess kurtosis (Fisher); normal -> 0
    xs = _to_np(values)
    n = _n(xs)
    if n < 4:
        return 0.0
    m = _mean(xs)
    s = _std_pop(xs, m)
    if s == 0.0:
        return -3.0  # degenerate distribution: excess kurtosis is -3
    m4 = float(((xs - m) ** 4).mean())
    return float(m4 / (s ** 4) - 3.0)


# ----------------- registry ----------------- #

AGGREGATION_FUNCS: Dict[str, Callable[[List[float]], float]] = {
    # Existing
    "mean": agg_mean,
    "std": agg_std,
    "slope_ols": agg_slope_ols,

    # A) Load level
    "median": agg_median,
    "ema_mean_hl": agg_ema_mean_hl,

    # B) Volatility
    "var": agg_var,
    "mad": agg_mad,
    "iqr": agg_iqr,

    # C) Extremes / tails
    "min": agg_min,
    "max": agg_max,
    "range": agg_range,
    "q10": agg_q10,
    "q90": agg_q90,
    "tail_exceed_p90_rate": agg_tail_exceed_p90_rate,

    # D) Trend / drift
    "trend_tstat": agg_trend_tstat,
    "delta_last_first": agg_delta_last_first,
    "pct_change_last_first": agg_pct_change_last_first,
    "spearman_rho_time": agg_spearman_rho_time,

    # E) Inertia / oscillation
    "autocorr_lag1": agg_autocorr_lag1,
    "vol_cluster_acf": agg_vol_cluster_acf,
    "zero_cross_rate": agg_zero_cross_rate,
    "sign_persistence": agg_sign_persistence,

    # F) Curvature / acceleration
    "mean_second_diff": agg_mean_second_diff,
    "abs_mean_second_diff": agg_abs_mean_second_diff,

    # G) Energy / effort
    "l1norm": agg_l1norm,
    "l2norm": agg_l2norm,

    # H) Regime shift
    "half_diff_mean": agg_half_diff_mean,
    "ema_crossover_rate": agg_ema_crossover_rate,

    # I) Spectral / periodicity
    "dom_freq_idx": agg_dom_freq_idx,
    "spectral_entropy": agg_spectral_entropy,
    "low_high_bandpower_ratio": agg_low_high_bandpower_ratio,

    # J) Robust exceedance counts
    "above_1_sigma_rate": agg_above_1_sigma_rate,
    "above_2_sigma_rate": agg_above_2_sigma_rate,
    "fano_factor": agg_fano_factor,

    # Additional
    "skew": agg_skew,
    "kurtosis": agg_kurtosis,
}

__all__ = [
    "AGGREGATION_FUNCS",
    # Existing
    "agg_mean", "agg_std", "agg_slope_ols",
    # A
    "agg_median", "agg_ema_mean_hl",
    # B
    "agg_var", "agg_mad", "agg_iqr",
    # C
    "agg_min", "agg_max", "agg_range", "agg_q10", "agg_q90", "agg_tail_exceed_p90_rate",
    # D
    "agg_trend_tstat", "agg_delta_last_first", "agg_pct_change_last_first", "agg_spearman_rho_time",
    # E
    "agg_autocorr_lag1", "agg_vol_cluster_acf", "agg_zero_cross_rate", "agg_sign_persistence",
    # F
    "agg_mean_second_diff", "agg_abs_mean_second_diff",
    # G
    "agg_l1norm", "agg_l2norm",
    # H
    "agg_half_diff_mean", "agg_ema_crossover_rate",
    # I
    "agg_dom_freq_idx", "agg_spectral_entropy", "agg_low_high_bandpower_ratio",
    # J
    "agg_above_1_sigma_rate", "agg_above_2_sigma_rate", "agg_fano_factor",
    # Additional
    "agg_skew", "agg_kurtosis",
]
