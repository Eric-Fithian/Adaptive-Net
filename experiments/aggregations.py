from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence
import math

# NumPy is used for FFT, percentiles, and some vector ops
import numpy as np


"""
Assumes oldest to newest.
"""

# -------- Helpers -------- #

def _as_float_list(values: Sequence[float]) -> List[float]:
    # Caller guarantees real numbers and no Nones, but keep this robust.
    if isinstance(values, np.ndarray):
        return [float(v) for v in values.tolist()]
    return [float(v) for v in values]


def _filter_nones(values: List[Optional[float]]) -> List[float]:
    # Kept for backward-compatibility with your original scaffolding.
    return [float(v) for v in values if v is not None]


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)


def _variance(xs: List[float], mean: Optional[float] = None) -> float:
    # population variance
    m = _mean(xs) if mean is None else mean
    return sum((x - m) ** 2 for x in xs) / len(xs)


def _std(xs: List[float], mean: Optional[float] = None) -> float:
    return math.sqrt(_variance(xs, mean))


def _percentile(xs: List[float], q: float) -> float:
    # q in [0, 100]
    return float(np.percentile(np.asarray(xs, dtype=float), q))


def _pearsonr(x: List[float], y: List[float]) -> Optional[float]:
    n = len(x)
    if n != len(y) or n < 2:
        return None
    mx, my = _mean(x), _mean(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    denx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    deny = math.sqrt(sum((yi - my) ** 2 for yi in y))
    den = denx * deny
    if den == 0.0:
        return 0.0
    return num / den


def _ranks(values: List[float]) -> List[float]:
    # Average-rank ties (Spearman)
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _ema_half_life(values: List[float], half_life: float) -> List[float]:
    # EMA with half-life: alpha = 1 - 2^(-1/H)
    if not values:
        return []
    alpha = 1.0 - 2.0 ** (-1.0 / float(half_life))
    out = [values[0]]
    for v in values[1:]:
        out.append(alpha * v + (1.0 - alpha) * out[-1])
    return out


def _split_blocks(n: int, blocks: int) -> List[range]:
    # Split indices [0..n-1] into ~equal contiguous blocks
    base = n // blocks
    rem = n % blocks
    ranges = []
    start = 0
    for b in range(blocks):
        extra = 1 if b < rem else 0
        end = start + base + extra
        ranges.append(range(start, end))
        start = end
    return ranges


# -------- Core aggregators (existing) -------- #

def agg_mean(values: List[Optional[float]]) -> Optional[float]:
    xs = _filter_nones(values)
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def agg_std(values: List[Optional[float]]) -> Optional[float]:
    xs = _filter_nones(values)
    n = len(xs)
    if n <= 1:
        return None
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    return float(var ** 0.5)


def agg_slope_ols(values: List[Optional[float]]) -> Optional[float]:
    """OLS slope against time index 0..n-1 for non-None entries (using their positions).
    If fewer than 2 valid points, returns None.
    """
    xs = []
    ys = []
    for i, v in enumerate(values):
        if v is not None:
            xs.append(float(i))
            ys.append(float(v))
    n = len(xs)
    if n <= 1:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0.0:
        return 0.0
    return float(num / den)


# -------- Temporal aggregation functions (A–J + skew/kurtosis) -------- #

# A) Load level
def agg_median(values: Sequence[float]) -> Optional[float]:
    xs = _as_float_list(values)
    if not xs:
        return None
    return float(np.median(xs))


def agg_ema_mean_hl(values: Sequence[float], half_life: float = 10.0) -> Optional[float]:
    xs = _as_float_list(values)
    if not xs:
        return None
    ema = _ema_half_life(xs, half_life)
    return float(ema[-1])


# B) Volatility
def agg_var(values: Sequence[float]) -> Optional[float]:
    xs = _as_float_list(values)
    if not xs:
        return None
    return _variance(xs)


def agg_mad(values: Sequence[float]) -> Optional[float]:
    xs = _as_float_list(values)
    if not xs:
        return None
    med = float(np.median(xs))
    return float(np.median([abs(x - med) for x in xs]))


def agg_iqr(values: Sequence[float]) -> Optional[float]:
    xs = _as_float_list(values)
    if not xs:
        return None
    q75 = _percentile(xs, 75.0)
    q25 = _percentile(xs, 25.0)
    return float(q75 - q25)


# C) Extremes / tails
def agg_min(values: Sequence[float]) -> Optional[float]:
    xs = _as_float_list(values)
    return None if not xs else float(min(xs))


def agg_max(values: Sequence[float]) -> Optional[float]:
    xs = _as_float_list(values)
    return None if not xs else float(max(xs))


def agg_range(values: Sequence[float]) -> Optional[float]:
    xs = _as_float_list(values)
    if not xs:
        return None
    return float(max(xs) - min(xs))


def agg_q10(values: Sequence[float]) -> Optional[float]:
    xs = _as_float_list(values)
    return None if not xs else _percentile(xs, 10.0)


def agg_q90(values: Sequence[float]) -> Optional[float]:
    xs = _as_float_list(values)
    return None if not xs else _percentile(xs, 90.0)


def agg_tail_exceed_p90_rate(values: Sequence[float]) -> Optional[float]:
    xs = _as_float_list(values)
    n = len(xs)
    if n == 0:
        return None
    q90 = _percentile(xs, 90.0)
    return float(sum(1 for v in xs if v > q90) / n)


# D) Trend / drift
def agg_trend_tstat(values: Sequence[float]) -> Optional[float]:
    xs = _as_float_list(values)
    n = len(xs)
    if n < 3:
        return None
    ts = list(range(n))
    mx, mt = _mean(xs), _mean(ts)
    sxx = sum((t - mt) ** 2 for t in ts)
    if sxx == 0.0:
        return 0.0
    beta = sum((t - mt) * (x - mx) for t, x in zip(ts, xs)) / sxx
    resid = [x - ( (_mean(xs) - beta * mt) + beta * t ) for t, x in zip(ts, xs)]
    s2 = sum(r * r for r in resid) / (n - 2)
    se_beta = math.sqrt(s2 / sxx) if s2 >= 0.0 else 0.0
    if se_beta == 0.0:
        return 0.0
    return float(beta / se_beta)


def agg_delta_last_first(values: Sequence[float]) -> Optional[float]:
    xs = _as_float_list(values)
    return None if len(xs) == 0 else float(xs[-1] - xs[0])


def agg_pct_change_last_first(values: Sequence[float], eps: float = 1e-12) -> Optional[float]:
    xs = _as_float_list(values)
    if len(xs) == 0:
        return None
    denom = xs[0]
    return float(xs[-1] / (denom if abs(denom) > eps else eps) - 1.0)


def agg_spearman_rho_time(values: Sequence[float]) -> Optional[float]:
    xs = _as_float_list(values)
    n = len(xs)
    if n < 2:
        return None
    rx = _ranks(xs)
    rt = list(range(n))  # already ranks
    return _pearsonr(rx, rt)


# E) Inertia / oscillation
def agg_autocorr_lag1(values: Sequence[float]) -> Optional[float]:
    xs = _as_float_list(values)
    if len(xs) < 2:
        return None
    x0, x1 = xs[:-1], xs[1:]
    return _pearsonr(x0, x1)


def agg_vol_cluster_acf(values: Sequence[float]) -> Optional[float]:
    xs = _as_float_list(values)
    n = len(xs)
    if n < 2:
        return None
    m = _mean(xs)
    y = [(x - m) ** 2 for x in xs]
    y0, y1 = y[:-1], y[1:]
    return _pearsonr(y0, y1)


def agg_zero_cross_rate(values: Sequence[float]) -> Optional[float]:
    xs = _as_float_list(values)
    n = len(xs)
    if n < 2:
        return None
    def sgn(v: float) -> int:
        return 1 if v > 0 else (-1 if v < 0 else 0)
    signs = [sgn(v) for v in xs]
    flips = 0
    pairs = 0
    for a, b in zip(signs[:-1], signs[1:]):
        if a == 0 or b == 0:
            # treat zeros as no definitive sign; still count transition if strict flip?
            # We'll ignore pairs where either is zero to focus on true sign flips.
            continue
        pairs += 1
        if a != b:
            flips += 1
    denom = max(pairs, 1)
    return float(flips / denom)


def agg_sign_persistence(values: Sequence[float]) -> Optional[float]:
    xs = _as_float_list(values)
    n = len(xs)
    if n == 0:
        return None
    def sgn(v: float) -> int:
        return 1 if v > 0 else (-1 if v < 0 else 0)
    signs = [sgn(v) for v in xs]
    max_run = 0
    run = 0
    prev = 0
    for s in signs:
        if s == 0:
            run = 0
            prev = 0
            continue
        if s == prev and s != 0:
            run += 1
        else:
            run = 1
        prev = s
        if run > max_run:
            max_run = run
    return float(max_run / n)


# F) Curvature / acceleration
def agg_mean_second_diff(values: Sequence[float]) -> Optional[float]:
    xs = _as_float_list(values)
    if len(xs) < 3:
        return None
    diffs2 = [xs[i] - 2 * xs[i - 1] + xs[i - 2] for i in range(2, len(xs))]
    return float(_mean(diffs2))


def agg_abs_mean_second_diff(values: Sequence[float]) -> Optional[float]:
    xs = _as_float_list(values)
    if len(xs) < 3:
        return None
    diffs2 = [abs(xs[i] - 2 * xs[i - 1] + xs[i - 2]) for i in range(2, len(xs))]
    return float(_mean(diffs2))


# G) Energy / effort
def agg_l1norm(values: Sequence[float]) -> Optional[float]:
    xs = _as_float_list(values)
    return None if not xs else float(sum(abs(x) for x in xs))


def agg_l2norm(values: Sequence[float]) -> Optional[float]:
    xs = _as_float_list(values)
    return None if not xs else float(math.sqrt(sum(x * x for x in xs)))


# H) Regime shift
def agg_half_diff_mean(values: Sequence[float]) -> Optional[float]:
    xs = _as_float_list(values)
    n = len(xs)
    if n == 0:
        return None
    half = n // 2
    early = xs[:half]
    late = xs[half:]
    if not early or not late:
        return 0.0
    return float(_mean(late) - _mean(early))


def agg_ema_crossover_rate(
    values: Sequence[float],
    short_half_life: float = 5.0,
    long_half_life: float = 20.0,
) -> Optional[float]:
    xs = _as_float_list(values)
    n = len(xs)
    if n < 2:
        return None
    s = _ema_half_life(xs, short_half_life)
    l = _ema_half_life(xs, long_half_life)
    diff = [si - li for si, li in zip(s, l)]
    # count sign changes in diff
    def sgn(v: float) -> int:
        return 1 if v > 0 else (-1 if v < 0 else 0)
    signs = [sgn(d) for d in diff]
    crossings = 0
    denom_pairs = 0
    for a, b in zip(signs[:-1], signs[1:]):
        if a == 0 or b == 0:
            continue
        denom_pairs += 1
        if a != b:
            crossings += 1
    denom = max(denom_pairs, 1)
    return float(crossings / denom)


# I) Spectral / periodicity
def agg_dom_freq_idx(values: Sequence[float]) -> Optional[float]:
    xs = _as_float_list(values)
    n = len(xs)
    if n < 2:
        return None
    x = np.asarray(xs, dtype=float)
    x = x - x.mean()  # remove DC
    # rFFT: bins 0..N/2 (inclusive). Exclude DC at index 0.
    spec = np.fft.rfft(x)
    power = (spec.real ** 2 + spec.imag ** 2)
    if power.size <= 1:
        return 0.0
    idx = int(np.argmax(power[1:]) + 1)
    return float(idx)


def agg_spectral_entropy(values: Sequence[float]) -> Optional[float]:
    xs = _as_float_list(values)
    n = len(xs)
    if n < 2:
        return None
    x = np.asarray(xs, dtype=float) - float(np.mean(xs))
    spec = np.fft.rfft(x)
    power = (spec.real ** 2 + spec.imag ** 2)
    power = power[1:]  # drop DC
    total = float(power.sum())
    if total <= 0.0 or power.size == 0:
        return 0.0
    p = power / total
    # avoid log(0)
    p = np.where(p > 0, p, 1.0)
    ent = float(-(p * np.log(p)).sum())
    return ent


def agg_low_high_bandpower_ratio(values: Sequence[float], cutoff_fraction: float = 0.25) -> Optional[float]:
    xs = _as_float_list(values)
    n = len(xs)
    if n < 2:
        return None
    x = np.asarray(xs, dtype=float) - float(np.mean(xs))
    power = np.abs(np.fft.rfft(x)) ** 2
    power = power[1:]  # exclude DC
    m = power.size
    if m == 0:
        return 0.0
    K = max(1, int(m * cutoff_fraction))
    low = float(power[:K].sum())
    high = float(power[K:].sum()) if K < m else 0.0
    if high == 0.0:
        return None
    return float(low / high)


# J) Robust exceedance counts
def agg_above_1_sigma_rate(values: Sequence[float]) -> Optional[float]:
    xs = _as_float_list(values)
    n = len(xs)
    if n == 0:
        return None
    m = _mean(xs)
    s = _std(xs, m)
    thr = m + 1.0 * s
    return float(sum(1 for v in xs if v > thr) / n)


def agg_above_2_sigma_rate(values: Sequence[float]) -> Optional[float]:
    xs = _as_float_list(values)
    n = len(xs)
    if n == 0:
        return None
    m = _mean(xs)
    s = _std(xs, m)
    thr = m + 2.0 * s
    return float(sum(1 for v in xs if v > thr) / n)


def agg_fano_factor(values: Sequence[float], blocks: int = 4, k_sigma: float = 1.0) -> Optional[float]:
    xs = _as_float_list(values)
    n = len(xs)
    if n == 0:
        return None
    m = _mean(xs)
    s = _std(xs, m)
    thr = m + k_sigma * s
    ranges = _split_blocks(n, max(1, blocks))
    counts = []
    for r in ranges:
        if len(r) == 0:
            continue
        c = sum(1 for i in r if xs[i] > thr)
        counts.append(float(c))
    if not counts:
        return None
    mu = _mean(counts)
    if mu == 0.0:
        return None
    var = _variance(counts, mu)
    return float(var / mu)


# Additional (from “Standard” set)
def agg_skew(values: Sequence[float]) -> Optional[float]:
    xs = _as_float_list(values)
    n = len(xs)
    if n < 3:
        return None
    m = _mean(xs)
    s = _std(xs, m)
    if s == 0.0:
        return 0.0
    m3 = sum((x - m) ** 3 for x in xs) / n
    return float(m3 / (s ** 3))


def agg_kurtosis(values: Sequence[float]) -> Optional[float]:
    xs = _as_float_list(values)
    n = len(xs)
    if n < 4:
        return None
    m = _mean(xs)
    s = _std(xs, m)
    if s == 0.0:
        return -3.0  # excess kurtosis for a degenerate distribution
    m4 = sum((x - m) ** 4 for x in xs) / n
    # Excess kurtosis (Fisher): normal -> 0
    return float(m4 / (s ** 4) - 3.0)


# -------- Registry -------- #

AGGREGATION_FUNCS: Dict[str, Callable[..., Optional[float]]] = {
    # Existing
    'mean': agg_mean,
    'std': agg_std,
    'slope_ols': agg_slope_ols,

    # A) Load level
    'median': agg_median,
    'ema_mean_hl': agg_ema_mean_hl,  # uses default half_life=10; can be called with kwargs

    # B) Volatility
    'var': agg_var,
    'mad': agg_mad,
    'iqr': agg_iqr,

    # C) Extremes / tails
    'min': agg_min,
    'max': agg_max,
    'range': agg_range,
    'q10': agg_q10,
    'q90': agg_q90,
    'tail_exceed_p90_rate': agg_tail_exceed_p90_rate,

    # D) Trend / drift
    'trend_tstat': agg_trend_tstat,
    'delta_last_first': agg_delta_last_first,
    'pct_change_last_first': agg_pct_change_last_first,
    'spearman_rho_time': agg_spearman_rho_time,

    # E) Inertia / oscillation
    'autocorr_lag1': agg_autocorr_lag1,
    'vol_cluster_acf': agg_vol_cluster_acf,
    'zero_cross_rate': agg_zero_cross_rate,
    'sign_persistence': agg_sign_persistence,

    # F) Curvature / acceleration
    'mean_second_diff': agg_mean_second_diff,
    'abs_mean_second_diff': agg_abs_mean_second_diff,

    # G) Energy / effort
    'l1norm': agg_l1norm,
    'l2norm': agg_l2norm,

    # H) Regime shift
    'half_diff_mean': agg_half_diff_mean,
    'ema_crossover_rate': agg_ema_crossover_rate,  # default short/long HLs; kwargs allowed

    # I) Spectral / periodicity
    'dom_freq_idx': agg_dom_freq_idx,
    'spectral_entropy': agg_spectral_entropy,
    'low_high_bandpower_ratio': agg_low_high_bandpower_ratio,  # default cutoff_fraction=0.25

    # J) Robust exceedance counts
    'above_1_sigma_rate': agg_above_1_sigma_rate,
    'above_2_sigma_rate': agg_above_2_sigma_rate,
    'fano_factor': agg_fano_factor,  # default blocks=4, k_sigma=1.0

    # Additional
    'skew': agg_skew,
    'kurtosis': agg_kurtosis,
}

__all__ = [
    'AGGREGATION_FUNCS',

    # Existing
    'agg_mean', 'agg_std', 'agg_slope_ols',

    # A
    'agg_median', 'agg_ema_mean_hl',

    # B
    'agg_var', 'agg_mad', 'agg_iqr',

    # C
    'agg_min', 'agg_max', 'agg_range', 'agg_q10', 'agg_q90', 'agg_tail_exceed_p90_rate',

    # D
    'agg_trend_tstat', 'agg_delta_last_first', 'agg_pct_change_last_first', 'agg_spearman_rho_time',

    # E
    'agg_autocorr_lag1', 'agg_vol_cluster_acf', 'agg_zero_cross_rate', 'agg_sign_persistence',

    # F
    'agg_mean_second_diff', 'agg_abs_mean_second_diff',

    # G
    'agg_l1norm', 'agg_l2norm',

    # H
    'agg_half_diff_mean', 'agg_ema_crossover_rate',

    # I
    'agg_dom_freq_idx', 'agg_spectral_entropy', 'agg_low_high_bandpower_ratio',

    # J
    'agg_above_1_sigma_rate', 'agg_above_2_sigma_rate', 'agg_fano_factor',

    # Additional
    'agg_skew', 'agg_kurtosis',
]