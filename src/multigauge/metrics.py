"""Goodness-of-fit metrics. All functions ignore pairs where either value is NaN."""
from __future__ import annotations

import numpy as np


def _clean(obs, sim):
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    ok = ~(np.isnan(obs) | np.isnan(sim))
    return obs[ok], sim[ok]


def nse(obs, sim) -> float:
    """Nash-Sutcliffe Efficiency (1 = perfect, 0 = as good as the mean, <0 worse)."""
    o, s = _clean(obs, sim)
    if len(o) < 2:
        return float("nan")
    denom = np.sum((o - o.mean()) ** 2)
    return float(1 - np.sum((s - o) ** 2) / denom) if denom > 0 else float("nan")


def kge(obs, sim) -> float:
    """Kling-Gupta Efficiency (Gupta et al., 2009)."""
    o, s = _clean(obs, sim)
    if len(o) < 2 or o.std() == 0 or o.mean() == 0:
        return float("nan")
    r = np.corrcoef(o, s)[0, 1] if s.std() > 0 else 0.0
    alpha = s.std() / o.std()
    beta = s.mean() / o.mean()
    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def rmse(obs, sim) -> float:
    o, s = _clean(obs, sim)
    return float(np.sqrt(np.mean((s - o) ** 2))) if len(o) else float("nan")


def pbias(obs, sim) -> float:
    """Percent bias: positive = model over-predicts total volume."""
    o, s = _clean(obs, sim)
    return float(100 * (s.sum() - o.sum()) / o.sum()) if len(o) and o.sum() != 0 else float("nan")


def peak_error(obs, sim, quantile: float = 0.98) -> float:
    """Mean percent error on high-flow days (observed flow above the quantile)."""
    o, s = _clean(obs, sim)
    if len(o) < 10:
        return float("nan")
    hi = o >= np.quantile(o, quantile)
    return float(100 * np.mean((s[hi] - o[hi]) / np.where(o[hi] == 0, np.nan, o[hi])))


ALL = {"NSE": nse, "KGE": kge, "RMSE": rmse, "PBIAS": pbias, "PeakError": peak_error}


def evaluate(obs, sim) -> dict:
    return {name: fn(obs, sim) for name, fn in ALL.items()}
