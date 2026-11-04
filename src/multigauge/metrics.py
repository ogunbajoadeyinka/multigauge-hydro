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


# ------------------------------------------------------------ probabilistic
def pinball(obs, q, level: float) -> float:
    o, s = _clean(obs, q)
    if not len(o):
        return float("nan")
    d = o - s
    return float(np.mean(np.maximum(level * d, (level - 1) * d)))


def crps_from_quantiles(obs, quantiles: dict) -> float:
    """CRPS approximated as twice the mean pinball loss over quantile levels.

    ``quantiles`` maps level (0-1) -> array of predicted quantiles. With
    evenly spaced levels this converges to the CRPS (Gneiting & Raftery, 2007).
    """
    if not quantiles:
        return float("nan")
    return float(2 * np.mean([pinball(obs, q, lv) for lv, q in quantiles.items()]))


def coverage(obs, lower, upper) -> float:
    """Share of observations inside [lower, upper] (target: nominal level)."""
    obs, lower, upper = (np.asarray(a, dtype=float) for a in (obs, lower, upper))
    ok = np.isfinite(obs) & np.isfinite(lower) & np.isfinite(upper)
    return float(np.mean((obs[ok] >= lower[ok]) & (obs[ok] <= upper[ok]))) if ok.any() else float("nan")


def interval_width(lower, upper) -> float:
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    ok = np.isfinite(lower) & np.isfinite(upper)
    return float(np.mean(upper[ok] - lower[ok])) if ok.any() else float("nan")


def evaluate_probabilistic(obs, pred) -> dict:
    """Deterministic metrics on ``pred['mean']`` plus interval metrics if
    quantile columns (q05 ... q95) are present."""
    out = evaluate(obs, pred["mean"])
    qcols = {int(c[1:]) / 100: pred[c].to_numpy() for c in pred.columns if c.startswith("q") and c[1:].isdigit()}
    if qcols:
        out["CRPS"] = crps_from_quantiles(np.asarray(obs, float), qcols)
        if 0.05 in qcols and 0.95 in qcols:
            out["Coverage90"] = coverage(obs, qcols[0.05], qcols[0.95])
            out["Width90"] = interval_width(qcols[0.05], qcols[0.95])
    return out
