"""The four rainfall input strategies compared by multigauge.

single         one gauge (chosen by permutation importance, or by the user)
lumped         simple arithmetic mean of all gauges
voronoi        Thiessen/Voronoi area-weighted mean
disaggregated  every gauge as its own model input

Missing gauge values are gap-filled the same way for every strategy so that
comparisons are fair: a missing value is replaced by the Voronoi-weighted
mean of the gauges that reported that day; days with no reports at all are
set to 0 mm. The share of filled values is reported by ``gap_fill_report``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

STRATEGIES = ("single", "lumped", "voronoi", "disaggregated")

LABELS = {
    "single": "Single gauge",
    "lumped": "Lumped average",
    "voronoi": "Voronoi weighted",
    "disaggregated": "Disaggregated",
}


def weighted_mean(rain: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Area-weighted mean that re-normalises weights over gauges reporting each day."""
    w = weights.reindex(rain.columns).fillna(0.0).values
    vals = rain.values
    present = ~np.isnan(vals)
    num = np.nansum(vals * w, axis=1)
    den = (present * w).sum(axis=1)
    out = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    # if all reporting gauges have zero weight, fall back to the plain mean
    plain = rain.mean(axis=1, skipna=True).values
    out = np.where((den == 0) & present.any(axis=1), plain, out)
    return pd.Series(out, index=rain.index)


def gap_fill(rain: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    fill = weighted_mean(rain, weights).fillna(0.0)
    return rain.apply(lambda col: col.fillna(fill)).fillna(0.0)


def gap_fill_report(rain: pd.DataFrame) -> pd.Series:
    return (rain.isna().mean() * 100).round(2).rename("percent_gap_filled")


def build_features(
    rain: pd.DataFrame,
    strategy: str,
    weights: pd.Series,
    station: Optional[str] = None,
    extra: Optional[pd.DataFrame] = None,
    drop: Optional[list] = None,
) -> pd.DataFrame:
    """Return the model input table for one strategy.

    ``drop`` removes gauges before the inputs are built (used by ablation).
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from {STRATEGIES}.")
    if drop:
        rain = rain.drop(columns=list(drop))
        weights = weights.drop(index=list(drop), errors="ignore")
        if weights.sum() > 0:
            weights = weights / weights.sum()

    filled = gap_fill(rain, weights)
    if strategy == "single":
        if station is None or station not in filled.columns:
            raise ValueError("The single-gauge strategy needs a valid station id.")
        feats = filled[[station]].rename(columns={station: f"P_{station}"})
    elif strategy == "lumped":
        feats = rain.mean(axis=1, skipna=True).fillna(0.0).to_frame("P_lumped")
    elif strategy == "voronoi":
        feats = weighted_mean(rain, weights).fillna(0.0).to_frame("P_voronoi")
    else:
        feats = filled.add_prefix("P_")

    if extra is not None:
        feats = feats.join(extra.add_prefix("X_"))
    return feats.astype("float32")
