"""Which gauges matter? Permutation importance and station ablation.

Permutation importance (Breiman, 2001) keeps a trained disaggregated model
fixed and shuffles one gauge's record in time; the drop in NSE measures how
much the model relies on that gauge. It is cheap (no retraining).

Station ablation retrains the disaggregated model with one gauge removed;
the change in NSE measures what the gauge adds that the others cannot
replace. It is more expensive (one training run per gauge) but answers the
network-design question directly.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from .metrics import nse
from .model import StreamflowLSTM


def permutation_importance(
    model: StreamflowLSTM,
    features: pd.DataFrame,
    flow: pd.Series,
    eval_mask: np.ndarray,
    columns: Optional[list] = None,
    n_repeats: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """NSE drop when each input column is shuffled over the whole record."""
    rng = np.random.default_rng(seed)
    obs = flow.to_numpy()
    base = nse(obs[eval_mask], model.predict(features).to_numpy()[eval_mask])
    columns = columns or [c for c in features.columns if c.startswith("P_")]
    rows = []
    for col in columns:
        drops = []
        for _ in range(n_repeats):
            shuffled = features.copy()
            shuffled[col] = rng.permutation(shuffled[col].to_numpy())
            score = nse(obs[eval_mask], model.predict(shuffled).to_numpy()[eval_mask])
            drops.append(base - score)
        rows.append(
            {
                "station": col[2:] if col.startswith("P_") else col,
                "importance": float(np.mean(drops)),
                "importance_std": float(np.std(drops)),
            }
        )
    out = pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)
    out.attrs["baseline_nse"] = base
    return out


def station_ablation(
    train_fn: Callable[[list], dict],
    stations: list,
    baseline: dict,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> pd.DataFrame:
    """Retrain without each gauge in turn.

    ``train_fn(drop)`` must train a disaggregated model with the gauges in
    ``drop`` removed and return ``{"validation": nse, "test": nse}``.
    ``baseline`` is the same dict for the model with all gauges.
    """
    rows = []
    for i, sid in enumerate(stations):
        if progress:
            progress(i, len(stations), sid)
        scores = train_fn([sid])
        rows.append(
            {
                "station": sid,
                "val_nse_without": scores["validation"],
                "test_nse_without": scores["test"],
                "val_nse_change": scores["validation"] - baseline["validation"],
                "test_nse_change": scores["test"] - baseline["test"],
            }
        )
    out = pd.DataFrame(rows).sort_values("test_nse_change").reset_index(drop=True)
    out.attrs["baseline"] = baseline
    return out
