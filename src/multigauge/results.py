"""Read saved study results without PyTorch.

The dashboard's pre-computed mode only needs these, so it stays light and
works even where PyTorch is not installed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .strategies import STRATEGIES

PERIODS = ("train", "validation", "test")
GAUGE_MODELS = {s: f"ft_{s}" for s in STRATEGIES}
LABELS = {
    "gr4j": "GR4J (conceptual)",
    "local": "Local LSTM (from scratch)",
    "pretrained": "Pre-trained, not fine-tuned",
    "finetuned": "Fine-tuned (gridded only)",
    "ft_single": "Fine-tuned + single gauge",
    "ft_lumped": "Fine-tuned + gauge average",
    "ft_voronoi": "Fine-tuned + Voronoi gauges",
    "ft_disaggregated": "Fine-tuned + all gauges",
    "dhbv": "δHBV hybrid",
    "ensemble": "LSTM + δHBV ensemble",
}
ORDER = list(LABELS)


@dataclass
class SavedStudy:
    """A study read back from ``StudyResult.save`` output (no models needed)."""

    basin_name: str
    flow_unit: str
    area_km2: float
    summary: pd.DataFrame
    predictions: dict          # model -> DataFrame in the user's flow unit (mean, q.., observed, period)
    gauge_value: Optional[pd.DataFrame]
    importance: Optional[pd.DataFrame]
    station_summary: pd.DataFrame
    info: dict
    extras: dict = field(default_factory=dict)

    @property
    def models(self) -> list:
        return [m for m in ORDER if m in self.predictions]

    def best_model(self, period: str = "test", metric: str = "NSE") -> str:
        return self.summary.xs(period, level="period")[metric].idxmax()

    def flow(self, model: str) -> pd.DataFrame:
        return self.predictions[model]


def add_seed_spread(gauge_value: pd.DataFrame, metrics: pd.DataFrame, min_gain: float = 0.01) -> pd.DataFrame:
    """Add the seed-to-seed spread of each NSE gain and a plain verdict.

    The gain is compared seed by seed (gauges vs no gauges, same seed), so the
    spread measures training noise. ``verdict`` is "helps"/"hurts" when the
    gain is more than twice that spread and at least ``min_gain`` in size,
    "negligible" when it is consistent but smaller, otherwise "within noise".
    """
    if gauge_value is None or metrics is None or metrics.empty:
        return gauge_value
    m = metrics[metrics["metric"] == "NSE"].pivot_table(index=["period", "seed"], columns="model", values="value")
    sd, verdict = [], []
    for _, r in gauge_value.iterrows():
        col = GAUGE_MODELS.get(r["gauge_input"])
        if col not in m or "finetuned" not in m or r["period"] not in m.index.get_level_values(0):
            sd.append(np.nan); verdict.append("")
            continue
        d = (m.loc[r["period"], col] - m.loc[r["period"], "finetuned"]).dropna()
        s_ = float(d.std(ddof=1)) if len(d) > 1 else np.nan
        sd.append(s_)
        if np.isnan(s_):
            verdict.append("one seed only")
        elif abs(r["NSE_gain"]) > 2 * s_:
            verdict.append("negligible" if abs(r["NSE_gain"]) < min_gain else ("helps" if r["NSE_gain"] > 0 else "hurts"))
        else:
            verdict.append("within noise")
    return gauge_value.assign(NSE_gain_seed_sd=sd, verdict=verdict)


def load_saved_study(folder) -> SavedStudy:
    folder = Path(folder)
    info = json.loads((folder / "run_info.json").read_text())
    summary = pd.read_csv(folder / "metrics_summary.csv").set_index(["model", "period"])
    pr = pd.read_csv(folder / "predictions.csv", index_col=0, parse_dates=True)
    preds = {}
    for m in ORDER:
        cols = [c for c in pr.columns if c.startswith(m + "_") and c[len(m) + 1:] in ("mean",) + tuple(
            f"q{i:02d}" for i in range(100))]
        if f"{m}_mean" in pr.columns:
            df = pr[cols].rename(columns=lambda c: c[len(m) + 1:])
            df["observed"] = pr["observed"]
            df["period"] = pr["period"]
            preds[m] = df
    opt = lambda n: pd.read_csv(folder / n) if (folder / n).exists() else None
    gv = opt("gauge_value.csv")
    if gv is not None and "verdict" not in gv and (folder / "metrics_by_seed.csv").exists():
        gv = add_seed_spread(gv, pd.read_csv(folder / "metrics_by_seed.csv"))
    extras = {}
    for n in ("subbasin_models.csv", "subbasin_contributions.csv", "forecast_skill.csv"):
        if (folder / n).exists():
            extras[n[:-4]] = pd.read_csv(folder / n)
    return SavedStudy(
        basin_name=info.get("basin", folder.parent.name), flow_unit=info.get("flow_unit", "m3/s"),
        area_km2=float(info.get("area_km2", np.nan)), summary=summary, predictions=preds,
        gauge_value=gv, importance=opt("importance.csv"),
        station_summary=pd.read_csv(folder / "stations.csv", index_col=0), info=info, extras=extras,
    )


def available_pretrained() -> list:
    """Names of the models shipped in the package (no PyTorch needed)."""
    d = Path(__file__).parent / "pretrained"
    return sorted(p.name for p in d.iterdir() if (p / "model.json").exists()) if d.exists() else []
