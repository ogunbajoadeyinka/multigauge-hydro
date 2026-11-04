"""Run the full comparison of rainfall input strategies for one basin."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from . import __version__
from .importance import permutation_importance, station_ablation
from .io import BasinData
from .metrics import evaluate, nse
from .model import StreamflowLSTM, TrainConfig
from .spatial import VoronoiResult, distance_km, voronoi_weights
from .strategies import LABELS, STRATEGIES, build_features, gap_fill_report

PERIODS = ("train", "validation", "test")


@dataclass
class ExperimentConfig:
    """Settings for a comparison run.

    Periods may be given as (start, end) date strings. Any period left as
    None is filled from ``split_fractions`` in chronological order.
    ``single_station`` may be a gauge id, "auto" (highest permutation
    importance on the validation period) or "nearest_outlet".
    """

    train_period: Optional[tuple] = None
    validation_period: Optional[tuple] = None
    test_period: Optional[tuple] = None
    split_fractions: tuple = (0.7, 0.15, 0.15)
    strategies: Sequence[str] = STRATEGIES
    single_station: str = "auto"
    n_seeds: int = 1
    permutation_repeats: int = 5
    run_ablation: bool = False
    training: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["strategies"] = list(self.strategies)
        return d


@dataclass
class ExperimentResult:
    basin_name: str
    config: ExperimentConfig
    periods: dict
    metrics: pd.DataFrame         # long table: strategy, period, metric, value, seed
    summary: pd.DataFrame         # strategy x (period metrics) for the ensemble
    predictions: pd.DataFrame     # date index: observed + one column per strategy
    voronoi: VoronoiResult
    single_station: str
    single_station_reason: str
    importance: Optional[pd.DataFrame]
    ablation: Optional[pd.DataFrame]
    station_summary: pd.DataFrame
    histories: dict
    notes: list
    models: dict = field(default_factory=dict, repr=False)

    def best_strategy(self, period: str = "test", metric: str = "NSE") -> str:
        s = self.summary.xs(period, level="period")[metric]
        return s.idxmax()

    def save(self, folder, save_models: bool = False) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.summary.to_csv(folder / "metrics_summary.csv")
        self.metrics.to_csv(folder / "metrics_by_seed.csv", index=False)
        self.predictions.to_csv(folder / "predictions.csv")
        self.station_summary.to_csv(folder / "stations.csv")
        if self.importance is not None:
            self.importance.to_csv(folder / "permutation_importance.csv", index=False)
        if self.ablation is not None:
            self.ablation.to_csv(folder / "station_ablation.csv", index=False)
        meta = {
            "multigauge_version": __version__,
            "basin": self.basin_name,
            "periods": {k: [str(v[0].date()), str(v[1].date())] for k, v in self.periods.items()},
            "single_station": self.single_station,
            "single_station_reason": self.single_station_reason,
            "voronoi_used_boundary": self.voronoi.used_boundary,
            "notes": self.notes,
            "config": self.config.to_dict(),
        }
        (folder / "run_info.json").write_text(json.dumps(meta, indent=2, default=str))
        if save_models:
            for (strategy, seed), m in self.models.items():
                m.save(folder / "models" / f"{strategy}_seed{seed}")
        return folder


# --------------------------------------------------------------------------
def make_periods(dates: pd.DatetimeIndex, cfg: ExperimentConfig) -> dict:
    given = {
        "train": cfg.train_period,
        "validation": cfg.validation_period,
        "test": cfg.test_period,
    }
    if all(v is not None for v in given.values()):
        periods = {k: (pd.Timestamp(v[0]), pd.Timestamp(v[1])) for k, v in given.items()}
    else:
        f = np.asarray(cfg.split_fractions, dtype=float)
        if len(f) != 3 or (f <= 0).any():
            raise ValueError("split_fractions needs three positive numbers.")
        f = f / f.sum()
        n = len(dates)
        cut1, cut2 = int(n * f[0]), int(n * (f[0] + f[1]))
        periods = {
            "train": (dates[0], dates[cut1 - 1]),
            "validation": (dates[cut1], dates[cut2 - 1]),
            "test": (dates[cut2], dates[-1]),
        }
        for k, v in given.items():
            if v is not None:
                periods[k] = (pd.Timestamp(v[0]), pd.Timestamp(v[1]))
    for k, (a, b) in periods.items():
        if a > b:
            raise ValueError(f"The {k} period starts after it ends.")
        if a < dates[0] or b > dates[-1]:
            raise ValueError(f"The {k} period ({a.date()} to {b.date()}) is outside the data ({dates[0].date()} to {dates[-1].date()}).")
    return periods


def _masks(dates, periods):
    return {k: np.asarray((dates >= a) & (dates <= b)) for k, (a, b) in periods.items()}


def run_experiment(
    data: BasinData,
    config: Optional[ExperimentConfig] = None,
    progress: Optional[Callable[[float, str], None]] = None,
    keep_models: bool = True,
) -> ExperimentResult:
    """Train one LSTM per strategy (and seed), evaluate, and rank gauges."""
    cfg = config or ExperimentConfig()
    tcfg = cfg.training
    strategies = [s for s in STRATEGIES if s in cfg.strategies]
    if not strategies:
        raise ValueError("Choose at least one strategy.")
    notes = list(data.notes)

    periods = make_periods(data.dates, cfg)
    masks = _masks(data.dates, periods)
    obs = data.streamflow
    vor = voronoi_weights(data.stations, data.boundary)
    if not vor.used_boundary:
        notes.append(
            "No basin boundary was supplied: Voronoi weights use a buffered convex hull of the gauges."
        )
    w = vor.weights
    stations = data.station_ids
    seeds = [tcfg.seed + k for k in range(max(1, cfg.n_seeds))]

    need_importance = ("disaggregated" in strategies and len(stations) > 1) or (
        "single" in strategies and cfg.single_station == "auto"
    )
    n_jobs = len(seeds) * len(strategies)
    if need_importance and "disaggregated" not in strategies:
        n_jobs += 1
    if need_importance:
        n_jobs += 1
    if cfg.run_ablation and len(stations) > 1:
        n_jobs += len(seeds) * len(stations)
    done = [0]

    def tick(msg):
        if progress:
            progress(min(done[0] / n_jobs, 0.999), msg)

    def train(strategy, seed, station=None, drop=None, label=None):
        feats = build_features(data.rainfall, strategy, w, station=station, extra=data.extra, drop=drop)
        model = StreamflowLSTM(TrainConfig(**{**asdict(tcfg), "seed": seed}))

        def cb(ep, total, score):
            if progress:
                frac = (done[0] + ep / total) / n_jobs
                progress(min(frac, 0.999), f"{label or LABELS[strategy]} - seed {seed} - epoch {ep}/{total} (val NSE {score:.2f})")

        model.fit(feats, obs, masks["train"], masks["validation"], progress=cb)
        done[0] += 1
        return model, feats

    models, preds, histories, rows = {}, {}, {}, []

    def record(strategy, seed, model, feats):
        p = model.predict(feats)
        preds.setdefault(strategy, []).append(p)
        histories[(strategy, seed)] = model.history
        if keep_models:
            models[(strategy, seed)] = model
        for per in PERIODS:
            m = masks[per]
            for metric, val in evaluate(obs.values[m], p.values[m]).items():
                rows.append({"strategy": strategy, "seed": seed, "period": per, "metric": metric, "value": val})

    # 1. disaggregated first: it drives gauge selection for the single-gauge model
    importance = None
    disagg_first = None
    if need_importance:
        tick("Training disaggregated model")
        disagg_first, feats_d = train("disaggregated", seeds[0])
        tick("Permutation importance")
        importance = permutation_importance(
            disagg_first, feats_d, obs, masks["validation"], n_repeats=cfg.permutation_repeats, seed=tcfg.seed
        )
        done[0] += 1
        if "disaggregated" in strategies:
            record("disaggregated", seeds[0], disagg_first, feats_d)

    # 2. choose the single gauge
    reason = ""
    single = None
    if "single" in strategies:
        choice = cfg.single_station
        if choice == "auto":
            if importance is not None:
                single = importance.iloc[0]["station"]
                reason = "highest permutation importance (validation period)"
            else:
                single, reason = stations[0], "only gauge available"
        elif choice == "nearest_outlet":
            if data.outlet is None:
                raise ValueError("'nearest_outlet' needs the outlet location (lat, lon).")
            d = distance_km(data.stations, *data.outlet)
            single, reason = d.idxmin(), f"closest to outlet ({d.min():.1f} km)"
        else:
            if choice not in stations:
                raise ValueError(f"Gauge '{choice}' is not in the rainfall data.")
            single, reason = choice, "chosen by user"

    # 3. train the remaining strategy/seed combinations
    for strategy in strategies:
        for seed in seeds:
            if strategy == "disaggregated" and seed == seeds[0] and disagg_first is not None:
                continue
            tick(f"Training {LABELS[strategy]} model")
            model, feats = train(strategy, seed, station=single)
            record(strategy, seed, model, feats)

    metrics = pd.DataFrame(rows)

    # ensemble (mean over seeds) predictions and summary metrics
    pred_df = pd.DataFrame({"observed": obs})
    for s in strategies:
        pred_df[s] = pd.concat(preds[s], axis=1).mean(axis=1)
    summ_rows = []
    for s in strategies:
        for per in PERIODS:
            m = masks[per]
            ev = evaluate(obs.values[m], pred_df[s].values[m])
            seed_nse = metrics.query("strategy == @s and period == @per and metric == 'NSE'")["value"]
            ev["NSE_seed_std"] = float(seed_nse.std()) if len(seed_nse) > 1 else np.nan
            summ_rows.append({"strategy": s, "period": per, **ev})
    summary = pd.DataFrame(summ_rows).set_index(["strategy", "period"])
    pred_df["period"] = "unused"
    for per in PERIODS:
        pred_df.loc[masks[per], "period"] = per

    # 4. station ablation
    ablation = None
    if cfg.run_ablation and len(stations) > 1:
        def train_without(drop):
            ps = []
            for seed in seeds:
                tick(f"Ablation: without {drop[0]}")
                m, f = train("disaggregated", seed, drop=drop, label=f"Ablation (without {drop[0]})")
                ps.append(m.predict(f))
            p = pd.concat(ps, axis=1).mean(axis=1).values
            return {per: nse(obs.values[masks[per]], p[masks[per]]) for per in ("validation", "test")}

        if "disaggregated" in strategies:
            base = {per: summary.loc[("disaggregated", per), "NSE"] for per in ("validation", "test")}
        else:
            base = train_without([])  # pragma: no cover
        ablation = station_ablation(train_without, stations, base)

    # station table
    st = data.summary().join(vor.weights).join(vor.areas_km2).join(gap_fill_report(data.rainfall))
    if data.outlet is not None:
        st = st.join(distance_km(data.stations, *data.outlet))
    if importance is not None:
        st = st.join(importance.set_index("station")[["importance", "importance_std"]])
    if ablation is not None:
        st = st.join(ablation.set_index("station")[["test_nse_change"]])

    if progress:
        progress(1.0, "Done")
    return ExperimentResult(
        basin_name=data.name,
        config=cfg,
        periods=periods,
        metrics=metrics,
        summary=summary,
        predictions=pred_df,
        voronoi=vor,
        single_station=single,
        single_station_reason=reason,
        importance=importance,
        ablation=ablation,
        station_summary=st,
        histories=histories,
        notes=notes,
        models=models,
    )
