"""The gauge-value study: how much do local rain gauges add to a
pre-trained deep-learning streamflow model, and which gauges matter?

For one basin it trains and compares:

=====================  =====================================================
``gr4j``               GR4J conceptual model calibrated on gridded rainfall
``local``              LSTM trained from scratch on this basin only, with
                       gridded forcings + every gauge (the conventional,
                       single-basin approach)
``pretrained``         the pre-trained multi-basin LSTM, used as-is
``finetuned``          pre-trained LSTM fine-tuned on this basin (gridded only)
``ft_single`` ...      fine-tuned with local gauges added as inputs, using
``ft_disaggregated``   each of the four gauge strategies
=====================  =====================================================

The *gauge value* is the change in skill of each ``ft_*`` model over
``finetuned``: what the gauges add on top of a strong model that already
sees gridded rainfall. Permutation importance and (optionally) station
ablation rank individual gauges on the disaggregated model.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from . import __version__
from .baselines import GR4JModel
from .basin import Basin
from .features import mm_per_day_to_flow
from .io import BasinData
from .metrics import evaluate_probabilistic, nse
from .nn import ModelConfig, StreamflowModel
from .spatial import VoronoiResult, distance_km, voronoi_weights
from .strategies import STRATEGIES, build_features, gap_fill_report
from .transfer import finetune, train_local

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
class StudyConfig:
    train_period: Optional[tuple] = None
    validation_period: Optional[tuple] = None
    test_period: Optional[tuple] = None
    split_fractions: tuple = (0.6, 0.2, 0.2)
    gauge_inputs: Sequence[str] = STRATEGIES
    single_station: str = "auto"
    n_seeds: int = 1
    finetune_epochs: int = 20
    finetune_lr: float = 2e-4
    include_zero_shot: bool = True
    include_local: bool = True
    include_gr4j: bool = True
    local_config: Optional[ModelConfig] = None
    permutation_repeats: int = 5
    run_ablation: bool = False
    gr4j_maxiter: int = 40

    def to_dict(self) -> dict:
        d = asdict(self)
        d["gauge_inputs"] = list(self.gauge_inputs)
        return d


@dataclass
class StudyResult:
    basin_name: str
    area_km2: float
    flow_unit: str
    config: StudyConfig
    periods: dict
    base_info: dict
    metrics: pd.DataFrame          # long: model, seed, period, metric, value
    summary: pd.DataFrame          # (model, period) -> metrics of the seed ensemble
    predictions: dict              # model -> DataFrame (mm/day): mean, q05.., observed, period
    gauge_value: Optional[pd.DataFrame]
    importance: Optional[pd.DataFrame]
    ablation: Optional[pd.DataFrame]
    voronoi: VoronoiResult
    single_station: Optional[str]
    single_station_reason: str
    station_summary: pd.DataFrame
    gr4j_params: Optional[dict]
    notes: list = field(default_factory=list)

    @property
    def models(self) -> list:
        return [m for m in ORDER if m in self.predictions]

    def best_model(self, period: str = "test", metric: str = "NSE") -> str:
        return self.summary.xs(period, level="period")[metric].idxmax()

    def flow(self, model: str) -> pd.DataFrame:
        """Predictions converted back to the user's flow unit."""
        df = self.predictions[model].copy()
        for c in df.columns:
            if c == "observed" or c == "mean" or (c.startswith("q") and c[1:].isdigit()):
                df[c] = mm_per_day_to_flow(df[c], self.area_km2, self.flow_unit)
        return df

    def save(self, folder) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.summary.to_csv(folder / "metrics_summary.csv")
        self.metrics.to_csv(folder / "metrics_by_seed.csv", index=False)
        preds = []
        for m in self.models:
            df = self.flow(m).drop(columns=["observed", "period"])
            df.columns = [f"{m}_{c}" for c in df.columns]
            preds.append(df)
        first = self.flow(self.models[0])
        pd.concat([first[["observed", "period"]], *preds], axis=1).to_csv(folder / "predictions.csv")
        self.station_summary.to_csv(folder / "stations.csv")
        for name in ("gauge_value", "importance", "ablation"):
            df = getattr(self, name)
            if df is not None:
                df.to_csv(folder / f"{name}.csv", index=False)
        info = {
            "multigauge_version": __version__,
            "basin": self.basin_name,
            "area_km2": self.area_km2,
            "flow_unit": self.flow_unit,
            "periods": {k: [str(a.date()), str(b.date())] for k, (a, b) in self.periods.items()},
            "pretrained_model": self.base_info,
            "single_station": self.single_station,
            "single_station_reason": self.single_station_reason,
            "gr4j_parameters": self.gr4j_params,
            "notes": self.notes,
            "config": self.config.to_dict(),
        }
        (folder / "run_info.json").write_text(json.dumps(info, indent=2, default=str))
        return folder


# =========================================================================
def make_periods(dates: pd.DatetimeIndex, cfg: StudyConfig, lookback: int) -> dict:
    given = {"train": cfg.train_period, "validation": cfg.validation_period, "test": cfg.test_period}
    if all(v is not None for v in given.values()):
        periods = {k: (pd.Timestamp(v[0]), pd.Timestamp(v[1])) for k, v in given.items()}
    else:
        f = np.asarray(cfg.split_fractions, float)
        f = f / f.sum()
        n = len(dates)
        c1, c2 = int(n * f[0]), int(n * (f[0] + f[1]))
        periods = {"train": (dates[0], dates[c1 - 1]), "validation": (dates[c1], dates[c2 - 1]), "test": (dates[c2], dates[-1])}
        for k, v in given.items():
            if v is not None:
                periods[k] = (pd.Timestamp(v[0]), pd.Timestamp(v[1]))
    for k, (a, b) in periods.items():
        if a > b:
            raise ValueError(f"The {k} period starts after it ends.")
        if a < dates[0] or b > dates[-1]:
            raise ValueError(f"The {k} period ({a.date()} to {b.date()}) is outside the data ({dates[0].date()} to {dates[-1].date()}).")
    n_train = int(((dates >= periods["train"][0]) & (dates <= periods["train"][1])).sum())
    if n_train < lookback + 365:
        raise ValueError(
            f"The training period has {n_train} days; at least {lookback + 365} are needed "
            f"(the model's {lookback}-day input window plus one year)."
        )
    return periods


def _masks(dates, periods):
    return {k: np.asarray((dates >= a) & (dates <= b)) for k, (a, b) in periods.items()}


def _ensemble(preds: list) -> pd.DataFrame:
    """Average predictions over seeds (quantiles are averaged, i.e. Vincentised)."""
    return sum(preds) / len(preds) if len(preds) > 1 else preds[0]


def run_study(
    data: BasinData,
    base: StreamflowModel,
    config: Optional[StudyConfig] = None,
    progress: Optional[Callable[[float, str], None]] = None,
    hybrid=None,
) -> StudyResult:
    """Run the full comparison. ``hybrid`` (a pre-trained ``HybridModel``)
    adds the fine-tuned δHBV model and an LSTM + δHBV ensemble."""
    cfg = config or StudyConfig()
    notes = list(data.notes)
    basin = data.to_basin()
    L = base.cfg.lookback
    periods = make_periods(basin.dates, cfg, L)
    masks = _masks(basin.dates, periods)
    obs = basin.q_mm.to_numpy()
    vor = voronoi_weights(data.stations, data.boundary)
    if not vor.used_boundary:
        notes.append("No basin boundary was supplied: Voronoi weights use a buffered convex hull of the gauges.")
    w = vor.weights
    stations = data.station_ids
    seeds = [base.cfg.seed + k for k in range(max(1, cfg.n_seeds))]
    gauge_inputs = [s for s in STRATEGIES if s in cfg.gauge_inputs]
    if base.info.get("source") == "synthetic" and data.meta.get("synthetic_zones") is None:
        notes.append("Warning: the pre-trained model was trained on synthetic data and is meant only for the demo.")

    # ---- job accounting for progress
    n_jobs = (len(gauge_inputs) + 1) * len(seeds)
    n_jobs += int(cfg.include_local) * len(seeds) + int(cfg.include_gr4j) + int(cfg.include_zero_shot)
    n_jobs += int(hybrid is not None)
    if cfg.run_ablation and "disaggregated" in gauge_inputs and len(stations) > 1:
        n_jobs += len(stations) * len(seeds)
    done = [0]

    def step(msg):
        if progress:
            progress(min(done[0] / max(n_jobs, 1), 0.999), msg)

    def ep_cb(label):
        def cb(e, n, s):
            if progress:
                progress(min((done[0] + e / n) / n_jobs, 0.999), f"{label}: epoch {e}/{n} (val NSE {s:.2f})")
        return cb

    preds_by_model: dict = {}
    rows = []

    def record(name, seed, pred):
        preds_by_model.setdefault(name, []).append(pred)
        for per in PERIODS:
            m = masks[per]
            for k, v in evaluate_probabilistic(obs[m], pred[m]).items():
                rows.append({"model": name, "seed": seed, "period": per, "metric": k, "value": v})

    def gauge_basin(strategy, station=None, drop=None):
        feats = build_features(data.rainfall, strategy, w, station=station, drop=drop)
        return basin.with_extra(feats)

    # ---- 1. baselines
    gr4j_params = None
    if cfg.include_gr4j:
        step("Calibrating GR4J")
        g = GR4JModel.calibrate(basin, masks["train"], maxiter=cfg.gr4j_maxiter)
        gr4j_params = g.to_dict()
        record("gr4j", 0, g.predict(basin))
        done[0] += 1
    if cfg.include_zero_shot:
        step("Pre-trained model (no fine-tuning)")
        record("pretrained", base.cfg.seed, base.predict(basin))
        done[0] += 1

    # ---- 2. fine-tuned, gridded only
    for sd in seeds:
        step("Fine-tuning (gridded only)")
        m = finetune(base, basin, masks["train"], masks["validation"], cfg.finetune_lr, cfg.finetune_epochs,
                     seed=sd, progress=ep_cb("Fine-tuning (gridded only)"))
        record("finetuned", sd, m.predict(basin))
        done[0] += 1

    # ---- 3. disaggregated first (drives gauge ranking)
    importance, disagg_model = None, None
    order = (["disaggregated"] if "disaggregated" in gauge_inputs else []) + [s for s in gauge_inputs if s != "disaggregated"]
    single, reason = None, ""
    for strategy in order:
        if strategy == "single":
            single, reason = _choose_single(cfg.single_station, importance, data, stations)
        for sd in seeds:
            b = gauge_basin(strategy, station=single)
            label = LABELS[GAUGE_MODELS[strategy]]
            step(label)
            m = finetune(base, b, masks["train"], masks["validation"], cfg.finetune_lr, cfg.finetune_epochs,
                         seed=sd, progress=ep_cb(label))
            record(GAUGE_MODELS[strategy], sd, m.predict(b))
            done[0] += 1
            if strategy == "disaggregated" and sd == seeds[0]:
                disagg_model = m
                if len(stations) > 1:
                    step("Permutation importance")
                    importance = permutation_importance(m, b, masks["validation"], cfg.permutation_repeats, seed=sd)

    # ---- 4. local LSTM from scratch (gridded + all gauges)
    if cfg.include_local:
        lcfg = cfg.local_config or ModelConfig.preset("local")
        lcfg.lookback = min(lcfg.lookback, L)
        b = gauge_basin("disaggregated")
        for sd in seeds:
            lcfg.seed = sd
            step("Local LSTM from scratch")
            m = train_local(b, masks["train"], masks["validation"], lcfg, progress=ep_cb("Local LSTM"))
            record("local", sd, m.predict(b))
            done[0] += 1

    # ---- 5. physics-guided hybrid and ensemble
    if hybrid is not None:
        from .hybrid import ensemble

        step("Fine-tuning the δHBV hybrid")
        hm = hybrid.finetune(basin, masks["train"], masks["validation"], epochs=cfg.finetune_epochs,
                             progress=ep_cb("δHBV hybrid"))
        hp = hm.predict(basin)
        record("dhbv", 0, hp)
        partner = "ft_disaggregated" if "ft_disaggregated" in preds_by_model else "finetuned"
        lstm_p = _ensemble(preds_by_model[partner])
        record("ensemble", 0, ensemble([lstm_p[["mean"]], hp]))
        done[0] += 1

    metrics = pd.DataFrame(rows)

    # ---- ensembles and summary
    predictions, summ = {}, []
    for name in ORDER:
        if name not in preds_by_model:
            continue
        p = _ensemble(preds_by_model[name])
        p["observed"] = obs
        p["period"] = "unused"
        for per in PERIODS:
            p.loc[masks[per], "period"] = per
        predictions[name] = p
        for per in PERIODS:
            m = masks[per]
            ev = evaluate_probabilistic(obs[m], p.drop(columns=["observed", "period"])[m])
            sn = metrics.query("model == @name and period == @per and metric == 'NSE'")["value"]
            ev["NSE_seed_std"] = float(sn.std()) if len(sn) > 1 else np.nan
            summ.append({"model": name, "period": per, **ev})
    summary = pd.DataFrame(summ).set_index(["model", "period"])

    # ---- gauge value over the fine-tuned gridded-only model
    gauge_value = None
    if "finetuned" in predictions and gauge_inputs:
        gv = []
        for s in gauge_inputs:
            mname = GAUGE_MODELS[s]
            for per in ("validation", "test"):
                a, b_ = summary.loc[(mname, per)], summary.loc[("finetuned", per)]
                gv.append({
                    "gauge_input": s, "period": per,
                    "NSE_gain": a["NSE"] - b_["NSE"], "KGE_gain": a["KGE"] - b_["KGE"],
                    "CRPS_reduction_pct": 100 * (b_["CRPS"] - a["CRPS"]) / b_["CRPS"] if "CRPS" in a and b_["CRPS"] else np.nan,
                    "PeakError_change": a["PeakError"] - b_["PeakError"],
                })
        gauge_value = add_seed_spread(pd.DataFrame(gv), metrics)

    # ---- ablation
    ablation = None
    if cfg.run_ablation and disagg_model is not None and len(stations) > 1:
        rows_ab = []
        base_scores = {per: summary.loc[("ft_disaggregated", per), "NSE"] for per in ("validation", "test")}
        for sid in stations:
            ps = []
            for sd in seeds:
                b = gauge_basin("disaggregated", drop=[sid])
                step(f"Ablation: without {sid}")
                m = finetune(base, b, masks["train"], masks["validation"], cfg.finetune_lr, cfg.finetune_epochs,
                             seed=sd, progress=ep_cb(f"Ablation without {sid}"))
                ps.append(m.predict(b)["mean"].to_numpy())
                done[0] += 1
            pm = np.mean(ps, axis=0)
            r = {"station": sid}
            for per in ("validation", "test"):
                s = nse(obs[masks[per]], pm[masks[per]])
                r[f"{per}_nse_without"] = s
                r[f"{per}_nse_change"] = s - base_scores[per]
            rows_ab.append(r)
        ablation = pd.DataFrame(rows_ab).sort_values("test_nse_change").reset_index(drop=True)

    # ---- station table
    st = data.summary().join(vor.weights).join(vor.areas_km2).join(gap_fill_report(data.rainfall))
    if data.outlet is not None:
        st = st.join(distance_km(data.stations, *data.outlet))
    if importance is not None:
        st = st.join(importance.set_index("station")[["importance", "importance_std"]])
    if ablation is not None:
        st = st.join(ablation.set_index("station")[["test_nse_change"]])

    if progress:
        progress(1.0, "Done")
    return StudyResult(
        basin_name=data.name, area_km2=float(data.area_km2), flow_unit=data.flow_unit, config=cfg,
        periods=periods, base_info=dict(base.info, lookback=L, hidden_size=base.cfg.hidden_size, head=base.cfg.head),
        metrics=metrics, summary=summary, predictions=predictions, gauge_value=gauge_value,
        importance=importance, ablation=ablation, voronoi=vor, single_station=single,
        single_station_reason=reason, station_summary=st, gr4j_params=gr4j_params, notes=notes,
    )


def _choose_single(choice, importance, data, stations):
    if choice == "auto":
        if importance is not None:
            return importance.iloc[0]["station"], "highest permutation importance (validation period)"
        return stations[0], "first gauge (importance not available)"
    if choice == "nearest_outlet":
        if data.outlet is None:
            raise ValueError("'nearest_outlet' needs the outlet location.")
        d = distance_km(data.stations, *data.outlet)
        return d.idxmin(), f"closest to outlet ({d.min():.1f} km)"
    if choice not in stations:
        raise ValueError(f"Gauge '{choice}' is not in the rainfall data.")
    return choice, "chosen by user"


def permutation_importance(model: StreamflowModel, basin: Basin, eval_mask: np.ndarray,
                           n_repeats: int = 5, seed: int = 0) -> pd.DataFrame:
    """Drop in validation NSE when one gauge's record is shuffled in time."""
    rng = np.random.default_rng(seed)
    obs = basin.q_mm.to_numpy()
    base_pred = model.predict(basin, quantiles=())["mean"].to_numpy()
    base_score = nse(obs[eval_mask], base_pred[eval_mask])
    rows = []
    for col in basin.extra.columns:
        drops = []
        for _ in range(n_repeats):
            ex = basin.extra.copy()
            ex[col] = rng.permutation(ex[col].to_numpy())
            p = model.predict(basin.with_extra(ex), quantiles=())["mean"].to_numpy()
            drops.append(base_score - nse(obs[eval_mask], p[eval_mask]))
        rows.append({"station": col[2:] if col.startswith("P_") else col,
                     "importance": float(np.mean(drops)), "importance_std": float(np.std(drops))})
    out = pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)
    out.attrs["baseline_nse"] = base_score
    return out


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
