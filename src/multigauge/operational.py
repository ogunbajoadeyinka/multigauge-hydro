"""Phase 3: running forecasts on a schedule.

Daymet, which the models are trained on, is released with a long delay,
so real-time runs take recent weather from Open-Meteo instead and correct
it to look like Daymet. The correction is fitted once, during setup, on
the years where both sources overlap: a monthly ratio for precipitation
and monthly offsets for temperature and radiation.

Typical use::

    multigauge forecast-setup my_basin/ --site 07316000 --out forecasts/washita
    multigauge forecast-run forecasts/            # daily, e.g. GitHub Actions

Each run writes ``latest.json`` and appends to ``history.csv`` in the
basin's forecast folder; the dashboard reads them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .basin import Basin
from .features import flow_to_mm_per_day, mm_per_day_to_flow
from .forecast import ForecastConfig, ForecastModel


@dataclass
class BiasCorrection:
    """Monthly correction of one weather source towards a reference."""

    prcp_ratio: dict
    offsets: dict  # variable -> {month: offset}

    @classmethod
    def fit(cls, source: pd.DataFrame, reference: pd.DataFrame) -> "BiasCorrection":
        idx = source.index.intersection(reference.index)
        if len(idx) < 365:
            raise ValueError("At least one year of overlap is needed to fit the bias correction.")
        s, r = source.loc[idx], reference.loc[idx]
        m = idx.month
        ratio = {}
        for mo in range(1, 13):
            a, b = s["prcp"][m == mo].sum(), r["prcp"][m == mo].sum()
            ratio[mo] = float(np.clip(b / a, 0.3, 3.0)) if a > 0 else 1.0
        offsets = {v: {mo: float((r[v][m == mo] - s[v][m == mo]).mean()) for mo in range(1, 13)}
                   for v in ("tmax", "tmin", "srad")}
        return cls(ratio, offsets)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        from .forecast import vp_from_tmin

        out = df.copy()
        m = out.index.month
        out["prcp"] = out["prcp"] * np.array([self.prcp_ratio[int(x)] for x in m])
        for v, off in self.offsets.items():
            out[v] = out[v] + np.array([off[int(x)] for x in m])
        out["vp"] = vp_from_tmin(out["tmin"])
        return out

    def to_dict(self):
        return {"prcp_ratio": self.prcp_ratio, "offsets": self.offsets}

    @classmethod
    def from_dict(cls, d):
        conv = lambda x: {int(k): float(v) for k, v in x.items()}
        return cls(conv(d["prcp_ratio"]), {k: conv(v) for k, v in d["offsets"].items()})


def setup_forecast(data, base, site: Optional[str], out_dir, cfg: Optional[ForecastConfig] = None,
                   split=(0.7, 0.15, 0.15), fetch_archive: bool = True, progress=None) -> dict:
    """Train a forecast model for a basin and store everything a daily run needs."""
    from .download import openmeteo_archive
    from .transfer import finetune

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    basin = data.to_basin()
    n = len(basin.dates)
    c1, c2 = int(n * split[0]), int(n * (split[0] + split[1]))
    tr = np.zeros(n, bool); tr[:c1] = True
    va = np.zeros(n, bool); va[c1:c2] = True
    te = ~(tr | va)
    ft = finetune(base, basin, tr, va, seed=base.cfg.seed)
    fm = ForecastModel(ft, cfg).fit(basin, tr, va, progress=progress)
    fm.save(out / "model")
    issue = np.where(te)[0]
    issue = issue[(issue >= ft.cfg.lookback - 1) & (issue < n - fm.cfg.horizon)]
    skill = fm.skill_by_lead(basin, issue)
    skill.to_csv(out / "hindcast_skill.csv", index=False)

    meta = {
        "name": data.name, "usgs_site": site, "area_km2": data.area_km2, "flow_unit": data.flow_unit,
        "lat": basin.lat, "lon": basin.lon, "statics": basin.statics.to_dict(),
        "boundary": json.loads(json.dumps(__import__("shapely").geometry.mapping(data.boundary))) if data.boundary is not None else None,
    }
    if fetch_archive and data.boundary is not None:
        om = openmeteo_archive(data.boundary, str(data.forcings.index[0].date()), str(data.forcings.index[-1].date()))
        meta["bias_correction"] = BiasCorrection.fit(om, data.forcings).to_dict()
    (out / "basin.json").write_text(json.dumps(meta, indent=1, default=float))
    return {"folder": str(out), "hindcast_skill": skill}


def run_forecast(folder, today: Optional[pd.Timestamp] = None) -> dict:
    """Issue a forecast for one configured basin (needs network access)."""
    from shapely.geometry import shape

    from .download import openmeteo_archive, openmeteo_forecast, usgs_recent_flow

    folder = Path(folder)
    meta = json.loads((folder / "basin.json").read_text())
    fm = ForecastModel.load(folder / "model")
    geom = shape(meta["boundary"]) if meta.get("boundary") else (meta["lat"], meta["lon"])
    L = fm.base.cfg.lookback
    today = (today or pd.Timestamp.utcnow().tz_localize(None)).normalize()

    hist_start = today - pd.Timedelta(days=L + 30)
    arch = openmeteo_archive(geom, str(hist_start.date()), str((today - pd.Timedelta(days=6)).date()))
    fc = openmeteo_forecast(geom, days=fm.cfg.horizon + 1, past_days=7)
    weather = pd.concat([arch, fc[~fc.index.isin(arch.index)]]).sort_index()
    if meta.get("bias_correction"):
        weather = BiasCorrection.from_dict(meta["bias_correction"]).apply(weather)
    hist = weather.loc[: today - pd.Timedelta(days=1)]
    future = weather.loc[today:].iloc[: fm.cfg.horizon]

    q = pd.Series(np.nan, index=hist.index)
    if meta.get("usgs_site"):
        try:
            obs = usgs_recent_flow(meta["usgs_site"], days=L + 40)
            q = flow_to_mm_per_day(obs.reindex(hist.index), meta["area_km2"], "m3/s")
        except Exception as exc:  # forecasts still run without recent flow
            meta.setdefault("warnings", []).append(f"USGS flow unavailable: {exc}")
    basin = Basin(meta["name"], hist, q, meta["area_km2"], meta["lat"], meta["lon"],
                  statics=pd.Series(meta["statics"]))
    df = fm.forecast(basin, future)
    for c in df.columns:
        if c == "mean" or (c.startswith("q") and c[1:].isdigit()):
            df[c] = mm_per_day_to_flow(df[c], meta["area_km2"], meta.get("flow_unit", "m3/s"))
    issued = {
        "basin": meta["name"], "issued": str(today.date()), "unit": meta.get("flow_unit", "m3/s"),
        "recent_observed": {str(k.date()): (None if not np.isfinite(v) else float(v))
                            for k, v in mm_per_day_to_flow(q.iloc[-30:], meta["area_km2"], "m3/s").items()},
        "forecast": [{"date": str(i.date()), **{k: float(v) for k, v in r.items()}} for i, r in df.iterrows()],
        "warnings": meta.get("warnings", []),
        "note": "Forecast skill is estimated from hindcasts with observed weather; real forecasts are less skilful.",
    }
    (folder / "latest.json").write_text(json.dumps(issued, indent=1))
    hist_file = folder / "history.csv"
    row = df.assign(issued=str(today.date())).reset_index()
    row.to_csv(hist_file, mode="a", header=not hist_file.exists(), index=False)
    return issued


def run_all(root) -> list:
    root = Path(root)
    results = []
    for f in sorted(root.glob("*/basin.json")):
        try:
            results.append(run_forecast(f.parent))
        except Exception as exc:
            results.append({"basin": f.parent.name, "error": str(exc)})
    return results
