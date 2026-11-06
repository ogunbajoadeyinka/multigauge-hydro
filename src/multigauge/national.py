"""National gauge-value survey: the same study on many U.S. basins.

For each basin, public data are downloaded (``prepare_usgs_basin``), the
pre-trained model is fine-tuned with and without the basin's NOAA gauges,
and the gain is recorded with its seed-to-seed spread. One row per basin
goes into ``national_summary.csv``; ``fig_national_map`` draws the map.

The run is resumable: a basin with ``results/run_info.json`` is skipped,
and a basin that failed is recorded in ``failed.csv`` and not retried.
"""
from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

PERIODS = dict(train_period=("2001-10-01", "2013-09-30"), validation_period=("2013-10-01", "2017-09-30"),
               test_period=("2017-10-01", "2023-09-30"))
START, END = "2000-10-01", "2023-09-30"

# USGS hydrologic regions (2-digit HUC) in the conterminous U.S.
HUC2_NAMES = {
    "01": "New England", "02": "Mid-Atlantic", "03": "South Atlantic-Gulf", "04": "Great Lakes",
    "05": "Ohio", "06": "Tennessee", "07": "Upper Mississippi", "08": "Lower Mississippi",
    "09": "Souris-Red-Rainy", "10": "Missouri", "11": "Arkansas-White-Red", "12": "Texas-Gulf",
    "13": "Rio Grande", "14": "Upper Colorado", "15": "Lower Colorado", "16": "Great Basin",
    "17": "Pacific Northwest", "18": "California",
}


_TRANSIENT = ("504", "502", "503", "500 Server", "429", "Timeout", "timed out", "Connection", "Max retries")


def _transient(reason: str) -> bool:
    return any(k in reason for k in _TRANSIENT)


def light_config(n_seeds: int = 2):
    """Study settings for a many-basin survey: fewer seeds and inputs."""
    from .study import StudyConfig

    return StudyConfig(n_seeds=n_seeds, gauge_inputs=("lumped", "voronoi", "disaggregated"), finetune_epochs=20,
                       include_local=False, include_gr4j=True, gr4j_maxiter=25, include_zero_shot=True,
                       permutation_repeats=2, **PERIODS)


def read_camels_catalog(name_file, topo_file) -> pd.DataFrame:
    """CAMELS gauge list with region, location and area (from camels_name.txt and camels_topo.txt)."""
    names = pd.read_csv(name_file, sep=";", dtype={"gauge_id": str, "huc_02": str})
    topo = pd.read_csv(topo_file, sep=";", dtype={"gauge_id": str})
    for df in (names, topo):
        df["gauge_id"] = df["gauge_id"].str.zfill(8)
    cat = names.merge(topo, on="gauge_id")
    cat["huc_02"] = cat["huc_02"].str.zfill(2)
    return cat.rename(columns={"gauge_lat": "lat", "gauge_lon": "lon", "area_gages2": "area_km2"})[
        ["gauge_id", "gauge_name", "huc_02", "lat", "lon", "area_km2"]].set_index("gauge_id")


def candidate_order(catalog: pd.DataFrame, per_region: int = 5, min_area_km2: float = 150.0,
                    max_area_km2: float = 5000.0, seed: int = 42) -> pd.DataFrame:
    """All eligible basins in the order to try them: regions are interleaved
    (round-robin) so that an early stop still covers the whole country, and
    within a region basins are shuffled with larger ones first (they are more
    likely to contain two or more NOAA gauges). ``rank`` says how far down
    its region's list a basin is; the runner stops a region after
    ``per_region`` successes."""
    rng = np.random.default_rng(seed)
    c = catalog[(catalog["area_km2"] >= min_area_km2) & (catalog["area_km2"] <= max_area_km2)].copy()
    c["key"] = rng.random(len(c)) / np.sqrt(c["area_km2"])  # random, larger basins tend to come first
    c = c.sort_values("key")
    c["rank"] = c.groupby("huc_02").cumcount()
    return c.sort_values(["rank", "huc_02"]).drop(columns="key")


def run_national(candidates: pd.DataFrame, out, model, per_region: int = 5, hybrid=None,
                 time_budget_h: float = 10.5, config=None, prepare: Optional[Callable] = None,
                 log=print) -> pd.DataFrame:
    """Run the study on candidates until ``per_region`` basins per region
    succeed or the time budget is spent. Returns the summary table."""
    from . import BasinData, run_study, write_report
    from .prepare import prepare_usgs_basin

    prepare = prepare or prepare_usgs_basin
    cfg = config or light_config()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    failed_f = out / "failed.csv"
    failed = pd.read_csv(failed_f, dtype={"site": str}) if failed_f.exists() else pd.DataFrame(columns=["site", "reason"])
    failed = failed[~failed["reason"].astype(str).map(_transient)]
    t0 = time.time()
    done = {p.parent.parent.name for p in out.glob("*/results/run_info.json")}
    for site, row in candidates.iterrows():
        n_ok = sum(1 for s in done if s in candidates.index and candidates.loc[s, "huc_02"] == row["huc_02"])
        if site in done or n_ok >= per_region or site in set(failed["site"]):
            continue
        if (time.time() - t0) / 3600 > time_budget_h:
            log(f"Time budget of {time_budget_h} h reached; re-run to continue.")
            break
        folder = out / site
        try:
            t = time.time()
            if (folder / "data" / "basin.json").exists():
                d = BasinData.from_folder(folder / "data")
            else:
                d = prepare(site, START, END, folder / "data", name=f"{row['gauge_name']}".strip(), progress=lambda m: None)
            res = run_study(d, model, cfg, hybrid=hybrid)
            res.save(folder / "results")
            write_report(res, d, folder / "results" / "report.html")
            done.add(site)
            log(f"[{len(done)}] {site} {row['gauge_name'][:40]} ({HUC2_NAMES.get(row['huc_02'], row['huc_02'])}): "
                f"{d.rainfall.shape[1]} gauges, {(time.time() - t) / 60:.1f} min")
        except Exception as exc:  # record and move on; one basin must not stop the survey
            reason = str(exc).splitlines()[0][:200] if str(exc) else type(exc).__name__
            if _transient(reason):  # a server or network hiccup: try this basin again on the next run
                log(f"  {site} postponed (download error, retried next run): {reason[:90]}")
                continue
            failed = pd.concat([failed, pd.DataFrame([{"site": site, "reason": reason}])], ignore_index=True)
            failed.to_csv(failed_f, index=False)
            log(f"  {site} skipped: {reason}")
            if not isinstance(exc, (RuntimeError, ValueError)):
                traceback.print_exc()
        national_summary(out, candidates).to_csv(out / "national_summary.csv", index=False)
    return national_summary(out, candidates)


def national_summary(out, catalog: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """One row per finished basin: skill with and without gauges, the gain,
    its seed spread and a verdict."""
    from .features import static_attributes
    from .io import BasinData
    from .results import load_saved_study

    rows = []
    for f in sorted(Path(out).glob("*/results/run_info.json")):
        site = f.parent.parent.name
        r = load_saved_study(f.parent)
        s = r.summary.xs("test", level="period")
        gv = r.gauge_value[r.gauge_value["period"] == "test"].set_index("gauge_input")
        best = gv["NSE_gain"].idxmax()
        st = r.station_summary
        row = {"site": site, "basin": r.basin_name, "area_km2": round(r.area_km2, 1), "n_gauges": len(st),
               "gauges_per_1000km2": round(1000 * len(st) / r.area_km2, 2)}
        if catalog is not None and site in catalog.index:
            row.update(huc_02=catalog.loc[site, "huc_02"], region=HUC2_NAMES.get(catalog.loc[site, "huc_02"], ""),
                       lat=catalog.loc[site, "lat"], lon=catalog.loc[site, "lon"])
        else:
            row.update(lat=np.nan, lon=np.nan)
        try:
            d = BasinData.from_folder(f.parent.parent / "data")
            a = static_attributes(d.forcings, d.area_km2, (d.outlet or d.centroid)[0])
            row.update(aridity=round(float(a["aridity"]), 2), frac_snow=round(float(a["frac_snow"]), 2),
                       p_mean=round(float(a["p_mean"]), 2), share_day_shifted=round(float((d.stations["day_shift"] != 0).mean()), 2))
            if np.isnan(row.get("lat", np.nan)):
                row["lat"], row["lon"] = d.outlet or d.centroid
        except Exception:
            pass
        for k in ("gr4j", "pretrained", "finetuned", "ft_lumped", "ft_voronoi", "ft_disaggregated"):
            if k in s.index:
                row[f"NSE_{k}"] = round(float(s.loc[k, "NSE"]), 3)
        row.update(best_gauge_input=best, NSE_gain=round(float(gv.loc[best, "NSE_gain"]), 3),
                   NSE_gain_seed_sd=round(float(gv.loc[best, "NSE_gain_seed_sd"]), 3) if "NSE_gain_seed_sd" in gv else np.nan,
                   verdict=gv.loc[best, "verdict"] if "verdict" in gv else "",
                   CRPS_reduction_pct=round(float(gv.loc[best, "CRPS_reduction_pct"]), 1))
        rows.append(row)
    return pd.DataFrame(rows)


def fig_national_map(summary: pd.DataFrame):
    """Map of the NSE gain from adding gauges, one marker per basin."""
    import plotly.graph_objects as go

    s = summary.dropna(subset=["lat", "lon"]).copy()
    gain = s["NSE_gain"].clip(-0.1, 0.1)
    sym = s["verdict"].map({"helps": "circle", "hurts": "x", "negligible": "circle-open",
                            "within noise": "circle-open"}).fillna("circle-open")
    text = (s["basin"] + "<br>USGS " + s["site"] + "<br>gauges: " + s["n_gauges"].astype(str)
            + "<br>NSE without gauges: " + s.get("NSE_finetuned", pd.Series(np.nan, index=s.index)).round(2).astype(str)
            + "<br>gain: " + s["NSE_gain"].round(3).astype(str) + " (" + s["verdict"].astype(str) + ")")
    fig = go.Figure(go.Scattergeo(
        lat=s["lat"], lon=s["lon"], text=text, hoverinfo="text", mode="markers",
        marker=dict(size=9 + 2 * np.sqrt(s["n_gauges"]), color=gain, cmin=-0.1, cmax=0.1, symbol=sym,
                    colorscale=[[0, "#e34948"], [0.5, "#f0efec"], [1, "#2a78d6"]], line=dict(width=0.6, color="#444"),
                    colorbar=dict(title="NSE gain<br>from gauges"))))
    fig.update_geos(scope="usa", projection_type="albers usa", showland=True, landcolor="#f2f2f2",
                    showlakes=True, lakecolor="#ffffff", subunitcolor="#bbbbbb")
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=30, b=0),
                      title="Gain in test NSE from adding NOAA rain gauges to the fine-tuned model "
                            "(filled = larger than training noise)")
    return fig
