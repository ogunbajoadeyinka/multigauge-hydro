"""Synthetic basins where the truth is known.

The simulator builds a basin polygon on a fine grid of ~4 km cells, rains
on it with a mix of small convective cells and broad frontal systems,
turns rain into runoff with a nonlinear (saturation-excess) soil bucket
whose capacity varies in space, adds a degree-day snow pack, and routes
each cell's runoff to the outlet with a travel time that grows with
distance.

Two kinds of rainfall information are produced, mimicking real data:

* a **gridded product** (like Daymet): the basin-average of the true field
  with multiplicative error and occasional one-day timing errors, as
  gridded products smooth and mistime convective storms;
* **rain gauges**: point samples of the true field with small error and
  random gaps.

Because storms are spatially variable and runoff generation is nonlinear,
the gauges carry information the gridded average misses, which is the
effect multigauge is designed to measure.

``generate_basin`` returns a full user-style basin (gauges, boundary,
gridded forcings, flow, sub-basins). ``generate_pretraining_set`` returns
many varied basins for testing the pre-training pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, mapping

from .basin import Basin
from .io import BasinData
from .spatial import KM_PER_DEG_LAT, KM_PER_DEG_LON_EQ


def _basin_polygon(rng, length_km, width_km, n=48):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    wobble = 1 + 0.12 * np.sin(3 * t + rng.uniform(0, 6)) + 0.06 * np.sin(7 * t + rng.uniform(0, 6))
    x = 0.5 * length_km * np.cos(t) * wobble
    y = 0.5 * width_km * np.sin(t) * wobble * (1 - 0.25 * np.cos(t))
    return Polygon(np.column_stack([x, y]))


def _simulate(
    rng,
    years: float,
    start: str,
    n_gauges: int,
    length_km: float,
    width_km: float,
    lat0: float,
    rain_scale: float = 1.0,
    cap_scale: float = 1.0,
    convective_share: float = 0.6,
    temp_offset: float = 0.0,
    cell_km: float = 4.0,
    n_zones: int = 0,
):
    poly = _basin_polygon(rng, length_km, width_km)
    minx, miny, maxx, maxy = poly.bounds
    edge = poly.exterior.interpolate(poly.exterior.project(Point(maxx, 0.0)))
    outlet = np.array([edge.x, edge.y])

    gx, gy = np.meshgrid(np.arange(minx, maxx, cell_km) + cell_km / 2, np.arange(miny, maxy, cell_km) + cell_km / 2)
    cells = np.column_stack([gx.ravel(), gy.ravel()])
    cells = cells[np.array([poly.contains(Point(p)) for p in cells])]
    G, cell_area = len(cells), cell_km**2

    gauges = []
    tries = 0
    while len(gauges) < n_gauges and tries < 20000:
        tries += 1
        p = np.array([rng.uniform(minx, maxx), rng.uniform(miny, maxy)])
        if poly.buffer(-2).contains(Point(p)) and all(np.hypot(*(p - q)) > 12 for q in gauges):
            gauges.append(p)
    gauges = np.array(gauges).reshape(-1, 2)

    dates = pd.date_range(start, periods=int(years * 365.25), freq="D")
    T = len(dates)
    doy = dates.dayofyear.values
    season = 0.6 + 0.5 * np.sin(2 * np.pi * (doy - 80) / 365.25)

    # ---- rain field on cells and gauges
    pts = np.vstack([cells, gauges])
    field = np.zeros((T, len(pts)), dtype=np.float32)
    wet_prev = False
    for t in range(T):
        wet_prev = rng.random() < (0.45 if wet_prev else 0.18) * season[t]
        if not wet_prev:
            continue
        day = np.zeros(len(pts))
        for _ in range(1 + rng.poisson(1.2)):
            conv = rng.random() < convective_share * season[t]
            c = np.array([rng.uniform(minx - 20, maxx + 20), rng.uniform(miny - 20, maxy + 20)])
            r = rng.uniform(6, 18) if conv else rng.uniform(40, 120)
            peak = rng.exponential(45 if conv else 15) * rain_scale
            day += peak * np.exp(-((pts - c) ** 2).sum(axis=1) / (2 * r * r))
        day *= rng.lognormal(0, 0.25, size=len(pts))
        day[day < 0.25] = 0.0
        field[t] = day
    rain_cells, rain_gauges = field[:, :G], field[:, G:]

    # ---- temperature, radiation, vapour pressure (basin-wide)
    lat_adj = -0.6 * (lat0 - 35)
    tmax = 22 + lat_adj + temp_offset + 12 * np.sin(2 * np.pi * (doy - 110) / 365.25) + rng.normal(0, 3, T)
    wet = rain_cells.mean(axis=1) > 1
    dtr = np.where(wet, 7.0, 13.0) + rng.normal(0, 1.5, T)
    tmin = tmax - np.clip(dtr, 3, 20)
    clear = 230 + 110 * np.sin(2 * np.pi * (doy - 80) / 365.25)
    srad = clear * np.where(wet, 0.45, 0.95) * rng.uniform(0.9, 1.05, T)
    vp = 611 * np.exp(17.27 * tmin / (tmin + 237.3))
    tmean = (tmax + tmin) / 2
    pet = np.clip(0.13 * (tmean + 5.0), 0.2, None) * (0.7 + 0.3 * srad / clear)

    # ---- snow + soil bucket per cell
    xn = (cells[:, 0] - minx) / max(maxx - minx, 1)
    cap = cap_scale * rng.uniform(0.8, 1.2, G) * (220 - 170 * xn)
    soil, gw, snow = cap * 0.5, np.full(G, 5.0), np.zeros(G)
    quick = np.zeros((T, G), dtype=np.float32)
    base = np.zeros((T, G), dtype=np.float32)
    for t in range(T):
        p = rain_cells[t].astype(float)
        if tmean[t] < 0:
            snow += p
            p = np.zeros(G)
        elif snow.any():
            melt = np.minimum(snow, 3.0 * tmean[t])
            snow -= melt
            p = p + melt
        infil_excess = np.maximum(p - 45.0, 0) * 0.5
        p_in = p - infil_excess
        frac = np.clip(soil / cap, 0, 1)
        sat_excess = p_in * frac**3
        soil = soil + p_in - sat_excess - pet[t] * frac
        perc = 0.02 * np.maximum(soil - 0.3 * cap, 0)
        soil = np.clip(soil - perc, 0, None)
        over = np.maximum(soil - cap, 0)
        soil -= over
        gw += perc
        bf = 0.015 * gw
        gw -= bf
        quick[t] = infil_excess + sat_excess + over
        base[t] = bf

    # ---- routing to the outlet
    dist = np.hypot(*(cells - outlet).T)
    delay = np.round(dist / 35.0).astype(int)
    q_cell = quick + base
    q_mm = np.zeros(T + delay.max() + 1)
    for d in np.unique(delay):
        q_mm[d : d + T] += quick[:, delay == d].sum(axis=1) * cell_area
    k = np.exp(-np.arange(8) / 1.5)
    q_mm = np.convolve(q_mm, k / k.sum())[:T] + base.sum(axis=1) * cell_area
    flow = q_mm * 1000 / 86400 * rng.lognormal(0, 0.05, T)  # m3/s

    # ---- gridded product: basin mean with error and timing slips
    gridded = rain_cells.mean(axis=1).astype(float) * rng.lognormal(0, 0.25, T)
    slip = (gridded > 0) & (rng.random(T) < 0.2)
    shifted = np.zeros(T)
    shifted[1:] = (gridded * slip)[:-1]
    gridded = gridded * ~slip + shifted

    # ---- optional sub-basins (bands ordered from outlet upstream)
    zones = None
    if n_zones and n_zones > 1:
        order = np.argsort(-dist)  # far first
        edges = np.quantile(dist, np.linspace(0, 1, n_zones + 1))
        zid = np.clip(np.searchsorted(edges, dist, side="right") - 1, 0, n_zones - 1)
        zone_rain = np.stack([rain_cells[:, zid == z].mean(axis=1) for z in range(n_zones)], axis=1)
        zone_rain = zone_rain * rng.lognormal(0, 0.2, zone_rain.shape)
        zone_q = np.stack([q_cell[:, zid == z].sum(axis=1) * cell_area for z in range(n_zones)], axis=1)
        zones = {"id": zid, "rain": zone_rain, "q_local_mm_km2": zone_q,
                 "area": np.array([(zid == z).sum() * cell_area for z in range(n_zones)])}

    forc = pd.DataFrame({"prcp": gridded.round(2), "srad": srad.round(1), "tmax": tmax.round(1),
                         "tmin": tmin.round(1), "vp": vp.round(0)}, index=pd.DatetimeIndex(dates, name="date"))
    return dict(poly=poly, cells=cells, gauges=gauges, rain_gauges=rain_gauges, forcings=forc,
                flow=flow, outlet=outlet, area_km2=G * cell_area, dates=dates, zones=zones, dist=dist)


def _to_lonlat(lat0, lon0):
    kx = KM_PER_DEG_LON_EQ * np.cos(np.radians(lat0))
    return lambda x, y: (lon0 + np.asarray(x) / kx, lat0 + np.asarray(y) / KM_PER_DEG_LAT)


def generate_basin(
    n_gauges: int = 10,
    years: int = 12,
    start: str = "2010-01-01",
    seed: int = 7,
    center: tuple = (35.0, -98.0),
    length_km: float = 120.0,
    width_km: float = 55.0,
    missing_fraction: float = 0.02,
    n_subbasins: int = 4,
    name: str = "Synthetic demo basin",
) -> BasinData:
    """A user-style basin: rain gauges, boundary, gridded forcings and flow."""
    rng = np.random.default_rng(seed)
    lat0, lon0 = center
    s = _simulate(rng, years, start, n_gauges, length_km, width_km, lat0, n_zones=n_subbasins)
    conv = _to_lonlat(lat0, lon0)

    ids = [f"G{i+1:02d}" for i in range(n_gauges)]
    rain = pd.DataFrame(s["rain_gauges"].round(2), index=pd.DatetimeIndex(s["dates"], name="date"), columns=ids)
    T = len(rain)
    if missing_fraction > 0:
        for j in range(n_gauges):
            for _ in range(rng.poisson(missing_fraction * T / 15)):
                a = rng.integers(0, T - 30)
                rain.iloc[a : a + rng.integers(3, 30), j] = np.nan
    glon, glat = conv(s["gauges"][:, 0], s["gauges"][:, 1])
    stations = pd.DataFrame({"lat": glat.round(5), "lon": glon.round(5)}, index=pd.Index(ids, name="id"))
    bx, by = s["poly"].exterior.xy
    blon, blat = conv(np.array(bx), np.array(by))
    olon, olat = conv(*s["outlet"])

    data = BasinData(
        rainfall=rain,
        stations=stations,
        streamflow=pd.Series(s["flow"].round(3), index=rain.index, name="flow"),
        boundary=Polygon(np.column_stack([blon, blat])),
        outlet=(float(olat), float(olon)),
        name=name,
        forcings=s["forcings"],
        area_km2=float(s["area_km2"]),
        flow_unit="m3/s",
    )
    if s["zones"] is not None:
        data.meta["synthetic_zones"] = s["zones"]
        data.meta["synthetic_cells_lonlat"] = np.column_stack(conv(s["cells"][:, 0], s["cells"][:, 1]))
    return data


def generate_pretraining_set(n_basins: int = 40, years: int = 12, start: str = "2000-01-01", seed: int = 100) -> list:
    """Many synthetic basins with varied size, climate and soils (no gauges)."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_basins):
        lat = rng.uniform(30, 46)
        s = _simulate(
            rng, years, start, 0,
            length_km=rng.uniform(25, 140), width_km=rng.uniform(15, 60), lat0=lat,
            rain_scale=rng.uniform(0.5, 1.6), cap_scale=rng.uniform(0.6, 1.6),
            convective_share=rng.uniform(0.2, 0.8), temp_offset=rng.uniform(-6, 4), cell_km=8.0,
        )
        q_mm = pd.Series(s["flow"] * 86400 / (s["area_km2"] * 1e6) * 1000, index=s["forcings"].index)
        out.append(Basin(f"S{i:03d}", s["forcings"], q_mm, s["area_km2"], lat, -98.0 + rng.uniform(-8, 8),
                         meta={"source": "synthetic"}))
    return out


def write_basin(data: BasinData, folder) -> Path:
    """Write a BasinData to the standard folder layout (also a format example)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    data.rainfall.to_csv(folder / "rainfall.csv", date_format="%Y-%m-%d")
    data.stations.to_csv(folder / "stations.csv")
    data.streamflow.rename("flow").to_frame().to_csv(folder / "streamflow.csv", date_format="%Y-%m-%d")
    if data.forcings is not None:
        data.forcings.to_csv(folder / "forcings.csv", date_format="%Y-%m-%d")
    if data.extra is not None:
        data.extra.to_csv(folder / "extra.csv", date_format="%Y-%m-%d")
    if data.boundary is not None:
        gj = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"name": data.name}, "geometry": mapping(data.boundary)}]}
        (folder / "boundary.geojson").write_text(json.dumps(gj))
    meta = {"name": data.name, "flow_unit": data.flow_unit, "rainfall_unit": "mm/day"}
    if data.area_km2:
        meta["area_km2"] = round(float(data.area_km2), 2)
    if data.outlet is not None:
        meta.update(outlet_lat=data.outlet[0], outlet_lon=data.outlet[1])
    (folder / "basin.json").write_text(json.dumps(meta, indent=2))
    return folder
