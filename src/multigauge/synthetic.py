"""Synthetic test basins with known, spatially variable rainfall.

The generator creates a basin polygon, a set of gauges, daily storm fields
(a mix of small convective cells and broad frontal systems) and routes
runoff from a fine grid of sub-areas to an outlet. Runoff generation is
nonlinear (a saturation-excess soil bucket), so where rain falls matters:
averaging gauges blurs storms and loses information, which is exactly the
effect the tool is designed to measure. Gauges only sample the true field.

Used for the built-in demo, the tests, and for anyone who wants to check
the method on a basin where the truth is known.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, mapping

from .io import BasinData
from .spatial import KM_PER_DEG_LAT, KM_PER_DEG_LON_EQ


def _basin_polygon(rng, length_km, width_km, n=48):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    wobble = 1 + 0.12 * np.sin(3 * t + rng.uniform(0, 6)) + 0.06 * np.sin(7 * t + rng.uniform(0, 6))
    x = 0.5 * length_km * np.cos(t) * wobble
    y = 0.5 * width_km * np.sin(t) * wobble * (1 - 0.25 * np.cos(t))
    return Polygon(np.column_stack([x, y]))


def generate_basin(
    n_gauges: int = 10,
    years: int = 12,
    start: str = "2010-01-01",
    seed: int = 7,
    center: tuple = (35.0, -98.0),
    length_km: float = 120.0,
    width_km: float = 55.0,
    missing_fraction: float = 0.02,
    name: str = "Synthetic demo basin",
) -> BasinData:
    rng = np.random.default_rng(seed)
    lat0, lon0 = center
    kx = KM_PER_DEG_LON_EQ * np.cos(np.radians(lat0))

    def to_lonlat(x, y):
        return lon0 + np.asarray(x) / kx, lat0 + np.asarray(y) / KM_PER_DEG_LAT

    poly = _basin_polygon(rng, length_km, width_km)
    minx, miny, maxx, maxy = poly.bounds
    # outlet: point on the boundary nearest the east end
    edge = poly.exterior.interpolate(poly.exterior.project(Point(maxx, 0.0)))
    outlet_xy = np.array([edge.x, edge.y])

    # fine "truth" grid (~4 km cells)
    gx, gy = np.meshgrid(np.arange(minx, maxx, 4.0) + 2, np.arange(miny, maxy, 4.0) + 2)
    cells = np.column_stack([gx.ravel(), gy.ravel()])
    inside = np.array([poly.contains(Point(p)) for p in cells])
    cells = cells[inside]
    G = len(cells)
    cell_area = 16.0  # km2

    # gauges: spread inside the basin, a couple near the edge
    gauges = []
    while len(gauges) < n_gauges:
        p = np.array([rng.uniform(minx, maxx), rng.uniform(miny, maxy)])
        if poly.buffer(-2).contains(Point(p)) and all(np.hypot(*(p - q)) > 12 for q in gauges):
            gauges.append(p)
    gauges = np.array(gauges)

    dates = pd.date_range(start, periods=int(years * 365.25), freq="D")
    T = len(dates)
    doy = dates.dayofyear.values
    season = 0.6 + 0.5 * np.sin(2 * np.pi * (doy - 80) / 365.25)  # wetter late spring

    # storm field on cells and gauges
    pts = np.vstack([cells, gauges])
    field = np.zeros((T, len(pts)), dtype=np.float32)
    wet_prev = False
    for t in range(T):
        p_wet = (0.45 if wet_prev else 0.18) * season[t]
        wet_prev = rng.random() < p_wet
        if not wet_prev:
            continue
        n_storms = 1 + rng.poisson(1.2)
        day = np.zeros(len(pts))
        for _ in range(n_storms):
            convective = rng.random() < 0.6 * season[t]
            c = np.array([rng.uniform(minx - 20, maxx + 20), rng.uniform(miny - 20, maxy + 20)])
            r = rng.uniform(6, 18) if convective else rng.uniform(40, 120)
            peak = rng.exponential(45 if convective else 15)
            d2 = ((pts - c) ** 2).sum(axis=1)
            day += peak * np.exp(-d2 / (2 * r * r))
        day *= rng.lognormal(0, 0.25, size=len(pts))  # small-scale noise
        day[day < 0.25] = 0.0
        field[t] = day
    rain_cells, rain_gauges = field[:, :G], field[:, G:]

    # temperature and potential ET (basin-wide)
    temp = 16 + 12 * np.sin(2 * np.pi * (doy - 110) / 365.25) + rng.normal(0, 3, T)
    pet = np.clip(0.13 * (temp + 5.0), 0.3, None)  # mm/day, simple temperature index

    # soil bucket per cell (spatially varying capacity) + groundwater store
    dist = np.hypot(*(cells - outlet_xy).T)
    # strong contrast: deep soils far from the outlet, shallow flashy soils near it
    xn = (cells[:, 0] - minx) / (maxx - minx)
    cap = rng.uniform(0.8, 1.2, G) * (220 - 170 * xn)
    soil = cap * 0.5
    gw = np.full(G, 5.0)
    quick = np.zeros((T, G), dtype=np.float32)
    base = np.zeros((T, G), dtype=np.float32)
    for t in range(T):
        p = rain_cells[t]
        infil_excess = np.maximum(p - 45.0, 0) * 0.5  # intense storms
        p_in = p - infil_excess
        frac = np.clip(soil / cap, 0, 1)
        sat_excess = p_in * frac**3
        soil = soil + p_in - sat_excess
        soil -= pet[t] * frac
        perc = 0.02 * np.maximum(soil - 0.3 * cap, 0)
        soil = np.clip(soil - perc, 0, None)
        over = np.maximum(soil - cap, 0)
        soil -= over
        gw += perc
        bf = 0.015 * gw
        gw -= bf
        quick[t] = infil_excess + sat_excess + over
        base[t] = bf

    # route: travel time by distance, then channel attenuation
    delay = np.round(dist / 35.0).astype(int)  # ~35 km/day
    q_mm = np.zeros(T + delay.max() + 1)
    for d in np.unique(delay):
        sel = delay == d
        q_mm[d : d + T] += quick[:, sel].sum(axis=1) * cell_area
    k = np.exp(-np.arange(8) / 1.5)
    k /= k.sum()
    q_mm = np.convolve(q_mm, k)[:T]
    q_mm += base.sum(axis=1) * cell_area
    flow = q_mm * 1000 / 86400  # m3/s (1 mm over 1 km2 = 1000 m3)
    flow *= rng.lognormal(0, 0.05, T)  # gauge measurement error

    # gauges: missing values
    ids = [f"G{i+1:02d}" for i in range(n_gauges)]
    rain_df = pd.DataFrame(rain_gauges.round(2), index=dates, columns=ids)
    if missing_fraction > 0:
        for c in ids:
            n_gaps = rng.poisson(missing_fraction * T / 15)
            for _ in range(n_gaps):
                s = rng.integers(0, T - 30)
                rain_df.iloc[s : s + rng.integers(3, 30), ids.index(c)] = np.nan
    rain_df.index.name = "date"

    glon, glat = to_lonlat(gauges[:, 0], gauges[:, 1])
    stations = pd.DataFrame({"lat": glat.round(5), "lon": glon.round(5)}, index=pd.Index(ids, name="id"))
    bx, by = poly.exterior.xy
    blon, blat = to_lonlat(np.array(bx), np.array(by))
    boundary = Polygon(np.column_stack([blon, blat]))
    olon, olat = to_lonlat(outlet_xy[0], outlet_xy[1])

    return BasinData(
        rainfall=rain_df,
        stations=stations,
        streamflow=pd.Series(flow.round(3), index=dates, name="flow"),
        boundary=boundary,
        extra=pd.DataFrame({"temperature_c": temp.round(1), "pet_mm": pet.round(2)}, index=dates),
        outlet=(float(olat), float(olon)),
        name=name,
    )


def write_basin(data: BasinData, folder, flow_unit: str = "m3/s") -> Path:
    """Write a BasinData to the standard folder layout (also a format example)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    data.rainfall.to_csv(folder / "rainfall.csv", date_format="%Y-%m-%d")
    data.stations.to_csv(folder / "stations.csv")
    data.streamflow.rename("flow").to_frame().to_csv(folder / "streamflow.csv", date_format="%Y-%m-%d")
    if data.extra is not None:
        data.extra.to_csv(folder / "extra.csv", date_format="%Y-%m-%d")
    if data.boundary is not None:
        gj = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"name": data.name}, "geometry": mapping(data.boundary)}]}
        (folder / "boundary.geojson").write_text(json.dumps(gj))
    meta = {"name": data.name, "flow_unit": flow_unit, "rainfall_unit": "mm/day"}
    if data.outlet is not None:
        meta.update(outlet_lat=data.outlet[0], outlet_lon=data.outlet[1])
    (folder / "basin.json").write_text(json.dumps(meta, indent=2))
    return folder
