"""Assemble a complete basin folder for any USGS streamgage from public data.

``prepare_usgs_basin("07326500", "2001-10-01", "2023-09-30", "basins/washita")``

1. basin boundary from the USGS Network-Linked Data Index (NLDI);
2. daily discharge from the USGS Water Data API;
3. basin-average Daymet weather (``forcings.csv``);
4. NOAA GHCN-Daily precipitation gauges inside the boundary (``rainfall.csv``,
   ``stations.csv``), chosen for record coverage;
5. ``basin.json`` with name, area, outlet and units.

The result can be passed straight to ``run_study``. Needs network access to
api.water.usgs.gov, daymet.ornl.gov and www.ncei.noaa.gov.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from shapely.geometry import Point, mapping, shape
from shapely.ops import unary_union

NLDI_URLS = [
    "https://api.water.usgs.gov/nldi/linked-data/nwissite/USGS-{site}/basin",
    "https://labs.waterdata.usgs.gov/api/nldi/linked-data/nwissite/USGS-{site}/basin",
]
NLDI_SITE = "https://api.water.usgs.gov/nldi/linked-data/nwissite/USGS-{site}"
GHCN_INVENTORY = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-inventory.txt"


def nldi_basin(site: str, timeout: int = 120):
    """Upstream basin polygon (lon/lat) and the gauge location (lat, lon)."""
    from .download import _session

    s = _session()
    last = None
    for url in NLDI_URLS:
        try:
            r = s.get(url.format(site=site), params={"f": "json", "simplified": "false"}, timeout=timeout)
            r.raise_for_status()
            js = r.json()
            feats = js.get("features", [js])
            geom = unary_union([shape(f["geometry"]) for f in feats])
            break
        except Exception as exc:  # try the next mirror
            last = exc
    else:
        raise RuntimeError(f"Could not get the basin boundary for USGS {site}: {last}")
    outlet = None
    try:
        r = s.get(NLDI_SITE.format(site=site), params={"f": "json"}, timeout=timeout)
        r.raise_for_status()
        f = r.json()["features"][0]
        lon, lat = f["geometry"]["coordinates"][:2]
        outlet = (float(lat), float(lon))
    except Exception:
        pass
    return geom, outlet


def read_ghcn_inventory(text: str) -> pd.DataFrame:
    """Parse ghcnd-inventory.txt (fixed width) into a DataFrame."""
    df = pd.read_fwf(io.StringIO(text), widths=[11, 9, 10, 5, 5, 5], header=None,
                     names=["id", "lat", "lon", "element", "first", "last"])
    return df


def select_gauges(inventory: pd.DataFrame, geom, start: str, end: str, buffer_km: float = 5.0,
                  max_gauges: int = 15, min_years_fraction: float = 0.8) -> pd.DataFrame:
    """GHCN precipitation stations inside the (buffered) basin whose record
    covers most of [start, end]; the longest records first."""
    from .spatial import LocalProjection

    p = inventory[inventory["element"] == "PRCP"].copy()
    minx, miny, maxx, maxy = geom.bounds
    pad = buffer_km / 100.0 + 0.05
    p = p[(p["lon"] >= minx - pad) & (p["lon"] <= maxx + pad) & (p["lat"] >= miny - pad) & (p["lat"] <= maxy + pad)]
    c = geom.centroid
    proj = LocalProjection(c.x, c.y)
    region = proj.geom_to_km(geom).buffer(buffer_km)
    x, y = proj.forward(p["lon"].values, p["lat"].values)
    p = p[[region.contains(Point(a, b)) for a, b in zip(x, y)]]
    y0, y1 = pd.Timestamp(start).year, pd.Timestamp(end).year
    overlap = (np.minimum(p["last"], y1) - np.maximum(p["first"], y0) + 1).clip(lower=0)
    p = p.assign(coverage=overlap / (y1 - y0 + 1))
    p = p[p["coverage"] >= min_years_fraction].sort_values(["coverage", "first"], ascending=[False, True])
    return p.head(max_gauges).reset_index(drop=True)


def prepare_usgs_basin(site: str, start: str, end: str, folder, name: Optional[str] = None,
                       max_gauges: int = 15, daymet_points: int = 16, min_coverage_days: float = 0.7,
                       progress=print):
    """Download everything for one gauge and write a basin folder; returns BasinData."""
    from .download import _session, daymet_basin, ghcn_daily_precip, usgs_daily_flow
    from .io import BasinData, boundary_area_km2

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    progress(f"[{site}] basin boundary")
    geom, outlet = nldi_basin(site)
    area = boundary_area_km2(geom)
    (folder / "boundary.geojson").write_text(json.dumps(
        {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"site": site}, "geometry": mapping(geom)}]}))

    progress(f"[{site}] streamflow ({area:,.0f} km2)")
    q = usgs_daily_flow(site, start, end, units="m3/s")
    q.to_frame("flow").to_csv(folder / "streamflow.csv", date_format="%Y-%m-%d")

    progress(f"[{site}] Daymet weather")
    f = daymet_basin(geom, start, end, n_points=daymet_points)
    f.to_csv(folder / "forcings.csv", date_format="%Y-%m-%d")

    progress(f"[{site}] rain gauges")
    inv_file = Path.home() / ".cache" / "multigauge" / "ghcnd-inventory.txt"
    if not inv_file.exists():
        inv_file.parent.mkdir(parents=True, exist_ok=True)
        r = _session().get(GHCN_INVENTORY, timeout=300)
        r.raise_for_status()
        inv_file.write_text(r.text)
    cand = select_gauges(read_ghcn_inventory(inv_file.read_text()), geom, start, end, max_gauges=max_gauges * 2)
    if cand.empty:
        raise RuntimeError(f"No GHCN precipitation gauges with enough record in or near basin {site}.")
    rain, st = ghcn_daily_precip(cand["id"].tolist(), start, end)
    rain = rain.loc[start:end]
    good = rain.notna().mean() >= min_coverage_days
    rain = rain.loc[:, good].iloc[:, :max_gauges]
    st = st.loc[rain.columns]
    st.index.name = "id"
    if rain.shape[1] < 2:
        raise RuntimeError(f"Fewer than two usable rain gauges for basin {site}.")
    rain.to_csv(folder / "rainfall.csv", date_format="%Y-%m-%d")
    st.to_csv(folder / "stations.csv")

    meta = {"name": name or f"USGS {site}", "usgs_site": site, "flow_unit": "m3/s", "area_km2": round(area, 1)}
    if outlet:
        meta.update(outlet_lat=outlet[0], outlet_lon=outlet[1])
    (folder / "basin.json").write_text(json.dumps(meta, indent=2))
    progress(f"[{site}] done: {rain.shape[1]} gauges, {len(f)} days")
    return BasinData.from_folder(folder)
