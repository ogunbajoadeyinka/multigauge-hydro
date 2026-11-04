"""Read the CAMELS-US data set (Newman et al., 2015; Addor et al., 2017).

Download ``basin_timeseries_v1p2_metForcing_obsFlow.zip`` and ``camels_topo.txt``
(CAMELS v1.2, NCAR GDEX: https://gdex.ucar.edu/dataset/camels.html; also on
Zenodo, record 15529996) and unzip into one folder.
The expected layout is::

    camels_root/
        basin_mean_forcing/daymet/<huc02>/<gauge_id>_lump_cida_forcing_leap.txt
        usgs_streamflow/<huc02>/<gauge_id>_streamflow_qc.txt
        camels_topo.txt            (optional, for accurate areas)

A nested ``basin_dataset_public_v1p2`` folder is also found automatically.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd

from .basin import Basin
from .features import flow_to_mm_per_day

# Daymet column names in the CAMELS files -> multigauge names
_COLS = {
    "dayl(s)": "dayl",
    "prcp(mm/day)": "prcp",
    "srad(W/m2)": "srad",
    "swe(mm)": "swe",
    "tmax(C)": "tmax",
    "tmin(C)": "tmin",
    "vp(Pa)": "vp",
}


_INDEX_CACHE: dict = {}


def _index(path) -> dict:
    """Map basin id -> {"forcing": file, "flow": file} over the whole tree.

    The CAMELS archive unpacks into more than one folder (for example
    ``basin_dataset_public`` and ``basin_dataset_public_v1p2``), and not
    every folder holds every basin, so files are found wherever they are;
    copies under a ``v1p2`` folder win when a basin appears twice.
    """
    key = str(Path(path).resolve())
    if key in _INDEX_CACHE:
        return _INDEX_CACHE[key]
    idx: dict = {}

    def add(kind, f, bid):
        cur = idx.setdefault(bid, {}).get(kind)
        if cur is None or ("v1p2" in str(f) and "v1p2" not in str(cur)):
            idx[bid][kind] = f

    for f in Path(path).rglob("*_lump_cida_forcing_leap.txt"):
        if f.parent.parent.name == "daymet":
            add("forcing", f, f.name.split("_")[0])
    for f in Path(path).rglob("*_streamflow_qc.txt"):
        add("flow", f, f.name.split("_")[0])
    if not idx:
        raise FileNotFoundError(f"No CAMELS Daymet forcing files found under {path}.")
    _INDEX_CACHE[key] = idx
    return idx


def _file(path, basin_id: str, kind: str) -> Path:
    f = _index(path).get(basin_id, {}).get(kind)
    if f is None:
        what = "Daymet forcing" if kind == "forcing" else "streamflow"
        raise FileNotFoundError(f"No CAMELS {what} file for basin {basin_id} under {path}.")
    return f


def _attr_root(path: Path) -> Optional[Path]:
    for p in [path, *path.parents][:3]:
        if (p / "camels_topo.txt").exists():
            return p
    for p in Path(path).rglob("camels_topo.txt"):
        return p.parent
    return None


def list_basins(path) -> list:
    return sorted(b for b, v in _index(path).items() if "forcing" in v and "flow" in v)


def read_forcing(path, basin_id: str) -> tuple[pd.DataFrame, float, float]:
    """Return (forcings, latitude, header area in km2)."""
    f = _file(path, basin_id, "forcing")
    with open(f) as fh:
        lat = float(fh.readline())
        fh.readline()  # elevation
        area_m2 = float(fh.readline())
    df = pd.read_csv(f, sep=r"\s+", skiprows=3)
    dates = pd.to_datetime(dict(year=df["Year"], month=df["Mnth"], day=df["Day"]))
    df = df.rename(columns=_COLS)
    df.index = pd.DatetimeIndex(dates, name="date")
    return df[[c for c in _COLS.values() if c in df.columns]], lat, area_m2 / 1e6


def read_streamflow(path, basin_id: str) -> pd.Series:
    """Observed discharge in ft3/s, NaN where missing."""
    f = _file(path, basin_id, "flow")
    df = pd.read_csv(f, sep=r"\s+", header=None, names=["id", "y", "m", "d", "q", "flag"], dtype={"id": str})
    q = df["q"].astype(float).where(df["q"] >= 0)
    q.index = pd.DatetimeIndex(pd.to_datetime(dict(year=df["y"], month=df["m"], day=df["d"])), name="date")
    return q


def read_topo(path) -> Optional[pd.DataFrame]:
    ar = _attr_root(Path(path))
    if ar is None:
        return None
    df = pd.read_csv(ar / "camels_topo.txt", sep=";", dtype={"gauge_id": str})
    df["gauge_id"] = df["gauge_id"].str.zfill(8)
    return df.set_index("gauge_id")


def load_basin(path, basin_id: str, topo: Optional[pd.DataFrame] = None) -> Basin:
    forc, lat, area_hdr = read_forcing(path, basin_id)
    area, lon = area_hdr, float("nan")
    if topo is not None and basin_id in topo.index:
        row = topo.loc[basin_id]
        area = float(row.get("area_gages2", area_hdr))
        lon = float(row.get("gauge_lon", np.nan))
    q_cfs = read_streamflow(path, basin_id)
    q_mm = flow_to_mm_per_day(q_cfs, area, "ft3/s")
    return Basin(basin_id, forc, q_mm, area, lat, lon, meta={"source": "CAMELS-US"})


def load_camels(path, basin_ids: Optional[Iterable[str]] = None, progress=None) -> list:
    """Load many CAMELS basins. ``basin_ids`` defaults to every basin found.

    The 531-basin subset used by most deep-learning studies (basins under
    2,000 km2 without large area discrepancies) can be passed as a list.
    """
    ids = list(basin_ids) if basin_ids is not None else list_basins(path)
    topo = read_topo(path)
    out, missing = [], []
    for i, b in enumerate(ids):
        try:
            out.append(load_basin(path, b, topo))
        except FileNotFoundError:
            missing.append(b)  # reported below; one absent basin should not stop pre-training
        if progress:
            progress(i + 1, len(ids), b)
    if missing:
        print(f"\nWarning: {len(missing)} basin(s) not found and skipped: {', '.join(missing[:20])}")
    if not out:
        raise FileNotFoundError(f"None of the requested basins were found under {path}.")
    return out
