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


def _root(path: Union[str, Path]) -> Path:
    path = Path(path)
    if (path / "basin_mean_forcing").exists():
        return path
    for p in path.rglob("basin_mean_forcing"):
        return p.parent
    raise FileNotFoundError(f"No 'basin_mean_forcing' folder found under {path}.")


def _attr_root(path: Path) -> Optional[Path]:
    for p in [path, *path.parents][:3]:
        if (p / "camels_topo.txt").exists():
            return p
    for p in Path(path).rglob("camels_topo.txt"):
        return p.parent
    return None


def list_basins(path) -> list:
    root = _root(path)
    files = sorted((root / "basin_mean_forcing" / "daymet").glob("*/*_lump_cida_forcing_leap.txt"))
    return [f.name.split("_")[0] for f in files]


def read_forcing(path, basin_id: str) -> tuple[pd.DataFrame, float, float]:
    """Return (forcings, latitude, header area in km2)."""
    root = _root(path)
    f = next((root / "basin_mean_forcing" / "daymet").glob(f"*/{basin_id}_lump_cida_forcing_leap.txt"))
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
    root = _root(path)
    f = next((root / "usgs_streamflow").glob(f"*/{basin_id}_streamflow_qc.txt"))
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
    out = []
    for i, b in enumerate(ids):
        out.append(load_basin(path, b, topo))
        if progress:
            progress(i + 1, len(ids), b)
    return out
