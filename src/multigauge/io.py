"""Loading and validating user data.

Every loader accepts a file path or a file-like object (so the same code
serves the command line, scripts and web uploads). Problems are raised as
``DataError`` with a message written for the person who supplied the file.

Expected formats (see docs/data_format.md for full detail):

* rainfall   - CSV, one ``date`` column plus one column per gauge (mm/day)
* stations   - CSV with ``id``, ``lat``, ``lon`` columns (WGS84 degrees)
* streamflow - CSV with ``date`` and one flow column (any consistent unit)
* extra      - optional CSV, ``date`` plus basin-wide inputs (e.g. temperature)
* boundary   - optional GeoJSON, or a zipped shapefile, in WGS84 lon/lat
"""
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Optional, Union

import numpy as np
import pandas as pd
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

PathOrFile = Union[str, Path, IO]

DATE_NAMES = ("date", "time", "datetime", "day", "timestamp")
ID_NAMES = ("id", "station", "station_id", "stid", "gauge", "gauge_id", "name")
LAT_NAMES = ("lat", "latitude", "y")
LON_NAMES = ("lon", "long", "longitude", "lng", "x")
FLOW_NAMES = ("flow", "streamflow", "discharge", "q", "qobs", "value")


class DataError(ValueError):
    """Raised when input data cannot be used. The message explains the fix."""


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _find_column(columns, candidates, what, required=True):
    lookup = {str(c).strip().lower(): c for c in columns}
    for name in candidates:
        if name in lookup:
            return lookup[name]
    if required:
        raise DataError(
            f"Could not find a {what} column. Expected one of: "
            f"{', '.join(candidates)}. Found: {', '.join(map(str, columns))}."
        )
    return None


def _read_csv(src: PathOrFile, what: str) -> pd.DataFrame:
    try:
        if hasattr(src, "seek"):
            src.seek(0)
        df = pd.read_csv(src)
    except Exception as exc:  # pragma: no cover - pandas error text varies
        raise DataError(f"Could not read the {what} file as CSV: {exc}") from exc
    if df.empty:
        raise DataError(f"The {what} file is empty.")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _date_index(df: pd.DataFrame, what: str) -> pd.DataFrame:
    col = _find_column(df.columns, DATE_NAMES, "date", required=False) or df.columns[0]
    dates = pd.to_datetime(df[col], errors="coerce")
    bad = dates.isna().sum()
    if bad == len(df):
        raise DataError(f"The {what} file's '{col}' column could not be read as dates.")
    if bad:
        raise DataError(
            f"{bad} row(s) in the {what} file have unreadable dates in '{col}'. "
            "Use a format like 2015-01-31."
        )
    df = df.drop(columns=[col])
    df.index = dates.dt.normalize()
    df.index.name = "date"
    dupes = df.index.duplicated()
    if dupes.any():
        first = df.index[dupes][0].date()
        raise DataError(
            f"The {what} file has {dupes.sum()} duplicated date(s) (first: {first}). "
            "Each day must appear once."
        )
    return df.sort_index()


def _numeric(df: pd.DataFrame, what: str) -> pd.DataFrame:
    out = df.apply(pd.to_numeric, errors="coerce")
    for c in df.columns:
        newly_missing = out[c].isna() & df[c].notna() & (df[c].astype(str).str.strip() != "")
        if newly_missing.sum() > 0.05 * len(df):
            raise DataError(
                f"Column '{c}' in the {what} file has many non-numeric values "
                f"(e.g. '{df.loc[newly_missing, c].iloc[0]}'). Use blank cells for missing data."
            )
    return out


# --------------------------------------------------------------------------
# individual loaders
# --------------------------------------------------------------------------
def load_rainfall(src: PathOrFile) -> pd.DataFrame:
    """Daily rainfall: index = date, one column per gauge (mm/day)."""
    df = _date_index(_read_csv(src, "rainfall"), "rainfall")
    if df.shape[1] == 0:
        raise DataError("The rainfall file needs at least one gauge column besides the date.")
    df.columns = [str(c) for c in df.columns]
    df = _numeric(df, "rainfall")
    neg = (df < 0).sum().sum()
    if neg:
        df = df.mask(df < 0)  # common missing-value codes such as -999
    return df


def load_stations(src: PathOrFile) -> pd.DataFrame:
    """Gauge locations: index = gauge id (str), columns lat, lon."""
    df = _read_csv(src, "stations")
    id_col = _find_column(df.columns, ID_NAMES, "station id")
    lat_col = _find_column(df.columns, LAT_NAMES, "latitude")
    lon_col = _find_column(df.columns, LON_NAMES, "longitude")
    out = pd.DataFrame(
        {
            "lat": pd.to_numeric(df[lat_col], errors="coerce").values,
            "lon": pd.to_numeric(df[lon_col], errors="coerce").values,
        },
        index=df[id_col].astype(str).str.strip(),
    )
    out.index.name = "id"
    if out[["lat", "lon"]].isna().any().any():
        raise DataError("Every station needs a numeric latitude and longitude.")
    if out.index.duplicated().any():
        raise DataError(f"Duplicate station ids: {sorted(set(out.index[out.index.duplicated()]))}.")
    if (out["lat"].abs() > 90).any() or (out["lon"].abs() > 180).any():
        raise DataError(
            "Station coordinates must be latitude/longitude in decimal degrees (WGS84). "
            "Projected coordinates (metres/feet) are not supported."
        )
    return out


def load_streamflow(src: PathOrFile) -> pd.Series:
    """Daily observed streamflow at the outlet, any consistent unit."""
    df = _date_index(_read_csv(src, "streamflow"), "streamflow")
    if df.shape[1] == 0:
        raise DataError("The streamflow file needs a flow column besides the date.")
    col = _find_column(df.columns, FLOW_NAMES, "flow", required=False) or df.columns[0]
    s = _numeric(df[[col]], "streamflow")[col]
    s = s.mask(s < 0)
    s.name = "flow"
    return s


def load_extra(src: PathOrFile) -> pd.DataFrame:
    """Optional basin-wide daily inputs (temperature, PET, ...)."""
    df = _date_index(_read_csv(src, "extra inputs"), "extra inputs")
    return _numeric(df, "extra inputs")


def load_boundary(src: PathOrFile, filename: Optional[str] = None) -> BaseGeometry:
    """Watershed boundary from GeoJSON or a zipped shapefile (WGS84 lon/lat)."""
    name = (filename or getattr(src, "name", None) or str(src)).lower()
    raw = src.read() if hasattr(src, "read") else Path(src).read_bytes()
    if isinstance(raw, str):
        raw = raw.encode()

    if name.endswith(".zip") or raw[:2] == b"PK":
        geom = _boundary_from_zip(raw)
    else:
        try:
            gj = json.loads(raw.decode("utf-8-sig"))
        except Exception as exc:
            raise DataError(
                "The boundary must be a GeoJSON file or a .zip containing a shapefile."
            ) from exc
        if gj.get("type") == "FeatureCollection":
            geoms = [shape(f["geometry"]) for f in gj["features"] if f.get("geometry")]
        elif gj.get("type") == "Feature":
            geoms = [shape(gj["geometry"])]
        else:
            geoms = [shape(gj)]
        geom = unary_union(geoms)

    geom = geom.buffer(0) if not geom.is_valid else geom
    if geom.is_empty or geom.area == 0:
        raise DataError("The boundary file contains no polygon.")
    minx, miny, maxx, maxy = geom.bounds
    if max(abs(minx), abs(maxx)) > 180 or max(abs(miny), abs(maxy)) > 90:
        raise DataError(
            "The boundary appears to be in projected coordinates. Re-save it in "
            "WGS84 (EPSG:4326) longitude/latitude."
        )
    return geom


def _boundary_from_zip(raw: bytes) -> BaseGeometry:
    import shapefile  # pyshp

    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        names = zf.namelist()
        shp = [n for n in names if n.lower().endswith(".shp")]
        if not shp:
            raise DataError("The .zip file does not contain a .shp shapefile.")
        stem = shp[0][:-4]
        prj = next((n for n in names if n.lower() == (stem + ".prj").lower()), None)
        if prj:
            text = zf.read(prj).decode(errors="ignore").upper()
            if "PROJCS" in text and "GEOGCS" in text:
                raise DataError(
                    "The shapefile is in a projected coordinate system. Re-save it in "
                    "WGS84 (EPSG:4326) longitude/latitude."
                )

        def part(ext):
            m = next((n for n in names if n.lower() == (stem + ext).lower()), None)
            return io.BytesIO(zf.read(m)) if m else None

        reader = shapefile.Reader(shp=part(".shp"), shx=part(".shx"), dbf=part(".dbf"))
        geoms = [shape(s.__geo_interface__) for s in reader.shapes()]
    return unary_union(geoms)


# --------------------------------------------------------------------------
# the bundle
# --------------------------------------------------------------------------
@dataclass
class BasinData:
    """All inputs for one watershed, aligned to a common daily index."""

    rainfall: pd.DataFrame
    stations: pd.DataFrame
    streamflow: pd.Series
    boundary: Optional[BaseGeometry] = None
    extra: Optional[pd.DataFrame] = None
    outlet: Optional[tuple] = None  # (lat, lon), optional
    name: str = "basin"
    notes: list = field(default_factory=list)

    def __post_init__(self):
        self._validate_and_align()

    # ------------------------------------------------------------------
    def _validate_and_align(self):
        rain_ids = set(self.rainfall.columns)
        station_ids = set(self.stations.index)
        missing_loc = sorted(rain_ids - station_ids)
        if missing_loc:
            raise DataError(
                "These rainfall columns have no entry in the stations file: "
                + ", ".join(missing_loc)
            )
        unused = sorted(station_ids - rain_ids)
        if unused:
            self.notes.append(f"Stations without rainfall data were ignored: {', '.join(unused)}.")
        self.stations = self.stations.loc[list(self.rainfall.columns)]

        start = max(self.rainfall.index.min(), self.streamflow.index.min())
        end = min(self.rainfall.index.max(), self.streamflow.index.max())
        if self.extra is not None:
            start = max(start, self.extra.index.min())
            end = min(end, self.extra.index.max())
        if start >= end:
            raise DataError("The rainfall and streamflow files do not share any dates.")

        idx = pd.date_range(start, end, freq="D", name="date")
        self.rainfall = self.rainfall.reindex(idx)
        self.streamflow = self.streamflow.reindex(idx)
        if self.extra is not None:
            self.extra = self.extra.reindex(idx).interpolate(limit_direction="both")

        if len(idx) < 3 * 365:
            raise DataError(
                f"Only {len(idx)} overlapping days were found. At least 3 years of "
                "daily data are needed to train and test a model."
            )
        all_missing = [c for c in self.rainfall.columns if self.rainfall[c].isna().all()]
        if all_missing:
            raise DataError(f"These gauges have no rainfall values in the overlap: {all_missing}.")
        if self.streamflow.notna().sum() < 2 * 365:
            raise DataError("Fewer than 2 years of streamflow values are available.")

    # ------------------------------------------------------------------
    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.rainfall.index

    @property
    def station_ids(self) -> list:
        return list(self.rainfall.columns)

    def summary(self) -> pd.DataFrame:
        """Per-gauge record summary (coverage and mean annual rainfall)."""
        r = self.rainfall
        return pd.DataFrame(
            {
                "lat": self.stations["lat"],
                "lon": self.stations["lon"],
                "percent_missing": (r.isna().mean() * 100).round(1),
                "mean_annual_mm": (r.mean() * 365.25).round(0),
            }
        )

    @classmethod
    def from_files(
        cls,
        rainfall: PathOrFile,
        stations: PathOrFile,
        streamflow: PathOrFile,
        boundary: Optional[PathOrFile] = None,
        extra: Optional[PathOrFile] = None,
        outlet: Optional[tuple] = None,
        name: str = "basin",
        boundary_filename: Optional[str] = None,
    ) -> "BasinData":
        return cls(
            rainfall=load_rainfall(rainfall),
            stations=load_stations(stations),
            streamflow=load_streamflow(streamflow),
            boundary=load_boundary(boundary, boundary_filename) if boundary is not None else None,
            extra=load_extra(extra) if extra is not None else None,
            outlet=outlet,
            name=name,
        )

    @classmethod
    def from_folder(cls, folder: Union[str, Path], name: Optional[str] = None) -> "BasinData":
        """Load a folder containing rainfall.csv, stations.csv, streamflow.csv
        and optionally boundary.geojson / boundary.zip / extra.csv."""
        folder = Path(folder)

        def opt(*names):
            for n in names:
                if (folder / n).exists():
                    return folder / n
            return None

        for req in ("rainfall.csv", "stations.csv", "streamflow.csv"):
            if not (folder / req).exists():
                raise DataError(f"'{req}' was not found in {folder}.")
        outlet = None
        meta = folder / "basin.json"
        if meta.exists():
            m = json.loads(meta.read_text())
            if "outlet_lat" in m and "outlet_lon" in m:
                outlet = (float(m["outlet_lat"]), float(m["outlet_lon"]))
            name = name or m.get("name")
        return cls.from_files(
            folder / "rainfall.csv",
            folder / "stations.csv",
            folder / "streamflow.csv",
            boundary=opt("boundary.geojson", "boundary.json", "boundary.zip"),
            extra=opt("extra.csv"),
            outlet=outlet,
            name=name or folder.name,
        )
