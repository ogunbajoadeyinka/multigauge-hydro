"""Optional helpers to fetch free U.S. data in multigauge's input format.

* USGS daily mean discharge from the USGS Water Data OGC API
  (https://api.waterdata.usgs.gov), which replaces the legacy
  waterservices.usgs.gov endpoints.
* NOAA GHCN-Daily precipitation and station locations from the NCEI
  Access Data Service (https://www.ncei.noaa.gov/access/services/data/v1).

Both services are public. USGS allows about 50 requests per hour without
a key; set the environment variable ``USGS_API_KEY`` (free at
https://api.waterdata.usgs.gov/signup) for heavier use. Requires the
``requests`` package (``pip install multigauge-hydro[download]``).

Oklahoma Mesonet and other networks have their own access terms, so they
are not downloaded here: export those records as CSV and load them with
``multigauge.io``.
"""
from __future__ import annotations

import os
import io
from typing import Iterable, Optional

import pandas as pd

USGS_DAILY = "https://api.waterdata.usgs.gov/ogcapi/v0/collections/daily/items"
NCEI_DATA = "https://www.ncei.noaa.gov/access/services/data/v1"
USER_AGENT = "multigauge-hydro (https://github.com/ogunbajoadeyinka/multigauge-hydro)"
CFS_TO_CMS = 0.0283168466


def _session():
    try:
        import requests
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Downloading needs 'requests': pip install multigauge-hydro[download]") from exc
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    key = os.environ.get("USGS_API_KEY")
    if key:
        s.headers["X-Api-Key"] = key
    return s


def usgs_daily_flow(
    site: str, start: str, end: str, units: str = "m3/s", timeout: int = 60
) -> pd.Series:
    """Daily mean discharge (parameter 00060, statistic 00003) for a USGS site.

    ``site`` may be given as '07316000' or 'USGS-07316000'.
    Returns a Series indexed by date, in m3/s (default) or ft3/s.
    """
    site_id = site if site.upper().startswith("USGS-") else f"USGS-{site}"
    s = _session()
    params = {
        "f": "json",
        "monitoring_location_id": site_id,
        "parameter_code": "00060",
        "statistic_id": "00003",
        "time": f"{pd.Timestamp(start).date()}/{pd.Timestamp(end).date()}",
        "limit": 10000,
    }
    url, rows = USGS_DAILY, []
    while url:
        r = s.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        js = r.json()
        for f in js.get("features", []):
            p = f["properties"]
            if p.get("statistic_id", "00003") == "00003" and p.get("value") not in (None, ""):
                rows.append((p["time"], float(p["value"])))
        url = next((l["href"] for l in js.get("links", []) if l.get("rel") == "next"), None)
        params = None  # the next link already carries the query
    if not rows:
        raise ValueError(f"No daily discharge returned for {site_id} between {start} and {end}.")
    ser = pd.Series(dict(rows), name="flow")
    ser.index = pd.to_datetime(ser.index)
    ser = ser.sort_index()
    ser = ser[~ser.index.duplicated()]
    if units == "m3/s":
        ser = ser * CFS_TO_CMS
    ser.index.name = "date"
    return ser


def ghcn_daily_precip(
    stations: Iterable[str], start: str, end: str, timeout: int = 120
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """GHCN-Daily precipitation (mm/day) and locations for station ids such as
    'USC00346139'. Returns (rainfall, stations) in multigauge's format."""
    stations = list(stations)
    s = _session()
    frames, locs = [], {}
    for i in range(0, len(stations), 25):
        chunk = stations[i : i + 25]
        r = s.get(
            NCEI_DATA,
            params={
                "dataset": "daily-summaries",
                "stations": ",".join(chunk),
                "dataTypes": "PRCP",
                "startDate": str(pd.Timestamp(start).date()),
                "endDate": str(pd.Timestamp(end).date()),
                "includeStationLocation": "1",
                "units": "metric",
                "format": "json",
            },
            timeout=timeout,
        )
        r.raise_for_status()
        recs = r.json()
        if recs:
            df = pd.DataFrame(recs)
            frames.append(df)
            for _, row in df.drop_duplicates("STATION").iterrows():
                if "LATITUDE" in row and "LONGITUDE" in row:
                    locs[row["STATION"]] = (float(row["LATITUDE"]), float(row["LONGITUDE"]))
    if not frames:
        raise ValueError("NOAA returned no precipitation for these stations and dates.")
    df = pd.concat(frames)
    df["DATE"] = pd.to_datetime(df["DATE"])
    df["PRCP"] = pd.to_numeric(df["PRCP"], errors="coerce")
    rain = df.pivot_table(index="DATE", columns="STATION", values="PRCP", aggfunc="first")
    rain = rain.reindex(pd.date_range(rain.index.min(), rain.index.max(), freq="D"))
    rain.index.name = "date"
    rain.columns.name = None
    missing = [st for st in stations if st not in rain.columns]
    if missing:
        rain = rain.reindex(columns=[c for c in stations if c in rain.columns])
    st = pd.DataFrame(
        {"lat": [locs.get(c, (None, None))[0] for c in rain.columns],
         "lon": [locs.get(c, (None, None))[1] for c in rain.columns]},
        index=pd.Index(rain.columns, name="id"),
    )
    if missing:
        st.attrs["missing"] = missing
    return rain, st


# =========================================================================
# Daymet (gridded daily weather, North America, 1 km)
# =========================================================================
DAYMET_URL = "https://daymet.ornl.gov/single-pixel/api/data"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def sample_points(geom, n: int = 16, seed: int = 0) -> list:
    """Up to ``n`` (lat, lon) points spread evenly inside a lon/lat polygon."""
    import numpy as np
    from shapely.geometry import Point

    minx, miny, maxx, maxy = geom.bounds
    k = int(np.ceil(np.sqrt(n * 2)))
    pts = []
    for fx in np.linspace(0.05, 0.95, k):
        for fy in np.linspace(0.05, 0.95, k):
            p = Point(minx + fx * (maxx - minx), miny + fy * (maxy - miny))
            if geom.contains(p):
                pts.append((p.y, p.x))
    if not pts:
        c = geom.representative_point()
        pts = [(c.y, c.x)]
    if len(pts) > n:
        idx = np.linspace(0, len(pts) - 1, n).round().astype(int)
        pts = [pts[i] for i in idx]
    return pts


def _parse_daymet(text: str) -> pd.DataFrame:
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.lower().startswith("year,yday"))
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])))
    df.columns = [c.split(" ")[0].strip() for c in df.columns]
    dates = pd.to_datetime(df["year"].astype(str), format="%Y") + pd.to_timedelta(df["yday"] - 1, unit="D")
    df.index = pd.DatetimeIndex(dates, name="date")
    out = df[["prcp", "srad", "tmax", "tmin", "vp"]].astype(float)
    # Daymet drops 31 December in leap years: fill it from 30 December
    end = out.index.max()
    if end.month == 12 and end.day == 30 and end.is_leap_year:
        end = end + pd.Timedelta(days=1)
    full = pd.date_range(out.index.min(), end, freq="D", name="date")
    return out.reindex(full).ffill(limit=1)


def daymet_point(lat: float, lon: float, start: str, end: str, timeout: int = 120) -> pd.DataFrame:
    s = _session()
    r = s.get(DAYMET_URL, params={
        "lat": f"{lat:.5f}", "lon": f"{lon:.5f}", "vars": "prcp,srad,tmax,tmin,vp",
        "start": str(pd.Timestamp(start).date()), "end": str(pd.Timestamp(end).date()),
    }, timeout=timeout)
    r.raise_for_status()
    return _parse_daymet(r.text)


def daymet_basin(geom, start: str, end: str, n_points: int = 16, progress=None) -> pd.DataFrame:
    """Basin-average Daymet forcings (mean over points sampled inside ``geom``)."""
    pts = sample_points(geom, n_points)
    frames = []
    for i, (la, lo) in enumerate(pts):
        frames.append(daymet_point(la, lo, start, end))
        if progress:
            progress(i + 1, len(pts))
    return sum(frames) / len(frames)


# =========================================================================
# Open-Meteo (weather forecasts, no key needed)
# =========================================================================
def daylight_seconds(lat: float, dates: pd.DatetimeIndex):
    import numpy as np

    doy = dates.dayofyear.values
    delta = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(np.radians(lat)) * np.tan(delta), -1, 1))
    return 24 * 3600 * ws / np.pi


def openmeteo_forecast(geom_or_point, days: int = 7, past_days: int = 0, n_points: int = 6, timeout: int = 60) -> pd.DataFrame:
    """Daily weather forecast averaged over a basin, in multigauge's units
    (prcp mm/day, srad W/m2 daylight mean, tmax/tmin degC, vp estimated).

    ``past_days`` (up to 92) also returns recent analysed days, useful to
    bridge the gap between the latest Daymet day and today.
    """
    import numpy as np

    from .forecast import vp_from_tmin

    if isinstance(geom_or_point, tuple):
        pts = [geom_or_point]
    else:
        pts = sample_points(geom_or_point, n_points)
    s = _session()
    r = s.get(OPEN_METEO_URL, params={
        "latitude": ",".join(f"{p[0]:.4f}" for p in pts),
        "longitude": ",".join(f"{p[1]:.4f}" for p in pts),
        "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min,shortwave_radiation_sum",
        "timezone": "UTC", "forecast_days": days, "past_days": past_days,
    }, timeout=timeout)
    r.raise_for_status()
    js = r.json()
    js = js if isinstance(js, list) else [js]
    frames = []
    for item in js:
        d = item["daily"]
        frames.append(pd.DataFrame({
            "prcp": d["precipitation_sum"], "tmax": d["temperature_2m_max"],
            "tmin": d["temperature_2m_min"], "rad_mj": d["shortwave_radiation_sum"],
        }, index=pd.DatetimeIndex(pd.to_datetime(d["time"]), name="date")).astype(float))
    df = sum(frames) / len(frames)
    lat = float(np.mean([p[0] for p in pts]))
    df["srad"] = df.pop("rad_mj") * 1e6 / daylight_seconds(lat, df.index)
    df["vp"] = vp_from_tmin(df["tmin"])
    return df[["prcp", "srad", "tmax", "tmin", "vp"]]


def usgs_recent_flow(site: str, days: int = 400, units: str = "m3/s") -> pd.Series:
    """The most recent ``days`` of daily mean discharge (provisional data)."""
    end = pd.Timestamp.utcnow().normalize().tz_localize(None)
    return usgs_daily_flow(site, str((end - pd.Timedelta(days=days)).date()), str(end.date()), units=units)


OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def openmeteo_archive(geom_or_point, start: str, end: str, n_points: int = 6, timeout: int = 120) -> pd.DataFrame:
    """Historical daily weather (ERA5-based, about 5 days behind real time),
    in the same units as ``openmeteo_forecast``."""
    import numpy as np

    from .forecast import vp_from_tmin

    pts = [geom_or_point] if isinstance(geom_or_point, tuple) else sample_points(geom_or_point, n_points)
    s = _session()
    r = s.get(OPEN_METEO_ARCHIVE_URL, params={
        "latitude": ",".join(f"{p[0]:.4f}" for p in pts),
        "longitude": ",".join(f"{p[1]:.4f}" for p in pts),
        "start_date": str(pd.Timestamp(start).date()), "end_date": str(pd.Timestamp(end).date()),
        "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min,shortwave_radiation_sum",
        "timezone": "UTC",
    }, timeout=timeout)
    r.raise_for_status()
    js = r.json()
    js = js if isinstance(js, list) else [js]
    frames = [pd.DataFrame({
        "prcp": i["daily"]["precipitation_sum"], "tmax": i["daily"]["temperature_2m_max"],
        "tmin": i["daily"]["temperature_2m_min"], "rad_mj": i["daily"]["shortwave_radiation_sum"],
    }, index=pd.DatetimeIndex(pd.to_datetime(i["daily"]["time"]), name="date")).astype(float) for i in js]
    df = sum(frames) / len(frames)
    lat = float(np.mean([p[0] for p in pts]))
    df["srad"] = df.pop("rad_mj") * 1e6 / daylight_seconds(lat, df.index)
    df["vp"] = vp_from_tmin(df["tmin"])
    return df[["prcp", "srad", "tmax", "tmin", "vp"]]
