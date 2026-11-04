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
