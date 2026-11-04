"""Input variables, basin climate attributes and unit conversions.

A pre-trained model can only be reused on a new basin if that basin can
supply exactly the same inputs. multigauge therefore uses inputs that are
available for every basin in the conterminous U.S.:

* dynamic (daily) forcings from Daymet: precipitation, shortwave radiation,
  max/min temperature and vapour pressure, the same variables CAMELS-US
  provides for its 671 basins;
* static attributes that are *computed from those forcings* plus the basin
  area and latitude, so no external attribute database is needed.

Streamflow is modelled as specific discharge (mm/day) so that one model
can learn from basins of very different size.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DYNAMIC = ["prcp", "srad", "tmax", "tmin", "vp"]
DYNAMIC_UNITS = {
    "prcp": "mm/day",
    "srad": "W/m2 (daylight mean)",
    "tmax": "degC",
    "tmin": "degC",
    "vp": "Pa",
}
STATIC = [
    "log_area",
    "lat",
    "p_mean",
    "pet_mean",
    "aridity",
    "t_mean",
    "frac_snow",
    "high_prec_freq",
    "low_prec_freq",
]

CFS_TO_CMS = 0.0283168466


# ---------------------------------------------------------------- units
def flow_to_mm_per_day(flow: pd.Series, area_km2: float, unit: str = "m3/s") -> pd.Series:
    """Convert discharge to basin-average depth (mm/day)."""
    unit = unit.lower().replace("^", "").replace(" ", "")
    if unit in ("mm/day", "mm/d", "mmd"):
        return flow.astype(float)
    if unit in ("ft3/s", "cfs", "ft3s"):
        flow = flow * CFS_TO_CMS
    elif unit not in ("m3/s", "cms", "m3s"):
        raise ValueError(f"Unknown flow unit '{unit}'. Use m3/s, ft3/s or mm/day.")
    return flow * 86400.0 / (area_km2 * 1e6) * 1000.0


def mm_per_day_to_flow(q_mm: pd.Series, area_km2: float, unit: str = "m3/s") -> pd.Series:
    out = q_mm * area_km2 * 1e6 / 1000.0 / 86400.0
    return out / CFS_TO_CMS if unit.lower() in ("ft3/s", "cfs") else out


# ------------------------------------------------------------- climate
def extraterrestrial_radiation(lat_deg: float, doy: np.ndarray) -> np.ndarray:
    """Daily extraterrestrial radiation Ra (MJ m-2 day-1), FAO-56 eq. 21."""
    phi = np.radians(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    delta = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1, 1))
    return (24 * 60 / np.pi) * 0.0820 * dr * (
        ws * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(ws)
    )


def pet_hargreaves(tmax: pd.Series, tmin: pd.Series, lat_deg: float) -> pd.Series:
    """Hargreaves-Samani potential evapotranspiration (mm/day)."""
    ra = extraterrestrial_radiation(lat_deg, tmax.index.dayofyear.values)
    tmean = (tmax + tmin) / 2
    dt = np.clip((tmax - tmin).values, 0, None)
    pet = 0.0023 * 0.408 * ra * (tmean.values + 17.8) * np.sqrt(dt)
    return pd.Series(np.clip(pet, 0, None), index=tmax.index, name="pet")


def static_attributes(forcings: pd.DataFrame, area_km2: float, lat: float) -> pd.Series:
    """Climate attributes computed from daily forcings (CAMELS definitions)."""
    p = forcings["prcp"].astype(float)
    tmean = (forcings["tmax"] + forcings["tmin"]) / 2
    pet = pet_hargreaves(forcings["tmax"], forcings["tmin"], lat)
    p_mean = float(p.mean())
    pet_mean = float(pet.mean())
    snow = p[tmean < 0].sum()
    return pd.Series(
        {
            "log_area": float(np.log10(max(area_km2, 1e-3))),
            "lat": float(lat),
            "p_mean": p_mean,
            "pet_mean": pet_mean,
            "aridity": pet_mean / p_mean if p_mean > 0 else 10.0,
            "t_mean": float(tmean.mean()),
            "frac_snow": float(snow / p.sum()) if p.sum() > 0 else 0.0,
            "high_prec_freq": float((p >= 5 * p_mean).mean() * 365.25),
            "low_prec_freq": float((p < 1).mean() * 365.25),
        }
    )[STATIC]


def check_forcings(df: pd.DataFrame, what: str = "forcings") -> pd.DataFrame:
    missing = [c for c in DYNAMIC if c not in df.columns]
    if missing:
        raise ValueError(
            f"The {what} table is missing columns {missing}. Expected Daymet variables: "
            + ", ".join(f"{k} ({v})" for k, v in DYNAMIC_UNITS.items())
        )
    out = df[DYNAMIC].astype(float)
    return out.interpolate(limit=5, limit_direction="both")
