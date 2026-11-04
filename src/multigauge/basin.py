"""A single basin's model-ready record.

``Basin`` is the common currency of the deep-learning pipeline: CAMELS
basins used for pre-training and the user's own basin used for
fine-tuning are both converted to it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .features import DYNAMIC, STATIC, check_forcings, static_attributes


@dataclass
class Basin:
    """Daily forcings and specific discharge for one basin.

    Attributes
    ----------
    basin_id : identifier (e.g. USGS gauge id)
    forcings : daily Daymet-style forcings, columns ``features.DYNAMIC``
    q_mm : observed specific discharge (mm/day), NaN where missing
    area_km2, lat, lon : basin area and outlet (or centroid) location
    extra : optional extra daily inputs, such as local rain gauges, added
        during fine-tuning (columns are free-form)
    statics : climate attributes; computed from the forcings if omitted
    """

    basin_id: str
    forcings: pd.DataFrame
    q_mm: pd.Series
    area_km2: float
    lat: float
    lon: float = float("nan")
    extra: Optional[pd.DataFrame] = None
    statics: Optional[pd.Series] = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.forcings = check_forcings(self.forcings, f"forcings of basin {self.basin_id}")
        idx = self.forcings.index
        self.q_mm = self.q_mm.reindex(idx).astype(float)
        self.q_mm[self.q_mm < 0] = np.nan
        if self.extra is not None:
            self.extra = self.extra.reindex(idx).astype(float)
        if self.statics is None:
            self.statics = static_attributes(self.forcings, self.area_km2, self.lat)
        else:
            self.statics = self.statics.reindex(STATIC)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.forcings.index

    def subset(self, start=None, end=None) -> "Basin":
        sl = slice(start, end)
        return Basin(
            self.basin_id,
            self.forcings.loc[sl],
            self.q_mm.loc[sl],
            self.area_km2,
            self.lat,
            self.lon,
            None if self.extra is None else self.extra.loc[sl],
            self.statics,  # keep the long-term climate attributes
            dict(self.meta),
        )

    def with_extra(self, extra: Optional[pd.DataFrame]) -> "Basin":
        return Basin(self.basin_id, self.forcings, self.q_mm, self.area_km2, self.lat, self.lon,
                     extra, self.statics, dict(self.meta))


def dynamic_columns() -> list:
    return list(DYNAMIC)
