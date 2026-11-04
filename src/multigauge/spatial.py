"""Spatial weighting of rain gauges (Thiessen / Voronoi polygons).

Coordinates are projected to a local equirectangular plane in kilometres
before any geometry is computed, so polygon areas are not distorted by
working in raw degrees. Results are returned in both km (areas) and
lon/lat (for mapping).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.geometry.base import BaseGeometry

KM_PER_DEG_LAT = 110.574
KM_PER_DEG_LON_EQ = 111.320


@dataclass
class LocalProjection:
    lon0: float
    lat0: float

    @property
    def kx(self) -> float:
        return KM_PER_DEG_LON_EQ * np.cos(np.radians(self.lat0))

    def forward(self, lon, lat):
        return (np.asarray(lon) - self.lon0) * self.kx, (np.asarray(lat) - self.lat0) * KM_PER_DEG_LAT

    def inverse(self, x, y):
        return np.asarray(x) / self.kx + self.lon0, np.asarray(y) / KM_PER_DEG_LAT + self.lat0

    def geom_to_km(self, geom: BaseGeometry) -> BaseGeometry:
        return shapely.transform(geom, lambda c: np.column_stack(self.forward(c[:, 0], c[:, 1])))

    def geom_to_lonlat(self, geom: BaseGeometry) -> BaseGeometry:
        return shapely.transform(geom, lambda c: np.column_stack(self.inverse(c[:, 0], c[:, 1])))


@dataclass
class VoronoiResult:
    weights: pd.Series            # station id -> area fraction (sums to 1)
    areas_km2: pd.Series          # station id -> clipped cell area
    cells_lonlat: dict            # station id -> shapely geometry (lon/lat)
    region_lonlat: BaseGeometry   # basin boundary or the fallback region used
    used_boundary: bool


def _fallback_region(points_km: np.ndarray) -> Polygon:
    """Region used when no boundary is supplied: the gauges' convex hull,
    buffered by 10 % of its characteristic size (minimum 5 km)."""
    mp = MultiPoint([tuple(p) for p in points_km])
    hull = mp.convex_hull
    size = np.sqrt(hull.area) if hull.area > 0 else hull.length
    return hull.buffer(max(5.0, 0.1 * size))


def voronoi_weights(
    stations: pd.DataFrame, boundary: Optional[BaseGeometry] = None
) -> VoronoiResult:
    """Thiessen-polygon weights for each gauge.

    Parameters
    ----------
    stations : DataFrame indexed by id with ``lat`` and ``lon`` columns.
    boundary : watershed polygon in lon/lat. If omitted, a buffered convex
        hull of the gauges is used and the result is flagged.
    """
    ids = list(stations.index)
    proj = LocalProjection(float(stations["lon"].mean()), float(stations["lat"].mean()))
    x, y = proj.forward(stations["lon"].values, stations["lat"].values)
    pts = np.column_stack([x, y])

    if boundary is not None:
        region = proj.geom_to_km(boundary)
        used_boundary = True
    else:
        region = _fallback_region(pts)
        used_boundary = False

    if len(ids) == 1:
        cells = {ids[0]: region}
    else:
        if len(np.unique(pts.round(6), axis=0)) < len(pts):
            raise ValueError("Two or more gauges share the same coordinates.")
        minx, miny, maxx, maxy = region.union(MultiPoint([tuple(p) for p in pts])).bounds
        pad = 10 * max(maxx - minx, maxy - miny, 1.0)
        envelope = shapely.box(minx - pad, miny - pad, maxx + pad, maxy + pad)
        vor = shapely.voronoi_polygons(MultiPoint([tuple(p) for p in pts]), extend_to=envelope)
        polys = list(vor.geoms)
        cells = {}
        for sid, p in zip(ids, pts):
            pt = Point(p)
            cell = next((g for g in polys if g.covers(pt)), None)
            if cell is None:  # numerical edge case: take nearest cell
                cell = min(polys, key=lambda g: g.distance(pt))
            cells[sid] = cell.intersection(region)

    areas = pd.Series({sid: cells[sid].area for sid in ids}, dtype=float)
    total = areas.sum()
    if total <= 0:
        raise ValueError("Voronoi cells do not overlap the basin boundary.")
    weights = areas / total
    return VoronoiResult(
        weights=weights.rename("weight"),
        areas_km2=areas.rename("area_km2"),
        cells_lonlat={sid: proj.geom_to_lonlat(g) for sid, g in cells.items()},
        region_lonlat=proj.geom_to_lonlat(region),
        used_boundary=used_boundary,
    )


def distance_km(stations: pd.DataFrame, lat: float, lon: float) -> pd.Series:
    """Great-circle distance (km) from each gauge to a point (e.g. the outlet)."""
    phi1, phi2 = np.radians(stations["lat"].values), np.radians(lat)
    dphi = phi2 - phi1
    dlmb = np.radians(lon - stations["lon"].values)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2) ** 2
    return pd.Series(6371.0 * 2 * np.arcsin(np.sqrt(a)), index=stations.index, name="distance_km")
