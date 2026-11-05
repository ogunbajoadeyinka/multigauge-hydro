# Input data format

multigauge needs three files. Two more are optional. Put them in one folder
with these names to use `multigauge study <folder>` or `BasinData.from_folder()`,
or upload them one by one in the web app.

| File | Required | Contents |
|---|---|---|
| `rainfall.csv` | yes | `date`, then one column per gauge, daily totals in **mm/day** |
| `stations.csv` | yes | `id`, `lat`, `lon` for every gauge (decimal degrees, WGS84) |
| `streamflow.csv` | yes | `date`, `flow`: daily mean flow at the basin outlet |
| `forcings.csv` | yes, for the pre-trained model | basin-average Daymet weather: `date, prcp, srad, tmax, tmin, vp` (mm/day, W/m², °C, °C, Pa). Create it with `multigauge daymet my_basin/` |
| `boundary.geojson` or `boundary.zip` | recommended | watershed polygon (GeoJSON, or a zipped shapefile), WGS84; gives the area and Voronoi weights |
| `basin.json` | no | `{"name": ..., "flow_unit": "m3/s", "area_km2": ..., "outlet_lat": ..., "outlet_lon": ...}` |
| `subbasins/` | no (phase 2) | `subbasins.geojson` (properties `id`, `downstream`, optional `distance_km`) and one `<id>.csv` of forcings per sub-basin |

Run `multigauge demo example/` to write a complete example folder.

The gridded forcings are what the pre-trained model needs; the rain gauges are what the study tests.

## Rules

- **Dates**: one row per day, any unambiguous format (`2015-01-31` is safest).
  Gaps in the date sequence are allowed; they are treated as missing.
- **Missing values**: leave the cell blank. Negative numbers (such as `-999`) are
  also treated as missing.
- **Day convention**: Daymet assigns rain to the day it fell. Many gauges, including NOAA
  cooperative observers who read the gauge at about 7 am, report it under the next day.
  multigauge compares each gauge with the Daymet precipitation and moves it one day earlier
  when that clearly matches better (the shift is listed as `day_shift` in the results and noted
  in the report). To set it yourself, add a `day_shift` column to `stations.csv`
  (`-1` = move one day earlier, `0` = leave as is; blank = detect).
- **Gauge ids** in the `rainfall.csv` header must match `id` in `stations.csv`.
  Column names are matched case-insensitively, and common alternatives are
  accepted (`Date`, `station_id`, `Latitude`, `Longitude`, `discharge`, ...).
- **Flow units**: m³/s (default) or ft³/s, set in `basin.json`. Flow is converted to
  specific discharge (mm/day) with the basin area for modelling, and converted back for results.
- **Basin area** comes from the boundary, or from `area_km2` in `basin.json`.
- **Coordinates** must be longitude/latitude in degrees. Re-project projected
  shapefiles to EPSG:4326 first (in QGIS: *Export → Save Features As → CRS 4326*).
- **Record length**: at least 3 years of overlapping rainfall and streamflow;
  8+ years is recommended. With the standard 365-day lookback, the first year
  of each split is used only as model input history.

## How missing rainfall is handled

For every strategy, a gauge's missing day is filled with the Voronoi-weighted
mean of the gauges that reported that day (weights re-normalised over those
gauges). Days with no reports at all are set to 0 mm. The percentage filled is
listed per gauge in `stations.csv` of the results.

## Without a boundary

If no boundary is given, Voronoi cells are clipped to the gauges' convex hull,
buffered by 10 % of its size. The run notes and the map say so. Supplying the
real boundary is recommended, especially when gauges lie near or outside the
basin edge.

## Getting U.S. data

```bash
pip install "multigauge-hydro[download]"
multigauge usgs 07316000 2010-01-01 2020-12-31 -o my_basin/streamflow.csv
multigauge ghcn USC00346139,USC00341750 2010-01-01 2020-12-31 -o my_basin/
multigauge daymet my_basin/
```

`multigauge daymet` averages Daymet over up to 16 points spread inside the boundary
(Daymet single-pixel service, https://daymet.ornl.gov). Daymet omits 31 December in
leap years; multigauge fills it from 30 December.

The first uses the USGS Water Data API (daily mean discharge, converted to m³/s).
The second uses NOAA's NCEI Access Data Service for GHCN-Daily precipitation and
writes both `rainfall.csv` and `stations.csv`. Watershed boundaries for U.S.
gauges are available from USGS StreamStats or the NHDPlus/WBD datasets.

Other networks (for example the Oklahoma Mesonet) have their own access and
redistribution terms. Export the records you are licensed to use as CSV.
