# Input data format

multigauge needs three files. Two more are optional. Put them in one folder
with these names to use `multigauge run <folder>` or `BasinData.from_folder()`,
or upload them one by one in the web app.

| File | Required | Contents |
|---|---|---|
| `rainfall.csv` | yes | `date`, then one column per gauge, daily totals in **mm/day** |
| `stations.csv` | yes | `id`, `lat`, `lon` for every gauge (decimal degrees, WGS84) |
| `streamflow.csv` | yes | `date`, `flow`: daily mean flow at the basin outlet |
| `boundary.geojson` or `boundary.zip` | no | watershed polygon (GeoJSON, or a zipped shapefile), WGS84 |
| `extra.csv` | no | `date` plus any basin-wide daily inputs (temperature, PET, ...) |
| `basin.json` | no | `{"name": ..., "flow_unit": "m3/s", "outlet_lat": ..., "outlet_lon": ...}` |

Run `multigauge demo example/` to write a complete example folder.

## Rules

- **Dates**: one row per day, any unambiguous format (`2015-01-31` is safest).
  Gaps in the date sequence are allowed; they are treated as missing.
- **Missing values**: leave the cell blank. Negative numbers (such as `-999`) are
  also treated as missing.
- **Gauge ids** in the `rainfall.csv` header must match `id` in `stations.csv`.
  Column names are matched case-insensitively, and common alternatives are
  accepted (`Date`, `station_id`, `Latitude`, `Longitude`, `discharge`, ...).
- **Flow units** can be anything consistent (m³/s, ft³/s, mm/day). NSE and KGE
  do not depend on the unit; RMSE is reported in the same unit.
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
```

The first uses the USGS Water Data API (daily mean discharge, converted to m³/s).
The second uses NOAA's NCEI Access Data Service for GHCN-Daily precipitation and
writes both `rainfall.csv` and `stations.csv`. Watershed boundaries for U.S.
gauges are available from USGS StreamStats or the NHDPlus/WBD datasets.

Other networks (for example the Oklahoma Mesonet) have their own access and
redistribution terms. Export the records you are licensed to use as CSV.
