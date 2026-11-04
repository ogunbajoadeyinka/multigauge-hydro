# Sub-basins and routing (phase 2)

Large basins respond to *where* rain falls. multigauge can split a basin into sub-basins, run the
pre-trained LSTM on each one with its own weather (and the rain gauges inside it), and route the
local runoff to the outlet. Everything is trained end to end on the outlet's discharge.

## Getting sub-basins

**Your own delineation (recommended).** Export sub-basins from USGS StreamStats, NHDPlus HR or a GIS
delineation and save them as `my_basin/subbasins/subbasins.geojson` with these properties:

| property | meaning |
|---|---|
| `id` | sub-basin id |
| `downstream` | id of the sub-basin it drains into, or `null` for the one at the outlet |
| `distance_km` | optional: flow distance from the sub-basin to the outlet (straight-line distance is used otherwise) |
| `area_km2` | optional: computed from the polygon otherwise |

Add one forcing file per sub-basin, `my_basin/subbasins/<id>.csv`, in the same format as `forcings.csv`.

**Automatic zones (quick approximation).** `multigauge zones my_basin/ -n 4` splits the basin into
distance bands from the outlet, links them in a chain and downloads Daymet weather for each band.

## Running

```bash
multigauge subbasins my_basin/ --routing gamma   # or --routing gnn
```

| routing | how it works |
|---|---|
| `gamma` | each sub-basin reaches the outlet through a gamma unit hydrograph with learned mean travel time and shape, initialised from distance (about 40 km/day) |
| `gnn` | a graph attention network passes information downstream along the sub-basin network (after Mosaffa et al., 2026) |

## Outputs

- test-period skill of the lumped model, the sub-basin model, and the sub-basin model with gauges;
- each sub-basin's share of area, total flow and flood flow (days above the 98th percentile);
- gauge value per sub-basin: the loss in outlet NSE when that sub-basin's gauges are removed;
- `priority_for_new_gauge`: the flood-flow share of sub-basins that have no gauge, a simple indicator
  of where a new gauge could add the most. Treat it as a screening tool, not a network design.
