# Changelog

## 0.3.0 — 2026-11-06

National results and new analyses.

- **National survey** of 88 USGS basins in all 18 U.S. hydrologic regions (`multigauge.national`,
  `notebooks/national_kaggle.ipynb`): adding NOAA rain gauges raised test NSE in 80 % of basins (median +0.021,
  Wilcoxon p = 3 × 10⁻⁹), +0.039 in rain-dominated basins and none in snow-dominated ones. National map in the
  README, report and dashboard.
- **Sub-basin study** on three large basins (`scripts/run_subbasins.py`, `notebooks/subbasins_kaggle.ipynb`):
  four routed sub-basins raised NSE by 0.12–0.21 with gridded weather alone; gauges then added nothing measurable.
- **Forecast tests** on the Iowa and Flint rivers (`scripts/run_forecast.py`).
- Gauge value now carries a seed-to-seed spread and a verdict (helps / hurts / negligible / within noise).
- Fill-in pass with a wider gauge search for sparsely gauged regions; download errors are retried.
- Periods partly outside a record are clipped instead of rejected.
- Pre-computed dashboard results load without PyTorch.
- `scripts/check_numbers.py` and `notebooks/check_numbers.ipynb` recompute every number in the manuscript.
- Affiliation: Independent Researcher.

## 0.2.1 — 2026-11-05

- Pre-trained CAMELS-US LSTM (531 basins) and δHBV hybrid shipped with the package.
- Automatic correction of gauge day conventions (morning-read cooperative gauges).
- Five-basin validation report and dashboard.
