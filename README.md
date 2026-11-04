# multigauge

**How much do multiple rain gauges improve machine-learning streamflow prediction, and which gauges matter?**

`multigauge` is an open-source Python package and web dashboard that answers this for any watershed.
Give it daily rainfall from several gauges, their locations, and outlet streamflow. It trains a
Long Short-Term Memory (LSTM) network for each of four ways of using the rainfall, compares them on
held-out data, and ranks the gauges by how much information they contribute.

| Rainfall input | What the model sees |
|---|---|
| **Single gauge** | One gauge, by default the one with the highest permutation importance |
| **Lumped average** | Arithmetic mean of all gauges |
| **Voronoi weighted** | Thiessen-polygon area-weighted mean, clipped to the basin |
| **Disaggregated** | Every gauge as a separate input, preserving where the rain fell |

Outputs: NSE, KGE, RMSE, percent bias and peak-flow error per strategy and period; observed vs.
predicted hydrographs; a map of gauge weights; permutation importance and station-ablation rankings;
and a shareable HTML report with every setting needed to reproduce the run.

## Who it is for

- **Researchers** testing how rainfall spatial variability affects ML hydrology models.
- **Agencies and network operators** deciding where gauges add the most forecasting value.
- **Engineers and water managers** wanting a quick, reproducible ML streamflow baseline for a basin.
- **Instructors** teaching data-driven hydrology.

## Install

```bash
pip install multigauge-hydro                 # package + command line
pip install "multigauge-hydro[app]"          # + web dashboard
pip install "multigauge-hydro[download]"     # + USGS / NOAA downloaders
```

Until the package is on PyPI, install from source:

```bash
git clone https://github.com/ogunbajoadeyinka/multigauge-hydro
cd multigauge-hydro
pip install -e ".[dev]"
```

A CPU is enough. On a 2-core machine, the `quick` preset compares all four inputs for a 10-gauge, 12-year basin in under a minute; the `standard` preset (365-day lookback) takes roughly 3-8 minutes per model, so use `--seeds` and `--ablation` with that in mind.

## Quick start

**Command line**

```bash
multigauge demo demo_basin/                       # synthetic example basin
multigauge run demo_basin/ -o results/ --preset quick --seeds 1
# open results/report.html
```

For your own basin, put `rainfall.csv`, `stations.csv` and `streamflow.csv` (plus an optional
`boundary.geojson`) in a folder. See **[docs/data_format.md](docs/data_format.md)**.

```bash
multigauge run my_basin/ -o results/ --seeds 3 --ablation \
    --train 2005-10-01:2012-09-30 --validation 2012-10-01:2014-09-30 --test 2014-10-01:2017-09-30
```

**Python**

```python
import multigauge as mg

data = mg.BasinData.from_folder("my_basin/")
cfg = mg.ExperimentConfig(n_seeds=3, run_ablation=True, training=mg.TrainConfig.preset("standard"))
result = mg.run_experiment(data, cfg)

print(result.summary.xs("test", level="period")[["NSE", "KGE"]])
print(result.importance.head())          # most informative gauges
result.save("results/")
mg.write_report(result, data, "results/report.html")
```

**Web dashboard**

```bash
streamlit run app/streamlit_app.py
```

Upload your files (or use the demo basin), pick settings, and press **Run comparison**.

## Method

- **Model.** Sequence-to-one LSTM (PyTorch): the previous *lookback* days of inputs predict the
  flow on the last day, the same set-up as NeuralHydrology's single-basin models. Presets:
  `quick` (90-day lookback, 32 hidden units), `standard` (365 days, 64 units), `thorough` (365 days, 128 units).
- **Training.** Adam, MSE loss on standardised flow (optionally log flow), gradient clipping, early
  stopping on validation NSE with best-weight restore. Normalisation uses training-period statistics only.
- **Fair comparison.** All strategies share the same periods, model settings, seeds, and gap filling.
  With `n_seeds > 1`, each strategy is trained several times; reported metrics use the mean prediction,
  and the spread of NSE across seeds is reported as `NSE_seed_std`.
- **Gauge selection.** The single-gauge model uses the gauge with the highest permutation importance
  on the *validation* period, so the test period is never used for selection.
- **Permutation importance** shuffles one gauge in time and measures the NSE drop of the trained
  disaggregated model. **Station ablation** retrains without each gauge.
- **Voronoi weights** are computed in a local kilometre projection and clipped to the basin boundary.

## Synthetic basins

`mg.generate_basin()` creates a basin with known, spatially variable storms (small convective cells and
broad frontal systems) and nonlinear, spatially varying runoff. It is used for the demo and the tests,
and lets you check the method where the truth is known. Results on real basins will differ.

## Tests

```bash
pytest
```

## Deploying the dashboard

- **Streamlit Community Cloud:** point it at `app/streamlit_app.py`; it installs `app/requirements.txt`.
- **Hugging Face Spaces:** create a Streamlit Space and copy the repository; `app/requirements.txt`
  installs CPU-only PyTorch to keep the image small.

Uploaded files are processed in memory for the session and are not stored.

## Citation

If you use multigauge, please cite it (see `CITATION.cff`). A DOI will be added via Zenodo.

## Licence

MIT. See `LICENSE`. Data you load remain subject to their providers' terms.
