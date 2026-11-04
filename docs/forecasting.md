# Forecasting (phase 3)

## Model

The forecast model follows the design of Google's operational flood model (Nearing et al., 2024):

1. a **hindcast LSTM** (the fine-tuned model) reads the past year of weather and yesterday's observed flow;
2. its memory is handed to a **forecast LSTM** that reads the weather forecast for lead days 1–7;
3. a mixture-density (CMAL) head gives a probability distribution for each lead day;
4. an **error correction** adds a lead-dependent fraction of the model's latest error (fitted on the
   training period), which mainly improves the first days.

The forecast LSTM starts as a copy of the trained model, so before forecast training it already
reproduces the simulation; training then learns forecast-specific behaviour.

## Training data and its limits

Archived weather forecasts are rarely available for a single basin, so the model is trained with
observed weather perturbed by errors that grow with lead time. **Hindcast skill (observed weather
used as the forecast) is therefore an upper bound**; skill with real forecasts will be lower,
especially for rainfall-driven peaks several days ahead.

## Setting up a basin

```bash
multigauge forecast-setup my_basin/ --site 07316000 --out forecasts/my_basin --model camels-us
```

This fine-tunes and trains the forecast model, saves hindcast skill by lead time, and fits a monthly
bias correction that maps Open-Meteo weather onto Daymet (needed because Daymet is published with a
long delay and cannot drive real-time runs).

## Daily forecasts

```bash
multigauge forecast-run forecasts/
```

downloads the last year of weather (Open-Meteo archive), the 7-day forecast, recent USGS flow, issues
a forecast and writes `latest.json` and `history.csv`. The GitHub Actions workflow
`.github/workflows/forecast.yml` does this every morning and commits the results; the dashboard shows
every basin under `forecasts/`. For heavy use of the USGS service, add a free API key as the
repository secret `USGS_API_KEY`.
