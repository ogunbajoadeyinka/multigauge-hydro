"""1-7 day forecast skill for basins already in validation/.

    python scripts/run_forecast.py validation 05453100 02347500

Fine-tunes the pre-trained LSTM on the basin, trains the hindcast/forecast
model with per-lead error correction, and scores it on every test-period
issue day against persistence. Writes validation/<site>/results/forecast_skill.csv.
Observed weather with growing noise stands in for weather forecasts.
"""
import sys
import time
from pathlib import Path

import numpy as np

import multigauge as mg
from multigauge.forecast import ForecastConfig, ForecastModel
from multigauge.nn import StreamflowModel
from multigauge.study import StudyConfig, make_periods

PERIODS = dict(train_period=("2001-10-01", "2013-09-30"), validation_period=("2013-10-01", "2017-09-30"),
               test_period=("2017-10-01", "2023-09-30"))

if __name__ == "__main__":
    root, sites = Path(sys.argv[1]), sys.argv[2:]
    model = StreamflowModel.load(Path(mg.__file__).parent / "pretrained" / "camels-us")
    for site in sites:
        t = time.time()
        d = mg.BasinData.from_folder(root / site / "data")
        periods = make_periods(d.dates, StudyConfig(**PERIODS), model.cfg.lookback)
        b = d.to_basin()
        m = {k: np.asarray((d.dates >= a) & (d.dates <= e)) for k, (a, e) in periods.items()}
        ft = mg.finetune(model, b, m["train"], m["validation"], epochs=20, seed=0)
        fm = ForecastModel(ft, ForecastConfig(epochs=15)).fit(b, m["train"], m["validation"])
        iss = np.where(m["test"])[0]
        iss = iss[iss < len(b.dates) - 8]
        skill = fm.skill_by_lead(b, iss)
        skill.to_csv(root / site / "results" / "forecast_skill.csv", index=False)
        print(f"[{site}] {d.name}: {(time.time() - t) / 60:.0f} min", flush=True)
        print(skill.round(3).to_string(index=False), flush=True)
