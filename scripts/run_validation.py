"""Re-run the five-basin validation from basin folders already on disk.

    python scripts/run_validation.py validation [site ...]

Each validation/<site>/data folder is loaded (gauges are aligned to the
Daymet day automatically), the full study is run with 3 seeds, and results
go to validation/<site>/results. The Washita (07326500) also gets the
sub-basin routing and forecast runs. Ends by writing validation/summary.csv.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import multigauge as mg
from multigauge.forecast import ForecastConfig, ForecastModel
from multigauge.hybrid import HybridModel
from multigauge.nn import StreamflowModel
from multigauge.report import summary_table
from multigauge.routing import SubbasinSet, run_routed_study
from multigauge.study import StudyConfig, load_saved_study, make_periods

PKG = Path(mg.__file__).parent / "pretrained"
PERIODS = dict(train_period=("2001-10-01", "2013-09-30"), validation_period=("2013-10-01", "2017-09-30"),
               test_period=("2017-10-01", "2023-09-30"))
SITES = ["07326500", "05453100", "02347500", "08167000", "01643000"]


def run_site(out: Path, site: str, model, hyb):
    d = mg.BasinData.from_folder(out / site / "data")
    print(f"[{site}] {d.name}: day shifts {d.stations['day_shift'].to_dict()}", flush=True)
    R = out / site / "results"
    t = time.time()
    res = mg.run_study(d, model, StudyConfig(n_seeds=3, finetune_epochs=20, **PERIODS), hybrid=hyb,
                       progress=lambda f, m: print(f"[{site}] {int(f*100):3d}% {m[:70]}", flush=True) if "epoch" not in m else None)
    res.save(R)
    mg.write_report(res, d, R / "report.html")
    print(summary_table(res).round(3)[["NSE", "KGE", "CRPS"]].to_string(), flush=True)
    if (out / site / "data" / "subbasins" / "subbasins.geojson").exists():
        periods = make_periods(d.dates, StudyConfig(**PERIODS), model.cfg.lookback)
        routed = run_routed_study(d, model, SubbasinSet.from_folder(out / site / "data" / "subbasins"), periods,
                                  routing="gamma", epochs=10)
        routed.summary.to_csv(R / "subbasin_models.csv")
        routed.contributions.to_csv(R / "subbasin_contributions.csv")
        print(routed.summary.xs("test", level="period")[["NSE", "KGE"]].round(3), flush=True)
        b = d.to_basin()
        m = {k: np.asarray((d.dates >= a) & (d.dates <= e)) for k, (a, e) in periods.items()}
        ft = mg.finetune(model, b, m["train"], m["validation"], epochs=20)
        fm = ForecastModel(ft, ForecastConfig(epochs=15)).fit(b, m["train"], m["validation"])
        iss = np.where(m["test"])[0]
        iss = iss[iss < len(b.dates) - 8]
        skill = fm.skill_by_lead(b, iss)
        skill.to_csv(R / "forecast_skill.csv", index=False)
        print(skill.round(3).to_string(index=False), flush=True)
        mg.write_report(res, d, R / "report.html", routed=routed, forecast_skill=skill)
    print(f"[{site}] done in {(time.time() - t) / 60:.0f} min", flush=True)


def summarise(out: Path):
    rows = []
    for site in SITES:
        if not (out / site / "results" / "run_info.json").exists():
            continue
        res = load_saved_study(out / site / "results")
        s = res.summary.xs("test", level="period")
        gv = res.gauge_value[res.gauge_value["period"] == "test"].set_index("gauge_input")
        r = {"site": site, "basin": res.basin_name, "area_km2": round(res.area_km2), "gauges": len(res.station_summary)}
        for k in ["gr4j", "local", "pretrained", "finetuned", "ft_single", "ft_lumped", "ft_voronoi",
                  "ft_disaggregated", "dhbv", "ensemble"]:
            if k in s.index:
                r[f"NSE_{k}"] = round(s.loc[k, "NSE"], 3)
        best = gv["NSE_gain"].idxmax()
        r.update(best_gauge_input=best, NSE_gain_from_gauges=round(gv.loc[best, "NSE_gain"], 3),
                 CRPS_reduction_pct=round(gv.loc[best, "CRPS_reduction_pct"], 1))
        rows.append(r)
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    print(pd.DataFrame(rows).to_string(), flush=True)


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "validation")
    sites = sys.argv[2:] or SITES
    model = StreamflowModel.load(PKG / "camels-us")
    hyb = HybridModel.load(PKG / "camels-us-dhbv")
    for s in sites:
        if s != "summary":
            run_site(out, s, model, hyb)
    summarise(out)
