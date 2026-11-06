"""Sub-basin (routed) comparison on large basins, with several seeds.

    python scripts/run_subbasins.py validation 05453100 02347500 --seeds 3

For each basin folder validation/<site>/data: make four distance-band
sub-basins with Daymet weather (if not there yet), then for each seed train
(1) the lumped fine-tuned model, (2) four routed sub-basin LSTMs on Daymet
only, (3) the same with each sub-basin's NOAA gauges. Sub-basin gauge value
(ablation) is computed for the first seed. Results go to
validation/<site>/results/subbasins/, with all seeds in seeds_summary.csv,
and a cross-basin table to validation/subbasin_summary.csv.
"""
import argparse
import time
from pathlib import Path

import pandas as pd

import multigauge as mg
from multigauge.nn import StreamflowModel
from multigauge.routing import SubbasinSet, make_zones, run_routed_study
from multigauge.study import StudyConfig, make_periods

PERIODS = dict(train_period=("2001-10-01", "2013-09-30"), validation_period=("2013-10-01", "2017-09-30"),
               test_period=("2017-10-01", "2023-09-30"))


def run_site(root: Path, site: str, model, seeds: int, epochs: int, log=print):
    data_dir = root / site / "data"
    if not (data_dir / "basin.json").exists():  # a basin not prepared yet: download it
        from multigauge.prepare import prepare_usgs_basin

        prepare_usgs_basin(site, "2000-10-01", "2023-09-30", data_dir, progress=log)
    if not (data_dir / "subbasins" / "subbasins.geojson").exists() or len(list((data_dir / "subbasins").glob("Z*.csv"))) < 4:
        make_zones(data_dir, n=4, progress=log)
    d = mg.BasinData.from_folder(data_dir)
    subs = SubbasinSet.from_folder(data_dir / "subbasins")
    periods = make_periods(d.dates, StudyConfig(**PERIODS), model.cfg.lookback)
    out = root / site / "results" / "subbasins"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in range(seeds):
        f = out / f"seed{seed}_summary.csv"
        if f.exists():
            s = pd.read_csv(f, index_col=[0, 1])
        else:
            t = time.time()
            r = run_routed_study(d, model, subs, periods, routing="gamma", epochs=epochs, seed=seed,
                                 gauge_ablation=(seed == 0))
            r.summary.to_csv(f)
            if seed == 0:
                r.contributions.to_csv(out / "contributions.csv")
            s = r.summary
            log(f"[{site}] seed {seed}: {(time.time() - t) / 60:.1f} min")
        for m in ("lumped", "routed", "routed_gauges"):
            rows.append({"site": site, "seed": seed, "model": m, "NSE": s.loc[(m, "test"), "NSE"],
                         "KGE": s.loc[(m, "test"), "KGE"]})
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "seeds_summary.csv", index=False)
    w = tab.pivot_table(index="seed", columns="model", values="NSE")
    res = {"site": site, "basin": d.name, "area_km2": round(d.area_km2), "n_gauges": d.rainfall.shape[1],
           "seeds": len(w), **{f"NSE_{m}": round(w[m].mean(), 3) for m in w},
           **{f"NSE_{m}_sd": round(w[m].std(ddof=1), 3) for m in w}}
    gain = w["routed_gauges"] - w["routed"]
    res.update(gauge_gain_routed=round(gain.mean(), 3), gauge_gain_routed_sd=round(gain.std(ddof=1), 3),
               routing_gain=round((w["routed"] - w["lumped"]).mean(), 3))
    log(pd.Series(res).to_string())
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("sites", nargs="+")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=10)
    a = p.parse_args()
    model = StreamflowModel.load(Path(mg.__file__).parent / "pretrained" / "camels-us")
    root = Path(a.root)
    rows = [run_site(root, s, model, a.seeds, a.epochs) for s in a.sites]
    f = root / "subbasin_summary.csv"
    old = pd.read_csv(f, dtype={"site": str}) if f.exists() else pd.DataFrame()
    new = pd.DataFrame(rows)
    pd.concat([old[~old.get("site", pd.Series(dtype=str)).isin(new["site"])] if len(old) else old, new]).to_csv(f, index=False)
    print(new.to_string())
