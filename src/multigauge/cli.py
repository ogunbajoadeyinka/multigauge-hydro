"""Command-line interface.

    multigauge demo     demo_basin/                     write a synthetic example basin
    multigauge study    my_basin/ -o results/           gauge-value study + HTML report
    multigauge subbasins my_basin/ -o results/          phase 2: sub-basins with routing
    multigauge pretrain /data/camels -o model/          pre-train on CAMELS-US (GPU recommended)
    multigauge pretrain-hybrid /data/camels -o dhbv/    pre-train the δHBV hybrid
    multigauge daymet   my_basin/                        download gridded forcings for a basin
    multigauge usgs     07316000 2000-01-01 2020-12-31  download USGS daily discharge
    multigauge ghcn     USC00346139,... 2000-01-01 2020-12-31 -o my_basin/
    multigauge forecast-setup my_basin/ --site 07316000 --out forecasts/my_basin
    multigauge forecast-run forecasts/                  issue today's forecasts
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__


def _progress_printer():
    last = [-1]

    def progress(frac, msg):
        pct = int(frac * 100)
        if pct != last[0]:
            last[0] = pct
            print(f"\r[{pct:3d}%] {msg[:72]:<72}", end="", flush=True)

    return progress


def _period(s):
    return None if s is None else tuple(s.split(":"))


def _load_model(name):
    from .transfer import load_pretrained

    return load_pretrained(name)


# ------------------------------------------------------------------ commands
def _cmd_demo(a):
    from .synthetic import generate_basin, write_basin

    d = generate_basin(n_gauges=a.gauges, years=a.years, seed=a.seed)
    write_basin(d, a.folder)
    print(f"Synthetic basin with {a.gauges} gauges and {a.years} years written to {a.folder}")


def _cmd_study(a):
    from .io import BasinData
    from .report import summary_table, write_report
    from .study import StudyConfig, run_study

    data = BasinData.from_folder(a.folder)
    base = _load_model(a.model)
    hybrid = None
    if a.hybrid:
        from .hybrid import HybridModel

        hybrid = HybridModel.load(a.hybrid)
    cfg = StudyConfig(
        train_period=_period(a.train), validation_period=_period(a.validation), test_period=_period(a.test),
        gauge_inputs=a.gauge_inputs.split(","), single_station=a.single, n_seeds=a.seeds,
        finetune_epochs=a.epochs, include_local=not a.no_local, include_gr4j=not a.no_gr4j,
        run_ablation=a.ablation,
    )
    res = run_study(data, base, cfg, progress=_progress_printer(), hybrid=hybrid)
    print()
    out = res.save(a.output)
    write_report(res, data, out / "report.html")
    print(summary_table(res).round(3).to_string())
    if res.gauge_value is not None:
        print("\nWhat the gauges add (test period):")
        print(res.gauge_value[res.gauge_value["period"] == "test"].round(3).to_string(index=False))
    print(f"\nResults and report.html written to {out}")


def _cmd_subbasins(a):
    import numpy as np

    from .io import BasinData
    from .routing import SubbasinSet, run_routed_study
    from .study import StudyConfig, make_periods

    data = BasinData.from_folder(a.folder)
    base = _load_model(a.model)
    sub_dir = Path(a.subbasins) if a.subbasins else Path(a.folder) / "subbasins"
    if not (sub_dir / "subbasins.geojson").exists():
        raise ValueError(f"No subbasins.geojson in {sub_dir}. Create sub-basins with `multigauge zones` or supply your own.")
    subs = SubbasinSet.from_folder(sub_dir)
    periods = make_periods(data.dates, StudyConfig(), base.cfg.lookback)
    res = run_routed_study(data, base, subs, periods, routing=a.routing, epochs=a.epochs,
                           gauge_ablation=not a.no_ablation, progress=_progress_printer())
    print()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    res.summary.to_csv(out / "subbasin_models.csv")
    res.contributions.to_csv(out / "subbasin_contributions.csv")
    print(res.summary.xs("test", level="period")[["NSE", "KGE"]].round(3).to_string())
    print("\n", res.contributions.round(3).to_string())


def _cmd_zones(a):
    """Approximate sub-basins as distance bands and download their forcings."""
    from shapely.geometry import mapping

    from .download import daymet_basin
    from .io import BasinData
    from .routing import auto_zones
    from .spatial import LocalProjection

    data = BasinData.from_folder(a.folder)
    if data.boundary is None or data.outlet is None:
        raise ValueError("Zones need a basin boundary and the outlet location (basin.json).")
    zones = auto_zones(data.boundary, data.outlet, a.n)
    out = Path(a.folder) / "subbasins"
    out.mkdir(exist_ok=True)
    feats = []
    for i, z in enumerate(zones):
        zid = f"Z{i + 1}"
        c = z.centroid
        dx, dy = LocalProjection(c.x, c.y).forward(data.outlet[1], data.outlet[0])
        feats.append({"type": "Feature", "geometry": mapping(z),
                      "properties": {"id": zid, "downstream": None if i == 0 else f"Z{i}", "distance_km": float((dx**2 + dy**2) ** 0.5)}})
        f = daymet_basin(z, str(data.dates[0].date()), str(data.dates[-1].date()), n_points=a.points)
        f.to_csv(out / f"{zid}.csv", date_format="%Y-%m-%d")
        print(f"{zid}: forcings downloaded")
    (out / "subbasins.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    print(f"{len(zones)} zones written to {out}")


def _cmd_pretrain(a):
    from .camels import load_camels
    from .nn import ModelConfig
    from .transfer import pretrain

    ids = Path(a.basins).read_text().split() if a.basins else None
    basins = load_camels(a.camels, ids, progress=lambda i, n, b: print(f"\rLoading {i}/{n} {b}", end=""))
    print()
    cfg = ModelConfig.preset(a.preset)
    if a.epochs:
        cfg.epochs = a.epochs
    if a.device != "cpu":
        print("Note: training runs on CPU in this version; use Colab's notebook for GPU training.")
    m = pretrain(basins, cfg, tuple(a.train.split(":")), tuple(a.validation.split(":")),
                 progress=lambda e, n, s: print(f"epoch {e}/{n}: median validation NSE {s:.3f}", flush=True))
    m.info.update(source="CAMELS-US", description=f"LSTM pre-trained on {len(basins)} CAMELS-US basins (Daymet forcings).")
    m.save(a.output)
    print(f"Saved to {a.output}")


def _cmd_pretrain_hybrid(a):
    from .camels import load_camels
    from .hybrid import HybridConfig, HybridModel
    from .transfer import period_mask

    ids = Path(a.basins).read_text().split() if a.basins else None
    basins = load_camels(a.camels, ids)
    tr = [period_mask(b.dates, tuple(a.train.split(":"))) for b in basins]
    va = [period_mask(b.dates, tuple(a.validation.split(":"))) for b in basins]
    m = HybridModel(HybridConfig(epochs=a.epochs)).fit(basins, tr, va, progress=lambda e, n, s: print(f"epoch {e}/{n}: {s:.3f}", flush=True))
    m.save(a.output)
    print(f"Saved to {a.output}")


def _cmd_daymet(a):
    from .download import daymet_basin
    from .io import load_boundary, load_rainfall, load_streamflow

    folder = Path(a.folder)
    b = next((folder / n for n in ("boundary.geojson", "boundary.json", "boundary.zip") if (folder / n).exists()), None)
    if b is None:
        raise ValueError("A boundary file is needed to download basin-average forcings.")
    geom = load_boundary(b)
    q = load_streamflow(folder / "streamflow.csv")
    start = a.start or str(q.index.min().date())
    end = a.end or str(q.index.max().date())
    f = daymet_basin(geom, start, end, n_points=a.points, progress=lambda i, n: print(f"\rpoint {i}/{n}", end=""))
    f.to_csv(folder / "forcings.csv", date_format="%Y-%m-%d")
    print(f"\nforcings.csv written ({len(f)} days)")


def _cmd_usgs(a):
    from .download import usgs_daily_flow

    s = usgs_daily_flow(a.site, a.start, a.end, units=a.units)
    s.to_frame().to_csv(a.output, date_format="%Y-%m-%d")
    print(f"{len(s)} days written to {a.output} ({a.units})")


def _cmd_ghcn(a):
    from .download import ghcn_daily_precip

    rain, st = ghcn_daily_precip(a.stations.split(","), a.start, a.end)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    rain.to_csv(out / "rainfall.csv", date_format="%Y-%m-%d")
    st.to_csv(out / "stations.csv")
    print(f"{rain.shape[1]} stations, {len(rain)} days written to {out}")


def _cmd_forecast_setup(a):
    from .io import BasinData
    from .operational import setup_forecast

    data = BasinData.from_folder(a.folder)
    res = setup_forecast(data, _load_model(a.model), a.site, a.out, fetch_archive=not a.no_bias_correction,
                         progress=lambda e, n, s: print(f"epoch {e}/{n}: mean validation NSE {s:.3f}", flush=True))
    print(res["hindcast_skill"].round(3).to_string(index=False))
    print(f"Forecast model saved in {res['folder']}")


def _cmd_forecast_run(a):
    from .operational import run_all

    for r in run_all(a.root):
        if "error" in r:
            print(f"{r['basin']}: ERROR {r['error']}")
        else:
            f = r["forecast"]
            print(f"{r['basin']}: issued {r['issued']}, day-1 median {f[0].get('q50', f[0]['mean']):.2f} {r['unit']}")


# ------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="multigauge", description="Value of local rain gauges for pre-trained deep-learning streamflow models.")
    p.add_argument("--version", action="version", version=f"multigauge {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="write a synthetic example basin")
    d.add_argument("folder")
    d.add_argument("--gauges", type=int, default=10)
    d.add_argument("--years", type=int, default=12)
    d.add_argument("--seed", type=int, default=7)
    d.set_defaults(func=_cmd_demo)

    s = sub.add_parser("study", help="gauge-value study on a basin folder")
    s.add_argument("folder")
    s.add_argument("-o", "--output", default="multigauge_results")
    s.add_argument("--model", default="camels-us", help="pre-trained model name or folder")
    s.add_argument("--hybrid", help="folder of a pre-trained δHBV model (adds hybrid + ensemble)")
    s.add_argument("--gauge-inputs", default="single,lumped,voronoi,disaggregated")
    s.add_argument("--single", default="auto")
    s.add_argument("--seeds", type=int, default=3)
    s.add_argument("--epochs", type=int, default=20, help="fine-tuning epochs")
    s.add_argument("--ablation", action="store_true")
    s.add_argument("--no-local", action="store_true")
    s.add_argument("--no-gr4j", action="store_true")
    s.add_argument("--train")
    s.add_argument("--validation")
    s.add_argument("--test")
    s.set_defaults(func=_cmd_study)

    r = sub.add_parser("subbasins", help="phase 2: sub-basin LSTMs with routing")
    r.add_argument("folder")
    r.add_argument("-o", "--output", default="multigauge_subbasins")
    r.add_argument("--model", default="camels-us")
    r.add_argument("--subbasins", help="folder with subbasins.geojson and <id>.csv forcings (default: <folder>/subbasins)")
    r.add_argument("--routing", choices=["gamma", "gnn"], default="gamma")
    r.add_argument("--epochs", type=int, default=15)
    r.add_argument("--no-ablation", action="store_true")
    r.set_defaults(func=_cmd_subbasins)

    z = sub.add_parser("zones", help="approximate sub-basins as distance bands and download their forcings")
    z.add_argument("folder")
    z.add_argument("-n", type=int, default=4)
    z.add_argument("--points", type=int, default=9)
    z.set_defaults(func=_cmd_zones)

    t = sub.add_parser("pretrain", help="pre-train the LSTM on CAMELS-US")
    t.add_argument("camels")
    t.add_argument("-o", "--output", default="camels-us")
    t.add_argument("--basins", help="text file of gauge ids (default: all)")
    t.add_argument("--preset", default="pretrain")
    t.add_argument("--epochs", type=int)
    t.add_argument("--train", default="1980-10-01:1995-09-30")
    t.add_argument("--validation", default="1995-10-01:2000-09-30")
    t.add_argument("--device", default="cpu")
    t.set_defaults(func=_cmd_pretrain)

    h = sub.add_parser("pretrain-hybrid", help="pre-train the δHBV hybrid on CAMELS-US")
    h.add_argument("camels")
    h.add_argument("-o", "--output", default="camels-us-dhbv")
    h.add_argument("--basins")
    h.add_argument("--epochs", type=int, default=30)
    h.add_argument("--train", default="1980-10-01:1995-09-30")
    h.add_argument("--validation", default="1995-10-01:2000-09-30")
    h.set_defaults(func=_cmd_pretrain_hybrid)

    dm = sub.add_parser("daymet", help="download basin-average Daymet forcings into a basin folder")
    dm.add_argument("folder")
    dm.add_argument("--start")
    dm.add_argument("--end")
    dm.add_argument("--points", type=int, default=16)
    dm.set_defaults(func=_cmd_daymet)

    u = sub.add_parser("usgs", help="download USGS daily discharge")
    u.add_argument("site")
    u.add_argument("start")
    u.add_argument("end")
    u.add_argument("-o", "--output", default="streamflow.csv")
    u.add_argument("--units", choices=["m3/s", "ft3/s"], default="m3/s")
    u.set_defaults(func=_cmd_usgs)

    g = sub.add_parser("ghcn", help="download NOAA GHCN-Daily precipitation")
    g.add_argument("stations")
    g.add_argument("start")
    g.add_argument("end")
    g.add_argument("-o", "--output", default=".")
    g.set_defaults(func=_cmd_ghcn)

    fs = sub.add_parser("forecast-setup", help="train a forecast model for a basin")
    fs.add_argument("folder")
    fs.add_argument("--site", help="USGS site number for recent observed flow")
    fs.add_argument("--out", required=True)
    fs.add_argument("--model", default="camels-us")
    fs.add_argument("--no-bias-correction", action="store_true")
    fs.set_defaults(func=_cmd_forecast_setup)

    fr = sub.add_parser("forecast-run", help="issue forecasts for every basin under a folder")
    fr.add_argument("root")
    fr.set_defaults(func=_cmd_forecast_run)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except Exception as exc:
        from .io import DataError

        if isinstance(exc, (DataError, ValueError, FileNotFoundError)):
            print(f"\nError: {exc}", file=sys.stderr)
            sys.exit(2)
        raise


if __name__ == "__main__":  # pragma: no cover
    main()
