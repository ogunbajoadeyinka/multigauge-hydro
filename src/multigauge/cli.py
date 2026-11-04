"""Command-line interface.

    multigauge demo  my_demo/                 # write a synthetic basin
    multigauge run   my_basin/ -o results/    # full comparison + HTML report
    multigauge usgs  07316000 2010-01-01 2020-12-31 -o streamflow.csv
    multigauge ghcn  USC00346139,USC00341750 2010-01-01 2020-12-31 -o rain/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__


def _cmd_demo(a):
    from .synthetic import generate_basin, write_basin

    d = generate_basin(n_gauges=a.gauges, years=a.years, seed=a.seed)
    write_basin(d, a.folder)
    print(f"Synthetic basin with {a.gauges} gauges and {a.years} years written to {a.folder}")


def _period(s):
    if s is None:
        return None
    a, b = s.split(":")
    return (a, b)


def _cmd_run(a):
    from .experiment import ExperimentConfig, run_experiment
    from .io import BasinData
    from .model import TrainConfig
    from .report import write_report

    data = BasinData.from_folder(a.folder)
    tcfg = TrainConfig.preset(a.preset)
    for k in ("lookback", "epochs", "hidden_size"):
        v = getattr(a, k)
        if v is not None:
            setattr(tcfg, k, v)
    if a.log_flow:
        tcfg.target_transform = "log"
    cfg = ExperimentConfig(
        train_period=_period(a.train),
        validation_period=_period(a.validation),
        test_period=_period(a.test),
        strategies=a.strategies.split(","),
        single_station=a.single,
        n_seeds=a.seeds,
        run_ablation=a.ablation,
        training=tcfg,
    )
    flow_unit = ""
    meta = Path(a.folder) / "basin.json"
    if meta.exists():
        flow_unit = json.loads(meta.read_text()).get("flow_unit", "")

    last = [-1]

    def progress(frac, msg):
        pct = int(frac * 100)
        if pct != last[0]:
            last[0] = pct
            print(f"\r[{pct:3d}%] {msg[:70]:<70}", end="", flush=True)

    res = run_experiment(data, cfg, progress=progress)
    print()
    out = res.save(a.output, save_models=a.save_models)
    write_report(res, data, out / "report.html", flow_unit=flow_unit)
    from .report import summary_table

    print(summary_table(res).round(3).to_string())
    print(f"\nResults and report.html written to {out}")


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
    if st.attrs.get("missing"):
        print("No data returned for:", ", ".join(st.attrs["missing"]))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="multigauge", description="Compare rain gauge input strategies for LSTM streamflow models.")
    p.add_argument("--version", action="version", version=f"multigauge {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="write a synthetic example basin")
    d.add_argument("folder")
    d.add_argument("--gauges", type=int, default=10)
    d.add_argument("--years", type=int, default=12)
    d.add_argument("--seed", type=int, default=7)
    d.set_defaults(func=_cmd_demo)

    r = sub.add_parser("run", help="run the comparison on a basin folder")
    r.add_argument("folder", help="folder with rainfall.csv, stations.csv, streamflow.csv (+ optional boundary.geojson, extra.csv, basin.json)")
    r.add_argument("-o", "--output", default="multigauge_results")
    r.add_argument("--preset", choices=["quick", "standard", "thorough"], default="standard")
    r.add_argument("--strategies", default="single,lumped,voronoi,disaggregated")
    r.add_argument("--single", default="auto", help="gauge id, 'auto' or 'nearest_outlet'")
    r.add_argument("--seeds", type=int, default=3, help="models per strategy (results are averaged)")
    r.add_argument("--ablation", action="store_true", help="also retrain without each gauge")
    r.add_argument("--train", help="START:END, e.g. 2005-10-01:2012-09-30")
    r.add_argument("--validation", help="START:END")
    r.add_argument("--test", help="START:END")
    r.add_argument("--lookback", type=int)
    r.add_argument("--epochs", type=int)
    r.add_argument("--hidden-size", dest="hidden_size", type=int)
    r.add_argument("--log-flow", action="store_true", help="train on log(1+flow)")
    r.add_argument("--save-models", action="store_true")
    r.set_defaults(func=_cmd_run)

    u = sub.add_parser("usgs", help="download USGS daily discharge")
    u.add_argument("site")
    u.add_argument("start")
    u.add_argument("end")
    u.add_argument("-o", "--output", default="streamflow.csv")
    u.add_argument("--units", choices=["m3/s", "ft3/s"], default="m3/s")
    u.set_defaults(func=_cmd_usgs)

    g = sub.add_parser("ghcn", help="download NOAA GHCN-Daily precipitation")
    g.add_argument("stations", help="comma-separated GHCN ids")
    g.add_argument("start")
    g.add_argument("end")
    g.add_argument("-o", "--output", default=".")
    g.set_defaults(func=_cmd_ghcn)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except Exception as exc:  # show a clean message, not a traceback
        from .io import DataError

        if isinstance(exc, (DataError, ValueError)):
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(2)
        raise


if __name__ == "__main__":  # pragma: no cover
    main()
