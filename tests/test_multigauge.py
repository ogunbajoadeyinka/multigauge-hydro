"""Test suite. Models are tiny so the whole suite runs in a few minutes on a CPU."""
import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest
import torch
from shapely.geometry import box

import multigauge as mg
from multigauge.cli import main as cli_main
from multigauge.io import DataError, load_boundary, load_rainfall, load_stations
from multigauge.metrics import coverage, crps_from_quantiles, kge, nse, pbias
from multigauge.nn import ModelConfig, StreamflowModel, cmal_mean, cmal_nll, cmal_sample
from multigauge.spatial import voronoi_weights
from multigauge.strategies import build_features, weighted_mean

TINY = ModelConfig.preset("tiny")


@pytest.fixture(scope="module")
def basin_data():
    return mg.generate_basin(n_gauges=3, years=5, seed=1, n_subbasins=3)


@pytest.fixture(scope="module")
def pretrain_set():
    return mg.generate_pretraining_set(5, years=5, seed=3)


@pytest.fixture(scope="module")
def base_model(pretrain_set):
    return mg.pretrain(pretrain_set, TINY, ("2000-01-01", "2003-06-30"), ("2003-07-01", "2004-12-31"))


def split(n, a=0.6, b=0.8):
    tr = np.zeros(n, bool); tr[: int(a * n)] = True
    va = np.zeros(n, bool); va[int(a * n): int(b * n)] = True
    return tr, va, ~(tr | va)


# ------------------------------------------------------------------ metrics
def test_metrics_known_values():
    obs = np.array([1.0, 2.0, 3.0, 4.0])
    assert nse(obs, obs) == pytest.approx(1.0)
    assert nse(obs, np.full(4, obs.mean())) == pytest.approx(0.0)
    assert kge(obs, obs) == pytest.approx(1.0)
    assert pbias(obs, obs * 1.1) == pytest.approx(10.0)
    assert coverage(obs, obs - 1, obs + 1) == 1.0
    # a perfect, sharp forecast has zero CRPS
    assert crps_from_quantiles(obs, {0.1: obs, 0.5: obs, 0.9: obs}) == pytest.approx(0.0)
    assert crps_from_quantiles(obs, {0.5: obs + 1}) > 0


# ------------------------------------------------------------------ features
def test_unit_conversions_round_trip():
    from multigauge.features import flow_to_mm_per_day, mm_per_day_to_flow

    q = pd.Series([10.0, 20.0])
    mm = flow_to_mm_per_day(q, 100.0, "m3/s")
    assert mm.iloc[0] == pytest.approx(10 * 86400 / 1e8 * 1000)
    assert np.allclose(mm_per_day_to_flow(mm, 100.0, "m3/s"), q)
    assert np.allclose(flow_to_mm_per_day(q / 0.0283168466, 100.0, "ft3/s"), mm)


def test_static_attributes_are_sensible(basin_data):
    s = basin_data.to_basin().statics
    assert s["p_mean"] > 0 and s["pet_mean"] > 0 and 0 <= s["frac_snow"] <= 1
    assert s["log_area"] == pytest.approx(np.log10(basin_data.area_km2))


# ------------------------------------------------------------------ spatial
def test_voronoi_weights():
    st = pd.DataFrame({"lat": [35.0, 35.0], "lon": [-98.3, -97.7]}, index=["A", "B"])
    vr = voronoi_weights(st, box(-99.0, 34.8, -97.6, 35.2))
    assert vr.weights.sum() == pytest.approx(1.0) and vr.weights["A"] > vr.weights["B"]
    assert voronoi_weights(st.iloc[:1]).weights.iloc[0] == pytest.approx(1.0)


def test_weighted_mean_and_features(basin_data):
    rain = pd.DataFrame({"A": [10.0, np.nan], "B": [0.0, 4.0]})
    out = weighted_mean(rain, pd.Series({"A": 0.75, "B": 0.25}))
    assert out.tolist() == pytest.approx([7.5, 4.0])
    w = voronoi_weights(basin_data.stations, basin_data.boundary).weights
    assert build_features(basin_data.rainfall, "disaggregated", w).shape[1] == 3
    assert list(build_features(basin_data.rainfall, "single", w, station="G02").columns) == ["P_G02"]
    assert not build_features(basin_data.rainfall, "voronoi", w).isna().any().any()


# ------------------------------------------------------------------ loading
def test_loaders_and_errors():
    rain = load_rainfall(io.StringIO("Date,S1,S2\n2020-01-01,1,-999\n2020-01-02,0,2\n"))
    assert np.isnan(rain.loc["2020-01-01", "S2"])
    st = load_stations(io.StringIO("Station_ID,Latitude,Longitude\nS1,35,-98\n"))
    assert list(st.columns) == ["lat", "lon"]
    with pytest.raises(DataError, match="duplicated date"):
        load_rainfall(io.StringIO("date,A\n2020-01-01,1\n2020-01-01,2\n"))
    with pytest.raises(DataError, match="decimal degrees"):
        load_stations(io.StringIO("id,lat,lon\nA,500000,4000000\n"))


def test_folder_round_trip(tmp_path, basin_data):
    mg.write_basin(basin_data, tmp_path)
    again = mg.BasinData.from_folder(tmp_path)
    assert again.has_forcings and again.area_km2 == pytest.approx(basin_data.area_km2, rel=1e-3)
    b = again.to_basin()
    assert b.q_mm.notna().sum() > 1000


def test_missing_forcings_message(basin_data):
    d = mg.BasinData(basin_data.rainfall, basin_data.stations, basin_data.streamflow, basin_data.boundary)
    with pytest.raises(DataError, match="Gridded forcings"):
        d.to_basin()


def test_zipped_shapefile_boundary(tmp_path):
    import shapefile

    w = shapefile.Writer(str(tmp_path / "b"), shapeType=shapefile.POLYGON)
    w.field("name", "C")
    w.poly([[[-98.5, 34.8], [-97.5, 34.8], [-97.5, 35.3], [-98.5, 35.3], [-98.5, 34.8]]])
    w.record("basin")
    w.close()
    with zipfile.ZipFile(tmp_path / "b.zip", "w") as zf:
        for ext in ("shp", "shx", "dbf"):
            zf.write(tmp_path / f"b.{ext}", f"b.{ext}")
    assert load_boundary(tmp_path / "b.zip").bounds == pytest.approx((-98.5, 34.8, -97.5, 35.3))


def test_camels_reader(tmp_path):
    from multigauge.camels import load_camels

    root = tmp_path / "basin_dataset_public_v1p2"
    (root / "basin_mean_forcing" / "daymet" / "11").mkdir(parents=True)
    (root / "usgs_streamflow" / "11").mkdir(parents=True)
    dates = pd.date_range("2000-01-01", periods=800)
    rng = np.random.default_rng(0)
    lines = ["35.0", "400", "1000000000",
             "Year Mnth Day Hr\tdayl(s)\tprcp(mm/day)\tsrad(W/m2)\tswe(mm)\ttmax(C)\ttmin(C)\tvp(Pa)"]
    for d in dates:
        lines.append(f"{d.year} {d.month:02d} {d.day:02d} 12\t40000\t{rng.exponential(2):.2f}\t250\t0\t25\t12\t1300")
    (root / "basin_mean_forcing" / "daymet" / "11" / "07000001_lump_cida_forcing_leap.txt").write_text("\n".join(lines))
    q = [f"07000001 {d.year} {d.month:02d} {d.day:02d} {(-999 if i == 5 else 100.0):.2f} A" for i, d in enumerate(dates)]
    (root / "usgs_streamflow" / "11" / "07000001_streamflow_qc.txt").write_text("\n".join(q))
    (tmp_path / "camels_topo.txt").write_text("gauge_id;gauge_lat;gauge_lon;area_gages2\n07000001;35.0;-98.0;1000\n")
    b = load_camels(tmp_path)[0]
    assert b.basin_id == "07000001" and b.area_km2 == 1000 and b.lon == -98.0
    assert np.isnan(b.q_mm.iloc[5])
    assert b.q_mm.iloc[0] == pytest.approx(100 * 0.0283168466 * 86400 / 1e9 * 1000)


# ------------------------------------------------------------------ network
def test_cmal_consistency():
    torch.manual_seed(0)
    p = {"mu": torch.tensor([[0.0, 2.0]]), "b": torch.tensor([[0.5, 0.3]]),
         "tau": torch.tensor([[0.3, 0.7]]), "pi": torch.tensor([[0.4, 0.6]])}
    s = cmal_sample(p, 40000, torch.Generator().manual_seed(1))
    assert s.mean().item() == pytest.approx(cmal_mean(p).item(), abs=0.03)
    assert torch.isfinite(cmal_nll(p, torch.tensor([1.0]))).all()


def test_extra_inputs_start_neutral(base_model, basin_data):
    b = basin_data.to_basin()
    tr, _, _ = split(len(b.dates))
    before = base_model.predict(b, quantiles=())["mean"]
    m = base_model.copy()
    g = build_features(basin_data.rainfall, "disaggregated", voronoi_weights(basin_data.stations).weights)
    m.add_extra_inputs(b.with_extra(g), tr)
    after = m.predict(b.with_extra(g), quantiles=())["mean"]
    np.testing.assert_allclose(before.dropna(), after.dropna(), rtol=1e-5)


def test_model_save_load_and_quantiles(tmp_path, base_model, basin_data):
    b = basin_data.to_basin()
    p = base_model.predict(b)
    assert {"mean", "q05", "q50", "q95"} <= set(p.columns)
    ok = p.dropna()
    assert (ok["q05"] <= ok["q95"] + 1e-9).all() and (ok["mean"] >= 0).all()
    base_model.save(tmp_path / "m")
    p2 = StreamflowModel.load(tmp_path / "m").predict(b, quantiles=())
    np.testing.assert_allclose(p["mean"].dropna(), p2["mean"].dropna(), rtol=1e-5)
    with pytest.raises(FileNotFoundError):
        mg.load_pretrained("no-such-model")
    assert "demo-synthetic" in mg.available_pretrained()


def test_finetune_and_local(base_model, basin_data):
    b = basin_data.to_basin()
    tr, va, _ = split(len(b.dates))
    ft = mg.finetune(base_model, b, tr, va, epochs=2)
    assert ft.info["finetuned_on"] == b.basin_id
    lo = mg.train_local(b, tr, va, TINY)
    assert lo.predict(b, quantiles=())["mean"].notna().sum() > 500


# ------------------------------------------------------------------ baselines
def test_gr4j():
    from multigauge.baselines import gr4j

    n = 800
    rng = np.random.default_rng(0)
    p, e, t = rng.exponential(2, n), np.full(n, 2.0), np.full(n, 10.0)
    q = gr4j(p, e, t, (300, 0.0, 80, 2.0, 3.0))
    assert (q >= 0).all() and q.sum() < p.sum()


# ------------------------------------------------------------------ study
def test_study_end_to_end(tmp_path, base_model, basin_data):
    cfg = mg.StudyConfig(finetune_epochs=2, local_config=ModelConfig.preset("tiny"), gr4j_maxiter=3,
                         permutation_repeats=2, run_ablation=True, n_seeds=2)
    msgs = []
    res = mg.run_study(basin_data, base_model, cfg, progress=lambda f, m: msgs.append(f))
    models = set(res.summary.index.get_level_values("model"))
    assert {"gr4j", "local", "pretrained", "finetuned", "ft_single", "ft_lumped", "ft_voronoi", "ft_disaggregated"} <= models
    assert set(res.gauge_value["gauge_input"]) == set(mg.STRATEGIES)
    assert len(res.importance) == 3 and len(res.ablation) == 3
    assert msgs[-1] == 1.0 and all(0 <= f <= 1 for f in msgs)
    assert "CRPS" in res.summary.columns
    out = res.save(tmp_path / "r")
    mg.write_report(res, basin_data, out / "report.html")
    for f in ("metrics_summary.csv", "predictions.csv", "gauge_value.csv", "importance.csv", "ablation.csv", "run_info.json", "report.html"):
        assert (out / f).exists(), f
    # flows come back in the user's unit
    fl = res.flow("finetuned")
    assert fl["observed"].mean() == pytest.approx(basin_data.streamflow.mean(), rel=1e-6)


def test_study_period_errors(base_model, basin_data):
    with pytest.raises(ValueError, match="outside the data"):
        mg.run_study(basin_data, base_model, mg.StudyConfig(train_period=("1980-01-01", "1981-01-01"),
                     validation_period=("1985-01-01", "1986-01-01"), test_period=("1990-01-01", "1991-01-01")))


# ------------------------------------------------------------------ phase 2
def test_subbasin_network_checks(basin_data):
    from multigauge.routing import Subbasin, SubbasinSet

    f = basin_data.forcings
    a = Subbasin("A", None, 10, f, "B", 5)
    b = Subbasin("B", None, 10, f, "A", 2)
    with pytest.raises(ValueError):
        SubbasinSet([a, b])  # no outlet / loop


@pytest.mark.parametrize("routing", ["gamma", "gnn"])
def test_routed_models(base_model, basin_data, routing):
    from multigauge.routing import RoutedConfig, RoutedModel, SubbasinSet, subbasin_gauge_inputs

    subs = SubbasinSet.from_synthetic(basin_data)
    assert len(subs.ids) == 3 and subs.depth() == 2
    gin, members = subbasin_gauge_inputs(basin_data, subs)
    assert sum(len(v) for v in members.values()) == 3
    q = mg.BasinData.to_basin(basin_data).q_mm
    tr, va, _ = split(len(q))
    rm = RoutedModel(base_model, subs, gin, RoutedConfig(routing=routing, epochs=1)).fit(q, tr, va)
    out, local = rm.predict(q.index)
    assert out.notna().sum() > 500 and local.shape[1] == 3
    if routing == "gamma":
        assert (rm.travel_times() > 0).all()


def test_routed_study(base_model, basin_data):
    from multigauge.routing import SubbasinSet, run_routed_study
    from multigauge.study import StudyConfig, make_periods

    subs = SubbasinSet.from_synthetic(basin_data)
    periods = make_periods(basin_data.dates, StudyConfig(), base_model.cfg.lookback)
    r = run_routed_study(basin_data, base_model, subs, periods, epochs=1, gauge_ablation=True)
    assert {"lumped", "routed", "routed_gauges"} <= set(r.summary.index.get_level_values("model"))
    assert r.contributions["flow_share"].sum() == pytest.approx(1.0)
    assert "gauge_value_test_nse" in r.contributions


def test_auto_zones(basin_data):
    zones = mg.auto_zones(basin_data.boundary, basin_data.outlet, 3)
    assert 2 <= len(zones) <= 3
    assert sum(z.area for z in zones) == pytest.approx(basin_data.boundary.area, rel=0.02)


# ------------------------------------------------------------------ phase 3
def test_forecast_model(tmp_path, base_model, basin_data):
    from multigauge.forecast import ForecastConfig, ForecastModel

    b = basin_data.to_basin()
    tr, va, te = split(len(b.dates))
    fm = ForecastModel(base_model, ForecastConfig(epochs=1, horizon=3)).fit(b, tr, va)
    iss = np.where(te)[0][:-5]
    sk = fm.skill_by_lead(b, iss)
    assert list(sk["lead_day"]) == [1, 2, 3]
    fc = fm.forecast(b.subset(None, b.dates[-4]), b.forcings.iloc[-3:])
    assert len(fc) == 3 and (fc["q05"] <= fc["q95"]).all()
    fm.save(tmp_path / "f")
    fm2 = ForecastModel.load(tmp_path / "f")
    np.testing.assert_allclose(fm.hindcast(b, iss[:4]), fm2.hindcast(b, iss[:4]), rtol=1e-5)


def test_hybrid_and_ensemble(tmp_path, pretrain_set, basin_data):
    from multigauge.hybrid import HybridConfig, HybridModel, ensemble
    from multigauge.transfer import period_mask

    tr = [period_mask(b.dates, ("2000-01-01", "2003-06-30")) for b in pretrain_set]
    va = [period_mask(b.dates, ("2003-07-01", "2004-12-31")) for b in pretrain_set]
    hm = HybridModel(HybridConfig(epochs=1, samples_per_epoch=64, predict_last=30)).fit(pretrain_set, tr, va)
    b = basin_data.to_basin()
    p = hm.predict(b)
    assert p["mean"].iloc[400:].notna().all() and (p["mean"].dropna() >= 0).all()
    assert hm.params(b).shape == (14, 4)
    hm.save(tmp_path / "h")
    p2 = HybridModel.load(tmp_path / "h").predict(b)
    np.testing.assert_allclose(p["mean"].dropna(), p2["mean"].dropna(), rtol=1e-5)
    e = ensemble([p, p2])
    np.testing.assert_allclose(e["mean"].dropna(), p["mean"].dropna(), rtol=1e-5)


def test_bias_correction():
    from multigauge.operational import BiasCorrection

    idx = pd.date_range("2000-01-01", periods=800)
    ref = pd.DataFrame({"prcp": 2.0, "srad": 200.0, "tmax": 25.0, "tmin": 10.0, "vp": 1200.0}, index=idx)
    src = ref.assign(prcp=1.0, tmax=23.0)
    bc = BiasCorrection.fit(src, ref)
    fixed = bc.apply(src)
    assert fixed["prcp"].mean() == pytest.approx(2.0) and fixed["tmax"].mean() == pytest.approx(25.0)
    assert BiasCorrection.from_dict(json.loads(json.dumps(bc.to_dict()))).prcp_ratio[1] == pytest.approx(2.0)


def test_operational_run_with_mocked_services(tmp_path, base_model, basin_data, monkeypatch):
    from multigauge import download, operational
    from multigauge.forecast import ForecastConfig

    res = operational.setup_forecast(basin_data, base_model, "07000000", tmp_path / "fc",
                                     ForecastConfig(epochs=1, horizon=3), fetch_archive=False)
    assert (tmp_path / "fc" / "model" / "forecast.json").exists()
    f = basin_data.forcings

    def fake_archive(geom, start, end, **k):
        idx = pd.date_range(start, end)
        return pd.DataFrame(np.tile(f.iloc[:1].to_numpy(), (len(idx), 1)), index=idx, columns=f.columns)

    def fake_forecast(geom, days=7, past_days=0, **k):
        today = pd.Timestamp("2030-06-01")
        idx = pd.date_range(today - pd.Timedelta(days=past_days), periods=past_days + days)
        return pd.DataFrame(np.tile(f.iloc[:1].to_numpy(), (len(idx), 1)), index=idx, columns=f.columns)

    monkeypatch.setattr(download, "openmeteo_archive", fake_archive)
    monkeypatch.setattr(download, "openmeteo_forecast", fake_forecast)
    monkeypatch.setattr(download, "usgs_recent_flow", lambda site, days=400, units="m3/s": pd.Series(dtype=float))
    out = operational.run_forecast(tmp_path / "fc", today=pd.Timestamp("2030-06-01"))
    assert out["issued"] == "2030-06-01" and len(out["forecast"]) == 3
    assert (tmp_path / "fc" / "latest.json").exists() and (tmp_path / "fc" / "history.csv").exists()


# ------------------------------------------------------------------ downloads
def test_usgs_parser(monkeypatch):
    from multigauge import download

    pages = [
        {"features": [{"properties": {"time": "2020-01-01", "value": "100", "statistic_id": "00003"}}],
         "links": [{"rel": "next", "href": "https://example/next"}]},
        {"features": [{"properties": {"time": "2020-01-02", "value": "200", "statistic_id": "00003"}}], "links": []},
    ]

    class R:
        def __init__(self, js): self.js = js
        def raise_for_status(self): pass
        def json(self): return self.js

    class S:
        headers = {}
        def get(self, url, params=None, timeout=None): return R(pages.pop(0))

    monkeypatch.setattr(download, "_session", lambda: S())
    s = download.usgs_daily_flow("07316000", "2020-01-01", "2020-01-02")
    assert list(s.round(3)) == [round(100 * download.CFS_TO_CMS, 3), round(200 * download.CFS_TO_CMS, 3)]


def test_daymet_parser():
    from multigauge.download import _parse_daymet

    text = "Latitude: 35  Longitude: -98\nX & Y\nTiles: 1\nElevation: 400\n\nyear,yday,prcp (mm/day),srad (W/m^2),tmax (deg c),tmin (deg c),vp (Pa)\n"
    rows = [f"2020,{d},{d % 3},200,20,10,1200" for d in range(1, 366)]
    df = _parse_daymet(text + "\n".join(rows))
    assert len(df) == 366 and df.index[-1] == pd.Timestamp("2020-12-31") and df.iloc[-1].notna().all()


# ------------------------------------------------------------------ cli
def test_cli(tmp_path, capsys, base_model):
    base_model.save(tmp_path / "model")
    cli_main(["demo", str(tmp_path / "b"), "--gauges", "3", "--years", "5"])
    cli_main(["study", str(tmp_path / "b"), "-o", str(tmp_path / "r"), "--model", str(tmp_path / "model"),
              "--seeds", "1", "--epochs", "1", "--no-local", "--no-gr4j", "--gauge-inputs", "voronoi"])
    assert (tmp_path / "r" / "report.html").exists()
    assert "Fine-tuned + Voronoi gauges" in capsys.readouterr().out


def test_ghcn_inventory_selection():
    from shapely.geometry import box as _box

    from multigauge.prepare import read_ghcn_inventory, select_gauges

    def row(i, lat, lon, el, a, b):
        return f"{i:<11}{lat:>9.4f}{lon:>10.4f} {el:<4}{a:>5}{b:>5}"

    text = "\n".join([
        row("USC00000001", 35.00, -98.00, "PRCP", 1990, 2024),   # inside, full record
        row("USC00000002", 35.05, -98.05, "PRCP", 2015, 2024),   # inside, short record
        row("USC00000003", 35.00, -98.00, "TMAX", 1990, 2024),   # wrong element
        row("USC00000004", 40.00, -90.00, "PRCP", 1990, 2024),   # far away
    ])
    inv = read_ghcn_inventory(text)
    assert list(inv.columns) == ["id", "lat", "lon", "element", "first", "last"]
    sel = select_gauges(inv, _box(-98.3, 34.8, -97.7, 35.2), "2001-10-01", "2023-09-30")
    assert list(sel["id"]) == ["USC00000001"]
