import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

import multigauge as mg
from multigauge.cli import main as cli_main
from multigauge.io import DataError, load_boundary, load_rainfall, load_stations
from multigauge.metrics import kge, nse, pbias
from multigauge.spatial import voronoi_weights
from multigauge.strategies import build_features, weighted_mean

FAST = mg.TrainConfig(lookback=30, hidden_size=8, epochs=2, patience=2, batch_size=128)


@pytest.fixture(scope="module")
def basin():
    return mg.generate_basin(n_gauges=4, years=4, seed=1)


# ---------------------------------------------------------------- metrics
def test_metrics_known_values():
    obs = np.array([1.0, 2.0, 3.0, 4.0])
    assert nse(obs, obs) == pytest.approx(1.0)
    assert nse(obs, np.full(4, obs.mean())) == pytest.approx(0.0)
    assert kge(obs, obs) == pytest.approx(1.0)
    assert pbias(obs, obs * 1.1) == pytest.approx(10.0)
    assert nse([1, np.nan, 3, 4], [1, 2, np.nan, 4]) == pytest.approx(1.0)


# ---------------------------------------------------------------- spatial
def test_voronoi_weights_sum_to_one_and_follow_area():
    st = pd.DataFrame({"lat": [35.0, 35.0], "lon": [-98.3, -97.7]}, index=["A", "B"])
    # boundary much larger on A's side
    vr = voronoi_weights(st, box(-99.0, 34.8, -97.6, 35.2))
    assert vr.weights.sum() == pytest.approx(1.0)
    assert vr.weights["A"] > vr.weights["B"]
    assert vr.used_boundary


def test_voronoi_without_boundary_and_single_gauge():
    st = pd.DataFrame({"lat": [35.0, 35.2, 34.9], "lon": [-98.0, -97.8, -97.6]}, index=["a", "b", "c"])
    vr = voronoi_weights(st)
    assert not vr.used_boundary and vr.weights.sum() == pytest.approx(1.0)
    one = voronoi_weights(st.iloc[:1])
    assert one.weights.iloc[0] == pytest.approx(1.0)


# ------------------------------------------------------------- strategies
def test_weighted_mean_renormalises_over_missing():
    rain = pd.DataFrame({"A": [10.0, np.nan], "B": [0.0, 4.0]})
    w = pd.Series({"A": 0.75, "B": 0.25})
    out = weighted_mean(rain, w)
    assert out.iloc[0] == pytest.approx(7.5)
    assert out.iloc[1] == pytest.approx(4.0)


def test_build_features_shapes(basin):
    w = voronoi_weights(basin.stations, basin.boundary).weights
    n = len(basin.station_ids)
    n_extra = basin.extra.shape[1]
    assert build_features(basin.rainfall, "disaggregated", w, extra=basin.extra).shape[1] == n + n_extra
    for s in ("lumped", "voronoi"):
        f = build_features(basin.rainfall, s, w)
        assert f.shape[1] == 1 and not f.isna().any().any()
    f = build_features(basin.rainfall, "single", w, station="G02")
    assert list(f.columns) == ["P_G02"]
    f = build_features(basin.rainfall, "disaggregated", w, drop=["G01"])
    assert "P_G01" not in f.columns
    with pytest.raises(ValueError):
        build_features(basin.rainfall, "single", w)


# ---------------------------------------------------------------- loading
def test_loaders_accept_common_column_names():
    rain = load_rainfall(io.StringIO("Date,S1,S2\n2020-01-01,1,-999\n2020-01-02,0,2\n"))
    assert np.isnan(rain.loc["2020-01-01", "S2"])  # negative codes become missing
    st = load_stations(io.StringIO("Station_ID,Latitude,Longitude\nS1,35,-98\nS2,35.1,-97.9\n"))
    assert list(st.columns) == ["lat", "lon"] and list(st.index) == ["S1", "S2"]


def test_loader_errors_are_helpful():
    with pytest.raises(DataError, match="duplicated date"):
        load_rainfall(io.StringIO("date,A\n2020-01-01,1\n2020-01-01,2\n"))
    with pytest.raises(DataError, match="latitude"):
        load_stations(io.StringIO("id,foo,bar\nA,1,2\n"))
    with pytest.raises(DataError, match="decimal degrees"):
        load_stations(io.StringIO("id,lat,lon\nA,500000,4000000\n"))


def test_basin_requires_matching_ids(basin):
    stations = basin.stations.rename(index={"G01": "X"})
    with pytest.raises(DataError, match="no entry in the stations file"):
        mg.BasinData(basin.rainfall, stations, basin.streamflow)


def test_folder_roundtrip_and_geojson(tmp_path, basin):
    mg.write_basin(basin, tmp_path)
    again = mg.BasinData.from_folder(tmp_path)
    assert again.station_ids == basin.station_ids
    assert again.outlet is not None and again.boundary is not None
    pd.testing.assert_frame_equal(again.rainfall, basin.rainfall, check_freq=False, check_dtype=False, atol=1e-6)


def test_zipped_shapefile_boundary(tmp_path):
    import shapefile

    w = shapefile.Writer(str(tmp_path / "b"), shapeType=shapefile.POLYGON)
    w.field("name", "C")
    w.poly([[[-98.5, 34.8], [-97.5, 34.8], [-97.5, 35.3], [-98.5, 35.3], [-98.5, 34.8]]])
    w.record("basin")
    w.close()
    zpath = tmp_path / "b.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for ext in ("shp", "shx", "dbf"):
            zf.write(tmp_path / f"b.{ext}", f"b.{ext}")
    geom = load_boundary(zpath)
    assert geom.bounds == pytest.approx((-98.5, 34.8, -97.5, 35.3))


# ----------------------------------------------------------------- model
def test_model_fit_predict_save_load(tmp_path, basin):
    w = voronoi_weights(basin.stations, basin.boundary).weights
    feats = build_features(basin.rainfall, "voronoi", w)
    n = len(feats)
    train = np.zeros(n, bool); train[: n // 2] = True
    val = np.zeros(n, bool); val[n // 2 : 3 * n // 4] = True
    m = mg.StreamflowLSTM(FAST).fit(feats, basin.streamflow, train, val)
    p = m.predict(feats)
    assert p.iloc[: FAST.lookback - 1].isna().all() and p.iloc[FAST.lookback :].notna().all()
    assert (p.dropna() >= 0).all()
    m.save(tmp_path / "m")
    p2 = mg.StreamflowLSTM.load(tmp_path / "m").predict(feats)
    np.testing.assert_allclose(p.dropna(), p2.dropna(), rtol=1e-5)


# ------------------------------------------------------------ experiment
def test_full_experiment_and_outputs(tmp_path, basin):
    cfg = mg.ExperimentConfig(training=FAST, n_seeds=2, run_ablation=True, permutation_repeats=2)
    msgs = []
    res = mg.run_experiment(basin, cfg, progress=lambda f, m: msgs.append(f))
    assert set(res.summary.index.get_level_values("strategy")) == set(mg.STRATEGIES)
    assert res.single_station in basin.station_ids
    assert len(res.importance) == len(basin.station_ids)
    assert len(res.ablation) == len(basin.station_ids)
    assert msgs[-1] == 1.0 and all(0 <= f <= 1 for f in msgs)
    # 2 seeds x 4 strategies x 3 periods x 5 metrics
    assert len(res.metrics) == 2 * 4 * 3 * 5
    out = res.save(tmp_path / "out")
    mg.write_report(res, basin, out / "report.html")
    for f in ("metrics_summary.csv", "predictions.csv", "stations.csv", "run_info.json",
              "permutation_importance.csv", "station_ablation.csv", "report.html"):
        assert (out / f).exists(), f
    info = json.loads((out / "run_info.json").read_text())
    assert info["single_station"] == res.single_station


def test_explicit_periods_and_single_choices(basin):
    d = basin.dates
    cfg = mg.ExperimentConfig(
        train_period=(str(d[0].date()), "2011-12-31"),
        validation_period=("2012-01-01", "2012-12-31"),
        test_period=("2013-01-01", str(d[-1].date())),
        strategies=["single", "lumped"], single_station="nearest_outlet", training=FAST,
    )
    res = mg.run_experiment(basin, cfg)
    assert res.periods["validation"][0] == pd.Timestamp("2012-01-01")
    assert "closest to outlet" in res.single_station_reason
    with pytest.raises(ValueError, match="outside the data"):
        mg.run_experiment(basin, mg.ExperimentConfig(test_period=("1990-01-01", "1991-01-01"),
                                                     train_period=("1980-01-01", "1981-01-01"),
                                                     validation_period=("1985-01-01", "1986-01-01")))


# -------------------------------------------------------------------- cli
def test_cli_demo_and_run(tmp_path, capsys):
    cli_main(["demo", str(tmp_path / "b"), "--gauges", "3", "--years", "4"])
    cli_main(["run", str(tmp_path / "b"), "-o", str(tmp_path / "r"), "--preset", "quick",
              "--seeds", "1", "--epochs", "1", "--lookback", "20"])
    assert (tmp_path / "r" / "report.html").exists()
    assert "Disaggregated" in capsys.readouterr().out


# --------------------------------------------------------------- download
def test_usgs_parser_with_mocked_api(monkeypatch):
    from multigauge import download

    pages = [
        {"features": [{"properties": {"time": "2020-01-01", "value": "100", "statistic_id": "00003"}}],
         "links": [{"rel": "next", "href": "https://example/next"}]},
        {"features": [{"properties": {"time": "2020-01-02", "value": "200", "statistic_id": "00003"}},
                      {"properties": {"time": "2020-01-02", "value": "999", "statistic_id": "00001"}}],
         "links": []},
    ]

    class Resp:
        def __init__(self, js): self.js = js
        def raise_for_status(self): pass
        def json(self): return self.js

    class Sess:
        headers = {}
        def get(self, url, params=None, timeout=None):
            return Resp(pages.pop(0))

    monkeypatch.setattr(download, "_session", lambda: Sess())
    s = download.usgs_daily_flow("07316000", "2020-01-01", "2020-01-02")
    assert list(s.round(3)) == [round(100 * download.CFS_TO_CMS, 3), round(200 * download.CFS_TO_CMS, 3)]
