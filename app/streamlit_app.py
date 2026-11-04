"""multigauge web dashboard.

Run locally:   streamlit run app/streamlit_app.py
Deploy:        Streamlit Community Cloud or Hugging Face Spaces (see README).
"""
from __future__ import annotations

import hashlib
import io
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

# allow running from a source checkout without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import multigauge as mg  # noqa: E402
from multigauge.io import DataError  # noqa: E402
from multigauge.report import (  # noqa: E402
    fig_ablation, fig_hydrograph, fig_importance, fig_map, fig_metric, summary_table, write_report,
)
from multigauge.spatial import voronoi_weights  # noqa: E402
from multigauge.strategies import LABELS, STRATEGIES  # noqa: E402
from multigauge.synthetic import write_basin  # noqa: E402

MAX_ABLATION_GAUGES = 15
PLOT_CFG = {"displaylogo": False, "responsive": True}

st.set_page_config(page_title="multigauge · rain gauge input explorer", page_icon="🌧️", layout="wide")


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Building the demo basin...")
def demo_basin():
    return mg.generate_basin()


@st.cache_data(show_spinner=False)
def template_zip() -> bytes:
    d = mg.generate_basin(years=4, n_gauges=4, seed=3)
    with tempfile.TemporaryDirectory() as tmp:
        write_basin(d, tmp)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in Path(tmp).iterdir():
                zf.write(f, f"example_basin/{f.name}")
    return buf.getvalue()


def load_uploaded(files: dict, outlet, name):
    def bio(f):
        return io.BytesIO(f.getvalue()) if f is not None else None

    return mg.BasinData.from_files(
        bio(files["rain"]), bio(files["stations"]), bio(files["flow"]),
        boundary=bio(files["boundary"]), extra=bio(files["extra"]),
        outlet=outlet, name=name or "My basin",
        boundary_filename=files["boundary"].name if files["boundary"] else None,
    )


def fingerprint(*parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p if isinstance(p, bytes) else str(p).encode())
    return h.hexdigest()


# --------------------------------------------------------------------------
# sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("1 · Data")
    source = st.radio("Data source", ["Demo basin", "Upload my basin"], horizontal=True,
                      help="The demo is a synthetic basin with 10 gauges and 12 years of daily data.")
    data, data_key, flow_unit = None, None, "m3/s"

    if source == "Demo basin":
        data = demo_basin()
        data_key = "demo"
    else:
        st.download_button("Download example files", template_zip(), "multigauge_example_basin.zip",
                           help="A small example of every file format, to copy for your own basin.")
        files = {
            "rain": st.file_uploader("Rainfall (CSV) *", type="csv",
                                     help="Columns: date, then one column per gauge, in mm/day. Blank = missing."),
            "stations": st.file_uploader("Gauge locations (CSV) *", type="csv",
                                         help="Columns: id, lat, lon (decimal degrees, WGS84). Ids must match rainfall columns."),
            "flow": st.file_uploader("Streamflow at outlet (CSV) *", type="csv",
                                     help="Columns: date, flow. Any consistent unit."),
            "boundary": st.file_uploader("Basin boundary (optional)", type=["geojson", "json", "zip"],
                                         help="GeoJSON or a zipped shapefile in WGS84. Used for Voronoi weights."),
            "extra": st.file_uploader("Other daily inputs (optional CSV)", type="csv",
                                      help="Basin-wide series such as temperature or PET; added to every model."),
        }
        name = st.text_input("Basin name", "My basin")
        flow_unit = st.text_input("Flow unit (for labels)", "m3/s")
        outlet = None
        if st.checkbox("I know the outlet location"):
            c1, c2 = st.columns(2)
            lat = c1.number_input("Outlet lat", value=35.0, format="%.5f")
            lon = c2.number_input("Outlet lon", value=-98.0, format="%.5f")
            outlet = (lat, lon)
        if all(files[k] is not None for k in ("rain", "stations", "flow")):
            try:
                data = load_uploaded(files, outlet, name)
                data_key = fingerprint(*(f.getvalue() for f in files.values() if f is not None), outlet)
            except DataError as e:
                st.error(str(e))
            except Exception as e:  # unexpected parsing problem
                st.error(f"Could not load the files: {e}")
        else:
            st.info("Upload the three required files (*) to begin.")

    st.header("2 · Settings")
    preset = st.selectbox("Training effort", ["quick", "standard"],
                          help="quick: 90-day lookback, small model (about 1 minute for 10 gauges and 12 years). "
                               "standard: 365-day lookback, as in NeuralHydrology (roughly 15-30 minutes on a "
                               "free cloud CPU). For research-grade runs, use the Python package on your own machine.")
    if preset == "standard":
        st.caption("⏱️ Standard runs can take 15-30 minutes here. Keep this tab open.")
    strategies = st.multiselect("Rainfall inputs to compare", list(STRATEGIES), default=list(STRATEGIES),
                                format_func=LABELS.get)
    single_opts = ["auto"]
    if data is not None:
        if data.outlet is not None:
            single_opts.append("nearest_outlet")
        single_opts += data.station_ids
    single = st.selectbox("Gauge for the single-gauge model", single_opts,
                          format_func=lambda s: {"auto": "Auto: most important gauge",
                                                 "nearest_outlet": "Closest to outlet"}.get(s, s))
    seeds = st.slider("Random seeds per input", 1, 3, 1,
                      help="Each model is trained this many times and the predictions averaged. More seeds = more reliable comparison, longer run.")
    ablation = st.checkbox("Station ablation (retrain without each gauge)",
                           help="Slow: one extra model per gauge per seed.")
    if ablation and data is not None and len(data.station_ids) > MAX_ABLATION_GAUGES:
        st.warning(f"Ablation is limited to {MAX_ABLATION_GAUGES} gauges in the web app; use the Python package for more.")
        ablation = False
    log_flow = st.checkbox("Train on log flow", help="Can help in basins dominated by low flows.")

    with st.expander("Train / validation / test periods"):
        custom = st.checkbox("Set dates myself", help="Default: first 70 % train, next 15 % validation, last 15 % test.")
        periods = {}
        if custom and data is not None:
            d0, d1 = data.dates[0].date(), data.dates[-1].date()
            for per, frac in (("train", (0, 0.7)), ("validation", (0.7, 0.85)), ("test", (0.85, 1.0))):
                n = len(data.dates) - 1
                default = (data.dates[int(frac[0] * n)].date(), data.dates[int(frac[1] * n)].date())
                periods[per] = st.date_input(per.title(), default, min_value=d0, max_value=d1, key=f"p_{per}")

    run = st.button("Run comparison", type="primary", width="stretch",
                    disabled=data is None or not strategies)


# --------------------------------------------------------------------------
# header
# --------------------------------------------------------------------------
st.title("multigauge")
st.markdown(
    "Does using **more rain gauges, and keeping them separate,** improve machine-learning streamflow "
    "prediction? Upload a watershed's gauges and outlet flow, and this tool trains an LSTM for each way of "
    "using the rainfall (one gauge, a simple average, Voronoi-weighted average, or every gauge as its own input), "
    "then shows which works best and which gauges matter most."
)

if data is None:
    st.stop()

for note in data.notes:
    st.caption(f"ℹ️ {note}")

# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------
settings_key = fingerprint(data_key, preset, strategies, single, seeds, ablation, log_flow, periods)
if run:
    tcfg = mg.TrainConfig.preset(preset)
    if log_flow:
        tcfg.target_transform = "log"
    cfg_kwargs = {}
    if periods and all(isinstance(v, tuple) and len(v) == 2 for v in periods.values()):
        cfg_kwargs = {f"{k}_period": (str(v[0]), str(v[1])) for k, v in periods.items()}
    cfg = mg.ExperimentConfig(strategies=strategies, single_station=single, n_seeds=seeds,
                              run_ablation=ablation, training=tcfg, **cfg_kwargs)
    bar = st.progress(0.0, "Starting...")
    try:
        res = mg.run_experiment(data, cfg, progress=lambda f, m: bar.progress(f, m), keep_models=False)
    except (DataError, ValueError) as e:
        bar.empty()
        st.error(str(e))
        st.stop()
    bar.empty()
    st.session_state["result"] = res
    st.session_state["result_key"] = settings_key
    st.session_state["result_data_key"] = data_key

res = st.session_state.get("result")
if res is not None and st.session_state.get("result_data_key") != data_key:
    res = None  # results belong to different data
stale = res is not None and st.session_state.get("result_key") != settings_key

tab_data, tab_res, tab_imp, tab_dl, tab_about = st.tabs(
    ["Basin & gauges", "Model comparison", "Gauge importance", "Download", "About the method"]
)

# --------------------------------------------------------------------------
with tab_data:
    vor = res.voronoi if res is not None else voronoi_weights(data.stations, data.boundary)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Gauges", len(data.station_ids))
    c2.metric("Days of record", f"{len(data.dates):,}")
    c3.metric("Period", f"{data.dates[0].year}–{data.dates[-1].year}")
    c4.metric("Boundary", "Supplied" if vor.used_boundary else "Estimated")
    left, right = st.columns([3, 2])
    with left:
        basemap = st.toggle("Street basemap", value=False, help="Adds map tiles from CARTO/OpenStreetMap.")
        st.plotly_chart(fig_map(data.stations, vor, data.outlet, basemap=basemap), config=PLOT_CFG, width="stretch")
        st.caption("Shading shows each gauge's Voronoi (Thiessen) area weight: darker = larger share of the basin.")
    with right:
        tbl = data.summary().join(vor.weights.round(3))
        st.dataframe(tbl, width="stretch")
        if not vor.used_boundary:
            st.caption("No boundary uploaded, so weights use a buffered hull around the gauges.")

# --------------------------------------------------------------------------
with tab_res:
    if res is None:
        st.info("Choose settings in the sidebar and press **Run comparison**.")
    else:
        if stale:
            st.warning("Settings changed since this run. Press **Run comparison** to update.")
        best = res.best_strategy("test")
        s_test = res.summary.xs("test", level="period")
        cols = st.columns(len(s_test))
        for col, strat in zip(cols, [s for s in STRATEGIES if s in s_test.index]):
            delta = s_test.loc[strat, "NSE"] - s_test.loc[best, "NSE"]
            col.metric(f"{LABELS[strat]} · test NSE", f"{s_test.loc[strat, 'NSE']:.3f}",
                       "best" if strat == best else f"{delta:+.3f} vs best",
                       delta_color="off" if strat == best else "normal")
        st.markdown(f"**Best on the test period: {LABELS[best]}.**"
                    + (f" Single-gauge model used **{res.single_station}** ({res.single_station_reason})."
                       if res.single_station else ""))
        c1, c2 = st.columns(2)
        metric = c1.selectbox("Metric", ["NSE", "KGE", "RMSE", "PBIAS", "PeakError"],
                              help="NSE/KGE: 1 is perfect. RMSE: lower is better. PBIAS: % volume bias. "
                                   "PeakError: % error on the highest 2 % of flows.")
        period = c2.selectbox("Period", ["test", "validation", "train"])
        st.plotly_chart(fig_metric(res.summary, metric, period), config=PLOT_CFG, width="stretch")
        st.dataframe(summary_table(res, period).round(3), width="stretch")
        hp = st.radio("Hydrograph period", ["test", "validation", "train", "all"], horizontal=True)
        st.plotly_chart(fig_hydrograph(res.predictions, hp, flow_unit=flow_unit), config=PLOT_CFG,
                        width="stretch")
        st.caption("Click legend entries to hide or show lines; drag the bottom slider to zoom in time.")

# --------------------------------------------------------------------------
with tab_imp:
    if res is None:
        st.info("Run a comparison to rank the gauges.")
    else:
        if res.importance is not None:
            st.plotly_chart(fig_importance(res.importance), config=PLOT_CFG, width="stretch")
            st.caption("Each gauge's record is shuffled in time and the trained disaggregated model is re-scored. "
                       "A larger NSE drop means the model relies more on that gauge.")
        else:
            st.info("Permutation importance needs the disaggregated model and at least two gauges.")
        if res.ablation is not None:
            st.plotly_chart(fig_ablation(res.ablation), config=PLOT_CFG, width="stretch")
            st.caption("The disaggregated model is retrained without each gauge. Positive bars: accuracy is lost "
                       "without the gauge (it adds unique information). Negative: the model did better without it.")
        else:
            st.caption("Tick *Station ablation* in the sidebar to also retrain without each gauge.")
        st.dataframe(res.station_summary.round(3), width="stretch")

# --------------------------------------------------------------------------
with tab_dl:
    if res is None:
        st.info("Run a comparison to download results.")
    else:
        @st.cache_data(show_spinner="Packaging results...")
        def results_zip(key: str) -> bytes:
            with tempfile.TemporaryDirectory() as tmp:
                out = res.save(Path(tmp) / "results")
                write_report(res, data, out / "report.html", flow_unit=flow_unit)
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    for f in out.rglob("*"):
                        if f.is_file():
                            zf.write(f, f.relative_to(out.parent))
            return buf.getvalue()

        st.download_button("Download results (ZIP)", results_zip(st.session_state["result_key"]),
                           f"multigauge_{res.basin_name.replace(' ', '_')}.zip", type="primary")
        st.markdown(
            "The ZIP contains `report.html` (a shareable report with all charts), `metrics_summary.csv`, "
            "`predictions.csv`, `stations.csv`, importance and ablation tables, and `run_info.json` with every "
            "setting needed to reproduce the run."
        )

# --------------------------------------------------------------------------
with tab_about:
    st.markdown(
        """
**Rainfall input strategies**

| Input | What the LSTM sees |
|---|---|
| Single gauge | One gauge's record (by default the gauge with the highest permutation importance) |
| Lumped average | The arithmetic mean of all gauges |
| Voronoi weighted | Area-weighted mean using Thiessen polygons clipped to the basin |
| Disaggregated | Every gauge as a separate input, preserving where rain fell |

**Model.** A sequence-to-one LSTM reads the previous *lookback* days of inputs and predicts that day's
flow. Inputs are standardised with training-period statistics only; training uses the Adam optimiser
with early stopping on validation NSE. Missing gauge values are filled the same way for every strategy
(Voronoi-weighted mean of reporting gauges) so the comparison is fair.

**Gauge importance.** Permutation importance shuffles one gauge's record and measures the NSE drop;
station ablation retrains without each gauge. Together they show which gauges carry unique information.

**Data.** Use any network you are licensed to use. U.S. users can fetch USGS streamflow and NOAA GHCN-Daily
rainfall with the Python package (`multigauge usgs`, `multigauge ghcn`). Uploaded files are processed
in memory for your session only and are not stored.

**Source code, documentation and citation:** see the project repository (MIT licence).
"""
    )
