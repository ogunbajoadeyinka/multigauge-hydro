"""multigauge web dashboard.

Run locally:   streamlit run app/streamlit_app.py
Deploy:        Streamlit Community Cloud or Hugging Face Spaces (see README).
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # run from a source checkout without installing

import multigauge as mg  # noqa: E402  (lazy: PyTorch loads only when a model runs)
from multigauge.io import DataError  # noqa: E402

PLOT = {"displaylogo": False, "responsive": True}
st.set_page_config(page_title="multigauge · rain gauges and AI streamflow models", page_icon="🌧️", layout="wide")


# ==========================================================================
# cached helpers
# ==========================================================================
@st.cache_resource(show_spinner="Building the demo basin...")
def demo_basin():
    return mg.generate_basin()


@st.cache_resource(show_spinner="Loading the pre-trained model...")
def pretrained(name: str):
    return mg.load_pretrained(name)


@st.cache_data(show_spinner=False)
def template_zip() -> bytes:
    d = mg.generate_basin(years=4, n_gauges=4, seed=3, n_subbasins=0)
    with tempfile.TemporaryDirectory() as tmp:
        mg.write_basin(d, tmp)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in Path(tmp).iterdir():
                zf.write(f, f"example_basin/{f.name}")
    return buf.getvalue()


def fingerprint(*parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p if isinstance(p, bytes) else str(p).encode())
    return h.hexdigest()


def bar(label):
    b = st.progress(0.0, label)
    return b, (lambda f, m: b.progress(min(max(f, 0.0), 1.0), m[:90]))


# ==========================================================================
# sidebar
# ==========================================================================
with st.sidebar:
    st.header("1 · Basin data")
    source = st.radio("Data source", ["Demo basin", "Upload my basin"], horizontal=True,
                      help="The demo is a synthetic basin where the true rainfall is known, with 10 gauges and 12 years of data.")
    data, data_key = None, None
    if source == "Demo basin":
        data, data_key = demo_basin(), "demo"
    else:
        st.download_button("Download example files", template_zip(), "multigauge_example_basin.zip",
                           help="A small example of every file format.")
        files = {
            "rain": st.file_uploader("Rain gauges (CSV) *", type="csv", help="date, then one column per gauge (mm/day)."),
            "stations": st.file_uploader("Gauge locations (CSV) *", type="csv", help="id, lat, lon (decimal degrees)."),
            "flow": st.file_uploader("Streamflow at the outlet (CSV) *", type="csv", help="date, flow."),
            "forcings": st.file_uploader("Gridded weather, Daymet (CSV) *", type="csv",
                                         help="date, prcp, srad, tmax, tmin, vp for the whole basin. "
                                              "Create it with `multigauge daymet my_basin/`."),
            "boundary": st.file_uploader("Basin boundary", type=["geojson", "json", "zip"],
                                         help="GeoJSON or zipped shapefile (WGS84). Gives the area and Voronoi weights."),
        }
        name = st.text_input("Basin name", "My basin")
        unit = st.selectbox("Flow unit", ["m3/s", "ft3/s"])
        area = st.number_input("Basin area (km², if no boundary)", min_value=0.0, value=0.0)
        outlet = None
        if st.checkbox("I know the outlet location"):
            c1, c2 = st.columns(2)
            outlet = (c1.number_input("Outlet lat", value=35.0, format="%.5f"), c2.number_input("Outlet lon", value=-98.0, format="%.5f"))
        if all(files[k] is not None for k in ("rain", "stations", "flow", "forcings")):
            try:
                bio = lambda f: io.BytesIO(f.getvalue()) if f is not None else None
                data = mg.BasinData.from_files(
                    bio(files["rain"]), bio(files["stations"]), bio(files["flow"]), boundary=bio(files["boundary"]),
                    outlet=outlet, name=name, boundary_filename=files["boundary"].name if files["boundary"] else None,
                    forcings=bio(files["forcings"]), area_km2=area or None, flow_unit=unit,
                )
                data_key = fingerprint(*(f.getvalue() for f in files.values() if f is not None), outlet, area, unit)
            except (DataError, ValueError) as e:
                st.error(str(e))
        else:
            st.info("Upload the four required files (*) to begin.")

    st.header("2 · Pre-trained model")
    models = mg.available_pretrained()
    default = "camels-us" if "camels-us" in models else (models[0] if models else None)
    model_name = st.selectbox("Model", models, index=models.index(default) if default else 0,
                              help="camels-us: trained on 531 U.S. basins. demo-synthetic: for the demo basin only.")
    if model_name == "demo-synthetic" and source != "Demo basin":
        st.warning("The demo model was trained on synthetic data. Use the CAMELS-US model for real basins.")

    st.header("3 · Settings")
    effort = st.radio("Effort", ["quick", "standard"], horizontal=True,
                      help="quick: 1 seed, short fine-tuning (a few minutes). standard: 3 seeds, longer training.")
    gauge_inputs = st.multiselect("Gauge inputs to test", list(mg.STRATEGIES), default=list(mg.STRATEGIES),
                                  format_func=lambda s: {"single": "Single gauge", "lumped": "Gauge average",
                                                         "voronoi": "Voronoi-weighted", "disaggregated": "All gauges"}[s])
    c1, c2 = st.columns(2)
    use_gr4j = c1.checkbox("GR4J", True, help="Calibrated conceptual model, the classic benchmark.")
    use_local = c2.checkbox("Local LSTM", True, help="LSTM trained from scratch on this basin only.")
    ablation = st.checkbox("Station ablation", help="Retrain without each gauge (slow).")
    run = st.button("Run gauge-value study", type="primary", width="stretch", disabled=data is None or not models)


# ==========================================================================
# header
# ==========================================================================
st.title("multigauge")
st.markdown(
    "**How much do local rain gauges add to a state-of-the-art AI streamflow model, and which gauges matter?** "
    "multigauge starts from an LSTM pre-trained on hundreds of U.S. basins, fine-tunes it on your basin, adds your "
    "rain gauges as extra inputs, and measures the gain against conceptual and single-basin baselines, with "
    "prediction intervals, sub-basin routing and forecasts."
)
if data is None:
    st.stop()
for note in data.notes:
    st.caption(f"ℹ️ {note}")


# ==========================================================================
# run the study
# ==========================================================================
settings_key = fingerprint(data_key, model_name, effort, gauge_inputs, use_gr4j, use_local, ablation)
if run:
    from multigauge.nn import ModelConfig

    quick = effort == "quick"
    lcfg = ModelConfig.preset("local")
    if quick:
        lcfg.epochs, lcfg.patience = 15, 4
    cfg = mg.StudyConfig(gauge_inputs=gauge_inputs or [], n_seeds=1 if quick else 3, finetune_epochs=8 if quick else 20,
                         include_gr4j=use_gr4j, include_local=use_local, local_config=lcfg, run_ablation=ablation,
                         gr4j_maxiter=20 if quick else 40, permutation_repeats=3 if quick else 5)
    b, cb = bar("Starting...")
    try:
        res = mg.run_study(data, pretrained(model_name), cfg, progress=cb)
    except (DataError, ValueError) as e:
        b.empty()
        st.error(str(e))
        st.stop()
    b.empty()
    st.session_state.update(result=res, result_key=settings_key, result_data=data_key)

res = st.session_state.get("result")
if res is not None and st.session_state.get("result_data") != data_key:
    res = None
stale = res is not None and st.session_state.get("result_key") != settings_key

tabs = st.tabs(["Basin & gauges", "Model comparison", "Gauge value", "Sub-basins", "Forecast", "Download", "Method"])
from multigauge import report as R  # noqa: E402

# ---------------------------------------------------------------- basin
with tabs[0]:
    vor = res.voronoi if res is not None else mg.voronoi_weights(data.stations, data.boundary)
    c = st.columns(4)
    c[0].metric("Rain gauges", len(data.station_ids))
    c[1].metric("Days of record", f"{len(data.dates):,}")
    c[2].metric("Period", f"{data.dates[0].year}–{data.dates[-1].year}")
    c[3].metric("Area", f"{data.area_km2:,.0f} km²" if data.area_km2 else "unknown")
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(R.fig_map(data.stations, vor, data.outlet), config=PLOT, width="stretch")
        st.caption("Shading: each gauge's Voronoi (Thiessen) share of the basin.")
    with right:
        st.dataframe(data.summary().join(vor.weights.round(3)), width="stretch")

# ----------------------------------------------------------- comparison
with tabs[1]:
    if res is None:
        st.info("Press **Run gauge-value study** in the sidebar.")
    else:
        if stale:
            st.warning("Settings changed since this run. Run the study again to update.")
        s = res.summary.xs("test", level="period")
        best = res.best_model("test")
        k = st.columns(4)
        k[0].metric(f"Best: {R.LABELS[best]}", f"{s.loc[best, 'NSE']:.3f}", help="Test-period NSE")
        if "local" in s.index:
            k[1].metric("Local LSTM (from scratch)", f"{s.loc['local', 'NSE']:.3f}")
        if "finetuned" in s.index:
            k[2].metric("Fine-tuned, gridded only", f"{s.loc['finetuned', 'NSE']:.3f}")
        if res.gauge_value is not None:
            gv = res.gauge_value[res.gauge_value["period"] == "test"]
            k[3].metric("Largest gain from gauges", f"{gv['NSE_gain'].max():+.3f} NSE")
        c1, c2 = st.columns(2)
        metric = c1.selectbox("Metric", ["NSE", "KGE", "CRPS", "Coverage90", "RMSE", "PBIAS", "PeakError"],
                              help="CRPS scores the whole prediction distribution (lower is better). "
                                   "Coverage90: share of days inside the 90% interval (ideal 0.90).")
        period = c2.selectbox("Period", ["test", "validation", "train"])
        if metric in s.columns:
            st.plotly_chart(R.fig_metric(res.summary, metric, period), config=PLOT, width="stretch")
        st.dataframe(R.summary_table(res, period).round(3), width="stretch")
        flows = {m: res.flow(m) for m in res.models}
        pick = st.multiselect("Models on the hydrograph", res.models, default=[m for m in ("finetuned", best, "local") if m in res.models][:3],
                              format_func=R.LABELS.get)
        hp = st.radio("Hydrograph period", ["test", "validation", "train", "all"], horizontal=True)
        band = best if "q05" in res.predictions[best] else None
        if pick:
            st.plotly_chart(R.fig_hydrograph(flows, list(dict.fromkeys(pick)), hp, res.flow_unit, band_model=band), config=PLOT, width="stretch")
            if band:
                st.caption(f"Shaded: 90% prediction interval of {R.LABELS[band]}.")

# ---------------------------------------------------------- gauge value
with tabs[2]:
    if res is None or res.gauge_value is None:
        st.info("Run the study with at least one gauge input to see what the gauges add.")
    else:
        st.markdown("Gain of each fine-tuned model **with gauges** over the same model **without gauges** "
                    "(gridded rainfall only). Positive = the gauges help.")
        c1, c2 = st.columns(2)
        c1.plotly_chart(R.fig_gauge_value(res.gauge_value, "NSE_gain"), config=PLOT, width="stretch")
        if res.gauge_value["CRPS_reduction_pct"].notna().any():
            c2.plotly_chart(R.fig_gauge_value(res.gauge_value, "CRPS_reduction_pct"), config=PLOT, width="stretch")
        if res.importance is not None:
            st.plotly_chart(R.fig_importance(res.importance), config=PLOT, width="stretch")
        if res.ablation is not None:
            st.plotly_chart(R.fig_ablation(res.ablation), config=PLOT, width="stretch")
        st.dataframe(res.station_summary.round(3), width="stretch")

# ------------------------------------------------------------ sub-basins
with tabs[3]:
    st.markdown("**Phase 2.** Split the basin into sub-basins, run the pre-trained LSTM on each, and route flow to the "
                "outlet with learned travel times (gamma) or a graph neural network (GNN). Shows where flood flow "
                "comes from and where gauges matter.")
    subs = None
    if data.meta.get("synthetic_zones") is not None:
        subs = mg.SubbasinSet.from_synthetic(data)
    else:
        z = st.file_uploader("Sub-basins (ZIP with subbasins.geojson and one <id>.csv of Daymet forcings per sub-basin)", type="zip",
                             help="Create approximate zones with `multigauge zones my_basin/`.")
        if z is not None:
            with tempfile.TemporaryDirectory() as tmp:
                zipfile.ZipFile(io.BytesIO(z.getvalue())).extractall(tmp)
                root = next((p.parent for p in Path(tmp).rglob("subbasins.geojson")), None)
                if root is None:
                    st.error("subbasins.geojson not found in the ZIP.")
                else:
                    try:
                        subs = mg.SubbasinSet.from_folder(root)
                    except Exception as e:
                        st.error(str(e))
    if subs is not None:
        routing = st.radio("Routing", ["gamma", "gnn"], horizontal=True,
                           format_func={"gamma": "Learned travel times (gamma)", "gnn": "Graph neural network"}.get)
        if st.button("Run sub-basin analysis"):
            from multigauge.study import StudyConfig, make_periods

            base = pretrained(model_name)
            periods = make_periods(data.dates, StudyConfig(), base.cfg.lookback)
            b, cb = bar("Starting...")
            st.session_state["routed"] = (data_key, routing, mg.run_routed_study(
                data, base, subs, periods, routing=routing, epochs=6 if effort == "quick" else 15,
                gauge_ablation=(effort != "quick"), progress=cb))
            b.empty()
        rr = st.session_state.get("routed")
        if rr is not None and rr[0] == data_key:
            r = rr[2]
            c1, c2 = st.columns([3, 2])
            c1.plotly_chart(R.fig_subbasin_map(subs, data.stations, r.contributions), config=PLOT, width="stretch")
            c2.dataframe(R._routed_table(r).round(3), width="stretch")
            if r.travel_times is not None:
                c2.dataframe(r.travel_times.round(2).to_frame(), width="stretch")
            st.plotly_chart(R.fig_contributions(r.contributions), config=PLOT, width="stretch")
            st.dataframe(r.contributions.round(3), width="stretch")
            st.caption("priority_for_new_gauge: share of flood flow from sub-basins that have no gauge yet, "
                       "a simple indicator of where a new gauge could help most.")

# --------------------------------------------------------------- forecast
with tabs[4]:
    st.markdown("**Phase 3.** Probabilistic forecasts 1–7 days ahead. The model reads the past year of weather and "
                "recent observed flow, then a forecast LSTM reads the weather forecast.")
    live = sorted((ROOT / "forecasts").glob("*/latest.json"))
    if live:
        st.subheader("Live forecasts")
        for f in live:
            issued = json.loads(f.read_text())
            st.plotly_chart(R.fig_forecast(issued), config=PLOT, width="stretch")
            if issued.get("note"):
                st.caption(issued["note"])
    st.subheader("Forecast skill for this basin")
    st.caption("Trains a forecast model and tests it on the test period, using observed weather as a perfect forecast.")
    if st.button("Train and test a forecast model"):
        import numpy as np

        from multigauge.forecast import ForecastConfig, ForecastModel
        from multigauge.study import StudyConfig, make_periods

        base = pretrained(model_name)
        basin = data.to_basin()
        periods = make_periods(data.dates, StudyConfig(), base.cfg.lookback)
        m = {k: np.asarray((data.dates >= a) & (data.dates <= e)) for k, (a, e) in periods.items()}
        b, cb = bar("Fine-tuning...")
        ft = mg.finetune(base, basin, m["train"], m["validation"], epochs=8)
        fm = ForecastModel(ft, ForecastConfig(epochs=4 if effort == "quick" else 12)).fit(
            basin, m["train"], m["validation"], progress=lambda e, n, s: cb(e / n, f"Forecast model: epoch {e}/{n}"))
        iss = np.where(m["test"])[0]
        iss = iss[iss < len(basin.dates) - fm.cfg.horizon - 1]
        skill = fm.skill_by_lead(basin, iss)
        last = len(basin.dates) - fm.cfg.horizon - 1
        fc = fm.forecast(basin.subset(None, basin.dates[last]), basin.forcings.iloc[last + 1 : last + 1 + fm.cfg.horizon])
        b.empty()
        st.session_state["forecast"] = (data_key, skill, fc, basin.dates[last], basin.q_mm.iloc[last - 30 : last + 1 + fm.cfg.horizon])
    fs = st.session_state.get("forecast")
    if fs is not None and fs[0] == data_key:
        _, skill, fc, when, recent = fs
        st.plotly_chart(R.fig_skill_by_lead(skill), config=PLOT, width="stretch")
        from multigauge.features import mm_per_day_to_flow

        fcf = fc.copy()
        for col in fcf.columns:
            if col != "lead_day":
                fcf[col] = mm_per_day_to_flow(fcf[col], data.area_km2, data.flow_unit)
        obs = mm_per_day_to_flow(recent, data.area_km2, data.flow_unit)
        issued = {"basin": data.name, "issued": str(when.date()), "unit": data.flow_unit,
                  "recent_observed": {str(k.date()): (None if pd.isna(v) else float(v)) for k, v in obs.items()},
                  "forecast": [{"date": str(i.date()), **{k: float(v) for k, v in r.items()}} for i, r in fcf.iterrows()]}
        st.plotly_chart(R.fig_forecast(issued, history_days=40), config=PLOT, width="stretch")
        st.caption("Example forecast issued on the last available date, compared with what actually happened.")

# ---------------------------------------------------------------- download
with tabs[5]:
    if res is None:
        st.info("Run the study to download results.")
    else:
        @st.cache_data(show_spinner="Packaging results...")
        def results_zip(key: str) -> bytes:
            with tempfile.TemporaryDirectory() as tmp:
                out = res.save(Path(tmp) / "results")
                rr = st.session_state.get("routed")
                R.write_report(res, data, out / "report.html", routed=rr[2] if rr and rr[0] == data_key else None)
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    for f in out.rglob("*"):
                        if f.is_file():
                            zf.write(f, f.relative_to(out.parent))
            return buf.getvalue()

        st.download_button("Download results (ZIP)", results_zip(st.session_state["result_key"]),
                           f"multigauge_{res.basin_name.replace(' ', '_')}.zip", type="primary")
        st.markdown("Contains `report.html` (all charts, works offline), metric tables, daily predictions with "
                    "prediction intervals, gauge rankings and `run_info.json` with every setting.")

# ------------------------------------------------------------------ method
with tabs[6]:
    st.markdown(
        """
| Model | What it is |
|---|---|
| GR4J | Four-parameter conceptual model (+ snow), calibrated on this basin: the classic benchmark |
| Local LSTM | LSTM trained from scratch on this basin only, the conventional single-basin approach |
| Pre-trained | LSTM trained on many basins, used without seeing this basin |
| Fine-tuned | The pre-trained LSTM fine-tuned on this basin with gridded (Daymet) weather |
| Fine-tuned + gauges | As above, plus local rain gauges as extra inputs: single, average, Voronoi-weighted or all gauges |
| δHBV hybrid (CLI) | Differentiable HBV with neural parameter learning, and an LSTM + δHBV ensemble |

**Why pre-train?** LSTMs trained on many basins are much more accurate than single-basin models
(Kratzert et al., 2019, 2024), and fine-tuning adapts them to local conditions (Ryd & Nearing, 2025).
**Gauge inputs** are added with zero initial weight, so fine-tuning starts from the pre-trained model and
learns how much the gauges help. **Prediction intervals** come from a mixture-density output (CMAL),
as in Google's operational flood model (Nearing et al., 2024). **Sub-basin routing** follows Yu et al.
(2024) and Mosaffa et al. (2026); **ensembles with a differentiable model** follow Li et al. (2025).

Uploaded files are processed in memory for your session only and are not stored.
Source code, documentation and citation: see the GitHub repository (MIT licence).
"""
    )
