"""Figures and a self-contained HTML report.

The same figure functions feed the web dashboard and the HTML report, so
what a user sees online is exactly what they can download.
"""
from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from . import __version__

LABELS = {
    "gr4j": "GR4J (conceptual)",
    "local": "Local LSTM (from scratch)",
    "pretrained": "Pre-trained, not fine-tuned",
    "finetuned": "Fine-tuned (gridded only)",
    "ft_single": "Fine-tuned + single gauge",
    "ft_lumped": "Fine-tuned + gauge average",
    "ft_voronoi": "Fine-tuned + Voronoi gauges",
    "ft_disaggregated": "Fine-tuned + all gauges",
    "lumped": "Lumped (one basin-wide model)",
    "routed": "Sub-basins + routing",
    "routed_gauges": "Sub-basins + routing + gauges",
    "dhbv": "δHBV hybrid",
    "ensemble": "LSTM + δHBV ensemble",
}
# Colour follows the model everywhere (validated categorical order).
COLORS = {
    "finetuned": "#2a78d6", "ft_disaggregated": "#eb6834", "ft_voronoi": "#1baf7a", "ft_lumped": "#eda100",
    "ft_single": "#e87ba4", "pretrained": "#008300", "local": "#6250d6", "gr4j": "#e34948",
    "lumped": "#2a78d6", "routed": "#eb6834", "routed_gauges": "#1baf7a", "dhbv": "#6250d6", "ensemble": "#eda100",
}
GAUGE_COLORS = {"single": "#e87ba4", "lumped": "#eda100", "voronoi": "#1baf7a", "disaggregated": "#eb6834"}
GAUGE_LABELS = {"single": "Single gauge", "lumped": "Gauge average", "voronoi": "Voronoi-weighted", "disaggregated": "All gauges"}
OBSERVED = "#52514e"
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
POS, NEG = "#2a78d6", "#e34948"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def _layout(fig: go.Figure, title: str = "", height: int = 380, **kw) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15, color=INK)) if title else None,
        height=height,
        margin=dict(l=8, r=8, t=72 if title else 16, b=8),
        font=dict(family=FONT, size=12, color=INK_2),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, font=dict(size=12)),
        hoverlabel=dict(font=dict(family=FONT)),
        **kw,
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont=dict(color=MUTED))
    return fig


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


# --------------------------------------------------------------------------
def fig_metric(summary: pd.DataFrame, metric: str = "NSE", period: str = "test", models: Optional[Sequence[str]] = None) -> go.Figure:
    s = summary.xs(period, level="period")
    order = [m for m in LABELS if m in s.index and (models is None or m in models)]
    vals = s.loc[order, metric]
    higher_better = metric in ("NSE", "KGE", "Coverage90")
    fmt = (lambda v: f"{v:.3f}") if metric in ("NSE", "KGE", "CRPS", "Coverage90") else (lambda v: f"{v:.2f}")
    fig = go.Figure(go.Bar(
        y=[LABELS[k] for k in order], x=vals.values, orientation="h",
        marker=dict(color=[COLORS.get(k, POS) for k in order], line=dict(width=0)),
        text=[fmt(v) for v in vals.values], textposition="outside", cliponaxis=False, textfont=dict(color=INK),
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))
    note = "higher is better" if higher_better else ("closer to 0 is better" if metric in ("PBIAS", "PeakError") else "lower is better")
    _layout(fig, f"{metric} by model, {period} period ({note})", height=max(280, 44 * len(order) + 90), bargap=0.35)
    fig.update_yaxes(autorange="reversed")
    if metric in ("NSE", "KGE"):
        lo = min(0.0, float(np.nanmin(vals.values)) - 0.05)
        fig.update_xaxes(range=[lo, 1.08])
    return fig


def fig_hydrograph(predictions: dict, models: Sequence[str], period: str = "test", flow_unit: str = "",
                   band_model: Optional[str] = None) -> go.Figure:
    """Observed vs predicted; ``band_model`` adds its 90 % prediction interval."""
    first = predictions[models[0]]
    sel = slice(None) if period == "all" else (first["period"] == period)
    fig = go.Figure()
    if band_model and band_model in predictions and "q05" in predictions[band_model]:
        p = predictions[band_model][sel]
        c = COLORS.get(band_model, POS)
        fig.add_trace(go.Scatter(x=p.index, y=p["q95"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=p.index, y=p["q05"], mode="lines", line=dict(width=0), fill="tonexty",
                                 fillcolor=_rgba(c, 0.18), name=f"90% interval ({LABELS[band_model]})", hoverinfo="skip"))
    obs = first[sel]["observed"]
    fig.add_trace(go.Scatter(x=obs.index, y=obs, name="Observed", mode="lines", line=dict(color=OBSERVED, width=2.5),
                             hovertemplate="%{y:.2f}<extra>Observed</extra>"))
    for m in models:
        p = predictions[m][sel]
        fig.add_trace(go.Scatter(x=p.index, y=p["mean"], name=LABELS[m], mode="lines",
                                 line=dict(color=COLORS.get(m, POS), width=1.6),
                                 hovertemplate="%{y:.2f}<extra>" + LABELS[m] + "</extra>"))
    unit = f" ({flow_unit})" if flow_unit else ""
    _layout(fig, f"Observed and predicted streamflow, {period} period", height=440, hovermode="x unified")
    fig.update_yaxes(title=dict(text="Flow" + unit, font=dict(color=MUTED)), rangemode="tozero")
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.06))
    return fig


def fig_gauge_value(gauge_value: pd.DataFrame, metric: str = "NSE_gain", period: str = "test") -> go.Figure:
    df = gauge_value[gauge_value["period"] == period]
    order = [g for g in GAUGE_LABELS if g in set(df["gauge_input"])]
    df = df.set_index("gauge_input").loc[order]
    label = {"NSE_gain": "NSE gain", "KGE_gain": "KGE gain", "CRPS_reduction_pct": "CRPS reduction (%)"}[metric]
    fig = go.Figure(go.Bar(
        x=[GAUGE_LABELS[g] for g in order], y=df[metric].values,
        marker=dict(color=[GAUGE_COLORS[g] for g in order], line=dict(width=0)),
        text=[f"{v:+.3f}" if metric != "CRPS_reduction_pct" else f"{v:+.1f}%" for v in df[metric].values],
        textposition="outside", cliponaxis=False, textfont=dict(color=INK),
        hovertemplate="%{x}: %{y:.3f}<extra></extra>",
    ))
    _layout(fig, f"What local gauges add to the fine-tuned model: {label}, {period} period", height=360, bargap=0.45)
    fig.update_yaxes(zeroline=True, zerolinecolor=AXIS)
    return fig


def fig_importance(importance: pd.DataFrame) -> go.Figure:
    df = importance.sort_values("importance")
    fig = go.Figure(go.Bar(
        y=df["station"], x=df["importance"], orientation="h", marker=dict(color=SEQ[3], line=dict(width=0)),
        error_x=dict(type="data", array=df["importance_std"], color=INK_2, thickness=1.2),
        hovertemplate="%{y}: NSE drop %{x:.3f}<extra></extra>",
    ))
    _layout(fig, "Gauge importance: drop in validation NSE when a gauge is shuffled", height=max(260, 34 * len(df) + 80), bargap=0.35)
    fig.update_xaxes(title=dict(text="NSE drop", font=dict(color=MUTED)))
    return fig


def fig_ablation(ablation: pd.DataFrame, period: str = "test") -> go.Figure:
    col = f"{period}_nse_change"
    df = ablation.assign(loss=-ablation[col]).sort_values("loss")
    fig = go.Figure(go.Bar(
        y=df["station"], x=df["loss"], orientation="h",
        marker=dict(color=[POS if v >= 0 else NEG for v in df["loss"]], line=dict(width=0)),
        hovertemplate="%{y}: NSE lost without this gauge %{x:.3f}<extra></extra>",
    ))
    _layout(fig, f"Station ablation: NSE lost when each gauge is removed ({period})", height=max(260, 34 * len(df) + 80), bargap=0.35)
    fig.update_xaxes(title=dict(text="NSE lost (negative = model improved without it)", font=dict(color=MUTED)), zeroline=True, zerolinecolor=AXIS)
    return fig


def fig_contributions(contrib: pd.DataFrame) -> go.Figure:
    df = contrib.sort_values("peak_flow_share")
    fig = go.Figure()
    fig.add_trace(go.Bar(y=df.index, x=df["area_share"] * 100, orientation="h", name="Share of basin area",
                         marker=dict(color=SEQ[1], line=dict(width=0))))
    fig.add_trace(go.Bar(y=df.index, x=df["peak_flow_share"] * 100, orientation="h", name="Share of flood flow (top 2% days)",
                         marker=dict(color=SEQ[4], line=dict(width=0))))
    _layout(fig, "Where flood flow comes from, by sub-basin", height=max(260, 46 * len(df) + 100), barmode="group", bargap=0.3)
    fig.update_xaxes(title=dict(text="%", font=dict(color=MUTED)))
    return fig


def fig_forecast(issued: dict, history_days: int = 30) -> go.Figure:
    fc = pd.DataFrame(issued["forecast"]).set_index("date")
    fc.index = pd.to_datetime(fc.index)
    obs = pd.Series(issued.get("recent_observed", {}), dtype=float)
    obs.index = pd.to_datetime(obs.index)
    fig = go.Figure()
    for lo, hi, a, name in (("q05", "q95", 0.15, "90% range"), ("q25", "q75", 0.3, "50% range")):
        if lo in fc and hi in fc:
            fig.add_trace(go.Scatter(x=fc.index, y=fc[hi], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=fc.index, y=fc[lo], mode="lines", line=dict(width=0), fill="tonexty",
                                     fillcolor=_rgba(POS, a), name=name, hoverinfo="skip"))
    if len(obs):
        fig.add_trace(go.Scatter(x=obs.index[-history_days:], y=obs.iloc[-history_days:], name="Observed", mode="lines+markers",
                                 line=dict(color=OBSERVED, width=2), marker=dict(size=4)))
    col = "q50" if "q50" in fc else "mean"
    fig.add_trace(go.Scatter(x=fc.index, y=fc[col], name="Forecast (median)", mode="lines+markers",
                             line=dict(color=POS, width=2), marker=dict(size=7)))
    _layout(fig, f"{issued['basin']}: forecast issued {issued['issued']}", height=400, hovermode="x unified")
    fig.update_yaxes(title=dict(text=f"Flow ({issued.get('unit', '')})", font=dict(color=MUTED)), rangemode="tozero")
    return fig


def fig_skill_by_lead(skill: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=skill["lead_day"], y=skill["NSE"], name="Forecast model", mode="lines+markers",
                             line=dict(color=POS, width=2), marker=dict(size=8)))
    fig.add_trace(go.Scatter(x=skill["lead_day"], y=skill["NSE_persistence"], name="Persistence (today's flow)", mode="lines+markers",
                             line=dict(color=NEG, width=2, dash="dot"), marker=dict(size=8)))
    _layout(fig, "Hindcast skill by lead time (observed weather used as forecast)", height=340)
    fig.update_xaxes(title=dict(text="Lead time (days)", font=dict(color=MUTED)), dtick=1)
    fig.update_yaxes(title=dict(text="NSE", font=dict(color=MUTED)))
    return fig


def _rings(geom):
    polys = getattr(geom, "geoms", [geom])
    for p in polys:
        if hasattr(p, "exterior"):
            x, y = p.exterior.xy
            yield list(x), list(y)


def fig_map(stations: pd.DataFrame, voronoi, outlet: Optional[tuple] = None, height: int = 460,
            basemap: bool = False) -> go.Figure:
    """Gauges, Voronoi cells shaded by weight, and the basin outline.

    ``basemap=False`` (default) draws on plain lon/lat axes, which works
    offline and in the HTML report. ``basemap=True`` adds street tiles
    (needs internet access in the viewer's browser).
    """
    Trace = go.Scattermap if basemap else go.Scatter

    def xy(x, y):
        return dict(lon=x, lat=y) if basemap else dict(x=x, y=y)

    fig = go.Figure()
    w = voronoi.weights
    bins = np.linspace(0, max(w.max(), 1e-9), len(SEQ) + 1)
    for sid, geom in voronoi.cells_lonlat.items():
        if geom.is_empty:
            continue
        k = min(int(np.searchsorted(bins, w[sid], side="right")) - 1, len(SEQ) - 1)
        for x, y in _rings(geom):
            kw = dict(mode="lines", fill="toself", fillcolor=SEQ[max(k, 0)], line=dict(color="#ffffff", width=2),
                      opacity=0.6 if basemap else 1.0, showlegend=False, hoverinfo="text",
                      text=f"{sid}: weight {w[sid]:.3f}")
            if not basemap:
                kw["hoveron"] = "fills"
            fig.add_trace(Trace(**xy(x, y), **kw))
    for x, y in _rings(voronoi.region_lonlat):
        fig.add_trace(Trace(**xy(x, y), mode="lines", line=dict(color=INK, width=2),
                            name="Basin boundary" if voronoi.used_boundary else "Gauge region (no boundary)",
                            hoverinfo="skip"))
    fig.add_trace(Trace(
        **xy(stations["lon"], stations["lat"]), mode="markers+text",
        marker=dict(size=10, color=INK), text=list(stations.index), textposition="top right",
        textfont=dict(color=INK, size=12), name="Rain gauges",
        customdata=np.column_stack([w.reindex(stations.index).values]),
        hovertemplate="%{text}<br>Voronoi weight %{customdata[0]:.3f}<extra></extra>",
    ))
    if outlet is not None:
        fig.add_trace(Trace(**xy([outlet[1]], [outlet[0]]), mode="markers",
                            marker=dict(size=14, color=NEG, symbol="circle"), name="Outlet",
                            hovertemplate="Outlet<extra></extra>"))
    minx, miny, maxx, maxy = voronoi.region_lonlat.bounds
    legend = dict(orientation="h", yanchor="bottom", y=0.01, xanchor="left", x=0.01,
                  bgcolor="rgba(255,255,255,0.85)", font=dict(color=INK))
    if basemap:
        span = max(maxx - minx, (maxy - miny) * 1.3, 1e-3)
        zoom = float(np.clip(np.log2(360 / span) - 0.6, 3, 14))
        fig.update_layout(
            map=dict(style="carto-positron", center=dict(lon=(minx + maxx) / 2, lat=(miny + maxy) / 2), zoom=zoom),
            height=height, margin=dict(l=0, r=0, t=0, b=0), legend=legend, font=dict(family=FONT),
        )
        return fig
    _layout(fig, "", height=height)
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.06, xanchor="left", x=0))
    # keep true shape: 1 degree of longitude is shorter than 1 degree of latitude
    ratio = 1 / np.cos(np.radians((miny + maxy) / 2))
    fig.update_yaxes(scaleanchor="x", scaleratio=ratio, showgrid=False, ticksuffix="°")
    fig.update_xaxes(showgrid=False, ticksuffix="°")
    return fig



def _table(df: pd.DataFrame, floatfmt="{:.3f}") -> str:  # wrapped for horizontal scroll
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else floatfmt.format(v)
        return html.escape(str(v))

    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in [df.index.name or ""] + list(df.columns))
    body = "".join(
        "<tr>" + f"<th>{html.escape(str(i))}</th>" + "".join(f"<td>{fmt(v)}</td>" for v in row) + "</tr>"
        for i, row in zip(df.index, df.values)
    )
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def fig_subbasin_map(subs, stations: pd.DataFrame, contrib: Optional[pd.DataFrame] = None, height: int = 440) -> go.Figure:
    """Sub-basins shaded by their share of flood flow; gauges as points."""
    fig = go.Figure()
    share = contrib["peak_flow_share"] if contrib is not None and "peak_flow_share" in contrib else None
    vmax = float(share.max()) if share is not None else 1.0
    for s in subs.subbasins:
        if s.geometry is None:
            continue
        v = float(share[s.sub_id]) if share is not None else 0.5
        k = min(int(v / max(vmax, 1e-9) * (len(SEQ) - 1)), len(SEQ) - 1)
        for x, y in _rings(s.geometry):
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", fill="toself", fillcolor=SEQ[k],
                                     line=dict(color="#ffffff", width=2), showlegend=False, hoveron="fills",
                                     hoverinfo="text", text=f"{s.sub_id}: {v * 100:.0f}% of flood flow, {s.area_km2:.0f} km²"))
        c = s.geometry.representative_point()
        fig.add_annotation(x=c.x, y=c.y, text=s.sub_id, showarrow=False, font=dict(color="#ffffff" if k >= 3 else INK, size=13))
    fig.add_trace(go.Scatter(x=stations["lon"], y=stations["lat"], mode="markers", marker=dict(size=9, color=INK),
                             name="Rain gauges", text=list(stations.index), hovertemplate="%{text}<extra></extra>"))
    _layout(fig, "", height=height)
    lat = float(stations["lat"].mean())
    fig.update_yaxes(scaleanchor="x", scaleratio=1 / np.cos(np.radians(lat)), showgrid=False, ticksuffix="°")
    fig.update_xaxes(showgrid=False, ticksuffix="°")
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.06, xanchor="left", x=0))
    return fig


def summary_table(result, period: str = "test") -> pd.DataFrame:
    s = result.summary.xs(period, level="period")
    cols = [c for c in ["NSE", "KGE", "RMSE", "PBIAS", "PeakError", "CRPS", "Coverage90"] if c in s.columns]
    s = s.loc[[m for m in LABELS if m in s.index], cols]
    s.index = [LABELS[k] for k in s.index]
    s.index.name = "Model"
    return s


def write_report(result, data, path, routed=None, forecast_skill: Optional[pd.DataFrame] = None) -> Path:
    """Single-file HTML report that works offline (plotly.js is embedded)."""
    path = Path(path)
    figs = [("Model comparison", fig_metric(result.summary, "NSE", "test"))]
    if result.gauge_value is not None:
        figs.append(("What the gauges add", fig_gauge_value(result.gauge_value, "NSE_gain", "test")))
        if "CRPS_reduction_pct" in result.gauge_value and result.gauge_value["CRPS_reduction_pct"].notna().any():
            figs.append(("", fig_gauge_value(result.gauge_value, "CRPS_reduction_pct", "test")))
    best = result.best_model("test")
    show = [m for m in ("finetuned", best, "local") if m in result.predictions]
    show = list(dict.fromkeys(show))
    flows = {m: result.flow(m) for m in result.models}
    figs.append(("Hydrograph", fig_hydrograph(flows, show, "test", result.flow_unit, band_model=best)))
    figs.append(("Gauges and Voronoi weights", fig_map(data.stations, result.voronoi, data.outlet)))
    if result.importance is not None:
        figs.append(("Gauge ranking", fig_importance(result.importance)))
    if result.ablation is not None:
        figs.append(("", fig_ablation(result.ablation)))
    if routed is not None:
        figs.append(("Sub-basins", fig_contributions(routed.contributions)))
    if forecast_skill is not None:
        figs.append(("Forecast skill", fig_skill_by_lead(forecast_skill)))

    parts = []
    for title, f in figs:
        h = f"<h2>{html.escape(title)}</h2>" if title else ""
        parts.append(h + f'<div class="card">{f.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False})}</div>')

    s = result.summary
    gain = ""
    if result.gauge_value is not None:
        gv = result.gauge_value[result.gauge_value["period"] == "test"].set_index("gauge_input")
        top = gv["NSE_gain"].idxmax()
        gain = (f" Adding local gauges changed test NSE by up to <b>{gv.loc[top, 'NSE_gain']:+.3f}</b> "
                f"({html.escape(GAUGE_LABELS[top].lower())}).")
    per = "".join(f"<li><b>{k.title()}</b>: {a.date()} to {b.date()}</li>" for k, (a, b) in result.periods.items())
    notes = "".join(f"<li>{html.escape(n)}</li>" for n in result.notes) or "<li>None</li>"
    bi = result.base_info
    routed_html = ""
    if routed is not None:
        routed_html = f'<h2>Sub-basin models</h2><div class="card">{_table(_routed_table(routed))}</div><div class="card">{_table(routed.contributions)}</div>'
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<script>{get_plotlyjs()}</script>
<title>multigauge report - {html.escape(result.basin_name)}</title>
<style>
body{{font-family:{FONT};color:{INK};background:#f9f9f7;margin:0}}
main{{max-width:1000px;margin:0 auto;padding:24px 16px 48px}}
h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:18px;margin:32px 0 8px}}
.sub{{color:{INK_2};margin:0 0 20px}} .card{{background:#fcfcfb;border:1px solid rgba(11,11,11,.1);border-radius:10px;padding:12px;margin:12px 0}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:14px}}
th,td{{padding:6px 10px;border-bottom:1px solid {GRID};text-align:right}} th:first-child{{text-align:left}}
thead th{{color:{INK_2};font-weight:600}} .scroll{{overflow-x:auto}} .kpi{{font-size:15px;line-height:1.5}} footer{{color:{MUTED};font-size:12px;margin-top:40px}}
</style></head><body><main>
<h1>{html.escape(result.basin_name)}</h1>
<p class="sub">Value of local rain gauges for pre-trained deep-learning streamflow models · generated {datetime.now():%Y-%m-%d %H:%M} with multigauge {__version__}</p>
<div class="card kpi">Best test-period model: <b>{html.escape(LABELS[best])}</b> (NSE {s.loc[(best, 'test'), 'NSE']:.2f}).{gain}
{len(data.station_ids)} gauges, {len(data.dates):,} days, {result.area_km2:,.0f} km².</div>
<h2>Test-period performance</h2><div class="card">{_table(summary_table(result))}</div>
{''.join(parts)}
{routed_html}
<h2>Station table</h2><div class="card">{_table(result.station_summary)}</div>
<h2>Run details</h2><div class="card"><ul>{per}</ul>
<p>Pre-trained model: {html.escape(str(bi.get('description', bi.get('source', 'see run_info.json'))))} Trained on {bi.get('trained_on', bi.get('n_basins', '?'))} basins; lookback {bi.get('lookback')} days; {html.escape(str(bi.get('head')))} output.</p>
<p>Single-gauge model used {html.escape(str(result.single_station))} ({html.escape(result.single_station_reason)}). Seeds per model: {result.config.n_seeds}.</p>
<p>Notes:</p><ul>{notes}</ul></div>
<footer>Produced with multigauge (MIT licence). NSE = Nash-Sutcliffe efficiency; KGE = Kling-Gupta efficiency; CRPS = continuous ranked probability score (mm/day, lower is better); Coverage90 = share of observations inside the 90% prediction interval (ideal 0.90); PeakError = mean % error on days above the 98th percentile of observed flow.</footer>
</main></body></html>"""
    path.write_text(doc, encoding="utf-8")
    return path


def _routed_table(routed) -> pd.DataFrame:
    s = routed.summary.xs("test", level="period")[["NSE", "KGE", "PeakError"]]
    s.index = [LABELS.get(k, k) for k in s.index]
    s.index.name = "Model (test period)"
    return s
