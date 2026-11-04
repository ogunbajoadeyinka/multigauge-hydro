"""Figures and a self-contained HTML report.

The same figure functions feed the web dashboard and the HTML report, so
what a user sees online is exactly what they can download.
"""
from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from . import __version__
from .strategies import LABELS, STRATEGIES

# Colour follows the strategy everywhere (validated categorical order).
COLORS = {
    "single": "#2a78d6",
    "lumped": "#eb6834",
    "voronoi": "#1baf7a",
    "disaggregated": "#eda100",
}
OBSERVED = "#52514e"
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
POS, NEG = "#2a78d6", "#e34948"  # diverging poles
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


def _ordered(cols):
    return [s for s in STRATEGIES if s in cols]


# --------------------------------------------------------------------------
def fig_hydrograph(predictions: pd.DataFrame, period: str = "test", strategies=None, flow_unit: str = "") -> go.Figure:
    df = predictions if period == "all" else predictions[predictions["period"] == period]
    strategies = _ordered(strategies or [c for c in df.columns if c in STRATEGIES])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df.index, y=df["observed"], name="Observed", mode="lines",
                   line=dict(color=OBSERVED, width=2.5),
                   hovertemplate="%{y:.2f}<extra>Observed</extra>")
    )
    for s in strategies:
        fig.add_trace(
            go.Scatter(x=df.index, y=df[s], name=LABELS[s], mode="lines",
                       line=dict(color=COLORS[s], width=1.6),
                       hovertemplate="%{y:.2f}<extra>" + LABELS[s] + "</extra>")
        )
    unit = f" ({flow_unit})" if flow_unit else ""
    _layout(fig, f"Observed and predicted streamflow - {period} period", height=420, hovermode="x unified")
    fig.update_yaxes(title=dict(text="Flow" + unit, font=dict(color=MUTED)), rangemode="tozero")
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.06))
    return fig


def fig_metric(summary: pd.DataFrame, metric: str = "NSE", period: str = "test") -> go.Figure:
    s = summary.xs(period, level="period")
    order = _ordered(s.index)
    vals = s.loc[order, metric]
    err = s.loc[order, "NSE_seed_std"] if metric == "NSE" and s["NSE_seed_std"].notna().any() else None
    fig = go.Figure(
        go.Bar(
            x=[LABELS[k] for k in order], y=vals.values,
            marker=dict(color=[COLORS[k] for k in order], line=dict(width=0)),
            error_y=dict(type="data", array=err.values, color=INK_2, thickness=1.2) if err is not None else None,
            text=[f"{v:.3f}" if metric in ("NSE", "KGE") else f"{v:.1f}" for v in vals.values], textposition="outside", cliponaxis=False,
            textfont=dict(color=INK),
            hovertemplate="%{x}: %{y:.3f}<extra></extra>",
        )
    )
    _layout(fig, f"{metric} by rainfall input - {period} period", height=340, bargap=0.45)
    if metric in ("NSE", "KGE"):
        lo = min(0.0, float(np.nanmin(vals.values)) - 0.1)
        fig.update_yaxes(range=[lo, 1.05])
    return fig


def fig_importance(importance: pd.DataFrame) -> go.Figure:
    df = importance.sort_values("importance")
    fig = go.Figure(
        go.Bar(
            y=df["station"], x=df["importance"], orientation="h",
            marker=dict(color=SEQ[3], line=dict(width=0)),
            error_x=dict(type="data", array=df["importance_std"], color=INK_2, thickness=1.2),
            hovertemplate="%{y}: NSE drop %{x:.3f}<extra></extra>",
        )
    )
    _layout(fig, "Permutation importance (drop in validation NSE when a gauge is shuffled)",
            height=max(260, 34 * len(df) + 80), bargap=0.35)
    fig.update_xaxes(title=dict(text="NSE drop", font=dict(color=MUTED)))
    return fig


def fig_ablation(ablation: pd.DataFrame, period: str = "test") -> go.Figure:
    col = f"{'test' if period == 'test' else 'val'}_nse_change"
    df = ablation.assign(loss=-ablation[col]).sort_values("loss")
    fig = go.Figure(
        go.Bar(
            y=df["station"], x=df["loss"], orientation="h",
            marker=dict(color=[POS if v >= 0 else NEG for v in df["loss"]], line=dict(width=0)),
            hovertemplate="%{y}: NSE lost without this gauge %{x:.3f}<extra></extra>",
        )
    )
    _layout(fig, f"Station ablation - NSE lost when each gauge is removed ({period})",
            height=max(260, 34 * len(df) + 80), bargap=0.35)
    fig.update_xaxes(title=dict(text="NSE lost (negative = model improved without it)", font=dict(color=MUTED)),
                     zeroline=True, zerolinecolor=AXIS)
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


# --------------------------------------------------------------------------
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


def summary_table(result, period: str = "test") -> pd.DataFrame:
    s = result.summary.xs(period, level="period")
    s = s.loc[_ordered(s.index), ["NSE", "KGE", "RMSE", "PBIAS", "PeakError"]]
    s.index = [LABELS[k] for k in s.index]
    s.index.name = "Rainfall input"
    return s


def write_report(result, data, path, flow_unit: str = "") -> Path:
    """Write a single-file HTML report that works offline (plotly.js is embedded)."""
    path = Path(path)
    figs = [
        fig_map(data.stations, result.voronoi, data.outlet),
        fig_metric(result.summary, "NSE", "test"),
        fig_hydrograph(result.predictions, "test", flow_unit=flow_unit),
    ]
    if result.importance is not None:
        figs.append(fig_importance(result.importance))
    if result.ablation is not None:
        figs.append(fig_ablation(result.ablation))
    parts = [f.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False})
             for i, f in enumerate(figs)]

    best = result.best_strategy("test")
    per = "".join(
        f"<li><b>{k.title()}</b>: {a.date()} to {b.date()}</li>" for k, (a, b) in result.periods.items()
    )
    notes = "".join(f"<li>{html.escape(n)}</li>" for n in result.notes) or "<li>None</li>"
    single = (f"<p>Single-gauge model used <b>{html.escape(str(result.single_station))}</b> "
              f"({html.escape(result.single_station_reason)}).</p>") if result.single_station else ""
    cfg = result.config.training
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
thead th{{color:{INK_2};font-weight:600}} .scroll{{overflow-x:auto}} .kpi{{font-size:15px}} footer{{color:{MUTED};font-size:12px;margin-top:40px}}
</style></head><body><main>
<h1>{html.escape(result.basin_name)}</h1>
<p class="sub">Rainfall input comparison for LSTM streamflow prediction · generated {datetime.now():%Y-%m-%d %H:%M} with multigauge {__version__}</p>
<div class="card kpi">Best test-period NSE: <b>{LABELS[best]}</b> ({result.summary.loc[(best, 'test'), 'NSE']:.2f}). {len(data.station_ids)} gauges, {len(data.dates):,} days.</div>
<h2>Test-period performance</h2><div class="card">{_table(summary_table(result))}</div>
<div class="card">{parts[1]}</div>
<h2>Hydrograph</h2><div class="card">{parts[2]}</div>
<h2>Gauges and Voronoi weights</h2><div class="card">{parts[0]}</div>
{single}
{''.join(f'<div class="card">{p}</div>' for p in parts[3:])}
<h2>Station table</h2><div class="card">{_table(result.station_summary)}</div>
<h2>Run details</h2><div class="card"><ul>{per}</ul>
<p>LSTM: lookback {cfg.lookback} days, hidden size {cfg.hidden_size}, up to {cfg.epochs} epochs with early stopping (patience {cfg.patience}), {result.config.n_seeds} seed(s).</p>
<p>Notes:</p><ul>{notes}</ul></div>
<footer>Produced with multigauge-hydro (MIT licence). NSE = Nash-Sutcliffe Efficiency; KGE = Kling-Gupta Efficiency; PBIAS = percent volume bias; PeakError = mean % error on days above the 98th percentile of observed flow.</footer>
</main></body></html>"""
    path.write_text(doc, encoding="utf-8")
    return path
