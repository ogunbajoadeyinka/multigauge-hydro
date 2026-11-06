"""Static PNG of the national gauge-value map (for the report and paper).

    python scripts/national_map_png.py validation/national/national_summary.csv states-10m.json docs/national_map.png

``states-10m.json`` is the U.S. states TopoJSON from the ``us-atlas`` npm package
(``npm pack us-atlas@3.0.1``). The interactive map in the dashboard needs nothing extra.
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize


def albers(lon, lat, lon0=-96.0, lat0=23.0, p1=29.5, p2=45.5):
    """USGS Albers equal-area conic for the conterminous U.S. (unit sphere)."""
    lon, lat = np.radians(np.asarray(lon, float)), np.radians(np.asarray(lat, float))
    l0, f0, f1, f2 = map(np.radians, (lon0, lat0, p1, p2))
    n = (np.sin(f1) + np.sin(f2)) / 2
    C = np.cos(f1) ** 2 + 2 * n * np.sin(f1)
    r0 = np.sqrt(C - 2 * n * np.sin(f0)) / n
    r = np.sqrt(C - 2 * n * np.sin(lat)) / n
    t = n * (lon - l0)
    return r * np.sin(t), r0 - r * np.cos(t)


def topo_lines(path, skip=("02", "15", "72", "78", "60", "66", "69")):
    """Decode TopoJSON arcs of the states object into lon/lat polylines."""
    tj = json.load(open(path))
    sx, sy = tj["transform"]["scale"]
    tx, ty = tj["transform"]["translate"]
    arcs = []
    for a in tj["arcs"]:
        q = np.cumsum(np.array(a, float), axis=0)
        arcs.append(np.column_stack([q[:, 0] * sx + tx, q[:, 1] * sy + ty]))
    used = set()
    for g in tj["objects"]["states"]["geometries"]:
        if str(g.get("id")) in skip:
            continue
        polys = g["arcs"] if g["type"] == "Polygon" else [r for p in g["arcs"] for r in p]
        for ring in polys:
            for i in ring:
                used.add(i if i >= 0 else ~i)
    return [arcs[i] for i in sorted(used)]


def main(summary_csv, topo, out):
    s = pd.read_csv(summary_csv, dtype={"site": str})
    fig, ax = plt.subplots(figsize=(11, 6.4), dpi=200)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for line in topo_lines(topo):
        x, y = albers(line[:, 0], line[:, 1])
        ax.plot(x, y, color="#c9c8c4", lw=0.5, zorder=1)
    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
    norm = Normalize(-0.1, 0.1)
    x, y = albers(s["lon"], s["lat"])
    real = s["verdict"].isin(["helps", "hurts"]).values
    size = 30 + 14 * np.sqrt(s["n_gauges"].values)
    c = cmap(norm(s["NSE_gain"].clip(-0.1, 0.1).values))
    ax.scatter(x[~real], y[~real], s=size[~real], facecolors="#fcfcfb", edgecolors=c[~real], linewidths=2, zorder=3)
    ax.scatter(x[real], y[real], s=size[real], c=c[real], edgecolors="#fcfcfb", linewidths=1.5, zorder=4)
    ax.set_aspect("equal")
    ax.axis("off")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.01, ticks=[-0.1, -0.05, 0, 0.05, 0.1])
    cb.set_label("Change in test NSE when gauges are added", color="#3a3a38")
    cb.ax.set_yticklabels(["−0.10 or less", "−0.05", "0", "+0.05", "+0.10 or more"], color="#3a3a38")
    cb.outline.set_visible(False)
    h = [plt.Line2D([], [], ls="", marker="o", ms=9, mfc="#2a78d6", mec="#fcfcfb", label="Change larger than training noise"),
         plt.Line2D([], [], ls="", marker="o", ms=9, mfc="#fcfcfb", mec="#2a78d6", mew=2, label="Within noise or negligible")]
    ax.legend(handles=h, loc="lower left", frameon=False, fontsize=9, labelcolor="#3a3a38")
    n_help = int((s["verdict"] == "helps").sum())
    ax.set_title(f"Where NOAA rain gauges improve the AI streamflow model ({len(s)} USGS basins)",
                 loc="left", fontsize=13, color="#1f1f1e", pad=22)
    ax.text(0, 1.01, f"Marker size: number of gauges. Gauges clearly helped in {n_help} basins; test period Oct 2017 to Sep 2023.",
            transform=ax.transAxes, fontsize=9, color="#6b6a66", va="bottom")
    fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main(*sys.argv[1:4])
