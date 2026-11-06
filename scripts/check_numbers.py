"""Recompute every number quoted in the HESS manuscript from the result files.

    python scripts/check_numbers.py            (from the repository folder)

Each line: [claim number] what the manuscript says -> the value computed here,
and the file it comes from. Compare them by eye with the manuscript.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "src"))
V = R / "validation"
n = 0


def show(text, value, source):
    global n
    n += 1
    print(f"[{n:2d}] {text}: {value}\n      from {source}")


# --- pre-trained model (Section 2.4)
m = pd.read_csv(R / "src/multigauge/pretrained/camels-us/test_metrics.csv")
show("Pre-training basins", m.shape[0], "src/multigauge/pretrained/camels-us/test_metrics.csv (one row per basin)")
show("Pre-trained model, median test NSE on CAMELS (2000-2010)", round(m["NSE"].median(), 2), "same file, NSE column")
cov = [c for c in m.columns if c.lower().startswith("coverage")]
if cov:
    show("Pre-trained model, median 90 % interval coverage", round(m[cov[0]].median(), 2), f"same file, {cov[0]} column")

# --- national survey (Sections 2.1, 3.1, 3.2)
f = "validation/national/national_summary.csv"
s = pd.read_csv(V / "national/national_summary.csv", dtype={"site": str, "huc_02": str})
g = s["NSE_ft_disaggregated"] - s["NSE_finetuned"]
show("Basins in the national survey", len(s), f + " (one row per basin)")
show("Hydrologic regions covered", s["region"].nunique(), f + ", region column")
show("Basins per region", s["region"].value_counts().to_dict(), f + ", region column")
show("Basins that are in the pre-training set", int(s["in_pretraining_set"].sum()), f + ", in_pretraining_set column")
show("Basins added by the 25 km fill-in search", int((s["gauge_search_km"] == 25).sum()), f + ", gauge_search_km column")
show("Basins where all gauges raised NSE", f"{int((g > 0).sum())} of {len(s)} = {(g > 0).mean():.0%}",
     f + ": NSE_ft_disaggregated minus NSE_finetuned")
show("Median / mean gain, all gauges", f"{g.median():+.3f} / {g.mean():+.3f}", "same difference")
show("Wilcoxon signed-rank p-value", f"{stats.wilcoxon(g).pvalue:.0e}", "same difference")
ga = s["NSE_ft_lumped"] - s["NSE_finetuned"]
show("Gauge average: median gain, share positive", f"{ga.median():+.3f}, {(ga > 0).mean():.0%}", f + ": NSE_ft_lumped minus NSE_finetuned")
show("Per-basin verdicts", s["verdict"].value_counts().to_dict(), f + ", verdict column")
for c, label in [("NSE_pretrained", "pre-trained"), ("NSE_gr4j", "GR4J"), ("NSE_finetuned", "fine-tuned"),
                 ("NSE_ft_disaggregated", "fine-tuned + all gauges")]:
    show(f"Median test NSE, {label}", round(s[c].median(), 3), f + f", {c} column")
show("Fine-tuned beats GR4J (without / with gauges)",
     f"{(s.NSE_finetuned > s.NSE_gr4j).mean():.0%} / {(s.NSE_ft_disaggregated > s.NSE_gr4j).mean():.0%}", f)
cls = pd.cut(s["frac_snow"], [-1, 0.1, 0.3, 1], labels=["rain-dominated", "mixed", "snow-dominated"])
for k in ["rain-dominated", "mixed", "snow-dominated"]:
    gk = g[cls == k]
    show(f"{k}: basins, median gain, share positive, median NSE without gauges",
         f"{len(gk)}, {gk.median():+.3f}, {(gk > 0).mean():.0%}, {s.loc[cls == k, 'NSE_finetuned'].median():.2f}",
         f + ", frac_snow column (<0.1, 0.1-0.3, >0.3)")
for c in ["frac_snow", "NSE_finetuned", "area_km2", "n_gauges", "gauges_per_1000km2", "aridity"]:
    r = stats.spearmanr(s[c], g)
    show(f"Spearman correlation of gain with {c}", f"{r.statistic:+.2f} (p = {r.pvalue:.0e})", f)
reg = g.groupby(s["region"]).median().sort_values()
show("Regional median gains (lowest and highest)", reg.round(3).to_dict(), f)
fi = s["gauge_search_km"] == 25
show("Fill-in vs standard search: share positive", f"{(g[fi] > 0).mean():.0%} vs {(g[~fi] > 0).mean():.0%}", f)
show("Fill-in vs standard search: median gain", f"{g[fi].median():+.3f} vs {g[~fi].median():+.3f}", f)
ip = s["in_pretraining_set"].astype(bool)
show("Outside vs inside pre-training set: share positive, median gain",
     f"{(g[~ip] > 0).mean():.0%} {g[~ip].median():+.3f} vs {(g[ip] > 0).mean():.0%} {g[ip].median():+.3f}", f)
show("Basins with a gauge moved one day", f"{(s['share_day_shifted'] > 0).mean():.0%}", f + ", share_day_shifted column")

# --- detailed basins (Sections 3.3, 3.4)
for site in ["05453100", "02347500", "08167000", "01643000"]:
    gv = pd.read_csv(V / site / "results/gauge_value.csv")
    t = gv[gv["period"] == "test"].set_index("gauge_input")
    b = t["NSE_gain"].idxmax()
    show(f"Detailed basin {site}: best gauge gain (input, CRPS reduction %)",
         f"{t.loc[b, 'NSE_gain']:+.3f} ({b}, {t.loc[b, 'CRPS_reduction_pct']:.1f} %)", f"validation/{site}/results/gauge_value.csv")

# --- sub-basins (Section 3.4)
ss = pd.read_csv(V / "subbasins/subbasin_summary.csv", dtype={"site": str})
for _, r in ss.iterrows():
    show(f"Sub-basins {r['basin']}: routed / routed + gauges / gauge gain +- s.d.",
         f"{r['NSE_routed']:.3f} / {r['NSE_routed_gauges']:.3f} / {r['gauge_gain_routed']:+.3f} +- {r['gauge_gain_routed_sd']:.3f}",
         "validation/subbasins/subbasin_summary.csv")
c = pd.read_csv(V / "subbasins/02347500/contributions.csv", index_col=0)
show("Flint: peak-flow share of the ungauged sub-basin", f"{c.loc[c['n_gauges'] == 0, 'peak_flow_share'].round(2).to_dict()}",
     "validation/subbasins/02347500/contributions.csv")
from multigauge.results import load_saved_study  # noqa: E402

for site, label in [("05453100", "Iowa"), ("02347500", "Flint")]:
    t = load_saved_study(V / site / "results").summary.loc[("finetuned", "test"), "NSE"]
    show(f"{label}: whole-basin fine-tuned NSE (3-seed ensemble)", round(t, 3), f"validation/{site}/results (summary table)")
lm = s.set_index("site").loc["05585000"]
show("La Moine: whole-basin fine-tuned NSE without / with gauges", f"{lm.NSE_finetuned:.3f} / {lm.NSE_ft_disaggregated:.3f}", f)
