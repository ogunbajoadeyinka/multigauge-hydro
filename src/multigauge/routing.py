"""Phase 2: sub-basin LSTMs with routing for large basins.

A large basin is split into sub-basins. The pre-trained LSTM (shared
weights) predicts local runoff for every sub-basin from that sub-basin's
own gridded forcings and, if present, the rain gauges inside it. Local
runoff is then routed to the outlet in one of two ways:

``gamma``  each sub-basin reaches the outlet through a gamma-distribution
           unit hydrograph whose mean travel time and shape are learned,
           initialised from flow distance (a differentiable version of the
           routing used by Yu et al., 2024);
``gnn``    a graph attention network passes information from upstream to
           downstream sub-basins along the river network and predicts the
           outlet flow (after Mosaffa et al., 2026).

Everything is trained end to end on outlet discharge, starting from the
pre-trained LSTM. Per-sub-basin results show where in the basin flow is
generated and where gauges add the most, which supports decisions about
where new gauges would help.
"""
from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from shapely.geometry import Point, shape
from shapely.geometry.base import BaseGeometry
from torch import nn

from .features import DYNAMIC, check_forcings, static_attributes
from .metrics import evaluate, nse
from .nn import StreamflowModel, cmal_mean
from .spatial import LocalProjection


# =========================================================================
# data
# =========================================================================
@dataclass
class Subbasin:
    sub_id: str
    geometry: Optional[BaseGeometry]   # lon/lat polygon (for maps and gauge assignment)
    area_km2: float
    forcings: pd.DataFrame             # gridded forcings for this sub-basin
    downstream: Optional[str]          # id of the next sub-basin downstream (None = outlet)
    distance_km: float                 # flow distance from sub-basin centroid to the basin outlet
    lat: float = float("nan")

    def __post_init__(self):
        self.forcings = check_forcings(self.forcings, f"forcings of sub-basin {self.sub_id}")
        if not np.isfinite(self.lat) and self.geometry is not None:
            self.lat = float(self.geometry.centroid.y)


@dataclass
class SubbasinSet:
    subbasins: list

    def __post_init__(self):
        ids = [s.sub_id for s in self.subbasins]
        if len(set(ids)) != len(ids):
            raise ValueError("Sub-basin ids must be unique.")
        for s in self.subbasins:
            if s.downstream is not None and s.downstream not in ids:
                raise ValueError(f"Sub-basin {s.sub_id} drains to unknown sub-basin '{s.downstream}'.")
        outlets = [s for s in self.subbasins if s.downstream is None]
        if len(outlets) != 1:
            raise ValueError("Exactly one sub-basin must drain to the outlet (downstream = null).")
        # cycle check
        down = {s.sub_id: s.downstream for s in self.subbasins}
        for s in ids:
            seen, cur = set(), s
            while cur is not None:
                if cur in seen:
                    raise ValueError("The sub-basin network contains a loop.")
                seen.add(cur)
                cur = down[cur]

    @property
    def ids(self) -> list:
        return [s.sub_id for s in self.subbasins]

    @property
    def total_area(self) -> float:
        return float(sum(s.area_km2 for s in self.subbasins))

    @property
    def outlet_id(self) -> str:
        return next(s.sub_id for s in self.subbasins if s.downstream is None)

    def depth(self) -> int:
        down = {s.sub_id: s.downstream for s in self.subbasins}
        best = 0
        for s in self.ids:
            d, cur = 0, s
            while down[cur] is not None:
                cur, d = down[cur], d + 1
            best = max(best, d)
        return best

    def assign_gauges(self, stations: pd.DataFrame) -> dict:
        """sub_id -> list of gauge ids located inside it."""
        out = {s.sub_id: [] for s in self.subbasins}
        for gid, row in stations.iterrows():
            pt = Point(row["lon"], row["lat"])
            hit = [s.sub_id for s in self.subbasins if s.geometry is not None and s.geometry.covers(pt)]
            if hit:
                out[hit[0]].append(gid)
            else:  # gauge just outside every polygon: nearest sub-basin
                cand = [s for s in self.subbasins if s.geometry is not None]
                if cand:
                    near = min(cand, key=lambda s: s.geometry.distance(pt))
                    out[near.sub_id].append(gid)
        return out

    # --------------------------------------------------------------- io
    @classmethod
    def from_folder(cls, folder) -> "SubbasinSet":
        """``subbasins.geojson`` (properties: id, downstream, optional
        distance_km) plus one ``<id>.csv`` of gridded forcings per sub-basin."""
        from .io import load_extra

        folder = Path(folder)
        gj = json.loads((folder / "subbasins.geojson").read_text())
        feats = gj["features"]
        geoms = {str(f["properties"]["id"]): shape(f["geometry"]) for f in feats}
        props = {str(f["properties"]["id"]): f["properties"] for f in feats}
        out_id = next(k for k, p in props.items() if p.get("downstream") in (None, "", "null"))
        outlet_pt = geoms[out_id].centroid
        subs = []
        for k, g in geoms.items():
            p = props[k]
            c = g.centroid
            proj = LocalProjection(c.x, c.y)
            dist = p.get("distance_km")
            if dist is None:  # straight-line distance as a fallback
                dx, dy = proj.forward(outlet_pt.x, outlet_pt.y)
                dist = float(np.hypot(dx, dy))
            area = p.get("area_km2") or float(proj.geom_to_km(g).area)
            ds = p.get("downstream")
            subs.append(Subbasin(k, g, float(area), load_extra(folder / f"{k}.csv"),
                                 None if ds in (None, "", "null") else str(ds), float(dist)))
        return cls(subs)

    @classmethod
    def from_synthetic(cls, data) -> "SubbasinSet":
        """Sub-basins of a basin made by ``synthetic.generate_basin``."""
        z = data.meta.get("synthetic_zones")
        if z is None:
            raise ValueError("This basin has no synthetic sub-basins.")
        from shapely.ops import unary_union
        from shapely.geometry import box

        cells = data.meta["synthetic_cells_lonlat"]
        n = z["rain"].shape[1]
        dlon = np.diff(np.unique(cells[:, 0].round(6)))
        dlat = np.diff(np.unique(cells[:, 1].round(6)))
        hx = (np.min(dlon[dlon > 1e-6]) if (dlon > 1e-6).any() else 0.04) / 2
        hy = (np.min(dlat[dlat > 1e-6]) if (dlat > 1e-6).any() else 0.04) / 2
        f = data.forcings
        subs = []
        olat, olon = data.outlet
        for k in range(n):
            pts = cells[z["id"] == k]
            geom = unary_union([box(x - hx * 1.02, y - hy * 1.02, x + hx * 1.02, y + hy * 1.02) for x, y in pts]).buffer(0).intersection(data.boundary)
            forc = f.copy()
            forc["prcp"] = np.round(z["rain"][:, k], 2)
            c = geom.centroid
            proj = LocalProjection(c.x, c.y)
            dx, dy = proj.forward(olon, olat)
            subs.append(Subbasin(f"Z{k + 1}", geom, float(z["area"][k]), forc,
                                 None if k == 0 else f"Z{k}", float(np.hypot(dx, dy))))
        return cls(subs)


def auto_zones(boundary: BaseGeometry, outlet: tuple, n: int = 4) -> list:
    """Split a basin polygon into ``n`` distance bands from the outlet, a
    simple sub-basin approximation when no delineation is available.
    Returns lon/lat polygons ordered from the outlet upstream."""
    olat, olon = outlet
    proj = LocalProjection(olon, olat)
    poly = proj.geom_to_km(boundary)
    minx, miny, maxx, maxy = poly.bounds
    far = max(np.hypot(x, y) for x, y in poly.exterior.coords)
    edges = np.quantile(np.linspace(0, far, 200), np.linspace(0, 1, n + 1))
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        ring = Point(0, 0).buffer(b, 64).difference(Point(0, 0).buffer(a, 64))
        piece = poly.intersection(ring)
        if not piece.is_empty and piece.area > 0:
            out.append(proj.geom_to_lonlat(piece))
    return out


# =========================================================================
# network
# =========================================================================
class GammaRouting(nn.Module):
    """Learnable gamma unit hydrograph per sub-basin."""

    def __init__(self, distances_km: Sequence[float], window: int, velocity_km_day: float = 40.0):
        super().__init__()
        t_mean = np.clip(np.asarray(distances_km, float) / velocity_km_day, 0.3, window - 1)
        self.log_mean = nn.Parameter(torch.tensor(np.log(t_mean), dtype=torch.float32))
        self.log_shape = nn.Parameter(torch.full((len(t_mean),), math.log(2.5)))
        self.window = window

    def kernels(self) -> torch.Tensor:
        k = torch.exp(self.log_shape).unsqueeze(1)                 # (N,1)
        theta = torch.exp(self.log_mean).unsqueeze(1) / k            # scale
        t = torch.arange(self.window, dtype=torch.float32).unsqueeze(0) + 0.5
        logpdf = (k - 1) * torch.log(t) - t / theta - torch.lgamma(k) - k * torch.log(theta)
        w = torch.softmax(logpdf, dim=1)                           # normalised over the window
        return w.flip(1)                                           # aligned with time order (oldest first)

    def forward(self, runoff_vol: torch.Tensor) -> torch.Tensor:
        """runoff_vol: (B, N, W) volume-weighted local runoff, oldest first -> (B,)"""
        return (runoff_vol * self.kernels().unsqueeze(0)).sum(dim=(1, 2))


class GATLayer(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.W = nn.Linear(d, d, bias=False)
        self.a_src = nn.Parameter(torch.randn(d) * 0.1)
        self.a_dst = nn.Parameter(torch.randn(d) * 0.1)

    def forward(self, h: torch.Tensor, adj: torch.Tensor):
        """h: (B, N, d); adj: (N, N) with adj[i, j] = 1 if j sends to i (incl. self)."""
        z = self.W(h)
        e = (z @ self.a_dst).unsqueeze(2) + (z @ self.a_src).unsqueeze(1)  # (B, N_dst, N_src)
        e = nn.functional.leaky_relu(e, 0.2).masked_fill(adj.unsqueeze(0) == 0, float("-inf"))
        alpha = torch.softmax(e, dim=2)
        return torch.relu(alpha @ z) + h, alpha


class GNNRouting(nn.Module):
    def __init__(self, n_nodes: int, adj: np.ndarray, outlet_idx: int, window: int, log_area: np.ndarray,
                 d: int = 16, hops: int = 3):
        super().__init__()
        self.register_buffer("adj", torch.tensor(adj, dtype=torch.float32))
        self.register_buffer("log_area", torch.tensor(log_area, dtype=torch.float32))
        self.embed = nn.Linear(window + 1, d)
        self.layers = nn.ModuleList([GATLayer(d) for _ in range(hops)])
        self.readout = nn.Linear(d, 1)
        self.outlet = outlet_idx
        self.last_attention = None

    def forward(self, runoff_vol: torch.Tensor) -> torch.Tensor:
        B, N, W = runoff_vol.shape
        x = torch.cat([runoff_vol, self.log_area.view(1, N, 1).expand(B, N, 1)], dim=2)
        h = torch.relu(self.embed(x))
        for layer in self.layers:
            h, alpha = layer(h, self.adj)
        self.last_attention = alpha.detach().mean(0)
        # residual on the total volume keeps the output on a sensible scale
        return nn.functional.softplus(self.readout(h[:, self.outlet, :]).squeeze(-1) + runoff_vol[:, :, -1].sum(1))


# =========================================================================
# the model
# =========================================================================
@dataclass
class RoutedConfig:
    routing: str = "gamma"   # "gamma" or "gnn"
    window: int = 10         # routing memory (days)
    learning_rate: float = 2e-4
    routing_lr: float = 5e-3
    epochs: int = 15
    patience: int = 4
    batch_size: int = 32
    seed: int = 0


class RoutedModel:
    """Pre-trained LSTM applied to every sub-basin + routing to the outlet."""

    def __init__(self, base: StreamflowModel, subs: SubbasinSet, gauges: Optional[dict] = None,
                 cfg: Optional[RoutedConfig] = None):
        self.cfg = cfg or RoutedConfig()
        torch.manual_seed(self.cfg.seed)
        self.base = base.copy()
        self.subs = subs
        self.gauge_inputs = gauges  # sub_id -> DataFrame of gauge features (or None)
        self.n_extra = 0
        if gauges:
            cols = {tuple(df.columns) for df in gauges.values() if df is not None}
            if len(cols) > 1:
                raise ValueError("All sub-basins must have the same gauge input columns.")
            self.extra_cols = list(next(iter(cols))) if cols else []
            self.n_extra = len(self.extra_cols)
            self.base.net.expand_extra(self.n_extra)
        ids = subs.ids
        areas = np.array([s.area_km2 for s in subs.subbasins])
        self.area_frac = torch.tensor(areas / areas.sum(), dtype=torch.float32)
        if self.cfg.routing == "gamma":
            self.router = GammaRouting([s.distance_km for s in subs.subbasins], self.cfg.window)
        elif self.cfg.routing == "gnn":
            idx = {k: i for i, k in enumerate(ids)}
            adj = np.eye(len(ids))
            for s in subs.subbasins:
                if s.downstream is not None:
                    adj[idx[s.downstream], idx[s.sub_id]] = 1.0
            self.router = GNNRouting(len(ids), adj, idx[subs.outlet_id], self.cfg.window,
                                     np.log10(areas), hops=max(1, min(3, subs.depth())))
        else:
            raise ValueError("routing must be 'gamma' or 'gnn'")
        self.history: list = []

    # ------------------------------------------------------------ arrays
    def _arrays(self, dates: pd.DatetimeIndex):
        sc = self.base.scaler
        xs = []
        for s in self.subs.subbasins:
            f = s.forcings.reindex(dates)
            x = (f.to_numpy() - sc.dyn_mean) / sc.dyn_std
            st = static_attributes(s.forcings, s.area_km2, s.lat).to_numpy()
            st = (st - sc.stat_mean) / sc.stat_std
            parts = [x, np.broadcast_to(st, (len(x), len(st)))]
            if self.n_extra:
                g = self.gauge_inputs.get(s.sub_id)
                e = np.zeros((len(x), self.n_extra)) if g is None else g.reindex(dates)[self.extra_cols].to_numpy()
                parts.append(np.nan_to_num((e - self._e_mean) / self._e_std))
            xs.append(np.nan_to_num(np.concatenate(parts, axis=1)))
        return np.stack(xs).astype(np.float32)  # (N, T, F)

    def _fit_extra_scaler(self, dates, mask):
        if not self.n_extra:
            return
        vals = np.concatenate([g.reindex(dates)[self.extra_cols].to_numpy()[mask]
                               for g in self.gauge_inputs.values() if g is not None])
        self._e_mean = np.nanmean(vals, axis=0)
        self._e_std = np.nanstd(vals, axis=0)
        self._e_std[~np.isfinite(self._e_std) | (self._e_std < 1e-6)] = 1.0

    def _local_runoff(self, xb: torch.Tensor) -> torch.Tensor:
        """xb: (B, N, L, F) -> local runoff in mm/day for the last W days: (B, N, W)."""
        B, N, L, F = xb.shape
        net = self.base.net
        out, _ = net.lstm(xb.reshape(B * N, L, F))
        h = net.dropout(out[:, -self.cfg.window :, :])
        if net.cfg.head == "cmal":
            y = cmal_mean(net.head(h))
        else:
            y = net.head(h).squeeze(-1)
        sc = self.base.scaler
        q = torch.relu(y * sc.q_std + sc.q_mean)
        return q.reshape(B, N, self.cfg.window)

    def _forward(self, xb):
        local = self._local_runoff(xb)                       # mm/day per sub-basin
        vol = local * self.area_frac.view(1, -1, 1)          # share of the outlet's specific discharge
        return self.router(vol), local

    def parameters(self):
        return list(self.base.net.parameters()) + list(self.router.parameters())

    # --------------------------------------------------------------- fit
    def fit(self, q_outlet_mm: pd.Series, train_mask: np.ndarray, val_mask: np.ndarray,
            progress: Optional[Callable[[int, int, float], None]] = None) -> "RoutedModel":
        cfg = self.cfg
        dates = q_outlet_mm.index
        self._fit_extra_scaler(dates, train_mask)
        X = self._arrays(dates)
        y = q_outlet_mm.to_numpy(np.float64)
        L = self.base.cfg.lookback
        t_ok = np.arange(len(y)) >= L - 1
        tr = np.where(train_mask & np.isfinite(y) & t_ok)[0]
        va = np.where(val_mask & np.isfinite(y) & t_ok)[0]
        y_std = float(np.nanstd(y[train_mask])) or 1.0
        rng = np.random.default_rng(cfg.seed)
        opt = torch.optim.Adam([
            {"params": self.base.net.parameters(), "lr": cfg.learning_rate},
            {"params": self.router.parameters(), "lr": cfg.routing_lr},
        ])
        best, best_state, bad = -np.inf, None, 0
        for ep in range(cfg.epochs):
            t0 = time.time()
            self.base.net.train()
            losses = []
            perm = rng.permutation(tr)
            for i in range(0, len(perm), cfg.batch_size):
                days = perm[i : i + cfg.batch_size]
                xb = torch.from_numpy(np.stack([X[:, t - L + 1 : t + 1] for t in days]))
                pred, _ = self._forward(xb)
                yt = torch.from_numpy(y[days].astype(np.float32))
                loss = (((pred - yt) / y_std) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                opt.step()
                losses.append(loss.item())
            score = nse(y[va], self._predict_days(X, va)[0]) if len(va) else -float(np.mean(losses))
            self.history.append({"epoch": ep + 1, "loss": float(np.mean(losses)), "val_nse": score, "seconds": round(time.time() - t0, 1)})
            if progress:
                progress(ep + 1, cfg.epochs, score)
            if score > best + 1e-4:
                best, bad = score, 0
                best_state = (copy.deepcopy(self.base.net.state_dict()), copy.deepcopy(self.router.state_dict()))
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
        if best_state:
            self.base.net.load_state_dict(best_state[0])
            self.router.load_state_dict(best_state[1])
        return self

    def _predict_days(self, X, days):
        L = self.base.cfg.lookback
        self.base.net.eval()
        outs, locs = [], []
        with torch.no_grad():
            for i in range(0, len(days), 128):
                d = days[i : i + 128]
                xb = torch.from_numpy(np.stack([X[:, t - L + 1 : t + 1] for t in d]))
                p, loc = self._forward(xb)
                outs.append(p.numpy())
                locs.append(loc[:, :, -1].numpy())
        if not outs:
            return np.array([]), np.zeros((0, X.shape[0]))
        return np.concatenate(outs), np.concatenate(locs)

    def predict(self, dates: pd.DatetimeIndex) -> tuple[pd.Series, pd.DataFrame]:
        """Outlet specific discharge (mm/day) and local runoff per sub-basin (mm/day)."""
        X = self._arrays(dates)
        L = self.base.cfg.lookback
        days = np.arange(max(L - 1, self.cfg.window - 1), len(dates))
        out, loc = self._predict_days(X, days)
        q = pd.Series(np.nan, index=dates, name="mean")
        q.iloc[days] = out
        local = pd.DataFrame(np.nan, index=dates, columns=self.subs.ids)
        local.iloc[days] = loc
        return q, local

    def travel_times(self) -> Optional[pd.Series]:
        if isinstance(self.router, GammaRouting):
            return pd.Series(torch.exp(self.router.log_mean).detach().numpy(), index=self.subs.ids, name="mean_travel_days")
        return None


# =========================================================================
# the phase-2 study
# =========================================================================
def subbasin_gauge_inputs(data, subs: SubbasinSet) -> tuple[dict, dict]:
    """Average of the gauges inside each sub-basin (gap-filled), plus a
    constant flag telling the model whether the sub-basin has a gauge."""
    from .spatial import voronoi_weights
    from .strategies import gap_fill

    w = voronoi_weights(data.stations, data.boundary).weights
    filled = gap_fill(data.rainfall, w)
    members = subs.assign_gauges(data.stations)
    out = {}
    for s in subs.subbasins:
        g = members[s.sub_id]
        df = pd.DataFrame(index=data.dates)
        df["P_gauges"] = filled[g].mean(axis=1) if g else 0.0
        df["has_gauge"] = 1.0 if g else 0.0
        out[s.sub_id] = df
    return out, members


@dataclass
class RoutedStudyResult:
    summary: pd.DataFrame            # model x period metrics
    predictions: dict                # model -> outlet mm/day Series
    local_runoff: dict               # model -> DataFrame per sub-basin
    contributions: pd.DataFrame      # per sub-basin: area, gauges, share of flow, gauge value
    travel_times: Optional[pd.Series]
    members: dict
    notes: list = field(default_factory=list)


def run_routed_study(
    data, base: StreamflowModel, subs: SubbasinSet, periods: dict,
    routing: str = "gamma", epochs: int = 15, gauge_ablation: bool = True,
    progress: Optional[Callable[[float, str], None]] = None, seed: int = 0,
) -> RoutedStudyResult:
    """Compare lumped vs. sub-basin models (with and without gauges) and
    estimate each sub-basin's flow contribution and gauge value."""
    from .features import flow_to_mm_per_day
    from .transfer import finetune

    dates = data.dates
    q = flow_to_mm_per_day(data.streamflow, data.area_km2, data.flow_unit)
    masks = {k: np.asarray((dates >= a) & (dates <= b)) for k, (a, b) in periods.items()}
    obs = q.to_numpy()
    gin, members = subbasin_gauge_inputs(data, subs)
    gauged = [k for k, v in members.items() if v]
    jobs = 3 + (len(gauged) if gauge_ablation else 0)
    done = [0]

    def cb(label):
        def f(e, n, s):
            if progress:
                progress(min((done[0] + e / n) / jobs, 0.999), f"{label}: epoch {e}/{n} (val NSE {s:.2f})")
        return f

    rcfg = RoutedConfig(routing=routing, epochs=epochs, seed=seed)
    summary, preds, local = [], {}, {}

    # lumped fine-tuned reference
    b = data.to_basin()
    m0 = finetune(base, b, masks["train"], masks["validation"], epochs=epochs, seed=seed, progress=cb("Lumped model"))
    preds["lumped"] = m0.predict(b, quantiles=())["mean"]
    done[0] += 1
    # sub-basins without and with gauges
    for name, g in (("routed", None), ("routed_gauges", gin)):
        rm = RoutedModel(base, subs, g, rcfg).fit(q, masks["train"], masks["validation"], progress=cb(name))
        preds[name], local[name] = rm.predict(dates)
        if name == "routed_gauges":
            final = rm
        done[0] += 1
    for name, p in preds.items():
        for per, m in masks.items():
            summary.append({"model": name, "period": per, **evaluate(obs[m], p.to_numpy()[m])})
    summary = pd.DataFrame(summary).set_index(["model", "period"])

    # contribution of each sub-basin to outlet flow (share of routed volume)
    loc = local["routed_gauges"]
    areas = pd.Series({s.sub_id: s.area_km2 for s in subs.subbasins})
    vol = loc.mul(areas / areas.sum(), axis=1)
    share = (vol.sum() / vol.sum().sum()).rename("flow_share")
    peak_days = q >= q.quantile(0.98)
    peak_share = (vol[peak_days].sum() / vol[peak_days].sum().sum()).rename("peak_flow_share")
    contrib = pd.DataFrame({
        "area_km2": areas, "area_share": areas / areas.sum(),
        "n_gauges": pd.Series({k: len(v) for k, v in members.items()}),
        "gauges": pd.Series({k: ", ".join(v) for k, v in members.items()}),
    }).join(share).join(peak_share)

    # gauge value per sub-basin: remove that sub-basin's gauges and retrain
    if gauge_ablation and gauged:
        base_test = summary.loc[("routed_gauges", "test"), "NSE"]
        vals = {}
        for k in gauged:
            g2 = {kk: (vv.assign(P_gauges=0.0, has_gauge=0.0) if kk == k else vv) for kk, vv in gin.items()}
            rm = RoutedModel(base, subs, g2, rcfg).fit(q, masks["train"], masks["validation"], progress=cb(f"Without gauges in {k}"))
            p, _ = rm.predict(dates)
            vals[k] = base_test - nse(obs[masks["test"]], p.to_numpy()[masks["test"]])
            done[0] += 1
        contrib["gauge_value_test_nse"] = pd.Series(vals)
    # simple siting indicator: high peak-flow share and no gauge
    contrib["priority_for_new_gauge"] = np.where(contrib["n_gauges"] == 0, contrib["peak_flow_share"], 0.0)
    if progress:
        progress(1.0, "Done")
    return RoutedStudyResult(summary, preds, local, contrib, final.travel_times(), members)
