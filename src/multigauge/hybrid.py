"""Phase 3: a differentiable, physics-guided model (δHBV) and LSTM ensembles.

The HBV conceptual model (Bergström, 1976; Seibert & Vis, 2012) is written
in PyTorch so its parameters can be learned by gradient descent. A small
neural network maps each basin's static attributes to HBV parameters, so
one model can be trained on many basins (parameter regionalisation) and
then fine-tuned on one, following the differentiable-modelling approach
of Feng et al. (2022) and Song et al. (2025). Several HBV components run
in parallel and are averaged.

The hybrid keeps mass balance and physically meaningful states (snow,
soil moisture, groundwater). Li et al. (2025) found that ensembling it
with an LSTM improves predictions, especially in basins outside the
training data; ``ensemble`` implements that combination.
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from .basin import Basin
from .features import STATIC, pet_hargreaves
from .metrics import nse

# parameter ranges (Feng et al., 2022)
RANGES = {
    "BETA": (1.0, 6.0), "FC": (50.0, 1000.0), "K0": (0.05, 0.9), "K1": (0.01, 0.5), "K2": (0.001, 0.2),
    "LP": (0.2, 1.0), "PERC": (0.0, 10.0), "UZL": (0.0, 100.0), "TT": (-2.5, 2.5), "CFMAX": (0.5, 10.0),
    "CFR": (0.0, 0.1), "CWH": (0.0, 0.2), "RT_A": (0.0, 2.9), "RT_B": (0.0, 6.5),
}
PNAMES = list(RANGES)


@dataclass
class HybridConfig:
    n_components: int = 4
    hidden: int = 64
    warmup: int = 365
    predict_last: int = 90
    route_window: int = 15
    learning_rate: float = 3e-3
    batch_size: int = 64
    epochs: int = 30
    patience: int = 6
    samples_per_epoch: Optional[int] = 4000
    seed: int = 0
    device: str = "auto"


def _hbv(P, T, PET, p: dict, n_comp: int):
    """Run HBV. P, T, PET: (B, L). p[name]: (B, n_comp). Returns runoff (B, L)."""
    B, L = P.shape
    z = lambda: torch.zeros(B, n_comp, device=P.device)
    SNOW, MELTW, SM, SUZ, SLZ = z(), z(), z() + 0.3 * p["FC"], z() + 5.0, z() + 10.0
    out = []
    for t in range(L):
        Pt, Tt, Et = P[:, t : t + 1], T[:, t : t + 1], PET[:, t : t + 1]
        rain = torch.where(Tt >= p["TT"], Pt, torch.zeros_like(Pt)).expand(B, n_comp)
        snow = torch.where(Tt < p["TT"], Pt, torch.zeros_like(Pt)).expand(B, n_comp)
        SNOW = SNOW + snow
        melt = torch.minimum(torch.clamp(p["CFMAX"] * (Tt - p["TT"]), min=0.0), SNOW)
        MELTW = MELTW + melt
        SNOW = SNOW - melt
        refr = torch.minimum(torch.clamp(p["CFR"] * p["CFMAX"] * (p["TT"] - Tt), min=0.0), MELTW)
        SNOW = SNOW + refr
        MELTW = MELTW - refr
        tosoil = torch.clamp(MELTW - p["CWH"] * SNOW, min=0.0)
        MELTW = MELTW - tosoil
        wet = torch.clamp((SM / p["FC"]) ** p["BETA"], 0.0, 1.0)
        recharge = (rain + tosoil) * wet
        SM = SM + rain + tosoil - recharge
        excess = torch.clamp(SM - p["FC"], min=0.0)
        SM = SM - excess
        evf = torch.clamp(SM / (p["LP"] * p["FC"]), 0.0, 1.0)
        et = torch.minimum(SM, Et * evf)
        SM = torch.clamp(SM - et, min=1e-5)
        SUZ = SUZ + recharge + excess
        perc = torch.minimum(SUZ, p["PERC"])
        SUZ = SUZ - perc
        q0 = p["K0"] * torch.clamp(SUZ - p["UZL"], min=0.0)
        SUZ = SUZ - q0
        q1 = p["K1"] * SUZ
        SUZ = SUZ - q1
        SLZ = SLZ + perc
        q2 = p["K2"] * SLZ
        SLZ = SLZ - q2
        out.append((q0 + q1 + q2).mean(dim=1))
    return torch.stack(out, dim=1)


def _gamma_route(q: torch.Tensor, a: torch.Tensor, b: torch.Tensor, window: int) -> torch.Tensor:
    """Convolve runoff (B, L) with a gamma unit hydrograph per basin."""
    k = a + 0.1
    theta = b + 0.5
    t = torch.arange(window, dtype=torch.float32, device=q.device).unsqueeze(0) + 0.5
    logw = (k.unsqueeze(1) - 1) * torch.log(t) - t / theta.unsqueeze(1)
    w = torch.softmax(logw, dim=1)                               # (B, W)
    qpad = nn.functional.pad(q, (window - 1, 0))
    windows = qpad.unfold(1, window, 1)                          # (B, L, W) oldest first
    return (windows * w.flip(1).unsqueeze(1)).sum(-1)


class _ParamNet(nn.Module):
    def __init__(self, n_static: int, cfg: HybridConfig):
        super().__init__()
        self.n = cfg.n_components
        self.mlp = nn.Sequential(nn.Linear(n_static, cfg.hidden), nn.Tanh(), nn.Linear(cfg.hidden, cfg.hidden), nn.Tanh(),
                                 nn.Linear(cfg.hidden, len(PNAMES) * cfg.n_components))

    def forward(self, s):
        raw = torch.sigmoid(self.mlp(s)).view(-1, len(PNAMES), self.n)
        out = {}
        for i, name in enumerate(PNAMES):
            lo, hi = RANGES[name]
            v = lo + (hi - lo) * raw[:, i, :]
            out[name] = v.mean(1) if name in ("RT_A", "RT_B") else v
        return out


class HybridModel:
    """δHBV: neural parameter learning + differentiable HBV."""

    def __init__(self, cfg: Optional[HybridConfig] = None):
        self.cfg = cfg or HybridConfig()
        torch.manual_seed(self.cfg.seed)
        self.net = _ParamNet(len(STATIC), self.cfg)
        self.stat_mean = np.zeros(len(STATIC))
        self.stat_std = np.ones(len(STATIC))
        self.history: list = []

    # ---------------------------------------------------------------- data
    @staticmethod
    def _series(b: Basin):
        f = b.forcings
        P = np.nan_to_num(f["prcp"].to_numpy(np.float32))
        T = ((f["tmax"] + f["tmin"]) / 2).to_numpy(np.float32)
        E = pet_hargreaves(f["tmax"], f["tmin"], b.lat).to_numpy(np.float32)
        return P, T, E, b.q_mm.to_numpy(np.float32)

    def _stat(self, b: Basin):
        return ((b.statics.to_numpy() - self.stat_mean) / self.stat_std).astype(np.float32)

    def params(self, basin: Basin) -> pd.DataFrame:
        with torch.no_grad():
            p = self.net(torch.from_numpy(self._stat(basin)[None]))
        n = self.cfg.n_components
        cols = {k: (v[0].numpy() if v.dim() > 1 else np.repeat(v.numpy(), n)) for k, v in p.items()}
        return pd.DataFrame(cols, index=[f"component_{i + 1}" for i in range(n)]).T

    # ----------------------------------------------------------------- fit
    def fit(self, basins: Sequence[Basin], train_masks, val_masks=None, learning_rate=None, epochs=None,
            progress: Optional[Callable[[int, int, float], None]] = None) -> "HybridModel":
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        if len(basins) > 1:
            S = np.stack([b.statics.to_numpy() for b in basins])
            self.stat_mean, self.stat_std = S.mean(0), np.where(S.std(0) < 1e-6, 1.0, S.std(0))
        elif not np.any(self.stat_mean):
            self.stat_mean = basins[0].statics.to_numpy()
        series = [self._series(b) for b in basins]
        stats = [self._stat(b) for b in basins]
        Lw = cfg.warmup + cfg.predict_last
        samples = []
        for i, (ser, m) in enumerate(zip(series, train_masks)):
            q = ser[3]
            ends = np.arange(Lw - 1, len(q), 7)  # weekly-spaced windows
            ends = ends[[m[e - cfg.predict_last + 1 : e + 1].all() and np.isfinite(q[e - cfg.predict_last + 1 : e + 1]).mean() > 0.8
                         for e in ends]]
            samples += [(i, e) for e in ends]
        if len(samples) < 5:
            raise ValueError("Too little training data for the hybrid model.")
        samples = np.array(samples)
        qstd = np.array([np.nanstd(s[3]) or 1.0 for s in series], dtype=np.float32)
        dev = torch.device("cuda" if cfg.device == "auto" and torch.cuda.is_available() else
                           ("cpu" if cfg.device == "auto" else cfg.device))
        self.net.to(dev)
        opt = torch.optim.Adam(self.net.parameters(), lr=learning_rate or cfg.learning_rate)
        best, best_state, bad = -np.inf, None, 0
        n_ep = epochs or cfg.epochs
        for ep in range(n_ep):
            order = rng.permutation(len(samples))
            if cfg.samples_per_epoch:
                order = order[: cfg.samples_per_epoch]
            losses = []
            for j in range(0, len(order), cfg.batch_size):
                sel = samples[order[j : j + cfg.batch_size]]
                P = torch.from_numpy(np.stack([series[i][0][e - Lw + 1 : e + 1] for i, e in sel]))
                T = torch.from_numpy(np.stack([series[i][1][e - Lw + 1 : e + 1] for i, e in sel]))
                E = torch.from_numpy(np.stack([series[i][2][e - Lw + 1 : e + 1] for i, e in sel]))
                Y = torch.from_numpy(np.stack([series[i][3][e - cfg.predict_last + 1 : e + 1] for i, e in sel]))
                W = torch.from_numpy(qstd[sel[:, 0]]).unsqueeze(1)
                s = torch.from_numpy(np.stack([stats[i] for i, _ in sel]))
                P, T, E, Y, W, s = (x.to(dev) for x in (P, T, E, Y, W, s))
                p = self.net(s)
                q = _gamma_route(_hbv(P, T, E, p, cfg.n_components), p["RT_A"], p["RT_B"], cfg.route_window)
                q = q[:, -cfg.predict_last :]
                ok = torch.isfinite(Y)
                loss = (((q - torch.nan_to_num(Y)) / (W + 0.1)) ** 2)[ok].mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                losses.append(loss.item())
            score = -float(np.mean(losses))
            if val_masks is not None:
                self.net.to("cpu")
                sc = []
                for b, vm in list(zip(basins, val_masks))[:40]:
                    pred = self.predict(b)["mean"].to_numpy()
                    sc.append(nse(b.q_mm.to_numpy()[vm], pred[vm]))
                score = float(np.nanmedian(sc))
                self.net.to(dev)
            self.history.append({"epoch": ep + 1, "loss": float(np.mean(losses)), "val_score": score})
            if progress:
                progress(ep + 1, n_ep, score)
            if score > best + 1e-4:
                best, bad, best_state = score, 0, copy.deepcopy(self.net.state_dict())
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
        if best_state:
            self.net.load_state_dict(best_state)
        self.net.to("cpu")
        return self

    def finetune(self, basin: Basin, train_mask, val_mask, epochs: int = 20, learning_rate: float = 1e-3,
                 progress=None) -> "HybridModel":
        m = copy.deepcopy(self)
        m.cfg.samples_per_epoch = None
        return m.fit([basin], [train_mask], [val_mask], learning_rate=learning_rate, epochs=epochs, progress=progress)

    # ------------------------------------------------------------- predict
    def predict(self, basin: Basin) -> pd.DataFrame:
        P, T, E, _ = self._series(basin)
        with torch.no_grad():
            p = self.net(torch.from_numpy(self._stat(basin)[None]))
            q = _gamma_route(_hbv(torch.from_numpy(P[None]), torch.from_numpy(T[None]), torch.from_numpy(E[None]),
                                  p, self.cfg.n_components), p["RT_A"], p["RT_B"], self.cfg.route_window)[0].numpy()
        q[: self.cfg.warmup] = np.nan
        return pd.DataFrame({"mean": q}, index=basin.dates)

    # ---------------------------------------------------------------- save
    def save(self, folder) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.cpu() for k, v in self.net.state_dict().items()}, folder / "weights.pt")
        (folder / "hybrid.json").write_text(json.dumps({
            "config": asdict(self.cfg), "stat_mean": self.stat_mean.tolist(), "stat_std": self.stat_std.tolist(), "static": STATIC,
        }, indent=1))
        return folder

    @classmethod
    def load(cls, folder) -> "HybridModel":
        folder = Path(folder)
        meta = json.loads((folder / "hybrid.json").read_text())
        m = cls(HybridConfig(**meta["config"]))
        m.stat_mean, m.stat_std = np.array(meta["stat_mean"]), np.array(meta["stat_std"])
        m.net.load_state_dict(torch.load(folder / "weights.pt", weights_only=True, map_location="cpu"))
        return m


def ensemble(preds: Sequence[pd.DataFrame], weights: Optional[Sequence[float]] = None) -> pd.DataFrame:
    """Weighted average of model predictions (mean columns; quantile columns
    are combined only when every member has them)."""
    weights = np.ones(len(preds)) / len(preds) if weights is None else np.asarray(weights) / np.sum(weights)
    out = pd.DataFrame(index=preds[0].index)
    out["mean"] = sum(w * p["mean"] for w, p in zip(weights, preds))
    common = set.intersection(*[set(c for c in p.columns if c.startswith("q")) for p in preds])
    for c in sorted(common):
        out[c] = sum(w * p[c] for w, p in zip(weights, preds))
    return out
