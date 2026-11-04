"""Phase 3: probabilistic streamflow forecasts several days ahead.

Architecture (after Nearing et al., 2024):

* a **hindcast LSTM** - the pre-trained (or fine-tuned) model - reads the
  past ``lookback`` days of observed forcings, and optionally yesterday's
  observed flow ("data assimilation"), up to the forecast issue day;
* its memory (hidden and cell state) is handed through a learned linear
  map to a **forecast LSTM** that reads the weather forecast for lead days
  1..H together with the basin attributes;
* a CMAL head gives a probability distribution for each lead day.

Training uses historical observed weather in place of forecasts, with
perturbations that grow with lead time to mimic forecast error, because
archived forecasts are rarely available for a single basin. Forecast
skill with real forecasts will be lower than these hindcast scores; the
report states this.

Forecast inputs are the variables global forecast services provide:
precipitation, max/min temperature and shortwave radiation; vapour
pressure is estimated from minimum temperature in both training and use,
so the inputs match.
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from .basin import Basin
from .features import DYNAMIC, STATIC
from .metrics import nse
from .nn import CMALHead, ModelConfig, StreamflowModel, _Arrays, cmal_mean, cmal_nll, cmal_sample


def vp_from_tmin(tmin):
    """Vapour pressure (Pa) assuming dew point = minimum temperature."""
    tmin = np.asarray(tmin, float)
    return 611.0 * np.exp(17.27 * tmin / (tmin + 237.3))


@dataclass
class ForecastConfig:
    horizon: int = 7
    use_recent_flow: bool = True
    learning_rate: float = 3e-4
    epochs: int = 20
    patience: int = 5
    batch_size: int = 128
    precip_error: float = 0.15   # lognormal sigma per lead day (grows with lead)
    temp_error: float = 0.6      # degC per lead day
    error_correction: bool = True  # AR(1) update with the latest observed flow
    seed: int = 0


class _ForecastNet(nn.Module):
    """Hindcast LSTM -> state handoff -> forecast LSTM -> CMAL head.

    The forecast LSTM and head start as copies of the trained hindcast
    model and the handoff starts as the identity, so before any forecast
    training the network simply continues the simulation with forecast
    weather; training then learns forecast-specific corrections.
    """

    def __init__(self, hind: nn.Module, n_fc_in: int, cfg: ForecastConfig, k: int):
        super().__init__()
        self.hind = hind                                     # HydroLSTM (pre-trained, maybe with flow inputs)
        H = hind.lstm.hidden_size
        self.fc_lstm = nn.LSTM(n_fc_in, H, batch_first=True)
        with torch.no_grad():
            self.fc_lstm.weight_ih_l0.copy_(hind.lstm.weight_ih_l0[:, :n_fc_in])
            self.fc_lstm.weight_hh_l0.copy_(hind.lstm.weight_hh_l0)
            self.fc_lstm.bias_ih_l0.copy_(hind.lstm.bias_ih_l0)
            self.fc_lstm.bias_hh_l0.copy_(hind.lstm.bias_hh_l0)
        self.handoff_h = nn.Linear(H, H)
        self.handoff_c = nn.Linear(H, H)
        for lin in (self.handoff_h, self.handoff_c):    # residual handoff starting at zero
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        if isinstance(hind.head, CMALHead):
            self.head = copy.deepcopy(hind.head)
        else:
            self.head = CMALHead(H, k)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x_hind, x_fc):
        _, (h, c) = self.hind.lstm(x_hind)
        h0 = (h[-1] + self.handoff_h(h[-1])).unsqueeze(0)
        c0 = (c[-1] + self.handoff_c(c[-1])).unsqueeze(0)
        out, _ = self.fc_lstm(x_fc, (h0.contiguous(), c0.contiguous()))
        return self.head(self.dropout(out))                  # params with shape (B, H, K)


class ForecastModel:
    def __init__(self, base: StreamflowModel, cfg: Optional[ForecastConfig] = None):
        self.cfg = cfg or ForecastConfig()
        self.base = base.copy()
        self.sim = base.copy()   # plain simulation model, used for error correction
        self.phi = 0.0
        if self.cfg.use_recent_flow:
            if self.base.scaler.extra_cols:
                raise ValueError("Use a model without gauge inputs as the base for forecasting.")
            self.base.net.expand_extra(2)
        n_fc_in = len(DYNAMIC) + len(STATIC)
        torch.manual_seed(self.cfg.seed)
        self.net = _ForecastNet(self.base.net, n_fc_in, self.cfg, self.base.cfg.n_mixtures)
        self.history: list = []

    # ------------------------------------------------------------- arrays
    def _hind_array(self, basin: Basin) -> np.ndarray:
        a = _Arrays(basin.with_extra(None), self.base.scaler, False).x  # dynamic + static
        if self.cfg.use_recent_flow:
            sc = self.base.scaler
            q = basin.q_mm.to_numpy(float)
            lag = np.r_[np.nan, q[:-1]]                       # yesterday's flow
            avail = np.isfinite(lag).astype(np.float32)
            lagn = np.nan_to_num((lag - sc.q_mean) / sc.q_std)
            a = np.concatenate([a, lagn[:, None], avail[:, None]], axis=1)
        return a.astype(np.float32)

    def _fc_inputs(self, forcings: pd.DataFrame, statics: np.ndarray) -> np.ndarray:
        sc = self.base.scaler
        f = forcings[DYNAMIC].copy()
        f["vp"] = vp_from_tmin(f["tmin"])
        x = (f.to_numpy() - sc.dyn_mean) / sc.dyn_std
        s = (statics - sc.stat_mean) / sc.stat_std
        return np.nan_to_num(np.concatenate([x, np.broadcast_to(s, (len(x), len(s)))], axis=1)).astype(np.float32)

    def _perturb(self, f: pd.DataFrame, rng) -> pd.DataFrame:
        f = f.copy()
        lead = np.arange(1, len(f) + 1)
        f["prcp"] = f["prcp"] * rng.lognormal(0, self.cfg.precip_error * np.sqrt(lead))
        dt = rng.normal(0, self.cfg.temp_error * np.sqrt(lead))
        f["tmax"] += dt
        f["tmin"] += dt
        return f

    # ---------------------------------------------------------------- fit
    def fit(self, basin: Basin, train_mask: np.ndarray, val_mask: np.ndarray,
            progress: Optional[Callable[[int, int, float], None]] = None) -> "ForecastModel":
        cfg, L, Hz = self.cfg, self.base.cfg.lookback, self.cfg.horizon
        rng = np.random.default_rng(cfg.seed)
        xh = self._hind_array(basin)
        stat = basin.statics.to_numpy()
        sc = self.base.scaler
        y = ((basin.q_mm.to_numpy(float) - sc.q_mean) / sc.q_std).astype(np.float32)
        T = len(y)
        issue = np.arange(L - 1, T - Hz)

        def ok(mask):
            sel = []
            for t in issue:
                if mask[t + 1 : t + Hz + 1].all() and np.isfinite(y[t + 1 : t + Hz + 1]).all():
                    sel.append(t)
            return np.array(sel)

        tr, va = ok(train_mask), ok(val_mask)
        if len(tr) < 50:
            raise ValueError("Not enough training days for forecasting with this horizon.")
        forc = basin.forcings

        def batch(ts, perturb=True):
            xb = np.stack([xh[t - L + 1 : t + 1] for t in ts])
            fb = np.stack([self._fc_inputs(self._perturb(forc.iloc[t + 1 : t + Hz + 1], rng) if perturb
                                           else forc.iloc[t + 1 : t + Hz + 1], stat) for t in ts])
            yb = np.stack([y[t + 1 : t + Hz + 1] for t in ts])
            return torch.from_numpy(xb), torch.from_numpy(fb), torch.from_numpy(yb)

        opt = torch.optim.Adam([
            {"params": self.net.hind.parameters(), "lr": cfg.learning_rate},
            {"params": [p for n, p in self.net.named_parameters() if not n.startswith("hind.")], "lr": cfg.learning_rate},
        ])
        best, best_state, bad = -np.inf, None, 0
        for ep in range(cfg.epochs):
            self.net.train()
            perm = rng.permutation(tr)
            losses = []
            for i in range(0, len(perm), cfg.batch_size):
                xb, fb, yb = batch(perm[i : i + cfg.batch_size])
                p = self.net(xb, fb)
                flat = {k: v.reshape(-1, v.shape[-1]) for k, v in p.items()}
                loss = cmal_nll(flat, yb.reshape(-1)).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                losses.append(loss.item())
            score = float(np.nanmean(self.skill_by_lead(basin, va)["NSE"])) if len(va) else -np.mean(losses)
            self.history.append({"epoch": ep + 1, "loss": float(np.mean(losses)), "val_mean_nse": score})
            if progress:
                progress(ep + 1, cfg.epochs, score)
            if score > best + 1e-4:
                best, bad, best_state = score, 0, copy.deepcopy(self.net.state_dict())
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
        if best_state:
            self.net.load_state_dict(best_state)
        if cfg.error_correction:
            self.phi = self._fit_phi(basin, train_mask)
        return self

    def _fit_phi(self, basin: Basin, mask: np.ndarray) -> list:
        """Per-lead regression coefficient of the error h days later on
        today's error (simulation errors on the training days)."""
        sim = self.sim.predict(basin, quantiles=())["mean"].to_numpy()
        e = basin.q_mm.to_numpy(float) - sim
        e[~mask] = np.nan
        out = []
        for h in range(1, self.cfg.horizon + 1):
            a, b = e[:-h], e[h:]
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() < 30 or np.var(a[ok]) == 0:
                out.append(0.0)
            else:
                out.append(float(np.clip(np.cov(a[ok], b[ok])[0, 1] / np.var(a[ok]), 0.0, 0.99)))
        return out

    def _correction(self, basin: Basin, issue_idx: np.ndarray) -> np.ndarray:
        """phi^h * (observed - simulated) on the issue day -> (n_issue, horizon) in mm/day."""
        Hz = self.cfg.horizon
        phi = np.atleast_1d(np.asarray(self.phi, float))
        if not self.cfg.error_correction or not np.any(phi > 0):
            return np.zeros((len(issue_idx), Hz))
        if phi.size == 1:  # older saved models
            phi = phi[0] ** np.arange(1, Hz + 1)
        sim = self.sim.predict(basin, quantiles=())["mean"].to_numpy()
        err = np.nan_to_num(basin.q_mm.to_numpy(float)[issue_idx] - sim[issue_idx])
        return err[:, None] * phi[None, :Hz]

    # ------------------------------------------------------------ predict
    def _params(self, xb, fb):
        self.net.eval()
        with torch.no_grad():
            return self.net(xb, fb)

    def hindcast(self, basin: Basin, issue_idx: np.ndarray) -> np.ndarray:
        """Mean forecasts (mm/day) for each issue day using observed weather
        as a perfect forecast -> array (n_issue, horizon)."""
        L, Hz, sc = self.base.cfg.lookback, self.cfg.horizon, self.base.scaler
        xh, stat, forc = self._hind_array(basin), basin.statics.to_numpy(), basin.forcings
        out = []
        for i in range(0, len(issue_idx), 256):
            ts = issue_idx[i : i + 256]
            xb = torch.from_numpy(np.stack([xh[t - L + 1 : t + 1] for t in ts]))
            fb = torch.from_numpy(np.stack([self._fc_inputs(forc.iloc[t + 1 : t + Hz + 1], stat) for t in ts]))
            p = self._params(xb, fb)
            out.append(cmal_mean(p).numpy() * sc.q_std + sc.q_mean)
        if not out:
            return np.zeros((0, Hz))
        return np.maximum(np.concatenate(out) + self._correction(basin, issue_idx), 0)

    def skill_by_lead(self, basin: Basin, issue_idx: np.ndarray) -> pd.DataFrame:
        Hz = self.cfg.horizon
        pred = self.hindcast(basin, issue_idx)
        q = basin.q_mm.to_numpy(float)
        rows = []
        for h in range(Hz):
            obs = q[issue_idx + h + 1]
            persistence = q[issue_idx]
            rows.append({"lead_day": h + 1, "NSE": nse(obs, pred[:, h]), "NSE_persistence": nse(obs, persistence)})
        return pd.DataFrame(rows)

    def forecast(self, history: Basin, forecast_forcings: pd.DataFrame,
                 quantiles: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95), n_samples: int = 500) -> pd.DataFrame:
        """Forecast from the end of ``history`` using ``forecast_forcings``
        (one row per lead day; columns prcp, srad, tmax, tmin; mm/day)."""
        L, sc = self.base.cfg.lookback, self.base.scaler
        Hz = min(self.cfg.horizon, len(forecast_forcings))
        if len(history.dates) < L:
            raise ValueError(f"At least {L} days of history are needed.")
        xh = self._hind_array(history)[-L:]
        f = forecast_forcings.iloc[:Hz].copy()
        if "vp" not in f:
            f["vp"] = vp_from_tmin(f["tmin"])
        fb = self._fc_inputs(f, history.statics.to_numpy())
        p = self._params(torch.from_numpy(xh[None]), torch.from_numpy(fb[None]))
        flat = {k: v[0] for k, v in p.items()}
        s = cmal_sample(flat, n_samples, torch.Generator().manual_seed(self.cfg.seed)) * sc.q_std + sc.q_mean
        corr = self._correction(history, np.array([len(history.dates) - 1]))[0, :Hz]
        s = (s + torch.from_numpy(corr).float().unsqueeze(1)).clamp(min=0)
        df = pd.DataFrame(index=forecast_forcings.index[:Hz])
        df.index.name = "date"
        df["lead_day"] = np.arange(1, Hz + 1)
        df["mean"] = s.mean(1).numpy()
        for qv in quantiles:
            df[f"q{int(round(qv * 100)):02d}"] = torch.quantile(s, qv, dim=1).numpy()
        return df

    # --------------------------------------------------------------- save
    def save(self, folder) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.base.save(folder / "hindcast")
        self.sim.save(folder / "simulation")
        torch.save(self.net.state_dict(), folder / "forecast_weights.pt")
        (folder / "forecast.json").write_text(json.dumps({**asdict(self.cfg), "phi": self.phi}, indent=1))
        return folder

    @classmethod
    def load(cls, folder) -> "ForecastModel":
        folder = Path(folder)
        meta = json.loads((folder / "forecast.json").read_text())
        phi = meta.pop("phi", 0.0)
        cfg = ForecastConfig(**meta)
        base = StreamflowModel.load(folder / "hindcast")
        m = cls.__new__(cls)
        m.cfg, m.base, m.history, m.phi = cfg, base, [], phi
        m.sim = StreamflowModel.load(folder / "simulation")
        m.net = _ForecastNet(base.net, len(DYNAMIC) + len(STATIC), cfg, base.cfg.n_mixtures)
        m.net.load_state_dict(torch.load(folder / "forecast_weights.pt", weights_only=True))
        return m
