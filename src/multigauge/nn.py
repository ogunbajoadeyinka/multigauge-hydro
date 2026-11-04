"""Multi-basin LSTM with a deterministic or probabilistic (CMAL) head.

Design follows the large-sample LSTM literature (Kratzert et al., 2019;
NeuralHydrology) and Google's operational flood model (Nearing et al.,
2024):

* one LSTM is trained on many basins at once; static basin attributes are
  appended to the daily inputs so the network can tell basins apart;
* the deterministic head is trained with the NSE* loss, which weights each
  basin by its flow variability so large rivers do not dominate;
* the probabilistic head outputs a Countable Mixture of Asymmetric
  Laplacians (CMAL) and is trained by negative log-likelihood, giving
  prediction intervals as well as a central estimate;
* extra inputs (for example local rain gauges) can be added to a trained
  model: their input weights start at zero, so fine-tuning begins from
  exactly the pre-trained behaviour and learns how much the new inputs help.
"""
from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from .basin import Basin
from .features import DYNAMIC, STATIC
from .metrics import nse


# =========================================================================
# configuration
# =========================================================================
@dataclass
class ModelConfig:
    hidden_size: int = 128
    dropout: float = 0.4
    lookback: int = 365
    head: str = "cmal"  # "cmal" (probabilistic) or "regression"
    n_mixtures: int = 3
    learning_rate: float = 1e-3
    batch_size: int = 256
    epochs: int = 30
    patience: int = 6
    samples_per_epoch: Optional[int] = None  # cap for very large data sets
    max_val_basins: int = 60
    seed: int = 0
    threads: Optional[int] = None
    device: str = "auto"  # "auto" uses a GPU when PyTorch can see one

    def torch_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    @classmethod
    def preset(cls, name: str) -> "ModelConfig":
        presets = {
            # used in tests and the web demo
            "tiny": dict(hidden_size=16, lookback=60, epochs=3, patience=2, batch_size=128, dropout=0.1),
            "quick": dict(hidden_size=64, lookback=180, epochs=15, patience=4),
            # the configuration used for the released pre-trained model
            "pretrain": dict(hidden_size=128, lookback=365, epochs=30, patience=6, samples_per_epoch=200_000),
            # single-basin LSTM trained from scratch (the 'local' baseline)
            "local": dict(hidden_size=64, lookback=365, epochs=60, patience=10, dropout=0.2),
        }
        if name not in presets:
            raise ValueError(f"Unknown preset '{name}'. Choose from {sorted(presets)}.")
        return cls(**presets[name])


# =========================================================================
# normalisation
# =========================================================================
@dataclass
class Scaler:
    dyn_mean: np.ndarray
    dyn_std: np.ndarray
    stat_mean: np.ndarray
    stat_std: np.ndarray
    q_mean: float
    q_std: float
    extra_cols: list = field(default_factory=list)
    extra_mean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    extra_std: np.ndarray = field(default_factory=lambda: np.ones(0))

    @classmethod
    def fit(cls, basins: Sequence[Basin], masks: Sequence[np.ndarray]) -> "Scaler":
        dyn = np.concatenate([b.forcings.to_numpy()[m] for b, m in zip(basins, masks)])
        q = np.concatenate([b.q_mm.to_numpy()[m] for b, m in zip(basins, masks)])
        stat = np.stack([b.statics.to_numpy() for b in basins])
        s_std = stat.std(axis=0) if len(basins) > 1 else np.ones(stat.shape[1])
        return cls(
            dyn_mean=np.nanmean(dyn, axis=0), dyn_std=_safe(np.nanstd(dyn, axis=0)),
            stat_mean=stat.mean(axis=0), stat_std=_safe(s_std),
            q_mean=float(np.nanmean(q)), q_std=float(_safe(np.array([np.nanstd(q)]))[0]),
        )

    def fit_extra(self, basin: Basin, mask: np.ndarray) -> None:
        x = basin.extra.to_numpy()[mask]
        self.extra_cols = list(basin.extra.columns)
        self.extra_mean = np.nanmean(x, axis=0)
        self.extra_std = _safe(np.nanstd(x, axis=0))

    def to_dict(self) -> dict:
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: dict) -> "Scaler":
        arr = {k: np.asarray(v, dtype=np.float64) for k, v in d.items()
               if k in ("dyn_mean", "dyn_std", "stat_mean", "stat_std", "extra_mean", "extra_std")}
        return cls(**{**d, **arr})


def _safe(std):
    std = np.asarray(std, dtype=np.float64)
    std[~np.isfinite(std) | (std < 1e-6)] = 1.0
    return std


# =========================================================================
# network
# =========================================================================
class CMALHead(nn.Module):
    """Countable mixture of asymmetric Laplacians (Klotz et al., 2022)."""

    def __init__(self, n_in: int, k: int):
        super().__init__()
        self.k = k
        self.fc = nn.Linear(n_in, 4 * k)

    def forward(self, h):
        mu, b, tau, pi = self.fc(h).chunk(4, dim=-1)
        b = nn.functional.softplus(b) + 1e-3
        tau = torch.sigmoid(tau) * 0.98 + 0.01
        pi = torch.softmax(pi, dim=-1)
        return {"mu": mu, "b": b, "tau": tau, "pi": pi}


def cmal_nll(params: dict, y: torch.Tensor) -> torch.Tensor:
    y = y.unsqueeze(-1)
    mu, b, tau, pi = params["mu"], params["b"], params["tau"], params["pi"]
    u = (y - mu) / b
    rho = u * (tau - (u < 0).float())
    log_f = torch.log(tau) + torch.log1p(-tau) - torch.log(b) - rho
    return -torch.logsumexp(torch.log(pi) + log_f, dim=-1)


def cmal_mean(params: dict) -> torch.Tensor:
    mu, b, tau, pi = params["mu"], params["b"], params["tau"], params["pi"]
    m = mu + b * (1 - 2 * tau) / (tau * (1 - tau))
    return (pi * m).sum(-1)


def cmal_sample(params: dict, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Draw n samples per prediction -> tensor (N, n)."""
    mu, b, tau, pi = (params[k] for k in ("mu", "b", "tau", "pi"))
    N, K = mu.shape
    comp = torch.multinomial(pi, n, replacement=True, generator=generator)  # (N, n)
    g = lambda t: torch.gather(t, 1, comp)
    m, s, t = g(mu), g(b), g(tau)
    p = torch.rand(N, n, generator=generator).clamp(1e-6, 1 - 1e-6)
    lo = m + s / (1 - t) * torch.log(p / t)
    hi = m - s / t * torch.log((1 - p) / (1 - t))
    return torch.where(p < t, lo, hi)


class HydroLSTM(nn.Module):
    def __init__(self, n_dynamic: int, n_static: int, cfg: ModelConfig, n_extra: int = 0):
        super().__init__()
        self.cfg = cfg
        self.n_dynamic, self.n_static, self.n_extra = n_dynamic, n_static, n_extra
        self.lstm = nn.LSTM(n_dynamic + n_static + n_extra, cfg.hidden_size, batch_first=True)
        self.dropout = nn.Dropout(cfg.dropout)
        if cfg.head == "cmal":
            self.head = CMALHead(cfg.hidden_size, cfg.n_mixtures)
        elif cfg.head == "regression":
            self.head = nn.Linear(cfg.hidden_size, 1)
        else:
            raise ValueError("head must be 'cmal' or 'regression'")
        with torch.no_grad():  # positive forget-gate bias helps long memory
            hs = cfg.hidden_size
            self.lstm.bias_hh_l0[hs : 2 * hs].fill_(3.0)

    def forward(self, x):
        out, _ = self.lstm(x)
        h = self.dropout(out[:, -1, :])
        return self.head(h) if self.cfg.head == "cmal" else {"y": self.head(h).squeeze(-1)}

    def expand_extra(self, n_new: int) -> None:
        """Append ``n_new`` input channels whose weights start at zero."""
        if n_new <= 0:
            return
        old = self.lstm
        new = nn.LSTM(old.input_size + n_new, old.hidden_size, batch_first=True)
        with torch.no_grad():
            new.weight_ih_l0.zero_()
            new.weight_ih_l0[:, : old.input_size] = old.weight_ih_l0
            new.weight_hh_l0.copy_(old.weight_hh_l0)
            new.bias_ih_l0.copy_(old.bias_ih_l0)
            new.bias_hh_l0.copy_(old.bias_hh_l0)
        self.lstm = new
        self.n_extra += n_new


# =========================================================================
# data windows
# =========================================================================
class _Arrays:
    """Normalised arrays for one basin."""

    def __init__(self, basin: Basin, scaler: Scaler, use_extra: bool):
        x = (basin.forcings.to_numpy() - scaler.dyn_mean) / scaler.dyn_std
        s = (basin.statics.to_numpy() - scaler.stat_mean) / scaler.stat_std
        parts = [x, np.broadcast_to(s, (len(x), len(s)))]
        if use_extra and scaler.extra_cols:
            e = basin.extra[scaler.extra_cols].to_numpy()
            e = (e - scaler.extra_mean) / scaler.extra_std
            parts.append(np.nan_to_num(e, nan=0.0))
        self.x = np.nan_to_num(np.concatenate(parts, axis=1)).astype(np.float32)
        self.q = basin.q_mm.to_numpy(dtype=np.float64)
        self.y = ((self.q - scaler.q_mean) / scaler.q_std).astype(np.float32)
        q_valid = self.q[np.isfinite(self.q)]
        self.q_std_norm = float(np.std(q_valid) / scaler.q_std) if len(q_valid) > 1 else 1.0


def _gather(arrs, idx, L):
    xb = np.stack([arrs[b].x[t - L + 1 : t + 1] for b, t in idx])
    yb = np.array([arrs[b].y[t] for b, t in idx], dtype=np.float32)
    wb = np.array([arrs[b].q_std_norm for b, t in idx], dtype=np.float32)
    return torch.from_numpy(xb), torch.from_numpy(yb), torch.from_numpy(wb)


def _sample_index(arrs, masks, L):
    rows = []
    for i, (a, m) in enumerate(zip(arrs, masks)):
        t = np.where(m & np.isfinite(a.y) & (np.arange(len(a.y)) >= L - 1))[0]
        rows.append(np.column_stack([np.full(len(t), i), t]))
    return np.concatenate(rows) if rows else np.zeros((0, 2), int)


# =========================================================================
# the model wrapper
# =========================================================================
class StreamflowModel:
    """A trained network together with its normalisation and settings."""

    def __init__(self, cfg: ModelConfig, scaler: Scaler, net: HydroLSTM, info: Optional[dict] = None):
        self.cfg, self.scaler, self.net = cfg, scaler, net
        self.info = info or {}
        self.history: list = []

    # ----------------------------------------------------------- creation
    @classmethod
    def new(cls, basins: Sequence[Basin], masks: Sequence[np.ndarray], cfg: ModelConfig) -> "StreamflowModel":
        torch.manual_seed(cfg.seed)
        scaler = Scaler.fit(basins, masks)
        net = HydroLSTM(len(DYNAMIC), len(STATIC), cfg)
        return cls(cfg, scaler, net, {"n_basins": len(basins)})

    def copy(self) -> "StreamflowModel":
        m = StreamflowModel(copy.deepcopy(self.cfg), copy.deepcopy(self.scaler), copy.deepcopy(self.net), dict(self.info))
        return m

    def add_extra_inputs(self, basin: Basin, train_mask: np.ndarray) -> None:
        """Prepare the model to use ``basin.extra`` (e.g. rain gauges)."""
        if basin.extra is None or basin.extra.shape[1] == 0:
            return
        if self.scaler.extra_cols:
            raise RuntimeError("This model already has extra inputs.")
        self.scaler.fit_extra(basin, train_mask)
        self.net.expand_extra(len(self.scaler.extra_cols))

    # ------------------------------------------------------------- train
    def fit(
        self,
        basins: Sequence[Basin],
        train_masks: Sequence[np.ndarray],
        val_masks: Optional[Sequence[np.ndarray]] = None,
        learning_rate: Optional[float] = None,
        epochs: Optional[int] = None,
        progress: Optional[Callable[[int, int, float], None]] = None,
    ) -> "StreamflowModel":
        cfg = self.cfg
        if cfg.threads:
            torch.set_num_threads(cfg.threads)
        torch.manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        L = cfg.lookback
        use_extra = bool(self.scaler.extra_cols)
        arrs = [_Arrays(b, self.scaler, use_extra) for b in basins]
        idx = _sample_index(arrs, train_masks, L)
        if len(idx) < 50:
            raise ValueError(f"Only {len(idx)} training samples with a {L}-day lookback. Use more data or a shorter lookback.")

        val_sel = None
        if val_masks is not None:
            order = rng.permutation(len(basins))[: cfg.max_val_basins]
            val_sel = [(i, val_masks[i]) for i in order if val_masks[i].any()]

        dev = cfg.torch_device()
        self.net.to(dev)
        opt = torch.optim.Adam(self.net.parameters(), lr=learning_rate or cfg.learning_rate)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=max(1, cfg.patience // 2))
        n_epochs = epochs or cfg.epochs
        best, best_state, bad = -np.inf, None, 0
        self.history = []
        for ep in range(n_epochs):
            t0 = time.time()
            self.net.train()
            order = rng.permutation(len(idx))
            if cfg.samples_per_epoch:
                order = order[: cfg.samples_per_epoch]
            losses = []
            for i in range(0, len(order), cfg.batch_size):
                xb, yb, wb = (t.to(dev) for t in _gather(arrs, idx[order[i : i + cfg.batch_size]], L))
                out = self.net(xb)
                if cfg.head == "cmal":
                    loss = cmal_nll(out, yb).mean()
                else:
                    loss = (((out["y"] - yb) / (wb + 0.1)) ** 2).mean()  # NSE* loss
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                losses.append(loss.item())
            score = float(-np.mean(losses))
            if val_sel:
                scores = []
                for i, m in val_sel:
                    pred = self._predict_arrays(arrs[i], np.where(m)[0])
                    scores.append(nse(arrs[i].q[m], pred))
                score = float(np.nanmedian(scores))
            sched.step(score)
            self.history.append({"epoch": ep + 1, "loss": float(np.mean(losses)), "val_score": score, "seconds": round(time.time() - t0, 2)})
            if progress:
                progress(ep + 1, n_epochs, score)
            if score > best + 1e-4:
                best, bad = score, 0
                best_state = copy.deepcopy(self.net.state_dict())
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.to("cpu")
        self.info["best_val_score"] = best
        return self

    # ----------------------------------------------------------- predict
    def _net_out(self, a: _Arrays, days: np.ndarray):
        L = self.cfg.lookback
        days = days[days >= L - 1]
        self.net.eval()
        dev = next(self.net.parameters()).device
        outs = []
        with torch.no_grad():
            for i in range(0, len(days), 1024):
                xb = torch.from_numpy(np.stack([a.x[t - L + 1 : t + 1] for t in days[i : i + 1024]])).to(dev)
                outs.append({k: v.cpu() for k, v in self.net(xb).items()})
        return days, outs

    def _predict_arrays(self, a: _Arrays, days: np.ndarray) -> np.ndarray:
        full = np.full(len(days), np.nan)
        kept, outs = self._net_out(a, days)
        if not outs:
            return full
        if self.cfg.head == "cmal":
            y = torch.cat([cmal_mean(o) for o in outs]).numpy()
        else:
            y = torch.cat([o["y"] for o in outs]).numpy()
        q = np.maximum(y * self.scaler.q_std + self.scaler.q_mean, 0.0)
        full[np.isin(days, kept)] = q
        return full

    def predict(self, basin: Basin, quantiles: Sequence[float] = (0.05, 0.5, 0.95), n_samples: int = 300) -> pd.DataFrame:
        """Daily predictions (mm/day): ``mean`` plus quantile columns for CMAL models."""
        a = _Arrays(basin, self.scaler, bool(self.scaler.extra_cols))
        days = np.arange(len(basin.dates))
        df = pd.DataFrame(index=basin.dates)
        df["mean"] = self._predict_arrays(a, days)
        if self.cfg.head == "cmal" and quantiles:
            kept, outs = self._net_out(a, days)
            gen = torch.Generator().manual_seed(self.cfg.seed)
            qs = []
            for o in outs:
                s = cmal_sample(o, n_samples, gen) * self.scaler.q_std + self.scaler.q_mean
                qs.append(torch.quantile(s.clamp(min=0), torch.tensor(list(quantiles)), dim=1).T)
            qarr = torch.cat(qs).numpy()
            for j, qv in enumerate(quantiles):
                col = np.full(len(days), np.nan)
                col[kept] = qarr[:, j]
                df[f"q{int(round(qv * 100)):02d}"] = col
        return df

    # -------------------------------------------------------------- save
    def save(self, folder) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.cpu() for k, v in self.net.state_dict().items()}, folder / "weights.pt")
        meta = {
            "format": "multigauge-model-1",
            "config": asdict(self.cfg),
            "scaler": self.scaler.to_dict(),
            "n_extra": self.net.n_extra,
            "dynamic": DYNAMIC,
            "static": STATIC,
            "info": self.info,
        }
        (folder / "model.json").write_text(json.dumps(meta, indent=1, default=float))
        return folder

    @classmethod
    def load(cls, folder) -> "StreamflowModel":
        folder = Path(folder)
        meta = json.loads((folder / "model.json").read_text())
        if meta.get("dynamic") != DYNAMIC or meta.get("static") != STATIC:
            raise ValueError("This saved model uses different input variables from this version of multigauge.")
        cfg = ModelConfig(**meta["config"])
        net = HydroLSTM(len(DYNAMIC), len(STATIC), cfg, n_extra=meta["n_extra"])
        net.load_state_dict(torch.load(folder / "weights.pt", weights_only=True, map_location="cpu"))
        return cls(cfg, Scaler.from_dict(meta["scaler"]), net, meta.get("info", {}))
