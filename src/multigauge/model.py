"""A compact sequence-to-one LSTM for daily streamflow.

The model reads the previous ``lookback`` days of inputs and predicts flow on
the last day, the same set-up used by NeuralHydrology's single-basin
``cudalstm`` models. Inputs are standardised with training-period statistics
only, early stopping uses validation-period NSE, and the best weights are
restored at the end of training.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn

from .metrics import nse


@dataclass
class TrainConfig:
    lookback: int = 365
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 256
    epochs: int = 40
    patience: int = 8
    seed: int = 42
    target_transform: str = "none"  # "none" or "log" (log1p of flow)
    threads: Optional[int] = None

    @classmethod
    def preset(cls, name: str) -> "TrainConfig":
        """'quick' for demos and the web app, 'standard' for research runs."""
        if name == "quick":
            return cls(lookback=90, hidden_size=32, epochs=25, patience=6)
        if name == "standard":
            return cls()
        if name == "thorough":
            return cls(hidden_size=128, epochs=100, patience=15)
        raise ValueError("preset must be 'quick', 'standard' or 'thorough'")


class _LSTMNet(nn.Module):
    def __init__(self, n_inputs: int, cfg: TrainConfig):
        super().__init__()
        self.lstm = nn.LSTM(
            n_inputs,
            cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(cfg.hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.dropout(out[:, -1, :])).squeeze(-1)


class StreamflowLSTM:
    """Fit/predict wrapper around the LSTM with its own normalisation."""

    def __init__(self, config: Optional[TrainConfig] = None):
        self.config = config or TrainConfig()
        self.net: Optional[_LSTMNet] = None
        self.columns: list = []
        self.history: list = []

    # ------------------------------------------------------------- transforms
    def _fwd_y(self, y):
        return np.log1p(np.maximum(y, 0)) if self.config.target_transform == "log" else y

    def _inv_y(self, y):
        return np.expm1(y) if self.config.target_transform == "log" else y

    def _x_array(self, features: pd.DataFrame) -> np.ndarray:
        x = features[self.columns].to_numpy(dtype=np.float32)
        return (x - self.x_mean) / self.x_std

    def _windows(self, x: np.ndarray) -> torch.Tensor:
        # (T, F) -> (T-L+1, L, F) view; sample i ends on day i+L-1
        t = torch.from_numpy(np.ascontiguousarray(x))
        return t.unfold(0, self.config.lookback, 1).permute(0, 2, 1)

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        features: pd.DataFrame,
        flow: pd.Series,
        train_mask: np.ndarray,
        val_mask: np.ndarray,
        progress: Optional[Callable[[int, int, float], None]] = None,
    ) -> "StreamflowLSTM":
        cfg = self.config
        if cfg.threads:
            torch.set_num_threads(cfg.threads)
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)

        self.columns = list(features.columns)
        raw = features.to_numpy(dtype=np.float32)
        self.x_mean = raw[train_mask].mean(axis=0)
        self.x_std = raw[train_mask].std(axis=0)
        self.x_std[self.x_std == 0] = 1.0

        y_all = flow.to_numpy(dtype=np.float64)
        yt = self._fwd_y(y_all)
        self.y_mean = float(np.nanmean(yt[train_mask]))
        self.y_std = float(np.nanstd(yt[train_mask])) or 1.0
        y_norm = ((yt - self.y_mean) / self.y_std).astype(np.float32)

        L = cfg.lookback
        if len(features) <= L:
            raise ValueError("The record is shorter than the lookback window.")
        win = self._windows(self._x_array(features))
        end_day = np.arange(L - 1, len(features))
        has_y = ~np.isnan(y_norm[end_day])
        tr_idx = np.where(train_mask[end_day] & has_y)[0]
        va_idx = np.where(val_mask[end_day] & has_y)[0]
        if len(tr_idx) < 100:
            raise ValueError(
                f"Only {len(tr_idx)} training samples after the {L}-day lookback. "
                "Use a longer training period or a shorter lookback."
            )
        y_t = torch.from_numpy(y_norm[end_day])

        self.net = _LSTMNet(len(self.columns), cfg)
        opt = torch.optim.Adam(self.net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        loss_fn = nn.MSELoss()

        best, best_state, bad = -np.inf, None, 0
        self.history = []
        for epoch in range(cfg.epochs):
            t0 = time.time()
            self.net.train()
            order = rng.permutation(tr_idx)
            losses = []
            for i in range(0, len(order), cfg.batch_size):
                b = order[i : i + cfg.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.net(win[b]), y_t[b])
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                losses.append(loss.item())

            score = self._score(win, va_idx, y_all[end_day]) if len(va_idx) else -float(np.mean(losses))
            self.history.append(
                {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_nse": score, "seconds": time.time() - t0}
            )
            if progress:
                progress(epoch + 1, cfg.epochs, score)
            if score > best + 1e-4:
                best, bad = score, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.best_val_nse = float(best)
        return self

    def _predict_windows(self, win: torch.Tensor, idx: np.ndarray) -> np.ndarray:
        self.net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(idx), 2048):
                out.append(self.net(win[idx[i : i + 2048]]).numpy())
        y = np.concatenate(out) if out else np.array([])
        return np.maximum(self._inv_y(y * self.y_std + self.y_mean), 0.0)

    def _score(self, win, idx, y_obs_end) -> float:
        return nse(y_obs_end[idx], self._predict_windows(win, idx))

    # -------------------------------------------------------------- predict
    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Predicted flow for every day that has a full lookback window."""
        if self.net is None:
            raise RuntimeError("Call fit() first.")
        L = self.config.lookback
        win = self._windows(self._x_array(features))
        pred = np.full(len(features), np.nan)
        pred[L - 1 :] = self._predict_windows(win, np.arange(win.shape[0]))
        return pd.Series(pred, index=features.index, name="predicted")

    # ----------------------------------------------------------------- save
    def save(self, path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path / "weights.pt")
        meta = {
            "config": asdict(self.config),
            "columns": self.columns,
            "x_mean": self.x_mean.tolist(),
            "x_std": self.x_std.tolist(),
            "y_mean": self.y_mean,
            "y_std": self.y_std,
        }
        (path / "model.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path) -> "StreamflowLSTM":
        path = Path(path)
        meta = json.loads((path / "model.json").read_text())
        m = cls(TrainConfig(**meta["config"]))
        m.columns = meta["columns"]
        m.x_mean = np.array(meta["x_mean"], dtype=np.float32)
        m.x_std = np.array(meta["x_std"], dtype=np.float32)
        m.y_mean, m.y_std = meta["y_mean"], meta["y_std"]
        m.net = _LSTMNet(len(m.columns), m.config)
        m.net.load_state_dict(torch.load(path / "weights.pt", weights_only=True))
        return m
