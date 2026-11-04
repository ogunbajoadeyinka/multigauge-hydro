"""Pre-training on many basins and fine-tuning on one.

Workflow
--------
1. ``pretrain`` trains one LSTM on many basins (CAMELS-US for the released
   model). This is done once, on a GPU, with ``notebooks/pretrain_camels.ipynb``.
2. ``finetune`` copies the pre-trained model and continues training on the
   user's basin at a lower learning rate, optionally adding local rain
   gauges as new inputs whose weights start at zero.
3. ``train_local`` trains the same kind of LSTM from scratch on the user's
   basin only: the conventional single-basin approach, kept as a baseline.
"""
from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from .basin import Basin
from .nn import ModelConfig, StreamflowModel


def period_mask(dates: pd.DatetimeIndex, period: Optional[tuple]) -> np.ndarray:
    if period is None:
        return np.ones(len(dates), bool)
    a, b = pd.Timestamp(period[0]), pd.Timestamp(period[1])
    return np.asarray((dates >= a) & (dates <= b))


def pretrain(
    basins: Sequence[Basin],
    cfg: Optional[ModelConfig] = None,
    train_period: tuple = ("1980-10-01", "1995-09-30"),
    val_period: tuple = ("1995-10-01", "2000-09-30"),
    progress: Optional[Callable[[int, int, float], None]] = None,
) -> StreamflowModel:
    """Train one model on many basins. Periods default to the CAMELS-US
    convention used by Kratzert et al. (2019)."""
    cfg = cfg or ModelConfig.preset("pretrain")
    train = [period_mask(b.dates, train_period) for b in basins]
    val = [period_mask(b.dates, val_period) for b in basins]
    model = StreamflowModel.new(basins, train, cfg)
    model.fit(basins, train, val, progress=progress)
    model.info.update(
        trained_on=len(basins), train_period=list(map(str, train_period)), val_period=list(map(str, val_period))
    )
    return model


def finetune(
    base: StreamflowModel,
    basin: Basin,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    learning_rate: float = 2e-4,
    epochs: int = 20,
    patience: Optional[int] = None,
    seed: Optional[int] = None,
    progress: Optional[Callable[[int, int, float], None]] = None,
) -> StreamflowModel:
    """Fine-tune a copy of ``base`` on one basin (``basin.extra`` = new inputs)."""
    m = base.copy()
    if seed is not None:
        m.cfg.seed = seed
    m.cfg.patience = patience or max(4, m.cfg.patience)
    m.cfg.samples_per_epoch = None
    if basin.extra is not None and basin.extra.shape[1]:
        m.add_extra_inputs(basin, train_mask)
    m.fit([basin], [train_mask], [val_mask], learning_rate=learning_rate, epochs=epochs, progress=progress)
    m.info["finetuned_on"] = basin.basin_id
    return m


def train_local(
    basin: Basin,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    cfg: Optional[ModelConfig] = None,
    progress: Optional[Callable[[int, int, float], None]] = None,
) -> StreamflowModel:
    """Single-basin LSTM trained from scratch (baseline)."""
    cfg = cfg or ModelConfig.preset("local")
    m = StreamflowModel.new([basin], [train_mask], cfg)
    if basin.extra is not None and basin.extra.shape[1]:
        m.add_extra_inputs(basin, train_mask)
    m.fit([basin], [train_mask], [val_mask], progress=progress)
    m.info["trained_on"] = basin.basin_id
    return m


# ----------------------------------------------------------------- registry
def pretrained_dir() -> Path:
    return Path(str(resources.files("multigauge"))) / "pretrained"


def available_pretrained() -> list:
    d = pretrained_dir()
    return sorted(p.name for p in d.iterdir() if (p / "model.json").exists()) if d.exists() else []


def load_pretrained(name: str = "camels-us") -> StreamflowModel:
    """Load a model shipped with the package, or from a folder path.

    ``camels-us`` is created by ``notebooks/pretrain_camels.ipynb`` and
    committed to the repository once trained. ``demo-synthetic`` is trained
    on synthetic basins and is only meant for the built-in demo.
    """
    p = Path(name)
    if not (p / "model.json").exists():
        p = pretrained_dir() / name
    if not (p / "model.json").exists():
        have = available_pretrained()
        raise FileNotFoundError(
            f"Pre-trained model '{name}' was not found. Available: {have or 'none'}. "
            "Run notebooks/pretrain_camels.ipynb to create the CAMELS-US model, "
            "or pass the folder of a saved model."
        )
    return StreamflowModel.load(p)
