"""Conceptual benchmark: GR4J (Perrin et al., 2003) with a degree-day snow pack.

A calibrated conceptual model is the standard yardstick for data-driven
hydrology: an LSTM is only interesting if it beats it. GR4J has four
parameters (production store capacity X1, groundwater exchange X2, routing
store capacity X3, unit-hydrograph time base X4); a fifth parameter, the
degree-day melt factor, handles snow. Potential evaporation comes from the
Hargreaves equation using the same Daymet temperatures as the LSTMs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from .basin import Basin
from .features import pet_hargreaves
from .metrics import nse

BOUNDS = [(10.0, 2000.0), (-8.0, 6.0), (5.0, 500.0), (0.5, 6.0), (0.5, 6.0)]
NAMES = ["X1", "X2", "X3", "X4", "DDF"]


def _sh1(t, x4):
    return np.where(t <= 0, 0.0, np.where(t < x4, (t / x4) ** 2.5, 1.0))


def _sh2(t, x4):
    return np.where(
        t <= 0, 0.0,
        np.where(t <= x4, 0.5 * (t / x4) ** 2.5,
                 np.where(t < 2 * x4, 1 - 0.5 * np.clip(2 - t / x4, 0, None) ** 2.5, 1.0)),
    )


def gr4j(prcp: np.ndarray, pet: np.ndarray, tmean: np.ndarray, params) -> np.ndarray:
    """Run GR4J (+ snow); returns simulated discharge in mm/day."""
    x1, x2, x3, x4, ddf = params
    n = len(prcp)
    # snow
    rain = np.empty(n)
    swe = 0.0
    for t in range(n):
        p = prcp[t]
        if tmean[t] <= 0.0:
            swe += p
            rain[t] = 0.0
        else:
            melt = min(swe, ddf * tmean[t])
            swe -= melt
            rain[t] = p + melt
    # production store
    pr = np.empty(n)
    s = 0.5 * x1
    for t in range(n):
        p, e = rain[t], pet[t]
        if p >= e:
            pn, ps = p - e, 0.0
            tw = np.tanh(pn / x1)
            ps = x1 * (1 - (s / x1) ** 2) * tw / (1 + s / x1 * tw)
            s += ps
        else:
            en, pn, ps = e - p, 0.0, 0.0
            tw = np.tanh(en / x1)
            es = s * (2 - s / x1) * tw / (1 + (1 - s / x1) * tw)
            s -= es
        perc = s * (1 - (1 + (4.0 / 9.0 * s / x1) ** 4) ** -0.25)
        s -= perc
        pr[t] = perc + (pn - ps)
    # unit hydrographs
    k1 = np.arange(int(np.ceil(x4)) + 1)
    uh1 = np.diff(_sh1(k1.astype(float), x4))
    k2 = np.arange(int(np.ceil(2 * x4)) + 1)
    uh2 = np.diff(_sh2(k2.astype(float), x4))
    q9 = np.convolve(0.9 * pr, uh1)[:n]
    q1 = np.convolve(0.1 * pr, uh2)[:n]
    # routing store
    q = np.empty(n)
    r = 0.5 * x3
    for t in range(n):
        f = x2 * (r / x3) ** 3.5
        r = max(0.0, r + q9[t] + f)
        qr = r * (1 - (1 + (r / x3) ** 4) ** -0.25)
        r -= qr
        q[t] = qr + max(0.0, q1[t] + f)
    return q


@dataclass
class GR4JModel:
    params: np.ndarray
    lat: float
    warmup_days: int = 365

    def _inputs(self, basin: Basin, prcp: Optional[pd.Series] = None):
        f = basin.forcings
        p = (prcp if prcp is not None else f["prcp"]).to_numpy(float)
        pet = pet_hargreaves(f["tmax"], f["tmin"], self.lat).to_numpy()
        tmean = ((f["tmax"] + f["tmin"]) / 2).to_numpy(float)
        return np.nan_to_num(p), pet, tmean

    def predict(self, basin: Basin, prcp: Optional[pd.Series] = None) -> pd.DataFrame:
        q = gr4j(*self._inputs(basin, prcp), self.params)
        q[: self.warmup_days] = np.nan
        return pd.DataFrame({"mean": q}, index=basin.dates)

    @classmethod
    def calibrate(
        cls,
        basin: Basin,
        train_mask: np.ndarray,
        prcp: Optional[pd.Series] = None,
        maxiter: int = 60,
        seed: int = 0,
    ) -> "GR4JModel":
        """Calibrate by maximising NSE on the training days (differential evolution)."""
        m = cls(np.zeros(5), basin.lat)
        inputs = m._inputs(basin, prcp)
        obs = basin.q_mm.to_numpy(float)
        sel = train_mask.copy()
        sel[: m.warmup_days] = False
        sel &= np.isfinite(obs)

        def loss(theta):
            sim = gr4j(*inputs, theta)
            return 1 - nse(obs[sel], sim[sel])

        res = differential_evolution(loss, BOUNDS, maxiter=maxiter, popsize=10, tol=1e-4, seed=seed, polish=False)
        m.params = res.x
        return m

    def to_dict(self) -> dict:
        return dict(zip(NAMES, map(float, self.params)))
