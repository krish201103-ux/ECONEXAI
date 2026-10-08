"""
explainability.py -- Why is the forecast moving, and where is the dust coming from?

  exact_group_shapley()  exact Shapley values over five human-meaningful driver
                         groups (recent trend, planned site activity, wind &
                         dispersion, regional background, time of day). With five
                         groups all 32 coalitions are evaluated exactly, so the
                         drivers add up to the forecast by construction -- no
                         sampling noise and no dependency on the `shap` package.
  SourceAttributor       an NNLS receptor model in the spirit of Chemical Mass
                         Balance for "AI-estimated contributors" (construction,
                         road dust, exhaust, commuter traffic, other), with
                         day-block bootstrap ranges and a confidence score.

Both are estimates and the UI labels them that way. On synthetic data the true
source shares are known, so train.py reports the attribution error honestly.
"""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import nnls

import site_model as site
from data_loader import ObservedData, ZONE_IDS
from ml_features import dispersion_proxy


# ==================================================================================================
# Exact group-Shapley explanations for forecasts.
# Features are grouped into a handful of human-meaningful drivers (recent trend,
# planned site activity, wind & dispersion, regional background, time of day).
# With five groups, all 2^5 = 32 coalitions can be evaluated exactly, so the
# attributions are exact Shapley values of the *interventional* value function
# v(S) = E_b[f(x_S, b_~S)] over a background sample b - no sampling noise, no
# dependency on the shap package, and additive by construction:
#
#     prediction = baseline + sum(group contributions)
#
# The background sample is "a typical comparable moment" (same zone, similar
# time of day, recent weeks), which makes the explanation read naturally:
# "compared with a typical afternoon in Zone B, the forecast is +X because ...".
# ==================================================================================================

def exact_group_shapley(predict: Callable[[pd.DataFrame], np.ndarray], x: pd.DataFrame, background: pd.DataFrame,
                        groups: Dict[str, List[str]]) -> Dict[str, object]:
    names = list(groups)
    n = len(names)
    bg = background.reset_index(drop=True)
    k = len(bg)
    coalitions = list(itertools.product([0, 1], repeat=n))
    rows = []
    for mask in coalitions:
        block = bg.copy()
        for g, on in zip(names, mask):
            if on:
                for col in groups[g]:
                    block[col] = x[col].iloc[0]
        rows.append(block)
    stacked = pd.concat(rows, ignore_index=True)
    preds = np.asarray(predict(stacked), dtype=float).reshape(len(coalitions), k).mean(axis=1)
    value = {mask: v for mask, v in zip(coalitions, preds)}

    phi = {}
    for i, g in enumerate(names):
        total = 0.0
        for mask in coalitions:
            if mask[i]:
                continue
            s = sum(mask)
            with_g = list(mask)
            with_g[i] = 1
            weight = math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)
            total += weight * (value[tuple(with_g)] - value[mask])
        phi[g] = float(total)
    baseline = float(value[tuple([0] * n)])
    full = float(value[tuple([1] * n)])
    return {"baseline": baseline, "prediction": full, "contributions": phi}


# ==================================================================================================
# Source attribution ("AI-estimated contributors").
# A receptor-style source-apportionment model in the spirit of Chemical Mass
# Balance (CMB). The zone's *local excess* PM10 and PM2.5 (zone reading minus the
# regional reference station) are regressed jointly on dispersion-scaled activity
# indicators with non-negativity constraints (NNLS):
#
#     excess_PM10(t) = D(t) * sum_k b_k * x_k(t)
#     excess_PM2.5(t) = D(t) * sum_k f_k * b_k * x_k(t)
#
#     x_construction = (own + wind-weighted upwind construction) x surface dryness
#     x_road_dust    = haul trucks x surface dryness   (resuspension, suppressed when wet)
#     x_exhaust      = haul trucks                      (diesel exhaust)
#     x_commuter     = commuter vehicles x road exposure
#     x_other        = 1                                (other / unexplained local)
#
# D(t) is the inverse-ventilation dispersion factor and f_k are literature-style
# PM2.5/PM10 *source profiles* (crustal dust is coarse, exhaust is fine). Using
# both size fractions plus surface-moisture variation is what makes road dust
# separable from exhaust. Weak emission-inventory priors (ridge penalty towards
# engineering estimates) stabilise the fit when activities share a schedule.
# A day-block bootstrap gives share intervals and the confidence score.
#
# These are *estimates* and are always labelled "AI-estimated contributors" -
# never presented as measured source shares.
# ==================================================================================================

logger = logging.getLogger(__name__)

TERMS = ["construction", "road_dust", "exhaust", "commuter_traffic", "other_local"]
CATEGORY_OF = {
    "construction": "Construction", "road_dust": "Road dust",
    "exhaust": "Traffic", "commuter_traffic": "Traffic", "other_local": "Other",
}
CATEGORIES = ["Construction", "Road dust", "Traffic", "Other"]
# PM2.5/PM10 source profiles (crustal dust coarse, combustion fine)
PM25_PROFILE = np.array([0.22, 0.22, 0.65, 0.65, 0.45])
BACKGROUND_PM25_FRACTION = 0.5
# Engineering emission-factor priors (ug/m3 at D = 1 per unit indicator) and prior scales
PRIOR_MEAN = np.array([25.0, 1.2, 0.5, 0.4, 2.0])
PRIOR_SCALE = np.array([15.0, 0.8, 0.4, 0.4, 3.0])
PRIOR_STRENGTH = 30.0


def design_matrix(obs: ObservedData, z: str, index: Optional[pd.DatetimeIndex] = None) -> pd.DataFrame:
    idx = index if index is not None else obs.index
    wx = obs.weather.loc[idx]
    D = dispersion_proxy(wx["ventilation"].to_numpy())
    zc = site.ZONE_BY_ID[z].centroid
    upwind = np.zeros(len(idx))
    for s in ZONE_IDS:
        if s == z:
            continue
        w = np.array([site.transport_weight(site.ZONE_BY_ID[s].centroid, zc, wd, ws)
                      for wd, ws in zip(wx["wind_dir_from"].to_numpy(), wx["wind_speed"].to_numpy())])
        upwind += w * obs.construction[s].loc[idx].to_numpy()
    trucks = obs.trucks[z].rolling(2, min_periods=1).mean().loc[idx].to_numpy()
    dry = 1.0 - obs.surface_moisture[z].loc[idx].to_numpy()
    vehicles = obs.weather["vehicles"].rolling(6, min_periods=1).mean().loc[idx].to_numpy()
    X = pd.DataFrame({
        "construction": D * (obs.construction[z].loc[idx].to_numpy() + upwind) * (1 - 0.75 * (1 - dry)),
        "road_dust": D * trucks * dry,
        "exhaust": D * trucks,
        "commuter_traffic": D * vehicles * site.ZONE_BY_ID[z].road_exposure,
        "other_local": D,
    }, index=idx)
    return X


@dataclass
class ZoneAttributionModel:
    zone_id: str
    coef: np.ndarray
    boot_coefs: np.ndarray
    r2: float
    n: int


@dataclass
class SourceAttributor:
    n_boot: int = 40
    models: Dict[str, ZoneAttributionModel] = field(default_factory=dict)
    validation: Dict[str, object] = field(default_factory=dict)

    def fit(self, obs: ObservedData, end: pd.Timestamp, days: int = 60, seed: int = 7) -> "SourceAttributor":
        rng = np.random.default_rng(seed)
        start = end - pd.Timedelta(days=days)
        for z in ZONE_IDS:
            idx = obs.index[(obs.index > start) & (obs.index <= end)]
            # thin to 15-min resolution: neighbouring 5-min rows are nearly duplicates
            idx = idx[::3]
            X = design_matrix(obs, z, idx)
            ref = obs.weather["reference_pm10"].loc[idx].to_numpy()
            y10 = obs.pm10[z].loc[idx].to_numpy() - ref
            y25 = obs.pm25[z].loc[idx].to_numpy() - BACKGROUND_PM25_FRACTION * ref
            A10 = X.to_numpy()
            A25 = A10 * PM25_PROFILE
            w25 = 2.0
            active = np.array([np.abs(X[c]).sum() > 1e-6 for c in X.columns])

            def solve(rows: np.ndarray) -> np.ndarray:
                A = np.vstack([A10[rows], w25 * A25[rows]])
                y = np.concatenate([y10[rows], w25 * y25[rows]])
                k = np.sqrt(PRIOR_STRENGTH * len(rows) / 1000.0) / PRIOR_SCALE
                A_aug = np.vstack([A[:, active], np.diag(k[active])])
                y_aug = np.concatenate([y, k[active] * PRIOR_MEAN[active]])
                c, _ = nnls(A_aug, y_aug)
                full = np.zeros(len(active))
                full[active] = c
                return full

            all_rows = np.arange(len(idx))
            coef = solve(all_rows)
            pred = A10 @ coef
            r2 = 1 - np.sum((y10 - pred) ** 2) / max(np.sum((y10 - y10.mean()) ** 2), 1e-9)
            days_arr = idx.normalize()
            unique_days = days_arr.unique()
            day_rows = [np.where(days_arr == d)[0] for d in unique_days]
            boots = []
            for _ in range(self.n_boot):
                pick = rng.choice(len(unique_days), len(unique_days), replace=True)
                boots.append(solve(np.concatenate([day_rows[i] for i in pick])))
            y = y10
            self.models[z] = ZoneAttributionModel(z, coef, np.array(boots), float(r2), len(y))
            logger.info(f"  attribution zone {z}: R2={r2:.3f} coef=" + ", ".join(f"{c}={v:.2f}" for c, v in zip(X.columns, coef)))
        return self

    # ------------------------------------------------------------------ #
    def explain(self, obs: ObservedData, z: str, t: pd.Timestamp, coverage: float = 1.0) -> Dict[str, object]:
        """Contributor shares for zone z at time t (30-minute window average)."""
        m = self.models[z]
        window = obs.index[(obs.index > t - pd.Timedelta(minutes=30)) & (obs.index <= t)]
        X = design_matrix(obs, z, window).mean().to_numpy()
        observed = float(obs.pm10[z].loc[window].mean())
        background = float(obs.weather["reference_pm10"].loc[window].mean())
        excess = max(observed - background, 0.0)

        def shares_for(coef: np.ndarray) -> Dict[str, float]:
            contrib = X * coef
            modelled = contrib.sum()
            cats = {c: 0.0 for c in CATEGORIES}
            for term, val in zip(TERMS, contrib):
                cats[CATEGORY_OF[term]] += max(val, 0.0)
            cats["Other"] += max(excess - modelled, 0.0)          # unexplained local excess
            total = sum(cats.values()) or 1.0
            return {c: v / total for c, v in cats.items()}

        point = shares_for(m.coef)
        boot = np.array([[shares_for(c)[k] for k in CATEGORIES] for c in m.boot_coefs])
        lo = np.percentile(boot, 10, axis=0)
        hi = np.percentile(boot, 90, axis=0)
        modelled_excess = float(X @ m.coef)
        fit_agreement = 1 - min(abs(excess - modelled_excess) / max(excess, 1.0), 1.0)
        width = float(np.mean(hi - lo))
        confidence = float(np.clip(0.35 * m.r2 + 0.35 * fit_agreement + 0.30 * (1 - 2.5 * width), 0, 1)) * \
            float(np.clip(0.6 + 0.4 * coverage, 0, 1))
        contributors = []
        for i, c in enumerate(CATEGORIES):
            contributors.append({
                "source": c,
                "share_pct": round(100 * point[c], 1),
                "range_pct": [round(100 * lo[i], 1), round(100 * hi[i], 1)],
                "ugm3": round(point[c] * excess, 1),
            })
        # normalise rounding so shares sum to 100
        diff = round(100 - sum(x["share_pct"] for x in contributors), 1)
        contributors[0]["share_pct"] = round(contributors[0]["share_pct"] + diff, 1)
        return {
            "zone_id": z,
            "time": str(t),
            "observed_pm10": round(observed, 1),
            "regional_background_pm10": round(background, 1),
            "local_excess_pm10": round(excess, 1),
            "contributors": contributors,
            "confidence": round(confidence, 2),
            "model_r2": round(m.r2, 3),
            "method": "NNLS receptor model on dispersion-scaled activity indicators, day-block bootstrap (n=%d)" % len(m.boot_coefs),
            "label": "AI-estimated contributors",
        }

    # ------------------------------------------------------------------ #
    def validate_against_truth(self, obs: ObservedData, start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, object]:
        """Synthetic data only: compare estimated shares with the simulator's true source contributions."""
        if obs.truth is None:
            return {"available": False}
        truth = obs.truth
        errors = {c: [] for c in CATEGORIES}
        n = 0
        for z in ("A", "B"):
            tz = truth[(truth["zone_id"] == z) & (truth["timestamp"] >= start) & (truth["timestamp"] <= end)].set_index("timestamp")
            tz = tz[(tz.index.hour >= 10) & (tz.index.hour <= 16) & (tz.index.minute == 0)]
            for t, row in tz.iterrows():
                local = {"Construction": row["contrib_construction"], "Road dust": row["contrib_road_dust"],
                         "Traffic": row["contrib_traffic"], "Other": row["contrib_other"]}
                tot = sum(local.values())
                if tot < 15:
                    continue
                est = self.explain(obs, z, t)
                for c in est["contributors"]:
                    errors[c["source"]].append(abs(c["share_pct"] - 100 * local[c["source"]] / tot))
                n += 1
        self.validation = {
            "available": True, "samples": n, "window": [str(start), str(end)],
            "mae_share_pp": {c: round(float(np.mean(v)), 1) if v else None for c, v in errors.items()},
            "mae_share_pp_overall": round(float(np.mean([e for v in errors.values() for e in v])), 1) if n else None,
            "note": "Mean absolute error (percentage points) of estimated vs simulator-true source shares, "
                    "zones A and B, working hours, hourly samples.",
        }
        return self.validation
