"""
ml_features.py -- Feature engineering, forecast horizons and evaluation metrics.

Two halves, both used by ml_models.py:

  1. Features for direct multi-horizon forecasting. For a forecast issued at
     time t for zone z and horizon h, every feature uses only information that
     is available at t (tests/test_platform.py::test_features_have_no_future_pm
     checks this): recent trend, planned site activity, dispersion (weather
     forecast for t+h), regional background and calendar, plus a few
     physics-informed interactions. Features are grouped into five drivers
     (GROUP_LABELS) so explainability.py can compute exact group-Shapley values.
  2. Metrics: RMSE, MAE, R2, MAPE, Willmott's index of agreement and
     rolling-origin (walk-forward) splits (Review Comment 5).
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple, Generator

import numpy as np
import pandas as pd

import site_model as site
from data_loader import ObservedData, ZONE_IDS


# ==================================================================================================
# Feature engineering for direct multi-horizon forecasting.
# For a forecast issued at time t for zone z and horizon h, the feature vector
# contains only information that is available at t:
#
# * recent trend      - current PM10/PM2.5, lags, rolling statistics, slopes
# * site activity     - current activity plus the *planned* construction schedule
#                       at t+h and the expected haul-truck load
# * dispersion        - current meteorology plus the weather forecast for t+h
#                       (an NWP feed; in the synthetic world this is the true
#                       future weather with horizon-dependent forecast error)
# * background        - regional reference station level and its trend
# * calendar/context  - time of day at t+h, work-day flag, zone
#
# Physics-informed interaction features (inverse-ventilation dispersion factor x
# planned emissions, wind-aligned upwind construction) give the tree models a
# head start without hard-coding the answer.
# ==================================================================================================

STEP_MIN = 5

# Horizon grid (steps of 5 minutes). Operational horizons drive DUSTWISE,
# planning horizons drive the 6h-72h outlook.
OPERATIONAL_HORIZONS: Dict[str, int] = {"15m": 3, "30m": 6, "45m": 9, "60m": 12, "75m": 15, "90m": 18}
PLANNING_HORIZONS: Dict[str, int] = {"2h": 24, "3h": 36, "4h": 48, "6h": 72, "9h": 108, "12h": 144, "18h": 216,
                                     "24h": 288, "36h": 432, "48h": 576, "60h": 720, "72h": 864}
ALL_HORIZONS: Dict[str, int] = {**OPERATIONAL_HORIZONS, **PLANNING_HORIZONS}
UI_TABS = ["15m", "30m", "60m", "90m", "6h", "24h", "72h"]

# Operational models (<= 90 min): rich short-term state.
FEATURE_GROUPS: Dict[str, List[str]] = {
    "recent_trend": ["pm10_now", "pm25_now", "pm10_lag_15m", "pm10_lag_30m", "pm10_lag_1h", "pm10_lag_2h",
                     "pm10_lag_3h", "pm10_lag_6h", "pm10_lag_24h", "pm10_mean_1h", "pm10_mean_3h", "pm10_std_1h",
                     "pm10_slope_30m", "pm10_slope_1h", "pm_ratio", "pm10_same_time_yday", "pm10_mean_24h"],
    "site_activity": ["construction_now", "trucks_30m", "vehicles_30m", "planned_construction_f",
                      "expected_trucks_f", "upwind_construction_f", "emission_index_f", "intervention_active",
                      "surface_moisture", "log_physics_ratio"],
    "dispersion": ["wind_speed", "wind_dir_sin", "wind_dir_cos", "mixing_height", "ventilation", "temperature",
                   "relative_humidity", "rain_1h", "wind_speed_f", "wind_dir_sin_f", "wind_dir_cos_f",
                   "mixing_height_f", "ventilation_f", "dispersion_f", "rain_f"],
    "background": ["reference_pm10", "reference_change_1h", "reference_mean_24h", "reference_f", "log_reference_change"],
    "calendar": ["hour_sin_f", "hour_cos_f", "workday_f", "zone_code"],
}
FEATURES: List[str] = [f for g in FEATURE_GROUPS.values() for f in g]

# Planning models (2-72 h): change-based features only. Absolute-level features
# let trees learn seasonal mean-reversion that fails under regime shifts, so the
# planning models see *changes* between now and the target time instead.
PLANNING_FEATURE_GROUPS: Dict[str, List[str]] = {
    "recent_trend": ["pm10_anom_24h", "local_enhancement_now", "pm10_slope_1h", "local_share"],
    "site_activity": ["activity_change", "log_emission_change", "planned_construction_f", "emission_index_f",
                      "upwind_construction_f", "surface_moisture", "log_physics_ratio"],
    "dispersion": ["log_dispersion_change", "dispersion_f", "wind_speed_f", "rain_f", "rain_1h"],
    "background": ["reference_f", "log_reference_change", "reference_anom_24h", "reference_change_1h"],
    "calendar": ["hour_sin_f", "hour_cos_f", "hour_sin", "hour_cos", "workday_f", "zone_code"],
}
PLANNING_FEATURES: List[str] = [f for g in PLANNING_FEATURE_GROUPS.values() for f in g]
ALL_FEATURE_COLUMNS: List[str] = sorted(set(FEATURES) | set(PLANNING_FEATURES))


def is_planning(h: int) -> bool:
    return h > 18


def features_for(h: int) -> List[str]:
    return PLANNING_FEATURES if is_planning(h) else FEATURES


def groups_for(h: int) -> Dict[str, List[str]]:
    return PLANNING_FEATURE_GROUPS if is_planning(h) else FEATURE_GROUPS


GROUP_LABELS = {
    "recent_trend": "Recent pollution trend",
    "site_activity": "Planned site activity",
    "dispersion": "Wind & dispersion conditions",
    "background": "Regional background",
    "calendar": "Time of day",
}


def dispersion_proxy(ventilation: np.ndarray | pd.Series) -> np.ndarray:
    """Inverse-ventilation concentration scaling (1.0 at ~1800 m2/s)."""
    v = np.asarray(ventilation, dtype=float)
    return np.clip(1800.0 / np.maximum(v, 120.0), 0.3, 4.5) ** 0.75


def _hash_normal(keys: np.ndarray, salt: int) -> np.ndarray:
    """Deterministic standard-normal draws keyed by integers (splitmix64 + Box-Muller)."""
    def mix(x: np.ndarray) -> np.ndarray:
        x = x.copy()
        x ^= x >> np.uint64(33)
        x *= np.uint64(0xFF51AFD7ED558CCD)
        x ^= x >> np.uint64(33)
        x *= np.uint64(0xC4CEB9FE1A85EC53)
        x ^= x >> np.uint64(33)
        return x
    with np.errstate(over="ignore"):
        k = keys.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15) + np.uint64(salt)
        u1 = (mix(k) >> np.uint64(11)).astype(np.float64) / float(1 << 53)
        u2 = (mix(k + np.uint64(0x632BE59BD9B4E019)) >> np.uint64(11)).astype(np.float64) / float(1 << 53)
    u1 = np.clip(u1, 1e-12, 1.0)
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2 * np.pi * u2)


_EPOCH = pd.Timestamp("2000-01-01")


def _nwp_noise(index: pd.DatetimeIndex, horizon_steps: int, scale: float, var: int = 0) -> np.ndarray:
    """Deterministic multiplicative forecast error for an NWP-style feed.

    The error is a smooth function of the *valid* time (t + h) scaled by a
    horizon-dependent magnitude, so (a) a forecast issued at t sees the same
    error whether features are built in bulk or for a single row, and (b) the
    forecast path across horizons is smooth, like real NWP output.
    """
    hours = horizon_steps * STEP_MIN / 60.0
    sigma = scale * (0.05 + 0.07 * np.log1p(hours))
    valid = index + pd.Timedelta(minutes=horizon_steps * STEP_MIN)
    keys = np.asarray((valid - _EPOCH) // pd.Timedelta(minutes=STEP_MIN), dtype=np.int64)
    raw = _hash_normal(keys, 7919 * (var + 1))
    smooth = pd.Series(raw).rolling(36, min_periods=1).mean().to_numpy() * np.sqrt(36 / 4)
    return np.exp(sigma * np.clip(smooth, -3, 3))


def _zone_frame(obs: ObservedData, z: str, h: int, wx_f: pd.DataFrame) -> pd.DataFrame:
    idx = obs.index
    pm10 = obs.pm10[z]
    pm25 = obs.pm25[z]
    wx = obs.weather
    f = pd.DataFrame(index=idx)
    f["pm10_now"] = pm10
    f["pm25_now"] = pm25
    for name, lag in (("15m", 3), ("30m", 6), ("1h", 12), ("2h", 24), ("3h", 36), ("6h", 72), ("24h", 288)):
        f[f"pm10_lag_{name}"] = pm10.shift(lag)
    f["pm10_mean_1h"] = pm10.rolling(12, min_periods=6).mean()
    f["pm10_mean_3h"] = pm10.rolling(36, min_periods=18).mean()
    f["pm10_std_1h"] = pm10.rolling(12, min_periods=6).std()
    f["pm10_mean_24h"] = pm10.rolling(288, min_periods=144).mean()
    f["pm10_slope_30m"] = pm10 - pm10.shift(6)
    f["pm10_slope_1h"] = pm10 - pm10.shift(12)
    f["pm_ratio"] = pm25 / pm10.clip(lower=1.0)
    f["pm10_same_time_yday"] = pm10.shift(288 - h) if h < 288 else pm10.shift(288 * int(np.ceil(h / 288)) - h)

    con = obs.construction[z]
    f["construction_now"] = con
    f["trucks_30m"] = obs.trucks[z].rolling(6, min_periods=1).mean()
    f["vehicles_30m"] = wx["vehicles"].rolling(6, min_periods=1).mean()
    # The site schedule is known ahead, so it may extend beyond the observation window.
    planned_now = obs.planned_construction[z].reindex(idx)
    planned_f = obs.planned_construction[z].shift(-h).reindex(idx)
    f["planned_construction_f"] = planned_f
    ratio = (planned_f / planned_now.clip(lower=0.05)).clip(0, 3)
    f["expected_trucks_f"] = (f["trucks_30m"] * ratio).where(planned_f > 0.02, 0.0)

    upwind = np.zeros(len(idx))
    zc = site.ZONE_BY_ID[z].centroid
    wd_f = wx_f["wind_dir_from"].to_numpy()
    ws_f = wx_f["wind_speed"].to_numpy()
    for s in ZONE_IDS:
        if s == z:
            continue
        sc = site.ZONE_BY_ID[s].centroid
        d = site.distance(sc, zc)
        wind_to = (wd_f + 180.0) % 360.0
        ang = np.radians(site.bearing_deg(sc, zc) - wind_to)
        align = (1.0 + np.cos(ang)) / 2.0
        directional = np.clip(ws_f / 1.5, 0, 1)
        weight = ((1 - directional) * 0.3 + directional * align ** 2) * np.exp(-d / (220 * (0.5 + 0.9 * directional * align)))
        upwind += weight * obs.planned_construction[s].shift(-h).reindex(idx).to_numpy()
    f["upwind_construction_f"] = upwind
    f["dispersion_f"] = dispersion_proxy(wx_f["ventilation"].to_numpy())
    f["emission_index_f"] = (planned_f.to_numpy() + 0.06 * f["expected_trucks_f"].to_numpy() + upwind) * f["dispersion_f"]
    f["intervention_active"] = obs.intervention_active[z]
    f["surface_moisture"] = obs.surface_moisture[z]

    f["wind_speed"] = wx["wind_speed"]
    rad = np.radians(wx["wind_dir_from"])
    f["wind_dir_sin"] = np.sin(rad)
    f["wind_dir_cos"] = np.cos(rad)
    f["mixing_height"] = wx["mixing_height"]
    f["ventilation"] = wx["ventilation"]
    f["temperature"] = wx["temperature"]
    f["relative_humidity"] = wx["relative_humidity"]
    f["rain_1h"] = wx["rain_mm"].rolling(12, min_periods=1).sum()
    f["wind_speed_f"] = wx_f["wind_speed"].to_numpy()
    rad_f = np.radians(wx_f["wind_dir_from"].to_numpy())
    f["wind_dir_sin_f"] = np.sin(rad_f)
    f["wind_dir_cos_f"] = np.cos(rad_f)
    f["mixing_height_f"] = wx_f["mixing_height"].to_numpy()
    f["ventilation_f"] = wx_f["ventilation"].to_numpy()
    f["rain_f"] = wx_f["rain_1h"].to_numpy()

    f["reference_pm10"] = wx["reference_pm10"]
    f["reference_change_1h"] = wx["reference_pm10"] - wx["reference_pm10"].shift(12)
    f["reference_mean_24h"] = wx["reference_pm10"].rolling(288, min_periods=144).mean()
    f["reference_f"] = wx_f["reference_pm10"].to_numpy()
    f["log_reference_change"] = np.log1p(f["reference_f"]) - np.log1p(f["reference_pm10"])
    f["reference_anom_24h"] = np.log1p(f["reference_pm10"]) - np.log1p(f["reference_mean_24h"])

    # Change-based features for the planning models
    disp_now = dispersion_proxy(wx["ventilation"].to_numpy())
    f["log_dispersion_change"] = np.log(f["dispersion_f"].to_numpy() / disp_now)
    f["activity_change"] = f["planned_construction_f"] - f["construction_now"]
    emission_now = (f["construction_now"] + 0.06 * f["trucks_30m"]).to_numpy() * disp_now
    f["log_emission_change"] = np.log1p(f["emission_index_f"].to_numpy()) - np.log1p(emission_now)
    f["pm10_anom_24h"] = np.log1p(pm10) - np.log1p(f["pm10_mean_24h"])
    f["local_enhancement_now"] = np.log1p(pm10) - np.log1p(f["reference_pm10"])
    f["local_share"] = 1 - f["reference_pm10"] / pm10.clip(lower=1.0)
    # Hybrid physics projection: regional forecast + current local increment scaled by the
    # planned change in (emissions x dispersion). Lets trees extrapolate in ratio space.
    local_now = (pm10 - f["reference_pm10"]).clip(lower=0.0).to_numpy()
    ratio = (f["emission_index_f"].to_numpy() + 0.05) / (emission_now + 0.05)
    projection = f["reference_f"].to_numpy() + local_now * np.clip(ratio, 0.0, 6.0)
    f["physics_projection_f"] = projection
    f["log_physics_ratio"] = np.log1p(projection) - np.log1p(pm10.to_numpy())

    tf = idx + pd.Timedelta(minutes=h * STEP_MIN)
    hour_f = tf.hour + tf.minute / 60.0
    f["hour_sin_f"] = np.sin(2 * np.pi * hour_f / 24)
    f["hour_cos_f"] = np.cos(2 * np.pi * hour_f / 24)
    f["workday_f"] = wx["workday"].shift(-h).to_numpy()
    hour_now = idx.hour + idx.minute / 60.0
    f["hour_sin"] = np.sin(2 * np.pi * hour_now / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour_now / 24)
    f["zone_code"] = float(ZONE_IDS.index(z))

    f["target_pm10"] = pm10.shift(-h)
    f["target_pm25"] = pm25.shift(-h)
    f["zone_id"] = z
    return f


def forecast_weather(obs: ObservedData, h: int) -> pd.DataFrame:
    """Weather forecast valid at t+h, issued at t (NWP proxy with growing error)."""
    wx = obs.weather
    idx = obs.index
    fut = wx.shift(-h)
    out = pd.DataFrame(index=idx)
    out["wind_speed"] = (fut["wind_speed"] * _nwp_noise(idx, h, 1.0, 1)).to_numpy()
    out["wind_dir_from"] = (fut["wind_dir_from"] + 25 * np.log(_nwp_noise(idx, h, 1.0, 2)) / 0.1).to_numpy() % 360
    out["mixing_height"] = (fut["mixing_height"] * _nwp_noise(idx, h, 1.0, 3)).to_numpy()
    out["ventilation"] = out["wind_speed"] * out["mixing_height"]
    out["temperature"] = (fut["temperature"] + 6 * np.log(_nwp_noise(idx, h, 1.0, 4))).to_numpy()
    out["relative_humidity"] = np.clip(fut["relative_humidity"] + 30 * np.log(_nwp_noise(idx, h, 1.0, 5)), 10, 99)
    out["rain_1h"] = (wx["rain_mm"].rolling(12, min_periods=1).sum().shift(-h) * _nwp_noise(idx, h, 2.0, 6)).to_numpy()
    # Regional air-quality forecast feed (e.g. CAMS / IITM early-warning system): background PM10 at t+h
    out["reference_pm10"] = (wx["reference_pm10"].shift(-h) * _nwp_noise(idx, h, 1.3, 7)).to_numpy()
    return out


def new_intervention_within(obs: ObservedData, z: str, h: int) -> np.ndarray:
    """True where a new intervention starts in zone z within (t, t+h]."""
    starts = np.zeros(len(obs.index))
    inter = obs.interventions
    if len(inter):
        for ts in inter.loc[inter["zone_id"] == z, "started_at"]:
            pos = obs.index.searchsorted(pd.Timestamp(ts))
            if 0 <= pos < len(starts):
                starts[pos] = 1
    cs = np.concatenate([[0], np.cumsum(starts)])
    n = len(starts)
    i = np.arange(n)
    hi = np.minimum(i + h, n - 1) + 1
    return (cs[hi] - cs[i + 1]) > 0


def build_features(obs: ObservedData, h: int, zones: Optional[List[str]] = None,
                   exclude_new_interventions: bool = True) -> pd.DataFrame:
    """Long feature frame (all zones) for horizon h, indexed by issue time."""
    zones = zones or ZONE_IDS
    wx_f = forecast_weather(obs, h)
    frames = []
    for z in zones:
        fz = _zone_frame(obs, z, h, wx_f)
        if exclude_new_interventions:
            fz["new_intervention"] = new_intervention_within(obs, z, h)
        else:
            fz["new_intervention"] = False
        frames.append(fz)
    out = pd.concat(frames)
    out.index.name = "issue_time"
    float_cols = [c for c in ALL_FEATURE_COLUMNS if c in out.columns]
    out[float_cols] = out[float_cols].astype("float32")
    return out


def feature_row(obs: ObservedData, t: pd.Timestamp, z: str, h: int) -> pd.DataFrame:
    """Single feature row for an issue time (used for live forecasts)."""
    window_start = t - pd.Timedelta(hours=26)
    window_end = t + pd.Timedelta(minutes=(h + 1) * STEP_MIN)
    sub = ObservedData(
        obs.pm10.loc[window_start:window_end], obs.pm25.loc[window_start:window_end],
        obs.weather.loc[window_start:window_end], obs.construction.loc[window_start:window_end],
        obs.planned_construction.loc[window_start:window_end], obs.trucks.loc[window_start:window_end],
        obs.intervention_active.loc[window_start:window_end], obs.surface_moisture.loc[window_start:window_end],
        obs.interventions, obs.qa, obs.meta,
    )
    f = build_features(sub, h, [z], exclude_new_interventions=False)
    return f.loc[[t], ALL_FEATURE_COLUMNS]


# ==================================================================================================
# Model Evaluation & Quantitative Metrics Suite.
# Implements standard atmospheric and statistical validation metrics:
# RMSE, MAE, R-squared, MAPE, Willmott's Index of Agreement (d),
# and Time-Series Walk-Forward Rolling-Origin Cross-Validation.
#
# Addresses Review Comment 5 (Quantitative Evaluation Metrics & Cross-Validation).
# ==================================================================================================

logger = logging.getLogger(__name__)


def compute_rmse(y_true: List[float], y_pred: List[float]) -> float:
    """Calculates Root Mean Squared Error (RMSE).

    RMSE penalizes large prediction errors more severely than MAE,
    making it essential for tracking severe pollution peak capture.
    """
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        return 0.0
    err = np.array(y_true, dtype=float) - np.array(y_pred, dtype=float)
    return float(np.sqrt(np.mean(err ** 2)))


def compute_mae(y_true: List[float], y_pred: List[float]) -> float:
    """Calculates Mean Absolute Error (MAE).

    Provides an unweighted, intuitive average magnitude of prediction errors.
    """
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        return 0.0
    err = np.abs(np.array(y_true, dtype=float) - np.array(y_pred, dtype=float))
    return float(np.mean(err))


def compute_r2_score(y_true: List[float], y_pred: List[float]) -> float:
    """Calculates Coefficient of Determination (R-squared).

    Measures the proportion of variance in observed air quality explained by the model.
    """
    if len(y_true) != len(y_pred) or len(y_true) < 2:
        return 0.0
    yt = np.array(y_true, dtype=float)
    yp = np.array(y_pred, dtype=float)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    if ss_tot == 0:
        return 1.0
    return float(1.0 - (ss_res / ss_tot))


def compute_mape(y_true: List[float], y_pred: List[float]) -> float:
    """Calculates Mean Absolute Percentage Error (MAPE)."""
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        return 0.0
    yt = np.array(y_true, dtype=float)
    yp = np.array(y_pred, dtype=float)
    # Avoid zero division
    denom = np.maximum(yt, 1.0)
    return float(np.mean(np.abs((yt - yp) / denom)) * 100.0)


def compute_index_of_agreement(y_true: List[float], y_pred: List[float]) -> float:
    """Calculates Willmott's Index of Agreement (d).

    A standardized measure of the degree of model prediction error in atmospheric
    modeling, bounded between 0 (no agreement) and 1 (perfect match).

    Formula:
        d = 1 - (sum((y_true - y_pred)^2) / sum((|y_pred - mean(y_true)| + |y_true - mean(y_true)|)^2))
    """
    if len(y_true) != len(y_pred) or len(y_true) < 2:
        return 0.0
    yt = np.array(y_true, dtype=float)
    yp = np.array(y_pred, dtype=float)
    mean_true = np.mean(yt)
    numerator = np.sum((yt - yp) ** 2)
    denominator = np.sum((np.abs(yp - mean_true) + np.abs(yt - mean_true)) ** 2)
    if denominator == 0:
        return 1.0
    return float(1.0 - (numerator / denominator))


def evaluate_forecast_performance(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
    """Generates a complete battery of evaluation metrics for a test split.

    Args:
        y_true: Ground truth observed pollutant concentrations.
        y_pred: Model predicted concentrations.

    Returns:
        Dict[str, float]: Dictionary of RMSE, MAE, R2, MAPE, and Index of Agreement.
    """
    return {
        "rmse": round(compute_rmse(y_true, y_pred), 2),
        "mae": round(compute_mae(y_true, y_pred), 2),
        "r2_score": round(compute_r2_score(y_true, y_pred), 3),
        "mape": round(compute_mape(y_true, y_pred), 2),
        "index_of_agreement": round(compute_index_of_agreement(y_true, y_pred), 3),
    }


def walk_forward_time_series_split(
    total_samples: int,
    initial_train_size: int,
    step_size: int = 24,
) -> Generator[Tuple[List[int], List[int]], None, None]:
    """Generates rolling-origin walk-forward cross-validation folds.

    Critical for temporal forecasting: standard k-fold randomly shuffles data,
    causing temporal data leakage from the future into the past. Walk-forward
    ensures strictly causal temporal integrity.

    Args:
        total_samples: Total number of sequential hourly observations.
        initial_train_size: Minimum initial training window size.
        step_size: Step size for advancing the origin (default: 24h).

    Yields:
        Tuple[List[int], List[int]]: (train_indices, test_indices)
    """
    current_train_end = initial_train_size
    while current_train_end + step_size <= total_samples:
        train_indices = list(range(0, current_train_end))
        test_indices = list(range(current_train_end, current_train_end + step_size))
        yield train_indices, test_indices
        current_train_end += step_size
