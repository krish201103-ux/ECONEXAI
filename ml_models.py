"""
ml_models.py -- Multi-horizon PM10 / PM2.5 forecaster and the walk-forward benchmark.

  MultiHorizonForecaster  one gradient-boosted model per (horizon, target):
                          18 horizons (15 min - 72 h) x PM10/PM2.5, each predicting
                          the log-change from the current zone level, with
                          split-conformal 90% prediction intervals.
  run_benchmark()         rolling-origin cross-validation against persistence,
                          seasonal-naive, diurnal climatology, ridge and random
                          forest -- same features and target for every learned
                          model, so the comparison isolates the algorithm.

XGBoost is used when installed, otherwise scikit-learn's
HistGradientBoostingRegressor (backend_name() reports which one ran). Every
number in artifacts/evaluation.json and artifacts/benchmark.json is computed
here by train.py; nothing is typed in by hand.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data_loader import ObservedData, ZONE_IDS
from ml_features import (ALL_HORIZONS, FEATURES, STEP_MIN, build_features, evaluate_forecast_performance, feature_row,
                         features_for, is_planning)


# ==================================================================================================
# Multi-horizon PM10 / PM2.5 forecaster.
# Direct multi-horizon strategy: one gradient-boosted model per (horizon, target).
# Every horizon predicts the log-change from the current zone level; planning
# horizons (2-72 h) use a change-based feature set (see features.py). Uncertainty is quantified with *split conformal
# prediction* on a held-out calibration window, which gives distribution-free
# 90% prediction intervals whose empirical coverage is then checked on a later
# test window (reported, not assumed).
#
# Operational forecasts (15-90 min) are **no-new-action baselines**: training
# rows in which a new intervention started inside the forecast window are
# excluded, so the model answers "what happens if we do nothing new?". This is
# exactly the quantity DUSTWISE needs to prioritise, and the counterfactual used
# for verification. Planning forecasts (2-72 h) are business-as-usual forecasts
# (see ``_exclude_new``).
#
# Backend: XGBoost (``tree_method=hist``) when installed, otherwise scikit-learn's
# HistGradientBoostingRegressor. The backend actually used is recorded in the
# model metadata and in every API response.
# ==================================================================================================

logger = logging.getLogger(__name__)

TARGETS = ("pm10", "pm25")
COVERAGE = 0.90

try:  # pragma: no cover - depends on environment
    import xgboost as xgb
    HAS_XGB = True
except Exception:  # pragma: no cover
    HAS_XGB = False


def make_gbm(n_estimators: int = 350, seed: int = 42, planning: bool = False):
    """Gradient-boosted tree regressor (XGBoost if available).

    Planning models are shallower and more regularised: their signal-to-noise
    ratio is lower and over-fitting to season-specific patterns is the main risk.
    """
    depth, mcw, lam, lr = (4, 20, 5.0, 0.05) if planning else (6, 5, 1.0, 0.06)
    if HAS_XGB:
        return xgb.XGBRegressor(
            n_estimators=n_estimators, max_depth=depth, learning_rate=lr, subsample=0.85,
            colsample_bytree=0.8, min_child_weight=mcw, reg_lambda=lam, tree_method="hist", max_bin=128,
            n_jobs=2, random_state=seed, verbosity=0,
        )
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=n_estimators, learning_rate=lr, max_depth=depth,
                                         l2_regularization=lam, min_samples_leaf=mcw, random_state=seed)


def backend_name() -> str:
    return f"XGBoost {xgb.__version__} (hist)" if HAS_XGB else "scikit-learn HistGradientBoosting"


def target_mode(h: int) -> str:
    """Targets are log-ratios to an anchor, never absolute levels.

    * operational (<= 90 min): anchor = current zone level        ("delta")
    * planning   (> 90 min)  : anchor = current zone level as well; the
      planning feature set is change-based (see features.py)

    Predicting a relative change keeps the trees robust to seasonal regime
    shifts (e.g. the post-monsoon rise in October) that absolute-level trees
    cannot extrapolate to.
    """
    return "delta"


def anchor_column(target: str, mode: str = "delta") -> str:
    return f"{target}_now"


def encode_target(y: np.ndarray, anchor: np.ndarray, mode: str = "delta") -> np.ndarray:
    return np.log1p(y) - np.log1p(anchor)


def decode_target(z: np.ndarray, anchor: np.ndarray, mode: str = "delta") -> np.ndarray:
    return np.expm1(z + np.log1p(anchor))


@dataclass
class HorizonModel:
    label: str
    steps: int
    target: str
    model: object
    mode: str
    q_conformal: float                # log-space half width for 90% interval
    n_train: int
    n_cal: int
    train_seconds: float
    features: List[str] = field(default_factory=lambda: list(FEATURES))

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        now = X[anchor_column(self.target, self.mode)].to_numpy(dtype=float)
        z = self.model.predict(X[self.features])
        med = decode_target(z, now, self.mode)
        lo = decode_target(z - self.q_conformal, now, self.mode)
        hi = decode_target(z + self.q_conformal, now, self.mode)
        return np.maximum(med, 0), np.maximum(lo, 0), hi


@dataclass
class MultiHorizonForecaster:
    horizons: Dict[str, int] = field(default_factory=lambda: dict(ALL_HORIZONS))
    targets: Tuple[str, ...] = TARGETS
    n_estimators: int = 350
    stride: int = 2
    models: Dict[str, Dict[str, HorizonModel]] = field(default_factory=dict)
    meta: Dict[str, object] = field(default_factory=dict)

    @staticmethod
    def _exclude_new(h: int) -> bool:
        """Operational horizons (<= 90 min) are trained as no-new-action baselines.

        Planning horizons are business-as-usual forecasts: on a busy site some
        intervention starts within almost every 24 h window, and excluding those
        windows would bias training towards unusually clean days.
        """
        return h <= 18

    # ------------------------------------------------------------------ #
    def fit(self, obs: ObservedData, train_end: pd.Timestamp, cal_end: pd.Timestamp) -> "MultiHorizonForecaster":
        t0 = time.time()
        for label, h in self.horizons.items():
            F = build_features(obs, h)
            feats = features_for(h)
            target_time = F.index + pd.Timedelta(minutes=h * STEP_MIN)
            usable = F[feats].notna().all(axis=1) & ~(F["new_intervention"] & self._exclude_new(h))
            train_mask = usable & (target_time <= train_end)
            cal_mask = usable & (F.index > train_end) & (target_time <= cal_end)
            self.models[label] = {}
            for target in self.targets:
                y = F[f"target_{target}"]
                tm = train_mask & y.notna()
                cm = cal_mask & y.notna()
                tr = F[tm].iloc[:: self.stride]
                ca = F[cm]
                mode = target_mode(h)
                anchor = anchor_column(target, mode)
                ytr = encode_target(tr[f"target_{target}"].to_numpy(float), tr[anchor].to_numpy(float), mode)
                start = time.time()
                model = make_gbm(self.n_estimators, planning=is_planning(h))
                model.fit(tr[feats], ytr)
                secs = time.time() - start
                zc = model.predict(ca[feats])
                yc = encode_target(ca[f"target_{target}"].to_numpy(float), ca[anchor].to_numpy(float), mode)
                res = np.abs(yc - zc)
                n = len(res)
                k = min(n, int(math.ceil((n + 1) * COVERAGE)))
                q = float(np.sort(res)[k - 1]) if n else 0.25
                self.models[label][target] = HorizonModel(label, h, target, model, mode, q, len(tr), n, secs, list(feats))
            logger.info(f"  horizon {label:>4}: trained {len(self.targets)} targets "
                        f"(n_train={self.models[label]['pm10'].n_train:,}, q90={self.models[label]['pm10'].q_conformal:.3f})")
        self.meta = {
            "backend": backend_name(), "train_end": str(train_end), "calibration_end": str(cal_end),
            "horizons": self.horizons, "targets": list(self.targets), "n_features": len(FEATURES),
            "features": FEATURES, "fit_seconds": round(time.time() - t0, 1), "interval": "split-conformal 90%",
            "baseline_definition": "no-new-intervention forecast",
        }
        return self

    # ------------------------------------------------------------------ #
    def predict_zone(self, obs: ObservedData, t: pd.Timestamp, zone: str,
                     labels: Optional[List[str]] = None) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Forecast for one zone issued at time t for the requested horizons."""
        labels = labels or list(self.horizons)
        out: Dict[str, Dict[str, Dict[str, float]]] = {}
        for label in labels:
            h = self.horizons[label]
            X = feature_row(obs, t, zone, h)
            out[label] = {"valid_time": str(t + pd.Timedelta(minutes=h * STEP_MIN)), "steps": h}
            for target in self.targets:
                med, lo, hi = self.models[label][target].predict(X)
                out[label][target] = {"median": round(float(med[0]), 1), "lo90": round(float(lo[0]), 1),
                                      "hi90": round(float(hi[0]), 1)}
        self.smooth_path(out, labels, self.targets)
        return out

    @staticmethod
    def smooth_path(out: Dict[str, Dict[str, object]], labels: List[str], targets) -> None:
        """Light horizon smoothing (log-space [1/4, 1/2, 1/4] kernel) within each tier.

        Direct per-horizon models are fitted independently, so neighbouring
        horizons can disagree by more than their true difference; smoothing
        across horizons reduces that variance without introducing lag.
        """
        for tier in (False, True):
            tier_labels = [l for l in labels if is_planning(ALL_HORIZONS.get(l, 0)) == tier and l in out]
            tier_labels.sort(key=lambda l: ALL_HORIZONS[l])
            if len(tier_labels) < 3:
                continue
            for target in targets:
                for key in ("median", "lo90", "hi90"):
                    vals = np.log1p([out[l][target][key] for l in tier_labels])
                    sm = vals.copy()
                    for i in range(len(vals)):
                        if i == 0:
                            continue                      # keep the shortest horizon anchored to the data
                        elif i == len(vals) - 1:
                            sm[i] = (2 * vals[i] + vals[i - 1]) / 3
                        else:
                            sm[i] = 0.25 * vals[i - 1] + 0.5 * vals[i] + 0.25 * vals[i + 1]
                    for l, v in zip(tier_labels, sm):
                        out[l][target][key] = round(float(np.expm1(v)), 1)

    def predict_frame(self, F: pd.DataFrame, label: str, target: str = "pm10") -> pd.DataFrame:
        med, lo, hi = self.models[label][target].predict(F)
        return pd.DataFrame({"median": med, "lo90": lo, "hi90": hi}, index=F.index)

    # ------------------------------------------------------------------ #
    def evaluate(self, obs: ObservedData, test_start: pd.Timestamp, test_end: pd.Timestamp,
                 labels: Optional[List[str]] = None, exceedance: float = 150.0) -> Dict[str, object]:

        labels = labels or list(self.horizons)
        results: Dict[str, object] = {}
        for label in labels:
            h = self.horizons[label]
            F = build_features(obs, h)
            target_time = F.index + pd.Timedelta(minutes=h * STEP_MIN)
            mask = (F[features_for(h)].notna().all(axis=1) & ~(F["new_intervention"] & self._exclude_new(h))
                    & (F.index >= test_start) & (target_time <= test_end))
            T = F[mask]
            res_h: Dict[str, object] = {}
            for target in self.targets:
                Tt = T[T[f"target_{target}"].notna()]
                y = Tt[f"target_{target}"].to_numpy(float)
                med, lo, hi = self.models[label][target].predict(Tt)
                persist = Tt[f"{target}_now"].to_numpy(float)
                m = evaluate_forecast_performance(y.tolist(), med.tolist())
                mp = evaluate_forecast_performance(y.tolist(), persist.tolist())
                m["coverage_90"] = round(float(np.mean((y >= lo) & (y <= hi))), 3)
                m["mean_interval_width"] = round(float(np.mean(hi - lo)), 1)
                m["skill_vs_persistence"] = round(1 - m["rmse"] / max(mp["rmse"], 1e-9), 3)
                m["persistence_rmse"] = mp["rmse"]
                m["n_test"] = int(len(y))
                if target == "pm10":
                    obs_ex = y > exceedance
                    pred_ex = med > exceedance
                    hits = int(np.sum(obs_ex & pred_ex))
                    misses = int(np.sum(obs_ex & ~pred_ex))
                    fas = int(np.sum(~obs_ex & pred_ex))
                    m["exceedance_threshold"] = exceedance
                    m["pod"] = round(hits / max(hits + misses, 1), 3)
                    m["far"] = round(fas / max(hits + fas, 1), 3)
                    m["csi"] = round(hits / max(hits + misses + fas, 1), 3)
                res_h[target] = m
            results[label] = res_h
        return results

    # ------------------------------------------------------------------ #
    def save(self, path) -> None:
        """Write the models in a portable, crash-safe way:

        * each XGBoost model is stored in XGBoost's own model format (UBJSON), which newer XGBoost
          releases are guaranteed to read -- a pickled Python object is not;
        * the file is written to a temporary name and then renamed, so an interrupted training run
          (Ctrl+C, a sync client locking the file) never leaves a half-written model behind;
        * its SHA-256 is recorded in artifacts/manifest.json, so a file that is damaged later is
          reported as damaged instead of failing deep inside XGBoost.
        """
        import joblib
        path = Path(path)
        packed = {label: {t: replace(hm, model=_pack_estimator(hm.model)) for t, hm in by_t.items()}
                  for label, by_t in self.models.items()}
        blob = {"format": MODEL_FORMAT, "horizons": self.horizons, "targets": self.targets, "models": packed,
                "meta": self.meta}
        tmp = path.with_name(path.name + ".tmp")
        joblib.dump(blob, tmp, compress=3)
        os.replace(tmp, path)
        record_artifact(path)

    @classmethod
    def load(cls, path) -> "MultiHorizonForecaster":
        """Load models saved by save() (or by earlier versions). Any problem -- a missing, damaged or
           incompatible file -- raises ModelLoadError with a message that says how to fix it."""
        import joblib
        path = Path(path)
        if not path.exists():
            raise ModelLoadError(f"{path.name} not found. Train the models with: python server.py --retrain")
        problem = verify_artifact(path)
        if problem:
            raise ModelLoadError(f"{path.name} is damaged or incomplete ({problem}). "
                                 "Retrain the models with: python server.py --retrain")
        try:
            blob = joblib.load(path)
            f = cls(horizons=blob["horizons"], targets=tuple(blob["targets"]))
            f.models = {label: {t: replace(hm, model=_unpack_estimator(hm.model)) for t, hm in by_t.items()}
                        for label, by_t in blob["models"].items()}
            f.meta = blob["meta"]
        except ModelLoadError:
            raise
        except Exception as e:  # damaged bytes, or written by an incompatible library version
            raise ModelLoadError(
                f"{path.name} could not be read ({type(e).__name__}: {str(e).strip().splitlines()[0][:160]}). "
                "The file is damaged, incomplete, or was written by an incompatible library version. "
                "Retrain the models with: python server.py --retrain") from e
        return f


# ---------------------------------------------------------------------------------------------------
# Model files: portable format + integrity record
# ---------------------------------------------------------------------------------------------------
MODEL_FORMAT = "econexus-forecaster/2"     # 2 = native XGBoost UBJSON per model (1 = pickled estimators)
MANIFEST_NAME = "manifest.json"


class ModelLoadError(RuntimeError):
    """The saved models cannot be used. The message is safe to show and says how to fix it."""


def _pack_estimator(est):
    if HAS_XGB and isinstance(est, xgb.XGBModel):
        return {"format": "xgboost-ubj", "raw": bytes(est.get_booster().save_raw(raw_format="ubj"))}
    return {"format": "pickle", "estimator": est}


def _unpack_estimator(packed):
    if isinstance(packed, dict) and packed.get("format") == "xgboost-ubj":
        if not HAS_XGB:
            raise ModelLoadError("These models were trained with XGBoost, which is not installed. "
                                 "Run: python server.py --reinstall (or --retrain to use scikit-learn instead)")
        reg = xgb.XGBRegressor()
        reg.load_model(bytearray(packed["raw"]))
        return reg
    if isinstance(packed, dict) and packed.get("format") == "pickle":
        return packed["estimator"]
    return packed  # format 1: the estimator object itself was pickled


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest_path(path: Path) -> Path:
    return Path(path).with_name(MANIFEST_NAME)


def record_artifact(path) -> None:
    """Store the size, SHA-256 and library versions of a model file in artifacts/manifest.json."""
    import platform
    import sklearn
    path = Path(path)
    mpath = _manifest_path(path)
    try:
        manifest = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}
    except (ValueError, OSError):
        manifest = {}
    manifest.setdefault("files", {})[path.name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    manifest["format"] = MODEL_FORMAT
    manifest["written_by"] = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                              "scikit-learn": sklearn.__version__, "xgboost": xgb.__version__ if HAS_XGB else None}
    tmp = mpath.with_name(mpath.name + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(tmp, mpath)


def verify_artifact(path) -> Optional[str]:
    """None if the file matches artifacts/manifest.json (or no record exists), else what is wrong."""
    path = Path(path)
    mpath = _manifest_path(path)
    try:
        rec = json.loads(mpath.read_text(encoding="utf-8")).get("files", {}).get(path.name) if mpath.exists() else None
    except (ValueError, OSError):
        return None
    if not rec:
        return None
    size = path.stat().st_size
    if size != rec.get("bytes"):
        return f"size {size:,} bytes, expected {rec.get('bytes'):,}"
    if _sha256(path) != rec.get("sha256"):
        return "checksum does not match the one recorded at training time"
    return None


# ==================================================================================================
# Walk-forward benchmark of the forecaster against baselines.
# Every number in the benchmark table is computed here, at training time, from
# the dataset in use. Nothing is hard-coded.
#
# Protocol
# --------
# Rolling-origin (expanding window) cross-validation: for each fold the models are
# trained on everything whose *target time* precedes the fold's test window and
# evaluated on the following window. Folds never shuffle time, so there is no
# leakage from the future.
#
# Models
# ------
# * Persistence            - the current value
# * Seasonal naive         - the value at the same time of day, 24 h earlier
# * Diurnal climatology    - zone x hour-of-day mean of the training window
# * Ridge regression       - L2 linear model on standardised features
# * Random forest          - bagged trees
# * XGBoost (proposed)     - gradient-boosted trees (production model family)
#
# All learned models use the same features and the same log-change target as the
# production forecaster, so the comparison isolates the learning algorithm.
# ==================================================================================================


DEFAULT_FOLDS: List[Tuple[str, str]] = [
    ("2026-08-27", "2026-09-05 23:55"),
    ("2026-09-06", "2026-09-15 23:55"),
    ("2026-09-16", "2026-09-25 23:55"),
    ("2026-09-26", "2026-10-05 23:55"),
]
DEFAULT_HORIZONS = {"60m": 12, "6h": 72, "24h": 288}

MODEL_INFO = {
    "persistence": ("Persistence", "Naive baseline"),
    "seasonal_naive": ("Seasonal naive (t-24h)", "Naive baseline"),
    "climatology": ("Diurnal climatology", "Statistical baseline"),
    "ridge": ("Ridge regression", "L2-regularised linear model"),
    "random_forest": ("Random forest", "Bagged decision trees"),
    "xgboost": ("XGBoost (proposed)", "Gradient-boosted trees"),
}


def _ridge():
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), Ridge(alpha=3.0))


def _forest():
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=80, max_depth=16, min_samples_leaf=5, max_features=0.5,
                                 max_samples=0.4, n_jobs=2, random_state=42)


def run_benchmark(obs: ObservedData, horizons: Dict[str, int] = None, folds: List[Tuple[str, str]] = None,
                  stride: int = 3, n_estimators: int = 300) -> Dict[str, object]:
    horizons = horizons or DEFAULT_HORIZONS
    folds = folds or DEFAULT_FOLDS
    t_start = time.time()
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    timings: Dict[str, Dict[str, List[float]]] = {m: {"train": [], "latency_ms": []} for m in MODEL_INFO}

    for label, h in horizons.items():
        F = build_features(obs, h)
        feats = features_for(h)
        target_time = F.index + pd.Timedelta(minutes=h * STEP_MIN)
        exclude = F["new_intervention"] if h <= 18 else False
        usable = F[feats].notna().all(axis=1) & F["target_pm10"].notna() & ~exclude
        preds: Dict[str, List[np.ndarray]] = {m: [] for m in MODEL_INFO}
        ys: List[np.ndarray] = []
        for test_start, test_end in folds:
            ts, te = pd.Timestamp(test_start), pd.Timestamp(test_end)
            tr = F[usable & (target_time < ts)].iloc[::stride]
            T = F[usable & (F.index >= ts) & (target_time <= te)]
            if len(T) == 0 or len(tr) == 0:
                continue
            y = T["target_pm10"].to_numpy(float)
            now = T["pm10_now"].to_numpy(float)
            ys.append(y)
            ztr = np.log1p(tr["target_pm10"].to_numpy(float)) - np.log1p(tr["pm10_now"].to_numpy(float))

            preds["persistence"].append(now)
            preds["seasonal_naive"].append(T["pm10_same_time_yday"].fillna(T["pm10_now"]).to_numpy(float))
            tgt_hour = ((T.index + pd.Timedelta(minutes=h * STEP_MIN)).hour).to_numpy()
            tr_hour = ((tr.index + pd.Timedelta(minutes=h * STEP_MIN)).hour).to_numpy()
            clim = pd.DataFrame({"zone": tr["zone_id"].to_numpy(), "hour": tr_hour, "y": tr["target_pm10"].to_numpy()})
            table = clim.groupby(["zone", "hour"])["y"].mean()
            preds["climatology"].append(np.array([table.get((z, hr), np.nan) for z, hr in zip(T["zone_id"], tgt_hour)]))

            for key, factory in (("ridge", _ridge), ("random_forest", _forest),
                                 ("xgboost", lambda: make_gbm(n_estimators, planning=is_planning(h)))):
                model = factory()
                t0 = time.time()
                model.fit(tr[feats], ztr)
                timings[key]["train"].append(time.time() - t0)
                z = model.predict(T[feats])
                preds[key].append(np.expm1(z + np.log1p(now)))
                one = T[feats].iloc[[0]]
                lat = []
                for _ in range(15):
                    t1 = time.perf_counter()
                    model.predict(one)
                    lat.append((time.perf_counter() - t1) * 1000)
                timings[key]["latency_ms"].append(float(np.median(lat)))

        y_all = np.concatenate(ys)
        res_h: Dict[str, Dict[str, float]] = {}
        persist_rmse = None
        for key in MODEL_INFO:
            p = np.concatenate(preds[key])
            ok = ~np.isnan(p)
            m = evaluate_forecast_performance(y_all[ok].tolist(), p[ok].tolist())
            if key == "persistence":
                persist_rmse = m["rmse"]
            m["skill_vs_persistence"] = round(1 - m["rmse"] / persist_rmse, 3) if persist_rmse else 0.0
            m["n"] = int(ok.sum())
            res_h[key] = m
        results[label] = res_h
        logger.info(f"  benchmark {label}: " + ", ".join(f"{k}={v['rmse']}" for k, v in res_h.items()))

    rows = []
    for key, (name, family) in MODEL_INFO.items():
        row = {"key": key, "model_name": name, "family": family,
               "selected_for_production": key == "xgboost",
               "train_seconds": round(float(np.mean(timings[key]["train"])), 2) if timings[key]["train"] else 0.0,
               "latency_ms": round(float(np.mean(timings[key]["latency_ms"])), 2) if timings[key]["latency_ms"] else 0.0}
        for label in horizons:
            for metric in ("rmse", "mae", "r2_score", "mape", "index_of_agreement", "skill_vs_persistence"):
                row[f"{metric}_{label}"] = results[label][key][metric]
        rows.append(row)

    return {
        "protocol": "rolling-origin walk-forward CV (expanding window)",
        "folds": [{"test_start": a, "test_end": b} for a, b in folds],
        "horizons": horizons,
        "target": "zone-level PM10 (ug/m3)",
        "xgboost_backend": backend_name(),
        "results": results,
        "table": rows,
        "runtime_seconds": round(time.time() - t_start, 1),
    }
