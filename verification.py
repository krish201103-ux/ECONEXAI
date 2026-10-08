"""
verification.py -- VERIFY and LEARN: did the intervention work, and what did we learn?

  verify_intervention()  observed before/after change at the zone sensors, kept
                         separate from the model-estimated effect against the
                         no-action counterfactual forecast (with its 90% band).
                         An effect is only called "verified" when the observation
                         falls below the counterfactual's lower bound.
  LearningLoop           a Bayesian field multiplier per action type
                         (measured = m x expected, prior m ~ N(1, 0.35^2)),
                         updated by every verified event; DUSTWISE scales its
                         expected reductions by the posterior mean.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from data_loader import ObservedData
from dustwise import ACTIONS
from ml_features import feature_row


# ==================================================================================================
# Impact verification ("Did the intervention work?").
# Two complementary measures are reported, and never conflated:
#
# * **Observed change** - before vs after at the zone sensors:
#   ``before`` = highest zone reading in the 30 minutes before the
#   intervention started, ``after`` = zone reading at verification time.
# * **Model-estimated effect** - observed after vs the *no-action counterfactual*
#   forecast issued at decision time (operational models are trained as
#   no-new-intervention baselines). The 90% conformal band of that forecast gives
#   an uncertainty range for the effect; the effect is only called "verified"
#   when the observation falls below the counterfactual's lower 90% bound.
#
# On synthetic data the true counterfactual is also known; it is reported
# separately as a validation of the method, never as the result.
# ==================================================================================================

def _interp_forecast(forecast: Dict[str, Dict[str, Any]], issued: pd.Timestamp, t: pd.Timestamp,
                     key: str, now_value: Optional[float] = None) -> Optional[float]:
    pts = [(0.0, now_value)] if now_value is not None else []
    for label, f in forecast.items():
        if "pm10" not in f:
            continue
        vt = pd.Timestamp(f["valid_time"])
        pts.append(((vt - issued).total_seconds() / 60, f["pm10"][key]))
    pts.sort()
    if not pts:
        return None
    m = (t - issued).total_seconds() / 60
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    if m < xs[0] or m > xs[-1]:
        return None
    return float(np.interp(m, xs, ys))


def verify_intervention(observed: pd.Series, forecast_at_decision: Dict[str, Dict[str, Any]], issued: pd.Timestamp,
                        started: pd.Timestamp, verify_at: pd.Timestamp, approved: pd.Timestamp,
                        detected: pd.Timestamp, water_rate_lpm: float, route_m: float,
                        true_counterfactual: Optional[pd.Series] = None, synthetic: bool = True) -> Dict[str, Any]:
    before_window = observed.loc[started - pd.Timedelta(minutes=30):started]
    before = float(before_window.max())
    before_time = before_window.idxmax()
    after = float(observed.loc[:verify_at].iloc[-1])
    observed_change = (after - before) / before
    cf = _interp_forecast(forecast_at_decision, issued, verify_at, "median")
    cf_lo = _interp_forecast(forecast_at_decision, issued, verify_at, "lo90")
    cf_hi = _interp_forecast(forecast_at_decision, issued, verify_at, "hi90")
    effect = (after - cf) / cf if cf else None
    effect_range = [(after - cf_hi) / cf_hi, (after - cf_lo) / cf_lo] if cf_lo and cf_hi else None
    verified = cf_lo is not None and after < cf_lo
    minutes_active = (verify_at - started).total_seconds() / 60
    series = observed.loc[detected - pd.Timedelta(minutes=60):verify_at + pd.Timedelta(minutes=10)]
    first_response = None
    post = observed.loc[started:verify_at]
    for t in post.index[1:]:
        if observed.loc[t] < before * 0.9:
            first_response = t
            break

    # predicted vs observed at the peak (forecast issued at decision time)
    peak_pred = _interp_forecast(forecast_at_decision, issued, before_time, "median",
                                 now_value=float(observed.loc[issued]))
    result: Dict[str, Any] = {
        "before_pm10": round(before, 1), "before_time": str(before_time),
        "after_pm10": round(after, 1), "after_time": str(verify_at),
        "observed_change_pct": round(100 * observed_change, 1),
        "counterfactual_pm10": round(cf, 1) if cf else None,
        "counterfactual_band": [round(cf_lo, 1), round(cf_hi, 1)] if cf_lo else None,
        "model_estimated_effect_pct": round(100 * effect, 1) if effect is not None else None,
        "model_estimated_effect_range_pct": [round(100 * effect_range[0], 1), round(100 * effect_range[1], 1)] if effect_range else None,
        "verified": bool(verified),
        "verdict": ("Effect verified - observed level is below the no-action forecast's 90% band"
                    if verified else "Inconclusive - observed level within the no-action forecast band"),
        "kpis": {
            "detection_to_action_min": int((started - detected).total_seconds() / 60),
            "approval_to_action_min": int((started - approved).total_seconds() / 60),
            "hotspot_duration_min": int((verify_at - detected).total_seconds() / 60),
            "first_response_time": str(first_response) if first_response is not None else None,
            "water_used_l": int(water_rate_lpm * minutes_active),
            "tanker_distance_km": round(route_m / 1000.0, 2),
            "peak_predicted_pm10": round(peak_pred, 1) if peak_pred else None,
            "peak_observed_pm10": round(before, 1),
            "peak_prediction_error_pct": round(100 * (peak_pred - before) / before, 1) if peak_pred else None,
        },
        "series": [{"time": str(t), "pm10": round(float(v), 1)} for t, v in series.items()],
        "data_label": "Demo simulation (synthetic data)" if synthetic else "Field measurement",
    }
    if true_counterfactual is not None and synthetic:
        tcf = float(true_counterfactual.loc[:verify_at].iloc[-1])
        result["validation_true_counterfactual_pm10"] = round(tcf, 1)
        result["validation_true_effect_pct"] = round(100 * (after - tcf) / tcf, 1)
    return result


# ==================================================================================================
# Closed learning loop.
# Prediction -> Intervention -> Observed outcome -> Prediction error ->
# Recommendation effectiveness -> Model refinement
#
# What is actually learned (no hand-waved "weight updates"):
#
# 1. **Field multiplier per action.** For every logged intervention DUSTWISE's
#    *expected* reduction is reconstructed (source shares at dispatch time x the
#    action's efficacy assumptions), and the *measured* reduction is estimated as
#    1 - observed / no-action forecast, 60 minutes after dispatch, using the
#    operational forecaster (trained without windows containing new
#    interventions, so it is a genuine no-action counterfactual). A Bayesian
#    regression through the origin, ``measured = m x expected + noise`` with
#    prior m ~ N(1, 0.35^2), gives a posterior multiplier per action type.
#    DUSTWISE multiplies its scenario estimates by it, so the recommendations
#    get more (or less) optimistic as field evidence accumulates.
#
#    Normalising by the expected reduction makes evidence comparable across
#    zones: spraying at a school perimeter with little local dust *should*
#    achieve little, and is not counted as a failure of the action.
#
# 2. **Forecast error tracking** - the peak prediction error of each verified
#    event is recorded.
#
# 3. **Refinement queue** - verified events are queued as labelled samples for
#    the next scheduled retrain; the posterior update itself is immediate.
# ==================================================================================================

logger = logging.getLogger(__name__)

EFFECT_HORIZON = "60m"
EFFECT_STEPS = 12
PRIOR_MEAN = 1.0
PRIOR_SD = 0.35


@dataclass
class MultiplierPosterior:
    action: str
    mean: float = PRIOR_MEAN
    sd: float = PRIOR_SD
    n: int = 0
    sum_xx: float = 0.0
    sum_xy: float = 0.0
    measured: List[float] = field(default_factory=list)
    expected: List[float] = field(default_factory=list)

    def update(self, expected_pct: float, measured_pct: float, noise_pct: float) -> None:
        self.sum_xx += expected_pct ** 2
        self.sum_xy += expected_pct * measured_pct
        prec = 1 / PRIOR_SD ** 2 + self.sum_xx / noise_pct ** 2
        self.mean = (PRIOR_MEAN / PRIOR_SD ** 2 + self.sum_xy / noise_pct ** 2) / prec
        self.sd = math.sqrt(1 / prec)
        self.n += 1
        self.measured.append(measured_pct)
        self.expected.append(expected_pct)

    def summary(self) -> Dict[str, Any]:
        return {
            "action": self.action, "label": ACTIONS[self.action]["label"], "n": self.n,
            "multiplier": round(self.mean, 3),
            "multiplier_lo": round(max(self.mean - 1.645 * self.sd, 0.05), 3),
            "multiplier_hi": round(self.mean + 1.645 * self.sd, 3),
            "median_measured_pct": round(float(np.median(self.measured)), 1) if self.measured else None,
            "median_expected_pct": round(float(np.median(self.expected)), 1) if self.expected else None,
            "evidence": "field data" if self.n >= 5 else "prior only (no field history yet)",
        }


def expected_reduction_pct(attribution: Dict[str, Any], action: str) -> float:
    shares = {c["source"]: c["share_pct"] / 100 for c in attribution["contributors"]}
    local_fraction = attribution["local_excess_pm10"] / max(attribution["observed_pm10"], 1.0)
    return 100 * local_fraction * sum(shares.get(s, 0) * r for s, r in ACTIONS[action]["reduction"].items())


class LearningLoop:
    def __init__(self) -> None:
        self.posteriors: Dict[str, MultiplierPosterior] = {k: MultiplierPosterior(k) for k in ACTIONS}
        self.records: List[Dict[str, Any]] = []
        self.noise_pct = 12.0
        self.history: List[Dict[str, Any]] = []

    def bootstrap_from_history(self, obs: ObservedData, forecaster, attributor, end: pd.Timestamp,
                               max_records: int = 400) -> None:
        """Estimate the field effect of every logged historical intervention and update posteriors."""
        model = forecaster.models[EFFECT_HORIZON]["pm10"]
        self.noise_pct = max(6.0, 100 * model.q_conformal / 1.645)
        inter = obs.interventions
        inter = inter[(inter["source"] == "historical") & (inter["dispatched_at"] < end)].tail(max_records)
        for _, rec in inter.iterrows():
            z = rec["zone_id"]
            t = pd.Timestamp(rec["dispatched_at"]).floor("5min")
            target_t = t + pd.Timedelta(minutes=5 * EFFECT_STEPS)
            if t not in obs.index or target_t not in obs.index:
                continue
            try:
                X = feature_row(obs, t, z, EFFECT_STEPS)
            except Exception:
                continue
            if X[model.features].isna().any(axis=None):
                continue
            med, _, _ = model.predict(X)
            observed = float(obs.pm10[z].loc[target_t])
            measured = 100 * (1 - observed / max(float(med[0]), 1.0))
            att = attributor.explain(obs, z, t)
            expected = expected_reduction_pct(att, rec["action"])
            self.posteriors[rec["action"]].update(expected, measured, self.noise_pct)
            self.records.append({"zone_id": z, "action": rec["action"], "dispatched_at": str(t),
                                 "observed": round(observed, 1), "counterfactual": round(float(med[0]), 1),
                                 "measured_pct": round(measured, 1), "expected_pct": round(expected, 1),
                                 "effect_pct": round(-measured, 1)})
        logger.info("Learning loop bootstrapped from %d interventions: %s", len(self.records),
                    {k: (v.n, round(v.mean, 2)) for k, v in self.posteriors.items()})

    def efficacy_table(self) -> Dict[str, Dict[str, Any]]:
        return {k: v.summary() for k, v in self.posteriors.items()}

    def record_outcome(self, action: str, observed: float, counterfactual: float, expected_pct: float,
                       event_id: str, prediction_error_pct: Optional[float]) -> Dict[str, Any]:
        before = self.posteriors[action].summary()
        measured = 100 * (1 - observed / max(counterfactual, 1.0))
        self.posteriors[action].update(expected_pct, measured, self.noise_pct)
        after = self.posteriors[action].summary()
        entry = {
            "event_id": event_id, "action": action, "label": ACTIONS[action]["label"],
            "expected_pct": round(expected_pct, 1), "measured_pct": round(measured, 1),
            "posterior_before": before, "posterior_after": after,
            "prediction_error_pct": prediction_error_pct,
            "refinement": "Labelled sample queued for the next scheduled retrain (Sun 02:00)",
        }
        self.history.append(entry)
        return entry

    def summary(self) -> Dict[str, Any]:
        measured = [r["measured_pct"] for r in self.records]
        good = [r for r in self.records if r["measured_pct"] >= 0.5 * max(r["expected_pct"], 1.0)]
        return {
            "interventions_evaluated": len(self.records),
            "recommendation_effectiveness_pct": round(100 * len(good) / max(len(self.records), 1), 1),
            "effectiveness_definition": "share of interventions achieving at least half of the expected reduction",
            "median_measured_reduction_pct": round(float(np.median(measured)), 1) if measured else None,
            "efficacy": self.efficacy_table(),
            "updates": self.history,
            "noise_pct": round(self.noise_pct, 1),
            "stages": ["Prediction", "Intervention", "Observed outcome", "Prediction error",
                       "Recommendation effectiveness", "Model refinement"],
            "recent": self.records[-12:],
        }
