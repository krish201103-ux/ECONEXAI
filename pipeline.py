"""
pipeline.py -- Builds the complete platform state from the trained models.

Builds the complete operational picture at a point in time from the trained
models: zone status, forecasts with intervals, events and their lifecycle,
AI insight text, DUSTWISE priorities / actions / fleet allocation, sensor
health, verification, learning loop and long-term sustainability metrics.

The same ``Platform`` object backs the REST API (server.py, through
forecast_service.py) and the static snapshot the web UI uses when no server is
running (train.export_snapshot), so both always show identical model output.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

import config
import digital_twin as twin
import site_model as site
from data_loader import ObservedData, ZONE_IDS, prepare_from_synthetic
from dustwise import ACTIONS, DustwiseEngine
from explainability import SourceAttributor, exact_group_shapley
from intelligence import (ACTION_LEVELS, CPCB_PM10_24H, PollutionEvent, TimelineEntry, detect_events, fleet_snapshot,
                          fmt0, level_for, level_rank, severity_for, sgn0, summarise_detected)
from ml_features import (ALL_HORIZONS, GROUP_LABELS, UI_TABS, build_features, feature_row, forecast_weather,
                         groups_for)
from ml_models import ModelLoadError, MultiHorizonForecaster, backend_name
from simulator import HERO_DATE, NOW, generate_dataset
from verification import LearningLoop, expected_reduction_pct, verify_intervention

logger = logging.getLogger("econexus.pipeline")
ARTIFACTS = config.ARTIFACTS_DIR

HERO_EVENT_ID = "EV-042"
HERO_ZONE = "B"
HERO_TIMES = {
    "detected": pd.Timestamp("2026-10-07 14:20"),
    "classified": pd.Timestamp("2026-10-07 14:24"),
    "predicted": pd.Timestamp("2026-10-07 14:27"),
    "recommended": pd.Timestamp("2026-10-07 14:29"),
    "approved": pd.Timestamp("2026-10-07 14:33"),
    "departed": pd.Timestamp("2026-10-07 14:38"),
    "initiated": pd.Timestamp("2026-10-07 14:42"),
    "response": pd.Timestamp("2026-10-07 15:05"),
    "verified": pd.Timestamp("2026-10-07 15:20"),
}
TAB_PATHS = {
    "15m": ["15m"], "30m": ["15m", "30m"], "60m": ["15m", "30m", "45m", "60m"],
    "90m": ["15m", "30m", "45m", "60m", "75m", "90m"],
    "6h": ["15m", "30m", "60m", "90m", "2h", "3h", "4h", "6h"],
    "24h": ["30m", "90m", "3h", "6h", "9h", "12h", "18h", "24h"],
    "72h": ["90m", "6h", "12h", "18h", "24h", "36h", "48h", "60h", "72h"],
}
TAB_HISTORY_H = {"15m": 2, "30m": 2, "60m": 3, "90m": 3, "6h": 12, "24h": 36, "72h": 96}
TAB_RES_MIN = {"15m": 5, "30m": 5, "60m": 5, "90m": 5, "6h": 15, "24h": 60, "72h": 120}


def _r(x: Optional[float], n: int = 1) -> Optional[float]:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    return round(float(x), n)


def json_safe(o: Any) -> Any:
    """Recursively converts numpy scalars to Python numbers and NaN/inf to None, so any platform
       output can go straight into a JSON response (or the offline snapshot)."""
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, np.floating):
        o = float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o


def _fmt_time(t: pd.Timestamp) -> str:
    return pd.Timestamp(t).strftime("%H:%M")


class Platform:
    """Loads data + models and assembles platform state."""

    def __init__(self, artifacts: Path = ARTIFACTS, seed: int = 42, train_if_missing: bool = True):
        t0 = time.time()
        self.artifacts = artifacts
        self.ds = generate_dataset(seed=seed)
        self.obs = prepare_from_synthetic(self.ds, cutoff=NOW)
        self.obs_after = prepare_from_synthetic(self.ds)          # includes the hero intervention once approved
        self.now = NOW
        fpath = artifacts / "forecaster.joblib"
        try:
            self.forecaster = MultiHorizonForecaster.load(fpath)
        except ModelLoadError as e:
            if not train_if_missing:
                raise
            # Last-resort safety net (bootstrap.py normally retrains before we get here): keep the
            # platform usable with a smaller model trained now, and say so in the console.
            logger.warning("%s -- training a quick replacement model now (about a minute). "
                           "For the full models run: python server.py --retrain", e)
            from train import PROD_CAL_END, PROD_TRAIN_END
            self.forecaster = MultiHorizonForecaster(n_estimators=120).fit(self.obs, PROD_TRAIN_END, PROD_CAL_END)
        apath = artifacts / "attribution.joblib"
        try:
            self.attributor: SourceAttributor = joblib.load(apath)
        except Exception as e:  # missing or unreadable -- refitting takes a few seconds
            if apath.exists():
                logger.warning("%s could not be read (%s) -- refitting the attribution model.", apath.name, e)
            self.attributor = SourceAttributor().fit(self.obs, pd.Timestamp("2026-10-06 23:55"))
        self.model_card = self._load_json("model_card.json")
        self.benchmark = self._load_json("benchmark.json")
        self.evaluation = self._load_json("evaluation.json")
        self.engine = DustwiseEngine()
        self.learning = LearningLoop()
        self.learning.bootstrap_from_history(self.obs, self.forecaster, self.attributor, NOW)
        self.sensors = fleet_snapshot(self.ds.sensor_frame, NOW)
        self._forecast_cache: Dict[tuple, Dict[str, Any]] = {}
        logger.info("Platform ready in %.1f s (backend: %s)", time.time() - t0, self.forecaster.meta.get("backend"))

    def _load_json(self, name: str) -> Dict[str, Any]:
        p = self.artifacts / name
        return json.loads(p.read_text()) if p.exists() else {}

    # ------------------------------------------------------------------ #
    # Basic readings
    # ------------------------------------------------------------------ #
    def zone_reading(self, z: str, t: pd.Timestamp, obs: Optional[ObservedData] = None) -> Dict[str, Any]:
        obs = obs or self.obs
        pm10 = float(obs.pm10[z].loc[t])
        pm25 = float(obs.pm25[z].loc[t])
        p30 = float(obs.pm10[z].loc[t - pd.Timedelta(minutes=30)])
        p60 = float(obs.pm10[z].loc[t - pd.Timedelta(minutes=60)])
        lvl = level_for(pm10)
        return {"pm10": _r(pm10), "pm25": _r(pm25), "change_30m_pct": _r(100 * (pm10 - p30) / p30),
                "change_60m_pct": _r(100 * (pm10 - p60) / p60), "level": lvl["key"], "level_label": lvl["label"],
                "level_color": lvl["color"]}

    def weather_now(self, t: pd.Timestamp) -> Dict[str, Any]:
        w = self.obs.weather.loc[t]
        return {"temperature": _r(w["temperature"]), "relative_humidity": _r(w["relative_humidity"]),
                "wind_speed": _r(w["wind_speed"], 2), "wind_dir_from": _r(w["wind_dir_from"], 0),
                "mixing_height": _r(w["mixing_height"], 0), "ventilation": _r(w["ventilation"], 0),
                "rain_mm_1h": _r(float(self.obs.weather["rain_mm"].loc[t - pd.Timedelta(minutes=55):t].sum()), 1),
                "reference_pm10": _r(w["reference_pm10"])}

    def forecast(self, z: str, t: pd.Timestamp, labels: Optional[List[str]] = None) -> Dict[str, Any]:
        labels = labels or list(self.forecaster.horizons)
        key = (z, str(t), tuple(labels))
        if key not in self._forecast_cache:
            self._forecast_cache[key] = self.forecaster.predict_zone(self.obs, t, z, labels)
        return self._forecast_cache[key]

    # ------------------------------------------------------------------ #
    # Forecast panel series
    # ------------------------------------------------------------------ #
    def forecast_panel(self, z: str, t: pd.Timestamp) -> Dict[str, Any]:
        fc_all = self.forecast(z, t)
        tabs = {}
        for tab in UI_TABS:
            hist_h = TAB_HISTORY_H[tab]
            res = TAB_RES_MIN[tab]
            t0 = t - pd.Timedelta(hours=hist_h)
            frame = pd.DataFrame({"pm10": self.obs.pm10[z].loc[t0:t], "pm25": self.obs.pm25[z].loc[t0:t]}).join(
                self.obs.weather.loc[t0:t, ["temperature", "relative_humidity", "wind_speed", "rain_mm"]])
            if res > 5:
                # Bin-averaged history (labelled at the bin end), but the most recent hour is kept at
                # full 5-min resolution so the observed line meets the forecast's starting point.
                binned = frame.resample(f"{res}min", origin="end").mean()
                cut = t - pd.Timedelta(minutes=60)
                binned = binned[binned.index <= cut]
                tail_start = (binned.index[-1] if len(binned) else t0) + pd.Timedelta(minutes=5)
                frame = pd.concat([binned, frame.loc[tail_start:t]])
            observed = [{"time": str(i), "pm10": _r(r.pm10), "pm25": _r(r.pm25),
                         "temperature": _r(r.temperature), "relative_humidity": _r(r.relative_humidity),
                         "wind_speed": _r(r.wind_speed, 2),
                         # 5-min accumulations expressed as an hourly rate (mm/h), same unit as the forecast
                         "rain_mm": _r(r.rain_mm * 12, 2)}
                        for i, r in frame.iterrows()]
            now_pt = {"time": str(t), "pm10": _r(self.obs.pm10[z].loc[t]), "pm25": _r(self.obs.pm25[z].loc[t])}
            predicted = [{**now_pt, "lo90": now_pt["pm10"], "hi90": now_pt["pm10"], "pm25_lo90": now_pt["pm25"],
                          "pm25_hi90": now_pt["pm25"], "horizon": "now"}]
            for lab in TAB_PATHS[tab]:
                f = fc_all[lab]
                h = ALL_HORIZONS[lab]
                wf = forecast_weather(self.obs, h).loc[t]
                p10 = f["pm10"]
                lvl = level_for(p10["median"])
                nxt = next((l for l in ACTION_LEVELS if l["min"] > lvl["min"]), None)
                prob_next = None
                if nxt is not None and p10["hi90"] > p10["median"]:
                    sigma = math.log(max(p10["hi90"], 1) / max(p10["median"], 1)) / 1.645
                    zscore = math.log(nxt["min"] / max(p10["median"], 1)) / max(sigma, 1e-6)
                    prob_next = 0.5 * math.erfc(zscore / math.sqrt(2))
                predicted.append({
                    "time": f["valid_time"], "horizon": lab, "pm10": p10["median"], "lo90": p10["lo90"],
                    "hi90": p10["hi90"], "pm25": f["pm25"]["median"], "pm25_lo90": f["pm25"]["lo90"],
                    "pm25_hi90": f["pm25"]["hi90"], "level": lvl["key"], "level_label": lvl["label"],
                    "prob_next_level": _r(prob_next, 2) if prob_next is not None else None,
                    "next_level": nxt["label"] if nxt else None,
                    "temperature": _r(wf["temperature"]), "relative_humidity": _r(wf["relative_humidity"]),
                    "wind_speed": _r(wf["wind_speed"], 2), "wind_dir_from": _r(wf["wind_dir_from"], 0),
                    "rain_mm": _r(max(wf["rain_1h"], 0) if not np.isnan(wf["rain_1h"]) else 0.0, 2),
                })
            tabs[tab] = {"observed": observed, "predicted": predicted, "history_hours": hist_h}
        return tabs

    # ------------------------------------------------------------------ #
    # Explanations
    # ------------------------------------------------------------------ #
    def forecast_explanation(self, z: str, t: pd.Timestamp, label: str = "60m") -> Dict[str, Any]:
        h = ALL_HORIZONS[label]
        model = self.forecaster.models[label]["pm10"]
        x = feature_row(self.obs, t, z, h)
        F = build_features(self.obs.slice(t), h, [z], exclude_new_interventions=False)
        hours = F.index.hour + F.index.minute / 60
        hh = t.hour + t.minute / 60
        bg = F[(abs(hours - hh) <= 1.0) & (F.index < t - pd.Timedelta(hours=12)) &
               (F.index > t - pd.Timedelta(days=15))].dropna(subset=model.features)
        if len(bg) > 40:
            bg = bg.sample(40, random_state=1)
        groups = {g: cols for g, cols in groups_for(h).items()}
        # make sure every feature the model uses is assigned to exactly one group
        res = exact_group_shapley(lambda X: model.predict(X)[0], x, bg[x.columns], groups)
        contrib = sorted(({"group": g, "label": GROUP_LABELS[g], "ugm3": _r(v)} for g, v in res["contributions"].items()),
                         key=lambda d: -abs(d["ugm3"]))
        return {"zone_id": z, "horizon": label, "typical_level": _r(res["baseline"]),
                "forecast": _r(res["prediction"]), "drivers": contrib,
                "method": "Exact group-Shapley (32 coalitions) vs typical same-hour conditions, last 14 days"}

    def why_text(self, z: str, wx: Dict[str, Any], attribution: Dict[str, Any], expl: Dict[str, Any],
                 reading: Dict[str, Any]) -> Dict[str, str]:
        shares = {c["source"]: c["share_pct"] for c in attribution["contributors"]}
        dust = shares.get("Construction", 0) + shares.get("Road dust", 0)
        drivers = {d["group"]: d["ugm3"] for d in expl["drivers"]}
        calm = wx["wind_speed"] < 1.5
        conds = []
        if calm:
            conds.append(f"low wind speed ({wx['wind_speed']:.1f} m/s)")
        if wx["mixing_height"] < 900:
            conds.append(f"a shallow mixing layer (~{fmt0(wx['mixing_height'])} m)")
        cond_txt = " and ".join(conds) if conds else "current dispersion conditions"
        activity = []
        trucks = float(self.obs.trucks[z].loc[self.now - pd.Timedelta(minutes=25):self.now].mean())
        if self.obs.construction[z].loc[self.now] > 0.6:
            activity.append("increased excavation activity")
        if trucks > 4:
            activity.append(f"haul-truck movements (~{fmt0(trucks)} per 5 min)")
        act_txt = " and ".join(activity) if activity else "site activity"
        why = (f"{cond_txt[0].upper() + cond_txt[1:]} {'is' if len(conds) <= 1 else 'are'} limiting pollutant dispersion, "
               f"while {act_txt} coincide{'s' if len(activity) == 1 else ''} with the observed PM10 increase.")
        return {
            "observation": f"PM10 concentration is increasing rapidly in Zone {z} "
                           f"({fmt0(reading['pm10'])} µg/m³, {sgn0(reading['change_30m_pct'])}% in 30 min).",
            "prediction": self._prediction_sentence(expl),
            "reason": why,
            "dust_share_pct": round(dust, 0),
        }

    @staticmethod
    def _prediction_sentence(expl: Dict[str, Any]) -> str:
        pos = [d for d in expl["drivers"] if d["ugm3"] >= 3][:2]
        if pos:
            parts = " and ".join(f"{d['label'].lower()} ({sgn0(d['ugm3'])} µg/m³)" for d in pos)
            lead = parts[0].upper() + parts[1:]
            return (f"{lead} push the 60-minute forecast to {fmt0(expl['forecast'])} µg/m³, versus "
                    f"{fmt0(expl['typical_level'])} µg/m³ under typical conditions at this hour.")
        return f"The 60-minute forecast is {fmt0(expl['forecast'])} µg/m³ (typical: {fmt0(expl['typical_level'])} µg/m³)."

    # ------------------------------------------------------------------ #
    # Events
    # ------------------------------------------------------------------ #
    def build_events(self, t: pd.Timestamp, zone_forecasts: Dict[str, Dict[str, Any]],
                     attributions: Dict[str, Dict[str, Any]], priorities: Dict[str, Dict[str, Any]],
                     comparisons: Dict[str, Dict[str, Any]], allocation: Dict[str, Any]) -> Dict[str, Any]:
        detected = detect_events(self.obs, t - pd.Timedelta(days=14), t)
        hero_idx = max(i for i, e in enumerate(detected) if e.zone_id == HERO_ZONE and e.end is None)
        events: List[PollutionEvent] = []
        inter = self.obs.interventions
        effects = {(r["zone_id"], r["dispatched_at"]): r for r in self.learning.records}
        for i, e in enumerate(detected):
            num = 42 - (hero_idx - i)
            if num < 1 or i > hero_idx and e.end is not None:
                continue
            eid = f"EV-{num:03d}"
            zname = site.ZONE_BY_ID[e.zone_id].short_name
            att = self.attributor.explain(self.obs, e.zone_id, e.peak_time if e.end is not None else t)
            top = max(att["contributors"], key=lambda c: c["share_pct"])
            second = sorted(att["contributors"], key=lambda c: -c["share_pct"])[1]
            local_frac = att["local_excess_pm10"] / max(att["observed_pm10"], 1)
            if local_frac < 0.25:
                contributor = "Regional background"
            elif top["source"] in ("Construction", "Road dust") and second["source"] in ("Construction", "Road dust") \
                    and second["share_pct"] > 10:
                contributor = "Construction / road dust"
            else:
                contributor = top["source"]
            end = e.end if e.end is not None else t
            matched = inter[(inter["zone_id"] == e.zone_id) & (inter["dispatched_at"] >= e.start - pd.Timedelta(minutes=10))
                            & (inter["dispatched_at"] <= end)]
            ev_type = {"rapid_rise": "PM10 spike", "sustained_elevation": "Sustained PM10 elevation",
                       "statistical_anomaly": "PM10 anomaly"}[e.trigger]
            base = dict(event_id=eid, zone_id=e.zone_id, zone_name=zname, pollutant="PM10", event_type=ev_type,
                        severity=severity_for(e.peak_pm10), level=level_for(e.peak_pm10)["label"],
                        detected_at=str(e.start), peak_pm10=_r(e.peak_pm10),
                        current_pm10=_r(float(self.obs.pm10[e.zone_id].loc[t])), predicted_peak=None,
                        probable_contributor=contributor, confidence=att["confidence"])
            if e.end is not None:
                if len(matched):
                    rec = matched.iloc[0]
                    eff = effects.get((e.zone_id, str(pd.Timestamp(rec["dispatched_at"]).floor("5min"))))
                    outcome = (f"Verified {sgn0(eff['effect_pct'])}% vs forecast" if eff and eff["effect_pct"] <= -10
                               else ("Inconclusive effect" if eff else "Intervention completed"))
                    status = "VERIFIED" if eff and eff["effect_pct"] <= -10 else "CLOSED"
                    ev = PollutionEvent(**base, intervention_status=status, status="CLOSED", outcome=outcome,
                                        assigned_resource=ACTIONS[rec["action"]]["label"], is_active=False)
                else:
                    outcome = ("Regional episode - no site action" if contributor == "Regional background"
                               else "Resolved without intervention")
                    ev = PollutionEvent(**base, intervention_status="NONE", status="CLOSED", outcome=outcome,
                                        is_active=False)
                events.append(ev)
            else:
                events.append(self._active_event(base, e, t, zone_forecasts, attributions, priorities, comparisons,
                                                 allocation, matched))
        # predicted (forecast-based) exposure events for sensitive zones
        next_num = 43
        for z in ZONE_IDS:
            if any(ev.zone_id == z and ev.is_active for ev in events):
                continue
            f = zone_forecasts[z]
            peak = max(f[k]["pm10"]["median"] for k in ("45m", "60m", "75m", "90m"))
            peak_hi = max(f[k]["pm10"]["hi90"] for k in ("45m", "60m", "75m", "90m"))
            cur = float(self.obs.pm10[z].loc[t])
            sensitive = any(r.zone_id == z and r.kind in ("school", "daycare", "clinic") for r in site.RECEPTORS)
            if sensitive and peak_hi >= 100 and peak > cur * 1.05:
                att = attributions[z]
                eid = f"EV-{next_num:03d}"
                next_num += 1
                ev = PollutionEvent(
                    event_id=eid, zone_id=z, zone_name=site.ZONE_BY_ID[z].short_name, pollutant="PM10",
                    event_type="Predicted exposure risk", severity="Low", level=level_for(peak)["label"],
                    detected_at=str(HERO_TIMES["predicted"]), peak_pm10=_r(peak), current_pm10=_r(cur),
                    predicted_peak={"pm10": _r(peak), "within_min": 90, "hi90": _r(peak_hi)},
                    probable_contributor="Downwind drift from Zone B", confidence=round(0.8 * att["confidence"], 2),
                    intervention_status="MONITORING", status="PREDICTED", outcome=None, is_active=True,
                    is_predicted=True,
                    timeline=[TimelineEntry(_fmt_time(HERO_TIMES["predicted"]), "PREDICTED", "Exposure risk predicted",
                                            f"90-min forecast {fmt0(peak)} µg/m³ (upper band {fmt0(peak_hi)}) near the school/daycare while occupied; "
                                            "plume from Zone B aligned with NW wind.", "done"),
                              TimelineEntry(_fmt_time(t), "MONITORING", "Monitoring",
                                            "No separate dispatch - mitigated at source by EV-042 action.", "current")],
                )
                events.append(ev)
        events.sort(key=lambda ev: (not ev.is_active, ev.event_id), reverse=False)
        active = [ev for ev in events if ev.is_active]
        history = sorted([ev for ev in events if not ev.is_active], key=lambda ev: ev.event_id, reverse=True)
        return {"active": [ev.to_dict() for ev in active], "history": [ev.to_dict() for ev in history],
                "counts": {"active": len(active), "history": len(history)},
                "detector": {"rules": ["Rapid rise: PM10 >= 150 and +20% in 30 min (confirmed)",
                                       "Sustained: PM10 >= 150 for 30 min",
                                       "Anomaly: robust z >= 3.5 vs 7-day same-hour baseline"]},
                "detected_raw": [summarise_detected(e) for e in detected[-12:]]}

    def _active_event(self, base: Dict[str, Any], e, t, zone_forecasts, attributions, priorities, comparisons,
                      allocation, matched) -> PollutionEvent:
        z = e.zone_id
        f = zone_forecasts[z]
        peak_med = max(f[k]["pm10"]["median"] for k in ("60m", "75m", "90m"))
        peak_hi = max(f[k]["pm10"]["hi90"] for k in ("60m", "75m", "90m"))
        base["predicted_peak"] = {"pm10": _r(peak_med), "hi90": _r(peak_hi), "within_min": 90}
        sev_levels = ["Low", "Medium", "High", "Critical"]
        sev = severity_for(max(e.peak_pm10, peak_med))
        if peak_hi >= {"Low": 150, "Medium": 200, "High": 250}.get(sev, 1e9):   # upper band reaches next level
            sev = sev_levels[min(sev_levels.index(sev) + 1, 3)]
        base["severity"] = sev
        if z == HERO_ZONE:
            alloc = next((a for a in allocation["allocations"] if a["zone_id"] == z and a["status"] == "Pending approval"), None)
            rec = comparisons[z]["options"][0]
            resource = alloc["resource"] if alloc else rec["resource"]
            tl = [
                TimelineEntry(_fmt_time(HERO_TIMES["detected"]), "DETECTED", "Pollution anomaly detected",
                              f"Zone B average crossed the Elevated action level (150 µg/m³) with a "
                              f"{sgn0(e.rise_pct_30m)}% rise in 30 min; first flagged by sensor EC-042 (haul road / pit edge).", "done"),
                TimelineEntry(_fmt_time(HERO_TIMES["classified"]), "CLASSIFIED", "AI classified event",
                              f"PM10 spike - probable {base['probable_contributor'].lower()} "
                              f"({fmt0(100 * base['confidence'])}% confidence).", "done"),
                TimelineEntry(_fmt_time(HERO_TIMES["predicted"]), "PREDICTED", "Hotspot predicted",
                              f"No-action forecast {fmt0(peak_med)} µg/m³ within 90 min (upper 90% band {fmt0(peak_hi)}); "
                              "plume drifting SE towards the school perimeter.", "done"),
                TimelineEntry(_fmt_time(HERO_TIMES["recommended"]), "RECOMMENDED", "DUSTWISE recommendation generated",
                              f"IPS {fmt0(priorities[z]['ips_score'])}/100 - {rec['label'].lower()} with {resource}.", "done"),
                TimelineEntry(_fmt_time(HERO_TIMES["approved"]), "APPROVED", "Operator approved intervention",
                              "Awaiting operator review.", "current"),
                TimelineEntry(_fmt_time(HERO_TIMES["initiated"]), "ACTIVE", "Intervention initiated",
                              f"{resource} on site, spraying haul road and pit edge.", "upcoming"),
                TimelineEntry(_fmt_time(HERO_TIMES["response"]), "RESPONSE_DETECTED", "Pollution response detected",
                              "Downwind sensors show a sustained decline.", "upcoming"),
                TimelineEntry(_fmt_time(HERO_TIMES["verified"]), "VERIFIED", "Impact verified",
                              "Observed level compared with the no-action forecast.", "upcoming"),
            ]
            return PollutionEvent(**base, intervention_status="PENDING", status="PENDING_APPROVAL", outcome=None,
                                  assigned_resource=resource, is_active=True, timeline=tl)
        # other active events (e.g. Zone A with Sweeper 01 already working)
        if len(matched):
            rec = matched.iloc[0]
            tl = [
                TimelineEntry(_fmt_time(e.start), "DETECTED", "Pollution anomaly detected",
                              f"Zone {z} exceeded its same-hour baseline (material unloading).", "done"),
                TimelineEntry(_fmt_time(e.start + pd.Timedelta(minutes=4)), "CLASSIFIED", "AI classified event",
                              f"Probable {base['probable_contributor'].lower()}.", "done"),
                TimelineEntry(_fmt_time(rec["dispatched_at"]), "APPROVED", "Operator approved intervention",
                              f"{ACTIONS[rec['action']]['label']} - Sweeper 01.", "done"),
                TimelineEntry(_fmt_time(rec["started_at"]), "ACTIVE", "Intervention initiated",
                              "Sweeping Tower Lane and the service track.", "done"),
                TimelineEntry(_fmt_time(t), "MONITORING", "Monitoring response",
                              f"PM10 {fmt0(base['current_pm10'])} µg/m³, down from a {fmt0(base['peak_pm10'])} µg/m³ peak.", "current"),
            ]
            return PollutionEvent(**base, intervention_status="ACTIVE", status="ACTIVE", outcome=None,
                                  assigned_resource="Sweeper 01", is_active=True, timeline=tl)
        return PollutionEvent(**base, intervention_status="NONE", status="DETECTED", outcome=None, is_active=True,
                              timeline=[TimelineEntry(_fmt_time(e.start), "DETECTED", "Pollution anomaly detected",
                                                      "Monitoring.", "current")])

    # ------------------------------------------------------------------ #
    # Full state
    # ------------------------------------------------------------------ #
    def build_state(self, t: pd.Timestamp = NOW) -> Dict[str, Any]:
        t0 = time.time()
        wx = self.weather_now(t)
        hour = t.hour + t.minute / 60
        workday = bool(self.obs.weather["workday"].loc[t] > 0.5)
        zone_fc = {z: self.forecast(z, t) for z in ZONE_IDS}
        attributions = {z: self.attributor.explain(self.obs, z, t, coverage=self.sensors["zones"][z]["data_confidence"])
                        for z in ZONE_IDS}
        vent_f = {z: float(np.mean([forecast_weather(self.obs, ALL_HORIZONS[k]).loc[t, "ventilation"]
                                    for k in ("30m", "60m", "90m")])) for z in ZONE_IDS}
        readings = {z: self.zone_reading(z, t) for z in ZONE_IDS}
        efficacy = self.learning.efficacy_table()
        priorities = {z: self.engine.priority(z, readings[z]["pm10"], zone_fc[z], attributions[z], wx, vent_f[z],
                                              hour, workday, readings[z]["change_30m_pct"]) for z in ZONE_IDS}
        comparisons = {z: self.engine.compare_actions(z, attributions[z], efficacy) for z in ZONE_IDS}
        allocation = self.engine.optimize_allocation(priorities, comparisons)
        events = self.build_events(t, zone_fc, attributions, priorities, comparisons, allocation)

        # predicted hotspots: zones forecast to escalate into >= Elevated within 90 min
        predicted_hotspots = []
        for z in ZONE_IDS:
            cur = readings[z]["pm10"]
            med = max(zone_fc[z][k]["pm10"]["median"] for k in ("60m", "75m", "90m"))
            hi = max(zone_fc[z][k]["pm10"]["hi90"] for k in ("60m", "75m", "90m"))
            cur_rank = level_rank(level_for(cur)["key"])
            nxt = next((l for l in ACTION_LEVELS if level_rank(l["key"]) == cur_rank + 1), None)
            if nxt and nxt["min"] >= 150 and hi >= nxt["min"] and med >= 0.9 * nxt["min"]:
                predicted_hotspots.append({"zone_id": z, "to_level": nxt["label"], "forecast_peak": _r(med),
                                           "upper_band": _r(hi), "within_min": 90})

        explanations = {z: self.forecast_explanation(z, t, "60m") for z in ZONE_IDS}
        hero_expl = explanations[HERO_ZONE]
        hero_text = self.why_text(HERO_ZONE, wx, attributions[HERO_ZONE], hero_expl, readings[HERO_ZONE])
        hero_event = next(ev for ev in events["active"] if ev["event_id"] == HERO_EVENT_ID)
        rec_opt = comparisons[HERO_ZONE]["options"][0]
        alloc_b = next(a for a in allocation["allocations"] if a["zone_id"] == HERO_ZONE and a["status"] == "Pending approval")
        fc_b = zone_fc[HERO_ZONE]
        # risk window: first horizon whose median reaches 95% of the High action level
        first_high = next((k for k in ("15m", "30m", "45m", "60m", "75m", "90m")
                           if fc_b[k]["pm10"]["median"] >= 190), None)
        risk_window = f"{first_high[:-1]}–90 min" if first_high and first_high != "90m" else "60–90 min"
        dust = hero_text["dust_share_pct"]
        insight = {
            "event_id": HERO_EVENT_ID, "zone_id": HERO_ZONE, "zone_name": site.ZONE_BY_ID[HERO_ZONE].short_name,
            "headline": "POLLUTION EVENT DETECTED",
            "statement": f"PM10 concentration is increasing rapidly in Zone {HERO_ZONE}. "
                         "Current conditions indicate elevated dust-event risk.",
            "current_pm10": readings[HERO_ZONE]["pm10"], "change_30m_pct": readings[HERO_ZONE]["change_30m_pct"],
            "forecast_text": f"High risk in {risk_window}",
            "forecast_peak": hero_event["predicted_peak"],
            "contributors": attributions[HERO_ZONE],
            "explanation": hero_text, "forecast_drivers": hero_expl,
            "priority": priorities[HERO_ZONE],
            "recommendation": {
                "action": rec_opt["label"], "action_key": rec_opt["action"],
                "title": "Targeted dust suppression" if rec_opt["action"] == "targeted_sprinkling" else rec_opt["label"],
                "priority": "Immediate" if priorities[HERO_ZONE]["ips_score"] >= 80 else "High",
                "location": f"Zone {HERO_ZONE} - haul road & pit edge",
                "resource": alloc_b["resource"], "asset_id": alloc_b["asset_id"], "eta_min": alloc_b["eta_min"],
                "route": alloc_b["route"],
                "reason": (f"Zone {HERO_ZONE} has high current PM10 ({fmt0(readings[HERO_ZONE]['pm10'])} µg/m³), "
                           f"a rising trend ({sgn0(readings[HERO_ZONE]['change_30m_pct'])}% in 30 min) and a probable "
                           f"construction/road-dust contribution of ~{fmt0(dust)}%."),
                "expected_outcome": f"AI scenario estimate: {fmt0(rec_opt['scenario_benefit_pct'])}% lower PM10 "
                                    f"(range {fmt0(rec_opt['scenario_benefit_range_pct'][0])}–{fmt0(rec_opt['scenario_benefit_range_pct'][1])}%) "
                                    f"within ~45 min of starting.",
                "approval_note": "AI recommendation — operator approval required",
                "water_l": rec_opt["water_l"],
            },
        }

        top_ips = max(p["ips_score"] for p in priorities.values())
        overall = max(ZONE_IDS, key=lambda z: readings[z]["pm10"])
        risk = "High" if top_ips >= 80 else ("Elevated" if top_ips >= 60 else ("Moderate" if top_ips >= 40 else "Low"))
        active_interventions = sum(1 for a in self.engine.fleet if a.status == "ACTIVE")
        counts = self.sensors["counts"]
        status_strip = {
            "overall_status": readings[overall]["level_label"].upper(),
            "overall_level": readings[overall]["level"],
            "overall_driver": f"Zone {overall} {fmt0(readings[overall]['pm10'])} µg/m³",
            "active_events": events["counts"]["active"],
            "predicted_hotspots": len(predicted_hotspots),
            "sensors_online": counts["online"], "sensors_total": counts["total"],
            "sensors_warning": counts["warning"], "sensors_offline": counts["offline"],
            "interventions_active": active_interventions,
            "environmental_risk": risk, "top_ips": top_ips,
        }

        zones_out = []
        for z in site.ZONES:
            zid = z.zone_id
            zones_out.append({
                "zone_id": zid, "name": z.name, "short_name": z.short_name, "land_use": z.land_use,
                **readings[zid],
                "forecast_60m": zone_fc[zid]["60m"]["pm10"], "forecast_90m": zone_fc[zid]["90m"]["pm10"],
                "ips": priorities[zid]["ips_score"], "ips_band": priorities[zid]["band"],
                "data_confidence": self.sensors["zones"][zid]["data_confidence"],
                "confidence_reduced": self.sensors["zones"][zid]["confidence_reduced"],
                "top_contributor": max(attributions[zid]["contributors"], key=lambda c: c["share_pct"])["source"],
                "predicted_hotspot": any(p["zone_id"] == zid for p in predicted_hotspots),
            })

        forecasts = {z: self.forecast_panel(z, t) for z in ZONE_IDS}
        state = {
            "meta": self.meta(t),
            "status_strip": status_strip,
            "weather": wx,
            "zones": zones_out,
            "predicted_hotspots": predicted_hotspots,
            "insight": insight,
            "events": events,
            "forecasts": forecasts,
            "attribution": attributions,
            "explanations": explanations,
            "dustwise": {"priorities": priorities, "comparisons": comparisons, "allocation": allocation,
                         "fleet": [self._asset_dict(a) for a in self.engine.fleet],
                         "actions_catalog": {k: {kk: vv for kk, vv in v.items() if kk != "reduction"} | {"reduction": v["reduction"]}
                                             for k, v in ACTIONS.items()}},
            "sensors": self.sensors,
            "learning": self.learning.summary(),
            "ai_status": self.ai_status(t),
        }
        logger.info("State built in %.1f s", time.time() - t0)
        return state

    def _asset_dict(self, a: site.FleetAsset) -> Dict[str, Any]:
        d = dict(a.__dict__)
        d["available_l"] = int(a.capacity_l * a.level_pct / 100) if a.asset_type == "tanker" else None
        d["zone_id"] = site.zone_of(a.position)
        return d

    def meta(self, t: pd.Timestamp) -> Dict[str, Any]:
        return {
            "platform": "EcoNexus AI", "module": "DUSTWISE • Intervention Intelligence",
            "site": site.SITE_NAME, "area_acres": site.SITE_AREA_ACRES, "now": str(t),
            "synthetic_data": True,
            "data_label": "Synthetic demo data (physics-based simulator) — not field measurements",
            "model_backend": self.forecaster.meta.get("backend", backend_name()),
            "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        }

    def ai_status(self, t: pd.Timestamp) -> Dict[str, Any]:
        ev = self.evaluation.get("metrics", {})
        cov = [ev[k]["pm10"]["coverage_90"] for k in ("15m", "60m", "90m") if k in ev]
        return {
            "status": "AI ENGINE ACTIVE",
            "last_model_update": self.model_card.get("trained_at"),
            "model_backend": self.forecaster.meta.get("backend"),
            "forecast_confidence": {"interval": "90% split-conformal",
                                    "test_coverage_operational": round(float(np.mean(cov)), 3) if cov else None},
            "data_freshness_s": 30,
            "sensor_coverage": f"{self.sensors['counts']['reporting']}/{self.sensors['counts']['total']} reporting",
            "model_status": "Production models loaded" if self.model_card else "Quick-trained fallback models",
            "horizons": list(self.forecaster.horizons),
        }

    # ------------------------------------------------------------------ #
    # Hero demo
    # ------------------------------------------------------------------ #
    def hero_frames(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Observed 5-min states from 14:15 to 15:30 for the hero demo playback.

        Values before 14:30 are history; values after 14:30 are what the
        (synthetic) sensors record once the operator approves the tanker.
        """
        obs = self.obs_after
        times = pd.date_range("2026-10-07 14:15", "2026-10-07 15:30", freq="5min")
        sf = self.ds.sensor_frame
        frames = []
        route = state["insight"]["recommendation"]["route"]
        for ts in times:
            zones = {z: {"pm10": _r(obs.pm10[z].loc[ts]), "pm25": _r(obs.pm25[z].loc[ts]),
                         "level": level_for(float(obs.pm10[z].loc[ts]))["key"]} for z in ZONE_IDS}
            srow = sf[sf["timestamp"] == ts]
            sensors = {r.sensor_id: (_r(r.pm10) if not np.isnan(r.pm10) else None) for r in srow.itertuples()}
            if ts < HERO_TIMES["departed"]:
                pos = 0.0
            elif ts >= HERO_TIMES["initiated"] - pd.Timedelta(minutes=3):
                pos = 1.0
            else:
                pos = (ts - HERO_TIMES["departed"]).total_seconds() / (
                    (HERO_TIMES["initiated"] - pd.Timedelta(minutes=3) - HERO_TIMES["departed"]).total_seconds())
            frames.append({"time": str(ts), "zones": zones, "sensors": sensors, "tanker_progress": round(pos, 2)})

        fc_b = self.forecast(HERO_ZONE, self.now)
        rec = state["insight"]["recommendation"]
        tanker = self.engine.asset(rec["asset_id"])
        true_cf = self.ds.counterfactual
        true_cf = true_cf[true_cf["zone_id"] == HERO_ZONE].set_index("timestamp")["pm10_true"]
        verification = verify_intervention(
            obs.pm10[HERO_ZONE], fc_b, self.now, HERO_TIMES["initiated"], HERO_TIMES["verified"],
            HERO_TIMES["approved"], HERO_TIMES["detected"], tanker.rate,
            2 * sum(site.distance(route[i], route[i + 1]) for i in range(len(route) - 1)),
            true_counterfactual=true_cf, synthetic=True)
        # counterfactual path (no-action forecast) for the impact chart
        cf_path = [{"time": str(self.now), "pm10": _r(obs.pm10[HERO_ZONE].loc[self.now]), "lo90": None, "hi90": None}]
        for k in ("15m", "30m", "45m", "60m", "75m", "90m"):
            cf_path.append({"time": fc_b[k]["valid_time"], **fc_b[k]["pm10"]})
        verification["counterfactual_path"] = cf_path

        # Preview of the learning update the demo will show; live state is restored afterwards so that
        # the API's real approve/verify flow applies the update exactly once.
        key = rec["action_key"]
        saved_post = copy.deepcopy(self.learning.posteriors[key])
        saved_hist = list(self.learning.history)
        expected = expected_reduction_pct(state["attribution"][HERO_ZONE], key)
        learning_update = self.learning.record_outcome(
            key, verification["after_pm10"], verification["counterfactual_pm10"], expected, HERO_EVENT_ID,
            verification["kpis"]["peak_prediction_error_pct"])
        self.learning.posteriors[key] = saved_post
        self.learning.history = saved_hist

        steps = [
            ("14:20", "OBSERVE", "Zone B begins showing PM10 increase", "Excavation and haul activity resume after lunch while the wind drops below 1 m/s."),
            ("14:20", "OBSERVE", "Sensor detects anomaly", "EC-042 at the haul road / pit edge records the steepest rise; the Zone B average crosses 150 µg/m³."),
            ("14:20", "UNDERSTAND", "Pollution event created", f"{HERO_EVENT_ID} opened for Zone B (rapid rise rule)."),
            ("14:24", "DIAGNOSE", "AI explains probable contributors", "Receptor model estimates construction, road dust and traffic shares with confidence."),
            ("14:27", "PREDICT", "Forecast predicts hotspot development", "No-action forecast and 90% band for the next 90 minutes."),
            ("14:29", "PRIORITIZE", "Risk score increases", "Intervention Priority Score computed from model outputs."),
            ("14:29", "OPTIMIZE", "DUSTWISE recommends targeted intervention", "Action comparison and fleet assignment."),
            ("14:33", "ACT", "Operator reviews and approves", "AI recommendation — operator approval required."),
            ("14:38", "ACT", "Tanker assignment appears", "Route issued to the tanker; ETA shown on the map."),
            ("14:42", "ACT", "Intervention status changes to active", "Spraying starts on the haul road and pit edge."),
            ("15:05", "VERIFY", "Sensor readings update", "Zone B readings fall while the no-action forecast keeps rising."),
            ("15:20", "VERIFY", "Before/after impact is displayed", "Observed change and model-estimated effect vs counterfactual."),
            ("15:20", "VERIFY", "Event becomes verified", "Effect is beyond the forecast's 90% band."),
            ("15:21", "LEARN", "AI learning loop records the outcome", "Efficacy posterior for targeted sprinkling updated."),
        ]
        return {"frames": frames, "times": {k: str(v) for k, v in HERO_TIMES.items()},
                "steps": [{"n": i + 1, "time": tm, "stage": st, "title": ti, "detail": de}
                          for i, (tm, st, ti, de) in enumerate(steps)],
                "verification": verification, "learning_update": learning_update,
                "tanker": {"asset_id": tanker.asset_id, "name": tanker.name, "route": route}}

    # ------------------------------------------------------------------ #
    # Sustainability
    # ------------------------------------------------------------------ #
    def sustainability(self) -> Dict[str, Any]:
        obs = self.obs
        end = self.now.normalize()
        pm = obs.pm10.loc[:end - pd.Timedelta(minutes=5)]
        daily = pm.resample("D").mean()
        # weeks start on Monday and are labelled by their first day (no labels beyond the data)
        weekly = pm.resample("W-MON", label="left", closed="left").mean()
        exceed = (pm >= 150).resample("W-MON", label="left", closed="left").sum() * 5 / 60   # hours/week at or above Elevated
        compliance = (daily <= CPCB_PM10_24H).mean()
        inter = obs.interventions[obs.interventions["source"] == "historical"].copy()
        inter["minutes"] = (inter["ended_at"] - inter["started_at"]).dt.total_seconds() / 60
        tanker = inter[inter["action"] != "mechanical_sweeping"]
        water_targeted = float((tanker["minutes"] * 160).sum())
        work_days = int(obs.weather.loc[:end, "workday"].resample("D").max().sum())
        water_blanket = work_days * 2 * 3 * 12000 * 0.85      # 3 tankers, 2 full loads per work day (routine schedule)
        events_hist = detect_events(obs, obs.index[0] + pd.Timedelta(days=7), self.now)
        heat = np.zeros((4, 24), dtype=int)
        for e in events_hist:
            heat[ZONE_IDS.index(e.zone_id), e.start.hour] += 1
        green_area = sum(site.polygon_area(g["polygon"]) for g in site.FEATURES["green"])
        site_area = site.polygon_area(site.BOUNDARY)
        eff = self.learning.summary()
        hotspot_counts = {z: int(heat[i].sum()) for i, z in enumerate(ZONE_IDS)}
        # composite score (transparent formula)
        components = [
            {"key": "compliance", "label": "Days within CPCB 24-h PM10", "weight": 35,
             "value": _r(100 * compliance.mean()), "unit": "% of zone-days", "points": 35 * compliance.mean()},
            {"key": "effectiveness", "label": "Intervention effectiveness", "weight": 25,
             "value": eff["recommendation_effectiveness_pct"], "unit": "% met half of expected (target 80%)",
             "points": 25 * min(1, eff["recommendation_effectiveness_pct"] / 80)},
            {"key": "water", "label": "Water saved vs routine spraying", "weight": 20,
             "value": _r(100 * (1 - min(1, water_targeted / max(water_blanket, 1)))), "unit": "% saved",
             "points": 20 * (1 - min(1, water_targeted / max(water_blanket, 1)))},
            {"key": "green", "label": "Green cover", "weight": 20,
             "value": _r(100 * green_area / site_area), "unit": "% of site (target 15%)",
             "points": 20 * min(1, (green_area / site_area) / 0.15)},
        ]
        for c in components:
            c["points"] = _r(c["points"], 1)
        score = sum(c["points"] for c in components)
        pave = twin.compare_scenarios()
        recs = [
            {"title": "Pave the Phase 3 haul road (first 300 m from Gate 2)", "horizon": "1–30 days",
             "benefit": "Scenario: haul-road dust −80%", "zone": "B",
             "estimate_zone_b_pct": round(100 * (twin.simulate(include_grid=False, base_trucks=9 * 0.2)["zones"]["B"] -
                                                 pave["scenarios"][0]["zones"]["B"]) / pave["scenarios"][0]["zones"]["B"], 1)},
            {"title": "Fixed misting line along the excavation pit edge", "horizon": "1–30 days",
             "benefit": "Scenario: pit construction dust −40% during operation", "zone": "B",
             "estimate_zone_b_pct": round(100 * (twin.simulate(include_grid=False, base_construction={"A": 0.72, "B": 0.86 * 0.6})["zones"]["B"] -
                                                 pave["scenarios"][0]["zones"]["B"]) / pave["scenarios"][0]["zones"]["B"], 1)},
            {"title": "Three-row vegetation buffer on the B–D boundary (school side)", "horizon": "6–12 months",
             "benefit": "Intercepts downwind drift towards the school and daycare", "zone": "D",
             "estimate_zone_b_pct": None},
            {"title": "Shift rock-breaking to before 11:00 on stagnant-forecast days", "horizon": "Immediate",
             "benefit": "Avoids low-ventilation afternoons flagged by the 24 h outlook", "zone": "B",
             "estimate_zone_b_pct": None},
        ]
        return {
            "period": {"start": str(pm.index[0]), "end": str(pm.index[-1])},
            "daily_pm10": {z: [{"date": str(d.date()), "pm10": _r(v)} for d, v in daily[z].items()] for z in ZONE_IDS},
            "weekly_pm10": {z: [{"week": str(d.date()), "pm10": _r(v)} for d, v in weekly[z].items()] for z in ZONE_IDS},
            "exceedance_hours_weekly": {z: [{"week": str(d.date()), "hours": _r(v)} for d, v in exceed[z].items()] for z in ZONE_IDS},
            "cpcb_daily_compliance_pct": {z: _r(100 * compliance[z]) for z in ZONE_IDS},
            "water": {"targeted_l": int(water_targeted), "routine_schedule_l": int(water_blanket),
                      "saving_pct": _r(100 * (1 - water_targeted / max(water_blanket, 1))),
                      "note": "Targeted dispatch volume vs a routine schedule of 3 tankers x 2 loads per work day (scenario baseline)."},
            "interventions": {"count": int(len(inter)), "by_action": inter["action"].value_counts().to_dict(),
                              "effectiveness_pct": eff["recommendation_effectiveness_pct"],
                              "median_measured_reduction_pct": eff["median_measured_reduction_pct"]},
            "recurring_hotspots": {"zones": ZONE_IDS, "by_hour": heat.tolist(), "counts": hotspot_counts},
            "green_coverage_pct": _r(100 * green_area / site_area),
            "performance_score": _r(score, 0),
            "performance_components": components,
            "performance_formula": "35×CPCB daily compliance + 25×intervention effectiveness/80% + 20×water saving + 20×green cover/15%",
            "recommendations": recs,
        }

    def model_summary(self) -> Dict[str, Any]:
        return {"card": self.model_card, "benchmark": self.benchmark, "evaluation": self.evaluation,
                "forecaster_meta": self.forecaster.meta,
                "conformal_q": {lab: {t: round(m.q_conformal, 4) for t, m in d.items()} for lab, d in self.forecaster.models.items()}}
