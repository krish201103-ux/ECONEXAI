"""
intelligence.py -- Site status vocabulary, pollution event detection and sensor health.

  ACTION_LEVELS / level_for()   Normal / Watch / Elevated / High / Critical for
                                zone PM10 (thresholds in config.py)
  detect_events()               rapid rise, sustained elevation and statistical
                                anomaly detection, with the event lifecycle
                                DETECTED -> ... -> VERIFIED and its audit timeline
  fleet_snapshot()              per-node ONLINE / WARNING / OFFLINE status, IDW
                                fallback estimates for offline nodes and the zone
                                "data confidence" score
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config
import site_model as site
from data_loader import ObservedData, ZONE_IDS


# ==================================================================================================
# Site dust action levels and status vocabulary.
# Action levels apply to the zone-average PM10 (ug/m3). They are site-management
# trigger levels for construction dust (configurable in config.py), not the
# national ambient standard; the CPCB 24-h PM10 standard (100 ug/m3) is reported
# separately in compliance metrics.
# ==================================================================================================

_T = config.ACTION_LEVEL_THRESHOLDS               # edit the thresholds in config.py
ACTION_LEVELS: List[Dict[str, object]] = [
    {"key": "normal", "label": "Normal", "min": 0, "max": _T["watch"], "color": "green"},
    {"key": "watch", "label": "Watch", "min": _T["watch"], "max": _T["elevated"], "color": "yellow"},
    {"key": "elevated", "label": "Elevated", "min": _T["elevated"], "max": _T["high"], "color": "orange"},
    {"key": "high", "label": "High", "min": _T["high"], "max": _T["critical"], "color": "red"},
    {"key": "critical", "label": "Critical", "min": _T["critical"], "max": 10_000, "color": "critical"},
]
CPCB_PM10_24H = 100.0
CPCB_PM25_24H = 60.0


def fmt0(x: float) -> str:
    """Whole number rounded half away from zero (174.5 -> 175), the rule the dashboard's JavaScript
       uses -- so text written here never disagrees with the numbers on screen."""
    return str(int(math.floor(abs(float(x)) + 0.5)) * (1 if x >= 0 else -1))


def sgn0(x: float) -> str:
    """fmt0 with an explicit sign: +31, -4, 0."""
    v = fmt0(x)
    return v if v.startswith('-') or v == '0' else '+' + v


def level_for(pm10: float) -> Dict[str, object]:
    for lvl in ACTION_LEVELS:
        if lvl["min"] <= pm10 < lvl["max"]:
            return lvl
    return ACTION_LEVELS[-1]


def level_rank(key: str) -> int:
    return [l["key"] for l in ACTION_LEVELS].index(key)


def ips_band(score: float) -> Dict[str, str]:
    if score >= 80:
        return {"key": "immediate", "label": "IMMEDIATE ACTION", "color": "red"}
    if score >= 60:
        return {"key": "high", "label": "HIGH PRIORITY", "color": "orange"}
    if score >= 40:
        return {"key": "watch", "label": "WATCH", "color": "yellow"}
    return {"key": "normal", "label": "ROUTINE", "color": "green"}


def factor_label(value: float) -> str:
    if value >= 70:
        return "High"
    if value >= 55:
        return "Medium/High"
    if value >= 40:
        return "Medium"
    return "Low"


# ==================================================================================================
# Pollution event detection and event lifecycle.
# Detection (per zone, 5-minute zone-average PM10):
#
# * **Rapid rise** - PM10 >= Elevated level (150) and a >= 20% rise over the
#   previous 30 minutes, confirmed on two consecutive readings.
# * **Sustained elevation** - PM10 >= 150 for 30 consecutive minutes.
# * **Statistical anomaly** - robust z-score >= 3.5 against the zone's same-hour
#   baseline (median / MAD of the previous 7 days).
#
# An event closes after PM10 stays below 130 for 30 minutes; a new event in the
# same zone needs a 60-minute quiet gap. Each event carries a lifecycle:
#
#     DETECTED -> CLASSIFIED -> PREDICTED -> RECOMMENDED -> APPROVED -> ACTIVE -> VERIFIED
#
# with a timestamped timeline that the UI renders as the event audit trail.
# ==================================================================================================

ELEVATED = 150.0
CLEAR = 130.0
RISE = 0.20

STATUS_ORDER = ["DETECTED", "CLASSIFIED", "PREDICTED", "RECOMMENDED", "PENDING_APPROVAL", "APPROVED", "ACTIVE",
                "RESPONSE_DETECTED", "VERIFIED", "CLOSED"]


@dataclass
class DetectedEvent:
    zone_id: str
    start: pd.Timestamp
    end: Optional[pd.Timestamp]
    trigger: str
    peak_pm10: float
    peak_time: pd.Timestamp
    start_pm10: float
    rise_pct_30m: float


def _baseline(series: pd.Series) -> pd.DataFrame:
    """Same-time-of-day median and MAD over the previous 7 days."""
    lags = pd.concat([series.shift(288 * d) for d in range(1, 8)], axis=1)
    med = lags.median(axis=1)
    mad = (lags.sub(med, axis=0)).abs().median(axis=1)
    return pd.DataFrame({"median": med, "mad": mad})


def detect_events(obs: ObservedData, start: pd.Timestamp, end: pd.Timestamp) -> List[DetectedEvent]:
    events: List[DetectedEvent] = []
    for z in ZONE_IDS:
        s = obs.pm10[z]
        base = _baseline(s)
        win = s.loc[start - pd.Timedelta(hours=1):end]
        rise = (s / s.shift(6) - 1.0)
        robust_z = 0.6745 * (s - base["median"]) / base["mad"].clip(lower=3.0)
        in_event = False
        cur: Optional[DetectedEvent] = None
        below_count = 0
        above_count = 0
        last_end = pd.Timestamp("2000-01-01")
        for t in win.index:
            if t < start:
                continue
            v = float(s.loc[t])
            if not in_event:
                above_count = above_count + 1 if v >= ELEVATED else 0
                r = float(rise.loc[t]) if not np.isnan(rise.loc[t]) else 0.0
                prev_ok = t - pd.Timedelta(minutes=5) in s.index and float(s.loc[t - pd.Timedelta(minutes=5)]) >= ELEVATED * 0.95
                trigger = None
                if v >= ELEVATED and r >= RISE and prev_ok:
                    trigger = "rapid_rise"
                elif above_count >= 6:
                    trigger = "sustained_elevation"
                elif robust_z.loc[t] >= 3.5 and v >= 120:
                    trigger = "statistical_anomaly"
                if trigger and t - last_end >= pd.Timedelta(minutes=60):
                    in_event = True
                    cur = DetectedEvent(z, t, None, trigger, v, t, v, round(100 * r, 1))
                    below_count = 0
            else:
                assert cur is not None
                if v > cur.peak_pm10:
                    cur.peak_pm10, cur.peak_time = v, t
                below_count = below_count + 1 if v < CLEAR else 0
                if below_count >= 6:
                    cur.end = t
                    events.append(cur)
                    last_end = t
                    in_event = False
                    above_count = 0
                    cur = None
        if in_event and cur is not None:
            events.append(cur)
    events.sort(key=lambda e: e.start)
    return events


# --------------------------------------------------------------------------- #
# Lifecycle store
# --------------------------------------------------------------------------- #

@dataclass
class TimelineEntry:
    time: str
    stage: str
    title: str
    detail: str
    status: str = "done"          # done | current | upcoming


@dataclass
class PollutionEvent:
    event_id: str
    zone_id: str
    zone_name: str
    pollutant: str
    event_type: str
    severity: str
    level: str
    detected_at: str
    peak_pm10: float
    current_pm10: float
    predicted_peak: Optional[Dict[str, object]]
    probable_contributor: str
    confidence: float
    intervention_status: str      # PENDING | APPROVED | ACTIVE | VERIFIED | NONE | MONITORING
    status: str                   # lifecycle stage
    outcome: Optional[str]
    assigned_resource: Optional[str] = None
    is_active: bool = True
    is_predicted: bool = False
    timeline: List[TimelineEntry] = field(default_factory=list)
    impact: Optional[Dict[str, object]] = None

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        return d


def severity_for(peak: float) -> str:
    if peak >= 250:
        return "Critical"
    if peak >= 200:
        return "High"
    if peak >= 150:
        return "Medium"
    return "Low"


def summarise_detected(e: DetectedEvent) -> Dict[str, object]:
    return {
        "zone_id": e.zone_id, "start": str(e.start), "end": str(e.end) if e.end is not None else None,
        "trigger": e.trigger, "peak_pm10": round(e.peak_pm10, 1), "peak_time": str(e.peak_time),
        "start_pm10": round(e.start_pm10, 1), "rise_pct_30m": e.rise_pct_30m,
        "duration_min": int((e.end - e.start).total_seconds() / 60) if e.end is not None else None,
        "level": level_for(e.peak_pm10)["label"],
    }


# ==================================================================================================
# Sensor fleet health, data confidence and offline fallback.
# * Status per node: ONLINE / WARNING / OFFLINE with a reason.
# * Offline fallback: inverse-distance-weighted (IDW, power 2) estimate from the
#   nearest online nodes in the same zone; its expected error is measured by
#   leave-one-out cross-validation over the online fleet at the same timestamp.
# * Zone data confidence: share of healthy nodes, penalised for warnings and
#   for nodes running on fallback estimates. Shown in the UI as
#   "Data confidence reduced" whenever it drops below 90%.
# ==================================================================================================

def idw(target: site.Point, neighbours: List[site.Point], values: List[float], power: float = 2.0) -> float:
    w = np.array([1.0 / max(site.distance(target, p), 5.0) ** power for p in neighbours])
    return float(np.sum(w * np.array(values)) / np.sum(w))


def fleet_snapshot(sensor_frame: pd.DataFrame, t: pd.Timestamp, history_hours: int = 6) -> Dict[str, object]:
    latest = sensor_frame[sensor_frame["timestamp"] == t].set_index("sensor_id")
    hist = sensor_frame[(sensor_frame["timestamp"] > t - pd.Timedelta(hours=history_hours)) &
                        (sensor_frame["timestamp"] <= t)]
    sensors_out = []
    online_vals: Dict[str, float] = {}
    for s in site.SENSORS:
        row = latest.loc[s.sensor_id] if s.sensor_id in latest.index else None
        if s.status != "OFFLINE" and row is not None and not np.isnan(row["pm10"]):
            online_vals[s.sensor_id] = float(row["pm10"])

    # leave-one-out IDW error over healthy online nodes (same zone neighbours)
    loo_err = []
    for s in site.SENSORS:
        if s.sensor_id not in online_vals or s.status != "ONLINE":
            continue
        neigh = [n for n in site.SENSORS if n.zone_id == s.zone_id and n.sensor_id != s.sensor_id
                 and n.sensor_id in online_vals and n.status == "ONLINE"]
        neigh = sorted(neigh, key=lambda n: site.distance(n.position, s.position))[:3]
        if len(neigh) < 2:
            continue
        est = idw(s.position, [n.position for n in neigh], [online_vals[n.sensor_id] for n in neigh])
        loo_err.append(abs(est - online_vals[s.sensor_id]) / max(online_vals[s.sensor_id], 1.0))
    loo_mape = float(np.mean(loo_err) * 100) if loo_err else None

    for s in site.SENSORS:
        row = latest.loc[s.sensor_id] if s.sensor_id in latest.index else None
        h = hist[hist["sensor_id"] == s.sensor_id].set_index("timestamp")
        spark = h["pm10"].resample("15min").mean().round(1)
        last_valid = h["pm10"].dropna()
        last_update = str(last_valid.index[-1]) if len(last_valid) else None
        entry = {
            "sensor_id": s.sensor_id, "zone_id": s.zone_id, "label": s.label, "position": s.position,
            "latlng": site.to_latlng(*s.position), "status": s.status, "status_reason": s.status_reason,
            "battery_pct": s.battery_pct, "signal_dbm": s.signal_dbm,
            "signal_quality": "Lost" if s.status == "OFFLINE" else ("Good" if s.signal_dbm > -72 else "Fair"),
            "last_update": last_update,
            "pm10": None, "pm25": None, "temperature": None, "relative_humidity": None,
            "sparkline_pm10": [None if np.isnan(v) else float(v) for v in spark.to_numpy()],
            "fallback": None,
        }
        if row is not None and s.status != "OFFLINE" and not np.isnan(row["pm10"]):
            entry.update({
                "pm10": round(float(row["pm10"]), 1), "pm25": round(float(row["pm25"]), 1),
                "temperature": round(float(row["temperature"]), 1),
                "relative_humidity": round(float(row["relative_humidity"]), 1),
            })
        if s.status == "OFFLINE":
            neigh = [n for n in site.SENSORS if n.zone_id == s.zone_id and n.sensor_id in online_vals]
            neigh = sorted(neigh, key=lambda n: site.distance(n.position, s.position))[:3]
            est = idw(s.position, [n.position for n in neigh], [online_vals[n.sensor_id] for n in neigh])
            entry["fallback"] = {
                "method": "IDW (power 2) from nearest online nodes in zone",
                "neighbours": [n.sensor_id for n in neigh],
                "estimated_pm10": round(est, 1),
                "expected_error_pct": round(loo_mape, 1) if loo_mape is not None else None,
            }
        sensors_out.append(entry)

    zones = {}
    for z in site.ZONES:
        members = [e for e in sensors_out if e["zone_id"] == z.zone_id]
        n = len(members)
        online = sum(1 for e in members if e["status"] == "ONLINE")
        warning = sum(1 for e in members if e["status"] == "WARNING")
        offline = sum(1 for e in members if e["status"] == "OFFLINE")
        confidence = (online + 0.6 * warning + 0.35 * offline) / max(n, 1)
        zones[z.zone_id] = {
            "sensors": n, "online": online, "warning": warning, "offline": offline,
            "data_confidence": round(confidence, 2),
            "confidence_reduced": confidence < 0.9,
        }

    counts = {
        "total": len(sensors_out),
        "online": sum(1 for e in sensors_out if e["status"] == "ONLINE"),
        "warning": sum(1 for e in sensors_out if e["status"] == "WARNING"),
        "offline": sum(1 for e in sensors_out if e["status"] == "OFFLINE"),
    }
    counts["reporting"] = counts["online"] + counts["warning"]
    return {"time": str(t), "counts": counts, "sensors": sensors_out, "zones": zones,
            "idw_loo_mape_pct": round(loo_mape, 1) if loo_mape is not None else None}
