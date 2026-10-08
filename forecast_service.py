"""
forecast_service.py -- Orchestration layer between the Flask routes and the models.

server.py never touches a model directly: it parses a request, calls one method on
PlatformService below, and turns the result (or a ServiceError) into an HTTP response. Everything
stateful lives here:

  - the trained platform (pipeline.Platform: forecaster, attribution, DUSTWISE, learning loop),
    loaded once and shared by every request (about 30 s on first use)
  - the mutable demo session: the hero event's lifecycle
    PENDING_APPROVAL -> APPROVED -> ACTIVE -> VERIFIED, where verifying applies a real update to
    the Bayesian learning loop
  - request validation for the JSON bodies (small, dependency-free schemas instead of pydantic,
    so a bad value comes back as a clear HTTP 422 naming the field)
  - the snapshot that both the live API and the offline dashboard (static/data/snapshot.js) use,
    so the two can never drift apart

Expected failures raise ServiceError subclasses whose message is safe to show to the user;
anything else is a bug and is logged server-side by server.log_and_hide().
"""
from __future__ import annotations

import copy
import csv
import io
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

import chatbot
import config
import digital_twin as twin
import site_model as site
from data_cleaning import TelemetryDataCleaner
from pipeline import HERO_EVENT_ID, HERO_TIMES, HERO_ZONE, Platform, json_safe
from verification import expected_reduction_pct

logger = logging.getLogger("econexus.service")

LIFECYCLE = ["PENDING_APPROVAL", "APPROVED", "ACTIVE", "VERIFIED"]


# -- ERRORS ----------------------------------------------------------------------------------------
class ServiceError(Exception):
    """An expected, explainable failure; ``status`` is the HTTP status server.py returns and the
       message is safe to show to the user."""
    status = 400

    def __init__(self, message: str, status: Optional[int] = None, details: Any = None) -> None:
        super().__init__(message)
        if status is not None:
            self.status = status
        self.details = details


class NotFoundError(ServiceError):
    status = 404


class ConflictError(ServiceError):
    status = 409


class ValidationError(ServiceError):
    status = 422


# -- REQUEST VALIDATION ----------------------------------------------------------------------------
_REQUIRED = object()


class Field:
    """One field of a JSON request body: type, default, and bounds."""

    def __init__(self, kind: str, default: Any = _REQUIRED, ge: float | None = None, le: float | None = None,
                 choices: tuple | None = None, max_length: int | None = None, nullable: bool = False):
        self.kind, self.default, self.ge, self.le = kind, default, ge, le
        self.choices, self.max_length, self.nullable = choices, max_length, nullable

    def parse(self, name: str, value: Any) -> Any:
        if value is None and (self.nullable or self.default is None):
            return None
        if self.kind == 'float':
            if isinstance(value, bool):
                raise ValueError('must be a number')
            try:
                value = float(value)
            except (TypeError, ValueError):
                raise ValueError('must be a number')
            if value != value or value in (float('inf'), float('-inf')):
                raise ValueError('must be a finite number')
            if self.ge is not None and value < self.ge:
                raise ValueError(f'must be >= {self.ge:g}')
            if self.le is not None and value > self.le:
                raise ValueError(f'must be <= {self.le:g}')
            return value
        if self.kind == 'bool':
            if isinstance(value, bool):
                return value
            if isinstance(value, str) and value.lower() in ('true', 'false', '1', '0', 'yes', 'no'):
                return value.lower() in ('true', '1', 'yes')
            raise ValueError('must be true or false')
        if self.kind == 'str':
            if not isinstance(value, str):
                raise ValueError('must be text')
            value = value.strip()
            if self.max_length is not None and len(value) > self.max_length:
                raise ValueError(f'must be at most {self.max_length} characters')
            if self.choices and value not in self.choices:
                raise ValueError(f'must be one of {", ".join(self.choices)}')
            return value
        if self.kind == 'datetime':
            try:
                return pd.Timestamp(value).to_pydatetime().isoformat()
            except (TypeError, ValueError):
                raise ValueError('must be an ISO 8601 date-time')
        raise AssertionError(self.kind)


TWIN_REQUEST = {
    'construction_pct': Field('float', 100.0, ge=0, le=200),
    'traffic_pct': Field('float', 100.0, ge=0, le=200),
    'wind_speed': Field('float', 0.85, ge=0.2, le=12.0),          # m/s
    'wind_from_deg': Field('float', 306.0, ge=0, le=360),
    'intervention': Field('str', 'none', choices=('none', 'targeted', 'combined')),
    'include_grid': Field('bool', True),
}
OPERATOR_ACTION = {
    'operator': Field('str', 'Site Operations Lead', max_length=80),
    'note': Field('str', None, max_length=500),
}
SENSOR_TELEMETRY = {
    'node_id': Field('str', max_length=40),
    'timestamp': Field('datetime', None),
    'latitude': Field('float', ge=-90, le=90),
    'longitude': Field('float', ge=-180, le=180),
    'installation_height_m': Field('float', 3.5, ge=0.5, le=50),
    'pm25': Field('float', ge=0, le=1000),                         # ug/m3
    'pm10': Field('float', ge=0, le=2000),
    'no2': Field('float', 0.0, ge=0, le=1000),
    'co': Field('float', 0.0, ge=0, le=100),
    'so2': Field('float', 0.0, ge=0, le=1000),
    'o3': Field('float', 0.0, ge=0, le=1000),
    'temperature': Field('float', ge=-20, le=60),
    'relative_humidity': Field('float', ge=0, le=100),
    'wind_speed': Field('float', 1.0, ge=0, le=100),
    'wind_direction_deg': Field('float', 0.0, ge=0, le=360),
    'battery_level_pct': Field('float', 100.0, ge=0, le=100),
    'firmware_version': Field('str', 'v2.4.1', max_length=20),
}


def validate(schema: Dict[str, Field], body: Any) -> Dict[str, Any]:
    """Returns the parsed body with defaults filled in, or raises ValidationError listing every
       bad field (HTTP 422). Unknown keys are ignored."""
    if body is None:
        body = {}
    if not isinstance(body, dict):
        raise ValidationError('Request body must be a JSON object', details=[])
    out, errors = {}, []
    for name, f in schema.items():
        if name not in body or (body[name] is None and not f.nullable and f.default is not _REQUIRED):
            if f.default is _REQUIRED:
                errors.append({'field': name, 'message': 'is required'})
            else:
                out[name] = f.default
            continue
        try:
            out[name] = f.parse(name, body[name])
        except ValueError as e:
            errors.append({'field': name, 'message': str(e), 'value': body[name]})
    if errors:
        summary = '; '.join(f"{e['field']} {e['message']}" for e in errors)
        raise ValidationError(f'Invalid request: {summary}', details=errors)
    return out


# -- THE SERVICE -----------------------------------------------------------------------------------
class PlatformService:
    """The trained platform plus the demo session state (thread-safe)."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.platform: Optional[Platform] = None
        self.base_state: Dict[str, Any] = {}
        self.state: Dict[str, Any] = {}
        self.hero: Dict[str, Any] = {}
        self.extras: Dict[str, Any] = {}
        self.static: Dict[str, Any] = {}
        self.cleaner = TelemetryDataCleaner()
        self.loaded_at: Optional[str] = None

    # ------------------------------------------------------------------ lifecycle of the service
    @property
    def ready(self) -> bool:
        return self.platform is not None

    def ensure(self) -> "PlatformService":
        """Loads the models and builds the site state once (about 30 s)."""
        if self.platform is None:
            with self.lock:
                if self.platform is None:
                    p = Platform()
                    self.base_state = json_safe(p.build_state())
                    self.hero = json_safe(p.hero_frames(self.base_state))
                    self.extras = json_safe({"sustainability": p.sustainability(), "model": p.model_summary()})
                    self.static = json_safe({"site": site.site_geojson_like(), "twin": twin.export_geometry(),
                                             "twin_default": twin.simulate(include_grid=False),
                                             "twin_comparison": twin.compare_scenarios()})
                    # the learning loop as bootstrapped from history, so a demo reset is instant
                    # (re-running the bootstrap takes ~10 s)
                    self._learning0 = copy.deepcopy(p.learning)
                    self.platform = p
                    self.loaded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
                    self._reset_state()
        return self

    def _reset_state(self) -> None:
        self.state = copy.deepcopy(self.base_state)
        self.state["lifecycle"] = {"event_id": HERO_EVENT_ID, "stage": "PENDING_APPROVAL", "log": []}

    def _st(self) -> Dict[str, Any]:
        self.ensure()
        return self.state

    # ------------------------------------------------------------------ read
    def health(self) -> Dict[str, Any]:
        return {"status": "HEALTHY" if self.ready else "STARTING", "platform": config.APP_NAME,
                "version": config.VERSION,
                "model_backend": self.platform.forecaster.meta.get("backend") if self.ready else None,
                "copilot": "groq" if config.GROQ_ENABLED else "local",
                "synthetic_data": True, "loaded_at": self.loaded_at,
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    def snapshot(self, live: bool = True) -> Dict[str, Any]:
        """Everything the dashboard needs in one call -- the same shape as static/data/snapshot.js
           (train.export_snapshot writes this with live=False)."""
        st = self._st()
        snap = {"version": 4, "live": live, "state": st, "hero": self.hero, **self.static, **self.extras}
        snap["copilot"] = chatbot.offline_pack(snap)
        return snap

    def context(self) -> Dict[str, Any]:
        """What the Copilot reasons over: the live state plus model card and sustainability."""
        return {"state": self._st(), "hero": self.hero, **self.static, **self.extras}

    def section(self, *path: str) -> Any:
        node: Any = self._st()
        for key in path:
            node = node[key]
        return node

    def zone(self, zone_id: str) -> Dict[str, Any]:
        st = self._st()
        zone_id = zone_id.upper()
        z = next((z for z in st["zones"] if z["zone_id"] == zone_id), None)
        if z is None:
            raise NotFoundError(f"Unknown zone {zone_id} (zones are {', '.join(site.ZONE_BY_ID)})")
        return {**z, "attribution": st["attribution"].get(zone_id),
                "priority": st["dustwise"]["priorities"].get(zone_id),
                "sensors": st["sensors"]["zones"].get(zone_id)}

    def forecast(self, zone_id: str, tab: str) -> Dict[str, Any]:
        fc = self._st()["forecasts"]
        zone_id = zone_id.upper()
        if zone_id not in fc:
            raise NotFoundError(f"Unknown zone {zone_id}")
        if tab not in fc[zone_id]:
            raise ServiceError(f"tab must be one of {', '.join(fc[zone_id])}")
        return {"zone_id": zone_id, "tab": tab, **fc[zone_id][tab]}

    def explain(self, zone_id: str, horizon: str) -> Dict[str, Any]:
        self.ensure()
        zone_id = zone_id.upper()
        if zone_id not in site.ZONE_BY_ID:
            raise NotFoundError(f"Unknown zone {zone_id}")
        try:
            return json_safe(self.platform.forecast_explanation(zone_id, self.platform.now, horizon))
        except KeyError:
            raise ServiceError(f"Unknown horizon {horizon}")

    def event(self, event_id: str) -> Dict[str, Any]:
        st = self._st()
        for ev in st["events"]["active"] + st["events"]["history"]:
            if ev["event_id"] == event_id:
                return ev
        raise NotFoundError(f"Unknown event {event_id}")

    def actions(self, zone_id: str) -> Dict[str, Any]:
        comps = self._st()["dustwise"]["comparisons"]
        zone_id = zone_id.upper()
        if zone_id not in comps:
            raise NotFoundError(f"Unknown zone {zone_id}")
        return comps[zone_id]

    def model(self) -> Dict[str, Any]:
        self.ensure()
        return self.extras["model"]

    def sustainability(self) -> Dict[str, Any]:
        self.ensure()
        return self.extras["sustainability"]

    def impact(self, event_id: str) -> Dict[str, Any]:
        ev = self.event(event_id)
        if ev.get("impact"):
            return ev["impact"]
        if event_id == HERO_EVENT_ID:
            return {"available": False, "reason": "Event not verified yet", "stage": self._st()["lifecycle"]["stage"]}
        raise NotFoundError("No impact record for this event")

    def briefing(self) -> Dict[str, Any]:
        return chatbot.briefing(self.context())

    # ------------------------------------------------------------------ digital twin
    @staticmethod
    def twin_simulate(body: Any) -> Dict[str, Any]:
        r = validate(TWIN_REQUEST, body)
        return json_safe(twin.simulate(construction_pct=r["construction_pct"], traffic_pct=r["traffic_pct"],
                                       wind_speed=r["wind_speed"], wind_from_deg=r["wind_from_deg"],
                                       intervention=r["intervention"], include_grid=r["include_grid"]))

    @staticmethod
    def twin_compare(body: Any) -> Dict[str, Any]:
        r = validate(TWIN_REQUEST, body)
        return json_safe(twin.compare_scenarios(construction_pct=r["construction_pct"], traffic_pct=r["traffic_pct"],
                                                wind_speed=r["wind_speed"], wind_from_deg=r["wind_from_deg"]))

    # ------------------------------------------------------------------ event lifecycle
    @staticmethod
    def _advance(ev: Dict[str, Any], stage: str) -> None:
        order = [i["stage"] for i in ev["timeline"]]
        if stage in order:
            idx = order.index(stage)
            for i, item in enumerate(ev["timeline"]):
                item["status"] = "done" if i <= idx else ("current" if i == idx + 1 else "upcoming")

    def _require_hero(self, event_id: str) -> None:
        self.event(event_id)  # 404 for unknown ids
        if event_id != HERO_EVENT_ID:
            raise ServiceError("Only the active DUSTWISE recommendation can be actioned in this demo build")

    def approve(self, event_id: str, body: Any = None) -> Dict[str, Any]:
        action = validate(OPERATOR_ACTION, body)
        self._require_hero(event_id)
        st = self._st()
        with self.lock:
            lc = st["lifecycle"]
            if lc["stage"] != "PENDING_APPROVAL":
                raise ConflictError(f"Event is already {lc['stage']}")
            ev = self.event(event_id)
            rec = st["insight"]["recommendation"]
            ev["intervention_status"] = ev["status"] = "APPROVED"
            for item in ev["timeline"]:
                if item["stage"] == "APPROVED":
                    item["detail"] = f"{action['operator']} approved {rec['action'].lower()} with {rec['resource']}."
            self._advance(ev, "APPROVED")
            for row in st["dustwise"]["allocation"]["allocations"]:
                if row["asset_id"] == rec["asset_id"]:
                    row["status"] = "Approved"
            lc["stage"] = "APPROVED"
            lc["log"].append({"time": str(HERO_TIMES["approved"]), "action": "approve", "operator": action["operator"],
                              "note": action["note"]})
            logger.info("%s approved by %s", event_id, action["operator"])
        return {"event": ev, "lifecycle": lc}

    def start(self, event_id: str) -> Dict[str, Any]:
        self._require_hero(event_id)
        st = self._st()
        with self.lock:
            lc = st["lifecycle"]
            if lc["stage"] != "APPROVED":
                raise ConflictError(f"Event must be APPROVED to start (is {lc['stage']})")
            ev = self.event(event_id)
            ev["intervention_status"] = ev["status"] = "ACTIVE"
            self._advance(ev, "ACTIVE")
            for row in st["dustwise"]["allocation"]["allocations"]:
                if row["asset_id"] == st["insight"]["recommendation"]["asset_id"]:
                    row["status"] = "Active"
            st["status_strip"]["interventions_active"] += 1
            lc["stage"] = "ACTIVE"
            lc["log"].append({"time": str(HERO_TIMES["initiated"]), "action": "start"})
        return {"event": ev, "lifecycle": lc}

    def verify(self, event_id: str) -> Dict[str, Any]:
        self._require_hero(event_id)
        st = self._st()
        with self.lock:
            lc = st["lifecycle"]
            if lc["stage"] != "ACTIVE":
                raise ConflictError(f"Event must be ACTIVE to verify (is {lc['stage']})")
            ev = self.event(event_id)
            v = self.hero["verification"]
            ev["impact"] = v
            ev["intervention_status"] = "VERIFIED" if v["verified"] else "INCONCLUSIVE"
            ev["status"] = "VERIFIED"
            ev["outcome"] = (f"Observed {v['observed_change_pct']:+.1f}%, "
                             f"model-estimated effect {v['model_estimated_effect_pct']:+.1f}%")
            self._advance(ev, "VERIFIED")
            # a real learning-loop update, applied once per verified event
            p = self.platform
            key = st["insight"]["recommendation"]["action_key"]
            upd = p.learning.record_outcome(key, v["after_pm10"], v["counterfactual_pm10"],
                                            expected_reduction_pct(st["attribution"][HERO_ZONE], key), event_id,
                                            v["kpis"]["peak_prediction_error_pct"])
            st["learning"] = json_safe(p.learning.summary())
            lc["stage"] = "VERIFIED"
            lc["log"].append({"time": str(HERO_TIMES["verified"]), "action": "verify"})
            logger.info("%s verified: observed %+.1f%%", event_id, v["observed_change_pct"])
        return {"event": ev, "impact": v, "learning_update": json_safe(upd), "learning": st["learning"], "lifecycle": lc}

    def escalate(self, event_id: str, body: Any = None) -> Dict[str, Any]:
        action = validate(OPERATOR_ACTION, body)
        ev = self.event(event_id)
        with self.lock:
            ev.setdefault("escalations", []).append({"by": action["operator"], "note": action["note"],
                                                     "at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        return {"event_id": event_id, "escalated": True,
                "note": "Escalation logged for the site environmental manager."}

    def reset_demo(self) -> Dict[str, Any]:
        self.ensure()
        with self.lock:
            p = self.platform
            p.learning = copy.deepcopy(self._learning0)
            self._reset_state()
        return {"reset": True, "stage": self.state["lifecycle"]["stage"]}

    # ------------------------------------------------------------------ ingest & export
    def ingest(self, body: Any) -> Dict[str, Any]:
        rec = validate(SENSOR_TELEMETRY, body)
        if rec["timestamp"] is None:
            rec["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if rec["pm10"] < rec["pm25"]:          # PM10 always includes PM2.5; small optical noise is clamped
            rec["pm10"] = rec["pm25"]
        return {"status": "INGESTED", "calibrated": json_safe(self.cleaner.clean_telemetry_record(rec))}

    def export_csv(self) -> str:
        """Last 24 h of zone telemetry as CSV text."""
        self.ensure()
        p, t = self.platform, self.platform.now
        out = io.StringIO()
        w = csv.writer(out)
        w.writerow(["timestamp", "zone_id", "pm10_ugm3", "pm25_ugm3", "wind_speed_ms", "wind_dir_from_deg",
                    "temperature_c", "relative_humidity_pct", "reference_pm10_ugm3", "synthetic"])
        sub = p.obs.pm10.loc[t - pd.Timedelta(hours=24):t]
        for ts in sub.index:
            wx = p.obs.weather.loc[ts]
            for z in sub.columns:
                w.writerow([ts.isoformat(), z, round(float(p.obs.pm10[z].loc[ts]), 1),
                            round(float(p.obs.pm25[z].loc[ts]), 1),
                            round(float(wx["wind_speed"]), 2), round(float(wx["wind_dir_from"])),
                            round(float(wx["temperature"]), 1), round(float(wx["relative_humidity"]), 1),
                            round(float(wx["reference_pm10"]), 1), True])
        return out.getvalue()

    def report(self) -> Dict[str, Any]:
        st = self._st()
        return {"platform": config.APP_NAME, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "site": site.SITE_NAME, "data_label": st["meta"]["data_label"], "status": st["status_strip"],
                "briefing": chatbot.briefing(self.context())["text"],
                "zones": st["zones"], "events": st["events"]["active"],
                "dustwise": {"priorities": st["dustwise"]["priorities"], "allocation": st["dustwise"]["allocation"]},
                "learning": st["learning"], "model": self.extras["model"]["card"]}

    def zone_ids(self) -> List[str]:
        return list(site.ZONE_BY_ID)
