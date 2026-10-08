"""
data_loader.py -- Observed dataset preparation (the data every model consumes).

Turns raw telemetry (synthetic simulator output or a real CSV export) into the
clean, aligned structure every model consumes:

* zone-level PM10 / PM2.5 (hygroscopic-corrected, gap-filled, QA-reported)
* site meteorology and the regional reference (background) station
* site activity logs (construction intensity, haul-truck trips, traffic counts)
* the intervention log and a modelled surface-moisture state per zone

Only information that would be observable on a real site is exposed here; the
simulator's ground-truth source contributions are kept separately in ``truth``
and are used solely to *evaluate* the attribution model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

import site_model as site
from data_cleaning import TelemetryDataCleaner, DataValidationError
from simulator import STEP_MIN, SyntheticDataset

logger = logging.getLogger(__name__)

ZONE_IDS = [z.zone_id for z in site.ZONES]


@dataclass
class ObservedData:
    pm10: pd.DataFrame                       # index=timestamp, columns=zone ids
    pm25: pd.DataFrame
    weather: pd.DataFrame                    # index=timestamp
    construction: pd.DataFrame               # actual intensity (columns=zones)
    planned_construction: pd.DataFrame       # site schedule (known in advance)
    trucks: pd.DataFrame                     # truck trips per step (columns=zones)
    intervention_active: pd.DataFrame        # 1 while spraying/sweeping (columns=zones)
    surface_moisture: pd.DataFrame           # modelled 0-1 wetness (columns=zones)
    interventions: pd.DataFrame
    qa: Dict[str, Dict[str, object]]
    meta: Dict[str, object]
    truth: Optional[pd.DataFrame] = None     # synthetic ground truth (evaluation only)
    counterfactual: Optional[pd.DataFrame] = None
    sensor_frame: Optional[pd.DataFrame] = None

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.pm10.index

    def slice(self, end: pd.Timestamp) -> "ObservedData":
        """Returns a copy containing only data up to and including ``end``."""
        def cut(df: pd.DataFrame) -> pd.DataFrame:
            return df.loc[:end]
        inter = self.interventions[self.interventions["dispatched_at"] <= end]
        return ObservedData(cut(self.pm10), cut(self.pm25), cut(self.weather), cut(self.construction),
                            self.planned_construction, cut(self.trucks), cut(self.intervention_active),
                            cut(self.surface_moisture), inter, self.qa, self.meta, self.truth,
                            self.counterfactual, self.sensor_frame)


def surface_moisture_model(index: pd.DatetimeIndex, rain_mm: np.ndarray, temperature: np.ndarray,
                           spraying: np.ndarray) -> np.ndarray:
    """Simple bucket model of road/soil surface moisture (0 dry - 1 saturated).

    Wetting from rain and water-tanker spraying, evaporation that scales with
    sunshine and temperature. Parameters are site-calibration assumptions, not
    the simulator's internal values.
    """
    hour = (index.hour + index.minute / 60.0).to_numpy()
    solar = np.clip(np.sin(np.pi * (hour - 6.5) / 12.0), 0, None)
    evap = 0.012 + 0.02 * solar * np.clip((temperature - 15) / 20, 0.2, 1.3)
    out = np.empty(len(index))
    cur = 0.0
    for i in range(len(index)):
        cur = min(1.0, cur + 0.5 * rain_mm[i] + 0.09 * spraying[i]) * (1 - evap[i])
        out[i] = cur
    return out


def _intervention_masks(index: pd.DatetimeIndex, interventions: pd.DataFrame) -> Dict[str, Dict[str, np.ndarray]]:
    spray = {z: np.zeros(len(index)) for z in ZONE_IDS}
    sweep = {z: np.zeros(len(index)) for z in ZONE_IDS}
    for _, rec in interventions.iterrows():
        z = rec["zone_id"]
        if z not in spray:
            continue
        m = (index >= rec["started_at"]) & (index < rec["ended_at"])
        if rec["action"] in ("targeted_sprinkling", "combined"):
            spray[z][m] = 1.0
        if rec["action"] in ("mechanical_sweeping", "combined"):
            sweep[z][m] = 1.0
    return {"spray": spray, "sweep": sweep}


def prepare_from_synthetic(ds: SyntheticDataset, cleaner: Optional[TelemetryDataCleaner] = None,
                           cutoff: Optional[pd.Timestamp] = None) -> ObservedData:
    """Builds ObservedData from the simulator output.

    ``cutoff`` limits which interventions are considered *known* (the hero
    intervention is only known after the operator approves it).
    """
    cleaner = cleaner or TelemetryDataCleaner()
    zf = ds.zone_frame
    weather = ds.weather.set_index("timestamp")
    index = weather.index

    qa: Dict[str, Dict[str, object]] = {}
    pm10 = pd.DataFrame(index=index)
    pm25 = pd.DataFrame(index=index)
    for z in ZONE_IDS:
        sub = zf[zf["zone_id"] == z].set_index("timestamp")
        c10, r10 = cleaner.clean_series(sub["pm10_raw"], weather["relative_humidity"], STEP_MIN)
        c25, r25 = cleaner.clean_series(sub["pm25_raw"], weather["relative_humidity"], STEP_MIN)
        c25 = np.minimum(c25, c10)                       # physical consistency PM2.5 <= PM10
        pm10[z] = c10
        pm25[z] = c25
        qa[z] = {"pm10": r10, "pm25": r25}

    inter = ds.interventions.copy()
    if cutoff is not None:
        inter = inter[inter["dispatched_at"] <= cutoff]
    masks = _intervention_masks(index, inter)

    construction = pd.DataFrame({z: weather[f"construction_{z}"] for z in ZONE_IDS}, index=index)
    planned = pd.DataFrame({z: weather[f"planned_construction_{z}"] for z in ZONE_IDS}, index=index)
    trucks = pd.DataFrame({z: weather[f"trucks_{z}"] for z in ZONE_IDS}, index=index)
    active = pd.DataFrame({z: np.maximum(masks["spray"][z], masks["sweep"][z]) for z in ZONE_IDS}, index=index)
    moisture = pd.DataFrame({
        z: surface_moisture_model(index, weather["rain_mm"].to_numpy(), weather["temperature"].to_numpy(),
                                  masks["spray"][z])
        for z in ZONE_IDS
    }, index=index)

    wx_cols = ["temperature", "relative_humidity", "wind_speed", "wind_dir_from", "mixing_height", "ventilation",
               "rain_mm", "reference_pm10", "vehicles", "workday"]
    meta = dict(ds.meta)
    meta["source"] = "synthetic"
    return ObservedData(pm10, pm25, weather[wx_cols].copy(), construction, planned, trucks, active, moisture,
                        inter.reset_index(drop=True), qa, meta, truth=zf, counterfactual=ds.counterfactual,
                        sensor_frame=ds.sensor_frame)


REQUIRED_CSV_COLUMNS = {"timestamp", "zone_id", "pm10", "pm25", "temperature", "relative_humidity",
                        "wind_speed", "wind_dir_from"}


def load_observed_csv(path: str | Path, cleaner: Optional[TelemetryDataCleaner] = None) -> ObservedData:
    """Loads real site telemetry from a long-format CSV.

    Required columns: timestamp, zone_id, pm10, pm25, temperature,
    relative_humidity, wind_speed (m/s), wind_dir_from (deg).
    Optional columns (sensible defaults are used when absent): mixing_height,
    rain_mm, reference_pm10, construction_intensity, planned_construction,
    truck_trips, vehicles, intervention_active.

    Zone IDs must match the site model (A-D) or the site model must be edited.
    """
    cleaner = cleaner or TelemetryDataCleaner()
    df = pd.read_csv(path, parse_dates=["timestamp"])
    missing = REQUIRED_CSV_COLUMNS - set(df.columns)
    if missing:
        raise DataValidationError(f"CSV is missing required columns: {sorted(missing)}")
    unknown = set(df["zone_id"].unique()) - set(ZONE_IDS)
    if unknown:
        raise DataValidationError(f"Unknown zone ids {sorted(unknown)}; expected {ZONE_IDS}")

    df = df.sort_values("timestamp")
    df["timestamp"] = df["timestamp"].dt.floor(f"{STEP_MIN}min")
    index = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq=f"{STEP_MIN}min")

    site_cols = ["temperature", "relative_humidity", "wind_speed", "wind_dir_from", "mixing_height", "rain_mm",
                 "reference_pm10", "vehicles"]
    wx = df.groupby("timestamp")[[c for c in site_cols if c in df.columns]].mean().reindex(index).interpolate(limit=12).ffill().bfill()
    hour = (index.hour + index.minute / 60.0).to_numpy()
    if "mixing_height" not in wx:
        solar = np.clip(np.sin(np.pi * (hour - 6.5) / 12.0), 0, None)
        wx["mixing_height"] = 250 + 900 * solar ** 1.3
    if "rain_mm" not in wx:
        wx["rain_mm"] = 0.0
    if "vehicles" not in wx:
        wx["vehicles"] = 20.0
    wx["ventilation"] = wx["wind_speed"] * wx["mixing_height"]
    wx["workday"] = ((index.dayofweek < 6)).astype(float)

    pm10 = pd.DataFrame(index=index)
    pm25 = pd.DataFrame(index=index)
    qa: Dict[str, Dict[str, object]] = {}
    per_zone: Dict[str, pd.DataFrame] = {}
    for z in ZONE_IDS:
        sub = df[df["zone_id"] == z].groupby("timestamp").mean(numeric_only=True).reindex(index)
        per_zone[z] = sub
        c10, r10 = cleaner.clean_series(sub["pm10"], wx["relative_humidity"], STEP_MIN)
        c25, r25 = cleaner.clean_series(sub["pm25"], wx["relative_humidity"], STEP_MIN)
        pm10[z], pm25[z] = c10, np.minimum(c25, c10)
        qa[z] = {"pm10": r10, "pm25": r25}
    if "reference_pm10" not in wx:
        wx["reference_pm10"] = pm10.min(axis=1).rolling(12, min_periods=1).median()

    def zcol(name: str, default: float) -> pd.DataFrame:
        return pd.DataFrame({z: per_zone[z][name].fillna(default) if name in per_zone[z] else default
                             for z in ZONE_IDS}, index=index)

    construction = zcol("construction_intensity", 0.0)
    planned = zcol("planned_construction", 0.0)
    if (planned.values == 0).all():
        planned = construction.copy()
    trucks = zcol("truck_trips", 0.0)
    active = zcol("intervention_active", 0.0)
    moisture = pd.DataFrame({z: surface_moisture_model(index, wx["rain_mm"].to_numpy(), wx["temperature"].to_numpy(),
                                                       active[z].to_numpy()) for z in ZONE_IDS}, index=index)
    inter = pd.DataFrame(columns=["zone_id", "action", "dispatched_at", "started_at", "ended_at", "asset_type", "source"])
    meta = {"synthetic": False, "source": str(path), "step_minutes": STEP_MIN, "zones": ZONE_IDS,
            "start": str(index[0]), "end": str(index[-1]), "now": str(index[-1])}
    return ObservedData(pm10, pm25, wx, construction, planned, trucks, active, moisture, inter, qa, meta)


def load_site_dataset(seed: int = 42, cutoff: Optional[pd.Timestamp] = None):
    """Generate the Green Valley dataset and prepare it, without booting the server -- handy in a
       notebook or an evaluation script:

           from data_loader import load_site_dataset
           raw, obs = load_site_dataset()          # obs.pm10 is a time x zone DataFrame

       Returns (raw SyntheticDataset, ObservedData). ``cutoff`` defaults to the demo "now"
       (simulator.NOW, 7 Oct 2026 14:30); pass a later time to include the hero intervention."""
    from simulator import NOW, generate_dataset
    raw = generate_dataset(seed=seed)
    return raw, prepare_from_synthetic(raw, cutoff=NOW if cutoff is None else cutoff)
