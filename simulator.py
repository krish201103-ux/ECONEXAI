"""
simulator.py -- Physically-motivated synthetic telemetry generator.

Real site telemetry is not available for this prototype, so the platform is
trained and demonstrated on a synthetic dataset for Green Valley Township that
is produced by a transparent, physics-motivated simulator. Everything that
comes out of this module is labelled ``synthetic`` downstream.

The simulator models, at 5-minute resolution:

* Meteorology - seasonal + diurnal temperature, humidity, wind (with monsoon
  easterlies and post-monsoon north-westerlies), mixing height, rain events.
* Site activity - work-day construction schedules per zone, haul-truck trips on
  the unpaved haul road, commuter traffic with rush-hour peaks.
* Concentrations - regional background (season, synoptic variability, rain
  wash-out) plus local increments from construction dust, haul-road dust
  resuspension and vehicle exhaust, scaled by atmospheric dispersion
  (ventilation = wind speed x mixing height) and transported between zones
  according to wind direction.
* Interventions - a historical dispatch policy (water tankers, sweepers,
  combined actions) whose effect is *emergent* from surface wetting, mist
  scavenging and sweeping in the physics, not a hard-coded percentage. The
  learning loop later estimates these effects from data.
* Sensors - zone readings are averages of noisy optical sensors with humidity
  bias (corrected later by the preprocessing module).

The final day (``HERO_DATE``) contains the scripted hero scenario: a Zone B
excavation and haul surge under stagnant afternoon winds.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import site_model as site

logger = logging.getLogger(__name__)

STEP_MIN = 5
STEPS_PER_HOUR = 60 // STEP_MIN
START = pd.Timestamp("2026-06-09 00:00")
HERO_DATE = pd.Timestamp("2026-10-07")
NOW = pd.Timestamp("2026-10-07 14:30")          # "current time" in the demo
END = pd.Timestamp("2026-10-10 23:55")          # 3 days beyond "now" for weather/schedule outlooks
HOLIDAYS = {pd.Timestamp("2026-08-15").date(), pd.Timestamp("2026-10-02").date()}

ZONE_IDS = [z.zone_id for z in site.ZONES]

# Emission coefficients (ug/m3 of local increment at reference dispersion D = 1)
K_CONSTRUCTION = {"A": 27.0, "B": 34.0, "C": 0.0, "D": 0.0}
K_ROAD_DUST_PER_TRIP = 1.5          # unpaved haul road resuspension per truck trip / 5 min
K_TRUCK_EXHAUST = 0.40              # diesel exhaust per truck trip / 5 min
K_COMMUTER = 0.05                   # per vehicle / 5 min x zone road exposure
ROAD_EXPOSURE = {z.zone_id: z.road_exposure for z in site.ZONES}
OTHER_RESIDENTIAL = {"A": 0.0, "B": 0.0, "C": 3.0, "D": 2.5}   # cooking, gensets (evening)

# PM2.5 / PM10 mass fractions by source
F25 = {"background": 0.52, "dust": 0.21, "traffic": 0.62, "other": 0.45}

# True hygroscopic growth of the site aerosol (the cleaner assumes kappa = 0.25)
TRUE_KAPPA = 0.22


@dataclass
class SyntheticDataset:
    zone_frame: pd.DataFrame            # long format: one row per (timestamp, zone)
    weather: pd.DataFrame               # site-wide meteorology + activity drivers
    interventions: pd.DataFrame         # historical + hero intervention log
    counterfactual: pd.DataFrame        # hero day without the hero intervention (truth, for evaluation only)
    sensor_frame: pd.DataFrame          # per-sensor readings for the last 36 hours
    meta: Dict[str, object]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _interp_season(doy: np.ndarray, knots: List[Tuple[int, float]]) -> np.ndarray:
    xs = np.array([k[0] for k in knots], dtype=float)
    ys = np.array([k[1] for k in knots], dtype=float)
    return np.interp(doy, xs, ys)


def _ar1(rng: np.random.Generator, n: int, phi: float, sigma: float) -> np.ndarray:
    out = np.empty(n)
    out[0] = rng.normal(0, sigma / math.sqrt(1 - phi ** 2))
    eps = rng.normal(0, sigma, n)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + eps[i]
    return out


def _solar(hour: np.ndarray) -> np.ndarray:
    return np.clip(np.sin(np.pi * (hour - 6.5) / 12.0), 0.0, None)


def _wind_transport(wind_from: np.ndarray, ws: np.ndarray, src: Tuple[float, float],
                    dst: Tuple[float, float], length_scale: float = 220.0) -> np.ndarray:
    """Vectorised version of :func:`site.transport_weight` over time."""
    d = site.distance(src, dst)
    wind_to = (wind_from + 180.0) % 360.0
    ang = np.radians(site.bearing_deg(src, dst) - wind_to)
    align = (1.0 + np.cos(ang)) / 2.0
    directional = np.clip(ws / 1.5, 0.0, 1.0)
    shape = (1 - directional) * 0.3 + directional * align ** 2
    reach = length_scale * (0.5 + 0.9 * directional * align)
    return shape * np.exp(-d / reach)


def dispersion_factor(ventilation: np.ndarray) -> np.ndarray:
    """Concentration multiplier from ventilation (m2/s). D = 1 at ~1800 m2/s."""
    return np.clip(1800.0 / np.maximum(ventilation, 120.0), 0.3, 4.5) ** 0.75


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #

class TownshipSimulator:
    """Generates the synthetic Green Valley Township dataset."""

    def __init__(self, seed: int = 42, start: pd.Timestamp = START, end: pd.Timestamp = END):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.index = pd.date_range(start, end, freq=f"{STEP_MIN}min")
        self.n = len(self.index)

    # ---- meteorology -------------------------------------------------------
    def _meteorology(self) -> pd.DataFrame:
        rng, idx, n = self.rng, self.index, self.n
        hour = (idx.hour + idx.minute / 60.0).to_numpy()
        doy = idx.dayofyear.to_numpy().astype(float)
        solar = _solar(hour)

        # Rain events (daily probability by season)
        rain = np.zeros(n)
        days = pd.Index(idx.normalize().unique())
        for day in days:
            if day.normalize() == HERO_DATE:
                continue
            day_doy = day.dayofyear
            if rng.random() < float(np.interp(day_doy, [160, 178, 250, 262, 275, 290], [0.12, 0.45, 0.45, 0.15, 0.04, 0.03])):
                n_events = 1 + int(rng.random() < 0.35)
                for _ in range(n_events):
                    start_h = rng.choice([rng.uniform(13, 20), rng.uniform(0, 24)])
                    dur_h = rng.uniform(0.7, 4.5)
                    intensity = rng.gamma(2.0, 2.2)                       # mm/h
                    t0 = day + pd.Timedelta(hours=float(start_h))
                    mask = (idx >= t0) & (idx < t0 + pd.Timedelta(hours=float(dur_h)))
                    shape = np.sin(np.linspace(0, np.pi, mask.sum()) + 1e-3) if mask.sum() else 0
                    rain[mask] += intensity * (0.4 + shape) * STEP_MIN / 60.0
        raining = rain > 0.02

        t_season = _interp_season(doy, [(160, 34.5), (182, 31.5), (215, 30.0), (245, 30.5), (270, 29.5), (290, 27.0)])
        t_amp = _interp_season(doy, [(160, 7.0), (182, 4.5), (245, 4.5), (270, 6.5), (290, 7.5)])
        rain_cool = pd.Series(rain).rolling(36, min_periods=1).sum().to_numpy() * 0.6
        temperature = t_season + t_amp * np.sin(2 * np.pi * (hour - 9) / 24) - np.minimum(rain_cool, 6) \
            + _ar1(rng, n, 0.995, 0.08)

        rh_season = _interp_season(doy, [(160, 42), (182, 74), (215, 82), (245, 78), (265, 66), (290, 52)])
        rh = rh_season - 2.1 * (temperature - t_season) + np.minimum(rain_cool * 4, 25) + _ar1(rng, n, 0.99, 0.6)
        rh = np.clip(rh, 15, 98)

        ws_season = _interp_season(doy, [(160, 2.4), (182, 2.0), (245, 1.8), (265, 1.6), (290, 1.35)])
        synoptic = np.exp(_ar1(rng, n, 0.9985, 0.025))
        ws = ws_season * (0.5 + 0.95 * solar) * synoptic * np.exp(rng.normal(0, 0.12, n))
        ws = np.clip(ws, 0.2, 9.0)

        wd_season = _interp_season(doy, [(160, 285), (182, 120), (240, 115), (258, 200), (270, 300), (290, 305)])
        wind_from = (wd_season + _ar1(rng, n, 0.995, 2.2)) % 360

        mh_night = _interp_season(doy, [(160, 330), (182, 380), (245, 360), (270, 260), (290, 200)])
        mh_day = _interp_season(doy, [(160, 1650), (182, 950), (245, 1050), (270, 1250), (290, 1050)])
        mixing = mh_night + (mh_day - mh_night) * solar ** 1.3
        mixing = mixing * np.where(raining, 0.75, 1.0) * np.exp(_ar1(rng, n, 0.997, 0.02))

        # Post-monsoon stagnation/haze days: suppressed afternoon mixing and weak winds
        stagnation = np.ones(n)
        for day in days:
            if day.normalize() == HERO_DATE or day.dayofyear < 250:
                continue
            if rng.random() < 0.22:
                m = np.asarray(idx.normalize() == day)
                bump = np.exp(-((hour[m] - 15.0) ** 2) / 8.0)
                stagnation[m] = 1 - rng.uniform(0.35, 0.5) * bump
        ws = np.clip(ws * stagnation, 0.2, 9.0)
        mixing = mixing * stagnation

        met = pd.DataFrame({
            "temperature": temperature, "relative_humidity": rh, "wind_speed": ws,
            "wind_dir_from": wind_from, "mixing_height": mixing, "rain_mm": rain,
        }, index=self.index)
        return met

    # ---- activity ------------------------------------------------------------
    def _activity(self) -> pd.DataFrame:
        rng, idx, n = self.rng, self.index, self.n
        hour = (idx.hour + idx.minute / 60.0).to_numpy()
        dow = idx.dayofweek.to_numpy()
        is_holiday = np.array([d in HOLIDAYS for d in idx.date])
        workday = (dow < 6) & ~is_holiday

        shift = ((hour >= 8) & (hour < 18)).astype(float)
        shift *= np.where((hour >= 13) & (hour < 13.75), 0.45, 1.0)

        act = pd.DataFrame(index=self.index)
        days = idx.normalize()
        unique_days = days.unique()
        for zid in ZONE_IDS:
            base = site.ZONE_BY_ID[zid].construction_base
            if base <= 0:
                act[f"construction_{zid}"] = 0.0
                continue
            daily = {d: float(np.exp(rng.normal(0, 0.22))) * (1.35 if rng.random() < 0.18 else 1.0) for d in unique_days}
            mult = np.array([daily[d] for d in days])
            jitter = np.exp(_ar1(rng, n, 0.97, 0.05))
            act[f"construction_{zid}"] = np.clip(base * mult * shift * workday * jitter, 0, 1.6)

        haul_window = ((hour >= 7) & (hour < 19)).astype(float)
        haul_shape = 1 + 0.25 * np.sin(2 * np.pi * (hour - 9) / 10)
        lam_b = 6.0 * act["construction_B"].to_numpy() / max(site.ZONE_BY_ID["B"].construction_base, 1e-6) \
            * haul_window * haul_shape * 0.75
        act["trucks_B"] = rng.poisson(np.clip(lam_b, 0, None)).astype(float)
        lam_a = 1.4 * act["construction_A"].to_numpy() / site.ZONE_BY_ID["A"].construction_base * haul_window
        act["trucks_A"] = rng.poisson(np.clip(lam_a, 0, None)).astype(float)
        act["trucks_C"] = 0.0
        act["trucks_D"] = 0.0

        commute = 14 + 30 * np.exp(-((hour - 9.0) ** 2) / 1.2) + 38 * np.exp(-((hour - 18.5) ** 2) / 1.6) \
            + 8 * _solar(hour)
        commute = commute * np.where(dow >= 5, 0.7, 1.0) * np.where(hour < 5.5, 0.25, 1.0)
        act["vehicles"] = rng.poisson(commute).astype(float)

        # Planned construction (what the site schedule says), used as a known-future feature
        for zid in ZONE_IDS:
            planned = act[f"construction_{zid}"].rolling(6, min_periods=1, center=True).mean().to_numpy()
            act[f"planned_construction_{zid}"] = np.clip(planned * np.exp(rng.normal(0, 0.06, n)), 0, 1.6)
        act["workday"] = workday.astype(float)
        return act

    # ---- hero scenario overrides ---------------------------------------------
    @staticmethod
    def _hero_profile(times: pd.DatetimeIndex, knots: List[Tuple[str, float]]) -> np.ndarray:
        base = HERO_DATE
        xs = np.array([(pd.Timestamp(f"{base.date()} {t}") - base).total_seconds() / 60 for t, _ in knots])
        ys = np.array([v for _, v in knots])
        tm = np.array([(t - base).total_seconds() / 60 for t in times])
        return np.interp(tm, xs, ys)

    def _apply_hero(self, met: pd.DataFrame, act: pd.DataFrame) -> None:
        mask = self.index.normalize() == HERO_DATE
        t = self.index[mask]
        rng = np.random.default_rng(self.seed + 7)
        met.loc[mask, "rain_mm"] = 0.0
        met.loc[mask, "wind_speed"] = self._hero_profile(t, [
            ("00:00", 0.7), ("07:00", 0.9), ("10:00", 1.7), ("12:00", 1.5), ("13:30", 1.15), ("14:00", 1.0),
            ("14:30", 0.85), ("15:30", 0.8), ("16:30", 0.95), ("17:30", 1.3), ("20:00", 0.8), ("23:55", 0.7)
        ]) * np.exp(rng.normal(0, 0.04, mask.sum()))
        met.loc[mask, "wind_dir_from"] = self._hero_profile(t, [("00:00", 300), ("12:00", 302), ("15:00", 306), ("23:55", 310)]) \
            + rng.normal(0, 4, mask.sum())
        met.loc[mask, "mixing_height"] = self._hero_profile(t, [
            ("00:00", 190), ("07:00", 220), ("10:00", 700), ("12:00", 820), ("14:00", 700), ("15:00", 650),
            ("16:30", 560), ("18:30", 260), ("23:55", 190)
        ])
        met.loc[mask, "temperature"] = self._hero_profile(t, [("00:00", 21), ("06:00", 20), ("14:30", 31.5), ("18:00", 28), ("23:55", 22.5)])
        met.loc[mask, "relative_humidity"] = self._hero_profile(t, [("00:00", 68), ("06:00", 72), ("14:30", 46), ("18:00", 52), ("23:55", 63)])

        act.loc[mask, "construction_B"] = self._hero_profile(t, [
            ("00:00", 0), ("07:55", 0), ("08:30", 0.62), ("12:55", 0.66), ("13:05", 0.3), ("13:45", 0.3),
            ("13:55", 0.47), ("14:05", 0.66), ("14:15", 0.8), ("14:30", 0.86), ("14:40", 1.12), ("17:30", 1.1),
            ("18:00", 0.35), ("18:10", 0.0), ("23:55", 0)
        ])
        act.loc[mask, "trucks_B"] = np.round(self._hero_profile(t, [
            ("00:00", 0), ("06:55", 0), ("07:30", 4), ("12:55", 5), ("13:05", 2), ("13:50", 2), ("14:05", 5),
            ("14:15", 7), ("14:30", 9), ("14:40", 13), ("17:30", 12), ("18:55", 3), ("19:05", 0), ("23:55", 0)
        ]) + rng.normal(0, 0.5, mask.sum())).clip(0)
        act.loc[mask, "construction_A"] = self._hero_profile(t, [
            ("00:00", 0), ("07:55", 0), ("08:30", 0.62), ("12:25", 0.7), ("12:35", 0.3), ("13:10", 0.3),
            ("13:20", 1.25), ("13:45", 1.35), ("13:55", 0.8), ("14:10", 0.72), ("17:55", 0.72), ("18:05", 0), ("23:55", 0)
        ])
        act.loc[mask, "trucks_A"] = np.round(self._hero_profile(t, [
            ("00:00", 0), ("06:55", 0), ("07:30", 1), ("12:25", 1), ("13:15", 2), ("13:25", 7), ("13:45", 7),
            ("13:55", 2), ("17:30", 1), ("19:00", 0), ("23:55", 0)
        ]) + rng.normal(0, 0.4, mask.sum())).clip(0)
        for zid in ("A", "B"):
            act.loc[mask, f"planned_construction_{zid}"] = act.loc[mask, f"construction_{zid}"]

    # ---- interventions ---------------------------------------------------------
    def _plan_interventions(self, peak_proxy: pd.DataFrame, met: pd.DataFrame) -> List[Dict[str, object]]:
        """Historical dispatch policy, decided on a proxy concentration (before effects)."""
        rng = np.random.default_rng(self.seed + 3)
        records: List[Dict[str, object]] = []
        busy_until = {z: pd.Timestamp.min for z in ZONE_IDS}
        hero_start = HERO_DATE
        idx = self.index
        hour = idx.hour + idx.minute / 60
        for zid in ("A", "B", "D"):
            series = peak_proxy[zid].to_numpy()
            for i in range(0, len(idx), 3):
                ts = idx[i]
                if ts >= hero_start:
                    break
                if not (7.5 <= hour[i] <= 18.0):
                    continue
                if ts < busy_until[zid]:
                    continue
                threshold = 150 if zid != "D" else 120
                if series[i] > threshold and rng.random() < 0.55:
                    if zid == "D":
                        action = "targeted_sprinkling"
                    else:
                        action = rng.choice(["targeted_sprinkling", "mechanical_sweeping", "combined"], p=[0.6, 0.22, 0.18])
                    delay = int(rng.integers(5, 18))
                    duration = int(rng.integers(45, 95))
                    start = ts + pd.Timedelta(minutes=delay)
                    records.append({
                        "zone_id": zid, "action": str(action), "dispatched_at": ts, "started_at": start,
                        "ended_at": start + pd.Timedelta(minutes=duration),
                        "asset_type": "tanker" if action != "mechanical_sweeping" else "sweeper",
                        "source": "historical",
                    })
                    busy_until[zid] = start + pd.Timedelta(minutes=duration + 60)
        # Hero-day context: Sweeper 01 already sweeping Zone A (EV-041)
        records.append({
            "zone_id": "A", "action": "mechanical_sweeping",
            "dispatched_at": pd.Timestamp("2026-10-07 13:27"), "started_at": pd.Timestamp("2026-10-07 13:34"),
            "ended_at": pd.Timestamp("2026-10-07 16:30"), "asset_type": "sweeper", "source": "hero_context",
        })
        return records

    # ---- concentration physics --------------------------------------------------
    def _concentrations(self, met: pd.DataFrame, act: pd.DataFrame,
                        interventions: List[Dict[str, object]], rng: np.random.Generator,
                        background: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        n = self.n
        idx = self.index
        hour = (idx.hour + idx.minute / 60.0).to_numpy()
        ws = met["wind_speed"].to_numpy()
        wd = met["wind_dir_from"].to_numpy()
        vent = ws * met["mixing_height"].to_numpy()
        disp = dispersion_factor(vent)
        rain = met["rain_mm"].to_numpy()
        temp = met["temperature"].to_numpy()

        # Intervention masks
        spray = {z: np.zeros(n) for z in ZONE_IDS}
        sweep = {z: np.zeros(n) for z in ZONE_IDS}
        slow = {z: np.ones(n) for z in ZONE_IDS}
        for rec in interventions:
            z = rec["zone_id"]
            m = (idx >= rec["started_at"]) & (idx < rec["ended_at"])
            if rec["action"] in ("targeted_sprinkling", "combined"):
                spray[z][m] = 1.0
            if rec["action"] in ("mechanical_sweeping", "combined"):
                sweep[z][m] = 1.0
            if rec["action"] == "combined":
                slow[z][m] = 0.8

        # Surface wetness per zone (rain + spraying), decays with evaporation
        evap = 0.010 + 0.022 * _solar(hour) * np.clip((temp - 15) / 20, 0.2, 1.3)
        wet = {}
        sweep_effect = {}
        for z in ZONE_IDS:
            w = np.zeros(n)
            s_eff = np.zeros(n)
            cur, cur_s = 0.0, 0.0
            for i in range(n):
                cur = cur + rain[i] * 0.55 + spray[z][i] * 0.097
                cur = min(cur, 1.0)
                cur = cur * (1 - evap[i])
                w[i] = cur
                cur_s = max(cur_s * 0.985, 0.48 * sweep[z][i])
                s_eff[i] = cur_s
            wet[z] = w
            sweep_effect[z] = s_eff

        washout = np.exp(-pd.Series(rain).rolling(72, min_periods=1).sum().to_numpy() / 9.0)
        bg = background * washout

        # Local emissions by source and zone
        emis: Dict[str, Dict[str, np.ndarray]] = {}
        for z in ZONE_IDS:
            wet_z = wet[z]
            con = K_CONSTRUCTION[z] * act[f"construction_{z}"].to_numpy() * slow[z] * (1 - 0.78 * wet_z)
            road = K_ROAD_DUST_PER_TRIP * act[f"trucks_{z}"].to_numpy() * (1 - 0.88 * wet_z) * (1 - sweep_effect[z])
            traffic = K_TRUCK_EXHAUST * act[f"trucks_{z}"].to_numpy() + K_COMMUTER * act["vehicles"].to_numpy() * ROAD_EXPOSURE[z] * 10
            other = OTHER_RESIDENTIAL[z] * (np.exp(-((hour - 20.5) ** 2) / 3.0) + 0.25)
            scav = 1 - 0.14 * spray[z]                          # airborne mist scavenging while spraying
            emis[z] = {"construction": con * scav, "road_dust": road * scav, "traffic": traffic, "other": other}

        out: Dict[str, Dict[str, np.ndarray]] = {}
        centroids = {z.zone_id: z.centroid for z in site.ZONES}
        for z in ZONE_IDS:
            parts = {k: disp * v for k, v in emis[z].items()}
            for s in ZONE_IDS:
                if s == z:
                    continue
                tw = _wind_transport(wd, ws, centroids[s], centroids[z])
                for k in ("construction", "road_dust", "traffic"):
                    parts[k] = parts[k] + tw * disp * emis[s][k]
            parts["background"] = bg
            pm10 = sum(parts.values())
            pm25 = (F25["background"] * parts["background"] + F25["dust"] * (parts["construction"] + parts["road_dust"])
                    + F25["traffic"] * parts["traffic"] + F25["other"] * parts["other"])
            out[z] = {**parts, "pm10": pm10, "pm25": pm25, "wetness": wet[z], "spraying": spray[z], "sweeping": sweep[z]}
        return out

    def _background(self, met: pd.DataFrame) -> np.ndarray:
        rng = np.random.default_rng(self.seed + 11)
        idx = self.index
        doy = idx.dayofyear.to_numpy().astype(float)
        hour = (idx.hour + idx.minute / 60.0).to_numpy()
        season = _interp_season(doy, [(160, 112), (178, 88), (195, 58), (243, 56), (263, 72), (278, 108), (290, 128)])
        synoptic = np.exp(_ar1(rng, self.n, 0.9988, 0.012))
        diurnal = 1.22 - 0.42 * _solar(hour)
        bg = season * diurnal * synoptic
        hero = idx.normalize() == HERO_DATE
        bg[hero] = self._hero_profile(idx[hero], [("00:00", 124), ("07:00", 118), ("12:00", 72), ("14:30", 66),
                                                   ("17:00", 76), ("20:00", 112), ("23:55", 122)])
        return bg

    # ---- public API -------------------------------------------------------------
    def generate(self) -> SyntheticDataset:
        logger.info(f"Generating synthetic township dataset: {self.n} steps x {len(ZONE_IDS)} zones")
        met = self._meteorology()
        act = self._activity()
        self._apply_hero(met, act)
        background = self._background(met)

        # Pass 1: no interventions -> decide historical dispatches on proxy levels
        proxy = self._concentrations(met, act, [], self.rng, background)
        proxy_frame = pd.DataFrame({z: proxy[z]["pm10"] for z in ZONE_IDS}, index=self.index)
        interventions = self._plan_interventions(proxy_frame, met)

        hero_record = {
            "zone_id": "B", "action": "targeted_sprinkling",
            "dispatched_at": pd.Timestamp("2026-10-07 14:33"), "started_at": pd.Timestamp("2026-10-07 14:42"),
            "ended_at": pd.Timestamp("2026-10-07 15:45"), "asset_type": "tanker", "source": "hero",
        }
        # Pass 2: with interventions (counterfactual = without the hero tanker)
        cf = self._concentrations(met, act, interventions, self.rng, background)
        actual = self._concentrations(met, act, interventions + [hero_record], self.rng, background)

        # Sensor-level measurement noise on zone averages + optical humidity bias
        rng = np.random.default_rng(self.seed + 5)
        rh = met["relative_humidity"].to_numpy()
        growth = np.where(rh > 60, 1 + TRUE_KAPPA * (np.minimum(rh, 98) / (100 - np.minimum(rh, 98))), 1.0)
        noise_cache = {z: np.exp(rng.normal(0, 0.035, self.n)) for z in ZONE_IDS}
        noise25 = {z: np.exp(rng.normal(0, 0.04, self.n)) for z in ZONE_IDS}
        ref_noise = np.exp(rng.normal(0, 0.05, self.n))
        washout = np.exp(-pd.Series(met["rain_mm"].to_numpy()).rolling(72, min_periods=1).sum().to_numpy() / 9.0)
        reference_bg = background * washout * ref_noise

        rows = []
        cf_rows = []
        for z in ZONE_IDS:
            for src, store in ((actual, rows), (cf, cf_rows)):
                d = src[z]
                frame = pd.DataFrame({
                    "timestamp": self.index, "zone_id": z,
                    "pm10_true": d["pm10"], "pm25_true": d["pm25"],
                    "pm10_raw": d["pm10"] * noise_cache[z] * growth,
                    "pm25_raw": d["pm25"] * noise25[z] * growth,
                    "contrib_construction": d["construction"], "contrib_road_dust": d["road_dust"],
                    "contrib_traffic": d["traffic"], "contrib_other": d["other"], "contrib_background": d["background"],
                    "wetness": d["wetness"], "spraying": d["spraying"], "sweeping": d["sweeping"],
                })
                store.append(frame)
        zone_frame = pd.concat(rows, ignore_index=True)
        cf_frame = pd.concat(cf_rows, ignore_index=True)

        # Random telemetry gaps (0.4% of zone readings) exercise the cleaner's imputation
        gap_mask = rng.random(len(zone_frame)) < 0.004
        hero_mask = zone_frame["timestamp"].dt.normalize() == HERO_DATE
        gap_mask &= ~hero_mask.to_numpy()
        zone_frame.loc[gap_mask, ["pm10_raw", "pm25_raw"]] = np.nan

        weather = pd.concat([met, act], axis=1)
        weather["ventilation"] = weather["wind_speed"] * weather["mixing_height"]
        weather["reference_pm10"] = reference_bg
        weather.index.name = "timestamp"
        weather = weather.reset_index()

        inter = pd.DataFrame(interventions + [hero_record])
        inter = inter.sort_values("dispatched_at").reset_index(drop=True)

        sensor_frame = self._sensor_readings(zone_frame, met)

        meta = {
            "synthetic": True,
            "generator": "TownshipSimulator v3 (physics-motivated)",
            "seed": self.seed,
            "start": str(self.index[0]), "end": str(self.index[-1]), "now": str(NOW),
            "step_minutes": STEP_MIN, "zones": ZONE_IDS, "rows_per_zone": self.n,
            "historical_interventions": int((inter["source"] == "historical").sum()),
        }
        logger.info(f"Generated {len(zone_frame):,} zone rows, {meta['historical_interventions']} historical interventions")
        return SyntheticDataset(zone_frame, weather, inter, cf_frame, sensor_frame, meta)

    def _sensor_readings(self, zone_frame: pd.DataFrame, met: pd.DataFrame) -> pd.DataFrame:
        """Per-sensor readings for the last 36 h (sensor cards, sparklines, IDW map)."""
        rng = np.random.default_rng(self.seed + 9)
        start = NOW - pd.Timedelta(hours=36)
        sub = zone_frame[(zone_frame["timestamp"] >= start)]
        frames = []
        met_sub = met.loc[met.index >= start]
        rh = met_sub["relative_humidity"].to_numpy()
        for s in site.SENSORS:
            zs = sub[sub["zone_id"] == s.zone_id].set_index("timestamp")
            n = len(zs)
            # micro-siting: sensors close to the haul road see more road dust
            factor = s.local_factor
            pm10 = zs["pm10_true"].to_numpy() * factor * np.exp(rng.normal(0, 0.06, n))
            pm25 = zs["pm25_true"].to_numpy() * (0.5 + 0.5 * factor) * np.exp(rng.normal(0, 0.07, n))
            temp = met_sub["temperature"].to_numpy() + rng.normal(0, 0.3, n)
            hum = rh + rng.normal(0, 1.2, n)
            if s.sensor_id == "EC-036":                                  # calibration drift
                pm10 = pm10 * 1.09
                pm25 = pm25 * 1.09
            status = np.array(["ONLINE"] * n, dtype=object)
            if s.status == "OFFLINE":
                cut = zs.index >= pd.Timestamp("2026-10-07 13:52")
                pm10 = np.where(cut, np.nan, pm10)
                pm25 = np.where(cut, np.nan, pm25)
                temp = np.where(cut, np.nan, temp)
                hum = np.where(cut, np.nan, hum)
                status = np.where(cut, "OFFLINE", "ONLINE")
            frames.append(pd.DataFrame({
                "timestamp": zs.index, "sensor_id": s.sensor_id, "zone_id": s.zone_id,
                "pm10": pm10, "pm25": pm25, "temperature": temp, "relative_humidity": hum, "status": status,
            }))
        return pd.concat(frames, ignore_index=True)


def generate_dataset(seed: int = 42) -> SyntheticDataset:
    return TownshipSimulator(seed=seed).generate()
