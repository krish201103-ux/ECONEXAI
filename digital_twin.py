"""
digital_twin.py -- Digital Environmental Twin (what-if scenario simulator).

A fast, steady-state dispersion model of the township on a 20 m grid:

* sources: construction work areas (Zones A and B, weighted towards the
  excavation pit), the unpaved haul road (line source) and commuter roads;
* transport: a Gaussian-plume-style kernel - crosswind spread that grows with
  downwind distance, a small near-field term for calm conditions, dilution by
  wind speed;
* interventions: targeted (Zone B wetting) and combined (wetting + haul
  limits + partial pause) reduce the relevant source strengths with the same
  efficacy assumptions DUSTWISE uses.

It is a *what-if* tool. Every output is labelled "Simulation / scenario
estimate - not field measurement". The browser runs an identical port of this
model (frontend/js/twin.js) so sliders respond instantly; ``tests`` check the
two implementations agree.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np

import site_model as site

CELL = 20.0
NX = int(site.SITE_WIDTH_M // CELL)
NY = int(site.SITE_HEIGHT_M // CELL)
BACKGROUND = 66.0

# Source strengths (model units). Calibrated so that the default scenario
# reproduces the observed 14:30 zone levels (A/B within ~5%, C/D within ~25%)
# while keeping Zone B's construction : haul-road : traffic mix near 61:28:11.
Q_CONSTRUCTION = {"A": 41.8, "B": 57.4}
Q_HAUL_PER_TRUCK = 2.55
Q_COMMUTER_PER_VEH = 0.635
KERNEL_SCALE = 260.0
PLUME_DECAY_M = 200.0

INTERVENTIONS = {
    "none": {"label": "Current condition", "con": {"A": 1.0, "B": 1.0}, "haul": 1.0, "commuter": 1.0},
    "targeted": {"label": "Targeted intervention", "con": {"A": 1.0, "B": 0.55}, "haul": 0.40, "commuter": 1.0},
    "combined": {"label": "Combined intervention", "con": {"A": 0.6, "B": 0.40}, "haul": 0.30, "commuter": 0.75},
}

DEFAULTS = {"construction_pct": 100.0, "traffic_pct": 100.0, "wind_speed": 0.85, "wind_from_deg": 306.0,
            "intervention": "none", "base_construction": {"A": 0.72, "B": 0.86}, "base_trucks": 9.0,
            "base_vehicles": 40.0, "background": BACKGROUND}


def _grid() -> Tuple[np.ndarray, np.ndarray]:
    xs = (np.arange(NX) + 0.5) * CELL
    ys = (np.arange(NY) + 0.5) * CELL
    return np.meshgrid(xs, ys)


def _sample_polygon(poly: List[site.Point], step: float) -> List[site.Point]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    pts = []
    x = min(xs) + step / 2
    while x < max(xs):
        y = min(ys) + step / 2
        while y < max(ys):
            if site.point_in_polygon((x, y), poly):
                pts.append((x, y))
            y += step
        x += step
    return pts


def _sample_line(a: site.Point, b: site.Point, step: float) -> List[site.Point]:
    d = site.distance(a, b)
    n = max(1, int(d // step))
    return [(a[0] + (b[0] - a[0]) * (i + 0.5) / n, a[1] + (b[1] - a[1]) * (i + 0.5) / n) for i in range(n)]


def source_points() -> Dict[str, List[site.Point]]:
    """Discretised source geometry (shared with the JS port via the API/snapshot)."""
    pit = next(w for w in site.FEATURES["works"] if w["kind"] == "pit")["polygon"]
    zone_b = site.ZONE_BY_ID["B"].polygon
    pts_b = _sample_polygon(zone_b, 45.0) + _sample_polygon(pit, 25.0) * 2
    pts_a = _sample_polygon(site.ZONE_BY_ID["A"].polygon, 45.0)
    haul, commuter = [], []
    for a, b, kind, name in site.ROAD_EDGES:
        pa, pb = site.ROAD_NODES[a], site.ROAD_NODES[b]
        if kind == "unpaved" and name == "Haul Road":
            haul += _sample_line(pa, pb, 25.0)
        elif kind == "paved":
            commuter += _sample_line(pa, pb, 40.0)
    return {"construction_A": pts_a, "construction_B": pts_b, "haul": haul, "commuter": commuter}


_SOURCES = source_points()


def _kernel(gx: np.ndarray, gy: np.ndarray, sx: float, sy: float, wind_to_rad: Tuple[float, float],
            u: float) -> np.ndarray:
    ux, uy = wind_to_rad
    dx, dy = gx - sx, gy - sy
    along = dx * ux + dy * uy
    cross = -dx * uy + dy * ux
    dist = np.hypot(dx, dy)
    sigma = 0.28 * np.maximum(along, 0) + 14.0
    plume = np.where(along > 0, np.exp(-cross ** 2 / (2 * sigma ** 2)) / sigma * np.exp(-along / PLUME_DECAY_M), 0.0)
    near = np.exp(-dist / 28.0) * 0.045
    calm = 1.0 / (1.0 + 2.0 * u)                      # calm air spreads omnidirectionally
    return (plume * (1 - 0.5 * calm) + near * (0.5 + calm)) / max(u, 0.3)


def simulate(construction_pct: float = 100.0, traffic_pct: float = 100.0, wind_speed: float = 0.85,
             wind_from_deg: float = 306.0, intervention: str = "none", base_construction: Dict[str, float] = None,
             base_trucks: float = 9.0, base_vehicles: float = 40.0, background: float = BACKGROUND,
             include_grid: bool = True) -> Dict[str, Any]:
    base_construction = base_construction or DEFAULTS["base_construction"]
    iv = INTERVENTIONS.get(intervention, INTERVENTIONS["none"])
    gx, gy = _grid()
    to = math.radians((wind_from_deg + 180) % 360)
    wind_to = (math.sin(to), math.cos(to))
    u = max(0.3, wind_speed)
    field = np.zeros_like(gx)
    c_scale = construction_pct / 100.0
    t_scale = traffic_pct / 100.0
    strengths = {
        "construction_A": Q_CONSTRUCTION["A"] * base_construction["A"] * c_scale * iv["con"]["A"],
        "construction_B": Q_CONSTRUCTION["B"] * base_construction["B"] * c_scale * iv["con"]["B"],
        "haul": Q_HAUL_PER_TRUCK * base_trucks * t_scale * iv["haul"],
        "commuter": Q_COMMUTER_PER_VEH * base_vehicles * t_scale * iv["commuter"],
    }
    for key, pts in _SOURCES.items():
        q = strengths[key] / max(len(pts), 1) * KERNEL_SCALE
        for sx, sy in pts:
            field += q * _kernel(gx, gy, sx, sy, wind_to, u)
    conc = background + field
    zones = {}
    for z in site.ZONES:
        mask = np.array([[site.point_in_polygon((x, y), z.polygon) for x, y in zip(rx, ry)] for rx, ry in zip(gx, gy)])
        zones[z.zone_id] = round(float(conc[mask].mean()), 1)
    receptors = {}
    for r in site.RECEPTORS:
        i = min(NY - 1, int(r.position[1] // CELL))
        j = min(NX - 1, int(r.position[0] // CELL))
        receptors[r.receptor_id] = {"name": r.name, "pm10": round(float(conc[i, j]), 1)}
    out = {
        "inputs": {"construction_pct": construction_pct, "traffic_pct": traffic_pct, "wind_speed": wind_speed,
                   "wind_from_deg": wind_from_deg, "intervention": intervention},
        "intervention_label": iv["label"], "zones": zones, "receptors": receptors,
        "site_max": round(float(conc.max()), 1), "background": background,
        "label": "Simulation / scenario estimate - not field measurement",
    }
    if include_grid:
        out["grid"] = {"cell_m": CELL, "nx": NX, "ny": NY, "values": np.round(conc, 1).tolist()}
    return out


def compare_scenarios(**kwargs: Any) -> Dict[str, Any]:
    kwargs = {k: v for k, v in kwargs.items() if k != "intervention"}
    runs = {k: simulate(intervention=k, include_grid=False, **kwargs) for k in INTERVENTIONS}
    base = runs["none"]
    rows = []
    for k, r in runs.items():
        rows.append({
            "scenario": k, "label": r["intervention_label"], "zones": r["zones"], "receptors": r["receptors"],
            "zone_b_change_pct": round(100 * (r["zones"]["B"] - base["zones"]["B"]) / base["zones"]["B"], 1),
            "school_change_pct": round(100 * (r["receptors"]["R1"]["pm10"] - base["receptors"]["R1"]["pm10"])
                                       / base["receptors"]["R1"]["pm10"], 1),
        })
    return {"scenarios": rows, "label": "Simulation / scenario estimate - not field measurement"}


def export_geometry() -> Dict[str, Any]:
    """Constants and source geometry for the browser port."""
    return {
        "cell": CELL, "nx": NX, "ny": NY, "background": BACKGROUND, "kernel_scale": KERNEL_SCALE,
        "plume_decay_m": PLUME_DECAY_M,
        "q_construction": Q_CONSTRUCTION, "q_haul_per_truck": Q_HAUL_PER_TRUCK,
        "q_commuter_per_veh": Q_COMMUTER_PER_VEH, "interventions": INTERVENTIONS, "defaults": DEFAULTS,
        "sources": {k: [list(p) for p in v] for k, v in _SOURCES.items()},
    }
