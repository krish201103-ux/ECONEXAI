"""
dustwise.py -- DUSTWISE decision & intervention optimisation engine.

DUSTWISE answers: *given a predicted pollution problem, what is the best
intervention and how should the available resources be deployed?*

1. Intervention Priority Score (IPS, 0-100) for every zone, computed from model
   outputs rather than typed-in values:

       IPS = 0.25 S_poll + 0.20 P_pred + 0.15 C_src + 0.15 E_exp + 0.10 P_pers + 0.15 F_feas

   S_poll  current severity           (zone PM10 vs the High action level)
   P_pred  predicted growth           (90-min forecast vs current, conformal upper band)
   C_src   source confidence          (attribution confidence x controllable share)
   E_exp   exposure                   (wind-weighted sensitive receptors + population)
   P_pers  atmospheric persistence    (forecast ventilation over the next 90 min)
   F_feas  intervention feasibility   (best resource ETA, availability, controllability)

2. Action comparison per hotspot - five candidate interventions with an
   *AI scenario estimate* of PM10 reduction (source shares x action efficacy x
   learned field multipliers), resource needs, cost level, response time and
   feasibility.

3. Fleet allocation - an assignment problem (Hungarian algorithm) that matches
   tankers / sweepers to hotspots maximising priority-weighted expected benefit
   minus travel time, subject to availability and water capacity, with routes
   on the site road network.

Every recommendation requires operator approval; nothing is dispatched
automatically.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np

import config
import site_model as site
from intelligence import factor_label, fmt0, ips_band

logger = logging.getLogger(__name__)

IPS_WEIGHTS = dict(config.IPS_WEIGHTS)          # edit the weights in config.py
_IPS_SYMBOLS = {"severity": "S_poll", "predicted_growth": "P_pred", "source_confidence": "C_src",
                "exposure": "E_exp", "persistence": "P_pers", "feasibility": "F_feas"}
IPS_FORMULA = "IPS = " + " + ".join(f"{IPS_WEIGHTS[k]:.2f}*{sym}" for k, sym in _IPS_SYMBOLS.items())
IPS_FACTOR_LABELS = {
    "severity": "Pollution severity", "predicted_growth": "Predicted growth", "source_confidence": "Source confidence",
    "exposure": "Exposure", "persistence": "Atmospheric persistence", "feasibility": "Intervention feasibility",
}
MOBILISATION_MIN = 5.0          # crew readiness / pre-departure checks

ACTIONS: Dict[str, Dict[str, Any]] = {
    "targeted_sprinkling": {
        "label": "Targeted sprinkling", "short": "Dust suppression", "resource": "tanker",
        "reduction": {"Construction": 0.45, "Road dust": 0.60, "Traffic": 0.0, "Other": 0.05},
        "setup_min": 3, "cost_level": "Low", "cost_rank": 0, "duration_min": 60,
        "disruption": "None - works continue",
    },
    "mechanical_sweeping": {
        "label": "Mechanical sweeping", "short": "Road sweeping", "resource": "sweeper",
        "reduction": {"Construction": 0.05, "Road dust": 0.40, "Traffic": 0.0, "Other": 0.0},
        "setup_min": 5, "cost_level": "Low", "cost_rank": 0, "duration_min": 120,
        "disruption": "Lane occupancy on haul road",
    },
    "construction_suppression": {
        "label": "Construction suppression", "short": "Pause high-dust tasks", "resource": "site_instruction",
        "reduction": {"Construction": 0.55, "Road dust": 0.15, "Traffic": 0.10, "Other": 0.0},
        "setup_min": 10, "cost_level": "High", "cost_rank": 3, "duration_min": 60,
        "disruption": "Rock-breaking & excavation paused 60 min",
    },
    "traffic_flow": {
        "label": "Traffic-flow recommendation", "short": "Haul speed limit & staggering", "resource": "site_instruction",
        "reduction": {"Construction": 0.0, "Road dust": 0.30, "Traffic": 0.25, "Other": 0.0},
        "setup_min": 15, "cost_level": "Low", "cost_rank": 0, "duration_min": 90,
        "disruption": "15 km/h haul limit, trucks staggered",
    },
    "combined": {
        "label": "Combined intervention", "short": "Sprinkling + traffic + partial pause", "resource": "tanker",
        "reduction": {"Construction": 0.60, "Road dust": 0.70, "Traffic": 0.25, "Other": 0.05},
        "setup_min": 10, "cost_level": "Medium-High", "cost_rank": 2, "duration_min": 60,
        "disruption": "Partial pause of excavation + haul limit",
    },
}
COST_PENALTY = {0: 0.0, 1: 0.08, 2: 0.18, 3: 0.32}


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def occupancy(r: site.Receptor, hour: float, workday: bool = True) -> float:
    start, end = r.hours
    if r.kind in ("school", "daycare") and not workday:
        return 0.15
    return 1.0 if start <= hour < end else 0.2


class DustwiseEngine:
    """Specialised intervention decision and resource optimisation engine."""

    def __init__(self, fleet: Optional[List[site.FleetAsset]] = None):
        self.fleet = [site.FleetAsset(**a.__dict__) for a in (fleet or site.FLEET)]

    # ------------------------------------------------------------------ #
    # Resources
    # ------------------------------------------------------------------ #
    def asset(self, asset_id: str) -> site.FleetAsset:
        return next(a for a in self.fleet if a.asset_id == asset_id)

    def eta_minutes(self, asset: site.FleetAsset, zone_id: str) -> Dict[str, Any]:
        z = site.ZONE_BY_ID[zone_id]
        route = site.shortest_route(asset.position, z.access_point)
        travel = route["length_m"] / max(asset.speed_m_per_min, 1.0)
        return {"route": route, "travel_min": round(travel, 1),
                "eta_min": round(MOBILISATION_MIN + travel, 1), "distance_m": route["length_m"]}

    def best_resource(self, zone_id: str, resource_type: str) -> Optional[Dict[str, Any]]:
        cands = [a for a in self.fleet if a.asset_type == resource_type and a.status in ("AVAILABLE", "STANDBY")]
        if not cands:
            return None
        scored = []
        for a in cands:
            e = self.eta_minutes(a, zone_id)
            penalty = 4.0 if a.status == "STANDBY" else 0.0
            scored.append((e["eta_min"] + penalty, a, e))
        scored.sort(key=lambda x: x[0])
        _, a, e = scored[0]
        return {"asset": a, **e}

    # ------------------------------------------------------------------ #
    # IPS
    # ------------------------------------------------------------------ #
    def exposure_score(self, zone_id: str, wind_from: float, wind_speed: float, hour: float,
                       workday: bool = True) -> Dict[str, Any]:
        z = site.ZONE_BY_ID[zone_id]
        receptor_terms = []
        for r in site.RECEPTORS:
            tw = 1.0 if r.zone_id == zone_id else site.transport_weight(z.centroid, r.position, wind_from, wind_speed)
            occ = occupancy(r, hour, workday)
            receptor_terms.append({"receptor": r.name, "kind": r.kind, "transport": round(tw, 3),
                                   "occupied": occ >= 1.0, "weighted": round(r.weight * occ * tw, 3)})
        rec = sum(t["weighted"] for t in receptor_terms)
        pop = z.population + sum(
            o.population * site.transport_weight(z.centroid, o.centroid, wind_from, wind_speed)
            for o in site.ZONES if o.zone_id != zone_id)
        score = 100 * _clip01(0.65 * _clip01(rec / 0.35) + 0.35 * _clip01(pop / 2500))
        top = sorted(receptor_terms, key=lambda t: -t["weighted"])
        return {"score": round(score, 1), "receptors": top, "exposed_population": int(pop)}

    def priority(self, zone_id: str, current_pm10: float, forecast: Dict[str, Dict[str, Any]],
                 attribution: Dict[str, Any], weather_now: Dict[str, float], ventilation_forecast: float,
                 hour: float, workday: bool = True, change_30m_pct: float = 0.0) -> Dict[str, Any]:
        window = [k for k in ("45m", "60m", "75m", "90m") if k in forecast]
        f90 = max([forecast[k]["pm10"]["median"] for k in window] or [current_pm10])
        hi90 = max([forecast[k]["pm10"]["hi90"] for k in window] or [current_pm10])
        # Severity: zone PM10 above a clean-air floor (40) relative to the High action level (200)
        severity = 100 * _clip01((current_pm10 - 40.0) / 160.0)
        # Predicted growth: +20% (median) / +30% (upper band) within 90 min counts as high growth
        growth_med = (f90 - current_pm10) / max(current_pm10, 1.0)
        growth_hi = (hi90 - current_pm10) / max(current_pm10, 1.0)
        forecast_growth = _clip01(0.6 * _clip01(growth_med / 0.20) + 0.4 * _clip01(growth_hi / 0.30))
        # Observed momentum: a +30% rise in 30 minutes is treated as maximal momentum
        momentum = _clip01(change_30m_pct / 30.0)
        growth = 100 * (0.5 * forecast_growth + 0.5 * momentum)
        shares = {c["source"]: c["share_pct"] / 100 for c in attribution["contributors"]}
        zone = site.ZONE_BY_ID[zone_id]
        # Locally controllable share: own construction only counts where the zone has works
        local_construction = shares.get("Construction", 0) if zone.construction_base > 0 else 0.0
        controllable = local_construction + shares.get("Road dust", 0) + shares.get("Traffic", 0)
        local_fraction = attribution["local_excess_pm10"] / max(attribution["observed_pm10"], 1.0)
        source_conf = 100 * _clip01(attribution["confidence"] * (0.5 + 0.5 * controllable))
        exp = self.exposure_score(zone_id, weather_now["wind_dir_from"], weather_now["wind_speed"], hour, workday)
        persistence = 100 * _clip01(1 - ventilation_forecast / 2500.0)
        best = self.best_resource(zone_id, "tanker")
        if best is None:
            feas = 0.25
            eta = None
        else:
            eta = best["eta_min"]
            feas = _clip01(1 - (eta - 6) / 40) * (0.85 if best["asset"].status == "STANDBY" else 1.0)
        feasibility = 100 * _clip01(feas * (0.3 + 0.7 * controllable) * (0.6 + 0.4 * _clip01(local_fraction / 0.4)))
        comps = {"severity": severity, "predicted_growth": growth, "source_confidence": source_conf,
                 "exposure": exp["score"], "persistence": persistence, "feasibility": feasibility}
        ips = sum(IPS_WEIGHTS[k] * v for k, v in comps.items())
        band = ips_band(ips)
        return {
            "zone_id": zone_id,
            "ips_score": round(ips, 1),
            "band": band,
            "components": {k: round(v, 1) for k, v in comps.items()},
            "factors": [{"key": k, "label": IPS_FACTOR_LABELS[k], "value": round(v, 1), "weight": IPS_WEIGHTS[k],
                         "level": factor_label(v), "contribution": round(IPS_WEIGHTS[k] * v, 1)} for k, v in comps.items()],
            "inputs": {"current_pm10": round(current_pm10, 1), "forecast_peak_90m": round(f90, 1),
                       "forecast_upper_90m": round(hi90, 1), "locally_controllable_share": round(controllable, 3),
                       "local_fraction": round(local_fraction, 3), "ventilation_forecast": round(ventilation_forecast),
                       "best_tanker_eta_min": eta, "change_30m_pct": round(change_30m_pct, 1)},
            "exposure": exp,
            "formula": IPS_FORMULA,
        }

    # ------------------------------------------------------------------ #
    # Action comparison
    # ------------------------------------------------------------------ #
    def compare_actions(self, zone_id: str, attribution: Dict[str, Any],
                        efficacy: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Any]:
        efficacy = efficacy or {}
        shares = {c["source"]: c["share_pct"] / 100 for c in attribution["contributors"]}
        ranges = {c["source"]: (c["range_pct"][0] / 100, c["range_pct"][1] / 100) for c in attribution["contributors"]}
        local_fraction = attribution["local_excess_pm10"] / max(attribution["observed_pm10"], 1.0)
        options = []
        for key, a in ACTIONS.items():
            base = sum(shares.get(s, 0) * r for s, r in a["reduction"].items())
            lo_share = sum(ranges.get(s, (0, 0))[0] * r for s, r in a["reduction"].items())
            hi_share = sum(ranges.get(s, (0, 0))[1] * r for s, r in a["reduction"].items())
            eff = efficacy.get(key, {"multiplier": 1.0, "multiplier_lo": 0.7, "multiplier_hi": 1.3, "n": 0})
            mult, mlo, mhi = eff["multiplier"], eff["multiplier_lo"], eff["multiplier_hi"]
            benefit = 100 * local_fraction * base * mult
            b_lo = 100 * local_fraction * lo_share * mlo
            b_hi = 100 * local_fraction * hi_share * mhi
            if a["resource"] in ("tanker", "sweeper"):
                best = self.best_resource(zone_id, a["resource"])
            else:
                best = None
            if a["resource"] in ("tanker", "sweeper") and best is None:
                feasibility, feas_label = 0.15, "Low - no %s available" % a["resource"]
                response = None
                resource_txt = "No %s available" % a["resource"]
            elif best is not None:
                response = round(best["eta_min"] + a["setup_min"], 0)
                feasibility = _clip01(1 - (response - 8) / 45)
                feas_label = "High" if feasibility >= 0.7 else ("Medium" if feasibility >= 0.45 else "Low")
                extra = " + site instruction" if key == "combined" else ""
                resource_txt = f"{best['asset'].name}{extra}"
            else:
                response = a["setup_min"]
                feasibility = 0.6 if a["cost_rank"] >= 2 else 0.85
                feas_label = "Medium - needs site manager sign-off" if a["cost_rank"] >= 2 else "High"
                resource_txt = "Site instruction (no vehicle)"
            water = None
            if a["resource"] == "tanker" and best is not None:
                water = int(best["asset"].rate * a["duration_min"])
            score = benefit / 100 * feasibility - COST_PENALTY[a["cost_rank"]] * 0.25
            options.append({
                "action": key, "label": a["label"], "short": a["short"],
                "scenario_benefit_pct": round(benefit, 1),
                "scenario_benefit_range_pct": [round(max(b_lo, 0), 1), round(b_hi, 1)],
                "expected_pm10_after": round(attribution["observed_pm10"] * (1 - benefit / 100), 1),
                "resource": resource_txt,
                "resource_asset_id": best["asset"].asset_id if best is not None else None,
                "water_l": water,
                "cost_level": a["cost_level"], "response_min": response,
                "feasibility": round(feasibility, 2), "feasibility_label": feas_label,
                "disruption": a["disruption"],
                "field_evidence_n": eff.get("n", 0),
                "score": round(score, 4),
            })
        options.sort(key=lambda o: -o["score"])
        recommended = options[0]
        return {"zone_id": zone_id, "options": options, "recommended": recommended["action"],
                "label": "AI scenario estimate - not a guaranteed outcome"}

    # ------------------------------------------------------------------ #
    # Fleet allocation
    # ------------------------------------------------------------------ #
    def optimize_allocation(self, priorities: Dict[str, Dict[str, Any]], comparisons: Dict[str, Dict[str, Any]],
                            min_ips: float = 40.0) -> Dict[str, Any]:
        from scipy.optimize import linear_sum_assignment

        hotspots = [z for z, p in sorted(priorities.items(), key=lambda kv: -kv[1]["ips_score"]) if p["ips_score"] >= min_ips]
        movable = [a for a in self.fleet if a.status in ("AVAILABLE", "STANDBY")]
        rows: List[Dict[str, Any]] = []
        value = np.zeros((len(movable), max(len(hotspots), 1)))
        info: Dict[tuple, Dict[str, Any]] = {}
        for i, a in enumerate(movable):
            action = "targeted_sprinkling" if a.asset_type == "tanker" else "mechanical_sweeping"
            for j, z in enumerate(hotspots):
                opt = next(o for o in comparisons[z]["options"] if o["action"] == action)
                e = self.eta_minutes(a, z)
                water_needed = a.rate * ACTIONS[action]["duration_min"] if a.asset_type == "tanker" else 0
                water_ok = (a.capacity_l * a.level_pct / 100) >= water_needed * 0.75 if a.asset_type == "tanker" else True
                v = priorities[z]["ips_score"] / 100 * opt["scenario_benefit_pct"] / 100 * _clip01(1 - e["eta_min"] / 60)
                if a.status == "STANDBY":
                    v *= 0.6                         # keep the reserve unless it clearly pays off
                if any(b.status == "ACTIVE" and b.assignment and site.zone_of(b.position) == z for b in self.fleet):
                    v *= 0.5                         # zone already has an active intervention: smaller marginal benefit
                if not water_ok:
                    v = 0.0
                value[i, j] = v
                info[(i, j)] = {"eta": e, "action": action, "water_needed": water_needed, "water_ok": water_ok}
        assigned: Dict[str, Dict[str, Any]] = {}
        if hotspots and movable:
            r_idx, c_idx = linear_sum_assignment(-value)
            for i, j in zip(r_idx, c_idx):
                if value[i, j] >= 0.01:
                    assigned[movable[i].asset_id] = {"zone": hotspots[j], **info[(i, j)], "value": float(value[i, j])}

        def priority_label(ips: float) -> str:
            return "Critical" if ips >= 80 else ("High" if ips >= 60 else ("Medium" if ips >= 40 else "Low"))

        for a in self.fleet:
            if a.asset_id in assigned:
                asg = assigned[a.asset_id]
                z = asg["zone"]
                ips = priorities[z]["ips_score"]
                rows.append({
                    "asset_id": a.asset_id, "resource": a.name, "type": a.asset_type, "destination": f"Zone {z}",
                    "zone_id": z, "action": ACTIONS[asg["action"]]["short"], "action_key": asg["action"],
                    "priority": priority_label(ips), "status": "Pending approval",
                    "eta_min": asg["eta"]["eta_min"], "distance_m": asg["eta"]["distance_m"],
                    "route": asg["eta"]["route"]["points"],
                    "water_l": int(asg["water_needed"]) if a.asset_type == "tanker" else None,
                    "reason": f"Highest priority-weighted benefit for Zone {z} (IPS {fmt0(ips)}), ETA {fmt0(asg['eta']['eta_min'])} min",
                })
            elif a.status == "ACTIVE":
                z = (a.assignment and "A") or None
                rows.append({
                    "asset_id": a.asset_id, "resource": a.name, "type": a.asset_type,
                    "destination": "Zone A", "zone_id": "A", "action": "Road sweeping", "action_key": "mechanical_sweeping",
                    "priority": priority_label(priorities.get("A", {"ips_score": 0})["ips_score"]),
                    "status": "Active", "eta_min": 0, "distance_m": 0, "route": [a.position], "water_l": None,
                    "reason": f"Continuing assignment {a.assignment} (Tower Lane sweeping)",
                })
            elif a.status == "MAINTENANCE":
                rows.append({
                    "asset_id": a.asset_id, "resource": a.name, "type": a.asset_type, "destination": "Depot",
                    "zone_id": None, "action": "None", "action_key": None, "priority": "-", "status": "Maintenance",
                    "eta_min": None, "distance_m": None, "route": [a.position], "water_l": None, "reason": a.note,
                })
            else:
                home = site.zone_of(a.position) or "C"
                rows.append({
                    "asset_id": a.asset_id, "resource": a.name, "type": a.asset_type, "destination": f"Zone {home}",
                    "zone_id": home, "action": "None", "action_key": None, "priority": "Low", "status": "Standby",
                    "eta_min": None, "distance_m": None, "route": [a.position], "water_l": None,
                    "reason": "Held in reserve - marginal benefit below dispatch threshold",
                })
        order = {"Pending approval": 0, "Active": 1, "Standby": 2, "Maintenance": 3}
        rows.sort(key=lambda r: (order.get(r["status"], 9), r["resource"]))
        return {"hotspots": hotspots, "allocations": rows, "method": "Hungarian assignment on priority-weighted benefit",
                "requires_operator_approval": True}
