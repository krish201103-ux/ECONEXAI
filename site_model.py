"""
site_model.py -- Site model (single source of truth for the demo site).

Green Valley Township is a 100-acre mixed-use development used as the reference
site for the platform. Every module (data generator, forecaster, DUSTWISE,
digital twin, API and the web UI) reads its geometry from here so that zone
names, sensor IDs, fleet assets and map coordinates are always consistent.

Coordinates are in metres in a local site frame: x points east, y points north,
the origin is the south-west corner of the site. ``to_latlng`` converts them to
WGS84 for export using an equirectangular approximation around ``ORIGIN_LATLNG``.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

Point = Tuple[float, float]

SITE_NAME = "Green Valley Township"
SITE_AREA_ACRES = 100
SITE_WIDTH_M = 720.0
SITE_HEIGHT_M = 560.0
ORIGIN_LATLNG = (28.4528, 77.0182)  # south-west corner (illustrative location)


def to_latlng(x: float, y: float) -> Tuple[float, float]:
    """Converts site metres to approximate WGS84 latitude/longitude."""
    lat0, lng0 = ORIGIN_LATLNG
    lat = lat0 + y / 111_320.0
    lng = lng0 + x / (111_320.0 * math.cos(math.radians(lat0)))
    return round(lat, 6), round(lng, 6)


def polygon_area(poly: List[Point]) -> float:
    area = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def polygon_centroid(poly: List[Point]) -> Point:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def bearing_deg(a: Point, b: Point) -> float:
    """Compass bearing (0 = north, 90 = east) of the vector a -> b."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    return (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0


def distance(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


# --------------------------------------------------------------------------- #
# Static site description
# --------------------------------------------------------------------------- #

BOUNDARY: List[Point] = [(0, 0), (690, 0), (720, 30), (720, 560), (40, 560), (0, 520)]


@dataclass
class Zone:
    zone_id: str
    name: str
    short_name: str
    land_use: str
    polygon: List[Point]
    access_point: Point
    population: int                 # people normally present in the zone
    construction_base: float        # typical construction intensity on a work day (0-1)
    has_unpaved_haul_road: bool
    controllable: bool              # whether site operations can intervene here
    road_exposure: float = 0.2      # share of the zone influenced by through-traffic roads (0-1)

    @property
    def centroid(self) -> Point:
        return polygon_centroid(self.polygon)

    @property
    def area_m2(self) -> float:
        return polygon_area(self.polygon)


ZONES: List[Zone] = [
    Zone("A", "Zone A", "Tower Construction (Phase 2)", "construction",
         [(385, 320), (700, 320), (700, 540), (385, 540)], (540, 420), 380, 0.55, False, True, 0.12),
    Zone("B", "Zone B", "Excavation & Haul Road (Phase 3)", "construction",
         [(40, 320), (345, 320), (345, 530), (60, 545), (40, 500)], (230, 445), 260, 0.75, True, True, 0.12),
    Zone("C", "Zone C", "Residential Block & Central Park", "residential",
         [(20, 30), (345, 30), (345, 270), (20, 270)], (250, 150), 2100, 0.0, False, True, 0.22),
    Zone("D", "Zone D", "School & Clinic Perimeter", "sensitive",
         [(385, 20), (685, 20), (700, 40), (700, 270), (385, 270)], (480, 150), 1650, 0.0, False, True, 0.40),
]
ZONE_BY_ID: Dict[str, Zone] = {z.zone_id: z for z in ZONES}


@dataclass
class Receptor:
    receptor_id: str
    name: str
    kind: str
    position: Point
    zone_id: str
    weight: float                    # vulnerability weight (0-1)
    occupants: int
    hours: Tuple[float, float]       # typical occupied hours (local time)


RECEPTORS: List[Receptor] = [
    Receptor("R1", "Green Valley Public School", "school", (480, 165), "D", 1.0, 1150, (7.5, 17.0)),
    Receptor("R2", "Community Health Clinic", "clinic", (630, 85), "D", 0.9, 140, (8.0, 20.0)),
    Receptor("R3", "Little Sprouts Daycare", "daycare", (560, 232), "D", 0.95, 85, (8.0, 18.0)),
    Receptor("R4", "Senior Living Residences", "residential", (110, 235), "C", 0.8, 320, (0.0, 24.0)),
]


@dataclass
class Sensor:
    sensor_id: str
    position: Point
    zone_id: str
    label: str
    status: str = "ONLINE"           # ONLINE | WARNING | OFFLINE
    status_reason: str = ""
    battery_pct: float = 90.0
    signal_dbm: float = -67.0
    local_factor: float = 1.0         # micro-siting factor relative to the zone average


SENSORS: List[Sensor] = [
    # Zone A - tower construction
    Sensor("EC-021", (420, 350), "A", "Tower 1 hoarding", local_factor=0.96),
    Sensor("EC-022", (520, 360), "A", "Tower 2 batching", local_factor=1.05),
    Sensor("EC-023", (640, 350), "A", "Tower 3 east", local_factor=0.94),
    Sensor("EC-024", (450, 500), "A", "North site office", local_factor=0.92),
    Sensor("EC-025", (580, 480), "A", "Material yard", local_factor=1.08),
    Sensor("EC-026", (680, 520), "A", "NE perimeter", local_factor=0.88),
    # Zone C - residential & park
    Sensor("EC-027", (60, 60), "C", "SW perimeter", local_factor=0.95),
    Sensor("EC-028", (150, 130), "C", "Central Park", local_factor=0.9),
    Sensor("EC-029", (280, 90), "C", "Block C3", status="WARNING", status_reason="Low battery (14%)",
           battery_pct=14, signal_dbm=-79, local_factor=1.0),
    Sensor("EC-030", (100, 240), "C", "Senior Living", local_factor=1.04),
    Sensor("EC-031", (300, 240), "C", "Block C1", local_factor=1.06),
    # Zone D - school & clinic
    Sensor("EC-032", (420, 60), "D", "Block D4", local_factor=0.93),
    Sensor("EC-033", (470, 190), "D", "School playground", local_factor=1.04),
    Sensor("EC-034", (560, 230), "D", "Daycare gate", status="OFFLINE", status_reason="No data since 13:52 (power fault)",
           battery_pct=0, signal_dbm=-120, local_factor=1.06),
    Sensor("EC-035", (640, 90), "D", "Clinic entrance", local_factor=0.95),
    Sensor("EC-036", (680, 200), "D", "East perimeter", status="WARNING", status_reason="Calibration drift +9% vs co-located reference",
           battery_pct=71, local_factor=0.99),
    Sensor("EC-037", (520, 40), "D", "South housing", local_factor=0.9),
    # Perimeter / roads
    Sensor("EC-038", (365, 12), "D", "Main gate", local_factor=1.02),
    Sensor("EC-039", (12, 300), "C", "West perimeter (upwind)", local_factor=0.86),
    # Zone B - excavation & haul road (hero zone)
    Sensor("EC-040", (90, 350), "B", "Batching plant", local_factor=1.0),
    Sensor("EC-041", (160, 410), "B", "Haul road west", local_factor=1.03),
    Sensor("EC-042", (232, 440), "B", "Haul road / pit edge", local_factor=1.12),
    Sensor("EC-043", (300, 400), "B", "Excavation pit south", local_factor=1.06),
    Sensor("EC-044", (120, 505), "B", "Stockpile north", local_factor=0.94),
    Sensor("EC-045", (320, 505), "B", "Pit north-east", local_factor=0.97),
    Sensor("EC-046", (55, 430), "B", "Gate 2 (haul entry)", local_factor=0.9),
    Sensor("EC-047", (712, 300), "A", "East perimeter / depot", local_factor=0.84),
]
SENSOR_BY_ID: Dict[str, Sensor] = {s.sensor_id: s for s in SENSORS}


@dataclass
class FleetAsset:
    asset_id: str
    name: str
    asset_type: str                  # tanker | sweeper
    description: str
    position: Point
    status: str                      # AVAILABLE | ACTIVE | STANDBY | MAINTENANCE
    capacity_l: float = 0.0
    level_pct: float = 0.0           # water level (tankers) or hopper fill (sweepers)
    rate: float = 0.0                # discharge L/min (tankers) or sweep width m (sweepers)
    speed_m_per_min: float = 300.0
    assignment: Optional[str] = None
    note: str = ""


FLEET: List[FleetAsset] = [
    FleetAsset("TANKER-01", "Tanker 01", "tanker", "Water tanker with rear spray bar", (700, 290), "AVAILABLE",
               capacity_l=12000, level_pct=95, rate=250, speed_m_per_min=280),
    FleetAsset("TANKER-02", "Tanker 02", "tanker", "Atomised mist tanker (fine droplet)", (320, 290), "AVAILABLE",
               capacity_l=8000, level_pct=82, rate=120, speed_m_per_min=300),
    FleetAsset("TANKER-03", "Tanker 03", "tanker", "Water tanker with spray bar", (250, 150), "STANDBY",
               capacity_l=10000, level_pct=64, rate=220, speed_m_per_min=280, note="Reserve unit"),
    FleetAsset("SWEEPER-01", "Sweeper 01", "sweeper", "Regenerative-air road sweeper", (540, 420), "ACTIVE",
               capacity_l=0, level_pct=45, rate=2.4, speed_m_per_min=170, assignment="EV-041"),
    FleetAsset("SWEEPER-02", "Sweeper 02", "sweeper", "Vacuum road sweeper", (700, 290), "MAINTENANCE",
               capacity_l=0, level_pct=10, rate=2.2, speed_m_per_min=170, note="Brush replacement, back 16:00"),
]
FLEET_BY_ID: Dict[str, FleetAsset] = {a.asset_id: a for a in FLEET}

DEPOT = {"name": "Operations Depot & Water Point", "position": (700, 290)}
GATES = [
    {"name": "Main Gate", "position": (365, 0)},
    {"name": "Gate 2 (Haul Entry)", "position": (0, 420)},
]

# Road network ---------------------------------------------------------------
# Nodes are named; edges carry a road type (paved/unpaved) used both for routing
# and for road-dust emissions in the simulator.
ROAD_NODES: Dict[str, Point] = {
    "W_END": (0, 290), "HAUL_J": (200, 290), "T2_STAGE": (320, 290), "CENTER": (365, 290),
    "A_J": (540, 290), "DEPOT": (700, 290),
    "MAIN_GATE": (365, 0), "CD_J": (365, 150), "NORTH_J": (365, 420), "N_END": (365, 560),
    "A_ACCESS": (540, 420), "A_EAST": (650, 420),
    "D_ACCESS": (480, 150), "D_EAST": (610, 150),
    "C_ACCESS": (250, 150), "C_WEST": (110, 150),
    "PIT": (230, 445), "PIT_N": (300, 455), "HAUL_W": (120, 420), "GATE2": (0, 420),
}

ROAD_EDGES: List[Tuple[str, str, str, str]] = [
    # (from, to, type, name)
    ("W_END", "HAUL_J", "paved", "Central Avenue"),
    ("HAUL_J", "T2_STAGE", "paved", "Central Avenue"),
    ("T2_STAGE", "CENTER", "paved", "Central Avenue"),
    ("CENTER", "A_J", "paved", "Central Avenue"),
    ("A_J", "DEPOT", "paved", "Central Avenue"),
    ("MAIN_GATE", "CD_J", "paved", "Park Street"),
    ("CD_J", "CENTER", "paved", "Park Street"),
    ("CENTER", "NORTH_J", "paved", "Park Street"),
    ("NORTH_J", "N_END", "paved", "Park Street"),
    ("NORTH_J", "A_ACCESS", "paved", "Tower Lane"),
    ("A_ACCESS", "A_EAST", "paved", "Tower Lane"),
    ("A_J", "A_ACCESS", "unpaved", "Tower service track"),
    ("CD_J", "D_ACCESS", "paved", "School Road"),
    ("D_ACCESS", "D_EAST", "paved", "School Road"),
    ("CD_J", "C_ACCESS", "paved", "Park Lane"),
    ("C_ACCESS", "C_WEST", "paved", "Park Lane"),
    ("HAUL_J", "PIT", "unpaved", "Haul Road"),
    ("PIT", "PIT_N", "unpaved", "Haul Road"),
    ("PIT", "HAUL_W", "unpaved", "Haul Road"),
    ("HAUL_W", "GATE2", "unpaved", "Haul Road"),
]

# External arterial shown for context on the map (not routable inside the site).
EXTERNAL_ROADS = [
    {"name": "Sector Road", "points": [(-60, -26), (780, -26)], "width": 20},
    {"name": "Gate 2 Link", "points": [(-60, 420), (0, 420)], "width": 10},
]

# Buildings, green areas and works areas (for the map / twin only) -----------
FEATURES = {
    "buildings": [
        {"name": "Tower 1 (under construction)", "zone": "A", "kind": "construction",
         "polygon": [(410, 370), (470, 370), (470, 430), (410, 430)]},
        {"name": "Tower 2 (under construction)", "zone": "A", "kind": "construction",
         "polygon": [(500, 450), (560, 450), (560, 510), (500, 510)]},
        {"name": "Tower 3 (foundation)", "zone": "A", "kind": "construction",
         "polygon": [(600, 360), (670, 360), (670, 400), (600, 400)]},
        {"name": "Site office", "zone": "A", "kind": "building",
         "polygon": [(420, 480), (470, 480), (470, 520), (420, 520)]},
        {"name": "Batching plant", "zone": "B", "kind": "industrial",
         "polygon": [(60, 330), (120, 330), (120, 375), (60, 375)]},
        {"name": "Block C1", "zone": "C", "kind": "residential", "polygon": [(260, 200), (330, 200), (330, 255), (260, 255)]},
        {"name": "Block C2", "zone": "C", "kind": "residential", "polygon": [(260, 45), (330, 45), (330, 120), (260, 120)]},
        {"name": "Senior Living", "zone": "C", "kind": "residential", "polygon": [(40, 205), (170, 205), (170, 260), (40, 260)]},
        {"name": "Green Valley Public School", "zone": "D", "kind": "school",
         "polygon": [(430, 165), (530, 165), (530, 215), (430, 215)]},
        {"name": "Community Health Clinic", "zone": "D", "kind": "clinic",
         "polygon": [(600, 60), (660, 60), (660, 110), (600, 110)]},
        {"name": "Little Sprouts Daycare", "zone": "D", "kind": "daycare",
         "polygon": [(545, 205), (585, 205), (585, 245), (545, 245)]},
        {"name": "Block D1", "zone": "D", "kind": "residential", "polygon": [(620, 180), (690, 180), (690, 255), (620, 255)]},
        {"name": "Block D2", "zone": "D", "kind": "residential", "polygon": [(400, 35), (470, 35), (470, 110), (400, 110)]},
        {"name": "Depot", "zone": None, "kind": "depot", "polygon": [(680, 300), (716, 300), (716, 335), (680, 335)]},
    ],
    "works": [
        {"name": "Excavation pit", "zone": "B", "kind": "pit",
         "polygon": [(150, 460), (300, 465), (320, 520), (170, 525)]},
        {"name": "Aggregate stockpile", "zone": "B", "kind": "stockpile",
         "polygon": [(70, 470), (120, 470), (120, 520), (70, 520)]},
    ],
    "green": [
        {"name": "Central Park", "zone": "C", "polygon": [(50, 50), (200, 50), (200, 180), (50, 180)]},
        {"name": "School green", "zone": "D", "polygon": [(430, 225), (530, 225), (530, 262), (430, 262)]},
        {"name": "Linear park", "zone": None, "polygon": [(372, 160), (382, 160), (382, 280), (372, 280)]},
    ],
}

PREVAILING_WIND_FROM_DEG = 305.0


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def point_in_polygon(pt: Point, poly: List[Point]) -> bool:
    x, y = pt
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi):
            inside = not inside
        j = i
    return inside


def zone_of(pt: Point) -> Optional[str]:
    for z in ZONES:
        if point_in_polygon(pt, z.polygon):
            return z.zone_id
    return None


def transport_weight(src: Point, dst: Point, wind_from_deg: float, wind_speed: float,
                     length_scale: float = 220.0) -> float:
    """Fraction of a source's local increment that reaches a receptor point.

    Downwind alignment and distance decay; calm winds spread pollution in all
    directions (less directional), stronger winds carry it further downwind.
    Identical in form to the simulator's transport term.
    """
    d = distance(src, dst)
    if d < 1.0:
        return 1.0
    wind_to = (wind_from_deg + 180.0) % 360.0
    ang = math.radians(bearing_deg(src, dst) - wind_to)
    align = (1.0 + math.cos(ang)) / 2.0                 # 1 downwind, 0 upwind
    directional = min(1.0, max(0.0, wind_speed / 1.5))
    shape = (1 - directional) * 0.3 + directional * align ** 2
    reach = length_scale * (0.5 + 0.9 * directional * align)
    return float(shape * math.exp(-d / reach))


# Routing ----------------------------------------------------------------------

def _graph() -> Dict[str, List[Tuple[str, float, str]]]:
    g: Dict[str, List[Tuple[str, float, str]]] = {k: [] for k in ROAD_NODES}
    for a, b, kind, _name in ROAD_EDGES:
        w = distance(ROAD_NODES[a], ROAD_NODES[b])
        g[a].append((b, w, kind))
        g[b].append((a, w, kind))
    return g


def nearest_node(pt: Point) -> str:
    return min(ROAD_NODES, key=lambda k: distance(ROAD_NODES[k], pt))


def shortest_route(start: Point, end: Point) -> Dict[str, object]:
    """Dijkstra over the internal road network; returns polyline and length (m)."""
    g = _graph()
    s, t = nearest_node(start), nearest_node(end)
    dist = {k: math.inf for k in g}
    prev: Dict[str, Optional[str]] = {k: None for k in g}
    dist[s] = 0.0
    heap = [(0.0, s)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        if u == t:
            break
        for v, w, _kind in g[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    path = []
    cur: Optional[str] = t
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    pts = [start] + [ROAD_NODES[n] for n in path] + [end]
    # remove consecutive duplicates
    clean: List[Point] = []
    for p in pts:
        if not clean or distance(clean[-1], p) > 0.5:
            clean.append((round(p[0], 1), round(p[1], 1)))
    length = sum(distance(clean[i], clean[i + 1]) for i in range(len(clean) - 1))
    return {"nodes": path, "points": clean, "length_m": round(length, 1)}


def site_geojson_like() -> Dict[str, object]:
    """Serialisable description of the whole site for the API and the UI."""
    return {
        "name": SITE_NAME,
        "area_acres": SITE_AREA_ACRES,
        "extent_m": [SITE_WIDTH_M, SITE_HEIGHT_M],
        "origin_latlng": ORIGIN_LATLNG,
        "boundary": BOUNDARY,
        "zones": [
            {**{k: v for k, v in asdict(z).items()}, "centroid": z.centroid, "area_m2": round(z.area_m2)}
            for z in ZONES
        ],
        "receptors": [asdict(r) for r in RECEPTORS],
        "sensors": [{**asdict(s), "latlng": to_latlng(*s.position)} for s in SENSORS],
        "fleet": [asdict(a) for a in FLEET],
        "depot": DEPOT,
        "gates": GATES,
        "road_nodes": ROAD_NODES,
        "roads": [
            {"from": a, "to": b, "type": kind, "name": name, "points": [ROAD_NODES[a], ROAD_NODES[b]]}
            for a, b, kind, name in ROAD_EDGES
        ],
        "external_roads": EXTERNAL_ROADS,
        "features": FEATURES,
        "prevailing_wind_from_deg": PREVAILING_WIND_FROM_DEG,
    }
