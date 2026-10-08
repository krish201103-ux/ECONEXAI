# Data Transparency

## 1. What data this build uses
**All telemetry in this build is synthetic.** No real site data was available for the prototype, so
EcoNexus AI is trained and demonstrated on data from a transparent, physics-motivated simulator
(`simulator.py`, seed 42) for a fictional site:

| | |
|---|---|
| Site | Green Valley Township, 100 acres, 4 zones (A Tower Construction, B Excavation & Haul Road, C Residential & Central Park, D School & Clinic Perimeter) |
| Period | 9 June – 10 Oct 2026 at 5-min resolution (35,712 rows per zone up to 6 Oct); the demo "now" is **7 Oct 14:30 IST** |
| Sensors | 27 nodes (24 online, 2 warning, 1 offline at 14:30), a site weather station and a regional reference station |
| Activity | Construction schedules per zone, haul-truck trips, commuter traffic, and a log of 296 historical interventions |

The UI says so everywhere it matters: the sidebar data chip, *Synthetic demo data* tags, *Demo
simulation* on verified outcomes, the Settings → Data source card, and the `synthetic_data: true`
flag in every API health response.

## 2. What the simulator models
- **Meteorology:** seasonal and diurnal temperature and humidity; monsoon easterlies and
  post-monsoon north-westerlies; mixing height; rain events with wash-out.
- **Site activity:** work-day construction schedules, haul-road truck trips and commuter traffic
  with rush-hour peaks.
- **Concentrations:** regional background (seasonal and synoptic) plus local increments from
  construction dust, haul-road resuspension and exhaust. These are scaled by dispersion
  (ventilation = wind speed × mixing height) and carried between zones by the wind.
- **Interventions:** a historical dispatch policy whose effect *emerges* from surface wetting, mist
  scavenging and sweeping in the physics. It is not a hard-coded percentage, so the learning loop
  has something real to estimate.
- **Sensors:** zone readings are averages of noisy optical sensors with a humidity bias, dropouts,
  one drifting node and one offline node.
- **Hero day (7 Oct):** a scripted Zone B excavation and haul surge under stagnant afternoon
  winds, plus an earlier Zone A event already being handled by Sweeper 01.

## 3. What the models are allowed to see
`data_loader.py` exposes only what a real site would observe: zone PM readings, weather,
the reference station, activity logs, the intervention log and a modelled surface-moisture state.
The simulator's *true* source contributions and counterfactuals are kept apart and are used **only
to evaluate** the models (attribution error, verification validation). A test checks that no
feature uses PM values from after the issue time, and another checks that the hero intervention is
not visible at 14:30.

**Weather forecasts.** Planning features use a forecast of the weather at the target time. In the
synthetic world this is the true future weather plus horizon-dependent error. It stands in for an
NWP feed (IMD/GFS/ECMWF) on a real deployment.

## 4. Preprocessing and quality control (`data_cleaning.py`)
1. **Physical bounds:** negative readings are clamped to 0. For single packets arriving at the API
   (`/api/v1/telemetry/ingest`), PM10 is also kept ≥ PM2.5.
2. **Hygroscopic correction** for optical sensors above 60% RH:
   `CF = 1 + κ·RH/(100 − RH)` with κ = 0.25 and RH clamped at 98%, then `PM_dry = PM_raw / CF`.
3. **Spike rejection:** modified z-score (median absolute deviation) on first differences, with a
   threshold of 3.5.
4. **Gap filling:** linear interpolation within a 4-hour imputation budget. Longer gaps are left
   missing and reported.

QA counts from the training run (zone B, PM10): 35,712 rows, 134 missing (0.38%), 134 filled by
interpolation, 28,696 humidity-corrected readings, 0 spikes rejected. The Sensors & Data screen
shows these totals across all zones.

## 5. Sensor health and fallback (`intelligence.py`)
- Each node is ONLINE, WARNING (low battery, calibration drift) or OFFLINE, with a reason.
- For an offline node, the value is estimated by inverse-distance weighting from online nodes in
  the same zone. Its expected error is measured by leave-one-out cross-validation (±11% at 14:30),
  and the UI shows **Data confidence reduced** on that node.
- Zone data confidence is the share of healthy nodes, penalised for warnings and fallbacks.
  Forecast and attribution confidence are scaled by it.

## 6. Using real data
Export site telemetry to a long-format CSV and run:
```bash
.venv/bin/python train.py --data your_site.csv     # Windows: .venv\Scripts\python train.py --data ...
```
Required columns: `timestamp, zone_id, pm10, pm25, temperature, relative_humidity, wind_speed,
wind_dir_from`. Optional columns: `mixing_height, rain_mm, reference_pm10, construction_intensity,
planned_construction, truck_trips, vehicles, intervention_active`. Missing optional columns get
documented defaults. Update `site_model.py` (zones, sensor positions, roads, receptors, fleet) to describe the real
site. The offline snapshot always shows the demo site, so it is not rewritten when `--data` is used.

## 7. Keys and secrets
- **Where keys live:** `.env` in the project folder (read by `config.py`, never committed —
  `.gitignore` lists it). `.env.example` documents every variable and must never contain real
  values.
- **Google Maps:** the browser key can live in `static/js/config.js` (needed when the dashboard is
  opened from disk) or in `.env` as `GOOGLE_MAPS_API_KEY`, served through `/api/v1/client-config`,
  or be pasted in Settings and saved in that browser. A browser key is always visible to page
  visitors, so restrict it in Google Cloud Console to the Maps JavaScript API and to your site's
  web addresses. The key used in v2 was committed in three files. If that repository was ever
  shared, create a new restricted key and put it in `config.js`.
- **Groq (Copilot LLM mode):** `GROQ_API_KEY` is read from `.env` only and is never sent to the
  browser. Without it the Copilot uses its built-in engine. If a Groq key was ever committed (for
  example in an `.env.example`), revoke it at console.groq.com and create a new one.
