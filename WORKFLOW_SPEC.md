# EcoNexus AI — Workflow Specification

**EcoNexus AI** observes the physical environment, predicts environmental risk, uses **DUSTWISE**
to optimise interventions, and closes the loop by proving whether each intervention worked.

## 1. The closed loop and where it lives

Code paths are modules in the project root (the module map is at the top of `server.py`).

| Stage | Question | Implementation | Screen |
|---|---|---|---|
| OBSERVE | What is happening? | 27 sensors, weather station, reference station → `data_cleaning.py`, `intelligence.fleet_snapshot` | Environmental Map, Sensors & Data |
| UNDERSTAND | Is this an event? | Rapid-rise, sustained and anomaly detection → `intelligence.detect_events` | Pollution Events |
| PREDICT | What happens next if we do nothing? | 18-horizon XGBoost with conformal 90% intervals → `ml_models.py` | Forecast & Intelligence |
| DIAGNOSE | Why? | NNLS receptor model (contributors), exact group-Shapley (forecast drivers) → `explainability.py` | Command Center, Forecast |
| PRIORITIZE | Where first? | Intervention Priority Score from model outputs → `dustwise.py` | DUSTWISE Optimizer |
| OPTIMIZE | Which action and which unit? | Action comparison × learned multipliers; Hungarian fleet assignment on the road graph | DUSTWISE Optimizer |
| ACT | Deploy | Operator approval → dispatch → active (`POST /events/{id}/approve`, `/start`) | Command Center, Resource Operations |
| VERIFY | Did it work? | Observed change and effect vs the no-action counterfactual → `verification.verify_intervention` | Impact & Verification |
| LEARN | Improve the next decision | Bayesian field multiplier per action → `verification.LearningLoop` | Impact & Verification |

The rail under the top bar shows the active event's position in this loop on every screen.

## 2. Hero demonstration (EV-042, Zone B)

| # | Time | What happens | Screen |
|---|---|---|---|
| 1 | 14:15 | Zone B PM10 begins rising | Command Center map |
| 2 | 14:20 | Sensor EC-042 flags the anomaly | Map (pulsing sensor) |
| 3 | 14:20 | Event EV-042 is created | AI insight, events badge |
| 4 | 14:24 | AI-estimated contributors: construction ≈ 64%, road dust, traffic, other | AI insight |
| 5 | 14:27 | Forecast: High risk within 15–90 min (peak ≈ 205, band to 228) | AI insight, Forecast |
| 6 | 14:29 | IPS rises to 86/100: immediate action | Intervention priority |
| 7 | 14:29 | DUSTWISE recommends targeted dust suppression with Tanker 02 | DUSTWISE card |
| 8 | 14:33 | Operator reviews and approves | Approve button: loading → approved |
| 9 | 14:38 | Tanker 02 is routed to Zone B (0.28 km, ETA 6 min) | Map route, allocation |
| 10 | 14:42 | Intervention active | Status Pending → Approved → Active |
| 11 | 15:05 | Sensor readings fall while the no-action forecast keeps rising | Map, charts |
| 12 | 15:20 | Before → after: 199 → 152 µg/m³ (−23.4% observed; −22.2% vs forecast) | Impact & Verification |
| 13 | 15:20 | Event verified (observation below the 90% band) | Event timeline |
| 14 | 15:20 | Learning loop records the outcome (sprinkling n 199 → 200) | Impact & Verification |

Run it from **Run hero demo** in the sidebar, either step by step or with auto-play. When the API
is running, steps 8–13 go through the real server endpoints.

## 3. Governance rules built into the product
- **Operator approval required.** DUSTWISE never dispatches by itself, and *Automatic physical
  control* is shown as disabled.
- **Estimates are labelled as estimates.** Contributors are *AI-estimated*, action benefits are
  *AI scenario estimates*, and twin outputs are *Simulation / scenario estimate — not field
  measurement*.
- **Verification is against a counterfactual.** An effect is only called *verified* when the
  observation is below the no-action forecast's 90% band. Otherwise it is *inconclusive*.
- **Degraded data is visible.** Offline nodes show *Data confidence reduced*, and zone confidence
  scales forecast and attribution confidence.

## 4. API surface (Flask, `server.py`; `GET /api/v1` lists every route)
`GET /api/v1/health, /client-config, /snapshot, /status, /briefing, /zones, /zones/{zone},
/forecast/{zone}?tab=, /explain/{zone}?horizon=, /attribution/{zone}, /events, /events/{id},
/events/{id}/impact, /dustwise/priorities, /dustwise/actions/{zone}, /dustwise/allocation, /sensors,
/learning, /model, /sustainability, /export/telemetry.csv, /export/report.json, /chat-status`

`POST /api/v1/events/{id}/approve | start | verify | escalate, /twin/simulate, /twin/compare,
/ingest, /demo/reset, /chat`

## 5. Copilot in the loop
The EcoNexus Copilot (`chatbot.py`) answers questions at any stage from the same state — "why is
Zone B rising?" (DIAGNOSE), "what should I do now?" (OPTIMIZE/ACT), "did it work?" (VERIFY) — and
links to the screen that holds the evidence. It recommends; it never approves or dispatches.
