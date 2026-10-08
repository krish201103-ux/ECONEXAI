# Methodology & Evaluation

This document describes how each EcoNexus AI model works and how it is evaluated. All numbers come
from `artifacts/*.json`, which training writes (`python train.py`, or `python server.py --retrain`).
Rerun training and they are regenerated. Nothing here is typed in by hand.

> All results below are on **synthetic** telemetry (see [DATA_TRANSPARENCY.md](DATA_TRANSPARENCY.md)).
> They show that the method works end to end on data with realistic structure. They are not a claim
> about accuracy on a real site.

---

## 1. Problem formulation

For each zone *z* (A–D) and issue time *t*, predict zone-average PM10 and PM2.5 at *t + h* for
18 horizons. There are 6 **operational** horizons (15–90 min) that drive DUSTWISE and 12
**planning** horizons (2–72 h) that drive the outlook. Each forecast has a calibrated 90% interval
and an explanation.

Operational forecasts are **no-new-action baselines**: they answer *"what happens if we do nothing
new?"*. Training rows in which a new intervention starts inside the forecast window are excluded.
That one quantity serves three jobs:
1. prioritisation (how bad will it get?),
2. the counterfactual for verification (did the action beat "doing nothing"?),
3. the reference for the learning loop (how much did each action really achieve?).

## 2. Forecaster (`ml_models.py`, `ml_features.py`)

| Item | Choice | Why |
|---|---|---|
| Strategy | Direct multi-horizon: one model per horizon and pollutant (36 models) | No error compounding from recursive roll-outs |
| Algorithm | XGBoost `hist` (scikit-learn HistGradientBoosting fallback) | Non-linear thresholds (stagnation, wet surfaces); fast; best in the benchmark |
| Target | `log1p(y[t+h]) − log1p(y[t])` (log-change from the current level) | Robust to seasonal regime shifts such as the October rise, which absolute-level trees cannot extrapolate to |
| Operational features (51) | Recent trend (lags, rolling stats, slopes, PM ratio); site activity (current plus *planned* construction at t+h, expected haul trucks, surface moisture); dispersion (current weather plus NWP forecast for t+h, ventilation = wind × mixing height); regional background; calendar | Only information available at *t* (enforced by a leakage test) |
| Planning features (26) | Change-based versions: activity change, log dispersion change, background change, anomalies vs 24 h | Absolute-level features learned seasonal mean-reversion that failed in October |
| Hyperparameters | 350 trees; depth 6 and learning rate 0.06 (operational); depth 4, λ = 5 and learning rate 0.05 (planning) | Planning models are more regularised because their signal-to-noise ratio is lower |
| Uncertainty | Split-conformal 90% intervals on log-residuals from a later calibration window | Distribution-free; coverage is *measured* on the test window, not assumed |

**Data splits** (time-ordered, never shuffled): train to 14 Sept → calibrate 15–25 Sept → test
26 Sept – 6 Oct. The production model retrains to 30 Sept and calibrates to 6 Oct. The demo day
(7 Oct) is never in training.

### 2.1 Hold-out test (PM10)

| Horizon | RMSE | MAE | R² | Willmott d | Skill vs persistence | 90% coverage | Mean interval width | POD (>150) | FAR | CSI |
|---|---|---|---|---|---|---|---|---|---|---|
| 15m | 7.0 | 5.3 | 0.953 | 0.988 | +19.6% | 94.3% | 25 | 0.90 | 0.10 | 0.82 |
| 30m | 7.8 | 5.8 | 0.942 | 0.985 | +23.8% | 94.9% | 28 | 0.88 | 0.11 | 0.80 |
| 60m | 9.1 | 6.8 | 0.921 | 0.979 | +29.0% | 93.5% | 31 | 0.77 | 0.09 | 0.72 |
| 90m | 9.5 | 7.2 | 0.914 | 0.977 | +36.5% | 94.2% | 34 | 0.80 | 0.11 | 0.73 |
| 6h | 16.2 | 12.2 | 0.736 | 0.937 | +48.3% | 93.4% | 56 | 0.74 | 0.24 | 0.61 |
| 24h | 21.9 | 16.9 | 0.468 | 0.859 | +33.6% | 85.3% | 60 | 0.55 | 0.29 | 0.45 |
| 72h | 28.4 | 21.7 | −0.026 | 0.705 | +42.0% | 82.4% | 72 | 0.35 | 0.48 | 0.26 |

*RMSE/MAE and interval width in µg/m³. Skill = 1 − RMSE/RMSE(persistence). POD/FAR/CSI are for
exceedance of the Elevated level (150 µg/m³). PM2.5 results are in `evaluation.json` and follow the
same pattern (e.g. 60 min: RMSE 4.7, R² 0.93, coverage 93%).*

**Reading the results honestly**
- At the operational horizons the model is accurate (R² ≥ 0.91) and its intervals are well
  calibrated (93–95%). These are the forecasts DUSTWISE acts on.
- At 24–72 h the model still beats persistence, but R² is low and coverage drops below 90% because
  the test window (the post-monsoon rise) differs from the calibration window. The UI shows these
  horizons as an *outlook* and flags any horizon that does not beat persistence.

### 2.2 Walk-forward benchmark (`ml_models.run_benchmark`)
Rolling-origin cross-validation with an expanding window: 4 folds of 10 days (27 Aug – 5 Oct). All
learned models share the same features and log-change target, so differences come from the
algorithm.

| Model | RMSE 60m | Skill 60m | RMSE 6h | Skill 6h | RMSE 24h | Skill 24h | Train s | Latency ms |
|---|---|---|---|---|---|---|---|---|
| Persistence | 13.8 | 0% | 32.1 | 0% | 29.4 | 0% | 0 | 0 |
| Seasonal naive (t−24h) | 29.7 | −116% | 30.0 | +7% | 29.4 | 0% | 0 | 0 |
| Diurnal climatology | 36.7 | −167% | 37.4 | −16% | 38.3 | −30% | 0 | 0 |
| Ridge regression | 8.8 | +36% | 14.6 | +55% | 16.2 | +45% | 0.03 | 0.9 |
| Random forest | 7.3 | +46% | 11.9 | +63% | 15.9 | +46% | 7.5 | 23.7 |
| **XGBoost (production)** | **7.0** | **+49%** | **11.8** | **+63%** | **15.4** | **+48%** | 1.0 | 2.2 |

XGBoost is best at every horizon. It is about 7× faster to train and 11× faster at inference than
random forest, which is why it is used in production. The benchmark folds cover a longer and more
varied period than the hold-out window, so their 24 h scores are higher than in §2.1.

### 2.3 Why the LSTM and SVR were removed
The v2 repository listed an LSTM and an SVR that were not trained or evaluated. Untested models
were removed rather than reported. A sequence model can be added through the same benchmark
harness if it beats XGBoost there.

## 3. Forecast explanations: exact group-Shapley (`explainability.exact_group_shapley`)
Features are grouped into five drivers: recent pollution trend, planned site activity, wind &
dispersion, regional background and time of day. With 5 groups, all 32 coalitions are evaluated,
so the attributions are **exact** interventional Shapley values against a background of
"typical comparable moments" (same zone, similar hour, recent weeks). They are additive:
`typical level + Σ driver contributions = forecast`. The *Why is the forecast moving?* panel shows
them. No `shap` dependency is needed.

## 4. Source attribution: "AI-estimated contributors" (`explainability.SourceAttributor`)
A receptor model in the spirit of Chemical Mass Balance. The local excess of PM10 *and* PM2.5 (zone
minus the regional reference station) is fitted jointly with non-negative least squares on
dispersion-scaled activity indicators: construction (own plus wind-weighted upwind, × surface
dryness), haul-road resuspension (trucks × dryness), exhaust, commuter traffic and other. Using
PM2.5/PM10 source profiles (crustal dust is coarse, exhaust is fine) together with surface-moisture
variation is what separates road dust from exhaust. Weak emission-inventory priors stabilise the
fit, and a day-block bootstrap (40 samples) gives ranges and a confidence score.

**Validation against the simulator's true shares** (75 hourly samples, zones A/B, working hours):
mean absolute error **6.2 percentage points** (construction 7.8, road dust 6.4, traffic 3.0, other
7.7). Fit R²: A 0.86, B 0.92, C 0.48, D 0.64. C and D are dominated by regional background, which
the model correctly does not assign to site sources. The UI always labels these as
*AI-estimated contributors*.

## 5. Event detection (`intelligence.detect_events`)
- **Rapid rise:** PM10 ≥ 150 and a ≥ 20% rise in 30 min, confirmed on two readings.
- **Sustained:** PM10 ≥ 150 for 30 min.
- **Anomaly:** robust z ≥ 3.5 against the zone's same-hour median/MAD over the past 7 days.

An event closes after 30 min below 130, and a new event in the same zone needs a 60-min gap. The
lifecycle is DETECTED → CLASSIFIED → PREDICTED → RECOMMENDED → APPROVED → ACTIVE → VERIFIED, with a
timestamped audit trail.

## 6. DUSTWISE (`dustwise.py`)
**Intervention Priority Score** (0–100), built from model outputs:

`IPS = 0.25·S_poll + 0.20·P_pred + 0.15·C_src + 0.15·E_exp + 0.10·P_pers + 0.15·F_feas`

| Factor | Computed from |
|---|---|
| S_poll severity | Zone PM10 between a clean-air floor (40) and the High level (200) |
| P_pred predicted growth | 90-min forecast vs now, including the conformal upper band |
| C_src source confidence | Attribution confidence × controllable (site) share |
| E_exp exposure | Wind-weighted sensitive receptors (school, clinic, daycare) and population |
| P_pers persistence | Forecast ventilation over the next 90 min |
| F_feas feasibility | Best resource ETA, availability and controllability |

Bands: ≥ 80 immediate action, 60–80 high, 40–60 watch, < 40 routine.

**Action comparison.** For each hotspot, five interventions (targeted sprinkling, mechanical
sweeping, construction suppression, traffic-flow recommendation, combined) are compared. Each
gets an *AI scenario estimate* of PM10 reduction (local source shares × the action's efficacy
assumptions × the **learned field multiplier**), plus resource needs, cost level, response time and
feasibility.

**Fleet allocation.** Hungarian assignment of tankers and sweepers to hotspots. It maximises
priority-weighted expected benefit minus travel time on the site road graph (Dijkstra), subject to
availability and water capacity. Units already working keep their task. DUSTWISE never dispatches
on its own: every recommendation needs operator approval.

## 7. Verification (`verification.verify_intervention`)
Two measures are reported and never conflated:
- **Observed change:** the highest zone reading in the 30 min before the start vs the reading at
  verification.
- **Model-estimated effect:** the observation vs the no-action forecast issued at decision time.
  It counts as *verified* only when the observation is below that forecast's lower 90% bound.

Hero event EV-042: observed −23.4% (199 → 152 µg/m³); model-estimated effect −22.2%
(−29% to −15%) against a no-action forecast of 196 µg/m³. On synthetic data the true counterfactual is known: 250 µg/m³, a true effect of
−39%. It is shown separately as **validation**, which shows the method is conservative.

## 8. Learning loop (`verification.LearningLoop`)
For each of the 288 logged historical interventions:
1. reconstruct DUSTWISE's *expected* reduction at dispatch time,
2. measure the reduction as `1 − observed / no-action forecast` 60 min later, and
3. fit `measured = m × expected + noise` per action with a Bayesian prior `m ~ N(1, 0.35²)`.

Current posteriors: targeted sprinkling 0.77 (0.71–0.83, n = 199), mechanical sweeping 0.39
(0.05–0.79, n = 53), combined 0.64 (0.56–0.72, n = 36). Construction suppression and traffic
management have no field history yet (prior only). DUSTWISE scales its scenario estimates by *m*.
Verifying EV-042 adds one observation and updates the sprinkling posterior live (n 199 → 200).

## 9. Digital twin (`digital_twin.py`, `static/js/twin.js`)
A steady-state, Gaussian-plume-style dispersion model on a 20 m grid. Sources are the construction
areas, the haul road (line source) and commuter roads. Source strengths are calibrated by
constrained least squares to the observed 14:30 zone levels. The browser runs a line-for-line port
so the sliders respond instantly, and a test checks the Python and JavaScript versions agree.
Every output is labelled *Simulation / scenario estimate — not field measurement*.

## 10. Tests (`tests/test_platform.py`, 37 tests)
They cover the site model (100 acres, consistent IDs), routing on the road network, generator
determinism, the hero scenario shape, **the hero intervention not being visible at 14:30**,
hygroscopic correction and gap filling, **no future PM values in the features**, group-Shapley
additivity, computed and skilful evaluation metrics, ordered forecast intervals, attribution shares
summing to 100%, the hero story, allocation respecting unit status, verification and the learning
update, IPS bounds, twin interventions reducing Zone B, **twin Python/JavaScript parity**, the `.env` loader
(UTF-16 and BOM files from Windows editors), IPS weights coming from `config.py`, the Flask API
lifecycle (approve → start → verify, 404/409/422 errors, CORS for a page opened from disk, CSV and
report exports) and the Copilot (scope filter, intent routing, **answers quoting the live numbers**,
the offline pack covering every intent, and falling back to the built-in engine when Groq fails).

## 11. EcoNexus Copilot (`chatbot.py`, `static/js/copilot.js`)
The assistant is built so that it cannot make numbers up:

1. **Scope filter.** A keyword/phrase heuristic declines clearly off-topic questions before any
   model is called (instant, free, and identical offline).
2. **Built-in engine.** Each on-topic question is routed to one of 20 intents (briefing,
   recommendation, why, forecast, priority, verification, learning, sensors, model accuracy,
   weather, sustainability, twin, events, fleet, exposure, data, methodology, action levels, zone,
   greeting). Each intent is a small function that reads the *live platform state* — the same
   dictionary the dashboard renders — and writes the answer from it, with deep links to the screen
   that shows the evidence. A test checks that the forecast answer quotes the forecaster's own
   60-minute median.
3. **Optional LLM.** With `GROQ_API_KEY` set, Groq (`llama-3.1-8b-instant` by default) receives a
   compact LIVE CONTEXT block plus the built-in engine's answer, and is instructed to use only those
   numbers. If Groq fails, the built-in answer is returned and labelled.
4. **Offline.** `train.py` exports every intent's answer (per zone, and per lifecycle stage for the
   briefing, recommendation and verification) plus the intent patterns into `static/data/snapshot.js`,
   so the dashboard opened from disk routes questions the same way with no server.

Rounding follows the dashboard's rule (half away from zero), so the Copilot and the screens never
disagree on a value such as 174.5.
