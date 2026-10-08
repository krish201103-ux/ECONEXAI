# EcoNexus AI

AI-powered environmental intelligence and dust-control platform for large construction sites.
It forecasts PM10/PM2.5 from 15 minutes to 72 hours ahead with XGBoost, explains *why* with
exact Shapley drivers and a receptor model, and uses its **DUSTWISE** engine to rank hotspots and
dispatch water tankers and sweepers. It then **proves** whether each intervention worked against
a no-action forecast and learns from the result. An integrated assistant, the **EcoNexus
Copilot**, answers questions about the site from the same live data.

Built with Flask, XGBoost, scikit-learn, SciPy, pandas and a dependency-free JavaScript dashboard.
Research and educational use only.

```
OBSERVE → UNDERSTAND → PREDICT → DIAGNOSE → PRIORITIZE → OPTIMIZE → ACT → VERIFY → LEARN
```

> **Data transparency.** This build runs on *synthetic* data for a fictional 100-acre site, Green
> Valley Township, produced by a physics-motivated simulator. Every number in the UI and in this
> README is **computed** from that data by the code here. Nothing is typed in by hand. See
> [DATA_TRANSPARENCY.md](DATA_TRANSPARENCY.md).

## Objectives

- Forecast zone-level dust with honest uncertainty (90% intervals whose coverage is *measured*),
  so site teams act before levels peak rather than after.
- Turn forecasts into a decision: which zone first, which action, which unit, and why.
  Every recommendation still needs operator approval.
- Verify every intervention against the counterfactual "what if we had done nothing", and feed
  the result back into future recommendations.
- Make all of this usable by non-specialists: plain-language answers, an AI shift briefing,
  a command palette and an assistant scoped strictly to the platform's own data.

## Project Structure

```
econexus_ai/
├── server.py               # Flask routes only -- parses requests, delegates to the modules below,
│                           #   turns results/exceptions into HTTP responses (python server.py)
├── bootstrap.py            # First run: Python check, .venv, install requirements, train if needed
├── config.py               # Central config (env-driven via .env; see Configuration)
├── site_model.py           # Green Valley Township: zones, 27 sensors, roads, receptors, fleet, routing
├── simulator.py            # Physics-motivated synthetic telemetry (stand-in for real site data)
├── data_cleaning.py        # QA/QC: hygroscopic correction, spike rejection, bounded gap filling
├── data_loader.py          # Builds the observed dataset every model consumes (or loads a real CSV)
├── ml_features.py          # Feature engineering, forecast horizons, metrics, walk-forward splits
├── ml_models.py            # MultiHorizonForecaster (XGBoost + split-conformal) and the benchmark
├── explainability.py       # Exact group-Shapley forecast drivers + NNLS source attribution
├── intelligence.py         # Action levels, pollution event detection, sensor fleet health
├── dustwise.py             # DUSTWISE: Intervention Priority Score, action comparison, fleet allocation
├── digital_twin.py         # What-if dispersion simulator (mirrored in static/js/twin.js)
├── verification.py         # Did it work? Counterfactual verification + Bayesian learning loop
├── pipeline.py             # Assembles the complete platform state from all of the above
├── forecast_service.py     # Orchestration: shared state, event lifecycle, request validation, exports
├── chatbot.py              # EcoNexus Copilot: scope filter, grounded engine, Groq system prompt
├── train.py                # Trains + evaluates the models, writes artifacts/ and the offline snapshot
├── artifacts/              # Trained models (+ manifest.json checksums) and computed metrics -- included
├── requirements.txt
├── .env.example            # Copy to .env; every setting is optional
├── index.html              # Opens static/index.html (dashboard without a server)
├── static/                 # The web app (served by server.py, or opened from disk)
│   ├── index.html          # One page, three sections: Home, Dashboard (11 screens), About
│   ├── css/app.css         # Design system: light theme, IBM Plex Sans, 12-column grid
│   ├── js/                 # app.js (shell + routing), mapview.js (real map), views-home.js
│   │                       #   (Home, About), views-a/b.js (dashboard screens), store, charts,
│   │                       #   twin, copilot.js (assistant), palette.js (command palette)
│   ├── vendor/             # Leaflet 1.9.4 (BSD-2) and Leaflet.GoogleMutant 0.16 (MIT), bundled
│   ├── data/snapshot.js    # Exported model output (offline mode, written by train.py)
│   └── fonts/              # IBM Plex Sans (SIL OFL 1.1)
├── tests/test_platform.py  # pytest suite
├── METHODOLOGY_AND_EVALUATION.md
├── DATA_TRANSPARENCY.md
└── WORKFLOW_SPEC.md
```

Each module has a single responsibility, so adding a new forecasting model only means touching
`ml_models.py` (and registering it in `run_benchmark`), and a new Copilot skill is one intent plus
one handler in `chatbot.py` -- not editing a server file that also handles HTTP routing.

## Installation

Requires Python 3.10 or newer (tested on 3.12 and 3.13). Nothing else: the first run installs
everything.

```bash
# 1. Unzip / clone and enter the project
cd econexus_ai

# 2. (Optional) configure keys -- everything works without them
cp .env.example .env          # Windows: copy .env.example .env
# open .env and paste a Groq key (free at https://console.groq.com) for the Copilot's LLM mode
```

On the first `python server.py` (next section), `bootstrap.py`:

1. checks the Python version,
2. creates a private environment in `.venv/` (your system Python stays clean) -- or uses the
   virtual/conda environment you already activated,
3. installs `requirements.txt` there (a few minutes; XGBoost is the largest). It tries the exact
   tested versions first and, if your Python has no ready-made packages for them, the newest
   compatible ones,
4. checks that the trained models in `artifacts/` load, and retrains them (~4 min) only if not.

Later starts skip all of this unless `requirements.txt` changes. Prefer to do it by hand?

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python server.py
```

## Running

```bash
python server.py
```

The server starts at `http://127.0.0.1:8000` and opens the app in your browser. The console
banner confirms the address, the model backend, whether the Copilot uses Groq or its built-in
engine, and where the Maps key comes from. Press **Ctrl+C** to stop.

The app has three sections, always in the top bar:

| Section | Address | What it shows |
|---|---|---|
| **Home** | `/` | What is happening on site now, a live map, the four zones, how the loop works, Copilot questions |
| **Dashboard** | `/dashboard` | All 11 working screens in a sidebar: Command Center, Environmental Map, Pollution Events, Forecast, DUSTWISE, Resource Operations, Digital Twin, Impact, Sensors, Sustainability, Settings |
| **About** | `/about` | Objectives, method, model accuracy, priority score, the map, data honesty, technology, limitations |

| Option | What it does |
|---|---|
| `--port 9000` | Use another port (the next free one is picked automatically if 8000 is busy) |
| `--host 0.0.0.0` | Let phones/laptops on the same Wi-Fi open the dashboard |
| `--no-browser` | Don't open a browser window |
| `--retrain` | Retrain the models and refresh the offline snapshot before starting |
| `--reinstall` | Reinstall the libraries |
| `--test` | Install pytest and run the test suite |
| `--system` | Install into the current Python instead of `.venv` |
| `--install-only` | Set everything up, then stop |

**No server?** Open `index.html` (or `static/index.html`) directly. Every screen, the hero demo and
the Copilot work offline from `static/data/snapshot.js`. If the server is running, a page opened
from disk connects to it automatically (`apiBaseUrl` in `static/js/config.js`).

## Workflow

1. **Command Center** -- status strip, an **AI shift briefing** written from live model output
   (press **Listen** to hear it), the site map, the AI insight for the active event and the
   DUSTWISE recommendation with **Approve**.
2. **Environmental Map, Pollution Events, Forecast & Intelligence** -- where, what and what next,
   with 90% intervals and the drivers of each forecast.
3. **DUSTWISE Optimizer, Resource Operations** -- priority scores, action comparison, fleet routes.
4. **Digital Twin** -- what-if sliders for construction, traffic, wind and the intervention.
5. **Impact & Verification** -- before/after vs the no-action forecast, KPIs, learning loop.
6. **Sensors & Data, Sustainability, Settings** -- data quality, long-term compliance and water,
   model card, benchmark, Copilot and map settings.

**Run hero demo** (sidebar, or press **D**) walks through the 14-step EV-042 scenario: anomaly →
event → contributors → forecast → priority → recommendation → approval → Tanker 02 dispatch →
active → sensor response → before/after → verified → learning update.

### What's new in the UI (v4)

| Feature | How to use it |
|---|---|
| **EcoNexus Copilot** | Top-bar **Copilot** button or **/**. Ask in plain words ("why is Zone B rising?", "what should I do in Zone A?", "is the school exposed?"). Answers quote live numbers, link to the screen that shows the evidence, and suggest follow-ups. Voice input (mic) and read-aloud where the browser supports them. |
| **Ask Copilot everywhere** | Every screen has an **Ask Copilot** button with a question for that screen; key cards have **Why?** / **Reliable?** chips. |
| **AI shift briefing** | Command Center. Updates with the event lifecycle (pending → approved → active → verified); **Listen** or press **B**. |
| **Command palette** | **Ctrl/⌘ K** or the search box: jump to any screen, zone, event or sensor, approve the recommendation, export data, or type a question to send it to the Copilot. |
| **Keyboard shortcuts** | **?** lists them; **Alt 1–9** jumps between screens. |

## Methodology (summary)

- **Forecasting:** one XGBoost model per (horizon, pollutant) -- 18 horizons from 15 min to 72 h
  for PM10 and PM2.5 -- each predicting the log-change from the current level. Features use only
  information available at forecast time (a test enforces this).
- **Uncertainty:** split-conformal 90% intervals, calibrated on a held-out window and checked on a
  later test window (coverage is reported, not assumed).
- **No-action baselines:** 15-90 min forecasts are trained without windows in which a new
  intervention started, so they answer "what if we do nothing?" -- exactly what DUSTWISE
  prioritises on and what verification compares against.
- **Validation:** a hold-out test plus rolling-origin (walk-forward, expanding window)
  cross-validation over 4 folds against 5 baselines -- never a random split, which would leak the
  future into training.
- **Explainability:** exact group-Shapley forecast drivers (5 groups, all 32 coalitions, no `shap`
  dependency) and an NNLS receptor model for "AI-estimated contributors" with bootstrap ranges.
- **DUSTWISE:** IPS = 0.25 severity + 0.20 predicted growth + 0.15 source confidence + 0.15
  exposure + 0.10 persistence + 0.15 feasibility; action comparison with learned field multipliers;
  Hungarian fleet assignment over Dijkstra routes on the site road network.
- **Learning:** a Bayesian field multiplier per action (prior N(1, 0.35²)) updated by each verified
  event.

Full detail: [METHODOLOGY_AND_EVALUATION.md](METHODOLOGY_AND_EVALUATION.md).

## Model results (synthetic data, computed by `train.py`)

**Hold-out test** -- trained to 14 Sept, calibrated 15-25 Sept, tested 26 Sept - 6 Oct
(11-12k zone-forecasts per horizon). PM10, zone averages.

| Horizon | RMSE µg/m³ | MAE | R² | Skill vs persistence | 90% interval coverage | Exceedance (>150) hit rate |
|---|---|---|---|---|---|---|
| 15 min | 7.0 | 5.3 | 0.95 | +20% | 94% | 90% |
| 60 min | 9.1 | 6.8 | 0.92 | +29% | 94% | 77% |
| 90 min | 9.5 | 7.2 | 0.91 | +36% | 94% | 80% |
| 6 h | 16.2 | 12.2 | 0.74 | +48% | 93% | 74% |
| 24 h | 21.9 | 16.9 | 0.47 | +34% | 85% | 55% |
| 72 h | 28.4 | 21.7 | −0.03 | +42% | 82% | 35% |

Strong at the operational horizons DUSTWISE acts on; beyond 24 h it still beats persistence but R²
falls to about zero, so the UI shows 24-72 h as an *outlook* with wide intervals.

**Walk-forward benchmark** (RMSE µg/m³, 4 expanding folds of 10 days): persistence 13.8 / 32.1 /
29.4, ridge 8.8 / 14.6 / 16.2, random forest 7.3 / 11.9 / 15.9, **XGBoost 7.0 / 11.8 / 15.4** at
60 min / 6 h / 24 h.

**Source attribution:** 6.2 percentage points mean error against the simulator's true shares.
**Hero event EV-042:** Zone B fell from 199 to 152 µg/m³ (−23.4% observed); the model-estimated
effect against the no-action forecast was −22.2% (range −29% to −15%) -- verified.

## API

All routes are under `/api/v1` and return JSON; `GET /api/v1` lists every one.

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/v1/health` | GET | Server and model status |
| `/api/v1/snapshot` | GET | Everything the dashboard needs in one call |
| `/api/v1/status`, `/briefing` | GET | Status strip and the AI shift briefing |
| `/api/v1/zones`, `/zones/<zone>` | GET | Zone readings, forecast, priority, attribution, sensors |
| `/api/v1/forecast/<zone>?tab=60m` | GET | Forecast path with 90% intervals (tabs 15m..72h) |
| `/api/v1/explain/<zone>?horizon=60m` | GET | Exact group-Shapley drivers of one forecast |
| `/api/v1/events`, `/events/<id>`, `/events/<id>/impact` | GET | Events, audit timeline, verified impact |
| `/api/v1/events/<id>/approve`, `/start`, `/verify`, `/escalate` | POST | Event lifecycle (409 if out of order) |
| `/api/v1/dustwise/priorities`, `/actions/<zone>`, `/allocation` | GET | IPS, action comparison, fleet plan |
| `/api/v1/twin/simulate`, `/twin/compare` | POST | Digital twin (422 names any out-of-range field) |
| `/api/v1/ingest` | POST | Validate and calibrate one sensor packet |
| `/api/v1/export/telemetry.csv`, `/export/report.json` | GET | Downloads |
| `/api/v1/chat` | POST | EcoNexus Copilot: `{"message": "...", "zone": "B", "view": "forecast"}` |
| `/api/v1/chat-status` | GET | Whether the Groq LLM is configured and reachable |
| `/api/v1/demo/reset` | POST | Back to 14:30, EV-042 pending approval |

Errors never leak a Python traceback: tracebacks are logged in the server console
(`server.log_and_hide`), clients get a clean message.

## EcoNexus Copilot

1. **Scope filter** -- off-topic questions (homework, jokes, news...) get a short decline before any
   LLM call.
2. **Built-in engine** -- each question is routed to one of 20 intents (briefing, what to do, why,
   forecast, priority, verification, sensors, model accuracy, weather, exposure, fleet...) and
   answered from the live platform state: the same numbers as the screens, never invented ones.
3. **Groq LLM (optional)** -- with `GROQ_API_KEY` set, Groq rephrases and expands the built-in
   engine's verified answer, constrained to the numbers in a LIVE CONTEXT block. If Groq is down,
   the built-in answer is returned and labelled.
4. **Offline** -- the same answers are pre-computed into `static/data/snapshot.js`, so the
   assistant works even without a server.

## Configuration

All configuration lives in `config.py` and is read from environment variables (via `.env`, see
`.env.example`). `.env` is parsed by a small built-in reader (no python-dotenv needed) that also
copes with the UTF-16/BOM files some Windows editors write.

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | *(empty)* | Optional: Copilot LLM mode (built-in engine without it) |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model name |
| `GROQ_URL` | Groq chat completions URL | Override the endpoint |
| `GROQ_TIMEOUT_S` | `20` | Seconds before falling back to the built-in engine |
| `GOOGLE_MAPS_API_KEY` | *(empty)* | Browser Maps key served to the dashboard (else `static/js/config.js`) |
| `ECONEXUS_HOST` | `127.0.0.1` | Flask bind host |
| `ECONEXUS_PORT` | `8000` | Flask bind port |
| `ECONEXUS_DEBUG` | `false` | Flask debug mode -- keep `false` in production |
| `ECONEXUS_OPEN_BROWSER` | `true` | Open the dashboard on start |
| `ECONEXUS_CORS_ORIGINS` | `null`, localhost:8000/8080/5500 | Other pages allowed to call the API |

Decision rules (action levels Watch 100 / Elevated 150 / High 200 / Critical 250 µg/m³ and the IPS
weights) are constants in `config.py`.

**Do not commit `.env`.** It is in `.gitignore`. Only `.env.example` (no real values) should ever be
committed. If a real key has ever been in a committed file, revoke it and create a new one.

## The map (Google Maps)

Every map is a real, zoomable map (Leaflet) with the platform's objects drawn as map features at
their positions — not a picture laid over it:

* **zones** coloured by PM10 level, **current hotspots** (pulsing rings), **predicted hotspots**
  (dashed rings with the forecast level and time), **27 sensors** coloured by their reading,
  **field units** with their routes, **sensitive sites**, **construction activity**, the
  **downwind exposure corridor** and, on the Digital Twin, the simulated **PM10 surface**.
* **Click anything** — a zone, hotspot, sensor, unit or site — and a panel opens inside the map with
  everything known about that place (readings, forecast, sources, recommended action, links to the
  right dashboard screen). There is no location drop-down.
* The **layers button** switches basemap (Map, Satellite, Site plan) and turns layers on and off.

Basemap: **Google Maps** (official Maps JavaScript API via Leaflet.GoogleMutant) when a key works;
otherwise **OpenStreetMap** (CARTO light tiles) or **Esri World Imagery** for satellite; with no
internet the site plan is drawn instead and all layers keep working. **Settings → Map** shows which
provider is in use and why. Set the key in `static/js/config.js` (needed for a page opened from
disk) or as `GOOGLE_MAPS_API_KEY` in `.env`. In Google Cloud Console restrict it to the *Maps
JavaScript API* and to `http://127.0.0.1:8000/*` and `http://localhost:8000/*`. Map data ©
OpenStreetMap contributors, © CARTO; imagery © Esri — the attribution is shown on every map.

## Using real site data

Provide a long-format CSV with `timestamp, zone_id, pm10, pm25, temperature, relative_humidity,
wind_speed, wind_dir_from` (optional: `mixing_height, rain_mm, reference_pm10,
construction_intensity, planned_construction, truck_trips, vehicles, intervention_active`), describe
your site in `site_model.py`, then:

```bash
.venv/bin/python train.py --data your_site.csv     # Windows: .venv\Scripts\python train.py --data ...
```

## Testing

```bash
python server.py --test        # or, inside the environment: python -m pytest
```

The suite covers data generation, leakage, models, attribution, DUSTWISE, twin Python/JavaScript
parity, verification/learning, the `.env` loader, the Flask API lifecycle and validation, and the
Copilot (scope, intents, grounded numbers, Groq fallback).

## Frontend

Plain HTML/CSS/JavaScript with no build step and no CDN: charts, the twin, the Copilot and the
command palette are in `static/js`; Leaflet, GoogleMutant and the IBM Plex Sans font are bundled in
`static/vendor` and `static/fonts`, so the app works offline and behind strict networks. Map tiles
(Google, OpenStreetMap/CARTO, Esri) are the only external resources, and the map falls back to the
site plan without them. Light theme; body text 15 px; all numbers use tabular figures.

## Troubleshooting

| Symptom | What to do |
|---|---|
| `XGBoostError: input stream corrupted`, or "the trained models can't be used" | The model file in `artifacts/` is damaged, half-downloaded, or was written by a different XGBoost install. `python server.py` now checks the file against `artifacts/manifest.json` and retrains automatically (~4 min); you can also force it with `python server.py --retrain`. |
| VS Code stops on an exception during setup | Run `python server.py` from a terminal, or untick **Raised Exceptions** in the debugger's breakpoint list. Setup handles these errors itself. |
| Project folder is inside OneDrive | OneDrive syncs `.venv` (hundreds of MB) and can lock or half-download large files. Move the folder outside OneDrive (for example `C:\Projects\econexus_ai`) and run again. |
| Port 8000 is busy | The next free port is used automatically, or pass `--port 9000`. |
| Libraries fail to install | Check the internet connection (set `HTTPS_PROXY` behind a proxy) and use Python 3.10-3.13; then run `python server.py` again. |
| "Google Maps unavailable — using OpenStreetMap" | The map still works on OpenStreetMap. For Google: enable the *Maps JavaScript API* and billing for the key, and allow `http://127.0.0.1:8000/*` in its website restrictions. **Settings → Map** shows the exact reason. |
| The map shows a plain site plan with no streets | Map tiles could not be downloaded (no internet or blocked). Every layer and click still works; the streets return when tiles load. |
| The page looks unstyled or old after updating | Hard-refresh the browser (Ctrl+F5) so it loads the new CSS and scripts. |

Models are saved in XGBoost's own model format (not a pickled Python object), written atomically,
and recorded with a SHA-256 checksum in `artifacts/manifest.json`, so a damaged file is reported as
damaged and repaired by retraining instead of crashing the server.

## Known limitations

- The data is synthetic; accuracy on a real site is unknown until it is trained on real telemetry.
- 24-72 h forecasts are an outlook (R² near zero at 72 h, coverage below 90%).
- The demo build actions only the hero event EV-042 through the API lifecycle.
- The Copilot's built-in engine answers the 20 question types it knows; free-form phrasing beyond
  those needs the optional Groq key. Voice input depends on the browser (Chrome/Edge).
- Retraining produces slightly different numbers with other library versions; the shipped
  artifacts were trained with the versions pinned in `requirements.txt`.

## License / Use

Research and educational use only. IBM Plex Sans font: SIL Open Font License 1.1. Icons: Lucide (ISC). Leaflet: BSD-2-Clause. Leaflet.GoogleMutant: MIT. Map data © OpenStreetMap contributors (ODbL), tiles © CARTO; imagery © Esri; Google Maps under the Google Maps Platform terms.
#   E C O N E X A I  
 