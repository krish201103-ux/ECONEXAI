#!/usr/bin/env python3
"""
EcoNexus AI - Flask Server
Live XGBoost forecasts | DUSTWISE decisions | Digital twin | EcoNexus Copilot
Run: python server.py  ->  http://127.0.0.1:8000
     (the first run creates .venv, installs requirements.txt and trains the models -- see bootstrap.py)

MODULE MAP (see README.md for the full picture):
  config.py             Environment-driven settings (.env: host/port, API keys) + decision rules
  bootstrap.py          First run: Python check, .venv, install libraries, train models if missing
  site_model.py         Green Valley Township: zones, 27 sensors, roads, receptors, fleet, routing
  simulator.py          Physics-motivated synthetic telemetry (the stand-in for real site data)
  data_cleaning.py      QA/QC: hygroscopic correction, spike rejection, bounded gap filling
  data_loader.py        Builds the observed dataset every model consumes (or loads a real CSV)
  ml_features.py        Feature engineering, forecast horizons, metrics, walk-forward splits
  ml_models.py          MultiHorizonForecaster (XGBoost + conformal intervals) and the benchmark
  explainability.py     Exact group-Shapley forecast drivers + NNLS source attribution
  intelligence.py       Action levels, pollution event detection, sensor fleet health
  dustwise.py           DUSTWISE: Intervention Priority Score, action comparison, fleet allocation
  digital_twin.py       What-if dispersion simulator (mirrored in static/js/twin.js)
  verification.py       Did it work? Counterfactual verification + Bayesian learning loop
  pipeline.py           Assembles the complete platform state from all of the above
  forecast_service.py   Orchestration: shared state, event lifecycle, request validation, exports
  chatbot.py            EcoNexus Copilot: scope filter, grounded local engine, Groq system prompt
  train.py              Trains and evaluates the models, writes artifacts/ and the offline snapshot
  server.py (this file) Flask routes only -- parses requests, calls the modules above,
                         and turns their results/exceptions into HTTP responses.
"""
import json
import logging
import os
import sys
import threading
import time
import traceback
import warnings
import webbrowser

import config

# -- FIRST-RUN SETUP -------------------------------------------------------------------------------
# Runs before any third-party import: bootstrap.py uses only the standard library, so a fresh Python
# can start this file. It installs the libraries into .venv (once), trains the models if they are
# missing, and re-runs this file inside .venv when needed. Importing server.py (tests, WSGI) skips it.
OPTIONS = None
if __name__ == '__main__':
    import bootstrap
    OPTIONS = bootstrap.ensure()

warnings.filterwarnings('ignore')

from flask import Flask, Response, jsonify, request, send_from_directory  # noqa: E402
from werkzeug.exceptions import HTTPException  # noqa: E402

import chatbot  # noqa: E402
from data_cleaning import EcoNexusBaseException  # noqa: E402
from forecast_service import PlatformService, ServiceError  # noqa: E402

# -- APP SETUP -------------------------------------------------------------------------------------
app = Flask(__name__, static_folder=str(config.STATIC_DIR), static_url_path='')
app.config['MAX_CONTENT_LENGTH'] = config.MAX_UPLOAD_MB * 1024 * 1024
app.json.sort_keys = False      # keep the platform's field order (zones A-D, horizons 15m..72h)
app.json.ensure_ascii = False   # send µg/m³ as-is

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logging.getLogger('werkzeug').setLevel(logging.WARNING)   # no line per request in the console
log = app.logger

service = PlatformService()     # loads the trained models on first use (~30 s); see main block


# -- ERROR HANDLING --------------------------------------------------------------------------------
# Unexpected errors never leak a Python traceback (file paths, internals) to the client. Tracebacks
# are logged server-side only; clients get a clean, safe message.
def log_and_hide(e, where):
    log.error("Unhandled error in %s: %s\n%s", where, e, traceback.format_exc())


def _is_api():
    return request.path.startswith('/api/')


@app.errorhandler(ServiceError)
def service_error(e):
    body = {'error': str(e)}
    if e.details:
        body['details'] = e.details
    return jsonify(body), e.status


@app.errorhandler(EcoNexusBaseException)
def domain_error(e):
    return jsonify({'error': e.message, 'type': e.__class__.__name__, 'details': e.details}), 400


@app.errorhandler(404)
def not_found(e):
    if _is_api():
        return jsonify({'error': 'Not found', 'hint': 'GET /api/v1 lists every endpoint'}), 404
    return send_from_directory(config.STATIC_DIR, 'index.html'), 404


@app.errorhandler(405)
def method_not_allowed(e):
    if _is_api():
        return jsonify({'error': f'{request.method} is not allowed on {request.path}'}), 405
    return e


@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': f'Request too large -- max {config.MAX_UPLOAD_MB} MB'}), 413


@app.errorhandler(Exception)
def handle_unexpected(e):
    """Last-resort catch-all so a truly unexpected error never surfaces as a raw Werkzeug page or
       an empty 500."""
    if isinstance(e, HTTPException):
        return e
    log_and_hide(e, request.path)
    if _is_api():
        return jsonify({'error': 'Internal server error. Please try again.'}), 500
    return Response('EcoNexus AI hit an unexpected error -- see the server console.', status=500)


# -- CORS ------------------------------------------------------------------------------------------
# Lets a dashboard opened from disk (Origin: null) or another local port call this API.
@app.after_request
def add_cors(resp):
    origin = request.headers.get('Origin')
    if origin and (origin in config.CORS_ORIGINS or '*' in config.CORS_ORIGINS):
        resp.headers['Access-Control-Allow-Origin'] = origin
        resp.headers['Vary'] = 'Origin'
        resp.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        resp.headers['Access-Control-Max-Age'] = '600'
    return resp


def _body():
    return request.get_json(silent=True)


# -- PAGE ROUTES -----------------------------------------------------------------------------------
# The UI is one page with three sections -- Home, Dashboard and About; these addresses open
# the page on the matching section (the page reads the path and switches to #/home, #/command
# or #/about).
@app.route('/')
@app.route('/home')
@app.route('/dashboard')
@app.route('/about')
def dashboard():
    return send_from_directory(config.STATIC_DIR, 'index.html')


# -- PLATFORM API ----------------------------------------------------------------------------------
@app.route('/api/v1')
def api_index():
    """Every endpoint, generated from the routing table (handy with curl or a browser)."""
    rows = sorted({(r.rule, ', '.join(sorted(r.methods - {'HEAD', 'OPTIONS'})))
                   for r in app.url_map.iter_rules() if r.rule.startswith('/api/')})
    return jsonify({'platform': config.APP_NAME, 'version': config.VERSION,
                    'endpoints': [{'path': p, 'methods': m} for p, m in rows]})


@app.route('/api/v1/health')
def api_health():
    return jsonify(service.health())


@app.route('/api/v1/client-config')
def api_client_config():
    """Browser-side settings the dashboard cannot read from .env itself."""
    return jsonify({'google_maps_api_key': config.GOOGLE_MAPS_API_KEY, 'version': config.VERSION,
                    'copilot': {'name': chatbot.COPILOT_NAME, 'mode': 'groq' if config.GROQ_ENABLED else 'local',
                                'model': config.GROQ_MODEL if config.GROQ_ENABLED else 'built-in engine'}})


@app.route('/api/v1/snapshot')
def api_snapshot():
    return jsonify(service.snapshot())


@app.route('/api/v1/status')
def api_status():
    st = service.section()
    return jsonify({'meta': st['meta'], 'status_strip': st['status_strip'], 'weather': st['weather'],
                    'lifecycle': st['lifecycle'], 'ai_status': st['ai_status']})


@app.route('/api/v1/briefing')
def api_briefing():
    return jsonify(service.briefing())


@app.route('/api/v1/zones')
def api_zones():
    return jsonify(service.section('zones'))


@app.route('/api/v1/zones/<zone_id>')
def api_zone(zone_id):
    return jsonify(service.zone(zone_id))


@app.route('/api/v1/forecast/<zone_id>')
def api_forecast(zone_id):
    return jsonify(service.forecast(zone_id, request.args.get('tab', '60m')))


@app.route('/api/v1/explain/<zone_id>')
def api_explain(zone_id):
    return jsonify(service.explain(zone_id, request.args.get('horizon', '60m')))


@app.route('/api/v1/attribution/<zone_id>')
def api_attribution(zone_id):
    return jsonify(service.zone(zone_id)['attribution'])


@app.route('/api/v1/events')
def api_events():
    ev = service.section('events')
    return jsonify({'active': ev['active'], 'history': ev['history'], 'counts': ev['counts']})


@app.route('/api/v1/events/<event_id>')
def api_event(event_id):
    return jsonify(service.event(event_id))


@app.route('/api/v1/events/<event_id>/impact')
def api_impact(event_id):
    return jsonify(service.impact(event_id))


@app.route('/api/v1/events/<event_id>/approve', methods=['POST'])
def api_approve(event_id):
    return jsonify(service.approve(event_id, _body()))


@app.route('/api/v1/events/<event_id>/start', methods=['POST'])
def api_start(event_id):
    return jsonify(service.start(event_id))


@app.route('/api/v1/events/<event_id>/verify', methods=['POST'])
def api_verify(event_id):
    return jsonify(service.verify(event_id))


@app.route('/api/v1/events/<event_id>/escalate', methods=['POST'])
def api_escalate(event_id):
    return jsonify(service.escalate(event_id, _body()))


@app.route('/api/v1/dustwise/priorities')
def api_priorities():
    return jsonify(service.section('dustwise', 'priorities'))


@app.route('/api/v1/dustwise/actions/<zone_id>')
def api_actions(zone_id):
    return jsonify(service.actions(zone_id))


@app.route('/api/v1/dustwise/allocation')
def api_allocation():
    dw = service.section('dustwise')
    return jsonify({'allocation': dw['allocation'], 'fleet': dw['fleet']})


@app.route('/api/v1/sensors')
def api_sensors():
    return jsonify(service.section('sensors'))


@app.route('/api/v1/learning')
def api_learning():
    return jsonify(service.section('learning'))


@app.route('/api/v1/model')
def api_model():
    return jsonify(service.model())


@app.route('/api/v1/sustainability')
def api_sustainability():
    return jsonify(service.sustainability())


@app.route('/api/v1/twin/simulate', methods=['POST'])
def api_twin_simulate():
    return jsonify(service.twin_simulate(_body()))


@app.route('/api/v1/twin/compare', methods=['POST'])
def api_twin_compare():
    return jsonify(service.twin_compare(_body()))


@app.route('/api/v1/ingest', methods=['POST'])
def api_ingest():
    return jsonify(service.ingest(_body()))


@app.route('/api/v1/demo/reset', methods=['POST'])
def api_reset():
    return jsonify(service.reset_demo())


@app.route('/api/v1/export/telemetry.csv')
def api_export_csv():
    return Response(service.export_csv(), mimetype='text/csv',
                    headers={'Content-Disposition': 'attachment; filename=econexus_telemetry_24h.csv'})


@app.route('/api/v1/export/report.json')
def api_export_report():
    return Response(json.dumps(service.report(), indent=2, ensure_ascii=False, default=str),
                    mimetype='application/json',
                    headers={'Content-Disposition': 'attachment; filename=econexus_report.json'})


# -- ECONEXUS COPILOT ------------------------------------------------------------------------------
@app.route('/api/v1/chat', methods=['POST'])
def api_chat():
    body = _body() or {}
    messages = body.get('messages') if isinstance(body.get('messages'), list) else []
    question = str(body.get('message') or (messages[-1].get('content', '') if messages else '')).strip()
    if not question:
        return jsonify({'error': 'Ask a question in "message" (or a "messages" chat history)'}), 400
    question = question[:1000]
    if not messages:
        messages = [{'role': 'user', 'content': question}]
    zone = str(body.get('zone') or '').upper()[:1] or None
    view = str(body.get('view') or '')[:30] or None

    # Fast, free, consistent pre-filter for clearly off-topic questions -- skips the LLM entirely.
    if not chatbot.is_on_topic(question):
        return jsonify({'answer': chatbot.off_topic_reply(), 'source': 'scope_filter', 'model': 'scope_filter',
                        'intent': 'off_topic', 'links': [],
                        'followups': ['Give me a situation briefing', 'What should I do now?']})

    ctx = service.context()
    local = chatbot.local_answer(question, ctx, zone=zone)
    if not config.GROQ_ENABLED or body.get('mode') == 'local':
        return jsonify({**local, 'source': 'local', 'model': 'econexus-local'})
    try:
        answer, model, tokens = chatbot.call_groq(messages, ctx, local['answer'], view)
        return jsonify({**local, 'answer': answer, 'source': 'groq', 'model': model, 'tokens': tokens})
    except Exception as e:
        log.warning("Groq call failed, answering with the built-in engine: %s", e)
        return jsonify({**local, 'source': 'local', 'model': 'econexus-local',
                        'note': 'Groq is unreachable right now — answered by the built-in engine.'})


@app.route('/api/v1/chat-status')
def api_chat_status():
    if not config.GROQ_ENABLED:
        return jsonify({'status': 'not_configured', 'mode': 'local',
                        'message': 'GROQ_API_KEY not set — the Copilot answers with its built-in engine '
                                   '(live platform data, no internet needed). See .env.example.'})
    try:
        code = chatbot.groq_ping()
        if code == 200:
            return jsonify({'status': 'connected', 'mode': 'groq', 'model': config.GROQ_MODEL})
        return jsonify({'status': 'error', 'mode': 'local', 'code': code,
                        'message': 'Groq rejected the request (check GROQ_API_KEY / GROQ_MODEL).'})
    except Exception as e:
        return jsonify({'status': 'unreachable', 'mode': 'local', 'error': str(e)[:200]})


# -- MAIN ------------------------------------------------------------------------------------------
if __name__ == '__main__':
    import bootstrap  # already imported above; used here for the port and address helpers

    host = OPTIONS.host
    port = bootstrap.pick_port(host, OPTIONS.port)
    print('\nLoading the trained models and building the site state (about 30 s) ...', flush=True)
    t0 = time.time()
    try:
        service.ensure()
    except Exception as e:  # models unreadable etc. -- explain instead of a bare traceback
        log_and_hide(e, 'startup')
        print(f'\nERROR  The platform could not start: {str(e).strip().splitlines()[0][:300]}')
        print('       Fix: python server.py --retrain\n')
        bootstrap.pause_if_double_clicked(1)
        sys.exit(1)
    local_url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{port}"
    backend = service.platform.forecaster.meta.get('backend', 'n/a')
    copilot = f'Groq {config.GROQ_MODEL}' if config.GROQ_ENABLED else 'built-in engine (no Groq key)'
    maps = ('Google (.env key)' if config.GOOGLE_MAPS_API_KEY else 'Google (config.js key)') + ', OpenStreetMap fallback'
    print('=' * 66)
    print(f'  {config.APP_NAME} {config.VERSION} - Environmental Intelligence Platform')
    print(f'  Open        {local_url}')
    print(f'  Sections    Home /  |  Dashboard /dashboard  |  About /about')
    if host == '0.0.0.0':
        print(f'  On Wi-Fi    http://{bootstrap.lan_ip()}:{port}')
    print(f'  API         {local_url}/api/v1')
    print(f'  Models      {backend}  |  ready in {time.time() - t0:.0f} s')
    print(f'  Copilot     {copilot}')
    print(f'  Map         {maps}')
    print(f'  Data        synthetic demo data (Green Valley Township)')
    print('=' * 66)
    print('  Press Ctrl+C to stop.\n', flush=True)
    if config.OPEN_BROWSER and not OPTIONS.no_browser and not os.environ.get('ECONEXUS_NO_BROWSER'):
        threading.Timer(1.2, lambda: webbrowser.open(local_url)).start()
    try:
        app.run(host=host, port=port, debug=OPTIONS.debug or config.DEBUG, threaded=True, use_reloader=False,
                load_dotenv=False)   # config.py already read .env
    except KeyboardInterrupt:
        pass
    print('\nEcoNexus AI stopped.')
