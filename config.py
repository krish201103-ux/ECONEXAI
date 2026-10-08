"""
config.py -- Centralized configuration for EcoNexus AI.

Every value that used to be hard-coded or spread across config/config.yaml
(host, port, CORS, API keys, action levels, IPS weights, file locations) now
lives here and, where it makes sense, is read from environment variables with
safe defaults for local development.

Setup:
  1. Copy .env.example to .env
  2. Put your keys in .env (Groq for the Copilot's LLM mode, Google Maps for the
     live basemap) -- both are optional, everything else works without them
  3. .env is in .gitignore -- it will never be committed

server.py and the other modules read everything through this module; nothing
downstream should read os.environ directly.

NOTE: this reads .env with a small built-in parser (_load_env_file below)
instead of the third-party python-dotenv package, and needs nothing outside the
standard library. That is deliberate: bootstrap.py imports this module *before*
any library is installed (to pick up ECONEXUS_PORT etc. from .env), and some
locked-down machines block `pip install` entirely, so every dependency we avoid
is one less way for the project to fail to start.
"""
import os
from pathlib import Path


def _read_env_text(path):
    """Read a .env file's text, tolerant of the encodings Windows tools commonly
       produce. PowerShell's `>` redirect writes UTF-16LE with a byte-order mark
       and Notepad's "UTF-8" option adds a UTF-8 BOM; a strict utf-8 read of
       either silently yields text that never matches KEY=VALUE, which looks
       like "the key just isn't loading". Tries UTF-8 (with/without BOM) first,
       then the UTF-16 variants."""
    raw = path.read_bytes()
    for encoding in ('utf-8-sig', 'utf-8', 'utf-16', 'utf-16-le', 'utf-16-be'):
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    # Last resort: load the lines that DO parse rather than silently nothing.
    return raw.decode('utf-8', errors='replace')


def _load_env_file(path):
    """Minimal stand-in for python-dotenv's load_dotenv(): reads KEY=VALUE lines
       into os.environ. Skips blank lines and # comments, strips one layer of
       matching quotes and an optional leading `export `. Never overwrites a
       variable that is already set in the real environment (e.g. by the OS or
       a hosting platform) -- .env is only a local-development fallback."""
    if not path.exists():
        return
    for line in _read_env_text(path).splitlines():
        line = line.strip().lstrip('﻿')
        if not line or line.startswith('#') or '=' not in line:
            continue
        if line.startswith('export '):
            line = line[len('export '):]
        key, _, value = line.partition('=')
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        elif ' #' in value:                      # inline comment: KEY=value  # note
            value = value.split(' #', 1)[0].rstrip()
        if key:
            os.environ.setdefault(key, value)


BASE_DIR = Path(__file__).resolve().parent
_load_env_file(BASE_DIR / '.env')


def _get_bool(name, default):
    val = os.environ.get(name)
    if val is None or not val.strip():
        return default
    return val.strip().lower() in ('1', 'true', 'yes', 'on')


def _get_int(name, default):
    try:
        return int(os.environ.get(name, '').strip() or default)
    except ValueError:
        print(f"[config] WARNING: {name} must be a whole number -- using {default}.")
        return default


def _get_list(name, default):
    val = os.environ.get(name, '').strip()
    return [v.strip() for v in val.split(',') if v.strip()] if val else list(default)


# -- Identity ------------------------------------------------------------------------------------
APP_NAME = 'EcoNexus AI'
VERSION = '4.1.0'
RANDOM_SEED = 42

# -- Files ---------------------------------------------------------------------------------------
STATIC_DIR = Path(os.environ.get('ECONEXUS_STATIC_DIR', str(BASE_DIR / 'static'))).resolve()
ARTIFACTS_DIR = BASE_DIR / 'artifacts'                 # trained models + computed metrics
SNAPSHOT_JS = STATIC_DIR / 'data' / 'snapshot.js'      # offline copy of the model output for the UI

# -- Flask server --------------------------------------------------------------------------------
HOST = os.environ.get('ECONEXUS_HOST', '127.0.0.1').strip() or '127.0.0.1'
PORT = _get_int('ECONEXUS_PORT', 8000)
DEBUG = _get_bool('ECONEXUS_DEBUG', False)
OPEN_BROWSER = _get_bool('ECONEXUS_OPEN_BROWSER', True)
MAX_UPLOAD_MB = _get_int('ECONEXUS_MAX_UPLOAD_MB', 16)

# Browsers allowed to call the API from another address. The UI served by this server needs none
# of these (same origin). "null" is what a page opened straight from disk (file://) sends, so
# static/index.html can talk to a server you started separately. The server listens on 127.0.0.1
# by default, so other computers cannot reach it anyway.
CORS_ORIGINS = _get_list('ECONEXUS_CORS_ORIGINS', [
    'null',
    'http://localhost:8000', 'http://127.0.0.1:8000',
    'http://localhost:8080', 'http://127.0.0.1:8080',
    'http://localhost:5500', 'http://127.0.0.1:5500',
])

# -- Google Maps (browser basemap) ---------------------------------------------------------------
# Handed to the dashboard via /api/v1/client-config when static/js/config.js does not set a key.
# A browser Maps key is public by design: restrict it to the Maps JavaScript API and your site's
# addresses (HTTP referrers) in Google Cloud Console.
GOOGLE_MAPS_API_KEY = os.environ.get('GOOGLE_MAPS_API_KEY', '').strip()

# -- EcoNexus Copilot (optional Groq LLM) --------------------------------------------------------
# Without a key the Copilot still answers from the live platform state with its built-in engine
# (chatbot.local_answer); the key only adds free-form LLM phrasing on top of the same numbers.
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '').strip()
GROQ_URL = os.environ.get('GROQ_URL', 'https://api.groq.com/openai/v1/chat/completions').strip()
GROQ_MODEL = os.environ.get('GROQ_MODEL', 'llama-3.1-8b-instant').strip()
GROQ_TIMEOUT_S = _get_int('GROQ_TIMEOUT_S', 20)
GROQ_ENABLED = bool(GROQ_API_KEY)

# -- Site decision rules -------------------------------------------------------------------------
# Site dust action levels for zone-average PM10 (ug/m3). These are site-management trigger levels,
# not the national ambient standard (CPCB 24-h PM10 = 100 ug/m3, reported separately).
ACTION_LEVEL_THRESHOLDS = {'watch': 100, 'elevated': 150, 'high': 200, 'critical': 250}

# DUSTWISE Intervention Priority Score: IPS = sum(weight x factor), factors on 0-100.
IPS_WEIGHTS = {
    'severity': 0.25,           # S_poll  current severity
    'predicted_growth': 0.20,   # P_pred  predicted growth (90-min forecast, conformal upper band)
    'source_confidence': 0.15,  # C_src   attribution confidence x controllable share
    'exposure': 0.15,           # E_exp   wind-weighted receptors + population
    'persistence': 0.10,        # P_pers  forecast ventilation over the next 90 min
    'feasibility': 0.15,        # F_feas  best resource ETA, availability, water
}
OPERATOR_APPROVAL_REQUIRED = True  # DUSTWISE recommends; a person approves every dispatch

assert abs(sum(IPS_WEIGHTS.values()) - 1.0) < 1e-9, 'IPS weights must sum to 1'
