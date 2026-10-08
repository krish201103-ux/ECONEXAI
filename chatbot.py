"""
chatbot.py -- EcoNexus Copilot: scope filter, grounded local engine, situation briefing, Groq LLM.

server.py's /api/chat route calls is_on_topic() / local_answer() / call_groq() from here rather
than embedding prompt text and keyword lists among the Flask route handlers.

How an answer is produced:

  1. Scope filter (is_on_topic). Clearly off-topic questions get a short, friendly decline
     instantly -- no LLM call, so refusals are free and consistent even offline.
  2. Local engine (local_answer). Every on-topic question is classified into an intent
     (briefing, recommendation, why, forecast, priority, verification, sensors, model accuracy,
     weather, sustainability, twin, fleet, exposure, methodology ...) and answered from the
     *live platform state* -- the same numbers the dashboard shows, never invented ones. It also
     returns deep links into the dashboard ("Open DUSTWISE") and follow-up suggestions. This
     engine needs no API key and no internet.
  3. Groq (call_groq), only when GROQ_API_KEY is set. The LLM gets a compact LIVE CONTEXT block
     plus the local engine's verified answer, and is told to use only those numbers; it adds
     natural phrasing and handles free-form questions. If Groq is unreachable the local answer
     is returned instead, so the Copilot never goes silent.

offline_pack() pre-computes the local engine's answers (and exports its intent patterns) into
static/data/snapshot.js, so the Copilot also works when the dashboard is opened from disk with
no server running.
"""
import itertools
import json
import re
from decimal import ROUND_HALF_UP, Decimal

import config

COPILOT_NAME = 'EcoNexus Copilot'
UGM3 = 'µg/m³'   # µg/m³

GROQ_SYSTEM = f"""You are {COPILOT_NAME}, the operations assistant built into EcoNexus AI, an environmental
command centre for construction sites. You ONLY discuss EcoNexus AI and the site it monitors.

PLATFORM KNOWLEDGE:
- Site: Green Valley Township, a fictional 100-acre township (zones A-D, 27 sensors). Zone A Tower Construction,
  Zone B Excavation & Haul Road, Zone C Residential Block & Central Park, Zone D School & Clinic Perimeter.
- DATA IS SYNTHETIC: produced by a physics-motivated simulator, not field measurements. Say so if asked about data.
- Workflow: OBSERVE -> UNDERSTAND -> PREDICT -> DIAGNOSE -> PRIORITIZE -> OPTIMIZE -> ACT -> VERIFY -> LEARN.
- Action levels (zone PM10, {UGM3}): Watch 100, Elevated 150, High 200, Critical 250. CPCB 24-h PM10 standard is 100.
- Forecasts: XGBoost direct multi-horizon models (18 horizons, 15 min - 72 h, PM10 and PM2.5) with split-conformal
  90% intervals. 15-90 min forecasts are "no-new-action" baselines. Beyond 24 h they are an outlook (wide intervals).
- Explainability: exact group-Shapley forecast drivers (5 groups); NNLS receptor model for "AI-estimated
  contributors" (construction, road dust, traffic, other) with bootstrap ranges. These are estimates.
- DUSTWISE: Intervention Priority Score IPS = 0.25 severity + 0.20 predicted growth + 0.15 source confidence
  + 0.15 exposure + 0.10 persistence + 0.15 feasibility (0-100). Actions are compared by scenario benefit,
  water, cost and disruption; tankers/sweepers are assigned with the Hungarian algorithm on the road network.
  Every recommendation requires operator approval.
- Verification: observed before/after change is reported separately from the model-estimated effect versus the
  no-action counterfactual forecast. "Verified" only when the observation is below the counterfactual's 90% band.
- Learning: a Bayesian field multiplier per action type (prior N(1, 0.35^2)) is updated by every verified event.
- Digital twin outputs are "Simulation / scenario estimate - not field measurement".
- Dashboard screens: Command Center, Environmental Map, Pollution Events, Forecast & Intelligence, DUSTWISE
  Optimizer, Resource Operations, Digital Environmental Twin, Impact & Verification, Sensors & Data,
  Sustainability, Settings.

GROUNDING RULE -- most important:
Use ONLY numbers that appear in LIVE CONTEXT or VERIFIED ANSWER below. Never invent readings, forecasts,
percentages or times. If a number is not there, say you don't have it and point to the screen that shows it.
Label estimates as estimates and simulations as simulations.

STRICT SCOPE RULE:
Only answer questions about EcoNexus AI, this site, its air quality, forecasts, sources, interventions, fleet,
sensors, models and methodology. General air-quality questions are fine only when they help interpret this
platform's output (e.g. "what is PM2.5"). For anything unrelated (chit-chat, coding help, news, homework,
medical, legal or financial advice) reply in 1-2 friendly sentences that you are scoped to EcoNexus and suggest
one relevant question instead. Never lecture.

STYLE: concise operations language, 2-5 sentences, or up to 5 short bullets for lists. Lead with the answer.
Use {UGM3} for concentrations. Do not use markdown headings."""

# -- DOMAIN RESTRICTION (heuristic pre-filter) --------------------------------------------------
# Catches clearly off-topic questions BEFORE any LLM call, so refusals are instant and free. The
# LLM's own scope instructions are a second line of defence for borderline cases.
ON_TOPIC_KEYWORDS = [
    'econexus', 'copilot', 'green valley', 'township', 'site', 'zone', 'dust', 'pm10', 'pm 10', 'pm2.5',
    'pm25', 'pm 2.5', 'particulate', 'pollut', 'air quality', 'aqi', 'emission', 'sensor', 'node', 'forecast',
    'predict', 'outlook', 'peak', 'trend', 'dustwise', 'ips', 'priorit', 'intervention', 'sprinkl', 'mist',
    'tanker', 'sweeper', 'water', 'fleet', 'dispatch', 'approve', 'approval', 'verif', 'impact',
    'counterfactual', 'learning', 'multiplier', 'xgboost', 'conformal', 'shapley', 'attribution', 'contribut',
    'source', 'construction', 'excavat', 'haul', 'traffic', 'truck', 'wind', 'humid', 'mixing', 'dispersion',
    'ventilat', 'weather', 'twin', 'what-if', 'what if', 'scenario', 'simulat', 'event', 'alert', 'hotspot',
    'exposure', 'school', 'clinic', 'daycare', 'resident', 'receptor', 'cpcb', 'complian', 'sustainab',
    'briefing', 'situation', 'dashboard', 'command center', 'map', 'benchmark', 'accuracy', 'rmse', 'r2',
    'synthetic', 'calibrat', 'battery', 'offline', 'rain', 'action level', 'elevated', 'critical', 'watch level',
    'ev-', 'ec-', 'workflow', 'recommend', 'report', 'export', 'csv', 'methodology', 'nnls', 'hungarian',
    'what should', 'what do i do', 'what can i do', 'what do we do', 'learn', 'next step', 'priority',
]
# Short or generic words that need WHOLE-WORD matching (as substrings they would false-positive).
ON_TOPIC_WHOLE_WORDS = [
    'status', 'now', 'why', 'next', 'model', 'models', 'risk', 'level', 'levels', 'help', 'unit', 'units',
    'data', 'action', 'actions', 'reading', 'readings', 'air', 'brief', 'summary', 'overview', 'today',
    'tonight', 'tomorrow', 'worse', 'better', 'safe', 'improve', 'happening', 'urgent', 'first',
]
OFF_TOPIC_SIGNALS = [
    'write me a', 'write a poem', 'write a story', 'write code', 'python code', 'javascript', 'translate',
    'recipe', 'stock price', 'stock market', 'crypto', 'bitcoin', 'movie', 'song', 'lyrics', 'tell me a joke',
    'who won', 'capital of', 'solve this', 'homework', 'essay', 'love advice', 'relationship advice',
    'girlfriend', 'boyfriend', 'medical diagnosis', 'prescribe', 'symptom of', 'legal advice', 'tax advice',
    'current president', 'prime minister', 'cricket score', 'football', 'horoscope',
]
GREETINGS = {'hi', 'hello', 'hey', 'hii', 'namaste', 'good morning', 'good afternoon', 'good evening',
             'thanks', 'thank you', 'ok', 'okay', 'cool', 'great', 'nice', 'got it', 'yes', 'no', 'bye'}


def is_on_topic(question):
    """Heuristic first-pass filter: does this question plausibly relate to EcoNexus? Errs toward
       YES for short or ambiguous questions (a false positive just means the local engine answers
       with a briefing; a false negative blocks a legitimate question, which is worse)."""
    q = (question or '').lower().strip()
    if len(q) < 2:
        return True
    if any(sig in q for sig in OFF_TOPIC_SIGNALS):
        return False
    if any(kw in q for kw in ON_TOPIC_KEYWORDS):
        return True
    if any(re.search(r'\b' + re.escape(w) + r'\b', q) for w in ON_TOPIC_WHOLE_WORDS):
        return True
    if q.rstrip('!.? ') in GREETINGS or re.search(r'\b(who are you|what can you do|how do (i|you) use)\b', q):
        return True
    return detect_zone(q) is not None


OFF_TOPIC_REPLIES = [
    "That's outside what I can help with here — I'm scoped to EcoNexus and Green Valley's air quality. "
    "Try: \"What should I do about Zone B right now?\"",
    "I'm built for EcoNexus operations questions, so I'll pass on that one. I can explain today's forecast, "
    "the DUSTWISE recommendation, or which sensors need attention.",
    "That falls outside my scope — I only cover this site's dust, forecasts, interventions and models. "
    "Ask me for a situation briefing to see what's happening right now.",
]
_off_topic_cycle = itertools.cycle(OFF_TOPIC_REPLIES)


def off_topic_reply():
    """Rotate through a few varied decline messages so repeated attempts don't get the exact
       same sentence every time."""
    return next(_off_topic_cycle)


# -- INTENTS --------------------------------------------------------------------------------------
# (intent, regex patterns, zone-aware). Patterns use only syntax that JavaScript also understands,
# because offline_pack() exports them for the dashboard's offline Copilot.
INTENTS = [
    ('greeting', [r'^(hi+|hello|hey|namaste|good (morning|afternoon|evening))\b', r'\bwho are you\b',
                  r'^(thanks|thank you|thx|ok|okay|cool|great|nice|got it)\b',
                  r'\bwhat can you do\b', r'^help\b', r'\bhow do (i|you) use\b'], False),
    ('method', [r'\bshapley\b', r'\bnnls\b', r'\bhungarian\b', r'\bdijkstra\b', r'\bsplit.?conformal\b',
                r'\breceptor model\b', r'\bmethodolog', r'\bhow does .* work\b', r'\balgorithm',
                r'\bhow does .*(attribution|shapley|conformal|interval|forecast\w*|fleet|routing) work',
                r'\bhow (is|are) .* (computed|calculated|estimated)\b'], False),
    ('briefing', [r'\bbrief', r'\bsummar', r'\boverview\b', r'\bstatus\b', r'\bsituation\b', r'\bsitrep\b',
                  r'happening', r'\bright now\b', r'\bupdate me\b', r'\bhow is the site\b', r'\bshift\b'], False),
    ('recommend', [r'\bwhat should\b', r'\bwhat (do|can) (i|we) do\b', r'\brecommend', r'\bnext step',
                   r'\bdispatch', r'\bapprov', r'\bwhich (unit|tanker|action)\b', r'\bdeploy', r'\brespond',
                   r'\bbest action\b', r'\bwhat action\b', r'\bfix\b', r'\breduce\b', r'\bmitigat',
                   r'\bwhy (this|that) action\b', r'\binstead of\b'], True),
    ('why', [r'\bwhy\b', r'\bcaus', r'\bsource', r'\bcontribut', r'\battribut', r'\bcoming from\b',
             r'\bresponsible\b', r'\bdriver', r'\brising\b', r'\bspike'], True),
    ('forecast', [r'\bforecast', r'\bpredict', r'\bnext (hour|90|60|30)', r'\bpeak\b', r'\bwill (it|pm|zone|the)\b',
                  r'\boutlook\b', r'\btonight\b', r'\btomorrow\b', r'\bgoing to\b', r'\bexpect', r'\bget worse\b',
                  r'\blater\b'], True),
    ('ips', [r'\bips\b', r'\bpriority score\b', r'\bpriorit', r'\brank', r'\bmost urgent\b', r'\bwhich zone first\b',
             r'\bworst zone\b', r'\bwhich zone\b'], True),
    ('verify', [r'\bverif', r'\bimpact\b', r'\bdid\b.*\bwork', r'\bworked\b', r'\beffect\b',
                r'\bcounterfactual', r'\bbefore and after\b', r'\boutcome\b', r'\bresult'], False),
    ('learning', [r'\blearn', r'\bmultiplier', r'\beffectiveness\b', r'\btrack record\b', r'\bfeedback loop\b'],
     False),
    ('sensors', [r'\bsensor', r'\bnodes?\b', r'\boffline\b', r'\bbattery\b', r'\bdrift', r'\bcalibrat',
                 r'\bdata quality\b', r'\bdata confidence\b', r'\bec-\d+', r'\bneed attention\b'], True),
    ('model', [r'\bmodels?\b', r'\baccura', r'\brmse\b', r'\bmae\b', r'\br2\b', r'\bbenchmark', r'\bxgboost\b',
               r'\breliab', r'\btrust', r'\bskill\b', r'\bcoverage\b', r'\bhow good\b', r'\buncertain',
               r'\b(reliab|accura|trust)\w*\b.*\bforecast', r'\bforecast\w*\b.*\b(reliab|accura|error)'], False),
    ('weather', [r'\bweather\b', r'\bwind', r'\bhumid', r'\btemperature\b', r'\bmixing\b', r'\bdispersion\b',
                 r'\bventilat', r'\brain', r'\bstagnant\b', r'\bmeteorolog'], False),
    ('sustainability', [r'\bwater\b', r'\bsustainab', r'\bcomplian', r'\bcpcb\b', r'\bperformance score\b',
                        r'\bgreen cover', r'\blong.?term\b', r'\bsav(e|ing)', r'\bmonth', r'\brecurring\b',
                        r'\brecurring hotspot'], False),
    ('twin', [r'\btwin\b', r'\bwhat.?if\b', r'\bscenario', r'\bsimulat'], False),
    ('events', [r'\bevents?\b', r'\balerts?\b', r'\bincident', r'\bev-\d+', r'\bhotspots?\b'], False),
    ('fleet', [r'\bfleet\b', r'\btankers?\b', r'\bsweepers?\b', r'\btrucks?\b', r'\bresources?\b',
               r'\bunits?\b', r'\bvehicles?\b', r'\bwhere is\b'], False),
    ('exposure', [r'\bschool', r'\bclinic', r'\bdaycare\b', r'\bchildren\b', r'\bkids\b', r'\bresidents?\b',
                  r'\bexpos', r'\breceptor', r'\bsenior', r'\bhealth\b', r'\bsafe\b'], False),
    ('data', [r'\bsynthetic\b', r'\breal data\b', r'\bsimulator\b', r'\bfake\b', r'\bmeasured\b',
              r'\bdata source', r'\bdataset\b', r'\bwhere does the data\b'], False),
    ('levels', [r'\baction levels?\b', r'\bthresholds?\b', r'\bwhat does (elevated|watch|high|critical) mean\b',
                r'\bwhat is pm', r'\bpm2\.?5\b.*\bpm10\b', r'\bcpcb standard\b'], False),
]
_COMPILED = [(name, [re.compile(p, re.I) for p in pats], zone_aware) for name, pats, zone_aware in INTENTS]
ZONE_AWARE = {name for name, _, za in INTENTS if za} | {'zone'}

# words that point at a zone without saying "zone X"
ZONE_ALIASES = {
    'A': ['tower', 'phase 2'],
    'B': ['excavation', 'haul road', 'haul-road', 'pit', 'phase 3'],
    'C': ['residential block', 'central park', 'park'],
    'D': ['school perimeter', 'clinic perimeter', 'perimeter'],
}


def detect_zone(question):
    q = (question or '').lower()
    m = re.search(r'\bzone\s*([abcd])\b', q)
    if m:
        return m.group(1).upper()
    for z, words in ZONE_ALIASES.items():
        if any(re.search(r'\b' + re.escape(w) + r'\b', q) for w in words):
            return z
    return None


def classify(question):
    """Returns (intent, zone). Scores every intent by matching patterns; ties go to the order of
       INTENTS. A bare zone mention with no other signal becomes the 'zone' intent."""
    q = (question or '').strip()
    zone = detect_zone(q)
    best, best_score = None, 0
    for name, pats, _ in _COMPILED:
        score = sum(1 for p in pats if p.search(q))
        if score > best_score:
            best, best_score = name, score
    if best is None or (best == 'briefing' and zone):
        best = 'zone' if zone else 'briefing'
    return best, zone


# -- SMALL FORMATTERS ------------------------------------------------------------------------------
def _n(x, d=0):
    """Number with thousands separators, rounded half away from zero -- the same rule the
       dashboard's JavaScript uses, so the Copilot and the screens never disagree on 174.5."""
    if x is None:
        return 'n/a'
    q = Decimal(str(x)).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    return f'{q:,.{d}f}'


def _signed(x, d=0):
    if x is None:
        return 'n/a'
    v = _n(x, d)
    return v if v.startswith('-') or float(v.replace(',', '')) == 0 else '+' + v


def _pct(x, d=0, sign=True):
    if x is None:
        return 'n/a'
    q = Decimal(str(x)).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    return f'{q:+.{d}f}%' if sign else f'{q:.{d}f}%'


def _hhmm(ts):
    s = str(ts or '')
    return s[11:16] if len(s) >= 16 else s


def _st(ctx):
    return ctx.get('state', {})


def _zone(ctx, zid):
    return next((z for z in _st(ctx).get('zones', []) if z['zone_id'] == zid), None)


def _short(name):
    """'Excavation & Haul Road (Phase 3)' -> 'Excavation & Haul Road' (avoids nested brackets)."""
    return re.sub(r'\s*\([^)]*\)\s*$', '', name or '')


def _tidy(text):
    """House style for user-facing text: real dashes instead of the ASCII '--'."""
    return text.replace(' -- ', ' \u2014 ').replace(' -> ', ' \u2192 ')


def _plural(n, word):
    return f"{n} {word}{'' if n == 1 else 's'}"


def _zname(ctx, zid):
    z = _zone(ctx, zid)
    return f"Zone {zid} ({_short(z['short_name'])})" if z else f'Zone {zid}'


def _hero_zone(ctx):
    return _st(ctx).get('insight', {}).get('zone_id', 'B')


def _stage(ctx):
    return _st(ctx).get('lifecycle', {}).get('stage', 'PENDING_APPROVAL')


def _link(label, nav, **extra):
    return {'label': label, 'nav': nav, **extra}


def _forecast_point(ctx, zid, horizon):
    for tab in ('90m', '24h', '72h', '60m'):
        for p in _st(ctx).get('forecasts', {}).get(zid, {}).get(tab, {}).get('predicted', []):
            if p.get('horizon') == horizon:
                return p
    return None


def _top_sources(att, k=2):
    rows = sorted((att or {}).get('contributors', []), key=lambda c: -c['share_pct'])[:k]
    return ', '.join(f"{c['source'].lower()} {_n(c['share_pct'])}%" for c in rows)


# -- BRIEFING -------------------------------------------------------------------------------------
def briefing(ctx):
    """A plain-language situation report built from the live state (Command Center card, the
       Copilot's 'briefing' intent and the exported report). Returns text plus structured bullets."""
    st = _st(ctx)
    if not st:
        return {'text': 'Platform data is still loading.', 'bullets': [], 'time': None}
    s, ins, wx = st['status_strip'], st['insight'], st.get('weather', {})
    now = _hhmm(st['meta']['now'])
    hz = ins['zone_id']
    z = _zone(ctx, hz)
    rec = ins['recommendation']
    pk = ins['forecast_peak']
    stage = _stage(ctx)
    hero = ctx.get('hero', {})
    times = hero.get('times', {})
    v = hero.get('verification', {})
    situation = (f"{_zname(ctx, hz)} is at {_n(z['pm10'])} {UGM3} ({z['level_label']}), "
                 f"{_pct(z['change_30m_pct'])} in 30 min; the no-new-action forecast peaks at "
                 f"{_n(pk['pm10'])} {UGM3} (90% up to {_n(pk['hi90'])}) within {pk['within_min']} min.")
    sources = (f"Probable contributors: {_top_sources(ins['contributors'])} (AI estimate, "
               f"confidence {_n(ins['contributors'].get('confidence', 0) * 100)}%).")
    # The lead bullet follows the event lifecycle, so the briefing always opens with what matters now.
    if stage == 'PENDING_APPROVAL':
        bullets = [situation, sources,
                   f"DUSTWISE ranks Zone {hz} first (IPS {_n(ins['priority']['ips_score'])}, "
                   f"{ins['priority']['band']['label'].lower()}) and recommends {rec['action'].lower()} with "
                   f"{rec['resource']} (ETA {_n(rec['eta_min'])} min) -- awaiting your approval."]
    elif stage == 'APPROVED':
        now = _hhmm(times.get('departed')) or now
        bullets = [f"{rec['resource']} is on its way to Zone {hz} for {rec['action'].lower()} (approved "
                   f"{_hhmm(times.get('approved'))}, ETA {_n(rec['eta_min'])} min). Without action Zone {hz} was "
                   f"forecast to peak at {_n(pk['pm10'])} {UGM3}.", sources]
    elif stage == 'ACTIVE':
        now = _hhmm(times.get('response')) or now
        bullets = [f"{rec['action']} is running in Zone {hz} with {rec['resource']} since "
                   f"{_hhmm(times.get('initiated'))}. Verification compares the readings with the no-action "
                   f"forecast (peak {_n(pk['pm10'])} {UGM3}, 90% up to {_n(pk['hi90'])}).", sources]
    else:
        now = _hhmm(times.get('verified')) or now
        bullets = [f"{ins['event_id']} verified: Zone {hz} fell from {_n(v.get('before_pm10'))} to "
                   f"{_n(v.get('after_pm10'))} {UGM3} ({_pct(v.get('observed_change_pct'), 1)} observed), below the "
                   f"no-action forecast's 90% band (model-estimated effect {_pct(v.get('model_estimated_effect_pct'), 1)}).",
                   f"The learning loop recorded the outcome for {rec['action'].lower()}; the loop is closed."]
    others = [h for h in st.get('predicted_hotspots', []) if h['zone_id'] != hz]
    for h in others[:2]:
        oz = _zone(ctx, h['zone_id'])
        bullets.append(f"Zone {h['zone_id']} is on {oz['level_label']} ({_n(oz['pm10'])} {UGM3}) and may reach "
                       f"{h['to_level']} within {h['within_min']} min (forecast {_n(h['forecast_peak'])}).")
    bad = [x for x in st['sensors']['sensors'] if x['status'] != 'ONLINE']
    sens = f"{s['sensors_online']}/{s['sensors_total']} sensors online"
    if bad:
        sens += '; ' + '; '.join(f"{x['sensor_id']} {x['status'].lower()}: {x['status_reason'][0].lower()}{x['status_reason'][1:]}"
                                 for x in bad[:3])
    bullets.append(sens + '.')
    if wx:
        bullets.append(f"Wind {_n(wx['wind_speed'], 1)} m/s from {_compass(wx['wind_dir_from'])}, mixing layer "
                       f"~{_n(wx['mixing_height'])} m -- {'poor' if wx['wind_speed'] < 1.5 else 'fair'} dispersion.")
    if stage == 'VERIFIED':
        head = f"{now} IST -- {ins['event_id']} closed and verified. Monitoring continues across the site."
    else:
        head = (f"{now} IST -- site status {s['overall_status']}. {_plural(s['active_events'], 'active event')}, "
                f"{_plural(s['predicted_hotspots'], 'predicted hotspot')}, "
                f"{_plural(s['interventions_active'] + _extra_running(ctx, stage), 'intervention')} running.")
    head, bullets = _tidy(head), [_tidy(b) for b in bullets]
    return {'text': head + ' ' + ' '.join(bullets), 'headline': head, 'bullets': bullets, 'time': now,
            'level': s['overall_level'], 'synthetic': True}


def _extra_running(ctx, stage):
    """+1 running intervention when an offline_pack() clone previews the ACTIVE stage of a state
       whose real stage is earlier (the live state already counts it once it is really active)."""
    real = _st(ctx).get('lifecycle', {}).get('real_stage', stage)
    return 1 if stage == 'ACTIVE' and real not in ('ACTIVE', 'VERIFIED') else 0


def _compass(deg):
    pts = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    return pts[int(((deg or 0) % 360) / 45 + 0.5) % 8]


# -- INTENT HANDLERS ------------------------------------------------------------------------------
def _h_greeting(ctx, zone):
    st = _st(ctx)
    if re.match(r'^(thanks|thank you|thx)', ctx.get('_question', '').strip().lower()):
        return ("You're welcome. I'll keep watching the site -- ask me any time for a briefing or what to do next.",
                [_link('Command Center', 'command')])
    return (f"Hi -- I'm {COPILOT_NAME}. I read the live state of {st.get('meta', {}).get('site', 'the site')}: "
            "readings, forecasts, probable sources, DUSTWISE recommendations, fleet and sensor health. "
            "Ask me what's happening, why a zone is rising, what to do next, or how reliable the forecast is.",
            [_link('Command Center', 'command')])


def _h_briefing(ctx, zone):
    b = briefing(ctx)
    return b['headline'] + '\n' + '\n'.join('- ' + x for x in b['bullets']), [_link('Command Center', 'command'),
                                                                             _link('Pollution events', 'events')]


def _h_recommend(ctx, zone):
    st = _st(ctx)
    hz = _hero_zone(ctx)
    zid = zone or hz
    if zid == hz:
        rec, pr = st['insight']['recommendation'], st['insight']['priority']
        stage = _stage(ctx)
        lead = {'PENDING_APPROVAL': 'Approve it to dispatch the unit.',
                'APPROVED': 'Already approved -- the unit is on its way.',
                'ACTIVE': 'Already approved and running; next step is verification.',
                'VERIFIED': 'This one is done and verified.'}.get(stage, '')
        opts = st['dustwise']['comparisons'].get(zid, {}).get('options', [])[1:3]
        alt = '; '.join(f"{o['label'].lower()} {_pct(-o['scenario_benefit_pct'])} ({o['disruption'].split(' - ')[0].lower()})"
                        for o in opts)
        text = (f"{rec['title']} in {rec['location']} with {rec['resource']} (ETA {_n(rec['eta_min'])} min, "
                f"~{_n(rec['water_l'])} L). {rec['expected_outcome']} Why: {rec['reason']} "
                f"IPS {_n(pr['ips_score'])} ({pr['band']['label'].lower()}). {lead}")
        if alt:
            text += f" Alternatives: {alt}."
        text += ' ' + rec['approval_note'] + '.'
        return text, [_link('Approve in Command Center', 'command'), _link('Compare actions', 'dustwise', zone=zid)]
    comp = st['dustwise']['comparisons'].get(zid, {})
    opts = comp.get('options', [])
    z = _zone(ctx, zid)
    alloc = next((a for a in st['dustwise']['allocation']['allocations'] if a.get('zone_id') == zid), None)
    if not opts:
        return f"{_zname(ctx, zid)} has no ranked action right now ({_n(z['pm10'])} {UGM3}, {z['level_label']}).", \
            [_link('DUSTWISE Optimizer', 'dustwise', zone=zid)]
    o = opts[0]
    text = (f"For {_zname(ctx, zid)}, now {_n(z['pm10'])} {UGM3} ({z['level_label']}, IPS {_n(z['ips'])}), DUSTWISE's "
            f"best option is {o['label'].lower()} with {o['resource']}: scenario estimate "
            f"{_pct(-o['scenario_benefit_pct'])} PM10 (range {_n(o['scenario_benefit_range_pct'][0])}-"
            f"{_n(o['scenario_benefit_range_pct'][1])}%), disruption: {o['disruption'].lower()}.")
    if alloc:
        text += f" Fleet plan: {alloc['resource']} -> {alloc['destination']} ({alloc['status'].lower()}, ETA {_n(alloc['eta_min'])} min)."
    text += f" It ranks below Zone {hz} (IPS {_n(st['insight']['priority']['ips_score'])}), which comes first."
    return text, [_link('Compare actions', 'dustwise', zone=zid), _link('Resource Operations', 'operations')]


def _h_why(ctx, zone):
    st = _st(ctx)
    zid = zone or _hero_zone(ctx)
    z, att = _zone(ctx, zid), st['attribution'].get(zid, {})
    ex = st.get('explanations', {}).get(zid, {})
    rows = sorted(att.get('contributors', []), key=lambda c: -c['share_pct'])
    src = ', '.join(f"{c['source'].lower()} {_n(c['share_pct'])}% ({_n(c['range_pct'][0])}-{_n(c['range_pct'][1])}%)"
                    for c in rows)
    drivers = sorted(ex.get('drivers', []), key=lambda d: -abs(d['ugm3']))[:3]
    drv = ', '.join(f"{d['label'].lower()} {_signed(d['ugm3'])}" for d in drivers)
    text = (f"{_zname(ctx, zid)} is at {_n(z['pm10'])} {UGM3} ({_pct(z['change_30m_pct'])} in 30 min). Of the local "
            f"excess over the regional background ({_n(att.get('regional_background_pm10'))} {UGM3}), the AI-estimated "
            f"contributors are {src}.")
    if zid == _hero_zone(ctx):
        text += ' ' + st['insight']['explanation']['reason']
    if drv:
        text += (f" Forecast drivers vs a typical {_hhmm(st['meta']['now'])} ({_n(ex.get('typical_level'))} {UGM3}): "
                 f"{drv} {UGM3} (exact group-Shapley).")
    return text, [_link('Forecast drivers', 'forecast', zone=zid), _link('Environmental map', 'map', zone=zid)]


def _h_forecast(ctx, zone):
    st = _st(ctx)
    zid = zone or _hero_zone(ctx)
    z = _zone(ctx, zid)
    f60, f90 = z['forecast_60m'], z['forecast_90m']
    p60 = _forecast_point(ctx, zid, '60m') or {}
    text = (f"{_zname(ctx, zid)}: now {_n(z['pm10'])} {UGM3} ({z['level_label']}). No-new-action forecast: "
            f"{_n(f60['median'])} in 60 min (90% interval {_n(f60['lo90'])}-{_n(f60['hi90'])}) and {_n(f90['median'])} "
            f"in 90 min ({_n(f90['lo90'])}-{_n(f90['hi90'])}).")
    if p60.get('level_label'):
        text += f" That is {p60['level_label']} level at 60 min"
        if p60.get('prob_next_level'):
            text += f", with a {_n(p60['prob_next_level'] * 100)}% chance of reaching {p60['next_level']}"
        text += '.'
    p6, p24 = _forecast_point(ctx, zid, '6h'), _forecast_point(ctx, zid, '24h')
    if p6:
        text += f" Outlook: {_n(p6['pm10'])} by {_hhmm(p6['time'])} ({p6['level_label']})"
        if p24:
            text += f", {_n(p24['pm10'])} in 24 h (wide interval {_n(p24['lo90'])}-{_n(p24['hi90'])})"
        text += '.'
    text += ' Operational forecasts assume no new action -- that is what DUSTWISE and verification compare against.'
    return text, [_link('Forecast & Intelligence', 'forecast', zone=zid)]


def _h_ips(ctx, zone):
    st = _st(ctx)
    pr = st['dustwise']['priorities']
    ranked = sorted(pr.values(), key=lambda p: -p['ips_score'])
    order = ', '.join(f"Zone {p['zone_id']} {_n(p['ips_score'])}" for p in ranked)
    zid = zone or ranked[0]['zone_id']
    p = pr[zid]
    top = sorted(p['factors'], key=lambda f: -f['contribution'])[:3]
    fac = ', '.join(f"{f['label'].lower()} {_n(f['value'])} (x{_n(f['weight'], 2)} = {_n(f['contribution'], 1)})" for f in top)
    text = (f"Intervention Priority Score ranking: {order}. Zone {zid} scores {_n(p['ips_score'])} "
            f"({p['band']['label'].lower()}); biggest contributions: {fac}. "
            f"{p.get('formula', 'IPS = 0.25 S + 0.20 P + 0.15 C + 0.15 E + 0.10 P + 0.15 F')}, each factor 0-100.")
    return text, [_link('DUSTWISE Optimizer', 'dustwise', zone=zid)]


def _h_verify(ctx, zone):
    st = _st(ctx)
    v = ctx.get('hero', {}).get('verification', {})
    ev = st['insight']['event_id']
    learn = st.get('learning', {})
    if _stage(ctx) == 'VERIFIED' and v:
        text = (f"{ev} verified. Zone {_hero_zone(ctx)} went from {_n(v['before_pm10'])} to {_n(v['after_pm10'])} {UGM3} "
                f"(observed {_pct(v['observed_change_pct'], 1)}). Against the no-action forecast "
                f"({_n(v['counterfactual_pm10'])}, 90% band {_n(v['counterfactual_band'][0])}-"
                f"{_n(v['counterfactual_band'][1])}) the model-estimated effect is {_pct(v['model_estimated_effect_pct'], 1)} "
                f"(range {_n(v['model_estimated_effect_range_pct'][0])} to {_n(v['model_estimated_effect_range_pct'][1])}%). "
                f"{v['verdict']}.")
    else:
        text = (f"{ev} is not verified yet (stage: {_stage(ctx).replace('_', ' ').lower()}). Once the intervention "
                "runs, I compare the zone reading with the no-action forecast issued at decision time; it only counts "
                "as verified if the observation falls below that forecast's 90% band.")
    if learn:
        text += (f" Track record: {learn.get('interventions_evaluated')} past interventions evaluated, "
                 f"{_n(learn.get('recommendation_effectiveness_pct'))}% achieved at least half the expected reduction.")
    return text, [_link('Impact & Verification', 'impact')]


def _h_learning(ctx, zone):
    learn = _st(ctx).get('learning', {})
    eff = learn.get('efficacy', {})
    rows = [f"{e['label'].lower()} x{_n(e['multiplier'], 2)} ({_n(e['multiplier_lo'], 2)}-{_n(e['multiplier_hi'], 2)}, n={e['n']})"
            for e in eff.values()][:4]
    text = ("The learning loop keeps a Bayesian field multiplier per action (measured = m x expected, prior "
            "N(1, 0.35^2)); DUSTWISE scales its scenario estimates by it. Current multipliers: " + '; '.join(rows) +
            f". {learn.get('effectiveness_definition', '')}".rstrip() + '.')
    if learn.get('updates'):
        u = learn['updates'][-1]
        if isinstance(u, dict) and u.get('action'):
            text += f" Latest update: {u.get('action')} from {u.get('event_id', 'a verified event')}."
    return text, [_link('Impact & Verification', 'impact')]


def _h_sensors(ctx, zone):
    st = _st(ctx)
    sen = st['sensors']
    c = sen['counts']
    bad = [x for x in sen['sensors'] if x['status'] != 'ONLINE' and (not zone or x['zone_id'] == zone)]
    text = f"{c['online']}/{c['total']} sensors online, {c['warning']} warning, {c['offline']} offline."
    if bad:
        parts = []
        for x in bad:
            p = f"{x['sensor_id']} (Zone {x['zone_id']}, {x['label']}): {x['status'].lower()} -- {x['status_reason']}"
            if x.get('fallback'):
                p += (f"; covered by {x['fallback']['method'].split(' from')[0]} estimate "
                      f"{_n(x['fallback']['estimated_pm10'])} {UGM3} (~{_n(x['fallback']['expected_error_pct'])}% error)")
            parts.append(p)
        text += ' Needs attention: ' + '; '.join(parts) + '.'
    elif zone:
        text += f' All Zone {zone} sensors are healthy.'
    zc = {k: v for k, v in sen['zones'].items() if v.get('confidence_reduced')}
    if zc:
        text += ' Data confidence reduced in ' + ', '.join(f"Zone {k} ({_n(v['data_confidence'] * 100)}%)" for k, v in zc.items()) + '.'
    return text, [_link('Sensors & Data', 'sensors')]


def _h_model(ctx, zone):
    model = ctx.get('model', {})
    m60 = model.get('card', {}).get('evaluation', {}).get('metrics', {}).get('60m', {}).get('pm10', {})
    bench = model.get('benchmark', {}).get('results', {}).get('60m', {})
    text = ''
    if m60:
        text = (f"60-minute PM10 forecasts on the hold-out test: RMSE {_n(m60['rmse'], 1)} {UGM3}, MAE {_n(m60['mae'], 1)}, "
                f"R² {_n(m60['r2_score'], 2)}, {_n(m60['skill_vs_persistence'] * 100)}% better than persistence; the 90% "
                f"intervals contained {_n(m60['coverage_90'] * 100)}% of outcomes; exceedances above 150 were caught "
                f"{_n(m60.get('pod', 0) * 100)}% of the time.")
    if bench:
        b = ', '.join(f"{k.replace('_', ' ')} {_n(v['rmse'], 1)}" for k, v in bench.items())
        text += f" Walk-forward benchmark RMSE at 60 min: {b}."
    text += (' Accuracy is strongest at 15-90 min (where DUSTWISE acts); 24-72 h is shown as an outlook because R² '
             'falls towards zero there. All numbers come from synthetic data.')
    return text.strip(), [_link('Model card', 'settings'), _link('Forecast & Intelligence', 'forecast')]


def _h_weather(ctx, zone):
    wx = _st(ctx).get('weather', {})
    if not wx:
        return 'No weather data in the current state.', [_link('Forecast & Intelligence', 'forecast')]
    text = (f"Now: {_n(wx['temperature'], 1)} °C, RH {_n(wx['relative_humidity'])}%, wind {_n(wx['wind_speed'], 2)} m/s from "
            f"{_compass(wx['wind_dir_from'])} ({_n(wx['wind_dir_from'])}°), mixing layer ~{_n(wx['mixing_height'])} m, "
            f"rain {_n(wx['rain_mm_1h'], 1)} mm/h, regional background PM10 {_n(wx['reference_pm10'])} {UGM3}. ")
    if wx['wind_speed'] < 1.5:
        text += ('Light wind and a shallow mixing layer mean poor ventilation: dust stays near the ground and '
                 'drifts south-east, towards Zone D.')
    else:
        text += 'Moderate wind is helping dispersion.'
    return text, [_link('Environmental map (wind layer)', 'map')]


def _h_sustainability(ctx, zone):
    s = ctx.get('sustainability', {})
    if not s:
        return 'Sustainability metrics are not loaded.', [_link('Sustainability', 'sustainability')]
    w, comp = s['water'], s['cpcb_daily_compliance_pct']
    text = (f"Since {str(s['period']['start'])[:10]}: site performance score {_n(s['performance_score'])}/100. "
            f"CPCB daily PM10 compliance: " + ', '.join(f"Zone {k} {_n(v)}%" for k, v in comp.items()) +
            f". Targeted dispatch used {_n(w['targeted_l'] / 1000, 0)} kL vs {_n(w['routine_schedule_l'] / 1000, 0)} kL for a "
            f"routine schedule ({_n(w['saving_pct'])}% saving, scenario baseline). {s['interventions']['count']} "
            f"interventions logged; most recurring hotspot: Zone {max(s['recurring_hotspots']['counts'], key=s['recurring_hotspots']['counts'].get)}.")
    recs = s.get('recommendations', [])[:2]
    if recs:
        text += ' Top structural fixes: ' + '; '.join(r['title'] for r in recs) + '.'
    return text, [_link('Sustainability', 'sustainability')]


def _h_twin(ctx, zone):
    comp = ctx.get('twin_comparison', {}).get('scenarios', [])
    text = ("The Digital Environmental Twin is a fast what-if dispersion model on a 20 m grid: change construction, "
            "traffic, wind or the intervention and see the PM10 field and receptor levels update.")
    if comp:
        parts = [f"{c['label'].lower()}: Zone B {_n(c['zones']['B'])}, school {_n(c['receptors']['R1']['pm10'])}" for c in comp]
        text += ' At current conditions -- ' + '; '.join(parts) + f' {UGM3}.'
    text += ' Simulation / scenario estimate -- not field measurement.'
    return text, [_link('Open the Digital Twin', 'twin')]


def _h_events(ctx, zone):
    ev = _st(ctx)['events']
    rows = [f"{e['event_id']} Zone {e['zone_id']} -- {e['event_type'].lower()}, {e['severity'].lower()} severity, "
            f"now {_n(e['current_pm10'])} {UGM3}, {e['intervention_status'].replace('_', ' ').lower()}"
            for e in ev['active']]
    text = f"{len(ev['active'])} active events: " + '; '.join(rows) + '.'
    hs = _st(ctx).get('predicted_hotspots', [])
    if hs:
        text += ' Predicted hotspots: ' + ', '.join(f"Zone {h['zone_id']} -> {h['to_level']} within {h['within_min']} min"
                                                    for h in hs) + '.'
    return text, [_link('Pollution Events', 'events')]


def _h_fleet(ctx, zone):
    st = _st(ctx)
    fleet = st['dustwise']['fleet']
    alloc = {a['asset_id']: a for a in st['dustwise']['allocation']['allocations']}
    rows = []
    for a in fleet:
        r = f"{a['name']} {a['status'].lower()}"
        if a['asset_id'] in alloc:
            al = alloc[a['asset_id']]
            r += f" -> {al['destination']} ({al['status'].lower()}, ETA {_n(al['eta_min'])} min)"
        elif a.get('assignment'):
            r += f" on {a['assignment']}"
        if a.get('available_l'):
            r += f", {_n(a['available_l'])} L water"
        rows.append(r)
    return 'Fleet: ' + '; '.join(rows) + '. Assignments use the Hungarian algorithm over road-network routes.', \
        [_link('Resource Operations', 'operations')]


def _h_exposure(ctx, zone):
    st = _st(ctx)
    exp = st['insight']['priority'].get('exposure', {})
    comp = ctx.get('twin_comparison', {}).get('scenarios', [])
    text = ''
    if exp:
        rec = sorted(exp.get('receptors', []), key=lambda r: -r['weighted'])
        text = (f"Zone {_hero_zone(ctx)}'s exposure score is {_n(exp.get('score'))}/100 (~{_n(exp.get('exposed_population'))} "
                f"people downwind). Most exposed: " + ', '.join(r['receptor'] for r in rec[:3]) + '.')
    if comp:
        now_, tgt = comp[0], next((c for c in comp if c['scenario'] == 'targeted'), None)
        r1 = now_['receptors']['R1']
        text += f" Twin estimate at the school: {_n(r1['pm10'])} {UGM3} now"
        if tgt:
            text += f", {_n(tgt['receptors']['R1']['pm10'])} with the targeted intervention ({_pct(tgt['school_change_pct'], 1)})"
        text += ' (simulation, not measurement).'
    d = _zone(ctx, 'D')
    if d:
        text += f" Zone D sensors read {_n(d['pm10'])} {UGM3} ({d['level_label']})."
    return text.strip(), [_link('Exposure layer on the map', 'map'), _link('Digital Twin', 'twin')]


def _h_data(ctx, zone):
    meta = _st(ctx).get('meta', {})
    return (f"{meta.get('data_label', 'Synthetic demo data')}. Green Valley Township is fictional; a physics-motivated "
            "simulator produces 5-minute weather, site activity and PM10/PM2.5 for 27 sensors, and every metric is "
            "computed from that data by the code -- nothing is typed in. Real telemetry can be swapped in with "
            "`python train.py --data your_site.csv`.", [_link('Settings -> Data source', 'settings')])


METHOD_KB = {
    'shapley': "Forecast drivers are exact group-Shapley values: features are grouped into 5 drivers, all 32 "
               "coalitions are evaluated, so the drivers add up exactly to forecast minus the typical level.",
    'nnls': "Source shares come from a non-negative least-squares receptor model (CMB-style): local excess PM10 "
            "and PM2.5 are regressed on dispersion-scaled activity indicators, with day-block bootstrap ranges.",
    'conformal': "Split-conformal prediction: the 90th percentile of absolute errors on a held-out calibration "
                 "window sets each horizon's interval width; coverage is then checked on a later test window.",
    'hungarian': "Fleet assignment uses the Hungarian algorithm (scipy linear_sum_assignment) on priority-weighted "
                 "benefit, with travel times from Dijkstra shortest paths on the site road graph.",
}


METHOD_SYNONYMS = {
    'shapley': ['shapley', 'driver', 'explain'],
    'nnls': ['nnls', 'attribution', 'source', 'contributor', 'receptor', 'cmb'],
    'conformal': ['conformal', 'interval', 'uncertain', 'band', 'forecast'],
    'hungarian': ['hungarian', 'dijkstra', 'fleet', 'routing', 'route', 'assign'],
}


def _h_method(ctx, zone):
    q = ctx.get('_question', '').lower()
    hits = [METHOD_KB[k] for k, words in METHOD_SYNONYMS.items() if any(w in q for w in words)]
    text = ' '.join(hits) if hits else ' '.join(METHOD_KB.values())
    return text, [_link('Settings -> Model card', 'settings')]


def _h_levels(ctx, zone):
    t = config.ACTION_LEVEL_THRESHOLDS
    return (f"Zone PM10 action levels ({UGM3}): Normal < {t['watch']}, Watch {t['watch']}-{t['elevated']}, Elevated "
            f"{t['elevated']}-{t['high']}, High {t['high']}-{t['critical']}, Critical >= {t['critical']}. They are "
            "site-management trigger levels; the CPCB 24-hour ambient standard (PM10 100, PM2.5 60) is tracked "
            "separately for compliance. PM10 is coarse dust (construction, roads); PM2.5 is the fine fraction "
            "(mostly combustion and background).", [_link('Settings -> Decision rules', 'settings')])


def _h_zone(ctx, zone):
    st = _st(ctx)
    zid = zone or _hero_zone(ctx)
    z = _zone(ctx, zid)
    sz = st['sensors']['zones'].get(zid, {})
    return (f"{_zname(ctx, zid)}: {_n(z['pm10'])} {UGM3} PM10 / {_n(z['pm25'])} PM2.5, {z['level_label']} "
            f"({_pct(z['change_30m_pct'])} in 30 min). 60-min forecast {_n(z['forecast_60m']['median'])} "
            f"({_n(z['forecast_60m']['lo90'])}-{_n(z['forecast_60m']['hi90'])}). IPS {_n(z['ips'])} "
            f"({z['ips_band']['label'].lower()}), top contributor {z['top_contributor'].lower()}, "
            f"{sz.get('online', '?')}/{sz.get('sensors', '?')} sensors online.",
            [_link(f'Zone {zid} on the map', 'map', zone=zid), _link('Forecast', 'forecast', zone=zid)])


HANDLERS = {
    'greeting': _h_greeting, 'briefing': _h_briefing, 'recommend': _h_recommend, 'why': _h_why,
    'forecast': _h_forecast, 'ips': _h_ips, 'verify': _h_verify, 'learning': _h_learning,
    'sensors': _h_sensors, 'model': _h_model, 'weather': _h_weather, 'sustainability': _h_sustainability,
    'twin': _h_twin, 'events': _h_events, 'fleet': _h_fleet, 'exposure': _h_exposure, 'data': _h_data,
    'method': _h_method, 'levels': _h_levels, 'zone': _h_zone,
}
FOLLOWUPS = {
    'greeting': ['Give me a situation briefing', 'What should I do now?', 'How reliable is the forecast?'],
    'briefing': ['Why is Zone B rising?', 'What should I do now?', 'Which sensors need attention?'],
    'recommend': ['Why this action and not a pause?', 'How is the priority score computed?', 'Is the school exposed?'],
    'why': ['Forecast for the next 90 minutes', 'What should I do now?', 'How does source attribution work?'],
    'forecast': ['How reliable is the forecast?', 'Why is it rising?', 'What should I do now?'],
    'ips': ['What should I do in Zone A?', 'Why is Zone B rising?', 'Show me the fleet'],
    'verify': ['What did the system learn?', 'Give me a situation briefing', 'How much water did we save?'],
    'learning': ['Did the last intervention work?', 'How reliable is the forecast?'],
    'sensors': ['How is data confidence computed?', 'Give me a situation briefing'],
    'model': ['How does split-conformal work?', 'Is this real data?', 'Forecast for Zone B'],
    'weather': ['Forecast for Zone B', 'Is the school exposed?'],
    'sustainability': ['Which zone is the recurring hotspot?', 'Did the last intervention work?'],
    'twin': ['Is the school exposed?', 'What should I do now?'],
    'events': ['Why is Zone B rising?', 'What should I do now?'],
    'fleet': ['What should I do now?', 'What should I do in Zone A?'],
    'exposure': ['What should I do now?', 'Open the what-if twin'],
    'data': ['How reliable is the forecast?', 'How does source attribution work?'],
    'method': ['How reliable is the forecast?', 'How is the priority score computed?'],
    'levels': ['Give me a situation briefing', 'Forecast for Zone B'],
    'zone': ['Why is it rising?', 'What should I do there?', 'Forecast for the next 90 minutes'],
}


def local_answer(question, ctx, zone=None):
    """Answer an on-topic question from the live platform state. Never calls the network.
       Returns {'answer', 'intent', 'zone', 'links', 'followups'}."""
    if not is_on_topic(question):
        return {'answer': off_topic_reply(), 'intent': 'off_topic', 'zone': None, 'links': [],
                'followups': ['Give me a situation briefing', 'What should I do now?']}
    intent, qzone = classify(question)
    zid = qzone or (zone if intent in ZONE_AWARE else None)
    if zid and not _zone(ctx, zid):
        zid = None
    try:
        text, links = HANDLERS[intent]({**ctx, '_question': question}, zid)
    except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError):
        # a section missing from this state (e.g. an older snapshot) -- fall back to the briefing
        intent = 'briefing'
        text, links = _h_briefing(ctx, None)
    return {'answer': _tidy(text), 'intent': intent, 'zone': zid, 'links': links,
            'followups': FOLLOWUPS.get(intent, [])[:3]}


# -- GROQ (optional LLM) --------------------------------------------------------------------------
def live_context(ctx, view=None):
    """Compact text summary of the live state for the LLM prompt (keeps tokens low)."""
    st = _st(ctx)
    if not st:
        return 'Platform state not loaded.'
    ins, s = st['insight'], st['status_strip']
    lines = [f"Time {_hhmm(st['meta']['now'])} IST, 7 Oct 2026. Site status {s['overall_status']}. "
             f"Event lifecycle for {ins['event_id']}: {_stage(ctx)}. Data: synthetic."]
    for z in st['zones']:
        lines.append(f"Zone {z['zone_id']} {z['short_name']}: PM10 {z['pm10']} ({z['level_label']}, {z['change_30m_pct']:+}% 30 min), "
                     f"PM2.5 {z['pm25']}, fc60 {z['forecast_60m']['median']} [{z['forecast_60m']['lo90']}-{z['forecast_60m']['hi90']}], "
                     f"fc90 {z['forecast_90m']['median']}, IPS {z['ips']}, top source {z['top_contributor']}.")
    lines.append('Recommendation: ' + json.dumps({k: ins['recommendation'][k] for k in
                                                   ('action', 'location', 'resource', 'eta_min', 'expected_outcome', 'reason')},
                                                  ensure_ascii=False))
    lines.append('Hero zone contributors: ' + _top_sources(ins['contributors'], 4))
    lines.append('Weather: ' + json.dumps(st.get('weather', {})))
    lines.append(f"Sensors {s['sensors_online']}/{s['sensors_total']} online, {s['sensors_warning']} warning, "
                 f"{s['sensors_offline']} offline.")
    if view:
        lines.append(f'The user is looking at the "{view}" screen.')
    return '\n'.join(lines)


def call_groq(messages, ctx, verified_answer='', view=None):
    """Call the Groq chat completion API. Returns (answer, model_name, total_tokens) or raises, so
       server.py can fall back to the local answer."""
    import requests  # imported here so the rest of the app never depends on it
    system = (GROQ_SYSTEM + '\n\nLIVE CONTEXT:\n' + live_context(ctx, view) +
              ('\n\nVERIFIED ANSWER from the platform engine for the latest question (rephrase or expand it, '
               'keep its numbers):\n' + verified_answer if verified_answer else ''))
    clean = [{'role': m.get('role'), 'content': str(m.get('content', ''))[:2000]}
             for m in messages if m.get('role') in ('user', 'assistant')]
    payload = {'model': config.GROQ_MODEL, 'messages': [{'role': 'system', 'content': system}] + clean[-10:],
               'max_tokens': 420, 'temperature': 0.4, 'stream': False}
    resp = requests.post(config.GROQ_URL, json=payload, timeout=config.GROQ_TIMEOUT_S,
                         headers={'Authorization': f'Bearer {config.GROQ_API_KEY}', 'Content-Type': 'application/json'})
    if resp.status_code == 200:
        data = resp.json()
        return data['choices'][0]['message']['content'].strip(), config.GROQ_MODEL, \
            data.get('usage', {}).get('total_tokens', 0)
    try:
        err = resp.json().get('error', {}).get('message') or resp.text[:160]
    except ValueError:
        err = resp.text[:160]
    raise RuntimeError(f'Groq HTTP {resp.status_code}: {err}')


def groq_ping():
    """Lightweight connectivity check used by /api/chat-status. Returns the HTTP status code;
       raises on network failure."""
    import requests
    resp = requests.post(config.GROQ_URL, timeout=8,
                         headers={'Authorization': f'Bearer {config.GROQ_API_KEY}', 'Content-Type': 'application/json'},
                         json={'model': config.GROQ_MODEL, 'messages': [{'role': 'user', 'content': 'Hi'}], 'max_tokens': 5})
    return resp.status_code


# -- OFFLINE PACK ---------------------------------------------------------------------------------
SUGGESTED = [
    'Give me a situation briefing',
    'Why is Zone B rising?',
    'What should I do now?',
    'Forecast for the next 90 minutes',
    'How reliable is the forecast?',
    'Which sensors need attention?',
    'Is the school exposed?',
    'Did the last intervention work?',
]


LIFECYCLE_STAGES = ['PENDING_APPROVAL', 'APPROVED', 'ACTIVE', 'VERIFIED']
STAGED_INTENTS = ['briefing', 'recommend', 'verify']


def offline_pack(ctx):
    """Everything the dashboard's Copilot needs with no server: the briefing, the local engine's
       answers for every intent (and every zone for zone-aware intents), and the intent patterns
       and scope lists so the browser can route free-text questions the same way."""
    if not _st(ctx):
        return {}
    answers = {}
    zones = [z['zone_id'] for z in _st(ctx)['zones']]
    for name in HANDLERS:
        try:
            text, links = HANDLERS[name]({**ctx, '_question': name}, None)
            answers[name] = {'answer': _tidy(text), 'links': links, 'followups': FOLLOWUPS.get(name, [])[:3]}
            if name in ZONE_AWARE:
                for zid in zones:
                    t, l = HANDLERS[name]({**ctx, '_question': name}, zid)
                    answers[f'{name}:{zid}'] = {'answer': _tidy(t), 'links': l, 'followups': FOLLOWUPS.get(name, [])[:3]}
        except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError):
            continue
    # Lifecycle-dependent answers, so the offline Copilot stays in step with the hero demo
    # (PENDING_APPROVAL -> APPROVED -> ACTIVE -> VERIFIED) even without a server.
    staged, briefings = {}, {}
    lc = _st(ctx).get('lifecycle', {})
    for stage in LIFECYCLE_STAGES:
        sctx = {**ctx, 'state': {**_st(ctx), 'lifecycle': {**lc, 'stage': stage,
                                                             'real_stage': lc.get('stage', 'PENDING_APPROVAL')}}}
        briefings[stage] = briefing(sctx)
        for name in STAGED_INTENTS:
            try:
                t, l = HANDLERS[name]({**sctx, '_question': name}, None)
                staged[f'{name}@{stage}'] = {'answer': _tidy(t), 'links': l, 'followups': FOLLOWUPS.get(name, [])[:3]}
            except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError):
                continue
    answers.update(staged)
    return {
        'name': COPILOT_NAME,
        'as_of': _st(ctx)['meta']['now'],
        'hero_zone': _hero_zone(ctx),
        'briefing': briefing(ctx),
        'briefings': briefings,
        'staged_intents': STAGED_INTENTS,
        'suggestions': SUGGESTED,
        'answers': answers,
        'intents': [[name, pats] for name, pats, _ in INTENTS],
        'zone_aware': sorted(ZONE_AWARE),
        'zone_aliases': ZONE_ALIASES,
        'scope': {'keywords': ON_TOPIC_KEYWORDS, 'whole_words': ON_TOPIC_WHOLE_WORDS,
                  'off_topic': OFF_TOPIC_SIGNALS, 'greetings': sorted(GREETINGS), 'declines': OFF_TOPIC_REPLIES},
    }
