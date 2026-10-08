"""
bootstrap.py -- First-run setup, called by server.py before any third-party library is imported.

Uses only the Python standard library (plus config.py, which is stdlib-only too), so
`python server.py` works on a fresh Python install:

  1. checks the Python version (3.10+)
  2. picks the environment: the project's own .venv (created on first run), or the virtual/conda
     environment you already activated, or this Python with --system
  3. installs requirements.txt there if anything is missing -- the exact tested versions first;
     if your Python has no ready-made packages for those, the newest compatible ones instead.
     Skipped on later runs unless requirements.txt changes.
  4. checks that the trained models load, and runs train.py (about 4 minutes) if they don't
  5. hands over to server.py, re-running it inside .venv when needed

Command-line options are parsed here and returned to server.py (python server.py --help).
"""

from __future__ import annotations

import argparse
import hashlib
import re
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SERVER_PY = ROOT / "server.py"
VENV_DIR = ROOT / ".venv"
REQ = ROOT / "requirements.txt"
ARTIFACTS = ROOT / "artifacts"
MIN_PY = (3, 10)
# import name -> distribution name (for version reporting)
LIBS = {"flask": "flask", "numpy": "numpy", "pandas": "pandas", "scipy": "scipy", "sklearn": "scikit-learn",
        "xgboost": "xgboost", "joblib": "joblib", "requests": "requests"}
DEV_LIBS = {"pytest": "pytest"}
TEST_REQ = ["pytest"]
READY_FLAG = "ECONEXUS_BOOTSTRAPPED"

IS_WIN = os.name == "nt"
if IS_WIN:
    os.system("")  # enables colours (ANSI) in the Windows console
USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


# --------------------------------------------------------------------------- console helpers
def c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def step(n: int, total: int, msg: str) -> None:
    print(f"\n{c(f'[{n}/{total}]', '1;36')} {c(msg, '1')}", flush=True)


def ok(msg: str) -> None:
    print(f"      {c('OK', '32')}  {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"      {c('!', '33')}   {msg}", flush=True)


def fail(msg: str, hints: list[str] | None = None) -> None:
    print(f"\n{c('ERROR', '1;31')}  {msg}", flush=True)
    for h in hints or []:
        print(f"       - {h}")
    print()
    pause_if_double_clicked(1)
    sys.exit(1)


def pause_if_double_clicked(code: int) -> None:
    """A console opened by double-clicking server.py on Windows closes at once; keep messages readable."""
    if IS_WIN and code and "PROMPT" not in os.environ:
        try:
            input("\nPress Enter to close this window...")
        except EOFError:
            pass


def run(cmd: list[str], **kw) -> int:
    return subprocess.call(cmd, cwd=ROOT, **kw)


def capture(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr).strip()


# --------------------------------------------------------------------------- options
def parse_args(argv: list[str]) -> argparse.Namespace:
    import config  # stdlib-only: reads ECONEXUS_HOST / ECONEXUS_PORT from .env
    ap = argparse.ArgumentParser(prog="python server.py",
                                 description="EcoNexus AI server. The first run installs everything it needs.")
    ap.add_argument("--host", default=config.HOST,
                    help=f"address to listen on (default {config.HOST}; 0.0.0.0 = whole Wi-Fi network)")
    ap.add_argument("--port", type=int, default=config.PORT,
                    help=f"port (default {config.PORT}; the next free port is used if it is busy)")
    ap.add_argument("--no-browser", action="store_true", help="don't open the dashboard in a browser")
    ap.add_argument("--debug", action="store_true", help="Flask debug mode (detailed errors in the console)")
    ap.add_argument("--system", action="store_true", help="install into this Python instead of .venv")
    ap.add_argument("--reinstall", action="store_true", help="reinstall the libraries")
    ap.add_argument("--retrain", action="store_true", help="retrain the models before starting (~4 min)")
    ap.add_argument("--install-only", action="store_true", help="install and check everything, then stop")
    ap.add_argument("--test", action="store_true", help="install pytest and run the test suite, then stop")
    return ap.parse_args(argv)


# --------------------------------------------------------------------------- environment
def in_active_env() -> bool:
    """True inside an activated venv/virtualenv, or a named conda environment (not conda's 'base')."""
    if sys.prefix != getattr(sys, "base_prefix", sys.prefix):
        return True
    conda = os.environ.get("CONDA_PREFIX")
    return bool(conda) and os.environ.get("CONDA_DEFAULT_ENV", "base") != "base" and \
        Path(conda).resolve() == Path(sys.prefix).resolve()


def venv_python() -> Path:
    return VENV_DIR / ("Scripts/python.exe" if IS_WIN else "bin/python")


def inside_project_venv() -> bool:
    return VENV_DIR.exists() and Path(sys.prefix).resolve() == VENV_DIR.resolve()


def choose_python(use_system: bool) -> tuple[Path, bool]:
    """Returns (python to use, whether server.py must be re-run with it)."""
    if use_system:
        ok(f"Using this Python directly (--system): {sys.executable}")
        return Path(sys.executable), False
    if inside_project_venv():
        ok(f"Running inside the project environment ({VENV_DIR.name})")
        return Path(sys.executable), False
    if in_active_env():
        ok(f"Using your active environment: {sys.prefix}")
        return Path(sys.executable), False
    py = venv_python()
    if py.exists():
        ok(f"Project environment: {VENV_DIR.name}{os.sep}")
        return py, True
    print(f"      Creating the project environment in {VENV_DIR.name}{os.sep} (keeps your system Python clean) ...", flush=True)
    code = run([sys.executable, "-m", "venv", str(VENV_DIR)])
    if code != 0 or not py.exists():
        shutil.rmtree(VENV_DIR, ignore_errors=True)
        hints = ["Or install into this Python instead: python server.py --system"]
        if platform.system() == "Linux":
            hints.insert(0, "On Ubuntu/Debian install the venv module first: sudo apt install python3-venv")
        fail("Could not create the project environment.", hints)
    ok(f"Created {VENV_DIR.name}{os.sep}")
    return py, True


def marker_path(py: Path) -> Path:
    """Where we remember what is installed for this interpreter."""
    if py == venv_python() or inside_project_venv():
        return VENV_DIR / ".installed.json"
    _, prefix = capture([str(py), "-c", "import sys; print(sys.prefix)"])
    tag = hashlib.sha256(prefix.encode()).hexdigest()[:10]
    return ROOT / f".installed-{tag}.json"


def fingerprint(py: Path, files: list[Path]) -> str:
    h = hashlib.sha256()
    for f in files:
        h.update(f.read_bytes())
    _, out = capture([str(py), "-c", "import sys; print(sys.version)"])
    h.update(out.encode())
    return h.hexdigest()


def load_marker(path: Path) -> dict:
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except (ValueError, OSError):
        return {}


def check_imports(py: Path, dev: bool) -> tuple[bool, dict | str]:
    libs = {**LIBS, **(DEV_LIBS if dev else {})}
    code = ("import importlib, importlib.metadata as md, json\n"
            f"libs = {libs!r}\nv = {{}}\n"
            "for mod, dist in libs.items():\n"
            "    importlib.import_module(mod)\n"
            "    v[dist] = md.version(dist)\n"
            "print(json.dumps(v))")
    rc, out = capture([str(py), "-c", code])
    if rc != 0:
        return False, out
    try:
        return True, json.loads(out.splitlines()[-1])
    except (ValueError, IndexError):
        return False, out


# Oldest versions the code is known to work with -- used only when the exact tested versions in
# requirements.txt have no ready-made packages for this Python (e.g. pandas 3 needs Python 3.11+).
MIN_VERSIONS = {"flask": "3.0", "numpy": "1.26", "pandas": "2.1", "scipy": "1.12", "scikit-learn": "1.4",
                "xgboost": "2.0", "joblib": "1.3", "requests": "2.31"}


def relaxed_requirements() -> Path:
    """requirements.txt with every exact pin (==x.y.z) replaced by the oldest supported version as a
       floor, so pip can pick whatever release exists for this Python."""
    out = ROOT / ".requirements-relaxed.txt"
    lines = []
    for line in REQ.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*([A-Za-z0-9_.\-]+)\s*==\s*([0-9][^\s;#]*)", line)
        if m:
            name = m.group(1)
            floor = MIN_VERSIONS.get(name.lower(), ".".join(m.group(2).split(".")[:2]))
            lines.append(f"{name}>={floor}")
        else:
            lines.append(line)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def ensure_libraries(py: Path, reinstall: bool, dev: bool) -> dict:
    marker = marker_path(py)
    done = load_marker(marker)
    fp_run = fingerprint(py, [REQ])
    fp_dev = fp_run + "+" + ",".join(TEST_REQ)
    if not reinstall and done.get("runtime") == fp_run and (not dev or done.get("dev") == fp_dev):
        good, versions = check_imports(py, dev)
        if good:
            ok("Libraries ready: " + ", ".join(f"{k} {versions[k]}" for k in ("flask", "pandas", "scikit-learn", "xgboost")))
            return done

    print("      Installing the libraries from requirements.txt (first run: a few minutes, XGBoost is the largest) ...", flush=True)
    run([str(py), "-m", "pip", "install", "--upgrade", "pip", "--disable-pip-version-check", "-q"])
    extra = TEST_REQ if dev else []
    code = run([str(py), "-m", "pip", "install", "--disable-pip-version-check", "-r", str(REQ), *extra])
    if code != 0:
        warn("The exact tested versions are not available for this Python; installing the newest compatible ones.")
        code = run([str(py), "-m", "pip", "install", "--disable-pip-version-check", "-r",
                    str(relaxed_requirements()), *extra])
    if code != 0:
        hints = ["Check your internet connection; behind a proxy, set HTTPS_PROXY first.",
                 "Use Python 3.10-3.13: very new Python versions may not have ready-made packages yet.",
                 "Then run python server.py again; it continues where it stopped."]
        fail("Installing the libraries failed (see the pip messages above).", hints)
    good, versions = check_imports(py, dev)
    if not good:
        fail("The libraries installed but could not be imported:\n" + str(versions)[-1500:],
             ["Run python server.py --reinstall.", "If that fails too, delete the .venv folder and run again."])
    done.update({"runtime": fp_run, "versions": versions, "installed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "models_checked": None})
    if dev:
        done["dev"] = fp_dev
    marker.write_text(json.dumps(done, indent=2))
    ok("Installed: " + ", ".join(f"{k} {v}" for k, v in versions.items()))
    return done


# --------------------------------------------------------------------------- models
def models_signature() -> str:
    parts = [f"{f.name}:{f.stat().st_size}:{int(f.stat().st_mtime)}" for f in sorted(ARTIFACTS.glob("*.joblib"))]
    return "|".join(parts)


def models_load(py: Path) -> tuple[bool, str]:
    """Loads the trained models in a child process (with the environment's libraries). Never raises:
       problems come back as a one-line reason, so a debugger attached to this process tree does not
       stop on an exception that setup already handles by retraining."""
    f, a = ARTIFACTS / "forecaster.joblib", ARTIFACTS / "attribution.joblib"
    if not f.exists() or not a.exists():
        return False, "no trained models found"
    code = ("import sys, warnings\nwarnings.filterwarnings('ignore')\n"
            "try:\n"
            "    import joblib\n"
            "    from ml_models import MultiHorizonForecaster\n"
            f"    m = MultiHorizonForecaster.load(r'{f}')\n"
            "    assert m.models, 'the forecaster file holds no models'\n"
            f"    joblib.load(r'{a}')\n"
            "except BaseException as e:\n"
            "    print('MODEL_CHECK_FAILED: ' + (str(e).strip().splitlines() or [type(e).__name__])[0][:400])\n"
            "    sys.exit(3)\n"
            "print('ok')")
    rc, out = capture([str(py), "-c", code])
    if rc == 0 and out.strip().endswith("ok"):
        return True, ""
    reason = next((ln.split("MODEL_CHECK_FAILED: ", 1)[1] for ln in out.splitlines() if "MODEL_CHECK_FAILED: " in ln),
                  (out.strip().splitlines() or ["unknown reason"])[-1])
    return False, reason


def ensure_models(py: Path, retrain: bool, done: dict) -> None:
    marker = marker_path(py)
    sig = f"{done.get('runtime')}|{models_signature()}"
    if not retrain:
        if done.get("models_checked") == sig:
            ok(f"Trained models ready ({ARTIFACTS.relative_to(ROOT)})")
            return
        good, why = models_load(py)
        if good:
            done["models_checked"] = sig
            marker.write_text(json.dumps(done, indent=2))
            ok(f"Trained models load correctly ({ARTIFACTS.relative_to(ROOT)})")
            return
        warn("The trained models can't be used: " + why)
        warn("Retraining them now -- this repairs damaged or incompatible model files.")
    print("      Training the forecasting models (about 4 minutes, needed once) ...", flush=True)
    if run([str(py), str(ROOT / "train.py")]) != 0:
        fail("Model training failed (see the messages above).", ["Run python server.py --retrain to try again."])
    ok("Models trained; offline dashboard data refreshed (static/data/snapshot.js)")
    done["models_checked"] = f"{done.get('runtime')}|{models_signature()}"
    marker.write_text(json.dumps(done, indent=2))


# --------------------------------------------------------------------------- ports
def port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1" if host in ("0.0.0.0", "localhost") else host, port))
            return True
        except OSError:
            return False


def pick_port(host: str, wanted: int) -> int:
    for p in range(wanted, wanted + 30):
        if port_free(host, p):
            if p != wanted:
                warn(f"Port {wanted} is busy; using {p} instead.")
            return p
    fail(f"No free port between {wanted} and {wanted + 29}.", ["Close the other program or use --port 9000."])
    return wanted


def lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        return "your-computer-ip"


# --------------------------------------------------------------------------- entry point
def ensure(argv: list[str] | None = None) -> argparse.Namespace:
    """Runs the setup steps; returns the options for server.py (or exits / re-runs inside .venv)."""
    argv = sys.argv[1:] if argv is None else argv
    args = parse_args(argv)
    if os.environ.get(READY_FLAG) == "1":  # already prepared by the parent process
        return args

    print(c("\nEcoNexus AI · server setup", "1;32"))
    total = 4
    step(1, total, "Checking Python")
    if sys.version_info < MIN_PY:
        fail(f"Python {platform.python_version()} is too old. EcoNexus AI needs Python 3.10 or newer.",
             ["Install Python 3.12 from https://www.python.org/downloads/ (Windows: tick 'Add python.exe to PATH').",
              "Then run: python server.py"])
    ok(f"Python {platform.python_version()} ({platform.system()} {platform.machine()})")
    if sys.version_info >= (3, 14):
        warn("Python 3.14+ is very new; if a library fails to install, use Python 3.12 or 3.13.")

    if "onedrive" in str(ROOT).lower():
        warn("This folder is inside OneDrive. OneDrive syncs .venv (hundreds of MB) and can lock or")
        warn("half-download large files such as artifacts/forecaster.joblib. A folder outside OneDrive")
        warn(r"(for example C:\Projects\econexus_ai) is faster and more reliable.")

    step(2, total, "Choosing the Python environment")
    py, reexec = choose_python(args.system)

    step(3, total, "Installing the libraries")
    done = ensure_libraries(py, args.reinstall, dev=args.test)

    step(4, total, "Checking the trained models")
    ensure_models(py, args.retrain, done)

    if args.test:
        print()
        code = run([str(py), "-m", "pytest"])
        pause_if_double_clicked(code)
        sys.exit(code)
    if args.install_only:
        print(f"\n{c('Ready.', '1;32')} Start the server with: python server.py\n")
        sys.exit(0)

    if reexec:
        # Re-run server.py with the project environment's Python (where the libraries are installed).
        env = dict(os.environ, **{READY_FLAG: "1"})
        proc = subprocess.Popen([str(py), str(SERVER_PY), *argv], cwd=ROOT, env=env)
        while True:
            try:
                code = proc.wait()
                break
            except KeyboardInterrupt:  # Ctrl+C reaches the server too; wait for it to stop cleanly
                continue
        pause_if_double_clicked(code)
        sys.exit(code)
    return args
