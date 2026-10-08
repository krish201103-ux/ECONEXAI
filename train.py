#!/usr/bin/env python3
"""
train.py -- Training & evaluation entry point (writes artifacts/ and the offline UI snapshot).

    python train.py                     # full run: ~3-5 min on a laptop
    python train.py --quick             # fewer trees, no benchmark (for development)
    python train.py --data my_site.csv  # train on real telemetry (format: data_loader.load_observed_csv)
    python train.py --snapshot-only     # just refresh static/data/snapshot.js from the saved models

You normally never run this yourself: `python server.py` (bootstrap.py) runs it once when the
trained models are missing or cannot be loaded.

Outputs (artifacts/):
    forecaster.joblib     production multi-horizon models (+ conformal widths)
    attribution.joblib    source attribution model
    evaluation.json       hold-out test metrics per horizon (computed, not typed)
    benchmark.json        walk-forward CV benchmark vs baselines
    model_card.json       consolidated summary used by the UI / API
and static/data/snapshot.js, the complete model output the dashboard uses when it is opened
straight from disk with no server running.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
import warnings

import joblib
import pandas as pd

import config
from data_loader import load_observed_csv, prepare_from_synthetic
from explainability import SourceAttributor
from ml_features import ALL_HORIZONS, UI_TABS
from ml_models import MultiHorizonForecaster, backend_name, record_artifact, run_benchmark
from simulator import NOW, generate_dataset

logger = logging.getLogger("econexus.train")
ARTIFACTS = config.ARTIFACTS_DIR

# Time splits (synthetic dataset). Training data never includes the demo day.
EVAL_TRAIN_END = pd.Timestamp("2026-09-14 23:55")
EVAL_CAL_END = pd.Timestamp("2026-09-25 23:55")
EVAL_TEST = (pd.Timestamp("2026-09-26 00:00"), pd.Timestamp("2026-10-06 23:55"))
PROD_TRAIN_END = pd.Timestamp("2026-09-30 23:55")
PROD_CAL_END = pd.Timestamp("2026-10-06 23:55")


def _json_default(o):
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def train(quick: bool = False, data: str | None = None, seed: int = config.RANDOM_SEED) -> dict:
    """Hold-out evaluation -> walk-forward benchmark -> production models -> attribution model.
       Returns the model card that is also written to artifacts/model_card.json."""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_trees = 150 if quick else 350

    if data:
        obs = load_observed_csv(data)
        end = obs.index[-1]
        prod_cal_end = end
        prod_train_end = end - pd.Timedelta(days=6)
        eval_test = (end - pd.Timedelta(days=10), end)
        eval_cal_end = eval_test[0] - pd.Timedelta(minutes=5)
        eval_train_end = eval_cal_end - pd.Timedelta(days=10)
    else:
        ds = generate_dataset(seed=seed)
        obs = prepare_from_synthetic(ds, cutoff=NOW)
        prod_train_end, prod_cal_end = PROD_TRAIN_END, PROD_CAL_END
        eval_train_end, eval_cal_end, eval_test = EVAL_TRAIN_END, EVAL_CAL_END, EVAL_TEST

    # 1. Hold-out evaluation (models trained only on data before the test window)
    logger.info("[1/4] Hold-out evaluation (train <= %s, calibrate <= %s, test %s..%s)",
                eval_train_end, eval_cal_end, *eval_test)
    eval_h = {k: ALL_HORIZONS[k] for k in UI_TABS}
    ev_model = MultiHorizonForecaster(horizons=eval_h, n_estimators=n_trees).fit(obs, eval_train_end, eval_cal_end)
    evaluation = {
        "split": {"train_end": str(eval_train_end), "calibration_end": str(eval_cal_end),
                  "test_start": str(eval_test[0]), "test_end": str(eval_test[1])},
        "backend": backend_name(),
        "metrics": ev_model.evaluate(obs, *eval_test),
    }
    (ARTIFACTS / "evaluation.json").write_text(json.dumps(evaluation, indent=2, default=_json_default))

    # 2. Walk-forward benchmark
    if not quick:
        logger.info("[2/4] Walk-forward benchmark vs baselines")
        bench = run_benchmark(obs, n_estimators=n_trees)
        (ARTIFACTS / "benchmark.json").write_text(json.dumps(bench, indent=2, default=_json_default))
    else:
        logger.info("[2/4] Benchmark skipped (--quick)")

    # 3. Production models (all horizons)
    logger.info("[3/4] Production models: train <= %s, calibrate <= %s", prod_train_end, prod_cal_end)
    prod = MultiHorizonForecaster(n_estimators=n_trees).fit(obs, prod_train_end, prod_cal_end)
    prod.save(ARTIFACTS / "forecaster.joblib")

    # 4. Attribution model
    logger.info("[4/4] Source attribution model")
    att = SourceAttributor().fit(obs, prod_cal_end)
    validation = att.validate_against_truth(obs, *eval_test) if obs.truth is not None else {"available": False}
    apath = ARTIFACTS / "attribution.joblib"
    joblib.dump(att, apath.with_name(apath.name + ".tmp"), compress=3)
    os.replace(apath.with_name(apath.name + ".tmp"), apath)      # atomic: never a half-written file
    record_artifact(apath)

    card = {
        "trained_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "data": obs.meta,
        "backend": backend_name(),
        "n_trees": n_trees,
        "horizons": list(prod.horizons),
        "production_split": {"train_end": str(prod_train_end), "calibration_end": str(prod_cal_end)},
        "evaluation": evaluation,
        "attribution_validation": validation,
        "attribution_fit_r2": {z: round(m.r2, 3) for z, m in att.models.items()},
        "qa": obs.qa,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (ARTIFACTS / "model_card.json").write_text(json.dumps(card, indent=2, default=_json_default))
    logger.info("Training done in %.1f s. Artifacts in %s", time.time() - t0, ARTIFACTS)
    return card


def export_snapshot(service=None) -> None:
    """Writes static/data/snapshot.js (``window.ECONEXUS_SNAPSHOT = {...}``): a complete,
       model-generated picture of the site at 14:30, the hero-demo playback frames and the
       Copilot's offline answers. The UI loads it with a plain <script> tag, so it also works when
       index.html is opened directly from disk (no server, no internet). When server.py is
       running, the UI uses the live API instead."""
    from forecast_service import PlatformService
    t0 = time.time()
    snap = (service or PlatformService()).ensure().snapshot(live=False)
    out = config.SNAPSHOT_JS
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(snap, separators=(",", ":"), default=str, ensure_ascii=False)
    out.write_text("/* Generated by `python train.py` - model output, do not edit by hand. */\n"
                   f"window.ECONEXUS_SNAPSHOT = {payload};\n", encoding="utf-8")
    logger.info("Snapshot written to %s (%.0f KB) in %.1f s", out, out.stat().st_size / 1024, time.time() - t0)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python train.py", description="Train and evaluate the EcoNexus AI models")
    ap.add_argument("--quick", action="store_true", help="fewer trees, skip the walk-forward benchmark")
    ap.add_argument("--data", type=str, default=None, help="CSV of real telemetry (long format)")
    ap.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    ap.add_argument("--snapshot-only", action="store_true", help="only refresh static/data/snapshot.js")
    ap.add_argument("--no-snapshot", action="store_true", help="skip writing static/data/snapshot.js")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    warnings.filterwarnings("ignore")
    if not args.snapshot_only:
        train(quick=args.quick, data=args.data, seed=args.seed)
    if not args.no_snapshot and not args.data:
        export_snapshot()


if __name__ == "__main__":
    main()
