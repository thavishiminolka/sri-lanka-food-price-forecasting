"""
model_persistence.py
====================
Saving and loading everything the Streamlit dashboard needs, so that the app
**loads** trained artefacts instead of retraining on every start.

Saved under ``models/saved_models/``:

===============================  =====================================================
``random_forest.joblib``          tuned RandomForestRegressor (bagging)
``xgboost.joblib``                tuned XGBRegressor (boosting)
``stacking.joblib``               fitted TimeAwareStackingRegressor
``hybrid.joblib``                 RandomForestARIMAHybrid, with its residual forecasts
``arima_info.json``               per-series ARIMA order, AIC and residual count
``preprocessing.joblib``          the OrdinalEncoder fitted on the training split
``metadata.json``                 split dates, feature list, versions, timings
``metrics.json``                  the four-model comparison and the best model
===============================  =====================================================

The hybrid's ``__getstate__`` converts its statsmodels results into a cached
residual-forecast array before pickling, so the file stays small and loads
without statsmodels having to rebuild state.
"""
from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import joblib
import pandas as pd

try:
    from . import utils
except ImportError:  # pragma: no cover
    import utils

LOGGER = utils.LOGGER

FILES = {
    "random_forest": "random_forest.joblib",
    "xgboost": "xgboost.joblib",
    "stacking": "stacking.joblib",
    "hybrid": "hybrid.joblib",
    "encoder": "preprocessing.joblib",
    "metadata": "metadata.json",
    "metrics": "metrics.json",
    "arima_info": "arima_info.json",
    "feature_importance": "feature_importance.json",
}


def _versions() -> Dict[str, str]:
    import numpy, sklearn, statsmodels, xgboost
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "statsmodels": statsmodels.__version__,
        "joblib": joblib.__version__,
    }


# --------------------------------------------------------------------------
def save_artifacts(
    models: Dict[str, object],
    encoder,
    metrics: Dict,
    metadata: Optional[Dict] = None,
    arima_info: Optional[Dict] = None,
    feature_importance: Optional[Dict] = None,
    outdir: Optional[Path] = None,
) -> Dict[str, str]:
    """Persist models, the encoder, metrics and metadata.  Returns the paths written."""
    outdir = Path(outdir or utils.MODELS_DIR)
    outdir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}

    for key in ("random_forest", "xgboost", "stacking", "hybrid"):
        if key in models and models[key] is not None:
            path = outdir / FILES[key]
            joblib.dump(models[key], path, compress=3)
            written[key] = str(path)
            LOGGER.info("saved %s (%.1f MB)", path.name, path.stat().st_size / 1024**2)

    path = outdir / FILES["encoder"]
    joblib.dump(
        {
            "categorical_encoder": encoder,
            "categorical_features": list(utils.CATEGORICAL_FEATURES),
            "feature_columns": list(utils.FEATURE_COLUMNS),
            "required_features": list(utils.REQUIRED_FEATURES),
            "fallback_features": dict(utils.FALLBACK_FEATURES),
            "target": utils.TARGET,
        },
        path,
        compress=3,
    )
    written["encoder"] = str(path)

    meta = {
        "project": "Ensemble Learning for Sri Lankan Food Price Analysis and Forecasting",
        "saved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": utils.DATASET_INFO,
        "target": utils.TARGET,
        "currency": utils.CURRENCY,
        "series_definition": utils.SERIES_KEYS,
        "feature_columns": list(utils.FEATURE_COLUMNS),
        "n_features": len(utils.FEATURE_COLUMNS),
        "train_end": str(utils.TRAIN_END.date()),
        "val_end": str(utils.VAL_END.date()),
        "cv_splits": utils.N_CV_SPLITS,
        "random_state": utils.RANDOM_STATE,
        "models": {k: utils.MODEL_LABELS[k] for k in utils.MODEL_ORDER},
        "versions": _versions(),
    }
    meta.update(metadata or {})
    written["metadata"] = str(utils.save_json(meta, outdir / FILES["metadata"]))
    written["metrics"] = str(utils.save_json(metrics, outdir / FILES["metrics"]))
    if arima_info is not None:
        written["arima_info"] = str(utils.save_json(arima_info, outdir / FILES["arima_info"]))
    if feature_importance is not None:
        written["feature_importance"] = str(
            utils.save_json(feature_importance, outdir / FILES["feature_importance"])
        )
    return written


# --------------------------------------------------------------------------
def load_artifacts(outdir: Optional[Path] = None, strict: bool = False) -> Dict:
    """Load whatever has been saved.  Missing pieces come back as ``None``.

    ``strict=True`` raises instead, which is what the test script uses.
    """
    outdir = Path(outdir or utils.MODELS_DIR)
    out: Dict = {"models": {}, "available": [], "missing": []}

    for key in ("random_forest", "xgboost", "stacking", "hybrid"):
        path = outdir / FILES[key]
        if path.exists():
            try:
                out["models"][key] = joblib.load(path)
                out["available"].append(key)
            except Exception as exc:  # noqa: BLE001
                if strict:
                    raise
                LOGGER.warning("could not load %s: %s", path.name, exc)
                out["models"][key] = None
                out["missing"].append(key)
        else:
            out["models"][key] = None
            out["missing"].append(key)

    path = outdir / FILES["encoder"]
    if path.exists():
        bundle = joblib.load(path)
        out["encoder"] = bundle.get("categorical_encoder")
        out["preprocessing_bundle"] = bundle
        out["feature_columns"] = bundle.get("feature_columns", list(utils.FEATURE_COLUMNS))
    else:
        if strict:
            raise FileNotFoundError(path)
        out["encoder"] = None
        out["feature_columns"] = list(utils.FEATURE_COLUMNS)
        out["missing"].append("encoder")

    for key in ("metadata", "metrics", "arima_info", "feature_importance"):
        path = outdir / FILES[key]
        out[key] = utils.load_json(path) if path.exists() else None
        if out[key] is None and key in ("metadata", "metrics"):
            out["missing"].append(key)

    if strict and out["missing"]:
        raise FileNotFoundError(f"missing artefacts: {out['missing']}")
    return out


def artifacts_exist(outdir: Optional[Path] = None) -> bool:
    outdir = Path(outdir or utils.MODELS_DIR)
    return all((outdir / FILES[k]).exists()
               for k in ("random_forest", "xgboost", "encoder", "metrics"))


def describe_artifacts(outdir: Optional[Path] = None) -> pd.DataFrame:
    """File inventory with sizes, for the dashboard's status panel."""
    outdir = Path(outdir or utils.MODELS_DIR)
    rows = []
    for key, name in FILES.items():
        p = outdir / name
        rows.append({
            "artefact": key,
            "file": name,
            "exists": p.exists(),
            "size_mb": round(p.stat().st_size / 1024**2, 3) if p.exists() else None,
        })
    return pd.DataFrame(rows)
