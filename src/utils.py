"""
utils.py
========
Shared configuration, paths, logging and metric helpers for the
"Ensemble Learning for Sri Lankan Food Price Analysis and Forecasting" project.

Everything that more than one module needs lives here so that the notebook,
the training pipeline and the Streamlit dashboard all agree on the same
constants (split dates, feature lists, file locations).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Project paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models" / "saved_models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

RAW_CSV_NAME = "wfp_food_prices_lka.csv"
RAW_CSV_PATH = RAW_DIR / RAW_CSV_NAME

CLEAN_PARQUET = PROCESSED_DIR / "food_prices_clean.parquet"
CLEAN_CSV = PROCESSED_DIR / "food_prices_clean.csv"
FEATURES_PARQUET = PROCESSED_DIR / "food_prices_features.parquet"
MODEL_TABLE_PARQUET = PROCESSED_DIR / "model_table.parquet"

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Dataset provenance (recorded for the assignment's documentation section)
# --------------------------------------------------------------------------
DATASET_INFO = {
    "title": "Sri Lanka - Food Prices",
    "provider": "World Food Programme (WFP)",
    "source_page": (
        "https://data.humdata.org/dataset/wfp-food-prices-for-sri-lanka/"
        "resource/3638f0d6-9969-48cf-a919-1d879d037ec6"
    ),
    "hdx_dataset": "wfp-food-prices-for-sri-lanka",
    "download_date": "2026-11-21",
    "file_supplied": RAW_CSV_NAME,
    "licence": (
        "Humanitarian Data Exchange entry for WFP food prices is published under "
        "the Creative Commons Attribution for Intergovernmental Organisations "
        "(CC BY-IGO) licence. Confirm the current licence text on the HDX resource "
        "page before redistribution."
    ),
    "attribution": "Source: World Food Programme (WFP), via the Humanitarian Data Exchange (HDX).",
    "notes": (
        "Monthly retail and wholesale food prices by market and commodity, in "
        "Sri Lankan Rupees (LKR), with a USD price column supplied by WFP."
    ),
}

# --------------------------------------------------------------------------
# Modelling configuration
# --------------------------------------------------------------------------
TARGET = "price"

# A single price "series" is one commodity, in one market, in one unit,
# at one price type. Lags/rolling statistics are always computed inside a series.
SERIES_KEYS = ["commodity", "market", "unit", "pricetype"]
SERIES_ID = "series_id"

# Chronological split boundaries (inclusive).  Chosen from the actual date
# coverage of the supplied file - see notebook section 11 for the reasoning.
TRAIN_END = pd.Timestamp("2024-12-31")
VAL_END = pd.Timestamp("2025-03-31")
# Test = everything after VAL_END (2025-04 .. 2025-09, six months)

# Calendar / index features
CALENDAR_FEATURES = [
    "year",
    "month",
    "quarter",
    "time_index",
    "month_sin",
    "month_cos",
    "obs_index",
    "months_since_prev_obs",
]

# Lag features
LAG_PERIODS = [1, 2, 3, 6, 12]
LAG_FEATURES = [f"lag_{p}" for p in LAG_PERIODS]

# Rolling window features (always computed on *shifted* prices)
ROLL_WINDOWS = [3, 6, 12]
ROLL_FEATURES = [f"roll_mean_{w}" for w in ROLL_WINDOWS] + [
    f"roll_std_{w}" for w in ROLL_WINDOWS
]

# Change / momentum features
CHANGE_FEATURES = ["diff_1", "pct_change_1", "pct_change_3"]

# Expanding (all-history) features - used as leakage-free fallbacks
EXPANDING_FEATURES = ["exp_mean", "exp_std"]

# Availability flags for the long-memory features
AVAILABILITY_FLAGS = ["has_lag_6", "has_lag_12", "has_roll_6", "has_roll_12"]

# Categorical features (ordinal-encoded, fitted on the training split only)
CATEGORICAL_FEATURES = ["commodity", "market", "unit", "pricetype", "category", "admin1"]
ENCODED_CATEGORICAL_FEATURES = [f"{c}_enc" for c in CATEGORICAL_FEATURES]

FEATURE_COLUMNS = (
    CALENDAR_FEATURES
    + LAG_FEATURES
    + ROLL_FEATURES
    + CHANGE_FEATURES
    + EXPANDING_FEATURES
    + AVAILABILITY_FLAGS
    + ENCODED_CATEGORICAL_FEATURES
)

# A row only enters the modelling table if ALL of these are observed.
# These need three prior months of history for the series.
REQUIRED_FEATURES = [
    "lag_1",
    "lag_2",
    "lag_3",
    "roll_mean_3",
    "roll_std_3",
    "exp_mean",
    "exp_std",
    "diff_1",
    "pct_change_1",
]

# Long-memory features: kept in the feature set, but filled from the same
# series' own expanding statistics when the 6/12-month history is not there.
FALLBACK_FEATURES = {
    "lag_6": "exp_mean",
    "lag_12": "exp_mean",
    "roll_mean_6": "exp_mean",
    "roll_mean_12": "exp_mean",
    "roll_std_6": "exp_std",
    "roll_std_12": "exp_std",
}

# Hybrid (Random Forest + ARIMA) configuration
MIN_RESIDUALS_FOR_ARIMA = 12   # minimum training residuals before fitting ARIMA
# Candidate ARIMA orders for the residual model.  d is held at 0 throughout:
# regression residuals should already be mean-reverting, and AIC is only
# comparable between models fitted to the *same* differenced series, so mixing
# d=0 and d=1 candidates would make the AIC selection meaningless.
ARIMA_CANDIDATE_ORDERS = [
    (0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1), (2, 0, 0), (0, 0, 2), (2, 0, 1),
]

RANDOM_STATE = 42
N_CV_SPLITS = 4       # TimeSeriesSplit folds for stacking/hybrid out-of-fold predictions
# Grid search uses fewer folds than the OOF step. Each extra fold multiplies the
# number of model fits, and the project has to stay runnable on a 2-core machine
# (Colab's free tier and Streamlit Community Cloud are both small). Three
# expanding-window folds still validate strictly forward in time.
TUNING_CV_SPLITS = 3

MODEL_LABELS = {
    "random_forest": "Bagging - Random Forest",
    "xgboost": "Boosting - XGBoost",
    "stacking": "Stacking - RF + XGB + LR (Ridge meta)",
    "hybrid": "Hybrid - Random Forest + ARIMA",
}
MODEL_ORDER = ["random_forest", "xgboost", "stacking", "hybrid"]

CURRENCY = "LKR"


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
def get_logger(name: str = "food_price") -> logging.Logger:
    """Return a configured logger that prints to stdout once."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


LOGGER = get_logger()


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot


def mape(y_true, y_pred) -> float:
    """Mean absolute percentage error (%), reported as supporting information."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def regression_metrics(y_true, y_pred) -> dict:
    """The three required metrics (MAE, RMSE, R2) plus MAPE and the sample size."""
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2(y_true, y_pred),
        "MAPE_pct": mape(y_true, y_pred),
        "n_obs": int(len(np.asarray(y_true))),
    }


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def make_series_id(df: pd.DataFrame) -> pd.Series:
    """Build a readable series identifier from the four series key columns."""
    return (
        df["commodity"].astype(str)
        + " | "
        + df["market"].astype(str)
        + " | "
        + df["unit"].astype(str)
        + " | "
        + df["pricetype"].astype(str)
    )


def save_json(obj, path: os.PathLike) -> Path:
    """Write a dict/list to JSON, converting numpy scalars on the way out."""

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        if isinstance(o, (pd.Timestamp,)):
            return o.isoformat()
        return str(o)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_default)
    return path


def load_json(path: os.PathLike):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def fmt_lkr(value: float, unit: str = "") -> str:
    """Format a price for titles and tables, keeping the unit visible."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    suffix = f"/{unit}" if unit else ""
    return f"{CURRENCY} {value:,.2f}{suffix}"
