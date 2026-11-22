"""
feature_engineering.py
======================
Builds the supervised-learning table for monthly food-price prediction.

Two properties matter more than anything else here:

1. **Calendar alignment.**  858 of the 1,444 series in the supplied file have at
   least one missing month.  If lags were taken as ``groupby().shift(1)`` on the
   observation order, ``lag_1`` for a series with a six-month gap would silently
   be a price from six months earlier.  Instead every series is reindexed onto a
   continuous monthly calendar, so ``lag_k`` is strictly "the price k calendar
   months before this row", and is missing when that month was not observed.

2. **No target leakage.**  Rolling and expanding statistics are computed on the
   *shifted* price series, so a row's window only ever covers months strictly
   before it.  Nothing is computed across the train/test boundary and no
   statistic is fitted on data the model has not yet seen.

Feature groups
--------------
``year, month, quarter, time_index, month_sin, month_cos``
    Calendar position.  ``time_index`` lets the trees pick up the long-run
    inflation trend; the sine/cosine pair gives month-of-year a smooth, cyclical
    encoding so that December and January are adjacent.
``obs_index, months_since_prev_obs``
    How much history the series has at this row, and whether the previous
    observation is genuinely last month.  These let a tree discount lag features
    for thin or gappy series instead of trusting them blindly.
``lag_1, lag_2, lag_3, lag_6, lag_12``
    The autoregressive core.  Monthly food prices are strongly persistent, so
    last month's price is the single most informative predictor; lag_12 carries
    the seasonal (same month last year) signal for vegetables.
``roll_mean_3/6/12, roll_std_3/6/12``
    Local level and local volatility.  The means smooth out one-month shocks;
    the standard deviations tell the model how unsettled the series has been,
    which is what separates a stable staple from a volatile vegetable.
``diff_1, pct_change_1, pct_change_3``
    Momentum.  The direction and size of recent movement, in absolute and
    relative terms.
``exp_mean, exp_std``
    The series' whole observed history up to the previous month.  Used directly
    as features and as the leakage-free fallback for the long-memory features.
``has_lag_6, has_lag_12, has_roll_6, has_roll_12``
    Indicators saying whether the long-memory feature was genuinely observed or
    filled from ``exp_mean``/``exp_std``.
``commodity_enc, market_enc, unit_enc, pricetype_enc, category_enc, admin1_enc``
    Ordinal encodings fitted on the training split only.

Why a fallback instead of dropping rows: requiring a full 12-month history for
every row cuts the usable table from 29,941 rows to 17,629 and removes most of
the dense 2023-2025 panel's early months.  The fallback keeps those rows while
only ever using the series' own past, and the ``has_*`` flags let the models
learn how much to trust a filled value.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

try:
    from . import utils
except ImportError:  # pragma: no cover
    import utils

LOGGER = utils.LOGGER


# --------------------------------------------------------------------------
def _month_index(ts: pd.Series) -> pd.Series:
    """Months since 2000-01 - an integer time axis that survives gaps."""
    ts = pd.to_datetime(ts)
    return (ts.dt.year - 2000) * 12 + (ts.dt.month - 1)


def build_features(df_clean: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Add every lag / rolling / calendar feature to the cleaned panel."""
    report: Dict = {}
    df = df_clean.copy()
    df["month_index"] = _month_index(df["month_start"]).astype(int)
    df = df.sort_values([utils.SERIES_ID, "month_index"]).reset_index(drop=True)

    # ---- reindex every series onto a continuous monthly calendar -----------
    wide = df.pivot_table(
        index="month_index", columns=utils.SERIES_ID, values="price", aggfunc="mean"
    )
    full_index = pd.RangeIndex(int(wide.index.min()), int(wide.index.max()) + 1)
    wide = wide.reindex(full_index)
    report["calendar_months_spanned"] = int(len(full_index))
    report["n_series"] = int(wide.shape[1])

    shifted = wide.shift(1)  # every window below starts from the previous month

    pieces: Dict[str, pd.DataFrame] = {}
    for lag in utils.LAG_PERIODS:
        pieces[f"lag_{lag}"] = wide.shift(lag)
    for win in utils.ROLL_WINDOWS:
        pieces[f"roll_mean_{win}"] = shifted.rolling(win, min_periods=win).mean()
        pieces[f"roll_std_{win}"] = shifted.rolling(win, min_periods=win).std()
    pieces["exp_mean"] = shifted.expanding(min_periods=2).mean()
    pieces["exp_std"] = shifted.expanding(min_periods=2).std()
    pieces["obs_index"] = shifted.notna().cumsum()
    # How many months back the previous observation actually is.
    prev_pos = pd.DataFrame(
        np.where(shifted.notna(), np.arange(len(shifted))[:, None], np.nan),
        index=shifted.index, columns=shifted.columns,
    ).ffill()
    pieces["months_since_prev_obs"] = (
        pd.DataFrame(np.arange(len(shifted))[:, None].repeat(shifted.shape[1], axis=1),
                     index=shifted.index, columns=shifted.columns)
        - prev_pos
    )

    # ---- melt the wide feature blocks back onto the observed rows ----------
    feat = df[[utils.SERIES_ID, "month_index"]].copy()
    for name, block in pieces.items():
        long = block.stack(future_stack=True).rename(name).reset_index()
        long.columns = ["month_index", utils.SERIES_ID, name]
        feat = feat.merge(long, on=[utils.SERIES_ID, "month_index"], how="left")

    df = df.merge(feat, on=[utils.SERIES_ID, "month_index"], how="left")

    # ---- momentum ---------------------------------------------------------
    df["diff_1"] = df["lag_1"] - df["lag_2"]
    df["pct_change_1"] = np.where(
        df["lag_2"] > 0, (df["lag_1"] - df["lag_2"]) / df["lag_2"] * 100, np.nan
    )
    df["pct_change_3"] = np.where(
        df["lag_3"] > 0, (df["lag_1"] - df["lag_3"]) / df["lag_3"] * 100, np.nan
    )
    # Realised change of the target itself (reporting / EDA only - never a feature)
    df["price_mom_pct"] = np.where(
        df["lag_1"] > 0, (df["price"] - df["lag_1"]) / df["lag_1"] * 100, np.nan
    )
    df["price_yoy_pct"] = np.where(
        df["lag_12"] > 0, (df["price"] - df["lag_12"]) / df["lag_12"] * 100, np.nan
    )

    # ---- calendar ---------------------------------------------------------
    df["year"] = df["month_start"].dt.year
    df["month"] = df["month_start"].dt.month
    df["quarter"] = df["month_start"].dt.quarter
    df["time_index"] = df["month_index"] - int(df["month_index"].min())
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # ---- availability flags + leakage-free fallback ------------------------
    for flag, src in (
        ("has_lag_6", "lag_6"), ("has_lag_12", "lag_12"),
        ("has_roll_6", "roll_mean_6"), ("has_roll_12", "roll_mean_12"),
    ):
        df[flag] = df[src].notna().astype(int)
        report[flag] = int(df[flag].sum())

    for target_col, fallback_col in utils.FALLBACK_FEATURES.items():
        n_filled = int(df[target_col].isna().sum())
        df[target_col] = df[target_col].fillna(df[fallback_col])
        report[f"{target_col}_filled_from_{fallback_col}"] = n_filled

    df["months_since_prev_obs"] = df["months_since_prev_obs"].fillna(0)
    df["obs_index"] = df["obs_index"].fillna(0)

    report["n_rows"] = int(len(df))
    report["feature_availability"] = {
        c: int(df[c].notna().sum()) for c in utils.FEATURE_COLUMNS
        if c in df.columns
    }
    LOGGER.info("Feature engineering produced %d rows x %d columns.", *df.shape)
    return df, report


# --------------------------------------------------------------------------
def build_model_table(df_feat: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Keep only rows whose required features are all observed."""
    report: Dict = {"n_rows_before": int(len(df_feat))}
    mask = df_feat[utils.REQUIRED_FEATURES].notna().all(axis=1)
    out = df_feat[mask].copy()
    report["required_features"] = list(utils.REQUIRED_FEATURES)
    report["n_rows_after"] = int(len(out))
    report["n_rows_dropped"] = int(len(df_feat) - len(out))
    report["pct_rows_kept"] = round(100 * len(out) / max(1, len(df_feat)), 2)
    report["n_series"] = int(out[utils.SERIES_ID].nunique())
    report["date_min"] = str(out["month_start"].min().date())
    report["date_max"] = str(out["month_start"].max().date())
    report["reason"] = (
        "A row needs at least three consecutive prior monthly observations in its "
        "own series before lag_1..lag_3 and the 3-month rolling statistics exist."
    )
    LOGGER.info(
        "Model table: %d rows (%.1f%% of featured rows), %d series, %s to %s",
        report["n_rows_after"], report["pct_rows_kept"], report["n_series"],
        report["date_min"], report["date_max"],
    )
    return out, report


# --------------------------------------------------------------------------
def chronological_split(
    df_model: pd.DataFrame,
    train_end: pd.Timestamp = utils.TRAIN_END,
    val_end: pd.Timestamp = utils.VAL_END,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """Split strictly by date: train <= train_end < val <= val_end < test.

    No shuffling anywhere.  Every test month is later than every training month,
    which is the only split that answers the question the project actually asks:
    can the model predict a price it has never seen a later month than?
    """
    train = df_model[df_model["month_start"] <= train_end].copy()
    val = df_model[
        (df_model["month_start"] > train_end) & (df_model["month_start"] <= val_end)
    ].copy()
    test = df_model[df_model["month_start"] > val_end].copy()

    n = len(df_model)
    report = {
        "train_end": str(pd.Timestamp(train_end).date()),
        "val_end": str(pd.Timestamp(val_end).date()),
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "pct_train": round(100 * len(train) / n, 1),
        "pct_val": round(100 * len(val) / n, 1),
        "pct_test": round(100 * len(test) / n, 1),
        "train_period": f"{train['month_start'].min().date()} .. {train['month_start'].max().date()}",
        "val_period": f"{val['month_start'].min().date()} .. {val['month_start'].max().date()}",
        "test_period": f"{test['month_start'].min().date()} .. {test['month_start'].max().date()}",
        "train_series": int(train[utils.SERIES_ID].nunique()),
        "val_series": int(val[utils.SERIES_ID].nunique()),
        "test_series": int(test[utils.SERIES_ID].nunique()),
        "test_months": sorted(str(d.date()) for d in test["month_start"].unique()),
        "shuffled": False,
        "note": (
            "Test series not present in training are predicted from their own lag "
            "features; the categorical encoder maps any unseen category to -1."
        ),
    }
    LOGGER.info(
        "Split -> train %d (%.0f%%) | val %d (%.0f%%) | test %d (%.0f%%)",
        len(train), report["pct_train"], len(val), report["pct_val"],
        len(test), report["pct_test"],
    )
    return train, val, test, report


# --------------------------------------------------------------------------
def fit_categorical_encoder(train: pd.DataFrame) -> OrdinalEncoder:
    """Fit the ordinal encoder on the training split only."""
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(train[utils.CATEGORICAL_FEATURES].astype(str))
    return enc


def apply_categorical_encoder(
    df: pd.DataFrame, encoder: OrdinalEncoder
) -> pd.DataFrame:
    out = df.copy()
    coded = encoder.transform(out[utils.CATEGORICAL_FEATURES].astype(str))
    for i, col in enumerate(utils.CATEGORICAL_FEATURES):
        out[f"{col}_enc"] = coded[:, i]
    return out


def get_xy(
    df: pd.DataFrame, feature_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    cols = feature_columns or utils.FEATURE_COLUMNS
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Feature columns absent from the frame: {missing}")
    X = df[cols].astype(float)
    y = df[utils.TARGET].astype(float)
    return X, y


def prepare_modelling_data(df_clean: pd.DataFrame) -> Dict:
    """End-to-end: features -> model table -> chronological split -> encoded X/y."""
    df_feat, feat_report = build_features(df_clean)
    df_model, table_report = build_model_table(df_feat)
    train, val, test, split_report = chronological_split(df_model)

    encoder = fit_categorical_encoder(train)
    train = apply_categorical_encoder(train, encoder)
    val = apply_categorical_encoder(val, encoder)
    test = apply_categorical_encoder(test, encoder)

    X_train, y_train = get_xy(train)
    X_val, y_val = get_xy(val)
    X_test, y_test = get_xy(test)

    return {
        "df_features": df_feat,
        "df_model": df_model,
        "train": train, "val": val, "test": test,
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "encoder": encoder,
        "feature_columns": list(utils.FEATURE_COLUMNS),
        "reports": {
            "features": feat_report,
            "model_table": table_report,
            "split": split_report,
        },
    }


if __name__ == "__main__":
    clean = pd.read_parquet(utils.CLEAN_PARQUET)
    data = prepare_modelling_data(clean)
    print(data["reports"]["model_table"])
    print(data["reports"]["split"])
