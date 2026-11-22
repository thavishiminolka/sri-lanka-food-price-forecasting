"""
forecasting.py
==============
Multi-step-ahead food price forecasting for a chosen commodity and market.

How it works
------------
The tabular ensembles predict one month ahead from lagged prices.  To reach a
horizon of 3, 6 or 12 months they are applied **recursively**: the model's own
prediction for month *t+1* is written into the series history and becomes the
``lag_1`` input for month *t+2*, and so on.

``_features_for_month`` rebuilds a feature row with exactly the definitions used
in :mod:`feature_engineering` - calendar-aligned lags, rolling windows on the
*shifted* series, expanding statistics as the fallback for missing long-memory
windows - so a forecast row looks to the model exactly like a training row.

What this means for the user, and why the dashboard says so plainly
-------------------------------------------------------------------
* Recursive forecasts **accumulate error**: month 12 is predicted from eleven
  earlier predictions, so its uncertainty is much wider than month 1's.
* The models were fitted on 2004-2025 data and know nothing about future policy
  changes, import restrictions, fuel prices, harvest failures or currency moves.
* A forecast here is a conditional estimate of what the recent price dynamics
  imply, not a guaranteed price.  The uncertainty band is derived from the
  model's own historical test-set error for that series, widened with the
  horizon; it is indicative, not a calibrated prediction interval.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from . import utils
except ImportError:  # pragma: no cover
    import utils

LOGGER = utils.LOGGER

SUPPORTED_HORIZONS = (1, 3, 6, 12)
MIN_HISTORY_FOR_FORECAST = 4  # lag_1..lag_3 plus the current month


def _month_index_from_timestamp(ts: pd.Timestamp) -> int:
    ts = pd.Timestamp(ts)
    return (ts.year - 2000) * 12 + (ts.month - 1)


def _timestamp_from_month_index(mi: int) -> pd.Timestamp:
    year = 2000 + int(mi) // 12
    month = int(mi) % 12 + 1
    return pd.Timestamp(year=year, month=month, day=1)


# --------------------------------------------------------------------------
def _features_for_month(
    hist: pd.Series,
    target_mi: int,
    meta: Dict[str, str],
    encoder,
    time_index_origin: int,
) -> pd.DataFrame:
    """Build a one-row feature frame for ``target_mi`` from a calendar-indexed history.

    ``hist`` is indexed by ``month_index`` and holds observed (or already
    forecast) prices.  Only months strictly before ``target_mi`` are used.
    """
    past = hist[hist.index < target_mi].sort_index()
    row: Dict[str, float] = {}

    # Calendar-aligned lags
    for lag in utils.LAG_PERIODS:
        want = target_mi - lag
        row[f"lag_{lag}"] = float(past[want]) if want in past.index else np.nan

    # Rolling windows over the previous W calendar months (all must be present)
    for win in utils.ROLL_WINDOWS:
        months = [target_mi - k for k in range(1, win + 1)]
        vals = [past[m] for m in months if m in past.index]
        if len(vals) == win:
            row[f"roll_mean_{win}"] = float(np.mean(vals))
            row[f"roll_std_{win}"] = float(np.std(vals, ddof=1)) if win > 1 else 0.0
        else:
            row[f"roll_mean_{win}"] = np.nan
            row[f"roll_std_{win}"] = np.nan

    # Expanding statistics over everything observed so far
    if len(past) >= 2:
        row["exp_mean"] = float(past.mean())
        row["exp_std"] = float(past.std(ddof=1))
    else:
        row["exp_mean"] = float(past.mean()) if len(past) else np.nan
        row["exp_std"] = np.nan

    row["obs_index"] = float(len(past))
    row["months_since_prev_obs"] = (
        float(target_mi - int(past.index.max())) if len(past) else 0.0
    )

    # Momentum
    l1, l2, l3 = row["lag_1"], row["lag_2"], row["lag_3"]
    row["diff_1"] = l1 - l2 if not (np.isnan(l1) or np.isnan(l2)) else np.nan
    row["pct_change_1"] = (
        (l1 - l2) / l2 * 100 if not (np.isnan(l1) or np.isnan(l2)) and l2 > 0 else np.nan
    )
    row["pct_change_3"] = (
        (l1 - l3) / l3 * 100 if not (np.isnan(l1) or np.isnan(l3)) and l3 > 0 else np.nan
    )

    # Calendar position
    ts = _timestamp_from_month_index(target_mi)
    row["year"] = float(ts.year)
    row["month"] = float(ts.month)
    row["quarter"] = float((ts.month - 1) // 3 + 1)
    row["time_index"] = float(target_mi - time_index_origin)
    row["month_sin"] = float(np.sin(2 * np.pi * ts.month / 12))
    row["month_cos"] = float(np.cos(2 * np.pi * ts.month / 12))

    # Availability flags, then the same leakage-free fallback as in training
    for flag, src in (("has_lag_6", "lag_6"), ("has_lag_12", "lag_12"),
                      ("has_roll_6", "roll_mean_6"), ("has_roll_12", "roll_mean_12")):
        row[flag] = 0.0 if np.isnan(row[src]) else 1.0
    for target_col, fallback_col in utils.FALLBACK_FEATURES.items():
        if np.isnan(row[target_col]):
            row[target_col] = row[fallback_col]

    # Categorical encodings via the encoder fitted on the training split
    cat_values = [[str(meta.get(c, "")) for c in utils.CATEGORICAL_FEATURES]]
    coded = encoder.transform(pd.DataFrame(cat_values, columns=utils.CATEGORICAL_FEATURES))
    for i, col in enumerate(utils.CATEGORICAL_FEATURES):
        row[f"{col}_enc"] = float(coded[0, i])

    return pd.DataFrame([row])[utils.FEATURE_COLUMNS].astype(float)


# ==========================================================================
class ForecastEngine:
    """Recursive forecaster over the cleaned monthly panel.

    Parameters
    ----------
    history
        The cleaned panel (one row per series-month) with ``series_id``,
        ``month_start``, ``price`` and the categorical columns.
    model
        Any fitted estimator exposing ``predict(X)``; by default the
        best-performing model from the evaluation step.
    encoder
        The ``OrdinalEncoder`` fitted on the training split.
    model_name
        Label shown in the dashboard and the returned payload.
    series_error
        Optional ``{series_id: mae}`` from the test set, used to size the
        indicative uncertainty band.
    """

    def __init__(
        self,
        history: pd.DataFrame,
        model,
        encoder,
        model_name: str = "model",
        series_error: Optional[Dict[str, float]] = None,
        global_error: Optional[float] = None,
    ):
        self.model = model
        self.encoder = encoder
        self.model_name = model_name
        self.series_error = series_error or {}
        self.global_error = global_error

        df = history.copy()
        if utils.SERIES_ID not in df.columns:
            df[utils.SERIES_ID] = utils.make_series_id(df)
        df["month_start"] = pd.to_datetime(df["month_start"])
        df["month_index"] = df["month_start"].map(_month_index_from_timestamp)
        self.history = df.sort_values([utils.SERIES_ID, "month_index"]).reset_index(drop=True)
        self.time_index_origin = int(self.history["month_index"].min())

        self._meta = (
            self.history.sort_values("month_index")
            .groupby(utils.SERIES_ID, observed=True)
            .last()[[c for c in utils.CATEGORICAL_FEATURES if c in self.history.columns]]
            .to_dict(orient="index")
        )

    # ------------------------------------------------------------ catalogue
    def available_commodities(self) -> List[str]:
        return sorted(self.history["commodity"].dropna().unique().tolist())

    def markets_for_commodity(self, commodity: str) -> List[str]:
        d = self.history[self.history["commodity"] == commodity]
        return sorted(d["market"].dropna().unique().tolist())

    def series_options(self, commodity: str, market: str) -> pd.DataFrame:
        """Unit / price-type combinations available for a commodity-market pair."""
        d = self.history[
            (self.history["commodity"] == commodity) & (self.history["market"] == market)
        ]
        if d.empty:
            return pd.DataFrame(columns=[utils.SERIES_ID, "unit", "pricetype", "n_obs",
                                         "first_month", "last_month"])
        out = (
            d.groupby([utils.SERIES_ID, "unit", "pricetype"], observed=True)
            .agg(n_obs=("price", "size"), first_month=("month_start", "min"),
                 last_month=("month_start", "max"))
            .reset_index()
            .sort_values("n_obs", ascending=False)
        )
        return out

    def get_series(self, series_id: str) -> pd.DataFrame:
        return self.history[self.history[utils.SERIES_ID] == series_id].sort_values("month_index")

    def supported_horizons(self, series_id: str) -> List[int]:
        """Horizons the data supports: every one, provided there is enough history."""
        n = len(self.get_series(series_id))
        return list(SUPPORTED_HORIZONS) if n >= MIN_HISTORY_FOR_FORECAST else []

    # ------------------------------------------------------------- forecast
    def forecast(
        self, series_id: str, horizon: int = 6, model=None, model_name: Optional[str] = None,
    ) -> Dict:
        """Recursive ``horizon``-month forecast for one series."""
        if horizon not in SUPPORTED_HORIZONS:
            raise ValueError(f"horizon must be one of {SUPPORTED_HORIZONS}, got {horizon}")

        model = model or self.model
        model_name = model_name or self.model_name
        s = self.get_series(series_id)
        if s.empty:
            raise KeyError(f"Unknown series: {series_id}")

        n_obs = len(s)
        if n_obs < MIN_HISTORY_FOR_FORECAST:
            return {
                "ok": False,
                "series_id": series_id,
                "reason": (
                    f"This series has only {n_obs} monthly observations. "
                    f"At least {MIN_HISTORY_FOR_FORECAST} are needed before lag_1 to lag_3 "
                    "and the 3-month rolling features exist."
                ),
                "history": s[["month_start", "price"]].rename(columns={"price": "actual"}),
                "forecast": pd.DataFrame(),
            }

        meta = self._meta.get(series_id, {})
        hist = pd.Series(s["price"].to_numpy(dtype=float), index=s["month_index"].to_numpy())
        last_mi = int(hist.index.max())

        base_err = self.series_error.get(series_id)
        if base_err is None or not np.isfinite(base_err):
            base_err = self.global_error if self.global_error is not None else float(
                np.nanmean(np.abs(np.diff(hist.to_numpy()))) if n_obs > 1 else 0.0
            )

        rows = []
        working = hist.copy()
        for step in range(1, horizon + 1):
            target_mi = last_mi + step
            X = _features_for_month(
                working, target_mi, meta, self.encoder, self.time_index_origin
            )
            if X.isna().any().any():
                missing = X.columns[X.isna().any()].tolist()
                LOGGER.warning(
                    "Stopping the forecast at step %d for %s: features unavailable %s",
                    step, series_id, missing,
                )
                break
            pred = float(np.asarray(model.predict(X)).ravel()[0])
            pred = max(pred, 0.0)  # a price cannot be negative
            # Error widens with the square root of the horizon - a random-walk
            # style heuristic, stated as indicative rather than calibrated.
            band = float(base_err) * np.sqrt(step)
            rows.append({
                "step": step,
                "month_start": _timestamp_from_month_index(target_mi),
                "forecast": pred,
                "lower": max(pred - 1.96 * band, 0.0),
                "upper": pred + 1.96 * band,
                "indicative_error": band,
            })
            working.loc[target_mi] = pred  # recursive: the prediction becomes an input

        fc = pd.DataFrame(rows)
        return {
            "ok": not fc.empty,
            "series_id": series_id,
            "commodity": meta.get("commodity"),
            "market": meta.get("market"),
            "unit": meta.get("unit"),
            "pricetype": meta.get("pricetype"),
            "model_name": model_name,
            "horizon_requested": horizon,
            "horizon_produced": int(len(fc)),
            "n_history_obs": n_obs,
            "last_observed_month": str(_timestamp_from_month_index(last_mi).date()),
            "last_observed_price": float(hist.loc[last_mi]),
            "history": s[["month_start", "price"]].rename(columns={"price": "actual"}),
            "forecast": fc,
            "method": "recursive one-step-ahead prediction, feeding each prediction back as lag_1",
            "caveats": [
                "Forecasts are model-based estimates, not guaranteed prices.",
                "Recursive forecasting compounds error: the 12th month rests on eleven "
                "earlier predictions.",
                "The model has no information about future policy, import rules, fuel "
                "prices, weather or exchange-rate movements.",
                "The band shown is scaled from this series' own historical error and "
                "widened with the horizon; it is indicative, not a calibrated interval.",
            ],
        }

    def forecast_table(self, result: Dict) -> pd.DataFrame:
        """Tidy, display-ready forecast table with the unit stated in the headers."""
        if not result.get("ok"):
            return pd.DataFrame()
        unit = result.get("unit", "")
        fc = result["forecast"].copy()
        last = result["last_observed_price"]
        fc["Month"] = fc["month_start"].dt.strftime("%Y-%m")
        fc[f"Forecast (LKR/{unit})"] = fc["forecast"].round(2)
        fc[f"Lower (LKR/{unit})"] = fc["lower"].round(2)
        fc[f"Upper (LKR/{unit})"] = fc["upper"].round(2)
        fc["Change vs last actual"] = (
            ((fc["forecast"] - last) / last * 100).round(2).astype(str) + "%"
        )
        fc["Months ahead"] = fc["step"]
        return fc[["Months ahead", "Month", f"Forecast (LKR/{unit})",
                   f"Lower (LKR/{unit})", f"Upper (LKR/{unit})", "Change vs last actual"]]


# --------------------------------------------------------------------------
def series_error_map(resid_df: pd.DataFrame, model_key: str) -> Dict[str, float]:
    """Per-series test MAE for one model - used to size the forecast bands."""
    d = resid_df[resid_df["model_key"] == model_key]
    if d.empty:
        return {}
    return (
        d.groupby(utils.SERIES_ID, observed=True)["abs_error"].mean().to_dict()
    )


def batch_forecast(
    engine: ForecastEngine, series_ids: Sequence[str], horizon: int = 6
) -> pd.DataFrame:
    """Forecast several series at once (used to pre-compute dashboard examples)."""
    frames = []
    for sid in series_ids:
        res = engine.forecast(sid, horizon=horizon)
        if not res.get("ok"):
            continue
        fc = res["forecast"].copy()
        fc[utils.SERIES_ID] = sid
        fc["commodity"] = res["commodity"]
        fc["market"] = res["market"]
        fc["unit"] = res["unit"]
        frames.append(fc)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
