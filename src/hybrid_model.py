"""
hybrid_model.py
===============
D. **Hybrid - Random Forest + ARIMA.**

The idea
--------
A Random Forest is good at the *cross-sectional* part of this problem.  Given
lags, rolling statistics, the commodity, the market and the month, it maps
features to a price, and it does so non-linearly and without being told the
functional form.  What it cannot do is model its own mistakes: a tree ensemble
predicts a constant within each leaf, it cannot extrapolate a trend beyond the
range it was trained on, and - crucially - it treats each row as independent, so
any *autocorrelation* left in its errors is simply thrown away.

That left-over autocorrelation is exactly what ARIMA is for.  If the forest has
been under-predicting a particular rice series for four months running, that is
information about next month.  So the hybrid:

1. trains the Random Forest on the training features,
2. computes **leakage-safe** training predictions (out-of-fold, with
   ``TimeSeriesSplit``) and takes the residuals ``actual - predicted``,
3. fits an ARIMA model **per price series** to that residual sequence, using
   training residuals only,
4. forecasts the residual forward to each test month,
5. adds the forecast residual to the forest's test prediction:

   ``y_hybrid = y_random_forest + r_arima``

Why the residuals must be out-of-fold
-------------------------------------
A Random Forest fitted on a row and then asked to predict that same row is
nearly exact: in-sample residuals are tiny and are mostly memorisation noise.
An ARIMA fitted to those would learn the pattern of the forest's *memorisation*
rather than of its *generalisation error*, and the correction would be useless -
or worse than useless - out of sample.  Out-of-fold residuals come from folds
where the forest had not seen the row, so they have the same character as the
errors it will make on the test months.

How leakage is prevented
------------------------
* The OOF folds are ``TimeSeriesSplit`` folds, so a residual for month *m* comes
  from a forest fitted only on months before *m*.
* ARIMA sees training residuals only.  Validation and test residuals are never
  computed, let alone fitted.
* The forecast origin is the last training month.  A test month *k* months after
  that origin uses the *k*-step-ahead forecast - the model is never re-anchored
  on an actual value it has not been given.
* The residual correction for a series is produced once, before any test actual
  is read.

Series with too little history
------------------------------
ARIMA on eight observations estimates nothing useful.  Series with fewer than
``utils.MIN_RESIDUALS_FOR_ARIMA`` training residuals therefore receive a zero
correction and fall back to the plain Random Forest prediction.  The count of
corrected versus uncorrected series is reported rather than hidden, because it
bounds how much of the hybrid's behaviour is actually hybrid.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

try:
    from . import utils
except ImportError:  # pragma: no cover
    import utils

LOGGER = utils.LOGGER

warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=RuntimeWarning)


# --------------------------------------------------------------------------
def out_of_fold_predictions(
    model, X: pd.DataFrame, y: pd.Series, n_splits: int = utils.N_CV_SPLITS
) -> Tuple[np.ndarray, np.ndarray]:
    """Expanding-window OOF predictions.  Returns ``(predictions, covered_mask)``.

    Rows in the earliest fold are never in a test fold, so they get no
    prediction; ``covered_mask`` marks the rows that do.
    """
    X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X, float))
    X_df = X_df.reset_index(drop=True)
    y_arr = np.asarray(y, dtype=float)
    preds = np.full(len(y_arr), np.nan)

    splitter = TimeSeriesSplit(n_splits=n_splits)
    for fold, (tr, te) in enumerate(splitter.split(X_df), start=1):
        est = clone(model)
        est.fit(X_df.iloc[tr], y_arr[tr])
        preds[te] = est.predict(X_df.iloc[te])
        LOGGER.info(
            "    hybrid OOF fold %d/%d: fitted on %d rows, predicted %d rows",
            fold, n_splits, len(tr), len(te),
        )
    covered = ~np.isnan(preds)
    return preds, covered


# --------------------------------------------------------------------------
def fit_arima_on_residuals(
    residuals: pd.Series,
    candidate_orders: Optional[List[Tuple[int, int, int]]] = None,
) -> Dict:
    """Fit ARIMA to one residual sequence, choosing the order by AIC.

    All candidates keep ``d = 0`` so their AIC values are comparable.  Returns a
    dict with the fitted result, or ``{"ok": False, ...}`` when nothing converges.
    """
    from statsmodels.tsa.arima.model import ARIMA

    candidate_orders = candidate_orders or utils.ARIMA_CANDIDATE_ORDERS
    resid = pd.Series(residuals).astype(float).dropna()
    if len(resid) < utils.MIN_RESIDUALS_FOR_ARIMA:
        return {"ok": False, "reason": f"only {len(resid)} residuals", "n_obs": len(resid)}
    if float(resid.std(ddof=0)) == 0.0:
        return {"ok": False, "reason": "residuals are constant", "n_obs": len(resid)}

    best = None
    tried = []
    for order in candidate_orders:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = ARIMA(
                    resid.to_numpy(), order=order, trend="c",
                    enforce_stationarity=True, enforce_invertibility=True,
                ).fit(method_kwargs={"warn_convergence": False})
            aic = float(res.aic)
            if not np.isfinite(aic):
                continue
            tried.append({"order": order, "aic": round(aic, 2)})
            if best is None or aic < best["aic"]:
                best = {"order": order, "aic": aic, "result": res}
        except Exception as exc:  # noqa: BLE001
            tried.append({"order": order, "error": type(exc).__name__})
            continue

    if best is None:
        return {"ok": False, "reason": "no candidate order converged", "n_obs": len(resid),
                "tried": tried}
    return {
        "ok": True,
        "order": best["order"],
        "aic": round(best["aic"], 2),
        "result": best["result"],
        "n_obs": int(len(resid)),
        "resid_std": float(resid.std(ddof=0)),
        "tried": tried,
    }


# --------------------------------------------------------------------------
class RandomForestARIMAHybrid:
    """Random Forest for the level, per-series ARIMA for the residual structure."""

    def __init__(
        self,
        rf_model: Optional[RandomForestRegressor] = None,
        min_residuals: int = utils.MIN_RESIDUALS_FOR_ARIMA,
        n_oof_splits: int = utils.N_CV_SPLITS,
        max_series: Optional[int] = None,
    ):
        self.rf_model = rf_model
        self.min_residuals = min_residuals
        self.n_oof_splits = n_oof_splits
        self.max_series = max_series  # cap for a fast run; None = every series

    # ----------------------------------------------------------------- fit
    def fit(
        self,
        train_df: pd.DataFrame,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        feature_columns: List[str],
        rf_already_fitted: bool = False,
    ) -> "RandomForestARIMAHybrid":
        """Fit the forest (unless supplied fitted) and an ARIMA per series.

        ``train_df`` must be row-aligned with ``X_train`` / ``y_train`` and carry
        ``series_id`` and ``month_index``.
        """
        self.feature_columns = list(feature_columns)

        if self.rf_model is None:
            self.rf_model = RandomForestRegressor(
                n_estimators=400, max_features=0.6, random_state=utils.RANDOM_STATE, n_jobs=-1
            )
            rf_already_fitted = False

        # Step 1-2: leakage-safe training residuals
        LOGGER.info("Hybrid step 1/4: out-of-fold Random Forest predictions on the training set")
        oof_pred, covered = out_of_fold_predictions(
            self.rf_model, X_train, y_train, n_splits=self.n_oof_splits
        )

        if not rf_already_fitted:
            LOGGER.info("Hybrid step 2/4: fitting the Random Forest on the full training set")
            self.rf_model.fit(X_train, np.asarray(y_train, float))

        resid_df = train_df.loc[:, [utils.SERIES_ID, "month_start", "month_index"]].copy()
        resid_df["actual"] = np.asarray(y_train, float)
        resid_df["rf_oof_pred"] = oof_pred
        resid_df["residual"] = resid_df["actual"] - resid_df["rf_oof_pred"]
        resid_df = resid_df[covered].sort_values([utils.SERIES_ID, "month_index"])

        self.training_residuals_ = resid_df
        self.n_oof_rows_ = int(covered.sum())
        self.n_rows_without_oof_ = int((~covered).sum())
        self.oof_residual_summary_ = {
            "n_residuals": int(len(resid_df)),
            "mean": float(resid_df["residual"].mean()),
            "std": float(resid_df["residual"].std(ddof=0)),
            "mae_of_oof_predictions": float(resid_df["residual"].abs().mean()),
        }
        self.forecast_origin_month_index_ = int(resid_df["month_index"].max())
        self.forecast_origin_month_ = str(
            pd.Timestamp(resid_df["month_start"].max()).date()
        )

        # Step 3: an ARIMA per series
        LOGGER.info(
            "Hybrid step 3/4: fitting ARIMA to the residuals of each series "
            "(minimum %d residuals)", self.min_residuals,
        )
        self.arima_info_: Dict[str, Dict] = {}
        self._arima_results: Dict[str, object] = {}
        self.series_last_residual_month_: Dict[str, int] = {}

        groups = list(resid_df.groupby(utils.SERIES_ID, observed=True))
        if self.max_series is not None:
            groups = sorted(groups, key=lambda kv: -len(kv[1]))[: self.max_series]

        n_fitted = n_skipped = 0
        for sid, grp in groups:
            grp = grp.sort_values("month_index")
            self.series_last_residual_month_[sid] = int(grp["month_index"].max())
            info = fit_arima_on_residuals(grp["residual"])
            if info.get("ok"):
                self._arima_results[sid] = info.pop("result")
                n_fitted += 1
            else:
                n_skipped += 1
            self.arima_info_[sid] = info

        self.n_series_with_arima_ = n_fitted
        self.n_series_without_arima_ = n_skipped
        LOGGER.info(
            "Hybrid step 4/4: ARIMA fitted for %d series; %d series fall back to "
            "Random Forest only (insufficient residual history).", n_fitted, n_skipped,
        )

        orders = [
            tuple(i["order"]) for i in self.arima_info_.values() if i.get("ok")
        ]
        self.arima_order_counts_ = (
            {str(o): int(c) for o, c in pd.Series(orders).value_counts().items()}
            if orders else {}
        )
        self.is_fitted_ = True
        return self

    # ------------------------------------------------------------- predict
    def _residual_forecast(self, sid: str, horizon: int) -> float:
        """``horizon``-step-ahead residual forecast for one series (0 if no model)."""
        res = self._arima_results.get(sid)
        if res is None or horizon < 1:
            return 0.0
        try:
            fc = res.forecast(steps=int(horizon))
            return float(np.asarray(fc)[-1])
        except Exception:  # noqa: BLE001
            return 0.0

    def predict(self, df: pd.DataFrame, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Predict for ``df`` (aligned with ``X``).

        Returns ``rf``, ``residual_forecast``, ``hybrid`` and ``corrected`` arrays.
        """
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("Hybrid model is not fitted yet.")

        rf_pred = self.rf_model.predict(X)

        sids = df[utils.SERIES_ID].to_numpy()
        month_idx = df["month_index"].to_numpy()
        resid_fc = np.zeros(len(df), dtype=float)
        corrected = np.zeros(len(df), dtype=bool)

        cache: Dict[Tuple[str, int], float] = {}
        for i, (sid, mi) in enumerate(zip(sids, month_idx)):
            if sid not in self._arima_results:
                continue
            origin = self.series_last_residual_month_.get(sid)
            if origin is None:
                continue
            horizon = int(mi) - int(origin)
            if horizon < 1:
                continue
            key = (sid, horizon)
            if key not in cache:
                cache[key] = self._residual_forecast(sid, horizon)
            resid_fc[i] = cache[key]
            corrected[i] = True

        return {
            "rf": rf_pred,
            "residual_forecast": resid_fc,
            "hybrid": rf_pred + resid_fc,
            "corrected": corrected,
        }

    # ------------------------------------------------------------- reports
    def summary(self) -> Dict:
        return {
            "model": "Hybrid: Random Forest (level) + per-series ARIMA (residuals)",
            "rf_params": {
                k: v for k, v in self.rf_model.get_params().items()
                if k in {"n_estimators", "max_depth", "max_features", "min_samples_leaf"}
            },
            "min_residuals_for_arima": self.min_residuals,
            "n_oof_folds": self.n_oof_splits,
            "n_training_rows_with_oof": self.n_oof_rows_,
            "n_training_rows_without_oof": self.n_rows_without_oof_,
            "oof_residual_summary": self.oof_residual_summary_,
            "forecast_origin_month": self.forecast_origin_month_,
            "n_series_with_arima": self.n_series_with_arima_,
            "n_series_without_arima": self.n_series_without_arima_,
            "arima_order_counts": self.arima_order_counts_,
            "candidate_orders": [list(o) for o in utils.ARIMA_CANDIDATE_ORDERS],
            "order_selection": "lowest AIC among candidates, all with d=0",
            "leakage_controls": [
                "Residuals come from TimeSeriesSplit out-of-fold predictions, not in-sample fits.",
                "ARIMA is fitted to training residuals only.",
                "Forecast origin is the last training month; test month k uses the k-step forecast.",
                "No test or validation actual is used to produce the correction.",
            ],
        }

    def arima_table(self) -> pd.DataFrame:
        """One row per series: whether ARIMA was fitted, its order and AIC."""
        rows = []
        for sid, info in self.arima_info_.items():
            rows.append({
                "series_id": sid,
                "arima_fitted": bool(info.get("ok", False)),
                "order": str(info.get("order")) if info.get("ok") else None,
                "aic": info.get("aic"),
                "n_residuals": info.get("n_obs"),
                "resid_std": info.get("resid_std"),
                "reason_skipped": None if info.get("ok") else info.get("reason"),
            })
        return pd.DataFrame(rows).sort_values(
            ["arima_fitted", "n_residuals"], ascending=[False, False]
        ).reset_index(drop=True)

    # ARIMA result objects hold references that pickle poorly in bulk; keep the
    # parameters needed to re-forecast instead of the full statsmodels objects.
    def __getstate__(self):
        state = self.__dict__.copy()
        results = state.pop("_arima_results", {})
        state["_arima_forecast_cache"] = {
            sid: np.asarray(res.forecast(steps=24), dtype=float).tolist()
            for sid, res in results.items()
        }
        return state

    def __setstate__(self, state):
        cache = state.pop("_arima_forecast_cache", {})
        self.__dict__.update(state)
        self._arima_results = {}
        self._forecast_cache = {sid: np.asarray(v) for sid, v in cache.items()}

        def _lookup(sid, horizon):
            arr = self._forecast_cache.get(sid)
            if arr is None or horizon < 1 or horizon > len(arr):
                return 0.0
            return float(arr[horizon - 1])

        self._residual_forecast = _lookup  # type: ignore[assignment]
        # restore membership checks used by predict()
        self._arima_results = dict.fromkeys(self._forecast_cache.keys())
