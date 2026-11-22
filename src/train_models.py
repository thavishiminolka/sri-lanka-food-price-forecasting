"""
train_models.py
===============
Trains the three tabular ensembles required by the brief:

A. **Bagging - RandomForestRegressor.**  Bagging ("bootstrap aggregating") grows
   many deep decision trees, each on a bootstrap resample of the training rows
   and each choosing its splits from a random subset of the features.  The trees
   are grown in parallel and are deliberately decorrelated; the forest's
   prediction is their average.  Averaging many high-variance, low-bias trees
   cuts variance without adding bias, which is why a forest generalises far
   better than any single tree.

B. **Boosting - XGBRegressor.**  Boosting builds trees *sequentially*.  Each new
   tree is fitted to the gradient of the loss left over from the trees already in
   the ensemble, so every tree concentrates on the rows the current model gets
   wrong.  XGBoost adds second-order (Newton) steps, L1/L2 penalties on leaf
   weights, shrinkage via the learning rate, and column/row subsampling.  The
   result is a sum of many shallow trees that reduces bias step by step, where
   bagging reduces variance all at once.

C. **Stacking - RF + XGBoost + LinearRegression, Ridge meta-learner.**  Stacking
   trains several different algorithms on the same data, then trains a second
   model (the meta-learner) on their predictions to learn how much to trust each
   one.  The meta-learner must be trained on *out-of-fold* predictions,
   otherwise it learns from base-model predictions of rows those models have
   already memorised, which inflates validation scores and leaks.

   ``sklearn.ensemble.StackingRegressor`` cannot be used directly here: it calls
   ``cross_val_predict`` internally, which rejects ``TimeSeriesSplit`` with
   "cross_val_predict only works for partitions" because the earliest fold is
   never in a test set.  :class:`TimeAwareStackingRegressor` below therefore
   implements the same algorithm with an expanding-window split, so the
   meta-learner only ever sees predictions for months that came *after* the
   months its base learners were fitted on.

Hyperparameter tuning uses ``GridSearchCV`` with ``TimeSeriesSplit`` so that
every validation fold is later in time than its training fold.  The held-out
test set is never touched during tuning.  ``FAST_MODE`` shrinks the grids for a
quick end-to-end run.
"""
from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

try:
    from . import utils
except ImportError:  # pragma: no cover
    import utils

LOGGER = utils.LOGGER

# Set FOOD_PRICE_FAST=1 to use the reduced grids (recommended on Colab's free tier).
FAST_MODE = os.environ.get("FOOD_PRICE_FAST", "0") == "1"


# ==========================================================================
# Time-aware stacking
# ==========================================================================
class TimeAwareStackingRegressor(BaseEstimator, RegressorMixin):
    """Stacking regressor whose meta-learner is trained on time-ordered OOF predictions.

    Fitting procedure
    -----------------
    1. Split the (chronologically ordered) training rows with
       ``TimeSeriesSplit``, giving expanding train folds and later test folds.
    2. For each fold, fit a fresh copy of every base learner on the fold's
       training part and predict its test part.  Those predictions are the
       out-of-fold (OOF) matrix; rows in the very first fold get no prediction
       and are excluded from the meta-learner's training data.
    3. Fit the meta-learner on the OOF matrix.
    4. Refit every base learner on the whole training set, for use at predict
       time.

    Because folds are time-ordered, no OOF prediction for month *m* is ever
    produced by a base learner that had seen month *m* or anything later.

    Parameters
    ----------
    estimators
        ``[(name, estimator), ...]`` - the base learners.
    final_estimator
        The meta-learner.  Ridge regression here: the base-model predictions are
        highly collinear (they all predict the same price), and Ridge's L2
        penalty keeps the blending weights stable instead of letting them swing
        to large opposing values the way ordinary least squares does.
    cv
        Number of ``TimeSeriesSplit`` folds.
    passthrough
        If True, the original features are concatenated with the base-model
        predictions before the meta-learner.  Default False - with 34 features
        and 3 predictions, passthrough lets the meta-learner re-learn the whole
        problem and usually overfits.
    """

    def __init__(
        self,
        estimators: List[Tuple[str, BaseEstimator]],
        final_estimator: Optional[BaseEstimator] = None,
        cv: int = utils.N_CV_SPLITS,
        passthrough: bool = False,
    ):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv
        self.passthrough = passthrough

    # -- sklearn plumbing --------------------------------------------------
    def get_params(self, deep=True):
        return {
            "estimators": self.estimators,
            "final_estimator": self.final_estimator,
            "cv": self.cv,
            "passthrough": self.passthrough,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    # -- fit / predict -----------------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples = X.shape[0]

        self.names_ = [name for name, _ in self.estimators]
        self.final_estimator_ = clone(
            self.final_estimator if self.final_estimator is not None else Ridge(alpha=1.0)
        )

        splitter = TimeSeriesSplit(n_splits=self.cv)
        oof = np.full((n_samples, len(self.estimators)), np.nan)

        for fold, (tr_idx, te_idx) in enumerate(splitter.split(X), start=1):
            for j, (name, est) in enumerate(self.estimators):
                model = clone(est)
                model.fit(X[tr_idx], y[tr_idx])
                oof[te_idx, j] = model.predict(X[te_idx])
            LOGGER.info(
                "    stacking OOF fold %d/%d: fitted on %d rows, predicted %d rows",
                fold, self.cv, len(tr_idx), len(te_idx),
            )

        covered = ~np.isnan(oof).any(axis=1)
        self.n_oof_rows_ = int(covered.sum())
        self.n_rows_without_oof_ = int((~covered).sum())

        # Kept so that the meta-learner can be re-tuned without rebuilding the
        # (expensive) out-of-fold matrix.
        self.oof_ = oof
        self.oof_mask_ = covered
        self.X_train_ = X
        self.y_train_ = y

        meta_X = oof[covered]
        if self.passthrough:
            meta_X = np.hstack([meta_X, X[covered]])
        self.final_estimator_.fit(meta_X, y[covered])

        # Refit every base learner on the full training set.
        self.fitted_estimators_ = []
        for name, est in self.estimators:
            model = clone(est)
            model.fit(X, y)
            self.fitted_estimators_.append((name, model))

        self._read_meta_coefficients()
        self.is_fitted_ = True
        return self

    def _read_meta_coefficients(self) -> None:
        """Pull coef_/intercept_ out of the meta-learner, Pipeline or bare estimator."""
        est = self.final_estimator_
        if isinstance(est, Pipeline):
            est = est[-1]
        self.meta_coef_ = getattr(est, "coef_", None)
        self.meta_intercept_ = getattr(est, "intercept_", None)

    def refit_meta(self, alpha: float) -> "TimeAwareStackingRegressor":
        """Refit only the Ridge meta-learner on the stored out-of-fold matrix.

        The out-of-fold predictions do not depend on the meta-learner's alpha, so
        searching over alpha costs one Ridge fit per candidate rather than a full
        rebuild of every base learner.
        """
        if not hasattr(self, "oof_"):
            raise RuntimeError("Call fit() before refit_meta().")
        meta = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
        meta_X = self.oof_[self.oof_mask_]
        if self.passthrough:
            meta_X = np.hstack([meta_X, self.X_train_[self.oof_mask_]])
        meta.fit(meta_X, self.y_train_[self.oof_mask_])
        self.final_estimator_ = meta
        self.ridge_alpha_ = float(alpha)
        self._read_meta_coefficients()
        return self

    def _base_predictions(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.column_stack([m.predict(X) for _, m in self.fitted_estimators_])

    def predict(self, X):
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("TimeAwareStackingRegressor is not fitted yet.")
        X = np.asarray(X, dtype=float)
        meta_X = self._base_predictions(X)
        if self.passthrough:
            meta_X = np.hstack([meta_X, X])
        return self.final_estimator_.predict(meta_X)

    def base_predictions_frame(self, X) -> pd.DataFrame:
        """Base-learner predictions as a labelled frame (used in the dashboard)."""
        return pd.DataFrame(self._base_predictions(X), columns=self.names_)

    def meta_weights(self) -> Dict[str, float]:
        """Fitted Ridge blending weight per base learner."""
        if self.meta_coef_ is None:
            return {}
        coefs = np.ravel(self.meta_coef_)[: len(self.names_)]
        out = {n: float(c) for n, c in zip(self.names_, coefs)}
        out["intercept"] = float(np.ravel(self.meta_intercept_)[0]) if self.meta_intercept_ is not None else 0.0
        return out


# ==========================================================================
# Hyperparameter grids
# ==========================================================================
def rf_param_grid() -> Dict[str, list]:
    """Random Forest grid.

    Deliberately compact. A single 300-tree fit on the 18,428 training rows takes
    about 25 seconds on two cores, so every parameter value added multiplies real
    minutes. The values kept are the ones that actually change a forest's
    behaviour on this data: tree depth (whether to let trees grow out fully),
    leaf size (how hard to stop them memorising single months), and the share of
    features each split may consider (how decorrelated the trees are). The number
    of trees is fixed at 300 because forest error flattens off well before that
    here - more trees cost time without changing the prediction.
    """
    if FAST_MODE:
        return {"n_estimators": [200], "max_depth": [None], "min_samples_leaf": [1],
                "max_features": [0.4, 0.7]}
    return {
        "n_estimators": [300],
        "max_depth": [None, 20],
        "min_samples_leaf": [1, 2],
        "max_features": [0.4, 0.7],
    }


def xgb_param_grid() -> Dict[str, list]:
    if FAST_MODE:
        return {"n_estimators": [400], "learning_rate": [0.05, 0.1], "max_depth": [6],
                "subsample": [0.8], "colsample_bytree": [0.8], "min_child_weight": [1],
                "reg_lambda": [1.0]}
    # XGBoost with tree_method="hist" fits this table in about 1.5 seconds, so a
    # much wider grid is affordable here than for the forest. The three
    # parameters that interact most - learning rate, depth and the number of
    # boosting rounds - are all varied, together with the two subsampling
    # controls and the minimum child weight that guard against overfitting the
    # short series.
    return {
        "n_estimators": [400, 800],
        "learning_rate": [0.03, 0.06, 0.1],
        "max_depth": [4, 6, 8],
        "subsample": [0.8],
        "colsample_bytree": [0.6, 0.9],
        "min_child_weight": [1, 5],
        "reg_lambda": [1.0],
    }


def ridge_alpha_grid() -> List[float]:
    """Ridge alphas for the stacking meta-learner. Cheap: only the meta-learner is
    refitted per value, not the base learners."""
    return [0.1, 1.0] if FAST_MODE else [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


# ==========================================================================
# Training helpers
# ==========================================================================
def _tscv(n_splits: int = utils.TUNING_CV_SPLITS) -> TimeSeriesSplit:
    return TimeSeriesSplit(n_splits=n_splits)


def _run_grid_search(estimator, grid, X, y, label: str, n_jobs: int = 1) -> Dict:
    """GridSearchCV with TimeSeriesSplit, scored by negative RMSE."""
    n_combos = int(np.prod([len(v) for v in grid.values()]))
    LOGGER.info(
        "Tuning %s: %d parameter combinations x %d time-series folds = %d fits%s",
        label, n_combos, utils.TUNING_CV_SPLITS, n_combos * utils.TUNING_CV_SPLITS,
        " [FAST_MODE]" if FAST_MODE else "",
    )
    t0 = time.time()
    gs = GridSearchCV(
        estimator=estimator,
        param_grid=grid,
        scoring="neg_root_mean_squared_error",
        cv=_tscv(),
        n_jobs=n_jobs,
        refit=True,
        verbose=0,
    )
    gs.fit(X, y)
    elapsed = time.time() - t0
    LOGGER.info(
        "  %s best CV RMSE = %.3f LKR | params = %s | %.1fs",
        label, -gs.best_score_, gs.best_params_, elapsed,
    )
    return {
        "best_estimator": gs.best_estimator_,
        "best_params": gs.best_params_,
        "best_cv_rmse": float(-gs.best_score_),
        "n_combinations": n_combos,
        "cv_folds": utils.TUNING_CV_SPLITS,
        "tuning_seconds": round(elapsed, 1),
        "cv_results_summary": _summarise_cv(gs),
    }


def _summarise_cv(gs: GridSearchCV, top: int = 5) -> List[Dict]:
    res = pd.DataFrame(gs.cv_results_)
    cols = ["params", "mean_test_score", "std_test_score", "rank_test_score"]
    res = res[cols].sort_values("rank_test_score").head(top)
    return [
        {
            "params": r["params"],
            "cv_rmse": round(float(-r["mean_test_score"]), 4),
            "cv_rmse_std": round(float(r["std_test_score"]), 4),
            "rank": int(r["rank_test_score"]),
        }
        for _, r in res.iterrows()
    ]


# --------------------------------------------------------------------------
def train_random_forest(X_train, y_train, tune: bool = True) -> Dict:
    """A. Bagging - Random Forest."""
    base = RandomForestRegressor(
        random_state=utils.RANDOM_STATE, n_jobs=-1, bootstrap=True, oob_score=False
    )
    if not tune:
        model = clone(base).set_params(n_estimators=400, max_features=0.6, min_samples_leaf=1)
        model.fit(X_train, y_train)
        return {"model": model, "best_params": model.get_params(), "tuned": False}
    out = _run_grid_search(base, rf_param_grid(), X_train, y_train, "RandomForest", n_jobs=1)
    return {"model": out.pop("best_estimator"), "tuned": True, **out}


def train_xgboost(X_train, y_train, tune: bool = True) -> Dict:
    """B. Boosting - XGBoost."""
    base = XGBRegressor(
        random_state=utils.RANDOM_STATE,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=2,
        verbosity=0,
    )
    if not tune:
        model = clone(base).set_params(
            n_estimators=600, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
        )
        model.fit(X_train, y_train)
        return {"model": model, "best_params": model.get_params(), "tuned": False}
    out = _run_grid_search(base, xgb_param_grid(), X_train, y_train, "XGBoost", n_jobs=1)
    return {"model": out.pop("best_estimator"), "tuned": True, **out}


def build_stacking_regressor(
    rf_params: Optional[Dict] = None,
    xgb_params: Optional[Dict] = None,
    ridge_alpha: float = 1.0,
) -> TimeAwareStackingRegressor:
    """C. Stacking - the exact configuration the brief specifies.

    Base learners : RandomForestRegressor, XGBRegressor, LinearRegression
    Meta-learner  : Ridge

    The LinearRegression base learner sits inside a ``StandardScaler`` pipeline,
    and so does the Ridge meta-learner: both are penalised/scale-sensitive
    linear models, and the raw features mix prices in the hundreds with
    percentage changes and 0/1 flags.  The tree learners need no scaling.
    """
    rf_kwargs = dict(n_estimators=400, max_features=0.6, min_samples_leaf=1)
    rf_kwargs.update(rf_params or {})
    xgb_kwargs = dict(
        n_estimators=600, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
    )
    xgb_kwargs.update(xgb_params or {})

    estimators = [
        ("random_forest", RandomForestRegressor(
            random_state=utils.RANDOM_STATE, n_jobs=-1, **rf_kwargs)),
        ("xgboost", XGBRegressor(
            random_state=utils.RANDOM_STATE, objective="reg:squarederror",
            tree_method="hist", n_jobs=2, verbosity=0, **xgb_kwargs)),
        ("linear_regression", Pipeline([
            ("scaler", StandardScaler()),
            ("linear", LinearRegression()),
        ])),
    ]
    meta = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=ridge_alpha))])
    return TimeAwareStackingRegressor(
        estimators=estimators, final_estimator=meta, cv=utils.N_CV_SPLITS, passthrough=False
    )


def train_stacking(
    X_train, y_train, X_val, y_val,
    rf_params: Optional[Dict] = None,
    xgb_params: Optional[Dict] = None,
    tune_alpha: bool = True,
) -> Dict:
    """Fit the stacking ensemble, choosing the Ridge alpha on the validation split.

    The alpha search uses the chronological validation months (never the test
    months), and each candidate is a complete refit so the OOF matrix is rebuilt
    with that alpha's meta-learner.
    """
    alphas = ridge_alpha_grid() if tune_alpha else [1.0]

    LOGGER.info(
        "Fitting the stacking ensemble (base learners + time-ordered OOF matrix) once"
    )
    model = build_stacking_regressor(rf_params, xgb_params, ridge_alpha=alphas[0])
    model.fit(X_train, y_train)

    # The base learners are fixed from here, so their validation predictions are
    # computed once and only the Ridge meta-learner is re-fitted per alpha.
    val_base = model._base_predictions(X_val)
    results = []
    best = None
    for alpha in alphas:
        model.refit_meta(alpha)
        meta_X = np.hstack([val_base, np.asarray(X_val, float)]) if model.passthrough else val_base
        val_rmse = utils.rmse(y_val, model.final_estimator_.predict(meta_X))
        results.append({"alpha": alpha, "val_rmse": round(val_rmse, 4)})
        LOGGER.info("  Ridge alpha=%g -> validation RMSE %.3f LKR", alpha, val_rmse)
        if best is None or val_rmse < best["val_rmse"]:
            best = {"alpha": alpha, "val_rmse": val_rmse}

    model.refit_meta(best["alpha"])
    LOGGER.info(
        "Stacking selected alpha=%g (validation RMSE %.3f). Meta weights: %s",
        best["alpha"], best["val_rmse"], model.meta_weights(),
    )
    return {
        "model": model,
        "tuned": tune_alpha,
        "best_params": {"ridge_alpha": best["alpha"]},
        "alpha_search": results,
        "meta_weights": model.meta_weights(),
        "n_oof_rows": model.n_oof_rows_,
        "n_rows_without_oof": model.n_rows_without_oof_,
        "base_learners": model.names_,
        "leakage_control": (
            "Meta-learner trained only on TimeSeriesSplit out-of-fold predictions; "
            f"{model.n_rows_without_oof_} earliest training rows had no OOF prediction "
            "and were excluded from meta-training."
        ),
    }


# --------------------------------------------------------------------------
def feature_importances(model, feature_names: List[str]) -> pd.DataFrame:
    """Impurity/gain-based importance, sorted, as a tidy frame."""
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float)
    else:
        raise AttributeError(f"{type(model).__name__} exposes no feature_importances_")
    return (
        pd.DataFrame({"feature": feature_names, "importance": imp})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def permutation_importance_df(
    model, X, y, feature_names: List[str], n_repeats: int = 5,
    random_state: int = utils.RANDOM_STATE, scoring: str = "neg_root_mean_squared_error",
) -> pd.DataFrame:
    """Permutation importance on held-out data (model-agnostic, no refit)."""
    from sklearn.inspection import permutation_importance

    res = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=random_state,
        scoring=scoring, n_jobs=1,
    )
    return (
        pd.DataFrame({
            "feature": feature_names,
            "importance_mean": res.importances_mean,
            "importance_std": res.importances_std,
        })
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
