"""
evaluate_models.py
==================
Evaluation and comparison of the four selected ensemble approaches.

Only these four enter the comparison:

1. Bagging  - Random Forest Regressor
2. Boosting - XGBoost Regressor
3. Stacking - RF + XGBoost + LinearRegression, Ridge meta-learner
4. Hybrid   - Random Forest + ARIMA

Standalone naive, seasonal-naive, linear-regression and ARIMA models are **not**
reported as competitors.  Linear Regression appears only as a stacking base
learner and ARIMA only as the hybrid's residual component.

All four are scored on exactly the same test rows (the chronological test split),
so the comparison table is like-for-like.  Where a model's coverage differs -
the hybrid only applies an ARIMA correction to series with enough residual
history - that is reported alongside the metrics instead of changing the row set.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from . import utils
except ImportError:  # pragma: no cover
    import utils

LOGGER = utils.LOGGER

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = {
    "random_forest": "#2E7D32",
    "xgboost": "#1565C0",
    "stacking": "#6A1B9A",
    "hybrid": "#E65100",
}


# --------------------------------------------------------------------------
def build_comparison_table(
    y_true: Iterable[float],
    predictions: Dict[str, np.ndarray],
    labels: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Metrics for every model on the same test rows, best MAE first."""
    labels = labels or utils.MODEL_LABELS
    y_true = np.asarray(list(y_true), dtype=float)
    rows = []
    for key in utils.MODEL_ORDER:
        if key not in predictions:
            continue
        pred = np.asarray(predictions[key], dtype=float)
        if len(pred) != len(y_true):
            raise ValueError(
                f"{key}: {len(pred)} predictions for {len(y_true)} test rows - "
                "models must be scored on the same rows."
            )
        m = utils.regression_metrics(y_true, pred)
        rows.append({
            "model_key": key,
            "Approach": labels.get(key, key),
            "MAE": m["MAE"],
            "RMSE": m["RMSE"],
            "R2": m["R2"],
            "MAPE_pct": m["MAPE_pct"],
            "n_test_obs": m["n_obs"],
        })
    table = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    table.insert(0, "Rank_by_MAE", np.arange(1, len(table) + 1))
    return table


def identify_best_model(table: pd.DataFrame, metric: str = "MAE") -> Dict:
    """Pick the best model from the actual results (lower MAE/RMSE, higher R2)."""
    ascending = metric in {"MAE", "RMSE", "MAPE_pct"}
    ordered = table.sort_values(metric, ascending=ascending).reset_index(drop=True)
    best, runner = ordered.iloc[0], ordered.iloc[1] if len(ordered) > 1 else None
    out = {
        "metric": metric,
        "best_model_key": best["model_key"],
        "best_model": best["Approach"],
        "best_value": float(best[metric]),
        "all_metrics": {
            "MAE": float(best["MAE"]), "RMSE": float(best["RMSE"]),
            "R2": float(best["R2"]), "MAPE_pct": float(best["MAPE_pct"]),
        },
    }
    if runner is not None:
        gap = float(runner[metric]) - float(best[metric])
        out["runner_up"] = runner["Approach"]
        out["runner_up_value"] = float(runner[metric])
        out["margin"] = gap
        out["margin_pct"] = round(100 * gap / float(runner[metric]), 2) if runner[metric] else None
    return out


def per_month_metrics(
    test_df: pd.DataFrame, y_true, predictions: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """Metrics broken down by test month - shows how error grows with horizon."""
    y_true = np.asarray(y_true, dtype=float)
    frame = test_df[["month_start"]].copy().reset_index(drop=True)
    frame["actual"] = y_true
    for k, v in predictions.items():
        frame[k] = np.asarray(v, dtype=float)
    rows = []
    for month, grp in frame.groupby("month_start"):
        for k in predictions:
            m = utils.regression_metrics(grp["actual"], grp[k])
            rows.append({
                "month": pd.Timestamp(month).date(), "model_key": k,
                "Approach": utils.MODEL_LABELS.get(k, k),
                "MAE": m["MAE"], "RMSE": m["RMSE"], "R2": m["R2"], "n_obs": m["n_obs"],
            })
    return pd.DataFrame(rows)


def per_group_metrics(
    test_df: pd.DataFrame, y_true, predictions: Dict[str, np.ndarray],
    group_col: str = "commodity", min_obs: int = 20,
) -> pd.DataFrame:
    """Metrics per commodity / market / category, for the error analysis section."""
    y_true = np.asarray(y_true, dtype=float)
    frame = test_df[[group_col, "unit"]].copy().reset_index(drop=True)
    frame["actual"] = y_true
    for k, v in predictions.items():
        frame[k] = np.asarray(v, dtype=float)
    rows = []
    for name, grp in frame.groupby(group_col, observed=True):
        if len(grp) < min_obs:
            continue
        for k in predictions:
            m = utils.regression_metrics(grp["actual"], grp[k])
            rows.append({
                group_col: name,
                "units": "/".join(sorted(grp["unit"].astype(str).unique())),
                "model_key": k, "Approach": utils.MODEL_LABELS.get(k, k),
                "MAE": m["MAE"], "RMSE": m["RMSE"], "R2": m["R2"],
                "MAPE_pct": m["MAPE_pct"], "n_obs": m["n_obs"],
                "mean_actual_price": float(grp["actual"].mean()),
            })
    return pd.DataFrame(rows)


def residual_frame(
    test_df: pd.DataFrame, y_true, predictions: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """Tidy actual/predicted/residual table used by the residual plots and dashboard."""
    y_true = np.asarray(y_true, dtype=float)
    base_cols = [c for c in [utils.SERIES_ID, "month_start", "commodity", "market",
                             "unit", "pricetype", "category", "admin1"]
                 if c in test_df.columns]
    base = test_df[base_cols].copy().reset_index(drop=True)
    base["actual"] = y_true
    out = []
    for k, v in predictions.items():
        part = base.copy()
        part["model_key"] = k
        part["Approach"] = utils.MODEL_LABELS.get(k, k)
        part["predicted"] = np.asarray(v, dtype=float)
        part["residual"] = part["actual"] - part["predicted"]
        part["abs_error"] = part["residual"].abs()
        part["pct_error"] = np.where(
            part["actual"] != 0, part["residual"] / part["actual"] * 100, np.nan
        )
        out.append(part)
    return pd.concat(out, ignore_index=True)


# ==========================================================================
# Figures
# ==========================================================================
def _save(fig, name: str, outdir: Optional[Path] = None) -> Path:
    outdir = Path(outdir or utils.FIGURES_DIR)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("figure saved: %s", path.name)
    return path


def plot_metric_comparison(
    table: pd.DataFrame, metric: str, outdir: Optional[Path] = None
) -> Path:
    """One bar chart per metric, as the brief requires (MAE, RMSE and R2 separately)."""
    ascending = metric in {"MAE", "RMSE", "MAPE_pct"}
    data = table.sort_values(metric, ascending=ascending)
    colors = [PALETTE.get(k, "#555") for k in data["model_key"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(data["Approach"], data[metric], color=colors, edgecolor="black", linewidth=0.6)
    unit = {"MAE": "LKR", "RMSE": "LKR", "R2": "", "MAPE_pct": "%"}[metric]
    pretty = {"MAE": "Mean Absolute Error", "RMSE": "Root Mean Squared Error",
              "R2": "R-squared", "MAPE_pct": "Mean Absolute Percentage Error"}[metric]
    ax.set_title(
        f"{pretty} on the chronological test set (2025-04 to 2025-09)\n"
        f"Four ensemble approaches, {int(data['n_test_obs'].iloc[0]):,} test observations",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylabel(f"{pretty}" + (f" ({unit})" if unit else ""))
    ax.set_xlabel("Ensemble approach")
    ax.tick_params(axis="x", labelrotation=18)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    span = float(data[metric].max() - data[metric].min()) or 1.0
    for bar, val in zip(bars, data[metric]):
        ax.annotate(
            f"{val:,.3f}" if metric == "R2" else f"{val:,.2f}",
            (bar.get_x() + bar.get_width() / 2, val),
            textcoords="offset points", xytext=(0, 4), ha="center", fontsize=10,
            fontweight="bold",
        )
    if metric == "R2":
        ax.set_ylim(max(0, float(data[metric].min()) - 0.05 * span), 1.0)
    fig.tight_layout()
    return _save(fig, f"model_comparison_{metric.lower()}.png", outdir)


def plot_all_metric_comparisons(table: pd.DataFrame, outdir: Optional[Path] = None) -> List[Path]:
    return [plot_metric_comparison(table, m, outdir) for m in ("MAE", "RMSE", "R2", "MAPE_pct")]


def plot_actual_vs_predicted(
    y_true, predictions: Dict[str, np.ndarray], outdir: Optional[Path] = None,
    sample: Optional[int] = 4000, random_state: int = utils.RANDOM_STATE,
) -> Path:
    """2x2 scatter of actual against predicted, one panel per approach."""
    y_true = np.asarray(y_true, dtype=float)
    keys = [k for k in utils.MODEL_ORDER if k in predictions]
    idx = np.arange(len(y_true))
    if sample is not None and len(idx) > sample:
        idx = np.random.default_rng(random_state).choice(idx, sample, replace=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    lo, hi = float(y_true.min()), float(y_true.max())
    for ax, key in zip(axes.ravel(), keys):
        pred = np.asarray(predictions[key], dtype=float)
        ax.scatter(y_true[idx], pred[idx], s=7, alpha=0.3,
                   color=PALETTE.get(key, "#555"), edgecolors="none")
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="perfect prediction")
        m = utils.regression_metrics(y_true, pred)
        ax.set_title(
            f"{utils.MODEL_LABELS[key]}\nMAE {m['MAE']:,.2f} | RMSE {m['RMSE']:,.2f} | "
            f"R² {m['R2']:.3f}", fontsize=11, fontweight="bold",
        )
        ax.set_xlabel("Actual price (LKR, per the row's own unit)")
        ax.set_ylabel("Predicted price (LKR)")
        ax.legend(loc="upper left", fontsize=9)
    for ax in axes.ravel()[len(keys):]:
        ax.axis("off")
    fig.suptitle(
        "Actual versus predicted food prices - chronological test set (2025-04 to 2025-09)",
        fontsize=13, fontweight="bold", y=0.995,
    )
    fig.tight_layout()
    return _save(fig, "actual_vs_predicted.png", outdir)


def plot_residual_analysis(
    resid_df: pd.DataFrame, outdir: Optional[Path] = None
) -> Path:
    """Residuals against fitted values, plus the error distribution, per model."""
    keys = [k for k in utils.MODEL_ORDER if k in resid_df["model_key"].unique()]
    fig, axes = plt.subplots(2, len(keys), figsize=(4.6 * len(keys), 9))
    if len(keys) == 1:
        axes = axes.reshape(2, 1)
    for j, key in enumerate(keys):
        d = resid_df[resid_df["model_key"] == key]
        ax = axes[0, j]
        ax.scatter(d["predicted"], d["residual"], s=6, alpha=0.3,
                   color=PALETTE.get(key, "#555"), edgecolors="none")
        ax.axhline(0, color="black", lw=1.1, ls="--")
        ax.set_title(utils.MODEL_LABELS[key], fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted price (LKR)")
        ax.set_ylabel("Residual (actual - predicted, LKR)")

        ax = axes[1, j]
        ax.hist(d["residual"], bins=60, color=PALETTE.get(key, "#555"),
                alpha=0.85, edgecolor="black", linewidth=0.3)
        ax.axvline(0, color="black", lw=1.1, ls="--")
        ax.set_title(
            f"mean {d['residual'].mean():,.2f} | sd {d['residual'].std(ddof=0):,.2f} LKR",
            fontsize=10,
        )
        ax.set_xlabel("Residual (LKR)")
        ax.set_ylabel("Number of test observations")
    fig.suptitle(
        "Residual analysis on the chronological test set - a flat band around zero "
        "means no systematic bias",
        fontsize=13, fontweight="bold", y=0.995,
    )
    fig.tight_layout()
    return _save(fig, "residual_analysis.png", outdir)


def plot_error_by_month(per_month: pd.DataFrame, outdir: Optional[Path] = None) -> Path:
    """MAE per test month - how prediction error behaves as the horizon lengthens."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for key, grp in per_month.groupby("model_key"):
        grp = grp.sort_values("month")
        ax.plot(grp["month"].astype(str), grp["MAE"], marker="o",
                label=utils.MODEL_LABELS.get(key, key), color=PALETTE.get(key, "#555"), lw=2)
    ax.set_title(
        "Mean absolute error by test month - all four approaches\n"
        "Models are trained to 2024-12, so later months are further from the training data",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Test month")
    ax.set_ylabel("MAE (LKR)")
    ax.legend(title="Approach", fontsize=9)
    ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    return _save(fig, "error_by_test_month.png", outdir)


def plot_feature_importance(
    importance_df: pd.DataFrame, title: str, filename: str,
    top_n: int = 20, outdir: Optional[Path] = None,
    value_col: str = "importance", color: str = "#2E7D32",
) -> Path:
    d = importance_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(5, 0.38 * len(d))))
    ax.barh(d["feature"], d[value_col], color=color, edgecolor="black", linewidth=0.5)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    return _save(fig, filename, outdir)


def plot_series_actual_vs_pred(
    resid_df: pd.DataFrame, series_id: str, outdir: Optional[Path] = None
) -> Optional[Path]:
    """Test-period actual and predicted lines for one commodity-market series."""
    d = resid_df[resid_df[utils.SERIES_ID] == series_id]
    if d.empty:
        return None
    unit = d["unit"].iloc[0]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    actual = d.drop_duplicates(subset=["month_start"]).sort_values("month_start")
    ax.plot(actual["month_start"], actual["actual"], "o-", color="black",
            lw=2.4, label="Actual price", zorder=5)
    for key, grp in d.groupby("model_key"):
        grp = grp.sort_values("month_start")
        ax.plot(grp["month_start"], grp["predicted"], "--", marker="s", ms=4,
                label=utils.MODEL_LABELS.get(key, key), color=PALETTE.get(key, "#555"))
    ax.set_title(f"Actual versus predicted - {series_id}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel(f"Price (LKR per {unit})")
    ax.legend(fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    safe = "".join(c if c.isalnum() else "_" for c in series_id)[:60]
    return _save(fig, f"series_actual_vs_pred_{safe}.png", outdir)


# ==========================================================================
def interpret_comparison(table: pd.DataFrame, hybrid_summary: Optional[Dict] = None) -> str:
    """Narrative written from the numbers in ``table`` - no result is hard-coded."""
    t = table.set_index("model_key")
    best = identify_best_model(table, "MAE")
    lines: List[str] = []
    lines.append(
        f"On the held-out test months the best approach by MAE is "
        f"**{best['best_model']}** (MAE {best['best_value']:,.2f} LKR, "
        f"RMSE {best['all_metrics']['RMSE']:,.2f}, R² {best['all_metrics']['R2']:.3f})."
    )
    if "runner_up" in best:
        lines.append(
            f"It is {best['margin']:,.2f} LKR ({best['margin_pct']:.1f}%) ahead of the "
            f"next approach, {best['runner_up']}."
        )
    best_rmse = identify_best_model(table, "RMSE")
    if best_rmse["best_model_key"] != best["best_model_key"]:
        lines.append(
            f"The ranking is metric-dependent: {best_rmse['best_model']} wins on RMSE "
            f"({best_rmse['best_value']:,.2f}). Because RMSE squares the errors, that "
            "means it keeps its largest misses smaller, while the MAE winner is better "
            "on the typical observation."
        )
    else:
        lines.append(
            "MAE and RMSE agree on the ranking, so the winner is better both on the "
            "typical observation and on its worst cases."
        )

    if {"random_forest", "xgboost"}.issubset(t.index):
        rf_mae, xgb_mae = float(t.loc["random_forest", "MAE"]), float(t.loc["xgboost", "MAE"])
        if xgb_mae < rf_mae:
            lines.append(
                f"Boosting beats bagging here ({xgb_mae:,.2f} against {rf_mae:,.2f} LKR MAE): "
                "sequentially correcting residuals extracts more from the lag features than "
                "averaging independent bootstrap trees does."
            )
        else:
            lines.append(
                f"Bagging beats boosting here ({rf_mae:,.2f} against {xgb_mae:,.2f} LKR MAE). "
                "With a panel this heterogeneous - 42 commodities across 44 markets and four "
                "units - variance reduction by averaging is worth more than sequential bias "
                "reduction, which can chase noise in the short, thin series."
            )
    if {"stacking"}.issubset(t.index):
        st = float(t.loc["stacking", "MAE"])
        others = {k: float(t.loc[k, "MAE"]) for k in ("random_forest", "xgboost") if k in t.index}
        if others and st < min(others.values()):
            lines.append(
                "Stacking improves on both of its own tree base learners, which is the "
                "expected payoff: the Ridge meta-learner learns a weighting that exploits "
                "where the forest and the booster disagree."
            )
        elif others:
            lines.append(
                "Stacking does not beat its best base learner. That happens when the base "
                "models' errors are highly correlated - a blend of two models that are wrong "
                "on the same rows cannot fix those rows - and when the linear base learner "
                "drags the blend on the extreme prices."
            )
    if "hybrid" in t.index and hybrid_summary:
        n_with = hybrid_summary.get("n_series_with_arima", 0)
        n_without = hybrid_summary.get("n_series_without_arima", 0)
        total = n_with + n_without
        share = 100 * n_with / total if total else 0
        hy = float(t.loc["hybrid", "MAE"])
        rf_mae = float(t.loc["random_forest", "MAE"]) if "random_forest" in t.index else float("nan")
        delta = rf_mae - hy
        coverage = (
            f"An ARIMA residual model could be fitted for {n_with:,} of {total:,} series "
            f"({share:.0f}%); the rest had fewer than "
            f"{hybrid_summary.get('min_residuals_for_arima')} training residuals and fall back "
            "to the plain forest, so on those rows the hybrid and the forest are identical by "
            "construction."
        )
        if np.isnan(delta):
            lines.append("The hybrid's Random Forest component is not in the table, so the two "
                         "cannot be compared directly. " + coverage)
        elif delta > 0:
            lines.append(
                f"The hybrid improves on its own Random Forest component "
                f"({hy:,.2f} against {rf_mae:,.2f} LKR MAE, a gain of {delta:,.2f}). "
                "That is the intended behaviour: the ARIMA correction is recovering serial "
                "structure the forest discards by treating rows as independent. " + coverage
            )
        else:
            lines.append(
                f"The hybrid is **worse** than its own Random Forest component "
                f"({hy:,.2f} against {rf_mae:,.2f} LKR MAE, a loss of {abs(delta):,.2f}). "
                "The ARIMA correction is adding error rather than removing it. The mechanism "
                "is sound but the conditions here work against it: the residual forecast is "
                "anchored on the last training month and has to reach four to nine months "
                "forward, and at that distance an ARIMA forecast has decayed to the mean of "
                "the training residuals. If those residuals are not centred on zero - and "
                "out-of-fold residuals from an inflationary series are not, because folds "
                "fitted on earlier, cheaper months under-predict later ones - then the "
                "correction becomes a near-constant bias added to every predicted month. "
                + coverage
            )
    return "\n\n".join(lines)


def save_all_results(
    table: pd.DataFrame, per_month: pd.DataFrame, resid_df: pd.DataFrame,
    extra: Optional[Dict] = None, outdir: Optional[Path] = None,
) -> Dict[str, str]:
    outdir = Path(outdir or utils.METRICS_DIR)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {}
    table.to_csv(outdir / "model_comparison.csv", index=False)
    paths["model_comparison"] = str(outdir / "model_comparison.csv")
    per_month.to_csv(outdir / "metrics_by_test_month.csv", index=False)
    paths["metrics_by_test_month"] = str(outdir / "metrics_by_test_month.csv")
    resid_df.to_csv(outdir / "test_predictions_residuals.csv", index=False)
    paths["test_predictions"] = str(outdir / "test_predictions_residuals.csv")
    payload = {
        "comparison": table.to_dict(orient="records"),
        "best_by_mae": identify_best_model(table, "MAE"),
        "best_by_rmse": identify_best_model(table, "RMSE"),
        "best_by_r2": identify_best_model(table, "R2"),
    }
    if extra:
        payload.update(extra)
    utils.save_json(payload, outdir / "evaluation_summary.json")
    paths["evaluation_summary"] = str(outdir / "evaluation_summary.json")
    return paths
