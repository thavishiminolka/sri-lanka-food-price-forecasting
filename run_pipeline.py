#!/usr/bin/env python
"""
run_pipeline.py
===============
End-to-end training pipeline: load -> clean -> features -> chronological split ->
train the four ensembles -> evaluate -> figures -> save artefacts.

Usage
-----
    python run_pipeline.py                # full hyperparameter grids
    python run_pipeline.py --fast         # reduced grids, for a quick run
    python run_pipeline.py --no-tune      # sensible defaults, no grid search

Running this once is what the Streamlit dashboard needs: ``app.py`` only loads
the artefacts it writes into ``models/saved_models/``.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import utils  # noqa: E402
import data_loader  # noqa: E402
import preprocessing  # noqa: E402
import feature_engineering as fe  # noqa: E402
import evaluate_models as ev  # noqa: E402
import model_persistence as mp  # noqa: E402

LOGGER = utils.LOGGER


def main() -> int:
    ap = argparse.ArgumentParser(description="Train and evaluate the four ensemble models.")
    ap.add_argument("--fast", action="store_true", help="reduced hyperparameter grids")
    ap.add_argument("--no-tune", action="store_true", help="skip grid search entirely")
    ap.add_argument("--csv", default=None, help="path to the WFP CSV (defaults to data/raw/)")
    ap.add_argument("--max-arima-series", type=int, default=None,
                    help="cap the number of series given an ARIMA residual model")
    args = ap.parse_args()

    if args.fast:
        os.environ["FOOD_PRICE_FAST"] = "1"
    import train_models as tm  # imported after FOOD_PRICE_FAST is set  # noqa: E402
    import hybrid_model as hm  # noqa: E402
    import forecasting as fc  # noqa: E402

    tune = not args.no_tune
    t_start = time.time()
    timings = {}

    # ---------------------------------------------------------------- 1. load
    LOGGER.info("STEP 1/8  Loading the dataset")
    raw, provenance = data_loader.load_raw_data(csv_path=args.csv)
    inspection = data_loader.inspect_raw(raw)
    utils.save_json(provenance, utils.METRICS_DIR / "dataset_provenance.json")
    utils.save_json(inspection, utils.METRICS_DIR / "dataset_inspection.json")

    # ---------------------------------------------------------------- 2. clean
    LOGGER.info("STEP 2/8  Cleaning")
    clean, clean_report = preprocessing.clean_food_prices(raw)
    clean.to_parquet(utils.CLEAN_PARQUET, index=False)
    clean.to_csv(utils.CLEAN_CSV, index=False)
    utils.save_json(clean_report, utils.METRICS_DIR / "preprocessing_report.json")

    # ------------------------------------------------- 3. features and split
    LOGGER.info("STEP 3/8  Feature engineering and chronological split")
    data = fe.prepare_modelling_data(clean)
    data["df_features"].to_parquet(utils.FEATURES_PARQUET, index=False)
    data["df_model"].to_parquet(utils.MODEL_TABLE_PARQUET, index=False)
    for name, rep in data["reports"].items():
        utils.save_json(rep, utils.METRICS_DIR / f"{name}_report.json")

    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    features = data["feature_columns"]

    # ------------------------------------------------------- 4. bagging (RF)
    LOGGER.info("STEP 4/8  A. Bagging - Random Forest")
    t0 = time.time()
    rf_out = tm.train_random_forest(X_train, y_train, tune=tune)
    rf = rf_out["model"]
    timings["random_forest"] = round(time.time() - t0, 1)

    # ------------------------------------------------------ 5. boosting (XGB)
    LOGGER.info("STEP 5/8  B. Boosting - XGBoost")
    t0 = time.time()
    xgb_out = tm.train_xgboost(X_train, y_train, tune=tune)
    xgb = xgb_out["model"]
    timings["xgboost"] = round(time.time() - t0, 1)

    # --------------------------------------------------------- 6. stacking
    LOGGER.info("STEP 6/8  C. Stacking - RF + XGB + LinearRegression, Ridge meta-learner")
    t0 = time.time()
    rf_best = {k: v for k, v in (rf_out.get("best_params") or {}).items()
               if k in {"n_estimators", "max_depth", "min_samples_leaf", "max_features"}}
    xgb_best = {k: v for k, v in (xgb_out.get("best_params") or {}).items()
                if k in {"n_estimators", "learning_rate", "max_depth", "subsample",
                         "colsample_bytree", "min_child_weight", "reg_lambda"}}
    stack_out = tm.train_stacking(
        X_train, y_train, X_val, y_val,
        rf_params=rf_best, xgb_params=xgb_best, tune_alpha=tune,
    )
    stacking = stack_out["model"]
    timings["stacking"] = round(time.time() - t0, 1)

    # ----------------------------------------------------------- 7. hybrid
    LOGGER.info("STEP 7/8  D. Hybrid - Random Forest + ARIMA")
    t0 = time.time()
    hybrid = hm.RandomForestARIMAHybrid(
        rf_model=rf, max_series=args.max_arima_series
    ).fit(data["train"], X_train, y_train, features, rf_already_fitted=True)
    timings["hybrid"] = round(time.time() - t0, 1)
    hybrid_summary = hybrid.summary()
    arima_table = hybrid.arima_table()
    arima_table.to_csv(utils.METRICS_DIR / "arima_per_series.csv", index=False)

    # -------------------------------------------------------- 8. evaluation
    LOGGER.info("STEP 8/8  Evaluation on the chronological test set")
    hybrid_test = hybrid.predict(data["test"], X_test)
    predictions = {
        "random_forest": rf.predict(X_test),
        "xgboost": xgb.predict(X_test),
        "stacking": stacking.predict(X_test),
        "hybrid": hybrid_test["hybrid"],
    }
    val_predictions = {
        "random_forest": rf.predict(X_val),
        "xgboost": xgb.predict(X_val),
        "stacking": stacking.predict(X_val),
    }

    # --- hybrid diagnostic: did the ARIMA correction help where it was applied? ---
    hy_mask = hybrid_test["corrected"]
    yt_arr = np.asarray(y_test, dtype=float)
    hybrid_diagnostic = {
        "n_test_rows": int(len(yt_arr)),
        "n_rows_corrected": int(hy_mask.sum()),
        "pct_rows_corrected": round(100 * float(hy_mask.mean()), 2),
    }
    if hy_mask.sum() > 0:
        rc = hybrid_test["residual_forecast"][hy_mask]
        hybrid_diagnostic.update({
            "corrected_rows_rf_mae": utils.mae(yt_arr[hy_mask], hybrid_test["rf"][hy_mask]),
            "corrected_rows_hybrid_mae": utils.mae(yt_arr[hy_mask], hybrid_test["hybrid"][hy_mask]),
            "correction_mean_lkr": float(rc.mean()),
            "correction_sd_lkr": float(rc.std()),
            "correction_min_lkr": float(rc.min()),
            "correction_max_lkr": float(rc.max()),
            "oof_residual_mean_lkr": hybrid_summary["oof_residual_summary"]["mean"],
        })
        hybrid_diagnostic["corrected_rows_mae_change"] = (
            hybrid_diagnostic["corrected_rows_rf_mae"]
            - hybrid_diagnostic["corrected_rows_hybrid_mae"]
        )
        print("\n" + "-" * 78)
        print("HYBRID DIAGNOSTIC - on the test rows that actually received a correction")
        print("-" * 78)
        print(f"Rows corrected            : {hybrid_diagnostic['n_rows_corrected']:,} of "
              f"{hybrid_diagnostic['n_test_rows']:,} "
              f"({hybrid_diagnostic['pct_rows_corrected']}%)")
        print(f"Random Forest alone   MAE : "
              f"{hybrid_diagnostic['corrected_rows_rf_mae']:,.2f} LKR")
        print(f"Hybrid (RF + ARIMA)   MAE : "
              f"{hybrid_diagnostic['corrected_rows_hybrid_mae']:,.2f} LKR")
        print(f"Change from correction    : "
              f"{hybrid_diagnostic['corrected_rows_mae_change']:+,.2f} LKR "
              f"({'improvement' if hybrid_diagnostic['corrected_rows_mae_change'] > 0 else 'degradation'})")
        print(f"Corrections applied       : mean {hybrid_diagnostic['correction_mean_lkr']:+,.2f}, "
              f"sd {hybrid_diagnostic['correction_sd_lkr']:,.2f}, range "
              f"[{hybrid_diagnostic['correction_min_lkr']:+,.2f}, "
              f"{hybrid_diagnostic['correction_max_lkr']:+,.2f}] LKR")
        print(f"Mean out-of-fold residual : "
              f"{hybrid_diagnostic['oof_residual_mean_lkr']:+,.2f} LKR "
              "(a non-zero mean becomes a near-constant bias at long horizons)")
        print("-" * 78)
    utils.save_json(hybrid_diagnostic, utils.METRICS_DIR / "hybrid_diagnostic.json")

    table = ev.build_comparison_table(y_test, predictions)
    best_mae = ev.identify_best_model(table, "MAE")
    best_rmse = ev.identify_best_model(table, "RMSE")
    best_r2 = ev.identify_best_model(table, "R2")
    per_month = ev.per_month_metrics(data["test"], y_test, predictions)
    per_commodity = ev.per_group_metrics(data["test"], y_test, predictions, "commodity")
    per_market = ev.per_group_metrics(data["test"], y_test, predictions, "market")
    resid = ev.residual_frame(data["test"], y_test, predictions)

    print("\n" + "=" * 78)
    print("FOUR-MODEL COMPARISON - chronological test set")
    print("=" * 78)
    print(table.to_string(index=False))
    print("\nBest by MAE :", best_mae["best_model"], f"({best_mae['best_value']:,.3f} LKR)")
    print("Best by RMSE:", best_rmse["best_model"], f"({best_rmse['best_value']:,.3f} LKR)")
    print("Best by R2  :", best_r2["best_model"], f"({best_r2['best_value']:.4f})")
    print("=" * 78 + "\n")

    # ------------------------------------------------------------- figures
    LOGGER.info("Writing figures")
    ev.plot_all_metric_comparisons(table)
    ev.plot_actual_vs_predicted(y_test, predictions)
    ev.plot_residual_analysis(resid)
    ev.plot_error_by_month(per_month)

    rf_imp = tm.feature_importances(rf, features)
    xgb_imp = tm.feature_importances(xgb, features)
    rf_imp.to_csv(utils.METRICS_DIR / "feature_importance_random_forest.csv", index=False)
    xgb_imp.to_csv(utils.METRICS_DIR / "feature_importance_xgboost.csv", index=False)
    ev.plot_feature_importance(
        rf_imp, "Random Forest feature importance (mean decrease in impurity)",
        "feature_importance_random_forest.png", color=ev.PALETTE["random_forest"])
    ev.plot_feature_importance(
        xgb_imp, "XGBoost feature importance (gain)",
        "feature_importance_xgboost.png", color=ev.PALETTE["xgboost"])

    LOGGER.info("Permutation importance on the test set (model-agnostic)")
    perm = tm.permutation_importance_df(rf, X_test, y_test, features, n_repeats=5)
    perm.to_csv(utils.METRICS_DIR / "permutation_importance_random_forest.csv", index=False)
    ev.plot_feature_importance(
        perm, "Permutation importance on the test set - Random Forest",
        "permutation_importance_random_forest.png",
        value_col="importance_mean", color="#00695C")

    # Example per-series actual-vs-predicted charts
    top_series = (
        resid.groupby(utils.SERIES_ID, observed=True).size().sort_values(ascending=False)
        .head(3).index.tolist()
    )
    for sid in top_series:
        ev.plot_series_actual_vs_pred(resid, sid)

    interpretation = ev.interpret_comparison(table, hybrid_summary)
    print(interpretation + "\n")

    # ------------------------------------------------------------- persist
    metrics_payload = {
        "comparison": table.to_dict(orient="records"),
        "best_by_mae": best_mae,
        "best_by_rmse": best_rmse,
        "best_by_r2": best_r2,
        "validation_metrics": {
            k: utils.regression_metrics(y_val, v) for k, v in val_predictions.items()
        },
        "test_metrics": {
            k: utils.regression_metrics(y_test, v) for k, v in predictions.items()
        },
        "per_test_month": per_month.to_dict(orient="records"),
        "hybrid_summary": hybrid_summary,
        "hybrid_diagnostic": hybrid_diagnostic,
        "interpretation": interpretation,
        "test_period": data["reports"]["split"]["test_period"],
        "n_test_obs": int(len(y_test)),
    }
    ev.save_all_results(table, per_month, resid,
                        extra={"hybrid_summary": hybrid_summary,
                               "interpretation": interpretation})
    per_commodity.to_csv(utils.METRICS_DIR / "metrics_by_commodity.csv", index=False)
    per_market.to_csv(utils.METRICS_DIR / "metrics_by_market.csv", index=False)

    tuning_payload = {
        "random_forest": {k: v for k, v in rf_out.items() if k != "model"},
        "xgboost": {k: v for k, v in xgb_out.items() if k != "model"},
        "stacking": {k: v for k, v in stack_out.items() if k != "model"},
    }
    utils.save_json(tuning_payload, utils.METRICS_DIR / "hyperparameter_tuning.json")

    # Pre-compute per-series test MAE so the dashboard can size forecast bands.
    series_err = {
        k: fc.series_error_map(resid, k) for k in utils.MODEL_ORDER
    }

    written = mp.save_artifacts(
        models={"random_forest": rf, "xgboost": xgb, "stacking": stacking, "hybrid": hybrid},
        encoder=data["encoder"],
        metrics=metrics_payload,
        metadata={
            "split": data["reports"]["split"],
            "model_table": data["reports"]["model_table"],
            "preprocessing": clean_report,
            "tuning": tuning_payload,
            "timings_seconds": timings,
            "fast_mode": bool(args.fast),
            "tuned": tune,
            "series_test_mae": series_err,
            "total_runtime_seconds": round(time.time() - t_start, 1),
        },
        arima_info={
            "summary": hybrid_summary,
            "per_series": arima_table.head(400).to_dict(orient="records"),
        },
        feature_importance={
            "random_forest": rf_imp.to_dict(orient="records"),
            "xgboost": xgb_imp.to_dict(orient="records"),
            "permutation_random_forest": perm.to_dict(orient="records"),
        },
    )

    LOGGER.info("Artefacts written: %s", list(written))
    LOGGER.info("Total runtime: %.1f s", time.time() - t_start)
    print("\nPipeline complete. The dashboard can now be started with: streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
