# Saved model artefacts

This folder holds the artefacts the Streamlit dashboard loads at start-up.

## What is in this package

The small files are included:

| File | Contents |
|---|---|
| `metrics.json` | Four-model comparison, best model, per-month metrics, hybrid summary and diagnostic |
| `metadata.json` | Split dates, feature list, tuning results, library versions, per-series test MAE |
| `arima_info.json` | Per-series ARIMA order, AIC and residual count |
| `feature_importance.json` | Impurity, gain and permutation importances |

## What is not, and how to create it

The four model binaries are **excluded from this package** because of their size:

| File | Size |
|---|---|
| `random_forest.joblib` | ~78 MB |
| `stacking.joblib` | ~81 MB |
| `hybrid.joblib` | ~79 MB |
| `xgboost.joblib` | ~1 MB |
| `preprocessing.joblib` | small (the fitted categorical encoder) |

Recreate all of them, exactly as they were produced, by running once from the project root:

```bash
python run_pipeline.py
```

That takes roughly 14 minutes on two CPU cores and writes every file listed above. Use
`python run_pipeline.py --fast` for a quicker run with reduced hyperparameter grids.

The dashboard reports clearly when these artefacts are absent rather than falling back to an
untrained model, so you will see a message telling you to run the pipeline.

## Reducing the model size for deployment

The Random Forest dominates the total. If you need to commit the artefacts to GitHub or deploy to
Streamlit Community Cloud, retrain with fewer or shallower trees first, for example by editing
`rf_param_grid()` in `src/train_models.py` to use `n_estimators: [150]` and
`max_depth: [12]`. Expect a small accuracy cost; re-run `run_pipeline.py` and quote the new
numbers rather than the ones in this package.
