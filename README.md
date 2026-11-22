# Ensemble Learning for Sri Lankan Food Price Analysis and Forecasting

Four ensemble learning approaches — **bagging**, **boosting**, **stacking** and a **Random Forest +
ARIMA hybrid** — applied to World Food Programme monthly food price data for Sri Lanka, compared on
a strictly chronological test set, with an interactive Streamlit dashboard for exploration and
forecasting.

---

## Project overview

Food prices are the sharpest everyday measure of economic stress. Between 2021 and 2023 Sri Lanka
went through a foreign-exchange crisis, fuel shortages and an import squeeze, and food prices moved
sharply as a result. This project treats the practical question as a supervised learning problem:
**given how a commodity has been priced recently in a particular market, what will it cost next
month?**

It is deliberately end-to-end — real data with real gaps, documented cleaning decisions,
leakage-free features, time-aware validation, an honest comparison, and a dashboard that loads
trained artefacts rather than retraining.

## Problem statement

Predict the monthly price (in Sri Lankan Rupees) of a given food commodity in a given market, using
only information available before that month. Three properties of the data make this harder than a
textbook regression:

1. **Extreme heterogeneity** — staples, vegetables, fish, pulses, sugar and fuel, measured in
   kilograms, litres, units and 750 ml bottles, with prices from about 11 LKR to over 3,700 LKR.
   Prices in different units must never be compared as though equivalent.
2. **Non-stationarity** — the 2022 inflation episode means a series' historical mean is a poor guide
   to its current level, and tree models cannot extrapolate above the range they were trained on.
3. **Short, uneven, numerous series** — over 1,400 commodity–market–unit–pricetype series, most
   beginning only in 2023, many with missing months. Fitting a separate classical time-series model
   per series is not viable, which is why every series is pooled into one supervised table.

## Dataset

| Field | Value |
|---|---|
| **Title** | Sri Lanka – Food Prices |
| **Provider** | World Food Programme (WFP) |
| **Distributed via** | Humanitarian Data Exchange (HDX) |
| **Resource page** | https://data.humdata.org/dataset/wfp-food-prices-for-sri-lanka/resource/3638f0d6-9969-48cf-a919-1d879d037ec6 |
| **File** | `wfp_food_prices_lka.csv` → `data/raw/` |
| **Download date** | 2026-11-21 |
| **Rows** | 34,262 monthly observations |
| **Coverage** | January 2004 – September 2025 (260 distinct months) |
| **Scope** | 42 commodities, 44 markets, 9 provinces |
| **Units** | KG, Unit, L, 750 ML — kept strictly separate |
| **Price types** | Retail and Wholesale |
| **Currency** | LKR, with a WFP-supplied `usdprice` column |
| **Price flag** | `actual` throughout (no aggregated or imputed WFP values) |
| **Licence** | HDX publishes WFP food prices under Creative Commons Attribution for Intergovernmental Organisations (CC BY-IGO). Confirm the current licence on the resource page before redistributing. |
| **Attribution** | Source: World Food Programme (WFP), via the Humanitarian Data Exchange (HDX). |

**Important structural fact:** about **84%** of all observations are dated June 2023 or later. The
dataset is a thin, intermittent history (2004–2023) followed by a dense but short panel of roughly
28 months. This single fact drives most of the modelling decisions below.

## Technologies used

| Purpose | Library |
|---|---|
| Data handling | pandas, numpy, pyarrow |
| Bagging, stacking, tuning, metrics | scikit-learn |
| Boosting | xgboost |
| Time-series residual model | statsmodels (ARIMA) |
| Static visualisation | matplotlib, seaborn |
| Interactive visualisation | plotly |
| Dashboard | streamlit |
| Persistence | joblib |
| Notebook | jupyter, nbformat |

## The four ensemble approaches

### A. Bagging — `RandomForestRegressor`

Many deep trees, each grown on a bootstrap resample of the training rows and restricted at every
split to a random subset of features, grown **in parallel** and averaged. Averaging decorrelated
high-variance trees reduces **variance** without adding bias. Random feature subsetting is what
stops every tree choosing `lag_1` at the root and becoming identical.

### B. Boosting — `XGBRegressor`

Trees built **sequentially**, each fitted to the gradient of the loss that the current ensemble
leaves behind, so every tree concentrates on the rows still being predicted badly. XGBoost adds
second-order (Newton) steps, L1/L2 penalties on leaf weights, shrinkage via the learning rate, and
row/column subsampling. This reduces **bias** stage by stage, where bagging reduces variance all at
once.

### C. Stacking — RF + XGBoost + LinearRegression, with a `Ridge` meta-learner

Three different algorithms predict the same target; a Ridge meta-learner trained on their
**out-of-fold** predictions learns how much to trust each. Ridge is used because the three base
predictions correlate above 0.99 — ordinary least squares on collinear inputs produces unstable,
wildly opposed coefficients, while Ridge's L2 penalty keeps them stable as blending weights.

> **Why not `sklearn.ensemble.StackingRegressor`?** It calls `cross_val_predict` internally, which
> requires the CV splitter to be a true partition. `TimeSeriesSplit` is not one — its earliest rows
> never appear in a test fold — so it raises `ValueError: cross_val_predict only works for
> partitions`. Substituting `KFold` would run, but would shuffle time and allow a fold to train on
> 2025 and predict 2024. `TimeAwareStackingRegressor` in `src/train_models.py` implements the same
> algorithm with expanding-window folds, excluding the uncovered earliest rows from meta-training
> rather than imputing them.

### D. Hybrid — Random Forest + ARIMA

The forest predicts the **level**; ARIMA models the autocorrelation left in the forest's residuals,
per price series:

```
y_hybrid = y_random_forest + r_arima
```

The residuals must be **out-of-fold**. A forest predicting a training row it has already seen is
nearly exact, so in-sample residuals are mostly memorisation noise; an ARIMA fitted to those would
model the forest's memorisation rather than its generalisation error. Out-of-fold residuals from
`TimeSeriesSplit` folds have the same character as the errors the forest will make on test months.

The forecast origin is the last training month, so a test month *k* months later uses the
*k*-step-ahead residual forecast — never a one-step forecast updated with an actual it has not been
given. Series with fewer than 12 training residuals receive a zero correction and fall back to the
plain forest; the counts are reported, not hidden.

> ARIMA is **not** reported as a fifth model. Linear Regression appears only as a stacking base
> learner. No standalone naive, seasonal-naive, linear or ARIMA model enters the main comparison.

## Methodology

### Cleaning decisions

| Decision | Reasoning |
|---|---|
| Parse `date`, derive `month_start` | Every observation falls on day 15 — the data is a monthly panel |
| Drop rows with missing/unparseable date or non-positive price | No time position or no target means the row is unusable (**0 rows affected in this file**) |
| Label missing `admin1`/`admin2` as `National`/`Unknown` | The 194 affected rows belong to the `National Average` market, which has no district — a structural absence, not a fault |
| Collapse repeated (series, month) pairs to their mean | A repeat for the same commodity–market–unit can only be an aggregation artefact (**0 found**) |
| **Flag** outliers, never remove them | The 2022–2023 spikes are real economic events. Detection uses a robust z-score (median and MAD, not mean and standard deviation, since the standard deviation is itself inflated by the outliers it should detect). **216 flagged, 0 removed.** |
| Keep all units and both price types | `unit` and `pricetype` form part of the series key, so they are never pooled |

### Feature engineering — 34 features

**Calendar-aligned lags.** 858 of 1,444 series have at least one missing month. A naive
`groupby().shift(1)` would take the previous *row*, so a series with a six-month gap would receive a
six-month-old price labelled `lag_1`. Every series is therefore reindexed onto a **continuous
monthly calendar** before any lag is taken.

**Leakage control.** All rolling and expanding statistics are computed on the *shifted* price
matrix, so a window for month *t* covers only months up to *t−1*. The target can never enter its own
feature. `test_project.py` verifies this on a random sample.

| Group | Features | Why |
|---|---|---|
| Calendar | `year`, `month`, `quarter`, `time_index`, `month_sin`, `month_cos` | `time_index` captures the inflation trend; sin/cos encodes month cyclically so December sits beside January |
| History depth | `obs_index`, `months_since_prev_obs` | Lets a tree discount lag features for thin or gappy series |
| Lags | `lag_1/2/3/6/12` | The autoregressive core; `lag_12` carries same-month-last-year seasonality |
| Rolling | `roll_mean_3/6/12`, `roll_std_3/6/12` | Local level and local volatility. The means proved to be among the strongest features; the standard deviations, added on the hypothesis that volatility would help, measured out in the bottom third (see results) |
| Momentum | `diff_1`, `pct_change_1`, `pct_change_3` | Direction and size of recent movement |
| Expanding | `exp_mean`, `exp_std` | Whole observed history to date; also the fallback below |
| Availability | `has_lag_6/12`, `has_roll_6/12` | Whether each long-memory feature was observed or filled |
| Categorical | `commodity`, `market`, `unit`, `pricetype`, `category`, `admin1` | Identity; encoder fitted on **training split only**, unseen categories → `-1` |

**The long-memory fallback — a measured trade-off.** Requiring a genuine 12-month history for every
row leaves **17,629** usable rows; requiring only the 3-month core leaves **29,941**. Rather than
discard two-thirds of the dense 2023–2025 panel for one feature, an unavailable long-memory feature
is filled with that series' **own expanding mean or standard deviation** — computed only from months
strictly before the row, so there is no leakage — and a `has_*` flag records the fill. The final
modelling table holds **26,188 rows** (76.4% of observations), covering 1,413 series.

### Chronological splitting

| Split | Period | Rows | Share |
|---|---|---|---|
| Train | up to 2024-12 | 18,428 | 70.4% |
| Validation | 2025-01 to 2025-03 | 2,754 | 10.5% |
| Test | 2025-04 to 2025-09 | 5,006 | 19.1% |

**No shuffling anywhere.** Every validation month is later than every training month, and every test
month later than every validation month — asserted in code. Tuning uses `GridSearchCV` with
`TimeSeriesSplit` (expanding windows, 3 folds), so no hyperparameter is chosen with hindsight. The
test set is touched once, at the end.

<!-- RESULTS_START -->
## Actual model comparison results

Measured on the chronological test set — **2025-04-01 .. 2025-09-01**, **5,006
observations**, the same rows for every model. These numbers come from executing
`run_pipeline.py` on the real dataset; nothing here is estimated.

| Rank (MAE) | Approach | MAE (LKR) | RMSE (LKR) | R² | MAPE (%) |
|---|---|---|---|---|---|
| 1 | Stacking - RF + XGB + LR (Ridge meta) **(best)** | 71.92 | 114.95 | 0.9723 | 14.35 |
| 2 | Bagging - Random Forest | 73.39 | 120.95 | 0.9693 | 13.80 |
| 3 | Boosting - XGBoost | 73.68 | 121.26 | 0.9691 | 14.14 |
| 4 | Hybrid - Random Forest + ARIMA | 91.78 | 163.26 | 0.9440 | 15.74 |

**Best by MAE:** Stacking - RF + XGB + LR (Ridge meta) (71.92 LKR)
**Best by RMSE:** Stacking - RF + XGB + LR (Ridge meta) (114.95 LKR)
**Best by R²:** Stacking - RF + XGB + LR (Ridge meta) (0.9723)

### Chosen hyperparameters

| Model | Best parameters | CV RMSE (LKR) | Fits | Tuning time |
|---|---|---|---|---|
| Random Forest | `{'max_depth': 20, 'max_features': 0.7, 'min_samples_leaf': 1, 'n_estimators': 300}` | 168.76 | 24 | 266.2 s |
| XGBoost | `{'colsample_bytree': 0.6, 'learning_rate': 0.03, 'max_depth': 8, 'min_child_weight': 5, 'n_estimators': 400, 'reg_lambda': 1.0, 'subsample': 0.8}` | 209.04 | 216 | 289.5 s |
| Stacking | Ridge `alpha=10.0` | — | — | — |

Stacking meta-learner weights on the standardised base predictions: `random_forest` +422.9, `xgboost` -240.7, `linear_regression` +460.7.

### Reading these results

**The top three finish within 1.76 LKR of each other.**
That is the most important observation in the table. When bagging, boosting and a learned blend of
both land this close, the binding constraint is not the choice of ensemble — it is the information
in the features. All three read the same lag and rolling features, and those features
carry most of the available signal. The hours that changed the outcome went into calendar-aligned
lags and leakage control, not model selection.

**Stacking's advantage is real but small**, and it shows up most clearly in RMSE
(114.95 against
120.95 for the forest), which means its
main benefit is keeping the *worst* misses smaller rather than improving the typical row. The Ridge
meta-learner puts a negative weight on one base learner — it is using that model as a correction
term rather than as a third opinion, which is exactly what a blend of highly correlated predictors
should do.

**The hybrid came last, and that is a finding rather than a defect.** ARIMA could be fitted for
634 of 1,129 series (56%); the rest had fewer than
12 training residuals and fall back to the plain forest. On just the 2,649 test rows that actually received an ARIMA correction (52.92% of the test set), the forest alone scored 78.44 LKR MAE and the hybrid 113.19 — so the correction **increased** error by 34.75 LKR on the rows it touched. The corrections applied had mean +53.74 LKR, and the mean out-of-fold training residual was +39.92 LKR.

The cause is the forecast horizon. The residual correction is anchored on the last training month
and must reach four to nine months forward. At that distance an ARIMA forecast has decayed to the
mean of the training residuals — and on an inflationary series those residuals are not centred on
zero, because out-of-fold folds fitted on earlier, cheaper months systematically under-predict
later ones. The correction therefore becomes close to a constant bias added to every predicted
month. The mechanism is sound; a residual-correction hybrid needs long series and a short forecast
horizon to show what it can do, and this dataset provides neither.

**The ensembles do beat trivial persistence.** A high R² alone would be unimpressive on a
persistent monthly series, since copying last month's price already scores well. The notebook
includes that benchmark explicitly (for context only — it is not one of the four compared
approaches) and the ensembles reduce MAE meaningfully below it, confirming they model price
*dynamics* rather than echoing the last observation.

**Error grows with the forecast horizon.** Every model is trained to December 2024, so a September
2025 prediction sits nine months from the training data. See
`results/figures/error_by_test_month.png`.

### Generated figures

All in `results/figures/`:

`model_comparison_mae.png` · `model_comparison_rmse.png` · `model_comparison_r2.png` ·
`model_comparison_mape_pct.png` · `actual_vs_predicted.png` · `residual_analysis.png` ·
`error_by_test_month.png` · `feature_importance_random_forest.png` ·
`feature_importance_xgboost.png` · `permutation_importance_random_forest.png` ·
per-series actual-versus-predicted charts

And in `results/metrics/`: `model_comparison.csv`, `metrics_by_test_month.csv`,
`metrics_by_commodity.csv`, `metrics_by_market.csv`, `test_predictions_residuals.csv`,
`hyperparameter_tuning.json`, `hybrid_diagnostic.json`, `arima_per_series.csv`,
`evaluation_summary.json`, and the dataset/preprocessing/split reports.
<!-- RESULTS_END -->

## Project structure

```
sri-lanka-food-price-ensemble/
├── app.py                        # Streamlit dashboard (6 pages) - loads saved models
├── run_pipeline.py               # end-to-end training pipeline
├── test_project.py               # component tests for every stage
├── requirements.txt
├── README.md
├── AI_USAGE.md
├── PRESENTATION.md               # 3-minute presentation outline + voiceover script
├── .gitignore
├── .streamlit/config.toml
├── data/
│   ├── raw/                      # place wfp_food_prices_lka.csv here
│   └── processed/                # cleaned panel, features, modelling table (generated)
├── notebooks/
│   └── food_price_ensemble_analysis.ipynb    # 26-section executable notebook
├── src/
│   ├── utils.py                  # config, paths, split dates, feature lists, metrics
│   ├── data_loader.py            # loading (local / upload / HDX), HXL handling, inspection
│   ├── preprocessing.py          # cleaning with a full decision report
│   ├── feature_engineering.py    # calendar-aligned features, model table, chronological split
│   ├── train_models.py           # RF, XGBoost, TimeAwareStackingRegressor, tuning
│   ├── hybrid_model.py           # RandomForestARIMAHybrid
│   ├── evaluate_models.py        # metrics, comparison table, figures, interpretation
│   ├── forecasting.py            # recursive multi-horizon ForecastEngine
│   └── model_persistence.py      # joblib save/load for the dashboard
├── models/saved_models/          # trained artefacts (generated)
└── results/
    ├── figures/                  # comparison charts, residuals, importances (generated)
    └── metrics/                  # CSV/JSON metrics and reports (generated)
```

## Installation

Python 3.10 or newer.

```bash
git clone <your-repository-url>
cd sri-lanka-food-price-ensemble

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Dataset setup

The raw CSV is excluded from git by `.gitignore` to keep the repository small. Get it in either way:

**Option 1 — manual (recommended).**
1. Open the [HDX resource page](https://data.humdata.org/dataset/wfp-food-prices-for-sri-lanka/resource/3638f0d6-9969-48cf-a919-1d879d037ec6)
2. Download the CSV
3. Save it as `data/raw/wfp_food_prices_lka.csv`

**Option 2 — automatic.** `data_loader.load_raw_data()` attempts the HDX download if no local file
is found. This needs outbound internet access to `data.humdata.org`; it is skipped silently when
unavailable.

If the file carries the HXL tag row (`#date`, `#adm1+name`, …), the loader detects and strips it
automatically. If no dataset can be found, the loader raises an error explaining the manual setup —
**it never substitutes synthetic data.**

## Running the pipeline

Train the four models, evaluate them, write the figures and save the artefacts the dashboard needs:

```bash
python run_pipeline.py              # full hyperparameter grids
python run_pipeline.py --fast       # reduced grids (quick run)
python run_pipeline.py --no-tune    # sensible defaults, no grid search
```

Useful options:

```bash
python run_pipeline.py --csv /path/to/wfp_food_prices_lka.csv
python run_pipeline.py --max-arima-series 200     # cap per-series ARIMA fitting
```

On two CPU cores the full run takes roughly 15–20 minutes, most of it Random Forest tuning
(a single 300-tree fit on 18,428 rows takes about 25 seconds, while a 600-round XGBoost fit takes
under two). `--fast` reduces this to a few minutes.

## Running the notebook

The notebook is **independent of the other commands** — it is a second way to see the whole project,
not a step in the pipeline. You do not have to run `run_pipeline.py` first.

```bash
# from the project root, with the virtual environment active
jupyter notebook notebooks/food_price_ensemble_analysis.ipynb
```

Then press **Run ▸ Run All Cells**. Equivalent alternatives:

```bash
jupyter lab                       # then open the file from the sidebar
code .                            # VS Code: open the .ipynb and pick the .venv kernel
```

### How long it takes depends on what is already there

The first code cell decides this automatically, and prints which path it took:

| Situation | What the notebook does | Time |
|---|---|---|
| `run_pipeline.py` has been run | **Loads** the saved models | under a minute |
| It has not | **Trains** all four models itself | about 5 minutes |

Both paths compute every number from the real dataset. The second uses reduced hyperparameter
grids, set by `QUICK_TUNING = True` in the first cell, so the metrics will differ slightly from the
full-grid results quoted above. Set `QUICK_TUNING = False` to reproduce them exactly.

> `QUICK_TUNING` is set in the **first** cell, before the imports, because `src/train_models.py`
> reads the `FOOD_PRICE_FAST` environment variable at import time to choose its grids. Setting it
> in a later cell would have no effect on the Random Forest and XGBoost searches.

### If `jupyter` is not found

`pip install -r requirements.txt` installs it, but the launcher script may not be on your `PATH`.
Run it through Python instead, which always works inside the virtual environment:

```bash
python -m jupyter notebook notebooks/food_price_ensemble_analysis.ipynb
```

If the notebook opens but the import cell fails, the kernel is probably not the virtual environment.
Register it explicitly:

```bash
python -m ipykernel install --user --name food-price --display-name "Python (food-price)"
```

then pick **Python (food-price)** under *Kernel ▸ Change Kernel*.

### In Google Colab

1. Upload and unzip the project into `/content`
2. Uncomment the `pip install` line in the first code cell
3. Run the cells in order — the data-loading cell offers a file upload if `data/raw/` is empty

Colab has no saved artefacts, so it trains in-notebook on the ~5-minute path.

## Running the dashboard

```bash
streamlit run app.py
```

Then open http://localhost:8501.

The dashboard **loads** saved models and never retrains at start-up. If artefacts are missing it
says so and explains how to create them, rather than silently using an untrained model.

**Pages**

1. **Project Overview** — introduction, problem statement, objectives, dataset description, the four techniques, headline result
2. **Food Price Explorer** — commodity / market / price-type selectors, date filter, historical charts, month-on-month changes, commodity comparison (within a single unit), market comparison
3. **Model Performance** — four-model comparison, MAE / RMSE / R² / MAPE charts, error by test month, actual-versus-predicted, best model from measured results, strengths and limitations, hybrid coverage
4. **Food Price Forecasting** — commodity, market, unit/price-type and horizon (1, 3, 6, 12 months) selectors; history-versus-forecast chart with actual and predicted clearly distinguished; forecast table; caveats
5. **Model Insights** — feature importance (impurity, gain, permutation), residual analysis, error distribution, per-series predictions, error by commodity, explanation of how the four approaches differ
6. **Dataset Explorer** — preview, commodities, markets, date range and coverage, data-quality report, artefact inventory

## Testing

```bash
python test_project.py            # full suite
python test_project.py --quick    # skip the slow training and startup tests
```

Covered: dataset loading · dataset inspection · preprocessing · feature engineering ·
**target-leakage checks** · model-table construction · chronological splitting · evaluation metrics
(verified against hand-computed values) · ARIMA residual modelling · Random Forest training ·
XGBoost training · stacking (including out-of-fold coverage) · hybrid prediction (asserting
`hybrid == RF + residual forecast`) · four-model comparison · forecasting at all four horizons ·
model saving and loading · dashboard imports · Streamlit startup.

Results are written to `results/metrics/test_report.json`. Tests requiring trained artefacts
**skip** rather than fail when `run_pipeline.py` has not been run.

## Deployment

### Streamlit Community Cloud

1. Push the repository to GitHub.
2. The dashboard needs the trained artefacts at run time. `.gitignore` excludes `*.joblib` by
   default, so choose one of:
   - **Commit the artefacts.** Remove the `models/saved_models/*.joblib` line from `.gitignore` and
     commit them. Check the total size first — the Random Forest pickle is the largest. Use
     [Git LFS](https://git-lfs.github.com/) if it is large.
   - **Reduce the model size** before saving (e.g. `n_estimators=150`), then commit.
   - **Commit the cleaned data instead** (`data/processed/food_price_clean.csv`) and train once on
     first launch — slower on cold start, and Community Cloud's memory limit applies.
3. Go to [share.streamlit.io](https://share.streamlit.io), connect the repository, and set:
   - **Main file path:** `app.py`
   - **Python version:** 3.10 or newer
4. `requirements.txt` is picked up automatically.

### Local or container

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### GitHub

Commit the source, the notebook, the documentation, `requirements.txt` and the small JSON/CSV
metrics. Keep the raw CSV and large pickles out unless deployment needs them. No credentials or
secrets are used anywhere in this project.

> This README does not claim the project has been uploaded or deployed. Those steps are the
> repository owner's to perform.

## Known limitations

**Data**

1. **Coverage is concentrated and recent** — ~84% of observations are June 2023 or later, so the
   dense panel is only ~28 months. That is thin for learning annual seasonality, and the models are
   effectively tuned to post-crisis dynamics.
2. **Gaps** — ~60% of series have at least one missing month. Calendar alignment handles this
   correctly but costs rows (26,188 of 34,262 retained).
3. **The `lag_12` fallback is a compromise** — leakage-free and flagged, but not the same
   information as a genuine 12-month lag.
4. **No exogenous drivers** — fuel prices, exchange rates, rainfall, import policy and harvest
   volumes all move food prices and none is in the dataset. The models react to price movements;
   they cannot anticipate shocks.
5. **Unit heterogeneity** — four units coexist. They are never pooled in a comparison, but one model
   still spans 11–3,700 LKR, which inflates absolute error for expensive commodities.
6. **No ground truth beyond September 2025.**

**Method**

7. **Recursive forecasting compounds error** — a 12-month forecast rests on eleven earlier
   predictions.
8. **Forecast bands are indicative, not calibrated** — scaled from historical error, not derived
   from a quantile or conformal procedure. Labelled as such throughout.
9. **One test window** — a rolling-origin backtest would give a more robust ranking, but was out of
   scope for the available compute (two cores).
10. **Grid sizes are hardware-constrained** — the forest's grid is compact because each fit costs
    ~25 seconds.
11. **No formal test of model differences** — the bootstrap intervals in the notebook are
    indicative; a Diebold-Mariano test would be the proper tool.
12. **Flagged outliers were retained** — defensible, but it means the metrics include genuinely
    hard rows.

## Generative AI acknowledgment

Generative AI tools, including ChatGPT and Claude, were used as supporting tools during this project, mainly for Streamlit dashboard development, coding assistance, and debugging.

AI assistance was used to develop the dashboard interface, implement interactive visualizations, and provide code suggestions for the ensemble learning models. It also helped identify and resolve programming errors during development.

All models were trained and evaluated using the actual World Food Programme (WFP) Sri Lankan food price dataset. **No synthetic data or fabricated evaluation results were used.** The reported performance metrics and visualizations were obtained by executing the implemented Python code.

AI was used as a technical assistance tool to support the development process. The modelling methodology, evaluation of results, and understanding of the implemented techniques remain the responsibility of the project author.

## Licence and attribution

- **Data:** World Food Programme (WFP), via the Humanitarian Data Exchange (HDX). Published under
  Creative Commons Attribution for Intergovernmental Organisations (CC BY-IGO). Confirm the current
  licence on the resource page before redistributing.
- **Code:** academic coursework submission.

**Forecasts produced by this project are model-based estimates, not guaranteed future prices.** They
reflect historical price dynamics only and cannot account for future policy changes, import
restrictions, weather shocks or currency movements.
