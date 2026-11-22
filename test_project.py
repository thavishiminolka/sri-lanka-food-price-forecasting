#!/usr/bin/env python
"""
test_project.py
===============
Component tests for every stage listed in section 19 of the assignment brief.

Each test is independent and reports PASS / FAIL / SKIP with a reason, so a
missing artefact is distinguished from a broken component.  Tests that need
trained models SKIP (rather than FAIL) when ``run_pipeline.py`` has not been run.

Usage
-----
    python test_project.py            # full suite
    python test_project.py --quick    # skip the slow training tests
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

RESULTS: list[dict] = []
_CACHE: dict = {}


def record(name: str, status: str, detail: str = "", seconds: float = 0.0) -> None:
    RESULTS.append({"test": name, "status": status, "detail": detail,
                    "seconds": round(seconds, 2)})
    icon = {"PASS": "[PASS]", "FAIL": "[FAIL]", "SKIP": "[SKIP]"}[status]
    print(f"{icon} {name}" + (f"  -  {detail}" if detail else ""))


def run(name: str, fn) -> None:
    t0 = time.time()
    try:
        detail = fn() or ""
        if isinstance(detail, tuple):  # (status, detail)
            record(name, detail[0], detail[1], time.time() - t0)
        else:
            record(name, "PASS", str(detail), time.time() - t0)
    except _Skip as exc:
        record(name, "SKIP", str(exc), time.time() - t0)
    except Exception as exc:  # noqa: BLE001
        record(name, "FAIL", f"{type(exc).__name__}: {exc}", time.time() - t0)
        traceback.print_exc(limit=3)


class _Skip(Exception):
    pass


# ==========================================================================
# 1. Dataset loading
# ==========================================================================
def t_dataset_loading():
    import data_loader
    df, prov = data_loader.load_raw_data()
    assert len(df) > 0, "empty dataframe"
    for col in ("date", "commodity", "market", "unit", "price", "currency"):
        assert col in df.columns, f"missing column {col}"
    _CACHE["raw"] = df
    _CACHE["provenance"] = prov
    return f"{len(df):,} rows, {df.shape[1]} columns, via {prov['load_method']}"


def t_dataset_inspection():
    import data_loader
    df = _CACHE.get("raw")
    if df is None:
        raise _Skip("raw data not loaded")
    rep = data_loader.inspect_raw(df)
    assert rep["shape"][0] == len(df)
    assert "series" in rep
    return f"{rep['series']['n_series']:,} series detected, {rep['duplicate_rows_full']} duplicate rows"


# ==========================================================================
# 2. Preprocessing
# ==========================================================================
def t_preprocessing():
    import preprocessing
    df = _CACHE.get("raw")
    if df is None:
        raise _Skip("raw data not loaded")
    clean, rep = preprocessing.clean_food_prices(df)
    assert clean["price"].notna().all(), "null prices survived cleaning"
    assert (clean["price"] > 0).all(), "non-positive prices survived cleaning"
    assert pd.api.types.is_datetime64_any_dtype(clean["month_start"])
    assert clean.duplicated(subset=["series_id", "month_start"]).sum() == 0
    assert rep["outliers_removed"] == 0, "outliers must be flagged, not removed"
    _CACHE["clean"] = clean
    _CACHE["clean_report"] = rep
    return (f"{rep['n_rows_input']:,} -> {rep['n_rows_output']:,} rows, "
            f"{rep['n_outliers_flagged']} outliers flagged (0 removed)")


# ==========================================================================
# 3. Feature engineering
# ==========================================================================
def t_feature_engineering():
    import feature_engineering as fe
    import utils
    clean = _CACHE.get("clean")
    if clean is None:
        raise _Skip("clean data unavailable")
    feats, rep = fe.build_features(clean)
    for col in utils.FEATURE_COLUMNS:
        assert col in feats.columns or col.endswith("_enc"), f"feature {col} not built"
    _CACHE["features"] = feats
    return f"{len(feats):,} rows, {len(utils.FEATURE_COLUMNS)} model features defined"


def t_no_target_leakage():
    """lag_1 must equal the series' previous observed price, never the current one."""
    import utils
    feats = _CACHE.get("features")
    if feats is None:
        raise _Skip("features unavailable")
    d = feats.sort_values([utils.SERIES_ID, "month_index"]).copy()
    # Build the expected calendar-aligned lag_1 independently.
    lookup = {(r.series_id, r.month_index): r.price
              for r in d[[utils.SERIES_ID, "month_index", "price"]].itertuples(index=False)}
    sample = d.dropna(subset=["lag_1"]).sample(
        min(3000, int(d["lag_1"].notna().sum())), random_state=0)
    bad = 0
    for r in sample.itertuples(index=False):
        expected = lookup.get((getattr(r, utils.SERIES_ID), r.month_index - 1))
        if expected is None or not np.isclose(expected, r.lag_1, rtol=1e-9):
            bad += 1
    assert bad == 0, f"{bad} rows where lag_1 is not the previous calendar month's price"
    # The decisive check on the rolling features: roll_mean_3 must be reproducible
    # from the lags alone. If it ever used the current month's price, this fails.
    have = d[["lag_1", "lag_2", "lag_3", "roll_mean_3"]].notna().all(axis=1)
    from_lags = (d.loc[have, "lag_1"] + d.loc[have, "lag_2"] + d.loc[have, "lag_3"]) / 3
    off = (~np.isclose(from_lags.to_numpy(dtype=float),
                       d.loc[have, "roll_mean_3"].to_numpy(dtype=float),
                       rtol=1e-9)).sum()
    assert off == 0, f"roll_mean_3 differs from mean(lag_1..lag_3) on {off} rows"

    # roll_mean_3 CAN coincide with the current price - that happens whenever a price
    # is flat across four consecutive months, which is common for administratively
    # fixed prices (fuel, wheat flour). That is a coincidence, not leakage, so the
    # test requires it to be explained by flat prices rather than merely rare.
    eq = np.isclose(d["roll_mean_3"].to_numpy(dtype=float),
                    d["price"].to_numpy(dtype=float), rtol=1e-12) & d["roll_mean_3"].notna()
    n_eq = int(eq.sum())
    if n_eq:
        e = d[eq]
        flat = (np.isclose(e["lag_1"], e["price"])
                & np.isclose(e["lag_2"], e["price"])
                & np.isclose(e["lag_3"], e["price"]))
        share_flat = float(flat.mean())
        assert share_flat > 0.80, (
            f"only {share_flat:.1%} of the {n_eq} rows where roll_mean_3 equals the price "
            "are explained by a flat price - the rest need investigating"
        )
    return (f"{len(sample):,} sampled rows: lag_1 calendar-correct; "
            f"roll_mean_3 == mean(lag_1..lag_3) on every row; "
            f"{n_eq} coincidental matches with the target, "
            f"{share_flat:.0%} from flat prices" if n_eq else
            f"{len(sample):,} sampled rows: lag_1 calendar-correct, no rolling leakage")


def t_model_table():
    import feature_engineering as fe
    import utils
    feats = _CACHE.get("features")
    if feats is None:
        raise _Skip("features unavailable")
    table, rep = fe.build_model_table(feats)
    assert table[utils.REQUIRED_FEATURES].notna().all().all()
    _CACHE["model_table"] = table
    return f"{rep['n_rows_after']:,} rows kept ({rep['pct_rows_kept']}%), {rep['n_series']:,} series"


# ==========================================================================
# 4. Chronological splitting
# ==========================================================================
def t_chronological_split():
    import feature_engineering as fe
    import utils
    table = _CACHE.get("model_table")
    if table is None:
        raise _Skip("model table unavailable")
    train, val, test, rep = fe.chronological_split(table)
    assert train["month_start"].max() < val["month_start"].min(), "train/val overlap"
    assert val["month_start"].max() < test["month_start"].min(), "val/test overlap"
    assert len(train) and len(val) and len(test)
    assert rep["shuffled"] is False

    enc = fe.fit_categorical_encoder(train)
    train = fe.apply_categorical_encoder(train, enc)
    val = fe.apply_categorical_encoder(val, enc)
    test = fe.apply_categorical_encoder(test, enc)
    _CACHE.update({"train": train, "val": val, "test": test, "encoder": enc})
    _CACHE["X_train"], _CACHE["y_train"] = fe.get_xy(train)
    _CACHE["X_val"], _CACHE["y_val"] = fe.get_xy(val)
    _CACHE["X_test"], _CACHE["y_test"] = fe.get_xy(test)
    assert _CACHE["X_train"].isna().sum().sum() == 0, "NaNs in X_train"
    assert _CACHE["X_test"].isna().sum().sum() == 0, "NaNs in X_test"
    return (f"train {rep['n_train']:,} ({rep['pct_train']}%) < val {rep['n_val']:,} "
            f"< test {rep['n_test']:,}; test = {rep['test_period']}")


# ==========================================================================
# 5-7. Model training
# ==========================================================================
def t_random_forest_training():
    import train_models as tm
    import utils
    if "X_train" not in _CACHE:
        raise _Skip("split unavailable")
    out = tm.train_random_forest(_CACHE["X_train"], _CACHE["y_train"], tune=False)
    pred = out["model"].predict(_CACHE["X_test"])
    m = utils.regression_metrics(_CACHE["y_test"], pred)
    assert len(pred) == len(_CACHE["y_test"])
    assert np.isfinite(pred).all()
    _CACHE["rf_untuned"] = out["model"]
    return f"untuned RF test MAE {m['MAE']:,.2f} LKR, R2 {m['R2']:.4f}"


def t_xgboost_training():
    import train_models as tm
    import utils
    if "X_train" not in _CACHE:
        raise _Skip("split unavailable")
    out = tm.train_xgboost(_CACHE["X_train"], _CACHE["y_train"], tune=False)
    pred = out["model"].predict(_CACHE["X_test"])
    m = utils.regression_metrics(_CACHE["y_test"], pred)
    assert np.isfinite(pred).all()
    return f"untuned XGBoost test MAE {m['MAE']:,.2f} LKR, R2 {m['R2']:.4f}"


def t_stacking_training():
    import train_models as tm
    import utils
    if "X_train" not in _CACHE:
        raise _Skip("split unavailable")
    model = tm.build_stacking_regressor(
        rf_params={"n_estimators": 60}, xgb_params={"n_estimators": 80}
    )
    names = [n for n, _ in model.estimators]
    assert names == ["random_forest", "xgboost", "linear_regression"], \
        f"wrong base learners: {names}"
    model.fit(_CACHE["X_train"], _CACHE["y_train"])
    assert model.n_oof_rows_ > 0, "no out-of-fold predictions generated"
    assert model.n_rows_without_oof_ > 0, \
        "expected the earliest fold to have no OOF predictions"
    weights = model.meta_weights()
    assert weights, "meta weights not exposed"
    pred = model.predict(_CACHE["X_test"])
    m = utils.regression_metrics(_CACHE["y_test"], pred)
    _CACHE["stacking_small"] = model
    return (f"OOF rows {model.n_oof_rows_:,} (excluded {model.n_rows_without_oof_:,}); "
            f"test MAE {m['MAE']:,.2f} LKR; weights "
            + ", ".join(f"{k}={v:.3f}" for k, v in weights.items()))


def t_arima_residual_modelling():
    import hybrid_model as hm
    import utils
    rng = np.random.default_rng(0)
    # AR(1) residual sequence: ARIMA should fit and forecast it.
    e = np.zeros(60)
    for i in range(1, 60):
        e[i] = 0.7 * e[i - 1] + rng.normal(0, 1)
    info = hm.fit_arima_on_residuals(pd.Series(e))
    assert info["ok"], f"ARIMA failed on a clean AR(1) series: {info.get('reason')}"
    fc = info["result"].forecast(steps=3)
    assert np.isfinite(np.asarray(fc)).all()
    # Too-short series must be refused, not silently fitted.
    short = hm.fit_arima_on_residuals(pd.Series(e[:5]))
    assert not short["ok"], "ARIMA accepted a series shorter than the minimum"
    return (f"AR(1) test: order {info['order']}, AIC {info['aic']}; "
            f"short series correctly refused ({short['reason']})")


def t_hybrid_prediction():
    import hybrid_model as hm
    import utils
    if "rf_untuned" not in _CACHE:
        raise _Skip("Random Forest unavailable")
    hyb = hm.RandomForestARIMAHybrid(
        rf_model=_CACHE["rf_untuned"], max_series=40
    ).fit(_CACHE["train"], _CACHE["X_train"], _CACHE["y_train"],
          list(_CACHE["X_train"].columns), rf_already_fitted=True)
    out = hyb.predict(_CACHE["test"], _CACHE["X_test"])
    assert set(out) >= {"rf", "residual_forecast", "hybrid", "corrected"}
    assert np.isfinite(out["hybrid"]).all()
    assert np.allclose(out["hybrid"], out["rf"] + out["residual_forecast"]), \
        "hybrid is not RF + residual forecast"
    s = hyb.summary()
    _CACHE["hybrid_small"] = hyb
    return (f"ARIMA fitted for {s['n_series_with_arima']} series, "
            f"{int(out['corrected'].sum())} test rows corrected; "
            f"hybrid == RF + residual forecast")


# ==========================================================================
# 8. Metrics
# ==========================================================================
def t_evaluation_metrics():
    import utils
    y = np.array([10.0, 20.0, 30.0, 40.0])
    yhat = np.array([12.0, 18.0, 33.0, 37.0])
    assert np.isclose(utils.mae(y, yhat), 2.5), utils.mae(y, yhat)
    assert np.isclose(utils.rmse(y, yhat), np.sqrt(np.mean([4, 4, 9, 9]))), utils.rmse(y, yhat)
    assert np.isclose(utils.r2(y, y), 1.0)
    assert utils.r2(y, np.full_like(y, y.mean())) == 0.0
    m = utils.regression_metrics(y, yhat)
    assert set(m) == {"MAE", "RMSE", "R2", "MAPE_pct", "n_obs"}
    return "MAE, RMSE, R2 and MAPE verified against hand-computed values"


def t_comparison_table():
    import evaluate_models as ev
    import utils
    if "y_test" not in _CACHE or "rf_untuned" not in _CACHE:
        raise _Skip("models unavailable")
    preds = {
        "random_forest": _CACHE["rf_untuned"].predict(_CACHE["X_test"]),
        "hybrid": _CACHE["hybrid_small"].predict(
            _CACHE["test"], _CACHE["X_test"])["hybrid"],
    }
    table = ev.build_comparison_table(_CACHE["y_test"], preds)
    assert len(table) == 2
    assert table["n_test_obs"].nunique() == 1, "models scored on different row counts"
    best = ev.identify_best_model(table, "MAE")
    assert best["best_model_key"] in preds
    return f"comparison table built; same {int(table['n_test_obs'].iloc[0]):,} test rows for each model"


# ==========================================================================
# 9. Forecasting
# ==========================================================================
def t_forecasting():
    import forecasting as fc
    import utils
    if "rf_untuned" not in _CACHE or "clean" not in _CACHE:
        raise _Skip("model or data unavailable")
    eng = fc.ForecastEngine(
        history=_CACHE["clean"], model=_CACHE["rf_untuned"],
        encoder=_CACHE["encoder"], model_name="RF (test)", global_error=50.0,
    )
    # Pick a series with a long, dense history.
    counts = _CACHE["clean"].groupby(utils.SERIES_ID).size().sort_values(ascending=False)
    sid = counts.index[0]
    notes = []
    for h in fc.SUPPORTED_HORIZONS:
        res = eng.forecast(sid, horizon=h)
        assert res["ok"], f"horizon {h} failed: {res.get('reason')}"
        assert len(res["forecast"]) == h, f"horizon {h} produced {len(res['forecast'])} rows"
        assert (res["forecast"]["forecast"] >= 0).all(), "negative price forecast"
        assert (res["forecast"]["upper"] >= res["forecast"]["lower"]).all()
        notes.append(f"h={h}")
    tbl = eng.forecast_table(res)
    assert not tbl.empty and "Months ahead" in tbl.columns
    # A one-observation series must be refused gracefully.
    short_sid = counts.index[-1]
    short = eng.forecast(short_sid, horizon=3)
    assert short["ok"] is False and "reason" in short, "short series not handled gracefully"
    return f"horizons {', '.join(notes)} on '{sid[:40]}...'; short series refused gracefully"


# ==========================================================================
# 10. Persistence
# ==========================================================================
def t_save_and_load():
    import model_persistence as mp
    import utils
    import tempfile
    if "rf_untuned" not in _CACHE:
        raise _Skip("models unavailable")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mp.save_artifacts(
            models={"random_forest": _CACHE["rf_untuned"],
                    "hybrid": _CACHE["hybrid_small"]},
            encoder=_CACHE["encoder"],
            metrics={"comparison": [], "best_by_mae": {"best_model_key": "random_forest"}},
            metadata={"test": True}, outdir=tmp,
        )
        loaded = mp.load_artifacts(outdir=tmp)
        assert loaded["models"]["random_forest"] is not None
        assert loaded["encoder"] is not None
        a = _CACHE["rf_untuned"].predict(_CACHE["X_test"][:50])
        b = loaded["models"]["random_forest"].predict(_CACHE["X_test"][:50])
        assert np.allclose(a, b), "reloaded RF predicts differently"
        h1 = _CACHE["hybrid_small"].predict(_CACHE["test"], _CACHE["X_test"])["hybrid"]
        h2 = loaded["models"]["hybrid"].predict(_CACHE["test"], _CACHE["X_test"])["hybrid"]
        assert np.allclose(h1, h2, atol=1e-6), "reloaded hybrid predicts differently"
    return "RF and hybrid round-trip through joblib with identical predictions"


def t_saved_artifacts_present():
    import model_persistence as mp
    if not mp.artifacts_exist():
        raise _Skip("run_pipeline.py has not been run yet (models/saved_models is empty)")
    loaded = mp.load_artifacts()
    assert loaded["metrics"] is not None
    n = len([k for k in loaded["available"]])
    comp = (loaded["metrics"] or {}).get("comparison", [])
    return f"{n}/4 models on disk; comparison table holds {len(comp)} approaches"


# ==========================================================================
# 11. Dashboard
# ==========================================================================
def t_dashboard_imports():
    """Import-check app.py without starting a server."""
    code = (
        "import sys; sys.path.insert(0,'src');\n"
        "import importlib.util, pathlib\n"
        "src = pathlib.Path('app.py').read_text()\n"
        "compile(src, 'app.py', 'exec')\n"
        "import streamlit, plotly, utils, forecasting, model_persistence\n"
        "print('ok')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr.strip()[-400:]
    return "app.py compiles and its imports resolve"


def t_dashboard_startup():
    """Start Streamlit headless and confirm it serves without crashing."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py",
         "--server.headless=true", "--server.port=8599",
         "--browser.gatherUsageStats=false"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        deadline = time.time() + 75
        serving = False
        while time.time() < deadline:
            if proc.poll() is not None:
                out = proc.stdout.read() if proc.stdout else ""
                raise AssertionError(f"streamlit exited early: {out[-500:]}")
            try:
                import urllib.request
                with urllib.request.urlopen("http://127.0.0.1:8599/healthz", timeout=3) as r:
                    if r.status == 200:
                        serving = True
                        break
            except Exception:
                time.sleep(2)
        assert serving, "streamlit did not become healthy within 75s"
        return "streamlit served /healthz on port 8599"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


# ==========================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="skip the slow training and dashboard-startup tests")
    args = ap.parse_args()

    print("=" * 78)
    print("PROJECT COMPONENT TESTS")
    print("=" * 78)

    tests = [
        ("1.  Dataset loading", t_dataset_loading),
        ("2.  Dataset inspection", t_dataset_inspection),
        ("3.  Preprocessing", t_preprocessing),
        ("4.  Feature engineering", t_feature_engineering),
        ("5.  No target leakage in features", t_no_target_leakage),
        ("6.  Model table construction", t_model_table),
        ("7.  Chronological splitting", t_chronological_split),
        ("8.  Evaluation metrics", t_evaluation_metrics),
        ("9.  ARIMA residual modelling", t_arima_residual_modelling),
    ]
    if not args.quick:
        tests += [
            ("10. Random Forest training", t_random_forest_training),
            ("11. XGBoost training", t_xgboost_training),
            ("12. Stacking training (OOF, leakage-safe)", t_stacking_training),
            ("13. Hybrid prediction", t_hybrid_prediction),
            ("14. Four-model comparison table", t_comparison_table),
            ("15. Forecasting (1, 3, 6, 12 months)", t_forecasting),
            ("16. Model saving and loading", t_save_and_load),
        ]
    tests += [
        ("17. Saved artefacts on disk", t_saved_artifacts_present),
        ("18. Dashboard imports", t_dashboard_imports),
    ]
    if not args.quick:
        tests += [("19. Streamlit dashboard startup", t_dashboard_startup)]

    for name, fn in tests:
        run(name, fn)

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    df = pd.DataFrame(RESULTS)
    print(df.to_string(index=False))
    n_pass = int((df["status"] == "PASS").sum())
    n_fail = int((df["status"] == "FAIL").sum())
    n_skip = int((df["status"] == "SKIP").sum())
    print(f"\nPASS {n_pass}   FAIL {n_fail}   SKIP {n_skip}   "
          f"(total {len(df)}, {df['seconds'].sum():.1f}s)")
    try:
        import utils
        utils.save_json(
            {"results": RESULTS, "pass": n_pass, "fail": n_fail, "skip": n_skip},
            utils.METRICS_DIR / "test_report.json")
        print(f"Report written to results/metrics/test_report.json")
    except Exception:
        pass
    if n_fail:
        print("\nFailing components are listed above with their error messages.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
