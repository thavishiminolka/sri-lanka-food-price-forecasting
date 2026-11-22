"""
app.py - Streamlit dashboard
============================
Ensemble Learning for Sri Lankan Food Price Analysis and Forecasting.

Six pages: Project Overview, Food Price Explorer, Model Performance,
Food Price Forecasting, Model Insights, Dataset Explorer.

The dashboard **loads** the artefacts written by ``run_pipeline.py`` - it never
trains a model at start-up.  If the artefacts are missing it says so and explains
how to produce them, rather than silently falling back to an untrained model.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import utils  # noqa: E402
import model_persistence as mp  # noqa: E402
import forecasting as fcast  # noqa: E402

# ==========================================================================
# Page configuration and styling
# ==========================================================================
st.set_page_config(
    page_title="Sri Lanka Food Price Ensemble Learning",
    page_icon="🍚",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT = "#1565C0"
MODEL_COLORS = {
    "random_forest": "#2E7D32",
    "xgboost": "#1565C0",
    "stacking": "#6A1B9A",
    "hybrid": "#E65100",
}

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1400px;}
      h1, h2, h3 {letter-spacing: -0.01em;}
      .hero {
        background: linear-gradient(135deg, #0D47A1 0%, #1565C0 55%, #00897B 100%);
        color: #fff; padding: 1.6rem 1.9rem; border-radius: 14px; margin-bottom: 1.2rem;
      }
      .hero h1 {color:#fff; margin:0 0 .35rem 0; font-size:1.85rem; line-height:1.2;}
      .hero p {color:#E3F2FD; margin:0; font-size:1rem;}
      .card {
        background:#fff; border:1px solid #E3E7EB; border-left:5px solid #1565C0;
        border-radius:10px; padding:1rem 1.15rem; margin-bottom:.85rem;
        box-shadow:0 1px 3px rgba(16,24,40,.05);
      }
      .card h4 {margin:.1rem 0 .45rem 0; font-size:1.03rem;}
      .card p  {margin:0; color:#42526E; font-size:.92rem; line-height:1.5;}
      .badge {
        display:inline-block; padding:.18rem .6rem; border-radius:999px;
        font-size:.76rem; font-weight:600; background:#E8F0FE; color:#0D47A1;
        margin-right:.35rem;
      }
      .note {
        background:#FFF8E1; border:1px solid #FFE082; border-radius:9px;
        padding:.8rem 1rem; font-size:.88rem; color:#5D4037;
      }
      div[data-testid="stMetricValue"] {font-size:1.55rem;}
      section[data-testid="stSidebar"] {background:#F7F9FC;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================================
# Cached loaders
# ==========================================================================
@st.cache_data(show_spinner="Loading the cleaned dataset...")
def load_clean_data() -> pd.DataFrame | None:
    """Cleaned monthly panel, from parquet if present, otherwise rebuilt from the raw CSV."""
    if utils.CLEAN_PARQUET.exists():
        df = pd.read_parquet(utils.CLEAN_PARQUET)
    elif utils.CLEAN_CSV.exists():
        df = pd.read_csv(utils.CLEAN_CSV, parse_dates=["month_start", "date"])
    else:
        try:
            import data_loader
            import preprocessing
            raw, _ = data_loader.load_raw_data()
            df, _ = preprocessing.clean_food_prices(raw)
        except Exception:
            return None
    df["month_start"] = pd.to_datetime(df["month_start"])
    return df


@st.cache_data(show_spinner=False)
def load_model_table() -> pd.DataFrame | None:
    if utils.MODEL_TABLE_PARQUET.exists():
        df = pd.read_parquet(utils.MODEL_TABLE_PARQUET)
        df["month_start"] = pd.to_datetime(df["month_start"])
        return df
    return None


@st.cache_resource(show_spinner="Loading trained models...")
def load_models() -> dict:
    return mp.load_artifacts()


@st.cache_data(show_spinner=False)
def load_csv_metric(name: str) -> pd.DataFrame | None:
    path = utils.METRICS_DIR / name
    return pd.read_csv(path) if path.exists() else None


@st.cache_resource(show_spinner=False)
def build_forecast_engine(_models: dict, best_key: str):
    """Forecast engine wired to the best-performing saved model."""
    clean = load_clean_data()
    if clean is None or _models.get("encoder") is None:
        return None
    model = _models["models"].get(best_key)
    if model is None:
        for k in utils.MODEL_ORDER:
            if _models["models"].get(k) is not None:
                model, best_key = _models["models"][k], k
                break
    if model is None:
        return None

    series_err = {}
    meta = _models.get("metadata") or {}
    series_err = (meta.get("series_test_mae") or {}).get(best_key, {})
    metrics = _models.get("metrics") or {}
    global_err = None
    for row in metrics.get("comparison", []):
        if row.get("model_key") == best_key:
            global_err = row.get("MAE")
    return fcast.ForecastEngine(
        history=clean, model=model, encoder=_models["encoder"],
        model_name=utils.MODEL_LABELS.get(best_key, best_key),
        series_error=series_err, global_error=global_err,
    )


def artefact_status(models: dict) -> tuple[bool, str]:
    missing = models.get("missing") or []
    critical = [m for m in missing if m in {"random_forest", "encoder", "metrics", "metadata"}]
    if critical:
        return False, (
            "Trained artefacts are missing from `models/saved_models/`: "
            + ", ".join(critical)
            + ".\n\nRun the training pipeline once to create them:\n"
            "```bash\npython run_pipeline.py        # or: python run_pipeline.py --fast\n```"
        )
    return True, ""


def best_model_key(models: dict) -> str:
    metrics = models.get("metrics") or {}
    best = metrics.get("best_by_mae") or {}
    key = best.get("best_model_key")
    if key in utils.MODEL_ORDER:
        return key
    return "random_forest"


def comparison_frame(models: dict) -> pd.DataFrame | None:
    metrics = models.get("metrics") or {}
    rows = metrics.get("comparison")
    if rows:
        return pd.DataFrame(rows)
    return load_csv_metric("model_comparison.csv")


# ==========================================================================
# Sidebar
# ==========================================================================
models = load_models()
clean = load_clean_data()
ok, problem = artefact_status(models)

st.sidebar.markdown("### Sri Lanka Food Prices")
st.sidebar.caption("Ensemble learning: analysis and forecasting")

PAGES = [
    "1 - Project Overview",
    "2 - Food Price Explorer",
    "3 - Model Performance",
    "4 - Food Price Forecasting",
    "5 - Model Insights",
    "6 - Dataset Explorer",
]
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")

st.sidebar.divider()
if clean is not None:
    st.sidebar.markdown("**Dataset loaded**")
    st.sidebar.caption(
        f"{len(clean):,} monthly observations  \n"
        f"{clean['commodity'].nunique()} commodities, {clean['market'].nunique()} markets  \n"
        f"{clean['month_start'].min():%b %Y} to {clean['month_start'].max():%b %Y}"
    )
else:
    st.sidebar.error("Dataset not found. See the Dataset Explorer page.")

if ok:
    st.sidebar.success(f"Models loaded: {len(models['available'])}/4")
    bk = best_model_key(models)
    st.sidebar.caption(f"Best on test MAE: **{utils.MODEL_LABELS.get(bk, bk)}**")
else:
    st.sidebar.warning("Trained models not found")

st.sidebar.divider()
st.sidebar.caption(
    "Data: World Food Programme (WFP), via the Humanitarian Data Exchange (HDX). "
    "Prices in Sri Lankan Rupees (LKR). Forecasts are model estimates, not guaranteed prices."
)


def require_data() -> bool:
    if clean is None:
        st.error(
            "The dataset could not be loaded. Place the WFP CSV at "
            f"`data/raw/{utils.RAW_CSV_NAME}` and run `python run_pipeline.py`."
        )
        return False
    return True


def require_models() -> bool:
    if not ok:
        st.warning(problem)
        return False
    return True


# ==========================================================================
# PAGE 1 - Project Overview
# ==========================================================================
if page == PAGES[0]:
    st.markdown(
        """
        <div class="hero">
          <h1>Ensemble Learning for Sri Lankan Food Price Analysis and Forecasting</h1>
          <p>Four ensemble approaches - bagging, boosting, stacking and a Random Forest + ARIMA
          hybrid - compared on World Food Programme monthly retail and wholesale food prices.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if clean is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Monthly observations", f"{len(clean):,}")
        c2.metric("Commodities", clean["commodity"].nunique())
        c3.metric("Markets", clean["market"].nunique())
        c4.metric(
            "Period covered",
            f"{clean['month_start'].min():%Y} - {clean['month_start'].max():%Y}",
        )

    st.subheader("Introduction")
    st.markdown(
        "Food prices are the sharpest everyday measure of economic stress. In Sri Lanka the "
        "2022 currency and fuel crisis pushed staple prices up steeply, and households, "
        "traders and relief agencies all had to make decisions against a moving target. "
        "This project treats that question as a supervised learning problem: given a "
        "commodity's recent price history in a specific market, what will it cost next month? "
        "It answers it with **ensemble learning** - combining many models instead of trusting one."
    )

    st.subheader("Problem statement")
    st.markdown(
        """
        Monthly food prices in Sri Lanka are hard to model for three reasons at once:

        * **Heterogeneity.** The panel covers staples, vegetables, fish, pulses and fuel,
          measured in kilograms, litres, units and 750 ml bottles. A single global model
          must handle prices from around 11 LKR to over 3,700 LKR.
        * **Non-stationarity.** The 2022 inflation episode means the recent level is
          nothing like the historical mean, so a model that assumes a stable average fails.
        * **Short, uneven series.** Most commodity-market series in the file begin in 2023,
          and many have missing months, which rules out fitting a classical time-series
          model independently to each one.

        A single decision tree overfits this; a single linear model cannot capture the
        non-linear interactions between commodity, market, season and recent momentum.
        Ensembles are the natural answer, and the project's job is to establish
        *which* ensemble strategy actually works best here, measured on data the models
        have never seen.
        """
    )

    st.subheader("Objectives")
    st.markdown(
        """
        1. Load and document the real WFP Sri Lanka food prices dataset, with provenance.
        2. Clean it and establish its structure without assuming column names or coverage.
        3. Analyse historical price behaviour across commodities, markets and time.
        4. Engineer leakage-free, calendar-aligned time-series features.
        5. Train four ensemble approaches with time-aware validation.
        6. Compare them on MAE, RMSE and R² over an identical, strictly future test period.
        7. Interpret what drives the predictions, and where the models fail.
        8. Deliver an interactive dashboard for exploration and forecasting.
        """
    )

    st.subheader("Dataset")
    info = utils.DATASET_INFO
    meta = models.get("metadata") or {}
    st.markdown(
        f"""
        | Field | Value |
        |---|---|
        | Title | {info['title']} |
        | Provider | {info['provider']} |
        | Source | [HDX resource page]({info['source_page']}) |
        | Download date | {info['download_date']} |
        | Frequency | Monthly (every observation dated the 15th) |
        | Currency | LKR, with a WFP-supplied USD column |
        | Price types | Retail and Wholesale |
        | Licence | {info['licence']} |
        | Attribution | {info['attribution']} |
        """
    )

    st.subheader("The four ensemble techniques")
    cols = st.columns(2)
    blurbs = [
        ("A. Bagging - Random Forest",
         "Many deep trees, each on a bootstrap resample of the rows and a random subset of "
         "features, grown in parallel and averaged. Averaging decorrelated high-variance "
         "trees cuts variance without adding bias."),
        ("B. Boosting - XGBoost",
         "Trees built sequentially, each fitted to the errors the ensemble still makes. "
         "Second-order gradient steps, shrinkage and regularisation reduce bias stage by "
         "stage, where bagging reduces variance all at once."),
        ("C. Stacking - RF + XGBoost + Linear Regression, Ridge meta-learner",
         "Three different algorithms predict the same target; a Ridge meta-learner trained "
         "on out-of-fold predictions learns how much to trust each. Here the folds are "
         "time-ordered, so the meta-learner never sees a prediction from a model that had "
         "seen that month."),
        ("D. Hybrid - Random Forest + ARIMA",
         "The forest predicts the price level from tabular features; ARIMA then models the "
         "autocorrelation left in the forest's out-of-fold residuals, per price series, and "
         "its residual forecast is added back. The forest captures structure, ARIMA captures "
         "what the forest's independence assumption discards."),
    ]
    for i, (title, body) in enumerate(blurbs):
        with cols[i % 2]:
            st.markdown(
                f'<div class="card"><h4>{title}</h4><p>{body}</p></div>',
                unsafe_allow_html=True,
            )

    if ok:
        comp = comparison_frame(models)
        if comp is not None and not comp.empty:
            st.subheader("Headline result")
            bk = best_model_key(models)
            row = comp[comp["model_key"] == bk].iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Best approach (test MAE)", utils.MODEL_LABELS.get(bk, bk))
            c2.metric("MAE", f"{row['MAE']:,.2f} LKR")
            c3.metric("RMSE", f"{row['RMSE']:,.2f} LKR")
            c4.metric("R²", f"{row['R2']:.4f}")
            st.caption(
                f"Measured on {int(row['n_test_obs']):,} observations from "
                f"{(models.get('metrics') or {}).get('test_period', 'the test period')}, "
                "none of which were used for training or tuning."
            )
    st.info(
        "Forecasts in this dashboard are model-based estimates. They reflect historical "
        "price dynamics only and cannot account for future policy changes, import "
        "restrictions, weather shocks or currency movements.",
        icon="ℹ️",
    )


# ==========================================================================
# PAGE 2 - Food Price Explorer
# ==========================================================================
elif page == PAGES[1]:
    st.title("Food Price Explorer")
    st.caption(
        "Historical monthly prices in Sri Lankan Rupees. Prices are only ever compared "
        "within the same measurement unit."
    )
    if require_data():
        commodities = sorted(clean["commodity"].dropna().unique())
        c1, c2, c3 = st.columns([2, 2, 1.4])
        with c1:
            commodity = st.selectbox("Commodity", commodities,
                                     index=commodities.index("Rice (white)")
                                     if "Rice (white)" in commodities else 0)
        sub_c = clean[clean["commodity"] == commodity]
        markets = sorted(sub_c["market"].dropna().unique())
        with c2:
            market = st.selectbox("Market", markets)
        with c3:
            ptypes = sorted(sub_c["pricetype"].dropna().unique())
            ptype = st.selectbox("Price type", ptypes)

        sub = sub_c[(sub_c["market"] == market) & (sub_c["pricetype"] == ptype)]
        if sub.empty:
            st.warning(
                f"No {ptype.lower()} observations for {commodity} in {market}. "
                "Try another market or price type."
            )
        else:
            dmin = sub["month_start"].min().date()
            dmax = sub["month_start"].max().date()
            if dmin == dmax:
                st.info(f"This series has a single observation, in {dmin:%B %Y}.")
                date_range = (dmin, dmax)
            else:
                date_range = st.slider(
                    "Date range", min_value=dmin, max_value=dmax,
                    value=(dmin, dmax), format="YYYY-MM",
                )
            view = sub[
                (sub["month_start"].dt.date >= date_range[0])
                & (sub["month_start"].dt.date <= date_range[1])
            ].sort_values("month_start")

            if view.empty:
                st.warning("No observations in the selected date range.")
            else:
                unit = view["unit"].iloc[0]
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Latest price", f"{view['price'].iloc[-1]:,.2f} LKR/{unit}",
                          help=f"as of {view['month_start'].iloc[-1]:%B %Y}")
                k2.metric("Mean over range", f"{view['price'].mean():,.2f} LKR/{unit}")
                first, last = view["price"].iloc[0], view["price"].iloc[-1]
                k3.metric("Change over range",
                          f"{(last - first) / first * 100:+.1f}%" if first else "n/a")
                k4.metric("Volatility (sd)", f"{view['price'].std(ddof=0):,.2f} LKR")

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=view["month_start"], y=view["price"], mode="lines+markers",
                    name=f"{commodity} - {market}", line=dict(color=ACCENT, width=2.4),
                    marker=dict(size=5),
                    hovertemplate="%{x|%b %Y}<br>LKR %{y:,.2f}/" + unit + "<extra></extra>",
                ))
                outl = view[view["is_outlier"]] if "is_outlier" in view.columns else pd.DataFrame()
                if not outl.empty:
                    fig.add_trace(go.Scatter(
                        x=outl["month_start"], y=outl["price"], mode="markers",
                        name="flagged as extreme", marker=dict(color="#C62828", size=10,
                                                               symbol="circle-open", line=dict(width=2)),
                        hovertemplate="%{x|%b %Y}<br>LKR %{y:,.2f}<extra>extreme</extra>",
                    ))
                fig.update_layout(
                    title=f"{commodity} - {ptype.lower()} price in {market}",
                    xaxis_title="Month", yaxis_title=f"Price (LKR per {unit})",
                    height=430, hovermode="x unified",
                    margin=dict(t=60, b=40), legend=dict(orientation="h", y=1.02, x=0),
                )
                st.plotly_chart(fig, use_container_width=True)

                mom = view[["month_start", "price"]].copy()
                mom["change_pct"] = mom["price"].pct_change() * 100
                fig2 = px.bar(
                    mom.dropna(), x="month_start", y="change_pct",
                    title=f"Month-on-month price change - {commodity} in {market}",
                    labels={"month_start": "Month", "change_pct": "Change (%)"},
                    color="change_pct", color_continuous_scale="RdYlGn_r",
                )
                fig2.update_layout(height=330, coloraxis_showscale=False, margin=dict(t=60))
                st.plotly_chart(fig2, use_container_width=True)

        st.divider()
        st.subheader("Commodity comparison")
        st.caption("Only commodities sharing a measurement unit are plotted together.")
        unit_choices = sorted(clean["unit"].dropna().unique())
        cc1, cc2 = st.columns([1, 3])
        with cc1:
            cmp_unit = st.selectbox("Measurement unit", unit_choices,
                                    index=unit_choices.index("KG") if "KG" in unit_choices else 0)
        pool = clean[(clean["unit"] == cmp_unit) & (clean["pricetype"] == "Retail")]
        if pool.empty:
            pool = clean[clean["unit"] == cmp_unit]
        options = sorted(pool["commodity"].dropna().unique())
        with cc2:
            default = [c for c in ["Rice (white)", "Wheat flour", "Coconut", "Tomatoes"]
                       if c in options][:4] or options[:3]
            picked = st.multiselect(f"Commodities measured in {cmp_unit}", options, default=default)
        if not picked:
            st.info("Select at least one commodity to compare.")
        else:
            nat = (
                pool[pool["commodity"].isin(picked)]
                .groupby(["month_start", "commodity"], observed=True)["price"]
                .mean().reset_index()
            )
            figc = px.line(
                nat, x="month_start", y="price", color="commodity",
                title=f"Average monthly price across all markets - commodities priced per {cmp_unit}",
                labels={"month_start": "Month", "price": f"Mean price (LKR per {cmp_unit})",
                        "commodity": "Commodity"},
            )
            figc.update_layout(height=430, hovermode="x unified", margin=dict(t=60))
            st.plotly_chart(figc, use_container_width=True)
            st.caption(
                "Each line averages every market reporting that commodity in a month, so it "
                "reflects both price moves and changes in which markets reported."
            )

        st.divider()
        st.subheader("Market comparison")
        mc1, mc2 = st.columns([2, 2])
        with mc1:
            mk_commodity = st.selectbox("Commodity to compare across markets", commodities,
                                        index=commodities.index(commodity)
                                        if commodity in commodities else 0,
                                        key="mkt_cmp_commodity")
        mk_pool = clean[clean["commodity"] == mk_commodity]
        mk_options = sorted(mk_pool["market"].dropna().unique())
        with mc2:
            mk_picked = st.multiselect("Markets", mk_options, default=mk_options[:5],
                                       key="mkt_cmp_markets")
        if not mk_picked:
            st.info("Select at least one market.")
        else:
            mdata = mk_pool[mk_pool["market"].isin(mk_picked)].sort_values("month_start")
            u = "/".join(sorted(mdata["unit"].astype(str).unique()))
            figm = px.line(
                mdata, x="month_start", y="price", color="market", line_dash="pricetype",
                title=f"{mk_commodity} across selected markets",
                labels={"month_start": "Month", "price": f"Price (LKR per {u})",
                        "market": "Market", "pricetype": "Price type"},
            )
            figm.update_layout(height=450, hovermode="x unified", margin=dict(t=60))
            st.plotly_chart(figm, use_container_width=True)

            latest = (
                mdata.sort_values("month_start").groupby("market", observed=True).tail(1)
                [["market", "month_start", "price", "unit", "pricetype"]]
                .sort_values("price", ascending=False)
            )
            latest["month_start"] = latest["month_start"].dt.strftime("%Y-%m")
            st.dataframe(
                latest.rename(columns={
                    "market": "Market", "month_start": "Latest month",
                    "price": "Price (LKR)", "unit": "Unit", "pricetype": "Price type"}),
                use_container_width=True, hide_index=True,
            )


# ==========================================================================
# PAGE 3 - Model Performance
# ==========================================================================
elif page == PAGES[2]:
    st.title("Model Performance")
    st.caption(
        "The four selected ensemble approaches, scored on the same chronological test set. "
        "No standalone baseline is reported as a competitor."
    )
    if require_models():
        comp = comparison_frame(models)
        metrics = models.get("metrics") or {}
        if comp is None or comp.empty:
            st.error("No comparison metrics found. Run `python run_pipeline.py`.")
        else:
            bk = best_model_key(models)
            best_row = comp[comp["model_key"] == bk].iloc[0]
            st.success(
                f"**Best-performing approach on actual test results: "
                f"{utils.MODEL_LABELS.get(bk, bk)}** - "
                f"MAE {best_row['MAE']:,.2f} LKR, RMSE {best_row['RMSE']:,.2f} LKR, "
                f"R² {best_row['R2']:.4f} over {int(best_row['n_test_obs']):,} test observations "
                f"({metrics.get('test_period', 'test period')})."
            )

            show = comp.copy()
            show["MAE (LKR)"] = show["MAE"].round(2)
            show["RMSE (LKR)"] = show["RMSE"].round(2)
            show["R²"] = show["R2"].round(4)
            show["MAPE (%)"] = show["MAPE_pct"].round(2)
            st.dataframe(
                show[["Approach", "MAE (LKR)", "RMSE (LKR)", "R²", "MAPE (%)", "n_test_obs"]]
                .rename(columns={"n_test_obs": "Test observations"}),
                use_container_width=True, hide_index=True,
            )

            t1, t2, t3, t4 = st.tabs(["MAE", "RMSE", "R²", "MAPE"])
            for tab, metric, label, asc in [
                (t1, "MAE", "Mean Absolute Error (LKR)", True),
                (t2, "RMSE", "Root Mean Squared Error (LKR)", True),
                (t3, "R2", "R² (coefficient of determination)", False),
                (t4, "MAPE_pct", "Mean Absolute Percentage Error (%)", True),
            ]:
                with tab:
                    d = comp.sort_values(metric, ascending=asc)
                    fig = px.bar(
                        d, x="Approach", y=metric, text=d[metric].round(4),
                        color="model_key",
                        color_discrete_map=MODEL_COLORS,
                        labels={metric: label, "Approach": "Ensemble approach"},
                        title=f"{label} - lower is better" if asc
                        else f"{label} - higher is better",
                    )
                    fig.update_traces(textposition="outside")
                    fig.update_layout(height=430, showlegend=False, margin=dict(t=60),
                                      xaxis_tickangle=-12)
                    if metric == "R2":
                        fig.update_yaxes(range=[max(0, d[metric].min() - 0.02), 1.0])
                    st.plotly_chart(fig, use_container_width=True)

            per_month = metrics.get("per_test_month")
            if per_month:
                st.subheader("Error by test month")
                pm = pd.DataFrame(per_month)
                figpm = px.line(
                    pm, x="month", y="MAE", color="model_key", markers=True,
                    color_discrete_map=MODEL_COLORS,
                    labels={"month": "Test month", "MAE": "MAE (LKR)", "model_key": "Model"},
                    title="Mean absolute error by test month - error grows with the forecast horizon",
                )
                figpm.update_layout(height=400, hovermode="x unified", margin=dict(t=60))
                st.plotly_chart(figpm, use_container_width=True)
                st.caption(
                    "All models are trained to December 2024, so later test months sit "
                    "further from the training data."
                )

            st.subheader("Actual versus predicted")
            preds = load_csv_metric("test_predictions_residuals.csv")
            if preds is not None and not preds.empty:
                sel = st.selectbox(
                    "Approach", [k for k in utils.MODEL_ORDER if k in set(preds["model_key"])],
                    format_func=lambda k: utils.MODEL_LABELS.get(k, k),
                )
                d = preds[preds["model_key"] == sel]
                if len(d) > 4000:
                    d = d.sample(4000, random_state=utils.RANDOM_STATE)
                lim = float(max(d["actual"].max(), d["predicted"].max()))
                figs = go.Figure()
                figs.add_trace(go.Scatter(
                    x=d["actual"], y=d["predicted"], mode="markers",
                    marker=dict(size=4, opacity=0.35, color=MODEL_COLORS.get(sel, ACCENT)),
                    name="test observations",
                    text=d["commodity"] + " - " + d["market"],
                    hovertemplate="%{text}<br>actual LKR %{x:,.2f}<br>predicted LKR %{y:,.2f}<extra></extra>",
                ))
                figs.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines",
                                          line=dict(dash="dash", color="black"),
                                          name="perfect prediction"))
                figs.update_layout(
                    title=f"Actual versus predicted - {utils.MODEL_LABELS.get(sel, sel)}",
                    xaxis_title="Actual price (LKR, in the row's own unit)",
                    yaxis_title="Predicted price (LKR)", height=520, margin=dict(t=60),
                )
                st.plotly_chart(figs, use_container_width=True)
            else:
                st.info("Per-observation predictions not found. Run `python run_pipeline.py`.")

            st.subheader("Why the models differ")
            st.markdown(metrics.get("interpretation", "_Interpretation not available._"))

            with st.expander("Strengths and limitations of each approach"):
                st.markdown(
                    """
                    **Bagging - Random Forest.** Strong on heterogeneous tabular panels, needs
                    little tuning, robust to outliers because it averages. Cannot extrapolate
                    beyond the price range it saw in training, and treats rows as independent,
                    so it ignores autocorrelation in its own errors.

                    **Boosting - XGBoost.** Usually the most accurate on tabular data, handles
                    interactions efficiently, and regularises explicitly. More sensitive to
                    hyperparameters, and because each tree chases the current residuals it can
                    fit noise in the short, thin series that dominate this file.

                    **Stacking.** Can only beat its base learners when their errors are
                    *different*. The RF and XGBoost base learners here see the same features and
                    tend to be wrong on the same rows, which caps the gain. It is also the most
                    expensive to train and the hardest to explain to a non-technical audience.

                    **Hybrid - RF + ARIMA.** Directly targets the forest's blind spot, serial
                    correlation in residuals, and is the only approach here with an explicit
                    time-series component. Its reach is limited by data: a series needs enough
                    training residuals for ARIMA to estimate anything, and most series in this
                    dataset only start in 2023.
                    """
                )

            hs = metrics.get("hybrid_summary") or {}
            if hs:
                with st.expander("Hybrid model coverage"):
                    h1, h2, h3 = st.columns(3)
                    h1.metric("Series with an ARIMA residual model",
                              f"{hs.get('n_series_with_arima', 0):,}")
                    h2.metric("Series falling back to Random Forest only",
                              f"{hs.get('n_series_without_arima', 0):,}")
                    h3.metric("Minimum residuals required",
                              hs.get("min_residuals_for_arima", "n/a"))
                    if hs.get("arima_order_counts"):
                        st.write("**Selected ARIMA orders (by AIC):**")
                        st.dataframe(
                            pd.DataFrame(
                                sorted(hs["arima_order_counts"].items(),
                                       key=lambda kv: -kv[1]),
                                columns=["ARIMA order (p,d,q)", "Number of series"]),
                            use_container_width=True, hide_index=True,
                        )


# ==========================================================================
# PAGE 4 - Food Price Forecasting
# ==========================================================================
elif page == PAGES[3]:
    st.title("Food Price Forecasting")
    st.markdown(
        '<div class="note"><b>These are estimates, not guaranteed prices.</b> '
        "The forecast extends recent price dynamics forward. It has no information about "
        "future policy, import rules, fuel costs, weather or exchange rates, and a recursive "
        "forecast compounds its own error as the horizon grows.</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    if require_data() and require_models():
        bk = best_model_key(models)
        engine = build_forecast_engine(models, bk)
        if engine is None:
            st.error("Could not build the forecasting engine. Run `python run_pipeline.py`.")
        else:
            available = [k for k in utils.MODEL_ORDER if models["models"].get(k) is not None
                         and k != "hybrid"]
            c1, c2, c3, c4 = st.columns([2, 2, 1.3, 1.6])
            commodities = engine.available_commodities()
            with c1:
                commodity = st.selectbox("Commodity", commodities,
                                         index=commodities.index("Rice (white)")
                                         if "Rice (white)" in commodities else 0)
            with c2:
                mkts = engine.markets_for_commodity(commodity)
                market = st.selectbox("Market", mkts)
            with c3:
                horizon = st.selectbox("Horizon (months)", list(fcast.SUPPORTED_HORIZONS),
                                       index=2)
            with c4:
                model_key = st.selectbox(
                    "Model", available, index=available.index(bk) if bk in available else 0,
                    format_func=lambda k: utils.MODEL_LABELS.get(k, k),
                )

            opts = engine.series_options(commodity, market)
            if opts.empty:
                st.warning(f"No price series for {commodity} in {market}.")
            else:
                if len(opts) > 1:
                    labels = {
                        r[utils.SERIES_ID]: f"{r['unit']} - {r['pricetype']} "
                                            f"({int(r['n_obs'])} observations)"
                        for _, r in opts.iterrows()
                    }
                    sid = st.radio("Unit and price type", list(labels),
                                   format_func=lambda s: labels[s], horizontal=True)
                else:
                    sid = opts.iloc[0][utils.SERIES_ID]

                series_rows = engine.get_series(sid)
                n_obs = len(series_rows)
                if n_obs < fcast.MIN_HISTORY_FOR_FORECAST:
                    st.warning(
                        f"This series has only {n_obs} monthly observation(s). At least "
                        f"{fcast.MIN_HISTORY_FOR_FORECAST} are needed before the lag and "
                        "rolling features a forecast depends on can be built. Choose another "
                        "market or commodity with a longer history."
                    )
                else:
                    with st.spinner("Forecasting..."):
                        res = engine.forecast(
                            sid, horizon=horizon,
                            model=models["models"][model_key],
                            model_name=utils.MODEL_LABELS.get(model_key, model_key),
                        )
                    if not res.get("ok"):
                        st.warning(res.get("reason", "Forecast unavailable for this series."))
                    else:
                        unit = res["unit"]
                        fc_df = res["forecast"]
                        k1, k2, k3, k4 = st.columns(4)
                        k1.metric("Last actual price",
                                  f"{res['last_observed_price']:,.2f} LKR/{unit}",
                                  help=f"observed {res['last_observed_month']}")
                        final = float(fc_df["forecast"].iloc[-1])
                        k2.metric(f"Forecast, +{res['horizon_produced']} months",
                                  f"{final:,.2f} LKR/{unit}",
                                  delta=f"{(final - res['last_observed_price']) / res['last_observed_price'] * 100:+.1f}%")
                        k3.metric("History available", f"{res['n_history_obs']} months")
                        k4.metric("Model", res["model_name"])

                        hist = res["history"].sort_values("month_start")
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=hist["month_start"], y=hist["actual"], mode="lines+markers",
                            name="Actual (observed)", line=dict(color="#263238", width=2.4),
                            marker=dict(size=5),
                            hovertemplate="%{x|%b %Y}<br>actual LKR %{y:,.2f}<extra></extra>",
                        ))
                        bridge_x = [hist["month_start"].iloc[-1]] + list(fc_df["month_start"])
                        bridge_y = [hist["actual"].iloc[-1]] + list(fc_df["forecast"])
                        fig.add_trace(go.Scatter(
                            x=bridge_x, y=bridge_y, mode="lines+markers",
                            name="Forecast (predicted)",
                            line=dict(color="#E65100", width=2.6, dash="dash"),
                            marker=dict(size=7, symbol="diamond"),
                            hovertemplate="%{x|%b %Y}<br>forecast LKR %{y:,.2f}<extra></extra>",
                        ))
                        fig.add_trace(go.Scatter(
                            x=list(fc_df["month_start"]) + list(fc_df["month_start"][::-1]),
                            y=list(fc_df["upper"]) + list(fc_df["lower"][::-1]),
                            fill="toself", fillcolor="rgba(230,81,0,0.14)",
                            line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
                            name="Indicative range",
                        ))
                        fig.add_vline(
                            x=hist["month_start"].iloc[-1], line_dash="dot",
                            line_color="#9E9E9E",
                            annotation_text="last actual", annotation_position="top left",
                        )
                        fig.update_layout(
                            title=f"{commodity} in {market} - history and {res['horizon_produced']}-month forecast",
                            xaxis_title="Month", yaxis_title=f"Price (LKR per {unit})",
                            height=500, hovermode="x unified", margin=dict(t=70),
                            legend=dict(orientation="h", y=1.03, x=0),
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        st.subheader("Forecast results")
                        st.dataframe(engine.forecast_table(res),
                                     use_container_width=True, hide_index=True)
                        st.caption(
                            f"Method: {res['method']}. "
                            "The indicative range is scaled from this series' own historical "
                            "test error and widened with the horizon; it is not a calibrated "
                            "prediction interval."
                        )
                        with st.expander("What can make this forecast wrong"):
                            for c in res["caveats"]:
                                st.markdown(f"- {c}")


# ==========================================================================
# PAGE 5 - Model Insights
# ==========================================================================
elif page == PAGES[4]:
    st.title("Model Insights")
    st.caption("What drives the predictions, and where the models go wrong.")
    if require_models():
        fi = models.get("feature_importance") or {}
        tabs = st.tabs(["Feature importance", "Residual analysis",
                        "Error by commodity", "How the four approaches differ"])

        with tabs[0]:
            c1, c2 = st.columns(2)
            rf_imp = pd.DataFrame(fi.get("random_forest") or []) \
                if fi.get("random_forest") else load_csv_metric("feature_importance_random_forest.csv")
            xgb_imp = pd.DataFrame(fi.get("xgboost") or []) \
                if fi.get("xgboost") else load_csv_metric("feature_importance_xgboost.csv")
            with c1:
                if rf_imp is not None and not rf_imp.empty:
                    d = rf_imp.head(18).iloc[::-1]
                    fig = px.bar(d, x="importance", y="feature", orientation="h",
                                 title="Random Forest - mean decrease in impurity",
                                 labels={"importance": "Importance", "feature": "Feature"})
                    fig.update_traces(marker_color=MODEL_COLORS["random_forest"])
                    fig.update_layout(height=560, margin=dict(t=60))
                    st.plotly_chart(fig, use_container_width=True)
            with c2:
                if xgb_imp is not None and not xgb_imp.empty:
                    d = xgb_imp.head(18).iloc[::-1]
                    fig = px.bar(d, x="importance", y="feature", orientation="h",
                                 title="XGBoost - gain",
                                 labels={"importance": "Importance", "feature": "Feature"})
                    fig.update_traces(marker_color=MODEL_COLORS["xgboost"])
                    fig.update_layout(height=560, margin=dict(t=60))
                    st.plotly_chart(fig, use_container_width=True)

            perm = pd.DataFrame(fi.get("permutation_random_forest") or []) \
                if fi.get("permutation_random_forest") \
                else load_csv_metric("permutation_importance_random_forest.csv")
            if perm is not None and not perm.empty:
                st.subheader("Permutation importance (test set)")
                st.caption(
                    "Impurity-based importance is computed on the training data and favours "
                    "high-cardinality features. Permutation importance shuffles one feature at "
                    "a time on the held-out test set and measures the drop in accuracy, so it "
                    "reflects what the model actually relies on to generalise."
                )
                d = perm.head(18).iloc[::-1]
                fig = px.bar(d, x="importance_mean", y="feature", orientation="h",
                             error_x="importance_std" if "importance_std" in d else None,
                             labels={"importance_mean": "Increase in RMSE when shuffled",
                                     "feature": "Feature"},
                             title="Permutation importance - Random Forest")
                fig.update_traces(marker_color="#00695C")
                fig.update_layout(height=560, margin=dict(t=60))
                st.plotly_chart(fig, use_container_width=True)

        with tabs[1]:
            preds = load_csv_metric("test_predictions_residuals.csv")
            if preds is None or preds.empty:
                st.info("Residual data not found. Run `python run_pipeline.py`.")
            else:
                sel = st.selectbox(
                    "Approach", [k for k in utils.MODEL_ORDER if k in set(preds["model_key"])],
                    format_func=lambda k: utils.MODEL_LABELS.get(k, k), key="resid_model",
                )
                d = preds[preds["model_key"] == sel]
                m1, m2, m3 = st.columns(3)
                m1.metric("Mean residual", f"{d['residual'].mean():,.2f} LKR",
                          help="Far from zero would mean a systematic bias.")
                m2.metric("Residual sd", f"{d['residual'].std(ddof=0):,.2f} LKR")
                m3.metric("Median absolute error", f"{d['abs_error'].median():,.2f} LKR")

                plot_d = d.sample(4000, random_state=utils.RANDOM_STATE) if len(d) > 4000 else d
                c1, c2 = st.columns(2)
                with c1:
                    fig = px.scatter(
                        plot_d, x="predicted", y="residual", opacity=0.35,
                        labels={"predicted": "Predicted price (LKR)",
                                "residual": "Residual (actual - predicted, LKR)"},
                        title="Residuals against predicted values",
                        hover_data=["commodity", "market"],
                    )
                    fig.update_traces(marker=dict(color=MODEL_COLORS.get(sel, ACCENT), size=4))
                    fig.add_hline(y=0, line_dash="dash", line_color="black")
                    fig.update_layout(height=430, margin=dict(t=60))
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    fig = px.histogram(
                        d, x="residual", nbins=70, title="Error distribution",
                        labels={"residual": "Residual (LKR)", "count": "Test observations"},
                    )
                    fig.update_traces(marker_color=MODEL_COLORS.get(sel, ACCENT))
                    fig.add_vline(x=0, line_dash="dash", line_color="black")
                    fig.update_layout(height=430, margin=dict(t=60))
                    st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    "A residual cloud that fans out with the predicted price means the model's "
                    "absolute error grows with price level - expected when a single model spans "
                    "commodities from 11 to over 3,700 LKR."
                )

                st.subheader("Actual versus predicted for one series")
                sids = sorted(d[utils.SERIES_ID].unique())
                pick = st.selectbox("Series", sids, key="series_pick")
                dd = preds[preds[utils.SERIES_ID] == pick]
                figs = go.Figure()
                act = dd.drop_duplicates(subset=["month_start"]).sort_values("month_start")
                figs.add_trace(go.Scatter(x=act["month_start"], y=act["actual"],
                                          mode="lines+markers", name="Actual",
                                          line=dict(color="#263238", width=3)))
                for k, g in dd.groupby("model_key"):
                    g = g.sort_values("month_start")
                    figs.add_trace(go.Scatter(
                        x=g["month_start"], y=g["predicted"], mode="lines+markers",
                        name=utils.MODEL_LABELS.get(k, k),
                        line=dict(color=MODEL_COLORS.get(k, "#888"), dash="dash")))
                figs.update_layout(title=f"Test-period predictions - {pick}",
                                   xaxis_title="Month", yaxis_title="Price (LKR)",
                                   height=440, hovermode="x unified", margin=dict(t=60))
                st.plotly_chart(figs, use_container_width=True)

        with tabs[2]:
            bycom = load_csv_metric("metrics_by_commodity.csv")
            if bycom is None or bycom.empty:
                st.info("Per-commodity metrics not found. Run `python run_pipeline.py`.")
            else:
                sel = st.selectbox(
                    "Approach", [k for k in utils.MODEL_ORDER if k in set(bycom["model_key"])],
                    format_func=lambda k: utils.MODEL_LABELS.get(k, k), key="bycom_model")
                d = bycom[bycom["model_key"] == sel].sort_values("MAPE_pct", ascending=False)
                fig = px.bar(
                    d.head(20), x="MAPE_pct", y="commodity", orientation="h",
                    color="mean_actual_price", color_continuous_scale="Viridis",
                    labels={"MAPE_pct": "MAPE (%)", "commodity": "Commodity",
                            "mean_actual_price": "Mean price (LKR)"},
                    title="Hardest commodities to predict (percentage error, test set)",
                    hover_data=["MAE", "RMSE", "R2", "units", "n_obs"],
                )
                fig.update_layout(height=620, margin=dict(t=60))
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    "Percentage error is used here because commodities are measured in "
                    "different units and at very different price levels, so comparing raw "
                    "LKR errors across them would be misleading."
                )
                st.dataframe(
                    d[["commodity", "units", "MAE", "RMSE", "R2", "MAPE_pct", "n_obs"]]
                    .round(3).rename(columns={"MAPE_pct": "MAPE (%)", "n_obs": "Test obs"}),
                    use_container_width=True, hide_index=True,
                )

        with tabs[3]:
            st.markdown(
                """
                ### Bagging, boosting, stacking and hybrid - as built here

                **Bagging (Random Forest)** reduces *variance*. It draws many bootstrap
                resamples of the 18,428 training rows, grows a deep tree on each while
                restricting every split to a random subset of the 34 features, and averages
                the predictions. The trees are individually unstable but their errors are
                largely independent, so averaging cancels much of that instability. Trees are
                grown **in parallel** and never see each other.

                **Boosting (XGBoost)** reduces *bias*. The first tree is a crude fit; each
                later tree is trained on the gradient of the loss that remains, so the
                ensemble is a sum of many shallow trees, each correcting its predecessors.
                Trees are grown **sequentially** and every one depends on all those before it.
                That is more powerful and more prone to fitting noise, which is why the
                learning rate, depth and subsampling all have to be tuned.

                **Stacking** reduces *model-choice risk*. Rather than deciding in advance
                whether the forest, the booster or a linear fit is right, it trains all three
                and lets a Ridge regression learn the weighting from out-of-fold predictions.
                The out-of-fold step is what makes it honest: the meta-learner must see
                predictions for rows the base models had not been fitted on, and here the folds
                are time-ordered as well, so it never learns from a prediction made with
                knowledge of the future.

                **Hybrid (RF + ARIMA)** addresses a *structural* gap rather than a
                bias-variance one. The other three all treat each row as independent, which
                throws away the fact that these rows form time series. If the forest has
                under-predicted a series for several months, that is information. ARIMA is
                fitted to the forest's out-of-fold residuals per series and its forecast of the
                next residuals is added back to the forest's prediction.

                ### Ensemble learning on tabular versus time-series data

                The tabular framing is what makes this project work at all. The dataset has
                over a thousand short commodity-market series, most beginning in 2023 - far too
                short to fit a separate time-series model to each. Pooling them into one
                supervised table, with the time structure carried in lag and rolling features
                and the identity carried in categorical features, lets every series borrow
                strength from every other. That is why the pooled tree ensembles achieve a high
                R² on genuinely future months.

                The cost of that framing is exactly what the hybrid tries to recover. A
                tabular model has no notion of a forecast horizon, cannot extrapolate past the
                price range it was trained on, and assumes its errors are independent. The
                sharpest practical lesson from this project is the asymmetry: features
                describing the recent *level* of a series - last month's price and the rolling
                means - take the top places in all three importance rankings, while calendar
                seasonality lands mid-pack and the rolling standard deviations fall into the
                bottom third. The volatility features were a reasonable hypothesis that the
                measurements did not support. For monthly food prices in an inflationary economy,
                where the price has recently been matters far more than the time of year, or than
                how much it has been moving around.
                """
            )


# ==========================================================================
# PAGE 6 - Dataset Explorer
# ==========================================================================
elif page == PAGES[5]:
    st.title("Dataset Explorer")
    info = utils.DATASET_INFO
    st.caption(f"{info['title']} - {info['provider']}. {info['attribution']}")

    if clean is None:
        st.error(
            f"Dataset not found. Download the CSV from {info['source_page']}, save it as "
            f"`data/raw/{utils.RAW_CSV_NAME}`, then run `python run_pipeline.py`."
        )
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Rows", f"{len(clean):,}")
        c2.metric("Commodities", clean["commodity"].nunique())
        c3.metric("Markets", clean["market"].nunique())
        c4.metric("Price series", clean[utils.SERIES_ID].nunique())
        c5.metric("Months covered", clean["month_start"].nunique())

        st.subheader("Date range and coverage")
        st.write(
            f"The cleaned panel runs from **{clean['month_start'].min():%B %Y}** to "
            f"**{clean['month_start'].max():%B %Y}**, with every observation dated the 15th "
            "of its month."
        )
        per_month = clean.groupby("month_start").size().reset_index(name="observations")
        fig = px.bar(per_month, x="month_start", y="observations",
                     title="Observations per month - coverage expands sharply from mid-2023",
                     labels={"month_start": "Month", "observations": "Number of observations"})
        fig.update_traces(marker_color=ACCENT)
        fig.update_layout(height=360, margin=dict(t=60))
        st.plotly_chart(fig, use_container_width=True)
        st.info(
            "This coverage shape matters for the modelling. Roughly 84% of all observations "
            "fall in June 2023 or later, so most series are short. That is why the project "
            "pools every series into one supervised table rather than fitting a separate "
            "time-series model per commodity and market.",
            icon="📌",
        )

        st.subheader("Data preview")
        cols = [c for c in ["month_start", "admin1", "market", "category", "commodity",
                            "unit", "pricetype", "currency", "price", "usdprice"]
                if c in clean.columns]
        st.dataframe(clean[cols].head(200), use_container_width=True, hide_index=True)

        t1, t2, t3 = st.tabs(["Commodities", "Markets", "Data quality"])
        with t1:
            com = (
                clean.groupby(["commodity", "unit", "category"], observed=True)
                .agg(observations=("price", "size"), first_month=("month_start", "min"),
                     last_month=("month_start", "max"), mean_price=("price", "mean"),
                     min_price=("price", "min"), max_price=("price", "max"))
                .reset_index().sort_values("observations", ascending=False)
            )
            com["first_month"] = com["first_month"].dt.strftime("%Y-%m")
            com["last_month"] = com["last_month"].dt.strftime("%Y-%m")
            st.dataframe(com.round(2), use_container_width=True, hide_index=True)
            figc = px.bar(
                com.head(25), x="observations", y="commodity", orientation="h", color="category",
                title="Commodities by number of observations",
                labels={"observations": "Observations", "commodity": "Commodity"},
            )
            figc.update_layout(height=640, margin=dict(t=60))
            st.plotly_chart(figc, use_container_width=True)
        with t2:
            mk = (
                clean.groupby(["market", "admin1"], observed=True)
                .agg(observations=("price", "size"), commodities=("commodity", "nunique"),
                     first_month=("month_start", "min"), last_month=("month_start", "max"))
                .reset_index().sort_values("observations", ascending=False)
            )
            mk["first_month"] = mk["first_month"].dt.strftime("%Y-%m")
            mk["last_month"] = mk["last_month"].dt.strftime("%Y-%m")
            st.dataframe(mk, use_container_width=True, hide_index=True)
            figm = px.bar(
                mk.head(25), x="observations", y="market", orientation="h", color="admin1",
                title="Markets by number of observations",
                labels={"observations": "Observations", "market": "Market",
                        "admin1": "Province"},
            )
            figm.update_layout(height=640, margin=dict(t=60))
            st.plotly_chart(figm, use_container_width=True)
        with t3:
            rep_path = utils.METRICS_DIR / "preprocessing_report.json"
            if rep_path.exists():
                rep = utils.load_json(rep_path)
                q1, q2, q3, q4 = st.columns(4)
                q1.metric("Rows in", f"{rep.get('n_rows_input', 0):,}")
                q2.metric("Rows out", f"{rep.get('n_rows_output', 0):,}")
                q3.metric("Duplicates removed", f"{rep.get('duplicate_full_rows', 0):,}")
                q4.metric("Extreme values flagged", f"{rep.get('n_outliers_flagged', 0):,}")
                st.markdown(
                    f"""
                    - **Missing prices or dates:** {rep.get('rows_dropped_missing_price', 0)} rows
                      dropped for a missing price, {rep.get('rows_dropped_unparseable_date', 0)}
                      for an unparseable date.
                    - **Missing geography:** {rep.get('admin1_missing_filled', 0)} rows had no
                      province. They belong to
                      {rep.get('admin1_missing_markets', ['the national-average market'])}, which
                      has no single district, so they were labelled
                      `{rep.get('admin1_fill_value', 'National')}` and kept.
                    - **Outliers:** {rep.get('n_outliers_flagged', 0)} flagged,
                      {rep.get('outliers_removed', 0)} removed. {rep.get('outlier_policy', '')}
                    - **Series with gaps:** {rep.get('n_series_with_gaps', 0):,} of
                      {rep.get('n_series', 0):,}. {rep.get('gap_policy', '')}
                    - **Units present:** {rep.get('units_present', [])} - never pooled into one
                      comparison.
                    - **Currency:** {rep.get('currencies_present', [])}.
                    """
                )
            miss = clean.isna().sum()
            miss = miss[miss > 0]
            if len(miss):
                st.write("**Remaining missing values in the cleaned panel:**")
                st.dataframe(
                    miss.rename("missing").reset_index()
                    .rename(columns={"index": "column"}),
                    use_container_width=True, hide_index=True,
                )
            else:
                st.success("No missing values remain in the cleaned panel.")

            st.write("**Saved model artefacts:**")
            st.dataframe(mp.describe_artifacts(), use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "Ensemble Learning for Sri Lankan Food Price Analysis and Forecasting - "
    "data: World Food Programme (WFP) via HDX. Built with scikit-learn, XGBoost, "
    "statsmodels, Plotly and Streamlit. Forecasts are estimates and may be affected by "
    "economic conditions, market changes and data limitations."
)
