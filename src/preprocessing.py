"""
preprocessing.py
================
Cleaning the raw WFP food-prices table into a tidy monthly panel.

Every decision taken here is recorded in the returned ``report`` dict so that
the notebook and the dashboard can show *what* was changed and *why*, rather
than asserting that the data was "cleaned".

Design decisions (and the reasoning):

* **Dates** are parsed to ``datetime`` and normalised to a month period.  Every
  observation in the supplied file falls on the 15th of the month, so the data
  is a monthly panel and ``month_start`` is the natural time axis.
* **Missing prices / dates** are dropped - a record without a price or a date
  cannot support supervised learning.  The count is reported.
* **Missing ``admin1`` / ``admin2`` / coordinates** are *not* dropped.  In this
  file they belong to the ``National Average`` market, which has no single
  district; they are labelled ``"National"`` / ``"Unknown"`` so the rows remain
  usable.
* **Duplicates** are checked both as whole rows and as repeated
  (series, month) pairs.  Repeated pairs are collapsed to their mean price,
  because a repeated monthly observation for the same commodity-market-unit
  can only be an aggregation artefact.
* **Outliers are flagged, never deleted.**  Food prices in Sri Lanka genuinely
  spiked during the 2022 economic crisis, so an extreme value is usually a real
  event.  A robust per-series MAD test adds an ``is_outlier`` flag for analysis.
* **Units and currency** are kept as columns and as part of the series key, so
  that prices measured in KG, L, Unit and 750 ML are never pooled into one
  comparison.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

try:
    from . import utils
except ImportError:  # pragma: no cover
    import utils

LOGGER = utils.LOGGER

NUMERIC_COLUMNS = ["price", "usdprice", "latitude", "longitude"]
TEXT_COLUMNS = [
    "admin1", "admin2", "market", "category", "commodity",
    "unit", "priceflag", "pricetype", "currency",
]


def clean_food_prices(
    df_raw: pd.DataFrame,
    outlier_mad_threshold: float = 6.0,
) -> Tuple[pd.DataFrame, Dict]:
    """Clean the raw frame.  Returns ``(clean_df, report)``."""
    report: Dict = {"n_rows_input": int(len(df_raw))}
    df = df_raw.copy()

    # ---------------------------------------------------------------- text
    for col in TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()
    # Collapse whitespace inside market/commodity names (e.g. "Keppetipola (DEC}")
    for col in ("market", "commodity"):
        if col in df.columns:
            df[col] = df[col].str.replace(r"\s+", " ", regex=True)

    # ---------------------------------------------------------------- dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    bad_dates = int(df["date"].isna().sum())
    report["rows_dropped_unparseable_date"] = bad_dates
    if bad_dates:
        df = df[df["date"].notna()]

    df["month_start"] = df["date"].dt.to_period("M").dt.to_timestamp()
    report["days_of_month_in_source"] = sorted(int(x) for x in df["date"].dt.day.unique())
    report["is_monthly_panel"] = len(report["days_of_month_in_source"]) == 1

    # ---------------------------------------------------------------- numerics
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            before_na = int(df[col].isna().sum())
            df[col] = pd.to_numeric(df[col], errors="coerce")
            report[f"{col}_newly_unparseable"] = int(df[col].isna().sum() - before_na)

    # Target must exist and be positive to be usable.
    n_before = len(df)
    df = df[df["price"].notna()]
    report["rows_dropped_missing_price"] = int(n_before - len(df))

    n_before = len(df)
    df = df[df["price"] > 0]
    report["rows_dropped_nonpositive_price"] = int(n_before - len(df))

    # ---------------------------------------------------------------- geography
    for col, filler in (("admin1", "National"), ("admin2", "Unknown")):
        if col in df.columns:
            n_missing = int(df[col].isna().sum())
            report[f"{col}_missing_filled"] = n_missing
            report[f"{col}_fill_value"] = filler
            if n_missing:
                missing_markets = sorted(df.loc[df[col].isna(), "market"].unique().tolist())
                report[f"{col}_missing_markets"] = missing_markets
            df[col] = df[col].fillna(filler)

    for col in ("latitude", "longitude"):
        if col in df.columns:
            report[f"{col}_missing_left_as_na"] = int(df[col].isna().sum())

    # ---------------------------------------------------------------- consistency
    if "currency" in df.columns:
        report["currencies_present"] = sorted(df["currency"].dropna().unique().tolist())
        report["single_currency"] = len(report["currencies_present"]) == 1
    if "priceflag" in df.columns:
        report["priceflags_present"] = sorted(df["priceflag"].dropna().unique().tolist())
    if "pricetype" in df.columns:
        report["pricetypes_present"] = sorted(df["pricetype"].dropna().unique().tolist())
    if "unit" in df.columns:
        report["units_present"] = sorted(df["unit"].dropna().unique().tolist())
        report["rows_per_unit"] = {
            str(k): int(v) for k, v in df["unit"].value_counts().items()
        }

    # ---------------------------------------------------------------- duplicates
    report["duplicate_full_rows"] = int(df.duplicated().sum())
    if report["duplicate_full_rows"]:
        df = df.drop_duplicates()

    df[utils.SERIES_ID] = utils.make_series_id(df)
    dup_mask = df.duplicated(subset=[utils.SERIES_ID, "month_start"], keep=False)
    report["rows_in_duplicate_series_month_groups"] = int(dup_mask.sum())

    if dup_mask.any():
        LOGGER.info(
            "Collapsing %d rows that share a (series, month) pair to their mean price.",
            int(dup_mask.sum()),
        )
        agg = {c: "first" for c in df.columns
               if c not in {"price", "usdprice", utils.SERIES_ID, "month_start"}}
        agg["price"] = "mean"
        if "usdprice" in df.columns:
            agg["usdprice"] = "mean"
        df = (
            df.groupby([utils.SERIES_ID, "month_start"], as_index=False, observed=True)
            .agg(agg)
        )
    report["n_rows_after_duplicate_handling"] = int(len(df))

    # ---------------------------------------------------------------- outlier flags
    df = df.sort_values([utils.SERIES_ID, "month_start"]).reset_index(drop=True)
    grp = df.groupby(utils.SERIES_ID, observed=True)["price"]
    med = grp.transform("median")
    mad = grp.transform(lambda s: np.median(np.abs(s - np.median(s))))
    # 1.4826 scales the MAD to be comparable with a standard deviation.
    scaled_mad = 1.4826 * mad
    robust_z = np.where(scaled_mad > 0, (df["price"] - med) / scaled_mad.replace(0, np.nan), 0.0)
    df["robust_z"] = pd.Series(robust_z, index=df.index).fillna(0.0)
    df["is_outlier"] = df["robust_z"].abs() > outlier_mad_threshold
    report["outlier_rule"] = (
        f"|robust z| > {outlier_mad_threshold}, where robust z uses the per-series "
        "median and median absolute deviation"
    )
    report["n_outliers_flagged"] = int(df["is_outlier"].sum())
    report["outliers_removed"] = 0
    report["outlier_policy"] = (
        "Flagged only. Extreme food prices in 2022-2023 reflect Sri Lanka's real "
        "inflation episode, so removing them would delete genuine signal."
    )
    if report["n_outliers_flagged"]:
        top = (
            df.loc[df["is_outlier"]]
            .nlargest(10, "robust_z")[[utils.SERIES_ID, "month_start", "price", "robust_z"]]
        )
        report["largest_positive_outliers"] = [
            {
                "series": r[utils.SERIES_ID],
                "month": str(pd.Timestamp(r["month_start"]).date()),
                "price": float(r["price"]),
                "robust_z": round(float(r["robust_z"]), 2),
            }
            for _, r in top.iterrows()
        ]

    # ---------------------------------------------------------------- series stats
    sizes = df.groupby(utils.SERIES_ID, observed=True).size()
    report["n_series"] = int(len(sizes))
    report["obs_per_series"] = {
        "min": int(sizes.min()),
        "median": float(sizes.median()),
        "max": int(sizes.max()),
        "ge_12": int((sizes >= 12).sum()),
        "ge_24": int((sizes >= 24).sum()),
        "ge_36": int((sizes >= 36).sum()),
    }

    # Gap diagnostics: does each series have a continuous run of months?
    spans = df.groupby(utils.SERIES_ID, observed=True)["month_start"].agg(["min", "max", "count"])
    spans["span_months"] = (
        (spans["max"].dt.year - spans["min"].dt.year) * 12
        + (spans["max"].dt.month - spans["min"].dt.month)
        + 1
    )
    spans["is_contiguous"] = spans["count"] == spans["span_months"]
    report["n_series_contiguous_monthly"] = int(spans["is_contiguous"].sum())
    report["n_series_with_gaps"] = int((~spans["is_contiguous"]).sum())
    report["gap_policy"] = (
        "Gaps are left as gaps. Lags are computed over the observed ordering within "
        "each series, and the pipeline records that a lag may therefore span more "
        "than one calendar month for series with missing months."
    )

    df["year"] = df["month_start"].dt.year
    df["month"] = df["month_start"].dt.month
    df["quarter"] = df["month_start"].dt.quarter

    report["n_rows_output"] = int(len(df))
    report["date_min"] = str(df["month_start"].min().date())
    report["date_max"] = str(df["month_start"].max().date())
    report["n_commodities"] = int(df["commodity"].nunique())
    report["n_markets"] = int(df["market"].nunique())

    keep = [
        "month_start", "date", "year", "month", "quarter",
        "admin1", "admin2", "market", "market_id", "latitude", "longitude",
        "category", "commodity", "commodity_id", "unit", "priceflag", "pricetype",
        "currency", "price", "usdprice", utils.SERIES_ID, "robust_z", "is_outlier",
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].sort_values(["month_start", utils.SERIES_ID]).reset_index(drop=True)

    LOGGER.info(
        "Cleaning complete: %d -> %d rows, %d series, %s to %s",
        report["n_rows_input"], report["n_rows_output"],
        report["n_series"], report["date_min"], report["date_max"],
    )
    return df, report


def print_cleaning_report(report: Dict) -> None:
    print("=" * 78)
    print("PREPROCESSING REPORT")
    print("=" * 78)
    print(f"Rows in                        : {report['n_rows_input']:,}")
    print(f"Dropped - unparseable date     : {report['rows_dropped_unparseable_date']:,}")
    print(f"Dropped - missing price        : {report['rows_dropped_missing_price']:,}")
    print(f"Dropped - price <= 0           : {report['rows_dropped_nonpositive_price']:,}")
    print(f"Duplicate full rows removed    : {report['duplicate_full_rows']:,}")
    print(f"Rows in duplicate series-month : {report['rows_in_duplicate_series_month_groups']:,}")
    print(f"Rows out                       : {report['n_rows_output']:,}")
    print()
    print(f"Monthly panel?                 : {report['is_monthly_panel']} "
          f"(days of month present: {report['days_of_month_in_source']})")
    print(f"Date coverage                  : {report['date_min']} -> {report['date_max']}")
    print(f"Currencies                     : {report.get('currencies_present')}")
    print(f"Price flags                    : {report.get('priceflags_present')}")
    print(f"Price types                    : {report.get('pricetypes_present')}")
    print(f"Units                          : {report.get('units_present')}")
    print(f"Commodities / markets          : {report['n_commodities']} / {report['n_markets']}")
    print()
    print(f"admin1 missing filled          : {report.get('admin1_missing_filled')} "
          f"-> '{report.get('admin1_fill_value')}' "
          f"(markets: {report.get('admin1_missing_markets')})")
    print()
    print(f"Series (commodity|market|unit|pricetype): {report['n_series']:,}")
    ops = report["obs_per_series"]
    print(f"  observations per series      : min={ops['min']}, median={ops['median']}, "
          f"max={ops['max']}")
    print(f"  series with >=12 / >=24 / >=36 obs : {ops['ge_12']:,} / {ops['ge_24']:,} / "
          f"{ops['ge_36']:,}")
    print(f"  contiguous monthly series    : {report['n_series_contiguous_monthly']:,}")
    print(f"  series containing gaps       : {report['n_series_with_gaps']:,}")
    print()
    print(f"Outlier rule                   : {report['outlier_rule']}")
    print(f"Outliers flagged / removed     : {report['n_outliers_flagged']:,} / "
          f"{report['outliers_removed']}")
    print(f"Policy                         : {report['outlier_policy']}")
    print("=" * 78)


if __name__ == "__main__":
    try:
        from . import data_loader
    except ImportError:
        import data_loader
    raw, _ = data_loader.load_raw_data()
    clean, rep = clean_food_prices(raw)
    print_cleaning_report(rep)
    clean.to_parquet(utils.CLEAN_PARQUET, index=False)
    print("saved:", utils.CLEAN_PARQUET)
