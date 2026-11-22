"""
data_loader.py
==============
Loading the WFP "Sri Lanka - Food Prices" dataset.

Three ways of getting the data, tried in this order:

1. A CSV already sitting in ``data/raw/`` (this is how the project was built -
   the file was supplied by the user and copied into ``data/raw/``).
2. A download from the HDX resource URL, when the environment has internet
   access to data.humdata.org.
3. A file path passed in by the caller, or a Streamlit/Colab file upload.

The WFP files published on HDX carry a second header row of HXL hash tags
(``#date``, ``#adm1+name``, ...).  ``read_food_prices`` detects and removes that
row if it is present, so the loader works with either variant of the file.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

try:  # allow both "import src.data_loader" and "import data_loader"
    from . import utils
except ImportError:  # pragma: no cover
    import utils

LOGGER = utils.LOGGER

HDX_RESOURCE_URL = (
    "https://data.humdata.org/dataset/wfp-food-prices-for-sri-lanka/"
    "resource/3638f0d6-9969-48cf-a919-1d879d037ec6/download/"
    "wfp_food_prices_lka.csv"
)

EXPECTED_COLUMNS = [
    "date",
    "admin1",
    "admin2",
    "market",
    "market_id",
    "latitude",
    "longitude",
    "category",
    "commodity",
    "commodity_id",
    "unit",
    "priceflag",
    "pricetype",
    "currency",
    "price",
    "usdprice",
]


# --------------------------------------------------------------------------
def _strip_hxl_row(df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    """Drop the HXL tag row (``#date``, ``#adm1+name`` ...) if it is the first row."""
    if len(df) == 0:
        return df, False
    first = df.iloc[0].astype(str)
    looks_like_hxl = first.str.startswith("#").sum() >= max(2, len(df.columns) // 3)
    if looks_like_hxl:
        LOGGER.info("HXL tag row detected in the CSV header - removing it.")
        return df.iloc[1:].reset_index(drop=True), True
    return df, False


def read_food_prices(path_or_buffer) -> pd.DataFrame:
    """Read a WFP food-prices CSV into a DataFrame of strings (no type coercion yet).

    Type conversion is deliberately left to :mod:`preprocessing` so that the
    notebook can inspect the raw values exactly as they arrive.
    """
    df = pd.read_csv(path_or_buffer, dtype=str, keep_default_na=True)
    df.columns = [str(c).strip().lower() for c in df.columns]
    df, _ = _strip_hxl_row(df)
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        LOGGER.warning(
            "Columns named in the WFP schema that are absent from this file: %s. "
            "The pipeline inspects whatever columns are present rather than assuming them.",
            missing,
        )
    return df


def download_from_hdx(target: Optional[Path] = None, timeout: int = 60) -> Optional[Path]:
    """Try to download the resource from HDX.  Returns the saved path, or None.

    Any network failure is reported and swallowed: the caller falls back to the
    local copy of the file.  The project never substitutes synthetic data.
    """
    target = Path(target or utils.RAW_CSV_PATH)
    try:
        import requests  # imported lazily so the package is optional
    except ImportError:
        LOGGER.warning("`requests` is not installed - cannot attempt the HDX download.")
        return None

    try:
        LOGGER.info("Attempting download from HDX: %s", HDX_RESOURCE_URL)
        resp = requests.get(HDX_RESOURCE_URL, timeout=timeout)
        resp.raise_for_status()
        # Sanity-check that we received a CSV and not an HTML error/login page.
        head = resp.content[:200].decode("utf-8", errors="ignore").lower()
        if "<html" in head:
            LOGGER.warning("HDX returned an HTML page rather than a CSV - download rejected.")
            return None
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(resp.content)
        LOGGER.info("Downloaded %.1f KB to %s", len(resp.content) / 1024, target)
        return target
    except Exception as exc:  # noqa: BLE001 - any failure falls back to the local file
        LOGGER.warning("HDX download failed (%s: %s).", type(exc).__name__, exc)
        LOGGER.warning(
            "Falling back to the local CSV. If none exists, place the WFP file at %s "
            "or pass its path to load_raw_data(csv_path=...).",
            utils.RAW_CSV_PATH,
        )
        return None


def load_raw_data(
    csv_path: Optional[str | Path] = None,
    uploaded_file: Optional[io.BytesIO] = None,
    try_download: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """Load the raw dataset and return ``(dataframe, provenance_dict)``.

    Parameters
    ----------
    csv_path
        Explicit path to a CSV. Takes priority over everything else.
    uploaded_file
        A file-like object, e.g. from ``st.file_uploader`` or Colab's
        ``files.upload()``.
    try_download
        If no local file is found, attempt the HDX download.
    """
    provenance = dict(utils.DATASET_INFO)

    if uploaded_file is not None:
        LOGGER.info("Reading the dataset from an uploaded file object.")
        df = read_food_prices(uploaded_file)
        provenance["load_method"] = "uploaded file object"
        return df, _finalise_provenance(df, provenance)

    if csv_path is not None:
        p = Path(csv_path)
        if not p.exists():
            raise FileNotFoundError(f"CSV not found: {p}")
        LOGGER.info("Reading the dataset from %s", p)
        df = read_food_prices(p)
        provenance["load_method"] = f"explicit path: {p}"
        return df, _finalise_provenance(df, provenance)

    if utils.RAW_CSV_PATH.exists():
        LOGGER.info("Reading the dataset from the project copy %s", utils.RAW_CSV_PATH)
        df = read_food_prices(utils.RAW_CSV_PATH)
        provenance["load_method"] = f"local file: data/raw/{utils.RAW_CSV_NAME}"
        return df, _finalise_provenance(df, provenance)

    # Nothing local - look for any CSV in data/raw before going to the network.
    candidates = sorted(utils.RAW_DIR.glob("*.csv"))
    if candidates:
        LOGGER.info("Using the CSV found in data/raw: %s", candidates[0].name)
        df = read_food_prices(candidates[0])
        provenance["load_method"] = f"local file: data/raw/{candidates[0].name}"
        return df, _finalise_provenance(df, provenance)

    if try_download and download_from_hdx() is not None:
        df = read_food_prices(utils.RAW_CSV_PATH)
        provenance["load_method"] = "downloaded from HDX"
        return df, _finalise_provenance(df, provenance)

    raise FileNotFoundError(
        "No dataset available.\n"
        "Manual setup:\n"
        f"  1. Download the CSV from {utils.DATASET_INFO['source_page']}\n"
        f"  2. Save it as {utils.RAW_CSV_PATH}\n"
        "  3. Re-run this function.\n"
        "The project will not generate synthetic data in place of the real dataset."
    )


def _finalise_provenance(df: pd.DataFrame, provenance: dict) -> dict:
    provenance["n_rows_raw"] = int(len(df))
    provenance["n_columns_raw"] = int(df.shape[1])
    provenance["columns_raw"] = list(df.columns)
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce")
        provenance["date_min"] = str(dates.min().date()) if dates.notna().any() else None
        provenance["date_max"] = str(dates.max().date()) if dates.notna().any() else None
    return provenance


# --------------------------------------------------------------------------
def inspect_raw(df: pd.DataFrame) -> dict:
    """Produce the dataset-inspection report required before preprocessing.

    Nothing here assumes a column exists: every lookup is guarded, so the report
    describes whatever the supplied file actually contains.
    """
    report: dict = {
        "shape": tuple(df.shape),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values": {c: int(df[c].isna().sum()) for c in df.columns},
        "missing_pct": {c: round(float(df[c].isna().mean() * 100), 3) for c in df.columns},
        "duplicate_rows_full": int(df.duplicated().sum()),
        "memory_mb": round(float(df.memory_usage(deep=True).sum() / 1024**2), 2),
    }

    if "date" in df.columns:
        d = pd.to_datetime(df["date"], errors="coerce")
        report["date"] = {
            "min": str(d.min()) if d.notna().any() else None,
            "max": str(d.max()) if d.notna().any() else None,
            "n_unique": int(d.nunique()),
            "unparseable": int(d.isna().sum()),
            "days_of_month_present": sorted(int(x) for x in d.dt.day.dropna().unique()),
        }

    for col in ("commodity", "market", "unit", "pricetype", "currency", "priceflag",
                "category", "admin1", "admin2"):
        if col in df.columns:
            vc = df[col].value_counts(dropna=False)
            report[col] = {
                "n_unique": int(df[col].nunique(dropna=True)),
                "top_values": {str(k): int(v) for k, v in vc.head(15).items()},
            }

    for col in ("price", "usdprice", "latitude", "longitude"):
        if col in df.columns:
            num = pd.to_numeric(df[col], errors="coerce")
            report[f"{col}_numeric"] = {
                "unparseable": int(num.isna().sum() - df[col].isna().sum()),
                "min": None if num.notna().sum() == 0 else float(num.min()),
                "max": None if num.notna().sum() == 0 else float(num.max()),
                "mean": None if num.notna().sum() == 0 else float(num.mean()),
                "median": None if num.notna().sum() == 0 else float(num.median()),
                "n_zero_or_negative": int((num <= 0).sum()),
            }

    key_cols = [c for c in utils.SERIES_KEYS if c in df.columns]
    if key_cols and "date" in df.columns:
        report["duplicate_rows_on_series_and_date"] = int(
            df.duplicated(subset=key_cols + ["date"]).sum()
        )
        grp = df.groupby(key_cols, observed=True).size()
        report["series"] = {
            "definition": " + ".join(key_cols) + " (one monthly price series each)",
            "n_series": int(len(grp)),
            "obs_per_series_min": int(grp.min()),
            "obs_per_series_median": float(grp.median()),
            "obs_per_series_max": int(grp.max()),
            "series_with_at_least_12_obs": int((grp >= 12).sum()),
            "series_with_at_least_24_obs": int((grp >= 24).sum()),
            "series_with_at_least_36_obs": int((grp >= 36).sum()),
        }

    if "commodity" in df.columns and "unit" in df.columns:
        multi = df.groupby("commodity", observed=True)["unit"].nunique()
        report["commodities_with_multiple_units"] = {
            str(k): int(v) for k, v in multi[multi > 1].items()
        }

    return report


def print_inspection(report: dict) -> None:
    """Human-readable printout of :func:`inspect_raw` for the notebook."""
    print("=" * 78)
    print("DATASET INSPECTION REPORT")
    print("=" * 78)
    print(f"Rows x columns      : {report['shape'][0]:,} x {report['shape'][1]}")
    print(f"Memory (deep)       : {report['memory_mb']} MB")
    print(f"Duplicate full rows : {report['duplicate_rows_full']:,}")
    if "duplicate_rows_on_series_and_date" in report:
        print(f"Duplicates on series+date : {report['duplicate_rows_on_series_and_date']:,}")
    print("\nColumns and dtypes:")
    for c in report["columns"]:
        miss = report["missing_values"][c]
        pct = report["missing_pct"][c]
        print(f"  {c:<14} {report['dtypes'][c]:<8} missing={miss:>6,} ({pct:>6.2f}%)")
    if "date" in report:
        d = report["date"]
        print(f"\nDate range          : {d['min']} -> {d['max']}")
        print(f"Unique dates        : {d['n_unique']}")
        print(f"Days of month seen  : {d['days_of_month_present']}")
    if "price_numeric" in report:
        p = report["price_numeric"]
        print(
            f"\nPrice (LKR)         : min={p['min']:,.2f}  median={p['median']:,.2f}  "
            f"max={p['max']:,.2f}  zero/negative={p['n_zero_or_negative']}"
        )
    for col in ("currency", "priceflag", "pricetype", "unit", "category"):
        if col in report:
            print(f"\n{col} ({report[col]['n_unique']} unique):")
            for k, v in report[col]["top_values"].items():
                print(f"    {k:<28} {v:>7,}")
    for col in ("commodity", "market", "admin1"):
        if col in report:
            print(f"\n{col}: {report[col]['n_unique']} unique. Most frequent:")
            for k, v in list(report[col]["top_values"].items())[:10]:
                print(f"    {k:<32} {v:>7,}")
    if "series" in report:
        s = report["series"]
        print("\nSeries structure")
        print(f"  Definition        : {s['definition']}")
        print(f"  Number of series  : {s['n_series']:,}")
        print(
            f"  Obs per series    : min={s['obs_per_series_min']}, "
            f"median={s['obs_per_series_median']}, max={s['obs_per_series_max']}"
        )
        print(f"  >=12 obs          : {s['series_with_at_least_12_obs']:,}")
        print(f"  >=24 obs          : {s['series_with_at_least_24_obs']:,}")
        print(f"  >=36 obs          : {s['series_with_at_least_36_obs']:,}")
    if report.get("commodities_with_multiple_units"):
        print(f"\nCommodities measured in more than one unit: "
              f"{report['commodities_with_multiple_units']}")
    else:
        print("\nEvery commodity is recorded in a single measurement unit.")
    print("=" * 78)


if __name__ == "__main__":
    frame, prov = load_raw_data()
    print_inspection(inspect_raw(frame))
    print("\nProvenance:", prov)
