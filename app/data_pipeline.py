"""
Data ingestion and cleaning pipeline.

Uses ALL Kaggle files from "Store Sales - Time Series Forecasting":
  - data/train.csv         : main sales records (3M rows)
  - data/stores.csv        : store metadata (type, cluster, city, state)
  - data/oil.csv           : daily oil price (Ecuador economy depends on oil)
  - data/holidays_events.csv: Ecuadorian holidays and events
  - data/transactions.csv  : number of transactions per store per day

All files are merged into one enriched daily time series for forecasting.
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = "data"


def load_all_files():
    """Load every Kaggle CSV file."""
    train = pd.read_csv(
        os.path.join(DATA_DIR, "train.csv"),
        parse_dates=["date"],
        usecols=["date", "store_nbr", "family", "sales", "onpromotion"],
    )

    stores = pd.read_csv(os.path.join(DATA_DIR, "stores.csv"))

    oil = pd.read_csv(
        os.path.join(DATA_DIR, "oil.csv"),
        parse_dates=["date"],
    )

    holidays = pd.read_csv(
        os.path.join(DATA_DIR, "holidays_events.csv"),
        parse_dates=["date"],
    )

    transactions = pd.read_csv(
        os.path.join(DATA_DIR, "transactions.csv"),
        parse_dates=["date"],
    )

    return train, stores, oil, holidays, transactions


def ingest_data():
    """
    Load and merge all files into one enriched daily sales DataFrame.

    Final columns:
        date, sales, onpromotion, oil_price, is_holiday, transactions
    """
    train, stores, oil, holidays, transactions = load_all_files()

    # Step 1: Aggregate train to daily totals (sum across all stores + families)
    daily = (
        train.groupby("date", as_index=False)
        .agg(sales=("sales", "sum"), onpromotion=("onpromotion", "sum"))
    )

    # Step 2: Merge oil prices (fill missing oil prices via forward-fill)
    oil = oil.rename(columns={"dcoilwtico": "oil_price"})
    daily = daily.merge(oil, on="date", how="left")

    # Step 3: Mark holidays
    # Keep only national holidays (most impactful on total sales)
    national_holidays = holidays[
        (holidays["locale"] == "National") & (holidays["transferred"] == False)
    ][["date"]].drop_duplicates()
    national_holidays["is_holiday"] = 1
    daily = daily.merge(national_holidays, on="date", how="left")
    daily["is_holiday"] = daily["is_holiday"].fillna(0).astype(int)

    # Step 4: Aggregate transactions to daily total
    daily_tx = transactions.groupby("date", as_index=False)["transactions"].sum()
    daily = daily.merge(daily_tx, on="date", how="left")

    return daily


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the merged DataFrame:
    - Sort by date
    - Remove duplicates
    - Fill missing values
    - Ensure continuous daily date range
    """
    df = df.drop_duplicates(subset="date", keep="first")
    df = df.sort_values("date").reset_index(drop=True)

    # Ensure continuous date range with no gaps
    full_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    df = df.set_index("date").reindex(full_range)
    df.index.name = "date"

    # Fill each column appropriately
    df["sales"] = df["sales"].interpolate(method="linear")
    df["oil_price"] = df["oil_price"].bfill().ffill()
    df["onpromotion"] = df["onpromotion"].fillna(0)
    df["is_holiday"] = df["is_holiday"].fillna(0).astype(int)
    df["transactions"] = df["transactions"].bfill().ffill()

    df = df.reset_index()
    return df


def run_pipeline() -> pd.DataFrame:
    """Full pipeline: load all files -> merge -> clean -> return."""
    print("  Loading all dataset files...")
    raw = ingest_data()
    print(f"  Raw merged shape: {raw.shape}")
    print("  Cleaning data...")
    clean = clean_data(raw)
    print(f"  Clean shape: {clean.shape}")
    print(f"  Date range: {clean['date'].min().date()} -> {clean['date'].max().date()}")
    print(f"  Missing values:\n{clean.isna().sum()}")
    return clean


if __name__ == "__main__":
    df = run_pipeline()
    print(df.head())
