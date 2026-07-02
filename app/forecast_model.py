"""
Sales forecasting model using Facebook Prophet.

Prophet handles trend + seasonality automatically.
We also pass in extra regressors (oil price, is_holiday, onpromotion)
using the additional data from the full dataset, which improves accuracy
compared to using only sales history.
"""

import pandas as pd
import numpy as np
import joblib
import os
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

MODEL_PATH = "models/forecast_model.pkl"
REGRESSORS = ["oil_price", "is_holiday", "onpromotion"]


def train_model(df: pd.DataFrame) -> Prophet:
    """
    Train a Prophet model with extra regressors.
    Prophet requires columns named 'ds' (date) and 'y' (target).
    """
    prophet_df = df.rename(columns={"date": "ds", "sales": "y"})

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
    )

    # Add extra regressors (from merged dataset files)
    for reg in REGRESSORS:
        if reg in prophet_df.columns:
            model.add_regressor(reg)

    model.fit(prophet_df)
    return model


def evaluate_model(df: pd.DataFrame):
    """
    Holdout evaluation: train on everything except last 30 days,
    predict those 30 days, compare to actuals.
    Returns MAE and MAPE (Mean Absolute Percentage Error).
    """
    train_df = df.iloc[:-30].copy()
    test_df = df.iloc[-30:].copy()

    eval_model = train_model(train_df)

    future = test_df.rename(columns={"date": "ds", "sales": "y"})[
        ["ds"] + REGRESSORS
    ]
    pred = eval_model.predict(future)

    y_true = test_df["sales"].values
    y_pred = pred["yhat"].values

    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return mae, mape


def forecast(model: Prophet, df: pd.DataFrame, periods: int = 30) -> pd.DataFrame:
    """
    Forecast future sales for `periods` days beyond the training data.
    Uses last known regressor values carried forward for future dates.
    """
    last_row = df.iloc[-1]

    future_dates = pd.date_range(
        start=df["date"].max() + pd.Timedelta(days=1),
        periods=periods,
        freq="D",
    )
    future = pd.DataFrame({"ds": future_dates})

    # Carry forward the last known regressor values into the future window
    for reg in REGRESSORS:
        future[reg] = last_row[reg] if reg in df.columns else 0

    result = model.predict(future)
    return result[["ds", "yhat", "yhat_lower", "yhat_upper"]]


def generate_recommendation(forecast_df: pd.DataFrame, history_df: pd.DataFrame) -> str:
    """
    Business recommendation based on forecasted vs recent average sales.
    """
    recent_avg = history_df["sales"].tail(30).mean()
    future_avg = forecast_df["yhat"].mean()
    pct = ((future_avg - recent_avg) / recent_avg) * 100

    if pct > 5:
        return (
            f"Sales are projected to INCREASE by {pct:.1f}% over the next period. "
            "Recommendation: Increase inventory levels and schedule additional staff "
            "to meet rising demand. Consider ramping up promotions on high-margin products."
        )
    elif pct < -5:
        return (
            f"Sales are projected to DECREASE by {abs(pct):.1f}% over the next period. "
            "Recommendation: Launch targeted promotional campaigns and discount strategies "
            "to stimulate demand. Review and optimize operational costs."
        )
    else:
        return (
            f"Sales are projected to remain STABLE ({pct:+.1f}% change). "
            "Recommendation: Maintain current inventory and staffing. "
            "Focus on customer retention and loyalty programs."
        )


def save_model(model: Prophet, path: str = MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str = MODEL_PATH) -> Prophet:
    return joblib.load(path)
