"""
FastAPI backend for the AI Sales Forecasting Dashboard.

Endpoints:
    GET  /                         -> dashboard HTML
    GET  /health                   -> API health check
    GET  /api/summary              -> dataset summary statistics
    GET  /api/historical           -> historical daily sales (JSON)
    GET  /api/forecast?days=N      -> forecast via query param
    POST /api/forecast             -> forecast via JSON body  <-- NEW

Run from project root:
    uvicorn app.main:app --reload
"""

import os
import logging
from pydantic import BaseModel, Field

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.data_pipeline import run_pipeline
from app.forecast_model import load_model, forecast, generate_recommendation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_PATH = "models/forecast_model.pkl"
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = FastAPI(
    title="AI Sales Forecasting Dashboard",
    description="Forecasts Corporación Favorita store sales using Prophet.",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ── Load model + data once at startup ──────────────────────────────────────
_model = None
_model_loaded = False
_history_df = None

if os.path.exists(MODEL_PATH):
    try:
        _model = load_model(MODEL_PATH)
        _model_loaded = True
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")

try:
    _history_df = run_pipeline()
    logger.info(f"Historical data loaded: {len(_history_df)} rows")
except Exception as e:
    logger.error(f"Failed to load data: {e}")


# ── Pydantic schema for POST body ───────────────────────────────────────────
class ForecastRequest(BaseModel):
    days: int = Field(default=30, ge=1, le=365, description="Number of days to forecast")


# ── Helper ──────────────────────────────────────────────────────────────────
def _build_forecast_response(days: int):
    if not _model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not trained. Run 'python train.py' first.",
        )
    if _history_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded.")

    logger.info(f"Forecast requested: {days} days")
    fc_df = forecast(_model, _history_df, periods=days)
    recommendation = generate_recommendation(fc_df, _history_df)

    return {
        "days_requested": days,
        "dates": fc_df["ds"].dt.strftime("%Y-%m-%d").tolist(),
        "forecast": fc_df["yhat"].round(2).tolist(),
        "lower_bound": fc_df["yhat_lower"].round(2).tolist(),
        "upper_bound": fc_df["yhat_upper"].round(2).tolist(),
        "recommendation": recommendation,
    }


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
def dashboard():
    """Serve the interactive dashboard."""
    path = os.path.join(STATIC_DIR, "dashboard.html")
    with open(path, "r") as f:
        return f.read()


@app.get("/health", tags=["Health"])
def health():
    """API health check."""
    return {
        "status": "ok",
        "model_loaded": _model_loaded,
        "data_loaded": _history_df is not None,
        "rows": len(_history_df) if _history_df is not None else 0,
    }


@app.get("/api/summary", tags=["Data"])
def summary():
    """Return dataset summary statistics."""
    if _history_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded.")
    df = _history_df
    return JSONResponse({
        "total_days": len(df),
        "date_from": str(df["date"].min().date()),
        "date_to": str(df["date"].max().date()),
        "avg_daily_sales": round(float(df["sales"].mean()), 2),
        "max_daily_sales": round(float(df["sales"].max()), 2),
        "min_daily_sales": round(float(df["sales"].min()), 2),
        "total_holidays": int(df["is_holiday"].sum()),
        "avg_oil_price": round(float(df["oil_price"].mean()), 2),
    })


@app.get("/api/historical", tags=["Data"])
def historical():
    """Return last 180 days of historical sales data."""
    if _history_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded.")
    df = _history_df.tail(180)
    logger.info("Historical data requested.")
    return JSONResponse({
        "dates": df["date"].dt.strftime("%Y-%m-%d").tolist(),
        "sales": df["sales"].round(2).tolist(),
        "oil_price": df["oil_price"].round(2).tolist(),
        "is_holiday": df["is_holiday"].tolist(),
    })


@app.get("/api/forecast", tags=["Forecast"])
def forecast_get(
    days: int = Query(30, ge=1, le=365, description="Number of days to forecast"),
):
    """
    GET forecast — pass days as a query parameter.
    Example: /api/forecast?days=60
    """
    return JSONResponse(_build_forecast_response(days))


@app.post("/api/forecast", tags=["Forecast"])
def forecast_post(body: ForecastRequest):
    """
    POST forecast — pass days in a JSON request body.
    Example body: {"days": 60}
    This is useful when calling the API from another app or script.
    """
    return JSONResponse(_build_forecast_response(body.days))
