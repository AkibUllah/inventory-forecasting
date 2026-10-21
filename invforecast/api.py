"""Optional FastAPI wrapper around a trained bundle.

    INVFORECAST_MODEL_PATH=models/model.joblib uvicorn invforecast.api:app

``POST /v1/forecast`` returns the next ``horizon`` days for one sku (or all), and
``POST /v1/reorder`` turns that into a replenishment recommendation.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .models import forecast_frame
from .reorder import recommend

MODEL_PATH = Path(os.environ.get("INVFORECAST_MODEL_PATH", "models/model.joblib"))
DEFAULT_HORIZON = int(os.environ.get("INVFORECAST_DEFAULT_HORIZON", "14"))

app = FastAPI(
    title="invforecast API",
    version=__version__,
    description="Demand forecasts and reorder recommendations from a trained bundle.",
)


@lru_cache(maxsize=1)
def get_bundle() -> dict:
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail=f"model bundle not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


class ForecastRequest(BaseModel):
    sku: str | None = Field(default=None, description="Omit for every trained sku.")
    horizon: int = Field(default=DEFAULT_HORIZON, ge=1, le=365)


class ForecastPoint(BaseModel):
    date: str
    forecast: float


class SkuForecast(BaseModel):
    sku: str
    model: str
    horizon: int
    total: float
    points: list[ForecastPoint]


class ReorderRequest(BaseModel):
    sku: str
    on_hand: float = Field(ge=0)
    on_order: float = Field(default=0, ge=0)
    lead_time_days: int = Field(default=7, ge=1, le=120)
    service_level: float = Field(default=0.95, gt=0.5, lt=1)
    review_period_days: int = Field(default=0, ge=0, le=60)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__, "model_loaded": MODEL_PATH.exists()}


@app.get("/v1/skus")
def list_skus() -> dict:
    bundle = get_bundle()
    return {"model": bundle["model_name"], "skus": sorted(bundle["models"])}


def _forecast(bundle: dict, sku: str, horizon: int) -> SkuForecast:
    if sku not in bundle["models"]:
        raise HTTPException(status_code=404, detail=f"sku '{sku}' not in trained bundle")
    preds = bundle["models"][sku].predict(horizon)
    frame = forecast_frame(bundle["history"][sku], preds, sku=sku)
    return SkuForecast(
        sku=sku,
        model=bundle["model_name"],
        horizon=horizon,
        total=round(float(frame["forecast"].sum()), 2),
        points=[
            ForecastPoint(date=d.strftime("%Y-%m-%d"), forecast=float(f))
            for d, f in zip(frame["date"], frame["forecast"], strict=False)
        ],
    )


@app.post("/v1/forecast", response_model=list[SkuForecast])
def forecast(request: ForecastRequest) -> list[SkuForecast]:
    bundle = get_bundle()
    targets = [request.sku] if request.sku else sorted(bundle["models"])
    return [_forecast(bundle, sku, request.horizon) for sku in targets]


@app.post("/v1/reorder")
def reorder(request: ReorderRequest) -> dict:
    bundle = get_bundle()
    if request.sku not in bundle["models"]:
        raise HTTPException(status_code=404, detail=f"sku '{request.sku}' not in trained bundle")
    horizon = request.lead_time_days + request.review_period_days
    preds = bundle["models"][request.sku].predict(horizon)
    rec = recommend(
        request.sku,
        history_units=bundle["history"][request.sku]["units_sold"],
        forecast_units=preds,
        on_hand=request.on_hand,
        on_order=request.on_order,
        lead_time_days=request.lead_time_days,
        service_level=request.service_level,
        review_period_days=request.review_period_days,
    )
    return rec.as_dict()
