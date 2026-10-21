import math

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from invforecast.cli import app
from invforecast.reorder import recommend, reorder_point, safety_stock, z_score

runner = CliRunner()


# --- reorder maths -------------------------------------------------------------
def test_z_scores_match_standard_table():
    assert z_score(0.50) == pytest.approx(0.0)
    assert z_score(0.90) == pytest.approx(1.2816, abs=1e-3)
    assert z_score(0.95) == pytest.approx(1.6449, abs=1e-3)
    assert z_score(0.99) == pytest.approx(2.3263, abs=1e-3)
    for bad in (0, 1, 1.5):
        with pytest.raises(ValueError):
            z_score(bad)


def test_safety_stock_and_reorder_point_formulas():
    ss = safety_stock(daily_std=4.0, lead_time_days=9, service_level=0.95)
    assert ss == pytest.approx(1.6449 * 4.0 * 3, abs=1e-2)
    assert reorder_point(mean_daily_demand=10, lead_time_days=9, safety=ss) == pytest.approx(
        90 + ss
    )
    with pytest.raises(ValueError):
        safety_stock(-1, 7, 0.95)
    with pytest.raises(ValueError):
        reorder_point(10, 0, 1)


def test_recommend_orders_when_position_below_reorder_point():
    history = np.array([10.0] * 50 + [12, 8, 10, 12, 8, 10])  # mean 10, small std
    forecast = np.full(14, 10.0)
    rec = recommend(
        "SKU",
        history_units=history,
        forecast_units=forecast,
        on_hand=30,
        lead_time_days=7,
        service_level=0.95,
    )
    assert rec.mean_daily_demand == pytest.approx(10.0, abs=0.05)
    assert rec.reorder_point > 70
    assert rec.should_order is True
    # order = forecast over 7 days (70) + safety stock - position (30)
    assert rec.recommended_order_qty == math.ceil(70 + rec.safety_stock - 30)
    assert rec.days_of_cover == 3.0


def test_recommend_no_order_when_well_stocked_and_counts_on_order():
    history = np.full(60, 5.0)
    rec = recommend(
        "SKU",
        history_units=history,
        forecast_units=np.full(7, 5.0),
        on_hand=20,
        on_order=40,
        lead_time_days=7,
        review_period_days=7,
    )
    assert rec.daily_std == 0 and rec.safety_stock == 0
    assert rec.reorder_point == 35
    assert rec.should_order is False  # position 60 > 35
    assert rec.forecast_over_lead_time == 70  # 14 days covered, horizon padded with the mean
    assert rec.recommended_order_qty == 10  # 70 - 60


def test_recommend_validation():
    with pytest.raises(ValueError):
        recommend("S", history_units=[], forecast_units=[1], on_hand=0, lead_time_days=7)
    with pytest.raises(ValueError):
        recommend("S", history_units=[1, 2], forecast_units=[1], on_hand=0, lead_time_days=0)


# --- CLI ----------------------------------------------------------------------
def test_cli_end_to_end(tmp_path):
    csv = tmp_path / "sales.csv"
    model = tmp_path / "model.joblib"

    result = runner.invoke(
        app, ["generate-sample", str(csv), "--days", "180", "--skus", "2", "--seed", "3"]
    )
    assert result.exit_code == 0, result.output
    assert "Wrote 360 rows" in result.output

    result = runner.invoke(
        app, ["train", str(csv), "--model", "seasonal_naive", "--out", str(model)]
    )
    assert result.exit_code == 0, result.output
    assert model.exists()

    result = runner.invoke(app, ["forecast", str(model), "--horizon", "7"])
    assert result.exit_code == 0, result.output
    assert "7-day forecast" in result.output and "RICE-MINI-5KG" in result.output

    out_csv = tmp_path / "fc.csv"
    result = runner.invoke(
        app,
        [
            "forecast",
            str(model),
            "--horizon",
            "5",
            "--sku",
            "OIL-SOY-1L",
            "--output",
            str(out_csv),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(pd.read_csv(out_csv)) == 5

    stock = tmp_path / "stock.csv"
    stock.write_text("sku,on_hand,on_order\nRICE-MINI-5KG,5,0\nOIL-SOY-1L,500,0\n")
    result = runner.invoke(app, ["reorder", str(model), "--lead-time", "5", "--stock", str(stock)])
    assert result.exit_code == 0, result.output
    assert "Reorder plan" in result.output and "yes" in result.output and "no" in result.output

    result = runner.invoke(
        app,
        [
            "backtest",
            str(csv),
            "--horizon",
            "7",
            "--splits",
            "2",
            "--models",
            "naive,moving_average",
            "--markdown",
        ],
    )
    assert result.exit_code == 0, result.output
    assert result.output.startswith("| Model | MAE |")


def test_cli_errors(tmp_path):
    assert runner.invoke(app, ["forecast", str(tmp_path / "missing.joblib")]).exit_code != 0
    csv = tmp_path / "s.csv"
    runner.invoke(app, ["generate-sample", str(csv), "--days", "90", "--skus", "1"])
    result = runner.invoke(
        app, ["train", str(csv), "--model", "arima", "--out", str(tmp_path / "m.joblib")]
    )
    assert result.exit_code != 0 and "unknown model" in result.output
    result = runner.invoke(app, ["backtest", str(csv), "--models", "naive,nope"])
    assert result.exit_code != 0


# --- API ----------------------------------------------------------------------
def test_api_forecast_and_reorder(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    csv = tmp_path / "sales.csv"
    model = tmp_path / "model.joblib"
    runner.invoke(app, ["generate-sample", str(csv), "--days", "120", "--skus", "1", "--seed", "1"])
    runner.invoke(app, ["train", str(csv), "--model", "moving_average", "--out", str(model)])

    from invforecast import api

    monkeypatch.setattr(api, "MODEL_PATH", model)
    api.get_bundle.cache_clear()
    client = TestClient(api.app)

    assert client.get("/health").json()["model_loaded"] is True
    assert client.get("/v1/skus").json()["skus"] == ["RICE-MINI-5KG"]

    response = client.post("/v1/forecast", json={"horizon": 5})
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1 and len(body[0]["points"]) == 5 and body[0]["model"] == "moving_average"
    assert client.post("/v1/forecast", json={"sku": "NOPE"}).status_code == 404
    assert client.post("/v1/forecast", json={"horizon": 0}).status_code == 422

    rec = client.post(
        "/v1/reorder", json={"sku": "RICE-MINI-5KG", "on_hand": 10, "lead_time_days": 7}
    ).json()
    assert rec["should_order"] is True and rec["recommended_order_qty"] > 0

    monkeypatch.setattr(api, "MODEL_PATH", tmp_path / "none.joblib")
    api.get_bundle.cache_clear()
    assert client.get("/v1/skus").status_code == 503
