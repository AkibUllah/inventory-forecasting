.PHONY: install test lint fmt sample train forecast reorder backtest api

PY ?= .venv/bin/python
CLI ?= .venv/bin/invforecast

install:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

fmt:
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

sample:
	$(CLI) generate-sample data/sample_sales.csv --days 365 --skus 4 --seed 42

train:
	$(CLI) train data/sample_sales.csv --model ensemble --out models/model.joblib

forecast:
	$(CLI) forecast models/model.joblib --horizon 14

reorder:
	$(CLI) reorder models/model.joblib --lead-time 7 --service-level 0.95

backtest:
	$(CLI) backtest data/sample_sales.csv --horizon 14 --splits 6 --markdown

api:
	$(PY) -m uvicorn invforecast.api:app --reload --port 8000
