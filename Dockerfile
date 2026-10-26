# Trains the ensemble on the bundled sample data at build time and serves the API.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY invforecast ./invforecast
RUN pip install ".[api]"

COPY data ./data
RUN invforecast train data/sample_sales.csv --model ensemble --out models/model.joblib

RUN addgroup --system app && adduser --system --ingroup app app && chown -R app:app /app
USER app

ENV INVFORECAST_MODEL_PATH=/app/models/model.joblib
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status == 200 else 1)"

CMD ["uvicorn", "invforecast.api:app", "--host", "0.0.0.0", "--port", "8000"]
