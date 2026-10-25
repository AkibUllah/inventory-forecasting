# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- CodeQL analysis for Python and GitHub Actions on pushes and pull requests to `main` and on a weekly schedule.
- Dependabot version updates for pip, Docker and GitHub Actions, with minor and patch updates grouped per ecosystem.
- README screenshots: a backtest/reorder run and the API reference.
- Roadmap items tracked as GitHub issues under the v0.2.0 milestone.

### Changed

- CI workflow runs with a read-only `GITHUB_TOKEN`.
- The sample tea sku is now `TEA-BLACK-400G` (generic name); the generated data is otherwise identical.

## [0.1.0] - 2026-10-25

First public release of Inventory Forecasting.

### Added

- `load_sales()` / `clean_sales()`: parses dates, trims sku codes, sums duplicate rows, clips returns to zero, fills missing days per sku with zero demand, forward-fills price
- Calendar features: day of week, Bangladeshi weekend (Fri/Sat), month start (salary days), public holidays, Eid days and a configurable pre-Eid run-up window, all from an editable YAML (`invforecast/data/bd_holidays.yaml`, fixed holidays recur, moveable ones listed per year)
- Lag features (1/7/14/28), trailing rolling mean/std/max shifted so they never see the target, promotion flag
- Models behind one interface: naive, seasonal naive (weekly), moving average, scikit-learn `GradientBoostingRegressor` with recursive multi-step forecasting, and a weighted ensemble
- Rolling-origin backtesting with contiguous, non-overlapping folds; MAE, RMSE, MAPE (zero-safe), sMAPE and bias per fold, model and sku; Markdown summary output
- Replenishment maths: service-level z-score (`statistics.NormalDist`), safety stock, reorder point, order quantity over lead time + review period, days of cover
- Synthetic POS generator with trend, weekly seasonality, holiday dips, pre-Eid lift, promotions and Poisson noise; the bundled `data/sample_sales.csv` is reproducible from it (`--seed 42`)
- CLI: `generate-sample`, `train`, `forecast`, `reorder`, `backtest`; FastAPI service: `POST /v1/forecast`, `POST /v1/reorder`, `GET /v1/skus`, `GET /health`
- Pure pandas/scikit-learn; no database, no external services

[Unreleased]: https://github.com/AkibUllah/inventory-forecasting/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/AkibUllah/inventory-forecasting/releases/tag/v0.1.0
