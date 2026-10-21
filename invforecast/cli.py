"""Typer command line: generate-sample, train, forecast, reorder, backtest."""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .backtest import backtest as run_backtest
from .backtest import summarize, to_markdown
from .data import load_sales, series_for, skus
from .holidays import load_calendar
from .models import MODELS, build_model, forecast_frame
from .reorder import recommend
from .sample import generate_sample

app = typer.Typer(
    help="Demand forecasting and reorder recommendations for small retailers.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")


def _version(value: bool) -> None:
    if value:
        console.print(f"invforecast {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", callback=_version, is_eager=True),
) -> None:
    """invforecast CLI."""


@app.command("generate-sample")
def generate_sample_cmd(
    output: Path = typer.Argument(Path("data/sample_sales.csv"), help="Where to write the CSV."),
    days: int = typer.Option(365, min=60),
    skus_count: int = typer.Option(4, "--skus", min=1, max=6),
    seed: int = typer.Option(42),
    start: str = typer.Option("2025-01-01", help="First day (YYYY-MM-DD)."),
) -> None:
    """Write a synthetic POS sales CSV with weekly seasonality, promotions and holidays."""
    df = generate_sample(start=date.fromisoformat(start), days=days, n_skus=skus_count, seed=seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    console.print(f"Wrote {len(df)} rows for {skus_count} sku(s) to [bold]{output}[/bold]")


@app.command()
def train(
    data: Path = typer.Argument(..., exists=True, readable=True, help="Cleaned or raw POS CSV."),
    model: str = typer.Option("ensemble", help=f"One of: {', '.join(MODELS)}"),
    out: Path = typer.Option(Path("models/model.joblib"), help="Where to save the fitted bundle."),
    holidays: Path | None = typer.Option(None, help="Custom holidays YAML."),
) -> None:
    """Fit one model per sku and save them as a single joblib bundle."""
    if model not in MODELS:
        raise typer.BadParameter(f"unknown model '{model}'; choose from {', '.join(MODELS)}")
    df = load_sales(data)
    calendar = load_calendar(holidays) if holidays else None
    bundle: dict = {
        "version": __version__,
        "model_name": model,
        "trained_at": pd.Timestamp.utcnow().isoformat(),
        "models": {},
        "history": {},
    }
    for sku in skus(df):
        history = series_for(df, sku)
        kwargs = {"calendar": calendar} if (calendar and model in {"gradient_boosting"}) else {}
        fitted = build_model(model, **kwargs).fit(history)
        bundle["models"][sku] = fitted
        bundle["history"][sku] = history
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out)
    console.print(f"Trained [bold]{model}[/bold] for {len(bundle['models'])} sku(s) -> {out}")


def _load_bundle(path: Path) -> dict:
    if not path.exists():
        raise typer.BadParameter(f"model bundle not found: {path} (run `invforecast train` first)")
    return joblib.load(path)


@app.command()
def forecast(
    model_path: Path = typer.Argument(Path("models/model.joblib")),
    horizon: int = typer.Option(14, min=1, max=365),
    sku: str | None = typer.Option(None, help="Only this sku (default: all)."),
    output: Path | None = typer.Option(None, help="Write the forecast CSV here."),
    as_json: bool = typer.Option(False, "--json", help="Print JSON instead of a table."),
) -> None:
    """Forecast the next HORIZON days for every trained sku."""
    bundle = _load_bundle(model_path)
    targets = [sku] if sku else sorted(bundle["models"])
    frames = []
    for name in targets:
        if name not in bundle["models"]:
            raise typer.BadParameter(
                f"sku '{name}' not in bundle; available: {', '.join(sorted(bundle['models']))}"
            )
        preds = bundle["models"][name].predict(horizon)
        frames.append(forecast_frame(bundle["history"][name], preds, sku=name))
    result = pd.concat(frames, ignore_index=True)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False)
    if as_json:
        console.print_json(
            json.dumps(
                result.assign(date=result["date"].dt.strftime("%Y-%m-%d")).to_dict("records")
            )
        )
        return
    table = Table(title=f"{horizon}-day forecast ({bundle['model_name']})")
    table.add_column("sku")
    table.add_column("total", justify="right")
    table.add_column("avg/day", justify="right")
    table.add_column("peak day")
    for name, group in result.groupby("sku"):
        peak = group.loc[group["forecast"].idxmax()]
        table.add_row(
            str(name),
            f"{group['forecast'].sum():.0f}",
            f"{group['forecast'].mean():.1f}",
            f"{peak['date']:%a %d %b} ({peak['forecast']:.0f})",
        )
    console.print(table)


@app.command()
def reorder(
    model_path: Path = typer.Argument(Path("models/model.joblib")),
    lead_time: int = typer.Option(7, min=1, help="Supplier lead time in days."),
    service_level: float = typer.Option(0.95, min=0.5, max=0.999),
    review_period: int = typer.Option(0, min=0, help="Days between order reviews."),
    stock: Path | None = typer.Option(None, help="CSV with sku,on_hand[,on_order]; defaults to 0."),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Safety stock, reorder point and order quantity per sku from the trained forecast."""
    bundle = _load_bundle(model_path)
    stock_levels: dict[str, tuple[float, float]] = {}
    if stock:
        stock_df = pd.read_csv(stock)
        for _, row in stock_df.iterrows():
            stock_levels[str(row["sku"])] = (
                float(row["on_hand"]),
                float(row.get("on_order", 0) or 0),
            )
    horizon = lead_time + review_period
    recs = []
    for name in sorted(bundle["models"]):
        history = bundle["history"][name]
        preds = bundle["models"][name].predict(horizon)
        on_hand, on_order = stock_levels.get(name, (0.0, 0.0))
        recs.append(
            recommend(
                name,
                history_units=history["units_sold"],
                forecast_units=preds,
                on_hand=on_hand,
                on_order=on_order,
                lead_time_days=lead_time,
                service_level=service_level,
                review_period_days=review_period,
            )
        )
    if as_json:
        console.print_json(json.dumps([r.as_dict() for r in recs]))
        return
    table = Table(title=f"Reorder plan (lead time {lead_time}d, service level {service_level:.0%})")
    for col in (
        "sku",
        "on hand",
        "avg/day",
        "safety stock",
        "reorder point",
        "order qty",
        "order?",
    ):
        table.add_column(col, justify="right" if col != "sku" else "left")
    for r in recs:
        table.add_row(
            r.sku,
            f"{r.on_hand:.0f}",
            f"{r.mean_daily_demand:.1f}",
            f"{r.safety_stock:.1f}",
            f"{r.reorder_point:.1f}",
            f"{r.recommended_order_qty:.0f}",
            "[bold red]yes[/bold red]" if r.should_order else "no",
        )
    console.print(table)


@app.command()
def backtest(
    data: Path = typer.Argument(..., exists=True, readable=True),
    horizon: int = typer.Option(14, min=1),
    splits: int = typer.Option(4, min=1),
    step: int | None = typer.Option(None, help="Days between fold cut-offs (default: horizon)."),
    models: str = typer.Option(",".join(MODELS), help="Comma-separated model names."),
    markdown: bool = typer.Option(False, help="Print the summary as a Markdown table."),
    output: Path | None = typer.Option(None, help="Write per-fold results CSV here."),
) -> None:
    """Rolling-origin backtest of every model on every sku."""
    names = [m.strip() for m in models.split(",") if m.strip()]
    unknown = [m for m in names if m not in MODELS]
    if unknown:
        raise typer.BadParameter(f"unknown model(s): {', '.join(unknown)}")
    df = load_sales(data)
    results = run_backtest(df, names, horizon=horizon, n_splits=splits, step=step)
    if results.empty:
        raise typer.BadParameter("no folds could be evaluated; reduce --splits or --horizon")
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output, index=False)
    summary = summarize(results)
    if markdown:
        typer.echo(to_markdown(summary))
        return
    table = Table(
        title=f"Backtest: {horizon}-day horizon, {splits} folds, {df['sku'].nunique()} sku(s)"
    )
    for col in ("model", "MAE", "RMSE", "MAPE %", "sMAPE %", "bias", "folds"):
        table.add_column(col, justify="right" if col != "model" else "left")
    for _, row in summary.iterrows():
        table.add_row(
            row["model"],
            f"{row['mae']:.2f}",
            f"{row['rmse']:.2f}",
            "n/a" if pd.isna(row["mape"]) else f"{row['mape']:.1f}",
            f"{row['smape']:.1f}",
            f"{row['bias']:+.2f}",
            str(int(row["folds"])),
        )
    console.print(table)


if __name__ == "__main__":  # pragma: no cover
    app()
