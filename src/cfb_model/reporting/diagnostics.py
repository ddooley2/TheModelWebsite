"""Generate data-coverage, sparsity, distribution, and seasonal-performance reports."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.cfb_model.train.score_model import MODEL_COLUMNS, build_model_features, load_historical_games, load_roster_priors
from src.cfb_model.train.experiments import choose_winner, run_candidate_backtest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT_ROOT / "public" / "assets" / "diagnostics"


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _source_coverage(db_path: Path) -> pd.DataFrame:
    with duckdb.connect(str(db_path), read_only=True) as connection:
        tables = [
            "stg_games_raw",
            "stg_team_game_stats_raw",
            "stg_roster_priors_raw",
            "stg_lines_raw",
            "stg_weather_raw",
        ]
        rows = []
        for table in tables:
            columns = connection.execute(f"DESCRIBE {table}").fetchdf()["column_name"].tolist()
            season_column = "season" if "season" in columns else "year" if "year" in columns else None
            if season_column is None:
                continue
            data = connection.execute(
                f"SELECT {season_column} AS season, COUNT(*) AS rows FROM {table} "
                f"WHERE {season_column} IS NOT NULL GROUP BY 1 ORDER BY 1"
            ).fetchdf()
            data["source"] = table.removeprefix("stg_").removesuffix("_raw")
            rows.append(data)
    return pd.concat(rows, ignore_index=True)


def _plot_source_coverage(coverage: pd.DataFrame, output: Path) -> None:
    pivot = coverage.pivot_table(index="season", columns="source", values="rows", fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 5))
    pivot.plot(kind="bar", ax=ax, width=0.85)
    ax.set(title="Raw input coverage by season", xlabel="Season", ylabel="Rows")
    ax.legend(title="Source", fontsize=8)
    _save(fig, output / "source_coverage.png")


def _plot_missingness(features: pd.DataFrame, games: pd.DataFrame, output: Path) -> pd.DataFrame:
    report = features.copy()
    report["season"] = games["season"].to_numpy()
    missing = report.groupby("season")[MODEL_COLUMNS].apply(lambda frame: frame.isna().mean() * 100)
    long = missing.reset_index().melt(id_vars="season", var_name="feature", value_name="missing_pct")
    fig, ax = plt.subplots(figsize=(16, 7))
    image = ax.imshow(missing.T.to_numpy(), aspect="auto", vmin=0, vmax=100, cmap="magma")
    ax.set(
        title="Model-feature missingness before LightGBM handling",
        xlabel="Season",
        ylabel="Feature",
        xticks=np.arange(len(missing.index)),
        xticklabels=missing.index.astype(str),
        yticks=np.arange(len(MODEL_COLUMNS)),
        yticklabels=MODEL_COLUMNS,
    )
    fig.colorbar(image, ax=ax, label="Missing values (%)")
    _save(fig, output / "feature_missingness.png")
    return long


def _plot_distributions(features: pd.DataFrame, output: Path) -> pd.DataFrame:
    summary = features[MODEL_COLUMNS].describe(percentiles=[0.01, 0.5, 0.99]).T.reset_index(names="feature")
    columns = MODEL_COLUMNS
    fig, axes = plt.subplots(7, 5, figsize=(16, 18))
    for axis, column in zip(axes.flat, columns):
        values = features[column].dropna()
        axis.hist(values, bins=30, color="#33b7b2", edgecolor="white")
        axis.set_title(column, fontsize=8)
        axis.tick_params(labelsize=6)
    for axis in axes.flat[len(columns):]:
        axis.set_visible(False)
    fig.suptitle("Input distributions for every model feature", y=1.01)
    _save(fig, output / "feature_distributions.png")
    return summary


def _plot_performance(metrics: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for candidate, data in metrics.groupby("candidate"):
        ax.plot(data["season"], data["score_mae"], marker="o", label=candidate.replace("_", " "))
    ax.set(title="Walk-forward score MAE by season", xlabel="Validation season", ylabel="Score MAE")
    ax.legend()
    _save(fig, output / "seasonal_score_mae.png")


def generate_diagnostics(db_path: Path, output_dir: Path = DEFAULT_OUTPUT) -> dict:
    """Create public report artifacts and return a compact machine-readable summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    games = load_historical_games(db_path)
    features = build_model_features(games, load_roster_priors(db_path))
    coverage = _source_coverage(db_path)
    missingness = _plot_missingness(features, games, output_dir)
    distributions = _plot_distributions(features, output_dir)
    _plot_source_coverage(coverage, output_dir)
    metrics, predictions = run_candidate_backtest(games, features)
    _plot_performance(metrics, output_dir)

    coverage.to_csv(output_dir / "source_coverage.csv", index=False)
    missingness.to_csv(output_dir / "feature_missingness.csv", index=False)
    distributions.to_csv(output_dir / "feature_distribution_summary.csv", index=False)
    metrics.to_csv(output_dir / "candidate_metrics.csv", index=False)
    predictions.to_csv(output_dir / "candidate_oof_predictions.csv", index=False)
    selection = choose_winner(metrics)
    summary = {
        "games": int(len(games)),
        "features": MODEL_COLUMNS,
        "winner": selection,
        "mean_score_mae": {
            candidate: float(group["score_mae"].mean())
            for candidate, group in metrics.groupby("candidate")
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
