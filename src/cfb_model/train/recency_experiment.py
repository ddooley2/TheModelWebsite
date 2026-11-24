"""Tune and report recency-weighted opponent-adjusted score ratings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.cfb_model.data.warehouse import ensure_warehouse
from src.cfb_model.features.opponent_adjusted import (
    FEATURE_COLUMNS as RATING_COLUMNS,
    RECENCY_HALF_LIFE_DAYS,
    RECENCY_RIDGE_ALPHAS,
    SEASON_PRIOR_STRENGTHS,
    build_recency_weighted_opponent_adjusted_features,
)
from src.cfb_model.reporting.diagnostics import plot_mae_by_season_comparison
from src.cfb_model.train.score_model import (
    BACKTEST_SEASONS,
    DB_PATH,
    build_model_features,
    load_advanced_stats,
    load_historical_games,
    load_roster_priors,
    load_schedule_history,
    run_walk_forward_predictions,
    score_metrics,
    _combine_model_history,
)

DEFAULT_OUTPUT_DIR = Path("results/baselines/recency_weighted_ratings_stage3")
BOOTSTRAP_REPLICATES = 5000


def _variant_name(half_life_days: int, season_prior_strength: float, ridge_alpha: float) -> str:
    return (
        f"half_life_{half_life_days}_prior_{season_prior_strength:g}"
        f"_ridge_{ridge_alpha:g}"
    )


def select_recency_parameters(
    grid_metrics: pd.DataFrame, tuning_seasons: list[int]
) -> dict[str, float | int]:
    """Select the lowest seasonal MAE among positive-prior variants on tuning years."""
    required = {
        "half_life_days", "season_prior_strength", "ridge_alpha", "season", "score_mae"
    }
    if not required <= set(grid_metrics.columns):
        raise ValueError(f"Tuning metrics need columns: {sorted(required)}")
    tuning = grid_metrics[
        grid_metrics["season"].isin(tuning_seasons)
        & (grid_metrics["season_prior_strength"] > 0)
    ]
    grouped = tuning.groupby(
        ["half_life_days", "season_prior_strength", "ridge_alpha"], as_index=False
    )["score_mae"].agg(["mean", "count"]).reset_index()
    if grouped.empty or (grouped["count"] != len(tuning_seasons)).any():
        raise ValueError("Each parameter set must have one score for every tuning season.")
    best = grouped.sort_values(
        ["mean", "half_life_days", "season_prior_strength", "ridge_alpha"],
        kind="stable",
    ).iloc[0]
    return {
        "half_life_days": int(best["half_life_days"]),
        "season_prior_strength": float(best["season_prior_strength"]),
        "ridge_alpha": float(best["ridge_alpha"]),
        "tuning_mean_score_mae": float(best["mean"]),
    }


def _season_metrics(predictions: pd.DataFrame, candidate: str) -> pd.DataFrame:
    rows = []
    for season, fold in predictions.groupby("season", sort=True):
        actual = fold[["actual_home_score", "actual_away_score"]].to_numpy(dtype=float)
        predicted = fold[["predicted_home_score", "predicted_away_score"]].to_numpy(dtype=float)
        rows.append({
            "candidate": candidate,
            "season": int(season),
            "games": len(fold),
            **score_metrics(actual, predicted),
        })
    return pd.DataFrame(rows)


def _week_block(week: int) -> str:
    if week <= 3:
        return "Weeks 0-3"
    if week <= 8:
        return "Weeks 4-8"
    return "Weeks 9+"


def _paired_final_season(
    baseline: pd.DataFrame, challenger: pd.DataFrame, final_season: int
) -> pd.DataFrame:
    key = ["game_id", "season", "week", "home_team", "away_team"]
    missing = set(key).difference(baseline.columns) | set(key).difference(challenger.columns)
    if missing:
        raise ValueError(f"Predictions are missing pairing columns: {sorted(missing)}")
    base = baseline.loc[baseline["season"] == final_season, key + [
        "game_date", "score_absolute_error"
    ]].rename(columns={"score_absolute_error": "baseline_score_absolute_error"})
    changed = challenger.loc[challenger["season"] == final_season, key + [
        "score_absolute_error"
    ]].rename(columns={"score_absolute_error": "challenger_score_absolute_error"})
    paired = base.merge(changed, on=key, how="outer", validate="one_to_one", indicator=True)
    if not paired["_merge"].eq("both").all():
        raise ValueError("Baseline and challenger final-season games do not match.")
    paired = paired.drop(columns="_merge")
    paired["score_absolute_error_delta"] = (
        paired["challenger_score_absolute_error"]
        - paired["baseline_score_absolute_error"]
    )
    paired["week"] = pd.to_numeric(paired["week"], errors="coerce").fillna(0).astype(int)
    paired["week_block"] = paired["week"].map(_week_block)
    return paired


def _week_cluster_bootstrap(
    paired: pd.DataFrame,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = 42,
) -> list[dict]:
    rows = []
    for label, subset in [
        ("Overall", paired),
        *[(name, paired[paired["week_block"] == name]) for name in (
            "Weeks 0-3", "Weeks 4-8", "Weeks 9+"
        )],
    ]:
        if subset.empty:
            rows.append({
                "block": label, "games": 0, "weeks": 0, "mean_delta": None,
                "ci_low": None, "ci_high": None,
            })
            continue
        by_week = subset.groupby("week", sort=True)["score_absolute_error_delta"].agg(
            ["sum", "count"]
        )
        sums = by_week["sum"].to_numpy(dtype=float)
        counts = by_week["count"].to_numpy(dtype=float)
        rng = np.random.default_rng(seed)
        sampled_weeks = rng.integers(0, len(by_week), size=(replicates, len(by_week)))
        estimates = sums[sampled_weeks].sum(axis=1) / counts[sampled_weeks].sum(axis=1)
        rows.append({
            "block": label,
            "games": int(len(subset)),
            "weeks": int(len(by_week)),
            "mean_delta": float(subset["score_absolute_error_delta"].mean()),
            "ci_low": float(np.quantile(estimates, 0.025)),
            "ci_high": float(np.quantile(estimates, 0.975)),
        })
    return rows


def _week_block_metrics(
    baseline: pd.DataFrame, challenger: pd.DataFrame, final_season: int
) -> pd.DataFrame:
    rows = []
    for label, predictions in (("Unweighted baseline", baseline), ("Recency-weighted", challenger)):
        final = predictions[predictions["season"] == final_season].copy()
        final["week"] = pd.to_numeric(final["week"], errors="coerce").fillna(0).astype(int)
        final["week_block"] = final["week"].map(_week_block)
        for block in ("Weeks 0-3", "Weeks 4-8", "Weeks 9+"):
            subset = final[final["week_block"] == block]
            if subset.empty:
                continue
            actual = subset[["actual_home_score", "actual_away_score"]].to_numpy(dtype=float)
            predicted = subset[["predicted_home_score", "predicted_away_score"]].to_numpy(dtype=float)
            rows.append({
                "candidate": label,
                "week_block": block,
                "games": len(subset),
                "score_mae": float(np.mean(np.abs(actual - predicted))),
            })
    return pd.DataFrame(rows)


def _replace_ratings(
    games: pd.DataFrame,
    base_features: pd.DataFrame,
    half_life_days: int,
    season_prior_strength: float,
    ridge_alpha: float,
) -> pd.DataFrame:
    timeline, target_positions = _combine_model_history(games, games)
    ratings = build_recency_weighted_opponent_adjusted_features(
        timeline,
        half_life_days=half_life_days,
        season_prior_strength=season_prior_strength,
        ridge_alpha=ridge_alpha,
    ).iloc[target_positions].reset_index(drop=True)
    if len(ratings) != len(base_features):
        raise ValueError("Weighted ratings are not aligned with the baseline feature rows.")
    features = base_features.copy()
    features.loc[:, RATING_COLUMNS] = ratings.to_numpy()
    return features


def run_experiment(
    db_path: Path = DB_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    reuse_grid: bool = False,
) -> dict:
    if len(BACKTEST_SEASONS) < 2:
        raise ValueError("At least two backtest seasons are required.")
    tuning_seasons = list(BACKTEST_SEASONS[:-1])
    final_season = int(BACKTEST_SEASONS[-1])

    ensure_warehouse(db_path)
    games = load_historical_games(db_path)
    baseline_features = build_model_features(
        games,
        load_roster_priors(db_path),
        load_advanced_stats(db_path),
        model_history=games,
        schedule_history=load_schedule_history(db_path),
    )
    baseline_prediction_path = output_dir / "baseline_predictions.csv"
    if reuse_grid and baseline_prediction_path.is_file():
        baseline_predictions = pd.read_csv(baseline_prediction_path)
    else:
        baseline_predictions = run_walk_forward_predictions(games, baseline_features)
    baseline_metrics = _season_metrics(baseline_predictions, "Unweighted baseline")

    grid_path = output_dir / "tuning_grid_metrics.csv"
    if reuse_grid:
        if not grid_path.is_file():
            raise FileNotFoundError(f"Cannot reuse missing tuning grid: {grid_path}")
        grid_metrics = pd.read_csv(grid_path)
    else:
        grid_rows: list[dict] = []
        for half_life in RECENCY_HALF_LIFE_DAYS:
            for prior_strength in SEASON_PRIOR_STRENGTHS:
                for ridge_alpha in RECENCY_RIDGE_ALPHAS:
                    features = _replace_ratings(
                        games, baseline_features, half_life, prior_strength, ridge_alpha
                    )
                    predictions = run_walk_forward_predictions(
                        games, features, val_seasons=tuning_seasons
                    )
                    metrics = _season_metrics(
                        predictions, _variant_name(half_life, prior_strength, ridge_alpha)
                    )
                    metrics["half_life_days"] = half_life
                    metrics["season_prior_strength"] = prior_strength
                    metrics["ridge_alpha"] = ridge_alpha
                    grid_rows.extend(metrics.to_dict("records"))
        grid_metrics = pd.DataFrame(grid_rows)
    selected = select_recency_parameters(grid_metrics, tuning_seasons)
    challenger_features = _replace_ratings(
        games,
        baseline_features,
        int(selected["half_life_days"]),
        float(selected["season_prior_strength"]),
        float(selected["ridge_alpha"]),
    )
    challenger_predictions = run_walk_forward_predictions(games, challenger_features)
    challenger_label = (
        f"Recency h={selected['half_life_days']}d, "
        f"prior={selected['season_prior_strength']:g}, "
        f"ridge={selected['ridge_alpha']:g}"
    )
    challenger_metrics = _season_metrics(challenger_predictions, challenger_label)

    output_dir.mkdir(parents=True, exist_ok=True)
    grid_metrics.to_csv(grid_path, index=False)
    baseline_predictions.to_csv(output_dir / "baseline_predictions.csv", index=False)
    challenger_predictions.to_csv(output_dir / "challenger_predictions.csv", index=False)
    metrics = pd.concat([baseline_metrics, challenger_metrics], ignore_index=True)
    metrics.to_csv(output_dir / "seasonal_metrics.csv", index=False)
    plot_mae_by_season_comparison(
        metrics[["candidate", "season", "score_mae"]],
        output_dir / "seasonal_score_mae_comparison.png",
    )

    paired = _paired_final_season(baseline_predictions, challenger_predictions, final_season)
    paired.to_csv(output_dir / f"paired_error_deltas_{final_season}.csv", index=False)
    bootstrap = _week_cluster_bootstrap(paired)
    (output_dir / f"paired_week_bootstrap_ci_{final_season}.json").write_text(
        json.dumps(bootstrap, indent=2) + "\n"
    )
    week_metrics = _week_block_metrics(
        baseline_predictions, challenger_predictions, final_season
    )
    week_metrics.to_csv(output_dir / f"week_block_metrics_{final_season}.csv", index=False)

    manifest = {
        "stage": 3,
        "production_model_changed": False,
        "rating_observation_policy": (
            "FBS-vs-FBS completed score records only; features are emitted before each UTC "
            "calendar-date batch, so same-date and future results are excluded."
        ),
        "baseline": "existing unweighted opponent-adjusted score ratings",
        "selection_metric": (
            "lowest mean seasonal score MAE across tuning seasons among positive-prior "
            "variants; zero prior is retained as a no-prior ablation"
        ),
        "tuning_seasons": tuning_seasons,
        "untouched_final_season": final_season,
        "tuning_grid_reused": reuse_grid,
        "parameter_grid": {
            "half_life_days": list(RECENCY_HALF_LIFE_DAYS),
            "season_prior_strength": list(SEASON_PRIOR_STRENGTHS),
            "ridge_alpha": list(RECENCY_RIDGE_ALPHAS),
        },
        "selected_parameters": selected,
        "selected_candidate": challenger_label,
        "baseline_mean_seasonal_score_mae": float(baseline_metrics["score_mae"].mean()),
        "challenger_mean_seasonal_score_mae": float(challenger_metrics["score_mae"].mean()),
        "baseline_overall_score_mae": float(
            baseline_predictions["score_absolute_error"].mean()
        ),
        "challenger_overall_score_mae": float(
            challenger_predictions["score_absolute_error"].mean()
        ),
        "final_season_baseline_score_mae": float(
            baseline_metrics.loc[baseline_metrics["season"] == final_season, "score_mae"].iloc[0]
        ),
        "final_season_challenger_score_mae": float(
            challenger_metrics.loc[challenger_metrics["season"] == final_season, "score_mae"].iloc[0]
        ),
        "artifacts": {
            "tuning_grid_metrics": "tuning_grid_metrics.csv",
            "seasonal_metrics": "seasonal_metrics.csv",
            "seasonal_comparison_chart": "seasonal_score_mae_comparison.png",
            "paired_final_season_errors": f"paired_error_deltas_{final_season}.csv",
            "week_block_metrics": f"week_block_metrics_{final_season}.csv",
            "week_cluster_bootstrap_ci": f"paired_week_bootstrap_ci_{final_season}.json",
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reuse-grid", action="store_true")
    args = parser.parse_args()
    print(json.dumps(
        run_experiment(args.db_path, args.output_dir, reuse_grid=args.reuse_grid),
        indent=2,
    ))


if __name__ == "__main__":
    main()
