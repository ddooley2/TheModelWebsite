"""Evaluate CFBD pace and efficiency features as separate Stage 4 challengers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.cfb_model.data.warehouse import ensure_warehouse
from src.cfb_model.features.advanced import FEATURE_COLUMNS as ADVANCED_COLUMNS
from src.cfb_model.features.cfbd_game_features import (
    BASE_GAME_STAT_COLUMNS,
    EFFICIENCY_FEATURE_COLUMNS,
    PACE_FEATURE_COLUMNS,
    SPLIT_STAT_COLUMNS,
    build_cfbd_efficiency_features,
    build_cfbd_pace_features,
    cfbd_feature_data_dictionary,
    cfbd_stat_coverage,
    combine_cfbd_game_stat_sources,
)
from src.cfb_model.reporting.diagnostics import plot_mae_by_season_comparison
from src.cfb_model.train.score_model import (
    BACKTEST_SEASONS,
    DB_PATH,
    MODEL_COLUMNS,
    build_model_features,
    load_advanced_stats,
    load_historical_games,
    load_roster_priors,
    load_schedule_history,
    run_walk_forward_predictions,
    score_metrics,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "baselines"
PACE_OUTPUT_NAME = "cfbd_pace_stage4a"
EFFICIENCY_OUTPUT_NAME = "cfbd_efficiency_stage4b"
EXCLUDE_GARBAGE_TIME = False


def load_cfbd_game_stats(db_path: Path = DB_PATH) -> tuple[pd.DataFrame, dict]:
    """Load and reconcile historical overall and play-split CFBD game stats."""
    with duckdb.connect(str(db_path), read_only=True) as connection:
        game_stats = connection.execute(
            "SELECT game_id, season, week, team, opponent, "
            + ", ".join(BASE_GAME_STAT_COLUMNS)
            + " FROM stg_team_game_stats_raw"
        ).fetchdf()
        split_stats = connection.execute(
            "SELECT game_id, season, week, team, opponent, "
            + ", ".join(SPLIT_STAT_COLUMNS)
            + " FROM stg_pbp_team_game_raw"
        ).fetchdf()

    stats = combine_cfbd_game_stat_sources(game_stats, split_stats)
    split_count_inference = {}
    for split in ("passingPlays", "rushingPlays"):
        for side in ("offense", "defense"):
            ppa = pd.to_numeric(
                stats[f"{side}_{split}_ppa"], errors="coerce"
            ).to_numpy(dtype=float)
            total = pd.to_numeric(
                stats[f"{side}_{split}_totalPPA"], errors="coerce"
            ).to_numpy(dtype=float)
            valid = np.isfinite(ppa) & np.isfinite(total) & (np.abs(ppa) > 1e-12)
            ratios = total[valid] / ppa[valid]
            integer_counts = (
                np.isfinite(ratios)
                & (np.rint(ratios) > 0)
                & np.isclose(ratios, np.rint(ratios), atol=1e-8, rtol=0)
            )
            split_count_inference[f"{side}_{split}"] = {
                "nonzero_ppa_rows": int(valid.sum()),
                "reliable_integer_count_rows": int(integer_counts.sum()),
                "zero_or_missing_ppa_rows_count_unavailable": int((~valid).sum()),
            }

    source_quality = {
        "overall_source_rows": int(len(game_stats)),
        "split_source_rows": int(len(split_stats)),
        "reconciled_team_game_rows": int(len(stats)),
        "split_count_inference": split_count_inference,
    }
    return stats, source_quality


def _season_metrics(predictions: pd.DataFrame, candidate: str) -> pd.DataFrame:
    rows = []
    for season, fold in predictions.groupby("season", sort=True):
        actual = fold[["actual_home_score", "actual_away_score"]].to_numpy(dtype=float)
        predicted = fold[
            ["predicted_home_score", "predicted_away_score"]
        ].to_numpy(dtype=float)
        rows.append({
            "candidate": candidate,
            "season": int(season),
            "games": int(len(fold)),
            **score_metrics(actual, predicted),
        })
    return pd.DataFrame(rows)


def _paired_errors(
    baseline: pd.DataFrame, challenger: pd.DataFrame
) -> pd.DataFrame:
    key = ["game_id", "season", "week", "home_team", "away_team"]
    baseline_rows = baseline[
        key + ["score_absolute_error"]
    ].rename(columns={"score_absolute_error": "baseline_score_absolute_error"})
    challenger_rows = challenger[
        key + ["score_absolute_error"]
    ].rename(columns={"score_absolute_error": "challenger_score_absolute_error"})
    paired = baseline_rows.merge(
        challenger_rows,
        on=key,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not paired["_merge"].eq("both").all():
        raise ValueError("Baseline and challenger predictions do not cover the same games.")
    return paired.drop(columns="_merge")


def _write_comparison(
    *,
    output_dir: Path,
    baseline_predictions: pd.DataFrame,
    challenger_predictions: pd.DataFrame,
    baseline_label: str,
    challenger_label: str,
    feature_columns: list[str],
    feature_dictionary: pd.DataFrame,
    source_coverage: dict,
    source_quality: dict,
    baseline_feature_columns: list[str],
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_metrics = _season_metrics(baseline_predictions, baseline_label)
    challenger_metrics = _season_metrics(challenger_predictions, challenger_label)
    metrics = pd.concat([baseline_metrics, challenger_metrics], ignore_index=True)
    metrics.to_csv(output_dir / "seasonal_metrics.csv", index=False)
    baseline_predictions.to_csv(output_dir / "baseline_predictions.csv", index=False)
    challenger_predictions.to_csv(output_dir / "challenger_predictions.csv", index=False)
    _paired_errors(baseline_predictions, challenger_predictions).to_csv(
        output_dir / "paired_error_deltas.csv", index=False
    )
    plot_mae_by_season_comparison(
        metrics[["candidate", "season", "score_mae"]],
        output_dir / "seasonal_score_mae_comparison.png",
    )
    feature_dictionary.loc[
        feature_dictionary["feature"].isin(feature_columns)
    ].to_csv(output_dir / "feature_data_dictionary.csv", index=False)
    (output_dir / "source_coverage.json").write_text(
        json.dumps(source_coverage, indent=2) + "\n"
    )
    manifest = {
        "stage": "4A" if "pace" in output_dir.name else "4B",
        "production_model_changed": False,
        "baseline": baseline_label,
        "challenger": challenger_label,
        "baseline_feature_columns": baseline_feature_columns,
        "challenger_feature_columns": feature_columns,
        "validation_seasons": [int(season) for season in sorted(metrics["season"].unique())],
        "selection_policy": (
            "Independent challenger evaluation on the same season-forward folds; "
            "no parameter tuning or production selection."
        ),
        "cfbd_source_policy": {
            "overall_table": "stg_team_game_stats_raw",
            "split_table": "stg_pbp_team_game_raw",
            "excludeGarbageTime": EXCLUDE_GARBAGE_TIME,
            "missing_stats": (
                "Retain the canonical game; emit missing CFBD features and missingness "
                "and sample-size features. LightGBM handles NaN values natively."
            ),
            "source_quality": source_quality,
            "matched_game_coverage": source_coverage,
        },
        "feature_parameters": {
            "recency_half_life_days": 56,
            "prior_season_shrinkage_games": 3,
            "season_to_date_cutoff": (
                "Only completed observations before the target UTC calendar-date batch."
            ),
        },
        "baseline_mean_seasonal_score_mae": float(
            baseline_metrics["score_mae"].mean()
        ),
        "challenger_mean_seasonal_score_mae": float(
            challenger_metrics["score_mae"].mean()
        ),
        "baseline_overall_score_mae": float(
            baseline_predictions["score_absolute_error"].mean()
        ),
        "challenger_overall_score_mae": float(
            challenger_predictions["score_absolute_error"].mean()
        ),
        "artifacts": {
            "seasonal_metrics": "seasonal_metrics.csv",
            "seasonal_comparison_chart": "seasonal_score_mae_comparison.png",
            "baseline_predictions": "baseline_predictions.csv",
            "challenger_predictions": "challenger_predictions.csv",
            "paired_errors": "paired_error_deltas.csv",
            "feature_data_dictionary": "feature_data_dictionary.csv",
            "source_coverage": "source_coverage.json",
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def run_experiment(
    db_path: Path = DB_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict:
    """Run separate pace and efficiency comparisons against the production baseline."""
    ensure_warehouse(db_path)
    games = load_historical_games(db_path).reset_index(drop=True)
    baseline_features = build_model_features(
        games,
        load_roster_priors(db_path),
        load_advanced_stats(db_path),
        model_history=games,
        schedule_history=load_schedule_history(db_path),
    )
    stats, source_quality = load_cfbd_game_stats(db_path)
    source_coverage = cfbd_stat_coverage(games, stats)
    pace_features = build_cfbd_pace_features(games, stats)
    efficiency_features = build_cfbd_efficiency_features(games, stats)

    pace_columns = MODEL_COLUMNS + PACE_FEATURE_COLUMNS
    efficiency_columns = (
        [column for column in MODEL_COLUMNS if column not in ADVANCED_COLUMNS]
        + EFFICIENCY_FEATURE_COLUMNS
    )
    baseline_label = "Production baseline"
    baseline_predictions = run_walk_forward_predictions(games, baseline_features)
    pace_predictions = run_walk_forward_predictions(
        games,
        pd.concat([baseline_features, pace_features], axis=1),
        feature_columns=pace_columns,
    )
    efficiency_base = baseline_features.drop(columns=ADVANCED_COLUMNS)
    efficiency_predictions = run_walk_forward_predictions(
        games,
        pd.concat([efficiency_base, efficiency_features], axis=1),
        feature_columns=efficiency_columns,
    )
    dictionary = cfbd_feature_data_dictionary()
    pace_manifest = _write_comparison(
        output_dir=output_root / PACE_OUTPUT_NAME,
        baseline_predictions=baseline_predictions,
        challenger_predictions=pace_predictions,
        baseline_label=baseline_label,
        challenger_label="CFBD pace challenger",
        feature_columns=PACE_FEATURE_COLUMNS,
        feature_dictionary=dictionary,
        source_coverage=source_coverage,
        source_quality=source_quality,
        baseline_feature_columns=MODEL_COLUMNS,
    )
    efficiency_manifest = _write_comparison(
        output_dir=output_root / EFFICIENCY_OUTPUT_NAME,
        baseline_predictions=baseline_predictions,
        challenger_predictions=efficiency_predictions,
        baseline_label=baseline_label,
        challenger_label="CFBD efficiency challenger",
        feature_columns=EFFICIENCY_FEATURE_COLUMNS,
        feature_dictionary=dictionary,
        source_coverage=source_coverage,
        source_quality=source_quality,
        baseline_feature_columns=[
            column for column in MODEL_COLUMNS if column not in ADVANCED_COLUMNS
        ],
    )
    result = {
        "pace": pace_manifest,
        "efficiency": efficiency_manifest,
    }
    (output_root / "cfbd_stage4_manifest.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    result = run_experiment(args.db_path, args.output_root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
