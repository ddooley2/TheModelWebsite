"""Evaluate stabilized role and neutral-site features as a Stage 6 challenger."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.cfb_model.data.warehouse import ensure_warehouse
from src.cfb_model.features.game_role_performance import (
    FEATURE_COLUMNS as ROLE_FEATURE_COLUMNS,
    ROLE_PRIOR_GAMES,
    build_game_role_features,
    stabilized_role_data_dictionary,
)
from src.cfb_model.reporting.diagnostics import plot_mae_by_season_comparison
from src.cfb_model.train.preseason_score_prior_experiment import (
    _safe_metrics,
    _schedule_after_fcs_flags,
)
from src.cfb_model.train.recency_experiment import (
    _paired_final_season,
    _week_cluster_bootstrap,
)
from src.cfb_model.train.score_model import (
    BACKTEST_SEASONS,
    DB_PATH,
    FIRST_TRAIN_SEASON,
    MODEL_COLUMNS,
    _make_model,
    build_model_features,
    load_advanced_stats,
    load_historical_games,
    load_roster_priors,
    load_schedule_history,
    run_walk_forward_predictions,
    score_metrics,
    validate_market_free_feature_columns,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "baselines" / "role_site_stabilization_stage6"
)
CANDIDATE_ID = "role_stabilized_relevant_and_overall_stage6"
BASELINE_ID = "production_unweighted_baseline"
BASELINE_LABEL = "Production baseline"
CANDIDATE_LABEL = "Stabilized role/site challenger"
FINAL_SEASON = max(BACKTEST_SEASONS)
STABILIZED_FEATURE_COLUMNS = ROLE_FEATURE_COLUMNS["stabilized"]


def _week_block(week: object) -> str:
    value = pd.to_numeric(week, errors="coerce")
    if pd.isna(value) or value <= 3:
        return "Weeks 0-3"
    if value <= 6:
        return "Weeks 4-6"
    return "Weeks 7+"


def _schema_hash(columns: list[str]) -> str:
    return hashlib.sha256("\n".join(columns).encode("utf-8")).hexdigest()


def _source_commit_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _json_records(frame: pd.DataFrame) -> list[dict]:
    return frame.astype(object).where(pd.notna(frame), None).to_dict("records")


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


def _build_subset_tags(
    games: pd.DataFrame,
    role_features: pd.DataFrame,
    schedule_history: pd.DataFrame,
) -> pd.DataFrame:
    tags = pd.DataFrame({"game_id": games["game_id"].to_numpy()})
    tags["week_block"] = pd.to_numeric(games["week"], errors="coerce").fillna(0).map(
        _week_block
    ).to_numpy()
    tags["neutral_site"] = games.get(
        "neutral_site", pd.Series(False, index=games.index)
    ).fillna(False).astype(bool).to_numpy()
    tags["after_fcs"] = _schedule_after_fcs_flags(
        games, schedule_history
    ).to_numpy()
    tags["role_features_missing"] = role_features[
        STABILIZED_FEATURE_COLUMNS
    ].isna().any(axis=1).to_numpy()
    return tags


def _subset_metrics(
    predictions: dict[str, pd.DataFrame], tags: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    definitions = {
        "Weeks 0-3": "week_block",
        "Weeks 4-6": "week_block",
        "Weeks 7+": "week_block",
        "Neutral site": "neutral_site",
        "After FCS opponent": "after_fcs",
        "Stabilized role feature missing": "role_features_missing",
    }
    tag_frame = tags.reset_index(drop=True)
    for candidate, frame in predictions.items():
        aligned = frame.merge(
            tag_frame, on="game_id", how="left", validate="one_to_one"
        )
        for season in ["all", *sorted(aligned["season"].unique())]:
            season_rows = (
                aligned if season == "all"
                else aligned[aligned["season"] == season]
            )
            for label, column in definitions.items():
                mask = (
                    season_rows[column].eq(label)
                    if column == "week_block"
                    else season_rows[column].fillna(False)
                )
                subset = season_rows[mask]
                if subset.empty:
                    continue
                actual = subset[
                    ["actual_home_score", "actual_away_score"]
                ].to_numpy(dtype=float)
                predicted = subset[
                    ["predicted_home_score", "predicted_away_score"]
                ].to_numpy(dtype=float)
                rows.append({
                    "candidate": candidate,
                    "season": season,
                    "subset": label,
                    "games": int(len(subset)),
                    **_safe_metrics(actual, predicted),
                })
    return pd.DataFrame(rows)


def _week_metrics(
    predictions: dict[str, pd.DataFrame], tags: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    tag_frame = tags[["game_id", "week_block"]].reset_index(drop=True)
    for candidate, frame in predictions.items():
        aligned = frame.merge(
            tag_frame, on="game_id", how="left", validate="one_to_one"
        )
        for (season, block), subset in aligned.groupby(
            ["season", "week_block"], sort=True
        ):
            actual = subset[
                ["actual_home_score", "actual_away_score"]
            ].to_numpy(dtype=float)
            predicted = subset[
                ["predicted_home_score", "predicted_away_score"]
            ].to_numpy(dtype=float)
            rows.append({
                "candidate": candidate,
                "season": int(season),
                "week_block": block,
                "games": int(len(subset)),
                **_safe_metrics(actual, predicted),
            })
    return pd.DataFrame(rows)


def _write_report(
    output_dir: Path,
    manifest: dict,
    seasonal: pd.DataFrame,
    subset_metrics: pd.DataFrame,
) -> None:
    overall = manifest["metrics_overall"]
    final = manifest["final_season_metrics"]
    pooled_neutral = subset_metrics[
        (subset_metrics["season"] == "all")
        & (subset_metrics["subset"] == "Neutral site")
    ].set_index("candidate")
    final_neutral = subset_metrics[
        (subset_metrics["season"] == FINAL_SEASON)
        & (subset_metrics["subset"] == "Neutral site")
    ].set_index("candidate")
    early = subset_metrics[
        (subset_metrics["season"] == "all")
        & (subset_metrics["subset"] == "Weeks 0-3")
    ].set_index("candidate")
    lines = [
        "# Stage 6: role/site-feature stabilization",
        "",
        "## Method",
        "",
        (
            f"Role-specific same-season estimates are blended toward overall team "
            f"estimates with {ROLE_PRIOR_GAMES:g} pseudo-games. Role observation "
            "counts and effective weights are included. Neutral-site targets use "
            "overall estimates directly; the existing neutral-site indicator is "
            "retained."
        ),
        "",
        "## Results",
        "",
        "| Model | Overall score MAE | Overall margin MAE | Overall total MAE | 2025 score MAE |",
        "|---|---:|---:|---:|---:|",
        (
            f"| {BASELINE_LABEL} | {overall['baseline']['score_mae']:.4f} | "
            f"{overall['baseline']['margin_mae']:.4f} | "
            f"{overall['baseline']['total_mae']:.4f} | "
            f"{final['baseline']['score_mae']:.4f} |"
        ),
        (
            f"| {CANDIDATE_LABEL} | {overall['candidate']['score_mae']:.4f} | "
            f"{overall['candidate']['margin_mae']:.4f} | "
            f"{overall['candidate']['total_mae']:.4f} | "
            f"{final['candidate']['score_mae']:.4f} |"
        ),
        "",
        "### Required subsets",
        "",
        "| Subset | Baseline MAE | Challenger MAE |",
        "|---|---:|---:|",
    ]
    for label, table in (
        ("Neutral site, all seasons", pooled_neutral),
        (f"Neutral site, {FINAL_SEASON}", final_neutral),
        ("Weeks 0-3, all seasons", early),
    ):
        if BASELINE_LABEL in table.index and CANDIDATE_LABEL in table.index:
            lines.append(
                f"| {label} | {table.loc[BASELINE_LABEL, 'score_mae']:.4f} | "
                f"{table.loc[CANDIDATE_LABEL, 'score_mae']:.4f} |"
            )
    lines.extend([
        "",
        "### Seasonal MAE",
        "",
        "| Season | Baseline | Challenger |",
        "|---:|---:|---:|",
    ])
    baseline = seasonal[seasonal["candidate"] == BASELINE_LABEL].set_index(
        "season"
    )["score_mae"]
    challenger = seasonal[seasonal["candidate"] == CANDIDATE_LABEL].set_index(
        "season"
    )["score_mae"]
    lines.extend(
        f"| {season} | {baseline.loc[season]:.4f} | {challenger.loc[season]:.4f} |"
        for season in baseline.index
    )
    lines.extend([
        "",
        "## Decision and limitations",
        "",
        (
            "This is an isolated evaluation only. The 2025 block has been examined "
            "in earlier stages and is not an untouched final test. No production "
            "or published candidate changed."
        ),
        "",
        f"Paired final-season bootstrap: `{output_dir.name}/paired_week_bootstrap_ci.json`.",
        "",
        "## Reproduction",
        "",
        "- `python -m src.cfb_model.train.role_site_stabilization_experiment`",
        "- `python -m pytest`",
        "- No new API fetch or cache is required; the runner uses the existing warehouse.",
        "",
    ])
    (output_dir / "experiment_report.md").write_text("\n".join(lines))


def run_experiment(
    db_path: Path = DB_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict:
    """Run the stabilized role/site challenger against the production baseline."""
    ensure_warehouse(db_path)
    games = load_historical_games(db_path).reset_index(drop=True)
    schedule_history = load_schedule_history(db_path)
    advanced = load_advanced_stats(db_path)
    baseline_features = build_model_features(
        games,
        load_roster_priors(db_path),
        advanced,
        model_history=games,
        schedule_history=schedule_history,
    )
    role_features = build_game_role_features(games, advanced)
    challenger_features = pd.concat([baseline_features, role_features], axis=1)
    challenger_columns = MODEL_COLUMNS + STABILIZED_FEATURE_COLUMNS
    validate_market_free_feature_columns(challenger_columns)

    baseline_predictions = run_walk_forward_predictions(games, baseline_features)
    challenger_predictions = run_walk_forward_predictions(
        games, challenger_features, feature_columns=challenger_columns
    )
    predictions = {
        BASELINE_LABEL: baseline_predictions,
        CANDIDATE_LABEL: challenger_predictions,
    }
    baseline_seasonal = _season_metrics(baseline_predictions, BASELINE_LABEL)
    challenger_seasonal = _season_metrics(
        challenger_predictions, CANDIDATE_LABEL
    )
    seasonal = pd.concat(
        [baseline_seasonal, challenger_seasonal], ignore_index=True
    )
    tags = _build_subset_tags(games, role_features, schedule_history)
    subset_metrics = _subset_metrics(predictions, tags)
    week_metrics = _week_metrics(predictions, tags)
    paired = _paired_final_season(
        baseline_predictions, challenger_predictions, FINAL_SEASON
    )
    bootstrap = _week_cluster_bootstrap(paired)

    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_predictions.to_csv(output_dir / "baseline_predictions.csv", index=False)
    challenger_predictions.to_csv(
        output_dir / "challenger_predictions.csv", index=False
    )
    seasonal.to_csv(output_dir / "seasonal_metrics.csv", index=False)
    subset_metrics.to_csv(output_dir / "metrics_by_subset.csv", index=False)
    week_metrics.to_csv(output_dir / "metrics_by_week_range.csv", index=False)
    paired.to_csv(output_dir / "paired_error_deltas_final_season.csv", index=False)
    (output_dir / "paired_week_bootstrap_ci.json").write_text(
        json.dumps(bootstrap, indent=2) + "\n"
    )
    stabilized_role_data_dictionary().to_csv(
        output_dir / "feature_data_dictionary.csv", index=False
    )
    plot_mae_by_season_comparison(
        seasonal[["candidate", "season", "score_mae"]],
        output_dir / "seasonal_score_mae_comparison.png",
    )

    baseline_overall = score_metrics(
        baseline_predictions[["actual_home_score", "actual_away_score"]].to_numpy(),
        baseline_predictions[
            ["predicted_home_score", "predicted_away_score"]
        ].to_numpy(),
    )
    challenger_overall = score_metrics(
        challenger_predictions[
            ["actual_home_score", "actual_away_score"]
        ].to_numpy(),
        challenger_predictions[
            ["predicted_home_score", "predicted_away_score"]
        ].to_numpy(),
    )
    final_baseline = baseline_seasonal[
        baseline_seasonal["season"] == FINAL_SEASON
    ].iloc[0].to_dict()
    final_challenger = challenger_seasonal[
        challenger_seasonal["season"] == FINAL_SEASON
    ].iloc[0].to_dict()
    training_seasons = {
        str(season): [
            int(value) for value in sorted(
                games.loc[
                    (games["season"] < season)
                    & (games["season"] >= FIRST_TRAIN_SEASON),
                    "season",
                ].unique()
            )
        ]
        for season in sorted(seasonal["season"].unique())
    }
    model = _make_model()
    manifest = {
        "candidate_id": CANDIDATE_ID,
        "baseline_candidate_id": BASELINE_ID,
        "source_commit_sha": _source_commit_sha(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_seasons_by_validation_season": training_seasons,
        "validation_seasons": [int(season) for season in sorted(
            seasonal["season"].unique()
        )],
        "untouched_test_block": None,
        "untouched_test_block_note": (
            "No new untouched block is available: the 2021-2025 seasons were "
            "examined in earlier stage comparisons. No tuning was performed."
        ),
        "prediction_cutoff_policy": (
            "Only completed same-season observations before the target UTC "
            "calendar-date batch are used in role and overall estimates."
        ),
        "neutral_site_policy": (
            "Neutral targets receive overall team estimates directly, with zero "
            "role effective weight; the existing neutral_site indicator remains."
        ),
        "garbage_time_policy": (
            "No new CFBD endpoint is used. Existing baseline advanced-game "
            "features use excludeGarbageTime=false."
        ),
        "market_independence_check_passed": True,
        "baseline_feature_schema_sha256": _schema_hash(MODEL_COLUMNS),
        "feature_schema_sha256": _schema_hash(challenger_columns),
        "baseline_feature_list": MODEL_COLUMNS,
        "feature_list": challenger_columns,
        "feature_configuration": {
            "overall_prior_pseudo_games": ROLE_PRIOR_GAMES,
            "neutral_site_policy": "overall estimates; nominal home/away roles ignored",
            "role_history": "same-season non-neutral games only",
        },
        "cfbd_endpoint_and_cache_versions": {
            "new_endpoint": None,
            "new_cache": None,
            "historical_advanced_stats": (
                "existing stg_team_game_stats_raw cache; excludeGarbageTime=false"
            ),
        },
        "rating_half_life_days": None,
        "rating_prior_strength": None,
        "model_hyperparameters": model.estimator.get_params(),
        "random_seed": 42,
        "metrics_overall": {
            "baseline": baseline_overall,
            "candidate": challenger_overall,
        },
        "metrics_by_season": seasonal.to_dict("records"),
        "metrics_by_week_range": _json_records(week_metrics),
        "metrics_by_subset": _json_records(subset_metrics),
        "metrics_by_neutral_site": _json_records(
            subset_metrics[subset_metrics["subset"] == "Neutral site"]
        ),
        "final_season_metrics": {
            "baseline": final_baseline,
            "candidate": final_challenger,
        },
        "paired_week_cluster_bootstrap_ci": bootstrap,
        "game_count": int(len(baseline_predictions)),
        "role_feature_coverage": {
            "validation_games_with_any_stabilized_role_missing": int(
                tags["role_features_missing"].sum()
            ),
            "neutral_site_validation_games": int(tags["neutral_site"].sum()),
        },
        "promotion_decision": (
            "not promoted; evaluation only, no untouched final block, and owner "
            "approval is required before changing production"
        ),
        "artifact_paths": {
            "seasonal_metrics": "seasonal_metrics.csv",
            "seasonal_chart": "seasonal_score_mae_comparison.png",
            "week_range_metrics": "metrics_by_week_range.csv",
            "subset_metrics": "metrics_by_subset.csv",
            "neutral_site_metrics": "metrics_by_subset.csv",
            "paired_errors": "paired_error_deltas_final_season.csv",
            "paired_week_bootstrap_ci": "paired_week_bootstrap_ci.json",
            "feature_data_dictionary": "feature_data_dictionary.csv",
            "human_report": "experiment_report.md",
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _write_report(output_dir, manifest, seasonal, subset_metrics)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    print(json.dumps(run_experiment(args.db_path, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
