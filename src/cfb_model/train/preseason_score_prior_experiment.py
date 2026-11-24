"""Evaluate point-in-time prior-season scoring features for early-season games."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.cfb_model.data.warehouse import ensure_warehouse
from src.cfb_model.features.advanced import ADVANCED_METRICS
from src.cfb_model.features.preseason_score_prior import (
    FEATURE_COLUMNS as PRIOR_FEATURE_COLUMNS,
    PRIOR_GAMES,
    build_preseason_score_prior_features,
    preseason_score_prior_data_dictionary,
)
from src.cfb_model.reporting.diagnostics import plot_mae_by_season_comparison
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
    PROJECT_ROOT / "results" / "baselines" / "preseason_score_prior_stage5"
)
BASELINE_ID = "production_unweighted_baseline"
CANDIDATE_ID = "prior_season_score_shrinkage_4_games_stage5"
BASELINE_LABEL = "Production baseline"
CANDIDATE_LABEL = "Prior-season score prior (4 pseudo-games)"
FINAL_SEASON = max(BACKTEST_SEASONS)


def _week_block(week: object) -> str:
    value = pd.to_numeric(week, errors="coerce")
    if pd.isna(value) or value <= 3:
        return "Weeks 0-3"
    if value <= 6:
        return "Weeks 4-6"
    return "Weeks 7+"


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


def _safe_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    margin = actual[:, 0] - actual[:, 1]
    decisive = margin != 0
    labels = (margin[decisive] > 0).astype(int)
    if labels.size and np.unique(labels).size == 2:
        return score_metrics(actual, predicted)
    actual_flat = actual.ravel()
    predicted_flat = predicted.ravel()
    return {
        "score_mae": float(mean_absolute_error(actual_flat, predicted_flat)),
        "score_rmse": float(np.sqrt(mean_squared_error(actual_flat, predicted_flat))),
        "score_r2": float(r2_score(actual_flat, predicted_flat)),
        "home_score_mae": float(mean_absolute_error(actual[:, 0], predicted[:, 0])),
        "away_score_mae": float(mean_absolute_error(actual[:, 1], predicted[:, 1])),
        "margin_mae": float(
            mean_absolute_error(margin, predicted[:, 0] - predicted[:, 1])
        ),
        "total_mae": float(
            mean_absolute_error(
                actual[:, 0] + actual[:, 1], predicted[:, 0] + predicted[:, 1]
            )
        ),
        "winner_auc": float("nan"),
    }


def _schedule_after_fcs_flags(
    games: pd.DataFrame, schedule_history: pd.DataFrame
) -> pd.Series:
    required = {
        "game_date", "home_team", "away_team", "home_is_fbs", "away_is_fbs"
    }
    if missing := required - set(schedule_history.columns):
        raise ValueError(f"Schedule history is missing columns: {sorted(missing)}")
    history = schedule_history.copy()
    history["_day"] = pd.to_datetime(
        history["game_date"], utc=True, errors="coerce", format="mixed"
    ).dt.normalize()
    history = history[history["_day"].notna()]
    history["_day_ns"] = history["_day"].astype("int64")
    events = []
    for side in ("home", "away"):
        other = "away" if side == "home" else "home"
        is_fbs = history[f"{side}_is_fbs"].fillna(False).astype(bool)
        events.append(pd.DataFrame({
            "team": history.loc[is_fbs, f"{side}_team"].to_numpy(),
            "day_ns": history.loc[is_fbs, "_day_ns"].to_numpy(),
            "last_opponent_was_fcs": (
                ~history.loc[is_fbs, f"{other}_is_fbs"].fillna(False).astype(bool)
            ).to_numpy(),
        }))
    prior_games = pd.concat(events, ignore_index=True).sort_values(
        ["team", "day_ns"], kind="stable"
    )
    targets = []
    canonical = games.reset_index(drop=True)
    target_days = pd.to_datetime(
        canonical["game_date"], utc=True, errors="coerce", format="mixed"
    ).dt.normalize()
    histories = {
        team: (
            rows["day_ns"].to_numpy(dtype=np.int64),
            rows["last_opponent_was_fcs"].to_numpy(dtype=bool),
        )
        for team, rows in prior_games.groupby("team", sort=False)
    }
    for position, row in canonical.iterrows():
        target_day = target_days.iloc[position]
        flags = []
        for side in ("home", "away"):
            team_history = histories.get(row[f"{side}_team"])
            if team_history is None or pd.isna(target_day):
                flags.append(False)
                continue
            days, last_opponent_was_fcs = team_history
            prior_end = int(np.searchsorted(days, target_day.value, side="left"))
            flags.append(
                bool(last_opponent_was_fcs[prior_end - 1])
                if prior_end
                else False
            )
        targets.append(any(flags))
    return pd.Series(targets, index=games.index, dtype=bool)


def _subset_tags(
    games: pd.DataFrame,
    prior_features: pd.DataFrame,
    baseline_features: pd.DataFrame,
    schedule_history: pd.DataFrame,
) -> pd.DataFrame:
    tags = pd.DataFrame(index=games.index)
    week = pd.to_numeric(games["week"], errors="coerce").fillna(0)
    tags["week_block"] = week.map(_week_block)
    tags["neutral_site"] = games.get(
        "neutral_site", pd.Series(False, index=games.index)
    ).fillna(False).astype(bool)
    tags["after_fcs"] = _schedule_after_fcs_flags(games, schedule_history)
    tags["prior_missing_any"] = (
        prior_features["home_prior_available"].eq(0)
        | prior_features["away_prior_available"].eq(0)
    )
    advanced_columns = [
        column for column in baseline_features
        if any(metric in column for metric in ADVANCED_METRICS)
    ]
    tags["baseline_advanced_missing"] = (
        baseline_features[advanced_columns].isna().any(axis=1)
        if advanced_columns
        else False
    )
    tags["game_id"] = games["game_id"].to_numpy()
    return tags


def _metrics_by_subset(
    predictions: dict[str, pd.DataFrame], tags: pd.DataFrame
) -> pd.DataFrame:
    subset_definitions = {
        "Weeks 0-3": tags["week_block"].eq("Weeks 0-3"),
        "Weeks 4-6": tags["week_block"].eq("Weeks 4-6"),
        "Weeks 7+": tags["week_block"].eq("Weeks 7+"),
        "Neutral site": tags["neutral_site"],
        "After FCS opponent": tags["after_fcs"],
        "Prior-season scoring unavailable": tags["prior_missing_any"],
        "Baseline advanced stats missing": tags["baseline_advanced_missing"],
    }
    rows = []
    tag_frame = tags.reset_index(drop=True)
    for label, frame in predictions.items():
        aligned = frame.merge(
            tag_frame[["game_id"]].assign(
                **{name: mask.to_numpy() for name, mask in subset_definitions.items()}
            ),
            on="game_id",
            how="left",
            validate="one_to_one",
        )
        for subset_name, mask in subset_definitions.items():
            subset = aligned[aligned[subset_name].fillna(False)]
            if subset.empty:
                continue
            actual = subset[
                ["actual_home_score", "actual_away_score"]
            ].to_numpy(dtype=float)
            predicted = subset[
                ["predicted_home_score", "predicted_away_score"]
            ].to_numpy(dtype=float)
            rows.append({
                "candidate": label,
                "subset": subset_name,
                "games": int(len(subset)),
                **_safe_metrics(actual, predicted),
            })
    return pd.DataFrame(rows)


def _week_range_metrics(
    predictions: dict[str, pd.DataFrame], tags: pd.DataFrame
) -> pd.DataFrame:
    tag_frame = tags[["game_id", "week_block"]].reset_index(drop=True)
    rows = []
    for label, frame in predictions.items():
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
                "candidate": label,
                "season": int(season),
                "week_block": block,
                "games": int(len(subset)),
                **_safe_metrics(actual, predicted),
            })
    return pd.DataFrame(rows)


def _schema_hash(columns: list[str]) -> str:
    return hashlib.sha256("\n".join(columns).encode("utf-8")).hexdigest()


def _json_records(frame: pd.DataFrame) -> list[dict]:
    return frame.astype(object).where(pd.notna(frame), None).to_dict("records")


def _source_commit_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_report(
    output_dir: Path,
    baseline_seasonal: pd.DataFrame,
    candidate_seasonal: pd.DataFrame,
    subset_metrics: pd.DataFrame,
    manifest: dict,
) -> None:
    early = subset_metrics[subset_metrics["subset"] == "Weeks 0-3"].set_index(
        "candidate"
    )
    baseline_early = early.loc[BASELINE_LABEL] if BASELINE_LABEL in early.index else None
    candidate_early = (
        early.loc[CANDIDATE_LABEL] if CANDIDATE_LABEL in early.index else None
    )
    final = manifest["final_season_metrics"]
    lines = [
        "# Stage 5: prior-season scoring prior",
        "",
        "## Method",
        "",
        (
            f"The challenger adds prior-season team scoring and scoring-allowed "
            f"averages, shrunk toward current-season results with {PRIOR_GAMES:g} "
            "prior pseudo-games. Features use only completed FBS-vs-FBS results "
            "before the target UTC date. The prior's weight is "
            f"`{PRIOR_GAMES:g} / (prior current-season games + {PRIOR_GAMES:g})` "
            "when prior-season data exist."
        ),
        "",
        (
            "Returning-production endpoint values were not used because historical "
            "publication-time snapshots could not be verified. Existing roster "
            "priors and Elo features remain unchanged."
        ),
        "",
        "## Results",
        "",
        "| Model | All-season score MAE | Final-season score MAE |",
        "|---|---:|---:|",
        (
            f"| {BASELINE_LABEL} | "
            f"{manifest['metrics_overall']['baseline']['score_mae']:.4f} | "
            f"{final['baseline']['score_mae']:.4f} |"
        ),
        (
            f"| {CANDIDATE_LABEL} | "
            f"{manifest['metrics_overall']['candidate']['score_mae']:.4f} | "
            f"{final['candidate']['score_mae']:.4f} |"
        ),
        "",
        "### Early-season Weeks 0–3",
        "",
    ]
    if baseline_early is not None and candidate_early is not None:
        lines.extend([
            (
                f"Baseline MAE: {baseline_early['score_mae']:.4f} "
                f"({int(baseline_early['games'])} games); "
                f"challenger MAE: {candidate_early['score_mae']:.4f} "
                f"({int(candidate_early['games'])} games)."
            ),
            "",
        ])
    lines.extend([
        "### Seasonal MAE",
        "",
        "| Season | Baseline | Challenger |",
        "|---:|---:|---:|",
    ])
    base = baseline_seasonal.set_index("season")["score_mae"]
    candidate = candidate_seasonal.set_index("season")["score_mae"]
    lines.extend(
        f"| {season} | {base.loc[season]:.4f} | {candidate.loc[season]:.4f} |"
        for season in base.index
    )
    lines.extend([
        "",
        "## Decision and limitations",
        "",
        (
            "This is an isolated challenger only; no production or published "
            "candidate changed. The 2025 block is not considered untouched because "
            "it was examined during earlier sequential experiments. The paired "
            "week-cluster interval and early-season subset should be considered "
            "before any promotion decision."
        ),
        "",
        f"Paired final-season bootstrap: `{output_dir.name}/paired_week_bootstrap_ci.json`.",
        "",
    ])
    (output_dir / "experiment_report.md").write_text("\n".join(lines))


def run_experiment(
    db_path: Path = DB_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict:
    """Run one predeclared prior-season scoring challenger and save diagnostics."""
    ensure_warehouse(db_path)
    games = load_historical_games(db_path).reset_index(drop=True)
    schedule_history = load_schedule_history(db_path)
    baseline_features = build_model_features(
        games,
        load_roster_priors(db_path),
        load_advanced_stats(db_path),
        model_history=games,
        schedule_history=schedule_history,
    )
    prior_features = build_preseason_score_prior_features(games)
    challenger_features = pd.concat(
        [baseline_features, prior_features], axis=1
    )
    challenger_columns = MODEL_COLUMNS + PRIOR_FEATURE_COLUMNS
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
    candidate_seasonal = _season_metrics(challenger_predictions, CANDIDATE_LABEL)
    seasonal = pd.concat([baseline_seasonal, candidate_seasonal], ignore_index=True)
    tags = _subset_tags(games, prior_features, baseline_features, schedule_history)
    subset_metrics = _metrics_by_subset(predictions, tags)
    week_metrics = _week_range_metrics(predictions, tags)
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
    preseason_score_prior_data_dictionary().to_csv(
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
    candidate_overall = score_metrics(
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
    final_candidate = candidate_seasonal[
        candidate_seasonal["season"] == FINAL_SEASON
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
            "Only completed canonical FBS-vs-FBS scores before the target UTC "
            "calendar-date batch are used in prior-score features."
        ),
        "garbage_time_policy": (
            "No additional CFBD endpoint is used by this challenger. Existing "
            "baseline advanced-game features use excludeGarbageTime=false."
        ),
        "market_independence_check_passed": True,
        "baseline_feature_schema_sha256": _schema_hash(MODEL_COLUMNS),
        "feature_schema_sha256": _schema_hash(challenger_columns),
        "baseline_feature_list": MODEL_COLUMNS,
        "feature_list": challenger_columns,
        "feature_configuration": {
            "prior_season_pseudo_games": PRIOR_GAMES,
            "prior_season_selection": "latest earlier season with completed scores",
            "current_season_cutoff": "before target UTC calendar-date batch",
            "returning_production_used": False,
            "roster_priors_changed": False,
        },
        "cfbd_endpoint_and_cache_versions": {
            "new_endpoint": None,
            "new_cache": None,
            "historical_availability_basis": (
                "prior-season results from canonical completed game scores; "
                "no current-day historical endpoint data used"
            ),
        },
        "rating_half_life_days": None,
        "rating_prior_strength": None,
        "model_hyperparameters": model.estimator.get_params(),
        "random_seed": 42,
        "metrics_overall": {
            "baseline": baseline_overall,
            "candidate": candidate_overall,
        },
        "metrics_by_season": seasonal.to_dict("records"),
        "metrics_by_week_range": _json_records(week_metrics),
        "metrics_by_subset": _json_records(subset_metrics),
        "final_season_metrics": {
            "baseline": final_baseline,
            "candidate": final_candidate,
        },
        "paired_week_cluster_bootstrap_ci": bootstrap,
        "game_count": int(len(baseline_predictions)),
        "prior_score_feature_coverage": {
            "both_teams_prior_available": int(
                (prior_features["home_prior_available"].eq(1)
                 & prior_features["away_prior_available"].eq(1)).sum()
            ),
            "either_team_prior_missing": int(
                (prior_features["home_prior_available"].eq(0)
                 | prior_features["away_prior_available"].eq(0)).sum()
            ),
            "mean_home_prior_weight": float(
                prior_features["home_prior_weight"].mean()
            ),
            "mean_away_prior_weight": float(
                prior_features["away_prior_weight"].mean()
            ),
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
            "paired_errors": "paired_error_deltas_final_season.csv",
            "paired_week_bootstrap_ci": "paired_week_bootstrap_ci.json",
            "feature_data_dictionary": "feature_data_dictionary.csv",
            "human_report": "experiment_report.md",
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _write_report(
        output_dir,
        baseline_seasonal,
        candidate_seasonal,
        subset_metrics,
        manifest,
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run_experiment(args.db_path, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
