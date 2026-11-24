"""Screen cumulative rating, pace, role, and optional feature combinations."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.cfb_model.data.warehouse import ensure_warehouse
from src.cfb_model.features.advanced import ADVANCED_METRICS, FEATURE_COLUMNS as ADVANCED_COLUMNS
from src.cfb_model.features.cfbd_game_features import (
    EFFICIENCY_FEATURE_COLUMNS,
    PACE_FEATURE_COLUMNS,
    build_cfbd_efficiency_features,
    build_cfbd_pace_features,
    cfbd_feature_data_dictionary,
    cfbd_stat_coverage,
)
from src.cfb_model.features.game_role_performance import (
    FEATURE_COLUMNS as ROLE_FEATURE_COLUMNS,
    ROLE_PRIOR_GAMES,
    build_game_role_features,
    stabilized_role_data_dictionary,
)
from src.cfb_model.features.opponent_adjusted import (
    FEATURE_COLUMNS as RATING_COLUMNS,
)
from src.cfb_model.features.preseason_score_prior import (
    FEATURE_COLUMNS as PRIOR_FEATURE_COLUMNS,
    PRIOR_GAMES,
    build_preseason_score_prior_features,
    preseason_score_prior_data_dictionary,
)
from src.cfb_model.reporting.diagnostics import plot_mae_by_season_comparison
from src.cfb_model.train.cfbd_game_experiment import load_cfbd_game_stats
from src.cfb_model.train.preseason_score_prior_experiment import (
    _safe_metrics,
    _schedule_after_fcs_flags,
)
from src.cfb_model.train.recency_experiment import (
    _paired_final_season,
    _replace_ratings,
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
    PROJECT_ROOT / "results" / "baselines" / "final_combined_optimization"
)
FINAL_SEASON = max(BACKTEST_SEASONS)
RECENCY_HALF_LIFE_DAYS = 56
RECENCY_PRIOR_STRENGTH = 20.0
RECENCY_RIDGE_ALPHA = 10.0
ROLE_COLUMNS = ROLE_FEATURE_COLUMNS["stabilized"]

BASELINE_ID = "production"
RECENCY_ID = "recency"
RECENCY_PACE_ID = "recency_pace"
CORE_ID = "recency_pace_role"
PRIOR_ABLATION_ID = "recency_pace_role_prior"
EFFICIENCY_ABLATION_ID = "recency_pace_role_efficiency"

CANDIDATE_LABELS = {
    BASELINE_ID: "Production baseline",
    RECENCY_ID: "Recency ratings",
    RECENCY_PACE_ID: "Recency + pace",
    CORE_ID: "Recency + pace + role/site",
    PRIOR_ABLATION_ID: "Core + Stage 5 score prior",
    EFFICIENCY_ABLATION_ID: "Core + Stage 4B efficiency",
}


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


def _season_metrics(
    predictions: pd.DataFrame, candidate_id: str
) -> pd.DataFrame:
    rows = []
    for season, fold in predictions.groupby("season", sort=True):
        actual = fold[["actual_home_score", "actual_away_score"]].to_numpy(dtype=float)
        predicted = fold[
            ["predicted_home_score", "predicted_away_score"]
        ].to_numpy(dtype=float)
        rows.append({
            "candidate_id": candidate_id,
            "candidate": CANDIDATE_LABELS[candidate_id],
            "season": int(season),
            "games": int(len(fold)),
            **score_metrics(actual, predicted),
        })
    return pd.DataFrame(rows)


def _build_tags(
    games: pd.DataFrame,
    advanced_features: pd.DataFrame,
    role_features: pd.DataFrame,
    prior_features: pd.DataFrame,
    schedule_history: pd.DataFrame,
) -> pd.DataFrame:
    tags = pd.DataFrame({"game_id": games["game_id"].to_numpy()})
    tags["week_block"] = pd.to_numeric(
        games["week"], errors="coerce"
    ).fillna(0).map(_week_block).to_numpy()
    tags["neutral_site"] = games.get(
        "neutral_site", pd.Series(False, index=games.index)
    ).fillna(False).astype(bool).to_numpy()
    tags["after_fcs"] = _schedule_after_fcs_flags(
        games, schedule_history
    ).to_numpy()
    tags["role_features_missing"] = role_features[ROLE_COLUMNS].isna().any(
        axis=1
    ).to_numpy()
    tags["prior_missing"] = (
        prior_features["home_prior_available"].eq(0)
        | prior_features["away_prior_available"].eq(0)
    ).to_numpy()
    tags["advanced_missing"] = advanced_features[ADVANCED_COLUMNS].isna().any(
        axis=1
    ).to_numpy()
    return tags


def _subset_metrics(
    predictions: dict[str, pd.DataFrame], tags: pd.DataFrame
) -> pd.DataFrame:
    definitions = {
        "Weeks 0-3": "week_block",
        "Weeks 4-6": "week_block",
        "Weeks 7+": "week_block",
        "Neutral site": "neutral_site",
        "After FCS opponent": "after_fcs",
        "Role features missing": "role_features_missing",
        "Stage 5 prior unavailable": "prior_missing",
        "Advanced features missing": "advanced_missing",
    }
    rows = []
    tag_frame = tags.reset_index(drop=True)
    for candidate_id, frame in predictions.items():
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
                    "candidate_id": candidate_id,
                    "candidate": CANDIDATE_LABELS[candidate_id],
                    "season": season,
                    "subset": label,
                    "games": int(len(subset)),
                    **_safe_metrics(actual, predicted),
                })
    return pd.DataFrame(rows)


def _paired_errors(
    baseline: pd.DataFrame,
    challengers: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    keys = ["game_id", "season", "week", "game_date", "home_team", "away_team"]
    baseline_rows = baseline[
        keys + ["score_absolute_error"]
    ].rename(columns={"score_absolute_error": "baseline_score_absolute_error"})
    paired_rows = []
    for candidate_id, predictions in challengers.items():
        candidate = predictions[
            keys + ["score_absolute_error"]
        ].rename(columns={"score_absolute_error": "candidate_score_absolute_error"})
        paired = baseline_rows.merge(
            candidate,
            on=keys,
            how="outer",
            validate="one_to_one",
            indicator=True,
        )
        if not paired["_merge"].eq("both").all():
            raise ValueError(
                f"{candidate_id} predictions do not match baseline game IDs."
            )
        paired = paired.drop(columns="_merge")
        paired["candidate_id"] = candidate_id
        paired["candidate"] = CANDIDATE_LABELS[candidate_id]
        paired["score_absolute_error_delta"] = (
            paired["candidate_score_absolute_error"]
            - paired["baseline_score_absolute_error"]
        )
        paired_rows.append(paired)
    return pd.concat(paired_rows, ignore_index=True)


def _feature_dictionary() -> pd.DataFrame:
    recency_rows = pd.DataFrame([
        {
            "feature": column,
            "definition": (
                "Recency-weighted opponent-adjusted offense or defense rating; "
                "56-day half-life, prior strength 20, Ridge alpha 10."
            ),
            "source": "completed FBS-vs-FBS scores before the target UTC date",
            "availability_cutoff": "before the target UTC calendar-date batch",
        }
        for column in RATING_COLUMNS
    ])
    dictionaries = [
        cfbd_feature_data_dictionary(),
        stabilized_role_data_dictionary(),
        preseason_score_prior_data_dictionary(),
        recency_rows,
    ]
    dictionary = pd.concat(dictionaries, ignore_index=True, sort=False)
    return dictionary.drop_duplicates("feature", keep="first")


def _write_candidate_chart(
    output_dir: Path,
    candidate_id: str,
    seasonal: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    challenger_predictions: pd.DataFrame,
) -> None:
    candidate_dir = output_dir / candidate_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    metrics = seasonal[
        seasonal["candidate_id"].isin([BASELINE_ID, candidate_id])
    ]
    metrics[["candidate", "season", "score_mae"]].to_csv(
        candidate_dir / "seasonal_metrics.csv", index=False
    )
    plot_mae_by_season_comparison(
        metrics[["candidate", "season", "score_mae"]],
        candidate_dir / "seasonal_score_mae_comparison.png",
    )
    baseline_predictions.to_csv(
        candidate_dir / "baseline_predictions.csv", index=False
    )
    challenger_predictions.to_csv(
        candidate_dir / "challenger_predictions.csv", index=False
    )


def run_experiment(
    db_path: Path = DB_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    validation_seasons: list[int] | None = None,
    source_metadata: dict | None = None,
) -> dict:
    """Run fixed Stage 3–6 combinations on identical walk-forward folds."""
    ensure_warehouse(db_path)
    games = load_historical_games(db_path).reset_index(drop=True)
    validation_seasons = sorted(set(
        BACKTEST_SEASONS if validation_seasons is None else validation_seasons
    ))
    if not validation_seasons:
        raise ValueError("At least one validation season is required.")
    missing_seasons = set(validation_seasons) - set(games["season"].unique())
    if missing_seasons:
        raise ValueError(
            f"No completed validation games for seasons: {sorted(missing_seasons)}"
        )
    final_season = max(validation_seasons)
    schedule_history = load_schedule_history(db_path)
    advanced = load_advanced_stats(db_path)
    baseline_features = build_model_features(
        games,
        load_roster_priors(db_path),
        advanced,
        model_history=games,
        schedule_history=schedule_history,
    )
    stats, stat_source_quality = load_cfbd_game_stats(db_path)
    stat_coverage = cfbd_stat_coverage(games, stats)
    recency_features = _replace_ratings(
        games,
        baseline_features,
        RECENCY_HALF_LIFE_DAYS,
        RECENCY_PRIOR_STRENGTH,
        RECENCY_RIDGE_ALPHA,
    )
    pace_features = build_cfbd_pace_features(games, stats)
    efficiency_features = build_cfbd_efficiency_features(games, stats)
    role_features = build_game_role_features(games, advanced)
    prior_features = build_preseason_score_prior_features(games)

    non_advanced_columns = [
        column for column in MODEL_COLUMNS if column not in ADVANCED_COLUMNS
    ]
    core_columns = MODEL_COLUMNS + PACE_FEATURE_COLUMNS + ROLE_COLUMNS
    efficiency_columns = (
        non_advanced_columns
        + PACE_FEATURE_COLUMNS
        + ROLE_COLUMNS
        + EFFICIENCY_FEATURE_COLUMNS
    )
    candidate_inputs = {
        BASELINE_ID: (baseline_features, MODEL_COLUMNS),
        RECENCY_ID: (recency_features, MODEL_COLUMNS),
        RECENCY_PACE_ID: (
            pd.concat([recency_features, pace_features], axis=1),
            MODEL_COLUMNS + PACE_FEATURE_COLUMNS,
        ),
        CORE_ID: (
            pd.concat([recency_features, pace_features, role_features], axis=1),
            core_columns,
        ),
        PRIOR_ABLATION_ID: (
            pd.concat(
                [recency_features, pace_features, role_features, prior_features],
                axis=1,
            ),
            core_columns + PRIOR_FEATURE_COLUMNS,
        ),
        EFFICIENCY_ABLATION_ID: (
            pd.concat(
                [
                    recency_features.drop(columns=ADVANCED_COLUMNS),
                    pace_features,
                    role_features,
                    efficiency_features,
                ],
                axis=1,
            ),
            efficiency_columns,
        ),
    }

    predictions = {}
    candidate_details = {}
    for candidate_id, (features, columns) in candidate_inputs.items():
        validate_market_free_feature_columns(columns)
        candidate_predictions = run_walk_forward_predictions(
            games,
            features,
            val_seasons=validation_seasons,
            feature_columns=columns,
        )
        predictions[candidate_id] = candidate_predictions
        candidate_details[candidate_id] = {
            "feature_columns": columns,
            "feature_schema_sha256": _schema_hash(columns),
            "predictions": candidate_predictions,
        }

    baseline_predictions = predictions[BASELINE_ID]
    challengers = {
        candidate_id: frame
        for candidate_id, frame in predictions.items()
        if candidate_id != BASELINE_ID
    }
    seasonal = pd.concat(
        [
            _season_metrics(frame, candidate_id)
            for candidate_id, frame in predictions.items()
        ],
        ignore_index=True,
    )
    tags = _build_tags(
        games, advanced_features=baseline_features,
        role_features=role_features, prior_features=prior_features,
        schedule_history=schedule_history,
    )
    subset_metrics = _subset_metrics(predictions, tags)
    paired_errors = _paired_errors(baseline_predictions, challengers)
    bootstrap_by_candidate = {}
    for candidate_id, frame in challengers.items():
        paired = _paired_final_season(
            baseline_predictions, frame, final_season
        )
        bootstrap_by_candidate[candidate_id] = _week_cluster_bootstrap(paired)

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(exist_ok=True)
    for candidate_id, frame in predictions.items():
        frame.to_csv(prediction_dir / f"{candidate_id}.csv", index=False)
    seasonal.to_csv(output_dir / "seasonal_metrics.csv", index=False)
    subset_metrics.to_csv(output_dir / "metrics_by_subset.csv", index=False)
    subset_metrics[
        subset_metrics["subset"].str.startswith("Weeks ")
    ].to_csv(output_dir / "metrics_by_week_range.csv", index=False)
    paired_errors.to_csv(output_dir / "paired_error_deltas.csv", index=False)
    (output_dir / "paired_week_bootstrap_ci.json").write_text(
        json.dumps(bootstrap_by_candidate, indent=2) + "\n"
    )
    feature_dictionary = _feature_dictionary()
    all_new_columns = list(dict.fromkeys(
        RATING_COLUMNS
        + PACE_FEATURE_COLUMNS
        + ROLE_COLUMNS
        + PRIOR_FEATURE_COLUMNS
        + EFFICIENCY_FEATURE_COLUMNS
    ))
    feature_dictionary[
        feature_dictionary["feature"].isin(all_new_columns)
    ].to_csv(output_dir / "feature_data_dictionary.csv", index=False)
    plot_mae_by_season_comparison(
        seasonal[["candidate", "season", "score_mae"]],
        output_dir / "seasonal_score_mae_all_candidates.png",
    )
    for candidate_id, frame in challengers.items():
        _write_candidate_chart(
            output_dir, candidate_id, seasonal, baseline_predictions, frame
        )

    overall_metrics = {}
    seasonal_metrics = {}
    for candidate_id, frame in predictions.items():
        actual = frame[["actual_home_score", "actual_away_score"]].to_numpy()
        predicted = frame[
            ["predicted_home_score", "predicted_away_score"]
        ].to_numpy()
        overall_metrics[candidate_id] = score_metrics(actual, predicted)
        seasonal_metrics[candidate_id] = seasonal[
            seasonal["candidate_id"] == candidate_id
        ].to_dict("records")

    provisional_2026 = validation_seasons == [2026]
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
        "experiment_id": "final_combined_optimization_screen",
        "source_commit_sha": _source_commit_sha(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_candidate_id": BASELINE_ID,
        "candidate_ids": list(candidate_inputs),
        "candidate_labels": CANDIDATE_LABELS,
        "training_seasons_by_validation_season": training_seasons,
        "validation_seasons": validation_seasons,
        "untouched_test_block": None,
        "untouched_test_block_note": (
            (
                "Provisional 2026 current-season screen; the season is incomplete "
                "and the Week 8 production evaluation predates this comparison. "
                "Do not tune or promote from this screen."
                if provisional_2026
                else (
                    "All 2021-2025 seasons were examined in prior sequential stages; "
                    "this is a screening experiment, not a promotion test."
                )
            )
        ),
        "source_data_snapshot": source_metadata,
        "prediction_cutoff_policy": (
            (
                "Feature builders exclude games on the target UTC calendar date. "
                "The 2026 CFBD statistics and roster priors are pinned current "
                "snapshots, not archived as-of-game-date snapshots; provider "
                "revision timing is unavailable."
            )
            if provisional_2026
            else (
                "Each feature builder uses results available before the target UTC "
                "calendar-date batch."
            )
        ),
        "garbage_time_policy": (
            "The 2026 advanced-game cache and historical Stage 4 CFBD tables use "
            "excludeGarbageTime=false."
            if provisional_2026
            else (
                "No new CFBD endpoint is queried; existing Stage 4 CFBD tables "
                "retain the excludeGarbageTime=false policy."
            )
        ),
        "market_independence_check_passed": True,
        "baseline_feature_schema_sha256": _schema_hash(MODEL_COLUMNS),
        "candidate_features": {
            candidate_id: {
                "feature_schema_sha256": detail["feature_schema_sha256"],
                "feature_list": detail["feature_columns"],
            }
            for candidate_id, detail in candidate_details.items()
        },
        "fixed_configuration": {
            "recency_half_life_days": RECENCY_HALF_LIFE_DAYS,
            "recency_prior_strength": RECENCY_PRIOR_STRENGTH,
            "recency_ridge_alpha": RECENCY_RIDGE_ALPHA,
            "role_overall_prior_pseudo_games": ROLE_PRIOR_GAMES,
            "preseason_score_prior_pseudo_games": PRIOR_GAMES,
            "selection_or_tuning_performed": False,
            "efficiency_ablation_replaces_legacy_advanced_features": True,
        },
        "cfbd_source_quality": stat_source_quality,
        "cfbd_stat_coverage": stat_coverage,
        "cfbd_endpoint_and_cache_versions": {
            "new_endpoint": (
                source_metadata.get("advanced_game_stats", {}).get("endpoint")
                if source_metadata else None
            ),
            "new_cache": (
                source_metadata.get("advanced_game_stats", {}).get("path")
                if source_metadata else None
            ),
            "historical_advanced_stats": (
                "stg_team_game_stats_raw and stg_pbp_team_game_raw; "
                "excludeGarbageTime=false"
            ),
        },
        "model_hyperparameters": model.estimator.get_params(),
        "random_seed": 42,
        "game_count": int(len(baseline_predictions)),
        "metrics_overall": overall_metrics,
        "metrics_by_season": seasonal_metrics,
        "metrics_by_subset": _json_records(subset_metrics),
        "paired_week_cluster_bootstrap_ci": bootstrap_by_candidate,
        "promotion_decision": (
            (
                "none; this is a provisional screen on an incomplete 2026 season "
                "with an earlier Week 8 production evaluation; do not tune or promote"
            )
            if provisional_2026
            else (
                "none; do not promote from this screen because no untouched block "
                "exists and all candidates were compared on previously examined seasons"
            )
        ),
        "artifact_paths": {
            "all_candidate_seasonal_chart": "seasonal_score_mae_all_candidates.png",
            "candidate_charts": {
                candidate_id: (
                    f"{candidate_id}/seasonal_score_mae_comparison.png"
                )
                for candidate_id in challengers
            },
            "metrics_by_season": "seasonal_metrics.csv",
            "metrics_by_week_range": "metrics_by_week_range.csv",
            "metrics_by_subset": "metrics_by_subset.csv",
            "paired_errors": "paired_error_deltas.csv",
            "paired_week_cluster_intervals": "paired_week_bootstrap_ci.json",
            "feature_dictionary": "feature_data_dictionary.csv",
            "human_report": "experiment_report.md",
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _write_report(
        output_dir,
        seasonal,
        subset_metrics,
        overall_metrics,
        bootstrap_by_candidate,
        validation_seasons=validation_seasons,
        provisional_2026=provisional_2026,
    )
    return manifest


def _write_report(
    output_dir: Path,
    seasonal: pd.DataFrame,
    subset_metrics: pd.DataFrame,
    overall_metrics: dict,
    bootstrap: dict,
    *,
    validation_seasons: list[int],
    provisional_2026: bool,
) -> None:
    final_season = max(validation_seasons)
    final_seasonal = seasonal[seasonal["season"] == final_season].set_index(
        "candidate_id"
    )
    subset_all = subset_metrics[subset_metrics["season"] == "all"]
    if provisional_2026:
        games = int(final_seasonal["games"].iloc[0])
        lines = [
            "# Provisional 2026 combined-model comparison",
            "",
            (
                f"All candidates were evaluated on the same {games} completed "
                "FBS-vs-FBS games in the pinned 2026 snapshot (latest game "
                "2026-11-22). Each candidate was trained only on prior seasons "
                "and no parameters were tuned. Feature builders exclude games "
                "on each target's UTC date, but CFBD statistics and roster priors "
                "are pinned current snapshots rather than archived as-of-game-date "
                "snapshots, so provider revisions are a limitation. The season is "
                "incomplete, and a Week 8 production evaluation predates this "
                "comparison; treat it as provisional, not a fully untouched test."
            ),
            "",
            "| Candidate | Games | 2026 score MAE | 2026 margin MAE | 2026 total MAE |",
            "|---|---:|---:|---:|---:|",
        ]
    else:
        lines = [
            "# Final combined optimization screen",
            "",
            (
                f"All candidates use the same {min(validation_seasons)}–"
                f"{max(validation_seasons)} walk-forward folds and the same "
                "score model. Parameters were fixed before this run. The "
                "seasons have already been examined in earlier stages, so "
                "results are screening-only."
            ),
            "",
            "| Candidate | Overall score MAE | "
            f"{final_season} score MAE | {final_season} margin MAE | "
            f"{final_season} total MAE |",
            "|---|---:|---:|---:|---:|",
        ]
    for candidate_id, label in CANDIDATE_LABELS.items():
        overall = overall_metrics[candidate_id]
        final = final_seasonal.loc[candidate_id]
        if provisional_2026:
            lines.append(
                f"| {label} | {int(final['games'])} | "
                f"{final['score_mae']:.4f} | {final['margin_mae']:.4f} | "
                f"{final['total_mae']:.4f} |"
            )
        else:
            lines.append(
                f"| {label} | {overall['score_mae']:.4f} | "
                f"{final['score_mae']:.4f} | {final['margin_mae']:.4f} | "
                f"{final['total_mae']:.4f} |"
            )
    lines.extend([
        "",
        "## Opening-week and neutral-site MAE",
        "",
        (
            "| Candidate | Weeks 0–3 | Neutral-site MAE |"
            if provisional_2026
            else (
                "| Candidate | Weeks 0–3 | Neutral-site all seasons | "
                f"Neutral-site {final_season} |"
            )
        ),
        "|---|---:|---:" + ("" if provisional_2026 else "---:") + "|",
    ])
    final_neutral = subset_metrics[
        (subset_metrics["season"] == final_season)
        & (subset_metrics["subset"] == "Neutral site")
    ].set_index("candidate_id")
    for candidate_id, label in CANDIDATE_LABELS.items():
        early = subset_all[
            (subset_all["candidate_id"] == candidate_id)
            & (subset_all["subset"] == "Weeks 0-3")
        ]
        neutral = subset_all[
            (subset_all["candidate_id"] == candidate_id)
            & (subset_all["subset"] == "Neutral site")
        ]
        if early.empty or neutral.empty or candidate_id not in final_neutral.index:
            continue
        if provisional_2026:
            lines.append(
                f"| {label} | {early.iloc[0]['score_mae']:.4f} | "
                f"{final_neutral.loc[candidate_id, 'score_mae']:.4f} |"
            )
        else:
            lines.append(
                f"| {label} | {early.iloc[0]['score_mae']:.4f} | "
                f"{neutral.iloc[0]['score_mae']:.4f} | "
                f"{final_neutral.loc[candidate_id, 'score_mae']:.4f} |"
            )
    lines.extend([
        "",
        "## Seasonal score MAE",
        "",
        "| Season | " + " | ".join(CANDIDATE_LABELS.values()) + " |",
        "|---:|" + "---:|" * len(CANDIDATE_LABELS),
    ])
    for season in sorted(seasonal["season"].unique()):
        by_candidate = seasonal[seasonal["season"] == season].set_index(
            "candidate_id"
        )["score_mae"]
        lines.append(
            f"| {season} | "
            + " | ".join(
                f"{by_candidate.loc[candidate_id]:.4f}"
                for candidate_id in CANDIDATE_LABELS
            )
            + " |"
        )
    lines.extend([
        "",
        (
            f"## Paired {final_season} intervals"
            if provisional_2026
            else "## Paired final-season intervals"
        ),
        "",
        (
            "| Candidate | Paired week-cluster 95% CI for score-MAE delta vs baseline |"
            if provisional_2026
            else "| Candidate | Overall 95% CI for score-MAE delta vs baseline |"
        ),
        "|---|---:|",
    ])
    for candidate_id, label in CANDIDATE_LABELS.items():
        if candidate_id == BASELINE_ID:
            continue
        overall = bootstrap[candidate_id][0]
        lines.append(
            f"| {label} | [{overall['ci_low']:.4f}, {overall['ci_high']:.4f}] |"
        )
    lines.extend([
        "",
        "## Recommendation",
        "",
        (
            (
                "Treat this as a provisional out-of-time signal only. Do not tune "
                "or promote using this incomplete season; keep production unchanged "
                "and repeat the locked comparison after the 2026 season and data "
                "reconciliation are complete."
            )
            if provisional_2026
            else (
                "Treat the cumulative stack as a screening candidate only. Prefer "
                "the simplest recency-plus-pace combination unless stabilized "
                "role/site features add repeatable value across seasons and neutral "
                "games. Keep Stage 5 and Stage 4B as ablations; do not select a "
                "winner from the previously examined 2025 block."
            )
        ),
        "",
        "## Reproduction",
        "",
        (
            "- `python -m src.cfb_model.train.final_combined_2026_experiment`"
            if provisional_2026
            else "- `python -m src.cfb_model.train.final_combined_experiment`"
        ),
        "- `python -m pytest`",
        "",
    ])
    (output_dir / "experiment_report.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--validation-seasons", type=int, nargs="+")
    args = parser.parse_args()
    print(json.dumps(run_experiment(
        args.db_path,
        args.output_dir,
        validation_seasons=args.validation_seasons,
    ), indent=2))


if __name__ == "__main__":
    main()
