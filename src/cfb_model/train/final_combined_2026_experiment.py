"""Run the fixed combined-model comparison on the current 2026 game snapshot."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import duckdb
import pandas as pd

from src.cfb_model.data.advanced_game_stats import (
    DEFAULT_CACHE_ROOT,
    load_cached_advanced_game_stats,
    reconcile_advanced_game_stats,
)
from src.cfb_model.data.warehouse import ensure_warehouse
from src.cfb_model.features.cfbd_game_features import (
    BASE_GAME_STAT_COLUMNS,
    SPLIT_STAT_COLUMNS,
)
from src.cfb_model.train.final_combined_experiment import (
    run_experiment,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "baselines" / "final_combined_2026_provisional"
)
SOURCE_DIR = OUTPUT_DIR / "source_data"
GAME_COLUMNS = [
    "game_id",
    "season",
    "week",
    "game_date",
    "home_team",
    "away_team",
    "home_team_id",
    "away_team_id",
    "home_score",
    "away_score",
    "venue_id",
    "status",
    "neutral_site",
    "home_is_fbs",
    "away_is_fbs",
]


def _read_games_snapshot() -> tuple[list[dict], dict]:
    games_path = SOURCE_DIR / "games_2026.json"
    manifest_path = SOURCE_DIR / "games_2026_manifest.json"
    games_bytes = games_path.read_bytes()
    manifest = json.loads(manifest_path.read_text())
    if hashlib.sha256(games_bytes).hexdigest() != manifest["payload_sha256"]:
        raise ValueError("2026 games snapshot hash does not match its manifest.")
    games = json.loads(games_bytes)
    if not isinstance(games, list):
        raise ValueError("2026 games snapshot must contain a JSON array.")
    return games, manifest


def _completed_fbs_games(payload: list[dict]) -> pd.DataFrame:
    rows = []
    seen_ids = set()
    for game in payload:
        home_is_fbs = game.get("homeClassification") == "fbs"
        away_is_fbs = game.get("awayClassification") == "fbs"
        if not (home_is_fbs or away_is_fbs):
            continue
        if game.get("homePoints") is None or game.get("awayPoints") is None:
            continue
        if not game.get("startDate") or not game.get("homeTeam") or not game.get("awayTeam"):
            raise ValueError("A completed 2026 FBS game is missing its date or team names.")
        game_id = pd.to_numeric(game.get("id"), errors="coerce")
        if pd.isna(game_id) or float(game_id) % 1:
            raise ValueError("A completed 2026 FBS game has an invalid game ID.")
        game_id = int(game_id)
        if game_id in seen_ids:
            raise ValueError(f"Duplicate 2026 game ID in source snapshot: {game_id}")
        seen_ids.add(game_id)
        rows.append({
            "game_id": game_id,
            "season": 2026,
            "week": game.get("week"),
            "game_date": game["startDate"],
            "home_team": game["homeTeam"],
            "away_team": game["awayTeam"],
            "home_team_id": game.get("homeId"),
            "away_team_id": game.get("awayId"),
            "home_score": game["homePoints"],
            "away_score": game["awayPoints"],
            "venue_id": game.get("venueId"),
            "status": "completed",
            "neutral_site": bool(game.get("neutralSite", False)),
            "home_is_fbs": home_is_fbs,
            "away_is_fbs": away_is_fbs,
        })
    games = pd.DataFrame(rows, columns=GAME_COLUMNS)
    if games.empty:
        raise ValueError("The 2026 snapshot contains no completed games involving FBS teams.")
    if games["week"].isna().any() or games["game_date"].isna().any():
        raise ValueError("Completed 2026 FBS games require week and game date.")
    return games


def _insert_frame(
    connection: duckdb.DuckDBPyConnection,
    table: str,
    frame: pd.DataFrame,
    columns: list[str],
    view_name: str,
) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{table} input is missing columns: {sorted(missing)}")
    connection.register(view_name, frame[columns])
    column_sql = ", ".join(f'"{column}"' for column in columns)
    connection.execute(
        f'INSERT INTO "{table}" ({column_sql}) '
        f"SELECT {column_sql} FROM {view_name}"
    )
    connection.unregister(view_name)


def _build_snapshot_warehouse(
    db_path: Path,
    games: pd.DataFrame,
    advanced: pd.DataFrame,
    roster_priors: pd.DataFrame,
) -> dict:
    ensure_warehouse(db_path)
    with duckdb.connect(str(db_path)) as connection:
        existing_ids = connection.execute(
            "SELECT game_id FROM stg_games_raw WHERE season = 2026"
        ).fetchdf()["game_id"]
        if not existing_ids.empty:
            raise ValueError("The base warehouse unexpectedly already contains 2026 games.")

        _insert_frame(
            connection,
            "stg_games_raw",
            games,
            GAME_COLUMNS,
            "current_games",
        )
        base_columns = [
            "game_id", "season", "week", "team", "opponent",
            *BASE_GAME_STAT_COLUMNS,
        ]
        split_columns = [
            "game_id", "season", "week", "team", "opponent",
            *SPLIT_STAT_COLUMNS,
        ]
        _insert_frame(
            connection,
            "stg_team_game_stats_raw",
            advanced,
            base_columns,
            "current_overall_stats",
        )
        _insert_frame(
            connection,
            "stg_pbp_team_game_raw",
            advanced,
            split_columns,
            "current_split_stats",
        )
        _insert_frame(
            connection,
            "stg_roster_priors_raw",
            roster_priors,
            [
                "season", "team", "team_talent",
                "recruiting_rank", "recruiting_points",
            ],
            "current_roster_priors",
        )
        fbs_game_count = connection.execute(
            """
            SELECT COUNT(*) FROM stg_games_raw
            WHERE season = 2026
              AND home_is_fbs = TRUE AND away_is_fbs = TRUE
              AND home_score IS NOT NULL AND away_score IS NOT NULL
            """
        ).fetchone()[0]
        schedule_game_count = connection.execute(
            """
            SELECT COUNT(*) FROM stg_games_raw
            WHERE season = 2026
              AND (home_is_fbs = TRUE OR away_is_fbs = TRUE)
              AND home_score IS NOT NULL AND away_score IS NOT NULL
            """
        ).fetchone()[0]
    return {
        "completed_fbs_vs_fbs_games": int(fbs_game_count),
        "completed_games_involving_fbs": int(schedule_game_count),
        "advanced_game_stat_rows": int(len(advanced)),
        "roster_prior_rows": int(len(roster_priors)),
    }


def run_2026_experiment(output_dir: Path = OUTPUT_DIR) -> dict:
    payload, game_snapshot_manifest = _read_games_snapshot()
    current_games = _completed_fbs_games(payload)
    expected_games = int(
        game_snapshot_manifest["completed_fbs_vs_fbs_games"]
    )
    actual_fbs_games = int(
        (current_games["home_is_fbs"] & current_games["away_is_fbs"]).sum()
    )
    if actual_fbs_games != expected_games:
        raise ValueError(
            f"Game snapshot has {actual_fbs_games} completed FBS-vs-FBS games; "
            f"manifest records {expected_games}."
        )

    cache_manifest_path = (
        DEFAULT_CACHE_ROOT / "year=2026" / "manifest.json"
    )
    cache_manifest = json.loads(cache_manifest_path.read_text())
    advanced = load_cached_advanced_game_stats(2026)
    if not advanced["season_type"].eq("regular").all():
        raise ValueError("The 2026 advanced-game cache contains non-regular games.")
    matched_stats, reconciliation = reconcile_advanced_game_stats(
        advanced,
        current_games,
    )
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    (SOURCE_DIR / "advanced_stats_reconciliation_2026.json").write_text(
        json.dumps(reconciliation, indent=2) + "\n"
    )
    if reconciliation["records_matched"] != len(matched_stats):
        raise ValueError("Advanced-stat reconciliation count is inconsistent.")

    roster_path = SOURCE_DIR / "roster_priors_2026.csv"
    roster_manifest_path = SOURCE_DIR / "roster_priors_2026_manifest.json"
    roster_bytes = roster_path.read_bytes()
    roster_manifest = json.loads(roster_manifest_path.read_text())
    if hashlib.sha256(roster_bytes).hexdigest() != roster_manifest["sha256"]:
        raise ValueError("2026 roster-prior snapshot hash does not match its manifest.")
    roster_priors = pd.read_csv(roster_path)
    if not roster_priors["season"].eq(2026).all():
        raise ValueError("Roster-prior snapshot contains a season other than 2026.")

    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_info = {
        "games": {
            "path": "source_data/games_2026.json",
            **{
                key: game_snapshot_manifest[key]
                for key in (
                    "fetched_at_utc",
                    "game_rows",
                    "completed_fbs_vs_fbs_games",
                    "latest_completed_game_date",
                    "payload_sha256",
                )
            },
        },
        "roster_priors": {
            "path": "source_data/roster_priors_2026.csv",
            **roster_manifest,
        },
        "advanced_game_stats": {
            "path": str(cache_manifest_path.relative_to(PROJECT_ROOT)),
            "endpoint": cache_manifest["endpoint"],
            "fetched_at": cache_manifest["fetched_at"],
            "row_count": cache_manifest["row_count"],
            "response_sha256": cache_manifest["response_sha256"],
            "normalized_sha256": cache_manifest["normalized_sha256"],
            "reconciliation": reconciliation,
        },
    }

    with tempfile.TemporaryDirectory(prefix="cfb-2026-comparison-") as temp_dir:
        db_path = Path(temp_dir) / "warehouse_2026.duckdb"
        warehouse_counts = _build_snapshot_warehouse(
            db_path, current_games, advanced, roster_priors
        )
        if warehouse_counts["completed_fbs_vs_fbs_games"] != expected_games:
            raise ValueError("Temporary warehouse game count does not match the snapshot.")
        snapshot_info["temporary_warehouse"] = warehouse_counts
        manifest = run_experiment(
            db_path=db_path,
            output_dir=output_dir,
            validation_seasons=[2026],
            source_metadata=snapshot_info,
        )

    return {
        "experiment_id": manifest["experiment_id"],
        "validation_seasons": manifest["validation_seasons"],
        "games": manifest["game_count"],
        "metrics_overall": manifest["metrics_overall"],
        "paired_week_cluster_bootstrap_ci": manifest[
            "paired_week_cluster_bootstrap_ci"
        ],
        "promotion_decision": manifest["promotion_decision"],
        "output_dir": str(output_dir),
    }


def main() -> None:
    print(json.dumps(run_2026_experiment(), indent=2))


if __name__ == "__main__":
    main()
