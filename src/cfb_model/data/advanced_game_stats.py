"""Fetch, cache, normalize, and validate CFBD game-level advanced statistics."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import unicodedata

import duckdb
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from src.cfb_model.data.warehouse import DEFAULT_DB_PATH, ensure_warehouse

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_ROOT = PROJECT_ROOT / "data" / "raw" / "cfbd" / "advanced_game"
ENDPOINT = "https://api.collegefootballdata.com/stats/game/advanced"
SEASON_TYPE = "both"
EXCLUDE_GARBAGE_TIME = False
SCHEMA_VERSION = 1
BASE_COLUMNS = [
    "game_id",
    "season",
    "season_type",
    "week",
    "team",
    "opponent",
    "source_game_date",
    "team_id",
    "opponent_id",
]


def _cache_paths(season: int, cache_root: Path) -> dict[str, Path]:
    directory = cache_root / f"year={season}"
    return {
        "directory": directory,
        "response": directory / "response.json",
        "normalized": directory / "normalized.parquet",
        "manifest": directory / "manifest.json",
        "reconciliation": directory / "reconciliation.json",
    }


def _normalize_team_name(value: Any) -> str:
    decomposed = unicodedata.normalize("NFKD", str(value))
    return "".join(
        character.lower()
        for character in decomposed
        if character.isalnum() and not unicodedata.combining(character)
    )


def _flatten_statistics(value: Any, prefix: str, output: dict[str, Any]) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Expected numeric CFBD stat at {prefix}, got {value!r}.")
        if not np.isfinite(value):
            raise ValueError(f"Non-finite CFBD stat at {prefix}.")
        output[prefix] = value
        return
    for key, child in value.items():
        _flatten_statistics(child, f"{prefix}_{key}", output)


def _validate_normalized_stats(stats: pd.DataFrame, season: int) -> pd.DataFrame:
    required = {"game_id", "season", "season_type", "week", "team", "opponent"}
    missing = required - set(stats.columns)
    if missing:
        raise ValueError(f"Normalized CFBD stats are missing columns: {sorted(missing)}")
    if stats.empty:
        return stats
    if stats["season"].isna().any() or (stats["season"] != season).any():
        raise ValueError(f"CFBD stats contain a season other than {season}.")
    if stats["season_type"].isna().any() or stats["week"].isna().any():
        raise ValueError("CFBD stats require season type and week for every record.")
    if stats[["team", "opponent"]].isna().any().any():
        raise ValueError("CFBD stats require team and opponent names for every record.")
    if (stats["team"].map(_normalize_team_name) == stats["opponent"].map(_normalize_team_name)).any():
        raise ValueError("CFBD stat record lists the same team as its opponent.")

    identified = stats[stats["game_id"].notna()]
    if identified.duplicated(["game_id", "team"]).any():
        raise ValueError("CFBD stats contain duplicate (game_id, team) records.")
    stats = stats.copy()
    stats["_validation_key"] = stats.apply(
        lambda row: (
            ("id", _canonical_id(row["game_id"]))
            if _canonical_id(row["game_id"]) is not None
            else (
                "pair",
                int(row["season"]),
                str(row["season_type"]),
                int(row["week"]),
                tuple(sorted((
                    _normalize_team_name(row["team"]),
                    _normalize_team_name(row["opponent"]),
                ))),
                None if pd.isna(row["source_game_date"]) else str(row["source_game_date"]),
            )
        ),
        axis=1,
    )
    game_groups = stats.groupby("_validation_key", sort=False)
    for game_key, group in game_groups:
        if len(group) != 2:
            raise ValueError(
                f"CFBD game {game_key!r} has {len(group)} team records; expected two."
            )
        first, second = group.iloc[0], group.iloc[1]
        if (
            _normalize_team_name(first["team"])
            != _normalize_team_name(second["opponent"])
            or _normalize_team_name(first["opponent"])
            != _normalize_team_name(second["team"])
        ):
            raise ValueError(f"CFBD teams are not reciprocal for game {game_key!r}.")
    stats = stats.drop(columns=["_validation_key"])

    metric_columns = [
        column for column in stats.columns
        if column.startswith(("offense_", "defense_"))
    ]
    for column in metric_columns:
        numeric = pd.to_numeric(stats[column], errors="coerce")
        invalid = stats[column].notna() & numeric.isna()
        if invalid.any() or not np.isfinite(numeric.dropna()).all():
            raise ValueError(f"CFBD stat column {column} contains invalid numeric values.")
        stats[column] = numeric
    return stats.reset_index(drop=True)


def normalize_advanced_game_stats(
    payload: Any, requested_season: int
) -> pd.DataFrame:
    """Flatten the documented response while preserving every numeric stat."""
    if not isinstance(payload, list):
        raise ValueError("CFBD advanced-game response must be a JSON array.")

    rows = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"CFBD advanced-game row {index} is not an object.")
        row = {
            "game_id": item.get("gameId"),
            "season": item.get("season", requested_season),
            "season_type": item.get("seasonType"),
            "week": item.get("week"),
            "team": item.get("team"),
            "opponent": item.get("opponent"),
            "source_game_date": item.get("gameDate", item.get("date")),
            "team_id": item.get("teamId"),
            "opponent_id": item.get("opponentId"),
        }
        for side in ("offense", "defense"):
            _flatten_statistics(item.get(side), side, row)
        rows.append(row)

    stats = pd.DataFrame(rows)
    for column in BASE_COLUMNS:
        if column not in stats:
            stats[column] = pd.Series(dtype="object")
    if stats.empty:
        return stats[BASE_COLUMNS]

    for column in ("game_id", "season", "week"):
        raw = stats[column]
        numeric = pd.to_numeric(raw, errors="coerce")
        if (raw.notna() & numeric.isna()).any():
            raise ValueError(f"CFBD field {column} must be an integer.")
        if (numeric.dropna() % 1 != 0).any():
            raise ValueError(f"CFBD field {column} must be an integer.")
        stats[column] = pd.array(numeric, dtype="Int64")
    stats["season_type"] = stats["season_type"].astype("string").str.lower()
    stats["team"] = stats["team"].astype("string")
    stats["opponent"] = stats["opponent"].astype("string")
    stats["source_game_date"] = stats["source_game_date"].astype("string")
    for column in ("team_id", "opponent_id"):
        stats[column] = stats[column].astype("string")
    metric_columns = sorted(
        column for column in stats.columns
        if column.startswith(("offense_", "defense_"))
    )
    stats = stats[BASE_COLUMNS + metric_columns]
    return _validate_normalized_stats(stats, requested_season)


def _canonical_id(value: Any) -> str | None:
    if pd.isna(value):
        return None
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.notna(numeric) and float(numeric).is_integer():
        return str(int(numeric))
    return str(value).strip()


def reconcile_advanced_game_stats(
    stats: pd.DataFrame, canonical_games: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Map CFBD stat rows to canonical completed games without many-to-many joins."""
    required_stats = {"game_id", "season", "week", "team", "opponent"}
    required_games = {
        "game_id", "season", "week", "game_date", "home_team", "away_team",
    }
    if not required_stats <= set(stats.columns):
        raise ValueError(f"Stats need columns: {sorted(required_stats)}")
    if not required_games <= set(canonical_games.columns):
        raise ValueError(f"Games need columns: {sorted(required_games)}")

    games = canonical_games.copy()
    if {"home_score", "away_score"} <= set(games.columns):
        games = games[games["home_score"].notna() & games["away_score"].notna()]
    elif "completed" in games:
        games = games[games["completed"].fillna(False).astype(bool)]
    if games["game_id"].duplicated().any():
        raise ValueError("Canonical games contain duplicate game IDs.")

    records_total = len(stats)
    if {"home_is_fbs", "away_is_fbs"} <= set(games.columns):
        fbs_teams = set(
            games.loc[games["home_is_fbs"].fillna(False), "home_team"].astype(str)
        ) | set(
            games.loc[games["away_is_fbs"].fillna(False), "away_team"].astype(str)
        )
        stats = stats[
            stats["team"].astype(str).isin(fbs_teams)
            | stats["opponent"].astype(str).isin(fbs_teams)
        ].copy()

    game_by_id = {
        _canonical_id(row["game_id"]): row
        for _, row in games.iterrows()
        if _canonical_id(row["game_id"]) is not None
    }

    def matches_pair(stat: pd.Series, game: pd.Series) -> bool:
        stat_team = _normalize_team_name(stat["team"])
        stat_opponent = _normalize_team_name(stat["opponent"])
        return (
            stat_team == _normalize_team_name(game["home_team"])
            and stat_opponent == _normalize_team_name(game["away_team"])
        ) or (
            stat_team == _normalize_team_name(game["away_team"])
            and stat_opponent == _normalize_team_name(game["home_team"])
        )

    def matches_id_identity(stat: pd.Series, game: pd.Series) -> bool:
        if matches_pair(stat, game):
            return True
        if {"home_is_fbs", "away_is_fbs"} <= set(game.index):
            fbs_sides = []
            if bool(game["home_is_fbs"]):
                fbs_sides.append(game["home_team"])
            if bool(game["away_is_fbs"]):
                fbs_sides.append(game["away_team"])
            observed = {
                _normalize_team_name(stat["team"]),
                _normalize_team_name(stat["opponent"]),
            }
            return bool(observed & {_normalize_team_name(team) for team in fbs_sides})
        return False

    matched_rows = []
    unmatched = []
    for _, stat in stats.iterrows():
        season_games = games[games["season"] == stat["season"]]
        week_games = season_games[season_games["week"] == stat["week"]]
        source_id = _canonical_id(stat["game_id"])
        id_game = game_by_id.get(source_id)
        if (
            id_game is not None
            and id_game["season"] == stat["season"]
            and id_game["week"] == stat["week"]
            and matches_id_identity(stat, id_game)
        ):
            game = id_game
            match_method = "game_id"
        else:
            candidates = [
                game for _, game in week_games.iterrows()
                if matches_pair(stat, game)
            ]
            if pd.notna(stat.get("source_game_date")) and candidates:
                source_date = pd.to_datetime(
                    stat["source_game_date"], utc=True, format="mixed"
                ).date()
                candidates = [
                    game for game in candidates
                    if pd.to_datetime(
                        game["game_date"], utc=True, format="mixed"
                    ).date() == source_date
                ]
            if len(candidates) > 1:
                raise ValueError(
                    "Advanced stats have multiple canonical game matches for "
                    f"{stat['season']} week {stat['week']} "
                    f"{stat['team']} vs {stat['opponent']}."
                )
            if not candidates:
                unmatched.append({
                    "game_id": source_id,
                    "season": int(stat["season"]),
                    "week": int(stat["week"]),
                    "team": str(stat["team"]),
                    "opponent": str(stat["opponent"]),
                    "reason": "no_unique_canonical_game_match",
                })
                continue
            game = candidates[0]
            match_method = "season_week_team_opponent"
        row = stat.to_dict()
        row["canonical_game_id"] = game["game_id"]
        row["game_date"] = game["game_date"]
        row["match_method"] = match_method
        matched_rows.append(row)

    matched = pd.DataFrame(matched_rows, columns=[
        *stats.columns, "canonical_game_id", "game_date", "match_method"
    ])
    if not matched.empty:
        for game_id, group in matched.groupby("canonical_game_id", sort=False):
            if len(group) != 2:
                raise ValueError(
                    f"Canonical game {game_id!r} has {len(group)} matched stat rows."
                )
            first, second = group.iloc[0], group.iloc[1]
            if (
                _normalize_team_name(first["team"])
                != _normalize_team_name(second["opponent"])
                or _normalize_team_name(first["opponent"])
                != _normalize_team_name(second["team"])
            ):
                raise ValueError(
                    f"Matched stat teams are not reciprocal for game {game_id!r}."
                )

    matched_ids = set(matched["canonical_game_id"]) if not matched.empty else set()
    report = {
        "records_total": int(records_total),
        "records_in_scope": int(len(stats)),
        "records_out_of_scope": int(records_total - len(stats)),
        "records_matched": int(len(matched)),
        "match_rate": float(len(matched) / len(stats)) if len(stats) else 0.0,
        "games_matched": int(len(matched_ids)),
        "canonical_completed_games": int(len(games)),
        "canonical_games_without_stats": int(len(set(games["game_id"]) - matched_ids)),
        "unmatched_records": unmatched,
    }
    return matched, report


def save_reconciliation_report(
    season: int,
    report: dict[str, Any],
    cache_root: Path = DEFAULT_CACHE_ROOT,
) -> Path:
    path = _cache_paths(season, cache_root)["reconciliation"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _source_commit_sha() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _write_cache(
    season: int,
    response_bytes: bytes,
    stats: pd.DataFrame,
    cache_root: Path,
) -> dict[str, Any]:
    paths = _cache_paths(season, cache_root)
    paths["directory"].mkdir(parents=True, exist_ok=True)
    response_tmp = paths["response"].with_suffix(".json.tmp")
    response_tmp.write_bytes(response_bytes)
    response_tmp.replace(paths["response"])

    normalized_tmp = paths["normalized"].with_suffix(".parquet.tmp")
    stats.to_parquet(normalized_tmp, index=False)
    normalized_tmp.replace(paths["normalized"])

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "season": season,
        "endpoint": ENDPOINT,
        "request_parameters": {
            "year": season,
            "seasonType": SEASON_TYPE,
            "excludeGarbageTime": EXCLUDE_GARBAGE_TIME,
        },
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source_commit_sha": _source_commit_sha(),
        "ingestion_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "response_sha256": hashlib.sha256(response_bytes).hexdigest(),
        "normalized_sha256": hashlib.sha256(paths["normalized"].read_bytes()).hexdigest(),
        "row_count": int(len(stats)),
        "exclude_garbage_time": EXCLUDE_GARBAGE_TIME,
        "raw_payload": paths["response"].name,
        "normalized_table": paths["normalized"].name,
    }
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return manifest


def refresh_advanced_game_stats(
    season: int,
    *,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    api_key: str | None = None,
    canonical_games: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch one season, validate it, and save the exact payload plus normalized data."""
    if api_key is None:
        load_dotenv(PROJECT_ROOT / ".env")
        api_key = os.environ.get("CFBD_API_KEY")
    if not api_key:
        raise RuntimeError("CFBD_API_KEY is required to refresh advanced game stats.")

    response = requests.get(
        ENDPOINT,
        headers={"Authorization": f"Bearer {api_key}"},
        params={
            "year": season,
            "seasonType": SEASON_TYPE,
            "excludeGarbageTime": EXCLUDE_GARBAGE_TIME,
        },
        timeout=60,
    )
    response.raise_for_status()
    response_bytes = response.content
    payload = json.loads(response_bytes)
    stats = normalize_advanced_game_stats(payload, season)
    report = None
    if canonical_games is not None:
        _, report = reconcile_advanced_game_stats(stats, canonical_games)
    manifest = _write_cache(season, response_bytes, stats, cache_root)
    if report is not None:
        save_reconciliation_report(season, report, cache_root)
        manifest["reconciliation"] = report
        _cache_paths(season, cache_root)["manifest"].write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    return stats, manifest


def load_cached_advanced_game_stats(
    season: int, cache_root: Path = DEFAULT_CACHE_ROOT
) -> pd.DataFrame:
    """Load and verify normalized data without making a network request."""
    paths = _cache_paths(season, cache_root)
    if not all(paths[key].exists() for key in ("response", "normalized", "manifest")):
        raise FileNotFoundError(
            f"No complete CFBD advanced-game cache for {season}; refresh it first."
        )
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    raw_bytes = paths["response"].read_bytes()
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported CFBD cache schema for {season}.")
    if manifest.get("season") != season:
        raise ValueError(f"CFBD cache manifest has the wrong season for {season}.")
    if manifest.get("exclude_garbage_time") != EXCLUDE_GARBAGE_TIME:
        raise ValueError("CFBD cache uses a different garbage-time policy.")
    if hashlib.sha256(raw_bytes).hexdigest() != manifest.get("response_sha256"):
        raise ValueError(f"CFBD raw response hash mismatch for {season}.")
    normalized_bytes = paths["normalized"].read_bytes()
    if hashlib.sha256(normalized_bytes).hexdigest() != manifest.get("normalized_sha256"):
        raise ValueError(f"CFBD normalized cache hash mismatch for {season}.")
    stats = pd.read_parquet(paths["normalized"])
    stats = _validate_normalized_stats(stats, season)
    if len(stats) != manifest.get("row_count"):
        raise ValueError(f"CFBD normalized row count mismatch for {season}.")
    return stats


def load_canonical_games(
    season: int, db_path: Path = DEFAULT_DB_PATH
) -> pd.DataFrame:
    """Load completed project games involving an FBS team for reconciliation."""
    with duckdb.connect(str(db_path), read_only=True) as connection:
        return connection.execute(
            """
            SELECT game_id, season, week, game_date, home_team, away_team,
                   home_score, away_score, home_is_fbs, away_is_fbs
            FROM stg_games_raw
            WHERE season = ?
              AND (home_is_fbs = TRUE OR away_is_fbs = TRUE)
              AND home_score IS NOT NULL AND away_score IS NOT NULL
            ORDER BY game_date, game_id
            """,
            [season],
        ).fetchdf()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--refresh", action="store_true", help="Fetch and replace this season's cache.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    args = parser.parse_args()

    if args.refresh:
        ensure_warehouse(args.db_path)
        games = load_canonical_games(args.season, args.db_path)
        stats, manifest = refresh_advanced_game_stats(
            args.season, canonical_games=games if not games.empty else None
        )
    else:
        stats = load_cached_advanced_game_stats(args.season)
        paths = _cache_paths(args.season, DEFAULT_CACHE_ROOT)
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        if args.db_path.exists():
            games = load_canonical_games(args.season, args.db_path)
            if not games.empty:
                _, report = reconcile_advanced_game_stats(stats, games)
                save_reconciliation_report(args.season, report)
                manifest["reconciliation"] = report
                paths["manifest"].write_text(
                    json.dumps(manifest, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8",
                )
    print(json.dumps({
        "season": args.season,
        "rows": len(stats),
        "cache_manifest": str(_cache_paths(args.season, DEFAULT_CACHE_ROOT)["manifest"]),
        "reconciliation": manifest.get("reconciliation", "not available in local game history"),
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
