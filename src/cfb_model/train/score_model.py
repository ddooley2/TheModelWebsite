"""Train, evaluate, and run a score-first college football model."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import logging
import os
import re
import subprocess
import unicodedata
from collections import defaultdict, deque
from datetime import date, datetime, timezone
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.multioutput import MultiOutputRegressor

from src.cfb_model.features.advanced import ADVANCED_METRICS, FEATURE_COLUMNS as ADVANCED_COLUMNS, build_advanced_features
from src.cfb_model.features.opponent_adjusted import build_opponent_adjusted_features
from src.cfb_model.data.warehouse import ensure_warehouse
from src.cfb_model.features.preseason import build_preseason_features


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DB_PATH = PROJECT_ROOT / "db" / "warehouse.duckdb"
ESPN_SCOREBOARD = (
    "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard"
)
FEATURE_COLUMNS = [
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_points_for_recent",
    "home_points_against_recent",
    "away_points_for_recent",
    "away_points_against_recent",
    "home_points_for_season",
    "home_points_against_season",
    "away_points_for_season",
    "away_points_against_season",
    "home_win_pct_season",
    "away_win_pct_season",
    "home_games_played",
    "away_games_played",
    "home_rest_days",
    "away_rest_days",
    "week",
    "neutral_site",
]
RATING_COLUMNS = [
    "home_offense_rating",
    "home_defense_rating",
    "away_offense_rating",
    "away_defense_rating",
    "offense_rating_diff",
    "defense_rating_diff",
]
PRESEASON_COLUMNS = [
    "home_preseason_elo",
    "away_preseason_elo",
    "preseason_elo_diff",
    "home_team_talent",
    "away_team_talent",
    "home_recruiting_rank",
    "away_recruiting_rank",
    "home_recruiting_points",
    "away_recruiting_points",
]
MODEL_COLUMNS = FEATURE_COLUMNS + RATING_COLUMNS + PRESEASON_COLUMNS + ADVANCED_COLUMNS
MARKET_FEATURES_ENABLED = False
FORBIDDEN_FEATURE_TERMS = (
    "line", "spread", "overunder", "total", "odds", "implied", "vegas",
    "market", "book", "consensus",
)
DISPLAY_ADVANCED_COLUMNS = [
    f"{side}_display_{metric}"
    for metric in ADVANCED_METRICS
    for side in ("home", "away")
]
ADVANCED_RANGE_COLUMNS = [
    f"advanced_{metric}_{bound}"
    for metric in ADVANCED_METRICS
    for bound in ("min", "max")
]
TARGET_COLUMNS = ["home_score", "away_score"]
CFBD_API = "https://api.collegefootballdata.com"
FIRST_TRAIN_SEASON = 2013
BACKTEST_SEASONS = [2021, 2022, 2023, 2024, 2025]
ESPN_ABBREVIATION_ALIASES = {
    "APP": "App State",
    "CONN": "UConn",
    "HAW": "Hawai'i",
    "M-OH": "Miami (OH)",
    "SJSU": "San José State",
    "UL": "Louisiana",
    "ULM": "UL Monroe",
    "UMASS": "Massachusetts",
}
LOGGER = logging.getLogger(__name__)


def _normalize_team_name(name: str) -> str:
    decomposed = unicodedata.normalize("NFKD", name)
    ascii_name = decomposed.encode("ascii", "ignore").decode("ascii")
    return "".join(character for character in ascii_name.lower() if character.isalnum())


def validate_market_free_feature_columns(columns: list[str] | pd.Index) -> None:
    """Reject feature names that could encode betting-market information."""
    forbidden = [
        column for column in columns
        if any(term in re.sub(r"[^a-z0-9]", "", column.lower()) for term in FORBIDDEN_FEATURE_TERMS)
    ]
    if forbidden:
        raise ValueError(f"Betting-market feature columns are forbidden: {forbidden}")


def _team_lookup(teams: pd.DataFrame) -> dict[str, dict]:
    lookup = {}
    for row in teams.to_dict("records"):
        canonical = row.get("school") or row.get("team_name")
        if not canonical:
            continue
        mapping = {
            "team_id": row.get("team_id", row.get("id")),
            "team_name": canonical,
        }
        for field in ("school", "team_name", "abbreviation", "alt_name1", "alt_name2", "alt_name3"):
            alias = row.get(field)
            if alias:
                lookup[_normalize_team_name(str(alias))] = mapping
    for abbreviation, team in ESPN_ABBREVIATION_ALIASES.items():
        mapping = lookup.get(_normalize_team_name(team))
        if mapping is not None:
            lookup[_normalize_team_name(abbreviation)] = mapping
    return lookup


def _build_espn_team_mapping(
    season: int, fbs_teams: pd.DataFrame, provider_teams: list[dict]
) -> pd.DataFrame:
    """Build and validate the season-scoped ESPN ID to CFBD team-ID crosswalk."""
    team_lookup = _team_lookup(fbs_teams)
    rows = []
    for provider_team in provider_teams:
        abbreviation = str(provider_team.get("abbreviation", "")).upper()
        provider_name = (
            provider_team.get("location")
            or provider_team.get("displayName")
            or provider_team.get("name")
            or provider_team.get("shortDisplayName")
            or ""
        )
        lookup_name = ESPN_ABBREVIATION_ALIASES.get(abbreviation, provider_name)
        match = team_lookup.get(_normalize_team_name(lookup_name))
        if match is None:
            continue
        rows.append({
            "season": season,
            "provider": "ESPN",
            "provider_team_id": str(provider_team.get("id", "")),
            "provider_team_name": provider_name,
            "canonical_team_id": match["team_id"],
            "canonical_team_name": match["team_name"],
        })

    mapping = pd.DataFrame(rows).drop_duplicates()
    if (
        mapping.empty
        or mapping["provider_team_id"].duplicated().any()
        or mapping["canonical_team_id"].duplicated().any()
    ):
        raise ValueError(f"ESPN-to-CFBD team mapping is not one-to-one for {season}.")
    expected_ids = set(fbs_teams["team_id"])
    mapped_ids = set(mapping["canonical_team_id"])
    if expected_ids != mapped_ids:
        missing_ids = sorted(expected_ids - mapped_ids)
        missing_teams = sorted(
            fbs_teams.loc[fbs_teams["team_id"].isin(missing_ids), "school"]
            .astype(str)
            .tolist()
        )
        raise ValueError(
            f"ESPN team mapping is missing season {season} CFBD teams: "
            f"{missing_teams} (IDs: {missing_ids})"
        )
    LOGGER.info(
        "Validated %s ESPN-to-CFBD FBS team mappings for %s",
        len(mapping), season,
    )
    return mapping


def load_historical_games(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Load completed FBS-vs-FBS games used for model statistics and targets."""
    with duckdb.connect(str(db_path), read_only=True) as connection:
        return connection.execute(
            """
            SELECT game_id, season, week, game_date, home_team, away_team,
                   home_score, away_score, neutral_site
            FROM stg_games_raw
            WHERE home_is_fbs = TRUE AND away_is_fbs = TRUE
              AND home_score IS NOT NULL AND away_score IS NOT NULL
              AND game_date IS NOT NULL
            ORDER BY game_date, game_id
            """
        ).fetchdf()


def load_schedule_history(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Load completed games involving an FBS team for schedule-derived features."""
    with duckdb.connect(str(db_path), read_only=True) as connection:
        return connection.execute(
            """
            SELECT game_id, season, week, game_date, home_team, away_team,
                   home_score, away_score, neutral_site, home_is_fbs, away_is_fbs
            FROM stg_games_raw
            WHERE (home_is_fbs = TRUE OR away_is_fbs = TRUE)
              AND home_score IS NOT NULL AND away_score IS NOT NULL
              AND game_date IS NOT NULL
            ORDER BY game_date, game_id
            """
        ).fetchdf()


def load_fbs_teams(season: int, db_path: Path = DB_PATH) -> pd.DataFrame:
    """Load the FBS team IDs for this season, fetching only when not cached."""
    with duckdb.connect(str(db_path), read_only=True) as connection:
        cached = connection.execute(
            """
            SELECT DISTINCT history.team_id, lookup.team_name
            FROM stg_conference_history_raw AS history
            JOIN stg_team_lookup_raw AS lookup USING (team_id)
            WHERE history.season = ?
            ORDER BY history.team_id
            """,
            [season],
        ).fetchdf()
    if not cached.empty:
        teams = cached.rename(columns={"team_name": "school"}).assign(season=season)
    else:
        load_dotenv()
        api_key = os.environ.get("CFBD_API_KEY")
        if not api_key:
            raise RuntimeError(f"CFBD_API_KEY is required to load FBS teams for {season}.")
        response = requests.get(
            f"{CFBD_API}/teams/fbs",
            headers={"Authorization": f"Bearer {api_key}"},
            params={"year": season},
            timeout=60,
        )
        response.raise_for_status()
        teams = pd.DataFrame(
            [
                {"team_id": item.get("id"), "school": item.get("school"), "season": season}
                for item in response.json()
            ]
        )

    if teams.empty or teams["team_id"].isna().any() or teams["school"].isna().any():
        raise ValueError(f"CFBD returned an invalid FBS team list for {season}.")
    if teams["team_id"].duplicated().any() or teams["school"].duplicated().any():
        raise ValueError(f"CFBD returned duplicate FBS team IDs or names for {season}.")
    return teams


def build_pregame_features(
    target_games: pd.DataFrame,
    model_history: pd.DataFrame,
    schedule_history: pd.DataFrame,
) -> pd.DataFrame:
    """Build target features from FBS model history and all-opponent schedule history.

    Both histories are processed in UTC calendar-date batches. Targets are
    evaluated before any result from their date updates either history.
    """
    required = {
        "season", "week", "game_date", "home_team", "away_team",
        "home_score", "away_score", "neutral_site",
    }
    for label, frame in (
        ("target_games", target_games),
        ("model_history", model_history),
        ("schedule_history", schedule_history),
    ):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{label} is missing game columns: {sorted(missing)}")

    def dated(frame: pd.DataFrame, label: str) -> pd.DataFrame:
        result = frame.copy().reset_index(drop=True)
        result["_history_date"] = pd.to_datetime(
            result["game_date"], utc=True, errors="coerce", format="mixed"
        ).dt.normalize()
        if result["_history_date"].isna().any():
            raise ValueError(f"{label} contains invalid game_date values.")
        return result

    targets = dated(target_games, "target_games")
    model_games = dated(model_history, "model_history")
    schedule_games = dated(schedule_history, "schedule_history")

    def sort_history(frame: pd.DataFrame) -> pd.DataFrame:
        ordered = frame.copy()
        ordered["_sort_game_id"] = ordered.get(
            "game_id", pd.Series("", index=ordered.index)
        ).fillna("").astype(str)
        return ordered.sort_values(
            ["_history_date", "_sort_game_id", "home_team", "away_team"],
            kind="stable",
        ).drop(columns="_sort_game_id").reset_index(drop=True)

    model_games = sort_history(model_games)
    schedule_games = sort_history(schedule_games)

    state = defaultdict(
        lambda: {
            "elo": 1500.0,
            "recent_for": deque(maxlen=4),
            "recent_against": deque(maxlen=4),
            "career_for": 0.0,
            "career_against": 0.0,
            "career_games": 0,
        }
    )
    model_season_stats = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0, 0]))
    schedule_season_games = defaultdict(lambda: defaultdict(int))
    last_schedule_date: dict[str, pd.Timestamp] = {}
    features: list[dict] = [{} for _ in range(len(targets))]

    target_dates = {
        day: rows
        for day, rows in targets.groupby("_history_date", sort=True).groups.items()
    }
    model_dates = {
        day: rows
        for day, rows in model_games.groupby("_history_date", sort=True).groups.items()
    }
    schedule_dates = {
        day: rows
        for day, rows in schedule_games.groupby("_history_date", sort=True).groups.items()
    }
    all_dates = sorted(set(target_dates) | set(model_dates) | set(schedule_dates))

    def average_or_prior(total: float, count: int, career_total: float, career_count: int) -> float:
        if count:
            return total / count
        if career_count:
            return career_total / career_count
        return 0.0

    for game_date in all_dates:
        for position in target_dates.get(game_date, []):
            row = targets.iloc[position]
            home = state[row["home_team"]]
            away = state[row["away_team"]]
            home_season = model_season_stats[row["season"]][row["home_team"]]
            away_season = model_season_stats[row["season"]][row["away_team"]]
            home_schedule_games = schedule_season_games[row["season"]][row["home_team"]]
            away_schedule_games = schedule_season_games[row["season"]][row["away_team"]]

            def rest_days(team: str) -> float:
                previous = last_schedule_date.get(team)
                if previous is None:
                    return 14.0
                return float(min(max((game_date - previous).days, 0), 14))

            features[position] = {
                "home_elo": home["elo"],
                "away_elo": away["elo"],
                "elo_diff": home["elo"] - away["elo"],
                "home_points_for_recent": (
                    float(np.mean(home["recent_for"])) if home["recent_for"]
                    else average_or_prior(
                        home_season[0], home_season[2], home["career_for"], home["career_games"]
                    )
                ),
                "home_points_against_recent": (
                    float(np.mean(home["recent_against"])) if home["recent_against"]
                    else average_or_prior(
                        home_season[1], home_season[2], home["career_against"], home["career_games"]
                    )
                ),
                "away_points_for_recent": (
                    float(np.mean(away["recent_for"])) if away["recent_for"]
                    else average_or_prior(
                        away_season[0], away_season[2], away["career_for"], away["career_games"]
                    )
                ),
                "away_points_against_recent": (
                    float(np.mean(away["recent_against"])) if away["recent_against"]
                    else average_or_prior(
                        away_season[1], away_season[2], away["career_against"], away["career_games"]
                    )
                ),
                "home_points_for_season": average_or_prior(
                    home_season[0], home_season[2], home["career_for"], home["career_games"]
                ),
                "home_points_against_season": average_or_prior(
                    home_season[1], home_season[2], home["career_against"], home["career_games"]
                ),
                "away_points_for_season": average_or_prior(
                    away_season[0], away_season[2], away["career_for"], away["career_games"]
                ),
                "away_points_against_season": average_or_prior(
                    away_season[1], away_season[2], away["career_against"], away["career_games"]
                ),
                "home_win_pct_season": home_season[3] / home_season[2] if home_season[2] else 0.5,
                "away_win_pct_season": away_season[3] / away_season[2] if away_season[2] else 0.5,
                "home_games_played": home_schedule_games,
                "away_games_played": away_schedule_games,
                "home_rest_days": rest_days(row["home_team"]),
                "away_rest_days": rest_days(row["away_team"]),
                "week": float(row["week"]),
                "neutral_site": float(
                    bool(row["neutral_site"]) if pd.notna(row["neutral_site"]) else False
                ),
            }
        if game_date in model_dates:
            daily_deltas = defaultdict(float)
            for position in model_dates[game_date]:
                row = model_games.iloc[position]
                home_team, away_team = row["home_team"], row["away_team"]
                home_score = pd.to_numeric(row["home_score"], errors="coerce")
                away_score = pd.to_numeric(row["away_score"], errors="coerce")
                if (
                    pd.isna(home_team) or pd.isna(away_team)
                    or pd.isna(home_score) or pd.isna(away_score)
                    or not np.isfinite(home_score) or not np.isfinite(away_score)
                ):
                    continue
                home, away = state[home_team], state[away_team]
                margin = float(home_score - away_score)
                expected_home = 1.0 / (1.0 + 10.0 ** ((away["elo"] - home["elo"]) / 400.0))
                actual_home = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
                margin_multiplier = np.log1p(abs(margin)) * 2.2 / (
                    abs(home["elo"] - away["elo"]) * 0.001 + 2.2
                )
                change = 20.0 * margin_multiplier * (actual_home - expected_home)
                daily_deltas[home_team] += change
                daily_deltas[away_team] -= change

                home["recent_for"].append(float(home_score))
                home["recent_against"].append(float(away_score))
                away["recent_for"].append(float(away_score))
                away["recent_against"].append(float(home_score))
                home["career_for"] += float(home_score)
                home["career_against"] += float(away_score)
                home["career_games"] += 1
                away["career_for"] += float(away_score)
                away["career_against"] += float(home_score)
                away["career_games"] += 1

                home_season = model_season_stats[row["season"]][home_team]
                away_season = model_season_stats[row["season"]][away_team]
                home_season[0] += float(home_score)
                home_season[1] += float(away_score)
                home_season[2] += 1
                home_season[3] += int(margin > 0)
                away_season[0] += float(away_score)
                away_season[1] += float(home_score)
                away_season[2] += 1
                away_season[3] += int(margin < 0)
            for team, change in daily_deltas.items():
                state[team]["elo"] += change

        if game_date in schedule_dates:
            for position in schedule_dates[game_date]:
                row = schedule_games.iloc[position]
                home_team, away_team = row["home_team"], row["away_team"]
                if pd.isna(home_team) or pd.isna(away_team):
                    continue
                schedule_season_games[row["season"]][home_team] += 1
                schedule_season_games[row["season"]][away_team] += 1
                last_schedule_date[home_team] = game_date
                last_schedule_date[away_team] = game_date

    return pd.DataFrame(features, columns=FEATURE_COLUMNS)


def load_roster_priors(db_path: Path = DB_PATH, season: int | None = None) -> pd.DataFrame:
    """Load roster priors from the warehouse, adding CFBD's current-season priors."""
    with duckdb.connect(str(db_path), read_only=True) as connection:
        priors = connection.execute(
            "SELECT season, team, team_talent, recruiting_rank, recruiting_points "
            "FROM stg_roster_priors_raw"
        ).fetchdf()
    if season is None or (priors["season"] == season).any():
        return priors

    load_dotenv()
    api_key = os.environ.get("CFBD_API_KEY")
    if not api_key:
        raise RuntimeError(f"CFBD_API_KEY is required to fetch {season} roster priors.")
    headers = {"Authorization": f"Bearer {api_key}"}

    def fetch(endpoint: str) -> pd.DataFrame:
        response = requests.get(
            f"{CFBD_API}/{endpoint}", headers=headers, params={"year": season}, timeout=60
        )
        response.raise_for_status()
        return pd.DataFrame(response.json())

    talent = fetch("talent").rename(columns={"year": "season", "talent": "team_talent"})
    recruiting = fetch("recruiting/teams").rename(
        columns={"year": "season", "rank": "recruiting_rank", "points": "recruiting_points"}
    )
    current = talent.merge(
        recruiting[["season", "team", "recruiting_rank", "recruiting_points"]],
        on=["season", "team"], how="outer", validate="one_to_one",
    )
    return pd.concat([priors, current], ignore_index=True)


def load_advanced_stats(db_path: Path = DB_PATH, season: int | None = None) -> pd.DataFrame:
    """Load warehouse stats and refresh the requested season from CFBD when available."""
    columns = ["game_id", "season", "week", "team", *ADVANCED_METRICS]
    with duckdb.connect(str(db_path), read_only=True) as connection:
        stats = connection.execute(
            f"SELECT {', '.join(columns)} FROM stg_team_game_stats_raw"
        ).fetchdf()
    if season is None:
        return stats

    load_dotenv()
    api_key = os.environ.get("CFBD_API_KEY")
    if not api_key:
        if (pd.to_numeric(stats["season"], errors="coerce") == season).any():
            return stats
        raise RuntimeError(f"CFBD_API_KEY is required to fetch {season} advanced stats.")
    response = requests.get(
        f"{CFBD_API}/stats/game/advanced",
        headers={"Authorization": "Bearer " + api_key},
        params={"year": season, "seasonType": "regular"},
        timeout=60,
    )
    response.raise_for_status()
    rows = []
    for item in response.json():
        row = {
            "season": item.get("season") or season,
            "week": item.get("week"),
            "team": item.get("team"),
        }
        for side in ("offense", "defense"):
            for metric in ("ppa", "successRate", "explosiveness"):
                row[f"{side}_{metric}"] = (item.get(side) or {}).get(metric)
        rows.append(row)
    # ESPN-sourced current games have no CFBD id; they join on season/week/team.
    current = pd.DataFrame(rows, columns=columns)
    if current.empty:
        return stats
    stats = stats[pd.to_numeric(stats["season"], errors="coerce") != season]
    return pd.concat([stats, current], ignore_index=True)


def _season_to_date_advanced_stats(
    predictions: pd.DataFrame,
    advanced: pd.DataFrame,
    fbs_teams: set[str],
    season: int,
    week: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, float | None]]]:
    prior_games = advanced[
        (pd.to_numeric(advanced["season"], errors="coerce") == season)
        & (pd.to_numeric(advanced["week"], errors="coerce") < week)
        & advanced["team"].isin(fbs_teams)
    ].drop_duplicates(["game_id", "season", "week", "team"])
    team_averages = prior_games.groupby("team")[list(ADVANCED_METRICS)].mean()

    display = pd.DataFrame(index=predictions.index)
    for side in ("home", "away"):
        for metric in ADVANCED_METRICS:
            display[f"{side}_display_{metric}"] = predictions[f"{side}_team"].map(
                team_averages[metric]
            )
    ranges = {}
    for metric in ADVANCED_METRICS:
        values = team_averages[metric].dropna()
        ranges[metric] = {
            "min": float(values.min()) if not values.empty else None,
            "max": float(values.max()) if not values.empty else None,
        }
    return display, ranges


def _combine_model_history(
    target_games: pd.DataFrame, model_history: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return unique FBS model events and target row positions in that timeline."""
    targets = target_games.copy().reset_index(drop=True)
    history = model_history.copy().reset_index(drop=True)
    targets["_target_position"] = np.arange(len(targets))
    history["_target_position"] = np.nan
    combined = pd.concat([targets, history], ignore_index=True, sort=False)

    def identity(row: pd.Series) -> tuple:
        game_date = pd.to_datetime(row["game_date"], utc=True, errors="coerce", format="mixed")
        if pd.notna(game_date):
            return row["season"], game_date.normalize(), row["home_team"], row["away_team"]
        return "game_id", row.get("game_id")

    combined["_identity"] = combined.apply(identity, axis=1)
    if combined.iloc[:len(targets)]["_identity"].duplicated().any():
        raise ValueError("target_games contains duplicate game identities.")
    timeline = combined.drop_duplicates("_identity", keep="first").drop(
        columns="_identity"
    ).reset_index(drop=True)
    positions = np.flatnonzero(timeline["_target_position"].notna().to_numpy())
    return timeline.drop(columns="_target_position"), positions


def build_model_features(
    target_games: pd.DataFrame,
    roster_priors: pd.DataFrame,
    advanced: pd.DataFrame | None = None,
    *,
    model_history: pd.DataFrame,
    schedule_history: pd.DataFrame,
) -> pd.DataFrame:
    """Build row-aligned FBS target features from explicit model and schedule histories."""
    model_timeline, target_positions = _combine_model_history(target_games, model_history)
    pregame = build_pregame_features(target_games, model_history, schedule_history)
    ratings = build_opponent_adjusted_features(model_timeline).iloc[target_positions].reset_index(drop=True)
    preseason = build_preseason_features(model_timeline, roster_priors).iloc[target_positions].reset_index(drop=True)
    preseason = preseason.rename(columns={
        "home_elo": "home_preseason_elo",
        "away_elo": "away_preseason_elo",
        "elo_diff": "preseason_elo_diff",
    })
    if advanced is None:
        advanced = pd.DataFrame(columns=["game_id", "season", "week", "team", *ADVANCED_METRICS])
    advanced_features = build_advanced_features(model_timeline, advanced).iloc[target_positions].reset_index(drop=True)
    features = pd.concat([pregame, ratings, preseason, advanced_features], axis=1)
    validate_market_free_feature_columns(MODEL_COLUMNS)
    return features[MODEL_COLUMNS]


def _make_model() -> MultiOutputRegressor:
    estimator = lgb.LGBMRegressor(
        objective="l1",
        n_estimators=800,
        learning_rate=0.015,
        max_depth=4,
        num_leaves=7,
        min_child_samples=80,
        reg_lambda=20.0,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    return MultiOutputRegressor(estimator)


def score_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Return score, margin, total, and derived winner metrics."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    actual_margin = actual[:, 0] - actual[:, 1]
    predicted_margin = predicted[:, 0] - predicted[:, 1]
    actual_total = actual[:, 0] + actual[:, 1]
    predicted_total = predicted[:, 0] + predicted[:, 1]
    winner = actual_margin != 0

    return {
        "score_mae": float(mean_absolute_error(actual.ravel(), predicted.ravel())),
        "score_rmse": float(np.sqrt(mean_squared_error(actual.ravel(), predicted.ravel()))),
        "score_r2": float(r2_score(actual.ravel(), predicted.ravel())),
        "home_score_mae": float(mean_absolute_error(actual[:, 0], predicted[:, 0])),
        "away_score_mae": float(mean_absolute_error(actual[:, 1], predicted[:, 1])),
        "margin_mae": float(mean_absolute_error(actual_margin, predicted_margin)),
        "total_mae": float(mean_absolute_error(actual_total, predicted_total)),
        "winner_auc": float(
            roc_auc_score((actual_margin[winner] > 0).astype(int), predicted_margin[winner])
        ),
    }


def run_walk_forward(
    games: pd.DataFrame,
    features: pd.DataFrame,
    val_seasons: list[int] = BACKTEST_SEASONS,
) -> pd.DataFrame:
    """Train only on earlier seasons and score each untouched validation season."""
    predictions = run_walk_forward_predictions(games, features, val_seasons)
    results = []
    for season, fold in predictions.groupby("season", sort=True):
        actual = fold[["actual_home_score", "actual_away_score"]].to_numpy(dtype=float)
        predicted = fold[["predicted_home_score", "predicted_away_score"]].to_numpy(dtype=float)
        results.append({
            "season": season,
            "games": len(fold),
            **score_metrics(actual, predicted),
            "baseline_score_mae": float(fold["baseline_score_mae"].iloc[0]),
        })
    return pd.DataFrame(results)


def run_walk_forward_predictions(
    games: pd.DataFrame,
    features: pd.DataFrame,
    val_seasons: list[int] = BACKTEST_SEASONS,
) -> pd.DataFrame:
    """Return row-level out-of-fold predictions for the direct-score baseline."""
    validate_market_free_feature_columns(MODEL_COLUMNS)
    missing = set(MODEL_COLUMNS) - set(features.columns)
    if missing:
        raise ValueError(f"Missing model feature columns: {sorted(missing)}")
    results = []
    for season in val_seasons:
        train_mask = (games["season"] < season) & (games["season"] >= FIRST_TRAIN_SEASON)
        valid_mask = games["season"] == season
        if not train_mask.any() or not valid_mask.any():
            continue

        train_y = games.loc[train_mask, TARGET_COLUMNS].to_numpy(dtype=float)
        valid_y = games.loc[valid_mask, TARGET_COLUMNS].to_numpy(dtype=float)
        model = _make_model()
        model.fit(features.loc[train_mask, MODEL_COLUMNS], train_y)
        predicted = model.predict(features.loc[valid_mask, MODEL_COLUMNS])
        baseline = np.tile(train_y.mean(axis=0), (len(valid_y), 1))
        fold = games.loc[valid_mask].copy()
        fold["actual_home_score"] = valid_y[:, 0]
        fold["actual_away_score"] = valid_y[:, 1]
        fold["predicted_home_score"] = predicted[:, 0]
        fold["predicted_away_score"] = predicted[:, 1]
        fold["home_score_absolute_error"] = np.abs(valid_y[:, 0] - predicted[:, 0])
        fold["away_score_absolute_error"] = np.abs(valid_y[:, 1] - predicted[:, 1])
        fold["score_absolute_error"] = (fold["home_score_absolute_error"] + fold["away_score_absolute_error"]) / 2
        fold["prediction_cutoff_utc"] = pd.to_datetime(
            fold["game_date"], utc=True, format="mixed"
        ).map(lambda value: value.isoformat())
        fold["baseline_score_mae"] = float(mean_absolute_error(valid_y.ravel(), baseline.ravel()))
        keep = [
            column for column in (
                "game_id", "season", "week", "game_date", "home_team", "away_team",
                "actual_home_score", "actual_away_score", "predicted_home_score",
                "predicted_away_score", "home_score_absolute_error",
                "away_score_absolute_error", "score_absolute_error",
                "prediction_cutoff_utc", "baseline_score_mae",
            ) if column in fold.columns
        ]
        results.append(fold[keep])
    if not results:
        raise ValueError("No walk-forward seasons had both training and validation games.")
    return pd.concat(results, ignore_index=True)


def write_baseline_run(
    games: pd.DataFrame,
    features: pd.DataFrame,
    run_dir: Path,
    val_seasons: list[int] = BACKTEST_SEASONS,
) -> dict:
    """Save reproducible baseline predictions, metrics, and run metadata."""
    if run_dir.exists():
        raise FileExistsError(f"Baseline run directory already exists: {run_dir}")
    predictions = run_walk_forward_predictions(games, features, val_seasons)
    run_dir.mkdir(parents=True)

    metrics = []
    for season, fold in predictions.groupby("season", sort=True):
        actual = fold[["actual_home_score", "actual_away_score"]].to_numpy(dtype=float)
        predicted = fold[["predicted_home_score", "predicted_away_score"]].to_numpy(dtype=float)
        metrics.append({"season": int(season), "games": len(fold), **score_metrics(actual, predicted)})
    metrics_frame = pd.DataFrame(metrics)
    actual = predictions[["actual_home_score", "actual_away_score"]].to_numpy(dtype=float)
    predicted = predictions[["predicted_home_score", "predicted_away_score"]].to_numpy(dtype=float)
    early = predictions[pd.to_numeric(predictions["week"], errors="coerce") <= 3]
    late = predictions[pd.to_numeric(predictions["week"], errors="coerce") >= 9]
    overall = score_metrics(actual, predicted)
    overall["early_season_score_mae"] = (
        float(early[["actual_home_score", "actual_away_score"]].sub(
            early[["predicted_home_score", "predicted_away_score"]].to_numpy()
        ).abs().to_numpy().mean()) if not early.empty else None
    )
    overall["late_season_score_mae"] = (
        float(late[["actual_home_score", "actual_away_score"]].sub(
            late[["predicted_home_score", "predicted_away_score"]].to_numpy()
        ).abs().to_numpy().mean()) if not late.empty else None
    )
    commit_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    working_tree_diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"], cwd=PROJECT_ROOT, check=True,
        capture_output=True,
    ).stdout
    working_tree_dirty = bool(subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=PROJECT_ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip())
    feature_list = list(MODEL_COLUMNS)
    feature_hash = hashlib.sha256(
        json.dumps(feature_list, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    metrics_frame.to_csv(run_dir / "metrics_by_season.csv", index=False)
    manifest = {
        "candidate_id": "direct_score_l1",
        "source_commit_sha": commit_sha,
        "source_worktree_dirty": working_tree_dirty,
        "source_worktree_diff_sha256": (
            hashlib.sha256(working_tree_diff).hexdigest() if working_tree_dirty else None
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "market_features_enabled": MARKET_FEATURES_ENABLED,
        "market_independence_check_passed": True,
        "training_cutoff_policy": "training seasons >= 2013 and strictly earlier than each validation season",
        "prediction_cutoff_policy": "point-in-time features at each game's recorded game_date; validation season excluded from fitting",
        "feature_schema_hash": feature_hash,
        "feature_list": feature_list,
        "validation_seasons": [int(season) for season in predictions["season"].drop_duplicates()],
        "held_out_game_ids": [str(game_id) for game_id in predictions.get("game_id", pd.Series(dtype=str))],
        "folds": [
            {
                "validation_season": int(season),
                "training_seasons": sorted(
                    int(value) for value in games.loc[
                        (games["season"] >= FIRST_TRAIN_SEASON)
                        & (games["season"] < season), "season"
                    ].dropna().unique()
                ),
                "held_out_game_ids": [
                    str(game_id) for game_id in fold["game_id"]
                ] if "game_id" in fold else [],
            }
            for season, fold in predictions.groupby("season", sort=True)
        ],
        "game_count": int(len(predictions)),
        "metrics_overall": overall,
        "metrics_by_season": metrics,
        "artifact_paths": {
            "predictions": "predictions.csv",
            "metrics_by_season": "metrics_by_season.csv",
        },
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return manifest


def _parse_espn_event(
    event: dict,
    season: int,
    week: int,
    team_lookup: dict[str, dict | str],
    espn_id_lookup: dict[str, dict] | None = None,
) -> dict:
    competition = event["competitions"][0]
    competitors = competition["competitors"]
    home = next(team for team in competitors if team["homeAway"] == "home")
    away = next(team for team in competitors if team["homeAway"] == "away")

    def candidate_name(lookup_name: str) -> str | None:
        alternatives = {
            key: value["team_name"] if isinstance(value, dict) else value
            for key, value in team_lookup.items()
        }
        candidates = difflib.get_close_matches(
            _normalize_team_name(lookup_name), list(alternatives), n=1, cutoff=0.9
        )
        return alternatives[candidates[0]] if candidates else None

    def mapped_team(team: dict) -> tuple[str, object, bool, str | None, str, str, str]:
        provider_team = team["team"]
        abbreviation = provider_team.get("abbreviation", "").upper()
        provider_name = (
            provider_team.get("location")
            or provider_team.get("displayName")
            or provider_team.get("name")
            or provider_team.get("shortDisplayName")
            or ""
        )
        lookup_name = ESPN_ABBREVIATION_ALIASES.get(abbreviation, provider_name)
        name_mapping = team_lookup.get(_normalize_team_name(lookup_name))
        provider_id = str(provider_team.get("id", ""))
        id_mapping = (espn_id_lookup or {}).get(provider_id)
        if id_mapping is not None:
            if name_mapping is not None and not isinstance(name_mapping, str):
                if name_mapping["team_id"] != id_mapping["canonical_team_id"]:
                    return (
                        id_mapping["canonical_team_name"],
                        id_mapping["canonical_team_id"],
                        True,
                        name_mapping["team_name"],
                        provider_id,
                        provider_name,
                        "provider_name_id_mismatch",
                    )
            elif (
                name_mapping is None
                and _normalize_team_name(lookup_name)
                != _normalize_team_name(id_mapping["provider_team_name"])
            ):
                return (
                    id_mapping["canonical_team_name"],
                    id_mapping["canonical_team_id"],
                    True,
                    candidate_name(lookup_name),
                    provider_id,
                    provider_name,
                    "provider_name_id_mismatch",
                )
            canonical_name = id_mapping["canonical_team_name"]
            team_id = id_mapping["canonical_team_id"]
            status = "mapped"
        elif isinstance(name_mapping, str):
            canonical_name, team_id = name_mapping, None
            status = "mapped_by_name"
        elif name_mapping is not None:
            canonical_name, team_id = name_mapping["team_name"], name_mapping.get("team_id")
            status = "unmapped_fbs_provider_id"
        else:
            canonical_name, team_id, status = provider_name, None, "not_fbs"
        candidate = (
            name_mapping["team_name"]
            if name_mapping is not None and not isinstance(name_mapping, str)
            else candidate_name(lookup_name)
        )
        return (
            canonical_name,
            team_id,
            status != "not_fbs",
            candidate,
            provider_id,
            provider_name,
            status,
        )

    (
        home_name, home_id, home_is_fbs, home_candidate,
        home_espn_id, home_provider_name, home_mapping_status,
    ) = mapped_team(home)
    (
        away_name, away_id, away_is_fbs, away_candidate,
        away_espn_id, away_provider_name, away_mapping_status,
    ) = mapped_team(away)

    completed = event["status"]["type"]["completed"]

    def score_for(team: dict) -> float:
        value = team.get("score")
        return float(value) if completed and value is not None and value != "" else np.nan

    return {
        "game_id": f"espn-{event['id']}",
        "season": season,
        "week": week,
        "game_date": event["date"],
        "home_team": home_name,
        "away_team": away_name,
        "home_team_id": home_id,
        "away_team_id": away_id,
        "home_is_fbs": home_is_fbs,
        "away_is_fbs": away_is_fbs,
        "home_espn_team_id": home_espn_id,
        "away_espn_team_id": away_espn_id,
        "home_espn_team_name": home_provider_name,
        "away_espn_team_name": away_provider_name,
        "home_mapping_status": home_mapping_status,
        "away_mapping_status": away_mapping_status,
        "home_mapping_candidate": home_candidate,
        "away_mapping_candidate": away_candidate,
        "home_score": score_for(home),
        "away_score": score_for(away),
        "neutral_site": competition.get("neutralSite", False),
        "completed": completed,
    }


def load_espn_season(season: int, teams: pd.DataFrame) -> pd.DataFrame:
    """Refresh the season schedule, retaining FBS/FCS events for calendar features."""
    team_lookup = _team_lookup(teams)
    session = requests.Session()
    weekly_events = []
    provider_teams_by_id = {}
    for week in range(1, 16):
        response = session.get(
            ESPN_SCOREBOARD,
            params={
                "dates": season,
                "seasontype": 2,
                "week": week,
                "groups": 80,
                "limit": 100,
            },
            timeout=20,
        )
        response.raise_for_status()
        for event in response.json().get("events", []):
            weekly_events.append((week, event))
            for competitor in event["competitions"][0]["competitors"]:
                provider_team = competitor["team"]
                provider_team_id = str(provider_team.get("id", ""))
                if provider_team_id:
                    provider_teams_by_id[provider_team_id] = provider_team

    team_mapping = _build_espn_team_mapping(
        season, teams, list(provider_teams_by_id.values())
    )
    espn_id_lookup = {
        row["provider_team_id"]: row for row in team_mapping.to_dict("records")
    }
    games_by_id = {}
    observed_games = {}
    for week, event in weekly_events:
        game = _parse_espn_event(event, season, week, team_lookup, espn_id_lookup)
        observed_games[game["game_id"]] = game
        for side in ("home", "away"):
            if game[f"{side}_mapping_status"] != "mapped":
                LOGGER.info(
                    "Unmapped ESPN team game=%s team=%s provider_id=%s "
                    "reason=%s mapping_candidate=%s",
                    game["game_id"], game[f"{side}_espn_team_name"],
                    game[f"{side}_espn_team_id"], game[f"{side}_mapping_status"],
                    game[f"{side}_mapping_candidate"],
                )
        if game["home_is_fbs"] or game["away_is_fbs"]:
            games_by_id[game["game_id"]] = game
        else:
            LOGGER.info("Ignoring non-FBS event game=%s", game["game_id"])

    unmatched = [
        (game, side)
        for game in observed_games.values()
        for side in ("home", "away")
        if game[f"{side}_mapping_status"] in {
            "unmapped_fbs_provider_id", "provider_name_id_mismatch"
        }
    ]
    for game, side in unmatched:
        LOGGER.error(
            "Unmatched FBS game=%s team=%s provider_id=%s reason=%s mapping_candidate=%s",
            game["game_id"], game[f"{side}_espn_team_name"],
            game[f"{side}_espn_team_id"], game[f"{side}_mapping_status"],
            game[f"{side}_mapping_candidate"],
        )
    if unmatched:
        raise ValueError(f"Unmatched FBS teams in ESPN schedule: {len(unmatched)}")

    if not any(game["home_is_fbs"] and game["away_is_fbs"] for game in games_by_id.values()):
        raise ValueError(f"ESPN returned no FBS-vs-FBS games for season {season}.")
    schedule = pd.DataFrame(games_by_id.values()).sort_values(
        ["game_date", "game_id"]
    ).reset_index(drop=True)
    schedule.attrs["provider_team_mapping"] = team_mapping
    return schedule


def run_backtest(db_path: Path = DB_PATH, run_dir: Path | None = None) -> pd.DataFrame:
    ensure_warehouse(db_path)
    model_history = load_historical_games(db_path)
    schedule_history = load_schedule_history(db_path)
    features = build_model_features(
        model_history,
        load_roster_priors(db_path),
        load_advanced_stats(db_path),
        model_history=model_history,
        schedule_history=schedule_history,
    )
    if run_dir is not None:
        manifest = write_baseline_run(model_history, features, run_dir)
        return pd.DataFrame(manifest["metrics_by_season"])
    return run_walk_forward(model_history, features)


def predict_week(
    season: int, week: int, db_path: Path = DB_PATH, candidate: str = "direct_score_l1"
) -> pd.DataFrame:
    """Predict one week with a registered market-free score candidate."""
    ensure_warehouse(db_path)
    historical = load_historical_games(db_path)
    schedule_history = load_schedule_history(db_path)
    season_teams = load_fbs_teams(season, db_path)
    current_schedule = load_espn_season(season, season_teams)
    current_targets = current_schedule[
        current_schedule["home_is_fbs"] & current_schedule["away_is_fbs"]
    ].copy()
    current_completed = current_targets[
        current_targets["home_score"].notna() & current_targets["away_score"].notna()
    ]
    current_schedule_history = current_schedule[
        current_schedule["home_score"].notna() & current_schedule["away_score"].notna()
    ]
    model_history = pd.concat([historical, current_completed], ignore_index=True, sort=False)
    schedule_history = pd.concat(
        [schedule_history, current_schedule_history], ignore_index=True, sort=False
    )
    games = pd.concat([historical, current_targets], ignore_index=True, sort=False)
    advanced = load_advanced_stats(db_path, season)
    features = build_model_features(
        games,
        load_roster_priors(db_path, season),
        advanced,
        model_history=model_history,
        schedule_history=schedule_history,
    )
    prediction_features = features
    if candidate.startswith("role_"):
        from src.cfb_model.features.game_role_performance import build_game_role_features
        prediction_features = pd.concat(
            [features, build_game_role_features(games, advanced)], axis=1
        )

    train_mask = (games["season"] >= FIRST_TRAIN_SEASON) & (games["season"] < season)
    predict_mask = (
        (games["season"] == season)
        & (games["week"] == week)
        & games["home_score"].isna()
        & games["away_score"].isna()
    )
    if not predict_mask.any():
        raise ValueError(f"No scheduled FBS-vs-FBS games found for {season}, week {week}.")

    if candidate == "direct_score_l1":
        model = _make_model()
        model.fit(
            prediction_features.loc[train_mask, MODEL_COLUMNS],
            games.loc[train_mask, TARGET_COLUMNS].to_numpy(dtype=float),
        )
        predictions = model.predict(prediction_features.loc[predict_mask, MODEL_COLUMNS])
    else:
        from src.cfb_model.train.experiments import candidate_feature_columns, predict_candidate
        columns = candidate_feature_columns(candidate)
        predictions = predict_candidate(
            candidate,
            prediction_features.loc[train_mask, columns],
            games.loc[train_mask, TARGET_COLUMNS].to_numpy(dtype=float),
            prediction_features.loc[predict_mask, columns],
        )
    output = games.loc[predict_mask, ["game_id", "season", "week", "game_date", "home_team", "away_team"]].copy()
    display_stats, display_ranges = _season_to_date_advanced_stats(
        output, advanced, set(season_teams["school"]), season, week
    )
    for column in DISPLAY_ADVANCED_COLUMNS:
        output[column] = display_stats[column].to_numpy()
    for metric, bounds in display_ranges.items():
        for bound, value in bounds.items():
            output[f"advanced_{metric}_{bound}"] = value
    for column in (
        "home_points_for_season",
        "home_points_against_season",
        "away_points_for_season",
        "away_points_against_season",
    ):
        output[column] = prediction_features.loc[predict_mask, column].to_numpy()
    for column in ADVANCED_COLUMNS:
        output[column] = prediction_features.loc[predict_mask, column].to_numpy()
    output["predicted_home_score"] = predictions[:, 0]
    output["predicted_away_score"] = predictions[:, 1]
    output["predicted_home_margin"] = predictions[:, 0] - predictions[:, 1]
    output["predicted_total"] = predictions[:, 0] + predictions[:, 1]
    return output.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--backtest", action="store_true", help="Run season-forward score evaluation.")
    mode.add_argument("--predict-week", type=int, help="Predict unplayed FBS games in this week.")
    parser.add_argument("--season", type=int, default=date.today().year)
    parser.add_argument("--output", type=Path, help="CSV path for weekly score predictions.")
    parser.add_argument(
        "--run-dir", type=Path,
        help="Save per-game predictions, metrics, and manifest for a reproducible backtest.",
    )
    parser.add_argument(
        "--candidate", default="direct_score_l1",
        choices=[
            "direct_score_l1",
            "environment_total_margin_l1",
            "role_points_score_l1",
            "role_form_score_l1",
            "role_advanced_score_l1",
            "role_both_roles_score_l1",
            "role_relevant_and_overall_score_l1",
        ],
        help="Market-free score candidate. Use the validated winner only after backtesting.",
    )
    args = parser.parse_args()

    if args.backtest:
        results = run_backtest(run_dir=args.run_dir)
        print(results.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
        if args.run_dir:
            print(f"\nSaved baseline artifacts to {args.run_dir}")
        print(
            "\nMean across validation seasons: "
            f"score MAE={results['score_mae'].mean():.3f}, "
            f"RMSE={results['score_rmse'].mean():.3f}, "
            f"R2={results['score_r2'].mean():.3f}, "
            f"winner AUC={results['winner_auc'].mean():.3f}"
        )
        return

    predictions = predict_week(args.season, args.predict_week, candidate=args.candidate)
    output_path = args.output or (
        PROJECT_ROOT / "results" / f"score_predictions_{args.season}_week{args.predict_week}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    print(predictions.to_string(index=False, float_format=lambda value: f"{value:.1f}"))
    print(f"\nWrote {len(predictions)} score predictions to {output_path}")


if __name__ == "__main__":
    main()
