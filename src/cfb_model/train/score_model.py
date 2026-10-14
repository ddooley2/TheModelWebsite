"""Train, evaluate, and run a score-first college football model."""

from __future__ import annotations

import argparse
import os
import unicodedata
from collections import defaultdict, deque
from datetime import date
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.multioutput import MultiOutputRegressor

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
MODEL_COLUMNS = FEATURE_COLUMNS + RATING_COLUMNS + PRESEASON_COLUMNS
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


def _normalize_team_name(name: str) -> str:
    decomposed = unicodedata.normalize("NFKD", name)
    ascii_name = decomposed.encode("ascii", "ignore").decode("ascii")
    return "".join(character for character in ascii_name.lower() if character.isalnum())


def _team_lookup(teams: set[str]) -> dict[str, str]:
    lookup = {_normalize_team_name(team): team for team in teams}
    for abbreviation, team in ESPN_ABBREVIATION_ALIASES.items():
        if team in teams:
            lookup[abbreviation.lower()] = team
    return lookup


def load_historical_games(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Load completed FBS-vs-FBS games from the local warehouse."""
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


def build_pregame_features(games: pd.DataFrame) -> pd.DataFrame:
    """Build each game's features before updating team state with its result."""
    required = {
        "season", "week", "game_date", "home_team", "away_team",
        "home_score", "away_score", "neutral_site",
    }
    missing = required - set(games.columns)
    if missing:
        raise ValueError(f"Missing game columns: {sorted(missing)}")

    ordered = games.copy()
    ordered["input_order"] = np.arange(len(ordered))
    ordered["game_date"] = pd.to_datetime(ordered["game_date"], utc=True, format="mixed")
    ordered = ordered.sort_values(["game_date", "input_order"], kind="stable")

    state = defaultdict(
        lambda: {
            "elo": 1500.0,
            "recent_for": deque(maxlen=4),
            "recent_against": deque(maxlen=4),
            "career_for": 0.0,
            "career_against": 0.0,
            "career_games": 0,
            "last_date": None,
        }
    )
    season_stats = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0, 0]))
    features = []

    for row in ordered.itertuples(index=False):
        home = state[row.home_team]
        away = state[row.away_team]
        home_season = season_stats[row.season][row.home_team]
        away_season = season_stats[row.season][row.away_team]

        def average_or_prior(total: float, count: int, career_total: float, career_count: int) -> float:
            if count:
                return total / count
            if career_count:
                return career_total / career_count
            return 0.0

        home_recent_for = list(home["recent_for"])
        home_recent_against = list(home["recent_against"])
        away_recent_for = list(away["recent_for"])
        away_recent_against = list(away["recent_against"])
        game_date = row.game_date

        def rest_days(team_state: dict) -> float:
            previous = team_state["last_date"]
            if previous is None:
                return 14.0
            return float(min(max((game_date - previous).days, 0), 14))

        features.append(
            {
                "input_order": row.input_order,
                "home_elo": home["elo"],
                "away_elo": away["elo"],
                "elo_diff": home["elo"] - away["elo"],
                "home_points_for_recent": (
                    float(np.mean(home_recent_for)) if home_recent_for
                    else average_or_prior(home_season[0], home_season[2], home["career_for"], home["career_games"])
                ),
                "home_points_against_recent": (
                    float(np.mean(home_recent_against)) if home_recent_against
                    else average_or_prior(home_season[1], home_season[2], home["career_against"], home["career_games"])
                ),
                "away_points_for_recent": (
                    float(np.mean(away_recent_for)) if away_recent_for
                    else average_or_prior(away_season[0], away_season[2], away["career_for"], away["career_games"])
                ),
                "away_points_against_recent": (
                    float(np.mean(away_recent_against)) if away_recent_against
                    else average_or_prior(away_season[1], away_season[2], away["career_against"], away["career_games"])
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
                "home_games_played": home_season[2],
                "away_games_played": away_season[2],
                "home_rest_days": rest_days(home),
                "away_rest_days": rest_days(away),
                "week": float(row.week),
                "neutral_site": float(bool(row.neutral_site)),
            }
        )

        if pd.notna(row.home_score) and pd.notna(row.away_score):
            home_score = float(row.home_score)
            away_score = float(row.away_score)
            home_margin = home_score - away_score
            expected_home = 1.0 / (1.0 + 10.0 ** ((away["elo"] - home["elo"]) / 400.0))
            actual_home = 1.0 if home_margin > 0 else 0.0 if home_margin < 0 else 0.5
            margin_multiplier = np.log1p(abs(home_margin)) * 2.2 / (
                abs(home["elo"] - away["elo"]) * 0.001 + 2.2
            )
            elo_change = 20.0 * margin_multiplier * (actual_home - expected_home)
            home["elo"] += elo_change
            away["elo"] -= elo_change

            home["recent_for"].append(home_score)
            home["recent_against"].append(away_score)
            away["recent_for"].append(away_score)
            away["recent_against"].append(home_score)
            home["career_for"] += home_score
            home["career_against"] += away_score
            home["career_games"] += 1
            away["career_for"] += away_score
            away["career_against"] += home_score
            away["career_games"] += 1

            home_season[0] += home_score
            home_season[1] += away_score
            home_season[2] += 1
            home_season[3] += int(home_margin > 0)
            away_season[0] += away_score
            away_season[1] += home_score
            away_season[2] += 1
            away_season[3] += int(home_margin < 0)
            home["last_date"] = game_date
            away["last_date"] = game_date

    return (
        pd.DataFrame(features)
        .sort_values("input_order")
        .drop(columns="input_order")
        .reset_index(drop=True)
    )


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


def build_model_features(games: pd.DataFrame, roster_priors: pd.DataFrame) -> pd.DataFrame:
    """Build all pregame features (no market inputs); output is row-aligned with games."""
    pregame = build_pregame_features(games)
    ratings = build_opponent_adjusted_features(games).reset_index(drop=True)
    preseason = build_preseason_features(games, roster_priors).reset_index(drop=True)
    preseason = preseason.rename(columns={
        "home_elo": "home_preseason_elo",
        "away_elo": "away_preseason_elo",
        "elo_diff": "preseason_elo_diff",
    })
    features = pd.concat([pregame, ratings, preseason], axis=1)
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
        results.append(
            {
                "season": season,
                "games": len(valid_y),
                **score_metrics(valid_y, predicted),
                "baseline_score_mae": float(mean_absolute_error(valid_y.ravel(), baseline.ravel())),
            }
        )
    if not results:
        raise ValueError("No walk-forward seasons had both training and validation games.")
    return pd.DataFrame(results)


def _parse_espn_event(event: dict, season: int, week: int, team_lookup: dict[str, str]) -> dict | None:
    competition = event["competitions"][0]
    competitors = competition["competitors"]
    home = next(team for team in competitors if team["homeAway"] == "home")
    away = next(team for team in competitors if team["homeAway"] == "away")

    def canonical_name(team: dict) -> str | None:
        abbreviation = team["team"].get("abbreviation", "").upper()
        name = ESPN_ABBREVIATION_ALIASES.get(abbreviation, team["team"]["location"])
        return team_lookup.get(_normalize_team_name(name))

    home_name = canonical_name(home)
    away_name = canonical_name(away)
    if home_name is None or away_name is None:
        return None

    completed = event["status"]["type"]["completed"]
    return {
        "game_id": f"espn-{event['id']}",
        "season": season,
        "week": week,
        "game_date": event["date"],
        "home_team": home_name,
        "away_team": away_name,
        "home_score": float(home["score"]) if completed else np.nan,
        "away_score": float(away["score"]) if completed else np.nan,
        "neutral_site": competition.get("neutralSite", False),
        "completed": completed,
    }


def load_espn_season(season: int, teams: set[str]) -> pd.DataFrame:
    """Refresh the regular-season FBS schedule and completed scores from ESPN."""
    team_lookup = _team_lookup(teams)
    session = requests.Session()
    games_by_id = {}
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
            game = _parse_espn_event(event, season, week, team_lookup)
            if game is not None:
                games_by_id[game["game_id"]] = game
    if not games_by_id:
        raise ValueError(f"ESPN returned no FBS-vs-FBS games for season {season}.")
    return pd.DataFrame(games_by_id.values()).sort_values(["game_date", "game_id"]).reset_index(drop=True)


def run_backtest(db_path: Path = DB_PATH) -> pd.DataFrame:
    ensure_warehouse(db_path)
    games = load_historical_games(db_path)
    features = build_model_features(games, load_roster_priors(db_path))
    return run_walk_forward(games, features)


def predict_week(
    season: int, week: int, db_path: Path = DB_PATH, candidate: str = "direct_score_l1"
) -> pd.DataFrame:
    """Predict one week with a registered market-free score candidate."""
    ensure_warehouse(db_path)
    historical = load_historical_games(db_path)
    teams = set(historical.loc[historical["season"] == 2025, "home_team"]) | set(
        historical.loc[historical["season"] == 2025, "away_team"]
    )
    current = load_espn_season(season, teams)
    games = pd.concat([historical, current], ignore_index=True, sort=False)
    features = build_model_features(games, load_roster_priors(db_path, season))

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
            features.loc[train_mask, MODEL_COLUMNS],
            games.loc[train_mask, TARGET_COLUMNS].to_numpy(dtype=float),
        )
        predictions = model.predict(features.loc[predict_mask, MODEL_COLUMNS])
    else:
        from src.cfb_model.train.experiments import predict_candidate
        predictions = predict_candidate(
            candidate,
            features.loc[train_mask, MODEL_COLUMNS],
            games.loc[train_mask, TARGET_COLUMNS].to_numpy(dtype=float),
            features.loc[predict_mask, MODEL_COLUMNS],
        )
    output = games.loc[predict_mask, ["game_id", "season", "week", "game_date", "home_team", "away_team"]].copy()
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
        "--candidate", default="direct_score_l1",
        choices=["direct_score_l1", "environment_total_margin_l1"],
        help="Market-free score candidate. Use the validated winner only after backtesting.",
    )
    args = parser.parse_args()

    if args.backtest:
        results = run_backtest()
        print(results.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
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
