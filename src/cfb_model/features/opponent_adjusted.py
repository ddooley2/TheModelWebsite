"""Point-in-time opponent-adjusted score ratings for college football."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.linear_model import Ridge

FEATURE_COLUMNS = [
    "home_offense_rating",
    "home_defense_rating",
    "away_offense_rating",
    "away_defense_rating",
    "offense_rating_diff",
    "defense_rating_diff",
]

RIDGE_ALPHA = 20.0


RECENCY_HALF_LIFE_DAYS = (28, 56, 84, 140, 224)
SEASON_PRIOR_STRENGTHS = (0.0, 5.0, 20.0)
RECENCY_RIDGE_ALPHAS = (10.0, 20.0, 40.0)


class _RatingObservation:
    def __init__(
        self,
        game_id: str,
        game_date: pd.Timestamp,
        season: int,
        home_team: str,
        away_team: str,
        home_score: float,
        away_score: float,
        neutral_site: bool,
        eligibility: str,
    ) -> None:
        self.game_id = game_id
        self.game_date = game_date
        self.season = season
        self.home_team = home_team
        self.away_team = away_team
        self.home_score = home_score
        self.away_score = away_score
        self.neutral_site = neutral_site
        self.eligibility = eligibility


def _fit_ratings(history: list[tuple[str, str, float, float, bool]]) -> dict[str, tuple[float, float]]:
    """Fit points-for ratings; defense is scaled as points allowed."""
    teams = sorted({team for home, away, *_ in history for team in (home, away)})
    team_index = {team: index for index, team in enumerate(teams)}
    n_teams = len(teams)
    n_columns = 2 * n_teams + 1

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    targets = np.empty(2 * len(history), dtype=float)

    for game_index, (home, away, home_score, away_score, neutral) in enumerate(history):
        home_row = 2 * game_index
        away_row = home_row + 1
        home_index = team_index[home]
        away_index = team_index[away]

        rows.extend((home_row, home_row))
        columns.extend((home_index, n_teams + away_index))
        values.extend((1.0, 1.0))
        if not neutral:
            rows.append(home_row)
            columns.append(2 * n_teams)
            values.append(1.0)

        rows.extend((away_row, away_row))
        columns.extend((away_index, n_teams + home_index))
        values.extend((1.0, 1.0))
        targets[home_row] = home_score
        targets[away_row] = away_score

    design = coo_matrix(
        (values, (rows, columns)), shape=(len(targets), n_columns)
    ).tocsr()
    model = Ridge(alpha=RIDGE_ALPHA, solver="lsqr")
    model.fit(design, targets)
    return {
        team: (float(model.coef_[index]), float(model.coef_[n_teams + index]))
        for team, index in team_index.items()
    }


def _solve_recency_weighted_ratings(
    team_index: dict[str, int],
    xtx: np.ndarray,
    xty: np.ndarray,
    ridge_alpha: float,
    season_prior: dict[str, tuple[float, float]] | None = None,
    season_prior_strength: float = 0.0,
) -> dict[str, tuple[float, float]]:
    n_teams = len(team_index)
    if not n_teams or not np.any(xtx):
        return {}
    intercept_index = 2 * n_teams + 1
    matrix = xtx.copy()
    target = xty.copy()
    regularized_columns = np.arange(intercept_index)
    matrix[regularized_columns, regularized_columns] += ridge_alpha
    if season_prior_strength:
        team_columns = np.arange(2 * n_teams)
        prior = season_prior or {}
        prior_values = np.zeros(2 * n_teams, dtype=float)
        for team, index in team_index.items():
            offense, defense = prior.get(team, (0.0, 0.0))
            prior_values[index] = offense
            prior_values[n_teams + index] = defense
        matrix[team_columns, team_columns] += season_prior_strength
        target[team_columns] += season_prior_strength * prior_values
    try:
        coefficients = np.linalg.solve(matrix, target)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(matrix, target, rcond=None)[0]
    return {
        team: (
            float(coefficients[index]),
            float(coefficients[n_teams + index]),
        )
        for team, index in team_index.items()
    }


def _add_observation_sufficient_stats(
    observation: _RatingObservation,
    team_index: dict[str, int],
    xtx: np.ndarray,
    xty: np.ndarray,
) -> None:
    n_teams = len(team_index)
    home_advantage_index = 2 * n_teams
    intercept_index = home_advantage_index + 1
    home_index = team_index[observation.home_team]
    away_index = team_index[observation.away_team]
    rows = [
        [home_index, n_teams + away_index, intercept_index],
        [away_index, n_teams + home_index, intercept_index],
    ]
    if not observation.neutral_site:
        rows[0].append(home_advantage_index)
    for columns, score in zip(rows, (observation.home_score, observation.away_score)):
        xtx[np.ix_(columns, columns)] += 1.0
        xty[columns] += score


def build_unweighted_opponent_adjusted_features(games: pd.DataFrame) -> pd.DataFrame:
    """Build the original date-batched, unweighted opponent-adjusted ratings.

    Requires ``game_date``, ``home_team``, ``away_team``, ``home_score``, and
    ``away_score``. Only finite score pairs from valid, named-team games enter
    the history; missing or invalid dates receive zero ratings and are not
    added to history. ``neutral_site`` is optional and defaults to False.
    For each UTC calendar date, features are emitted before any results from
    that date are added, so games on the same date never update one another.
    A sparse Ridge model (alpha=20) estimates points-for offense and points-
    allowed defense, with a fitted intercept and non-neutral home advantage.
    Teams not seen in earlier completed games receive zero (the regularized
    preseason baseline). The function uses only team, date, neutral-site, and
    score columns; market or line inputs are ignored.

    Returns only the six rating features, with the original row order, index,
    and one-to-one row alignment preserved.
    """
    required = {"game_date", "home_team", "away_team", "home_score", "away_score"}
    missing = required - set(games.columns)
    if missing:
        raise ValueError(f"Missing game columns: {sorted(missing)}")

    n_games = len(games)
    output = np.zeros((n_games, len(FEATURE_COLUMNS)), dtype=float)
    if n_games == 0:
        return pd.DataFrame(output, index=games.index, columns=FEATURE_COLUMNS)

    dates = pd.to_datetime(
        games["game_date"], utc=True, errors="coerce", format="mixed"
    ).dt.normalize()
    home_scores = pd.to_numeric(games["home_score"], errors="coerce").to_numpy(dtype=float)
    away_scores = pd.to_numeric(games["away_score"], errors="coerce").to_numpy(dtype=float)

    home_values = games["home_team"].to_numpy(dtype=object)
    away_values = games["away_team"].to_numpy(dtype=object)
    neutral_values = (
        np.asarray(
            [False if pd.isna(value) else bool(value) for value in games["neutral_site"]],
            dtype=bool,
        )
        if "neutral_site" in games
        else np.zeros(n_games, dtype=bool)
    )

    history: list[tuple[str, str, float, float, bool]] = []
    dated_positions = pd.Series(np.arange(n_games), index=dates)
    for game_date, batch_positions in dated_positions.groupby(level=0, sort=True):
        if pd.isna(game_date):
            continue
        positions = batch_positions.to_numpy(dtype=int)
        ratings = _fit_ratings(history) if history else {}

        for position in positions:
            home = home_values[position]
            away = away_values[position]
            home_rating = ratings.get(str(home), (0.0, 0.0)) if pd.notna(home) else (0.0, 0.0)
            away_rating = ratings.get(str(away), (0.0, 0.0)) if pd.notna(away) else (0.0, 0.0)
            output[position] = (
                home_rating[0],
                home_rating[1],
                away_rating[0],
                away_rating[1],
                home_rating[0] - away_rating[0],
                home_rating[1] - away_rating[1],
            )

        # Only completed, valid games update ratings, and only after the date batch.
        for position in positions:
            home = home_values[position]
            away = away_values[position]
            home_score = home_scores[position]
            away_score = away_scores[position]
            if (
                pd.isna(home)
                or pd.isna(away)
                or not np.isfinite(home_score)
                or not np.isfinite(away_score)
            ):
                continue
            history.append(
                (
                    str(home),
                    str(away),
                    home_score,
                    away_score,
                    bool(neutral_values[position]),
                )
            )

    return pd.DataFrame(output, index=games.index, columns=FEATURE_COLUMNS)


build_opponent_adjusted_features = build_unweighted_opponent_adjusted_features


def build_recency_weighted_opponent_adjusted_features(
    games: pd.DataFrame,
    *,
    half_life_days: int,
    season_prior_strength: float,
    ridge_alpha: float,
) -> pd.DataFrame:
    """Build leakage-safe exponential ratings with a fixed prior-season rating.

    Only completed FBS-vs-FBS observations from prior UTC calendar dates are
    used. Empty or one-game history produces finite zero-centered ratings.
    """
    required = {
        "game_date", "season", "home_team", "away_team", "home_score", "away_score"
    }
    missing = required - set(games.columns)
    if missing:
        raise ValueError(f"Missing game columns: {sorted(missing)}")
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive.")
    if season_prior_strength < 0 or ridge_alpha < 0:
        raise ValueError("season_prior_strength and ridge_alpha must be non-negative.")

    n_games = len(games)
    output = np.zeros((n_games, len(FEATURE_COLUMNS)), dtype=float)
    if n_games == 0:
        return pd.DataFrame(output, index=games.index, columns=FEATURE_COLUMNS)

    dates = pd.to_datetime(
        games["game_date"], utc=True, errors="coerce", format="mixed"
    ).dt.normalize()
    seasons = pd.to_numeric(games["season"], errors="coerce").to_numpy(dtype=float)
    home_scores = pd.to_numeric(games["home_score"], errors="coerce").to_numpy(dtype=float)
    away_scores = pd.to_numeric(games["away_score"], errors="coerce").to_numpy(dtype=float)
    home_values = games["home_team"].to_numpy(dtype=object)
    away_values = games["away_team"].to_numpy(dtype=object)
    neutral_values = (
        np.asarray(
            [False if pd.isna(value) else bool(value) for value in games["neutral_site"]],
            dtype=bool,
        )
        if "neutral_site" in games
        else np.zeros(n_games, dtype=bool)
    )
    fbs_values = (
        games["home_is_fbs"].fillna(False).to_numpy(dtype=bool)
        & games["away_is_fbs"].fillna(False).to_numpy(dtype=bool)
        if {"home_is_fbs", "away_is_fbs"} <= set(games.columns)
        else np.ones(n_games, dtype=bool)
    )
    game_ids = (
        games["game_id"].to_numpy(dtype=object)
        if "game_id" in games
        else np.arange(n_games).astype(str)
    )

    dated_positions = pd.Series(np.arange(n_games), index=dates)
    positions_by_date = {
        game_date: batch.to_numpy(dtype=int)
        for game_date, batch in dated_positions.groupby(level=0, sort=True)
        if pd.notna(game_date)
    }
    observations_by_date: dict[pd.Timestamp, list[_RatingObservation]] = {}
    for game_date, positions in positions_by_date.items():
        observations = []
        for position in positions:
            home = home_values[position]
            away = away_values[position]
            home_score = home_scores[position]
            away_score = away_scores[position]
            if (
                not fbs_values[position]
                or not np.isfinite(seasons[position])
                or pd.isna(home)
                or pd.isna(away)
                or not np.isfinite(home_score)
                or not np.isfinite(away_score)
            ):
                continue
            observations.append(
                _RatingObservation(
                    game_id=str(game_ids[position]),
                    game_date=game_date,
                    season=int(seasons[position]),
                    home_team=str(home),
                    away_team=str(away),
                    home_score=home_score,
                    away_score=away_score,
                    neutral_site=bool(neutral_values[position]),
                    eligibility="fbs_vs_fbs",
                )
            )
        observations_by_date[game_date] = observations

    teams = sorted({
        team
        for observations in observations_by_date.values()
        for observation in observations
        for team in (observation.home_team, observation.away_team)
    })
    team_index = {team: index for index, team in enumerate(teams)}
    n_columns = 2 * len(teams) + 2
    xtx = np.zeros((n_columns, n_columns), dtype=float)
    xty = np.zeros(n_columns, dtype=float)
    season_priors: dict[int, dict[str, tuple[float, float]]] = {}
    previous_date: pd.Timestamp | None = None
    for game_date, positions in positions_by_date.items():
        if previous_date is not None:
            days = (game_date - previous_date).days
            decay = float(np.exp2(-days / half_life_days))
            xtx *= decay
            xty *= decay
        ratings_by_season: dict[int, dict[str, tuple[float, float]]] = {}

        for position in positions:
            if not np.isfinite(seasons[position]):
                continue
            season = int(seasons[position])
            if season not in season_priors:
                season_priors[season] = _solve_recency_weighted_ratings(
                    team_index, xtx, xty, ridge_alpha
                )
            if season not in ratings_by_season:
                ratings_by_season[season] = _solve_recency_weighted_ratings(
                    team_index,
                    xtx,
                    xty,
                    ridge_alpha,
                    season_prior=season_priors[season],
                    season_prior_strength=season_prior_strength,
                )
            ratings = ratings_by_season[season]
            home = home_values[position]
            away = away_values[position]
            home_rating = ratings.get(str(home), (0.0, 0.0)) if pd.notna(home) else (0.0, 0.0)
            away_rating = ratings.get(str(away), (0.0, 0.0)) if pd.notna(away) else (0.0, 0.0)
            output[position] = (
                home_rating[0],
                home_rating[1],
                away_rating[0],
                away_rating[1],
                home_rating[0] - away_rating[0],
                home_rating[1] - away_rating[1],
            )

        for observation in observations_by_date[game_date]:
            _add_observation_sufficient_stats(observation, team_index, xtx, xty)
        previous_date = game_date

    return pd.DataFrame(output, index=games.index, columns=FEATURE_COLUMNS)
