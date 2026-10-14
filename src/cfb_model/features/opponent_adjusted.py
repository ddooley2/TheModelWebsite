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


def build_opponent_adjusted_features(games: pd.DataFrame) -> pd.DataFrame:
    """Build date-batched, point-in-time CFB offense/defense rating features.

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
