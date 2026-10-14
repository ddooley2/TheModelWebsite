"""Pregame CFB roster priors and season-carrying Elo features.

The roster inputs are the annual ``/talent?year=...`` and
``/recruiting/teams?year=...`` fields loaded by ``fetch_milestone1_pt2``.
Returning-production PPA and usage fields are intentionally excluded because
they summarize on-field production rather than roster composition alone.
"""

import numpy as np
import pandas as pd


ROSTER_COLUMNS = ("team_talent", "recruiting_rank", "recruiting_points")
FEATURE_COLUMNS = (
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_team_talent",
    "away_team_talent",
    "home_recruiting_rank",
    "away_recruiting_rank",
    "home_recruiting_points",
    "away_recruiting_points",
)

_ELO_BASE = 1500.0
_ELO_SCALE = 400.0


def build_preseason_features(
    games: pd.DataFrame,
    roster_priors: pd.DataFrame | None = None,
    *,
    elo_k: float = 20.0,
    elo_regression: float = 0.75,
) -> pd.DataFrame:
    """Return row-aligned home/away preseason priors and pregame Elo.

    Elo starts at 1500 for first-seen teams. The prior season's final rating is
    regressed 25% toward 1500 at each season boundary. Games on a shared date
    all receive ratings from the start of that date; their Elo changes are
    applied together afterward. Rows with missing scores still receive
    features and do not update ratings. Missing roster fields remain NaN.

    ``games`` requires ``season``, ``game_date``, ``home_team``, and
    ``away_team``. Scores are optional; if present, both must be non-null for
    an Elo update. ``roster_priors`` uses ``season`` and ``team`` keys and any
    available subset of ``ROSTER_COLUMNS``.
    """
    required = {"season", "game_date", "home_team", "away_team"}
    missing = required.difference(games.columns)
    if missing:
        raise ValueError(f"games is missing required columns: {sorted(missing)}")
    if not 0.0 <= elo_regression <= 1.0:
        raise ValueError("elo_regression must be between 0 and 1")

    count = len(games)
    values = {column: np.full(count, np.nan, dtype=float) for column in FEATURE_COLUMNS}
    if count == 0:
        return pd.DataFrame(values, index=games.index)

    schedule = games.reset_index(drop=True)
    dates = pd.to_datetime(schedule["game_date"], errors="coerce", utc=True, format="mixed").dt.date
    scores = {}
    for column in ("home_score", "away_score"):
        scores[column] = (
            pd.to_numeric(schedule[column], errors="coerce").to_numpy(dtype=float)
            if column in schedule
            else np.full(count, np.nan)
        )

    prior_lookup = _roster_lookup(roster_priors)
    season_values = pd.to_numeric(schedule["season"], errors="coerce")
    if season_values.isna().any():
        raise ValueError("games.season must contain valid seasons")

    def set_game_features(position, home_rating, away_rating):
        values["home_elo"][position] = home_rating
        values["away_elo"][position] = away_rating
        values["elo_diff"][position] = home_rating - away_rating
        season = schedule.at[position, "season"]
        for side, team in (("home", schedule.at[position, "home_team"]),
                           ("away", schedule.at[position, "away_team"])):
            prior = prior_lookup.get((season, team), {})
            for column in ROSTER_COLUMNS:
                values[f"{side}_{column}"][position] = prior.get(column, np.nan)

    carried_ratings = {}
    previous_season = None
    for season in sorted(season_values.unique()):
        year_positions = np.flatnonzero(season_values.to_numpy() == season)
        gap = 1 if previous_season is None else int(season - previous_season)
        shrink = elo_regression ** gap
        season_ratings = {
            team: _ELO_BASE + (rating - _ELO_BASE) * shrink
            for team, rating in carried_ratings.items()
        }
        year_dates = {}
        for position in year_positions:
            date = dates.iloc[position]
            if pd.isna(date):
                set_game_features(
                    position,
                    _team_rating(season_ratings, schedule.at[position, "home_team"]),
                    _team_rating(season_ratings, schedule.at[position, "away_team"]),
                )
            else:
                year_dates.setdefault(date, []).append(position)

        for date in sorted(year_dates):
            positions = year_dates[date]
            day_ratings = {}
            for position in positions:
                home = schedule.at[position, "home_team"]
                away = schedule.at[position, "away_team"]
                home_rating = _team_rating(season_ratings, home)
                away_rating = _team_rating(season_ratings, away)
                set_game_features(position, home_rating, away_rating)
                day_ratings[position] = (home_rating, away_rating)

            deltas = {}
            for position in positions:
                home_score = scores["home_score"][position]
                away_score = scores["away_score"][position]
                if not np.isfinite(home_score) or not np.isfinite(away_score):
                    continue
                home = schedule.at[position, "home_team"]
                away = schedule.at[position, "away_team"]
                if _missing_team(home) or _missing_team(away):
                    continue
                home_rating, away_rating = day_ratings[position]
                expected_home = 1.0 / (
                    1.0 + 10.0 ** ((away_rating - home_rating) / _ELO_SCALE)
                )
                actual_home = 1.0 if home_score > away_score else (
                    0.0 if home_score < away_score else 0.5
                )
                change = elo_k * (actual_home - expected_home)
                deltas[home] = deltas.get(home, 0.0) + change
                deltas[away] = deltas.get(away, 0.0) - change
            for team, change in deltas.items():
                season_ratings[team] = _team_rating(season_ratings, team) + change

        # Preserve first-seen teams (including teams whose games have no score)
        # and carry every known team's rating into the next season.
        for position in year_positions:
            for side in ("home", "away"):
                team = schedule.at[position, f"{side}_team"]
                if not _missing_team(team):
                    season_ratings.setdefault(team, _ELO_BASE)
        carried_ratings = season_ratings
        previous_season = season

    return pd.DataFrame(values, index=games.index, columns=FEATURE_COLUMNS)


def _roster_lookup(roster_priors):
    if roster_priors is None or roster_priors.empty:
        return {}
    if not {"season", "team"}.issubset(roster_priors.columns):
        raise ValueError("roster_priors must contain season and team columns")

    available = [column for column in ROSTER_COLUMNS if column in roster_priors]
    priors = roster_priors[["season", "team"] + available].copy()
    for column in available:
        priors[column] = pd.to_numeric(priors[column], errors="coerce")
    priors = priors.drop_duplicates(["season", "team"], keep="first")
    return {
        (row.season, row.team): {
            column: getattr(row, column) for column in available
        }
        for row in priors.itertuples(index=False)
    }


def _missing_team(team):
    return bool(pd.isna(team))


def _team_rating(ratings, team):
    return _ELO_BASE if _missing_team(team) else ratings.get(team, _ELO_BASE)
