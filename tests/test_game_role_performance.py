import numpy as np
import pandas as pd

from src.cfb_model.features.advanced import ADVANCED_METRICS
from src.cfb_model.features.game_role_performance import (
    FEATURE_COLUMNS,
    build_game_role_features,
)


def _games():
    return pd.DataFrame([
        {
            "game_id": 1, "season": 2024, "week": 1,
            "game_date": "2024-09-01T12:00:00Z",
            "home_team": "A", "away_team": "B",
            "home_score": 30, "away_score": 10, "neutral_site": True,
        },
        {
            "game_id": 2, "season": 2024, "week": 1,
            "game_date": "2024-09-01T15:00:00Z",
            "home_team": "A", "away_team": "C",
            "home_score": 20, "away_score": 14, "neutral_site": False,
        },
        {
            "game_id": 3, "season": 2024, "week": 2,
            "game_date": "2024-09-08T12:00:00Z",
            "home_team": "B", "away_team": "A",
            "home_score": 24, "away_score": 17, "neutral_site": False,
        },
        {
            "game_id": 4, "season": 2024, "week": 3,
            "game_date": "2024-09-15T12:00:00Z",
            "home_team": "A", "away_team": "B",
            "home_score": np.nan, "away_score": np.nan, "neutral_site": False,
        },
    ])


def _advanced(rows):
    frame = pd.DataFrame(rows, columns=["game_id", "season", "week", "team", "value"])
    for metric in ADVANCED_METRICS:
        frame[metric] = frame["value"]
    return frame.drop(columns="value")


def test_features_are_split_by_team_game_role_and_prior_only():
    games = _games()

    features = build_game_role_features(games)

    assert np.isnan(features.loc[1, "home_team_prior_home_points_for"])
    assert features.loc[3, "home_team_prior_home_points_for"] == 20
    assert features.loc[3, "home_team_prior_home_points_against"] == 14
    assert features.loc[3, "home_team_prior_home_win_pct"] == 1
    assert features.loc[3, "home_team_prior_away_points_for"] == 17
    assert features.loc[3, "home_team_prior_away_points_against"] == 24
    assert features.loc[3, "home_team_prior_away_win_pct"] == 0
    assert np.isnan(features.loc[3, "away_team_prior_away_points_for"])
    assert features.loc[3, "home_team_prior_overall_points_for"] == (30 + 20 + 17) / 3


def test_advanced_role_features_use_game_id_matched_prior_stats():
    games = _games()
    advanced = _advanced([
        [1, 2024, 1, "A", 0.4],
        [1, 2024, 1, "B", 0.2],
        [2, 2024, 1, "A", 0.5],
        [3, 2024, 2, "A", 0.25],
    ])

    features = build_game_role_features(games, advanced)

    assert features.loc[3, "home_team_prior_home_offense_ppa"] == 0.5
    assert np.isnan(features.loc[3, "away_team_prior_away_offense_ppa"])
    assert len(FEATURE_COLUMNS["advanced"]) == 18


def test_role_form_resets_at_the_start_of_each_season():
    games = _games()
    next_season = games.iloc[[3]].copy()
    next_season["season"] = 2025
    next_season["week"] = 1
    next_season["game_date"] = "2025-08-31T12:00:00Z"
    games = pd.concat([games, next_season], ignore_index=True)

    features = build_game_role_features(games)

    assert np.isnan(features.loc[4, "home_team_prior_home_points_for"])
    assert np.isnan(features.loc[4, "home_team_prior_overall_points_for"])
