import numpy as np
import pandas as pd

from src.cfb_model.features.advanced import ADVANCED_METRICS, FEATURE_COLUMNS, build_advanced_features


def _stats(rows):
    frame = pd.DataFrame(rows, columns=["game_id", "season", "week", "team", "ppa"])
    for metric in ADVANCED_METRICS:
        frame[metric] = frame["ppa"]
    return frame.drop(columns="ppa")


def test_features_use_only_earlier_games_and_prior_season():
    games = pd.DataFrame([
        {"game_id": 1, "season": 2024, "week": 1, "game_date": "2024-09-01", "home_team": "A", "away_team": "B"},
        {"game_id": 2, "season": 2025, "week": 1, "game_date": "2025-09-01", "home_team": "A", "away_team": "B"},
        {"game_id": "espn-3", "season": 2025, "week": 2, "game_date": "2025-09-08", "home_team": "A", "away_team": "B"},
    ])
    stats = _stats([
        [1, 2024, 1, "A", 0.3], [1, 2024, 1, "B", 0.1],
        [2, 2025, 1, "A", 0.9], [2, 2025, 1, "B", 0.5],
        [np.nan, 2025, 2, "A", 5.0], [np.nan, 2025, 2, "B", 5.0],
    ])
    features = build_advanced_features(games, stats)

    assert list(features.columns) == FEATURE_COLUMNS
    assert np.isnan(features.loc[0, "home_offense_ppa"])
    assert features.loc[1, "home_offense_ppa"] == 0.3
    expected = (3 * 0.3 + 0.9) / 4
    assert np.isclose(features.loc[2, "home_offense_ppa"], expected)
    assert np.isclose(features.loc[2, "away_offense_ppa"], (3 * 0.1 + 0.5) / 4)
