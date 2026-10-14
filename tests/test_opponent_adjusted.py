import pandas as pd

from src.cfb_model.features.opponent_adjusted import (
    FEATURE_COLUMNS,
    build_opponent_adjusted_features,
)


def _games():
    return pd.DataFrame(
        [
            {
                "game_date": "2024-08-31",
                "home_team": "Alpha",
                "away_team": "Beta",
                "home_score": 35,
                "away_score": 7,
                "neutral_site": False,
                "market_spread": -10,
            },
            {
                "game_date": "2024-09-07T12:00:00Z",
                "home_team": "Alpha",
                "away_team": "Gamma",
                "home_score": 21,
                "away_score": 17,
                "neutral_site": False,
            },
            {
                "game_date": "2024-09-07T20:00:00Z",
                "home_team": "Beta",
                "away_team": "Alpha",
                "home_score": 14,
                "away_score": 28,
                "neutral_site": False,
            },
            {
                "game_date": "2024-09-14",
                "home_team": "Gamma",
                "away_team": "Beta",
                "home_score": 24,
                "away_score": 10,
                "neutral_site": True,
            },
        ],
        index=[12, 4, 19, 2],
    )


def test_features_use_only_prior_date_results_and_ignore_market_inputs():
    games = _games()
    features = build_opponent_adjusted_features(games)

    assert list(features.columns) == FEATURE_COLUMNS
    assert features.index.tolist() == games.index.tolist()
    assert len(features) == len(games)
    # The two September 7 games share exactly the same pre-date team ratings.
    assert features.loc[4, "home_offense_rating"] == features.loc[19, "away_offense_rating"]
    assert features.loc[4, "home_defense_rating"] == features.loc[19, "away_defense_rating"]
    assert "market_spread" not in features.columns


def test_target_and_future_scores_do_not_change_target_features():
    games = _games()
    original = build_opponent_adjusted_features(games)

    changed = games.copy()
    changed.loc[1, ["home_score", "away_score"]] = [0, 70]
    changed.loc[2, ["home_score", "away_score"]] = [63, 3]
    changed.loc[3, ["home_score", "away_score"]] = [2, 59]
    updated = build_opponent_adjusted_features(changed)

    pd.testing.assert_series_equal(original.loc[4], updated.loc[4])


def test_output_preserves_input_row_order_and_index():
    games = _games().iloc[[3, 1, 0, 2]]
    features = build_opponent_adjusted_features(games)

    assert features.index.tolist() == [2, 4, 12, 19]
    assert len(features) == len(games)
