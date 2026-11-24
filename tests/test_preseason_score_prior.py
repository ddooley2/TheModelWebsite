import numpy as np
import pandas as pd
import pytest

from src.cfb_model.features.preseason_score_prior import (
    FEATURE_COLUMNS,
    build_preseason_score_prior_features,
    preseason_score_prior_data_dictionary,
)


def _games():
    return pd.DataFrame(
        [
            {"season": 2023, "game_date": "2023-09-01", "home_team": "Alpha",
             "away_team": "Beta", "home_score": 30, "away_score": 10},
            {"season": 2023, "game_date": "2023-09-08", "home_team": "Alpha",
             "away_team": "Gamma", "home_score": 20, "away_score": 10},
            {"season": 2024, "game_date": "2024-08-31", "home_team": "Alpha",
             "away_team": "Beta", "home_score": 10, "away_score": 20},
            {"season": 2024, "game_date": "2024-09-07", "home_team": "Alpha",
             "away_team": "Gamma", "home_score": 12, "away_score": 7},
            {"season": 2024, "game_date": "2024-09-07", "home_team": "Alpha",
             "away_team": "Delta", "home_score": 50, "away_score": 0},
            {"season": 2024, "game_date": "2024-09-14", "home_team": "Alpha",
             "away_team": "Beta", "home_score": 20, "away_score": 13},
        ],
        index=[30, 10, 20, 40, 50, 60],
    )


def test_prior_season_is_used_at_season_start_then_weight_decays():
    games = _games()
    features = build_preseason_score_prior_features(games)

    assert features.index.tolist() == games.index.tolist()
    assert list(features.columns) == FEATURE_COLUMNS
    assert features.loc[20, "home_prior_points_for_shrunk"] == pytest.approx(25.0)
    assert features.loc[20, "home_prior_points_against_shrunk"] == pytest.approx(10.0)
    assert features.loc[20, "home_prior_weight"] == pytest.approx(1.0)
    assert features.loc[20, "home_prior_current_games"] == 0
    assert features.loc[20, "home_prior_season_games"] == 2

    assert features.loc[40, "home_prior_points_for_shrunk"] == pytest.approx(22.0)
    assert features.loc[40, "home_prior_points_against_shrunk"] == pytest.approx(12.0)
    assert features.loc[40, "home_prior_weight"] == pytest.approx(0.8)
    assert features.loc[60, "home_prior_weight"] == pytest.approx(4 / 7)


def test_target_same_date_and_future_results_do_not_change_features():
    games = _games()
    original = build_preseason_score_prior_features(games)
    changed = games.copy()
    changed.loc[[40, 50, 60], ["home_score", "away_score"]] = [
        [100, 0], [0, 100], [100, 0]
    ]
    updated = build_preseason_score_prior_features(changed)

    pd.testing.assert_series_equal(original.loc[40], updated.loc[40])


def test_missing_prior_is_explicit_and_current_season_can_seed_estimate():
    games = pd.DataFrame(
        [
            {"season": 2024, "game_date": "2024-08-31", "home_team": "New",
             "away_team": "Other", "home_score": 20, "away_score": 7},
            {"season": 2024, "game_date": "2024-09-07", "home_team": "New",
             "away_team": "Other", "home_score": 10, "away_score": 3},
        ]
    )
    features = build_preseason_score_prior_features(games)

    assert np.isnan(features.loc[0, "home_prior_points_for_shrunk"])
    assert features.loc[0, "home_prior_available"] == 0
    assert features.loc[1, "home_prior_points_for_shrunk"] == 20
    assert features.loc[1, "home_prior_weight"] == 0
    assert features.loc[1, "home_prior_current_games"] == 1


def test_invalid_prior_strength_is_rejected_and_dictionary_is_complete():
    with pytest.raises(ValueError, match="positive finite"):
        build_preseason_score_prior_features(_games(), prior_games=0)

    dictionary = preseason_score_prior_data_dictionary()
    assert dictionary["feature"].tolist() == FEATURE_COLUMNS
    assert dictionary["availability_cutoff"].eq(
        "before the target UTC calendar-date batch"
    ).all()
