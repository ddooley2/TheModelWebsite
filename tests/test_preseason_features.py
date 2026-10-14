import numpy as np
import pandas as pd

from src.cfb_model.features.preseason import (
    FEATURE_COLUMNS,
    build_preseason_features,
)


def _games():
    return pd.DataFrame(
        [
            {
                "season": 2023,
                "game_date": "2023-09-02",
                "home_team": "Alpha",
                "away_team": "Beta",
                "home_score": 28,
                "away_score": 14,
            },
            {
                "season": 2023,
                "game_date": "2023-09-02",
                "home_team": "Gamma",
                "away_team": "Delta",
                "home_score": 10,
                "away_score": 20,
            },
            {
                "season": 2023,
                "game_date": "2023-09-09",
                "home_team": "Alpha",
                "away_team": "Gamma",
                "home_score": 17,
                "away_score": 21,
            },
            {
                "season": 2024,
                "game_date": "2024-08-31",
                "home_team": "Alpha",
                "away_team": "Beta",
                "home_score": 7,
                "away_score": 3,
            },
        ]
    )


def test_same_date_results_wait_until_all_features_are_emitted():
    games = _games()
    original = build_preseason_features(games)
    changed = games.copy()
    changed.loc[0, ["home_score", "away_score"]] = [0, 50]
    changed.loc[1, ["home_score", "away_score"]] = [40, 0]
    changed_features = build_preseason_features(changed)

    pd.testing.assert_frame_equal(original.iloc[:2], changed_features.iloc[:2])
    assert original.loc[2, "home_elo"] != changed_features.loc[2, "home_elo"]


def test_own_and_future_results_do_not_change_pregame_features():
    games = _games()
    original = build_preseason_features(games)

    own_result_changed = games.copy()
    own_result_changed.loc[2, ["home_score", "away_score"]] = [50, 0]
    changed = build_preseason_features(own_result_changed)
    pd.testing.assert_series_equal(original.loc[2], changed.loc[2])

    future_result_changed = games.copy()
    future_result_changed.loc[3, ["home_score", "away_score"]] = [0, 50]
    changed = build_preseason_features(future_result_changed)
    pd.testing.assert_frame_equal(original.iloc[:3], changed.iloc[:3])


def test_season_carryover_regresses_final_elo_toward_1500():
    games = pd.DataFrame(
        [
            {
                "season": 2023,
                "game_date": "2023-09-01",
                "home_team": "Alpha",
                "away_team": "Beta",
                "home_score": 1,
                "away_score": 0,
            },
            {
                "season": 2024,
                "game_date": "2024-09-01",
                "home_team": "Alpha",
                "away_team": "Beta",
                "home_score": np.nan,
                "away_score": np.nan,
            },
        ]
    )

    features = build_preseason_features(games)

    assert features.loc[0, "home_elo"] == 1500
    assert features.loc[1, "home_elo"] == 1507.5
    assert features.loc[1, "away_elo"] == 1492.5


def test_first_seen_teams_and_missing_roster_fields_are_preserved():
    games = pd.DataFrame(
        [
            {
                "season": 2025,
                "game_date": "2025-08-30",
                "home_team": "New Team",
                "away_team": "No Roster",
                "home_score": np.nan,
                "away_score": np.nan,
            }
        ],
        index=[17],
    )
    priors = pd.DataFrame(
        [
            {
                "season": 2025,
                "team": "New Team",
                "team_talent": 900,
                "recruiting_rank": 10,
                "recruiting_points": 250,
                "totalPPA": 1e9,
                "percentPPA": 1e9,
            }
        ]
    )

    features = build_preseason_features(games, priors)

    assert list(features.index) == [17]
    assert tuple(features.columns) == FEATURE_COLUMNS
    assert features.loc[17, "home_elo"] == features.loc[17, "away_elo"] == 1500
    assert features.loc[17, "home_team_talent"] == 900
    assert features.loc[17, "home_recruiting_rank"] == 10
    assert features.loc[17, "home_recruiting_points"] == 250
    assert np.isnan(features.loc[17, "away_team_talent"])
    assert np.isnan(features.loc[17, "away_recruiting_rank"])
    assert np.isnan(features.loc[17, "away_recruiting_points"])


def test_features_remain_aligned_after_input_reordering_and_keep_missing_games():
    canonical = build_preseason_features(_games())
    games = _games().iloc[[3, 0, 2, 1]].copy()
    games.index = [41, 7, 99, 12]
    games.loc[41, ["home_score", "away_score"]] = [np.nan, np.nan]

    features = build_preseason_features(games)
    expected = canonical.iloc[[3, 0, 2, 1]].copy()
    expected.index = games.index

    assert list(features.index) == [41, 7, 99, 12]
    assert tuple(features.columns) == FEATURE_COLUMNS
    pd.testing.assert_frame_equal(features, expected)
    assert np.isnan(features["home_team_talent"]).all()
