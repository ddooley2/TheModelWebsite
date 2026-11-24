import numpy as np
import pandas as pd

from src.cfb_model.features.opponent_adjusted import (
    FEATURE_COLUMNS,
    build_opponent_adjusted_features,
    build_recency_weighted_opponent_adjusted_features,
    build_unweighted_opponent_adjusted_features,
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
    named_baseline = build_unweighted_opponent_adjusted_features(games)

    assert list(features.columns) == FEATURE_COLUMNS
    pd.testing.assert_frame_equal(features, named_baseline)
    assert features.index.tolist() == games.index.tolist()
    assert len(features) == len(games)
    # The two September 7 games share exactly the same pre-date team ratings.
    assert features.loc[4, "home_offense_rating"] == features.loc[19, "away_offense_rating"]
    assert features.loc[4, "home_defense_rating"] == features.loc[19, "away_defense_rating"]
    assert "market_spread" not in features.columns
    np.testing.assert_allclose(
        features.loc[2].to_numpy(),
        [
            -0.10530774354336778,
            -0.01970971269801734,
            -0.8089227794745109,
            0.9339402357158959,
            0.703615035931143,
            -0.9536499484139133,
        ],
    )


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


def _recency_games():
    return pd.DataFrame(
        [
            {
                "game_id": "old-game",
                "game_date": "2024-08-31",
                "season": 2024,
                "home_team": "Alpha",
                "away_team": "Beta",
                "home_score": 35,
                "away_score": 7,
                "neutral_site": False,
                "home_is_fbs": True,
                "away_is_fbs": True,
            },
            {
                "game_id": "target",
                "game_date": "2024-09-07",
                "season": 2024,
                "home_team": "Alpha",
                "away_team": "Gamma",
                "home_score": float("nan"),
                "away_score": float("nan"),
                "neutral_site": False,
                "home_is_fbs": True,
                "away_is_fbs": True,
            },
        ]
    )


def _recency_features(games):
    return build_recency_weighted_opponent_adjusted_features(
        games,
        half_life_days=56,
        season_prior_strength=0.0,
        ridge_alpha=20.0,
    )


def test_recency_weighting_reduces_the_effect_of_an_older_observation():
    recent = pd.DataFrame(
        [
            {
                "game_id": "first",
                "game_date": "2024-08-31",
                "season": 2024,
                "home_team": "Alpha",
                "away_team": "Beta",
                "home_score": 35,
                "away_score": 7,
                "neutral_site": False,
                "home_is_fbs": True,
                "away_is_fbs": True,
            },
            {
                "game_id": "second",
                "game_date": "2024-09-07",
                "season": 2024,
                "home_team": "Gamma",
                "away_team": "Alpha",
                "home_score": 7,
                "away_score": 35,
                "neutral_site": False,
                "home_is_fbs": True,
                "away_is_fbs": True,
            },
            {
                "game_id": "third",
                "game_date": "2024-09-14",
                "season": 2024,
                "home_team": "Beta",
                "away_team": "Gamma",
                "home_score": 14,
                "away_score": 21,
                "neutral_site": False,
                "home_is_fbs": True,
                "away_is_fbs": True,
            },
            {
                "game_id": "target",
                "game_date": "2024-09-21",
                "season": 2024,
                "home_team": "Alpha",
                "away_team": "Gamma",
                "home_score": float("nan"),
                "away_score": float("nan"),
                "neutral_site": False,
                "home_is_fbs": True,
                "away_is_fbs": True,
            },
        ]
    )
    older = recent.copy()
    older.loc[0, ["game_date", "season"]] = ["2020-08-31", 2020]

    recent_rating = _recency_features(recent).loc[3, "home_offense_rating"]
    older_rating = _recency_features(older).loc[3, "home_offense_rating"]

    assert recent_rating > older_rating


def test_weighted_features_exclude_target_future_and_same_date_results():
    games = _recency_games()
    games.loc[1, "game_date"] = "2024-09-07T20:00:00Z"
    games = pd.concat(
        [
            games,
            pd.DataFrame(
                [
                    {
                        "game_id": "same-date",
                        "game_date": "2024-09-07T12:00:00Z",
                        "season": 2024,
                        "home_team": "Alpha",
                        "away_team": "Gamma",
                        "home_score": 0,
                        "away_score": 70,
                        "neutral_site": False,
                        "home_is_fbs": True,
                        "away_is_fbs": True,
                    },
                    {
                        "game_id": "future",
                        "game_date": "2024-09-14",
                        "season": 2024,
                        "home_team": "Alpha",
                        "away_team": "Gamma",
                        "home_score": 63,
                        "away_score": 3,
                        "neutral_site": False,
                        "home_is_fbs": True,
                        "away_is_fbs": True,
                    },
                ]
            ),
        ],
        ignore_index=True,
    )

    original = _recency_features(games).loc[1]
    truncated = _recency_features(games.iloc[:2]).loc[1]
    changed = games.copy()
    changed.loc[[1, 2, 3], ["home_score", "away_score"]] = [
        [100, 0],
        [70, 0],
        [3, 63],
    ]
    updated = _recency_features(changed).loc[1]
    pd.testing.assert_series_equal(original, truncated)
    pd.testing.assert_series_equal(original, updated)


def test_weighted_ratings_ignore_ineligible_fbs_fcs_observations():
    games = _recency_games()
    fbs_fcs_game = pd.DataFrame(
        [
            {
                "game_id": "fbs-fcs",
                "game_date": "2024-09-03",
                "season": 2024,
                "home_team": "Alpha",
                "away_team": "FCS Team",
                "home_score": 0,
                "away_score": 70,
                "neutral_site": False,
                "home_is_fbs": True,
                "away_is_fbs": False,
            }
        ]
    )
    with_ineligible = pd.concat([games, fbs_fcs_game], ignore_index=True)

    expected = _recency_features(games).loc[1]
    actual = _recency_features(with_ineligible).loc[1]

    pd.testing.assert_series_equal(expected, actual)


def test_weighted_features_have_finite_empty_and_one_game_fallbacks():
    empty = _recency_features(_recency_games().iloc[0:0])
    one_game = _recency_features(_recency_games().iloc[[0]])

    assert empty.empty
    assert one_game.to_numpy().tolist() == [[0.0] * len(FEATURE_COLUMNS)]
    assert np.isfinite(empty.to_numpy()).all()
    assert np.isfinite(one_game.to_numpy()).all()


def test_season_prior_is_fixed_at_the_season_boundary():
    games = _recency_games()
    games.loc[0, ["game_date", "season"]] = ["2023-08-31", 2023]
    without_prior = build_recency_weighted_opponent_adjusted_features(
        games,
        half_life_days=56,
        season_prior_strength=0.0,
        ridge_alpha=20.0,
    )
    with_prior = build_recency_weighted_opponent_adjusted_features(
        games,
        half_life_days=56,
        season_prior_strength=20.0,
        ridge_alpha=20.0,
    )

    assert without_prior.loc[1, "home_offense_rating"] != with_prior.loc[1, "home_offense_rating"]
