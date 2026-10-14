import numpy as np
import pandas as pd

from src.cfb_model.train.score_model import (
    FEATURE_COLUMNS,
    MODEL_COLUMNS,
    _parse_espn_event,
    build_model_features,
    build_pregame_features,
    score_metrics,
)


def _games():
    return pd.DataFrame(
        [
            {
                "season": 2024,
                "week": 1,
                "game_date": "2024-08-31T12:00:00Z",
                "home_team": "Alpha",
                "away_team": "Beta",
                "home_score": 28,
                "away_score": 14,
                "neutral_site": False,
            },
            {
                "season": 2024,
                "week": 2,
                "game_date": "2024-09-07T12:00:00Z",
                "home_team": "Alpha",
                "away_team": "Beta",
                "home_score": 21,
                "away_score": 24,
                "neutral_site": False,
            },
        ]
    )


def test_features_use_only_prior_game_results():
    games = _games()
    features = build_pregame_features(games)

    assert features.loc[0, "home_points_for_recent"] == 0
    assert features.loc[1, "home_points_for_recent"] == 28
    assert features.loc[1, "home_points_against_recent"] == 14
    assert features.loc[1, "home_games_played"] == 1
    assert features.loc[1, "home_win_pct_season"] == 1
    assert list(features.columns) == FEATURE_COLUMNS


def test_a_game_result_does_not_change_its_own_features():
    original = build_pregame_features(_games())
    changed_games = _games()
    changed_games.loc[0, ["home_score", "away_score"]] = [3, 45]
    changed = build_pregame_features(changed_games)

    pd.testing.assert_series_equal(original.loc[0], changed.loc[0])


def test_model_features_are_aligned_and_exclude_market_inputs():
    games = _games()
    games["market_spread_home"] = [-7.0, -3.0]
    games["market_total"] = [50.0, 48.0]
    features = build_model_features(games, pd.DataFrame())

    assert list(features.columns) == MODEL_COLUMNS
    assert len(features) == len(games)
    assert not any("market" in column for column in features.columns)


def test_score_metrics_report_score_and_derived_quality():
    actual = np.array([[28, 14], [17, 20], [35, 10]], dtype=float)
    predicted = np.array([[26, 16], [19, 18], [33, 12]], dtype=float)

    metrics = score_metrics(actual, predicted)

    assert metrics["score_mae"] == 2
    assert metrics["margin_mae"] == 4
    assert metrics["total_mae"] == 0
    assert metrics["winner_auc"] == 1


def test_espn_event_parser_keeps_scheduled_score_targets_missing():
    event = {
        "id": "123",
        "date": "2026-10-24T16:00Z",
        "competitions": [
            {
                "neutralSite": False,
                "competitors": [
                    {
                        "homeAway": "home",
                        "team": {"location": "Alpha", "abbreviation": "ALP"},
                        "score": "0",
                    },
                    {
                        "homeAway": "away",
                        "team": {"location": "Beta", "abbreviation": "BET"},
                        "score": "0",
                    },
                ],
            }
        ],
        "status": {"type": {"completed": False}},
    }

    game = _parse_espn_event(
        event,
        season=2026,
        week=8,
        team_lookup={"alpha": "Alpha", "beta": "Beta"},
    )

    assert game["game_id"] == "espn-123"
    assert np.isnan(game["home_score"])
    assert np.isnan(game["away_score"])
