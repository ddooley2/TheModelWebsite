import numpy as np
import pandas as pd

from src.cfb_model.train.score_model import (
    ADVANCED_COLUMNS,
    ADVANCED_METRICS,
    FEATURE_COLUMNS,
    MODEL_COLUMNS,
    _parse_espn_event,
    build_model_features,
    build_pregame_features,
    predict_week,
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


def test_predict_week_saves_prediction_time_features(monkeypatch, tmp_path):
    import src.cfb_model.train.score_model as score_model

    historical = pd.DataFrame([
        {
            "game_id": 1, "season": 2025, "week": 1, "game_date": "2025-08-30",
            "home_team": "Alpha", "away_team": "Beta", "home_score": 28,
            "away_score": 14, "neutral_site": False,
        },
        {
            "game_id": 3, "season": 2025, "week": 1, "game_date": "2025-08-30",
            "home_team": "Gamma", "away_team": "Delta", "home_score": 21,
            "away_score": 7, "neutral_site": False,
        },
    ])
    scheduled = pd.DataFrame([{
        "game_id": 2, "season": 2026, "week": 8, "game_date": "2026-10-24",
        "home_team": "Alpha", "away_team": "Beta", "home_score": np.nan,
        "away_score": np.nan, "neutral_site": False,
    }])
    features = pd.DataFrame(0.5, index=range(3), columns=MODEL_COLUMNS)
    features.loc[2, "home_points_for_season"] = 28.0
    features.loc[2, "home_points_against_season"] = 14.0
    features.loc[2, "away_points_for_season"] = 14.0
    features.loc[2, "away_points_against_season"] = 28.0
    for column in ADVANCED_COLUMNS:
        features.loc[2, column] = 0.25

    advanced_rows = []
    for game_id, week, team, ppa in [
        (11, 1, "Alpha", 0.2),
        (12, 2, "Alpha", 0.4),
        (13, 1, "Beta", 0.1),
        (14, 2, "Beta", 0.2),
        (15, 1, "Gamma", 0.8),
        (16, 1, "Delta", -0.1),
        (17, 8, "Alpha", 9.0),
        (18, 1, "FCS Team", 100.0),
    ]:
        advanced_rows.append({
            "game_id": game_id, "season": 2026, "week": week, "team": team,
            **{
                metric: ppa if metric == "offense_ppa" else 0.5
                for metric in ADVANCED_METRICS
            },
        })
    advanced = pd.DataFrame(advanced_rows)

    class Model:
        def fit(self, *args):
            pass

        def predict(self, values):
            return np.array([[30.0, 20.0]])

    monkeypatch.setattr(score_model, "ensure_warehouse", lambda path: path)
    monkeypatch.setattr(score_model, "load_historical_games", lambda path: historical)
    monkeypatch.setattr(score_model, "load_espn_season", lambda season, teams: scheduled)
    monkeypatch.setattr(score_model, "load_advanced_stats", lambda *args: advanced)
    monkeypatch.setattr(score_model, "load_roster_priors", lambda *args: pd.DataFrame())
    monkeypatch.setattr(score_model, "build_model_features", lambda *args: features)
    monkeypatch.setattr(score_model, "_make_model", Model)

    predictions = predict_week(2026, 8, tmp_path / "warehouse.duckdb")

    assert predictions.loc[0, "home_points_against_season"] == 14.0
    assert predictions.loc[0, "away_points_for_season"] == 14.0
    assert all(predictions.loc[0, column] == 0.25 for column in ADVANCED_COLUMNS)
    assert np.isclose(predictions.loc[0, "home_display_offense_ppa"], 0.3)
    assert np.isclose(predictions.loc[0, "away_display_offense_ppa"], 0.15)
    assert np.isclose(predictions.loc[0, "advanced_offense_ppa_min"], -0.1)
    assert np.isclose(predictions.loc[0, "advanced_offense_ppa_max"], 0.8)
    assert predictions.loc[0, "predicted_home_score"] == 30.0
