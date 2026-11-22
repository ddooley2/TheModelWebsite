import numpy as np
import pandas as pd

from src.cfb_model.features.advanced import build_advanced_features
from src.cfb_model.train.score_model import (
    ADVANCED_COLUMNS,
    ADVANCED_METRICS,
    FEATURE_COLUMNS,
    MODEL_COLUMNS,
    _parse_espn_event,
    build_model_features,
    build_pregame_features,
    load_advanced_stats,
    predict_week,
    run_walk_forward_predictions,
    score_metrics,
    validate_market_free_feature_columns,
    write_baseline_run,
)
from src.cfb_model.train.experiments import run_candidate_backtest


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


def _walk_forward_games():
    return pd.DataFrame([
        {
            "game_id": 1, "season": 2013, "week": 1, "game_date": "2013-08-31T12:00Z",
            "home_team": "Alpha", "away_team": "Beta", "home_score": 28,
            "away_score": 14, "neutral_site": False,
        },
        {
            "game_id": 2, "season": 2013, "week": 1, "game_date": "2013-08-31T12:00Z",
            "home_team": "Gamma", "away_team": "Delta", "home_score": 7,
            "away_score": 21, "neutral_site": False,
        },
        {
            "game_id": 3, "season": 2014, "week": 1, "game_date": "2014-08-30T12:00Z",
            "home_team": "Alpha", "away_team": "Beta", "home_score": 31,
            "away_score": 10, "neutral_site": False,
        },
        {
            "game_id": 4, "season": 2014, "week": 1, "game_date": "2014-08-30T12:00Z",
            "home_team": "Gamma", "away_team": "Delta", "home_score": 10,
            "away_score": 24, "neutral_site": False,
        },
    ])


def test_features_use_only_prior_game_results():
    games = _games()
    features = build_pregame_features(games, games, games)

    assert features.loc[0, "home_points_for_recent"] == 0
    assert features.loc[1, "home_points_for_recent"] == 28
    assert features.loc[1, "home_points_against_recent"] == 14
    assert features.loc[1, "home_games_played"] == 1
    assert features.loc[1, "home_win_pct_season"] == 1
    assert list(features.columns) == FEATURE_COLUMNS


def test_a_game_result_does_not_change_its_own_features():
    original = build_pregame_features(_games(), _games(), _games())
    changed_games = _games()
    changed_games.loc[0, ["home_score", "away_score"]] = [3, 45]
    changed = build_pregame_features(changed_games, changed_games, changed_games)

    pd.testing.assert_series_equal(original.loc[0], changed.loc[0])


def test_future_model_and_schedule_results_do_not_change_earlier_targets():
    model_history = _games()
    schedule_history = _games()
    target = model_history.iloc[[0]].reset_index(drop=True)
    original = build_pregame_features(target, model_history, schedule_history)

    changed_model = model_history.copy()
    changed_model.loc[1, ["home_score", "away_score"]] = [100, 0]
    changed_schedule = schedule_history.copy()
    changed_schedule.loc[1, ["home_score", "away_score"]] = [100, 0]
    changed = build_pregame_features(target, changed_model, changed_schedule)

    pd.testing.assert_frame_equal(original, changed)


def test_fcs_game_updates_schedule_rest_and_game_count_not_model_stats():
    model_history = pd.DataFrame([
        {
            "season": 2024, "week": 1, "game_date": "2024-09-01T12:00Z",
            "home_team": "Alpha", "away_team": "Beta", "home_score": 28,
            "away_score": 14, "neutral_site": False,
        },
        {
            "season": 2024, "week": 3, "game_date": "2024-09-15T12:00Z",
            "home_team": "Alpha", "away_team": "Beta", "home_score": 21,
            "away_score": 24, "neutral_site": False,
        },
    ])
    schedule_history = pd.concat([
        model_history.iloc[[0]],
        pd.DataFrame([{
            "season": 2024, "week": 2, "game_date": "2024-09-08T12:00Z",
            "home_team": "Alpha", "away_team": "FCS Team", "home_score": 35,
            "away_score": 7, "neutral_site": False,
        }]),
        model_history.iloc[[1]],
    ], ignore_index=True)
    target = model_history.iloc[[1]].reset_index(drop=True)

    features = build_pregame_features(target, model_history, schedule_history)

    assert features.loc[0, "home_rest_days"] == 7
    assert features.loc[0, "home_games_played"] == 2
    assert features.loc[0, "home_points_for_season"] == 28
    assert features.loc[0, "home_win_pct_season"] == 1


def test_fbs_fcs_games_are_calendar_history_not_model_targets(tmp_path):
    import duckdb
    from src.cfb_model.train.score_model import load_schedule_history, load_historical_games

    db_path = tmp_path / "warehouse.duckdb"
    with duckdb.connect(str(db_path)) as connection:
        connection.execute("""
            CREATE TABLE stg_games_raw (
                game_id BIGINT, season BIGINT, week BIGINT, game_date VARCHAR,
                home_team VARCHAR, away_team VARCHAR, home_score DOUBLE,
                away_score DOUBLE, neutral_site BOOLEAN,
                home_is_fbs BOOLEAN, away_is_fbs BOOLEAN
            )
        """)
        connection.execute("""
            INSERT INTO stg_games_raw VALUES
            (1, 2024, 1, '2024-09-01T12:00Z', 'Alpha', 'Beta', 28, 14, FALSE, TRUE, TRUE),
            (2, 2024, 2, '2024-09-08T12:00Z', 'Alpha', 'FCS Team', 35, 7, FALSE, TRUE, FALSE),
            (3, 2024, 2, '2024-09-08T12:00Z', 'FCS One', 'FCS Two', 21, 10, FALSE, FALSE, FALSE)
        """)

    model_history = load_historical_games(db_path)
    schedule_history = load_schedule_history(db_path)

    assert model_history["game_id"].tolist() == [1]
    assert schedule_history["game_id"].tolist() == [1, 2]


def test_fbs_team_population_is_loaded_for_the_requested_season(tmp_path, monkeypatch):
    import duckdb
    import src.cfb_model.train.score_model as score_model

    db_path = tmp_path / "warehouse.duckdb"
    with duckdb.connect(str(db_path)) as connection:
        connection.execute(
            "CREATE TABLE stg_conference_history_raw (team_id BIGINT, season BIGINT)"
        )
        connection.execute(
            "CREATE TABLE stg_team_lookup_raw (team_id BIGINT, team_name VARCHAR)"
        )
        connection.execute(
            "INSERT INTO stg_conference_history_raw VALUES (1, 2024), (2, 2025)"
        )
        connection.execute(
            "INSERT INTO stg_team_lookup_raw VALUES (1, 'Old Team'), (2, 'Current Team')"
        )

    teams = score_model.load_fbs_teams(2025, db_path)

    assert teams[["team_id", "school"]].to_dict("records") == [
        {"team_id": 2, "school": "Current Team"}
    ]

    monkeypatch.setattr(score_model, "load_dotenv", lambda: None)
    monkeypatch.setenv("CFBD_API_KEY", "test")

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return [{"id": 3, "school": "New Team"}]

    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs["params"]))
        return Response()

    monkeypatch.setattr(score_model.requests, "get", get)
    fetched = score_model.load_fbs_teams(2026, db_path)

    assert fetched[["team_id", "school"]].to_dict("records") == [
        {"team_id": 3, "school": "New Team"}
    ]
    assert calls == [(f"{score_model.CFBD_API}/teams/fbs", {"year": 2026})]


def test_same_date_results_do_not_update_features_for_other_targets():
    targets = pd.DataFrame([
        {
            "season": 2024, "week": 1, "game_date": "2024-08-31T12:00Z",
            "home_team": "Alpha", "away_team": "Beta", "home_score": 70,
            "away_score": 0, "neutral_site": False,
        },
        {
            "season": 2024, "week": 1, "game_date": "2024-08-31T22:00Z",
            "home_team": "Beta", "away_team": "Gamma", "home_score": 28,
            "away_score": 14, "neutral_site": False,
        },
    ])

    features = build_pregame_features(targets, targets, targets)

    assert features.loc[1, "home_games_played"] == 0
    assert features.loc[1, "home_points_for_recent"] == 0


def test_model_features_are_aligned_and_exclude_market_inputs():
    games = _games()
    games["market_spread_home"] = [-7.0, -3.0]
    games["market_total"] = [50.0, 48.0]
    features = build_model_features(
        games, pd.DataFrame(), model_history=games, schedule_history=games
    )

    assert list(features.columns) == MODEL_COLUMNS
    assert len(features) == len(games)
    assert not any("market" in column for column in features.columns)


def test_market_feature_validator_catches_normalized_prohibited_terms():
    import pytest

    with pytest.raises(ValueError, match="over_under"):
        validate_market_free_feature_columns(["team_over_under_edge"])


def test_line_data_does_not_change_features_training_or_predictions(monkeypatch, tmp_path):
    import src.cfb_model.train.score_model as score_model

    games = _walk_forward_games()
    with_lines = games.assign(
        market_spread_home=[-1000, 1000, -1000, 1000],
        market_total=[9999, -9999, 9999, -9999],
    )
    plain_features = build_model_features(
        games, pd.DataFrame(), model_history=games, schedule_history=games
    )
    line_features = build_model_features(
        with_lines, pd.DataFrame(), model_history=with_lines, schedule_history=with_lines
    )
    pd.testing.assert_frame_equal(plain_features, line_features)

    fit_sizes = []
    candidate_inputs = []

    class Model:
        def fit(self, values, targets):
            fit_sizes.append((len(values), len(targets)))
            self.prediction = targets.mean(axis=0)

        def predict(self, values):
            return np.tile(self.prediction, (len(values), 1))

    monkeypatch.setattr(score_model, "_make_model", Model)
    def candidate_fit(train_x, train_y, valid_x):
        candidate_inputs.append(train_x.copy())
        return np.tile(train_y.mean(axis=0), (len(valid_x), 1))

    monkeypatch.setattr(
        "src.cfb_model.train.experiments.CANDIDATES",
        {"direct_score_l1": candidate_fit},
    )
    plain_metrics, plain_candidate_predictions = run_candidate_backtest(
        games, plain_features, val_seasons=[2014]
    )
    line_metrics, line_candidate_predictions = run_candidate_backtest(
        with_lines, line_features, val_seasons=[2014]
    )
    pd.testing.assert_frame_equal(plain_metrics, line_metrics)
    pd.testing.assert_frame_equal(plain_candidate_predictions, line_candidate_predictions)
    pd.testing.assert_frame_equal(candidate_inputs[0], candidate_inputs[1])

    plain_predictions = run_walk_forward_predictions(
        games, plain_features, val_seasons=[2014]
    )
    line_predictions = run_walk_forward_predictions(
        with_lines, line_features, val_seasons=[2014]
    )

    pd.testing.assert_frame_equal(plain_predictions, line_predictions)
    assert fit_sizes == [(2, 2), (2, 2)]
    assert len(plain_predictions) == 2
    assert plain_predictions["game_id"].tolist() == [3, 4]
    assert all(plain_predictions["prediction_cutoff_utc"].str.endswith("+00:00"))


def test_baseline_run_writes_candidate_sha_and_row_level_artifacts(monkeypatch, tmp_path):
    import json
    import src.cfb_model.train.score_model as score_model

    games = _walk_forward_games()
    features = pd.DataFrame(0.0, index=games.index, columns=MODEL_COLUMNS)

    class Model:
        def fit(self, values, targets):
            self.prediction = targets.mean(axis=0)

        def predict(self, values):
            return np.tile(self.prediction, (len(values), 1))

    monkeypatch.setattr(score_model, "_make_model", Model)
    run_dir = tmp_path / "baseline"
    manifest = write_baseline_run(games, features, run_dir, val_seasons=[2014])

    saved = json.loads((run_dir / "manifest.json").read_text())
    assert saved["candidate_id"] == "direct_score_l1"
    assert saved["source_commit_sha"]
    assert "source_worktree_dirty" in saved
    assert saved["market_features_enabled"] is False
    assert saved["market_independence_check_passed"] is True
    assert manifest["game_count"] == 2
    assert len(pd.read_csv(run_dir / "predictions.csv")) == 2
    assert (run_dir / "metrics_by_season.csv").exists()


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
    assert game["home_is_fbs"] is True
    assert game["away_is_fbs"] is True


def _espn_event(event_id, home_name, away_name, completed=True):
    return {
        "id": event_id,
        "date": "2026-09-05T16:00:00Z",
        "competitions": [{
            "neutralSite": False,
            "competitors": [
                {
                    "homeAway": "home",
                    "team": {
                        "id": f"espn-{home_name.lower().replace(' ', '-')}",
                        "location": home_name,
                        "abbreviation": "",
                    },
                    "score": "28",
                },
                {
                    "homeAway": "away",
                    "team": {
                        "id": (
                            "espn-beta" if away_name == "Alphaa"
                            else f"espn-{away_name.lower().replace(' ', '-')}"
                        ),
                        "location": away_name,
                        "abbreviation": "",
                    },
                    "score": "14",
                },
            ],
        }],
        "status": {"type": {"completed": completed}},
    }


def test_load_espn_season_keeps_fcs_game_for_schedule_and_maps_fbs_ids(monkeypatch):
    import src.cfb_model.train.score_model as score_model

    events = [
        _espn_event("fbs-game", "Alpha", "Beta"),
        _espn_event("fcs-game", "Alpha", "FCS Team"),
    ]

    class Response:
        def __init__(self, week=None, is_teams=False):
            self.week = week
            self.is_teams = is_teams

        def raise_for_status(self):
            pass

        def json(self):
            if self.is_teams:
                return {"teams": [
                    {"team": {"id": "espn-alpha", "location": "Alpha"}},
                    {"team": {"id": "espn-beta", "location": "Beta"}},
                ]}
            if self.week is None:
                return {"events": []}
            return {"events": events if self.week == 1 else []}

    class Session:
        def get(self, *args, **kwargs):
            if args[0] == score_model.ESPN_TEAMS:
                return Response(is_teams=True)
            return Response(kwargs["params"].get("week")) if args[0] == score_model.ESPN_SCOREBOARD else Response()

    monkeypatch.setattr(score_model.requests, "Session", Session)
    teams = pd.DataFrame([
        {"team_id": 1, "school": "Alpha"},
        {"team_id": 2, "school": "Beta"},
    ])

    schedule = score_model.load_espn_season(2026, teams)

    assert len(schedule) == 2
    assert schedule["home_is_fbs"].all()
    assert schedule.loc[schedule["game_id"] == "espn-fbs-game", "away_is_fbs"].item()
    assert not schedule.loc[schedule["game_id"] == "espn-fcs-game", "away_is_fbs"].item()
    mapping = schedule.attrs["provider_team_mapping"]
    assert set(mapping["canonical_team_id"]) == {1, 2}
    assert set(mapping["provider_team_name"]) == {"Alpha", "Beta"}


def test_espn_schedule_fails_and_logs_unmatched_potential_fbs_team(monkeypatch, caplog):
    import pytest
    import src.cfb_model.train.score_model as score_model

    class Response:
        def __init__(self, is_teams=False):
            self.is_teams = is_teams

        def raise_for_status(self):
            pass

        def json(self):
            if self.is_teams:
                return {"teams": [
                    {"team": {"id": "espn-alpha", "location": "Alpha"}},
                    {"team": {"id": "espn-beta", "location": "Beta"}},
                ]}
            return {"events": [_espn_event("unknown-game", "Alpha", "Alphaa")]}

    class Session:
        def get(self, *args, **kwargs):
            if args[0] == score_model.ESPN_TEAMS:
                return Response(is_teams=True)
            return Response() if kwargs["params"]["week"] == 1 else type(
                "EmptyResponse", (), {"raise_for_status": lambda self: None,
                                      "json": lambda self: {"events": []}}
            )()

    monkeypatch.setattr(score_model.requests, "Session", Session)
    teams = pd.DataFrame([
        {"team_id": 1, "school": "Alpha"},
        {"team_id": 2, "school": "Beta"},
    ])

    with pytest.raises(ValueError, match="Unmatched FBS teams"):
        score_model.load_espn_season(2026, teams)
    assert "unknown-game" in caplog.text
    assert "Alphaa" in caplog.text
    assert "mapping_candidate=Alpha" in caplog.text


def test_advanced_features_use_prior_current_season_stats_and_prior_year_shrinkage():
    games = pd.DataFrame([
        {
            "game_id": 1, "season": 2025, "week": 1, "game_date": "2025-08-30",
            "home_team": "Alpha", "away_team": "Beta",
        },
        {
            "game_id": 2, "season": 2026, "week": 1, "game_date": "2026-08-29",
            "home_team": "Alpha", "away_team": "Beta",
        },
        {
            "game_id": 3, "season": 2026, "week": 2, "game_date": "2026-09-05",
            "home_team": "Alpha", "away_team": "Beta",
        },
    ])
    advanced_rows = []
    for game_id, season, week, team, ppa in [
        (1, 2025, 1, "Alpha", 0.1),
        (1, 2025, 1, "Beta", 0.2),
        (np.nan, 2026, 1, "Alpha", 0.5),
        (np.nan, 2026, 1, "Beta", 0.6),
        (np.nan, 2026, 2, "Alpha", 99.0),
        (np.nan, 2026, 2, "Beta", 99.0),
    ]:
        advanced_rows.append({
            "game_id": game_id, "season": season, "week": week, "team": team,
            **{
                metric: ppa if metric == "offense_ppa" else 0.5
                for metric in ADVANCED_METRICS
            },
        })

    features = build_advanced_features(games, pd.DataFrame(advanced_rows))

    assert np.isclose(features.loc[2, "home_offense_ppa"], 0.2)
    assert np.isclose(features.loc[2, "away_offense_ppa"], 0.3)


def test_advanced_features_exclude_other_results_from_the_same_date():
    games = pd.DataFrame([
        {
            "game_id": 1, "season": 2026, "week": 1, "game_date": "2026-09-05T12:00Z",
            "home_team": "Alpha", "away_team": "Beta",
        },
        {
            "game_id": 2, "season": 2026, "week": 1, "game_date": "2026-09-05T22:00Z",
            "home_team": "Alpha", "away_team": "Gamma",
        },
    ])
    advanced = pd.DataFrame([
        {
            "game_id": game_id, "season": 2026, "week": 1, "team": team,
            **{metric: value for metric in ADVANCED_METRICS},
        }
        for game_id, teams, value in (
            (1, ("Alpha", "Beta"), 99.0),
            (2, ("Alpha", "Gamma"), 0.5),
        )
        for team in teams
    ])

    features = build_advanced_features(games, advanced)

    assert pd.isna(features.loc[1, "home_offense_ppa"])


def test_load_advanced_stats_refreshes_partial_requested_season(tmp_path, monkeypatch):
    import duckdb
    import src.cfb_model.train.score_model as score_model

    db_path = tmp_path / "warehouse.duckdb"
    with duckdb.connect(str(db_path)) as connection:
        connection.execute(
            "CREATE TABLE stg_team_game_stats_raw ("
            "game_id BIGINT, season INTEGER, week INTEGER, team VARCHAR, "
            + ", ".join(f"{metric} DOUBLE" for metric in ADVANCED_METRICS)
            + ")"
        )
        connection.execute(
            "INSERT INTO stg_team_game_stats_raw VALUES "
            "(1, 2025, 1, 'Alpha', 0.1, 0.1, 0.1, 0.1, 0.1, 0.1), "
            "(2, 2026, 1, 'Alpha', 0.2, 0.2, 0.2, 0.2, 0.2, 0.2)"
        )

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return [
                {
                    "season": 2026, "week": 1, "team": "Alpha",
                    "offense": {"ppa": 0.4, "successRate": 0.5, "explosiveness": 1.2},
                    "defense": {"ppa": -0.1, "successRate": 0.3, "explosiveness": 0.9},
                },
                {
                    "season": 2026, "week": 2, "team": "Beta",
                    "offense": {"ppa": 0.7, "successRate": 0.6, "explosiveness": 1.3},
                    "defense": {"ppa": -0.2, "successRate": 0.4, "explosiveness": 0.8},
                },
            ]

    monkeypatch.setattr(score_model, "load_dotenv", lambda: None)
    monkeypatch.setattr(score_model.requests, "get", lambda *args, **kwargs: Response())
    monkeypatch.setenv("CFBD_API_KEY", "test")

    stats = load_advanced_stats(db_path, season=2026)
    current = stats[stats["season"] == 2026].set_index(["week", "team"])

    assert stats[stats["season"] == 2025].shape[0] == 1
    assert current.loc[(1, "Alpha"), "offense_ppa"] == 0.4
    assert current.loc[(2, "Beta"), "offense_ppa"] == 0.7
    assert len(current) == 2


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
    scheduled = pd.DataFrame([
        {
            "game_id": 2, "season": 2026, "week": 8, "game_date": "2026-10-24",
            "home_team": "Alpha", "away_team": "Beta", "home_score": np.nan,
            "away_score": np.nan, "neutral_site": False,
            "home_is_fbs": True, "away_is_fbs": True,
        },
        {
            "game_id": 4, "season": 2026, "week": 7, "game_date": "2026-10-17",
            "home_team": "Alpha", "away_team": "FCS Team", "home_score": 35,
            "away_score": 7, "neutral_site": False,
            "home_is_fbs": True, "away_is_fbs": False,
        },
    ])
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

    captured = {}
    def build_features(target_games, *args, **kwargs):
        captured["targets"] = target_games.copy()
        captured["model_history"] = kwargs["model_history"].copy()
        captured["schedule_history"] = kwargs["schedule_history"].copy()
        return features

    monkeypatch.setattr(score_model, "ensure_warehouse", lambda path: path)
    monkeypatch.setattr(score_model, "load_historical_games", lambda path: historical)
    monkeypatch.setattr(score_model, "load_schedule_history", lambda path: historical)
    monkeypatch.setattr(
        score_model,
        "load_fbs_teams",
        lambda season, path: pd.DataFrame({
            "team_id": [1, 2, 3, 4],
            "school": ["Alpha", "Beta", "Gamma", "Delta"],
            "season": [season] * 4,
        }),
    )
    monkeypatch.setattr(score_model, "load_espn_season", lambda season, teams: scheduled)
    monkeypatch.setattr(score_model, "load_advanced_stats", lambda *args: advanced)
    monkeypatch.setattr(score_model, "load_roster_priors", lambda *args: pd.DataFrame())
    monkeypatch.setattr(score_model, "build_model_features", build_features)
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
    assert 4 not in captured["targets"]["game_id"].tolist()
    assert 4 not in captured["model_history"]["game_id"].tolist()
    assert 4 in captured["schedule_history"]["game_id"].tolist()
