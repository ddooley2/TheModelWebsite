import json

import pandas as pd
import pytest

import src.cfb_model.site.build_site as site_builder
from src.cfb_model.site.build_site import (
    _grade_ats,
    _grade_straight_up,
    _market_payload,
    _performance,
    _snapshot_lines,
)


def test_line_snapshot_preserves_first_observation_as_opening():
    history = {}
    initial = [{"id": 9, "lines": [{"provider": "DraftKings", "spread": -3.5, "overUnder": 48.5}]}]
    by_game, changed = _snapshot_lines(initial, history, "2026-09-01T00:00:00+00:00")
    assert changed
    assert by_game["9"]["id"] == 9

    later = [{"id": 9, "lines": [{"provider": "DraftKings", "spread": -5.0, "overUnder": 49.5}]}]
    _, changed = _snapshot_lines(later, history, "2026-09-02T00:00:00+00:00")
    assert changed

    market = _market_payload(later[0], history, "Alpha", "Beta")
    assert market["opening"]["spread_favorite"] == -3.5
    assert market["opening"]["favorite_team"] == "Alpha"
    assert market["current"]["spread_favorite"] == -5.0
    assert market["current"]["favorite_team"] == "Alpha"
    assert market["provider"] == "DraftKings"


def test_market_payload_converts_away_favorite_to_favorite_perspective():
    game = {"id": 9, "lines": [{"provider": "DraftKings", "spread": 6.5, "overUnder": 44.5}]}
    history = {}
    _snapshot_lines([game], history, "2026-09-01T00:00:00+00:00")

    market = _market_payload(game, history, "Alpha", "Beta")

    assert market["favorite_team"] == "Beta"
    assert market["underdog_team"] == "Alpha"
    assert market["opening"]["spread_favorite"] == -6.5
    assert market["current"]["spread_favorite"] == -6.5
    assert market["current"]["total"] == 44.5


def test_line_snapshot_uses_published_opening_line_once():
    history = {}
    first = [{
        "id": 9,
        "lines": [{
            "provider": "DraftKings",
            "spread": 7.5,
            "overUnder": 48.5,
            "spreadOpen": 6.5,
            "overUnderOpen": 46.5,
        }],
    }]
    _snapshot_lines(first, history, "2026-09-01T00:00:00+00:00")
    later = [{
        "id": 9,
        "lines": [{"provider": "DraftKings", "spread": 8.5, "overUnder": 49.5}],
    }]
    _snapshot_lines(later, history, "2026-09-02T00:00:00+00:00")

    market = _market_payload(later[0], history, "Alpha", "Beta")

    assert market["opening"]["spread_favorite"] == -6.5
    assert market["opening"]["total"] == 46.5
    assert market["current"]["spread_favorite"] == -8.5
    assert market["current"]["total"] == 49.5


def test_site_payload_uses_favorite_perspective_and_current_total(tmp_path, monkeypatch):
    prediction_path = tmp_path / "week9.csv"
    pd.DataFrame([{
        "game_id": "espn-9",
        "season": 2026,
        "week": 9,
        "game_date": "2026-10-31T12:00:00Z",
        "home_team": "Alpha",
        "away_team": "Beta",
        "predicted_home_score": 20.0,
        "predicted_away_score": 30.0,
        "predicted_home_margin": -10.0,
        "predicted_total": 50.0,
    }]).to_csv(prediction_path, index=False)
    game = {
        "id": 9,
        "homeTeam": "Alpha",
        "homeId": 12,
        "awayTeam": "Beta",
        "awayId": 34,
        "startDate": "2026-10-31T12:00:00Z",
        "lines": [{"provider": "DraftKings", "spread": 9.0, "overUnder": 44.5}],
    }
    monkeypatch.setattr(site_builder, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(site_builder, "SITE_DIR", tmp_path / "public")
    monkeypatch.setattr(site_builder, "DATA_DIR", tmp_path / "public" / "data")
    monkeypatch.setattr(site_builder, "HISTORY_PATH", tmp_path / "line_history.json")
    monkeypatch.setattr(site_builder, "ATS_RESULTS_PATH", tmp_path / "ats_results.json")
    monkeypatch.setattr(site_builder, "_prediction_file", lambda season, week: prediction_path)
    monkeypatch.setattr(site_builder, "_api_get", lambda endpoint, params, api_key: (
        [game] if endpoint in {"games", "lines"} else []
    ))
    monkeypatch.setattr(site_builder, "_read_history", lambda: {})
    monkeypatch.setattr(site_builder, "_write_history", lambda history: None)

    output = site_builder.build_site(2026, 9, None, False)
    payload = json.loads(output.read_text())
    result = payload["games"][0]

    assert result["favorite_team"] == "Beta"
    assert result["underdog_team"] == "Alpha"
    assert result["home"]["logo_id"] == 12
    assert result["away"]["logo_id"] == 34
    assert result["projected_favorite_margin"] == 10
    assert result["market"]["current"]["spread_favorite"] == -9.0
    assert result["market"]["current"]["total"] == 44.5
    assert result["model_edge_favorite"] == 1.0
    assert result["pick"] == "Beta"


def test_line_refresh_preserves_prediction_timestamp(tmp_path, monkeypatch):
    prediction_path = tmp_path / "week9.csv"
    pd.DataFrame([{
        "game_id": "espn-9",
        "season": 2026,
        "week": 9,
        "game_date": "2026-10-31T12:00:00Z",
        "home_team": "Alpha",
        "away_team": "Beta",
        "predicted_home_score": 20.0,
        "predicted_away_score": 30.0,
        "predicted_home_margin": -10.0,
        "predicted_total": 50.0,
    }]).to_csv(prediction_path, index=False)
    line_game = {
        "id": 9,
        "homeTeam": "Alpha",
        "awayTeam": "Beta",
        "lines": [{"provider": "DraftKings", "spread": 9.0, "overUnder": 44.5}],
    }
    dashboard_dir = tmp_path / "public" / "data"
    dashboard_dir.mkdir(parents=True)
    (dashboard_dir / "dashboard.json").write_text(json.dumps({
        "generated_at": "2026-10-26T14:17:00+00:00",
        "season": 2026,
        "week": 9,
        "games": [],
    }))
    monkeypatch.setattr(site_builder, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(site_builder, "SITE_DIR", tmp_path / "public")
    monkeypatch.setattr(site_builder, "DATA_DIR", dashboard_dir)
    monkeypatch.setattr(site_builder, "HISTORY_PATH", tmp_path / "line_history.json")
    monkeypatch.setattr(site_builder, "ATS_RESULTS_PATH", tmp_path / "ats_results.json")
    monkeypatch.setattr(site_builder, "_prediction_file", lambda season, week: prediction_path)
    monkeypatch.setattr(site_builder, "_api_get", lambda endpoint, params, api_key: (
        [line_game] if endpoint == "lines" else []
    ))
    monkeypatch.setattr(site_builder, "_read_history", lambda: {})
    monkeypatch.setattr(site_builder, "_write_history", lambda history: None)

    output = site_builder.build_site(2026, 9, None, False)

    payload = json.loads(output.read_text())
    assert payload["generated_at"] == "2026-10-26T14:17:00+00:00"
    assert payload["games"][0]["market"]["current"]["observed_at"] != payload["generated_at"]


def test_new_predictions_refresh_prediction_timestamp(tmp_path, monkeypatch):
    prediction_path = tmp_path / "week9.csv"
    pd.DataFrame([{
        "game_id": "espn-9",
        "season": 2026,
        "week": 9,
        "game_date": "2026-10-31T12:00:00Z",
        "home_team": "Alpha",
        "away_team": "Beta",
        "predicted_home_score": 20.0,
        "predicted_away_score": 30.0,
        "predicted_home_margin": -10.0,
        "predicted_total": 50.0,
    }]).to_csv(prediction_path, index=False)
    dashboard_dir = tmp_path / "public" / "data"
    dashboard_dir.mkdir(parents=True)
    (dashboard_dir / "dashboard.json").write_text(json.dumps({
        "generated_at": "2026-10-26T14:17:00+00:00",
        "season": 2026,
        "week": 9,
        "games": [],
    }))
    monkeypatch.setattr(site_builder, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(site_builder, "SITE_DIR", tmp_path / "public")
    monkeypatch.setattr(site_builder, "DATA_DIR", dashboard_dir)
    monkeypatch.setattr(site_builder, "HISTORY_PATH", tmp_path / "line_history.json")
    monkeypatch.setattr(site_builder, "ATS_RESULTS_PATH", tmp_path / "ats_results.json")
    monkeypatch.setattr(site_builder, "_prediction_file", lambda season, week: prediction_path)
    monkeypatch.setattr(site_builder, "_api_get", lambda endpoint, params, api_key: [])
    monkeypatch.setattr(site_builder, "_read_history", lambda: {})
    monkeypatch.setattr(site_builder, "_write_history", lambda history: None)

    output = site_builder.build_site(2026, 9, None, False, predictions_generated=True)

    assert json.loads(output.read_text())["generated_at"] != "2026-10-26T14:17:00+00:00"


@pytest.mark.parametrize(
    ("pick", "game", "expected"),
    [
        ("Alpha", {"homeTeam": "Alpha", "awayTeam": "Beta", "homePoints": 24, "awayPoints": 20}, "win"),
        ("Alpha", {"homeTeam": "Alpha", "awayTeam": "Beta", "homePoints": 20, "awayPoints": 18}, "loss"),
        ("Alpha", {"homeTeam": "Alpha", "awayTeam": "Beta", "homePoints": 23, "awayPoints": 20}, "push"),
        (None, {"homeTeam": "Alpha", "awayTeam": "Beta", "homePoints": 24, "awayPoints": 20}, None),
    ],
)
def test_grade_ats_uses_spread_and_handles_pushes_and_no_pick(pick, game, expected):
    opening = {"favorite_team": "Alpha", "spread_favorite": -3}

    assert _grade_ats(pick, game, opening) == expected


@pytest.mark.parametrize(
    ("predicted_home_margin", "game", "expected"),
    [
        (0.1, {"homeTeam": "Alpha", "awayTeam": "Beta", "homePoints": 24, "awayPoints": 20}, "win"),
        (0.1, {"homeTeam": "Alpha", "awayTeam": "Beta", "homePoints": 20, "awayPoints": 24}, "loss"),
        (-0.1, {"homeTeam": "Alpha", "awayTeam": "Beta", "homePoints": 20, "awayPoints": 24}, "win"),
        (0.1, {"homeTeam": "Alpha", "awayTeam": "Beta"}, None),
        (0.0, {"homeTeam": "Alpha", "awayTeam": "Beta", "homePoints": 24, "awayPoints": 20}, None),
    ],
)
def test_grade_straight_up_uses_predicted_score_winner(predicted_home_margin, game, expected):
    assert _grade_straight_up(predicted_home_margin, game) == expected


def test_performance_ats_percent_excludes_pushes_and_weeks_before_nine():
    results = {"2026": {
        "9:1": {"week": 9, "result": "win"},
        "9:2": {"week": 9, "result": "loss"},
        "9:3": {"week": 9, "result": "push"},
        "8:1": {"week": 8, "result": "win"},
    }}

    performance = _performance(results, 2026, 9)

    assert performance["current_week"]["ats_pct"] == 0.5
    assert performance["current_week"]["pushes"] == 1
    assert performance["current_week"]["su_wins"] == 0
    assert performance["current_week"]["su_losses"] == 0
    assert performance["current_week"]["su_pct"] is None
    assert [item["week"] for item in performance["weeks"]] == [9]


def test_line_refresh_persists_opening_line_ats_results_and_weekly_performance(
    tmp_path, monkeypatch
):
    prediction_paths = {}
    week9_predictions = [{
        "game_id": "game-9",
        "season": 2026,
        "week": 9,
        "game_date": "2026-10-31T12:00:00Z",
        "home_team": "Alpha",
        "away_team": "Beta",
        "predicted_home_score": 20.1,
        "predicted_away_score": 20.0,
        "predicted_home_margin": 0.1,
        "predicted_total": 50.0,
    }, {
        "game_id": "game-10",
        "season": 2026,
        "week": 9,
        "game_date": "2026-10-31T12:00:00Z",
        "home_team": "Gamma",
        "away_team": "Delta",
        "predicted_home_score": 20.0,
        "predicted_away_score": 20.0,
        "predicted_home_margin": 0.0,
        "predicted_total": 40.0,
    }]
    week10_predictions = [{
        "game_id": "game-11",
        "season": 2026,
        "week": 10,
        "game_date": "2026-11-07T12:00:00Z",
        "home_team": "Epsilon",
        "away_team": "Zeta",
        "predicted_home_score": 30.0,
        "predicted_away_score": 20.0,
        "predicted_home_margin": 10.0,
        "predicted_total": 50.0,
    }]
    for week, rows in ((9, week9_predictions), (10, week10_predictions)):
        path = tmp_path / f"week{week}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        prediction_paths[week] = path

    schedules = {
        9: [{
            "id": 9, "homeTeam": "Alpha", "awayTeam": "Beta",
            "homePoints": 20, "awayPoints": 24,
        }, {
            "id": 10, "homeTeam": "Gamma", "awayTeam": "Delta",
            "homePoints": 24, "awayPoints": 10,
        }],
        10: [{"id": 11, "homeTeam": "Epsilon", "awayTeam": "Zeta"}],
    }
    lines = {
        9: [{
            "id": 9,
            "homeTeam": "Alpha",
            "awayTeam": "Beta",
            "lines": [{
                "provider": "DraftKings",
                "spread": -12,
                "spreadOpen": -1.5,
                "overUnder": 50,
                "overUnderOpen": 48,
            }],
        }],
        10: [{
            "id": 11,
            "homeTeam": "Epsilon",
            "awayTeam": "Zeta",
            "lines": [{"provider": "DraftKings", "spread": -9, "overUnder": 50}],
        }],
    }

    def api_get(endpoint, params, api_key):
        week = params.get("week")
        if endpoint == "games":
            return schedules.get(week, [])
        if endpoint == "lines":
            return lines.get(week, [])
        return []

    monkeypatch.setattr(site_builder, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(site_builder, "SITE_DIR", tmp_path / "public")
    monkeypatch.setattr(site_builder, "DATA_DIR", tmp_path / "public" / "data")
    monkeypatch.setattr(site_builder, "HISTORY_PATH", tmp_path / "line_history.json")
    monkeypatch.setattr(site_builder, "ATS_RESULTS_PATH", tmp_path / "ats_results.json")
    (tmp_path / "ats_results.json").write_text(json.dumps({
        "2026": {
            "9:9": {
                "week": 9, "game_id": "9", "result": "loss", "su_result": "win",
            },
        },
    }))
    monkeypatch.setattr(
        site_builder, "_prediction_file", lambda season, week: prediction_paths[week]
    )
    monkeypatch.setattr(site_builder, "_api_get", api_get)

    week9 = json.loads(site_builder.build_site(2026, 9, None, False).read_text())

    assert week9["games"][0]["pick"] == "Beta"
    assert week9["games"][0]["market"]["opening"]["spread_favorite"] == -1.5
    assert week9["games"][0]["market"]["current"]["spread_favorite"] == -12
    assert week9["games"][0]["model_edge_favorite"] == -1.4
    assert week9["games"][0]["ats_result"] == "loss"
    assert week9["games"][0]["su_result"] == "loss"
    assert week9["games"][1]["pick"] is None
    assert week9["games"][1]["ats_result"] is None
    assert week9["games"][1]["su_result"] is None
    assert week9["performance"]["current_week"] == {
        "week": 9, "wins": 0, "losses": 1, "pushes": 0, "ats_pct": 0.0,
        "su_wins": 0, "su_losses": 1, "su_pct": 0.0,
    }

    week10 = json.loads(site_builder.build_site(2026, 10, None, False).read_text())
    saved_results = json.loads((tmp_path / "ats_results.json").read_text())

    assert saved_results["2026"]["9:9"]["result"] == "loss"
    assert saved_results["2026"]["9:9"]["su_result"] == "loss"
    assert week10["performance"]["current_week"]["week"] == 10
    assert week10["performance"]["weeks"] == [
        {"week": 9, "wins": 0, "losses": 1, "pushes": 0, "ats_pct": 0.0,
         "su_wins": 0, "su_losses": 1, "su_pct": 0.0},
        {"week": 10, "wins": 0, "losses": 0, "pushes": 0, "ats_pct": None,
         "su_wins": 0, "su_losses": 0, "su_pct": None},
    ]
