import json

import pandas as pd

import src.cfb_model.site.build_site as site_builder
from src.cfb_model.site.build_site import _market_payload, _snapshot_lines


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
        "awayTeam": "Beta",
        "startDate": "2026-10-31T12:00:00Z",
        "lines": [{"provider": "DraftKings", "spread": 7.5, "overUnder": 44.5}],
    }
    monkeypatch.setattr(site_builder, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(site_builder, "SITE_DIR", tmp_path / "public")
    monkeypatch.setattr(site_builder, "DATA_DIR", tmp_path / "public" / "data")
    monkeypatch.setattr(site_builder, "HISTORY_PATH", tmp_path / "line_history.json")
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
    assert result["projected_favorite_margin"] == 10
    assert result["market"]["current"]["spread_favorite"] == -7.5
    assert result["market"]["current"]["total"] == 44.5
    assert result["model_edge_favorite"] == 2.5
    assert result["pick"] == "Beta"
