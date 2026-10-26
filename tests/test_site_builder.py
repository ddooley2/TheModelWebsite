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

    market = _market_payload(later[0], history)
    assert market["opening"]["spread"] == -3.5
    assert market["current"]["spread"] == -5.0
    assert market["provider"] == "DraftKings"
