import json
from datetime import datetime, timezone
from unittest.mock import Mock

from src.cfb_model.site.schedule import resolve_publish_schedule


def test_daily_refresh_uses_current_dashboard_schedule(tmp_path):
    dashboard = tmp_path / "dashboard.json"
    dashboard.write_text(json.dumps({"season": 2026, "week": 9}))

    assert resolve_publish_schedule("daily", dashboard_path=dashboard) == (2026, 9)


def test_daily_refresh_accepts_explicit_schedule(tmp_path):
    assert resolve_publish_schedule(
        "daily", "2027", "3", dashboard_path=tmp_path / "missing.json"
    ) == (2027, 3)


def test_weekly_resolves_next_upcoming_fbs_week(monkeypatch):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = [
        {"homeClassification": "fbs", "awayClassification": "fbs",
         "startDate": "2026-10-20T17:00:00Z", "week": 8},
        {"homeClassification": "fcs", "awayClassification": "fbs",
         "startDate": "2026-10-27T16:00:00Z", "week": 9},
        {"homeClassification": "fbs", "awayClassification": "fbs",
         "startDate": "2026-10-27T23:00:00Z", "week": 9},
    ]
    get = Mock(return_value=response)
    monkeypatch.setattr("src.cfb_model.site.schedule.requests.get", get)

    assert resolve_publish_schedule(
        "weekly",
        api_key="test-key",
        now=datetime(2026, 10, 26, tzinfo=timezone.utc),
    ) == (2026, 9)
    assert get.call_args.kwargs["params"] == {"year": 2026, "seasonType": "regular"}


def test_weekly_explicit_schedule_does_not_need_api():
    assert resolve_publish_schedule("weekly", "2026", "9") == (2026, 9)
