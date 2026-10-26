"""Resolve the season and week for scheduled dashboard publications."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests

CFBD_API = "https://api.collegefootballdata.com"
DEFAULT_DASHBOARD = Path(__file__).resolve().parents[3] / "public" / "data" / "dashboard.json"


def _upcoming_week(season: int, api_key: str, now: datetime) -> int | None:
    response = requests.get(
        f"{CFBD_API}/games",
        params={"year": season, "seasonType": "regular"},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=60,
    )
    response.raise_for_status()
    upcoming = []
    for game in response.json():
        if game.get("homeClassification") != "fbs" or game.get("awayClassification") != "fbs":
            continue
        start = game.get("startDate")
        week = game.get("week")
        if not start or week is None:
            continue
        game_time = datetime.fromisoformat(start.replace("Z", "+00:00"))
        if game_time >= now:
            upcoming.append((game_time, int(week)))
    return min(upcoming)[1] if upcoming else None


def resolve_publish_schedule(
    run_mode: str,
    season_override: str | None = None,
    week_override: str | None = None,
    dashboard_path: Path = DEFAULT_DASHBOARD,
    api_key: str | None = None,
    now: datetime | None = None,
) -> tuple[int, int]:
    """Resolve an explicit schedule, the current dashboard, or the next FBS week."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)

    season = int(season_override) if season_override else None
    week = int(week_override) if week_override else None
    if run_mode == "daily":
        if season is not None and week is not None:
            return season, week
        dashboard = json.loads(dashboard_path.read_text())
        season = season or int(dashboard["season"])
        week = week or int(dashboard["week"])
        return season, week
    if run_mode not in {"daily", "weekly"}:
        raise ValueError(f"Unsupported publish run mode: {run_mode}")
    if run_mode == "weekly" and season is not None and week is not None:
        return season, week

    api_key = api_key or os.environ.get("CFBD_API_KEY")
    if not api_key:
        raise RuntimeError("CFBD_API_KEY is required to resolve the upcoming schedule.")
    seasons = [season] if season is not None else [now.year, now.year + 1]
    for candidate_season in seasons:
        resolved_week = _upcoming_week(candidate_season, api_key, now)
        if resolved_week is not None:
            return candidate_season, week if week is not None else resolved_week
    raise ValueError(f"No upcoming regular-season FBS games found for seasons {seasons}.")
