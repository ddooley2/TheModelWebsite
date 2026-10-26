"""Build the static GitHub Pages dashboard from model output and CFBD API data."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.cfb_model.reporting.diagnostics import generate_diagnostics
from src.cfb_model.train.score_model import PROJECT_ROOT

CFBD_API = "https://api.collegefootballdata.com"
SITE_DIR = PROJECT_ROOT / "public"
DATA_DIR = SITE_DIR / "data"
HISTORY_PATH = PROJECT_ROOT / "data" / "site" / "line_history.json"
LINE_PROVIDER_PREFERENCE = ("DraftKings", "FanDuel", "BetMGM", "Bovada", "Caesars")


def _normalise(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _api_get(endpoint: str, params: dict[str, Any], api_key: str | None) -> list[dict]:
    if not api_key:
        return []
    response = requests.get(
        f"{CFBD_API}/{endpoint.lstrip('/')}",
        params=params,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return data if isinstance(data, list) else []


def _read_history() -> dict:
    if not HISTORY_PATH.exists():
        return {}
    try:
        return json.loads(HISTORY_PATH.read_text())
    except json.JSONDecodeError:
        raise ValueError(f"Invalid line history JSON: {HISTORY_PATH}")


def _write_history(history: dict) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(history, indent=2, sort_keys=True))


def _line_key(game: dict, line: dict) -> str:
    provider = _normalise(line.get("provider") or line.get("source") or "unknown")
    return ":".join((str(game.get("id", "")), provider))


def _select_line(lines: list[dict]) -> dict | None:
    numeric = [
        line for line in lines
        if pd.notna(pd.to_numeric(line.get("spread"), errors="coerce"))
    ]
    if not numeric:
        return None
    ranks = {_normalise(name): index for index, name in enumerate(LINE_PROVIDER_PREFERENCE)}
    return min(
        numeric,
        key=lambda line: ranks.get(_normalise(line.get("provider") or line.get("source")), len(ranks)),
    )


def _snapshot_lines(lines_games: list[dict], history: dict, timestamp: str) -> tuple[dict, bool]:
    """Store first-seen opening lines and latest observed current lines by provider."""
    changed = False
    by_game = {}
    for game in lines_games:
        lines = game.get("lines") or []
        for line in lines:
            if not isinstance(line, dict):
                continue
            key = _line_key(game, line)
            observed = {
                "spread": pd.to_numeric(line.get("spread"), errors="coerce"),
                "total": pd.to_numeric(line.get("overUnder"), errors="coerce"),
                "provider": line.get("provider") or line.get("source"),
                "observed_at": timestamp,
            }
            observed = {name: (None if pd.isna(value) else value) for name, value in observed.items()}
            record = history.get(key)
            if record is None:
                opening = observed.copy()
                for field, source in (
                    ("spread", "spreadOpen"),
                    ("total", "overUnderOpen"),
                ):
                    value = pd.to_numeric(line.get(source), errors="coerce")
                    if pd.notna(value):
                        opening[field] = float(value)
                history[key] = {"opening": opening, "current": observed}
                changed = True
            elif record.get("current") != observed:
                record["current"] = observed
                changed = True
        by_game[str(game.get("id"))] = game
    return by_game, changed


def _records(games: list[dict], season: int) -> dict[str, str]:
    record = defaultdict(lambda: [0, 0])
    for game in games:
        home_score = pd.to_numeric(game.get("homePoints"), errors="coerce")
        away_score = pd.to_numeric(game.get("awayPoints"), errors="coerce")
        if pd.isna(home_score) or pd.isna(away_score):
            continue
        home, away = _normalise(game.get("homeTeam")), _normalise(game.get("awayTeam"))
        if home_score > away_score:
            record[home][0] += 1
            record[away][1] += 1
        elif away_score > home_score:
            record[away][0] += 1
            record[home][1] += 1
    return {team: f"{wins}-{losses}" for team, (wins, losses) in record.items()}


def _rankings(rows: list[dict]) -> dict[str, int]:
    ranks: dict[str, int] = {}
    for row in rows:
        for poll in row.get("polls", []):
            if str(poll.get("poll", "")).lower() not in {"ap top 25", "associated press"}:
                continue
            for ranked in poll.get("ranks", []):
                team = ranked.get("school") or ranked.get("team")
                rank = ranked.get("rank")
                if team and rank:
                    ranks[_normalise(team)] = int(rank)
    return ranks


def _match_schedule_game(prediction: pd.Series, games: list[dict]) -> dict:
    home = _normalise(prediction.home_team)
    away = _normalise(prediction.away_team)
    for game in games:
        if _normalise(game.get("homeTeam")) == home and _normalise(game.get("awayTeam")) == away:
            return game
    return {}


def _favorite_line(record: dict | None, home_team: str, away_team: str) -> dict | None:
    if not record:
        return None
    spread = pd.to_numeric(record.get("spread"), errors="coerce")
    if pd.isna(spread):
        return None
    favorite = home_team if spread < 0 else away_team if spread > 0 else None
    underdog = away_team if spread < 0 else home_team if spread > 0 else None
    return {
        "spread_favorite": -abs(float(spread)),
        "favorite_team": favorite,
        "underdog_team": underdog,
        "total": record.get("total"),
        "provider": record.get("provider"),
        "observed_at": record.get("observed_at"),
    }


def _market_payload(
    game: dict, history: dict, home_team: str, away_team: str
) -> dict:
    selected = _select_line(game.get("lines") or [])
    if not selected:
        return {
            "opening": None,
            "current": None,
            "provider": None,
            "favorite_team": None,
            "underdog_team": None,
        }
    key = _line_key(game, selected)
    record = history.get(key, {})
    opening = _favorite_line(record.get("opening"), home_team, away_team)
    current = _favorite_line(record.get("current"), home_team, away_team)
    favorite = current or opening
    return {
        "opening": opening,
        "current": current,
        "provider": selected.get("provider") or selected.get("source"),
        "favorite_team": favorite["favorite_team"] if favorite else None,
        "underdog_team": favorite["underdog_team"] if favorite else None,
    }


def _prediction_file(season: int, week: int | None) -> Path:
    if week is not None:
        path = PROJECT_ROOT / "results" / f"score_predictions_{season}_week{week}.csv"
        if path.exists():
            return path
        raise FileNotFoundError(f"Prediction file does not exist: {path}")
    choices = sorted((PROJECT_ROOT / "results").glob(f"score_predictions_{season}_week*.csv"))
    if not choices:
        raise FileNotFoundError(f"No prediction CSV for {season}. Run the score model first.")
    return choices[-1]


def build_site(season: int, week: int | None, db_path: Path | None, include_diagnostics: bool) -> Path:
    """Build static data for the dashboard; CFBD is display-only, never a model input."""
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    api_key = os.environ.get("CFBD_API_KEY")
    predictions = pd.read_csv(_prediction_file(season, week))
    resolved_week = int(predictions["week"].iloc[0])

    schedule = _api_get("games", {"year": season, "week": resolved_week, "seasonType": "regular"}, api_key)
    all_games = _api_get("games", {"year": season, "seasonType": "regular"}, api_key)
    line_games = _api_get("lines", {"year": season, "week": resolved_week, "seasonType": "regular"}, api_key)
    ranking_rows = _api_get("rankings", {"year": season, "week": resolved_week}, api_key)

    history = _read_history()
    timestamp = datetime.now(timezone.utc).isoformat()
    lines_by_id, changed = _snapshot_lines(line_games, history, timestamp)
    if changed:
        _write_history(history)

    schedule_by_teams = schedule or [
        {"id": game.get("id"), "homeTeam": game.get("homeTeam"), "awayTeam": game.get("awayTeam")}
        for game in line_games
    ]
    record_lookup = _records(all_games, season)
    rank_lookup = _rankings(ranking_rows)
    games = []
    for prediction in predictions.itertuples(index=False):
        schedule_game = _match_schedule_game(prediction, schedule_by_teams)
        line_game = lines_by_id.get(str(schedule_game.get("id")))
        if line_game is None:
            line_game = next((
                game for game in line_games
                if _normalise(game.get("homeTeam")) == _normalise(prediction.home_team)
                and _normalise(game.get("awayTeam")) == _normalise(prediction.away_team)
            ), {})
        market = _market_payload(
            line_game, history, prediction.home_team, prediction.away_team
        )
        home_margin = float(prediction.predicted_home_margin)
        current_spread = (market.get("current") or {}).get("spread_favorite")
        favorite_team = market.get("favorite_team")
        underdog_team = market.get("underdog_team")
        favorite_margin = (
            home_margin
            if favorite_team == prediction.home_team
            else -home_margin
            if favorite_team == prediction.away_team
            else None
        )
        edge = (
            None
            if current_spread is None or favorite_margin is None
            else favorite_margin + float(current_spread)
        )
        pick = None
        if edge is not None and abs(edge) >= 1.0:
            pick = favorite_team if edge > 0 else underdog_team
        games.append({
            "id": str(schedule_game.get("id") or prediction.game_id),
            "start": schedule_game.get("startDate") or prediction.game_date,
            "venue": schedule_game.get("venue") or None,
            "location": schedule_game.get("venue") or None,
            "neutral_site": bool(schedule_game.get("neutralSite", False)),
            "home": {
                "name": prediction.home_team,
                "logo_id": schedule_game.get("homeId"),
                "record": record_lookup.get(_normalise(prediction.home_team)),
                "rank": rank_lookup.get(_normalise(prediction.home_team)),
                "projected_score": round(float(prediction.predicted_home_score), 1),
            },
            "away": {
                "name": prediction.away_team,
                "logo_id": schedule_game.get("awayId"),
                "record": record_lookup.get(_normalise(prediction.away_team)),
                "rank": rank_lookup.get(_normalise(prediction.away_team)),
                "projected_score": round(float(prediction.predicted_away_score), 1),
            },
            "projected_home_margin": round(home_margin, 1),
            "projected_favorite_margin": (
                None if favorite_margin is None else round(favorite_margin, 1)
            ),
            "projected_total": round(float(prediction.predicted_total), 1),
            "favorite_team": favorite_team,
            "underdog_team": underdog_team,
            "market": market,
            "model_edge_favorite": None if edge is None else round(edge, 1),
            "pick": pick,
        })
    payload = {
        "generated_at": timestamp,
        "season": season,
        "week": resolved_week,
        "games": games,
        "market_data_status": "live CFBD snapshot" if api_key else "no CFBD_API_KEY; market fields unavailable",
        "methodology": {
            "model": "Market-free score model. Lines are shown only after prediction generation.",
            "pick_threshold_points": 1.0,
            "spread_convention": "A negative spread means the favorite is favored.",
        },
    }
    (DATA_DIR / "dashboard.json").write_text(json.dumps(payload, indent=2))

    if include_diagnostics:
        if db_path is None:
            raise ValueError("db_path is required when generating diagnostics")
        diagnostics = generate_diagnostics(db_path)
        (DATA_DIR / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    return DATA_DIR / "dashboard.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int)
    parser.add_argument("--db-path", type=Path)
    parser.add_argument("--skip-diagnostics", action="store_true")
    args = parser.parse_args()
    output = build_site(args.season, args.week, args.db_path, not args.skip_diagnostics)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
