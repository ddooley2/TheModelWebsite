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

from src.cfb_model.features.advanced import ADVANCED_METRICS, build_advanced_features
from src.cfb_model.reporting.diagnostics import generate_diagnostics
from src.cfb_model.train.score_model import PROJECT_ROOT

CFBD_API = "https://api.collegefootballdata.com"
SITE_DIR = PROJECT_ROOT / "public"
DATA_DIR = SITE_DIR / "data"
HISTORY_PATH = PROJECT_ROOT / "data" / "site" / "line_history.json"
ATS_RESULTS_PATH = PROJECT_ROOT / "data" / "site" / "ats_results.json"
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


def _read_ats_results() -> dict:
    if not ATS_RESULTS_PATH.exists():
        return {}
    try:
        return json.loads(ATS_RESULTS_PATH.read_text())
    except json.JSONDecodeError:
        raise ValueError(f"Invalid ATS results JSON: {ATS_RESULTS_PATH}")


def _write_ats_results(results: dict) -> None:
    ATS_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ATS_RESULTS_PATH.write_text(json.dumps(results, indent=2, sort_keys=True))


def _grade_ats(pick: str | None, game: dict, opening: dict | None) -> str | None:
    if not pick or not opening or opening.get("favorite_team") is None:
        return None
    home_score = pd.to_numeric(game.get("homePoints"), errors="coerce")
    away_score = pd.to_numeric(game.get("awayPoints"), errors="coerce")
    spread = pd.to_numeric(opening.get("spread_favorite"), errors="coerce")
    if pd.isna(home_score) or pd.isna(away_score) or pd.isna(spread):
        return None
    if pick == game.get("homeTeam"):
        margin = home_score - away_score
    elif pick == game.get("awayTeam"):
        margin = away_score - home_score
    else:
        return None
    margin += spread if pick == opening["favorite_team"] else -spread
    return "win" if margin > 0 else "loss" if margin < 0 else "push"


def _grade_straight_up(predicted_home_margin: float, game: dict) -> str | None:
    if predicted_home_margin == 0:
        return None
    home_score = pd.to_numeric(game.get("homePoints"), errors="coerce")
    away_score = pd.to_numeric(game.get("awayPoints"), errors="coerce")
    if pd.isna(home_score) or pd.isna(away_score):
        return None
    predicted_home_winner = predicted_home_margin > 0
    actual_home_winner = home_score > away_score
    return "win" if predicted_home_winner == actual_home_winner else "loss"


def _performance(results: dict, season: int, current_week: int) -> dict:
    season_results = results.get(str(season), {})
    weeks = [
        int(record["week"])
        for record in season_results.values()
        if int(record["week"]) >= 9
    ]
    final_week = max([current_week, *weeks], default=current_week)

    def totals(week: int) -> dict:
        outcomes = [
            record.get("result")
            for record in season_results.values()
            if int(record["week"]) == week
        ]
        wins = outcomes.count("win")
        losses = outcomes.count("loss")
        pushes = outcomes.count("push")
        decided = wins + losses
        straight_up_outcomes = [
            record.get("su_result")
            for record in season_results.values()
            if int(record["week"]) == week
        ]
        straight_up_wins = straight_up_outcomes.count("win")
        straight_up_losses = straight_up_outcomes.count("loss")
        straight_up_decided = straight_up_wins + straight_up_losses
        return {
            "week": week,
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "ats_pct": None if decided == 0 else wins / decided,
            "su_wins": straight_up_wins,
            "su_losses": straight_up_losses,
            "su_pct": (
                None
                if straight_up_decided == 0
                else straight_up_wins / straight_up_decided
            ),
        }

    return {
        "current_week": (
            totals(current_week)
            if current_week >= 9
            else {
                "week": current_week,
                "wins": 0,
                "losses": 0,
                "pushes": 0,
                "ats_pct": None,
                "su_wins": 0,
                "su_losses": 0,
                "su_pct": None,
            }
        ),
        "weeks": [totals(week) for week in range(9, final_week + 1)],
    }


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


def capture_reference_lines(season: int, week: int) -> None:
    """Capture live line history as the immutable reference for new predictions."""
    api_key = os.environ.get("CFBD_API_KEY")
    line_games = _api_get(
        "lines", {"year": season, "week": week, "seasonType": "regular"}, api_key
    )
    if not any(game.get("lines") for game in line_games):
        raise RuntimeError(f"No CFBD lines returned for {season} week {week}")
    history = _read_history()
    timestamp = datetime.now(timezone.utc).isoformat()
    _snapshot_lines(line_games, history, timestamp)
    for game in line_games:
        selected = _select_line(game.get("lines") or [])
        if selected is None:
            continue
        selected_key = _line_key(game, selected)
        for line in game.get("lines") or []:
            if not isinstance(line, dict):
                continue
            record = history.get(_line_key(game, line))
            if record and record.get("current"):
                record["opening"] = record["current"].copy()
                record.pop("reference", None)
                if _line_key(game, line) == selected_key:
                    record["reference"] = record["current"].copy()
    _write_history(history)


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


def _matchup_key(home_team: str, away_team: str) -> tuple[str, str]:
    return _normalise(home_team), _normalise(away_team)


def _advanced_stats_payload(
    predictions: pd.DataFrame,
    historical_games: pd.DataFrame,
    advanced: pd.DataFrame,
    schedule: list[dict],
) -> dict[tuple[str, str], dict]:
    upcoming = predictions[
        ["game_id", "season", "week", "game_date", "home_team", "away_team"]
    ].copy()
    upcoming["home_score"] = float("nan")
    upcoming["away_score"] = float("nan")
    upcoming["neutral_site"] = [
        bool(_match_schedule_game(prediction, schedule).get("neutralSite", False))
        for _, prediction in predictions.iterrows()
    ]
    historical_count = len(historical_games)
    games = pd.concat([historical_games, upcoming], ignore_index=True, sort=False)
    feature_rows = build_advanced_features(games, advanced).iloc[historical_count:]

    payload = {}
    for prediction, feature_row in zip(
        predictions.itertuples(index=False), feature_rows.itertuples(index=False)
    ):
        teams = {}
        for side in ("home", "away"):
            teams[side] = {}
            for metric in ADVANCED_METRICS:
                value = getattr(feature_row, f"{side}_{metric}")
                teams[side][metric] = (
                    None if pd.isna(value) else round(float(value), 4)
                )
        payload[_matchup_key(prediction.home_team, prediction.away_team)] = teams
    return payload


def _pregame_stats_payload(
    predictions: pd.DataFrame,
    historical_games: pd.DataFrame,
    season_games: list[dict],
    season: int,
    schedule: list[dict],
) -> dict[tuple[str, str], dict]:
    completed = []
    for game in season_games:
        if int(game.get("season") or 0) != season:
            continue
        home_score = pd.to_numeric(game.get("homePoints"), errors="coerce")
        away_score = pd.to_numeric(game.get("awayPoints"), errors="coerce")
        if (
            pd.isna(home_score)
            or pd.isna(away_score)
            or game.get("completed") is False
            or not game.get("startDate")
            or game.get("week") is None
            or not game.get("homeTeam")
            or not game.get("awayTeam")
        ):
            continue
        completed.append({
            "game_id": game.get("id"),
            "season": season,
            "week": game.get("week"),
            "game_date": game["startDate"],
            "home_team": game.get("homeTeam"),
            "away_team": game.get("awayTeam"),
            "home_score": home_score,
            "away_score": away_score,
            "neutral_site": bool(game.get("neutralSite", False)),
        })

    previous_seasons = historical_games[historical_games["season"] < season]
    current_results = pd.DataFrame(completed, columns=historical_games.columns)
    upcoming = predictions[
        ["game_id", "season", "week", "game_date", "home_team", "away_team"]
    ].copy()
    upcoming["home_score"] = float("nan")
    upcoming["away_score"] = float("nan")
    upcoming["neutral_site"] = [
        bool(_match_schedule_game(prediction, schedule).get("neutralSite", False))
        for _, prediction in predictions.iterrows()
    ]

    history_count = len(previous_seasons) + len(current_results)
    games = pd.concat(
        [previous_seasons, current_results, upcoming], ignore_index=True, sort=False
    )
    from src.cfb_model.train.score_model import build_pregame_features

    feature_rows = build_pregame_features(games).iloc[history_count:]
    payload = {}
    for prediction, row in zip(
        predictions.itertuples(index=False), feature_rows.itertuples(index=False)
    ):
        teams = {}
        for side in ("home", "away"):
            teams[side] = {
                "points_for": round(float(getattr(row, f"{side}_points_for_season")), 1),
                "points_against": round(
                    float(getattr(row, f"{side}_points_against_season")), 1
                ),
            }
        payload[_matchup_key(prediction.home_team, prediction.away_team)] = teams
    return payload


def _saved_pregame_stats_payload(predictions: pd.DataFrame) -> dict[tuple[str, str], dict]:
    fields = {
        "home": ("home_points_for_season", "home_points_against_season"),
        "away": ("away_points_for_season", "away_points_against_season"),
    }
    if not {column for columns in fields.values() for column in columns}.issubset(
        predictions.columns
    ):
        return {}
    payload = {}
    for prediction in predictions.itertuples(index=False):
        teams = {}
        for side, (for_column, against_column) in fields.items():
            points_for = pd.to_numeric(getattr(prediction, for_column), errors="coerce")
            points_against = pd.to_numeric(
                getattr(prediction, against_column), errors="coerce"
            )
            teams[side] = {
                "points_for": None if pd.isna(points_for) else round(float(points_for), 1),
                "points_against": (
                    None if pd.isna(points_against) else round(float(points_against), 1)
                ),
            }
        payload[_matchup_key(prediction.home_team, prediction.away_team)] = teams
    return payload


def _normalize_advanced_stats(
    stats_by_matchup: dict[tuple[str, str], dict],
) -> tuple[dict[tuple[str, str], dict], dict[str, dict[str, float | None]]]:
    ranges = {}
    for metric in ADVANCED_METRICS:
        values = [
            value
            for teams in stats_by_matchup.values()
            for side in ("home", "away")
            if (value := teams.get(side, {}).get(metric)) is not None
        ]
        ranges[metric] = {
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }

    normalized = {}
    for matchup, teams in stats_by_matchup.items():
        normalized[matchup] = {}
        for side in ("home", "away"):
            normalized[matchup][side] = {}
            for metric in ADVANCED_METRICS:
                value = teams.get(side, {}).get(metric)
                low, high = ranges[metric]["min"], ranges[metric]["max"]
                if value is None or low is None or high is None:
                    score = None
                elif low == high:
                    score = 50.0
                else:
                    score = round((value - low) / (high - low) * 100, 1)
                normalized[matchup][side][metric] = score
    return normalized, ranges


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
    reference_record = next(
        (
            value for history_key, value in history.items()
            if history_key.startswith(f"{game.get('id', '')}:")
            and value.get("reference") is not None
        ),
        record,
    )
    opening = _favorite_line(
        reference_record.get("reference") or reference_record.get("opening"),
        home_team,
        away_team,
    )
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


def build_site(
    season: int,
    week: int | None,
    db_path: Path | None,
    include_diagnostics: bool,
    predictions_generated: bool = False,
) -> Path:
    """Build static data for the dashboard; CFBD is display-only, never a model input."""
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    api_key = os.environ.get("CFBD_API_KEY")
    predictions = pd.read_csv(_prediction_file(season, week))
    resolved_week = int(predictions["week"].iloc[0])
    dashboard_path = DATA_DIR / "dashboard.json"
    previous = (
        json.loads(dashboard_path.read_text()) if dashboard_path.exists() else None
    )
    matching_previous = (
        previous
        if previous
        and previous.get("season") == season
        and previous.get("week") == resolved_week
        else None
    )

    schedule = _api_get("games", {"year": season, "week": resolved_week, "seasonType": "regular"}, api_key)
    all_games = _api_get("games", {"year": season, "seasonType": "regular"}, api_key)
    line_games = _api_get("lines", {"year": season, "week": resolved_week, "seasonType": "regular"}, api_key)
    ranking_rows = _api_get("rankings", {"year": season, "week": resolved_week}, api_key)

    history = _read_history()
    line_timestamp = datetime.now(timezone.utc).isoformat()
    lines_by_id, changed = _snapshot_lines(line_games, history, line_timestamp)
    if changed:
        _write_history(history)

    generated_at = line_timestamp
    if not predictions_generated and matching_previous:
        generated_at = matching_previous.get("generated_at") or generated_at

    schedule_by_teams = schedule or [
        {"id": game.get("id"), "homeTeam": game.get("homeTeam"), "awayTeam": game.get("awayTeam")}
        for game in line_games
    ]
    advanced_by_game = {
        _matchup_key(game["home"]["name"], game["away"]["name"]): game["advanced_stats"]
        for game in (matching_previous or {}).get("games", [])
        if game.get("home", {}).get("name")
        and game.get("away", {}).get("name")
        and game.get("advanced_stats") is not None
    } if not predictions_generated else {}
    pregame_by_game = {
        _matchup_key(game["home"]["name"], game["away"]["name"]): game["pregame_stats"]
        for game in (matching_previous or {}).get("games", [])
        if game.get("home", {}).get("name")
        and game.get("away", {}).get("name")
        and game.get("pregame_stats") is not None
    } if not predictions_generated else {}
    missing_advanced = [
        _matchup_key(game.home_team, game.away_team)
        for game in predictions.itertuples(index=False)
        if _matchup_key(game.home_team, game.away_team) not in advanced_by_game
    ]
    missing_pregame = [
        _matchup_key(game.home_team, game.away_team)
        for game in predictions.itertuples(index=False)
        if _matchup_key(game.home_team, game.away_team) not in pregame_by_game
    ]
    feature_db_path = db_path or PROJECT_ROOT / "db" / "warehouse.duckdb"
    if missing_pregame:
        saved_pregame = _saved_pregame_stats_payload(predictions)
        if saved_pregame:
            pregame_by_game = {**saved_pregame, **pregame_by_game}
            missing_pregame = [
                matchup
                for matchup in missing_pregame
                if matchup not in pregame_by_game
            ]
    if (
        not feature_db_path.exists()
        and db_path is None
        and api_key
        and (missing_pregame or missing_advanced)
    ):
        from src.cfb_model.data.warehouse import ensure_warehouse

        feature_db_path = ensure_warehouse(feature_db_path)
    if feature_db_path.exists() and (missing_pregame or (missing_advanced and api_key)):
        from src.cfb_model.train.score_model import load_historical_games
        historical_games = load_historical_games(feature_db_path)
        if missing_pregame:
            pregame_by_game = {
                **_pregame_stats_payload(
                    predictions, historical_games, all_games, season, schedule_by_teams
                ),
                **pregame_by_game,
            }
        if missing_advanced and api_key:
            from src.cfb_model.train.score_model import load_advanced_stats

            advanced = load_advanced_stats(feature_db_path, season)
            advanced_by_game = {
                **_advanced_stats_payload(
                    predictions, historical_games, advanced, schedule_by_teams
                ),
                **advanced_by_game,
            }
    normalized_by_game, advanced_stat_ranges = _normalize_advanced_stats(advanced_by_game)
    record_lookup = _records(all_games, season)
    rank_lookup = _rankings(ranking_rows)
    ats_results = _read_ats_results()
    season_results = ats_results.setdefault(str(season), {})
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
        reference_spread = (market.get("opening") or {}).get("spread_favorite")
        favorite_team = (market.get("opening") or {}).get("favorite_team")
        underdog_team = (market.get("opening") or {}).get("underdog_team")
        favorite_margin = (
            home_margin
            if favorite_team == prediction.home_team
            else -home_margin
            if favorite_team == prediction.away_team
            else None
        )
        edge = (
            None
            if reference_spread is None or favorite_margin is None
            else favorite_margin + float(reference_spread)
        )
        pick = None
        if edge is not None and abs(edge) >= 1.0:
            pick = favorite_team if edge > 0 else underdog_team
        game_id = str(schedule_game.get("id") or prediction.game_id)
        result_key = f"{resolved_week}:{game_id}"
        if resolved_week >= 9:
            opening = market.get("opening")
            stored_result = season_results.get(result_key, {})
            ats_result = (
                stored_result.get("result")
                or (_grade_ats(pick, schedule_game, opening) if pick else None)
            )
            su_result = _grade_straight_up(home_margin, schedule_game)
            if su_result is None:
                su_result = stored_result.get("su_result")
            if ats_result is not None or su_result is not None:
                season_results[result_key] = {
                    **stored_result,
                    "week": resolved_week,
                    "game_id": game_id,
                    **({"result": ats_result} if ats_result is not None else {}),
                    **({"su_result": su_result} if su_result is not None else {}),
                }
        ats_result = (
            season_results.get(result_key, {}).get("result")
            if resolved_week >= 9
            else None
        )
        su_result = (
            season_results.get(result_key, {}).get("su_result")
            if resolved_week >= 9
            else None
        )
        games.append({
            "id": game_id,
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
            "advanced_stats": advanced_by_game.get(
                _matchup_key(prediction.home_team, prediction.away_team)
            ),
            "advanced_stats_normalized": normalized_by_game.get(
                _matchup_key(prediction.home_team, prediction.away_team)
            ),
            "pregame_stats": pregame_by_game.get(
                _matchup_key(prediction.home_team, prediction.away_team)
            ),
            "model_edge_favorite": None if edge is None else round(edge, 1),
            "pick": pick,
            "ats_result": ats_result,
            "su_result": su_result,
        })
    _write_ats_results(ats_results)
    payload = {
        "generated_at": generated_at,
        "season": season,
        "week": resolved_week,
        "games": games,
        "advanced_stat_ranges": advanced_stat_ranges,
        "performance": _performance(ats_results, season, resolved_week),
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
    parser.add_argument("--predictions-generated", action="store_true")
    parser.add_argument("--capture-reference", action="store_true")
    args = parser.parse_args()
    if args.capture_reference:
        if args.week is None:
            raise ValueError("--week is required with --capture-reference")
        capture_reference_lines(args.season, args.week)
        return
    output = build_site(
        args.season,
        args.week,
        args.db_path,
        not args.skip_diagnostics,
        args.predictions_generated,
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
