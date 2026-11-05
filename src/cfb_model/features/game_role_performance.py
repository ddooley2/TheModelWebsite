"""Point-in-time team form split by historical home/away game role."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


FORMULATIONS = ("points", "form", "advanced", "both_roles", "relevant_and_overall")
BASE_METRICS = ("points_for", "points_against", "win_pct")
ADVANCED_METRICS = (
    "offense_ppa",
    "offense_successRate",
    "offense_explosiveness",
    "defense_ppa",
    "defense_successRate",
    "defense_explosiveness",
)


def _role_columns(matchup_side: str, historical_role: str, metrics: tuple[str, ...]) -> list[str]:
    return [
        f"{matchup_side}_team_prior_{historical_role}_{metric}"
        for metric in metrics
    ]


def _overall_columns(matchup_side: str, metrics: tuple[str, ...]) -> list[str]:
    return [f"{matchup_side}_team_prior_overall_{metric}" for metric in metrics]


FEATURE_COLUMNS = {
    "points": (
        _role_columns("home", "home", ("points_for", "points_against"))
        + _role_columns("away", "away", ("points_for", "points_against"))
    ),
    "form": (
        _role_columns("home", "home", BASE_METRICS)
        + _role_columns("away", "away", BASE_METRICS)
    ),
    "advanced": (
        _role_columns("home", "home", BASE_METRICS + ADVANCED_METRICS)
        + _role_columns("away", "away", BASE_METRICS + ADVANCED_METRICS)
    ),
    "both_roles": (
        _role_columns("home", "home", BASE_METRICS)
        + _role_columns("home", "away", BASE_METRICS)
        + _role_columns("away", "home", BASE_METRICS)
        + _role_columns("away", "away", BASE_METRICS)
    ),
    "relevant_and_overall": (
        _role_columns("home", "home", BASE_METRICS)
        + _overall_columns("home", BASE_METRICS)
        + _role_columns("away", "away", BASE_METRICS)
        + _overall_columns("away", BASE_METRICS)
    ),
}
ALL_FEATURE_COLUMNS = list(dict.fromkeys(
    column for columns in FEATURE_COLUMNS.values() for column in columns
))


def _summarize(stats: dict[str, list[float]]) -> dict[str, float]:
    return {
        metric: total / count
        for metric, (total, count) in stats.items()
        if count
    }


def _game_id_key(value: object) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _advanced_lookup(advanced: pd.DataFrame | None) -> tuple[dict, dict]:
    by_id: dict[tuple[str, str], dict[str, float]] = {}
    by_week: dict[tuple[object, object, str], dict[str, float]] = {}
    if advanced is None or advanced.empty:
        return by_id, by_week

    for row in advanced.to_dict("records"):
        team = row.get("team")
        if pd.isna(team):
            continue
        values = {}
        for metric in ADVANCED_METRICS:
            value = pd.to_numeric(row.get(metric), errors="coerce")
            if pd.notna(value) and np.isfinite(value):
                values[metric] = float(value)
        key = _game_id_key(row.get("game_id"))
        if key is not None:
            by_id[(key, team)] = values
        else:
            by_week[(row.get("season"), row.get("week"), team)] = values
    return by_id, by_week


def build_game_role_features(
    games: pd.DataFrame, advanced: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Build season-to-date role-split and overall form from earlier games only.

    Overall history includes neutral-site games; home/away-role history does
    not. Same-date games are evaluated before any results from that date enter
    history.
    """
    required = {
        "game_date", "home_team", "away_team", "home_score", "away_score",
        "season", "week",
    }
    missing = required - set(games.columns)
    if missing:
        raise ValueError(f"Missing game columns: {sorted(missing)}")

    output = np.full((len(games), len(ALL_FEATURE_COLUMNS)), np.nan, dtype=float)
    column_index = {column: index for index, column in enumerate(ALL_FEATURE_COLUMNS)}
    if games.empty:
        return pd.DataFrame(output, index=games.index, columns=ALL_FEATURE_COLUMNS)

    dates = pd.to_datetime(games["game_date"], utc=True, errors="coerce", format="mixed")
    home_teams = games["home_team"].to_numpy()
    away_teams = games["away_team"].to_numpy()
    home_scores = pd.to_numeric(games["home_score"], errors="coerce").to_numpy(dtype=float)
    away_scores = pd.to_numeric(games["away_score"], errors="coerce").to_numpy(dtype=float)
    neutral = (
        games["neutral_site"].fillna(False).astype(bool).to_numpy()
        if "neutral_site" in games
        else np.zeros(len(games), dtype=bool)
    )
    game_ids = games["game_id"].to_numpy() if "game_id" in games else np.full(len(games), np.nan)
    seasons = games["season"].to_numpy()
    weeks = games["week"].to_numpy()
    by_id, by_week = _advanced_lookup(advanced)
    history = defaultdict(lambda: {
        "overall": defaultdict(lambda: [0.0, 0.0]),
        "home": defaultdict(lambda: [0.0, 0.0]),
        "away": defaultdict(lambda: [0.0, 0.0]),
    })

    ordered = pd.DataFrame({"position": np.arange(len(games)), "date": dates})
    ordered = ordered.dropna(subset=["date"]).sort_values(["date", "position"], kind="stable")
    for _, date_rows in ordered.groupby(ordered["date"].dt.normalize(), sort=True):
        positions = date_rows["position"].to_numpy(dtype=int)
        for position in positions:
            for matchup_side, team, role in (
                ("home", home_teams[position], "home"),
                ("away", away_teams[position], "away"),
            ):
                if pd.isna(team):
                    continue
                team_history = history[(team, seasons[position])]
                overall = _summarize(team_history["overall"])
                role_stats = _summarize(team_history[role])
                for formulation in FORMULATIONS:
                    if formulation in ("points", "form", "advanced"):
                        metrics = (
                            ("points_for", "points_against")
                            if formulation == "points"
                            else BASE_METRICS + (ADVANCED_METRICS if formulation == "advanced" else ())
                        )
                        source = role_stats
                        names = _role_columns(matchup_side, role, metrics)
                        for metric, name in zip(metrics, names):
                            if metric in source:
                                output[position, column_index[name]] = source[metric]
                    elif formulation == "both_roles":
                        for historical_role in ("home", "away"):
                            stats = _summarize(team_history[historical_role])
                            for metric, name in zip(
                                BASE_METRICS,
                                _role_columns(matchup_side, historical_role, BASE_METRICS),
                            ):
                                if metric in stats:
                                    output[position, column_index[name]] = stats[metric]
                    else:
                        for stats, names in (
                            (role_stats, _role_columns(matchup_side, role, BASE_METRICS)),
                            (overall, _overall_columns(matchup_side, BASE_METRICS)),
                        ):
                            for metric, name in zip(BASE_METRICS, names):
                                if metric in stats:
                                    output[position, column_index[name]] = stats[metric]

        for position in positions:
            home, away = home_teams[position], away_teams[position]
            home_score, away_score = home_scores[position], away_scores[position]
            if pd.isna(home) or pd.isna(away) or not np.isfinite(home_score) or not np.isfinite(away_score):
                continue
            margin = float(home_score - away_score)
            game_key = _game_id_key(game_ids[position])
            observations = (
                (home, "home", home_score, away_score, float(margin > 0)),
                (away, "away", away_score, home_score, float(margin < 0)),
            )
            for team, role, points_for, points_against, win in observations:
                values = {
                    "points_for": float(points_for),
                    "points_against": float(points_against),
                    "win_pct": win,
                }
                advanced_values = (
                    by_id.get((game_key, team), {})
                    if game_key is not None
                    else {}
                )
                if not advanced_values:
                    advanced_values = by_week.get((seasons[position], weeks[position], team), {})
                values.update(advanced_values)
                team_history = history[(team, seasons[position])]
                for scope in ("overall",):
                    for metric, value in values.items():
                        accumulator = team_history[scope][metric]
                        accumulator[0] += value
                        accumulator[1] += 1
                if not neutral[position]:
                    for metric, value in values.items():
                        accumulator = team_history[role][metric]
                        accumulator[0] += value
                        accumulator[1] += 1

    return pd.DataFrame(output, index=games.index, columns=ALL_FEATURE_COLUMNS)
