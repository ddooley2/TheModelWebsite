"""Point-in-time team form split by historical home/away game role."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


FORMULATIONS = (
    "points", "form", "advanced", "both_roles", "relevant_and_overall", "stabilized"
)
BASE_METRICS = ("points_for", "points_against", "win_pct")
ROLE_PRIOR_GAMES = 4.0
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


def _stabilized_columns(matchup_side: str, historical_role: str) -> list[str]:
    return (
        [
            f"{matchup_side}_team_prior_{historical_role}_{metric}_shrunk"
            for metric in BASE_METRICS
        ]
        + [
            f"{matchup_side}_team_prior_{historical_role}_games",
            f"{matchup_side}_team_prior_overall_games",
            f"{matchup_side}_team_prior_{historical_role}_effective_weight",
        ]
        + _overall_columns(matchup_side, BASE_METRICS)
    )


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
    "stabilized": (
        _stabilized_columns("home", "home")
        + _stabilized_columns("away", "away")
    ),
}
ALL_FEATURE_COLUMNS = list(dict.fromkeys(
    column for columns in FEATURE_COLUMNS.values() for column in columns
))


def stabilized_role_data_dictionary() -> pd.DataFrame:
    """Describe stabilized role features and the neutral-site fallback."""
    rows = []
    for side, role in (("home", "home"), ("away", "away")):
        for metric in BASE_METRICS:
            rows.extend([
                {
                    "feature": f"{side}_team_prior_{role}_{metric}_shrunk",
                    "definition": (
                        f"Prior {role}-role {metric}, shrunk toward the team's "
                        "same-season overall estimate."
                    ),
                    "orientation": metric,
                    "source_field": (
                        "team points for" if metric == "points_for"
                        else "opponent points" if metric == "points_against"
                        else "game outcome"
                    ),
                    "numerator": (
                        "role estimate × role games + overall estimate × 4 "
                        "pseudo-games"
                    ),
                    "denominator": "role games + 4 pseudo-games, when both estimates exist",
                    "aggregation_window": (
                        "same-season completed games before the target UTC date"
                    ),
                    "missing_value_policy": (
                        "Fallback to the overall estimate when role data are absent; "
                        "otherwise retain NaN for the model-native missing path."
                    ),
                    "neutral_site_policy": (
                        "Use overall estimate directly and set role effective weight to 0."
                    ),
                    "availability_cutoff": (
                        "before the target UTC calendar-date batch"
                    ),
                }
            ])
        rows.extend([
            {
                "feature": f"{side}_team_prior_{role}_games",
                "definition": f"Prior non-neutral {role}-role games in the season.",
                "orientation": "sample count",
                "source_field": "completed game scores and neutral_site",
                "numerator": "count of valid prior role observations",
                "denominator": "not applicable",
                "aggregation_window": "same season, before the target UTC date",
                "missing_value_policy": "0 when no role observations exist.",
                "neutral_site_policy": "Count is descriptive; role value is not applied.",
                "availability_cutoff": "before the target UTC calendar-date batch",
            },
            {
                "feature": f"{side}_team_prior_overall_games",
                "definition": "Prior scored team games in any venue role.",
                "orientation": "sample count",
                "source_field": "completed game scores",
                "numerator": "count of valid prior team observations",
                "denominator": "not applicable",
                "aggregation_window": "same season, before the target UTC date",
                "missing_value_policy": "0 when no overall observations exist.",
                "neutral_site_policy": "Includes neutral-site results.",
                "availability_cutoff": "before the target UTC calendar-date batch",
            },
            {
                "feature": f"{side}_team_prior_{role}_effective_weight",
                "definition": "Effective contribution weight of the role estimate.",
                "orientation": "role estimate weight",
                "source_field": "role and overall game counts",
                "numerator": "role observation count",
                "denominator": "role observation count + 4 overall pseudo-games",
                "aggregation_window": "same season, before the target UTC date",
                "missing_value_policy": "0 when role data are absent.",
                "neutral_site_policy": "0; neutral targets use overall estimates directly.",
                "availability_cutoff": "before the target UTC calendar-date batch",
            },
        ])
        for metric in BASE_METRICS:
            rows.append({
                "feature": f"{side}_team_prior_overall_{metric}",
                "definition": f"Overall same-season team {metric}, including neutral games.",
                "orientation": metric,
                "source_field": (
                    "team points for" if metric == "points_for"
                    else "opponent points" if metric == "points_against"
                    else "game outcome"
                ),
                "numerator": "sum of prior metric observations",
                "denominator": "number of valid prior observations",
                "aggregation_window": "same season, before the target UTC date",
                "missing_value_policy": "NaN when no overall observations exist.",
                "neutral_site_policy": "Includes neutral-site results.",
                "availability_cutoff": "before the target UTC calendar-date batch",
            })
    return pd.DataFrame(rows).set_index("feature").loc[
        FEATURE_COLUMNS["stabilized"]
    ].reset_index()


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
                    elif formulation == "stabilized":
                        role_history = team_history[role]
                        overall_history = team_history["overall"]
                        role_games = int(
                            role_history.get("points_for", [0.0, 0.0])[1]
                        )
                        overall_games = int(
                            overall_history.get("points_for", [0.0, 0.0])[1]
                        )
                        effective_weight = (
                            role_games / (role_games + ROLE_PRIOR_GAMES)
                            if role_games
                            else 0.0
                        )
                        if neutral[position]:
                            effective_weight = 0.0
                        for metric in BASE_METRICS:
                            overall_value = overall.get(metric, np.nan)
                            role_value = role_stats.get(metric, np.nan)
                            if neutral[position]:
                                value = overall_value
                            elif np.isfinite(overall_value):
                                value = (
                                    (
                                        role_games * role_value
                                        + ROLE_PRIOR_GAMES * overall_value
                                    )
                                    / (role_games + ROLE_PRIOR_GAMES)
                                    if role_games
                                    else overall_value
                                )
                            elif role_games:
                                value = role_value
                                effective_weight = 1.0
                            else:
                                value = np.nan
                            name = (
                                f"{matchup_side}_team_prior_{role}_"
                                f"{metric}_shrunk"
                            )
                            output[position, column_index[name]] = value
                            overall_name = (
                                f"{matchup_side}_team_prior_overall_{metric}"
                            )
                            if metric in overall:
                                output[position, column_index[overall_name]] = (
                                    overall_value
                                )
                        output[
                            position,
                            column_index[
                                f"{matchup_side}_team_prior_{role}_games"
                            ],
                        ] = role_games
                        output[
                            position,
                            column_index[
                                f"{matchup_side}_team_prior_overall_games"
                            ],
                        ] = overall_games
                        output[
                            position,
                            column_index[
                                f"{matchup_side}_team_prior_{role}_effective_weight"
                            ],
                        ] = effective_weight
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
