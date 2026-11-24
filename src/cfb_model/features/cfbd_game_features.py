"""Point-in-time CFBD pace and efficiency features from per-game observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

RECENCY_HALF_LIFE_DAYS = 56
PRIOR_GAMES = 3.0

PACE_METRICS = (
    "offense_plays_per_game",
    "offense_drives_per_game",
    "defense_plays_allowed_per_game",
    "defense_drives_allowed_per_game",
)

EFFICIENCY_METRICS = (
    "off_ppa",
    "def_ppa",
    "off_success_rate",
    "def_success_rate",
    "off_explosiveness",
    "def_explosiveness",
    "off_pass_ppa",
    "def_pass_ppa",
    "off_pass_success_rate",
    "def_pass_success_rate",
    "off_pass_explosiveness",
    "def_pass_explosiveness",
    "off_rush_ppa",
    "def_rush_ppa",
    "off_rush_success_rate",
    "def_rush_success_rate",
    "off_rush_explosiveness",
    "def_rush_explosiveness",
)

BASE_GAME_STAT_COLUMNS = (
    "offense_plays",
    "offense_drives",
    "offense_ppa",
    "offense_totalPPA",
    "offense_successRate",
    "offense_explosiveness",
    "defense_plays",
    "defense_drives",
    "defense_ppa",
    "defense_totalPPA",
    "defense_successRate",
    "defense_explosiveness",
)
SPLIT_STAT_COLUMNS = tuple(
    f"{side}_{split}_{metric}"
    for side in ("offense", "defense")
    for split in ("passingPlays", "rushingPlays")
    for metric in ("ppa", "totalPPA", "successRate", "explosiveness")
)

PACE_FEATURE_COLUMNS = [
    f"{side}_pace_{metric}_{window}"
    for side in ("home", "away")
    for metric in PACE_METRICS
    for window in ("s2d", "recent")
] + [
    f"{side}_pace_{support}"
    for side in ("home", "away")
    for support in (
        "n_games_s2d",
        "n_plays_s2d",
        "n_drives_s2d",
        "n_games_recent_ess",
        "missing_s2d",
        "missing_recent",
    )
] + [
    f"{feature}_{window}"
    for feature in (
        "home_off_plays_minus_away_def_plays",
        "away_off_plays_minus_home_def_plays",
        "offense_plays_sum",
    )
    for window in ("s2d", "recent")
]

EFFICIENCY_FEATURE_COLUMNS = [
    f"{side}_eff_{metric}_{window}"
    for side in ("home", "away")
    for metric in EFFICIENCY_METRICS
    for window in ("s2d", "recent")
] + [
    f"{side}_eff_{support}"
    for side in ("home", "away")
    for support in (
        "n_games_s2d",
        "n_plays_s2d",
        "n_pass_plays_s2d",
        "n_rush_plays_s2d",
        "n_games_recent_ess",
        "missing_s2d",
        "missing_recent",
        "missing_metrics_s2d",
        "missing_metrics_recent",
    )
] + [
    f"{feature}_{window}"
    for feature in (
        "home_off_ppa_minus_away_def_ppa",
        "away_off_ppa_minus_home_def_ppa",
    )
    for window in ("s2d", "recent")
]


@dataclass(frozen=True)
class _MetricValues:
    numerator: np.ndarray
    denominator: np.ndarray
    source: str
    orientation: str
    units: str


def _game_id_key(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.notna(numeric) and float(numeric).is_integer():
        return str(int(numeric))
    return str(value).strip()


def combine_cfbd_game_stat_sources(
    game_stats: pd.DataFrame, split_stats: pd.DataFrame
) -> pd.DataFrame:
    """Join overall and split observations using validated game/team keys."""
    required_base = {
        "game_id", "season", "week", "team", "opponent", *BASE_GAME_STAT_COLUMNS
    }
    required_split = {
        "game_id", "season", "week", "team", "opponent", *SPLIT_STAT_COLUMNS
    }
    if missing := required_base - set(game_stats.columns):
        raise ValueError(f"Game stats are missing columns: {sorted(missing)}")
    if missing := required_split - set(split_stats.columns):
        raise ValueError(f"Split stats are missing columns: {sorted(missing)}")

    base = game_stats.copy()
    splits = split_stats[["game_id", "season", "week", "team", "opponent", *SPLIT_STAT_COLUMNS]].copy()
    for frame in (base, splits):
        frame["_game_id_key"] = frame["game_id"].map(_game_id_key)
        frame["team"] = frame["team"].astype(str)
    if base["_game_id_key"].isna().any() or splits["_game_id_key"].isna().any():
        raise ValueError("CFBD game stat sources require game IDs.")
    if base.duplicated(["_game_id_key", "team"]).any():
        raise ValueError("CFBD game stats contain duplicate (game_id, team) rows.")
    if splits.duplicated(["_game_id_key", "team"]).any():
        raise ValueError("CFBD split stats contain duplicate (game_id, team) rows.")

    splits = splits.rename(columns={
        "season": "_split_season",
        "week": "_split_week",
        "opponent": "_split_opponent",
    })
    merged = base.merge(
        splits[["_game_id_key", "team", "_split_season", "_split_week",
                "_split_opponent", *SPLIT_STAT_COLUMNS]],
        on=["_game_id_key", "team"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        counts = merged["_merge"].value_counts().to_dict()
        raise ValueError(f"CFBD overall/split game rows do not reconcile: {counts}")
    identity_matches = (
        merged["season"].eq(merged["_split_season"])
        & merged["week"].eq(merged["_split_week"])
        & merged["opponent"].eq(merged["_split_opponent"])
    )
    if not identity_matches.all():
        raise ValueError("CFBD overall and split rows disagree on game identity.")
    return merged.drop(
        columns=[
            "_game_id_key", "_split_season", "_split_week", "_split_opponent", "_merge"
        ]
    )


def _numeric(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    numeric[~np.isfinite(numeric)] = np.nan
    return numeric


def _metric_values(
    numerator: pd.Series | np.ndarray,
    denominator: pd.Series | np.ndarray,
    source: str,
    orientation: str,
    units: str,
) -> _MetricValues:
    numerator = np.asarray(numerator, dtype=float).copy()
    denominator = np.asarray(denominator, dtype=float).copy()
    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (denominator > 0)
    )
    numerator[~valid] = np.nan
    denominator[~valid] = np.nan
    return _MetricValues(numerator, denominator, source, orientation, units)


def _make_pace_metrics(long: pd.DataFrame) -> dict[str, _MetricValues]:
    one_game = np.ones(len(long), dtype=float)
    definitions = {
        "offense_plays_per_game": (
            "offense_plays", "offense_plays", "offense", "plays per game"
        ),
        "offense_drives_per_game": (
            "offense_drives", "offense_drives", "offense", "drives per game"
        ),
        "defense_plays_allowed_per_game": (
            "defense_plays", "defense_plays", "defense allowed", "plays per game"
        ),
        "defense_drives_allowed_per_game": (
            "defense_drives", "defense_drives", "defense allowed", "drives per game"
        ),
    }
    result = {}
    for name, (field, _, orientation, units) in definitions.items():
        result[name] = _metric_values(
            _numeric(long[field]), one_game, field, orientation, units
        )
    return result


def _split_play_count(total: np.ndarray, rate: np.ndarray) -> np.ndarray:
    count = np.full(len(rate), np.nan, dtype=float)
    valid = np.isfinite(total) & np.isfinite(rate) & (np.abs(rate) > 1e-12)
    ratios = total[valid] / rate[valid]
    rounded = np.rint(ratios)
    reliable = np.isfinite(ratios) & (rounded > 0) & np.isclose(
        ratios, rounded, atol=1e-8, rtol=0
    )
    positions = np.flatnonzero(valid)
    count[positions[reliable]] = rounded[reliable]
    return count


def _make_efficiency_metrics(long: pd.DataFrame) -> dict[str, _MetricValues]:
    result: dict[str, _MetricValues] = {}
    for side, prefix in (("offense", "off"), ("defense", "def")):
        plays = _numeric(long[f"{side}_plays"])
        ppa = _numeric(long[f"{side}_ppa"])
        total_ppa = _numeric(long[f"{side}_totalPPA"])
        rate = _numeric(long[f"{side}_successRate"])
        explosiveness = _numeric(long[f"{side}_explosiveness"])
        total_ppa = np.where(
            np.isfinite(total_ppa),
            total_ppa,
            ppa * plays,
        )
        successful_plays = rate * plays
        result[f"{prefix}_ppa"] = _metric_values(
            total_ppa, plays, f"{side}_totalPPA / {side}_plays",
            "offense" if side == "offense" else "defense allowed", "PPA per play"
        )
        result[f"{prefix}_success_rate"] = _metric_values(
            rate * plays, plays, f"{side}_successRate weighted by {side}_plays",
            "offense" if side == "offense" else "defense allowed", "successful plays / plays"
        )
        result[f"{prefix}_explosiveness"] = _metric_values(
            explosiveness * successful_plays, successful_plays,
            f"{side}_explosiveness weighted by successful plays",
            "offense" if side == "offense" else "defense allowed",
            "PPA per successful play",
        )

        for split, split_key in (("passingPlays", "pass"), ("rushingPlays", "rush")):
            split_ppa = _numeric(long[f"{side}_{split}_ppa"])
            split_total = _numeric(long[f"{side}_{split}_totalPPA"])
            split_count = _split_play_count(split_total, split_ppa)
            split_success = _numeric(long[f"{side}_{split}_successRate"])
            split_explosive = _numeric(long[f"{side}_{split}_explosiveness"])
            split_successful = split_success * split_count
            result[f"{prefix}_{split_key}_ppa"] = _metric_values(
                split_total,
                split_count,
                f"{side}_{split}_totalPPA / inferred {split} plays",
                "offense" if side == "offense" else "defense allowed",
                "PPA per split play",
            )
            result[f"{prefix}_{split_key}_success_rate"] = _metric_values(
                split_success * split_count,
                split_count,
                f"{side}_{split}_successRate weighted by inferred {split} plays",
                "offense" if side == "offense" else "defense allowed",
                "successful split plays / split plays",
            )
            result[f"{prefix}_{split_key}_explosiveness"] = _metric_values(
                split_explosive * split_successful,
                split_successful,
                f"{side}_{split}_explosiveness weighted by successful {split} plays",
                "offense" if side == "offense" else "defense allowed",
                "PPA per successful split play",
            )
    return result


def _prepare_team_games(
    games: pd.DataFrame, stats: pd.DataFrame
) -> pd.DataFrame:
    required_games = {
        "game_id", "season", "week", "game_date", "home_team", "away_team"
    }
    if missing := required_games - set(games.columns):
        raise ValueError(f"Games are missing columns: {sorted(missing)}")
    if games["game_id"].duplicated().any():
        raise ValueError("Canonical games contain duplicate game IDs.")
    if stats.empty and games.empty:
        return pd.DataFrame(columns=[
            "_row", "_game_id_key", "_day", "season", "week", "team", "opponent",
            "_side", "_stats_present", *BASE_GAME_STAT_COLUMNS, *SPLIT_STAT_COLUMNS,
        ])
    stats = stats.copy()
    stats["_game_id_key"] = stats["game_id"].map(_game_id_key)
    stats["team"] = stats["team"].astype(str)
    if stats["_game_id_key"].isna().any():
        raise ValueError("CFBD game stats require game IDs.")
    if stats.duplicated(["_game_id_key", "team"]).any():
        raise ValueError("CFBD stats contain duplicate (game_id, team) rows.")

    canonical = games.reset_index(drop=True).copy()
    canonical["_row"] = np.arange(len(canonical))
    canonical["_game_id_key"] = canonical["game_id"].map(_game_id_key)
    canonical["_day"] = pd.to_datetime(
        canonical["game_date"], utc=True, errors="coerce", format="mixed"
    ).dt.normalize()
    if canonical["_day"].isna().any():
        raise ValueError("Canonical games contain invalid game dates.")

    team_rows = []
    for side in ("home", "away"):
        frame = canonical[[
            "_row", "_game_id_key", "_day", "season", "week",
            f"{side}_team", f"{'away' if side == 'home' else 'home'}_team",
        ]].copy()
        frame = frame.rename(columns={
            f"{side}_team": "team",
            f"{'away' if side == 'home' else 'home'}_team": "opponent",
        })
        frame["team"] = frame["team"].astype(str)
        frame["opponent"] = frame["opponent"].astype(str)
        frame["_side"] = side
        team_rows.append(frame)
    long = pd.concat(team_rows, ignore_index=True)

    source = stats
    source = source.rename(columns={
        "season": "_stats_season",
        "week": "_stats_week",
        "opponent": "_stats_opponent",
    })
    source_columns = [
        "_game_id_key", "team", "_stats_season", "_stats_week",
        "_stats_opponent", *BASE_GAME_STAT_COLUMNS, *SPLIT_STAT_COLUMNS,
    ]
    long = long.merge(
        source[source_columns],
        on=["_game_id_key", "team"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    matched = long["_merge"].eq("both")
    identity_matches = (
        long.loc[matched, "season"].eq(long.loc[matched, "_stats_season"])
        & long.loc[matched, "week"].eq(long.loc[matched, "_stats_week"])
        & long.loc[matched, "opponent"].eq(long.loc[matched, "_stats_opponent"])
    )
    if not identity_matches.all():
        raise ValueError("CFBD team-game records disagree with canonical game identity.")
    long["_stats_present"] = matched.to_numpy()
    return long.drop(columns=["_merge", "_game_id_key"])


def _team_summaries(
    long: pd.DataFrame,
    metrics: dict[str, _MetricValues],
    group: str,
    core_metric: str,
) -> tuple[pd.DataFrame, list[str]]:
    working = long[["_row", "_day", "season", "team", "_side", "_stats_present"]].copy()
    for name, values in metrics.items():
        working[f"_num_{name}"] = values.numerator
        working[f"_den_{name}"] = values.denominator
    working = working.sort_values(["team", "_day", "_row"], kind="stable")
    summary_rows = []
    result_columns = []

    for team, team_rows in working.groupby("team", sort=False):
        team_rows = team_rows.reset_index(drop=True)
        day_ns = team_rows["_day"].astype("int64").to_numpy()
        seasons = pd.to_numeric(team_rows["season"], errors="coerce").to_numpy(dtype=float)
        target_days = team_rows["_day"].drop_duplicates().sort_values()
        arrays = {
            name: (
                team_rows[f"_num_{name}"].to_numpy(dtype=float),
                team_rows[f"_den_{name}"].to_numpy(dtype=float),
            )
            for name in metrics
        }

        for target_day in target_days:
            target_ns = int(pd.Timestamp(target_day).value)
            end = int(np.searchsorted(day_ns, target_ns, side="left"))
            target_seasons = team_rows.loc[
                team_rows["_day"] == target_day, "season"
            ].dropna().unique()
            if not len(target_seasons):
                continue
            target_season = int(target_seasons[0])
            prior_seasons = seasons[:end]
            ages = (target_ns - day_ns[:end]) / 86_400_000_000_000
            weights = np.exp2(-ages / RECENCY_HALF_LIFE_DAYS)
            summary: dict[str, float | int | str | pd.Timestamp] = {
                "team": team,
                "_day": target_day,
                "season": target_season,
            }
            core_current_count = 0
            core_recent_ess = 0.0
            core_current_denominator = 0.0
            core_current_numerator = 0.0
            for name, (numerator, denominator) in arrays.items():
                numerator = numerator[:end]
                denominator = denominator[:end]
                valid = (
                    np.isfinite(numerator)
                    & np.isfinite(denominator)
                    & (denominator > 0)
                )
                current = valid & (prior_seasons == target_season)
                current_count = int(current.sum())
                current_num = float(numerator[current].sum()) if current_count else 0.0
                current_den = float(denominator[current].sum()) if current_count else 0.0

                earlier = valid & (prior_seasons < target_season)
                prior_rate = np.nan
                prior_denominator = 0.0
                if earlier.any():
                    latest_prior_season = int(prior_seasons[earlier].max())
                    prior_rows = earlier & (prior_seasons == latest_prior_season)
                    prior_den_sum = float(denominator[prior_rows].sum())
                    prior_count = int(prior_rows.sum())
                    if prior_den_sum > 0 and prior_count:
                        prior_rate = float(numerator[prior_rows].sum() / prior_den_sum)
                        prior_denominator = (
                            PRIOR_GAMES * prior_den_sum / prior_count
                        )

                shrink_denominator = prior_denominator if np.isfinite(prior_rate) else 0.0
                season_denominator = current_den + shrink_denominator
                season_value = (
                    (current_num + prior_rate * shrink_denominator)
                    / season_denominator
                    if season_denominator > 0
                    else np.nan
                )

                recent_valid = valid
                recent_denominator = float(np.sum(weights[recent_valid] * denominator[recent_valid]))
                recent_value = (
                    float(np.sum(weights[recent_valid] * numerator[recent_valid]) / recent_denominator)
                    if recent_denominator > 0
                    else np.nan
                )
                weight_sum = float(weights[recent_valid].sum())
                weight_square_sum = float(np.square(weights[recent_valid]).sum())
                recent_ess = (
                    weight_sum * weight_sum / weight_square_sum
                    if weight_square_sum > 0
                    else 0.0
                )
                summary[f"{group}_{name}_s2d"] = float(season_value)
                summary[f"{group}_{name}_recent"] = float(recent_value)
                result_columns.extend(
                    [f"{group}_{name}_s2d", f"{group}_{name}_recent"]
                )
                if name == core_metric:
                    core_current_count = current_count
                    core_recent_ess = recent_ess
                    core_current_denominator = current_den
                    core_current_numerator = current_num

            support = {
                f"{group}_n_games_s2d": core_current_count,
                f"{group}_n_games_recent_ess": core_recent_ess,
                f"{group}_n_plays_s2d": (
                    core_current_numerator if group == "pace" else core_current_denominator
                ),
            }
            if group == "pace":
                support[f"{group}_missing_s2d"] = int(
                    not any(
                        np.isfinite(summary[f"{group}_{name}_s2d"])
                        for name in metrics
                    )
                )
                support[f"{group}_missing_recent"] = int(
                    not any(
                        np.isfinite(summary[f"{group}_{name}_recent"])
                        for name in metrics
                    )
                )
                drive_array = arrays["offense_drives_per_game"][0][:end]
                drive_valid = (
                    np.isfinite(drive_array)
                    & np.isfinite(arrays["offense_drives_per_game"][1][:end])
                    & (prior_seasons == target_season)
                )
                support[f"{group}_n_drives_s2d"] = float(drive_array[drive_valid].sum())
            else:
                pass_den = arrays["off_pass_ppa"][1][:end]
                rush_den = arrays["off_rush_ppa"][1][:end]
                support[f"{group}_n_pass_plays_s2d"] = float(
                    pass_den[np.isfinite(pass_den) & (prior_seasons == target_season)].sum()
                )
                support[f"{group}_n_rush_plays_s2d"] = float(
                    rush_den[np.isfinite(rush_den) & (prior_seasons == target_season)].sum()
                )
                s2d_values = [
                    summary[f"{group}_{name}_s2d"] for name in metrics
                ]
                recent_values = [
                    summary[f"{group}_{name}_recent"] for name in metrics
                ]
                support[f"{group}_missing_s2d"] = int(
                    not any(np.isfinite(value) for value in s2d_values)
                )
                support[f"{group}_missing_recent"] = int(
                    not any(np.isfinite(value) for value in recent_values)
                )
                support[f"{group}_missing_metrics_s2d"] = int(
                    sum(not np.isfinite(value) for value in s2d_values)
                )
                support[f"{group}_missing_metrics_recent"] = int(
                    sum(not np.isfinite(value) for value in recent_values)
                )
            summary.update(support)
            result_columns.extend(support)
            summary_rows.append(summary)

    summarized = pd.DataFrame(summary_rows)
    result_columns = list(dict.fromkeys(result_columns))
    wide = pd.DataFrame(index=np.arange(long["_row"].max() + 1))
    expanded = long.merge(
        summarized,
        on=["team", "season", "_day"],
        how="left",
        validate="many_to_one",
    )
    for side in ("home", "away"):
        side_rows = expanded[expanded["_side"] == side].set_index("_row")
        for column in result_columns:
            wide[f"{side}_{column}"] = side_rows[column].reindex(wide.index).to_numpy()

    return wide, result_columns


def _add_pace_matchups(features: pd.DataFrame) -> pd.DataFrame:
    for window in ("s2d", "recent"):
        home_off = features[f"home_pace_offense_plays_per_game_{window}"]
        away_off = features[f"away_pace_offense_plays_per_game_{window}"]
        home_def = features[f"home_pace_defense_plays_allowed_per_game_{window}"]
        away_def = features[f"away_pace_defense_plays_allowed_per_game_{window}"]
        features[f"home_off_plays_minus_away_def_plays_{window}"] = home_off - away_def
        features[f"away_off_plays_minus_home_def_plays_{window}"] = away_off - home_def
        features[f"offense_plays_sum_{window}"] = home_off + away_off
    return features[PACE_FEATURE_COLUMNS]


def _add_efficiency_matchups(features: pd.DataFrame) -> pd.DataFrame:
    for window in ("s2d", "recent"):
        features[f"home_off_ppa_minus_away_def_ppa_{window}"] = (
            features[f"home_eff_off_ppa_{window}"]
            - features[f"away_eff_def_ppa_{window}"]
        )
        features[f"away_off_ppa_minus_home_def_ppa_{window}"] = (
            features[f"away_eff_off_ppa_{window}"]
            - features[f"home_eff_def_ppa_{window}"]
        )
    return features[EFFICIENCY_FEATURE_COLUMNS]


def _team_rows(games: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    return _prepare_team_games(games, stats)


def build_cfbd_pace_features(games: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Build 4A pace features using only prior-date FBS-vs-FBS observations."""
    if games.empty:
        return pd.DataFrame(index=games.index, columns=PACE_FEATURE_COLUMNS, dtype=float)
    long = _team_rows(games, stats)
    metrics = _make_pace_metrics(long)
    wide, _ = _team_summaries(long, metrics, "pace", "offense_plays_per_game")
    wide.index = games.index
    return _add_pace_matchups(wide)


def build_cfbd_efficiency_features(
    games: pd.DataFrame, stats: pd.DataFrame
) -> pd.DataFrame:
    """Build 4B season-to-date and 56-day weighted CFBD efficiency features."""
    if games.empty:
        return pd.DataFrame(index=games.index, columns=EFFICIENCY_FEATURE_COLUMNS, dtype=float)
    long = _team_rows(games, stats)
    metrics = _make_efficiency_metrics(long)
    wide, _ = _team_summaries(long, metrics, "eff", "off_ppa")
    wide.index = games.index
    return _add_efficiency_matchups(wide)


def cfbd_feature_data_dictionary() -> pd.DataFrame:
    """Describe feature orientation, source, denominators, window, and missingness."""
    rows = []
    efficiency_sources = {
        "off_ppa": ("offense_totalPPA", "offense_plays"),
        "def_ppa": ("defense_totalPPA", "defense_plays"),
        "off_success_rate": ("offense_successRate", "offense_plays"),
        "def_success_rate": ("defense_successRate", "defense_plays"),
        "off_explosiveness": ("offense_explosiveness", "successful offense plays"),
        "def_explosiveness": ("defense_explosiveness", "successful defense plays allowed"),
        "off_pass_ppa": (
            "offense_passingPlays_totalPPA",
            "inferred passing plays (totalPPA / PPA)",
        ),
        "def_pass_ppa": (
            "defense_passingPlays_totalPPA",
            "inferred passing plays allowed (totalPPA / PPA)",
        ),
        "off_pass_success_rate": (
            "offense_passingPlays_successRate",
            "inferred passing plays",
        ),
        "def_pass_success_rate": (
            "defense_passingPlays_successRate",
            "inferred passing plays allowed",
        ),
        "off_pass_explosiveness": (
            "offense_passingPlays_explosiveness",
            "successful inferred passing plays",
        ),
        "def_pass_explosiveness": (
            "defense_passingPlays_explosiveness",
            "successful inferred passing plays allowed",
        ),
        "off_rush_ppa": (
            "offense_rushingPlays_totalPPA",
            "inferred rushing plays (totalPPA / PPA)",
        ),
        "def_rush_ppa": (
            "defense_rushingPlays_totalPPA",
            "inferred rushing plays allowed (totalPPA / PPA)",
        ),
        "off_rush_success_rate": (
            "offense_rushingPlays_successRate",
            "inferred rushing plays",
        ),
        "def_rush_success_rate": (
            "defense_rushingPlays_successRate",
            "inferred rushing plays allowed",
        ),
        "off_rush_explosiveness": (
            "offense_rushingPlays_explosiveness",
            "successful inferred rushing plays",
        ),
        "def_rush_explosiveness": (
            "defense_rushingPlays_explosiveness",
            "successful inferred rushing plays allowed",
        ),
    }
    for group, metrics in (
        ("pace", PACE_METRICS),
        ("efficiency", EFFICIENCY_METRICS),
    ):
        sides = ("home", "away")
        for side in sides:
            for metric in metrics:
                if group == "pace":
                    source = {
                        "offense_plays_per_game": "offense_plays",
                        "offense_drives_per_game": "offense_drives",
                        "defense_plays_allowed_per_game": "defense_plays",
                        "defense_drives_allowed_per_game": "defense_drives",
                    }[metric]
                    numerator, denominator = source, "valid completed game count"
                    orientation = (
                        "team offense" if metric.startswith("offense_")
                        else "team defense plays/drives allowed"
                    )
                else:
                    orientation = (
                        "team offense" if metric.startswith("off_")
                        else "team defense allowed"
                    )
                    source = efficiency_sources[metric][0]
                    denominator = efficiency_sources[metric][1]
                    numerator = (
                        "totalPPA"
                        if metric.endswith("_ppa")
                        else "successRate x plays"
                        if metric.endswith("_success_rate")
                        else "explosiveness x successful plays"
                    )
                for window in ("s2d", "recent"):
                    rows.append({
                        "feature": f"{side}_{group}_{metric}_{window}",
                        "definition": f"{side} team {orientation} {source}",
                        "source_field": source,
                        "numerator": numerator,
                        "denominator": denominator,
                        "aggregation_window": (
                            "prior same-season games, shrunk by 3 pseudo-games from the "
                            "latest earlier season"
                            if window == "s2d"
                            else f"all prior games, exponential half-life {RECENCY_HALF_LIFE_DAYS} days"
                        ),
                        "availability_cutoff": "before the target UTC calendar-date batch",
                        "missing_policy": "LightGBM native NaN; sample-size and missing-count features included",
                        "neutral_site": "venue does not alter team-level rate orientation",
                    })
    for feature in (*PACE_FEATURE_COLUMNS, *EFFICIENCY_FEATURE_COLUMNS):
        if any(feature == row["feature"] for row in rows):
            continue
        support_definitions = {
            "pace_n_games_s2d": (
                "Number of valid prior same-season team-games for pace rates."
            ),
            "pace_n_plays_s2d": (
                "Offensive plays observed in valid prior same-season pace records."
            ),
            "pace_n_drives_s2d": (
                "Offensive drives observed in valid prior same-season pace records."
            ),
            "pace_n_games_recent_ess": (
                "Effective prior-game sample size: (sum weights)^2 / sum squared weights."
            ),
            "pace_missing_s2d": "1 when no season-to-date or prior-season pace rate is available.",
            "pace_missing_recent": "1 when no recency-weighted pace rate is available.",
            "eff_n_games_s2d": (
                "Number of valid prior same-season team-games contributing efficiency rates."
            ),
            "eff_n_plays_s2d": "Prior same-season offensive plays used as the overall PPA denominator.",
            "eff_n_pass_plays_s2d": (
                "Prior same-season inferred passing plays used as split denominators."
            ),
            "eff_n_rush_plays_s2d": (
                "Prior same-season inferred rushing plays used as split denominators."
            ),
            "eff_n_games_recent_ess": (
                "Effective prior-game sample size: (sum weights)^2 / sum squared weights."
            ),
            "eff_missing_s2d": "1 when no season-to-date or prior-season efficiency rate is available.",
            "eff_missing_recent": "1 when no recency-weighted efficiency rate is available.",
            "eff_missing_metrics_s2d": "Count of efficiency rates unavailable in the season-to-date window.",
            "eff_missing_metrics_recent": "Count of efficiency rates unavailable in the recency window.",
        }
        feature_tail = feature.removeprefix("home_").removeprefix("away_")
        if feature_tail in support_definitions:
            definition = support_definitions[feature_tail]
        elif feature in {
            f"{name}_{window}"
            for name in (
                "home_off_plays_minus_away_def_plays",
                "away_off_plays_minus_home_def_plays",
                "offense_plays_sum",
                "home_off_ppa_minus_away_def_ppa",
                "away_off_ppa_minus_home_def_ppa",
            )
            for window in ("s2d", "recent")
        }:
            definition = f"Explicit matchup contrast: {feature.removesuffix('_s2d').removesuffix('_recent')}."
        else:
            definition = "Derived sample-size or missingness feature."
        rows.append({
            "feature": feature,
            "definition": definition,
            "source_field": "Derived from the listed CFBD per-game observations",
            "numerator": "See definition",
            "denominator": "See definition",
            "aggregation_window": (
                "Season-to-date or recency window corresponding to the feature name"
            ),
            "availability_cutoff": "before the target UTC calendar-date batch",
            "missing_policy": "Counts use zero when no prior observation exists; rate inputs remain NaN",
            "neutral_site": "No home/away role split is applied to neutral-site history",
        })
    return pd.DataFrame(rows)


def cfbd_stat_coverage(games: pd.DataFrame, stats: pd.DataFrame) -> dict:
    """Count FBS team-game rows with and without a validated per-game stat record."""
    long = _prepare_team_games(games, stats)
    season_coverage = {}
    for season, rows in long.groupby("season", sort=True):
        matched = int(rows["_stats_present"].sum())
        season_coverage[str(int(season))] = {
            "canonical_team_game_rows": int(len(rows)),
            "matched_team_game_rows": matched,
            "missing_team_game_rows": int(len(rows) - matched),
            "coverage_rate": float(matched / len(rows)) if len(rows) else 0.0,
        }
    return {
        "canonical_team_game_rows": int(len(long)),
        "matched_team_game_rows": int(long["_stats_present"].sum()),
        "missing_team_game_rows": int((~long["_stats_present"]).sum()),
        "coverage_rate": (
            float(long["_stats_present"].mean()) if len(long) else 0.0
        ),
        "by_season": season_coverage,
    }
