"""Point-in-time preseason scoring priors with season-to-date shrinkage."""

from __future__ import annotations

import numpy as np
import pandas as pd

PRIOR_GAMES = 4.0
METRICS = ("points_for", "points_against")
SUPPORT_COLUMNS = (
    "prior_weight",
    "prior_current_games",
    "prior_season_games",
    "prior_available",
)
FEATURE_COLUMNS = [
    f"{side}_prior_{metric}_shrunk"
    for side in ("home", "away")
    for metric in METRICS
] + [
    f"{side}_{support}"
    for side in ("home", "away")
    for support in SUPPORT_COLUMNS
]


def build_preseason_score_prior_features(
    games: pd.DataFrame,
    *,
    prior_games: float = PRIOR_GAMES,
) -> pd.DataFrame:
    """Blend current-season scoring with the latest completed season's averages.

    Only completed results before each UTC calendar-date batch contribute. The
    prior-season pseudo-game weight is fixed, so its share falls as current-
    season games accumulate.
    """
    required = {
        "season", "game_date", "home_team", "away_team", "home_score", "away_score"
    }
    if missing := required - set(games.columns):
        raise ValueError(f"Games are missing columns: {sorted(missing)}")
    if not np.isfinite(prior_games) or prior_games <= 0:
        raise ValueError("prior_games must be a positive finite number.")
    if games.empty:
        return pd.DataFrame(index=games.index, columns=FEATURE_COLUMNS, dtype=float)

    canonical = games.reset_index(drop=True).copy()
    canonical["_day"] = pd.to_datetime(
        canonical["game_date"], utc=True, errors="coerce", format="mixed"
    ).dt.normalize()
    if canonical["_day"].isna().any():
        raise ValueError("Games contain invalid game dates.")
    canonical["_season"] = pd.to_numeric(canonical["season"], errors="coerce")
    if canonical["_season"].isna().any():
        raise ValueError("Games contain invalid seasons.")
    if canonical[["home_team", "away_team"]].isna().any().any():
        raise ValueError("Games contain missing team names.")

    team_rows = []
    for side in ("home", "away"):
        frame = canonical[["_day", "_season", f"{side}_team"]].copy()
        opponent_side = "away" if side == "home" else "home"
        frame = frame.rename(columns={f"{side}_team": "_team"})
        frame["_row"] = np.arange(len(canonical))
        frame["_side"] = side
        frame["_points_for"] = pd.to_numeric(
            canonical[f"{side}_score"], errors="coerce"
        ).to_numpy(dtype=float)
        frame["_points_against"] = pd.to_numeric(
            canonical[f"{opponent_side}_score"], errors="coerce"
        ).to_numpy(dtype=float)
        team_rows.append(frame)
    long = pd.concat(team_rows, ignore_index=True)
    long = long.sort_values(["_team", "_day", "_row"], kind="stable")

    summaries = []
    for team, team_rows in long.groupby("_team", sort=False, dropna=True):
        team_rows = team_rows.reset_index(drop=True)
        days = team_rows["_day"].astype("int64").to_numpy()
        seasons = team_rows["_season"].to_numpy(dtype=float)
        target_days = team_rows["_day"].drop_duplicates().sort_values()
        points_for = team_rows["_points_for"].to_numpy(dtype=float)
        points_against = team_rows["_points_against"].to_numpy(dtype=float)

        for target_day in target_days:
            target_ns = int(pd.Timestamp(target_day).value)
            end = int(np.searchsorted(days, target_ns, side="left"))
            target_seasons = team_rows.loc[
                team_rows["_day"] == target_day, "_season"
            ].unique()
            target_season = float(target_seasons[0])
            previous_seasons = seasons[:end]
            valid = np.isfinite(points_for[:end]) & np.isfinite(points_against[:end])
            current = valid & (previous_seasons == target_season)
            prior = valid & (previous_seasons < target_season)
            current_count = int(current.sum())

            prior_count = 0
            prior_for = np.nan
            prior_against = np.nan
            if prior.any():
                latest_prior_season = np.max(previous_seasons[prior])
                prior_rows = prior & (previous_seasons == latest_prior_season)
                prior_count = int(prior_rows.sum())
                prior_for = float(points_for[:end][prior_rows].mean())
                prior_against = float(points_against[:end][prior_rows].mean())

            current_for = float(points_for[:end][current].sum()) if current_count else 0.0
            current_against = (
                float(points_against[:end][current].sum()) if current_count else 0.0
            )
            has_prior = prior_count > 0
            denominator = current_count + prior_games if has_prior else current_count
            weight = prior_games / denominator if has_prior else 0.0
            if denominator:
                shrunk_for = (
                    current_for + prior_games * prior_for
                ) / denominator if has_prior else current_for / denominator
                shrunk_against = (
                    current_against + prior_games * prior_against
                ) / denominator if has_prior else current_against / denominator
            else:
                shrunk_for = np.nan
                shrunk_against = np.nan

            summaries.append({
                "_team": team,
                "_day": target_day,
                "_season": target_season,
                "prior_points_for_shrunk": shrunk_for,
                "prior_points_against_shrunk": shrunk_against,
                "prior_weight": weight,
                "prior_current_games": current_count,
                "prior_season_games": prior_count,
                "prior_available": int(has_prior),
            })

    summary = pd.DataFrame(summaries)
    expanded = long.merge(
        summary,
        on=["_team", "_day", "_season"],
        how="left",
        validate="many_to_one",
    )
    features = pd.DataFrame(index=np.arange(len(canonical)))
    for side in ("home", "away"):
        side_rows = expanded[expanded["_side"] == side].set_index("_row")
        for column in (
            "prior_points_for_shrunk",
            "prior_points_against_shrunk",
            *SUPPORT_COLUMNS,
        ):
            features[f"{side}_{column}"] = side_rows[column].reindex(
                features.index
            ).to_numpy()
    features.index = games.index
    return features[FEATURE_COLUMNS]


def preseason_score_prior_data_dictionary() -> pd.DataFrame:
    """Describe the score-prior features and their point-in-time semantics."""
    rows = []
    for side in ("home", "away"):
        for metric, score_field, orientation in (
            ("points_for", f"{side}_score", "team scoring"),
            ("points_against", f"{'away' if side == 'home' else 'home'}_score",
             "opponent scoring allowed"),
        ):
            rows.append({
                "feature": f"{side}_prior_{metric}_shrunk",
                "definition": (
                    "Current-season cumulative score blended with the latest "
                    "completed season's per-game average."
                ),
                "orientation": orientation,
                "source": "completed canonical FBS-vs-FBS game scores",
                "numerator": (
                    "current-season score sum + latest prior-season mean × 4"
                ),
                "denominator": (
                    "current-season games + 4 prior pseudo-games when prior "
                    "scores exist; otherwise current-season games"
                ),
                "aggregation_window": (
                    "current season to date, shrunk toward latest prior season"
                ),
                "missing_value_policy": (
                    "NaN when neither current-season nor prior-season scores exist; "
                    "LightGBM native missing-value path."
                ),
                "availability_cutoff": (
                    "before the target UTC calendar-date batch"
                ),
                "source_field": score_field,
            })
        for support, definition in (
            ("prior_weight", "Share of the shrunk estimate contributed by the prior."),
            ("prior_current_games", "Scored team games earlier in the current season."),
            ("prior_season_games", "Scored team games in the latest prior season."),
            ("prior_available", "Whether a completed prior-season score history exists."),
        ):
            rows.append({
                "feature": f"{side}_{support}",
                "definition": definition,
                "orientation": "support or missingness indicator",
                "source": "completed canonical FBS-vs-FBS game scores",
                "numerator": "not applicable",
                "denominator": "not applicable",
                "aggregation_window": (
                    "current season and latest prior season, before target date"
                ),
                "missing_value_policy": "0 when unavailable; scores remain NaN if absent.",
                "availability_cutoff": (
                    "before the target UTC calendar-date batch"
                ),
                "source_field": "game_date, season, team, home_score, away_score",
            })
    return pd.DataFrame(rows).set_index("feature").loc[FEATURE_COLUMNS].reset_index()
