"""Point-in-time team advanced-stat features (CFBD ``/stats/game/advanced``).

Each game's features summarize only the team's earlier games, shrunk toward the
team's most recent prior-season average so early-season rows stay stable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ADVANCED_METRICS = (
    "offense_ppa",
    "offense_successRate",
    "offense_explosiveness",
    "defense_ppa",
    "defense_successRate",
    "defense_explosiveness",
)
FEATURE_COLUMNS = [f"{side}_{metric}" for metric in ADVANCED_METRICS for side in ("home", "away")]

PRIOR_WEIGHT = 3.0


def _team_games(games: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(games["game_date"], utc=True, format="mixed")
    keys = pd.to_numeric(games.get("game_id", pd.Series(np.nan, index=games.index)), errors="coerce")
    frames = []
    for side in ("home", "away"):
        frames.append(pd.DataFrame({
            "row": np.arange(len(games)),
            "side": side,
            "season": games["season"].to_numpy(),
            "week": games["week"].to_numpy(),
            "game_date": dates.to_numpy(),
            "game_id": keys.to_numpy(),
            "team": games[f"{side}_team"].to_numpy(),
        }))
    return pd.concat(frames, ignore_index=True)


def _attach_stats(long: pd.DataFrame, advanced: pd.DataFrame) -> pd.DataFrame:
    """Join stats by CFBD game id, falling back to season/week/team for id-less rows."""
    metrics = list(ADVANCED_METRICS)
    by_id = advanced.dropna(subset=["game_id"]).drop_duplicates(["game_id", "team"])
    by_week = advanced[advanced["game_id"].isna()].drop_duplicates(["season", "week", "team"])
    merged = long.merge(
        by_id[["game_id", "team", *metrics]], on=["game_id", "team"], how="left"
    )
    fallback = long[["season", "week", "team"]].merge(
        by_week[["season", "week", "team", *metrics]], on=["season", "week", "team"], how="left"
    )
    for metric in metrics:
        merged[metric] = merged[metric].fillna(fallback[metric])
    return merged


def build_advanced_features(games: pd.DataFrame, advanced: pd.DataFrame) -> pd.DataFrame:
    """Return row-aligned home/away advanced features known before each game.

    ``advanced`` needs ``game_id`` (NaN for current-season rows), ``season``,
    ``week``, ``team``, and every column in ``ADVANCED_METRICS``.
    """
    long = _attach_stats(_team_games(games), advanced)
    long = long.sort_values(["team", "game_date", "row"], kind="stable").reset_index(drop=True)

    for metric in ADVANCED_METRICS:
        values = long[metric]
        present = values.notna().astype(float)
        total = values.fillna(0.0).groupby([long["team"], long["season"]]).cumsum() - values.fillna(0.0)
        count = present.groupby([long["team"], long["season"]]).cumsum() - present

        season_mean = long.groupby(["team", "season"])[metric].mean().reset_index()
        season_mean = season_mean.sort_values(["team", "season"])
        season_mean["prior_mean"] = season_mean.groupby("team")[metric].shift(1)
        prior = long[["team", "season"]].merge(
            season_mean[["team", "season", "prior_mean"]], on=["team", "season"], how="left"
        )["prior_mean"]

        weight = np.where(prior.notna(), PRIOR_WEIGHT, 0.0)
        denominator = count + weight
        numerator = total + weight * prior.fillna(0.0)
        long[f"_{metric}"] = np.where(denominator > 0, numerator / denominator.where(denominator > 0, 1), np.nan)

    output = pd.DataFrame(index=np.arange(len(games)))
    for side in ("home", "away"):
        rows = long[long["side"] == side].set_index("row")
        for metric in ADVANCED_METRICS:
            output[f"{side}_{metric}"] = rows[f"_{metric}"].reindex(output.index).to_numpy()
    return output[FEATURE_COLUMNS]
