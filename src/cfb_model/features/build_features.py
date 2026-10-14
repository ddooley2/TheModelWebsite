"""
build_features.py — CFB Model V5

Builds feature_table_modeling in DuckDB from:
  - stg_games_raw
  - stg_team_game_stats_raw
  - stg_roster_priors_raw
  - stg_weather_raw
  - stg_lines_normalized
  - weekly_team_ratings (built by build_ratings.py)

NaN strategy:
  - Rolling stats early-season: fill with prior-season conference-peer median,
    else 0.
  - Market features: DROP rows where market_spread_home is NaN.
  - Rating features: fill with 0 if still NaN.
  - Never fill with global mean across all data.
"""

import duckdb
import pandas as pd
import numpy as np

DB_PATH = "/Users/ddooley/Documents/SportsBetting/TheModel/cfb-model/db/warehouse.duckdb"

ROLLING_COLS = [
    "offense_ppa", "defense_ppa",
    "offense_successRate", "defense_successRate",
    "offense_explosiveness", "defense_explosiveness",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_tables(con):
    df_games = con.execute("""
        SELECT game_id, season, week, game_date, home_team, away_team,
               home_score, away_score, neutral_site, home_is_fbs, away_is_fbs
        FROM stg_games_raw
        WHERE season BETWEEN 2013 AND 2025
          AND home_score IS NOT NULL AND away_score IS NOT NULL
          AND home_is_fbs = TRUE AND away_is_fbs = TRUE
        ORDER BY season, week
    """).fetchdf()
    
    df_games["game_date"] = pd.to_datetime(df_games["game_date"])

    df_stats = con.execute("SELECT * FROM stg_team_game_stats_raw").fetchdf()
    df_weather = con.execute("SELECT game_id, weather_severity_composite, wind_mph FROM stg_weather_raw").fetchdf()
    df_lines = con.execute("""
        SELECT game_id,
               AVG(market_spread_home)  AS market_spread_home,
               AVG(market_total)        AS market_total
        FROM stg_lines_normalized
        GROUP BY game_id
    """).fetchdf()
    df_ratings = con.execute("SELECT * FROM weekly_team_ratings").fetchdf()
    df_priors = con.execute("""
        SELECT season, team, conference, team_talent, percentPPA,
               recruiting_rank, recruiting_points
        FROM stg_roster_priors_raw
    """).fetchdf()
    return df_games, df_stats, df_weather, df_lines, df_ratings, df_priors


# ---------------------------------------------------------------------------
# rest days
# ---------------------------------------------------------------------------

def _add_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rest days for home and away teams, capped at 14."""
    pieces = []
    for side, col in [("home", "home_team"), ("away", "away_team")]:
        tmp = df[["game_id", "game_date", col, "season"]].rename(columns={col: "team"})
        pieces.append(tmp)
    all_dates = pd.concat(pieces).sort_values(["team", "game_date"])
    all_dates["prev"] = all_dates.groupby("team")["game_date"].shift(1)
    all_dates["rest_days"] = (all_dates["game_date"] - all_dates["prev"]).dt.days
    all_dates["rest_days"] = all_dates["rest_days"].fillna(14).clip(upper=14)
    rest_map = all_dates.set_index(["game_id", "team"])["rest_days"].to_dict()

    df["home_rest_days"] = df.apply(
        lambda r: rest_map.get((r["game_id"], r["home_team"]), 14), axis=1
    )
    df["away_rest_days"] = df.apply(
        lambda r: rest_map.get((r["game_id"], r["away_team"]), 14), axis=1
    )
    return df


# ---------------------------------------------------------------------------
# rolling stats (shift-1 lag)
# ---------------------------------------------------------------------------

def _build_rolling_stats(df_stats: pd.DataFrame, df_games: pd.DataFrame) -> pd.DataFrame:
    """Compute per-team rolling stats with strict shift(1) lag."""
    date_map = df_games.set_index("game_id")["game_date"].to_dict()
    df_stats = df_stats.copy()
    df_stats["game_date"] = df_stats["game_id"].map(date_map)
    df_stats = df_stats.dropna(subset=["game_date"]).sort_values(["team", "game_date"])

    out_frames = []
    for team, grp in df_stats.groupby("team"):
        grp = grp.copy().sort_values("game_date")
        for col in ROLLING_COLS:
            if col not in grp.columns:
                grp[f"{col}_std"] = np.nan
                grp[f"{col}_l4"] = np.nan
                continue
            shifted = grp[col].shift(1)
            grp[f"{col}_std"] = grp.groupby("season")[col].transform(
                lambda x: x.shift(1).expanding().mean()
            )
            grp[f"{col}_l4"] = shifted.rolling(window=4, min_periods=1).mean()
        out_frames.append(grp)

    df_roll = pd.concat(out_frames, ignore_index=True)
    keep = ["game_id", "team"] + [f"{c}_{w}" for c in ROLLING_COLS for w in ("std", "l4")]
    return df_roll[keep]


def _build_conference_medians(df_stats: pd.DataFrame, df_priors: pd.DataFrame) -> pd.DataFrame:
    """For each (season, conference, stat), compute the median of the full-season
    values.  Used to fill early-game NaN in the NEXT season."""
    # Get conference assignments
    conf_map = df_priors.set_index(["season", "team"])["conference"].to_dict()
    df_s = df_stats.copy()
    df_s["conference"] = df_s.apply(
        lambda r: conf_map.get((r.get("season", 0), r["team"]), None), axis=1
    )
    df_s = df_s.dropna(subset=["conference"])
    records = []
    for (season, conf), g in df_s.groupby(["season", "conference"]):
        rec = {"season": season, "conference": conf}
        for col in ROLLING_COLS:
            if col in g.columns:
                rec[col] = g[col].median()
        records.append(rec)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# NaN filling for rolling stats
# ---------------------------------------------------------------------------

def _fill_rolling_nans(
    df: pd.DataFrame,
    df_priors: pd.DataFrame,
    conf_medians: pd.DataFrame,
    fill_log: dict,
) -> pd.DataFrame:
    """Fill NaN in rolling stat columns with prior-season conference-peer medians,
    else 0."""
    # Build lookup: (season, team) -> conference
    conf_map = df_priors.set_index(["season", "team"])["conference"].to_dict()

    # Build lookup: (season, conference, stat_base) -> median
    med_lookup: dict = {}
    for _, r in conf_medians.iterrows():
        for col in ROLLING_COLS:
            if col in r and pd.notna(r[col]):
                med_lookup[(r["season"], r["conference"], col)] = r[col]

    rolling_feat_cols = [f"{side}_{c}_{w}" for side in ("home", "away")
                         for c in ROLLING_COLS for w in ("std", "l4")]

    for fc in rolling_feat_cols:
        nan_mask = df[fc].isna()
        n_nan = nan_mask.sum()
        if n_nan == 0:
            continue

        filled = 0
        # Determine the side and base stat
        parts = fc.split("_", 1)  # home_ or away_
        side = parts[0]
        rest = parts[1]  # e.g. offense_ppa_std
        # base stat is everything up to the last _std or _l4
        base_stat = rest.rsplit("_", 1)[0]  # offense_ppa

        team_col = f"{side}_team"
        for idx in df[nan_mask].index:
            season = df.loc[idx, "season"]
            team = df.loc[idx, team_col]
            conf = conf_map.get((season, team), None)
            prior_season = season - 1
            val = None
            if conf:
                val = med_lookup.get((prior_season, conf, base_stat), None)
            if val is not None:
                df.loc[idx, fc] = val
                filled += 1

        # Fill remaining with 0
        still_nan = df[fc].isna().sum()
        df[fc] = df[fc].fillna(0.0)
        fill_log[fc] = f"{n_nan} NaN total → {filled} filled with conf median, {still_nan} filled with 0"

    return df


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def build_features():
    con = duckdb.connect(DB_PATH)
    print("Loading tables …")
    df_games, df_stats, df_weather, df_lines, df_ratings, df_priors = _load_tables(con)

    fill_log: dict = {}

    # Targets
    df_games["home_margin"] = df_games["home_score"] - df_games["away_score"]
    df_games["total_points"] = df_games["home_score"] + df_games["away_score"]

    # 1. Rest days
    df_games = _add_rest_days(df_games)

    # 2. Rolling stats
    print("Building rolling stats …")
    df_roll = _build_rolling_stats(df_stats, df_games)
    conf_medians = _build_conference_medians(df_stats, df_priors)

    # Join rolling stats for home
    df_roll_h = df_roll.copy()
    df_roll_h.columns = ["game_id", "home_team"] + [f"home_{c}" for c in df_roll.columns[2:]]
    df_games = df_games.merge(df_roll_h, on=["game_id", "home_team"], how="left")

    # Join rolling stats for away
    df_roll_a = df_roll.copy()
    df_roll_a.columns = ["game_id", "away_team"] + [f"away_{c}" for c in df_roll.columns[2:]]
    df_games = df_games.merge(df_roll_a, on=["game_id", "away_team"], how="left")

    # 3. Fill rolling NaN
    print("Filling rolling stat NaN …")
    df_games = _fill_rolling_nans(df_games, df_priors, conf_medians, fill_log)

    # 4. Weather
    df_games = df_games.merge(df_weather, on="game_id", how="left")
    n_w = df_games["weather_severity_composite"].isna().sum()
    df_games["weather_severity_composite"] = df_games["weather_severity_composite"].fillna(0.0)
    df_games["wind_mph"] = df_games["wind_mph"].fillna(0.0)
    fill_log["weather_severity_composite"] = f"{n_w} NaN → filled with 0 (indoor/missing)"
    fill_log["wind_mph"] = f"{df_games['wind_mph'].isna().sum()} NaN → filled with 0"

    # 5. Market lines
    df_games = df_games.merge(df_lines, on="game_id", how="left")
    n_before = len(df_games)
    df_games = df_games.dropna(subset=["market_spread_home", "market_total"]).copy()
    n_after = len(df_games)
    fill_log["market_spread_home"] = f"Dropped {n_before - n_after} rows with NaN market_spread_home or market_total"

    # Fill remaining market NaN
    n_m = df_games["market_total"].isna().sum()
    if n_m > 0:
        df_games["market_total"] = df_games["market_total"].fillna(0.0)
        fill_log["market_total"] = f"{n_m} NaN → filled with 0"

    # 6. Ratings
    print("Joining ratings …")
    rat_cols = [
        "team", "season", "week",
        "off_rating", "def_rating", "rating_uncertainty", "hfa", "points_intercept",
        "pace_rating", "pace_def_rating",
        "explosiveness_rating", "explosiveness_def_rating",
        "finishing_rating", "finishing_def_rating",
    ]
    # Only keep cols that exist
    rat_cols = [c for c in rat_cols if c in df_ratings.columns]
    df_rat = df_ratings[rat_cols].copy()

    # Home ratings
    df_rat_h = df_rat.rename(columns={
        "team": "home_team",
        "off_rating": "home_off_rating",
        "def_rating": "home_def_rating",
        "rating_uncertainty": "home_rating_uncertainty",
        "hfa": "hfa",
        "points_intercept": "points_intercept",
        "pace_rating": "home_pace_rating",
        "pace_def_rating": "home_pace_def_rating",
        "explosiveness_rating": "home_explosiveness_rating",
        "explosiveness_def_rating": "home_explosiveness_def_rating",
        "finishing_rating": "home_finishing_rating",
        "finishing_def_rating": "home_finishing_def_rating",
    })
    df_games = df_games.merge(df_rat_h, on=["season", "week", "home_team"], how="left")

    # Away ratings
    df_rat_a = df_rat.rename(columns={
        "team": "away_team",
        "off_rating": "away_off_rating",
        "def_rating": "away_def_rating",
        "rating_uncertainty": "away_rating_uncertainty",
        "pace_rating": "away_pace_rating",
        "pace_def_rating": "away_pace_def_rating",
        "explosiveness_rating": "away_explosiveness_rating",
        "explosiveness_def_rating": "away_explosiveness_def_rating",
        "finishing_rating": "away_finishing_rating",
        "finishing_def_rating": "away_finishing_def_rating",
    })
    # Drop hfa/intercept from away (already have from home join)
    for c in ["hfa", "points_intercept"]:
        if c in df_rat_a.columns:
            df_rat_a = df_rat_a.drop(columns=[c])
    df_games = df_games.merge(df_rat_a, on=["season", "week", "away_team"], how="left")

    # Fill rating NaN with 0
    rating_features = [c for c in df_games.columns if "rating" in c or c in ("hfa", "points_intercept")]
    for rc in rating_features:
        n_r = df_games[rc].isna().sum()
        if n_r > 0:
            df_games[rc] = df_games[rc].fillna(0.0)
            fill_log[rc] = f"{n_r} NaN → filled with 0"

    # 7. Engineered rating deltas
    df_games["off_delta"] = df_games["home_off_rating"] - df_games["away_off_rating"]
    df_games["def_delta"] = df_games["home_def_rating"] - df_games["away_def_rating"]
    df_games["net_rating_delta"] = (
        (df_games["home_off_rating"] + df_games["home_def_rating"])
        - (df_games["away_off_rating"] + df_games["away_def_rating"])
    )

    # rating_spread_implied: net_rating_delta + hfa for non-neutral, else net_rating_delta
    # Negative means home favored to match market_spread_home convention
    hfa_val = df_games["hfa"].fillna(0.0)
    df_games["rating_spread_implied"] = np.where(
        df_games["neutral_site"],
        -1.0 * df_games["net_rating_delta"],
        -1.0 * (df_games["net_rating_delta"] + hfa_val),
    )

    # rating_total_implied: sum of both offensive ratings + 2*intercept
    df_games["rating_total_implied"] = (
        df_games["home_off_rating"]
        + df_games["away_off_rating"]
        + 2 * df_games["points_intercept"].fillna(0.0)
    )

    # market_rating_disagreement: difference between implied and market
    # Negative disagreement means model thinks home should be favored by MORE than market
    df_games["market_rating_disagreement"] = (
        df_games["rating_spread_implied"] - df_games["market_spread_home"]
    )

    # 8. Preseason priors for both teams
    prior_cols_map = {
        "team_talent": "team_talent",
        "recruiting_rank": "recruiting_rank",
        "percentPPA": "returning_production_pct",
    }
    df_pr = df_priors[["season", "team"] + list(prior_cols_map.keys())].copy()
    df_pr = df_pr.rename(columns=prior_cols_map)
    df_pr = df_pr.drop_duplicates(["season", "team"])

    # Home priors
    df_pr_h = df_pr.rename(columns={
        "team": "home_team",
        "team_talent": "home_team_talent",
        "recruiting_rank": "home_recruiting_rank",
        "returning_production_pct": "home_returning_production_pct",
    })
    df_games = df_games.merge(df_pr_h, on=["season", "home_team"], how="left")

    # Away priors
    df_pr_a = df_pr.rename(columns={
        "team": "away_team",
        "team_talent": "away_team_talent",
        "recruiting_rank": "away_recruiting_rank",
        "returning_production_pct": "away_returning_production_pct",
    })
    df_games = df_games.merge(df_pr_a, on=["season", "away_team"], how="left")

    # Fill prior NaN
    for pc in ["home_team_talent", "away_team_talent", "home_recruiting_rank",
               "away_recruiting_rank", "home_returning_production_pct", "away_returning_production_pct"]:
        n_p = df_games[pc].isna().sum()
        if n_p > 0:
            df_games[pc] = df_games[pc].fillna(0.0)
            fill_log[pc] = f"{n_p} NaN → filled with 0"

    # 9. neutral_site as int for modeling
    df_games["neutral_site"] = df_games["neutral_site"].astype(int)

    # 10. Drop columns we don't want
    drop_cols = [
        "home_is_fbs", "away_is_fbs",  # keep for diagnostics but will exclude from features
        "home_team_id", "away_team_id", "status", "venue_id",
    ]
    for dc in drop_cols:
        if dc in df_games.columns:
            df_games = df_games.drop(columns=[dc], errors="ignore")

    # 11. Drop games with all-zero off_ppa artifact
    n_before_zero = len(df_games)
    zero_ppa_mask = (
        (df_games["home_offense_ppa_std"] == 0.0) & (df_games["home_offense_ppa_l4"] == 0.0)
    ) | (
        (df_games["away_offense_ppa_std"] == 0.0) & (df_games["away_offense_ppa_l4"] == 0.0)
    )
    df_games = df_games[~zero_ppa_mask].copy()
    fill_log["zero_ppa_artifact"] = f"Dropped {n_before_zero - len(df_games)} rows with 0.0 PPA stats"

    # Final write
    print(f"\n{'='*60}")
    print(f"Feature table summary:")
    print(f"  Rows:     {len(df_games)}")
    print(f"  Columns:  {len(df_games.columns)}")
    print(f"\nNaN fill log:")
    for k, v in sorted(fill_log.items()):
        print(f"  {k}: {v}")

    # Check remaining NaN
    nan_counts = df_games.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        print(f"\nRemaining NaN columns:")
        for c, n in nan_cols.items():
            print(f"  {c}: {n}")
    else:
        print("\n✓ No remaining NaN values")

    con.execute("DROP TABLE IF EXISTS feature_table_modeling")
    con.execute("CREATE TABLE feature_table_modeling AS SELECT * FROM df_games")
    print("\n✓ feature_table_modeling written to DuckDB")
    con.close()


if __name__ == "__main__":
    build_features()
