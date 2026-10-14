"""
build_ratings.py — CFB Model V5

Builds point-in-time weekly team ratings across 4 dimensions:
  1. Points  (home_score / away_score)
  2. Pace    (offense_plays)
  3. Explosiveness (offense_explosiveness)
  4. Finishing (points / drives)

Cross-season carry-forward:
  prior_S = 0.5 * final_rating_(S-1) + 0.3 * returning_production + 0.2 * talent

Recency weighting:
  w_g = exp(-ln2 * delta_weeks / tau),  tau=52

Sigmoid shrinkage:
  s_k = 1 / (1 + exp(-(k - 5) / 2))
  rating = (1 - s_k) * prior + s_k * in_season_ridge_rating
"""

import sys, os, warnings
import numpy as np
import pandas as pd
import duckdb

warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.insert(0, os.path.dirname(__file__))
from ridge_engine import RidgeRatingsEngine, _sigmoid_shrinkage, _recency_weight

DB_PATH = "/Users/ddooley/Documents/SportsBetting/TheModel/cfb-model/db/warehouse.duckdb"
START_SEASON = 2013
END_SEASON = 2025
TAU = 52.0


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

def load_games(con):
    q = f"""
    SELECT
        g.game_id, g.season, g.week, g.game_date,
        g.home_team, g.away_team,
        g.home_score, g.away_score, g.neutral_site,
        s_h.offense_plays   AS home_off_plays,
        s_h.offense_explosiveness AS home_off_exp,
        s_h.offense_drives  AS home_off_drives,
        s_a.offense_plays   AS away_off_plays,
        s_a.offense_explosiveness AS away_off_exp,
        s_a.offense_drives  AS away_off_drives
    FROM stg_games_raw g
    LEFT JOIN stg_team_game_stats_raw s_h
        ON g.game_id = s_h.game_id AND g.home_team = s_h.team
    LEFT JOIN stg_team_game_stats_raw s_a
        ON g.game_id = s_a.game_id AND g.away_team = s_a.team
    WHERE g.season BETWEEN {START_SEASON} AND {END_SEASON}
      AND g.home_score IS NOT NULL AND g.away_score IS NOT NULL
    ORDER BY g.season, g.week
    """
    df = con.execute(q).fetchdf()
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def load_priors(con):
    q = f"""
    SELECT season, team, team_talent, totalPPA, percentPPA
    FROM stg_roster_priors_raw
    WHERE season BETWEEN {START_SEASON} AND {END_SEASON}
    """
    return con.execute(q).fetchdf()


# ---------------------------------------------------------------------------
# prior construction
# ---------------------------------------------------------------------------

def _zscore(s):
    std = s.std()
    if std is None or std == 0 or pd.isna(std):
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def build_preseason_priors(df_priors_season, prev_final_ratings, metric_name):
    """Returns {team: (prior_off, prior_def)}."""
    df = df_priors_season.copy()
    for c in ["team_talent", "totalPPA"]:
        med = df[c].median()
        df[c] = df[c].fillna(med if pd.notna(med) else 0.0)

    z_talent = _zscore(df["team_talent"]).values
    z_prod = _zscore(df["totalPPA"]).values
    teams = df["team"].values

    scale = {"points": 4.0, "pace": 3.0, "explosiveness": 0.04, "finishing": 0.5}.get(metric_name, 1.0)

    priors = {}
    for i in range(len(df)):
        team = teams[i]
        tc = z_talent[i] * scale
        pc = z_prod[i] * scale

        if prev_final_ratings and team in prev_final_ratings:
            prev_off, prev_def = prev_final_ratings[team]
            p_off = 0.5 * prev_off + 0.3 * pc + 0.2 * tc
            p_def = 0.5 * prev_def + 0.3 * (-pc) + 0.2 * (-tc)
        else:
            p_off = 0.6 * pc + 0.4 * tc
            p_def = 0.6 * (-pc) + 0.4 * (-tc)
        priors[team] = (p_off, p_def)
    return priors


# ---------------------------------------------------------------------------
# vectorised observation builder
# ---------------------------------------------------------------------------

def _build_obs_vectorized(df_history, metric_name):
    """Build 2-row-per-game observations for a given metric, vectorized."""
    if len(df_history) == 0:
        return pd.DataFrame(columns=["team", "opponent", "is_home", "target", "game_week", "game_season"])

    df = df_history.copy()

    if metric_name == "points":
        df["h_val"] = df["home_score"]
        df["a_val"] = df["away_score"]
    elif metric_name == "pace":
        df["h_val"] = df["home_off_plays"]
        df["a_val"] = df["away_off_plays"]
    elif metric_name == "explosiveness":
        df["h_val"] = df["home_off_exp"]
        df["a_val"] = df["away_off_exp"]
    elif metric_name == "finishing":
        df["h_val"] = np.where(
            df["home_off_drives"].fillna(0) > 0,
            df["home_score"] / df["home_off_drives"],
            np.nan,
        )
        df["a_val"] = np.where(
            df["away_off_drives"].fillna(0) > 0,
            df["away_score"] / df["away_off_drives"],
            np.nan,
        )

    # Home rows
    home_mask = df["h_val"].notna()
    h = df[home_mask]
    home_rows = pd.DataFrame({
        "team": h["home_team"].values,
        "opponent": h["away_team"].values,
        "is_home": np.where(h["neutral_site"].values, 0, 1),
        "target": h["h_val"].values.astype(float),
        "game_week": h["week"].values.astype(int),
        "game_season": h["season"].values.astype(int),
    })

    # Away rows
    away_mask = df["a_val"].notna()
    a = df[away_mask]
    away_rows = pd.DataFrame({
        "team": a["away_team"].values,
        "opponent": a["home_team"].values,
        "is_home": np.zeros(len(a), dtype=int),
        "target": a["a_val"].values.astype(float),
        "game_week": a["week"].values.astype(int),
        "game_season": a["season"].values.astype(int),
    })

    return pd.concat([home_rows, away_rows], ignore_index=True)


def _compute_recency_weights_vec(df_obs, target_week, target_season):
    """Vectorized recency weights."""
    gs = df_obs["game_season"].values
    gw = df_obs["game_week"].values
    same_season = gs == target_season
    delta = np.where(
        same_season,
        target_week - gw,
        (16 - gw) + 34 + target_week,
    ).astype(float)
    delta = np.maximum(delta, 0)
    return np.exp(-np.log(2) * delta / TAU)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run_weekly_ratings():
    con = duckdb.connect(DB_PATH)
    print("Loading data …")
    df_games = load_games(con)
    df_priors_all = load_priors(con)

    metrics = ["points", "pace", "explosiveness", "finishing"]
    all_ratings = []
    seasons = sorted(df_games["season"].unique())
    prev_final = {m: None for m in metrics}

    for season in seasons:
        df_szn = df_games[df_games["season"] == season]
        df_priors_szn = df_priors_all[df_priors_all["season"] == season]
        
        prev_szn = season - 1
        if prev_szn == 2020:
            prev_szn = 2019
        df_prev = df_games[df_games["season"] == prev_szn]

        weeks = sorted(df_szn["week"].unique())
        if not weeks:
            continue
        max_week = max(weeks)

        for target_week in range(1, max_week + 2):
            df_history_szn = df_szn[df_szn["week"] < target_week]

            # Game counts for shrinkage
            game_counts = {}
            if len(df_history_szn) > 0:
                for t in df_history_szn["home_team"].values:
                    game_counts[t] = game_counts.get(t, 0) + 1
                for t in df_history_szn["away_team"].values:
                    game_counts[t] = game_counts.get(t, 0) + 1

            week_ratings = None
            for m_name in metrics:
                priors_dict = build_preseason_priors(df_priors_szn, prev_final[m_name], m_name)

                df_obs_szn = _build_obs_vectorized(df_history_szn, m_name)
                df_obs_prev = _build_obs_vectorized(df_prev, m_name)
                df_obs = pd.concat([df_obs_prev, df_obs_szn], ignore_index=True)

                weights = _compute_recency_weights_vec(df_obs, target_week, season) if len(df_obs) > 0 else None

                engine = RidgeRatingsEngine(dimension_name=m_name, ridge_alpha=1.0)
                pw = max(1.0, 6.0 - 0.3 * (target_week - 1))
                engine.fit(
                    df_obs[["team", "opponent", "is_home", "target"]] if len(df_obs) > 0
                    else pd.DataFrame(columns=["team", "opponent", "is_home", "target"]),
                    priors=priors_dict,
                    sample_weights=weights,
                    prior_weight=pw,
                )
                r_df = engine.get_ratings()

                # Sigmoid shrinkage
                final_off = np.empty(len(r_df))
                final_def = np.empty(len(r_df))
                for i, (_, rr) in enumerate(r_df.iterrows()):
                    team = rr["team"]
                    k = game_counts.get(team, 0)
                    s_k = _sigmoid_shrinkage(k, k0=5, beta=2)
                    p_off, p_def = priors_dict.get(team, (0.0, 0.0))
                    final_off[i] = (1 - s_k) * p_off + s_k * rr["offense_rating"]
                    final_def[i] = (1 - s_k) * p_def + s_k * rr["defense_rating"]
                r_df["offense_rating"] = final_off
                r_df["defense_rating"] = final_def

                rename_map = {
                    "offense_rating": f"{m_name}_offense_rating",
                    "defense_rating": f"{m_name}_defense_rating",
                    "rating_uncertainty": f"{m_name}_rating_uncertainty",
                    "intercept": f"{m_name}_intercept",
                    "hfa": f"{m_name}_hfa",
                }
                r_df = r_df.rename(columns=rename_map)

                if week_ratings is None:
                    week_ratings = r_df
                else:
                    week_ratings = pd.merge(week_ratings, r_df, on="team", how="outer")

            if week_ratings is not None:
                week_ratings["season"] = season
                week_ratings["week"] = target_week
                all_ratings.append(week_ratings)

        # Store final ratings for carry-forward
        if all_ratings:
            last = all_ratings[-1]
            for m_name in metrics:
                prev_final[m_name] = {}
                off_col = f"{m_name}_offense_rating"
                def_col = f"{m_name}_defense_rating"
                if off_col in last.columns and def_col in last.columns:
                    for _, rr in last.iterrows():
                        prev_final[m_name][rr["team"]] = (rr[off_col], rr[def_col])

        print(f"  Season {season} done  ({len(weeks)} weeks)")

    final_df = pd.concat(all_ratings, ignore_index=True)

    # Rename to final column names
    final_df = final_df.rename(columns={
        "points_offense_rating": "off_rating",
        "points_defense_rating": "def_rating",
        "points_rating_uncertainty": "rating_uncertainty",
        "points_hfa": "hfa",
        "pace_offense_rating": "pace_rating",
        "pace_defense_rating": "pace_def_rating",
        "pace_rating_uncertainty": "pace_uncertainty",
        "explosiveness_offense_rating": "explosiveness_rating",
        "explosiveness_defense_rating": "explosiveness_def_rating",
        "explosiveness_rating_uncertainty": "explosiveness_uncertainty",
        "finishing_offense_rating": "finishing_rating",
        "finishing_defense_rating": "finishing_def_rating",
        "finishing_rating_uncertainty": "finishing_uncertainty",
    })

    keep_cols = [
        "team", "season", "week",
        "off_rating", "def_rating", "rating_uncertainty", "hfa", "points_intercept",
        "pace_rating", "pace_def_rating",
        "explosiveness_rating", "explosiveness_def_rating",
        "finishing_rating", "finishing_def_rating",
    ]
    final_df = final_df[[c for c in keep_cols if c in final_df.columns]]

    print(f"\nTotal rating rows: {len(final_df)}")
    print(f"Teams: {final_df['team'].nunique()}")

    con.execute("DROP TABLE IF EXISTS weekly_team_ratings")
    con.execute("CREATE TABLE weekly_team_ratings AS SELECT * FROM final_df")
    print("✓ weekly_team_ratings written to DuckDB")
    con.close()


if __name__ == "__main__":
    run_weekly_ratings()
