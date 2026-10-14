import requests
import pandas as pd
import duckdb
import os
import time

import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("CFBD_API_KEY")
if not API_KEY:
    raise ValueError("CFBD_API_KEY environment variable not set")
HEADERS = {'Authorization': f'Bearer {API_KEY}'}
BASE_URL = 'https://api.collegefootballdata.com'
YEARS = [y for y in range(2001, 2026) if y != 2020]

BASE_DIR = "/Users/ddooley/Documents/SportsBetting/TheModel/cfb-model"
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
DB_PATH = os.path.join(BASE_DIR, "db", "warehouse.duckdb")

def make_request(endpoint, params):
    url = f"{BASE_URL}{endpoint}"
    for _ in range(3):
        r = requests.get(url, headers=HEADERS, params=params)
        if r.status_code == 200:
            return r.json()
        elif r.status_code == 429:
            time.sleep(2)
        else:
            print(f"Error {r.status_code} for {url} {params}")
            return []
    return []

def american_to_prob(ml):
    if pd.isna(ml):
        return None
    ml = float(ml)
    if ml > 0:
        return 100.0 / (ml + 100.0)
    elif ml < 0:
        return -ml / (-ml + 100.0)
    return None

def main():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    con = duckdb.connect(DB_PATH)
    
    # === Step 1: Teams, Conferences, Venues ===
    fbs_teams_list = []
    print("Fetching FBS teams...")
    for yr in YEARS:
        teams = make_request("/teams/fbs", {"year": yr})
        for t in teams:
            fbs_teams_list.append({
                'team_id': t['id'],
                'team_name': t['school'],
                'season': yr,
                'conference': t.get('conference')
            })
        time.sleep(0.1)
    
    df_fbs_teams = pd.DataFrame(fbs_teams_list)
    
    # stg_team_lookup_raw
    df_team_lookup = df_fbs_teams[['team_id', 'team_name']].drop_duplicates()
    df_team_lookup.to_parquet(os.path.join(RAW_DATA_DIR, "stg_team_lookup_raw.parquet"))
    con.execute("CREATE OR REPLACE TABLE stg_team_lookup_raw AS SELECT * FROM df_team_lookup")
    
    # stg_conference_history_raw
    df_conf_history = df_fbs_teams[['team_id', 'season', 'conference']].copy()
    df_conf_history.rename(columns={'conference': 'conference_name_raw'}, inplace=True)
    df_conf_history['conference_name_canonical'] = df_conf_history['conference_name_raw']
    df_conf_history.sort_values(['team_id', 'season'], inplace=True)
    
    # Realignment flag
    df_conf_history['prior_conference'] = df_conf_history.groupby('team_id')['conference_name_raw'].shift(1)
    df_conf_history['prior_season'] = df_conf_history.groupby('team_id')['season'].shift(1)
    df_conf_history['realignment_flag'] = (df_conf_history['prior_conference'] != df_conf_history['conference_name_raw']) & (df_conf_history['prior_season'] == df_conf_history['season'] - 1)
    df_conf_history['notes'] = None
    df_conf_history.drop(columns=['prior_conference', 'prior_season'], inplace=True)
    
    df_conf_history.to_parquet(os.path.join(RAW_DATA_DIR, "stg_conference_history_raw.parquet"))
    con.execute("CREATE OR REPLACE TABLE stg_conference_history_raw AS SELECT * FROM df_conf_history")
    
    # Venues
    print("Fetching venues...")
    venues = make_request("/venues", {})
    venues_list = []
    for v in venues:
        loc = v.get('location') or {}
        venues_list.append({
            'venue_id': v.get('id'),
            'venue_name': v.get('name'),
            'latitude': loc.get('y'),
            'longitude': loc.get('x'),
            'altitude_ft': v.get('elevation'),
            'indoor_flag': v.get('dome'),
            'timezone': v.get('timezone')
        })
    df_venues = pd.DataFrame(venues_list)
    df_venues.to_parquet(os.path.join(RAW_DATA_DIR, "venue_dimension_shell.parquet"))
    con.execute("CREATE OR REPLACE TABLE venue_dimension_shell AS SELECT * FROM df_venues")
    
    # === Step 2: Games and final scores ===
    print("Fetching games...")
    games_list = []
    for yr in YEARS:
        for st in ['regular', 'postseason']:
            games = make_request("/games", {"year": yr, "seasonType": st})
            for g in games:
                games_list.append({
                    'game_id': g.get('id'),
                    'season': g.get('season'),
                    'week': g.get('week'),
                    'game_date': g.get('startDate'),
                    'home_team': g.get('homeTeam'),
                    'away_team': g.get('awayTeam'),
                    'home_team_id': g.get('homeId'),
                    'away_team_id': g.get('awayId'),
                    'home_score': g.get('homePoints'),
                    'away_score': g.get('awayPoints'),
                    'venue_id': g.get('venueId'),
                    'status': 'completed' if g.get('homePoints') is not None and g.get('awayPoints') is not None else 'scheduled',
                    'neutral_site': g.get('neutralSite', False)
                })
        time.sleep(0.1)
    
    df_games = pd.DataFrame(games_list)
    
    fbs_set = set(zip(df_fbs_teams['team_name'], df_fbs_teams['season']))
    df_games['home_is_fbs'] = df_games.apply(lambda x: (x['home_team'], x['season']) in fbs_set, axis=1)
    df_games['away_is_fbs'] = df_games.apply(lambda x: (x['away_team'], x['season']) in fbs_set, axis=1)
    df_games_fbs = df_games[(df_games['home_is_fbs']) | (df_games['away_is_fbs'])].copy()
    
    df_games_fbs.to_parquet(os.path.join(RAW_DATA_DIR, "stg_games_raw.parquet"))
    con.execute("CREATE OR REPLACE TABLE stg_games_raw AS SELECT * FROM df_games_fbs")
    
    # Margin Distribution
    print("Computing margin distribution...")
    df_fbs_vs_fbs = df_games_fbs[(df_games_fbs['home_is_fbs']) & (df_games_fbs['away_is_fbs']) & (df_games_fbs['home_score'].notnull()) & (df_games_fbs['away_score'].notnull())].copy()
    df_fbs_vs_fbs['margin'] = df_fbs_vs_fbs['home_score'] - df_fbs_vs_fbs['away_score']
    
    margin_counts = df_fbs_vs_fbs['margin'].value_counts().reset_index()
    margin_counts.columns = ['margin_bucket', 'historical_frequency']
    
    all_margins = pd.DataFrame({'margin_bucket': range(-60, 61)})
    margin_dist = all_margins.merge(margin_counts, on='margin_bucket', how='left').fillna(0)
    total_games = margin_dist['historical_frequency'].sum()
    margin_dist['relative_frequency'] = margin_dist['historical_frequency'] / total_games
    margin_dist['cumulative_frequency'] = margin_dist['relative_frequency'].cumsum()
    margin_dist['seasons_included'] = f"{min(YEARS)}-{max(YEARS)} (excl. 2020)"
    margin_dist['sample_size'] = total_games
    
    margin_dist.to_parquet(os.path.join(RAW_DATA_DIR, "margin_distribution_reference.parquet"))
    con.execute("CREATE OR REPLACE TABLE margin_distribution_reference AS SELECT * FROM margin_dist")
    
    # === Step 3: Betting Lines ===
    print("Fetching betting lines...")
    lines_list = []
    for yr in YEARS:
        for st in ['regular', 'postseason']:
            lines_data = make_request("/lines", {"year": yr, "seasonType": st})
            for g in lines_data:
                for line in g.get('lines', []):
                    lines_list.append({
                        'game_id': g.get('id'),
                        'season': g.get('season'),
                        'week': g.get('week'),
                        'source': line.get('provider'),
                        'spread': line.get('spread'),
                        'total': line.get('overUnder'),
                        'moneyline_home': line.get('homeMoneyline'),
                        'moneyline_away': line.get('awayMoneyline'),
                        'home_team': g.get('homeTeam'),
                        'away_team': g.get('awayTeam')
                    })
        time.sleep(0.1)
        
    df_lines = pd.DataFrame(lines_list)
    df_lines.to_parquet(os.path.join(RAW_DATA_DIR, "stg_lines_raw.parquet"))
    con.execute("CREATE OR REPLACE TABLE stg_lines_raw AS SELECT * FROM df_lines")
    
    # Normalized lines
    print("Normalizing betting lines...")
    df_norm = df_lines.copy()
    df_norm['market_spread_home'] = pd.to_numeric(df_norm['spread'], errors='coerce')
    df_norm['market_total'] = pd.to_numeric(df_norm['total'], errors='coerce')
    
    df_norm['implied_home_prob_raw'] = df_norm['moneyline_home'].apply(american_to_prob)
    df_norm['implied_away_prob_raw'] = df_norm['moneyline_away'].apply(american_to_prob)
    
    df_norm['market_overround'] = df_norm['implied_home_prob_raw'] + df_norm['implied_away_prob_raw'] - 1.0
    
    total_implied = df_norm['implied_home_prob_raw'] + df_norm['implied_away_prob_raw']
    df_norm['implied_home_prob'] = df_norm['implied_home_prob_raw'] / total_implied
    df_norm['implied_away_prob'] = df_norm['implied_away_prob_raw'] / total_implied
    
    df_norm.to_parquet(os.path.join(RAW_DATA_DIR, "stg_lines_normalized.parquet"))
    con.execute("CREATE OR REPLACE TABLE stg_lines_normalized AS SELECT * FROM df_norm")
    
    con.close()
    print("Done!")

if __name__ == "__main__":
    main()
