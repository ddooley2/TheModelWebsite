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
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=10)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                time.sleep(2)
            else:
                print(f"Error {r.status_code} for {url} {params}")
                return []
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            time.sleep(2)
    return []

def extract_nested_stats(row, prefix, source_dict):
    if not isinstance(source_dict, dict):
        return
    for k, v in source_dict.items():
        if not isinstance(v, dict):
            row[f"{prefix}_{k}"] = v
            
def extract_pbp_stats(row, prefix, source_dict):
    if not isinstance(source_dict, dict):
        return
    for k, v in source_dict.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                row[f"{prefix}_{k}_{sub_k}"] = sub_v

def main():
    con = duckdb.connect(DB_PATH)
    
    # --- Step 4: Game Advanced Stats & PBP ---
    print("Fetching advanced stats...")
    stats_list = []
    pbp_list = []
    
    for yr in YEARS:
        print(f"Fetching advanced stats for {yr}...")
        for st in ['regular', 'postseason']:
            data = make_request("/stats/game/advanced", {"year": yr, "seasonType": st})
            for d in data:
                base = {
                    'game_id': d.get('gameId'),
                    'season': d.get('season'),
                    'week': d.get('week'),
                    'team': d.get('team'),
                    'opponent': d.get('opponent')
                }
                
                # stats
                stat_row = base.copy()
                extract_nested_stats(stat_row, "offense", d.get('offense'))
                extract_nested_stats(stat_row, "defense", d.get('defense'))
                stats_list.append(stat_row)
                
                # pbp
                pbp_row = base.copy()
                extract_pbp_stats(pbp_row, "offense", d.get('offense'))
                extract_pbp_stats(pbp_row, "defense", d.get('defense'))
                pbp_list.append(pbp_row)
        time.sleep(0.1)
        
    df_stats = pd.DataFrame(stats_list)
    df_pbp = pd.DataFrame(pbp_list)
    
    df_stats.to_parquet(os.path.join(RAW_DATA_DIR, "stg_team_game_stats_raw.parquet"))
    con.execute("CREATE OR REPLACE TABLE stg_team_game_stats_raw AS SELECT * FROM df_stats")
    
    df_pbp.to_parquet(os.path.join(RAW_DATA_DIR, "stg_pbp_team_game_raw.parquet"))
    con.execute("CREATE OR REPLACE TABLE stg_pbp_team_game_raw AS SELECT * FROM df_pbp")
    
    # --- Step 5: Roster Priors and Weather ---
    print("Fetching roster priors...")
    returning_list = []
    talent_list = []
    recruiting_list = []
    
    for yr in YEARS:
        print(f"Fetching roster priors for {yr}...")
        returning = make_request("/player/returning", {"year": yr})
        for r in returning:
            returning_list.append(r)
        
        talent = make_request("/talent", {"year": yr})
        for t in talent:
            talent_list.append(t)
            
        recruiting = make_request("/recruiting/teams", {"year": yr})
        for rec in recruiting:
            recruiting_list.append(rec)
            
        time.sleep(0.1)
        
    df_returning = pd.DataFrame(returning_list)
    df_talent = pd.DataFrame(talent_list)
    df_recruiting = pd.DataFrame(recruiting_list)
    
    if not df_talent.empty:
        df_talent.rename(columns={'year': 'season', 'talent': 'team_talent'}, inplace=True)
    if not df_recruiting.empty:
        df_recruiting.rename(columns={'year': 'season', 'points': 'recruiting_points', 'rank': 'recruiting_rank'}, inplace=True)
    
    if not df_returning.empty:
        df_priors = df_returning.copy()
    else:
        df_priors = pd.DataFrame(columns=['team', 'season'])
    
    if not df_talent.empty:
        df_priors = pd.merge(df_priors, df_talent[['team', 'season', 'team_talent']], on=['team', 'season'], how='outer')
    if not df_recruiting.empty:
        df_priors = pd.merge(df_priors, df_recruiting[['team', 'season', 'recruiting_points', 'recruiting_rank']], on=['team', 'season'], how='outer')
        
    df_priors.to_parquet(os.path.join(RAW_DATA_DIR, "stg_roster_priors_raw.parquet"))
    con.execute("CREATE OR REPLACE TABLE stg_roster_priors_raw AS SELECT * FROM df_priors")
    
    print("Creating empty weather table due to CFBD authorization tier...")
    df_weather = pd.DataFrame(columns=['game_id', 'timestamp', 'weather_variables', 'source'])
    df_weather.to_parquet(os.path.join(RAW_DATA_DIR, "stg_weather_raw.parquet"))
    con.execute("CREATE OR REPLACE TABLE stg_weather_raw AS SELECT * FROM df_weather")
    
    print("Step 4 and 5 done. Tables created in DuckDB.")
    con.close()

if __name__ == "__main__":
    main()
