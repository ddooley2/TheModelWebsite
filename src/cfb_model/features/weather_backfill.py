import os
import duckdb
import pandas as pd
import requests
import time
from datetime import datetime

# Adjust to absolute path based on workspace
DB_PATH = "/Users/ddooley/Documents/SportsBetting/TheModel/cfb-model/db/warehouse.duckdb"

def compute_severity(row):
    if row['indoor_flag'] == True or pd.isna(row['temperature_f']):
        return 0.0
    temp_dev = abs(row['temperature_f'] - 72.0) / 10.0
    wind_dev = (row['wind_mph'] or 0.0) / 10.0
    precip_dev = (row['precip_mm'] or 0.0) * 5.0
    return temp_dev + wind_dev + precip_dev

def fetch_weather():
    con = duckdb.connect(DB_PATH)
    
    # 1. Fetch games and venues
    query = """
    SELECT 
        g.game_id, 
        g.game_date, 
        v.indoor_flag, 
        v.latitude, 
        v.longitude
    FROM stg_games_raw g
    LEFT JOIN venue_dimension_shell v ON g.venue_id = v.venue_id
    WHERE g.season >= 2013 AND g.season <= 2025
    """
    df = con.execute(query).df()
    
    print(f"Total games to process: {len(df)}")
    
    results = []
    
    # Pre-process dates and split into outdoor vs indoor
    # Handle potentially missing or unparseable dates by coercing to NaT
    df['game_datetime'] = pd.to_datetime(df['game_date'], utc=True, errors='coerce')
    
    # Drop rows without a valid date
    df = df.dropna(subset=['game_datetime'])
    
    df['utc_date'] = df['game_datetime'].dt.strftime('%Y-%m-%d')
    df['utc_hour'] = df['game_datetime'].dt.strftime('%Y-%m-%dT%H:00')
    
    # Assign indoor/missing weather
    # Note: outdoor requires valid lat/long AND indoor_flag == False/None
    df['indoor_flag'] = df['indoor_flag'].fillna(False)
    
    indoor_or_missing = df[df['indoor_flag'] | df['latitude'].isna() | df['longitude'].isna()]
    for _, row in indoor_or_missing.iterrows():
        results.append({
            'game_id': row['game_id'],
            'temperature_f': 72.0,
            'wind_mph': 0.0,
            'precip_mm': 0.0,
            'relative_humidity_pct': 50.0,
            'surface_pressure_hpa': 1013.0,
            'indoor_flag': True
        })
        
    outdoor = df[~df['indoor_flag'] & df['latitude'].notna() & df['longitude'].notna()]
    print(f"Outdoor games with valid lat/long: {len(outdoor)}")
    
    # Group outdoor games by date to batch API requests
    grouped = outdoor.groupby('utc_date')
    
    import concurrent.futures
    import json
    
    url = 'https://archive-api.open-meteo.com/v1/archive'
    
    # Prepare all API calls
    api_calls = []
    
    count = 0
    for date_str, group in grouped:
        chunk_size = 30
        for i in range(0, len(group), chunk_size):
            chunk = group.iloc[i:i+chunk_size]
            lats = chunk['latitude'].astype(str).tolist()
            lons = chunk['longitude'].astype(str).tolist()
            
            params = {
                'latitude': ','.join(lats),
                'longitude': ','.join(lons),
                'start_date': date_str,
                'end_date': date_str,
                'hourly': 'temperature_2m,wind_speed_10m,precipitation,relative_humidity_2m,surface_pressure',
                'temperature_unit': 'fahrenheit',
                'wind_speed_unit': 'mph',
                'precipitation_unit': 'mm',
                'timezone': 'GMT'
            }
            api_calls.append((chunk, params))

    print(f"Total API calls to make: {len(api_calls)}")

    def fetch_chunk(item):
        chunk, params = item
        retries = 3
        for r in range(retries):
            try:
                resp = requests.get(url, params=params, timeout=15)
                if resp.status_code == 200:
                    return chunk, resp.json()
                elif resp.status_code == 429:
                    time.sleep(5)
                elif resp.status_code == 400:
                    print(f"Bad Request: {resp.text}")
                    return chunk, None
                else:
                    time.sleep(2)
            except Exception as e:
                time.sleep(2)
        return chunk, None

    processed_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for chunk, data in executor.map(fetch_chunk, api_calls):
            if data is None:
                for _, row in chunk.iterrows():
                    results.append({
                        'game_id': row['game_id'],
                        'temperature_f': None,
                        'wind_mph': None,
                        'precip_mm': None,
                        'relative_humidity_pct': None,
                        'surface_pressure_hpa': None,
                        'indoor_flag': False
                    })
            else:
                if isinstance(data, dict):
                    data = [data]
                for idx, (_, row) in enumerate(chunk.iterrows()):
                    if idx >= len(data):
                        break
                    loc_data = data[idx]
                    hourly = loc_data.get('hourly', {})
                    times = hourly.get('time', [])
                    target_hour = row['utc_hour']
                    
                    temp, wind, precip, rh, pressure = None, None, None, None, None
                    if target_hour in times:
                        t_idx = times.index(target_hour)
                        try:
                            temp = hourly.get('temperature_2m', [])[t_idx]
                            wind = hourly.get('wind_speed_10m', [])[t_idx]
                            precip = hourly.get('precipitation', [])[t_idx]
                            rh = hourly.get('relative_humidity_2m', [])[t_idx]
                            pressure = hourly.get('surface_pressure', [])[t_idx]
                        except IndexError:
                            pass
                    
                    results.append({
                        'game_id': row['game_id'],
                        'temperature_f': temp,
                        'wind_mph': wind,
                        'precip_mm': precip,
                        'relative_humidity_pct': rh,
                        'surface_pressure_hpa': pressure,
                        'indoor_flag': False
                    })
            
            processed_count += len(chunk)
            if processed_count % 1000 < 30:
                print(f"Processed {processed_count}/{len(outdoor)} outdoor games")

    df_results = pd.DataFrame(results)
    df_results['weather_severity_composite'] = df_results.apply(compute_severity, axis=1)
    
    print("Writing stg_weather_raw to duckdb...")
    con.execute("CREATE OR REPLACE TABLE stg_weather_raw AS SELECT * FROM df_results")
    
    # Verification
    res = con.execute("SELECT COUNT(*) FROM stg_weather_raw").fetchone()[0]
    print(f"Total rows in stg_weather_raw: {res}")
    con.close()
    print("Done!")

if __name__ == "__main__":
    fetch_weather()
