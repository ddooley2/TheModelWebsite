import duckdb
import pandas as pd
import requests
import time
from datetime import datetime

DB_PATH = "/Users/ddooley/Documents/SportsBetting/TheModel/cfb-model/db/warehouse.duckdb"

def compute_severity(row):
    if row['indoor_flag'] == True or pd.isna(row['temperature_f']):
        return 0.0
    temp_dev = abs(row['temperature_f'] - 72.0) / 10.0
    wind_dev = (row['wind_mph'] or 0.0) / 10.0
    precip_dev = (row['precip_mm'] or 0.0) * 5.0
    return temp_dev + wind_dev + precip_dev

def fix_missing_weather():
    con = duckdb.connect(DB_PATH)
    
    # 1. Fetch missing records
    query = """
    SELECT 
        w.game_id, 
        g.game_date, 
        v.latitude, 
        v.longitude
    FROM stg_weather_raw w
    JOIN stg_games_raw g ON w.game_id = g.game_id
    JOIN venue_dimension_shell v ON g.venue_id = v.venue_id
    WHERE w.temperature_f IS NULL 
      AND w.indoor_flag = False
      AND v.latitude IS NOT NULL
      AND v.longitude IS NOT NULL
    """
    df = con.execute(query).df()
    print(f"Total missing games to fix: {len(df)}")
    
    if len(df) == 0:
        con.close()
        return

    df['game_datetime'] = pd.to_datetime(df['game_date'], utc=True, errors='coerce')
    df = df.dropna(subset=['game_datetime'])
    df['utc_date'] = df['game_datetime'].dt.strftime('%Y-%m-%d')
    df['utc_hour'] = df['game_datetime'].dt.strftime('%Y-%m-%dT%H:00')

    url = 'https://archive-api.open-meteo.com/v1/archive'
    
    # Group by date to batch
    grouped = df.groupby('utc_date')
    api_calls = []
    
    for date_str, group in grouped:
        chunk_size = 50
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

    print(f"Total API calls: {len(api_calls)}")
    
    updates = []
    for i, (chunk, params) in enumerate(api_calls):
        retries = 5
        success = False
        for r in range(retries):
            try:
                resp = requests.get(url, params=params, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    success = True
                    break
                elif resp.status_code == 429:
                    print(f"Rate limited, sleeping 15s... (Call {i+1}/{len(api_calls)})")
                    time.sleep(15)
                else:
                    print(f"Error {resp.status_code}: {resp.text}")
                    time.sleep(5)
            except Exception as e:
                time.sleep(5)
                
        if not success:
            print(f"Failed to fetch chunk {i}")
            continue

        if isinstance(data, dict):
            data = [data]
            
        for idx, (_, row) in enumerate(chunk.iterrows()):
            if idx >= len(data):
                break
            loc_data = data[idx]
            hourly = loc_data.get('hourly', {})
            times = hourly.get('time', [])
            target_hour = row['utc_hour']
            
            if target_hour in times:
                t_idx = times.index(target_hour)
                try:
                    temp = hourly.get('temperature_2m', [])[t_idx]
                    wind = hourly.get('wind_speed_10m', [])[t_idx]
                    precip = hourly.get('precipitation', [])[t_idx]
                    rh = hourly.get('relative_humidity_2m', [])[t_idx]
                    pressure = hourly.get('surface_pressure', [])[t_idx]
                    
                    if temp is not None:
                        updates.append({
                            'game_id': row['game_id'],
                            'temperature_f': temp,
                            'wind_mph': wind,
                            'precip_mm': precip,
                            'relative_humidity_pct': rh,
                            'surface_pressure_hpa': pressure
                        })
                except IndexError:
                    pass
                    
        time.sleep(0.1) # Be gentle to the API
        if i % 10 == 0:
            print(f"Progress: {i}/{len(api_calls)} chunks")

    print(f"Writing {len(updates)} updates to duckdb...")
    df_updates = pd.DataFrame(updates)
    if len(df_updates) > 0:
        df_updates['indoor_flag'] = False
        df_updates['weather_severity_composite'] = df_updates.apply(compute_severity, axis=1)
        
        # Temp table for updates
        con.execute("CREATE OR REPLACE TEMP TABLE weather_updates AS SELECT * FROM df_updates")
        
        con.execute("""
            UPDATE stg_weather_raw 
            SET 
                temperature_f = u.temperature_f,
                wind_mph = u.wind_mph,
                precip_mm = u.precip_mm,
                relative_humidity_pct = u.relative_humidity_pct,
                surface_pressure_hpa = u.surface_pressure_hpa,
                weather_severity_composite = u.weather_severity_composite
            FROM weather_updates u
            WHERE stg_weather_raw.game_id = u.game_id
        """)
        print("Update complete.")
    con.close()

if __name__ == "__main__":
    fix_missing_weather()
