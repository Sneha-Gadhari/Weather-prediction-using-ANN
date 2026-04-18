import requests
import pandas as pd

def fetch_pollution(start, end):
    url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude=19.076&longitude=72.877"
        f"&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,ozone,european_aqi"
        f"&start_date={start}&end_date={end}&timezone=Asia%2FKolkata"
    )
    r = requests.get(url, timeout=30).json()
    if 'hourly' not in r:
        print(f"Pollution error {start}-{end}:", r)
        return None
    h = r['hourly']
    return pd.DataFrame({
        'time':  h['time'],
        'PM2.5': h['pm2_5'],
        'PM10':  h['pm10'],
        'NO2':   h['nitrogen_dioxide'],
        'CO':    [v / 1000 if v is not None else None for v in h['carbon_monoxide']],
        'O3':    h['ozone'],
        'AQI':   [v * 2.0 if v is not None else None for v in h['european_aqi']],
    })

def fetch_weather(start, end):
    # open-meteo historical ERA5 endpoint
    url = (
        f"https://historical-forecast-api.open-meteo.com/v1/forecast"
        f"?latitude=19.076&longitude=72.877"
        f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,"
        f"shortwave_radiation,surface_pressure"
        f"&start_date={start}&end_date={end}&timezone=Asia%2FKolkata"
    )
    r = requests.get(url, timeout=30).json()
    if 'hourly' not in r:
        print(f"Weather error {start}-{end}:", r)
        return None
    h = r['hourly']
    return pd.DataFrame({
        'time':           h['time'],
        'Temperature':    h['temperature_2m'],
        'Humidity':       h['relative_humidity_2m'],
        'WindSpeed':      h['wind_speed_10m'],
        'SolarRadiation': h['shortwave_radiation'],
        'Pressure':       h['surface_pressure'],
    })

all_dfs = []

for year in ['2022', '2023']:
    print(f"\nFetching {year}...")
    start = f"{year}-01-01"
    end   = f"{year}-12-31"

    df_p = fetch_pollution(start, end)
    df_w = fetch_weather(start, end)

    if df_p is None or df_w is None:
        print(f"Skipping {year} due to error")
        continue

    df = pd.merge(df_p, df_w, on='time')
    df.dropna(inplace=True)
    print(f"{year}: {len(df)} rows")
    all_dfs.append(df)

if not all_dfs:
    print("No data fetched at all. Check your internet connection.")
else:
    final = pd.concat(all_dfs, ignore_index=True)
    final.drop(columns=['time'], inplace=True)
    final.to_csv('mumbai_aqi_dataset.csv', index=False)
    print(f"\nDone! Saved {len(final)} rows to mumbai_aqi_dataset.csv")
    print(final.head())