"""
train_pollution.py
==================
Trains an ANN (sklearn MLPClassifier) to predict AQI category from
air pollution + weather features.

HOW TO RUN:
    python train_pollution.py

WHAT IT NEEDS:
    mumbai_aqi_dataset.csv   (see DATASET section below)

WHAT IT PRODUCES:
    best_pollution_model.pkl   (loaded automatically by app.py)

DATASET
-------
Your CSV must have these columns (column names are case-insensitive):
    PM2.5, PM10, NO2, CO, O3, AQI, Temperature, Humidity,
    WindSpeed, SolarRadiation, Pressure

You can download free Mumbai AQI data from:
  - OpenAQ   : https://openaq.org/#/countries/IN (filter city=Mumbai)
  - CPCB     : https://cpcb.nic.in (Central Pollution Control Board)
  - Kaggle   : search "Mumbai air quality dataset"

SOFT COMPUTING CONCEPTS USED
------------------------------
1. ANN (Artificial Neural Network) — MLPClassifier, same family as main weather model
2. Feature Engineering — SmogIndex, PollutionRisk, VisibilityProxy etc.
3. Genetic Algorithm integration — uses GA-selected features for best subset
4. Fuzzy Logic — fuzzy_aqi_classify() in app.py handles soft boundaries
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ── 1. Load dataset ───────────────────────────────────────────────────────────
print("Loading dataset...")
try:
    df = pd.read_csv('mumbai_aqi_dataset.csv')
except FileNotFoundError:
    print("\n[ERROR] mumbai_aqi_dataset.csv not found.")
    print("  Please download Mumbai AQI data and save as mumbai_aqi_dataset.csv")
    print("  See the DATASET section at the top of this file for sources.")
    exit(1)

# Normalise column names
df.columns = [c.strip().lower().replace('.','').replace(' ','_') for c in df.columns]
print(f"Loaded {len(df)} rows. Columns: {list(df.columns)}")

# ── 2. Column mapping (handles common naming variations) ─────────────────────
COL_MAP = {
    'pm25':         ['pm25','pm2_5','pm2.5','pm 2.5'],
    'pm10':         ['pm10','pm 10'],
    'no2':          ['no2','nitrogen_dioxide'],
    'co':           ['co','carbon_monoxide'],
    'o3':           ['o3','ozone'],
    'aqi':          ['aqi','air_quality_index','aqindex'],
    'temperature':  ['temperature','temp'],
    'humidity':     ['humidity','relative_humidity'],
    'windspeed':    ['windspeed','wind_speed','wind'],
    'solar':        ['solar','solarradiation','shortwave_radiation'],
    'pressure':     ['pressure','atmospheric_pressure'],
}

def find_col(df, candidates):
    for c in candidates:
        if c in df.columns: return c
    return None

col = {k: find_col(df, v) for k, v in COL_MAP.items()}
missing = [k for k, v in col.items() if v is None]
if missing:
    print(f"[ERROR] Missing columns: {missing}")
    print(f"  Available columns: {list(df.columns)}")
    exit(1)

# ── 3. AQI category labels ────────────────────────────────────────────────────
def aqi_to_category(aqi):
    if aqi <= 50:   return 'Good'
    if aqi <= 100:  return 'Moderate'
    if aqi <= 200:  return 'Unhealthy'
    return 'Hazardous'

df['aqi_category'] = df[col['aqi']].apply(aqi_to_category)
print("Class distribution:")
print(df['aqi_category'].value_counts())

# ── 4. Feature engineering ────────────────────────────────────────────────────
def engineer_features(row):
    pm25      = row[col['pm25']]
    pm10      = row[col['pm10']]
    no2       = row[col['no2']]
    co        = row[col['co']]
    o3        = row[col['o3']]
    aqi       = row[col['aqi']]
    temp      = row[col['temperature']]
    humidity  = row[col['humidity']]
    windspeed = row[col['windspeed']]
    solar     = row[col['solar']]
    pressure  = row[col['pressure']]

    # Raw features
    feats = [pm25, pm10, no2, co, o3, aqi]

    # SC-engineered features (same logic as app.py)
    feats += [
        (pm25 * humidity) / max(windspeed + 0.1, 1),                     # SmogIndex
        aqi * (1.0 - min(solar / 500.0, 1.0)),                           # PollutionRisk
        100.0 / (1 + pm10/50.0 + humidity/100.0),                        # VisibilityProxy
        (temp * pm25) / 100.0,                                            # HeatSmog
        o3 * solar / 500.0,                                               # OzoneWeatherIndex
        pm25 / max(pressure - 990.0, 1),                                  # PressureDispersion
        no2 * humidity / 100.0,                                           # NOx_Humidity
        co / max(windspeed + 0.1, 1),                                     # CO_Wind
        pm25 / max(pm10, 1),                                              # PM_Ratio
        temp, humidity, windspeed, solar, pressure,                       # weather context
    ]
    return feats

FEATURE_NAMES = [
    'PM2.5','PM10','NO2','CO','O3','AQI',
    'SmogIndex','PollutionRisk','VisibilityProxy',
    'HeatSmog','OzoneWeather','PressureDispersion',
    'NOx_Humidity','CO_Wind','PM_Ratio',
    'Temperature','Humidity','WindSpeed','Solar','Pressure'
]

print("\nEngineering features...")
df_clean = df.dropna(subset=list(col.values()))

# Force all feature columns to numeric, drop any remaining bad rows
for k, v in col.items():
    df_clean[v] = pd.to_numeric(df_clean[v], errors='coerce')
df_clean.dropna(subset=list(col.values()), inplace=True)
df_clean.reset_index(drop=True, inplace=True)
print(f"After cleaning: {len(df_clean)} rows")

X = np.array([engineer_features(row) for _, row in df_clean.iterrows()], dtype=np.float64)
y = df_clean['aqi_category'].values

print(f"Feature matrix: {X.shape}")

# ── 5. Train / test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── 6. ANN training ───────────────────────────────────────────────────────────
print("\nTraining ANN (MLPClassifier)...")
ann = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=300,
    early_stopping=False,
    batch_size=64,
    random_state=42,
    verbose=True
)
ann.fit(X_train, y_train)

y_pred = ann.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
print(f"\nPollution ANN Test Accuracy: {acc*100:.2f}%")
print(classification_report(y_test, y_pred))

# ── 7. Save model ─────────────────────────────────────────────────────────────
bundle = {'model': ann, 'scaler': scaler, 'feature_names': FEATURE_NAMES}
with open('best_pollution_model.pkl', 'wb') as f:
    pickle.dump(bundle, f)

print("\nSaved: best_pollution_model.pkl")
print("Now run: python app.py — the pollution model will load automatically.")
