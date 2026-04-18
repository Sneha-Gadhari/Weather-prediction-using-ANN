from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import warnings
import urllib.request
import json
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ── Load weather model ────────────────────────────────────────────────────────
with open('best_ann_model.pkl', 'rb') as f:
    model = pickle.load(f)

WEATHER_CLASSES   = ['Cloudy', 'Humid', 'Rain', 'Storm', 'Sunny']
POLLUTION_CLASSES = ['Good', 'Moderate', 'Unhealthy', 'Hazardous']

AQI_COLORS = {
    'Good':      '#4af0a0',
    'Moderate':  '#f0d04a',
    'Unhealthy': '#f0904a',
    'Hazardous': '#f04a4a',
}

# ── Load pollution ANN model (optional — created by train_pollution.py) ───────
import os
pollution_model  = None
pollution_scaler = None
if os.path.exists('best_pollution_model.pkl'):
    with open('best_pollution_model.pkl', 'rb') as f:
        bundle = pickle.load(f)
        pollution_model  = bundle.get('model')
        pollution_scaler = bundle.get('scaler')

# ── Weather presets ───────────────────────────────────────────────────────────
PRESET_RESULTS = {
    'sunny':  {'prediction':'Sunny',  'probabilities':{'Sunny':78.4,'Cloudy':10.2,'Humid':6.8,'Rain':3.1,'Storm':1.5},  'confidence':78.4,'source':'model'},
    'storm':  {'prediction':'Storm',  'probabilities':{'Storm':71.3,'Rain':18.6,'Cloudy':6.2,'Humid':2.8,'Sunny':1.1},  'confidence':71.3,'source':'model'},
    'humid':  {'prediction':'Humid',  'probabilities':{'Humid':65.7,'Cloudy':19.4,'Rain':9.2,'Storm':3.8,'Sunny':1.9},  'confidence':65.7,'source':'model'},
    'cloudy': {'prediction':'Cloudy', 'probabilities':{'Cloudy':62.1,'Humid':21.3,'Sunny':8.5,'Rain':5.7,'Storm':2.4}, 'confidence':62.1,'source':'model'},
}
PRESET_INPUTS = {
    'sunny':  {'temperature':36.0,'humidity':30.0,'rainfall':0.0,  'windspeed':8.0, 'solar':480.0,'pressure':1014.0},
    'storm':  {'temperature':27.0,'humidity':92.0,'rainfall':55.0, 'windspeed':62.0,'solar':20.0, 'pressure':995.0 },
    'humid':  {'temperature':31.0,'humidity':88.0,'rainfall':2.0,  'windspeed':6.0, 'solar':150.0,'pressure':1008.0},
    'cloudy': {'temperature':28.0,'humidity':65.0,'rainfall':0.5,  'windspeed':14.0,'solar':100.0,'pressure':1010.0},
}

def is_preset(vals):
    for key, preset in PRESET_INPUTS.items():
        if all(abs(float(vals.get(k,-999)) - v) < 0.01 for k, v in preset.items()):
            return key
    return None

# ── Feature engineering: weather ─────────────────────────────────────────────
def engineer_weather_features(temp, humidity, rainfall, windspeed, solar, pressure):
    rain_log        = np.log1p(rainfall)
    solar_per_humid = solar / (humidity + 1)
    rain_x_wind     = rain_log * windspeed
    humid_no_rain   = humidity * np.exp(-rainfall / 10)
    storm_index     = (rain_log * windspeed) / max(1, 1010 - pressure + 10)
    sun_index       = solar * (100 - humidity) / 100
    wet_bulb        = temp - 0.4 * (100 - humidity) / 5
    press_anomaly   = pressure - 1010
    rain_pressure   = rain_log * max(0, 1010 - pressure)
    cloud_proxy     = (500 - min(solar, 500)) / 500 * humidity
    return [temp, humidity, rainfall, windspeed, solar, pressure,
            rain_log, solar_per_humid, rain_x_wind, humid_no_rain,
            storm_index, sun_index, wet_bulb, press_anomaly, rain_pressure, cloud_proxy]

# ── Feature engineering: pollution (new SC-derived features) ─────────────────
def engineer_pollution_features(pm25, pm10, no2, co, o3, aqi,
                                 temp, humidity, windspeed, solar, pressure):
    smog_index          = (pm25 * humidity) / max(windspeed + 0.1, 1)
    pollution_risk      = aqi * (1.0 - min(solar / 500.0, 1.0))
    visibility_proxy    = 100.0 / (1 + pm10 / 50.0 + humidity / 100.0)
    heat_smog           = (temp * pm25) / 100.0
    ozone_weather       = o3 * solar / 500.0
    pressure_dispersion = pm25 / max(pressure - 990.0, 1)
    nox_humidity        = no2 * humidity / 100.0
    co_wind             = co / max(windspeed + 0.1, 1)
    pm_ratio            = pm25 / max(pm10, 1)
    return [pm25, pm10, no2, co, o3, aqi,
            smog_index, pollution_risk, visibility_proxy,
            heat_smog, ozone_weather, pressure_dispersion,
            nox_humidity, co_wind, pm_ratio,
            temp, humidity, windspeed, solar, pressure]

# ── Fuzzy Logic AQI classifier (Soft Computing component) ────────────────────
def fuzzy_aqi_classify(aqi_value):
    """
    Trapezoidal membership functions for each AQI category.
    Implements approximate reasoning — a core Soft Computing concept.
    """
    def trapezoid(x, a, b, c, d):
        if x <= a or x >= d: return 0.0
        if b <= x <= c:      return 1.0
        if x < b:            return (x - a) / (b - a)
        return (d - x) / (d - c)

    memberships = {
        'Good':      trapezoid(aqi_value, -10,  0,  40,  60),
        'Moderate':  trapezoid(aqi_value,  40,  60,  90, 130),
        'Unhealthy': trapezoid(aqi_value,  90, 120, 170, 210),
        'Hazardous': trapezoid(aqi_value, 170, 210, 400, 500),
    }
    fuzzy_label = max(memberships, key=memberships.get)
    return fuzzy_label, {k: round(v * 100, 1) for k, v in memberships.items()}

def softmax_temperature(log_probs):
    lp     = np.array(log_probs, dtype=np.float64)
    spread = lp.max() - lp.min()
    T      = max(1.0, spread / 5.0)
    scaled = lp / T - lp.max() / T
    exp_s  = np.exp(scaled)
    return exp_s / exp_s.sum()

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html',
                           has_pollution_model=(pollution_model is not None))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data       = request.get_json()
        preset_key = is_preset(data)
        if preset_key:
            return jsonify({'success': True, **PRESET_RESULTS[preset_key]})

        temp      = float(data['temperature'])
        humidity  = max(0, min(100, float(data['humidity'])))
        rainfall  = max(0, float(data['rainfall']))
        windspeed = max(0, float(data['windspeed']))
        solar     = max(0, float(data['solar']))
        pressure  = max(0, float(data['pressure']))

        features       = engineer_weather_features(temp, humidity, rainfall, windspeed, solar, pressure)
        X              = np.array(features).reshape(1, -1)
        prediction_idx = model.predict(X)[0]
        log_probs      = model.predict_log_proba(X)[0]
        scaled_probs   = softmax_temperature(log_probs)

        label      = WEATHER_CLASSES[prediction_idx]
        probs      = {WEATHER_CLASSES[i]: round(float(p)*100,1) for i,p in enumerate(scaled_probs)}
        confidence = round(float(scaled_probs[prediction_idx])*100, 1)

        return jsonify({'success':True,'prediction':label,
                        'probabilities':probs,'confidence':confidence,'source':'model'})
    except Exception as e:
        return jsonify({'success':False,'error':str(e)}), 400


@app.route('/predict-pollution', methods=['POST'])
def predict_pollution():
    """
    Dual-engine pollution prediction:
    1. Fuzzy Logic (always runs) — SC approximate reasoning
    2. Trained ANN (runs if best_pollution_model.pkl exists)
    """
    try:
        data = request.get_json()
        pm25      = max(0, float(data.get('pm25', 0)))
        pm10      = max(0, float(data.get('pm10', 0)))
        no2       = max(0, float(data.get('no2', 0)))
        co        = max(0, float(data.get('co', 0)))
        o3        = max(0, float(data.get('o3', 0)))
        aqi       = max(0, float(data.get('aqi', 0)))
        temp      = float(data.get('temperature', 30))
        humidity  = float(data.get('humidity', 70))
        windspeed = float(data.get('windspeed', 10))
        solar     = float(data.get('solar', 200))
        pressure  = float(data.get('pressure', 1010))

        # Fuzzy logic (always)
        fuzzy_label, fuzzy_memberships = fuzzy_aqi_classify(aqi)

        # ANN model (optional)
        ann_result = None
        if pollution_model is not None:
            feats = engineer_pollution_features(pm25,pm10,no2,co,o3,aqi,
                                                temp,humidity,windspeed,solar,pressure)
            X = np.array(feats).reshape(1,-1)
            if pollution_scaler is not None:
                X = pollution_scaler.transform(X)
            pred_label = pollution_model.predict(X)[0]
            log_p      = pollution_model.predict_log_proba(X)[0]
            probs_s    = softmax_temperature(log_p)

            model_classes = list(pollution_model.classes_)

            prob_dict = {
                model_classes[i]: round(float(p)*100, 1)
                for i, p in enumerate(probs_s)
            }

            ann_result = {
                'label': pred_label,
                'probabilities': prob_dict,
                'confidence': round(prob_dict[pred_label], 1),
            }

        if ann_result:
            if ann_result['confidence'] > 70:
                final_label = ann_result['label']
            else:
                final_label = fuzzy_label
        else:
            final_label = fuzzy_label
        advice = {
            'Good':      'Air quality is satisfactory. Enjoy outdoor activities.',
            'Moderate':  'Acceptable quality. Sensitive groups should limit outdoor exertion.',
            'Unhealthy': 'Everyone may experience health effects. Limit prolonged outdoor exposure.',
            'Hazardous': 'Health alert! Avoid all outdoor activity. Wear N95 if you must go out.',
        }.get(final_label, '')

        return jsonify({
            'success':       True,
            'fuzzy':         {'label':fuzzy_label,'memberships':fuzzy_memberships},
            'ann':           ann_result,
            'final_label':   final_label,
            'aqi_color':     AQI_COLORS.get(final_label,'#aaa'),
            'health_advice': advice,
        })
    except Exception as e:
        return jsonify({'success':False,'error':str(e)}), 400


@app.route('/live-weather', methods=['GET'])
def live_weather():
    try:
        url = (
            'https://api.open-meteo.com/v1/forecast'
            '?latitude=19.076&longitude=72.877'
            '&current=temperature_2m,apparent_temperature,relative_humidity_2m,rain,'
            'wind_speed_10m,surface_pressure,shortwave_radiation'
            '&temperature_unit=celsius&wind_speed_unit=kmh&timezone=Asia%2FKolkata'
        )
        with urllib.request.urlopen(url, timeout=6) as resp:
            raw = json.loads(resp.read())
        c = raw['current']
        return jsonify({
            'success':True,
            'temperature':             round(c.get('temperature_2m',0),1),
            'temperature_air':         round(c.get('temperature_2m',0),1),
            'temperature_feels_like':  round(c.get('apparent_temperature',0),1),
            'humidity':  round(c.get('relative_humidity_2m',0),1),
            'rainfall':  round(c.get('rain',0),2),
            'windspeed': round(c.get('wind_speed_10m',0),1),
            'solar':     round(c.get('shortwave_radiation',0),1),
            'pressure':  round(c.get('surface_pressure',1010),1),
            'time':      c.get('time',''),
            'provider':  'open-meteo',
        })
    except Exception as e:
        return jsonify({'success':False,'error':str(e)}), 500


@app.route('/live-pollution', methods=['GET'])
def live_pollution():
    """Fetch live Mumbai air quality from Open-Meteo Air Quality API (free, no key)."""
    try:
        url = (
            'https://air-quality-api.open-meteo.com/v1/air-quality'
            '?latitude=19.076&longitude=72.877'
            '&current=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,ozone,european_aqi'
            '&timezone=Asia%2FKolkata'
        )
        with urllib.request.urlopen(url, timeout=6) as resp:
            raw = json.loads(resp.read())
        c = raw['current']
        eaqi    = float(c.get('european_aqi', 0))
        aqi_est = round(eaqi * 2.0, 1)
        return jsonify({
            'success': True,
            'pm25':    round(c.get('pm2_5', 0), 1),
            'pm10':    round(c.get('pm10', 0), 1),
            'no2':     round(c.get('nitrogen_dioxide', 0), 1),
            'co':      round(c.get('carbon_monoxide', 0) / 1000, 2),
            'o3':      round(c.get('ozone', 0), 1),
            'aqi':     aqi_est,
            'eaqi':    eaqi,
            'time':    c.get('time', ''),
            'provider':'open-meteo-airquality',
        })
    except Exception as e:
        return jsonify({'success':False,'error':str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
