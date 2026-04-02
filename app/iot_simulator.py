# ============================================================
# ROADSENSE INDIA — IoT Highway Traffic Simulator
# File: app/iot_simulator.py
# ============================================================

import random
import pandas as pd
from datetime import datetime

# ── Constants ────────────────────────────────────────────────
HIGHWAYS = ['NH-44', 'NH-48', 'NH-19', 'NH-27', 'NH-16', 'NH-52', 'NH-66', 'NH-58']

WEATHER_CONDITIONS  = ['Clear', 'Foggy', 'Rainy', 'Hazy', 'Drizzle', 'Stormy']
VEHICLE_TYPES       = ['Car', 'Truck', 'Bus', 'Two-Wheeler', 'Auto-Rickshaw', 'Tractor']
INCIDENT_TYPES      = ['No Incident', 'Minor Collision', 'Vehicle Breakdown',
                       'Severe Accident', 'Road Obstruction']

# Speed range (km/h) per weather condition
SPEED_RANGE = {
    'Clear'   : (60, 110),
    'Foggy'   : (20, 50),
    'Rainy'   : (30, 65),
    'Hazy'    : (35, 70),
    'Drizzle' : (40, 75),
    'Stormy'  : (15, 45),
}

# Incident probability per weather condition
INCIDENT_PROB = {
    'Clear'   : 0.04,
    'Foggy'   : 0.22,
    'Rainy'   : 0.18,
    'Hazy'    : 0.12,
    'Drizzle' : 0.10,
    'Stormy'  : 0.30,
}

# Risk score weights
RISK_WEIGHTS = {
    'No Incident'       : 0,
    'Minor Collision'   : 25,
    'Vehicle Breakdown' : 15,
    'Severe Accident'   : 75,
    'Road Obstruction'  : 30,
}


def generate_reading() -> dict:
    """Generate a single simulated IoT sensor reading."""
    highway = random.choice(HIGHWAYS)
    weather = random.choice(WEATHER_CONDITIONS)
    vehicle = random.choice(VEHICLE_TYPES)

    speed_lo, speed_hi = SPEED_RANGE[weather]
    speed = round(random.uniform(speed_lo, speed_hi), 1)

    # Incident determined by weather probability
    if random.random() < INCIDENT_PROB[weather]:
        incident = random.choice(INCIDENT_TYPES[1:])   # skip 'No Incident'
    else:
        incident = 'No Incident'

    # Risk score: base from incident + speed penalty + weather penalty
    base_risk  = RISK_WEIGHTS[incident]
    speed_risk = max(0, (speed - 80) * 0.5)    # penalty above 80 km/h
    weather_penalty = {'Clear': 0, 'Drizzle': 5, 'Hazy': 8,
                       'Rainy': 12, 'Foggy': 18, 'Stormy': 25}[weather]
    risk_score = min(100, round(base_risk + speed_risk + weather_penalty, 1))

    # Risk level label
    if risk_score < 20:
        risk_level = '🟢 Low'
    elif risk_score < 50:
        risk_level = '🟡 Medium'
    elif risk_score < 75:
        risk_level = '🟠 High'
    else:
        risk_level = '🔴 Critical'

    # Simulated GPS coordinates (rough bounding box of India)
    lat = round(random.uniform(8.5, 35.5), 4)
    lon = round(random.uniform(68.0, 97.5), 4)

    return {
        'timestamp'  : datetime.now().strftime('%H:%M:%S'),
        'highway'    : highway,
        'vehicle'    : vehicle,
        'weather'    : weather,
        'speed_kmh'  : speed,
        'incident'   : incident,
        'risk_score' : risk_score,
        'risk_level' : risk_level,
        'latitude'   : lat,
        'longitude'  : lon,
    }


def generate_batch(n: int = 10) -> pd.DataFrame:
    """Generate n readings as a DataFrame (used by dashboard)."""
    return pd.DataFrame([generate_reading() for _ in range(n)])


def get_alert_color(risk_level: str) -> str:
    """Return a hex colour for a given risk level label."""
    return {
        '🟢 Low'     : '#27AE60',
        '🟡 Medium'  : '#F39C12',
        '🟠 High'    : '#E67E22',
        '🔴 Critical': '#C0392B',
    }.get(risk_level, '#888888')


# ── Stand-alone demo ─────────────────────────────────────────
if __name__ == '__main__':
    print("Generating 5 sample IoT readings...\n")
    batch = generate_batch(5)
    print(batch.to_string(index=False))