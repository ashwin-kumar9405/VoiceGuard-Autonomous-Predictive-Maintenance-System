import json
import time
import random
from pathlib import Path
import pandas as pd
import streamlit as st
from voiceguard.pipeline import build_pipeline
from voiceguard.model import load_model


st.set_page_config(page_title="VoiceGuard", layout="wide")
st.title("VoiceGuard: Autonomous Predictive Maintenance System")
col1, col2 = st.columns(2)
with col1:
    voice_text = st.text_area("Voice transcript", value="My car has been overheating and there is rattling vibration at low speeds. Oil light flickered. It's urgent.", height=150)
    cust_id = st.text_input("Customer ID", value="CUST-2002")
    lat = st.number_input("Customer latitude", value=12.9716, format="%.6f")
    lon = st.number_input("Customer longitude", value=77.5946, format="%.6f")
with col2:
    engine_temp_c = st.number_input("Engine temp (°C)", value=102.5)
    battery_voltage = st.number_input("Battery voltage (V)", value=11.9)
    oil_pressure_psi = st.number_input("Oil pressure (psi)", value=28.0)
    vibration_g = st.number_input("Vibration (g)", value=0.85)
    speed_kph = st.number_input("Speed (kph)", value=25.0)
    odometer_km = st.number_input("Odometer (km)", value=120000)
    error_codes = st.text_input("Error codes CSV", value="P0520,P0302")

telemetry = {
    "engine_temp_c": engine_temp_c,
    "battery_voltage": battery_voltage,
    "oil_pressure_psi": oil_pressure_psi,
    "vibration_g": vibration_g,
    "speed_kph": speed_kph,
    "odometer_km": odometer_km,
    "error_codes": [c.strip() for c in error_codes.split(",") if c.strip()],
    "location": [lat, lon],
}
customer = {"id": cust_id, "location": [lat, lon]}

model_path = "models/lg.pkl"
model_obj = None
if Path(model_path).exists():
    model_obj = load_model(model_path)

run_once = st.button("Run VoiceGuard")
live_mode = st.checkbox("Live mode (auto-update risk)")
start_live = st.button("Start live simulation")

if run_once:
    pipeline = build_pipeline(model_obj=model_obj)
    result = pipeline.run(voice_text, telemetry, customer)
    st.subheader("Diagnosis")
    d = result["diagnosis"]
    st.metric("Risk score", f'{d["risk_score"]:.3f}')
    st.write(f'Issue category: {d["issue_category"]}')
    st.subheader("Scheduling")
    st.json(result["schedule"])
    st.subheader("Voice Summary")
    st.json(result["voice_summary"])
    st.subheader("Signals")
    st.json(d["signals"])
    st.subheader("Security Alerts")
    st.json(result["security_alerts"])
    st.subheader("OEM Feedback")
    st.json(result["oem_feedback"])
    st.subheader("Analytics")
    st.json(result["analytics"])
    st.download_button("Download JSON", data=json.dumps(result, indent=2), file_name="voiceguard_result.json", mime="application/json")

if live_mode and start_live:
    pipeline = build_pipeline(model_obj=model_obj)
    chart = st.line_chart(pd.DataFrame({"risk": []}))
    out = st.empty()
    base = telemetry.copy()
    for i in range(20):
        jitter = lambda v, s: float(v + random.gauss(0, s))
        base["engine_temp_c"] = jitter(base["engine_temp_c"], 0.7)
        base["battery_voltage"] = jitter(base["battery_voltage"], 0.05)
        base["oil_pressure_psi"] = jitter(base["oil_pressure_psi"], 0.6)
        base["vibration_g"] = max(0.0, jitter(base["vibration_g"], 0.05))
        res = pipeline.run(voice_text, base, customer)
        risk = res["diagnosis"]["risk_score"]
        chart.add_rows(pd.DataFrame({"risk": [risk]}))
        out.json(res)
        time.sleep(0.5)
