# VoiceGuard: Autonomous Predictive Maintenance System (EY Techathon 6.0)

## Quick Start
- Prerequisites: Python 3.10+ (tested on 3.13)
- Install dependencies:
  - `python -m pip install streamlit scikit-learn pandas numpy`
- Train the explainable ML model:
  - `python train.py`
  - Saves `models/lg.pkl` and prints test accuracy
- Run the Streamlit demo:
  - `streamlit run app.py --server.address 127.0.0.1 --server.port 8501`
  - Open `http://127.0.0.1:8501`
- Optional HTTP JSON API:
  - `python web/server.py` then POST to `http://127.0.0.1:8000/api/predict`
- CLI simulation:
  - `python simulate.py` (prints a full JSON result)

## What You Will Demo
- Enter a customer voice transcript and telemetry values
- Click “Run VoiceGuard” to see:
  - Risk score (probability), issue category, nearest center, slot time, ETA, priority
  - Security alerts and OEM feedback flags
  - Feature signals used by the diagnosis for transparency
- Real‑time mode:
  - Enable “Live mode (auto‑update risk)” and click “Start live simulation”
  - Watch a live risk chart and continuously updating JSON output

## Architecture Overview
- Frontend: Streamlit single‑page app (`app.py`)
- Agents (`voiceguard/agents.py`):
  - Conversational Agent (`VoiceCustomerAgent`) parses transcript for symptoms and intent
  - Monitoring Agent (`TelemetryAgent`) normalizes IoT telemetry into bounded signals
  - Decision Agent (`DiagnosisAgent`) computes risk (Logistic Regression or rule fallback) and issue category
  - Scheduling Agent selects nearest center, slot, urgency
  - UEBA Agent flags intent–risk anomalies
  - Feedback Agent emits `oem_quality_flag` and `recommended_action`
- Orchestration: `VoiceGuardPipeline` (`voiceguard/pipeline.py`) coordinates agents end‑to‑end
- ML: `LogisticRegression` (`voiceguard/model.py`) trained on synthetic telemetry (`train.py`)
- Data: JSON/CSV inputs and outputs in `data/` and `models/`

## Data & Model
- Features used:
  - `engine_temp_c`, `battery_voltage`, `oil_pressure_psi`, `vibration_g`, `speed_kph`, `odometer_km`, `error_code_count`
- Model:
  - `LogisticRegression` with `StandardScaler`, trained via `train.py`
  - Saves artifact to `models/lg.pkl`, loaded automatically by `app.py`
- Explainability:
  - Streamlit displays normalized signals; coefficients are interpretable and monotonic trends are intuitive

## API (Optional)
- Endpoint: `POST /api/predict` (when `web/server.py` is running)
- Request:
```
{
  "voice_text": "My car overheats and rattles at low speed",
  "telemetry": {
    "engine_temp_c": 102.5,
    "battery_voltage": 11.9,
    "oil_pressure_psi": 28.0,
    "vibration_g": 0.85,
    "speed_kph": 25.0,
    "odometer_km": 120000,
    "error_codes": ["P0520", "P0302"],
    "location": [12.9716, 77.5946]
  },
  "customer": { "id": "CUST-2002", "location": [12.9716, 77.5946] }
}
```
- Response (excerpt):
```
{
  "diagnosis": {
    "risk_score": 0.94,
    "issue_category": "Cooling/Overheat",
    "signals": { "engine_temp_norm": 0.65, "battery_drop_norm": 0.2, ... }
  },
  "schedule": { "center_id": "BLR-01", "slot": "2025-12-17T14:03:11Z", "priority": "urgent" },
  "security_alerts": [],
  "oem_feedback": { "oem_quality_flag": "investigate", "recommended_action": "pre-stock parts" }
}
```

## Project Structure
- `app.py` — Streamlit UI (demo and live mode)
- `simulate.py` — CLI demo printing JSON
- `train.py` — synthetic dataset generator + model training
- `voiceguard/` — core package
  - `agents.py` — all agents (conversational, monitoring, decision, scheduling, UEBA, feedback)
  - `pipeline.py` — orchestrator
  - `model.py` — Logistic Regression training/loading/prediction
- `web/` — optional HTTP server + minimal UI
  - `server.py`, `index.html`
- `data/` — sample telemetry and generated CSV
- `models/` — trained model artifacts

## Troubleshooting
- Streamlit connection:
  - Use explicit bind: `--server.address 127.0.0.1 --server.port 8501`
  - If the port is busy, change `--server.port`
  - Ensure firewall allows local loopback
- Missing libraries:
  - Re‑run `python -m pip install streamlit scikit-learn pandas numpy`
- Model not found:
  - Run `python train.py` to create `models/lg.pkl`

## Alignment With EY Round‑2 Requirements
- Working prototype: Streamlit app + agents + ML + optional API
- Agentic behavior: autonomous monitoring, decisioning, scheduling, feedback, UEBA
- Predictive logic: Logistic Regression probability + transparent signals
- Smart scheduling: nearest center, urgent vs normal, slot and ETA
- Conversational interface: text transcript drives severity and category
- Scalability: stateless agents; cloud‑ready APIs and model hosting

