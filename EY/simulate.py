import json
from pathlib import Path
from voiceguard.pipeline import build_pipeline
from voiceguard.model import load_model


def main():
    try:
        with open("data/telemetry_sample.json", "r", encoding="utf-8") as f:
            telemetry = json.load(f)
    except FileNotFoundError:
        telemetry = {
            "engine_temp_c": 98,
            "battery_voltage": 12.1,
            "oil_pressure_psi": 32,
            "vibration_g": 0.7,
            "speed_kph": 30,
            "odometer_km": 80000,
            "error_codes": ["P0301", "P0420"],
            "location": [12.985, 77.605],
        }
    voice_text = (
        "My car has been overheating and there's a rattling vibration at low speeds. "
        "I think the oil pressure light came on once. It's urgent."
    )
    customer = {"id": "CUST-1001", "location": (12.99, 77.59)}
    model_obj = load_model("models/lg.pkl") if Path("models/lg.pkl").exists() else None
    pipeline = build_pipeline(model_obj=model_obj)
    result = pipeline.run(voice_text, telemetry, customer)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
