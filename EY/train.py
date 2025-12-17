import random
import csv
from pathlib import Path
from voiceguard.model import train_model


def generate_synthetic(path: str, n: int = 1200):
    rng = random.Random(42)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "engine_temp_c",
            "battery_voltage",
            "oil_pressure_psi",
            "vibration_g",
            "speed_kph",
            "odometer_km",
            "error_code_count",
            "label"
        ])
        for _ in range(n):
            engine = rng.gauss(92, 8)
            battery = rng.gauss(12.4, 0.5)
            oil = rng.gauss(35, 6)
            vib = max(0.0, rng.gauss(0.5, 0.25))
            speed = max(0.0, rng.gauss(45, 20))
            odo = max(0.0, rng.gauss(60000, 30000))
            codes = max(0, int(abs(rng.gauss(1.0, 1.0))))
            risk_score = (
                (engine - 90) * 0.03
                + (12.5 - battery) * 0.4
                + (40 - oil) * 0.02
                + vib * 0.6
                + (odo / 200000) * 0.3
                + (codes / 5.0) * 0.5
            )
            noise = rng.gauss(0, 0.2)
            label = 1 if (risk_score + noise) > 0.9 else 0
            w.writerow([round(engine, 2), round(battery, 2), round(oil, 2), round(vib, 3), round(speed, 1), int(odo), codes, label])


def main():
    data_path = "data/sim_telemetry.csv"
    model_path = "models/lg.pkl"
    Path("data").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    if not Path(data_path).exists():
        generate_synthetic(data_path, 2000)
    report = train_model(data_path, model_path)
    print("trained", report)


if __name__ == "__main__":
    main()

