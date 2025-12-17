from typing import Optional, Tuple
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "engine_temp_c",
    "battery_voltage",
    "oil_pressure_psi",
    "vibration_g",
    "speed_kph",
    "odometer_km",
    "error_code_count",
]


def _prepare(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    X = df[FEATURES].values.astype(float)
    y = df["label"].values.astype(int)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, y, scaler


def train_model(input_csv: str, output_pkl: str) -> dict:
    df = pd.read_csv(input_csv)
    Xs, y, scaler = _prepare(df)
    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=500)
    clf.fit(X_train, y_train)
    acc = float(clf.score(X_test, y_test))
    with open(output_pkl, "wb") as f:
        pickle.dump({"model": clf, "scaler": scaler, "features": FEATURES}, f)
    return {"accuracy": acc, "features": FEATURES}


def load_model(path: str):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def predict_proba(model_obj, row: dict) -> float:
    scaler = model_obj["scaler"]
    clf = model_obj["model"]
    feats = np.array([[row[k] for k in FEATURES]], dtype=float)
    feats_s = scaler.transform(feats)
    p = float(clf.predict_proba(feats_s)[0, 1])
    return p

