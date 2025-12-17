from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import time


@dataclass
class Telemetry:
    engine_temp_c: float
    battery_voltage: float
    oil_pressure_psi: float
    vibration_g: float
    speed_kph: float
    odometer_km: float
    error_codes: List[str]
    location: Tuple[float, float]  # lat, lon


@dataclass
class VoiceCall:
    customer_id: str
    text: str
    timestamp: float
    location: Tuple[float, float]


@dataclass
class VoiceSummary:
    customer_id: str
    symptoms: List[str]
    severity: float  # 0..1
    intent: str  # e.g., "service_request"


@dataclass
class DiagnosisResult:
    risk_score: float  # 0..1
    issue_category: str
    contributing_signals: Dict[str, float]


@dataclass
class ScheduleResult:
    center_name: str
    center_id: str
    eta_minutes: int
    slot_iso: str
    priority: str  # "normal" | "urgent"


class VoiceCustomerAgent:
    KEYWORDS = {
        "overheat": ["overheat", "hot", "temperature", "smell burning"],
        "battery": ["battery", "won't start", "no power", "low voltage"],
        "vibration": ["vibration", "shaking", "rattle", "noise"],
        "oil": ["oil", "leak", "pressure"],
        "stall": ["stall", "engine stopped", "cut off"],
        "brake": ["brake", "squeak", "soft pedal"],
    }

    def process(self, call: VoiceCall) -> VoiceSummary:
        text = call.text.lower()
        symptoms: List[str] = []
        for cat, kws in self.KEYWORDS.items():
            if any(kw in text for kw in kws):
                symptoms.append(cat)
        # naive severity estimate: number of symptoms + presence of "urgent"
        base = min(len(symptoms) / 4.0, 1.0)
        if "urgent" in text or "immediately" in text or "breakdown" in text or "won't start" in text:
            base = max(base, 0.8)
        severity = round(base, 2)
        intent = "service_request" if ("service" in text or "appointment" in text or symptoms) else "general_inquiry"
        return VoiceSummary(customer_id=call.customer_id, symptoms=symptoms, severity=severity, intent=intent)


class TelemetryAgent:
    def process(self, telemetry: Telemetry) -> Dict[str, float]:
        # Normalize features roughly to 0..1 ranges
        features = {
            "engine_temp_norm": max(0.0, min(1.0, (telemetry.engine_temp_c - 70) / 50)),
            "battery_drop_norm": max(0.0, min(1.0, (12.5 - telemetry.battery_voltage) / 3)),
            "oil_pressure_low_norm": max(0.0, min(1.0, (40 - telemetry.oil_pressure_psi) / 40)),
            "vibration_norm": max(0.0, min(1.0, telemetry.vibration_g / 2)),
            "speed_norm": max(0.0, min(1.0, telemetry.speed_kph / 180)),
            "odometer_norm": max(0.0, min(1.0, telemetry.odometer_km / 200000)),
            "error_code_count": float(len(telemetry.error_codes)),
        }
        return features


class DiagnosisAgent:
    def __init__(self, model_obj=None):
        self.model_obj = model_obj

    def process(self, voice: VoiceSummary, telem_features: Dict[str, float]) -> DiagnosisResult:
        # Simple logistic-style risk model combining telemetry + voice severity
        if self.model_obj is not None:
            raw = {
                "engine_temp_c": telem_features["engine_temp_norm"] * 50 + 70,
                "battery_voltage": 12.5 - telem_features["battery_drop_norm"] * 3,
                "oil_pressure_psi": 40 - telem_features["oil_pressure_low_norm"] * 40,
                "vibration_g": telem_features["vibration_norm"] * 2,
                "speed_kph": telem_features["speed_norm"] * 180,
                "odometer_km": telem_features["odometer_norm"] * 200000,
                "error_code_count": telem_features["error_code_count"],
            }
            from .model import predict_proba
            risk = predict_proba(self.model_obj, raw)
        else:
            w = {
                "engine_temp_norm": 0.6,
                "battery_drop_norm": 0.5,
                "oil_pressure_low_norm": 0.4,
                "vibration_norm": 0.3,
                "error_code_count": 0.25,
                "odometer_norm": 0.2,
            }
            linear = (
                w["engine_temp_norm"] * telem_features["engine_temp_norm"]
                + w["battery_drop_norm"] * telem_features["battery_drop_norm"]
                + w["oil_pressure_low_norm"] * telem_features["oil_pressure_low_norm"]
                + w["vibration_norm"] * telem_features["vibration_norm"]
                + w["error_code_count"] * (telem_features["error_code_count"] / 5.0)
                + w["odometer_norm"] * telem_features["odometer_norm"]
                + 0.5 * voice.severity
            )
            risk = 1 / (1 + math.exp(-3 * (linear - 0.6)))
            risk = max(0.0, min(1.0, risk))
        # Map symptoms + telemetry to issue category
        if "battery" in voice.symptoms or telem_features["battery_drop_norm"] > 0.6:
            category = "Electrical/Battery"
        elif "overheat" in voice.symptoms or telem_features["engine_temp_norm"] > 0.7:
            category = "Cooling/Overheat"
        elif "oil" in voice.symptoms or telem_features["oil_pressure_low_norm"] > 0.6:
            category = "Lubrication/Oil Pressure"
        elif "vibration" in voice.symptoms or telem_features["vibration_norm"] > 0.6:
            category = "Drivetrain/Mechanical Vibration"
        else:
            category = "General Inspection"
        contrib = {**telem_features, "voice_severity": voice.severity}
        return DiagnosisResult(risk_score=round(risk, 3), issue_category=category, contributing_signals=contrib)


class SchedulingAgent:
    CENTERS = [
        {"id": "BLR-01", "name": "Bengaluru Central", "lat": 12.9716, "lon": 77.5946},
        {"id": "DEL-02", "name": "Delhi West", "lat": 28.7041, "lon": 77.1025},
        {"id": "MUM-03", "name": "Mumbai Andheri", "lat": 19.0760, "lon": 72.8777},
        {"id": "CHE-04", "name": "Chennai North", "lat": 13.0827, "lon": 80.2707},
    ]

    @staticmethod
    def _dist_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        # rough planar approximation for small distances
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) * 111

    def schedule(self, customer_loc: Tuple[float, float], diag: DiagnosisResult) -> ScheduleResult:
        nearest = min(self.CENTERS, key=lambda c: self._dist_km(customer_loc, (c["lat"], c["lon"])))
        urgent = diag.risk_score >= 0.7 or diag.issue_category != "General Inspection"
        priority = "urgent" if urgent else "normal"
        # simple slot: now + offset minutes
        offset = 60 if urgent else 3 * 24 * 60
        eta = int(self._dist_km(customer_loc, (nearest["lat"], nearest["lon"])) / 40 * 60)  # assume 40km/h
        slot_ts = time.time() + (offset * 60)
        slot_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(slot_ts))
        return ScheduleResult(center_name=nearest["name"], center_id=nearest["id"], eta_minutes=eta, slot_iso=slot_iso, priority=priority)


class UEBAAgent:
    def monitor(self, voice: VoiceSummary, diag: DiagnosisResult) -> List[str]:
        alerts: List[str] = []
        if voice.intent != "service_request" and diag.risk_score > 0.8:
            alerts.append("High risk without explicit service intent")
        if voice.severity > 0.9 and diag.risk_score < 0.3:
            alerts.append("Mismatched high severity vs low model risk")
        return alerts


class FeedbackAgent:
    def generate(self, diag: DiagnosisResult, schedule: ScheduleResult) -> Dict[str, str]:
        return {
            "oem_quality_flag": "investigate" if diag.issue_category in {"Lubrication/Oil Pressure", "Cooling/Overheat"} else "monitor",
            "recommended_action": "pre-stock parts" if schedule.priority == "urgent" else "standard-prep",
        }


class DataAnalysisAgent:
    def aggregate(self, voice: VoiceSummary, diag: DiagnosisResult, schedule: ScheduleResult) -> Dict[str, object]:
        return {
            "customer_id": voice.customer_id,
            "issue_category": diag.issue_category,
            "risk_score": diag.risk_score,
            "priority": schedule.priority,
            "center_id": schedule.center_id,
        }
