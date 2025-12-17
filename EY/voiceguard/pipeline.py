from typing import Dict, Tuple
from .agents import (
    Telemetry,
    VoiceCall,
    VoiceCustomerAgent,
    TelemetryAgent,
    DiagnosisAgent,
    SchedulingAgent,
    UEBAAgent,
    FeedbackAgent,
    DataAnalysisAgent,
)
import time


class VoiceGuardPipeline:
    def __init__(self, model_obj=None):
        self.voice_agent = VoiceCustomerAgent()
        self.telemetry_agent = TelemetryAgent()
        self.diagnosis_agent = DiagnosisAgent(model_obj=model_obj)
        self.scheduling_agent = SchedulingAgent()
        self.ueba_agent = UEBAAgent()
        self.feedback_agent = FeedbackAgent()
        self.data_agent = DataAnalysisAgent()

    def run(self, voice_text: str, telemetry_payload: Dict, customer: Dict) -> Dict:
        voice = VoiceCall(
            customer_id=customer.get("id", "unknown"),
            text=voice_text,
            timestamp=time.time(),
            location=tuple(customer.get("location", (12.9716, 77.5946))),
        )
        telem = Telemetry(
            engine_temp_c=float(telemetry_payload.get("engine_temp_c", 90)),
            battery_voltage=float(telemetry_payload.get("battery_voltage", 12.4)),
            oil_pressure_psi=float(telemetry_payload.get("oil_pressure_psi", 35)),
            vibration_g=float(telemetry_payload.get("vibration_g", 0.4)),
            speed_kph=float(telemetry_payload.get("speed_kph", 40)),
            odometer_km=float(telemetry_payload.get("odometer_km", 45000)),
            error_codes=list(telemetry_payload.get("error_codes", [])),
            location=tuple(telemetry_payload.get("location", customer.get("location", (12.9716, 77.5946)))),
        )

        voice_summary = self.voice_agent.process(voice)
        telem_features = self.telemetry_agent.process(telem)
        diagnosis = self.diagnosis_agent.process(voice_summary, telem_features)
        schedule = self.scheduling_agent.schedule(voice.location, diagnosis)
        ueba_alerts = self.ueba_agent.monitor(voice_summary, diagnosis)
        feedback = self.feedback_agent.generate(diagnosis, schedule)
        aggregate = self.data_agent.aggregate(voice_summary, diagnosis, schedule)

        return {
            "voice_summary": voice_summary.__dict__,
            "telemetry_features": telem_features,
            "diagnosis": {
                "risk_score": diagnosis.risk_score,
                "issue_category": diagnosis.issue_category,
                "signals": diagnosis.contributing_signals,
            },
            "schedule": {
                "center_id": schedule.center_id,
                "center_name": schedule.center_name,
                "slot": schedule.slot_iso,
                "eta_minutes": schedule.eta_minutes,
                "priority": schedule.priority,
            },
            "security_alerts": ueba_alerts,
            "oem_feedback": feedback,
            "analytics": aggregate,
        }


def build_pipeline(model_obj=None) -> VoiceGuardPipeline:
    return VoiceGuardPipeline(model_obj=model_obj)

