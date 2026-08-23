from typing import Dict

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class PredictResponse(BaseModel):
    label: str
    probabilities: Dict[str, float]
    latency_ms: float
