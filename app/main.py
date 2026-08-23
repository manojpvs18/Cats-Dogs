"""FastAPI inference service for the cats-vs-dogs model.

Endpoints:
    GET  /health   - liveness/readiness check
    POST /predict  - multipart image upload -> {label, probabilities}
    GET  /metrics  - Prometheus scrape endpoint

Run:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
import io
import os
import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

from app.schemas import HealthResponse, PredictResponse
from monitoring.logging_config import get_logger
from monitoring.metrics import (
    ERROR_COUNT,
    LATENCY,
    REQUEST_COUNT,
    render_latest,
)
from src.models.inference import load_model, predict as run_predict

MODEL_PATH = os.environ.get("MODEL_PATH", "artifacts/model.pt")

app = FastAPI(title="Cats vs Dogs Inference API", version="1.0.0")
logger = get_logger("cats-dogs-api")

_model = None
_class_names = None


@app.on_event("startup")
def _load_model_on_startup():
    global _model, _class_names
    try:
        _model, _class_names = load_model(MODEL_PATH)
        logger.info({"event": "model_loaded", "path": MODEL_PATH})
    except Exception as e:
        # Service still starts so /health reports model_loaded=False rather
        # than crashing the container; /predict will 503 until a model exists.
        logger.error({"event": "model_load_failed", "error": str(e)})
        _model, _class_names = None, None


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_loaded=_model is not None)


@app.get("/metrics")
def metrics():
    return Response(content=render_latest(), media_type="text/plain")


@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    start = time.time()
    REQUEST_COUNT.labels(endpoint="/predict").inc()

    if _model is None:
        ERROR_COUNT.labels(endpoint="/predict").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        raw = await file.read()
        image = Image.open(io.BytesIO(raw))
        result = run_predict(_model, _class_names, image)
    except Exception as e:
        ERROR_COUNT.labels(endpoint="/predict").inc()
        logger.error({"event": "predict_error", "error": str(e)})
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    latency_ms = (time.time() - start) * 1000
    LATENCY.labels(endpoint="/predict").observe(latency_ms / 1000.0)

    # Log request/response metadata only — never raw image bytes or PII.
    logger.info({
        "event": "predict",
        "filename": file.filename,
        "predicted_label": result["label"],
        "latency_ms": round(latency_ms, 2),
    })

    return PredictResponse(
        label=result["label"],
        probabilities=result["probabilities"],
        latency_ms=round(latency_ms, 2),
    )
