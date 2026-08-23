"""Prometheus metrics for the inference service, exposed at GET /metrics."""
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

REQUEST_COUNT = Counter(
    "inference_requests_total", "Total number of requests", ["endpoint"]
)
ERROR_COUNT = Counter(
    "inference_errors_total", "Total number of failed requests", ["endpoint"]
)
LATENCY = Histogram(
    "inference_latency_seconds", "Request latency in seconds", ["endpoint"]
)


def render_latest() -> bytes:
    return generate_latest()


CONTENT_TYPE = CONTENT_TYPE_LATEST
