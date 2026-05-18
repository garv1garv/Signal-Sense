"""
Prometheus metrics for the SignalSense API.

Exposes standard RED (Rate, Errors, Duration) metrics for monitoring
API health and performance. Scraped by Prometheus at /metrics endpoint.
"""

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
)

# ── Request metrics ──

REQUEST_COUNT = Counter(
    "signalsense_requests_total",
    "Total number of video analysis requests processed",
)

REQUEST_LATENCY = Histogram(
    "signalsense_request_duration_ms",
    "Request processing duration in milliseconds",
    buckets=[50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000],
)

ACTIVE_REQUESTS = Gauge(
    "signalsense_active_requests",
    "Number of currently processing requests",
)

# ── Model metrics ──

MODEL_INFERENCE_LATENCY = Histogram(
    "signalsense_model_inference_ms",
    "Per-model inference latency in milliseconds",
    labelnames=["model_name"],
    buckets=[5, 10, 25, 50, 100, 250, 500, 1000],
)

DETECTIONS_PER_FRAME = Histogram(
    "signalsense_detections_per_frame",
    "Number of objects detected per frame",
    buckets=[1, 2, 5, 10, 20, 50, 100],
)

ANOMALY_SCORE = Histogram(
    "signalsense_anomaly_score",
    "Distribution of anomaly scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# ── HPO metrics ──

HPO_TRIAL_COUNT = Counter(
    "signalsense_hpo_trials_total",
    "Total number of HPO trials completed",
)

HPO_ACTIVE_TRIAL = Gauge(
    "signalsense_hpo_active_trial",
    "Currently active HPO trial number",
)

HPO_BEST_AUROC = Gauge(
    "signalsense_hpo_best_auroc",
    "AUROC of the currently active HPO config",
)

# ── System info ──

BUILD_INFO = Info(
    "signalsense_build",
    "Build information",
)
BUILD_INFO.info({
    "version": "1.0.0",
    "stack": "YOLOv9+CLIP+DINOv2+Phi3Vision",
})
