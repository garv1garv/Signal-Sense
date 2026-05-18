"""
FastAPI application — SignalSense video analysis API.

Loads CV and LLM models at startup using the active HPO config from Redis.
Processes video uploads through the full pipeline: frame extraction →
YOLO detection → CLIP classification → Phi-3-Vision narration.

Key features:
  - Lifespan-managed model registry (load once, serve many)
  - Hot-swappable config via Redis (no restart needed)
  - Prometheus metrics endpoint for monitoring
  - Async narration via asyncio.to_thread for non-blocking inference
"""

import time
import uuid
import torch
import asyncio
import logging
import tempfile
import shutil
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware

from serving.schemas import (
    AnalyzeResponse,
    VideoEvent,
    DetectedObject,
    HealthResponse,
)
from serving.middleware import verify_api_key, rate_limit
from hpo.hot_swap import get_active_config
from cv_pipeline.frame_extractor import FrameExtractor
from cv_pipeline.detector import YOLODetector
from cv_pipeline.clip_classifier import CLIPSceneClassifier
from cv_pipeline.dino_embedder import DINOEmbedder
from llm_pipeline.inference import NarrationEngine
from monitoring.metrics import REQUEST_LATENCY, REQUEST_COUNT, ACTIVE_REQUESTS

logger = logging.getLogger(__name__)

# ── Model registry (loaded once at startup) ──
models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all ML models at startup, release on shutdown."""
    cfg = get_active_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading models on device: {device}")
    models["detector"] = YOLODetector(
        conf_threshold=cfg.get("yolo_conf", 0.35),
        iou_threshold=cfg.get("yolo_iou", 0.45),
        device=device,
    )
    models["clip"] = CLIPSceneClassifier(
        temperature=cfg.get("clip_temp", 0.07),
        device=device,
    )
    models["dino"] = DINOEmbedder(device=device)
    models["narrator"] = NarrationEngine(
        adapter_path=cfg.get("adapter_path", "checkpoints/base"),
        device=device,
    )
    models["active_cfg"] = cfg
    logger.info(
        f"Models loaded. Active HPO config: "
        f"trial {cfg.get('trial_number', 'default')}"
    )

    yield

    models.clear()
    logger.info("Models unloaded.")


app = FastAPI(
    title="SignalSense API",
    description=(
        "Real-time video understanding API with auto-adapting models. "
        "Processes surveillance video through YOLOv9 detection, CLIP scene "
        "classification, and Phi-3-Vision narration. Model hyperparameters "
        "are automatically optimized via nightly Optuna NSGA-II runs."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ──

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Health check with GPU availability info."""
    return HealthResponse(
        status="ok",
        gpu=(
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "cpu"
        ),
    )


@app.post(
    "/v1/analyze",
    response_model=AnalyzeResponse,
    dependencies=[Depends(verify_api_key), Depends(rate_limit)],
    tags=["Analysis"],
    summary="Analyze a video for events and anomalies",
)
async def analyze_video(file: UploadFile = File(...)):
    """
    Upload a video file for full-pipeline analysis.

    Returns timestamped events with severity classifications,
    object detections, scene probabilities, and narrations.

    Supported formats: .mp4, .avi, .mov, .mkv
    """
    SUPPORTED = (".mp4", ".avi", ".mov", ".mkv")
    if not file.filename or not file.filename.lower().endswith(SUPPORTED):
        raise HTTPException(
            400,
            f"Unsupported file format. Supported: {', '.join(SUPPORTED)}"
        )

    ACTIVE_REQUESTS.inc()
    t_start = time.perf_counter()
    video_id = str(uuid.uuid4())

    # Save upload to temp file
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        extractor  = FrameExtractor(target_fps=2.0)
        detector   = models["detector"]
        clip_clf   = models["clip"]
        narrator   = models["narrator"]
        events     = []
        frame_buffer = []

        for frame in extractor.extract(tmp_path):
            # Run CV inference
            dets      = detector.detect(frame.image)
            scene     = clip_clf.classify(frame.image)
            top_scene = max(scene, key=scene.get)
            score     = scene[top_scene]

            frame_buffer.append(frame)

            # Narrate every 4 frames (sliding window)
            if len(frame_buffer) >= 4:
                narration = await asyncio.to_thread(
                    narrator.narrate,
                    [f.image for f in frame_buffer[-4:]],
                )
                events.append(VideoEvent(
                    timestamp_ms=frame.timestamp_ms,
                    event=narration["event"],
                    severity=narration["severity"],
                    reasoning=narration["reasoning"],
                    anomaly_score=score,
                    detections=[
                        DetectedObject(
                            class_name=d.class_name,
                            confidence=d.confidence,
                            bbox_xyxy=d.bbox_xyxy,
                        )
                        for d in dets
                    ],
                    scene_probs=scene,
                ))

        processing_ms = (time.perf_counter() - t_start) * 1000
        REQUEST_LATENCY.observe(processing_ms)
        REQUEST_COUNT.inc()

        return AnalyzeResponse(
            video_id=video_id,
            duration_ms=processing_ms,
            events=events,
            active_model_config=models.get("active_cfg", {}),
            processing_ms=processing_ms,
        )
    except Exception as e:
        logger.error(f"Analysis failed for {video_id}: {e}", exc_info=True)
        raise HTTPException(500, f"Analysis failed: {str(e)}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)
        ACTIVE_REQUESTS.dec()


@app.get("/v1/config", tags=["System"])
def active_config():
    """Returns which HPO trial is currently serving."""
    return models.get("active_cfg", {})


@app.get("/v1/config/history", tags=["System"])
def config_history():
    """Returns the last 10 HPO config swaps for audit."""
    from hpo.hot_swap import get_config_history
    return get_config_history(last_n=10)


@app.get("/metrics", tags=["Monitoring"])
def metrics():
    """Prometheus metrics endpoint."""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
