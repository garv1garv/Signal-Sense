"""
Celery tasks for nightly HPO and background processing.

The nightly-hpo beat task fires at 2 AM, runs 30 Optuna trials,
selects the best Pareto-optimal config meeting the latency SLA,
and pushes it to Redis for zero-downtime serving.
"""

import os
import logging
from celery import Celery
from celery.schedules import crontab

logger = logging.getLogger(__name__)

_REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
_REDIS_PORT = os.environ.get("REDIS_PORT", "6379")

app = Celery(
    "signalsense",
    broker=f"redis://{_REDIS_HOST}:{_REDIS_PORT}/0",
    backend=f"redis://{_REDIS_HOST}:{_REDIS_PORT}/1",
)

# ── Celery configuration ──
app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,   # one task at a time (GPU bound)
)

# ── Beat schedule: nightly HPO at 2 AM UTC ──
app.conf.beat_schedule = {
    "nightly-hpo": {
        "task":     "serving.tasks.run_nightly_hpo",
        "schedule": crontab(hour=2, minute=0),
    }
}


@app.task(
    name="serving.tasks.run_nightly_hpo",
    bind=True,
    max_retries=2,
    default_retry_delay=300,
)
def run_nightly_hpo(self, n_trials: int = 30, latency_sla_ms: float = 200.0):
    """
    Nightly HPO task: run Optuna trials and hot-swap the best config.

    Args:
        n_trials: Number of optimization trials to run.
        latency_sla_ms: Maximum acceptable p95 latency in milliseconds.
            Only Pareto trials meeting this SLA are considered.

    Returns:
        Dict with swap status and selected trial info.
    """
    from hpo.run_study import run_hpo
    from hpo.hot_swap import push_best_config

    try:
        logger.info(f"Starting nightly HPO: {n_trials} trials")
        pareto = run_hpo(n_trials=n_trials)

        # Filter by latency SLA: -values[2] is negated latency
        candidates = [t for t in pareto if -t.values[2] < latency_sla_ms]

        if candidates:
            # Pick highest AUROC among SLA-compliant candidates
            best = max(candidates, key=lambda t: t.values[0])
            push_best_config(best)
            result = {
                "status": "swapped",
                "trial": best.number,
                "auroc": best.values[0],
                "bert_f1": best.values[1],
                "p95_latency": -best.values[2],
            }
            logger.info(f"HPO complete — swapped to trial {best.number}")
            return result
        else:
            logger.warning(
                f"No Pareto trial meets latency SLA of {latency_sla_ms}ms. "
                f"Keeping current config."
            )
            return {"status": "no_valid_candidate", "pareto_count": len(pareto)}

    except Exception as exc:
        logger.error(f"Nightly HPO failed: {exc}", exc_info=True)
        raise self.retry(exc=exc)


@app.task(name="serving.tasks.analyze_video_async")
def analyze_video_async(video_path: str) -> dict:
    """
    Async video analysis task for batch processing.

    Can be enqueued via the API for non-blocking analysis of
    large video files or batch processing jobs.
    """
    from cv_pipeline.frame_extractor import FrameExtractor
    from cv_pipeline.detector import YOLODetector
    from cv_pipeline.clip_classifier import CLIPSceneClassifier
    from hpo.hot_swap import get_active_config

    cfg = get_active_config()
    extractor = FrameExtractor(target_fps=2.0)
    detector = YOLODetector(
        conf_threshold=cfg.get("yolo_conf", 0.35),
        iou_threshold=cfg.get("yolo_iou", 0.45),
    )
    clip_clf = CLIPSceneClassifier(temperature=cfg.get("clip_temp", 0.07))

    events = []
    for frame in extractor.extract(video_path):
        dets = detector.detect(frame.image)
        scene = clip_clf.classify(frame.image)
        top_scene = max(scene, key=scene.get)
        events.append({
            "timestamp_ms": frame.timestamp_ms,
            "detections": len(dets),
            "top_scene": top_scene,
            "anomaly_score": scene[top_scene],
        })

    return {"video": video_path, "event_count": len(events), "events": events}
