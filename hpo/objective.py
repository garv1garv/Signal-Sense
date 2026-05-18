"""
Multi-objective function for Optuna NSGA-II optimization.

Objectives (all maximized after negation where needed):
  1. AUROC          — maximize anomaly detection accuracy
  2. BERTScore F1   — maximize narration quality
  3. -p95_latency   — minimize inference latency (negated for maximization)

NSGA-II finds the Pareto frontier across these three objectives.
The operator picks which trade-off suits the deployment SLA.
"""

import time
import logging
import numpy as np
from pathlib import Path
from typing import Optional

from hpo.search_spaces import suggest_cv_params, suggest_lora_params
from cv_pipeline.detector import YOLODetector
from cv_pipeline.clip_classifier import CLIPSceneClassifier
from cv_pipeline.frame_extractor import FrameExtractor
from llm_pipeline.trainer import train
from llm_pipeline.evaluate import evaluate_narration, evaluate_anomaly_detection

logger = logging.getLogger(__name__)

EVAL_VIDEO_DIR = "data/eval_videos/"
TRAIN_JSONL    = "data/annotations/train.jsonl"
EVAL_JSONL     = "data/annotations/eval.jsonl"


def load_eval_videos(eval_dir: str) -> list[tuple[str, int]]:
    """
    Load evaluation video paths with binary anomaly labels.

    Convention: videos in 'anomaly/' subdirectory are labeled 1,
    videos in 'normal/' subdirectory are labeled 0.
    """
    pairs = []
    eval_path = Path(eval_dir)
    for video in sorted(eval_path.glob("normal/**/*.mp4")):
        pairs.append((str(video), 0))
    for video in sorted(eval_path.glob("anomaly/**/*.mp4")):
        pairs.append((str(video), 1))
    return pairs


def load_one_frame(video_path: str) -> np.ndarray:
    """Extract a single representative frame from a video."""
    extractor = FrameExtractor(target_fps=1.0)
    for frame in extractor.extract(video_path):
        return frame.image
    raise ValueError(f"No frames extracted from {video_path}")


def objective(trial) -> tuple[float, float, float]:
    """
    Multi-objective function evaluated by Optuna NSGA-II.

    Each trial:
      1. Suggests CV params (YOLO thresholds, CLIP temperature, etc.)
      2. Suggests LoRA params (rank, alpha, LR, etc.)
      3. Runs CV inference on eval videos → computes AUROC + latency
      4. Runs QLoRA fine-tuning → computes BERTScore F1

    Returns:
        Tuple of (AUROC, BERTScore_F1, -p95_latency_ms).
        All three are maximized by NSGA-II.
    """
    cv_params   = suggest_cv_params(trial)
    lora_params = suggest_lora_params(trial)

    logger.info(f"Trial {trial.number} — CV: {cv_params}")
    logger.info(f"Trial {trial.number} — LoRA: {lora_params}")

    # ── CV evaluation ──
    detector = YOLODetector(
        conf_threshold=cv_params["yolo_conf_threshold"],
        iou_threshold=cv_params["yolo_iou_threshold"],
    )
    clip_clf = CLIPSceneClassifier(temperature=cv_params["clip_temperature"])

    anomaly_scores, labels = [], []
    latencies = []
    eval_videos = load_eval_videos(EVAL_VIDEO_DIR)

    if not eval_videos:
        logger.warning("No eval videos found — using dummy metrics.")
        auroc = 0.5
        p95_latency = 999.0
    else:
        for video_path, label in eval_videos:
            t0 = time.perf_counter()
            frame = load_one_frame(video_path)
            dets  = detector.detect(frame)
            scene = clip_clf.classify(frame)
            score = max(scene.values())   # crude anomaly proxy
            latencies.append((time.perf_counter() - t0) * 1000)
            anomaly_scores.append(score)
            labels.append(label)

        auroc = evaluate_anomaly_detection(anomaly_scores, labels)["auroc"]
        p95_latency = float(np.percentile(latencies, 95))

    # ── LLM evaluation ──
    output_dir = f"data/hpo_logs/trial_{trial.number}"
    try:
        eval_metrics = train(
            jsonl_path=TRAIN_JSONL,
            output_dir=output_dir,
            **lora_params,
        )
        bert_f1 = eval_metrics.get("eval_bert_f1_mean", 0.0)
    except Exception as e:
        logger.error(f"Training failed for trial {trial.number}: {e}")
        bert_f1 = 0.0

    logger.info(
        f"Trial {trial.number} results — "
        f"AUROC: {auroc:.4f}, BERTScore F1: {bert_f1:.4f}, "
        f"p95 latency: {p95_latency:.1f}ms"
    )

    # Return (maximize AUROC, maximize BERTScore, minimize latency via negation)
    return auroc, bert_f1, -p95_latency
