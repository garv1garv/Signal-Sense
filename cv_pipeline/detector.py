"""
YOLOv9 object & action detection wrapper.

Provides a clean interface over Ultralytics YOLO with HPO-tunable
confidence and IoU thresholds. Returns normalized bounding boxes
and an optional anomaly score placeholder for downstream HPO logic.
"""

import torch
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass, field
from typing import List


@dataclass
class Detection:
    """Single detected object in a frame."""
    class_name: str
    confidence: float
    bbox_xyxy: List[float]      # [x1, y1, x2, y2] normalized 0-1


@dataclass
class FrameDetections:
    """All detections for a single frame, plus an anomaly score."""
    timestamp_ms: float
    detections: List[Detection] = field(default_factory=list)
    anomaly_score: float = 0.0  # filled by HPO-tunable threshold logic


class YOLODetector:
    """
    YOLOv9 wrapper with HPO-tunable parameters.

    The conf_threshold and iou_threshold are primary targets for
    Optuna optimization — they directly affect precision/recall
    trade-off and thus AUROC on the anomaly detection task.
    """

    def __init__(
        self,
        model_path: str = "yolov9c.pt",
        conf_threshold: float = 0.35,   # <-- HPO target param
        iou_threshold: float = 0.45,    # <-- HPO target param
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path)
        self.conf = conf_threshold
        self.iou = iou_threshold

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run detection on a single BGR frame.

        Args:
            frame: HWC BGR uint8 numpy array.

        Returns:
            List of Detection objects with normalized coordinates.
        """
        h, w = frame.shape[:2]
        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(Detection(
                class_name=self.model.names[int(box.cls)],
                confidence=float(box.conf),
                bbox_xyxy=[x1 / w, y1 / h, x2 / w, y2 / h],
            ))
        return detections

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """
        Run detection on a batch of frames.

        Args:
            frames: List of HWC BGR uint8 numpy arrays.

        Returns:
            List of detection lists, one per frame.
        """
        return [self.detect(f) for f in frames]

    def update_thresholds(self, conf: float, iou: float) -> None:
        """Hot-update thresholds from HPO without reloading model weights."""
        self.conf = conf
        self.iou = iou
