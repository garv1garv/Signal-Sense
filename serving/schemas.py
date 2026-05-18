"""
Pydantic schemas for the SignalSense API.

Defines request/response models with full validation for the
video analysis endpoint. All models use strict typing and
descriptive field metadata for auto-generated OpenAPI docs.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from enum import Enum


class Severity(str, Enum):
    """Event severity classification."""
    normal   = "normal"
    warning  = "warning"
    critical = "critical"


class DetectedObject(BaseModel):
    """Single detected object within a video frame."""
    class_name:  str = Field(..., description="Object class name from YOLO")
    confidence:  float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    bbox_xyxy:   List[float] = Field(
        ...,
        min_length=4, max_length=4,
        description="Bounding box [x1, y1, x2, y2] normalized 0-1"
    )


class VideoEvent(BaseModel):
    """A single detected event within the video timeline."""
    timestamp_ms:   float = Field(..., description="Event timestamp in milliseconds")
    event:          str = Field(..., description="Concise event description")
    severity:       Severity = Field(..., description="Event severity level")
    reasoning:      str = Field(..., description="One-sentence reasoning for the classification")
    anomaly_score:  float = Field(..., ge=0.0, le=1.0, description="Anomaly likelihood score")
    detections:     List[DetectedObject] = Field(
        default_factory=list,
        description="Objects detected in the event's key frame"
    )
    scene_probs:    Dict[str, float] = Field(
        default_factory=dict,
        description="CLIP scene category probabilities"
    )


class AnalyzeResponse(BaseModel):
    """Full response from the /v1/analyze endpoint."""
    video_id:            str = Field(..., description="Unique identifier for this analysis")
    duration_ms:         float = Field(..., description="Total video duration in milliseconds")
    events:              List[VideoEvent] = Field(
        default_factory=list,
        description="Chronological list of detected events"
    )
    active_model_config: Dict = Field(
        default_factory=dict,
        description="Active HPO trial configuration used for this analysis"
    )
    processing_ms:       float = Field(..., description="Server-side processing time in milliseconds")


class AnalyzeRequest(BaseModel):
    """Optional parameters for the analyze endpoint."""
    target_fps:     float = Field(default=2.0, ge=0.5, le=30.0, description="Frame extraction FPS")
    narration_window: int = Field(default=4, ge=2, le=16, description="Frames per narration window")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    gpu:    str = Field(..., description="GPU device name or 'cpu'")


class ConfigResponse(BaseModel):
    """Active HPO configuration response."""
    trial_number: Optional[int] = Field(None, description="Active Optuna trial number")
    auroc:        Optional[float] = Field(None, description="Trial AUROC score")
    bert_f1:      Optional[float] = Field(None, description="Trial BERTScore F1")
    p95_latency:  Optional[float] = Field(None, description="Trial p95 latency in ms")
