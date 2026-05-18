# SignalSense — Computer Vision Pipeline
from .frame_extractor import FrameExtractor, Frame
from .detector import YOLODetector, Detection, FrameDetections
from .clip_classifier import CLIPSceneClassifier, SCENE_CATEGORIES
from .dino_embedder import DINOEmbedder
from .temporal_model import TemporalEventDetector

__all__ = [
    "FrameExtractor", "Frame",
    "YOLODetector", "Detection", "FrameDetections",
    "CLIPSceneClassifier", "SCENE_CATEGORIES",
    "DINOEmbedder",
    "TemporalEventDetector",
]
