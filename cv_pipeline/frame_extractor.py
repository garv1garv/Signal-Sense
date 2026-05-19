"""
Frame extraction with perceptual hash deduplication.

Extracts frames at a configurable FPS from video files, filtering
near-duplicate frames via DCT-based perceptual hashing to avoid
wasting downstream CV compute on static scenes.
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Generator


@dataclass
class Frame:
    """Single extracted video frame with metadata."""
    idx: int
    timestamp_ms: float
    image: np.ndarray          # HWC BGR uint8


class FrameExtractor:
    """Extracts frames at a fixed FPS, optionally filtering near-duplicates
    using perceptual hash distance to avoid wasting CV compute."""

    def __init__(self, target_fps: float = 2.0, phash_threshold: int = 3):
        """
        Args:
            target_fps: Target extraction rate. Source frames are subsampled
                        to approximate this FPS.
            phash_threshold: Minimum Hamming distance between consecutive
                             perceptual hashes to accept a frame. Lower values
                             keep more frames; higher values are more aggressive.
        """
        self.target_fps = target_fps
        self.phash_threshold = phash_threshold

    def _phash(self, img: np.ndarray) -> np.ndarray:
        """Compute a 64-bit perceptual hash via low-frequency DCT coefficients."""
        gray = cv2.cvtColor(cv2.resize(img, (32, 32)), cv2.COLOR_BGR2GRAY)
        dct = cv2.dct(gray.astype(np.float32))
        dct_low = dct[:8, :8]
        return (dct_low > dct_low.mean()).flatten()

    def extract(self, video_path: str) -> Generator[Frame, None, None]:
        """
        Yield deduplicated frames from a video file.

        Args:
            video_path: Path to the input video file.

        Yields:
            Frame objects with index, timestamp, and BGR image data.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(src_fps / self.target_fps))
        prev_hash = None
        idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                h = self._phash(frame)
                if prev_hash is None or np.sum(h ^ prev_hash) > self.phash_threshold:
                    ts = cap.get(cv2.CAP_PROP_POS_MSEC)
                    yield Frame(idx=idx, timestamp_ms=ts, image=frame)
                    prev_hash = h
            idx += 1
        cap.release()

    def extract_to_dir(
        self, video_path: str, output_dir: str, fmt: str = "jpg"
    ) -> list[str]:
        """
        Extract frames and save to disk. Returns list of saved file paths.

        Args:
            video_path: Path to the input video file.
            output_dir: Directory to save extracted frames.
            fmt: Image format (jpg, png).

        Returns:
            List of absolute paths to saved frame images.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        saved = []
        for frame in self.extract(video_path):
            p = out / f"frame_{frame.idx:06d}.{fmt}"
            cv2.imwrite(str(p), frame.image)
            saved.append(str(p))
        return saved
