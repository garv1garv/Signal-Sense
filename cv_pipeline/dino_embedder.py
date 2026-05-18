"""
DINOv2 dense feature extraction.

Extracts CLS token embeddings from DINOv2 for use as temporal input
to the TemporalEventDetector transformer. DINOv2 provides rich,
self-supervised visual features that generalize well to downstream
tasks without fine-tuning.
"""

import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from typing import List


class DINOEmbedder:
    """Extracts CLS token embedding from DINOv2 — used as temporal input."""

    def __init__(
        self,
        model_id: str = "facebook/dinov2-base",
        device: str = "cuda",
    ):
        self.device = device
        self.model_id = model_id
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device)
        self.model.eval()

    @torch.no_grad()
    def embed(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract CLS token embedding from a single BGR frame.

        Args:
            frame: HWC BGR uint8 numpy array.

        Returns:
            1-D numpy array of shape (768,) — the CLS token embedding.
        """
        img = Image.fromarray(frame[..., ::-1])  # BGR → RGB
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        cls_token = outputs.last_hidden_state[:, 0, :]   # (1, 768)
        return cls_token.squeeze().cpu().numpy()

    @torch.no_grad()
    def embed_batch(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Extract CLS token embeddings from a batch of BGR frames.

        Args:
            frames: List of HWC BGR uint8 numpy arrays.

        Returns:
            2-D numpy array of shape (N, 768).
        """
        images = [Image.fromarray(f[..., ::-1]) for f in frames]
        inputs = self.processor(
            images=images, return_tensors="pt", padding=True
        ).to(self.device)
        outputs = self.model(**inputs)
        cls_tokens = outputs.last_hidden_state[:, 0, :]   # (N, 768)
        return cls_tokens.cpu().numpy()

    @property
    def embed_dim(self) -> int:
        """Return the dimensionality of the CLS token embedding."""
        return self.model.config.hidden_size
