"""
CLIP zero-shot scene classification.

Uses OpenAI CLIP to classify video frames against a predefined set of
surveillance scene categories. The temperature parameter controls softmax
sharpness and is an HPO target — lower temperatures produce more confident
(peaked) distributions.
"""

import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import List, Dict


SCENE_CATEGORIES = [
    "normal activity",
    "person in restricted area",
    "crowd gathering",
    "equipment malfunction",
    "safety violation",
    "unattended object",
]


class CLIPSceneClassifier:
    """
    Zero-shot scene classifier using CLIP embeddings.

    Pre-encodes text labels at init time for efficiency. Only image
    encoding runs per-frame. Temperature is an HPO parameter that
    controls the softmax sharpness of the similarity distribution.
    """

    def __init__(
        self,
        model_id: str = "openai/clip-vit-large-patch14",
        temperature: float = 0.07,      # <-- HPO target param
        device: str | None = None,
        categories: List[str] | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = temperature
        self.categories = categories or SCENE_CATEGORIES
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        # Pre-encode text labels once
        self._text_features = self._encode_labels(self.categories)

    @torch.no_grad()
    def _encode_labels(self, labels: List[str]) -> torch.Tensor:
        """Encode text labels into normalized CLIP embeddings."""
        inputs = self.processor(
            text=labels, return_tensors="pt", padding=True
        ).to(self.device)
        feats = self.model.get_text_features(**inputs)
        return feats / feats.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def classify(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Classify a single BGR frame against scene categories.

        Args:
            frame: HWC BGR uint8 numpy array.

        Returns:
            Dict mapping category name to probability.
        """
        img = Image.fromarray(frame[..., ::-1])  # BGR → RGB
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        img_feats = self.model.get_image_features(**inputs)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        logits = (img_feats @ self._text_features.T) / self.temperature
        probs = logits.softmax(dim=-1).squeeze().tolist()
        return dict(zip(self.categories, probs))

    @torch.no_grad()
    def classify_batch(self, frames: List[np.ndarray]) -> List[Dict[str, float]]:
        """Classify a batch of frames. Returns list of probability dicts."""
        images = [Image.fromarray(f[..., ::-1]) for f in frames]
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
        img_feats = self.model.get_image_features(**inputs)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        logits = (img_feats @ self._text_features.T) / self.temperature
        probs = logits.softmax(dim=-1).tolist()
        return [dict(zip(self.categories, p)) for p in probs]

    def update_temperature(self, temperature: float) -> None:
        """Hot-update temperature from HPO without reloading model."""
        self.temperature = temperature
