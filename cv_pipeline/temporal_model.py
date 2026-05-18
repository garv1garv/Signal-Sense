"""
Temporal event detector using a lightweight transformer.

Operates over a sliding window of DINOv2 CLS embeddings to classify
whether a temporal sequence contains a notable event. Window size,
dropout, and layer count are HPO target parameters.
"""

import torch
import torch.nn as nn
from typing import Optional


# Default number of scene categories (matches SCENE_CATEGORIES in clip_classifier.py)
DEFAULT_NUM_CLASSES = 6


class TemporalEventDetector(nn.Module):
    """
    Lightweight transformer over a sliding window of DINOv2 CLS embeddings.
    Classifies whether a temporal sequence contains a notable event.

    HPO target parameters:
        - window_size: Number of frames in the sliding window
        - dropout: Transformer encoder dropout rate
        - num_layers: Number of transformer encoder layers
        - num_heads: Number of attention heads
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 4,
        num_layers: int = 2,
        window_size: int = 8,       # <-- HPO target param
        num_classes: int = DEFAULT_NUM_CLASSES,
        dropout: float = 0.1,       # <-- HPO target param
    ):
        super().__init__()
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # Learnable positional embeddings for temporal ordering
        self.pos_embed = nn.Parameter(
            torch.randn(1, window_size, embed_dim) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Classify a temporal window of frame embeddings.

        Args:
            x: (batch, window_size, embed_dim) tensor of DINOv2 CLS tokens.
            mask: Optional attention mask for padded sequences.

        Returns:
            (batch, num_classes) logits tensor.
        """
        # Add positional encoding
        x = x + self.pos_embed[:, :x.size(1), :]
        out = self.transformer(x, src_key_padding_mask=mask)
        out = self.norm(out[:, -1, :])        # classify on last frame token
        return self.head(out)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities instead of raw logits."""
        self.eval()
        logits = self.forward(x)
        return logits.softmax(dim=-1)
