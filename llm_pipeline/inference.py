"""
Narration inference engine.

Loads the fine-tuned Phi-3-Vision QLoRA adapter and generates
structured event narrations from frame sequences. Designed for
real-time serving — the model stays loaded in VRAM and accepts
frame batches for inference.
"""

import json
import logging
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Optional

from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

logger = logging.getLogger(__name__)


class NarrationEngine:
    """
    Inference engine for structured video narration.

    Loads the base Phi-3-Vision model in 4-bit quantization and merges
    the fine-tuned LoRA adapter. Generates structured event descriptions
    with severity classifications.
    """

    def __init__(
        self,
        adapter_path: str = "checkpoints/base",
        base_model_id: str = "microsoft/Phi-3.5-vision-instruct",
        device: str = "cuda",
        max_new_tokens: int = 150,
    ):
        self.device = device
        self.max_new_tokens = max_new_tokens

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        logger.info(f"Loading base model: {base_model_id}")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.processor = AutoProcessor.from_pretrained(
            base_model_id, trust_remote_code=True
        )

        # Load fine-tuned LoRA adapter if it exists
        adapter = Path(adapter_path)
        if adapter.exists() and (adapter / "adapter_model.safetensors").exists():
            logger.info(f"Loading LoRA adapter from: {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        else:
            logger.warning(
                f"No adapter found at {adapter_path}, using base model only."
            )

        self.model.eval()

    @torch.no_grad()
    def narrate(self, frames: list[np.ndarray]) -> dict:
        """
        Generate a structured narration for a sequence of BGR frames.

        Args:
            frames: List of HWC BGR uint8 numpy arrays (typically 4-8 frames).

        Returns:
            Dict with keys: event, severity, reasoning.
            Falls back to defaults on parse failure.
        """
        # Use the middle frame as the representative image
        mid = len(frames) // 2
        img = Image.fromarray(frames[mid][..., ::-1])  # BGR → RGB

        prompt = (
            "<|user|>\n<|image_1|>\n"
            "What is happening in this video sequence?\n<|end|>\n"
            "<|assistant|>\n"
        )

        inputs = self.processor(
            text=prompt, images=[img], return_tensors="pt"
        ).to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.1,
        )

        # Decode only the newly generated tokens
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        text = self.processor.decode(generated[0], skip_special_tokens=True).strip()

        return self._parse_output(text)

    def _parse_output(self, text: str) -> dict:
        """
        Parse the model's text output into a structured dict.

        Tries to extract Event/Severity/Reasoning fields. Falls back
        to wrapping the raw text as a normal-severity event.
        """
        result = {
            "event": "unknown",
            "severity": "normal",
            "reasoning": "",
        }

        # Try structured field extraction
        for line in text.split("\n"):
            line = line.strip()
            lower = line.lower()
            if lower.startswith("event:"):
                result["event"] = line.split(":", 1)[1].strip().rstrip(".")
            elif lower.startswith("severity:"):
                sev = line.split(":", 1)[1].strip().lower().rstrip(".")
                if sev in ("normal", "warning", "critical"):
                    result["severity"] = sev
            elif lower.startswith("reasoning:"):
                result["reasoning"] = line.split(":", 1)[1].strip()

        # Fallback: if no structured fields found, use raw text
        if result["event"] == "unknown" and text:
            result["event"] = text[:200]

        return result

    def reload_adapter(self, adapter_path: str) -> None:
        """
        Hot-reload a new LoRA adapter without restarting the process.
        Used by the HPO hot-swap system.
        """
        adapter = Path(adapter_path)
        if not adapter.exists():
            logger.error(f"Adapter not found: {adapter_path}")
            return

        logger.info(f"Hot-reloading adapter from: {adapter_path}")
        # Unload existing adapter if present
        if hasattr(self.model, "unload"):
            self.model.unload()

        self.model = PeftModel.from_pretrained(
            self.model, adapter_path
        )
        self.model.eval()
        logger.info("Adapter reloaded successfully.")
