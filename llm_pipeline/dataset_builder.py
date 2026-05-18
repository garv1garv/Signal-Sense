"""
Synthetic dataset generation using a local open-source VLM.

Strategy: use a powerful local Vision-Language Model (like Phi-3-Vision) to watch
frame sequences and write structured narration. Then fine-tune the same model
architecture on those (frames, narration) pairs using QLoRA for task-specific adaptation.
This gives you a labeled dataset without manual annotation or API costs.
"""

import json
import logging
import torch
from pathlib import Path
from typing import Optional
from PIL import Image

import cv2
from transformers import AutoModelForCausalLM, AutoProcessor

from cv_pipeline.frame_extractor import FrameExtractor

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a surveillance analyst. Given a sequence of video frames,
write a single timestamped event description in this exact JSON format:
{
  "event": "<concise action description>",
  "severity": "normal|warning|critical",
  "reasoning": "<one sentence why>"
}
Be specific. Mention objects, people counts, and actions. Never say 'I see'."""


class LocalTeacherVLM:
    """
    Local Vision-Language Model teacher for generating synthetic labels.
    Uses Phi-3-Vision (or similar) to avoid expensive API calls to OpenAI.
    """
    def __init__(self, model_id: str = "microsoft/Phi-3.5-vision-instruct", device: str = "cuda"):
        self.device = device
        logger.info(f"Loading teacher model {model_id}...")
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()

    @torch.no_grad()
    def generate_narration(self, frame_paths: list[str], timestamp_ms: float) -> dict:
        # Phi-3-Vision prompt format for multiple images
        images = [Image.open(p).convert("RGB") for p in frame_paths[:8]]
        
        # Build prompt with image placeholders
        image_tags = "\n".join([f"<|image_{i+1}|>" for i in range(len(images))])
        text = (
            f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
            f"<|user|>\n{image_tags}\n"
            f"Timestamp: {timestamp_ms / 1000:.1f}s. Describe what is happening in the exact JSON format requested.<|end|>\n"
            f"<|assistant|>\n"
        )
        
        inputs = self.processor(text=text, images=images, return_tensors="pt").to(self.device)

        output_ids = self.model.generate(
            **inputs, 
            max_new_tokens=200, 
            do_sample=False,
            temperature=1.0,
        )
        
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        output_text = self.processor.decode(generated[0], skip_special_tokens=True).strip()
        
        # Try to parse JSON from output
        try:
            # Simple heuristic to extract JSON block if the model babbles
            if "{" in output_text and "}" in output_text:
                json_str = output_text[output_text.find("{"):output_text.rfind("}")+1]
                return json.loads(json_str)
            else:
                return json.loads(output_text)
        except Exception as e:
            logger.warning(f"Failed to parse teacher output as JSON: {output_text}. Error: {e}")
            return {
                "event": output_text[:100],
                "severity": "normal",
                "reasoning": "Failed to parse JSON"
            }


def build_dataset(
    video_dir: str,
    output_path: str,
    samples: int = 2000,
    target_fps: float = 1.0,
    window_size: int = 8,
    stride: int = 4,
    temp_dir: Optional[str] = None,
    teacher_model_id: str = "microsoft/Phi-3.5-vision-instruct",
):
    """
    Iterate videos, sample sliding windows, generate labels via local VLM,
    and save as JSONL for downstream QLoRA fine-tuning.
    """
    if temp_dir is None:
        temp_dir = str(Path(output_path).parent / "tmp_frames")
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    records = []
    extractor = FrameExtractor(target_fps=target_fps)
    
    # Initialize local teacher VLM
    teacher = LocalTeacherVLM(model_id=teacher_model_id)

    for video in sorted(Path(video_dir).glob("**/*.mp4")):
        logger.info(f"Processing: {video.name}")
        frames = list(extractor.extract(str(video)))

        for i in range(0, len(frames) - window_size, stride):
            window = frames[i : i + window_size]
            # Save frames temporarily
            paths = []
            for f in window:
                p = str(Path(temp_dir) / f"frame_{f.idx:06d}.jpg")
                cv2.imwrite(p, f.image)
                paths.append(p)

            try:
                label = teacher.generate_narration(paths, window[0].timestamp_ms)
            except Exception as e:
                logger.warning(f"Local VLM call failed at frame {window[0].idx}: {e}")
                continue

            records.append({
                "frame_paths": paths,
                "timestamp_ms": window[0].timestamp_ms,
                "video": str(video),
                "label": label,
            })
            if len(records) >= samples:
                break
        if len(records) >= samples:
            break

    with open(output_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    logger.info(f"Saved {len(records)} samples to {output_path}")
    return len(records)

