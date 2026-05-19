import logging
import sys
from llm_pipeline.dataset_builder import build_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

print("Starting synthetic dataset generation...")

# Generate train annotations from raw training videos
print("Building training dataset...")
train_count = build_dataset(
    video_dir="data/raw",
    output_path="data/annotations/train.jsonl",
    samples=25,
    target_fps=2.0,      # match target fps in serving
    window_size=4,       # sliding window of 4 frames
    stride=2,
)
print(f"Generated {train_count} samples for train.jsonl")

# Generate eval annotations from normal/anomaly eval videos
print("Building evaluation dataset...")
eval_count = build_dataset(
    video_dir="data/eval_videos",
    output_path="data/annotations/eval.jsonl",
    samples=15,
    target_fps=2.0,      # match target fps in serving
    window_size=4,       # sliding window of 4 frames
    stride=2,
)
print(f"Generated {eval_count} samples for eval.jsonl")
print("Synthetic dataset generation complete!")
