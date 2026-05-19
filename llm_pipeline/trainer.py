"""
QLoRA fine-tuning on Phi-3-Vision.

Loads the base model in 4-bit quantization (NF4 + double quant),
applies LoRA adapters to attention + MLP projection layers, and
trains using SFTTrainer with cosine LR schedule and W&B logging.

All LoRA hyperparameters (rank, alpha, dropout) and training params
(LR, batch size, warmup) are HPO target parameters tuned by Optuna.
"""

import torch
import json
import logging
from pathlib import Path
from PIL import Image

from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from datasets import Dataset

logger = logging.getLogger(__name__)


def load_model_and_processor(
    model_id: str = "microsoft/Phi-3.5-vision-instruct",
    lora_rank: int = 16,            # <-- HPO target param
    lora_alpha: int = 32,           # <-- HPO target param
    lora_dropout: float = 0.05,     # <-- HPO target param
):
    """
    Load Phi-3-Vision in 4-bit quantization and attach LoRA adapters.

    Returns:
        Tuple of (peft_model, processor).
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    # Flash Attention 2 requires the `flash-attn` package (not in requirements).
    # Detect availability and fall back to eager attention gracefully.
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "eager"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        _attn_implementation=attn_impl,
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_rslora=True,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, processor


def make_dataset(
    jsonl_path: str,
    processor,
    max_samples: int = 2000,
) -> Dataset:
    """
    Load JSONL training data and tokenize into model-ready format.

    Each sample consists of a single representative frame paired with
    a structured narration in the Phi-3 chat format.

    Args:
        jsonl_path: Path to the JSONL file from dataset_builder.
        processor: The model's processor/tokenizer.
        max_samples: Maximum number of samples to load.

    Returns:
        HuggingFace Dataset ready for SFTTrainer.
    """
    records = []
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            label = r["label"]
            text = (
                f"<|user|>\n<|image_1|>\n"
                f"What is happening in this video sequence?\n<|end|>\n"
                f"<|assistant|>\n"
                f"Event: {label['event']}. "
                f"Severity: {label['severity']}. "
                f"Reasoning: {label['reasoning']}<|end|>"
            )
            # Use first frame as representative image
            frame_path = r["frame_paths"][0] if r["frame_paths"] else None
            if frame_path is None or not Path(frame_path).exists():
                continue

            images = [Image.open(frame_path).convert("RGB")]
            inputs = processor(text=text, images=images, return_tensors="pt")
            records.append({
                "input_ids": inputs["input_ids"][0],
                "attention_mask": inputs["attention_mask"][0],
                "pixel_values": inputs["pixel_values"][0],
                "labels": inputs["input_ids"][0].clone(),
                "reference_event": label["event"],
                "frame_path": frame_path,
            })
            if len(records) >= max_samples:
                break

    logger.info(f"Loaded {len(records)} training samples from {jsonl_path}")
    return Dataset.from_list(records)


def train(
    jsonl_path: str,
    output_dir: str,
    learning_rate: float = 2e-4,    # <-- HPO target param
    batch_size: int = 4,            # <-- HPO target param
    num_epochs: int = 3,
    warmup_steps: int = 50,         # <-- HPO target param
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
):
    """
    Full QLoRA fine-tuning pipeline.

    Loads model, prepares dataset, trains with SFTTrainer, saves
    adapter weights, and returns evaluation metrics.

    Args:
        jsonl_path: Path to JSONL training data.
        output_dir: Directory to save adapter weights and checkpoints.
        learning_rate: AdamW learning rate (HPO param).
        batch_size: Per-device training batch size (HPO param).
        num_epochs: Number of training epochs.
        warmup_steps: LR warmup steps (HPO param).
        lora_rank: LoRA rank (HPO param).
        lora_alpha: LoRA alpha scaling (HPO param).
        lora_dropout: LoRA dropout rate (HPO param).

    Returns:
        Dict of evaluation metrics from the trainer.
    """
    model, processor = load_model_and_processor(
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    dataset = make_dataset(jsonl_path, processor)
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_ds, eval_ds = split["train"], split["test"]

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=max(1, 16 // batch_size),
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="wandb",
        run_name=f"signalsense-r{lora_rank}-lr{learning_rate:.0e}",
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    eval_results = trainer.evaluate()
    
    # Custom evaluation on eval_ds to compute bert_f1_mean
    try:
        from llm_pipeline.evaluate import evaluate_narration
        
        logger.info("Running custom BERTScore evaluation on validation split...")
        model.eval()
        predictions = []
        references = []
        
        prompt_text = (
            "<|user|>\n<|image_1|>\n"
            "What is happening in this video sequence?\n<|end|>\n"
            "<|assistant|>\n"
        )
        
        for idx in range(len(eval_ds)):
            sample = eval_ds[idx]
            ref_event = sample["reference_event"]
            frame_path = sample["frame_path"]
            
            # Load real image
            img = Image.open(frame_path).convert("RGB")
            
            inputs = processor(
                text=prompt_text,
                images=[img],
                return_tensors="pt"
            ).to(model.device)
            
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False,
                    temperature=1.0,
                )
                
            generated = output_ids[:, inputs["input_ids"].shape[1]:]
            text = processor.decode(generated[0], skip_special_tokens=True).strip()
            
            # Parse output
            pred_event = "unknown"
            for line in text.split("\n"):
                line = line.strip()
                if line.lower().startswith("event:"):
                    pred_event = line.split(":", 1)[1].strip().rstrip(".")
                    break
            if pred_event == "unknown" and text:
                pred_event = text[:200]
                
            predictions.append(pred_event)
            references.append(ref_event)
            
        metrics = evaluate_narration(predictions, references, device=str(model.device))
        eval_results["eval_bert_f1_mean"] = metrics.get("bert_f1_mean", 0.0)
        logger.info(f"Custom evaluation BERTScore F1: {eval_results['eval_bert_f1_mean']:.4f}")
    except Exception as eval_err:
        logger.error(f"Failed to run custom evaluation: {eval_err}", exc_info=True)
        eval_results["eval_bert_f1_mean"] = 0.0

    logger.info(f"Final Eval results: {eval_results}")
    return eval_results
