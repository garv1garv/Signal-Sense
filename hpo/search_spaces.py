"""
Optuna search space definitions for CV and LoRA hyperparameters.

Each function takes an Optuna Trial and returns a dict of suggested
parameter values. These are consumed by the multi-objective function
in objective.py.
"""

import optuna


def suggest_cv_params(trial: optuna.Trial) -> dict:
    """
    Suggest CV pipeline hyperparameters.

    Tunable parameters:
        - yolo_conf: YOLO confidence threshold (precision/recall trade-off)
        - yolo_iou: YOLO IoU threshold for NMS
        - clip_temp: CLIP softmax temperature (distribution sharpness)
        - window_size: Temporal transformer sliding window length
        - t_dropout: Temporal transformer dropout rate
    """
    return {
        "yolo_conf_threshold":  trial.suggest_float("yolo_conf", 0.20, 0.60),
        "yolo_iou_threshold":   trial.suggest_float("yolo_iou", 0.30, 0.65),
        "clip_temperature":     trial.suggest_float("clip_temp", 0.02, 0.20, log=True),
        "temporal_window_size": trial.suggest_int("window_size", 4, 16),
        "temporal_dropout":     trial.suggest_float("t_dropout", 0.0, 0.3),
    }


def suggest_lora_params(trial: optuna.Trial) -> dict:
    """
    Suggest QLoRA fine-tuning hyperparameters.

    Tunable parameters:
        - lora_rank: LoRA decomposition rank (capacity vs. efficiency)
        - lora_alpha: LoRA scaling factor (typically rank * multiplier)
        - lora_dropout: LoRA dropout for regularization
        - lr: AdamW learning rate (log-uniform)
        - bs: Per-device batch size
        - warmup: LR warmup steps
    """
    rank = trial.suggest_categorical("lora_rank", [16, 32, 64, 128, 256])
    return {
        "lora_rank":     rank,
        "lora_alpha":    rank * trial.suggest_int("alpha_mult", 1, 8),
        "lora_dropout":  trial.suggest_float("lora_dropout", 0.0, 0.15),
        "learning_rate": trial.suggest_float("lr", 5e-5, 5e-4, log=True),
        "batch_size":    trial.suggest_categorical("bs", [2, 4, 8]),
        "warmup_steps":  trial.suggest_int("warmup", 20, 150),
    }
