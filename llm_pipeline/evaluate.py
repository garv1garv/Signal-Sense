"""
Evaluation metrics for narration quality and anomaly detection.

- BERTScore: Measures semantic similarity between generated narrations
  and reference descriptions. F1 variant is the primary metric.
- AUROC: Measures anomaly detection accuracy using continuous scores
  against binary labels.
"""

import logging
import numpy as np
from bert_score import score as bert_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

logger = logging.getLogger(__name__)


def evaluate_narration(
    predictions: list[str],
    references: list[str],
    device: str = "cuda",
) -> dict:
    """
    Evaluate narration quality using BERTScore.

    Args:
        predictions: List of generated narration strings.
        references: List of ground-truth narration strings.
        device: Device for BERTScore model inference.

    Returns:
        Dict with bert_f1_mean, bert_f1_std, bert_precision_mean,
        bert_recall_mean.
    """
    if not predictions or not references:
        logger.warning("Empty prediction or reference list for BERTScore.")
        return {"bert_f1_mean": 0.0, "bert_f1_std": 0.0}

    P, R, F1 = bert_score(
        predictions, references, lang="en", device=device, verbose=False
    )
    return {
        "bert_f1_mean": F1.mean().item(),
        "bert_f1_std": F1.std().item(),
        "bert_precision_mean": P.mean().item(),
        "bert_recall_mean": R.mean().item(),
    }


def evaluate_anomaly_detection(
    scores: list[float],
    labels: list[int],   # 1 = anomaly, 0 = normal
) -> dict:
    """
    Evaluate binary anomaly detection using AUROC and AUPRC.

    Args:
        scores: Continuous anomaly scores (higher = more anomalous).
        labels: Binary ground-truth labels (1 = anomaly, 0 = normal).

    Returns:
        Dict with auroc and auprc metrics.
    """
    if len(set(labels)) < 2:
        logger.warning("Only one class present in labels; AUROC is undefined.")
        return {"auroc": 0.5, "auprc": 0.0}

    auroc = roc_auc_score(labels, scores)

    # Also compute area under precision-recall curve (more informative
    # for imbalanced datasets like surveillance anomaly detection)
    precision, recall, _ = precision_recall_curve(labels, scores)
    auprc = auc(recall, precision)

    return {"auroc": auroc, "auprc": auprc}


def evaluate_severity_accuracy(
    predicted_severities: list[str],
    true_severities: list[str],
) -> dict:
    """
    Simple accuracy metric for severity classification.

    Args:
        predicted_severities: List of predicted severity strings.
        true_severities: List of ground-truth severity strings.

    Returns:
        Dict with accuracy and per-class accuracy.
    """
    assert len(predicted_severities) == len(true_severities)
    if not predicted_severities:
        return {"severity_accuracy": 0.0}

    correct = sum(p == t for p, t in zip(predicted_severities, true_severities))
    total = len(predicted_severities)

    # Per-class accuracy
    classes = set(true_severities)
    per_class = {}
    for cls in classes:
        cls_mask = [t == cls for t in true_severities]
        cls_correct = sum(
            p == t for p, t, m in zip(predicted_severities, true_severities, cls_mask) if m
        )
        cls_total = sum(cls_mask)
        per_class[cls] = cls_correct / cls_total if cls_total > 0 else 0.0

    return {
        "severity_accuracy": correct / total,
        "severity_per_class": per_class,
    }
