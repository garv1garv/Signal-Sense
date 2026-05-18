"""
Weights & Biases experiment tracking for SignalSense.

Provides a unified logger for tracking training runs, HPO trials,
and serving metrics. Supports logging scalars, images, tables,
and artifacts to W&B for experiment comparison.
"""

import logging
from typing import Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class SignalSenseLogger:
    """
    W&B logger wrapper for consistent experiment tracking.

    Initializes a W&B run and provides methods for logging
    metrics, images, and model artifacts.
    """

    def __init__(
        self,
        project: str = "signalsense",
        run_name: Optional[str] = None,
        config: Optional[dict] = None,
        tags: Optional[list[str]] = None,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self._run = None

        if not enabled:
            logger.info("W&B logging disabled.")
            return

        try:
            import wandb
            self._wandb = wandb
            self._run = wandb.init(
                project=project,
                name=run_name,
                config=config or {},
                tags=tags or [],
                reinit=True,
            )
            logger.info(f"W&B run initialized: {self._run.name}")
        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}. Logging disabled.")
            self.enabled = False

    def log(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        """Log a dict of scalar metrics."""
        if not self.enabled:
            return
        self._wandb.log(metrics, step=step)

    def log_image(
        self,
        key: str,
        image,
        caption: Optional[str] = None,
        step: Optional[int] = None,
    ) -> None:
        """Log an image (PIL, numpy, or path)."""
        if not self.enabled:
            return
        img = self._wandb.Image(image, caption=caption)
        self._wandb.log({key: img}, step=step)

    def log_table(
        self,
        key: str,
        columns: list[str],
        data: list[list],
    ) -> None:
        """Log a table of data."""
        if not self.enabled:
            return
        table = self._wandb.Table(columns=columns, data=data)
        self._wandb.log({key: table})

    def log_artifact(
        self,
        name: str,
        artifact_type: str,
        path: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """Log a model or data artifact."""
        if not self.enabled:
            return
        artifact = self._wandb.Artifact(
            name=name,
            type=artifact_type,
            metadata=metadata or {},
        )
        artifact_path = Path(path)
        if artifact_path.is_dir():
            artifact.add_dir(str(artifact_path))
        else:
            artifact.add_file(str(artifact_path))
        self._run.log_artifact(artifact)
        logger.info(f"Logged artifact: {name} ({artifact_type})")

    def log_hpo_trial(self, trial) -> None:
        """Log an Optuna trial's params and values."""
        if not self.enabled:
            return
        self._wandb.log({
            "trial_number": trial.number,
            "auroc": trial.values[0] if len(trial.values) > 0 else None,
            "bert_f1": trial.values[1] if len(trial.values) > 1 else None,
            "p95_latency": -trial.values[2] if len(trial.values) > 2 else None,
            **trial.params,
        })

    def finish(self) -> None:
        """Finish the W&B run."""
        if self.enabled and self._run:
            self._run.finish()
            logger.info("W&B run finished.")
