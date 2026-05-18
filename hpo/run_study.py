"""
Optuna NSGA-II study runner.

Entry point for launching hyperparameter optimization. Uses NSGA-II
multi-objective sampler to find the Pareto frontier across AUROC,
BERTScore F1, and inference latency. Results are logged to both
an SQLite database (for Optuna Dashboard) and W&B.
"""

import logging
import optuna
from optuna.samplers import NSGAIISampler
from hpo.objective import objective

logger = logging.getLogger(__name__)

# Suppress noisy Optuna logs during trials
optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_hpo(
    n_trials: int = 60,
    study_name: str = "signalsense-v1",
    storage: str = "sqlite:///data/hpo_logs/study.db?timeout=30&journal_mode=WAL",
    population_size: int = 20,
    seed: int = 42,
    log_to_wandb: bool = True,
):
    """
    Run the multi-objective HPO study.

    Args:
        n_trials: Number of optimization trials to run.
        study_name: Name for the Optuna study (used in DB and dashboard).
        storage: SQLite URL for persisting study state.
        population_size: NSGA-II population size.
        seed: Random seed for reproducibility.
        log_to_wandb: Whether to log Pareto frontier to W&B.

    Returns:
        List of Pareto-optimal trials.
    """
    if log_to_wandb:
        try:
            import wandb
            wandb.init(project="signalsense-hpo", name=study_name)
        except Exception as e:
            logger.warning(f"W&B init failed: {e}. Continuing without W&B.")
            log_to_wandb = False

    sampler = NSGAIISampler(
        population_size=population_size,
        mutation_prob=None,      # auto-calculated
        crossover_prob=0.9,
        swapping_prob=0.5,
        seed=seed,
    )

    study = optuna.create_study(
        study_name=study_name,
        directions=["maximize", "maximize", "maximize"],  # AUROC, BERTScore, -latency
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )

    logger.info(
        f"Starting HPO: {n_trials} trials, study='{study_name}', "
        f"population={population_size}"
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=1,               # 1 GPU — sequential
        show_progress_bar=True,
    )

    # Extract Pareto frontier
    pareto = study.best_trials
    logger.info(f"Pareto-optimal trials: {len(pareto)}")

    # Log Pareto frontier to W&B
    if log_to_wandb:
        import wandb
        for t in pareto:
            wandb.log({
                "auroc":       t.values[0],
                "bert_f1":     t.values[1],
                "p95_latency": -t.values[2],
                "trial":       t.number,
                **t.params,
            })

    # Print summary
    print(f"\n{'='*60}")
    print(f"Pareto-optimal trials: {len(pareto)}")
    print(f"{'='*60}")
    for t in pareto:
        print(
            f"  Trial {t.number:3d} │ "
            f"AUROC: {t.values[0]:.4f} │ "
            f"BERTScore: {t.values[1]:.4f} │ "
            f"p95 latency: {-t.values[2]:.1f}ms"
        )
    print(f"{'='*60}\n")

    return pareto


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_hpo()
