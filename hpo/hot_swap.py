"""
Zero-downtime configuration hot-swap via Redis.

After nightly HPO, the winning config is written to a shared Redis key.
The FastAPI serving layer reads it on each request — no restart needed.
This enables the system to autonomously improve its own model config
without any manual intervention or service interruption.
"""

import json
import logging
import os
from typing import Optional

import redis

logger = logging.getLogger(__name__)

ACTIVE_CONFIG_KEY = "signalsense:active_config"
CONFIG_HISTORY_KEY = "signalsense:config_history"

_REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
_REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))


def _get_redis(host: str = _REDIS_HOST, port: int = _REDIS_PORT) -> redis.Redis:
    """Get a Redis connection with decode_responses enabled."""
    return redis.Redis(host=host, port=port, decode_responses=True)


def push_best_config(
    pareto_trial,
    host: str = "redis",
    port: int = 6379,
) -> None:
    """
    Push the best Pareto trial config to Redis for live serving.

    Picks the Pareto trial with highest AUROC that meets the latency SLA.
    Also appends to a config history list for audit trail.

    Args:
        pareto_trial: An Optuna FrozenTrial from the Pareto frontier.
        host: Redis host.
        port: Redis port.
    """
    r = _get_redis(host, port)
    config = {
        **pareto_trial.params,
        "trial_number": pareto_trial.number,
        "auroc":        pareto_trial.values[0],
        "bert_f1":      pareto_trial.values[1],
        "p95_latency":  -pareto_trial.values[2],
    }

    # Store as active config
    r.set(ACTIVE_CONFIG_KEY, json.dumps(config))

    # Append to history for audit trail
    r.rpush(CONFIG_HISTORY_KEY, json.dumps(config))

    logger.info(
        f"Hot-swapped to trial {pareto_trial.number} — "
        f"AUROC {config['auroc']:.3f}, "
        f"p95 latency {config['p95_latency']:.1f}ms"
    )


def get_active_config(
    host: str = "redis",
    port: int = 6379,
) -> dict:
    """
    Retrieve the currently active HPO config from Redis.

    Returns an empty dict if no config has been pushed yet,
    allowing the serving layer to fall back to defaults.
    """
    try:
        r = _get_redis(host, port)
        raw = r.get(ACTIVE_CONFIG_KEY)
        return json.loads(raw) if raw else {}
    except redis.ConnectionError:
        logger.warning("Redis unavailable — returning empty config (defaults).")
        return {}


def get_config_history(
    host: str = "redis",
    port: int = 6379,
    last_n: int = 10,
) -> list[dict]:
    """
    Retrieve the last N config swaps for audit/debugging.

    Args:
        host: Redis host.
        port: Redis port.
        last_n: Number of recent configs to retrieve.

    Returns:
        List of config dicts, most recent last.
    """
    try:
        r = _get_redis(host, port)
        raw_list = r.lrange(CONFIG_HISTORY_KEY, -last_n, -1)
        return [json.loads(raw) for raw in raw_list]
    except redis.ConnectionError:
        logger.warning("Redis unavailable — returning empty history.")
        return []


def set_manual_config(
    config: dict,
    host: str = "redis",
    port: int = 6379,
) -> None:
    """
    Manually override the active config (for debugging or rollback).

    Args:
        config: Full config dict to set as active.
        host: Redis host.
        port: Redis port.
    """
    r = _get_redis(host, port)
    config["_manual_override"] = True
    r.set(ACTIVE_CONFIG_KEY, json.dumps(config))
    r.rpush(CONFIG_HISTORY_KEY, json.dumps(config))
    logger.info(f"Manual config override applied: {config}")
