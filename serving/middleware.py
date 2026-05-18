"""
Authentication and rate-limiting middleware.

Validates API keys from the x-api-key header against a comma-separated
allowlist in the API_KEYS environment variable. Rate limiting uses a
simple in-memory sliding window (replace with Redis-backed limiter
in production for multi-worker deployments).
"""

import os
import time
import logging
from collections import defaultdict
from fastapi import Header, HTTPException, Request

logger = logging.getLogger(__name__)

# API key allowlist from environment
VALID_KEYS = set(os.environ.get("API_KEYS", "dev-key-123").split(","))

# Simple in-memory rate limiter
_rate_limits: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT_WINDOW = 60   # seconds
RATE_LIMIT_MAX = 100       # requests per window


async def verify_api_key(x_api_key: str = Header(...)):
    """
    Validate the x-api-key header against the allowlist.

    Raises 401 if the key is missing or invalid.
    """
    if x_api_key not in VALID_KEYS:
        logger.warning(f"Rejected invalid API key: {x_api_key[:8]}...")
        raise HTTPException(status_code=401, detail="Invalid API key")


async def rate_limit(request: Request):
    """
    Simple sliding-window rate limiter keyed by client IP.

    In production, replace with a Redis-backed solution (e.g.,
    slowapi or custom Redis ZSET) for multi-worker consistency.
    """
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW

    # Clean old entries
    _rate_limits[client_ip] = [
        ts for ts in _rate_limits[client_ip] if ts > window_start
    ]

    if len(_rate_limits[client_ip]) >= RATE_LIMIT_MAX:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {RATE_LIMIT_MAX} requests per {RATE_LIMIT_WINDOW}s.",
        )

    _rate_limits[client_ip].append(now)
