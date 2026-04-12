"""Centralized logging configuration for the Scripture Translation system.

This module provides a single source of truth for logging configuration.
Library code calls get_logger(); entrypoints call configure_logging() once.

This prevents the anti-pattern of multiple basicConfig() calls scattered
throughout the codebase, which are non-deterministic based on import order.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

_configured = False


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a named logger without calling basicConfig.

    Args:
        name: The logger name (typically __name__).
        level: Log level (used only if configure_logging not yet called).

    Returns:
        A configured logger instance.
    """
    logger = logging.getLogger(name)
    # If no handler configured yet, add a default stderr handler
    if not logger.handlers and not logging.root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


def configure_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
) -> None:
    """Configure root logger once. Call from entrypoints only.

    This should be called exactly once at application startup (app.py main(),
    cli.py main(), training scripts), before importing modules that use logging.

    Args:
        level: Root log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to log file. If provided, logs to both file and stderr.
        fmt: Log message format string.
    """
    global _configured
    if _configured:
        return

    handlers = [logging.StreamHandler(sys.stderr)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=handlers,
    )
    _configured = True
