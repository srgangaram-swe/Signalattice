"""Centralised logging configuration.

A single :func:`configure_logging` entry-point keeps log formatting consistent
across the CLI, library calls and tests. We use :class:`rich.logging.RichHandler`
when available for readable console output, and fall back to the standard
library otherwise so the package has no hard dependency on ``rich`` at runtime.
"""

from __future__ import annotations

import logging
import os

_CONFIGURED = False
_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: str | None = None, *, force: bool = False) -> None:
    """Configure the root logger once for the whole process.

    Parameters
    ----------
    level:
        Logging level name (e.g. ``"INFO"``). If ``None`` the ``QRDP_LOG_LEVEL``
        environment variable is consulted, defaulting to ``"INFO"``.
    force:
        Re-configure even if logging was already set up. Useful in tests.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    level_name = (level or os.getenv("QRDP_LOG_LEVEL") or "INFO").upper()
    log_level = getattr(logging, level_name, logging.INFO)

    handler: logging.Handler
    try:  # pragma: no cover - depends on optional dependency presence
        from rich.logging import RichHandler

        handler = RichHandler(
            rich_tracebacks=True,
            show_path=False,
            markup=False,
            log_time_format="[%X]",
        )
        fmt = "%(message)s"
    except Exception:  # pragma: no cover - fallback path
        handler = logging.StreamHandler()
        fmt = _DEFAULT_FORMAT

    handler.setFormatter(logging.Formatter(fmt, datefmt=_DATE_FORMAT))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(log_level)

    # Tame noisy third-party loggers.
    for noisy in ("urllib3", "matplotlib", "yfinance", "peewee"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, ensuring logging is configured first."""
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
