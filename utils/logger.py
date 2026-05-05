"""
utils/logger.py — Rich coloured logging
"""
import logging
import sys
from datetime import datetime
from pathlib import Path

try:
    from rich.logging import RichHandler
    from rich.console import Console
    _RICH = True
except ImportError:
    _RICH = False


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # File handler
    fh = logging.FileHandler(
        log_dir / f"alphabot_{datetime.now().strftime('%Y%m%d')}.log",
        encoding="utf-8"
    )
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    ))
    logger.addHandler(fh)

    # Console handler
    if _RICH:
        ch = RichHandler(rich_tracebacks=True, show_path=False)
    else:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%H:%M:%S"
        ))
    logger.addHandler(ch)
    return logger
