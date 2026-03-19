"""Utilidades generales de E/S, JSON, texto y manejo de errores."""
from __future__ import annotations

import json
import logging
import os
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('chat_zeus')
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    file_handler = logging.FileHandler(log_dir / 'chat_zeus.log', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def read_json(path: Path, default: Any) -> Any:
    try:
        if not path.exists():
            return default
        with path.open('r', encoding='utf-8') as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def safe_error_message(exc: Exception) -> str:
    return f'{exc.__class__.__name__}: {exc}'


def format_exception(exc: Exception) -> str:
    return ''.join(traceback.format_exception(exc))


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def sanitize_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def soft_memory_limit_bytes(megabytes: int) -> int:
    return max(32, megabytes) * 1024 * 1024


def ensure_environment_defaults() -> None:
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('PYTHONUNBUFFERED', '1')
