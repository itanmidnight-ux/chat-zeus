"""Utilidades generales de E/S, JSON, texto y manejo de errores."""
from __future__ import annotations

import contextlib
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
    temp_path = path.with_suffix(path.suffix + '.tmp')
    with temp_path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    temp_path.replace(path)


def safe_error_message(exc: Exception) -> str:
    return f'{exc.__class__.__name__}: {exc}'


def format_exception(exc: Exception) -> str:
    return ''.join(traceback.format_exception(exc))


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def sanitize_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def extract_numeric_value(text: str, pattern: str, default: float | int | None = None) -> float | int | None:
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return default
    raw = match.group(1).replace(',', '.')
    if raw.isdigit():
        return int(raw)
    with contextlib.suppress(ValueError):
        value = float(raw)
        return int(value) if value.is_integer() else value
    return default


def soft_memory_limit_bytes(megabytes: int) -> int:
    return max(32, megabytes) * 1024 * 1024


def ensure_environment_defaults() -> None:
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('PYTHONUNBUFFERED', '1')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
    os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
    os.environ.setdefault('BLIS_NUM_THREADS', '1')
    os.environ.setdefault('MALLOC_ARENA_MAX', '2')


def apply_soft_memory_limit(megabytes: int) -> None:
    try:
        import resource
    except Exception:
        return
    limit = soft_memory_limit_bytes(megabytes)
    with contextlib.suppress(Exception):
        current_soft, current_hard = resource.getrlimit(resource.RLIMIT_AS)
        target_soft = limit if current_soft in (-1, resource.RLIM_INFINITY) else min(current_soft, limit)
        target_hard = current_hard if current_hard not in (-1, resource.RLIM_INFINITY) else max(limit, target_soft)
        resource.setrlimit(resource.RLIMIT_AS, (target_soft, target_hard))


def detect_linker_memory_issue(exc: BaseException) -> bool:
    message = safe_error_message(exc)
    markers = [
        'create_new_page',
        'MAP_FAILED',
        'linker_block_allocator',
        'cannot allocate memory',
        'std::bad_alloc',
    ]
    lowered = message.lower()
    return any(marker.lower() in lowered for marker in markers)
