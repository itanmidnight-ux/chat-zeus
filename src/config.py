"""Configuración central y rutas para el sistema compatible con Termux."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    base_dir: Path
    chatbot_dir: Path
    models_dir: Path
    data_dir: Path
    logs_dir: Path
    db_path: Path
    report_dir: Path
    checkpoint_dir: Path
    max_task_memory_mb: int = 1024
    simulation_chunk_size: int = 64
    default_steps: int = 500
    max_workers: int = 2
    worker_poll_interval_ms: int = 200
    internet_timeout_sec: int = 8
    max_external_queries: int = 36
    internet_max_retries: int = 3
    max_history_messages: int = 8
    max_inline_context_chars: int = 2000
    max_checkpoint_history: int = 200


ROOT_DIR = Path(__file__).resolve().parent.parent
TERMUX_DATA_ROOT = Path(os.environ.get('CHAT_ZEUS_DATA_ROOT', ROOT_DIR / 'data'))
CONFIG = AppConfig(
    base_dir=ROOT_DIR,
    chatbot_dir=TERMUX_DATA_ROOT / 'chatbot',
    models_dir=TERMUX_DATA_ROOT / 'models',
    data_dir=TERMUX_DATA_ROOT / 'data',
    logs_dir=TERMUX_DATA_ROOT / 'logs',
    db_path=TERMUX_DATA_ROOT / 'data' / 'knowledge.sqlite3',
    report_dir=TERMUX_DATA_ROOT / 'data' / 'reports',
    checkpoint_dir=TERMUX_DATA_ROOT / 'data' / 'checkpoints',
)


def ensure_directories() -> None:
    """Crea la estructura de carpetas exigida por la app."""
    for path in [
        CONFIG.chatbot_dir,
        CONFIG.models_dir,
        CONFIG.data_dir,
        CONFIG.logs_dir,
        CONFIG.report_dir,
        CONFIG.checkpoint_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)
