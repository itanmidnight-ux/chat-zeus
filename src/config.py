"""Configuración central y rutas para el sistema compatible con Termux."""
from __future__ import annotations

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
    simulation_chunk_size: int = 100
    default_steps: int = 500
    max_workers: int = 2
    internet_timeout_sec: int = 8


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT_DIR / 'data'
CONFIG = AppConfig(
    base_dir=ROOT_DIR,
    chatbot_dir=DATA_ROOT / 'chatbot',
    models_dir=DATA_ROOT / 'models',
    data_dir=DATA_ROOT / 'data',
    logs_dir=DATA_ROOT / 'logs',
    db_path=DATA_ROOT / 'data' / 'knowledge.sqlite3',
    report_dir=DATA_ROOT / 'data' / 'reports',
    checkpoint_dir=DATA_ROOT / 'data' / 'checkpoints',
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
