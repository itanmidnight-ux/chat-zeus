"""Configuración central y perfil adaptativo de recursos para Linux."""
from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, '').strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _detect_memory_mb() -> int:
    page_size = os.sysconf('SC_PAGE_SIZE') if hasattr(os, 'sysconf') else 4096
    physical_pages = os.sysconf('SC_PHYS_PAGES') if hasattr(os, 'sysconf') else 0
    if not page_size or not physical_pages:
        return 2048
    return max(512, int(page_size * physical_pages / (1024 * 1024)))


def _detect_cpu_count() -> int:
    return max(1, os.cpu_count() or 1)


CPU_COUNT = _detect_cpu_count()
TOTAL_MEMORY_MB = _detect_memory_mb()
IS_LINUX = platform.system().lower() == 'linux'
DEFAULT_MEMORY_BUDGET_MB = min(max(384, TOTAL_MEMORY_MB // 3), max(512, TOTAL_MEMORY_MB - 512))
DEFAULT_WORKERS = min(max(2, CPU_COUNT - 1), 6)
DEFAULT_SIM_CHUNK_SIZE = min(256, max(32, (TOTAL_MEMORY_MB // 256) * 16))
DEFAULT_HARD_STEP_CAP = min(4096, max(720, DEFAULT_SIM_CHUNK_SIZE * max(8, CPU_COUNT * 3)))
DEFAULT_OPT_ITERATIONS = min(24, max(8, CPU_COUNT * 2))
DEFAULT_EXTERNAL_QUERIES = min(24, max(8, CPU_COUNT * 3))
DEFAULT_STORAGE_BATCH = min(128, max(16, CPU_COUNT * 8))


@dataclass(frozen=True)
class RuntimeProfile:
    platform_name: str
    cpu_count: int
    total_memory_mb: int
    recommended_workers: int
    recommended_chunk_size: int
    recommended_step_cap: int
    recommended_memory_budget_mb: int


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
    runtime_profile: RuntimeProfile
    max_task_memory_mb: int = DEFAULT_MEMORY_BUDGET_MB
    simulation_chunk_size: int = DEFAULT_SIM_CHUNK_SIZE
    default_steps: int = 480
    max_workers: int = DEFAULT_WORKERS
    worker_poll_interval_ms: int = 200
    internet_timeout_sec: int = 8
    max_external_queries: int = DEFAULT_EXTERNAL_QUERIES
    internet_max_retries: int = 3
    max_history_messages: int = 8
    max_conversation_context_chars: int = 4096
    storage_stream_batch_size: int = DEFAULT_STORAGE_BATCH
    storage_retry_attempts: int = 3
    max_inline_context_chars: int = 2000
    max_checkpoint_history: int = 96
    hard_step_cap: int = DEFAULT_HARD_STEP_CAP
    checkpoint_retention_per_prefix: int = 36
    optimizer_iterations: int = DEFAULT_OPT_ITERATIONS
    ml_checkpoint_file: str = 'lightweight_ml_state.json'
    enable_native_ml_backend_probe: bool = os.environ.get('CHAT_ZEUS_ENABLE_NATIVE_ML_BACKEND_PROBE', '').strip().lower() in {'1', 'true', 'yes', 'on'}


ROOT_DIR = Path(__file__).resolve().parent.parent
TERMUX_DATA_ROOT = Path(os.environ.get('CHAT_ZEUS_DATA_ROOT', ROOT_DIR / 'data'))
RUNTIME_PROFILE = RuntimeProfile(
    platform_name=platform.system() or 'Unknown',
    cpu_count=CPU_COUNT,
    total_memory_mb=TOTAL_MEMORY_MB,
    recommended_workers=DEFAULT_WORKERS,
    recommended_chunk_size=DEFAULT_SIM_CHUNK_SIZE,
    recommended_step_cap=DEFAULT_HARD_STEP_CAP,
    recommended_memory_budget_mb=DEFAULT_MEMORY_BUDGET_MB,
)
CONFIG = AppConfig(
    base_dir=ROOT_DIR,
    chatbot_dir=TERMUX_DATA_ROOT / 'chatbot',
    models_dir=TERMUX_DATA_ROOT / 'models',
    data_dir=TERMUX_DATA_ROOT / 'data',
    logs_dir=TERMUX_DATA_ROOT / 'logs',
    db_path=TERMUX_DATA_ROOT / 'data' / 'knowledge.sqlite3',
    report_dir=TERMUX_DATA_ROOT / 'data' / 'reports',
    checkpoint_dir=TERMUX_DATA_ROOT / 'data' / 'checkpoints',
    runtime_profile=RUNTIME_PROFILE,
    max_task_memory_mb=_env_int('CHAT_ZEUS_MAX_TASK_MEMORY_MB', DEFAULT_MEMORY_BUDGET_MB),
    simulation_chunk_size=_env_int('CHAT_ZEUS_SIMULATION_CHUNK_SIZE', DEFAULT_SIM_CHUNK_SIZE),
    default_steps=_env_int('CHAT_ZEUS_DEFAULT_STEPS', 480),
    max_workers=_env_int('CHAT_ZEUS_MAX_WORKERS', DEFAULT_WORKERS),
    internet_timeout_sec=_env_int('CHAT_ZEUS_INTERNET_TIMEOUT_SEC', 8),
    max_external_queries=_env_int('CHAT_ZEUS_MAX_EXTERNAL_QUERIES', DEFAULT_EXTERNAL_QUERIES),
    storage_stream_batch_size=_env_int('CHAT_ZEUS_STORAGE_BATCH_SIZE', DEFAULT_STORAGE_BATCH),
    hard_step_cap=_env_int('CHAT_ZEUS_HARD_STEP_CAP', DEFAULT_HARD_STEP_CAP),
    optimizer_iterations=_env_int('CHAT_ZEUS_OPT_ITERATIONS', DEFAULT_OPT_ITERATIONS),
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
