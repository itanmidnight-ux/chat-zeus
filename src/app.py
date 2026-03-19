"""CLI entrypoint for the autonomous reasoning agent."""
from __future__ import annotations

from src.autonomous_system import AutonomousReasoningSystem
from src.chatbot import ChatbotInterface
from src.config import CONFIG, ensure_directories
from src.storage import StorageManager
from src.termux_ui import TermuxUI
from src.utils import apply_soft_memory_limit, ensure_environment_defaults, setup_logging
from src.worker import BackgroundExecutor


def build_app() -> ChatbotInterface:
    ensure_environment_defaults()
    apply_soft_memory_limit(CONFIG.max_task_memory_mb)
    ensure_directories()
    logger = setup_logging(CONFIG.logs_dir)
    storage = StorageManager(CONFIG.db_path, CONFIG.checkpoint_dir)
    background_executor = BackgroundExecutor(max_workers=CONFIG.max_workers)
    autonomous_system = AutonomousReasoningSystem(storage=storage, memory_path=CONFIG.models_dir / 'agent_memory.json')
    return ChatbotInterface(storage, autonomous_system, background_executor, logger)


def main() -> None:
    app = build_app()
    ui = TermuxUI()
    print(ui.render_welcome())
    try:
        while True:
            try:
                question = input(ui.prompt()).strip()
            except EOFError:
                break
            except KeyboardInterrupt:
                print()
                break
            if not question:
                continue
            if question.lower() in {'salir', 'exit', 'quit'}:
                break
            print(app.safe_answer(question).get('response_text', ''))
    finally:
        app.background_executor.shutdown()
